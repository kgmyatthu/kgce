//! Native conversion of the ETW world-map display assets into the NTW `campaign_maps/nap_world` layout
//! (`display/` tree, theatre TGAs in the map root and the MTGI `metadata.dat` battle-terrain grid).
//!
//! Optional generated extras of earlier experiments (`display/world.markers`, `display/traderoutes/*`) are intentionally
//! NOT produced: the accepted installation ships neither.

mod art;
mod fixed_art;
pub mod formats;
mod raster;
mod snow;
pub mod theatre;

pub use theatre::{
    CanvasSizes, default_sea_point, extend_canvases, extend_canvases_in, import_painted_terrain, recanvas_tga_bytes,
    recanvas_tgas, theatre_stem,
};

use crate::{
    model::{Progress, TerrainRelief},
    snapshot::{Game, Snapshot},
};
use anyhow::{Context, Result, bail, ensure};
use formats as F;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

/// NTW coastline meshes sit at y = 0 (ETW's sit at 1.7534).
const COAST_Y: f32 = 0.0;
const META_W: usize = 4096;
const META_H: usize = 2048;

fn wr(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    fs::write(path, data).with_context(|| format!("Write {}", path.display()))
}

/// Names that are direct children of `prefix` (a directory ending in `/`) and end with `suffix`.
fn children(snapshot: &Snapshot, game: Game, prefix: &str, suffix: &str) -> Vec<String> {
    snapshot
        .names(game, prefix)
        .into_iter()
        .filter(|n| {
            n.len() > prefix.len() && !n[prefix.len()..].contains('/') && n.ends_with(suffix)
        })
        .collect()
}

fn base(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

/// Run `f` over `items` on all cores, preserving order.
fn parallel_map<T: Sync, R: Send>(
    items: &[T],
    f: impl Fn(&T) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let per = items.len().div_ceil(threads).max(1);
    let parts: Vec<Result<Vec<R>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(per)
            .map(|part| {
                let f = &f;
                scope.spawn(move || part.iter().map(f).collect::<Result<Vec<R>>>())
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| bail!("Worker thread panicked")))
            .collect()
    });
    let mut out = Vec::with_capacity(items.len());
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------- map-root TGAs

fn step_root_tgas(snapshot: &Snapshot, map_dir: &Path, progress: &Progress) -> Result<Vec<String>> {
    let prefix = "campaign_maps/global_map/";
    let names = children(snapshot, Game::Etw, prefix, ".tga");
    ensure!(
        !names.is_empty(),
        "ETW global_map contains no theatre TGA files"
    );
    for n in &names {
        wr(&map_dir.join(base(n)), &snapshot.read(Game::Etw, n)?)?;
    }
    progress(&format!(
        "Map root: {} theatre TGA files from ETW global_map",
        names.len()
    ));
    Ok(names.iter().map(|n| base(n).to_string()).collect())
}

// ---------------------------------------------------------------------------------------------- heightmap

/// Display heightmap with rows top-first: `raw` is the elevation after the fixed theatre repairs (what the terrain is),
/// `shaped` the relief-curved result actually written (absent when the original relief curve is kept).
struct Heights {
    raw: Vec<u8>,
    shaped: Option<Vec<u8>>,
}

impl Heights {
    fn shaped(&self) -> &[u8] {
        self.shaped.as_deref().unwrap_or(&self.raw)
    }
}

/// Apply the fixed map's raw terrain repairs, then the project's relief curve and land-only smoothing.
/// Relief preserves the repaired land/sea outline. Fog derives from the shaped heightmap, snow from the repaired one.
fn step_heightmap(
    snapshot: &Snapshot,
    display: &Path,
    relief: &TerrainRelief,
    fixed: &fixed_art::FixedArt,
    progress: &Progress,
) -> Result<Heights> {
    let mut b = snapshot.read(Game::Etw, "heightmaps/default.tga")?;
    let h = F::tga_header(&b)?;
    ensure!(
        (h.width, h.height, h.bpp, h.image_type) == (8192, 4096, 8, 3),
        "Unexpected ETW heightmap format"
    );
    ensure!(b.len() == 18 + 8192 * 4096, "Unexpected ETW heightmap size");
    ensure!(
        h.id_len == 0 && h.desc & 0x30 == 0,
        "Fixed terrain artwork requires the original bottom-left heightmap"
    );
    let painted = fixed.apply_heights(&mut b[18..])?;
    progress(&format!("Display: fixed theatre terrain applied to {painted} heightmap pixels"));
    if relief.is_original() {
        wr(&display.join("heightmap/heightmap.tga"), &b)?;
        progress("Display: heightmap (fixed theatre terrain, 8192x4096, original relief curve)");
        let (_, _, top) = F::load_tga8(&b)?;
        return Ok(Heights { raw: top, shaped: None });
    }
    relief.validate()?;
    let (w, hh, top) = F::load_tga8(&b)?;
    let shaped = relief_heights(&top, w, hh, relief);
    // write back in the file's own row order (load_tga8 returns top-down rows; the file is bottom-up unless the
    // descriptor's top-left bit is set)
    let mut out = b[..18 + h.id_len].to_vec();
    if h.desc & 0x20 == 0 {
        for row in shaped.chunks_exact(w).rev() {
            out.extend_from_slice(row);
        }
    } else {
        out.extend_from_slice(&shaped);
    }
    wr(&display.join("heightmap/heightmap.tga"), &out)?;
    let land: Vec<&u8> = shaped.iter().filter(|&&v| v > 0).collect();
    let mean = land.iter().map(|&&v| v as f64).sum::<f64>() / land.len().max(1) as f64;
    progress(&format!(
        "Display: heightmap relief shaped (knee {}, uplands x{:.2}, blur {:.1} px): land mean {:.1}",
        relief.knee, relief.upland_scale, relief.smooth_sigma, mean
    ));
    Ok(Heights { raw: top, shaped: Some(shaped) })
}

/// Relief curve + land-only Gaussian blur; sea pixels are excluded from the blur so coastlines never move.
fn relief_heights(top: &[u8], w: usize, h: usize, relief: &TerrainRelief) -> Vec<u8> {
    let curved: Vec<f32> = top
        .iter()
        .map(|&v| if v == 0 { 0.0 } else { relief.curve(v) })
        .collect();
    let smoothed = if relief.smooth_sigma > 0.0 {
        let mask: Vec<f32> = top.iter().map(|&v| if v == 0 { 0.0 } else { 1.0 }).collect();
        let num = raster::gaussian_blur(&curved, w, h, relief.smooth_sigma as f64);
        let den = raster::gaussian_blur(&mask, w, h, relief.smooth_sigma as f64);
        num.iter()
            .zip(&den)
            .map(|(&n, &d)| if d > 1e-6 { n / d } else { 0.0 })
            .collect()
    } else {
        curved
    };
    top.iter()
        .zip(&smoothed)
        .map(|(&orig, &v)| {
            if orig == 0 {
                0
            } else {
                v.round().clamp(1.0, 255.0) as u8
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------- supertexture

fn step_supertexture(
    snapshot: &Snapshot,
    display: &Path,
    fixed: &fixed_art::FixedArt,
    progress: &Progress,
) -> Result<()> {
    let stpi = F::parse_stpi(&snapshot.read(Game::Etw, "world_.stpi")?)?;
    ensure!(
        stpi.tile == 512 && stpi.tile_bytes == 0x40080,
        "Unexpected ETW supertexture tile layout"
    );
    ensure!(
        (stpi.width, stpi.height, stpi.levels) == (65536, 32768, 7)
            && stpi.layers.iter().enumerate().all(|(level, layer)| {
                layer.tx == (128 >> level) && layer.ty == (64 >> level)
            }),
        "Fixed theatre artwork requires the original seven-level world supertexture layout"
    );
    let stpd_path = snapshot.path(Game::Etw, "world_.stpd")?;
    let mut stpd =
        fs::File::open(&stpd_path).with_context(|| format!("Open {}", stpd_path.display()))?;
    let stpd_len = stpd.metadata()?.len();
    progress(&format!(
        "Display: supertexture, {} levels {}x{}",
        stpi.levels, stpi.width, stpi.height
    ));

    // ETW dedupes identical tiles to one (offset, size); convert each source tile once.
    let mut unique: Vec<(u32, u32)> = stpi
        .layers
        .iter()
        .flat_map(|l| l.entries.iter().map(|e| (e[0], e[1])))
        .collect();
    unique.sort_unstable();
    unique.dedup();
    progress(&format!(
        "Display: converting {} unique supertexture tiles",
        unique.len()
    ));

    let mut converted: HashMap<(u32, u32), Vec<u8>> = HashMap::with_capacity(unique.len());
    let batch = 256usize;
    for (done, keys) in unique.chunks(batch).enumerate() {
        let mut sources = Vec::with_capacity(keys.len());
        for &(off, size) in keys {
            ensure!(
                off as u64 + size as u64 <= stpd_len,
                "Supertexture tile outside world_.stpd"
            );
            stpd.seek(SeekFrom::Start(off as u64))?;
            let mut buf = vec![0u8; size as usize];
            stpd.read_exact(&mut buf)?;
            sources.push(buf);
        }
        let tiles = parallel_map(&sources, |z| F::etw_tile_to_ntw(z))?;
        for (k, t) in keys.iter().zip(tiles) {
            converted.insert(*k, t);
        }
        if done % 8 == 0 {
            progress(&format!(
                "Display: supertexture tiles {}/{}",
                ((done + 1) * batch).min(unique.len()),
                unique.len()
            ));
        }
    }

    let out_path = display.join("supertexture/supertexture.stpd");
    fs::create_dir_all(out_path.parent().unwrap())?;
    let mut writer = BufWriter::new(
        fs::File::create(&out_path).with_context(|| format!("Create {}", out_path.display()))?,
    );
    let mut written: u64 = 0;
    let mut tile_id: u32 = 0;
    let mut painted_tiles = 0usize;
    let mut header = Vec::new();
    for v in [
        stpi.levels,
        stpi.width,
        stpi.height,
        512,
        0x40000,
        stpi.levels,
    ] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    for (li, layer) in stpi.layers.iter().enumerate() {
        header.extend_from_slice(&layer.tx.to_le_bytes());
        header.extend_from_slice(&layer.ty.to_le_bytes());
        for e in &layer.entries {
            let original = converted[&(e[0], e[1])].as_slice();
            let replacement = fixed.tile(tile_id as usize, original)?;
            painted_tiles += usize::from(replacement.is_some());
            let tile = replacement.unwrap_or(original);
            tile_id += 1;
            ensure!(
                written <= u32::MAX as u64,
                "Supertexture exceeds the 4 GiB stpd offset range"
            );
            let colour: u32 = if li == 0 { 0xFFFF_FF00 } else { 0xEB1C_F400 };
            for v in [written as u32, tile.len() as u32, 0x40000, tile_id, colour] {
                header.extend_from_slice(&v.to_le_bytes());
            }
            writer.write_all(tile)?;
            written += tile.len() as u64;
        }
    }
    writer.flush()?;
    header.extend_from_slice(&1u32.to_le_bytes());
    wr(&display.join("supertexture/supertexture.stpi"), &header)?;
    progress(&format!(
        "Display: supertexture written ({} MB)",
        written / 1_000_000
    ));
    progress(&format!(
        "Display: fixed theatre frames and terrain applied to {painted_tiles} supertexture tiles across all mip levels"
    ));
    Ok(())
}

// ---------------------------------------------------------------------------------------------- detail / snow / fog

fn step_detail(snapshot: &Snapshot, display: &Path, progress: &Progress) -> Result<()> {
    let read = |n: &str| snapshot.read(Game::Etw, n);
    wr(
        &display.join("detail/campaign_detail.dds"),
        &read("campaign/terrain/campaign_detail.dds")?,
    )?;
    wr(
        &display.join("detail/campaign_detail_mask.dds"),
        &read("campaign/terrain/campaign_detail_mask.dds")?,
    )?;
    wr(
        &display.join("detail/campaign_rocks.dds"),
        &F::dds_with_linear_size(&read("campaign/terrain/rocks1_diffuse.dds")?)?,
    )?;
    // The summer mask is blank over America (nothing to continue); the winter mask is written with the snow.
    wr(
        &display.join("snowmask/campaign_snow_mask_summer.dds"),
        &read("campaign/terrain/campaign_snow_mask_summer.dds")?,
    )?;
    for n in [
        "river_diffuse",
        "river_distortion",
        "river_normal",
        "river_surface",
    ] {
        wr(
            &display.join(format!("rivers/textures/{n}.dds")),
            &read(&format!("rigidmodels/campaignrivers/textures/{n}.dds"))?,
        )?;
    }
    progress("Display: detail, rocks, summer snow mask and river textures");
    Ok(())
}

/// `snowmask/campaign_snow_mask_winter.dds`, `snow/winter.dds` and `fogmask/fogmask.dds`.
///
/// The ETW winter snow mask is painted for the original America rectangle and has a rounded edge inside its old west
/// border; [`snow::extend_winter_mask`] continues its latitude zone over the enlarged theatre (with a terrain-driven
/// snow line) and the result feeds both the DXT5 mask pair member and the 4096x2048 luminance `snow/winter.dds`
/// (bicubic upsampling, masked by land). The rest of the mask and every mip block outside the changed area is kept
/// bit-exact. The summer mask carries no America snow and is copied unchanged by [`step_detail`].
fn step_snow_fog(
    snapshot: &Snapshot,
    display: &Path,
    heights: &Heights,
    progress: &Progress,
) -> Result<()> {
    const HW: usize = 8192;
    const HH: usize = 4096;
    let heightmap_top = heights.shaped();
    ensure!(heightmap_top.len() == HW * HH, "Unexpected heightmap size");

    let winter = snapshot.read(Game::Etw, "campaign/terrain/campaign_snow_mask_winter.dds")?;
    let (sw, sh, source) = raster::dxt5_red(&winter)?;
    ensure!(
        (sw, sh) == (snow::MASK_W, snow::MASK_H),
        "Unexpected ETW winter snow mask size {sw}x{sh}"
    );
    let extended = snow::extend_winter_mask(&source, &heights.raw)?;
    wr(
        &display.join("snowmask/campaign_snow_mask_winter.dds"),
        &snow::patch_winter_dds(&winter, &source, &extended)?,
    )?;
    let big = raster::resize_l8(&extended, sw, sh, 4096, 2048, raster::Filter::Bicubic)?;
    let mut land = vec![false; 4096 * 2048];
    for y in 0..2048 {
        for x in 0..4096 {
            let mut any = false;
            for dy in 0..2 {
                let row = (y * 2 + dy) * HW + x * 2;
                any |= heightmap_top[row] > 0 || heightmap_top[row + 1] > 0;
            }
            land[y * 4096 + x] = any;
        }
    }
    let land = raster::dilate_cross(&land, 4096, 2048);
    let snow_pixels: Vec<u8> = big
        .iter()
        .zip(&land)
        .map(|(&v, &l)| if l { v } else { 0 })
        .collect();
    let mut snow_dds = F::dds_header_l8(4096, 2048);
    snow_dds.extend_from_slice(&snow_pixels);
    wr(&display.join("snow/winter.dds"), &snow_dds)?;

    // Fog: NTW fog correlates with blurred terrain height: fog = min(119, 1.6 * gauss(height)) at 8x8 block means.
    let (bw, bh) = (HW / 8, HH / 8);
    let mut blocks = vec![0f32; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let mut sum = 0u32;
            for dy in 0..8 {
                let row = (by * 8 + dy) * HW + bx * 8;
                sum += heightmap_top[row..row + 8]
                    .iter()
                    .map(|&v| v as u32)
                    .sum::<u32>();
            }
            blocks[by * bw + bx] = sum as f32 / 64.0;
        }
    }
    let blur = raster::gaussian_blur(&blocks, bw, bh, 3.0);
    let fog_small: Vec<f32> = blur.iter().map(|v| (1.6 * v).clamp(0.0, 119.0)).collect();
    let fog = raster::resize_f32(&fog_small, bw, bh, 4096, 2048, raster::Filter::Bilinear)?;
    let mut fog_dds = F::dds_header_l8(4096, 2048);
    fog_dds.extend(fog.iter().map(|v| v.clamp(0.0, 255.0) as u8));
    wr(&display.join("fogmask/fogmask.dds"), &fog_dds)?;
    progress("Display: winter snow mask extended over the enlarged America theatre; snow/winter.dds and fogmask/fogmask.dds written");
    Ok(())
}

// ---------------------------------------------------------------------------------------------- coastline

fn step_coast(snapshot: &Snapshot, display: &Path, progress: &Progress) -> Result<()> {
    let mut groups = Vec::new();
    for name in children(snapshot, Game::Etw, "campaign/coastline/", ".rigid_mesh") {
        let file = base(&name).to_string();
        if file.ends_with(".cs2.rigid_mesh") {
            continue;
        }
        let Some(num) = file
            .strip_prefix("coastline_group")
            .and_then(|s| s.strip_suffix(".rigid_mesh"))
            .and_then(|s| s.parse::<usize>().ok())
        else {
            bail!("Unexpected coastline mesh name {file}");
        };
        let b = snapshot.read(Game::Etw, &name)?;
        F::parse_coast_mesh(&b).with_context(|| file.clone())?;
        wr(
            &display.join("coastline").join(&file),
            &F::set_coast_height(&b, COAST_Y)?,
        )?;
        groups.push(num);
    }
    groups.sort_unstable();
    // The loader walks `%d` from 0 until the first missing file.
    ensure!(
        groups.iter().enumerate().all(|(i, &g)| i == g) && !groups.is_empty(),
        "Coastline groups are not contiguous from 0: {groups:?}"
    );
    progress(&format!(
        "Display: {} coastline groups (vertex y = {COAST_Y})",
        groups.len()
    ));
    Ok(())
}

// ---------------------------------------------------------------------------------------------- borders / roads / rivers

fn step_splines(
    snapshot: &Snapshot,
    display: &Path,
    keys: &BTreeSet<String>,
    progress: &Progress,
) -> Result<()> {
    // Borders: <region>.rigid_spline or <region>_1.._N; only regions present in global_map.
    let mut borders = 0;
    for name in children(
        snapshot,
        Game::Etw,
        "rigidmodels/campaignborders/",
        ".rigid_spline",
    ) {
        let file = base(&name);
        let stem = &file[..file.len() - ".rigid_spline".len()];
        let key = match stem.rsplit_once('_') {
            Some((head, tail)) if !tail.is_empty() && tail.bytes().all(|c| c.is_ascii_digit()) => {
                head
            }
            _ => stem,
        };
        if !keys.contains(key) {
            continue;
        }
        let b = snapshot.read(Game::Etw, &name)?;
        F::parse_spln(&b).with_context(|| file.to_string())?;
        wr(&display.join("borders").join(file), &b)?;
        borders += 1;
    }

    // Roads: ETW `main`; parts renumbered so they are contiguous from 1.
    let mut by_region: BTreeMap<String, Vec<(u32, String)>> = BTreeMap::new();
    for name in children(
        snapshot,
        Game::Etw,
        "rigidmodels/campaignroads/main/",
        ".rigid_spline",
    ) {
        let file = base(&name);
        let stem = &file[..file.len() - ".rigid_spline".len()];
        let (key, num) = stem
            .rsplit_once('_')
            .with_context(|| format!("Unexpected road name {file}"))?;
        let num: u32 = num
            .parse()
            .with_context(|| format!("Unexpected road number in {file}"))?;
        if keys.contains(key) {
            by_region
                .entry(key.to_string())
                .or_default()
                .push((num, name.clone()));
        }
    }
    let mut roads = 0;
    for (key, parts) in by_region.iter_mut() {
        parts.sort();
        for (i, (num, name)) in parts.iter().enumerate() {
            let index = i as u32 + 1;
            let b = snapshot.read(Game::Etw, name)?;
            let (version, mut splines) = F::parse_spln(&b)?;
            let out = if *num != index {
                for s in &mut splines {
                    s.name = format!("road:{key}:{index}");
                }
                F::write_spln(version, &splines)
            } else {
                b
            };
            wr(
                &display.join(format!("roads/{key}_{index}.rigid_spline")),
                &out,
            )?;
            roads += 1;
        }
    }

    // Rivers 01..NN (loader: %02d from 01 until the first missing file).
    let rivers = children(
        snapshot,
        Game::Etw,
        "rigidmodels/campaignrivers/",
        ".rigid_spline",
    );
    for (i, name) in rivers.iter().enumerate() {
        ensure!(
            base(name) == format!("{:02}.rigid_spline", i + 1),
            "River splines are not contiguous: {}",
            base(name)
        );
        let b = snapshot.read(Game::Etw, name)?;
        F::parse_spln(&b)?;
        wr(&display.join("rivers").join(base(name)), &b)?;
    }
    progress(&format!(
        "Display: {borders} border, {roads} road ({} regions) and {} river splines",
        by_region.len(),
        rivers.len()
    ));
    Ok(())
}

// ---------------------------------------------------------------------------------------------- trees / bridges / props

fn step_trees_bridges(snapshot: &Snapshot, display: &Path, progress: &Progress) -> Result<()> {
    let trees = snapshot.read(Game::Etw, "rigidmodels/campaigntrees/campaign.rigid_trees")?;
    let models = F::parse_trees(&trees)?;
    let mut missing: Vec<&str> = Vec::new();
    for m in models.iter().filter(|m| m.instances > 0) {
        if !snapshot.has_in_source(Game::Ntw, &m.name.replace('\\', "/")) {
            missing.push(m.name.as_str());
        }
    }
    ensure!(
        missing.is_empty(),
        "Tree models missing from the NTW packs: {missing:?}"
    );
    wr(&display.join("trees/campaign.rigid_trees"), &trees)?;
    let instances: usize = models.iter().map(|m| m.instances).sum();
    let bridges = snapshot.read(Game::Etw, "rigidmodels/campaignbridges/bridge.markers")?;
    let n_bridges = F::parse_bridges(&bridges)?;
    wr(&display.join("bridges/bridge.markers"), &bridges)?;
    progress(&format!(
        "Display: {instances} tree instances / {} models, {n_bridges} bridges",
        models.len()
    ));
    Ok(())
}

fn step_env_props(snapshot: &Snapshot, display: &Path, progress: &Progress) -> Result<()> {
    for season in ["spring", "summer", "autumn", "winter"] {
        let b = snapshot.read(
            Game::Ntw,
            &format!("campaign_maps/nap_europe/display/environment/{season}.lighting"),
        )?;
        wr(&display.join(format!("environment/{season}.lighting")), &b)?;
    }
    // Unreferenced by Napoleon.exe but present in every NTW map folder: keep the folder layout identical.
    wr(
        &display.join("arrows/arrows.rigid_model"),
        &snapshot.read(
            Game::Ntw,
            "campaign_maps/nap_egypt/display/arrows/arrows.rigid_model",
        )?,
    )?;
    wr(
        &display.join("tradenodes/all_tradenode.rigid_model"),
        &snapshot.read(
            Game::Etw,
            "rigidmodels/campaignbuildings/campaignprops/all_tradenode.rigid_model",
        )?,
    )?;
    progress("Display: environment lighting, arrows and trade-node models");
    Ok(())
}

// ---------------------------------------------------------------------------------------------- driver

/// Build `<map_dir>/display/**` and the theatre TGAs of `<map_dir>`; returns the TGA names written to the map root.
/// `art_cache` is the workspace-owned terrain cache directory holding the locally generated fixed theatre art
/// (`<art_cache>/fixed_art`); it is generated from the snapshot's ETW supertexture/heightmap once per unchanged source.
pub fn build_display(
    snapshot: &Snapshot,
    map_dir: &Path,
    region_keys: &BTreeSet<String>,
    relief: &TerrainRelief,
    art_cache: &Path,
    progress: &Progress,
) -> Result<Vec<String>> {
    let display = map_dir.join("display");
    fs::create_dir_all(&display)?;
    let fixed = fixed_art::FixedArt::load_or_generate(snapshot, art_cache, progress)?;
    let tgas = step_root_tgas(snapshot, map_dir, progress)?;
    let heightmap = step_heightmap(snapshot, &display, relief, &fixed, progress)?;
    step_supertexture(snapshot, &display, &fixed, progress)?;
    step_detail(snapshot, &display, progress)?;
    step_snow_fog(snapshot, &display, &heightmap, progress)?;
    step_coast(snapshot, &display, progress)?;
    step_splines(snapshot, &display, region_keys, progress)?;
    step_trees_bridges(snapshot, &display, progress)?;
    step_env_props(snapshot, &display, progress)?;
    Ok(tgas)
}

// ---------------------------------------------------------------------------------------------- metadata.dat

/// ETW keeps the battle-terrain class grid of the world map in battleterrain.pack (MTGI 8192x4096 over exactly the
/// world coordinates kept here). Downsample 2x2 -> 4096x2048 (most frequent class, ties prefer land) and write
/// `<map_dir>/metadata.dat`. Class 9 (Americas only, absent from every NTW metadata.dat) becomes the temperate class 6.
pub fn build_metadata(snapshot: &Snapshot, map_dir: &Path, progress: &Progress) -> Result<()> {
    let b = snapshot.read(Game::Etw, "battleterrain/tiles/metadata.dat")?;
    ensure!(
        b.get(0..4) == Some(b"MTGI"),
        "ETW metadata.dat is not an MTGI file"
    );
    let (w, h) = (F::u32_at(&b, 4)? as usize, F::u32_at(&b, 8)? as usize);
    ensure!(
        (w, h) == (META_W * 2, META_H * 2) && b.len() == 12 + w * h,
        "Unexpected ETW metadata.dat size {w}x{h}"
    );
    let grid: Vec<u8> = b[12..]
        .iter()
        .map(|&v| if v == 9 { 6 } else { v })
        .collect();
    let mut out = Vec::with_capacity(12 + META_W * META_H);
    out.extend_from_slice(b"MTGI");
    out.extend_from_slice(&(META_W as u32).to_le_bytes());
    out.extend_from_slice(&(META_H as u32).to_le_bytes());
    for y in 0..META_H {
        for x in 0..META_W {
            let cells = [
                grid[2 * y * w + 2 * x],
                grid[2 * y * w + 2 * x + 1],
                grid[(2 * y + 1) * w + 2 * x],
                grid[(2 * y + 1) * w + 2 * x + 1],
            ];
            let (mut best, mut class) = (0i32, 0u8);
            for c in 0u8..16 {
                let count = cells.iter().filter(|&&v| v == c).count() as i32;
                let score = count * 2 + i32::from(c != 0);
                if score > best {
                    best = score;
                    class = c;
                }
            }
            out.push(class);
        }
    }
    wr(&map_dir.join("metadata.dat"), &out)?;
    progress(&format!("Map: metadata.dat {META_W}x{META_H}"));
    Ok(())
}

#[cfg(test)]
mod relief_tests {
    use super::*;

    #[test]
    fn relief_keeps_sea_and_lowlands_and_flattens_rough_uplands() {
        // 64x64: sea on the left half, a noisy plateau on the right, a lowland strip in between
        let (w, h) = (64usize, 64usize);
        let mut top = vec![0u8; w * h];
        for y in 0..h {
            for x in 32..w {
                top[y * w + x] = if x < 40 { 20 } else if (x + y) % 2 == 0 { 180 } else { 60 };
            }
        }
        let relief = TerrainRelief::default();
        let out = relief_heights(&top, w, h, &relief);
        assert!(out.iter().zip(&top).all(|(&o, &t)| (o == 0) == (t == 0)), "coastline moved");
        assert!(out.iter().zip(&top).all(|(&o, &t)| t == 0 || o >= 1), "land dropped to sea level");
        let rough = |g: &[u8]| -> f64 {
            let mut sum = 0.0;
            let mut n = 0;
            for y in 0..h {
                for x in 44..w - 1 {
                    sum += (g[y * w + x] as f64 - g[y * w + x + 1] as f64).abs();
                    n += 1;
                }
            }
            sum / n as f64
        };
        assert!(rough(&out) < rough(&top) / 4.0, "uplands not smoothed: {} vs {}", rough(&out), rough(&top));
        let mean = |g: &[u8], x0: usize, x1: usize| -> f64 {
            let mut v = Vec::new();
            for y in 8..h - 8 {
                v.extend(g[y * w + x0..y * w + x1].iter().map(|&p| p as f64));
            }
            v.iter().sum::<f64>() / v.len() as f64
        };
        assert!((mean(&out, 32, 35) - 20.0).abs() < 1.0, "lowland changed: {}", mean(&out, 32, 35));
        let before = mean(&top, 48, 60);
        let after = mean(&out, 48, 60);
        let expected = relief.curve(120) as f64;
        assert!((after - expected).abs() < 3.0, "upland mean {after} vs expected {expected} (was {before})");
        let same = relief_heights(&top, w, h, &TerrainRelief::original());
        assert_eq!(same, top);
    }
}
