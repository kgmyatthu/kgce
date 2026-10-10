//! Display half of the connected atlas: every map-aligned raster (heightmap with the accepted bilinear smoothing,
//! `metadata.dat`, fog/snow L8 masks, DXT5 detail and snow masks, the seven-level supertexture) and every vector asset
//! (borders, roads, rivers, trade routes, coastline, trees, bridges, `world.markers`) of
//! `<data>/campaign_maps/nap_world` is rewritten through the same [`Plan`](super::Plan) as the geography and campaign.
//! Nothing is sampled geometry-only: the accepted `raster_scale.py` / `vector_scale.py` atlas branches are reproduced
//! natively from the generated source assets. Unchanged by design: repeating material textures, theatre/lookup/radar
//! TGAs, trade-node/arrow models and environment lighting.

mod dxt;
mod mapping;
mod rasters;
mod supertexture;
mod vector;

use super::Plan;
use crate::model::Progress;
use anyhow::{Context, Result, bail, ensure};
use rasters::{DxtMask, L8, L8Kind};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
};

const STAGE_SUFFIX: &str = ".atlas-new";

pub(crate) fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32)
}

/// Run `f(chunk_index, chunk)` over disjoint `chunk`-byte pieces of `data` on all cores.
pub(crate) fn par_chunks_mut<F>(data: &mut [u8], chunk: usize, f: F)
where
    F: Fn(usize, &mut [u8]) + Sync,
{
    let queue = Mutex::new(data.chunks_mut(chunk.max(1)).enumerate().rev().collect::<Vec<_>>());
    std::thread::scope(|s| {
        for _ in 0..threads() {
            s.spawn(|| {
                // A panicking sibling must not hide its panic behind a poisoned queue; the guard is a statement
                // temporary so the lock is not held while `f` runs.
                loop {
                    let item = queue.lock().unwrap_or_else(|e| e.into_inner()).pop();
                    let Some((i, c)) = item else { break };
                    f(i, c);
                }
            });
        }
    });
}

fn sha256_bytes(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

fn sha256_file(p: &Path) -> Result<String> {
    let mut f = fs::File::open(p).with_context(|| format!("Open {}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            return Ok(hex::encode(h.finalize()));
        }
        h.update(&buf[..n]);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Role {
    Transform,
    Unchanged,
    NotRaster,
}

/// Classify every file of the map folder from the real inventory first; unknown raster-like files raise.
fn classify(rel: &str) -> Result<Role> {
    const EXACT: [&str; 7] = [
        "metadata.dat",
        "display/heightmap/heightmap.tga",
        "display/fogmask/fogmask.dds",
        "display/snow/winter.dds",
        "display/detail/campaign_detail_mask.dds",
        "display/supertexture/supertexture.stpi",
        "display/supertexture/supertexture.stpd",
    ];
    let ext = rel.rsplit_once('.').map(|(_, e)| format!(".{}", e.to_lowercase())).unwrap_or_default();
    let name = rel.rsplit('/').next().unwrap_or(rel);
    if EXACT.contains(&rel) || (rel.starts_with("display/snowmask/") && rel.matches('/').count() == 2 && ext == ".dds") {
        return Ok(Role::Transform);
    }
    if matches!(rel, "display/detail/campaign_detail.dds" | "display/detail/campaign_rocks.dds")
        || (rel.starts_with("display/rivers/textures/") && ext == ".dds")
    {
        return Ok(Role::Unchanged);
    }
    if !rel.contains('/') && ext == ".tga" && (name.ends_with("_map.tga") || name.ends_with("_lookup.tga") || name.starts_with("stratradar_")) {
        return Ok(Role::Unchanged);
    }
    if matches!(ext.as_str(), ".tga" | ".dds" | ".stpi" | ".stpd" | ".dat") {
        bail!("unclassified raster-like input file {rel:?}: refusing to skip or guess");
    }
    const NONRASTER: [&str; 13] = [
        ".esf", ".rigid_spline", ".rigid_mesh", ".rigid_model", ".rigid_trees", ".markers", ".candidate", ".lighting", ".pack", ".exe",
        ".lua", ".json", ".txt",
    ];
    if NONRASTER.contains(&ext.as_str()) || ext.is_empty() || ext == ".md" {
        return Ok(Role::NotRaster);
    }
    bail!("unclassified input file {rel:?}")
}

fn inventory(map_dir: &Path) -> Result<Value> {
    let mut classes: BTreeMap<&'static str, usize> = BTreeMap::new();
    for e in walkdir::WalkDir::new(map_dir).sort_by_file_name() {
        let e = e?;
        if !e.file_type().is_file() {
            continue;
        }
        let rel = e.path().strip_prefix(map_dir)?.to_string_lossy().replace('\\', "/");
        let name = match classify(&rel)? {
            Role::Transform => "transform",
            Role::Unchanged => "unchanged",
            Role::NotRaster => "not_raster_untouched",
        };
        *classes.entry(name).or_default() += 1;
    }
    Ok(json!(classes))
}

fn remove_stale_stage_files(map_dir: &Path) -> Result<()> {
    for e in walkdir::WalkDir::new(map_dir) {
        let e = e?;
        if e.file_type().is_file() && e.file_name().to_string_lossy().ends_with(STAGE_SUFFIX) {
            fs::remove_file(e.path())?;
        }
    }
    Ok(())
}

struct Stage {
    files: Vec<(PathBuf, PathBuf)>,
    hashes: BTreeMap<String, String>,
}

impl Stage {
    fn add_bytes(&mut self, map_dir: &Path, rel: &str, bytes: &[u8]) -> Result<()> {
        let dst = map_dir.join(rel);
        let tmp = PathBuf::from(format!("{}{STAGE_SUFFIX}", dst.display()));
        fs::create_dir_all(dst.parent().unwrap())?;
        fs::write(&tmp, bytes).with_context(|| format!("Write {}", tmp.display()))?;
        self.hashes.insert(rel.to_string(), sha256_bytes(bytes));
        self.files.push((tmp, dst));
        Ok(())
    }
}

/// Transform the generated display assets of `<data>/campaign_maps/nap_world` in place (once, after source-space
/// edits and before packing). Returns the factual validation report.
pub fn transform(data: &Path, plan: &Plan, progress: &Progress<'_>) -> Result<Value> {
    plan.validate()?;
    let map_dir = data.join("campaign_maps").join("nap_world");
    ensure!(map_dir.is_dir(), "{} not found", map_dir.display());
    remove_stale_stage_files(&map_dir)?;
    progress("Atlas display: inventorying raster and vector assets");
    let classes = inventory(&map_dir)?;
    let read = |rel: &str| -> Result<Vec<u8>> { fs::read(map_dir.join(rel)).with_context(|| format!("Read {rel}")) };

    // ---- single-resolution rasters (all sources are read before anything is replaced)
    progress("Atlas display: heightmap, metadata, fog and snow rasters");
    let mut stage = Stage { files: Vec::new(), hashes: BTreeMap::new() };
    let mut outputs = serde_json::Map::new();
    let heightmap = L8::load(L8Kind::Heightmap, &read(L8Kind::Heightmap.rel())?, plan)?;
    for kind in [L8Kind::Heightmap, L8Kind::Metadata, L8Kind::Fogmask, L8Kind::Snow] {
        let asset = if kind == L8Kind::Heightmap { None } else { Some(L8::load(kind, &read(kind.rel())?, plan)?) };
        let asset = asset.as_ref().unwrap_or(&heightmap);
        let bytes = asset.render()?;
        if kind == L8Kind::Heightmap {
            outputs.insert("heightmap_check".into(), asset.check_bilinear(&bytes)?);
        }
        let mut rep = asset.report();
        rep["bytes"] = json!(bytes.len());
        stage.add_bytes(&map_dir, kind.rel(), &bytes)?;
        outputs.insert(kind.name().into(), rep);
    }

    progress("Atlas display: detail and snow masks");
    let mut mask_rels = vec![("detail_mask", "display/detail/campaign_detail_mask.dds".to_string())];
    let mut snow: Vec<String> = fs::read_dir(map_dir.join("display/snowmask"))
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    snow.sort();
    mask_rels.extend(snow.into_iter().map(|n| ("snowmask", format!("display/snowmask/{n}"))));
    let hm = Some((heightmap.top.as_slice(), heightmap.w, heightmap.h));
    for (kind, rel) in &mask_rels {
        let mask = DxtMask::load(kind, rel, &read(rel)?, plan)?;
        let (bytes, rep) = mask.render(hm)?;
        stage.add_bytes(&map_dir, rel, &bytes)?;
        outputs.insert(format!("{kind}:{}", rel.rsplit('/').next().unwrap()), rep);
    }

    // ---- supertexture (streamed tile by tile, staged beside the source)
    let st = supertexture::transform(&map_dir, plan, progress)?;
    let (mut stpi_stage, mut stpd_stage) = (None, None);
    for (tmp, dst) in &st.files {
        match dst.extension().and_then(|e| e.to_str()) {
            Some("stpi") => stpi_stage = Some((tmp.clone(), dst.clone())),
            _ => stpd_stage = Some((tmp.clone(), dst.clone())),
        }
    }
    let (stpi_stage, stpd_stage) = (stpi_stage.context("stpi stage missing")?, stpd_stage.context("stpd stage missing")?);
    let out_px = st.report["output_px"].as_array().context("supertexture report lacks output_px")?;
    let out_dims = (out_px[0].as_u64().unwrap_or(0) as usize, out_px[1].as_u64().unwrap_or(0) as usize);
    let st_check = supertexture::verify(&fs::read(&stpi_stage.0)?, fs::metadata(&stpd_stage.0)?.len(), out_dims)?;
    let mut st_report = st.report.clone();
    st_report["verification"] = st_check;
    st_report["sha256"] = json!({
        "display/supertexture/supertexture.stpi": sha256_file(&stpi_stage.0)?,
        "display/supertexture/supertexture.stpd": sha256_file(&stpd_stage.0)?,
    });

    // ---- vector assets (validated in memory, then committed in place)
    let vec_report = vector::transform(&map_dir, plan, progress)?;

    // ---- commit rasters
    progress("Atlas display: replacing display assets");
    for (tmp, dst) in stage.files.iter().chain(st.files.iter()) {
        fs::rename(tmp, dst).with_context(|| format!("Replace {}", dst.display()))?;
    }
    Ok(json!({
        "format": 1,
        "scale": plan.scale,
        "source_world": plan.source_world,
        "world": plan.world,
        "inventory": classes,
        "rasters": outputs,
        "raster_sha256": stage.hashes,
        "supertexture": st_report,
        "vector": vec_report,
        "unchanged": [
            "display/detail/campaign_detail.dds",
            "display/detail/campaign_rocks.dds",
            "display/rivers/textures/*.dds",
            "*_map.tga", "*_lookup.tga", "stratradar_*.tga",
            "display/tradenodes", "display/arrows", "display/environment"
        ],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_refuses_unknown_raster_like_files_and_classifies_known_ones() {
        assert_eq!(classify("display/heightmap/heightmap.tga").unwrap(), Role::Transform);
        assert_eq!(classify("display/snowmask/campaign_snow_mask_winter.dds").unwrap(), Role::Transform);
        assert_eq!(classify("display/rivers/textures/river_normal.dds").unwrap(), Role::Unchanged);
        assert_eq!(classify("europe_lookup.tga").unwrap(), Role::Unchanged);
        assert_eq!(classify("display/world.markers.candidate").unwrap(), Role::NotRaster);
        assert!(classify("display/detail/new_mask.dds").is_err());
        assert!(classify("display/snowmask/nested/x.dds").is_err());
        assert!(classify("notes.bin").is_err());
    }
}
