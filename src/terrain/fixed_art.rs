//! Source-bound artwork for repaired source theatres: India `"3"`, plus extended America `"1"` in the unscaled layout.
//!
//! Replacement artwork is not shipped: repaired supertexture tiles and raw heightmap runs are generated natively
//! ([`super::art`]) from the user's own ETW snapshot (`world_.stpi`/`world_.stpd`, `heightmaps/default.tga`) and a
//! public NOAA elevation grid, then cached below the workspace-owned terrain cache directory:
//!
//! ```text
//! <art_cache>/fixed_art/
//!   manifest.json   format, binding, source hashes, per-theatre reports, tile list, heights entry
//!   tiles/{index}.bin   complete NTW tile = 8 chunks of [u32 csize][u32 raw = 32768][zlib]  (262144 raw DXT5 bytes)
//!   heights.bin     u32 LE run count, then [u32 offset][u32 len][len bytes] runs over the bottom-first raw pixel
//!                   stream (sorted, disjoint, maximal runs of pixels that differ from the original heightmap)
//! ```
//! The atlas keeps America's original source bounds/artwork and uses `<art_cache>/original-america/fixed_art/`
//! with an India-only manifest. Selection is bound into the cache key; the two layouts never reuse each other's repairs.
//!
//! The manifest `binding` is SHA-256 over the recipe version, the selected [`FIXED_BOUNDS`], the manifest hashes of the three ETW
//! source files and the fingerprint of the shipped public terrain data. A cache whose binding, layout or file hashes
//! do not match is rejected and regenerated from the snapshot (never used as a stale fallback); every tile is also
//! bound to the SHA-256 of its baseline uncompressed DXT5 tile and the height runs to the SHA-256 of the whole
//! original raw heightmap.

use super::art::{
    self, TheatreResult,
    dem::{self, Dem, Theatre},
    stpx::{self, TileSource},
};
use super::formats as F;
use crate::{
    model::Progress,
    regionmap::theatres::FIXED_BOUNDS,
    snapshot::{Game, Snapshot},
};
use anyhow::{Context, Result, bail, ensure};
use flate2::{Decompress, FlushDecompress, Status, read::ZlibDecoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Mutex,
};

/// Bump whenever a generator change alters the produced pixels; part of the cache binding.
const RECIPE_VERSION: u32 = 4;
const MANIFEST_FORMAT: u32 = 1;
const CACHE_DIR: &str = "fixed_art";

const INDIA_THEATRE: &str = "3";
const AMERICA_THEATRE: &str = "1";

/// World supertexture the repairs patch: 512px tiles over the 7-level 65536x32768 pyramid.
const SUPER_W: u32 = 65_536;
const SUPER_H: u32 = 32_768;
const SUPER_LEVELS: u32 = 7;
/// One 512x512 DXT5 tile: 16 bytes per 4x4 block = 1 byte per pixel.
const TILE_RAW: usize = 512 * 512;
/// NTW tiles store the raw tile as 8 zlib chunks of 0x8000 bytes (`[u32 csize][u32 raw][zlib]`).
const CHUNK_RAW: usize = 0x8000;
const TILE_CHUNKS: usize = TILE_RAW / CHUNK_RAW;
/// Original 8-bit 8192x4096 heightmap: raw pixel bytes without the 18-byte TGA header.
const HEIGHTMAP_LEN: usize = 8192 * 4096;

const SRC_STPI: &str = "world_.stpi";
const SRC_STPD: &str = "world_.stpd";
const SRC_HEIGHTMAP: &str = "heightmaps/default.tga";

// ------------------------------------------------------------------ public API

/// Validated artwork for the source theatres extended by the selected layout.
pub struct FixedArt {
    tiles: HashMap<u32, Tile>,
    heights_source: [u8; 32],
    /// `(offset, bytes)` ascending and disjoint.
    runs: Vec<(usize, Vec<u8>)>,
}

struct Tile {
    /// SHA-256 of the baseline tile's 262144 raw DXT5 bytes.
    source: [u8; 32],
    payload: Vec<u8>,
}

impl FixedArt {
    /// Open the cached artwork when it is bound to this snapshot and recipe, otherwise generate it natively from the
    /// snapshot's ETW supertexture and heightmap (several minutes, once per unchanged source) and publish it
    /// atomically below `art_cache`.
    pub fn load_or_generate(
        snapshot: &Snapshot,
        art_cache: &Path,
        extend_america: bool,
        progress: &Progress,
    ) -> Result<FixedArt> {
        let binding = binding(snapshot, extend_america)?;
        let original_america_cache = (!extend_america).then(|| art_cache.join("original-america"));
        let art_cache = original_america_cache.as_deref().unwrap_or(art_cache);
        let dir = art_cache.join(CACHE_DIR);
        if dir.join("manifest.json").is_file() {
            match load(&dir, &binding, extend_america) {
                Ok(art) => {
                    progress("Display: fixed theatre terrain taken from the local art cache");
                    return Ok(art);
                }
                Err(e) => progress(&format!(
                    "Display: local fixed theatre art cache rejected ({e:#}); regenerating from the ETW snapshot"
                )),
            }
        }
        progress(
            "Display: generating fixed theatre terrain from the ETW snapshot (once per unchanged source)",
        );
        let heights = original_heights(snapshot)?;
        let source = EtwTiles::open(snapshot)?;
        let india_dem = Dem::load(Theatre::India)?;
        let india = art::india::generate(&source, &heights, &india_dem, progress)
            .context("generate the fixed India theatre art")?;
        drop(india_dem);
        let mut parts = Vec::with_capacity(if extend_america { 2 } else { 1 });
        if extend_america {
            let america_dem = Dem::load(Theatre::America)?;
            let america = art::america::generate(&source, &heights, &america_dem, progress)
                .context("generate the fixed America theatre art")?;
            parts.push((AMERICA_THEATRE, america));
        }
        parts.push((INDIA_THEATRE, india));
        let shas = source_hashes(snapshot)?;
        publish(
            art_cache, &binding, &shas, &parts, &source, &heights, progress,
        )?;
        load(&dir, &binding, extend_america).context("verify the freshly generated fixed theatre art cache")
    }

    /// Replacement bytes for supertexture tile `index`, after proving that `original_ntw_tile` is the baseline tile
    /// the cache was generated from (SHA-256 streamed over the 262144 raw DXT5 bytes of its 8 chunks). `Ok(None)`
    /// when neither repair touches the tile. The caller writes the slice out verbatim.
    pub fn tile(&self, index: usize, original_ntw_tile: &[u8]) -> Result<Option<&[u8]>> {
        let Ok(key) = u32::try_from(index) else {
            return Ok(None);
        };
        let Some(tile) = self.tiles.get(&key) else {
            return Ok(None);
        };
        let digest = tile_digest(original_ntw_tile)
            .with_context(|| format!("fixed art: original supertexture tile {index}"))?;
        ensure!(
            digest == tile.source,
            "fixed art cache: supertexture tile {index} is not the baseline artwork the cache was generated from \
             (expected sha256 {}, found {})",
            hex::encode(tile.source),
            hex::encode(digest)
        );
        Ok(Some(&tile.payload))
    }

    /// Apply both theatres' height repairs to the original raw heightmap pixels (the bottom-first TGA pixel
    /// stream, 18-byte header already stripped). The whole buffer is bound to the baseline SHA-256 first; repairs
    /// apply exactly once, so the input must still be the original. Returns the number of pixels whose value
    /// changed; on any error the heightmap is returned untouched.
    pub fn apply_heights(&self, original_raw_pixels: &mut [u8]) -> Result<usize> {
        ensure!(
            original_raw_pixels.len() == HEIGHTMAP_LEN,
            "heightmap has {} raw pixels, expected {HEIGHTMAP_LEN} (8192x4096, 8-bit)",
            original_raw_pixels.len()
        );
        let digest: [u8; 32] = Sha256::digest(&original_raw_pixels[..]).into();
        ensure!(
            digest == self.heights_source,
            "fixed art cache: heightmap is not the original baseline (expected sha256 {}, found {})",
            hex::encode(self.heights_source),
            hex::encode(digest)
        );
        let mut changed = 0usize;
        for (offset, bytes) in &self.runs {
            for (dst, &src) in original_raw_pixels[*offset..offset + bytes.len()]
                .iter_mut()
                .zip(bytes)
            {
                if *dst != src {
                    *dst = src;
                    changed += 1;
                }
            }
        }
        Ok(changed)
    }
}

// ------------------------------------------------------------------ binding / manifest

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: u32,
    recipe_version: u32,
    binding: String,
    /// Manifest SHA-256 of the three ETW snapshot files the art was generated from.
    sources: SourceHashes,
    /// SHA-256 of the whole original raw heightmap pixel stream.
    heightmap_source_sha256: String,
    theatres: Vec<TheatreEntry>,
    tiles: Vec<TileEntry>,
    heights: HeightsEntry,
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SourceHashes {
    world_stpi: String,
    world_stpd: String,
    heightmap: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TheatreEntry {
    key: String,
    bounds: [[f32; 2]; 2],
    tiles: usize,
    height_pixels: usize,
    /// Numbers the native generator measured about its own result.
    report: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TileEntry {
    index: u32,
    source_sha256: String,
    file_sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeightsEntry {
    file_sha256: String,
    runs: usize,
    pixels: usize,
}

fn source_hashes(snapshot: &Snapshot) -> Result<SourceHashes> {
    let sha = |p: &str| -> Result<String> { Ok(snapshot.entry(Game::Etw, p)?.sha256.clone()) };
    Ok(SourceHashes {
        world_stpi: sha(SRC_STPI)?,
        world_stpd: sha(SRC_STPD)?,
        heightmap: sha(SRC_HEIGHTMAP)?,
    })
}

/// SHA-256 over everything the generated art depends on besides the code: recipe version, fixed bounds, the three
/// ETW source file hashes and the shipped public terrain data.
fn binding(snapshot: &Snapshot, extend_america: bool) -> Result<[u8; 32]> {
    let shas = source_hashes(snapshot)?;
    Ok(binding_of(&shas, &dem::recipe_fingerprint(), extend_america))
}

fn binding_of(shas: &SourceHashes, terrain_data: &[u8; 32], extend_america: bool) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"kgc-fixed-art-binding\0");
    h.update(RECIPE_VERSION.to_le_bytes());
    for (key, rect) in FIXED_BOUNDS {
        if !extend_america && *key == AMERICA_THEATRE {
            continue;
        }
        h.update(key.as_bytes());
        h.update([0]);
        for coordinate in rect.iter().flatten() {
            h.update(coordinate.to_le_bytes());
        }
    }
    for s in [&shas.world_stpi, &shas.world_stpd, &shas.heightmap] {
        h.update(s.as_bytes());
        h.update([0]);
    }
    h.update(terrain_data);
    h.finalize().into()
}

fn sha256_field(value: &str, what: &str) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(value, &mut out)
        .with_context(|| format!("{what} is not a SHA-256 hex digest"))?;
    Ok(out)
}

// ------------------------------------------------------------------ load

fn load(dir: &Path, binding: &[u8; 32], extend_america: bool) -> Result<FixedArt> {
    let manifest_path = dir.join("manifest.json");
    let manifest: Manifest = serde_json::from_slice(
        &fs::read(&manifest_path).with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .context("parse fixed art manifest")?;
    ensure!(
        manifest.format == MANIFEST_FORMAT && manifest.recipe_version == RECIPE_VERSION,
        "fixed art cache format {}/recipe {} is not the current {MANIFEST_FORMAT}/{RECIPE_VERSION}",
        manifest.format,
        manifest.recipe_version
    );
    ensure!(
        sha256_field(&manifest.binding, "manifest binding")? == *binding,
        "fixed art cache is bound to a different ETW source or recipe"
    );
    let expected: &[&str] = if extend_america { &[AMERICA_THEATRE, INDIA_THEATRE] } else { &[INDIA_THEATRE] };
    ensure!(
        manifest.theatres.iter().map(|t| t.key.as_str()).eq(expected.iter().copied()),
        "fixed art manifest does not match the selected source theatre extensions {expected:?}"
    );
    for t in &manifest.theatres {
        let (_, bounds) = FIXED_BOUNDS
            .iter()
            .find(|(key, _)| *key == t.key)
            .with_context(|| format!("no fixed bounds for theatre {:?}", t.key))?;
        ensure!(
            t.bounds == *bounds,
            "fixed art theatre {} was generated for bounds {:?}, the fixed bounds are {:?}",
            t.key,
            t.bounds,
            bounds
        );
    }
    let heights_source = sha256_field(&manifest.heightmap_source_sha256, "heightmap source hash")?;

    let tile_total = tile_count(SUPER_W, SUPER_H, SUPER_LEVELS);
    let mut tiles = HashMap::with_capacity(manifest.tiles.len());
    for entry in &manifest.tiles {
        ensure!(
            (entry.index as u64) < tile_total,
            "fixed art tile index {} is outside the {tile_total}-tile supertexture",
            entry.index
        );
        let path = dir.join("tiles").join(format!("{}.bin", entry.index));
        let payload = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            Sha256::digest(&payload)[..] == sha256_field(&entry.file_sha256, "tile file hash")?[..],
            "fixed art tile file {} does not match its manifest hash",
            path.display()
        );
        validate_tile(&payload).with_context(|| format!("fixed art tile {}", entry.index))?;
        let source = sha256_field(&entry.source_sha256, "tile source hash")?;
        ensure!(
            tiles
                .insert(entry.index, Tile { source, payload })
                .is_none(),
            "fixed art tile {} is listed twice",
            entry.index
        );
    }
    ensure!(
        manifest.theatres.iter().map(|t| t.tiles).sum::<usize>() == tiles.len(),
        "fixed art manifest tile counts disagree with its tile list"
    );

    let heights_path = dir.join("heights.bin");
    let data =
        fs::read(&heights_path).with_context(|| format!("read {}", heights_path.display()))?;
    ensure!(
        Sha256::digest(&data)[..]
            == sha256_field(&manifest.heights.file_sha256, "heights file hash")?[..],
        "fixed art heights.bin does not match its manifest hash"
    );
    let layout = parse_runs(&data)?;
    ensure!(
        layout.len() == manifest.heights.runs,
        "fixed art manifest run count disagrees with heights.bin"
    );
    ensure!(
        layout.iter().map(|r| r.1).sum::<usize>() == manifest.heights.pixels,
        "fixed art manifest pixel count disagrees with heights.bin"
    );
    let mut runs = Vec::with_capacity(layout.len());
    let mut p = 4usize;
    for (offset, len) in layout {
        runs.push((offset, data[p + 8..p + 8 + len].to_vec()));
        p += 8 + len;
    }
    Ok(FixedArt {
        tiles,
        heights_source,
        runs,
    })
}

/// Total tiles of the level pyramid (row-major levels concatenated: 128x64 + 64x32 + ... = 10922).
fn tile_count(width: u32, height: u32, levels: u32) -> u64 {
    let (mut tx, mut ty) = (width / 512, height / 512);
    let mut total = 0u64;
    for _ in 0..levels {
        total += tx as u64 * ty as u64;
        tx /= 2;
        ty /= 2;
    }
    total
}

// ------------------------------------------------------------------ generation / publication

/// Tiles of the ETW world supertexture (one zlib stream of a DDS per stored tile) as raw DXT5 baseline tiles.
struct EtwTiles {
    /// `(offset, compressed size)` of every tile, levels concatenated row-major.
    entries: Vec<(u32, u32)>,
    stpd_len: u64,
    file: Mutex<fs::File>,
}

impl EtwTiles {
    fn open(snapshot: &Snapshot) -> Result<EtwTiles> {
        let stpi = F::parse_stpi(&snapshot.read(Game::Etw, SRC_STPI)?)?;
        ensure!(
            stpi.tile == 512 && stpi.tile_bytes == 0x40080,
            "Unexpected ETW supertexture tile layout"
        );
        ensure!(
            (stpi.width, stpi.height, stpi.levels) == (SUPER_W, SUPER_H, SUPER_LEVELS)
                && stpi.layers.iter().enumerate().all(|(level, layer)| {
                    layer.tx == (128 >> level) && layer.ty == (64 >> level)
                }),
            "Fixed theatre artwork requires the original seven-level world supertexture layout"
        );
        let path = snapshot.path(Game::Etw, SRC_STPD)?;
        let file = fs::File::open(&path).with_context(|| format!("Open {}", path.display()))?;
        let stpd_len = file.metadata()?.len();
        let entries: Vec<(u32, u32)> = stpi
            .layers
            .iter()
            .flat_map(|l| l.entries.iter().map(|e| (e[0], e[1])))
            .collect();
        ensure!(
            entries.len() as u64 == tile_count(SUPER_W, SUPER_H, SUPER_LEVELS),
            "ETW supertexture tile count"
        );
        Ok(EtwTiles {
            entries,
            stpd_len,
            file: Mutex::new(file),
        })
    }
}

impl TileSource for EtwTiles {
    fn raw_tile(&self, index: usize) -> Result<Vec<u8>> {
        let &(off, size) = self
            .entries
            .get(index)
            .with_context(|| format!("supertexture tile {index} does not exist"))?;
        ensure!(
            off as u64 + size as u64 <= self.stpd_len,
            "Supertexture tile {index} outside world_.stpd"
        );
        let mut z = vec![0u8; size as usize];
        {
            let mut f = self
                .file
                .lock()
                .map_err(|_| anyhow::anyhow!("supertexture file lock poisoned"))?;
            f.seek(SeekFrom::Start(off as u64))?;
            f.read_exact(&mut z)?;
        }
        let mut raw = Vec::with_capacity(TILE_RAW + 128);
        ZlibDecoder::new(&z[..])
            .read_to_end(&mut raw)
            .with_context(|| format!("Inflate ETW supertexture tile {index}"))?;
        ensure!(
            raw.len() == TILE_RAW + 128 && &raw[0..4] == b"DDS " && &raw[84..88] == b"DXT5",
            "ETW supertexture tile {index} is not a 512x512 DXT5 DDS (inflated {} bytes)",
            raw.len()
        );
        raw.drain(..128);
        Ok(raw)
    }
}

/// The original raw heightmap pixels (bottom-first, header stripped) with the format checks of the build.
fn original_heights(snapshot: &Snapshot) -> Result<Vec<u8>> {
    let mut b = snapshot.read(Game::Etw, SRC_HEIGHTMAP)?;
    let h = F::tga_header(&b)?;
    ensure!(
        (h.width, h.height, h.bpp, h.image_type) == (8192, 4096, 8, 3),
        "Unexpected ETW heightmap format"
    );
    ensure!(
        b.len() == 18 + HEIGHTMAP_LEN,
        "Unexpected ETW heightmap size"
    );
    ensure!(
        h.id_len == 0 && h.desc & 0x30 == 0,
        "Fixed terrain artwork requires the original bottom-left heightmap"
    );
    b.drain(..18);
    Ok(b)
}

/// Run `f` over `items` on all cores, preserving order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> Result<R> + Sync) -> Result<Vec<R>> {
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

/// Maximal runs `(offset, len)` of pixels where `new` differs from `base`.
fn diff_runs(base: &[u8], new: &[u8]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    for (i, (a, b)) in base.iter().zip(new).enumerate() {
        match (a != b, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                runs.push((s, i - s));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push((s, base.len() - s));
    }
    runs
}

/// Encode the generated theatres into a fresh cache directory and publish it atomically.
fn publish(
    art_cache: &Path,
    binding: &[u8; 32],
    shas: &SourceHashes,
    parts: &[(&str, TheatreResult)],
    source: &dyn TileSource,
    original_heights: &[u8],
    progress: &Progress,
) -> Result<()> {
    ensure!(
        original_heights.len() == HEIGHTMAP_LEN,
        "original heightmap has the wrong size"
    );
    // tiles: each theatre's set, proven disjoint
    let mut owner: BTreeMap<usize, usize> = BTreeMap::new();
    for (pi, (key, part)) in parts.iter().enumerate() {
        for &index in part.tiles.keys() {
            if let Some(prev) = owner.insert(index, pi) {
                bail!(
                    "supertexture tile {index} is repaired by both theatre {} and theatre {key}; each theatre must \
                     stay inside its own half of the world",
                    parts[prev].0
                );
            }
        }
    }
    // heights: each theatre's differing runs, proven disjoint, then merged
    let mut merged = original_heights.to_vec();
    let mut all: Vec<Vec<(usize, usize)>> = Vec::new();
    for (key, part) in parts {
        ensure!(
            part.heights.len() == HEIGHTMAP_LEN,
            "theatre {key} produced a heightmap of {} pixels",
            part.heights.len()
        );
        all.push(diff_runs(original_heights, &part.heights));
    }
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            ensure_runs_disjoint(parts[i].0, &all[i], parts[j].0, &all[j])?;
        }
    }
    for (pi, runs) in all.iter().enumerate() {
        for &(offset, len) in runs {
            merged[offset..offset + len]
                .copy_from_slice(&parts[pi].1.heights[offset..offset + len]);
        }
    }
    let runs = diff_runs(original_heights, &merged);

    let tmp = art_cache.join(format!("{CACHE_DIR}.tmp-{}", std::process::id()));
    if tmp.exists() {
        fs::remove_dir_all(&tmp).with_context(|| format!("clear {}", tmp.display()))?;
    }
    fs::create_dir_all(tmp.join("tiles")).with_context(|| format!("create {}", tmp.display()))?;
    let result = (|| -> Result<()> {
        let indexed: Vec<(&usize, &Vec<u8>)> =
            parts.iter().flat_map(|(_, p)| p.tiles.iter()).collect();
        progress(&format!(
            "Display: encoding {} repaired supertexture tiles",
            indexed.len()
        ));
        let encoded = par_map(
            &indexed,
            |&(&index, raw)| -> Result<(u32, [u8; 32], Vec<u8>)> {
                ensure!(
                    raw.len() == TILE_RAW,
                    "generated tile {index} has {} raw bytes",
                    raw.len()
                );
                let baseline = source.raw_tile(index)?;
                let src: [u8; 32] = Sha256::digest(&baseline).into();
                let payload = stpx::encode_ntw_tile(raw)?;
                Ok((u32::try_from(index)?, src, payload))
            },
        )?;
        let mut entries = Vec::with_capacity(encoded.len());
        for (index, src, payload) in &encoded {
            fs::write(tmp.join("tiles").join(format!("{index}.bin")), payload)
                .with_context(|| format!("write tile {index}"))?;
            entries.push(TileEntry {
                index: *index,
                source_sha256: hex::encode(src),
                file_sha256: hex::encode(Sha256::digest(payload)),
            });
        }
        entries.sort_by_key(|e| e.index);
        let mut data = Vec::new();
        data.extend_from_slice(&(runs.len() as u32).to_le_bytes());
        for &(offset, len) in &runs {
            data.extend_from_slice(&(offset as u32).to_le_bytes());
            data.extend_from_slice(&(len as u32).to_le_bytes());
            data.extend_from_slice(&merged[offset..offset + len]);
        }
        fs::write(tmp.join("heights.bin"), &data).context("write heights.bin")?;
        let manifest = Manifest {
            format: MANIFEST_FORMAT,
            recipe_version: RECIPE_VERSION,
            binding: hex::encode(binding),
            sources: SourceHashes {
                world_stpi: shas.world_stpi.clone(),
                world_stpd: shas.world_stpd.clone(),
                heightmap: shas.heightmap.clone(),
            },
            heightmap_source_sha256: hex::encode(Sha256::digest(original_heights)),
            theatres: parts
                .iter()
                .enumerate()
                .map(|(pi, (key, part))| -> Result<TheatreEntry> {
                    let (_, bounds) = FIXED_BOUNDS
                        .iter()
                        .find(|(k, _)| k == key)
                        .with_context(|| format!("no fixed bounds for theatre {key:?}"))?;
                    Ok(TheatreEntry {
                        key: key.to_string(),
                        bounds: *bounds,
                        tiles: part.tiles.len(),
                        height_pixels: all[pi].iter().map(|r| r.1).sum(),
                        report: part.report.clone(),
                    })
                })
                .collect::<Result<_>>()?,
            tiles: entries,
            heights: HeightsEntry {
                file_sha256: hex::encode(Sha256::digest(&data)),
                runs: runs.len(),
                pixels: runs.iter().map(|r| r.1).sum(),
            },
        };
        fs::write(
            tmp.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )
        .context("write manifest")?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = fs::remove_dir_all(&tmp);
        return Err(e);
    }
    let dir = art_cache.join(CACHE_DIR);
    let old = art_cache.join(format!("{CACHE_DIR}.old-{}", std::process::id()));
    if dir.exists() {
        if old.exists() {
            fs::remove_dir_all(&old).ok();
        }
        fs::rename(&dir, &old).with_context(|| format!("replace {}", dir.display()))?;
    }
    if let Err(e) = fs::rename(&tmp, &dir) {
        // a concurrent run may have published the same cache first
        if load(&dir, binding, parts.iter().any(|(key, _)| *key == AMERICA_THEATRE)).is_ok() {
            let _ = fs::remove_dir_all(&tmp);
        } else {
            return Err(e).with_context(|| format!("publish {}", dir.display()));
        }
    }
    if old.exists() {
        let _ = fs::remove_dir_all(&old);
    }
    progress("Display: fixed theatre art cache published");
    Ok(())
}

// ------------------------------------------------------------------ NTW tiles

/// Walk the NTW tile layout (8 chunks of `[u32 csize][u32 raw][zlib]`), inflating each chunk into
/// a reused buffer and passing its raw bytes to `sink`. Rejects any other chunk count, raw length
/// or zlib stream.
fn scan_tile(tile: &[u8], mut sink: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let mut buf = [0u8; CHUNK_RAW];
    let mut decoder = Decompress::new(true);
    let (mut p, mut chunks, mut total) = (0usize, 0usize, 0usize);
    while p < tile.len() {
        let csize = F::u32_at(tile, p)? as usize;
        let declared = F::u32_at(tile, p + 4)? as usize;
        let data = tile
            .get(p + 8..)
            .and_then(|rest| rest.get(..csize))
            .with_context(|| format!("supertexture tile chunk {chunks} is truncated"))?;
        ensure!(
            declared == CHUNK_RAW,
            "supertexture tile chunk {chunks} holds {declared} raw bytes, expected {CHUNK_RAW}"
        );
        if chunks > 0 {
            decoder.reset(true);
        }
        inflate_chunk(&mut decoder, data, &mut buf)
            .with_context(|| format!("supertexture tile chunk {chunks}"))?;
        sink(&buf)?;
        p += 8 + csize;
        chunks += 1;
        total += declared;
    }
    ensure!(
        chunks == TILE_CHUNKS && total == TILE_RAW,
        "supertexture tile has {chunks} chunks inflating to {total} bytes, expected \
         {TILE_CHUNKS} chunks of {CHUNK_RAW}"
    );
    Ok(())
}

/// Require one complete zlib stream, consuming exactly its declared chunk and producing 32 KiB.
fn inflate_chunk(decoder: &mut Decompress, data: &[u8], buf: &mut [u8]) -> Result<()> {
    let status = decoder
        .decompress(data, buf, FlushDecompress::Finish)
        .context("inflate supertexture tile chunk")?;
    ensure!(
        status == Status::StreamEnd
            && decoder.total_in() == data.len() as u64
            && decoder.total_out() == buf.len() as u64,
        "supertexture tile chunk must contain one complete zlib stream yielding {} bytes",
        buf.len()
    );
    Ok(())
}

/// Validate a stored replacement payload (layout and every zlib stream) without copying it.
fn validate_tile(tile: &[u8]) -> Result<()> {
    scan_tile(tile, |_| Ok(()))
}

/// SHA-256 of a tile's 262144 raw DXT5 bytes, streamed chunk by chunk.
fn tile_digest(tile: &[u8]) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    scan_tile(tile, |raw| {
        hasher.update(raw);
        Ok(())
    })?;
    Ok(hasher.finalize().into())
}

// ------------------------------------------------------------------ height runs

/// Parse and fully validate the run payload of `heights.bin`: ascending disjoint offsets,
/// positive lengths inside the heightmap, exact payload consumption. Returns `(offset, len)` pairs.
fn parse_runs(data: &[u8]) -> Result<Vec<(usize, usize)>> {
    ensure!(data.len() >= 4, "heights.bin is truncated (no run count)");
    let count = F::u32_at(data, 0)? as u64;
    ensure!(
        data.len() as u64 >= 4 + count * 8,
        "heights.bin declares {count} runs but holds only {} bytes",
        data.len()
    );
    let mut runs = Vec::with_capacity(count as usize);
    let (mut p, mut prev_end) = (4usize, 0usize);
    for i in 0..count as usize {
        let offset = F::u32_at(data, p)? as usize;
        let len = F::u32_at(data, p + 4)? as usize;
        ensure!(len > 0, "height run {i} is empty");
        ensure!(
            offset >= prev_end,
            "height run {i} at {offset} precedes or overlaps the previous run (ends at {prev_end})"
        );
        let end = offset as u64 + len as u64;
        ensure!(
            end <= HEIGHTMAP_LEN as u64,
            "height run {i} reaches pixel {end}, past the {HEIGHTMAP_LEN}-pixel heightmap"
        );
        ensure!(
            p + 8 + len <= data.len(),
            "height run {i} payload is truncated"
        );
        p += 8 + len;
        prev_end = end as usize;
        runs.push((offset, len));
    }
    ensure!(
        p == data.len(),
        "heights.bin has {} trailing bytes after the last run",
        data.len() - p
    );
    Ok(runs)
}

/// Both theatres repair disjoint halves of the world; runs touching the same pixel are a
/// generation mistake and must not depend on application order.
fn ensure_runs_disjoint(
    left_label: &str,
    left: &[(usize, usize)],
    right_label: &str,
    right: &[(usize, usize)],
) -> Result<()> {
    let (mut i, mut j) = (0usize, 0usize);
    while i < left.len() && j < right.len() {
        let (l0, l1) = (left[i].0, left[i].0 + left[i].1);
        let (r0, r1) = (right[j].0, right[j].0 + right[j].1);
        ensure!(
            l1 <= r0 || r1 <= l0,
            "height runs overlap between theatre {left_label} and theatre {right_label} at pixels {}..{}; each \
             theatre's repair must stay inside its own bounds",
            l0.max(r0),
            l1.min(r1)
        );
        if l1 <= r1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Deterministic fake baseline tiles; tile `i` is filled with a pattern depending on `i`.
    struct FakeTiles;

    fn fake_raw(index: usize) -> Vec<u8> {
        (0..TILE_RAW)
            .map(|k| ((k * 31 + index * 7) % 251) as u8)
            .collect()
    }

    impl TileSource for FakeTiles {
        fn raw_tile(&self, index: usize) -> Result<Vec<u8>> {
            Ok(fake_raw(index))
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let p =
            std::env::temp_dir().join(format!("kgc-fixed-art-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn shas() -> SourceHashes {
        SourceHashes {
            world_stpi: "a".repeat(64),
            world_stpd: "b".repeat(64),
            heightmap: "c".repeat(64),
        }
    }

    fn heights() -> Vec<u8> {
        (0..HEIGHTMAP_LEN).map(|i| (i % 200) as u8).collect()
    }

    fn part(tiles: &[usize], pixels: std::ops::Range<usize>, base: &[u8]) -> TheatreResult {
        let mut h = base.to_vec();
        for i in pixels {
            h[i] = h[i].wrapping_add(1);
        }
        TheatreResult {
            tiles: tiles.iter().map(|&t| (t, fake_raw(t + 1))).collect(),
            heights: h,
            report: serde_json::json!({ "tiles": tiles.len() }),
        }
    }

    fn quiet(_: &str) {}

    #[test]
    fn publish_load_roundtrip_binds_tiles_and_heights() {
        let root = scratch("roundtrip");
        let base = heights();
        let parts = [
            ("1", part(&[10, 11], 1000..1010, &base)),
            ("3", part(&[20], 5000..5003, &base)),
        ];
        let binding = binding_of(&shas(), &[7u8; 32], true);
        publish(&root, &binding, &shas(), &parts, &FakeTiles, &base, &quiet).unwrap();
        let art = load(&root.join(CACHE_DIR), &binding, true).unwrap();
        let original = stpx::encode_ntw_tile(&fake_raw(10)).unwrap();
        let replacement = art.tile(10, &original).unwrap().unwrap();
        assert_eq!(
            replacement,
            &stpx::encode_ntw_tile(&fake_raw(11)).unwrap()[..]
        );
        assert!(art.tile(12, &original).unwrap().is_none());
        // a different baseline tile is refused
        let other = stpx::encode_ntw_tile(&fake_raw(99)).unwrap();
        assert!(art.tile(10, &other).is_err());
        let mut raw = base.clone();
        assert_eq!(art.apply_heights(&mut raw).unwrap(), 13);
        assert_eq!(raw[1000], base[1000].wrapping_add(1));
        assert_eq!(raw[5002], base[5002].wrapping_add(1));
        assert_eq!(raw[999], base[999]);
        // repairs apply once: the already-repaired buffer is no longer the baseline
        assert!(art.apply_heights(&mut raw).is_err());
        // another binding (different source or recipe) is rejected
        assert!(load(&root.join(CACHE_DIR), &binding_of(&shas(), &[8u8; 32], true), true).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn original_america_layout_keeps_its_pixels_and_refuses_extended_layout_caches() {
        let root = scratch("original-america");
        let base = heights();
        let binding = binding_of(&shas(), &[7u8; 32], false);
        let parts = [(INDIA_THEATRE, part(&[20], 5000..5003, &base))];
        publish(&root, &binding, &shas(), &parts, &FakeTiles, &base, &quiet).unwrap();
        let dir = root.join(CACHE_DIR);
        let art = load(&dir, &binding, false).unwrap();
        let original = stpx::encode_ntw_tile(&fake_raw(10)).unwrap();
        assert!(art.tile(10, &original).unwrap().is_none());
        let mut raw = base.clone();
        assert_eq!(art.apply_heights(&mut raw).unwrap(), 3);
        assert_eq!(raw[1000], base[1000]);
        assert_eq!(raw[5002], base[5002].wrapping_add(1));
        assert!(load(&dir, &binding, true).is_err());
        assert!(load(&dir, &binding_of(&shas(), &[7u8; 32], true), false).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn overlapping_tiles_or_heights_are_refused() {
        let root = scratch("overlap");
        let base = heights();
        let binding = binding_of(&shas(), &[7u8; 32], true);
        let tiles_overlap = [
            ("1", part(&[10], 1000..1010, &base)),
            ("3", part(&[10], 5000..5003, &base)),
        ];
        let err = publish(
            &root,
            &binding,
            &shas(),
            &tiles_overlap,
            &FakeTiles,
            &base,
            &quiet,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("repaired by both"), "{err:#}");
        let heights_overlap = [
            ("1", part(&[10], 1000..1010, &base)),
            ("3", part(&[20], 1005..1020, &base)),
        ];
        let err = publish(
            &root,
            &binding,
            &shas(),
            &heights_overlap,
            &FakeTiles,
            &base,
            &quiet,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("overlap"), "{err:#}");
        assert!(!root.join(CACHE_DIR).exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn corrupt_or_partial_caches_are_rejected() {
        let root = scratch("corrupt");
        let base = heights();
        let parts = [
            ("1", part(&[10], 1000..1010, &base)),
            ("3", part(&[20], 5000..5003, &base)),
        ];
        let binding = binding_of(&shas(), &[7u8; 32], true);
        publish(&root, &binding, &shas(), &parts, &FakeTiles, &base, &quiet).unwrap();
        let dir = root.join(CACHE_DIR);
        assert!(load(&dir, &binding, true).is_ok());
        let tile = dir.join("tiles/10.bin");
        let good = fs::read(&tile).unwrap();
        let mut bad = good.clone();
        let n = bad.len();
        bad[n - 5] ^= 0x55;
        fs::write(&tile, &bad).unwrap();
        assert!(load(&dir, &binding, true).is_err());
        fs::write(&tile, &good).unwrap();
        fs::remove_file(dir.join("tiles/20.bin")).unwrap();
        assert!(load(&dir, &binding, true).is_err());
        fs::write(
            dir.join("tiles/20.bin"),
            fs::read(dir.join("tiles/10.bin")).unwrap(),
        )
        .unwrap();
        assert!(load(&dir, &binding, true).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn diff_runs_are_maximal_and_sorted() {
        let a = [0u8, 0, 0, 0, 0, 0, 0];
        let b = [0u8, 1, 1, 0, 0, 2, 2];
        assert_eq!(diff_runs(&a, &b), vec![(1, 2), (5, 2)]);
        assert!(diff_runs(&a, &a).is_empty());
    }

    #[test]
    fn height_runs_parse_strictly() {
        let mut data = 1u32.to_le_bytes().to_vec();
        data.extend_from_slice(&5u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&[9, 9]);
        assert_eq!(parse_runs(&data).unwrap(), vec![(5, 2)]);
        let mut trailing = data.clone();
        trailing.push(0);
        assert!(parse_runs(&trailing).is_err());
        let truncated = &data[..data.len() - 1];
        assert!(parse_runs(truncated).is_err());
    }
}
