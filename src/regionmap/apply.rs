//! Applies the project's region splits to the converted world map (`<data>/campaign_maps/nap_world`).
//!
//! The map builder copies the ETW `global_map` geometry verbatim (ETW coordinates, ring vertex orders), so a split
//! recorded against the ETW rings applies unchanged. For every split, in project order, this module updates
//!  * `regions.esf` (geometry, settlement, theatre region list, query index),
//!  * `pathfinding.esf` (pieces / empty cells of the theatre grid that contain the new settlement),
//!  * `poi.esf` (re-tagged points, neighbour lists, new settlement point),
//!  * `trade_routes.esf` (new settlement node, direct links to the neighbours' settlements),
//!  * `sea_grids.esf` (land lists of the coastal cells),
//!  * the theatre's lookup TGA (new palette colour, pixels of the new region),
//!  * `display/borders` (border splines of the source area and of the new region),
//! and appends the settlement DB row, the localisation rows and the slot mapping of the new region to the map
//! artifacts. Nothing is written unless every split succeeded.
//!
//! [`apply_activations`] runs first: it turns ETW wilderness polygons (land regions without settlement) into playable
//! regions in the same files (settlement from a donor region, POI points, trade node, lookup colour, border splines),
//! after which a wilderness key is an ordinary region that a split may cut.
//!
//! Measured on the ETW `global_map` (see `examples/apply_verify.rs`):
//!  * lookup TGA: the theatre rectangle of `regions.esf` `theatres` (europe: (-180,140)..(440,520)) is stretched
//!    over the whole image (490x300, 1.265 world units per pixel in both directions); pixel row 0 is the southern
//!    edge for a bottom-origin TGA (descriptor bit 5 clear);
//!  * border splines: `x = world * 0.0254`, `y = -0.000254`, `z = world_y * 0.0254` (one world unit is one inch,
//!    the splines are in metres); each file holds one closed loop; a region with one loop is `<key>.rigid_spline`,
//!    with several `<key>_1 .. <key>_N`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::{Value, json};

use super::geometry::{P, point_segment_distance, signed_area2};
use super::pathfinding::Grid;
use super::regions::{NONE, Outline, Region, RegionData, SettlementSpec, Split, add_theatre_region_key, settlement_center};
use super::{poi, sea_grids, trade_routes};
use crate::esf::{self, Node, Value as EsfValue};
use crate::map::MapArtifacts;
use crate::model::{RegionActivation, RegionSplit, Rules};
use crate::terrain::formats::{Spline, parse_spln, write_spln};

/// Border spline units per world unit (the ETW splines are in metres, the world in inches).
pub const BORDER_UNITS_PER_WORLD: f64 = 0.0254;
/// Constant height of every ETW border spline point.
const BORDER_Y: f32 = -0.000254;
/// Resampling step of a border ring, in world units.
const BORDER_STEP: f64 = 0.3;
/// A border spline belongs to a region area when this share of its sampled points lies within
/// [`BORDER_MATCH_DISTANCE`] world units of the area's outer ring.
const BORDER_MATCH_SHARE: f64 = 0.9;
const BORDER_MATCH_DISTANCE: f64 = 2.0;
/// Share of the source colour's lookup pixels that must lie on the source polygons (within 1.5 pixels).
const LOOKUP_CHECK_SHARE: f64 = 0.95;

/// What one split changed, for the start position builder.
#[derive(Clone, Debug)]
pub struct AppliedSplit {
    pub key: String,
    pub source: String,
    /// index of the new region in `regions.esf`
    pub new_index: usize,
    /// `settlement:<key>:<settlement>`
    pub settlement_key: String,
    /// settlement slot keys of the new region, in `slot_descriptions` order
    pub slot_keys: Vec<String>,
    /// world position (x, y) of each slot
    pub slot_positions: Vec<[f32; 2]>,
    pub center: [f32; 2],
    /// keys of the regions sharing a border with the new region (includes the source), by `regions.esf` index
    pub neighbours: Vec<String>,
    /// polygon border length of the new region per neighbour
    pub border_lengths: BTreeMap<String, f64>,
    /// polygon border length of the source per neighbour before / after this split
    pub source_border_lengths_before: BTreeMap<String, f64>,
    pub source_border_lengths_after: BTreeMap<String, f64>,
    /// regions.esf theatre key that lists the region
    pub theatre: String,
}

/// Lookup TGA repaint result.
#[derive(Clone, Debug, Default)]
pub struct PaintReport {
    pub painted: usize,
    pub palette_index: usize,
    /// pixels of the source colour that were checked against the source polygons / how many were on them
    pub checked: usize,
    pub on_polygon: usize,
}

/// Per-split statistics.
#[derive(Clone, Debug)]
pub struct SplitStats {
    pub key: String,
    pub new_index: usize,
    pub pathfinding_area: usize,
    pub pieces_relabelled: usize,
    pub empty_cells_relabelled: usize,
    pub poi: poi::SplitReport,
    pub trade_node: u32,
    pub trade_routes: usize,
    pub lookup_file: String,
    pub lookup: PaintReport,
    pub border_files_written: Vec<String>,
    pub border_files_removed: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ApplyStats {
    pub splits: Vec<SplitStats>,
    pub query_cells: usize,
    pub sea_cells_changed: usize,
}

/// Applies every split to the converted map files under `map_dir` and appends the new regions' rows to `map`.
pub fn apply_splits(
    map_dir: &Path,
    splits: &[RegionSplit],
    rules: &Rules,
    map: &mut MapArtifacts,
) -> Result<Vec<AppliedSplit>> {
    Ok(apply_splits_with_stats(map_dir, splits, rules, map)?.0)
}

// ------------------------------------------------------------------------------------------------ regions.esf helpers

struct Theatre {
    key: String,
    lo: [f32; 2],
    hi: [f32; 2],
}

/// `theatres` of regions.esf: items hold the scalars [key s16, min XY, max XY, bool] (directly or inside one record).
fn theatres(root: &Node) -> Result<Vec<Theatre>> {
    let mut out = Vec::new();
    for item in root.get("theatres")?.items()? {
        let ch: &[Node] = if let [Node::Record { children, .. }] = item.as_slice() { children } else { item };
        let v: Vec<&Node> = ch.iter().filter(|n| matches!(n, Node::Value(_))).collect();
        ensure!(v.len() >= 3, "theatre record has fewer than 3 scalars");
        out.push(Theatre {
            key: v[0].as_str()?.to_string(),
            lo: v[1].as_xy()?,
            hi: v[2].as_xy()?,
        });
    }
    Ok(out)
}

/// Key of the theatre whose `theatres_and_region_keys` entry lists `region`.
fn theatre_listing(root: &Node, region: &str) -> Result<String> {
    for item in root.get("theatres_and_region_keys")?.items()? {
        let theatre = &item[0];
        let lists = theatre
            .get("region_keys")?
            .items()?
            .iter()
            .any(|it| it[0].as_str().ok() == Some(region));
        if lists {
            let key = theatre
                .children()?
                .iter()
                .find_map(|c| match c {
                    Node::Value(EsfValue::String16(s)) | Node::Value(EsfValue::String8(s)) => Some(s.clone()),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("theatre entry listing {region} has no key string"))?;
            return Ok(key);
        }
    }
    bail!("no theatre lists region {region}")
}

fn ring_points(rd: &RegionData, ring: &[u32]) -> Vec<P> {
    ring.iter().map(|&i| rd.v(i)).collect()
}

fn outer_ring(rd: &RegionData, ri: usize, area: usize) -> Result<&Vec<u32>> {
    let a = rd.regions[ri]
        .areas
        .get(area)
        .ok_or_else(|| anyhow!("{}: no area {area}", rd.regions[ri].key))?;
    Ok(&a
        .outlines
        .iter()
        .find(|o| o.outer)
        .ok_or_else(|| anyhow!("{}: area {area} has no outer ring", rd.regions[ri].key))?
        .ring)
}

fn bbox_of(ring: &[P]) -> (P, P) {
    let mut lo = [f64::MAX; 2];
    let mut hi = [f64::MIN; 2];
    for p in ring {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

fn named_borders(rd: &RegionData, ri: usize) -> BTreeMap<String, f64> {
    rd.border_lengths(ri)
        .into_iter()
        .map(|(i, l)| (rd.regions[i].key.clone(), l))
        .collect()
}

fn xz_to_world(p: [f32; 3]) -> P {
    [p[0] as f64 / BORDER_UNITS_PER_WORLD, p[2] as f64 / BORDER_UNITS_PER_WORLD]
}

// ------------------------------------------------------------------------------------------------ lookup TGA

/// Repaints a palettised region lookup map in place: gives `colour` a free palette entry (the unused index directly
/// below the first used one, index 0 is left alone; it is an error when `colour` is already in the palette) and sets
/// every pixel whose world-space centre lies inside region `new_index` of `rd` to it. `rect` is the world rectangle
/// `(x0, y0, x1, y1)` that the image covers. With `source = Some((index, rgb))` only pixels that currently carry the
/// source region's colour are repainted, after checking that at least [`LOOKUP_CHECK_SHARE`] of them lie on the
/// source / new region polygons (this proves the pixel-world mapping). Everything else (header, palette order, other
/// pixels, footer) is kept byte for byte.
pub fn paint_lookup_tga(
    src_tga: &Path,
    dst_tga: &Path,
    rd: &RegionData,
    new_index: usize,
    rect: [f64; 4],
    colour: [u8; 3],
    source: Option<(usize, [u8; 3])>,
) -> Result<PaintReport> {
    let mut d = fs::read(src_tga).with_context(|| format!("read {}", src_tga.display()))?;
    let report = paint_lookup(&mut d, &src_tga.display().to_string(), rd, new_index, rect, colour, source)?;
    if let Some(dir) = dst_tga.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    fs::write(dst_tga, &d).with_context(|| format!("write {}", dst_tga.display()))?;
    Ok(report)
}

/// [`paint_lookup_tga`] on file bytes; `name` is only used in messages.
fn paint_lookup(
    d: &mut Vec<u8>,
    name: &str,
    rd: &RegionData,
    new_index: usize,
    rect: [f64; 4],
    colour: [u8; 3],
    source: Option<(usize, [u8; 3])>,
) -> Result<PaintReport> {
    let id_len = d[0] as usize;
    let (cmap_type, image_type) = (d[1], d[2]);
    let first = u16::from_le_bytes([d[3], d[4]]) as usize;
    let count = u16::from_le_bytes([d[5], d[6]]) as usize;
    let entry_bits = d[7];
    let (w, h) = (
        u16::from_le_bytes([d[12], d[13]]) as usize,
        u16::from_le_bytes([d[14], d[15]]) as usize,
    );
    let (bpp, desc) = (d[16], d[17]);
    ensure!(
        cmap_type == 1 && image_type == 1 && entry_bits == 24 && bpp == 8 && first == 0 && count == 256,
        "{} is not an uncompressed 8-bit colour-mapped TGA with a 256 entry 24-bit palette",
        name
    );
    ensure!(w > 1 && h > 1, "{} is a {w}x{h} placeholder, not a region lookup map", name);
    let pal_at = 18 + id_len;
    let px_at = pal_at + 3 * count;
    ensure!(d.len() >= px_at + w * h, "{} pixel data is truncated", name);
    let bgr = |c: [u8; 3]| [c[2], c[1], c[0]];
    let find = |d: &[u8], c: [u8; 3]| (0..count).find(|&i| d[pal_at + 3 * i..pal_at + 3 * i + 3] == bgr(c));
    ensure!(
        find(d, colour).is_none(),
        "lookup colour {colour:?} is already a palette entry of {}; choose another colour for the new region",
        name
    );
    let mut used = [false; 256];
    for &p in &d[px_at..px_at + w * h] {
        used[p as usize] = true;
    }
    let first_used = used.iter().position(|&u| u).unwrap_or(count);
    let index = (1..first_used)
        .rev()
        .find(|&i| !used[i])
        .or_else(|| (1..count).find(|&i| !used[i]))
        .ok_or_else(|| anyhow!("the palette of {} has no unused entry", name))?;
    d[pal_at + 3 * index..pal_at + 3 * index + 3].copy_from_slice(&bgr(colour));

    let [x0, y0, x1, y1] = rect;
    let (sx, sy) = ((x1 - x0) / w as f64, (y1 - y0) / h as f64);
    let top_origin = desc & 0x20 != 0;
    // file row of image row `py` counted from the world's southern edge
    let row_of = |py: usize| if top_origin { h - 1 - py } else { py };
    let centre = |px: usize, py: usize| [x0 + (px as f64 + 0.5) * sx, y0 + (py as f64 + 0.5) * sy];

    let mut report = PaintReport { palette_index: index, ..Default::default() };
    let source_index = match source {
        Some((ri, rgb)) => {
            let at = find(d, rgb)
                .ok_or_else(|| anyhow!("source colour {rgb:?} is not in the palette of {}", name))?;
            let slack = 1.5 * sx.max(sy);
            for py in 0..h {
                for px in 0..w {
                    if d[px_at + row_of(py) * w + px] as usize != at {
                        continue;
                    }
                    let pt = centre(px, py);
                    report.checked += 1;
                    if rd.in_region(ri, pt)
                        || rd.in_region(new_index, pt)
                        || rd.distance_to_region(ri, pt).min(rd.distance_to_region(new_index, pt)) <= slack
                    {
                        report.on_polygon += 1;
                    }
                }
            }
            ensure!(report.checked > 0, "no pixel of {} carries the colour {rgb:?} of region {}", name, rd.regions[ri].key);
            ensure!(
                report.on_polygon as f64 >= LOOKUP_CHECK_SHARE * report.checked as f64,
                "only {} of {} pixels of {} with the colour of region {} lie on its polygon: the theatre rectangle {rect:?} does not match the image",
                report.on_polygon,
                report.checked,
                name,
                rd.regions[ri].key
            );
            Some(at)
        }
        None => None,
    };

    // only pixels inside the bounding box of the new region can belong to it
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for a in &rd.regions[new_index].areas {
        for o in &a.outlines {
            for &v in &o.ring {
                let p = rd.vertices[v as usize];
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
    }
    ensure!(lo[0] <= hi[0], "region {} has no outline", rd.regions[new_index].key);
    for py in 0..h {
        let wy = y0 + (py as f64 + 0.5) * sy;
        if wy < lo[1] as f64 - sy || wy > hi[1] as f64 + sy {
            continue;
        }
        for px in 0..w {
            let wx = x0 + (px as f64 + 0.5) * sx;
            if wx < lo[0] as f64 - sx || wx > hi[0] as f64 + sx {
                continue;
            }
            let at = px_at + row_of(py) * w + px;
            if source_index.is_some_and(|s| d[at] as usize != s) {
                continue;
            }
            if rd.in_region(new_index, [wx, wy]) {
                d[at] = index as u8;
                report.painted += 1;
            }
        }
    }
    ensure!(
        report.painted > 0 || source.is_none(),
        "region {} covers no pixel of the lookup map {name} ({sx:.2} x {sy:.2} world units per pixel); cut a larger region",
        rd.regions[new_index].key
    );
    Ok(report)
}

// ------------------------------------------------------------------------------------------------ border splines

fn border_point(p: P) -> [f32; 3] {
    [(p[0] * BORDER_UNITS_PER_WORLD) as f32, BORDER_Y, (p[1] * BORDER_UNITS_PER_WORLD) as f32]
}

/// Points of the edge `a` -> `b` every [`BORDER_STEP`] world units, `a` included, `b` excluded.
fn edge_points(a: P, b: P) -> impl Iterator<Item = [f32; 3]> {
    let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
    let steps = (len / BORDER_STEP).ceil().max(1.0) as usize;
    (0..steps).map(move |s| {
        let t = s as f64 / steps as f64;
        border_point([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
    })
}

/// The border lines of an outline: ETW draws a border only where the region touches another land region (never
/// along a coast), so the maximal runs of such edges become open polylines; an outline that touches land
/// everywhere becomes one closed loop. Resampled every [`BORDER_STEP`] world units.
fn border_lines(rd: &RegionData, o: &Outline) -> Vec<Vec<[f32; 3]>> {
    let n = o.ring.len();
    let mut drawn = vec![false; n];
    for run in o.runs.iter().filter(|r| r.code != NONE && rd.regions[(r.code >> 16) as usize].kind != "sea") {
        for k in run.from..=run.to {
            drawn[k as usize % n] = true;
        }
    }
    let start = drawn.iter().position(|&d| !d).unwrap_or(0);
    let mut lines = Vec::new();
    let mut cur: Vec<[f32; 3]> = Vec::new();
    for i in 1..=n {
        let k = (start + i) % n;
        if !drawn[k] {
            continue;
        }
        let (a, b) = (rd.v(o.ring[k]), rd.v(o.ring[(k + 1) % n]));
        cur.extend(edge_points(a, b));
        if !drawn[(k + 1) % n] || i == n {
            cur.push(border_point(b));
            lines.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

fn outer_outline(rd: &RegionData, ri: usize, area: usize) -> Result<&Outline> {
    rd.regions[ri]
        .areas
        .get(area)
        .and_then(|a| a.outlines.iter().find(|o| o.outer))
        .ok_or_else(|| anyhow!("{}: area {area} has no outer ring", rd.regions[ri].key))
}

fn spline_area2(points: &[[f32; 3]]) -> f64 {
    let ring: Vec<P> = points.iter().map(|p| [p[0] as f64, p[2] as f64]).collect();
    signed_area2(&ring)
}

/// Gives `points` the winding of `like` (reversing when they differ).
fn match_winding(points: &mut [[f32; 3]], like: &[[f32; 3]]) {
    let closed = |p: &[[f32; 3]]| p.len() > 2 && p.first() == p.last();
    if closed(points) && closed(like) && spline_area2(points).signum() != spline_area2(like).signum() {
        points.reverse();
    }
}

/// True when most of the spline's points lie on `ring` (world coordinates).
fn spline_follows(spline: &Spline, ring: &[P]) -> bool {
    let n = spline.points.len();
    if n == 0 || ring.is_empty() {
        return false;
    }
    let samples = n.min(48);
    let near = (0..samples)
        .filter(|&s| {
            let pt = xz_to_world(spline.points[s * n / samples]);
            (0..ring.len()).any(|k| {
                point_segment_distance(pt, ring[k], ring[(k + 1) % ring.len()]) <= BORDER_MATCH_DISTANCE
            })
        })
        .count();
    near as f64 >= BORDER_MATCH_SHARE * samples as f64
}

/// Number of a border file name: 0 for `<key>.rigid_spline`, N for `<key>_N.rigid_spline`.
fn border_number(name: &str, key: &str) -> Option<usize> {
    let stem = name.strip_suffix(".rigid_spline")?;
    if stem == key {
        return Some(0);
    }
    stem.strip_prefix(key)?
        .strip_prefix('_')
        .filter(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))?
        .parse()
        .ok()
}

/// `display/borders` with the edits of all splits held in memory until [`BorderStore::flush`].
struct BorderStore {
    dir: PathBuf,
    /// file name -> new content (`None` = delete)
    pending: BTreeMap<String, Option<Vec<u8>>>,
}

impl BorderStore {
    fn new(map_dir: &Path) -> Self {
        Self { dir: map_dir.join("display/borders"), pending: BTreeMap::new() }
    }

    /// Names of the files that belong to region `key`, ordered by number.
    fn files(&self, key: &str) -> Result<Vec<String>> {
        let mut names: BTreeSet<String> = BTreeSet::new();
        if self.dir.is_dir() {
            for entry in fs::read_dir(&self.dir)? {
                if let Some(name) = entry?.file_name().to_str() {
                    names.insert(name.to_string());
                }
            }
        }
        names.extend(self.pending.iter().filter(|(_, v)| v.is_some()).map(|(k, _)| k.clone()));
        names.retain(|n| !matches!(self.pending.get(n), Some(None)));
        let mut out: Vec<(usize, String)> =
            names.into_iter().filter_map(|n| border_number(&n, key).map(|i| (i, n))).collect();
        out.sort();
        Ok(out.into_iter().map(|(_, n)| n).collect())
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        match self.pending.get(name) {
            Some(Some(bytes)) => Ok(bytes.clone()),
            _ => fs::read(self.dir.join(name)).with_context(|| format!("read border {name}")),
        }
    }

    fn flush(&self) -> Result<()> {
        fs::create_dir_all(&self.dir)?;
        for (name, content) in &self.pending {
            let path = self.dir.join(name);
            match content {
                Some(bytes) => fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?,
                None => {
                    if path.exists() {
                        fs::remove_file(&path)?;
                    }
                }
            }
        }
        Ok(())
    }
}

struct BorderEdit {
    written: Vec<String>,
    removed: Vec<String>,
}

/// Replaces the border of the split area. `old_ring` is the area's outer ring BEFORE the split (world coordinates);
/// the source's loop that follows it is replaced by the shrunken ring, the other loops (other areas) stay; the new
/// region gets its own file.
fn write_borders(
    store: &mut BorderStore,
    rd: &RegionData,
    old_ring: &[P],
    source: (usize, usize),
    new_index: usize,
) -> Result<BorderEdit> {
    let (src_index, area) = source;
    let (src_key, new_key) = (rd.regions[src_index].key.clone(), rd.regions[new_index].key.clone());
    let files = store.files(&src_key)?;
    let mut loops: Vec<(u32, Spline)> = Vec::new();
    for name in &files {
        let (version, splines) = parse_spln(&store.read(name)?).with_context(|| name.clone())?;
        loops.extend(splines.into_iter().map(|s| (version, s)));
    }
    let mut src_lines = border_lines(rd, outer_outline(rd, src_index, area)?);
    let mut new_lines = border_lines(rd, outer_outline(rd, new_index, 0)?);
    let matched: Vec<usize> = (0..loops.len()).filter(|&i| spline_follows(&loops[i].1, old_ring)).collect();
    let (mut version, mut flag, mut at) = (1u32, 1u32, loops.len());
    if let Some(&first) = matched.first() {
        let (v, old) = &loops[first];
        (version, flag, at) = (*v, old.flag, first);
        for line in src_lines.iter_mut().chain(new_lines.iter_mut()) {
            match_winding(line, &old.points);
        }
    }
    // the matched loops go; the new source lines take the place of the first one (empty name = not named yet)
    let mut kept: Vec<(u32, Spline)> = Vec::new();
    let mut inserted = false;
    for (i, l) in loops.into_iter().enumerate() {
        if i == at && !inserted {
            inserted = true;
            kept.extend(src_lines.drain(..).map(|points| (version, Spline { name: String::new(), flag, points })));
        }
        if !matched.contains(&i) {
            kept.push(l);
        }
    }
    if !inserted {
        kept.extend(src_lines.drain(..).map(|points| (version, Spline { name: String::new(), flag, points })));
    }

    let mut edit = BorderEdit { written: Vec::new(), removed: Vec::new() };
    for name in files.into_iter().chain(store.files(&new_key)?) {
        store.pending.insert(name.clone(), None);
        edit.removed.push(name);
    }
    let new_loops = new_lines.into_iter().map(|points| (version, Spline { name: String::new(), flag, points }));
    for (key, loops) in [(&src_key, kept), (&new_key, new_loops.collect())] {
        let numbered = loops.len() > 1;
        for (i, (v, mut spline)) in loops.into_iter().enumerate() {
            let name = if numbered { format!("{key}_{}.rigid_spline", i + 1) } else { format!("{key}.rigid_spline") };
            if spline.name.is_empty() {
                spline.name = if numbered { format!("border:{key}:{}", i + 1) } else { format!("border:{key}") };
            }
            store.pending.insert(name.clone(), Some(write_spln(v, &[spline])));
            edit.written.push(name);
        }
    }
    edit.removed.retain(|n| !edit.written.contains(n));
    Ok(edit)
}

// ------------------------------------------------------------------------------------------------ DB rows

/// The parts of [`MapArtifacts`] that the new regions extend.
struct MapRows {
    db_rows: BTreeMap<String, Value>,
    loc_rows: Vec<(String, String, u8, String)>,
    slot_mapping: Value,
}

fn row_str<'a>(row: &'a Value, field: &str) -> Result<&'a str> {
    row.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("row has no text field {field}: {row}"))
}

fn table_rows_mut<'a>(map: &'a mut MapRows, table: &str) -> Result<&'a mut Vec<Value>> {
    map.db_rows
        .get_mut(table)
        .and_then(|t| t.get_mut("rows"))
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow!("map artifacts have no {table} rows"))
}

fn set_loc(map: &mut MapRows, key: String, text: &str) {
    match map.loc_rows.iter_mut().find(|r| r.0 == key) {
        Some(row) => row.1 = text.to_string(),
        None => map.loc_rows.push((key, text.to_string(), 0, "localisation".to_string())),
    }
}

/// Names and slots of a region that gets its settlement row, localisation rows and slot mapping entry.
struct NewRows<'a> {
    /// playable region whose settlement row / slot mapping entry is the template (split source or activation donor)
    template: &'a str,
    key: &'a str,
    name: &'a str,
    settlement_key: String,
    settlement_name: &'a str,
    slot_keys: &'a [String],
}

/// Adds the settlement row (a clone of the template region's settlement row), the three localisation rows and the
/// slot mapping entry of a new region.
fn add_rows(map: &mut MapRows, rules: &Rules, new: &NewRows) -> Result<()> {
    let source_settlement = rules
        .regions
        .get(new.template)
        .and_then(|r| r.get("capital_settlement"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("rules have no capital settlement for region {}", new.template))?
        .to_string();
    let settlement_key = new.settlement_key.clone();
    let rows = table_rows_mut(map, "campaign_map_settlements_tables")?;
    ensure!(
        !rows.iter().any(|r| r.get("settlement_id").and_then(Value::as_str) == Some(settlement_key.as_str())),
        "settlement row {settlement_key} already exists"
    );
    let mut row = rows
        .iter()
        .find(|r| r.get("settlement_id").and_then(Value::as_str) == Some(source_settlement.as_str()))
        .cloned()
        .ok_or_else(|| anyhow!("campaign_map_settlements_tables has no row {source_settlement}"))?;
    let general_slots = row.get("slot_count").cloned();
    row["settlement_id"] = json!(settlement_key);
    row["region"] = json!(new.key);
    row["default_onscreen_name"] = json!(new.settlement_name);
    let at = rows
        .iter()
        .position(|r| r.get("settlement_id").and_then(Value::as_str).is_some_and(|k| k > settlement_key.as_str()))
        .unwrap_or(rows.len());
    rows.insert(at, row);

    set_loc(map, format!("regions_onscreen_{}", new.key), new.name);
    set_loc(map, format!("regions_battle_name_{}", new.key), new.settlement_name);
    set_loc(
        map,
        format!("start_pos_settlements_onscreen_name_{settlement_key}"),
        new.settlement_name,
    );

    // slot mapping (region -> settlement slot keys as written to regions.esf)
    let regions = if map.slot_mapping.get("regions").is_some() {
        map.slot_mapping.get_mut("regions")
    } else {
        Some(&mut map.slot_mapping)
    };
    if let Some(regions) = regions.and_then(Value::as_object_mut) {
        let source_entry = regions.get(new.template).cloned().unwrap_or(Value::Null);
        let slots: Vec<Value> = new
            .slot_keys
            .iter()
            .map(|k| {
                let ty = k.split(':').nth(3).unwrap_or_default();
                json!({"etw_key": k, "key": k, "type": ty, "dropped": false})
            })
            .collect();
        regions.insert(
            new.key.to_string(),
            json!({
                "settlement_id": settlement_key,
                "settlement_slot_type": source_entry.get("settlement_slot_type").cloned().unwrap_or(Value::Null),
                "general_slots": source_entry.get("general_slots").cloned().or(general_slots).unwrap_or(Value::Null),
                "slots": slots,
            }),
        );
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ driver

fn read_esf(path: &Path) -> Result<esf::Esf> {
    esf::read(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

fn write_esf(path: &Path, file: &esf::Esf) -> Result<()> {
    fs::write(path, esf::write(file)?).with_context(|| format!("write {}", path.display()))
}

fn lookup_rgb(rules: &Rules, region: &str) -> Result<[u8; 3]> {
    let v = rules
        .regions
        .get(region)
        .and_then(|r| r.get("lookup_rgb"))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("rules have no lookup_rgb for region {region}"))?;
    ensure!(v.len() == 3, "lookup_rgb of {region} is not an RGB triple");
    let mut out = [0u8; 3];
    for (o, c) in out.iter_mut().zip(v) {
        *o = c.as_u64().and_then(|c| u8::try_from(c).ok()).ok_or_else(|| anyhow!("bad lookup_rgb of {region}"))?;
    }
    Ok(out)
}

/// `overlay_file` of the theatre's `campaign_map_playable_areas_tables` row (a land theatre with its own lookup map).
fn overlay_file(map: &MapRows, theatre: &str) -> Result<String> {
    let rows = map
        .db_rows
        .get("campaign_map_playable_areas_tables")
        .and_then(|t| t.get("rows"))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("map artifacts have no campaign_map_playable_areas_tables rows"))?;
    let row = rows
        .iter()
        .find(|r| match r.get("index") {
            Some(Value::String(s)) => s == theatre,
            Some(Value::Number(n)) => n.to_string() == theatre,
            _ => false,
        })
        .ok_or_else(|| anyhow!("no playable area row for theatre {theatre}"))?;
    ensure!(
        row.get("sea_trade").and_then(Value::as_bool) != Some(true),
        "theatre {theatre} is a sea-trade theatre without a land lookup map"
    );
    Ok(row_str(row, "overlay_file")?.to_string())
}

fn validate(split: &RegionSplit) -> Result<()> {
    ensure!(!split.key.is_empty() && !split.settlement.is_empty(), "split {:?}: empty key or settlement", split.key);
    for (what, text) in [("key", &split.key), ("settlement", &split.settlement)] {
        ensure!(
            !text.contains([':', '\t', '\n', '\r']),
            "split {}: the {what} {text:?} must not contain ':' or control characters",
            split.key
        );
    }
    for (what, text) in [("name", &split.name), ("settlement name", &split.settlement_name)] {
        ensure!(
            !text.is_empty() && !text.contains(['\t', '\n', '\r']),
            "split {}: the {what} {text:?} is empty or contains a tab / newline",
            split.key
        );
    }
    Ok(())
}

/// [`apply_splits`] plus the per-split statistics.
pub fn apply_splits_with_stats(
    map_dir: &Path,
    splits: &[RegionSplit],
    rules: &Rules,
    map: &mut MapArtifacts,
) -> Result<(Vec<AppliedSplit>, ApplyStats)> {
    let mut stats = ApplyStats::default();
    if splits.is_empty() {
        return Ok((Vec::new(), stats));
    }
    let mut regions_file = read_esf(&map_dir.join("regions.esf"))?;
    let mut rd = RegionData::parse(&regions_file.root)?;
    let theatre_rects: HashMap<String, [f64; 4]> = theatres(&regions_file.root)?
        .into_iter()
        .map(|t| (t.key, [t.lo[0] as f64, t.lo[1] as f64, t.hi[0] as f64, t.hi[1] as f64]))
        .collect();
    let mut pf_file = read_esf(&map_dir.join("pathfinding.esf"))?;
    let mut grids: BTreeMap<usize, Grid> = BTreeMap::new();
    let mut poi_file = read_esf(&map_dir.join("poi.esf"))?;
    let mut pois = poi::parse(&poi_file.root)?;
    let mut trade_file = read_esf(&map_dir.join("trade_routes.esf"))?;
    let mut sea_file = read_esf(&map_dir.join("sea_grids.esf"))?;
    let mut tgas: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut borders = BorderStore::new(map_dir);
    // the map artifacts only change once every file edit succeeded
    let mut staged = MapRows {
        db_rows: map.db_rows.clone(),
        loc_rows: map.loc_rows.clone(),
        slot_mapping: map.slot_mapping.clone(),
    };
    let mut applied: Vec<AppliedSplit> = Vec::new();
    // world rectangles (old outer rings of the cut areas) whose index cells / sea cells need recomputing
    let mut changed_rects: Vec<(P, P)> = Vec::new();
    // pathfinding: grids whose untouched state passes the rule checker are checked again after the splits; every
    // relabelled split gets its settlement node and border band from `pathsynth::synthesize_split`
    let rd_before = rd.clone();
    let mut grid_baseline: BTreeMap<usize, std::collections::BTreeSet<String>> = BTreeMap::new();
    let mut relabelled: Vec<(usize, usize, usize, String)> = Vec::new();

    for split in splits {
        validate(split)?;
        let source = rd.index(&split.source)?;
        let src_before = named_borders(&rd, source);
        let theatre = theatre_listing(&regions_file.root, &split.source)?;
        let rect = *theatre_rects
            .get(&theatre)
            .ok_or_else(|| anyhow!("theatre {theatre} has no rectangle in regions.esf"))?;
        let old_ring = ring_points(&rd, outer_ring(&rd, source, split.area)?);
        changed_rects.push(bbox_of(&old_ring));
        let old_settlement = settlement_center(&rd.regions[source]).ok();
        let overlay = overlay_file(&staged, &theatre)?;
        let source_rgb = lookup_rgb(rules, &split.source)?;

        // ---- regions.esf
        let new_index = rd.split(&Split {
            key: split.source.clone(),
            area: split.area,
            a: split.a,
            b: split.b,
            interior: split.interior.clone(),
            new_key: split.key.clone(),
            settlement: SettlementSpec { name: split.settlement.clone(), center: split.center },
        })?;
        add_theatre_region_key(&mut regions_file.root, &split.source, &split.key, split.center)?;
        let centre = [split.center[0] as f64, split.center[1] as f64];
        ensure!(
            rd.in_region(new_index, centre),
            "split {}: the settlement position {:?} is not inside the new region",
            split.key,
            split.center
        );
        if let Some(old) = old_settlement {
            ensure!(
                !rd.in_region(new_index, [old[0] as f64, old[1] as f64]),
                "split {}: the settlement of {} ({:?}) ends up inside the new region; cut so that it stays with {}",
                split.key,
                split.source,
                old,
                split.source
            );
        }
        let adjacent = rd.adjacent(new_index);

        // ---- pathfinding.esf
        let area = Grid::area_at(&pf_file.root, centre)?
            .ok_or_else(|| anyhow!("split {}: the settlement position lies outside every pathfinding area", split.key))?;
        if let std::collections::btree_map::Entry::Vacant(v) = grids.entry(area) {
            let parsed = Grid::parse_area(&pf_file.root, area)?;
            grid_baseline.insert(area, super::stages::problem_identities(&parsed, &rd_before));
            v.insert(parsed);
        }
        let grid = grids.get_mut(&area).expect("grid just inserted");
        let in_new = |p: P| rd.in_region(new_index, p);
        // a source that has no path yet (an activation whose paths the format-8 synthesis creates afterwards) has
        // nothing to relabel: both halves are synthesised together then
        let (pieces, empties) = if grid.region_ids.contains(&(source as i16)) {
            relabelled.push((area, source, new_index, split.key.clone()));
            grid.relabel(source as i16, new_index as i16, &in_new)?
        } else {
            (0, 0)
        };

        // ---- poi.esf
        let poi_report = poi::apply_split(&mut pois, &rd, source, new_index, centre, &adjacent)?;

        // ---- trade_routes.esf
        let mut linked: Vec<(String, [f32; 2])> = Vec::new();
        for &i in &adjacent {
            let r = &rd.regions[i];
            if r.settlement.is_some() && trade_routes::settlement_node_index(&trade_file.root, &r.key).is_ok() {
                linked.push((r.key.clone(), settlement_center(r)?));
            }
        }
        let refs: Vec<(&str, [f32; 2])> = linked.iter().map(|(k, p)| (k.as_str(), *p)).collect();
        let trade = trade_routes::add_settlement(&mut trade_file.root, &split.key, split.center, &refs)?;

        // ---- lookup TGA
        if !tgas.contains_key(&overlay) {
            let path = map_dir.join(&overlay);
            tgas.insert(overlay.clone(), fs::read(&path).with_context(|| format!("read {}", path.display()))?);
        }
        let tga = tgas.get_mut(&overlay).expect("tga just inserted");
        let lookup = paint_lookup(tga, &overlay, &rd, new_index, rect, split.colour, Some((source, source_rgb)))?;

        // ---- display/borders
        let edit = write_borders(&mut borders, &rd, &old_ring, (source, split.area), new_index)?;

        // ---- result
        let (slot_keys, slot_positions) = settlement_slots(&rd.regions[new_index])?;
        let this = AppliedSplit {
            key: split.key.clone(),
            source: split.source.clone(),
            new_index,
            settlement_key: split.settlement_key(),
            slot_keys,
            slot_positions,
            center: split.center,
            neighbours: adjacent.iter().map(|&i| rd.regions[i].key.clone()).collect(),
            border_lengths: named_borders(&rd, new_index),
            source_border_lengths_before: src_before,
            source_border_lengths_after: named_borders(&rd, source),
            theatre,
        };
        let rows = NewRows {
            template: &split.source,
            key: &split.key,
            name: &split.name,
            settlement_key: split.settlement_key(),
            settlement_name: &split.settlement_name,
            slot_keys: &this.slot_keys,
        };
        add_rows(&mut staged, rules, &rows)?;
        stats.splits.push(SplitStats {
            key: split.key.clone(),
            new_index,
            pathfinding_area: area,
            pieces_relabelled: pieces,
            empty_cells_relabelled: empties,
            poi: poi_report,
            trade_node: trade.node,
            trade_routes: trade.routes_added,
            lookup_file: overlay,
            lookup,
            border_files_written: edit.written,
            border_files_removed: edit.removed,
        });
        applied.push(this);
    }

    // ---- pathfinding of the new regions: node, border band, group path
    for (area, source, new_index, key) in &relabelled {
        let kind_name = staged.slot_mapping["regions"][key.as_str()]["settlement_slot_type"]
            .as_str()
            .ok_or_else(|| anyhow!("split {key}: the settlement slot type of the new region is unknown"))?;
        let kind = super::pathsynth::SlotKind::from_name(kind_name)
            .ok_or_else(|| anyhow!("split {key}: settlement slot type {kind_name} has no node shape"))?;
        let grid = grids.get_mut(area).expect("grid of a relabelled split");
        super::pathsynth::synthesize_split(grid, &rd, *source, *new_index, kind, &super::pathsynth::Options::default())
            .with_context(|| format!("split {key}: pathfinding node and border band"))?;
    }
    for (area, grid) in &grids {
        // problems the untouched grid had stay (America / Europe fail the checker); any new one aborts
        let baseline = grid_baseline.get(area).cloned().unwrap_or_default();
        let problems = super::stages::new_problems(&baseline, grid, &rd);
        ensure!(
            problems.is_empty(),
            "pathfinding area {area} gets {} new rule-checker problem(s) from the splits:\n  {}",
            problems.len(),
            problems.iter().take(20).cloned().collect::<Vec<_>>().join("\n  ")
        );
    }

    // ---- everything computed: serialise and write
    rd.store(&mut regions_file.root)?;
    stats.query_cells = rd.rebuild_query_index_within(&mut regions_file.root, Some(&changed_rects))?;
    stats.sea_cells_changed =
        sea_grids::add_region_within(&mut sea_file.root, &rd, rd.regions.len() - 1, Some(&changed_rects))?;
    for grid in grids.values() {
        grid.store(&mut pf_file.root)?;
    }
    poi::store(&mut poi_file.root, &pois)?;
    write_esf(&map_dir.join("regions.esf"), &regions_file)?;
    write_esf(&map_dir.join("pathfinding.esf"), &pf_file)?;
    write_esf(&map_dir.join("poi.esf"), &poi_file)?;
    write_esf(&map_dir.join("trade_routes.esf"), &trade_file)?;
    write_esf(&map_dir.join("sea_grids.esf"), &sea_file)?;
    for (name, bytes) in &tgas {
        let path = map_dir.join(name);
        fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    }
    borders.flush()?;
    map.db_rows = staged.db_rows;
    map.loc_rows = staged.loc_rows;
    map.slot_mapping = staged.slot_mapping;
    Ok((applied, stats))
}

// ------------------------------------------------------------------------------------------------ activations

/// What one activation changed, for the start position builder.
#[derive(Clone, Debug)]
pub struct AppliedActivation {
    pub key: String,
    pub donor: String,
    /// index of the region in `regions.esf`
    pub index: usize,
    /// `settlement:<key>:<settlement>`
    pub settlement_key: String,
    /// settlement slot keys of the region, in `slot_descriptions` order
    pub slot_keys: Vec<String>,
    /// world position (x, y) of each slot
    pub slot_positions: Vec<[f32; 2]>,
    pub center: [f32; 2],
    /// bordering regions that are playable (present in `rules.regions`, which includes activated and split ones)
    pub neighbours: Vec<String>,
    /// polygon border length of the region per bordering region (sea and wilderness included)
    pub border_lengths: BTreeMap<String, f64>,
    /// polygon border length of the donor per bordering region, same method
    pub donor_border_lengths: BTreeMap<String, f64>,
    /// regions.esf theatre key that lists the region
    pub theatre: String,
}

/// Per-activation statistics.
#[derive(Clone, Debug)]
pub struct ActivationStats {
    pub key: String,
    pub index: usize,
    /// `tail` written to regions.esf (the donor's: the id of the theatre that lists the region)
    pub tail: i32,
    pub pathfinding_area: usize,
    pub poi: poi::ActivationReport,
    pub trade_node: u32,
    pub trade_routes: usize,
    pub lookup_file: String,
    pub lookup: PaintReport,
    pub border_files_written: Vec<String>,
    pub border_files_removed: Vec<String>,
    /// cells of `sea_grids.esf` that list the region (checked, not changed)
    pub sea_cells_listing: usize,
}

/// Slot keys and world positions (x, y) of the settlement slots of `region`, in `slot_descriptions` order.
fn settlement_slots(region: &Region) -> Result<(Vec<String>, Vec<[f32; 2]>)> {
    let slots = region
        .settlement
        .as_ref()
        .ok_or_else(|| anyhow!("region {} has no settlement", region.key))?
        .children()?
        .get(2)
        .ok_or_else(|| anyhow!("settlement record of {} has no slot list", region.key))?
        .items()?;
    let mut keys = Vec::new();
    let mut positions = Vec::new();
    for s in slots {
        keys.push(s[0].as_str()?.to_string());
        positions.push(match s[2].value()? {
            EsfValue::XYZ(v) => [v[0], v[2]],
            EsfValue::XY(v) => *v,
            other => bail!("slot position of {} is {other:?}", region.key),
        });
    }
    Ok((keys, positions))
}

/// Border file(s) of region `index` (which has none yet): one closed loop or open polyline per land border run of
/// every area, written like the split regions' borders (version / flag / winding copied from the first loop of the
/// `like` region's file).
fn write_region_borders(store: &mut BorderStore, rd: &RegionData, like: &str, index: usize) -> Result<BorderEdit> {
    let key = rd.regions[index].key.clone();
    let style = match store.files(like)?.first() {
        Some(name) => {
            let (version, splines) = parse_spln(&store.read(name)?).with_context(|| name.clone())?;
            splines.into_iter().next().map(|s| (version, s))
        }
        None => None,
    };
    let (version, flag) = style.as_ref().map_or((1, 1), |(v, s)| (*v, s.flag));
    let mut lines: Vec<Vec<[f32; 3]>> = Vec::new();
    for a in &rd.regions[index].areas {
        for o in a.outlines.iter().filter(|o| o.outer) {
            lines.extend(border_lines(rd, o));
        }
    }
    lines.retain(|l| l.len() >= 2);
    if let Some((_, like)) = &style {
        for l in &mut lines {
            match_winding(l, &like.points);
        }
    }
    let mut edit = BorderEdit { written: Vec::new(), removed: Vec::new() };
    for name in store.files(&key)? {
        store.pending.insert(name.clone(), None);
        edit.removed.push(name);
    }
    let numbered = lines.len() > 1;
    for (i, points) in lines.into_iter().enumerate() {
        let name = if numbered { format!("{key}_{}.rigid_spline", i + 1) } else { format!("{key}.rigid_spline") };
        let spline_name = if numbered { format!("border:{key}:{}", i + 1) } else { format!("border:{key}") };
        let spline = Spline { name: spline_name, flag, points };
        store.pending.insert(name.clone(), Some(write_spln(version, &[spline])));
        edit.written.push(name);
    }
    edit.removed.retain(|n| !edit.written.contains(n));
    Ok(edit)
}

fn validate_activation(a: &RegionActivation) -> Result<()> {
    ensure!(
        !a.key.is_empty() && !a.donor.is_empty() && !a.settlement.is_empty(),
        "activation {:?}: empty key, donor or settlement",
        a.key
    );
    for (what, text) in [("key", &a.key), ("donor", &a.donor), ("settlement", &a.settlement)] {
        ensure!(
            !text.contains([':', '\t', '\n', '\r']),
            "activation {}: the {what} {text:?} must not contain ':' or control characters",
            a.key
        );
    }
    for (what, text) in [("name", &a.name), ("settlement name", &a.settlement_name)] {
        ensure!(
            !text.is_empty() && !text.contains(['\t', '\n', '\r']),
            "activation {}: the {what} {text:?} is empty or contains a tab / newline",
            a.key
        );
    }
    ensure!(a.key != a.donor, "activation {}: the region cannot be its own donor", a.key);
    Ok(())
}

/// Checks that every populated cell overlapping `ri`'s bounding box lists the region in its land list exactly when
/// the region overlaps the cell (the converter keeps ETW's lists, which already contain wilderness regions).
/// Returns the number of cells listing it.
fn check_sea_grids(root: &Node, rd: &RegionData, ri: usize) -> Result<usize> {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for a in &rd.regions[ri].areas {
        for o in &a.outlines {
            for &v in &o.ring {
                let p = rd.vertices[v as usize];
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
    }
    let key = &rd.regions[ri].key;
    let sg = sea_grids::SeaGrids::parse(root)?;
    let mut listing = 0;
    for g in &sg.grids {
        for (cell, area) in g.cells.iter().zip(&g.areas) {
            if area.lands.is_empty() && area.seas.is_empty() {
                continue;
            }
            if cell.hi[0] < lo[0] || cell.lo[0] > hi[0] || cell.hi[1] < lo[1] || cell.lo[1] > hi[1] {
                continue;
            }
            let listed = area.lands.contains(key);
            let overlaps = sea_grids::covering_regions(rd, cell.lo, cell.hi, false).contains(&ri);
            ensure!(
                listed == overlaps,
                "sea_grids.esf cell ({}, {}) of theatre {} {} region {key} although it {} the cell",
                cell.xi,
                cell.yi,
                g.name,
                if listed { "lists" } else { "does not list" },
                if overlaps { "overlaps" } else { "does not overlap" }
            );
            listing += listed as usize;
        }
    }
    Ok(listing)
}

/// Applies every activation to the converted map files under `map_dir` and appends the regions' rows to `map`.
/// Run it before [`apply_splits`]: afterwards an activated key is an ordinary region (a split may cut it).
pub fn apply_activations(
    map_dir: &Path,
    activations: &[RegionActivation],
    rules: &Rules,
    map: &mut MapArtifacts,
) -> Result<Vec<AppliedActivation>> {
    Ok(apply_activations_with_stats(map_dir, activations, rules, map)?.0)
}

/// [`apply_activations`] plus the per-activation statistics.
///
/// Per activation: `regions.esf` gets the donor's settlement (translated to `center`, slot keys renamed), the donor's
/// theatre id in `tail` and a theatre region-key entry (geometry and query index are untouched);
/// `pathfinding.esf` and `sea_grids.esf` are only asserted to already contain the region; `poi.esf` gets the
/// settlement, interior and border points of [`poi::apply_activation`]; `trade_routes.esf` a settlement node
/// linked to the playable neighbours; the theatre's lookup TGA the region's pixels in `colour`; `display/borders` the
/// region's border splines; the map artifacts the settlement row, localisation rows and slot mapping.
pub fn apply_activations_with_stats(
    map_dir: &Path,
    activations: &[RegionActivation],
    rules: &Rules,
    map: &mut MapArtifacts,
) -> Result<(Vec<AppliedActivation>, Vec<ActivationStats>)> {
    apply_activations_staged(map_dir, activations, rules, map, true)
}

/// [`apply_activations_with_stats`] for a chain whose `pathfinding.esf` / `sea_grids.esf` are regenerated after this
/// step: with `verify_grids == false` the checks that both files already hold the region (a path for it, matching
/// land lists) are skipped; the pathfinding area containing the settlement must still exist.
pub fn apply_activations_staged(
    map_dir: &Path,
    activations: &[RegionActivation],
    rules: &Rules,
    map: &mut MapArtifacts,
    verify_grids: bool,
) -> Result<(Vec<AppliedActivation>, Vec<ActivationStats>)> {
    if activations.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut regions_file = read_esf(&map_dir.join("regions.esf"))?;
    let mut rd = RegionData::parse(&regions_file.root)?;
    let theatre_rects: HashMap<String, [f64; 4]> = theatres(&regions_file.root)?
        .into_iter()
        .map(|t| (t.key, [t.lo[0] as f64, t.lo[1] as f64, t.hi[0] as f64, t.hi[1] as f64]))
        .collect();
    let pf_file = read_esf(&map_dir.join("pathfinding.esf"))?;
    let mut grids: BTreeMap<usize, Grid> = BTreeMap::new();
    let mut poi_file = read_esf(&map_dir.join("poi.esf"))?;
    let mut pois = poi::parse(&poi_file.root)?;
    let mut trade_file = read_esf(&map_dir.join("trade_routes.esf"))?;
    let sea_file = read_esf(&map_dir.join("sea_grids.esf"))?;
    let mut tgas: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut borders = BorderStore::new(map_dir);
    let mut staged = MapRows {
        db_rows: map.db_rows.clone(),
        loc_rows: map.loc_rows.clone(),
        slot_mapping: map.slot_mapping.clone(),
    };
    let rule_regions = rules.regions.as_object().context("rules have no region table")?;
    let mut applied: Vec<AppliedActivation> = Vec::new();
    let mut stats: Vec<ActivationStats> = Vec::new();

    for act in activations {
        validate_activation(act)?;
        ensure!(
            rule_regions.contains_key(&act.key),
            "activation {}: the rules have no region {} (apply the project to the rules first)",
            act.key,
            act.key
        );
        let ri = rd.index(&act.key)?;
        let donor = rd.index(&act.donor)?;
        ensure!(
            rule_regions.contains_key(&act.donor) && rd.regions[donor].settlement.is_some(),
            "activation {}: donor {} is not a playable region with a settlement",
            act.key,
            act.donor
        );
        ensure!(
            theatre_listing(&regions_file.root, &act.key).is_err(),
            "activation {}: the region is already listed in a theatre, it is playable already",
            act.key
        );
        let theatre = theatre_listing(&regions_file.root, &act.donor)?;
        let rect = *theatre_rects
            .get(&theatre)
            .ok_or_else(|| anyhow!("theatre {theatre} has no rectangle in regions.esf"))?;
        let centre = [act.center[0] as f64, act.center[1] as f64];
        ensure!(
            centre[0] >= rect[0] && centre[0] <= rect[2] && centre[1] >= rect[1] && centre[1] <= rect[3],
            "activation {}: the settlement position {:?} lies outside the playable rectangle of donor {}'s theatre {theatre}",
            act.key,
            act.center,
            act.donor
        );

        // ---- regions.esf
        // (the format-8 geometry stage has attached the settlement already)
        if rd.regions[ri].settlement.is_none() {
            rd.attach_settlement(ri, donor, &SettlementSpec { name: act.settlement.clone(), center: act.center })?;
        }
        add_theatre_region_key(&mut regions_file.root, &act.donor, &act.key, act.center)?;
        let (slot_keys, slot_positions) = settlement_slots(&rd.regions[ri])?;
        for (k, p) in slot_keys.iter().zip(&slot_positions) {
            ensure!(
                rd.in_region(ri, [p[0] as f64, p[1] as f64]),
                "activation {}: slot {k} would lie at {p:?}, outside the region; move the settlement further inland",
                act.key
            );
        }

        // ---- pathfinding.esf: wilderness regions already have their own path in the theatre grid
        let area = Grid::area_at(&pf_file.root, centre)?
            .ok_or_else(|| anyhow!("activation {}: the settlement position lies outside every pathfinding area", act.key))?;
        if verify_grids {
            if let std::collections::btree_map::Entry::Vacant(v) = grids.entry(area) {
                v.insert(Grid::parse_area(&pf_file.root, area)?);
            }
            let grid = &grids[&area];
            ensure!(
                grid.region_ids.contains(&(ri as i16)) && grid.single_path(ri as i16).is_some(),
                "activation {}: pathfinding.esf area {area} has no path for the region",
                act.key
            );
        }

        // ---- sea_grids.esf
        let sea_cells_listing = if verify_grids { check_sea_grids(&sea_file.root, &rd, ri)? } else { 0 };

        // ---- poi.esf
        let adjacent = rd.adjacent(ri);
        let playable = |i: usize| rule_regions.contains_key(&rd.regions[i].key);
        let poi_report = poi::apply_activation(&mut pois, &rd, donor, ri, centre, &playable)?;

        // ---- trade_routes.esf
        let neighbours: Vec<usize> = adjacent.iter().copied().filter(|&i| playable(i)).collect();
        let mut linked: Vec<(String, [f32; 2])> = Vec::new();
        for &i in &neighbours {
            let r = &rd.regions[i];
            if r.settlement.is_some() && trade_routes::settlement_node_index(&trade_file.root, &r.key).is_ok() {
                linked.push((r.key.clone(), settlement_center(r)?));
            }
        }
        let refs: Vec<(&str, [f32; 2])> = linked.iter().map(|(k, p)| (k.as_str(), *p)).collect();
        let trade = trade_routes::add_settlement(&mut trade_file.root, &act.key, act.center, &refs)?;

        // ---- lookup TGA
        let overlay = overlay_file(&staged, &theatre)?;
        if !tgas.contains_key(&overlay) {
            let path = map_dir.join(&overlay);
            tgas.insert(overlay.clone(), fs::read(&path).with_context(|| format!("read {}", path.display()))?);
        }
        let tga = tgas.get_mut(&overlay).expect("tga just inserted");
        let lookup = paint_lookup(tga, &overlay, &rd, ri, rect, act.colour, None)?;
        ensure!(
            lookup.painted > 0,
            "activation {}: the region covers no pixel of the lookup map {overlay}",
            act.key
        );

        // ---- display/borders
        let edit = write_region_borders(&mut borders, &rd, &act.donor, ri)?;

        // ---- result
        let this = AppliedActivation {
            key: act.key.clone(),
            donor: act.donor.clone(),
            index: ri,
            settlement_key: act.settlement_key(),
            slot_keys,
            slot_positions,
            center: act.center,
            neighbours: neighbours.iter().map(|&i| rd.regions[i].key.clone()).collect(),
            border_lengths: named_borders(&rd, ri),
            donor_border_lengths: named_borders(&rd, donor),
            theatre,
        };
        let rows = NewRows {
            template: &act.donor,
            key: &act.key,
            name: &act.name,
            settlement_key: act.settlement_key(),
            settlement_name: &act.settlement_name,
            slot_keys: &this.slot_keys,
        };
        add_rows(&mut staged, rules, &rows)?;
        stats.push(ActivationStats {
            key: act.key.clone(),
            index: ri,
            tail: rd.regions[ri].tail,
            pathfinding_area: area,
            poi: poi_report,
            trade_node: trade.node,
            trade_routes: trade.routes_added,
            lookup_file: overlay,
            lookup,
            border_files_written: edit.written,
            border_files_removed: edit.removed,
            sea_cells_listing,
        });
        applied.push(this);
    }

    // ---- everything computed: serialise and write (pathfinding.esf and sea_grids.esf are unchanged)
    rd.store(&mut regions_file.root)?;
    poi::store(&mut poi_file.root, &pois)?;
    write_esf(&map_dir.join("regions.esf"), &regions_file)?;
    write_esf(&map_dir.join("poi.esf"), &poi_file)?;
    write_esf(&map_dir.join("trade_routes.esf"), &trade_file)?;
    for (name, bytes) in &tgas {
        let path = map_dir.join(name);
        fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    }
    borders.flush()?;
    map.db_rows = staged.db_rows;
    map.loc_rows = staged.loc_rows;
    map.slot_mapping = staged.slot_mapping;
    Ok((applied, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_file_numbers() {
        assert_eq!(border_number("alsace.rigid_spline", "alsace"), Some(0));
        assert_eq!(border_number("alsace_12.rigid_spline", "alsace"), Some(12));
        assert_eq!(border_number("alsace_south.rigid_spline", "alsace"), None);
        assert_eq!(border_number("alsace_south_1.rigid_spline", "alsace"), None);
        assert_eq!(border_number("alsace_1.txt", "alsace"), None);
    }

    #[test]
    fn border_points_follow_the_edge() {
        let pts: Vec<_> = edge_points([0.0, 0.0], [1.0, 0.0]).collect();
        assert_eq!(pts.len(), 4);
        assert_eq!(pts[0], [0.0, BORDER_Y, 0.0]);
        assert!((pts[1][0] as f64 - 0.25 * BORDER_UNITS_PER_WORLD).abs() < 1e-6);
    }
}
