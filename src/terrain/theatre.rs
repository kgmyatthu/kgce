//! Terrain assets that follow a theatre rectangle extension and the import of externally painted terrain.
//!
//! * [`extend_canvases`]: the `climate_map` / `wind_map` rasters inside the theatre record of `regions.esf` are laid
//!   over the new rectangle at the same pixels per world unit; existing pixels keep their place.
//! * [`recanvas_tgas`]: `<theatre>_lookup.tga`, `<theatre>_map.tga` and `stratradar_<theatre>.tga` in the map folder
//!   get the same treatment (new pixels = the pixel at a sea point of the old canvas).
//! * [`import_painted_terrain`]: copy a painted supertexture / heightmap / fogmask (see `tools/terrain_paint`) over
//!   the generated `display/` assets after validating sizes and formats.
//!
//! Rules are those of `work/exp3/exp3_india_big.py` (`recanvas`, `climate_fill`, `tga_recanvas`).

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};

use super::formats as F;
use crate::{
    esf::{self, Node, Value},
    regionmap::{
        geometry::{P, Rect},
        regions::RegionData,
    },
};

/// File-name stem of a theatre's TGAs (`<stem>_lookup.tga`) for its regions.esf key (`"3"` -> `india`).
pub fn theatre_stem(key: &str) -> Option<&'static str> {
    match key {
        "1" => Some("america"),
        "2" => Some("europe"),
        "3" => Some("india"),
        _ => None,
    }
}

/// World point inside the open sea of each theatre's pre-extension canvas, used as the fill pixel of
/// [`recanvas_tgas`] (the Bay of Bengal / Arabian Sea point exp3 used for India). `None` for other theatres: pass
/// your own point.
pub fn default_sea_point(stem: &str) -> Option<[f32; 2]> {
    match stem {
        "india" => Some([600.0, 40.0]),
        _ => None,
    }
}

/// Python's `round` (half to even), which the reference script used for every canvas size and offset.
fn py_round(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        (2.0 * (x / 2.0).round()) as i64
    } else {
        r as i64
    }
}

fn check_growth(old_rect: Rect, new_rect: Rect) -> Result<()> {
    for k in 0..2 {
        ensure!(old_rect[1][k] > old_rect[0][k] && new_rect[1][k] > new_rect[0][k], "empty theatre rectangle");
    }
    ensure!(
        new_rect[0][0] <= old_rect[0][0]
            && new_rect[0][1] <= old_rect[0][1]
            && new_rect[1][0] >= old_rect[1][0]
            && new_rect[1][1] >= old_rect[1][1],
        "the new theatre rectangle {new_rect:?} does not contain the old one {old_rect:?} (extensions only grow)"
    );
    Ok(())
}

/// Geometry of an old raster re-laid over a new rectangle at the same pixels per world unit.
struct Canvas {
    w: usize,
    h: usize,
    nw: usize,
    nh: usize,
    ox: i64,
    oy: i64,
    sx: f64,
    sy: f64,
    new_min: [f64; 2],
}

impl Canvas {
    fn new(w: usize, h: usize, old: Rect, new: Rect) -> Result<Self> {
        ensure!(w > 0 && h > 0, "empty raster");
        let f = |r: Rect, i: usize, k: usize| r[i][k] as f64;
        let sx = w as f64 / (f(old, 1, 0) - f(old, 0, 0));
        let sy = h as f64 / (f(old, 1, 1) - f(old, 0, 1));
        Ok(Self {
            w,
            h,
            nw: py_round((f(new, 1, 0) - f(new, 0, 0)) * sx) as usize,
            nh: py_round((f(new, 1, 1) - f(new, 0, 1)) * sy) as usize,
            ox: py_round((f(old, 0, 0) - f(new, 0, 0)) * sx),
            oy: py_round((f(old, 0, 1) - f(new, 0, 1)) * sy),
            sx,
            sy,
            new_min: [f(new, 0, 0), f(new, 0, 1)],
        })
    }

    /// Source pixel of new pixel `(px, py)` when it lies on the old raster.
    fn source(&self, px: usize, py: usize) -> Option<(usize, usize)> {
        let (qx, qy) = (px as i64 - self.ox, py as i64 - self.oy);
        (0 <= qx && qx < self.w as i64 && 0 <= qy && qy < self.h as i64).then_some((qx as usize, qy as usize))
    }

    /// New raster (bottom-left origin, `bpx` bytes per pixel): old pixels copied; the others nearest old pixel
    /// (`nearest`) or zero (filled by the caller through [`Canvas::source`]).
    fn build(&self, data: &[u8], bpx: usize, nearest: bool) -> Vec<u8> {
        let mut out = vec![0u8; self.nw * self.nh * bpx];
        for py in 0..self.nh {
            for px in 0..self.nw {
                let src = match self.source(px, py) {
                    Some(s) => s,
                    None if nearest => {
                        let qx = (px as i64 - self.ox).clamp(0, self.w as i64 - 1) as usize;
                        let qy = (py as i64 - self.oy).clamp(0, self.h as i64 - 1) as usize;
                        (qx, qy)
                    }
                    None => continue,
                };
                let s = (src.1 * self.w + src.0) * bpx;
                let d = (py * self.nw + px) * bpx;
                out[d..d + bpx].copy_from_slice(&data[s..s + bpx]);
            }
        }
        out
    }

    /// World coordinates of the centre of new pixel `(px, py)`.
    fn world(&self, px: usize, py: usize) -> P {
        [self.new_min[0] + (px as f64 + 0.5) / self.sx, self.new_min[1] + (py as f64 + 0.5) / self.sy]
    }
}

// ------------------------------------------------------------------------------------------ climate fill

struct LandRing {
    region: usize,
    area: usize,
    outer: bool,
    pts: Vec<P>,
    bb: (P, P),
    bbox_area: f64,
}

/// "Which region is at this point" of the reference script (`locate`): the outer ring with the smallest bounding box
/// area that contains the point, unless a hole of the same area contains it. Evaluated one scanline at a time.
struct LandLocator<'a> {
    rd: &'a RegionData,
    rings: Vec<LandRing>,
}

struct Scanline {
    /// (ring index, sorted x of the edge crossings of the scanline)
    outer: Vec<(usize, Vec<f64>)>,
    holes: Vec<(usize, Vec<f64>)>,
}

fn odd_beyond(crossings: &[f64], x: f64) -> bool {
    (crossings.len() - crossings.partition_point(|&c| c <= x)) % 2 == 1
}

impl<'a> LandLocator<'a> {
    fn new(rd: &'a RegionData) -> Self {
        let mut rings = Vec::new();
        for (region, r) in rd.regions.iter().enumerate() {
            for (area, a) in r.areas.iter().enumerate() {
                for o in &a.outlines {
                    let pts: Vec<P> = o.ring.iter().map(|&i| rd.v(i)).collect();
                    if pts.is_empty() {
                        continue;
                    }
                    let mut bb = ([f64::MAX; 2], [f64::MIN; 2]);
                    for p in &pts {
                        for k in 0..2 {
                            bb.0[k] = bb.0[k].min(p[k]);
                            bb.1[k] = bb.1[k].max(p[k]);
                        }
                    }
                    let bbox_area = (bb.1[0] - bb.0[0]) * (bb.1[1] - bb.0[1]);
                    rings.push(LandRing { region, area, outer: o.outer, pts, bb, bbox_area });
                }
            }
        }
        Self { rd, rings }
    }

    fn scanline(&self, y: f64) -> Scanline {
        let mut s = Scanline { outer: Vec::new(), holes: Vec::new() };
        for (i, r) in self.rings.iter().enumerate() {
            if y < r.bb.0[1] || y > r.bb.1[1] {
                continue;
            }
            let n = r.pts.len();
            let mut xs = Vec::new();
            for k in 0..n {
                let [x1, y1] = r.pts[k];
                let [x2, y2] = r.pts[(k + 1) % n];
                if (y1 > y) != (y2 > y) {
                    xs.push(x1 + (y - y1) * (x2 - x1) / (y2 - y1));
                }
            }
            xs.sort_by(f64::total_cmp);
            if r.outer { &mut s.outer } else { &mut s.holes }.push((i, xs));
        }
        s
    }

    /// Is the point (x, scanline y) inside a land region?
    fn is_land(&self, s: &Scanline, x: f64) -> bool {
        let mut best: Option<usize> = None;
        for (i, xs) in &s.outer {
            let r = &self.rings[*i];
            if x < r.bb.0[0] || x > r.bb.1[0] || !odd_beyond(xs, x) {
                continue;
            }
            if best.is_none_or(|b| r.bbox_area < self.rings[b].bbox_area) {
                best = Some(*i);
            }
        }
        let Some(b) = best else { return false };
        let (region, area) = (self.rings[b].region, self.rings[b].area);
        if s.holes.iter().any(|(i, xs)| self.rings[*i].region == region && self.rings[*i].area == area && odd_beyond(xs, x)) {
            return false;
        }
        self.rd.regions[region].kind == "land"
    }
}

// ------------------------------------------------------------------------------------------ canvases

/// Raster sizes written by [`extend_canvases`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanvasSizes {
    pub climate: (usize, usize),
    pub wind: (usize, usize),
}

fn set_u32(node: &mut Node, v: usize) -> Result<()> {
    match node.value_mut()? {
        Value::U32(x) => {
            *x = v as u32;
            Ok(())
        }
        other => bail!("expected u32, found {other:?}"),
    }
}

fn set_array(node: &mut Node, data: Vec<u8>) -> Result<()> {
    match node.value_mut()? {
        Value::Array { data: d, .. } => {
            *d = data;
            Ok(())
        }
        other => bail!("expected packed array, found {other:?}"),
    }
}

/// Extend the climate and wind canvases of theatre `theatre` (regions.esf key, e.g. `"3"`) in
/// `<map_dir>/regions.esf` from `old_rect` to `new_rect`:
/// * `climate_map` records: 1 byte per pixel, new pixels get the climate of the point (land north of y = 250: 1,
///   other land: 7, sea or outside every region: 34; land = the innermost region polygon at the pixel centre, taken
///   from the file's own geometry, so call this BEFORE regions are clipped / activated);
/// * `wind_map`: 2 bytes per pixel, new pixels take the nearest existing pixel.
/// Pixels per world unit stay as they were (about 2.44 for climate, 0.4 for wind), sizes are rounded half-to-even.
/// The theatre rectangle itself is not touched (that is `RegionData::extend_theatre`); the file is rewritten in
/// place. Call this before the pipeline loads `regions.esf` into a `RegionData` it intends to store back.
pub fn extend_canvases(map_dir: &Path, theatre: &str, old_rect: Rect, new_rect: Rect) -> Result<CanvasSizes> {
    let path = map_dir.join("regions.esf");
    let bytes = fs::read(&path).with_context(|| format!("Read {}", path.display()))?;
    let mut file = esf::read(&bytes).with_context(|| format!("Parse {}", path.display()))?;
    let sizes = extend_canvases_in(&mut file.root, theatre, old_rect, new_rect)?;
    fs::write(&path, esf::write(&file)?).with_context(|| format!("Write {}", path.display()))?;
    Ok(sizes)
}

/// [`extend_canvases`] on an already parsed `regions.esf` root.
pub fn extend_canvases_in(root: &mut Node, theatre: &str, old_rect: Rect, new_rect: Rect) -> Result<CanvasSizes> {
    check_growth(old_rect, new_rect)?;
    let rd = RegionData::parse(root)?;
    let locator = LandLocator::new(&rd);
    let index = root
        .get("theatres")?
        .items()?
        .iter()
        .position(|it| it.first().and_then(|r| r.children().ok()).and_then(|c| c.first()).and_then(|k| k.as_str().ok()) == Some(theatre))
        .with_context(|| format!("regions.esf has no theatre {theatre}"))?;
    let record = &mut root.get_mut("theatres")?.items_mut()?[index][0];
    let mut sizes = CanvasSizes { climate: (0, 0), wind: (0, 0) };
    let mut winds = 0;
    for child in record.children_mut()? {
        match child.name() {
            Some("climate_map") => {
                let ch = child.children_mut()?;
                let (w, h) = (ch[0].as_u32()? as usize, ch[1].as_u32()? as usize);
                let data = ch[2].raw_array()?;
                ensure!(data.len() == w * h, "climate_map holds {} bytes for {w}x{h}", data.len());
                let cv = Canvas::new(w, h, old_rect, new_rect)?;
                let mut out = cv.build(data, 1, false);
                for py in 0..cv.nh {
                    if (0..cv.nw).all(|px| cv.source(px, py).is_some()) {
                        continue;
                    }
                    let scan = locator.scanline(cv.world(0, py)[1]);
                    for px in 0..cv.nw {
                        if cv.source(px, py).is_none() {
                            let [wx, wy] = cv.world(px, py);
                            out[py * cv.nw + px] = if locator.is_land(&scan, wx) { if wy > 250.0 { 1 } else { 7 } } else { 34 };
                        }
                    }
                }
                set_u32(&mut ch[0], cv.nw)?;
                set_u32(&mut ch[1], cv.nh)?;
                set_array(&mut ch[2], out)?;
                sizes.climate = (cv.nw, cv.nh);
            }
            Some("wind_map") => {
                let ch = child.children_mut()?;
                let (w, h) = (ch[0].as_u32()? as usize, ch[1].as_u32()? as usize);
                let data = ch[3].raw_array()?;
                ensure!(data.len() == w * h * 2, "wind_map holds {} bytes for {w}x{h}x2", data.len());
                let cv = Canvas::new(w, h, old_rect, new_rect)?;
                let out = cv.build(data, 2, true);
                set_u32(&mut ch[0], cv.nw)?;
                set_u32(&mut ch[1], cv.nh)?;
                set_array(&mut ch[3], out)?;
                sizes.wind = (cv.nw, cv.nh);
                winds += 1;
            }
            _ => {}
        }
    }
    ensure!(sizes.climate.0 > 0, "theatre {theatre} has no climate_map");
    ensure!(winds == 1, "theatre {theatre} has {winds} wind_map records, expected 1");
    Ok(sizes)
}

// ------------------------------------------------------------------------------------------ TGAs

/// Re-canvas an uncompressed TGA (`bytes`) from `old_rect` to `new_rect` at the same pixels per world unit; new
/// pixels copy the pixel at `sea_point` (world coordinates inside the old rectangle). The header (id field, colour
/// map, descriptor) and any footer are kept, only width and height change. Returns the new file and its size.
pub fn recanvas_tga_bytes(bytes: &[u8], old_rect: Rect, new_rect: Rect, sea_point: [f32; 2]) -> Result<(Vec<u8>, (usize, usize))> {
    check_growth(old_rect, new_rect)?;
    ensure!(bytes.len() >= 18, "TGA header truncated");
    let (id_len, cm_type, image_type) = (bytes[0] as usize, bytes[1], bytes[2]);
    let cm_len = u16::from_le_bytes([bytes[5], bytes[6]]) as usize;
    let cm_bits = bytes[7] as usize;
    let (w, h) = (u16::from_le_bytes([bytes[12], bytes[13]]) as usize, u16::from_le_bytes([bytes[14], bytes[15]]) as usize);
    let (bpp, desc) = (bytes[16] as usize, bytes[17]);
    ensure!(matches!(image_type, 1 | 2 | 3), "TGA type {image_type} is not an uncompressed image (RLE TGAs are not supported)");
    ensure!(cm_type == 0 || image_type == 1, "unexpected colour map in TGA type {image_type}");
    ensure!(bpp > 0 && bpp % 8 == 0, "unsupported TGA depth {bpp}");
    ensure!(desc & 0x10 == 0, "right-to-left TGA rows are not supported");
    let bpx = bpp / 8;
    let off = 18 + id_len + cm_len * cm_bits.div_ceil(8);
    let end = off + w * h * bpx;
    ensure!(bytes.len() >= end, "TGA pixel data truncated ({} of {end} bytes)", bytes.len());
    let bottom_up = desc & 0x20 == 0;
    let px = &bytes[off..end];
    let logical: Vec<u8> = if bottom_up {
        px.to_vec()
    } else {
        (0..h).rev().flat_map(|y| px[y * w * bpx..(y + 1) * w * bpx].iter().copied()).collect()
    };
    let cv = Canvas::new(w, h, old_rect, new_rect)?;
    let sxp = ((sea_point[0] - old_rect[0][0]) as f64 * cv.sx) as i64;
    let syp = ((sea_point[1] - old_rect[0][1]) as f64 * cv.sy) as i64;
    ensure!(
        0 <= sxp && sxp < w as i64 && 0 <= syp && syp < h as i64,
        "sea point {sea_point:?} lies outside the old theatre rectangle {old_rect:?}"
    );
    let sea = (syp as usize * w + sxp as usize) * bpx;
    let sea = logical[sea..sea + bpx].to_vec();
    let mut out = cv.build(&logical, bpx, false);
    for py in 0..cv.nh {
        for pxl in 0..cv.nw {
            if cv.source(pxl, py).is_none() {
                let d = (py * cv.nw + pxl) * bpx;
                out[d..d + bpx].copy_from_slice(&sea);
            }
        }
    }
    ensure!(cv.nw <= u16::MAX as usize && cv.nh <= u16::MAX as usize, "re-canvased TGA {}x{} exceeds 65535", cv.nw, cv.nh);
    let mut file = bytes[..off].to_vec();
    file[12..14].copy_from_slice(&(cv.nw as u16).to_le_bytes());
    file[14..16].copy_from_slice(&(cv.nh as u16).to_le_bytes());
    if bottom_up {
        file.extend_from_slice(&out);
    } else {
        for y in (0..cv.nh).rev() {
            file.extend_from_slice(&out[y * cv.nw * bpx..(y + 1) * cv.nw * bpx]);
        }
    }
    file.extend_from_slice(&bytes[end..]);
    Ok((file, (cv.nw, cv.nh)))
}

/// Re-canvas `<theatre>_lookup.tga`, `<theatre>_map.tga` and (when present) `stratradar_<theatre>.tga` in `map_dir`
/// in place. `theatre` is the TGA name stem (`india`, see [`theatre_stem`]); `sea_point` is a world point in open sea
/// inside `old_rect` ([`default_sea_point`] for India). Returns the files rewritten with their new sizes.
pub fn recanvas_tgas(
    map_dir: &Path,
    theatre: &str,
    old_rect: Rect,
    new_rect: Rect,
    sea_point: [f32; 2],
) -> Result<Vec<(String, (usize, usize))>> {
    let mut done = Vec::new();
    for (name, required) in [
        (format!("{theatre}_lookup.tga"), true),
        (format!("{theatre}_map.tga"), true),
        (format!("stratradar_{theatre}.tga"), false),
    ] {
        let path = map_dir.join(&name);
        if !path.exists() {
            ensure!(!required, "{} is missing", path.display());
            continue;
        }
        let bytes = fs::read(&path).with_context(|| format!("Read {}", path.display()))?;
        let (out, size) = recanvas_tga_bytes(&bytes, old_rect, new_rect, sea_point).with_context(|| name.clone())?;
        fs::write(&path, out).with_context(|| format!("Write {}", path.display()))?;
        done.push((name, size));
    }
    Ok(done)
}

// ------------------------------------------------------------------------------------------ painted terrain

const STP_TILE: u32 = 512;
const STP_TILE_BYTES: u32 = 0x40000;
const HEIGHTMAP_SIZE: (usize, usize) = (8192, 4096);
const FOG_SIZE: (u32, u32) = (4096, 2048);

/// Files of a painted-terrain folder, relative to both the folder and the map folder.
const SUPERTEXTURE: [&str; 2] = ["display/supertexture/supertexture.stpi", "display/supertexture/supertexture.stpd"];
const HEIGHTMAP: &str = "display/heightmap/heightmap.tga";
const FOGMASK: &str = "display/fogmask/fogmask.dds";

fn validate_supertexture(stpi: &Path, stpd: &Path, existing: Option<&Path>) -> Result<()> {
    let b = fs::read(stpi).with_context(|| format!("Read {}", stpi.display()))?;
    let idx = F::parse_stpi(&b).with_context(|| format!("{}", stpi.display()))?;
    ensure!(
        idx.tile == STP_TILE && idx.tile_bytes == STP_TILE_BYTES,
        "{}: tile layout {}/{:#x}, expected {STP_TILE}/{STP_TILE_BYTES:#x}",
        stpi.display(),
        idx.tile,
        idx.tile_bytes
    );
    ensure!(idx.levels > 0 && idx.layers.len() == idx.levels as usize, "{}: no levels", stpi.display());
    let len = fs::metadata(stpd).with_context(|| format!("Stat {}", stpd.display()))?.len();
    let mut tiles = 0usize;
    for (li, l) in idx.layers.iter().enumerate() {
        for e in &l.entries {
            ensure!(
                e[0] as u64 + e[1] as u64 <= len && e[1] > 0,
                "{}: level {li} tile at {}+{} lies outside {} ({len} bytes)",
                stpi.display(),
                e[0],
                e[1],
                stpd.display()
            );
            ensure!(e[2] == STP_TILE_BYTES, "{}: level {li} tile raw size {:#x}", stpi.display(), e[2]);
        }
        tiles += l.entries.len();
    }
    ensure!(tiles > 0, "{}: no tiles", stpi.display());
    // every stored tile is 8 chunks [u32 csize][u32 32768][zlib]; check the first one of the file
    let first = idx.layers[0].entries[0];
    let mut f = fs::File::open(stpd)?;
    std::io::Seek::seek(&mut f, std::io::SeekFrom::Start(first[0] as u64))?;
    let mut head = [0u8; 8];
    f.read_exact(&mut head)?;
    ensure!(
        u32::from_le_bytes(head[4..8].try_into().unwrap()) == 0x8000,
        "{}: first tile is not stored as 32 KiB zlib chunks",
        stpd.display()
    );
    if let Some(cur) = existing.filter(|p| p.exists()) {
        let old = F::parse_stpi(&fs::read(cur)?)?;
        ensure!(
            (old.levels, old.width, old.height) == (idx.levels, idx.width, idx.height)
                && old.layers.iter().zip(&idx.layers).all(|(a, b)| (a.tx, a.ty) == (b.tx, b.ty)),
            "{}: supertexture is {}x{} with {} levels, the generated one is {}x{} with {} levels (tile grids must be equal)",
            stpi.display(),
            idx.width,
            idx.height,
            idx.levels,
            old.width,
            old.height,
            old.levels
        );
    }
    Ok(())
}

fn validate_heightmap(path: &Path) -> Result<()> {
    let mut head = vec![0u8; 18];
    fs::File::open(path).with_context(|| format!("Open {}", path.display()))?.read_exact(&mut head)?;
    let h = F::tga_header(&head)?;
    ensure!(
        (h.width, h.height, h.bpp, h.image_type) == (HEIGHTMAP_SIZE.0, HEIGHTMAP_SIZE.1, 8, 3),
        "{}: {}x{} {}-bit type {} TGA, expected {}x{} 8-bit greyscale (type 3)",
        path.display(),
        h.width,
        h.height,
        h.bpp,
        h.image_type,
        HEIGHTMAP_SIZE.0,
        HEIGHTMAP_SIZE.1
    );
    let len = fs::metadata(path)?.len() as usize;
    ensure!(
        len >= 18 + h.id_len + h.width * h.height,
        "{}: pixel data truncated ({len} bytes)",
        path.display()
    );
    Ok(())
}

fn validate_fogmask(path: &Path) -> Result<()> {
    let mut head = vec![0u8; 128];
    fs::File::open(path).with_context(|| format!("Open {}", path.display()))?.read_exact(&mut head)?;
    ensure!(&head[0..4] == b"DDS ", "{}: not a DDS file", path.display());
    let (h, w) = (F::u32_at(&head, 12)?, F::u32_at(&head, 16)?);
    ensure!((w, h) == FOG_SIZE, "{}: {w}x{h}, expected {}x{}", path.display(), FOG_SIZE.0, FOG_SIZE.1);
    ensure!(
        F::u32_at(&head, 88)? == 8 && F::u32_at(&head, 80)? & 0x20000 != 0,
        "{}: not an 8-bit luminance DDS",
        path.display()
    );
    ensure!(
        fs::metadata(path)?.len() == 128 + (w * h) as u64,
        "{}: unexpected size {}",
        path.display(),
        fs::metadata(path)?.len()
    );
    Ok(())
}

/// Import painted terrain from `dir` (a folder shaped like a map folder: `display/supertexture/supertexture.{stpd,stpi}`,
/// `display/heightmap/heightmap.tga`, `display/fogmask/fogmask.dds`, as written by `tools/terrain_paint`) over the
/// generated `<map_dir>/display/**`. Every present item is validated first (supertexture pair: tile layout, entries
/// inside the `.stpd`, tile grids equal to the generated supertexture when one exists; heightmap: 8192x4096 8-bit
/// greyscale TGA; fogmask: 4096x2048 8-bit luminance DDS); nothing is copied unless all of them pass. At least one
/// item must be present. Returns the copied files relative to the map folder.
pub fn import_painted_terrain(dir: &Path, map_dir: &Path) -> Result<Vec<String>> {
    let has = |rel: &str| dir.join(rel).is_file();
    let pair = (has(SUPERTEXTURE[0]), has(SUPERTEXTURE[1]));
    ensure!(
        pair.0 == pair.1,
        "{}: supertexture.stpi and supertexture.stpd must both be present",
        dir.join("display/supertexture").display()
    );
    let mut items: Vec<&str> = Vec::new();
    if pair.0 {
        validate_supertexture(&dir.join(SUPERTEXTURE[0]), &dir.join(SUPERTEXTURE[1]), Some(&map_dir.join(SUPERTEXTURE[0])))?;
        items.extend(SUPERTEXTURE);
    }
    if has(HEIGHTMAP) {
        validate_heightmap(&dir.join(HEIGHTMAP))?;
        items.push(HEIGHTMAP);
    }
    if has(FOGMASK) {
        validate_fogmask(&dir.join(FOGMASK))?;
        items.push(FOGMASK);
    }
    ensure!(
        !items.is_empty(),
        "{} contains no painted terrain (expected display/supertexture/supertexture.{{stpd,stpi}}, display/heightmap/heightmap.tga or display/fogmask/fogmask.dds)",
        dir.display()
    );
    let mut copied = Vec::new();
    for rel in items {
        let to: PathBuf = map_dir.join(rel);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
        }
        fs::copy(dir.join(rel), &to).with_context(|| format!("Copy {} to {}", rel, to.display()))?;
        copied.push(rel.to_string());
    }
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORK: &str = "F:/NTW2/work";
    const OLD: Rect = [[440.0, 20.0], [680.0, 280.0]];
    const NEW: Rect = [[440.0, -80.0], [790.0, 300.0]];

    fn exists(p: &str) -> bool {
        let ok = Path::new(p).exists();
        if !ok {
            eprintln!("skipped: {p} not available");
        }
        ok
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kgc_terrain_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn python_rounding_is_half_even() {
        assert_eq!([py_round(0.5), py_round(1.5), py_round(2.5), py_round(-0.5), py_round(2.4), py_round(2.6)], [0, 2, 2, 0, 2, 3]);
    }

    #[test]
    fn canvases_equal_exp3_records() {
        let (van, exp3) = (
            format!("{WORK}/exp1/backup/campaign_maps/nap_world/regions.esf"),
            format!("{WORK}/exp3/out/data/campaign_maps/nap_world/regions.esf"),
        );
        if !exists(&van) || !exists(&exp3) {
            return;
        }
        let dir = scratch("canvas");
        fs::copy(&van, dir.join("regions.esf")).unwrap();
        let sizes = extend_canvases(&dir, "3", OLD, NEW).unwrap();
        eprintln!("canvas sizes {sizes:?}");
        let got = esf::read(&fs::read(dir.join("regions.esf")).unwrap()).unwrap();
        let want = esf::read(&fs::read(&exp3).unwrap()).unwrap();
        let pick = |f: &esf::Esf| -> Vec<Node> {
            let t = &f.root.get("theatres").unwrap().items().unwrap();
            let rec = t.iter().map(|it| &it[0]).find(|r| r.children().unwrap()[0].as_str().unwrap() == "3").unwrap();
            rec.children().unwrap().iter().filter(|c| matches!(c.name(), Some("climate_map" | "wind_map"))).cloned().collect()
        };
        let (g, w) = (pick(&got), pick(&want));
        assert!(!g.is_empty() && g.len() == w.len());
        assert_eq!(g, w, "climate/wind records differ from exp3");
        // the original canvas content survives (the old rect is at +0, +round(100 * sy))
        let orig = esf::read(&fs::read(&van).unwrap()).unwrap();
        let o = pick(&orig);
        let cm = |n: &Node| (n.children().unwrap()[0].as_u32().unwrap() as usize, n.children().unwrap()[2].raw_array().unwrap().to_vec());
        let (ow, od) = cm(&o[0]);
        let (nw, nd) = cm(&g[0]);
        let oy = py_round(100.0 * (od.len() / ow) as f64 / 260.0) as usize;
        for y in [0, od.len() / ow / 2, od.len() / ow - 1] {
            assert_eq!(od[y * ow..(y + 1) * ow], nd[(y + oy) * nw..(y + oy) * nw + ow]);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tgas_equal_exp3_outside_the_later_paint() {
        let src = format!("{WORK}/exp1/backup/campaign_maps/nap_world");
        let exp3 = format!("{WORK}/exp3/out/data/campaign_maps/nap_world");
        if !exists(&format!("{src}/india_lookup.tga")) || !exists(&format!("{exp3}/india_lookup.tga")) {
            return;
        }
        let dir = scratch("tga");
        for n in ["india_lookup.tga", "india_map.tga", "stratradar_india.tga"] {
            fs::copy(format!("{src}/{n}"), dir.join(n)).unwrap();
        }
        let done = recanvas_tgas(&dir, "india", OLD, NEW, default_sea_point("india").unwrap()).unwrap();
        eprintln!("{done:?}");
        assert_eq!(done[0], ("india_lookup.tga".to_string(), (404, 438)));
        for (name, _) in &done {
            let (got, want) = (fs::read(dir.join(name)).unwrap(), fs::read(format!("{exp3}/{name}")).unwrap());
            assert_eq!(got.len(), want.len(), "{name}");
            assert_eq!(got, want, "{name} differs from exp3");
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tga_recanvas_top_down_and_validation() {
        // 2x2 RGB-less grey TGA, top-down: rows (top) [1,2] / [3,4]; old rect 0..2 x 0..2, grows by 1 unit to the left/bottom
        let mut b = vec![0u8; 18];
        b[2] = 3;
        b[12] = 2;
        b[14] = 2;
        b[16] = 8;
        b[17] = 0x20;
        b.extend_from_slice(&[1, 2, 3, 4]);
        let (out, size) = recanvas_tga_bytes(&b, [[0.0, 0.0], [2.0, 2.0]], [[-1.0, -1.0], [2.0, 2.0]], [1.5, 0.5]).unwrap();
        assert_eq!(size, (3, 3));
        // sea pixel = logical (1, 0) = bottom row second pixel = 4; top-down rows
        assert_eq!(&out[18..], &[4, 1, 2, 4, 3, 4, 4, 4, 4]);
        assert!(recanvas_tga_bytes(&b, [[0.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [1.0, 1.0]], [1.5, 0.5]).is_err());
        assert!(recanvas_tga_bytes(&b, [[0.0, 0.0], [2.0, 2.0]], [[-1.0, -1.0], [2.0, 2.0]], [5.0, 0.5]).is_err());
        b[2] = 10;
        assert!(recanvas_tga_bytes(&b, [[0.0, 0.0], [2.0, 2.0]], [[-1.0, -1.0], [2.0, 2.0]], [1.5, 0.5]).is_err());
    }

    #[test]
    fn import_validates_and_copies() {
        let painted = format!("{WORK}/exp2/out/data/campaign_maps/nap_world");
        if !exists(&format!("{painted}/display/supertexture/supertexture.stpi")) {
            return;
        }
        let map = scratch("import");
        let copied = import_painted_terrain(Path::new(&painted), &map).unwrap();
        assert!(copied.iter().any(|c| c.ends_with("supertexture.stpd")) && copied.iter().any(|c| c.ends_with("heightmap.tga")));
        for c in &copied {
            assert_eq!(fs::metadata(map.join(c)).unwrap().len(), fs::metadata(format!("{painted}/{c}")).unwrap().len(), "{c}");
        }
        // empty folder, broken pairs and wrong sizes are refused
        let empty = scratch("import_empty");
        assert!(import_painted_terrain(&empty, &map).is_err());
        fs::create_dir_all(empty.join("display/heightmap")).unwrap();
        let mut bad = vec![0u8; 18];
        bad[2] = 3;
        bad[12..14].copy_from_slice(&100u16.to_le_bytes());
        bad[14..16].copy_from_slice(&100u16.to_le_bytes());
        bad[16] = 8;
        bad.extend(vec![0u8; 10_000]);
        fs::write(empty.join(HEIGHTMAP), bad).unwrap();
        let err = import_painted_terrain(&empty, &map).unwrap_err().to_string();
        assert!(err.contains("8192x4096"), "{err}");
        fs::create_dir_all(empty.join("display/supertexture")).unwrap();
        fs::write(empty.join(SUPERTEXTURE[0]), b"x").unwrap();
        assert!(import_painted_terrain(&empty, &map).is_err());
        fs::remove_dir_all(map).unwrap();
        fs::remove_dir_all(empty).unwrap();
    }
}
