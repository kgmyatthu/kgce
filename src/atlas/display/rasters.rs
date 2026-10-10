//! Map-aligned single-resolution rasters of the connected atlas: the heightmap (node-registered bilinear
//! reconstruction), `metadata.dat` class bytes, the L8 fog/snow DDS masks (nearest) and the DXT5 mip-chained detail and
//! snow masks. Port of `L8Asset` / `DxtMaskAsset` of the accepted `raster_scale.py`.

use super::{
    dxt,
    mapping::{Mapping, touch_grid},
    par_chunks_mut,
};
use crate::atlas::Plan;
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};
use std::collections::HashMap;

/// Pixels per world unit of every map-aligned raster family (measured inventory of the accepted POC).
pub fn density(kind: &str) -> f64 {
    match kind {
        "heightmap" => 3.2,
        "metadata" | "fogmask" | "snow" => 1.6,
        "detail_mask" | "snowmask" => 0.8,
        "supertexture" => 25.6,
        _ => f64::NAN,
    }
}

pub fn check_dims(plan: &Plan, kind: &str, rel: &str, w: usize, h: usize) -> Result<()> {
    let swx = plan.source_world[1][0] - plan.source_world[0][0];
    let swy = plan.source_world[1][1] - plan.source_world[0][1];
    let (ew, eh) = (density(kind) * swx, density(kind) * swy);
    ensure!(
        (w as f64 - ew).abs() <= 1e-6 && (h as f64 - eh).abs() <= 1e-6,
        "{rel}: {w}x{h} is not the expected {ew}x{eh} for {} px/unit over the source world; not map-aligned?",
        density(kind)
    );
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum L8Kind {
    Heightmap,
    Metadata,
    Fogmask,
    Snow,
}

impl L8Kind {
    pub fn name(self) -> &'static str {
        match self {
            L8Kind::Heightmap => "heightmap",
            L8Kind::Metadata => "metadata",
            L8Kind::Fogmask => "fogmask",
            L8Kind::Snow => "snow",
        }
    }

    pub fn rel(self) -> &'static str {
        match self {
            L8Kind::Heightmap => "display/heightmap/heightmap.tga",
            L8Kind::Metadata => "metadata.dat",
            L8Kind::Fogmask => "display/fogmask/fogmask.dds",
            L8Kind::Snow => "display/snow/winter.dds",
        }
    }
}

fn u32le(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

/// `heightmap.tga` / `metadata.dat` / L8 DDS: header bytes plus an 8-bit raster with explicit row orientation.
pub struct L8 {
    pub kind: L8Kind,
    header: Vec<u8>,
    pub w: usize,
    pub h: usize,
    bottom_up: bool,
    /// Source raster, rows top-first (engine memory image).
    pub top: Vec<u8>,
    pub map: Mapping,
}

impl L8 {
    pub fn load(kind: L8Kind, bytes: &[u8], plan: &Plan) -> Result<Self> {
        let rel = kind.rel();
        let (w, h, header, bottom_up, data): (usize, usize, Vec<u8>, bool, &[u8]) = match kind {
            L8Kind::Heightmap => {
                ensure!(bytes.len() >= 18, "{rel}: TGA header truncated");
                let (id_len, cmtype, imgtype, bpp, desc) = (bytes[0] as usize, bytes[1], bytes[2], bytes[16], bytes[17]);
                let (w, h) = (
                    u16::from_le_bytes([bytes[12], bytes[13]]) as usize,
                    u16::from_le_bytes([bytes[14], bytes[15]]) as usize,
                );
                ensure!(
                    imgtype == 3 && bpp == 8 && cmtype == 0 && desc & 0x10 == 0,
                    "{rel}: only uncompressed 8-bit greyscale TGA is supported"
                );
                let off = 18 + id_len;
                ensure!(
                    bytes.len() == off + w * h,
                    "{rel}: unexpected size {} (trailing bytes after TGA pixel data; refusing to guess)",
                    bytes.len()
                );
                (w, h, bytes[..off].to_vec(), desc & 0x20 == 0, &bytes[off..])
            }
            L8Kind::Metadata => {
                ensure!(bytes.len() >= 12 && &bytes[0..4] == b"MTGI", "{rel}: magic is not MTGI");
                let (w, h) = (u32le(bytes, 4) as usize, u32le(bytes, 8) as usize);
                ensure!(bytes.len() == 12 + w * h, "{rel}: size != 12 + w*h class bytes; unknown extra data");
                (w, h, bytes[..12].to_vec(), false, &bytes[12..])
            }
            L8Kind::Fogmask | L8Kind::Snow => {
                ensure!(bytes.len() >= 128 && &bytes[0..4] == b"DDS ", "{rel}: not a DDS");
                let (h, w) = (u32le(bytes, 12) as usize, u32le(bytes, 16) as usize);
                ensure!(
                    u32le(bytes, 4) == 124
                        && u32le(bytes, 76) == 32
                        && u32le(bytes, 8) == 0x81007
                        && u32le(bytes, 20) as usize == w * h
                        && u32le(bytes, 80) == 0x20000
                        && u32le(bytes, 88) == 8
                        && u32le(bytes, 92) == 0xFF
                        && matches!(u32le(bytes, 28), 0 | 1)
                        && bytes.len() == 128 + w * h,
                    "{rel}: unsupported L8 DDS layout"
                );
                (w, h, bytes[..128].to_vec(), false, &bytes[128..])
            }
        };
        check_dims(plan, kind.name(), rel, w, h)?;
        let top = if bottom_up {
            let mut t = Vec::with_capacity(data.len());
            for y in (0..h).rev() {
                t.extend_from_slice(&data[y * w..(y + 1) * w]);
            }
            t
        } else {
            data.to_vec()
        };
        Ok(Self { kind, header, w, h, bottom_up, top, map: Mapping::new(plan, w, h, kind.name())? })
    }

    fn out_header(&self) -> Result<Vec<u8>> {
        let mut h = self.header.clone();
        let (w, hh) = (self.map.out_w as usize, self.map.out_h as usize);
        match self.kind {
            L8Kind::Heightmap => {
                ensure!(w <= 0xFFFF && hh <= 0xFFFF, "TGA dimensions exceed u16");
                h[12..14].copy_from_slice(&(w as u16).to_le_bytes());
                h[14..16].copy_from_slice(&(hh as u16).to_le_bytes());
            }
            L8Kind::Metadata => {
                h[4..8].copy_from_slice(&(w as u32).to_le_bytes());
                h[8..12].copy_from_slice(&(hh as u32).to_le_bytes());
            }
            L8Kind::Fogmask | L8Kind::Snow => {
                h[12..16].copy_from_slice(&(hh as u32).to_le_bytes());
                h[16..20].copy_from_slice(&(w as u32).to_le_bytes());
                h[20..24].copy_from_slice(&((w * hh) as u32).to_le_bytes());
            }
        }
        Ok(h)
    }

    /// Complete transformed file.
    pub fn render(&self) -> Result<Vec<u8>> {
        let (ow, oh) = (self.map.out_w as usize, self.map.out_h as usize);
        let top = if self.kind == L8Kind::Heightmap {
            bilinear_render(&self.top, self.w, &self.map, 0)
        } else {
            nearest_render(&self.top, self.w, &self.map, 0)
        };
        let mut out = self.out_header()?;
        out.reserve(ow * oh);
        if self.bottom_up {
            for y in (0..oh).rev() {
                out.extend_from_slice(&top[y * ow..(y + 1) * ow]);
            }
        } else {
            out.extend_from_slice(&top);
        }
        Ok(out)
    }

    /// Source bytes that are not representable in the output (outside every crop) and differ from the fill value.
    pub fn dropped_nonfill(&self) -> usize {
        let mut covered = vec![false; self.w * self.h];
        for r in &self.map.rects {
            for y in r.sy0..r.sy1 {
                covered[(y as usize) * self.w + r.sx0 as usize..(y as usize) * self.w + r.sx1 as usize].fill(true);
            }
        }
        self.top.iter().zip(&covered).filter(|&(&v, &c)| v != 0 && !c).count()
    }

    /// Independent check of a rendered heightmap on sampled rows: every output pixel lies within the min/max of its
    /// four source neighbours (no invented extrema) and every pixel that sits exactly on a source node equals it.
    pub fn check_bilinear(&self, file: &[u8]) -> Result<Value> {
        ensure!(self.kind == L8Kind::Heightmap, "bilinear check applies to the heightmap only");
        let (ow, oh) = (self.map.out_w as usize, self.map.out_h as usize);
        let hl = self.header.len();
        ensure!(file.len() == hl + ow * oh, "heightmap output has the wrong size");
        let at = |y: usize, x: usize| file[hl + (if self.bottom_up { oh - 1 - y } else { y }) * ow + x];
        let s = self.map.s;
        let (mut checked, mut overshoot, mut anchors, mut anchor_bad) = (0usize, 0usize, 0usize, 0usize);
        for (ri, r) in self.map.rects.iter().enumerate() {
            let mut rows: Vec<i64> = (r.oy0..r.oy1).step_by(5).collect();
            rows.push(r.oy1 - 1);
            for y in rows {
                let yv = ((y as f64 - r.ky) / s).max(r.sy0 as f64).min((r.sy1 - 1) as f64);
                let (r0, fy) = (yv.floor() as i64, yv - yv.floor());
                let r1 = (r0 + 1).min(r.sy1 - 1);
                for x in r.ox0..r.ox1 {
                    // pixels rewritten by a later rectangle belong to that rectangle's own check
                    if self.map.rects[ri + 1..].iter().any(|l| x >= l.ox0 && x < l.ox1 && y >= l.oy0 && y < l.oy1) {
                        continue;
                    }
                    let xv = ((x as f64 - r.kx) / s).max(r.sx0 as f64).min((r.sx1 - 1) as f64);
                    let (c0, fx) = (xv.floor() as i64, xv - xv.floor());
                    let c1 = (c0 + 1).min(r.sx1 - 1);
                    let n = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)].map(|(a, b)| self.top[a as usize * self.w + b as usize]);
                    let got = at(y as usize, x as usize);
                    checked += 1;
                    if got < *n.iter().min().unwrap() || got > *n.iter().max().unwrap() {
                        overshoot += 1;
                    }
                    if fx == 0.0 && fy == 0.0 {
                        anchors += 1;
                        anchor_bad += (got != n[0]) as usize;
                    }
                }
            }
        }
        ensure!(overshoot == 0, "heightmap: {overshoot}/{checked} sampled pixels lie outside the min/max of their source neighbours");
        ensure!(anchor_bad == 0, "heightmap: {anchor_bad}/{anchors} native anchor samples differ from the source");
        Ok(json!({"pixels_checked": checked, "overshoot_pixels": overshoot, "anchor_samples": anchors, "anchor_mismatches": anchor_bad}))
    }

    pub fn report(&self) -> Value {
        json!({
            "rel": self.kind.rel(),
            "source_px": [self.w, self.h],
            "output_px": [self.map.out_w, self.map.out_h],
            "orientation": if self.bottom_up { "bottom-up" } else { "top-down" },
            "resampler": if self.kind == L8Kind::Heightmap { "node-registered bilinear, round half up" } else { "nearest" },
            "fill_value": 0,
            "source_nonfill_px_dropped_outside_crops": self.dropped_nonfill(),
            "mapping": self.map.report(),
        })
    }
}

fn nearest_render(top: &[u8], w: usize, map: &Mapping, fill: u8) -> Vec<u8> {
    let (ow, oh) = (map.out_w as usize, map.out_h as usize);
    let mut out = vec![fill; ow * oh];
    let cols: Vec<Vec<usize>> = map
        .rects
        .iter()
        .map(|r| (r.ox0..r.ox1).map(|ox| map.src_col(r, ox) as usize).collect())
        .collect();
    par_chunks_mut(&mut out, ow, |y, row| {
        let y = y as i64;
        for (ri, r) in map.rects.iter().enumerate() {
            if y < r.oy0 || y >= r.oy1 {
                continue;
            }
            let sy = map.src_row(r, y) as usize;
            let src = &top[sy * w..(sy + 1) * w];
            for (i, ox) in (r.ox0..r.ox1).enumerate() {
                row[ox as usize] = src[cols[ri][i]];
            }
        }
    });
    out
}

struct Col {
    c0: usize,
    c1: usize,
    fx: f64,
}

/// Convex bilinear reconstruction at inverse affine nodes (texel `i` sits AT coordinate `i`), clamped to each source
/// crop; float64 weights, round half up, evaluated at the unrounded world-to-pixel translation (`bilinear_rows`).
fn bilinear_render(top: &[u8], w: usize, map: &Mapping, fill: u8) -> Vec<u8> {
    let (ow, oh) = (map.out_w as usize, map.out_h as usize);
    let mut out = vec![fill; ow * oh];
    let s = map.s;
    let cols: Vec<Vec<Col>> = map
        .rects
        .iter()
        .map(|r| {
            (r.ox0..r.ox1)
                .map(|ox| {
                    let x = ((ox as f64 - r.kx) / s).max(r.sx0 as f64).min((r.sx1 - 1) as f64);
                    let c0 = x.floor() as i64;
                    let c1 = (c0 + 1).min(r.sx1 - 1);
                    Col { c0: c0 as usize, c1: c1 as usize, fx: x - c0 as f64 }
                })
                .collect()
        })
        .collect();
    par_chunks_mut(&mut out, ow, |y, row| {
        let y = y as i64;
        for (ri, r) in map.rects.iter().enumerate() {
            if y < r.oy0 || y >= r.oy1 {
                continue;
            }
            let yv = ((y as f64 - r.ky) / s).max(r.sy0 as f64).min((r.sy1 - 1) as f64);
            let r0 = yv.floor() as i64;
            let r1 = (r0 + 1).min(r.sy1 - 1);
            let fy = yv - r0 as f64;
            let (row0, row1) = (&top[r0 as usize * w..][..w], &top[r1 as usize * w..][..w]);
            for (i, ox) in (r.ox0..r.ox1).enumerate() {
                let c = &cols[ri][i];
                let mut a = row0[c.c0] as f64 * (1.0 - c.fx);
                a += row0[c.c1] as f64 * c.fx;
                let mut b = row1[c.c0] as f64 * (1.0 - c.fx);
                b += row1[c.c1] as f64 * c.fx;
                let v = a * (1.0 - fy) + b * fy;
                row[ox as usize] = (v + 0.5 + 1e-9).floor() as u8;
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------- DXT5 masks

pub fn mip_count(w: usize, h: usize) -> usize {
    (w.max(h) as f64).log2().floor() as usize + 1
}

pub fn dxt_chain_bytes(w: usize, h: usize, mips: usize) -> usize {
    let (mut total, mut ww, mut hh) = (0, w, h);
    for _ in 0..mips {
        total += ww.div_ceil(4).max(1) * hh.div_ceil(4).max(1) * 16;
        ww = (ww >> 1).max(1);
        hh = (hh >> 1).max(1);
    }
    total
}

/// Streaming "most frequent 16-byte block" (ties: smallest bytes, like `np.unique` + `argmax`).
#[derive(Default)]
pub struct ModeCounter {
    counts: HashMap<[u8; 16], usize>,
    n: usize,
}

impl ModeCounter {
    pub fn add(&mut self, block: &[u8]) {
        *self.counts.entry(block.try_into().expect("16-byte block")).or_default() += 1;
        self.n += 1;
    }

    pub fn total(&self) -> usize {
        self.n
    }

    /// `(block, fraction, distinct)`.
    pub fn finish(&self) -> Option<([u8; 16], f64, usize)> {
        let best = self.counts.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))?;
        Some((*best.0, *best.1 as f64 / self.n as f64, self.counts.len()))
    }
}

pub struct DxtMask {
    pub rel: String,
    header: Vec<u8>,
    pub w: usize,
    pub h: usize,
    mips: usize,
    blocks0: Vec<u8>,
    pub map: Mapping,
}

impl DxtMask {
    pub fn load(kind: &str, rel: &str, bytes: &[u8], plan: &Plan) -> Result<Self> {
        ensure!(bytes.len() >= 128 && &bytes[0..4] == b"DDS ", "{rel}: not a DDS");
        let (h, w, mips) = (u32le(bytes, 12) as usize, u32le(bytes, 16) as usize, u32le(bytes, 28) as usize);
        ensure!(
            u32le(bytes, 4) == 124 && u32le(bytes, 76) == 32 && u32le(bytes, 80) & 4 != 0 && &bytes[84..88] == b"DXT5",
            "{rel}: expected a DXT5 DDS"
        );
        ensure!(
            mips == mip_count(w, h),
            "{rel}: expected a complete DXT5 mip chain, got mips={mips} for {w}x{h}"
        );
        ensure!(
            bytes.len() == 128 + dxt_chain_bytes(w, h, mips),
            "{rel}: DXT5 size {} != header chain {}",
            bytes.len(),
            128 + dxt_chain_bytes(w, h, mips)
        );
        check_dims(plan, kind, rel, w, h)?;
        let map = Mapping::new(plan, w, h, rel)?;
        ensure!(map.out_w % 4 == 0 && map.out_h % 4 == 0, "{rel}: output dims not multiples of 4");
        Ok(Self {
            rel: rel.to_string(),
            header: bytes[..128].to_vec(),
            w,
            h,
            mips,
            blocks0: bytes[128..128 + (h / 4) * (w / 4) * 16].to_vec(),
            map,
        })
    }

    /// Mode block among blocks that lie outside every crop (restricted to source sea blocks when the heightmap allows).
    fn choose_fill(&self, heightmap: Option<(&[u8], usize, usize)>) -> Result<(Option<[u8; 16]>, Value)> {
        let m = &self.map;
        if m.covers_output() {
            return Ok((None, json!({"needed": false})));
        }
        let (bh, bw) = (self.h / 4, self.w / 4);
        let touched = touch_grid(&m.source_boxes(), 0, 0, bh, bw, 4);
        let mut outside: Vec<bool> = touched.iter().map(|t| !t).collect();
        let mut info = json!({"needed": true, "blocks_outside_crops": outside.iter().filter(|&&o| o).count()});
        if let Some((hm, hw, hh)) = heightmap {
            let (fy, fx) = (hh / bh, hw / bw);
            if fy > 0 && fx > 0 && hh == bh * fy && hw == bw * fx {
                let mut sea = vec![false; bh * bw];
                for by in 0..bh {
                    for bx in 0..bw {
                        let mut mx = 0u8;
                        for dy in 0..fy {
                            let row = (by * fy + dy) * hw + bx * fx;
                            mx = mx.max(*hm[row..row + fx].iter().max().unwrap());
                        }
                        sea[by * bw + bx] = mx == 0;
                    }
                }
                if outside.iter().zip(&sea).any(|(&o, &s)| o && s) {
                    for (o, s) in outside.iter_mut().zip(&sea) {
                        *o &= *s;
                    }
                    info["restricted_to_source_sea_blocks"] = json!(outside.iter().filter(|&&o| o).count());
                }
            }
        }
        if !outside.iter().any(|&o| o) {
            bail!("{}: output has uncovered area but the source has no block outside all crops", self.rel);
        }
        let mut counter = ModeCounter::default();
        for (i, _) in outside.iter().enumerate().filter(|&(_, &o)| o) {
            counter.add(&self.blocks0[i * 16..i * 16 + 16]);
        }
        let (blk, frac, distinct) = counter.finish().expect("non-empty pool");
        info["fraction_of_candidates"] = json!(frac);
        info["distinct_candidate_blocks"] = json!(distinct);
        info["fill_rgba"] = json!(dxt::decode_block(&blk)[0]);
        Ok((Some(blk), info))
    }

    fn out_header(&self, out_mips: usize) -> Vec<u8> {
        let mut h = self.header.clone();
        let (w, hh) = (self.map.out_w as usize, self.map.out_h as usize);
        h[12..16].copy_from_slice(&(hh as u32).to_le_bytes());
        h[16..20].copy_from_slice(&(w as u32).to_le_bytes());
        h[20..24].copy_from_slice(&(((w / 4) * (hh / 4) * 16) as u32).to_le_bytes());
        h[28..32].copy_from_slice(&(out_mips as u32).to_le_bytes());
        h
    }

    /// Complete transformed DDS: nearest-sampled level 0 (fill blocks kept verbatim) and re-derived, re-encoded mips.
    pub fn render(&self, heightmap: Option<(&[u8], usize, usize)>) -> Result<(Vec<u8>, Value)> {
        let m = &self.map;
        let (ow, oh) = (m.out_w as usize, m.out_h as usize);
        let (fill, fill_info) = self.choose_fill(heightmap)?;
        let src_px = dxt::blocks_to_pixels(&self.blocks0, self.h / 4, self.w / 4);
        let fill_px = match &fill {
            Some(b) => dxt::decode_block(b),
            None => [[0u8; 4]; 16],
        };
        let mut canvas = vec![0u8; ow * oh * 4];
        for y in 0..oh {
            for x in 0..ow {
                let o = (y * ow + x) * 4;
                canvas[o..o + 4].copy_from_slice(&fill_px[(y & 3) * 4 + (x & 3)]);
            }
        }
        for r in &m.rects {
            let cols: Vec<usize> = (r.ox0..r.ox1).map(|ox| m.src_col(r, ox) as usize).collect();
            for oy in r.oy0..r.oy1 {
                let sy = m.src_row(r, oy) as usize;
                for (i, ox) in (r.ox0..r.ox1).enumerate() {
                    let (s, d) = ((sy * self.w + cols[i]) * 4, (oy as usize * ow + ox as usize) * 4);
                    canvas[d..d + 4].copy_from_slice(&src_px[s..s + 4]);
                }
            }
        }
        let (bh, bw) = (oh / 4, ow / 4);
        let mut touched = vec![false; bh * bw];
        for r in &m.rects {
            let (cx0, cx1) = ((r.ox0 >> 2).max(0), super::mapping::ceildiv(r.ox1, 4).min(bw as i64));
            let (cy0, cy1) = ((r.oy0 >> 2).max(0), super::mapping::ceildiv(r.oy1, 4).min(bh as i64));
            for y in cy0..cy1 {
                for x in cx0..cx1 {
                    touched[y as usize * bw + x as usize] = true;
                }
            }
        }
        if fill.is_none() && !touched.iter().all(|&t| t) {
            bail!("{}: region needs a fill block but none could be derived from the source", self.rel);
        }
        let out_mips = mip_count(ow, oh);
        let mut out = self.out_header(out_mips);
        let mut level0 = vec![0u8; bh * bw * 16];
        par_chunks_mut(&mut level0, bw * 16, |by, row| {
            for bx in 0..bw {
                let dst = &mut row[bx * 16..bx * 16 + 16];
                if touched[by * bw + bx] {
                    dst.copy_from_slice(&dxt::encode_block(&dxt::gather_block(&canvas, ow, bx, by)));
                } else {
                    dst.copy_from_slice(&fill.expect("checked above"));
                }
            }
        });
        out.extend_from_slice(&level0);
        let encoded = touched.iter().filter(|&&t| t).count();
        let (mut level, mut lh, mut lw) = (canvas, oh, ow);
        for _ in 1..out_mips {
            let (next, nh, nw) = dxt::box_half(&level, lh, lw);
            let (padded, ph, pw) = dxt::pad_edge4(next.clone(), nh, nw);
            out.extend_from_slice(&dxt::pixels_to_blocks(&padded, ph, pw));
            level = next;
            lh = nh;
            lw = nw;
        }
        ensure!(
            out.len() == 128 + dxt_chain_bytes(ow, oh, out_mips),
            "{}: wrong output size {}",
            self.rel,
            out.len()
        );
        let report = json!({
            "rel": self.rel,
            "source_px": [self.w, self.h],
            "source_mips": self.mips,
            "output_px": [ow, oh],
            "output_mips": out_mips,
            "blocks_reencoded": encoded,
            "blocks_fill": bh * bw - encoded,
            "mip_derivation": "box filter of canvas level 0, re-encoded",
            "fill": fill_info,
            "mapping": self.map.report(),
        });
        Ok((out, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::display::mapping::PxRect;

    fn map(s: f64, src: i64, out: i64, k: f64) -> Mapping {
        Mapping {
            s,
            density: 1.0,
            src_w: src,
            src_h: src,
            out_w: out,
            out_h: out,
            rects: vec![PxRect {
                chart: "a".into(),
                sx0: 0,
                sy0: 0,
                sx1: src,
                sy1: src,
                ox0: 0,
                oy0: 0,
                ox1: out,
                oy1: out,
                kx: k,
                ky: k,
            }],
        }
    }

    #[test]
    fn fractional_bilinear_keeps_source_nodes_and_never_leaves_neighbour_range() {
        let n = 20usize;
        let top: Vec<u8> = (0..n * n).map(|i| ((i % n) * 7 + (i / n) * 5 + ((i * 31) % 11)) as u8).collect();
        let m = map(1.5, n as i64, 30, 0.0);
        let out = bilinear_render(&top, n, &m, 0);
        for y in 0..30usize {
            for x in 0..30usize {
                let (yv, xv) = ((y as f64 / 1.5).min(19.0), (x as f64 / 1.5).min(19.0));
                let (r0, c0) = (yv.floor() as usize, xv.floor() as usize);
                let (r1, c1) = ((r0 + 1).min(n - 1), (c0 + 1).min(n - 1));
                let nb = [top[r0 * n + c0], top[r0 * n + c1], top[r1 * n + c0], top[r1 * n + c1]];
                let v = out[y * 30 + x];
                assert!(v >= *nb.iter().min().unwrap() && v <= *nb.iter().max().unwrap(), "({y},{x})");
                if yv.fract() == 0.0 && xv.fract() == 0.0 {
                    assert_eq!(v, top[r0 * n + c0], "node ({y},{x})");
                }
            }
        }
    }

    #[test]
    fn bilinear_clamps_at_the_crop_edge_and_fills_outside() {
        let n = 8usize;
        let top = vec![10u8; n * n];
        let mut m = map(1.5, n as i64, 20, 2.0);
        m.rects[0].ox0 = 2;
        m.rects[0].oy0 = 2;
        let out = bilinear_render(&top, n, &m, 0);
        assert_eq!(out[0], 0, "left of the translated crop stays fill");
        assert_eq!(out[19 * 20 + 19], 10, "far side clamps to the last source row/column");
    }

    #[test]
    fn categorical_rasters_are_replicated_not_blended() {
        let n = 6usize;
        let top: Vec<u8> = (0..n * n).map(|i| if (i % n + i / n) % 2 == 0 { 1 } else { 9 }).collect();
        let out = nearest_render(&top, n, &map(1.5, n as i64, 9, 0.0), 0);
        assert!(out.iter().all(|&v| v == 1 || v == 9));
        assert_eq!(out[3], top[2], "output column 3 samples source column floor(3/1.5) = 2");
    }

    #[test]
    fn mode_block_prefers_the_smallest_block_on_ties() {
        let mut c = ModeCounter::default();
        c.add(&[2u8; 16]);
        c.add(&[1u8; 16]);
        assert_eq!(c.finish().unwrap().0, [1u8; 16]);
    }
}
