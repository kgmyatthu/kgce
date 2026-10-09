//! Supertexture tile layout, DXT5 (BC3) decode/encode, mip rebuild and the changed-tile builder shared by the
//! fixed-theatre painters. Native port of `tools/terrain_paint/stp_write.py` (`dxt5_decode_rgba`, `dxt5_encode`,
//! `encode_alpha_blocks`, `downsample2`, `encode_tile_bytes`) and of `load_canvas` / `encode_changed` /
//! `build_tiles` from `paint_india_east.py` / `fixed_america.py`.
//!
//! The DXT5 encoder reproduces the numpy float32 arithmetic of the Python encoder operation for operation
//! (same evaluation order, `np.add.reduce` pairwise order for the 16-element sums, sequential `einsum`
//! accumulation, round-half-even), so its output is meant to be byte-identical to `stp_write.dxt5_encode`.

use super::grid::{Arr2, Rgb8};
use anyhow::{Context, Result, ensure};
use flate2::{Compression, write::ZlibEncoder};
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const TILE_PX: usize = 512;
pub const TILE_RAW: usize = 512 * 512;
pub const LEVELS: usize = 7;

const CHUNK: usize = 32768;
const BLOCKS_PER_ROW: usize = TILE_PX / 4; // 128
const HALF: usize = TILE_PX / 2; // 256

// ------------------------------------------------------------------------------------------------ tile layout

#[inline]
fn level_dims(level: usize) -> (usize, usize) {
    (128usize >> level, 64usize >> level)
}

fn level_start(level: usize) -> usize {
    (0..level)
        .map(|l| {
            let (w, h) = level_dims(l);
            w * h
        })
        .sum()
}

/// Flat tile index: levels concatenated, row-major, level `L` has `(128>>L) x (64>>L)` tiles (tile row 0 = north).
pub fn tile_index(level: usize, tx: usize, ty: usize) -> usize {
    let (w, h) = level_dims(level);
    assert!(
        level < LEVELS && tx < w && ty < h,
        "tile ({level},{tx},{ty}) outside the supertexture"
    );
    level_start(level) + ty * w + tx
}

/// Inverse of [`tile_index`]: `(level, tx, ty)`.
#[cfg(test)]
pub fn tile_xy(index: usize) -> (usize, usize, usize) {
    for level in 0..LEVELS {
        let (w, h) = level_dims(level);
        let start = level_start(level);
        if index < start + w * h {
            let k = index - start;
            return (level, k % w, k / w);
        }
    }
    panic!("tile index {index} outside the supertexture");
}

/// Baseline tile provider (implemented over the ETW snapshot by the owner).
pub trait TileSource: Sync {
    /// 262144 raw DXT5 bytes of BASELINE tile `index`.
    fn raw_tile(&self, index: usize) -> Result<Vec<u8>>;
}

// ------------------------------------------------------------------------------------------------ parallel helper

/// Evaluate `f(0..n)` on a scoped thread pool; results come back in index order.
fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4)
        .min(n)
        .max(1);
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break out;
                        }
                        out.push((i, f(i)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(v) => v,
                Err(e) => std::panic::resume_unwind(e),
            })
            .collect()
    });
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for (i, v) in parts.into_iter().flatten() {
        slots[i] = Some(v);
    }
    slots
        .into_iter()
        .map(|v| v.expect("par_map slot filled"))
        .collect()
}

// ------------------------------------------------------------------------------------------------ DXT5 decode

#[inline]
fn expand565(c: u16) -> [i32; 3] {
    let c = c as i32;
    [
        (c >> 11) * 255 / 31,
        ((c >> 5) & 63) * 255 / 63,
        (c & 31) * 255 / 31,
    ]
}

#[inline]
fn palette(c0: u16, c1: u16) -> [[i32; 3]; 4] {
    let (p0, p1) = (expand565(c0), expand565(c1));
    let mut p2 = [0; 3];
    let mut p3 = [0; 3];
    for c in 0..3 {
        p2[c] = (2 * p0[c] + p1[c]) / 3;
        p3[c] = (p0[c] + 2 * p1[c]) / 3;
    }
    [p0, p1, p2, p3]
}

/// BC3 alpha palette of stp_write._alpha_palette (float32 arithmetic).
#[inline]
fn alpha_palette(a0: u8, a1: u8) -> [f32; 8] {
    let (f0, f1) = (a0 as f32, a1 as f32);
    let mut p = [0f32; 8];
    p[0] = f0;
    p[1] = f1;
    if a0 > a1 {
        for k in 1..=6 {
            let kf = k as f32;
            p[1 + k] = ((7.0 - kf) * f0 + kf * f1) / 7.0;
        }
    } else {
        for k in 1..=4 {
            let kf = k as f32;
            p[1 + k] = ((5.0 - kf) * f0 + kf * f1) / 5.0;
        }
        p[6] = 0.0;
        p[7] = 255.0;
    }
    p
}

/// Decode one 512x512 tile into `rgb`/`alpha` destinations with row stride `stride` (pixels).
fn decode_into(raw: &[u8], rgb: &mut [Rgb8], alpha: &mut [u8], stride: usize) {
    assert_eq!(raw.len(), TILE_RAW, "DXT5 tile must be {TILE_RAW} bytes");
    for (i, b) in raw.chunks_exact(16).enumerate() {
        let (by, bx) = (i / BLOCKS_PER_ROW, i % BLOCKS_PER_ROW);
        let c0 = b[8] as u16 | (b[9] as u16) << 8;
        let c1 = b[10] as u16 | (b[11] as u16) << 8;
        let pal = palette(c0, c1);
        let bits = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
        let mut ab = 0u64;
        for k in 0..6 {
            ab |= (b[2 + k] as u64) << (8 * k);
        }
        let ap = alpha_palette(b[0], b[1]);
        for k in 0..16 {
            let (py, px) = (k / 4, k % 4);
            let o = (by * 4 + py) * stride + bx * 4 + px;
            let q = pal[((bits >> (2 * k)) & 3) as usize];
            rgb[o] = [q[0] as u8, q[1] as u8, q[2] as u8];
            alpha[o] = ap[((ab >> (3 * k)) & 7) as usize].round_ties_even() as u8;
        }
    }
}

/// Raw DXT5 tile (262144 bytes) -> 512x512 `(rgb, alpha)`; alpha = BC3 8-level interpolation, rounded
/// (`stp_write.dxt5_decode_rgba`).
pub fn dxt5_decode_rgba(raw: &[u8]) -> (Arr2<Rgb8>, Arr2<u8>) {
    let mut rgb = Arr2::new(TILE_PX, TILE_PX, [0u8; 3]);
    let mut al = Arr2::new(TILE_PX, TILE_PX, 0u8);
    decode_into(raw, &mut rgb.d, &mut al.d, TILE_PX);
    (rgb, al)
}

// ------------------------------------------------------------------------------------------------ DXT5 encode

/// float32 `np.add.reduce` of 16 contiguous values (pairwise, 8 accumulators).
#[inline]
fn sum16(a: &[f32; 16]) -> f32 {
    let mut r = [0f32; 8];
    r.copy_from_slice(&a[..8]);
    for i in 0..8 {
        r[i] += a[8 + i];
    }
    ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
}

#[inline]
fn quant565(e: [f32; 3]) -> u16 {
    #[inline]
    fn q(x: f32, mul: f32, max: f32) -> u16 {
        (((x.clamp(0.0, 255.0) * mul) / 255.0).round_ties_even()).clamp(0.0, max) as u16
    }
    let (r5, g6, b5) = (
        q(e[0], 31.0, 31.0),
        q(e[1], 63.0, 63.0),
        q(e[2], 31.0, 31.0),
    );
    (r5 << 11) | (g6 << 5) | b5
}

/// Nearest palette entry per pixel (lowest index on ties) and the summed squared error.
#[inline]
fn assign(p: &[[i32; 3]; 16], c0: u16, c1: u16) -> ([u8; 16], i32) {
    let pal = palette(c0, c1);
    let mut idx = [0u8; 16];
    let mut err = 0;
    for k in 0..16 {
        let mut best = i32::MAX;
        let mut bi = 0u8;
        for (j, q) in pal.iter().enumerate() {
            let d = (p[k][0] - q[0]).pow(2) + (p[k][1] - q[1]).pow(2) + (p[k][2] - q[2]).pow(2);
            if d < best {
                best = d;
                bi = j as u8;
            }
        }
        idx[k] = bi;
        err += best;
    }
    (idx, err)
}

const W0: [f32; 4] = [1.0, 0.0, (2.0f64 / 3.0) as f32, (1.0f64 / 3.0) as f32];
const W1: [f32; 4] = [1.0 - W0[0], 1.0 - W0[1], 1.0 - W0[2], 1.0 - W0[3]];

/// PCA endpoint seed + `refine` least-squares refits for one 4x4 block -> `(c0, c1, index bits)`.
fn encode_color_block(px: &[Rgb8; 16], refine: usize) -> (u16, u16, u32) {
    let mut p = [[0f32; 3]; 16];
    let mut pi = [[0i32; 3]; 16];
    for k in 0..16 {
        for c in 0..3 {
            p[k][c] = px[k][c] as f32;
            pi[k][c] = px[k][c] as i32;
        }
    }
    // mean (sums of integers < 2^24: exact), deviations, covariance (sequential float32 einsum order)
    let mut mean = [0f32; 3];
    let mut mx = [f32::MIN; 3];
    let mut mn = [f32::MAX; 3];
    for c in 0..3 {
        let mut s = 0f32;
        for k in 0..16 {
            s += p[k][c];
            mx[c] = mx[c].max(p[k][c]);
            mn[c] = mn[c].min(p[k][c]);
        }
        mean[c] = s / 16.0;
    }
    let mut d = [[0f32; 3]; 16];
    for k in 0..16 {
        for c in 0..3 {
            d[k][c] = p[k][c] - mean[c];
        }
    }
    let mut cov = [[0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0f32;
            for k in 0..16 {
                s += d[k][i] * d[k][j];
            }
            cov[i][j] = s;
        }
    }
    // principal axis by 6 power iterations seeded with the bounding-box diagonal
    let mut v = [
        (mx[0] - mn[0]) + 1e-3,
        (mx[1] - mn[1]) + 1e-3,
        (mx[2] - mn[2]) + 1e-3,
    ];
    for _ in 0..6 {
        let mut nv = [0f32; 3];
        for i in 0..3 {
            nv[i] = cov[i][0] * v[0] + cov[i][1] * v[1] + cov[i][2] * v[2];
        }
        let norm = (((nv[0] * nv[0]) + (nv[1] * nv[1])) + (nv[2] * nv[2])).sqrt();
        let den = norm + 1e-9;
        for i in 0..3 {
            v[i] = nv[i] / den;
        }
    }
    let mut tmax = f32::MIN;
    let mut tmin = f32::MAX;
    for k in 0..16 {
        let t = d[k][0] * v[0] + d[k][1] * v[1] + d[k][2] * v[2];
        tmax = tmax.max(t);
        tmin = tmin.min(t);
    }
    let mut e0 = [0f32; 3];
    let mut e1 = [0f32; 3];
    for c in 0..3 {
        e0[c] = mean[c] + v[c] * tmax;
        e1[c] = mean[c] + v[c] * tmin;
    }
    // inset like squish/stb
    for c in 0..3 {
        let r = (e0[c] - e1[c]) / 16.0;
        e0[c] -= r;
        e1[c] += r;
    }
    let mut c0 = quant565(e0);
    let mut c1 = quant565(e1);
    let (mut idx, mut err) = assign(&pi, c0, c1);
    for _ in 0..refine {
        let mut a = [0f32; 16];
        let mut b = [0f32; 16];
        for k in 0..16 {
            a[k] = W0[idx[k] as usize];
            b[k] = W1[idx[k] as usize];
        }
        let mut aa16 = [0f32; 16];
        let mut bb16 = [0f32; 16];
        let mut ab16 = [0f32; 16];
        for k in 0..16 {
            aa16[k] = a[k] * a[k];
            bb16[k] = b[k] * b[k];
            ab16[k] = a[k] * b[k];
        }
        let (aa, bb, ab) = (sum16(&aa16), sum16(&bb16), sum16(&ab16));
        let det = aa * bb - ab * ab;
        let ok = det > 1e-6;
        let detc = if ok { det } else { 1.0 };
        let mut n0 = e0;
        let mut n1 = e1;
        if ok {
            for c in 0..3 {
                let (mut pa, mut pb) = (0f32, 0f32);
                for k in 0..16 {
                    pa += a[k] * p[k][c];
                    pb += b[k] * p[k][c];
                }
                n0[c] = (bb * pa - ab * pb) / detc;
                n1[c] = (aa * pb - ab * pa) / detc;
            }
        }
        let (nc0, nc1) = (quant565(n0), quant565(n1));
        let (nidx, nerr) = assign(&pi, nc0, nc1);
        if nerr < err {
            c0 = nc0;
            c1 = nc1;
            idx = nidx;
            err = nerr;
        }
    }
    // canonical order c0 > c1 (swap endpoints, remap 0<->1, 2<->3); equal endpoints -> all index 0
    let sw = c0 < c1;
    let (c0s, c1s) = if sw { (c1, c0) } else { (c0, c1) };
    const REMAP: [u8; 4] = [1, 0, 3, 2];
    let mut bits = 0u32;
    if c0s != c1s {
        for k in 0..16 {
            let i = if sw { REMAP[idx[k] as usize] } else { idx[k] };
            bits |= (i as u32) << (2 * k);
        }
    }
    (c0s, c1s, bits)
}

/// `stp_write.encode_alpha_blocks` for one block: a0 = max, a1 = min, 8-level mode, nearest index.
fn encode_alpha_block(al: &[u8; 16]) -> [u8; 8] {
    let a0 = *al.iter().max().unwrap();
    let a1 = *al.iter().min().unwrap();
    let (f0, f1) = (a0 as f32, a1 as f32);
    let mut pal = [0f32; 8];
    pal[0] = f0;
    pal[1] = f1;
    for k in 1..=6 {
        let kf = k as f32;
        pal[1 + k] = ((7.0 - kf) * f0 + kf * f1) / 7.0;
    }
    let mut bits = 0u64;
    if a0 != a1 {
        for k in 0..16 {
            let x = al[k] as f32;
            let mut best = f32::INFINITY;
            let mut bi = 0u64;
            for (j, q) in pal.iter().enumerate() {
                let d = (x - q).abs();
                if d < best {
                    best = d;
                    bi = j as u64;
                }
            }
            bits |= bi << (3 * k);
        }
    }
    let mut out = [0u8; 8];
    out[0] = a0;
    out[1] = a1;
    for i in 0..6 {
        out[2 + i] = ((bits >> (8 * i)) & 255) as u8;
    }
    out
}

/// Encode block `(bx, by)` of an image with row stride `w`.
#[inline]
fn encode_block_at(
    rgb: &[Rgb8],
    alpha: &[u8],
    w: usize,
    bx: usize,
    by: usize,
    refine: usize,
    channels: u8,
) -> [u8; 16] {
    let mut out = [0u8; 16];
    if channels & 1 != 0 {
        let mut px = [[0u8; 3]; 16];
        for k in 0..16 {
            px[k] = rgb[(by * 4 + k / 4) * w + bx * 4 + k % 4];
        }
        let (c0, c1, bits) = encode_color_block(&px, refine);
        out[8..10].copy_from_slice(&c0.to_le_bytes());
        out[10..12].copy_from_slice(&c1.to_le_bytes());
        out[12..16].copy_from_slice(&bits.to_le_bytes());
    }
    if channels & 2 != 0 {
        let mut al = [0u8; 16];
        for k in 0..16 {
            al[k] = alpha[(by * 4 + k / 4) * w + bx * 4 + k % 4];
        }
        out[..8].copy_from_slice(&encode_alpha_block(&al));
    }
    out
}

/// `stp_write.dxt5_encode(rgb, alpha, refine=2)`: raw DXT5 bytes, 16 B per 4x4 block, row-major blocks.
#[cfg(test)]
pub fn dxt5_encode(rgb: &Arr2<Rgb8>, alpha: &Arr2<u8>) -> Vec<u8> {
    assert!(
        rgb.h == alpha.h && rgb.w == alpha.w && rgb.h % 4 == 0 && rgb.w % 4 == 0,
        "dxt5_encode: bad image size"
    );
    let (bh, bw) = (rgb.h / 4, rgb.w / 4);
    let mut out = vec![0u8; bh * bw * 16];
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * 16;
            out[o..o + 16].copy_from_slice(&encode_block_at(&rgb.d, &alpha.d, rgb.w, bx, by, 2, 3));
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ NTW tile chunks

/// 262144 raw DXT5 bytes -> 8 x `[u32 csize][u32 0x8000][zlib level 9]` (`stp_write.encode_tile_bytes`).
pub fn encode_ntw_tile(raw: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        raw.len() == TILE_RAW,
        "NTW tile source must be {TILE_RAW} bytes, got {}",
        raw.len()
    );
    let mut out = Vec::new();
    for chunk in raw.chunks(CHUNK) {
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(9));
        enc.write_all(chunk)?;
        let c = enc.finish()?;
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
        out.extend_from_slice(&(CHUNK as u32).to_le_bytes());
        out.extend_from_slice(&c);
    }
    Ok(out)
}

/// Inverse of [`encode_ntw_tile`] (`stp_write.decode_tile_bytes`): validates every chunk header.
#[cfg(test)]
pub fn decode_ntw_tile(b: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::with_capacity(TILE_RAW);
    let mut pos = 0;
    while pos < b.len() {
        ensure!(pos + 8 <= b.len(), "NTW tile chunk header truncated");
        let c = u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap()) as usize;
        let r = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
        ensure!(r == CHUNK, "NTW tile chunk raw size {r} != {CHUNK}");
        let z = b
            .get(pos + 8..pos + 8 + c)
            .context("NTW tile chunk truncated")?;
        let mut dec = flate2::read::ZlibDecoder::new(z);
        let start = out.len();
        dec.read_to_end(&mut out)
            .context("inflate NTW tile chunk")?;
        ensure!(
            out.len() - start == CHUNK,
            "NTW tile chunk inflated to {} bytes",
            out.len() - start
        );
        pos += 8 + c;
    }
    ensure!(
        out.len() == TILE_RAW,
        "NTW tile inflated to {} bytes",
        out.len()
    );
    Ok(out)
}

// ------------------------------------------------------------------------------------------------ canvas / build_tiles

/// Level-0 tile window of a painter canvas.
#[derive(Clone, Copy, Debug)]
pub struct CanvasWindow {
    pub tx0: usize,
    pub ty0: usize,
    pub ntx: usize,
    pub nty: usize,
}

impl CanvasWindow {
    fn check(&self) -> Result<()> {
        let (w, h) = level_dims(0);
        ensure!(
            self.ntx > 0 && self.nty > 0 && self.tx0 + self.ntx <= w && self.ty0 + self.nty <= h,
            "canvas window {self:?} lies outside the {w}x{h} level-0 tile grid"
        );
        Ok(())
    }
}

/// Decode the level-0 tiles of `win` into one canvas `(rgb, alpha)` of size `(nty*512, ntx*512)` (parallel).
pub fn load_canvas(src: &dyn TileSource, win: CanvasWindow) -> Result<(Arr2<Rgb8>, Arr2<u8>)> {
    win.check()?;
    let (h, w) = (win.nty * TILE_PX, win.ntx * TILE_PX);
    let mut rgb = Arr2::new(h, w, [0u8; 3]);
    let mut al = Arr2::new(h, w, 0u8);
    // one tile row at a time (bounded memory): decode the row's tiles in parallel, then paste
    for ti in 0..win.nty {
        let tiles: Vec<Result<(Arr2<Rgb8>, Arr2<u8>)>> = par_map(win.ntx, |tj| {
            let n = tile_index(0, win.tx0 + tj, win.ty0 + ti);
            let raw = src
                .raw_tile(n)
                .with_context(|| format!("baseline tile {n}"))?;
            ensure!(
                raw.len() == TILE_RAW,
                "baseline tile {n} has {} bytes",
                raw.len()
            );
            Ok(dxt5_decode_rgba(&raw))
        });
        for (tj, t) in tiles.into_iter().enumerate() {
            let (r, a) = t?;
            rgb.paste(ti * TILE_PX, tj * TILE_PX, &r);
            al.paste(ti * TILE_PX, tj * TILE_PX, &a);
        }
    }
    Ok((rgb, al))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelStat {
    pub level: usize,
    pub tiles: usize,
    pub changed_blocks: usize,
}

/// A full repaint reconstructs both channels over its footprint. A colour repair must not
/// resample the artist-authored coastline mips merely because nearby RGB pixels changed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChannelUpdates {
    Repaint,
    ColourRepair,
}

/// Compact mip contribution of one tile for the next level: 2x box downsample of the tile and the 2x2-changed mask.
struct Mip {
    r: Vec<Rgb8>,
    a: Vec<u8>,
    m: Vec<u8>, // bit 0: colour; bit 1: alpha
}

fn mip_of(r: &[Rgb8], a: &[u8], mask: &[u8]) -> Mip {
    let mut m = vec![0u8; HALF * HALF];
    let mut mr = vec![[0u8; 3]; HALF * HALF];
    let mut ma = vec![0u8; HALF * HALF];
    for y in 0..HALF {
        for x in 0..HALF {
            let (o00, o01) = ((2 * y) * TILE_PX + 2 * x, (2 * y) * TILE_PX + 2 * x + 1);
            let (o10, o11) = (o00 + TILE_PX, o01 + TILE_PX);
            let channels = mask[o00] | mask[o01] | mask[o10] | mask[o11];
            if channels != 0 {
                let i = y * HALF + x;
                m[i] = channels;
                if channels & 1 != 0 {
                    for c in 0..3 {
                        mr[i][c] = ((r[o00][c] as u16
                            + r[o10][c] as u16
                            + r[o01][c] as u16
                            + r[o11][c] as u16
                            + 2)
                            >> 2) as u8;
                    }
                }
                if channels & 2 != 0 {
                    ma[i] = ((a[o00] as u16 + a[o10] as u16 + a[o01] as u16 + a[o11] as u16 + 2)
                        >> 2) as u8;
                }
            }
        }
    }
    Mip { r: mr, a: ma, m }
}

/// `encode_changed` + mip preparation for one tile: blocks whose pixels differ from the baseline are re-encoded,
/// every other block keeps its baseline bytes.
struct TileJob {
    new_r: Vec<Rgb8>,
    new_a: Vec<u8>,
    old_r: Vec<Rgb8>,
    old_a: Vec<u8>,
    raw_old: Vec<u8>,
    mask: Vec<u8>,
}

fn finish_tile(
    job: TileJob,
    want_mip: bool,
    channels: ChannelUpdates,
) -> (Vec<u8>, usize, Option<Mip>) {
    let TileJob {
        new_r,
        new_a,
        old_r,
        old_a,
        mut raw_old,
        mask,
    } = job;
    let mut changed = 0;
    for by in 0..BLOCKS_PER_ROW {
        for bx in 0..BLOCKS_PER_ROW {
            let mut diff = 0u8;
            for py in 0..4 {
                let o = (by * 4 + py) * TILE_PX + bx * 4;
                for px in 0..4 {
                    diff |= (new_r[o + px] != old_r[o + px]) as u8;
                    diff |= ((new_a[o + px] != old_a[o + px]) as u8) << 1;
                }
            }
            if diff != 0 {
                changed += 1;
                if channels == ChannelUpdates::Repaint {
                    diff = 3;
                }
                let i = (by * BLOCKS_PER_ROW + bx) * 16;
                let block = encode_block_at(&new_r, &new_a, TILE_PX, bx, by, 2, diff);
                if diff & 1 != 0 {
                    raw_old[i + 8..i + 16].copy_from_slice(&block[8..]);
                }
                if diff & 2 != 0 {
                    raw_old[i..i + 8].copy_from_slice(&block[..8]);
                }
            }
        }
    }
    let mip = want_mip.then(|| mip_of(&new_r, &new_a, &mask));
    (raw_old, changed, mip)
}

fn fetch_baseline(src: &dyn TileSource, n: usize) -> Result<Vec<u8>> {
    let raw = src
        .raw_tile(n)
        .with_context(|| format!("baseline tile {n}"))?;
    ensure!(
        raw.len() == TILE_RAW,
        "baseline tile {n} has {} bytes",
        raw.len()
    );
    Ok(raw)
}

/// Port of `build_tiles()`: `rgb`/`alpha` is the repainted canvas of `win` (same layout as [`load_canvas`]).
/// Level-0 tiles whose pixels differ from the baseline are re-encoded only where a 4x4 block differs; mips are
/// rebuilt for changed regions only (2x box downsample of the changed 2x2 pixel groups into the decoded baseline
/// mip tile), levels 1..6. The baseline is re-decoded per tile from `src`. Returns RAW 262144-byte DXT5 per
/// affected tile index (mip tiles whose downsampled pixels equal the baseline are included with baseline bytes,
/// exactly like the Python) and per-level statistics.
pub fn build_tiles(
    src: &dyn TileSource,
    win: CanvasWindow,
    rgb: &Arr2<Rgb8>,
    alpha: &Arr2<u8>,
    channels: ChannelUpdates,
) -> Result<(BTreeMap<usize, Vec<u8>>, Vec<LevelStat>)> {
    win.check()?;
    ensure!(
        rgb.h == win.nty * TILE_PX
            && rgb.w == win.ntx * TILE_PX
            && alpha.h == rgb.h
            && alpha.w == rgb.w,
        "canvas {}x{} does not match window {win:?}",
        rgb.h,
        rgb.w
    );
    let mut out: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    let mut stats = Vec::with_capacity(LEVELS);

    // ---- level 0: tiles whose canvas pixels differ from the baseline
    let results: Vec<Result<Option<(usize, usize, usize, Vec<u8>, usize, Option<Mip>)>>> =
        par_map(win.ntx * win.nty, |k| {
            let (i, j) = (k / win.ntx, k % win.ntx);
            let (tx, ty) = (win.tx0 + j, win.ty0 + i);
            let n = tile_index(0, tx, ty);
            let raw_old = fetch_baseline(src, n)?;
            let (old_rgb, old_al) = dxt5_decode_rgba(&raw_old);
            let mut new_r = Vec::with_capacity(TILE_RAW);
            let mut new_a = Vec::with_capacity(TILE_RAW);
            for y in 0..TILE_PX {
                let o = (i * TILE_PX + y) * rgb.w + j * TILE_PX;
                new_r.extend_from_slice(&rgb.d[o..o + TILE_PX]);
                new_a.extend_from_slice(&alpha.d[o..o + TILE_PX]);
            }
            let mask: Vec<u8> = (0..TILE_RAW)
                .map(|p| {
                    let diff =
                        (new_r[p] != old_rgb.d[p]) as u8 | (((new_a[p] != old_al.d[p]) as u8) << 1);
                    if diff != 0 && channels == ChannelUpdates::Repaint {
                        3
                    } else {
                        diff
                    }
                })
                .collect();
            if !mask.iter().any(|&m| m != 0) {
                return Ok(None);
            }
            let job = TileJob {
                new_r,
                new_a,
                old_r: old_rgb.d,
                old_a: old_al.d,
                raw_old,
                mask,
            };
            let (raw, nb, mip) = finish_tile(job, true, channels);
            Ok(Some((n, tx, ty, raw, nb, mip)))
        });
    let mut cur: BTreeMap<(usize, usize), Mip> = BTreeMap::new();
    let mut changed_blocks = 0;
    for r in results {
        if let Some((n, tx, ty, raw, nb, mip)) = r? {
            out.insert(n, raw);
            changed_blocks += nb;
            cur.insert((tx, ty), mip.expect("level-0 mip"));
        }
    }
    stats.push(LevelStat {
        level: 0,
        tiles: cur.len(),
        changed_blocks,
    });

    // ---- mips
    for level in 1..LEVELS {
        let parents: Vec<(usize, usize)> = {
            let mut s: Vec<(usize, usize)> = cur.keys().map(|&(x, y)| (x / 2, y / 2)).collect();
            s.sort_unstable();
            s.dedup();
            s
        };
        let want_mip = level + 1 < LEVELS;
        let results: Vec<Result<(usize, Vec<u8>, usize, Option<Mip>)>> =
            par_map(parents.len(), |k| {
                let (tx, ty) = parents[k];
                let n = tile_index(level, tx, ty);
                let raw_old = fetch_baseline(src, n)?;
                let (old_rgb, old_al) = dxt5_decode_rgba(&raw_old);
                let mut new_r = old_rgb.d.clone();
                let mut new_a = old_al.d.clone();
                let mut mask = vec![0u8; TILE_RAW];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let Some(m) = cur.get(&(tx * 2 + dx, ty * 2 + dy)) else {
                            continue;
                        };
                        for y in 0..HALF {
                            for x in 0..HALF {
                                let s = y * HALF + x;
                                if m.m[s] != 0 {
                                    let o = (dy * HALF + y) * TILE_PX + dx * HALF + x;
                                    if m.m[s] & 1 != 0 {
                                        new_r[o] = m.r[s];
                                    }
                                    if m.m[s] & 2 != 0 {
                                        new_a[o] = m.a[s];
                                    }
                                    mask[o] = m.m[s];
                                }
                            }
                        }
                    }
                }
                let job = TileJob {
                    new_r,
                    new_a,
                    old_r: old_rgb.d,
                    old_a: old_al.d,
                    raw_old,
                    mask,
                };
                let (raw, nb, mip) = finish_tile(job, want_mip, channels);
                Ok((n, raw, nb, mip))
            });
        let mut next: BTreeMap<(usize, usize), Mip> = BTreeMap::new();
        let mut cb = 0;
        for (k, r) in results.into_iter().enumerate() {
            let (n, raw, nb, mip) = r?;
            out.insert(n, raw);
            cb += nb;
            if let Some(m) = mip {
                next.insert(parents[k], m);
            }
        }
        stats.push(LevelStat {
            level,
            tiles: parents.len(),
            changed_blocks: cb,
        });
        cur = next;
    }
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_repairs_preserve_shoreline_mips_but_coverage_edits_reach_them() {
        struct Source {
            levels: Vec<Vec<u8>>,
        }
        impl TileSource for Source {
            fn raw_tile(&self, index: usize) -> Result<Vec<u8>> {
                Ok(self.levels[tile_xy(index).0].clone())
            }
        }
        // The artist's coarser coverage is deliberately not a box average of level zero.
        let rgb = Arr2::new(TILE_PX, TILE_PX, [100, 140, 80]);
        let source = Source {
            levels: (0..LEVELS)
                .map(|level| {
                    dxt5_encode(&rgb, &Arr2::new(TILE_PX, TILE_PX, 20 * (level as u8 + 1)))
                })
                .collect(),
        };
        let (original_rgb, original_alpha) = dxt5_decode_rgba(&source.levels[0]);
        let painted = Arr2::new(TILE_PX, TILE_PX, [180, 60, 45]);
        let window = CanvasWindow {
            tx0: 0,
            ty0: 0,
            ntx: 1,
            nty: 1,
        };
        let (colour_tiles, _) = build_tiles(
            &source,
            window,
            &painted,
            &original_alpha,
            ChannelUpdates::ColourRepair,
        )
        .unwrap();
        let coverage = Arr2::new(TILE_PX, TILE_PX, 180);
        let (coverage_tiles, _) = build_tiles(
            &source,
            window,
            &original_rgb,
            &coverage,
            ChannelUpdates::ColourRepair,
        )
        .unwrap();
        for level in 0..LEVELS {
            let index = tile_index(level, 0, 0);
            let colour = &colour_tiles[&index];
            for (new, old) in colour
                .chunks_exact(16)
                .zip(source.levels[level].chunks_exact(16))
            {
                assert_eq!(
                    &new[..8],
                    &old[..8],
                    "RGB repair rewrote coastline at mip {level}"
                );
            }
            let (decoded, _) = dxt5_decode_rgba(colour);
            for (value, expected) in decoded.d[0].iter().zip([180, 60, 45]) {
                assert!(
                    (*value as i16 - expected).abs() <= 4,
                    "colour failed to reach mip {level}"
                );
            }
            let coverage_tile = &coverage_tiles[&index];
            for (new, old) in coverage_tile
                .chunks_exact(16)
                .zip(source.levels[level].chunks_exact(16))
            {
                assert_eq!(
                    &new[8..],
                    &old[8..],
                    "coverage edit changed colour at mip {level}"
                );
            }
            let (_, decoded) = dxt5_decode_rgba(coverage_tile);
            assert_eq!(
                decoded.d[0], 180,
                "new coastline failed to reach mip {level}"
            );
        }
    }

    #[test]
    fn tile_index_roundtrip() {
        assert_eq!(tile_index(1, 0, 0), 128 * 64);
        for n in [0, 1, 8191, 8192, 10239, 10240, 10919, 10921] {
            let (l, x, y) = tile_xy(n);
            assert_eq!(tile_index(l, x, y), n);
        }
        assert_eq!(tile_xy(10921), (6, 1, 0));
    }

    #[test]
    fn flat_blocks_roundtrip() {
        let rgb = Arr2::new(512, 512, [200u8, 100, 50]);
        let al = Arr2::new(512, 512, 255u8);
        let raw = dxt5_encode(&rgb, &al);
        assert_eq!(raw.len(), TILE_RAW);
        let (r2, a2) = dxt5_decode_rgba(&raw);
        assert!(a2.d.iter().all(|&a| a == 255));
        let d = r2.d[0];
        assert!(
            (d[0] as i32 - 200).abs() <= 4
                && (d[1] as i32 - 100).abs() <= 2
                && (d[2] as i32 - 50).abs() <= 4
        );
    }

    #[test]
    fn ntw_chunks_roundtrip() {
        let raw: Vec<u8> = (0..TILE_RAW).map(|i| (i * 7 >> 3) as u8).collect();
        assert_eq!(
            decode_ntw_tile(&encode_ntw_tile(&raw).unwrap()).unwrap(),
            raw
        );
    }
}
