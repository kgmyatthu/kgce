//! DXT5 (BC3) block decode/encode used for the atlas mask and supertexture resampling.
//!
//! The encoder is the accepted POC encoder (`raster_scale.py`): colour and alpha endpoints are fitted independently,
//! colour uses a min/max-inset seed plus one least-squares refit (float32 arithmetic), alpha chooses between the
//! max/min eight-level palette and the six-level palette over the inner values (exact 0 and 255 survive).

pub type Block = [u8; 16];

const W0: [f32; 4] = [1.0, 0.0, (2.0f64 / 3.0) as f32, (1.0f64 / 3.0) as f32];

#[inline]
fn alpha_palette(a0: i32, a1: i32) -> [i32; 8] {
    let mut pal = [0i32; 8];
    pal[0] = a0;
    pal[1] = a1;
    if a0 > a1 {
        for k in 1..=6 {
            pal[1 + k] = ((7 - k as i32) * a0 + k as i32 * a1) / 7;
        }
    } else {
        for k in 1..=4 {
            pal[1 + k] = ((5 - k as i32) * a0 + k as i32 * a1) / 5;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    pal
}

#[inline]
fn expand565(c: u32) -> [i32; 3] {
    let (r, g, b) = ((c >> 11) & 31, (c >> 5) & 63, c & 31);
    [((r << 3) | (r >> 2)) as i32, ((g << 2) | (g >> 4)) as i32, ((b << 3) | (b >> 2)) as i32]
}

/// One block -> 16 RGBA pixels, row-major inside the block. Colour is always decoded as four-colour.
pub fn decode_block(blk: &[u8]) -> [[u8; 4]; 16] {
    let pal = alpha_palette(blk[0] as i32, blk[1] as i32);
    let mut bits = 0u64;
    for i in 0..6 {
        bits |= (blk[2 + i] as u64) << (8 * i);
    }
    let c0 = blk[8] as u32 | ((blk[9] as u32) << 8);
    let c1 = blk[10] as u32 | ((blk[11] as u32) << 8);
    let (e0, e1) = (expand565(c0), expand565(c1));
    let mut cp = [[0i32; 3]; 4];
    cp[0] = e0;
    cp[1] = e1;
    for c in 0..3 {
        cp[2][c] = (2 * e0[c] + e1[c]) / 3;
        cp[3][c] = (e0[c] + 2 * e1[c]) / 3;
    }
    let ci = u32::from_le_bytes([blk[12], blk[13], blk[14], blk[15]]);
    let mut out = [[0u8; 4]; 16];
    for k in 0..16 {
        let col = cp[((ci >> (2 * k)) & 3) as usize];
        let a = pal[((bits >> (3 * k)) & 7) as usize];
        out[k] = [col[0] as u8, col[1] as u8, col[2] as u8, a as u8];
    }
    out
}

/// `bh x bw` blocks (row-major, 16 bytes each) -> `(bh*4) x (bw*4)` RGBA pixels.
pub fn blocks_to_pixels(blocks: &[u8], bh: usize, bw: usize) -> Vec<u8> {
    let w = bw * 4;
    let mut out = vec![0u8; bh * 4 * w * 4];
    for by in 0..bh {
        for bx in 0..bw {
            let px = decode_block(&blocks[(by * bw + bx) * 16..][..16]);
            for k in 0..16 {
                let o = ((by * 4 + k / 4) * w + bx * 4 + k % 4) * 4;
                out[o..o + 4].copy_from_slice(&px[k]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------- encoder

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
fn q565(x: [f32; 3]) -> u32 {
    let f = [(31.0f64 / 255.0) as f32, (63.0f64 / 255.0) as f32, (31.0f64 / 255.0) as f32];
    let hi = [31.0f32, 63.0, 31.0];
    let mut q = [0u32; 3];
    for c in 0..3 {
        q[c] = (x[c] * f[c]).round_ties_even().clamp(0.0, hi[c]) as u32;
    }
    (q[0] << 11) | (q[1] << 5) | q[2]
}

#[inline]
fn e565f(c: u32) -> [f32; 3] {
    let e = expand565(c);
    [e[0] as f32, e[1] as f32, e[2] as f32]
}

/// Palette fit: endpoints ordered `q0 >= q1`, nearest-index assignment (first minimum), summed squared error.
fn color_candidate(rgb: &[[f32; 3]; 16], q0: u32, q1: u32) -> (u32, u32, [u8; 16], f32) {
    let (q0, q1) = if q0 < q1 { (q1, q0) } else { (q0, q1) };
    let (e0, e1) = (e565f(q0), e565f(q1));
    let mut pal = [[0f32; 3]; 4];
    pal[0] = e0;
    pal[1] = e1;
    for c in 0..3 {
        pal[2][c] = (2.0f32 * e0[c] + e1[c]) / 3.0f32;
        pal[3][c] = (e0[c] + 2.0f32 * e1[c]) / 3.0f32;
    }
    let mut idx = [0u8; 16];
    let mut best = [0f32; 16];
    for k in 0..16 {
        let mut bd = f32::INFINITY;
        let mut bi = 0u8;
        for (j, p) in pal.iter().enumerate() {
            let (dr, dg, db) = (rgb[k][0] - p[0], rgb[k][1] - p[1], rgb[k][2] - p[2]);
            let d = (dr * dr + dg * dg) + db * db;
            if d < bd {
                bd = d;
                bi = j as u8;
            }
        }
        idx[k] = bi;
        best[k] = bd;
    }
    (q0, q1, idx, sum16(&best))
}

fn encode_color(rgb: &[[f32; 3]; 16]) -> (u32, u32, [u8; 16]) {
    let mut mx = [f32::MIN; 3];
    let mut mn = [f32::MAX; 3];
    for p in rgb {
        for c in 0..3 {
            mx[c] = mx[c].max(p[c]);
            mn[c] = mn[c].min(p[c]);
        }
    }
    let mut hi = [0f32; 3];
    let mut lo = [0f32; 3];
    for c in 0..3 {
        let inset = (mx[c] - mn[c]) / 16.0f32;
        hi[c] = mx[c] - inset;
        lo[c] = mn[c] + inset;
    }
    let (q0, q1, idx, err) = color_candidate(rgb, q565(hi), q565(lo));
    let mut w0 = [0f32; 16];
    let mut w1 = [0f32; 16];
    for k in 0..16 {
        w0[k] = W0[idx[k] as usize];
        w1[k] = 1.0f32 - w0[k];
    }
    let mut t = [[0f32; 16]; 3];
    for k in 0..16 {
        t[0][k] = w0[k] * w0[k];
        t[1][k] = w0[k] * w1[k];
        t[2][k] = w1[k] * w1[k];
    }
    let (a11, a12, a22) = (sum16(&t[0]), sum16(&t[1]), sum16(&t[2]));
    let mut b0 = [0f32; 3];
    let mut b1 = [0f32; 3];
    for c in 0..3 {
        b0[c] = w0[0] * rgb[0][c];
        b1[c] = w1[0] * rgb[0][c];
        for k in 1..16 {
            b0[c] += w0[k] * rgb[k][c];
            b1[c] += w1[k] * rgb[k][c];
        }
    }
    let det = a11 * a22 - a12 * a12;
    let ok = det.abs() > 1e-6f32;
    let sd = if ok { det } else { 1.0f32 };
    let mut a = hi;
    let mut b = lo;
    if ok {
        for c in 0..3 {
            a[c] = (b0[c] * a22 - b1[c] * a12) / sd;
            b[c] = (b1[c] * a11 - b0[c] * a12) / sd;
        }
    }
    let (r0, r1, ridx, rerr) = color_candidate(rgb, q565(a), q565(b));
    if rerr < err { (r0, r1, ridx) } else { (q0, q1, idx) }
}

fn alpha_candidate(a: &[i32; 16], a0: i32, a1: i32) -> ([u8; 16], i64) {
    let pal = alpha_palette(a0, a1);
    let mut idx = [0u8; 16];
    let mut err = 0i64;
    for k in 0..16 {
        let mut bd = i32::MAX;
        let mut bi = 0u8;
        for (j, p) in pal.iter().enumerate() {
            let d = (a[k] - p).abs();
            if d < bd {
                bd = d;
                bi = j as u8;
            }
        }
        idx[k] = bi;
        err += (bd as i64) * (bd as i64);
    }
    (idx, err)
}

fn encode_alpha(a: &[i32; 16]) -> (i32, i32, [u8; 16]) {
    let mx = *a.iter().max().unwrap();
    let mn = *a.iter().min().unwrap();
    let (i1, e1) = alpha_candidate(a, mx, mn);
    let inner: Vec<i32> = a.iter().copied().filter(|&v| v > 0 && v < 255).collect();
    let (lo, hi) = if inner.is_empty() {
        (0, 0)
    } else {
        (*inner.iter().min().unwrap(), *inner.iter().max().unwrap())
    };
    let (i2, e2) = alpha_candidate(a, lo, hi);
    if e2 < e1 { (lo, hi, i2) } else { (mx, mn, i1) }
}

/// Encode one 4x4 block of RGBA pixels (row-major inside the block).
pub fn encode_block(px: &[[u8; 4]; 16]) -> Block {
    let mut rgb = [[0f32; 3]; 16];
    let mut a = [0i32; 16];
    for k in 0..16 {
        rgb[k] = [px[k][0] as f32, px[k][1] as f32, px[k][2] as f32];
        a[k] = px[k][3] as i32;
    }
    let (a0, a1, aidx) = encode_alpha(&a);
    let (c0, c1, cidx) = encode_color(&rgb);
    let mut out = [0u8; 16];
    out[0] = a0 as u8;
    out[1] = a1 as u8;
    let mut abits = 0u64;
    let mut cbits = 0u32;
    for k in 0..16 {
        abits |= (aidx[k] as u64) << (3 * k);
        cbits |= (cidx[k] as u32) << (2 * k);
    }
    for i in 0..6 {
        out[2 + i] = ((abits >> (8 * i)) & 0xFF) as u8;
    }
    out[8..10].copy_from_slice(&(c0 as u16).to_le_bytes());
    out[10..12].copy_from_slice(&(c1 as u16).to_le_bytes());
    out[12..16].copy_from_slice(&cbits.to_le_bytes());
    out
}

/// Gather block `(bx, by)` out of an RGBA canvas with row stride `w` pixels.
#[inline]
pub fn gather_block(img: &[u8], w: usize, bx: usize, by: usize) -> [[u8; 4]; 16] {
    let mut px = [[0u8; 4]; 16];
    for k in 0..16 {
        let o = ((by * 4 + k / 4) * w + bx * 4 + k % 4) * 4;
        px[k].copy_from_slice(&img[o..o + 4]);
    }
    px
}

/// RGBA image (`h`, `w` multiples of 4) -> encoded blocks `(h/4) x (w/4) x 16`.
pub fn pixels_to_blocks(img: &[u8], h: usize, w: usize) -> Vec<u8> {
    let (bh, bw) = (h / 4, w / 4);
    let mut out = vec![0u8; bh * bw * 16];
    super::par_chunks_mut(&mut out, bw * 16, |by, row| {
        for bx in 0..bw {
            row[bx * 16..bx * 16 + 16].copy_from_slice(&encode_block(&gather_block(img, w, bx, by)));
        }
    });
    out
}

/// 2x2 box filter with rounding (`raster_scale.box_half`); odd trailing rows/columns are dropped, a size of 1 is kept.
pub fn box_half(img: &[u8], h: usize, w: usize) -> (Vec<u8>, usize, usize) {
    let (h2, w2) = ((h / 2).max(1), (w / 2).max(1));
    let (rows, cols) = (if h > 1 { h2 * 2 } else { 1 }, if w > 1 { w2 * 2 } else { 1 });
    let (ny, nx) = (if h > 1 { 2usize } else { 1 }, if w > 1 { 2usize } else { 1 });
    let n = (ny * nx) as u32;
    let mut out = vec![0u8; h2 * w2 * 4];
    for y in 0..h2 {
        for x in 0..w2 {
            for c in 0..4 {
                let mut sum = 0u32;
                for dy in 0..ny {
                    for dx in 0..nx {
                        let (yy, xx) = (y * ny + dy, x * nx + dx);
                        debug_assert!(yy < rows && xx < cols);
                        sum += img[(yy * w + xx) * 4 + c] as u32;
                    }
                }
                out[(y * w2 + x) * 4 + c] = ((sum + n / 2) / n) as u8;
            }
        }
    }
    (out, h2, w2)
}

/// Replicate the last row/column until both sizes are multiples of 4 (`np.pad(..., mode="edge")`).
pub fn pad_edge4(img: Vec<u8>, h: usize, w: usize) -> (Vec<u8>, usize, usize) {
    let (ph, pw) = ((4 - h % 4) % 4, (4 - w % 4) % 4);
    if ph == 0 && pw == 0 {
        return (img, h, w);
    }
    let (nh, nw) = (h + ph, w + pw);
    let mut out = vec![0u8; nh * nw * 4];
    for y in 0..nh {
        let sy = y.min(h - 1);
        for x in 0..nw {
            let sx = x.min(w - 1);
            let (s, d) = ((sy * w + sx) * 4, (y * nw + x) * 4);
            out[d..d + 4].copy_from_slice(&img[s..s + 4]);
        }
    }
    (out, nh, nw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_and_sharp_blocks_round_trip_exactly() {
        // channel values that are exact RGB565 expansions survive a flat block
        let flat = [[8u8, 0, 255, 255]; 16];
        assert_eq!(decode_block(&encode_block(&flat)), flat);
        let mut sharp = [[0u8, 0, 0, 0]; 16];
        for k in 0..16 {
            if k % 5 == 0 {
                sharp[k] = [255, 255, 255, 255];
            }
        }
        let back = decode_block(&encode_block(&sharp));
        for k in 0..16 {
            assert_eq!(back[k][3], sharp[k][3], "alpha {k}");
            assert!(back[k][0].abs_diff(sharp[k][0]) <= 8);
        }
    }

    #[test]
    fn mask_blocks_keep_exact_zero_and_full_alpha_with_inner_values() {
        let mut px = [[0u8, 1, 0, 255]; 16];
        px[3][3] = 0;
        px[7][3] = 100;
        px[9][3] = 37;
        let back = decode_block(&encode_block(&px));
        assert_eq!(back[3][3], 0);
        assert_eq!(back[0][3], 255);
        assert!(back[7][3].abs_diff(100) <= 20);
    }

    #[test]
    fn box_half_follows_the_even_crop_and_unit_rules() {
        let mut img = vec![0u8; 3 * 1 * 4];
        img[0] = 10;
        img[4] = 21;
        img[8] = 200;
        let (out, h, w) = box_half(&img, 1, 3);
        assert_eq!((h, w), (1, 1));
        assert_eq!(out[0], 16); // (10 + 21 + 1) / 2, third column dropped
    }
}
