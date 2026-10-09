//! Raster algorithms used for the optional terrain masks (snow / fog) and DXT decoding.
//!
//! * `resize_l8` / `resize_f32` follow the separable convolution resampler used by Pillow
//!   (support-scaled filter, coefficients renormalised at the borders, 22-bit fixed point for 8-bit data),
//!   so the upsampled masks are real bicubic / bilinear interpolations.
//! * `gaussian_blur` is a separable Gaussian with a 4-sigma kernel and mirrored (scipy "reflect") borders.

use anyhow::{Result, bail};

#[derive(Clone, Copy)]
pub enum Filter {
    Bilinear,
    Bicubic,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Filter::Bicubic => {
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

struct Coeffs {
    bounds: Vec<(usize, usize)>, // (xmin, count)
    kernels: Vec<Vec<f64>>,
}

fn coeffs(in_size: usize, out_size: usize, filter: Filter) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ss = 1.0 / filterscale;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kernels = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax =
            (((center + support + 0.5) as i64).min(in_size as i64)).max(xmin as i64) as usize;
        let n = xmax - xmin;
        let mut k: Vec<f64> = (0..n)
            .map(|x| filter.eval(((x + xmin) as f64 - center + 0.5) * ss))
            .collect();
        let total: f64 = k.iter().sum();
        if total != 0.0 {
            for v in &mut k {
                *v /= total;
            }
        }
        bounds.push((xmin, n));
        kernels.push(k);
    }
    Coeffs { bounds, kernels }
}

/// Resample an 8-bit single channel image (rows top-first) with fixed point arithmetic.
pub fn resize_l8(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
) -> Result<Vec<u8>> {
    if src.len() != sw * sh {
        bail!(
            "resize_l8: source has {} bytes, expected {}",
            src.len(),
            sw * sh
        );
    }
    const PREC: u32 = 22;
    let fix = |kernels: &Vec<Vec<f64>>| -> Vec<Vec<i64>> {
        kernels
            .iter()
            .map(|k| {
                k.iter()
                    .map(|&v| {
                        let s = v * (1u64 << PREC) as f64;
                        if v < 0.0 {
                            (-0.5 + s) as i64
                        } else {
                            (0.5 + s) as i64
                        }
                    })
                    .collect()
            })
            .collect()
    };
    let clip = |ss: i64| -> u8 { (ss >> PREC).clamp(0, 255) as u8 };
    let ch = coeffs(sw, dw, filter);
    let ck = fix(&ch.kernels);
    let mut tmp = vec![0u8; dw * sh];
    for y in 0..sh {
        let row = &src[y * sw..(y + 1) * sw];
        for x in 0..dw {
            let (xmin, n) = ch.bounds[x];
            let mut acc: i64 = 1 << (PREC - 1);
            for i in 0..n {
                acc += row[xmin + i] as i64 * ck[x][i];
            }
            tmp[y * dw + x] = clip(acc);
        }
    }
    let cv = coeffs(sh, dh, filter);
    let kv = fix(&cv.kernels);
    let mut out = vec![0u8; dw * dh];
    for y in 0..dh {
        let (ymin, n) = cv.bounds[y];
        for x in 0..dw {
            let mut acc: i64 = 1 << (PREC - 1);
            for i in 0..n {
                acc += tmp[(ymin + i) * dw + x] as i64 * kv[y][i];
            }
            out[y * dw + x] = clip(acc);
        }
    }
    Ok(out)
}

/// Resample a float image (rows top-first) in floating point arithmetic.
pub fn resize_f32(
    src: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    filter: Filter,
) -> Result<Vec<f32>> {
    if src.len() != sw * sh {
        bail!(
            "resize_f32: source has {} values, expected {}",
            src.len(),
            sw * sh
        );
    }
    let ch = coeffs(sw, dw, filter);
    let mut tmp = vec![0f32; dw * sh];
    for y in 0..sh {
        let row = &src[y * sw..(y + 1) * sw];
        for x in 0..dw {
            let (xmin, n) = ch.bounds[x];
            let mut acc = 0f64;
            for i in 0..n {
                acc += row[xmin + i] as f64 * ch.kernels[x][i];
            }
            tmp[y * dw + x] = acc as f32;
        }
    }
    let cv = coeffs(sh, dh, filter);
    let mut out = vec![0f32; dw * dh];
    for y in 0..dh {
        let (ymin, n) = cv.bounds[y];
        for x in 0..dw {
            let mut acc = 0f64;
            for i in 0..n {
                acc += tmp[(ymin + i) * dw + x] as f64 * cv.kernels[y][i];
            }
            out[y * dw + x] = acc as f32;
        }
    }
    Ok(out)
}

fn reflect(mut i: i64, n: i64) -> usize {
    // scipy "reflect": d c b a | a b c d | d c b a
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - i - 1;
        } else {
            return i as usize;
        }
    }
}

pub fn gaussian_blur(data: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let mut k: Vec<f64> = (-radius..=radius)
        .map(|x| (-0.5 * (x as f64) * (x as f64) / (sigma * sigma)).exp())
        .collect();
    let total: f64 = k.iter().sum();
    for v in &mut k {
        *v /= total;
    }
    let mut tmp = vec![0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let xi = reflect(x as i64 + j as i64 - radius, w as i64);
                acc += data[y * w + xi] as f64 * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let yi = reflect(y as i64 + j as i64 - radius, h as i64);
                acc += tmp[yi * w + x] * kv;
            }
            out[y * w + x] = acc as f32;
        }
    }
    out
}

/// 4-connected binary dilation, one iteration, zero outside the image.
pub fn dilate_cross(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            out[i] = mask[i]
                || (x > 0 && mask[i - 1])
                || (x + 1 < w && mask[i + 1])
                || (y > 0 && mask[i - w])
                || (y + 1 < h && mask[i + w]);
        }
    }
    out
}

fn expand565(c: u16) -> [i32; 3] {
    let r = ((c >> 11) & 31) as i32;
    let g = ((c >> 5) & 63) as i32;
    let b = (c & 31) as i32;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// Decode the red channel of the first mip level of a BC3 (DXT5) DDS file.
pub fn dxt5_red(dds: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    if dds.len() < 128 || &dds[0..4] != b"DDS " || &dds[84..88] != b"DXT5" {
        bail!("Expected a DXT5 DDS file");
    }
    let h = u32::from_le_bytes(dds[12..16].try_into()?) as usize;
    let w = u32::from_le_bytes(dds[16..20].try_into()?) as usize;
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let need = 128 + bw * bh * 16;
    if dds.len() < need {
        bail!("Truncated DXT5 DDS: {} bytes, need {need}", dds.len());
    }
    let mut out = vec![0u8; w * h];
    for by in 0..bh {
        for bx in 0..bw {
            let blk = &dds[128 + (by * bw + bx) * 16 + 8..128 + (by * bw + bx) * 16 + 16];
            let c0 = u16::from_le_bytes([blk[0], blk[1]]);
            let c1 = u16::from_le_bytes([blk[2], blk[3]]);
            let bits = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
            let (a, b) = (expand565(c0)[0], expand565(c1)[0]);
            // BC2/BC3 colour blocks always use the four colour mode.
            let pal = [a, b, (2 * a + b) / 3, (a + 2 * b) / 3];
            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx * 4 + px, by * 4 + py);
                    if x < w && y < h {
                        out[y * w + x] = pal[((bits >> (2 * (py * 4 + px))) & 3) as usize] as u8;
                    }
                }
            }
        }
    }
    Ok((w, h, out))
}
