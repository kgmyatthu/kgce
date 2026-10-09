//! Winter snow coverage of the enlarged America theatre.
//!
//! The ETW world snow mask (`campaign_snow_mask_winter.dds`, 2048x1024 DXT5, grey, one pixel = 1.25 world units) was
//! painted for the original America playable rectangle: its snow zone ends in a rounded edge about 18 units inside the
//! old west border (x -760), so everything the theatre enlargement adds to the west would be snow free next to a snow
//! field. This module continues that snow zone westwards.
//!
//! Measurements of the shipped ETW data this is derived from:
//! * America's snow is a pure latitude zone (255 north of y ~ 300, a 30 unit ramp to 0 near y ~ 270, nothing south);
//!   regressing it on the heightmap shows no elevation dependence (best lift 0), so the zone is continued as a
//!   latitude profile taken from the original mask itself, row by row, just inside the old west edge.
//! * Elsewhere on the same mask (Europe/Asia) the painter did follow the terrain: snow = lowland profile of
//!   `y + 0.6 * height` (best-fit lift over the ETW heightmap, residual 0.038 against 0.050 without). Terrain that the
//!   enlargement adds above the anchor strip's own elevation therefore moves the snow line south by that law, fading
//!   in with distance from the join so that nothing changes at the old data.
//! * North of the profile the original mask is clipped at the old north limit, so the profile is made monotone
//!   towards the north (coverage never decreases northwards) when extended.
//!
//! Only a band west of the detected old edge is rewritten: full replacement up to 6 pixels east of the edge, smooth
//! blend back to the untouched original over the next 24 pixels. Rows outside the fixed America theatre, sea and
//! all other continents keep their original values, and a patched DXT5 file keeps every untouched block bit-exact.

use super::raster;
use crate::regionmap::theatres::FIXED_BOUNDS;
use anyhow::{Context, Result, bail, ensure};

pub const MASK_W: usize = 2048;
pub const MASK_H: usize = 1024;
const HM_W: usize = 8192;
const HM_H: usize = 4096;
/// Heightmap pixels per mask pixel in each direction.
const BLOCK: usize = HM_W / MASK_W;

/// World placement of the mask: top-left corner and world units per mask pixel (the whole 2560 x 1280 world).
const WORLD_X0: f32 = -1280.0;
const WORLD_Y_TOP: f32 = 640.0;
const UNIT: f32 = 2560.0 / MASK_W as f32;

/// West edge of the original America playable rectangle; the painted snow edge is searched around it.
const OLD_WEST_X: f32 = -760.0;
const EDGE_SEARCH: (f32, f32) = (OLD_WEST_X - 20.0, OLD_WEST_X + 70.0);
/// A pixel counts as snow when it reaches this level (of 255) over a run of `EDGE_RUN` pixels.
const EDGE_LEVEL: u8 = 16;
const EDGE_RUN: usize = 4;
/// Fewest detected rows for the source mask to be accepted as the known America mask (the shipped one has 133).
const EDGE_MIN_ROWS: usize = 32;
const EDGE_MEDIAN_ROWS: usize = 9;

/// Latitude profile anchor strip, in pixels east of the detected edge (clear of the painted edge ramp).
const ANCHOR: (usize, usize) = (12, 28);
/// Pure continuation up to `FULL` pixels east of the edge, blended back to the original until `END`.
const BLEND_FULL: usize = 6;
const BLEND_END: usize = 30;
/// The terrain lift fades in over this many pixels west of `BLEND_END`.
const LIFT_FADE: f32 = 40.0;
/// World units of snow-line shift per heightmap unit of elevation above the anchor strip (fitted on ETW Europe).
const LIFT_UNITS_PER_HEIGHT: f32 = 0.6;
const LIFT_MAX_HEIGHT: f32 = 110.0;

const HEIGHT_SIGMA: f64 = 5.0;
const PROFILE_SIGMA: f32 = 1.5;
const REFERENCE_HEIGHT_SIGMA: f32 = 4.0;
const LAND_DILATIONS: usize = 2;
/// Pixels around the altered area that the window also covers (4 sigma of the widest blur is 20).
const WINDOW_MARGIN: usize = 24;
const LAND_SIGMA: f64 = 1.2;

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn blur_1d(data: &[f32], sigma: f32) -> Vec<f32> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|x| (-0.5 * (x as f32 / sigma).powi(2)).exp())
        .collect();
    let total: f32 = kernel.iter().sum();
    let n = data.len() as i64;
    (0..n)
        .map(|i| {
            kernel
                .iter()
                .enumerate()
                .map(|(j, k)| {
                    let at = (i + j as i64 - radius).clamp(0, n - 1);
                    data[at as usize] * k
                })
                .sum::<f32>()
                / total
        })
        .collect()
}

/// Per-row column of the painted west snow edge of the original America block, smoothed over neighbouring rows.
fn west_edges(mask: &[u8]) -> Result<Vec<usize>> {
    let col = |x: f32| ((x - WORLD_X0) / UNIT).round() as usize;
    let (c0, c1) = (col(EDGE_SEARCH.0), col(EDGE_SEARCH.1));
    let mut found: Vec<Option<usize>> = vec![None; MASK_H];
    for (r, slot) in found.iter_mut().enumerate() {
        let row = &mask[r * MASK_W..(r + 1) * MASK_W];
        *slot = (c0..c1).find(|&c| row[c..c + EDGE_RUN].iter().all(|&v| v >= EDGE_LEVEL));
    }
    let mut seen: Vec<usize> = found.iter().flatten().copied().collect();
    ensure!(
        seen.len() >= EDGE_MIN_ROWS,
        "The ETW winter snow mask has no America west edge near x {OLD_WEST_X} ({} rows)",
        seen.len()
    );
    seen.sort_unstable();
    let fallback = seen[seen.len() / 2];
    let raw: Vec<usize> = found.iter().map(|f| f.unwrap_or(fallback)).collect();
    let half = EDGE_MEDIAN_ROWS / 2;
    Ok((0..MASK_H)
        .map(|r| {
            let mut window: Vec<usize> =
                raw[r.saturating_sub(half)..(r + half + 1).min(MASK_H)].to_vec();
            window.sort_unstable();
            window[window.len() / 2]
        })
        .collect())
}

/// Mask-pixel window `[r0, r1) x [c0, c1)` over which the continuation is computed. It is the altered area plus a
/// margin of the widest filter reach (4 sigma of the elevation blur), so every pixel that is written sees exactly the
/// filter inputs a whole-mask computation would give it.
struct Window {
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
}

impl Window {
    fn width(&self) -> usize {
        self.c1 - self.c0
    }

    fn height(&self) -> usize {
        self.r1 - self.r0
    }

    /// Index of mask pixel (`r`, `c`) in a window-sized buffer.
    fn at(&self, r: usize, c: usize) -> usize {
        (r - self.r0) * self.width() + (c - self.c0)
    }
}

/// Mean of each `BLOCK` x `BLOCK` heightmap cell inside the window, as the mask-resolution elevation field.
fn cell_heights(heights: &[u8], win: &Window) -> Vec<f32> {
    let mut out = Vec::with_capacity(win.width() * win.height());
    for r in win.r0..win.r1 {
        for c in win.c0..win.c1 {
            let mut sum = 0u32;
            for dy in 0..BLOCK {
                let row = (r * BLOCK + dy) * HM_W + c * BLOCK;
                sum += heights[row..row + BLOCK]
                    .iter()
                    .map(|&v| v as u32)
                    .sum::<u32>();
            }
            out.push(sum as f32 / (BLOCK * BLOCK) as f32);
        }
    }
    out
}

/// Soft land coverage inside the window: any land in a cell, grown a little, then blurred so coasts fade.
fn soft_land(heights: &[u8], win: &Window) -> Vec<f32> {
    let mut land = Vec::with_capacity(win.width() * win.height());
    for r in win.r0..win.r1 {
        for c in win.c0..win.c1 {
            land.push((0..BLOCK).any(|dy| {
                let row = (r * BLOCK + dy) * HM_W + c * BLOCK;
                heights[row..row + BLOCK].iter().any(|&v| v > 0)
            }));
        }
    }
    for _ in 0..LAND_DILATIONS {
        land = raster::dilate_cross(&land, win.width(), win.height());
    }
    let field: Vec<f32> = land.iter().map(|&l| if l { 1.0 } else { 0.0 }).collect();
    raster::gaussian_blur(&field, win.width(), win.height(), LAND_SIGMA)
        .into_iter()
        .map(|v| (v * 1.6).clamp(0.0, 1.0))
        .collect()
}

/// Continue the winter snow zone of the original America mask over the area added by the theatre enlargement.
///
/// `mask` is the 2048x1024 red channel of the ETW winter mask (rows top-first), `heights` the 8192x4096 elevation of
/// the finished map (rows top-first, unshaped fixed-art elevation, 0 = sea). Returns the extended mask.
pub fn extend_winter_mask(mask: &[u8], heights: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        mask.len() == MASK_W * MASK_H,
        "Unexpected winter snow mask size"
    );
    ensure!(heights.len() == HM_W * HM_H, "Unexpected heightmap size");
    let [[x0, y0], [_, y1]] = FIXED_BOUNDS
        .iter()
        .find(|(key, _)| *key == "1")
        .map(|(_, rect)| *rect)
        .context("No fixed America theatre bounds")?;

    let edge = west_edges(mask)?;
    let theatre_rows: Vec<usize> = (0..MASK_H)
        .filter(|&r| {
            let y = WORLD_Y_TOP - (r as f32 + 0.5) * UNIT;
            y >= y0 && y <= y1
        })
        .collect();
    ensure!(
        !theatre_rows.is_empty(),
        "America theatre bounds lie outside the snow mask"
    );
    let (rows_lo, rows_hi) = (theatre_rows[0], theatre_rows[theatre_rows.len() - 1] + 1);
    let first_col = (((x0 - WORLD_X0) / UNIT).floor() as usize).saturating_sub(2);
    let (r0, r1) = (
        rows_lo.saturating_sub(WINDOW_MARGIN),
        (rows_hi + WINDOW_MARGIN).min(MASK_H),
    );
    let reach = edge[r0..r1].iter().max().copied().unwrap_or(0) + BLEND_END;
    let win = Window {
        r0,
        r1,
        c0: first_col.saturating_sub(WINDOW_MARGIN),
        c1: (reach + WINDOW_MARGIN).min(MASK_W),
    };
    let hs = raster::gaussian_blur(
        &cell_heights(heights, &win),
        win.width(),
        win.height(),
        HEIGHT_SIGMA,
    );
    let land = soft_land(heights, &win);

    // latitude profile (snow) per row, coverage never decreasing northwards (rows count southwards): suffix maximum
    let mut profile = vec![0f32; MASK_H];
    for (r, p) in profile.iter_mut().enumerate() {
        let (a, b) = (edge[r] + ANCHOR.0, edge[r] + ANCHOR.1);
        *p = mask[r * MASK_W + a..r * MASK_W + b]
            .iter()
            .map(|&v| v as f32)
            .sum::<f32>()
            / (b - a) as f32;
    }
    let mut envelope = blur_1d(&profile, PROFILE_SIGMA);
    for r in (0..MASK_H - 1).rev() {
        envelope[r] = envelope[r].max(envelope[r + 1]);
    }
    let sample = |row: f32| -> f32 {
        let row = row.clamp(0.0, (MASK_H - 1) as f32);
        let (i, f) = (row.floor() as usize, row.fract());
        let j = (i + 1).min(MASK_H - 1);
        envelope[i] * (1.0 - f) + envelope[j] * f
    };
    // reference elevation of the anchor strip just inside the old edge, per window row
    let reference: Vec<f32> = (win.r0..win.r1)
        .map(|r| {
            let (a, b) = (edge[r] + ANCHOR.0, edge[r] + ANCHOR.1);
            (a..b).map(|c| hs[win.at(r, c)]).sum::<f32>() / (b - a) as f32
        })
        .collect();
    let reference = blur_1d(&reference, REFERENCE_HEIGHT_SIGMA);

    let mut out = mask.to_vec();
    for r in rows_lo..rows_hi {
        let end = edge[r] + BLEND_END;
        for c in first_col..end {
            let t = (end - c) as f32;
            let blend = smoothstep(t / (BLEND_END - BLEND_FULL) as f32);
            let fade = smoothstep(t / LIFT_FADE);
            let w = win.at(r, c);
            let lift = (hs[w] - reference[r - win.r0]).clamp(0.0, LIFT_MAX_HEIGHT) * fade;
            let snow = sample(r as f32 - lift * LIFT_UNITS_PER_HEIGHT / UNIT) * land[w];
            let i = r * MASK_W + c;
            out[i] = (blend * snow + (1.0 - blend) * mask[i] as f32)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------- DXT5 patching

fn expand565(c: u16) -> [i32; 3] {
    let (r, g, b) = (
        ((c >> 11) & 31) as i32,
        ((c >> 5) & 63) as i32,
        (c & 31) as i32,
    );
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

fn grey565(v: u8) -> u16 {
    let r = (v as u32 * 31 + 127) / 255;
    let g = (v as u32 * 63 + 127) / 255;
    ((r << 11) | (g << 5) | r) as u16
}

/// Colour half (8 bytes) of a BC3 block for 16 grey values (rows top-first): max/min endpoints, nearest of the four
/// palette greys. `c0 >= c1` always holds, so every decoder uses the four colour mode.
fn encode_grey_color(px: &[u8; 16]) -> [u8; 8] {
    let (lo, hi) = (*px.iter().min().unwrap(), *px.iter().max().unwrap());
    let (c0, c1) = (grey565(hi), grey565(lo));
    let (p0, p1) = (expand565(c0), expand565(c1));
    let palette: Vec<i32> = [
        p0,
        p1,
        std::array::from_fn(|k| (2 * p0[k] + p1[k]) / 3),
        std::array::from_fn(|k| (p0[k] + 2 * p1[k]) / 3),
    ]
    .iter()
    .map(|p| p[0] + p[1] + p[2])
    .collect();
    let mut bits = 0u32;
    for (i, &v) in px.iter().enumerate() {
        let idx = (0..4)
            .min_by_key(|&k| (palette[k] - 3 * v as i32).abs())
            .unwrap();
        bits |= (idx as u32) << (2 * i);
    }
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
    out
}

struct Level {
    w: usize,
    h: usize,
    px: Vec<u8>,
    changed: Vec<bool>,
}

fn next_level(l: &Level) -> Level {
    let (w, h) = ((l.w / 2).max(1), (l.h / 2).max(1));
    let mut px = vec![0u8; w * h];
    let mut changed = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut n, mut any) = (0u32, 0u32, false);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = ((2 * x + dx).min(l.w - 1), (2 * y + dy).min(l.h - 1));
                    sum += l.px[sy * l.w + sx] as u32;
                    any |= l.changed[sy * l.w + sx];
                    n += 1;
                }
            }
            px[y * w + x] = ((sum + n / 2) / n) as u8;
            changed[y * w + x] = any;
        }
    }
    Level { w, h, px, changed }
}

/// Rewrite the colour blocks of a grey 2048x1024 DXT5 mip chain that cover pixels where `extended` differs from
/// `original` (the decoded top level); every other block and the whole header stay bit-identical. Mip levels are
/// box-filtered from `extended` over the same area.
pub fn patch_winter_dds(dds: &[u8], original: &[u8], extended: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        dds.len() >= 128 && &dds[0..4] == b"DDS " && &dds[84..88] == b"DXT5",
        "Expected a DXT5 DDS file"
    );
    let dim = |at: usize| u32::from_le_bytes(dds[at..at + 4].try_into().unwrap()) as usize;
    ensure!(
        (dim(16), dim(12)) == (MASK_W, MASK_H),
        "Winter snow mask is {}x{}, expected {MASK_W}x{MASK_H}",
        dim(16),
        dim(12)
    );
    ensure!(
        original.len() == MASK_W * MASK_H && extended.len() == original.len(),
        "Mask size mismatch"
    );
    let mips = dim(28).max(1);

    let mut level = Level {
        w: MASK_W,
        h: MASK_H,
        px: extended.to_vec(),
        changed: extended.iter().zip(original).map(|(a, b)| a != b).collect(),
    };
    let mut out = dds.to_vec();
    let mut offset = 128;
    for n in 0..mips {
        let (bw, bh) = (level.w.div_ceil(4), level.h.div_ceil(4));
        let size = bw * bh * 16;
        if offset + size > out.len() {
            bail!("Truncated DXT5 mip chain at level {n}");
        }
        // blocks of the 4x2, 2x1 and 1x1 tail levels are partial: their missing pixels repeat the edge
        for by in 0..bh {
            for bx in 0..bw {
                let mut px = [0u8; 16];
                let mut touched = false;
                for (i, p) in px.iter_mut().enumerate() {
                    let (x, y) = (
                        (bx * 4 + i % 4).min(level.w - 1),
                        (by * 4 + i / 4).min(level.h - 1),
                    );
                    *p = level.px[y * level.w + x];
                    touched |= level.changed[y * level.w + x];
                }
                if touched {
                    let block = offset + (by * bw + bx) * 16;
                    out[block + 8..block + 16].copy_from_slice(&encode_grey_color(&px));
                }
            }
        }
        offset += size;
        if level.w == 1 && level.h == 1 {
            break;
        }
        level = next_level(&level);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EDGE_COL: usize = 430;
    const NORTH_ROW: usize = 170; // y 427.5
    const SNOW_ROW: usize = 280; // y 290: last full row, 20 rows of ramp follow

    /// Hand-made America-like winter mask: rounded 4 pixel west ramp at `EDGE_COL`, full snow north of `SNOW_ROW`,
    /// a 20 row southern ramp, nothing else; 255 east of the edge up to x ~ -270.
    fn source_mask() -> Vec<u8> {
        let mut mask = vec![0u8; MASK_W * MASK_H];
        for r in NORTH_ROW..SNOW_ROW + 20 {
            let level = if r <= SNOW_ROW {
                255.0
            } else {
                255.0 * (1.0 - (r - SNOW_ROW) as f32 / 20.0)
            };
            for c in EDGE_COL..900 {
                let across = ((c - EDGE_COL + 1) as f32 * 64.0).min(255.0) / 255.0;
                mask[r * MASK_W + c] = (level * across) as u8;
            }
        }
        mask
    }

    /// Low land over the whole enlarged theatre (x -888..), a high massif south of the snow zone and one low spot.
    fn terrain() -> Vec<u8> {
        let mut h = vec![0u8; HM_W * HM_H];
        let first = ((-888.0 - WORLD_X0) * 3.2) as usize;
        for y in 0..HM_H {
            for x in first..first + 2000 {
                h[y * HM_W + x] = 10;
            }
        }
        for y in 330 * 4..350 * 4 {
            for x in 340 * 4..360 * 4 {
                h[y * HM_W + x] = 150;
            }
        }
        h
    }

    #[test]
    fn continues_the_snow_zone_westwards_without_a_seam() {
        let mask = source_mask();
        let out = extend_winter_mask(&mask, &terrain()).unwrap();

        // old rows/columns far from the join are untouched: east of the blend, and outside the theatre rows
        for r in 0..MASK_H {
            let beyond = 500;
            assert_eq!(
                out[r * MASK_W + beyond..(r + 1) * MASK_W],
                mask[r * MASK_W + beyond..(r + 1) * MASK_W]
            );
        }
        let outside = [0..120, 520..MASK_H];
        for rows in outside {
            for r in rows {
                assert_eq!(
                    out[r * MASK_W..(r + 1) * MASK_W],
                    mask[r * MASK_W..(r + 1) * MASK_W],
                    "row {r}"
                );
            }
        }

        // snow zone continues over the former void: full in the north, none far south, the 0 -> 255 ramp in between
        assert!(out[200 * MASK_W + 330] >= 250, "north of the snow line");
        assert!(
            out[200 * MASK_W + 420] >= 250,
            "between old edge and old west border"
        );
        assert_eq!(out[320 * MASK_W + 400], 0, "south of the snow line");
        let ramp = out[(SNOW_ROW + 10) * MASK_W + 330];
        assert!(
            (100..160).contains(&ramp),
            "latitude ramp continues, got {ramp}"
        );

        // no seam: neighbouring pixels never jump along rows that cross the join
        for r in [200, 240, SNOW_ROW + 5, SNOW_ROW + 12] {
            for c in 320..520 {
                let step = (out[r * MASK_W + c] as i32 - out[r * MASK_W + c + 1] as i32).abs();
                assert!(step <= 40, "row {r} col {c}: step {step}");
            }
        }
        // sea (no land) west of the enlarged theatre stays empty
        assert_eq!(out[200 * MASK_W + 200], 0);
    }

    #[test]
    fn high_new_terrain_lowers_the_snow_line() {
        let out = extend_winter_mask(&source_mask(), &terrain()).unwrap();
        assert!(
            out[340 * MASK_W + 350] > 100,
            "massif {}",
            out[340 * MASK_W + 350]
        );
        assert_eq!(out[340 * MASK_W + 400], 0, "same latitude, lowland");
    }

    fn test_dds(mask: &[u8]) -> Vec<u8> {
        let mut dds = vec![0u8; 128];
        dds[0..4].copy_from_slice(b"DDS ");
        dds[12..16].copy_from_slice(&(MASK_H as u32).to_le_bytes());
        dds[16..20].copy_from_slice(&(MASK_W as u32).to_le_bytes());
        dds[28..32].copy_from_slice(&12u32.to_le_bytes());
        dds[84..88].copy_from_slice(b"DXT5");
        let mut level = Level {
            w: MASK_W,
            h: MASK_H,
            px: mask.to_vec(),
            changed: vec![false; mask.len()],
        };
        loop {
            for by in 0..level.h.div_ceil(4) {
                for bx in 0..level.w.div_ceil(4) {
                    let mut px = [0u8; 16];
                    for (i, p) in px.iter_mut().enumerate() {
                        let (x, y) = (
                            (bx * 4 + i % 4).min(level.w - 1),
                            (by * 4 + i / 4).min(level.h - 1),
                        );
                        *p = level.px[y * level.w + x];
                    }
                    dds.extend_from_slice(&[255, 255, 0, 0, 0, 0, 0, 0]);
                    dds.extend_from_slice(&encode_grey_color(&px));
                }
            }
            if level.w == 1 && level.h == 1 {
                return dds;
            }
            level = next_level(&level);
        }
    }

    #[test]
    fn patched_dds_changes_only_the_extended_blocks() {
        let mask = source_mask();
        let dds = test_dds(&mask);
        let (_, _, decoded) = raster::dxt5_red(&dds).unwrap();
        let extended = extend_winter_mask(&decoded, &terrain()).unwrap();
        let patched = patch_winter_dds(&dds, &decoded, &extended).unwrap();
        assert_eq!(patched.len(), dds.len());
        assert_eq!(patched[..128], dds[..128]);
        let (_, _, after) = raster::dxt5_red(&patched).unwrap();
        // untouched area decodes exactly as before
        for r in 0..MASK_H {
            assert_eq!(
                after[r * MASK_W + 512..(r + 1) * MASK_W],
                decoded[r * MASK_W + 512..(r + 1) * MASK_W]
            );
        }
        // the extension is in the file, within the block quantisation error of the grey encoding
        let near = |r: usize, c: usize| {
            (after[r * MASK_W + c] as i32 - extended[r * MASK_W + c] as i32).abs()
        };
        assert!(near(200, 330) <= 6 && near(200, 420) <= 6 && near(320, 400) <= 6);
        // alpha halves of every block are untouched
        for (a, b) in patched[128..].chunks(16).zip(dds[128..].chunks(16)) {
            assert_eq!(a[..8], b[..8]);
        }
    }

    fn mip_offset(level: usize) -> usize {
        128 + (0..level)
            .map(|k| (MASK_W >> k).max(1).div_ceil(4) * (MASK_H >> k).max(1).div_ceil(4) * 16)
            .sum::<usize>()
    }

    /// First pixel (red) of the first block of a mip level.
    fn first_pixel(file: &[u8], level: usize) -> i32 {
        let block = &file[mip_offset(level) + 8..mip_offset(level) + 16];
        let (c0, c1) = (
            u16::from_le_bytes([block[0], block[1]]),
            u16::from_le_bytes([block[2], block[3]]),
        );
        let (a, b) = (expand565(c0)[0], expand565(c1)[0]);
        [a, b, (2 * a + b) / 3, (a + 2 * b) / 3][(block[4] & 3) as usize]
    }

    fn box_filtered(px: &[u8], level: usize) -> i32 {
        let mut l = Level {
            w: MASK_W,
            h: MASK_H,
            px: px.to_vec(),
            changed: vec![false; px.len()],
        };
        for _ in 0..level {
            l = next_level(&l);
        }
        l.px[0] as i32
    }

    #[test]
    fn coarse_mips_including_partial_block_levels_follow_the_extension() {
        let dds = test_dds(&source_mask());
        let (_, _, decoded) = raster::dxt5_red(&dds).unwrap();
        let extended = extend_winter_mask(&decoded, &terrain()).unwrap();
        let patched = patch_winter_dds(&dds, &decoded, &extended).unwrap();
        // levels 9 (4x2), 10 (2x1) and 11 (1x1) consist of partial blocks; 4 grey levels per block bound the error
        // by a sixth of the block's range (level 9) or the 5-bit endpoint quantisation (levels 10, 11)
        for (level, tolerance) in [(9, 10), (10, 5), (11, 5)] {
            let want = box_filtered(&extended, level);
            let got = first_pixel(&patched, level);
            assert!(
                (got - want).abs() <= tolerance,
                "level {level}: {got} vs box-filtered extension {want}"
            );
        }
        // the added snow is large enough at level 9 for a stale tail to be told apart from a patched one
        let (stale, want) = (first_pixel(&dds, 9), box_filtered(&extended, 9));
        assert!(want - stale >= 12, "fixture too weak: {stale} -> {want}");
    }
}
