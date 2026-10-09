//! Native port of `tools/terrain_paint/fixed_america.py`: enlarge the America theatre (theatre "1") from the
//! playable rect (-760,0)..(-340,460) to (-888,0)..(-340,460).
//!
//! Everything is derived at run time from the user's own baseline tiles / raw heightmap (read only, through
//! [`TileSource`] and `baseline_heights`) and the public DEM:
//!   * the western decorative frame (brown band + parchment margin) is relocated 128 units west and the north /
//!     south frame bands are extended with seam-optimised chunks of the same vanilla bands;
//!   * the old frame, the inner parchment margin, the dark sea vignette along the old playable edge, the
//!     unexplored-parchment void inside the old rect and the torn-parchment remnants are repainted with the shared
//!     painter (DEM terrain/sea, palette / textures LEARNED from the vanilla interior);
//!   * only changed 4x4 DXT5 blocks are re-encoded, mips are rebuilt for changed regions;
//!   * the raw heightmap is rewritten over the painter ownership window, tapered to the kept vanilla heights.
//!
//! Canvas pixel grids use row 0 = NORTH; the heightmap stream row 0 = SOUTH.

use std::collections::BTreeMap;

use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

use super::TheatreResult;
use super::dem::{Dem, z_to_h};
use super::grid::{Arr2, Rgb8, Rgbf};
use super::painter::{
    Canvas, DemTerrain, Edge, Learned, MatchOpts, Painter, SeaSigma, Variant, colour_match,
    fade_seam, fill_cols, smoothstep,
};
use super::prims::{self, Mode};
use super::resample::Pcg64;
use super::stpx::{self, CanvasWindow, TileSource};

mod seam;

// ---------------------------------------------------------------------------------------------- constants
const S0: f64 = 25.6;
const WORLD_X0: f64 = -1280.0;
const WORLD_Y0: f64 = -640.0;
const HM_PX: f64 = 3.2;
const HM_W: usize = 8192;
const HM_H: usize = 4096;

const CANVAS_TX0: usize = 18;
const CANVAS_TY0: usize = 8;
const CANVAS_NTX: usize = 30;
const CANVAS_NTY: usize = 26;
const CANVAS_X0: f64 = WORLD_X0 + CANVAS_TX0 as f64 * 20.0; // -920
const CANVAS_YT: f64 = -WORLD_Y0 - CANVAS_TY0 as f64 * 20.0; // 480
const W_C: usize = CANVAS_NTX * 512; // 15360
const H_C: usize = CANVAS_NTY * 512; // 13312

const C_FB0: usize = 3520; // px_x(-782.5): west edge of the frame copy
const C_OLD_W: usize = 4096; // px_x(-760): old playable west edge / east end of the frame copy
const C_NEW_W: usize = 819; // px_x(-888): new playable west edge
const SHIFT_W: i64 = -3277;
const C_FB1: usize = 243; // C_FB0 + SHIFT_W
const C_BAND_SEAM: usize = 3840; // px_x(-770)
const C_BAND_SRC_END: usize = 6400; // px_x(-670)
const R_BAND_N: usize = 640; // px_y(455)
const R_BAND_S: usize = 12237; // px_y(2)
const R_WIN0: usize = 640; // px_y(455)
const R_WIN1: usize = 12250; // px_y(1.5)
const C_WIN0: usize = 4160; // px_x(-757.5)
const C_WIN1: usize = 14784; // px_x(-342.5)
const C_HEAL1: usize = 4403; // px_x(-748)
const HEAL_PX: f32 = 150.0;
const REM_PX: usize = 140;
const SEA_SIGMA_NEAR: f32 = 45.0;
const SEA_SIGMA_STRIP: f32 = 320.0;
const STRIP_CHUNK_END: usize = 4992; // px_x(-725)
const VOID_HM_STRAY: u8 = 13;
const EDGE_BASE: f64 = 2.6;
const P0_AMER: [f32; 3] = [156.7, 145.1, 116.2];

const HM_ROW_TOP: usize = 3584; // round((CANVAS_YT - WORLD_Y0) * HM_PX)
const HM_OFF: usize = 9216; // round((CANVAS_X0 - WORLD_X0) * HM_PX) * 8

const CANVAS_WIN: CanvasWindow = CanvasWindow {
    tx0: CANVAS_TX0,
    ty0: CANVAS_TY0,
    ntx: CANVAS_NTX,
    nty: CANVAS_NTY,
};

fn px_x(x: f64) -> usize {
    ((x - CANVAS_X0) * S0).round_ties_even() as usize
}

fn px_y(y: f64) -> usize {
    ((CANVAS_YT - y) * S0).round_ties_even() as usize
}

// ---------------------------------------------------------------------------------------------- small helpers
fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16)
}

/// `f(row, out_row)` over every row of `data` on all cores.
fn par_rows<T: Send>(data: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = data.len() / row_len;
    if rows == 0 {
        return;
    }
    let per = rows.div_ceil(threads().min(rows));
    std::thread::scope(|s| {
        for (ci, chunk) in data.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * per + k, row);
                }
            });
        }
    });
}

/// Vanilla heightmap value under canvas pixel `(r, c)` (8x8 canvas px per heightmap px, hm row 0 = south).
#[inline]
fn hm_at(hm: &[u8], r: usize, c: usize) -> u8 {
    hm[(HM_ROW_TOP - 1 - r / 8) * HM_W + (c + HM_OFF) / 8]
}

fn rnd(v: f32) -> f32 {
    v.round_ties_even()
}

/// np.median of a 256-bin histogram of integer samples.
fn hist_median(h: &[u32]) -> f64 {
    let n: u64 = h.iter().map(|&v| v as u64).sum();
    if n == 0 {
        return f64::NAN;
    }
    let kth = |k: u64| -> usize {
        let mut acc = 0u64;
        for (i, &v) in h.iter().enumerate() {
            acc += v as u64;
            if acc > k {
                return i;
            }
        }
        255
    };
    (kth((n - 1) / 2) + kth(n / 2)) as f64 / 2.0
}

fn jr(v: f64, nd: i32) -> Value {
    let k = 10f64.powi(nd);
    json!((v * k).round_ties_even() / k)
}

// ---------------------------------------------------------------------------------------------- packed bit masks
/// Full-canvas packed bool mask (`np.packbits(axis=1)` layout: MSB first).
struct Bits {
    stride: usize,
    d: Vec<u8>,
}

impl Bits {
    fn new() -> Bits {
        let stride = W_C.div_ceil(8);
        Bits {
            stride,
            d: vec![0; H_C * stride],
        }
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> bool {
        self.d[r * self.stride + c / 8] & (0x80 >> (c % 8)) != 0
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize) {
        self.d[r * self.stride + c / 8] |= 0x80 >> (c % 8);
    }

    /// OR a window mask whose top-left corner is canvas `(r0, c0)`.
    fn or_window(&mut self, r0: usize, c0: usize, m: &Arr2<bool>) {
        for r in 0..m.h {
            let row = m.row(r);
            for (c, &v) in row.iter().enumerate() {
                if v {
                    self.set(r0 + r, c0 + c);
                }
            }
        }
    }

    fn set_rect(&mut self, r0: usize, r1: usize, c0: usize, c1: usize) {
        for r in r0..r1 {
            for c in c0..c1 {
                self.set(r, c);
            }
        }
    }

    fn unpack(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> Arr2<bool> {
        let mut out = Arr2::new(r1 - r0, c1 - c0, false);
        for r in r0..r1 {
            let o = out.row_mut(r - r0);
            for (k, v) in o.iter_mut().enumerate() {
                *v = self.get(r, c0 + k);
            }
        }
        out
    }

    fn any_in(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> bool {
        (r0..r1).any(|r| (c0..c1).any(|c| self.get(r, c)))
    }

    /// `(rmin, rmax, cmin, cmax)` of the set pixels.
    fn bounds(&self) -> Option<(usize, usize, usize, usize)> {
        let mut b: Option<(usize, usize, usize, usize)> = None;
        for r in 0..H_C {
            let row = &self.d[r * self.stride..(r + 1) * self.stride];
            let Some(fb) = row.iter().position(|&v| v != 0) else {
                continue;
            };
            let lb = row.iter().rposition(|&v| v != 0).unwrap();
            let cmin = fb * 8 + row[fb].leading_zeros() as usize;
            let cmax = lb * 8 + 7 - row[lb].trailing_zeros() as usize;
            b = Some(match b {
                None => (r, r, cmin, cmax),
                Some((a, _, c, d)) => (a, r, c.min(cmin), d.max(cmax)),
            });
        }
        b
    }
}

struct Masks {
    mp: Bits,
    vc: Bits,
    vland: Bits,
}

// ---------------------------------------------------------------------------------------------- learning
struct LearnInfo {
    p0: [f64; 3],
    land_lut_samples_total: u64,
    h_max_with_samples: usize,
    hp_std: [f64; 2],
    sea_patch_world_xyxy: [f64; 4],
    land_lum_std: f64,
}

/// Mean/std (population) accumulators over f64 values.
fn std_of(sum: f64, sumsq: f64, n: f64) -> f64 {
    let m = sum / n;
    (sumsq / n - m * m).max(0.0).sqrt()
}

/// High-pass texture window: `clip(w - gaussian(w, 5), -26, 26)` over the full-res canvas window `Q x Q` at `(rr, cc)`.
fn hp_window(rgb: &Arr2<Rgb8>, rr: usize, cc: usize, q: usize) -> Arr2<Rgbf> {
    let mut w = Arr2::new(q, q, [0f32; 3]);
    for r in 0..q {
        let src = &rgb.row(rr + r)[cc..cc + q];
        for (d, s) in w.row_mut(r).iter_mut().zip(src) {
            *d = [s[0] as f32, s[1] as f32, s[2] as f32];
        }
    }
    let g = prims::gaussian_filter_rgb(&w, 5.0);
    for (d, b) in w.d.iter_mut().zip(&g.d) {
        for ch in 0..3 {
            d[ch] = (d[ch] - b[ch]).clamp(-26.0, 26.0);
        }
    }
    w
}

/// Learn America's land / sea / parchment statistics from the vanilla interior (half-resolution statistics
/// pass; the sea clone patch and texture patches are re-extracted at full resolution from the canvas).
fn learn(rgb: &Arr2<Rgb8>, al: &Arr2<u8>, hm: &[u8]) -> Result<(Learned, LearnInfo)> {
    let (r0, r1, c0, c1) = (R_WIN0, R_WIN1, C_WIN0, C_WIN1);
    let hh = (r1 - r0).div_ceil(2);
    let hw = (c1 - c0).div_ceil(2);
    let sub = |i: usize, j: usize| -> Rgb8 { rgb.get(r0 + 2 * i, c0 + 2 * j) };
    let lsum = |p: Rgb8| p[0] as u32 + p[1] as u32 + p[2] as u32;

    let mut land = Arr2::new(hh, hw, false);
    let mut hm2 = vec![0u8; hh * hw];
    for i in 0..hh {
        for j in 0..hw {
            let (r, c) = (r0 + 2 * i, c0 + 2 * j);
            let h = hm_at(hm, r, c);
            hm2[i * hw + j] = h;
            land.d[i * hw + j] = al.get(r, c) < 8 && h > 8;
        }
    }
    let d2_sea = prims::edt(&land);
    let d2_land = prims::edt(&land.map(|v| !v));

    // parchment strip
    let (pr0, pr1, pc0, pc1) = (px_y(350.0), px_y(150.0), px_x(-765.0), px_x(-762.0));
    let parch_src = rgb.crop(pr0, pr1, pc0, pc1);
    let mut ph = [[0u32; 256]; 3];
    for p in &parch_src.d {
        for ch in 0..3 {
            ph[ch][p[ch] as usize] += 1;
        }
    }
    let p0 = [
        hist_median(&ph[0]),
        hist_median(&ph[1]),
        hist_median(&ph[2]),
    ];

    // ---- land colour LUT: heightmap bin x latitude band
    let mut valid = Arr2::new(hh, hw, false);
    const NH: usize = 256;
    let mut hist = vec![0u32; NH * 3 * 3 * 256];
    let mut cnt = vec![0u64; NH * 3];
    let mut n_valid = 0u64;
    let (mut lsum_acc, mut lsq_acc) = (0f64, 0f64);
    for i in 0..hh {
        let rows_y = CANVAS_YT - ((r0 + 2 * i) as f64 + 1.0) / S0;
        let band = prims::digitize(rows_y, &[150.0, 290.0]);
        for j in 0..hw {
            let k = i * hw + j;
            if !(land.d[k] && d2_sea.d[k] * 2.0 > 14.0) {
                continue;
            }
            let p = sub(i, j);
            if al.get(r0 + 2 * i, c0 + 2 * j) != 0 || lsum(p) <= 135 {
                continue;
            }
            valid.d[k] = true;
            n_valid += 1;
            let lum = lsum(p) as f32 / 3.0;
            lsum_acc += lum as f64;
            lsq_acc += lum as f64 * lum as f64;
            let g = hm2[k] as usize * 3 + band;
            cnt[g] += 1;
            for ch in 0..3 {
                hist[(g * 3 + ch) * 256 + p[ch] as usize] += 1;
            }
        }
    }
    let land_lum_std = if n_valid > 0 {
        std_of(lsum_acc, lsq_acc, n_valid as f64)
    } else {
        0.0
    };
    let nan3 = [f32::NAN; 3];
    let mut lut = vec![vec![nan3; NH]; 3];
    let mut lut_all = vec![nan3; NH];
    let mut cnt_all = vec![0u64; NH];
    let mut lut_cnt_total = 0u64;
    for h in 0..NH {
        let mut merged = [[0u32; 256]; 3];
        let mut n_all = 0u64;
        for b in 0..3 {
            let g = h * 3 + b;
            n_all += cnt[g];
            for ch in 0..3 {
                let hs = &hist[(g * 3 + ch) * 256..(g * 3 + ch + 1) * 256];
                for (m, &v) in merged[ch].iter_mut().zip(hs) {
                    *m += v;
                }
            }
            if cnt[g] >= 1200 {
                lut_cnt_total += cnt[g];
                for ch in 0..3 {
                    lut[b][h][ch] =
                        hist_median(&hist[(g * 3 + ch) * 256..(g * 3 + ch + 1) * 256]) as f32;
                }
            }
        }
        if n_all >= 1200 {
            cnt_all[h] = n_all;
            for ch in 0..3 {
                lut_all[h][ch] = hist_median(&merged[ch]) as f32;
            }
        }
    }
    let good_all: Vec<usize> = (0..NH).filter(|&h| !lut_all[h][0].is_nan()).collect();
    ensure!(
        good_all.len() > 60,
        "land LUT too thin ({} height bins)",
        good_all.len()
    );
    let xp: Vec<f64> = good_all.iter().map(|&h| h as f64).collect();
    let mut filled_all = vec![[0f32; 3]; NH];
    for ch in 0..3 {
        let fp: Vec<f64> = good_all.iter().map(|&h| lut_all[h][ch] as f64).collect();
        for h in 0..NH {
            filled_all[h][ch] = prims::interp(h as f64, &xp, &fp) as f32;
        }
    }
    let mut land_lut: [Vec<Rgbf>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for b in 0..3 {
        let v: Vec<Rgbf> = (0..NH)
            .map(|h| {
                if lut[b][h][0].is_nan() {
                    filled_all[h]
                } else {
                    lut[b][h]
                }
            })
            .collect();
        land_lut[b] = prims::gaussian_filter1d_lut(&v, 1.5, Mode::Nearest);
    }
    let h_max_with_samples = (0..NH).filter(|&h| cnt_all[h] > 0).max().unwrap_or(0);

    // ---- sand rim colour
    let mut sh = [[0u32; 256]; 3];
    let mut n_rim = 0u64;
    for i in 0..hh {
        for j in 0..hw {
            let k = i * hw + j;
            let d = d2_sea.d[k] * 2.0;
            if !(land.d[k] && d >= 1.0 && d <= 5.0) {
                continue;
            }
            let p = sub(i, j);
            if al.get(r0 + 2 * i, c0 + 2 * j) != 0 || lsum(p) <= 180 {
                continue;
            }
            n_rim += 1;
            for ch in 0..3 {
                sh[ch][p[ch] as usize] += 1;
            }
        }
    }
    ensure!(
        n_rim > 0,
        "no sand-rim samples (land next to sea) in the vanilla America interior"
    );
    let sand = [
        hist_median(&sh[0]) as f32,
        hist_median(&sh[1]) as f32,
        hist_median(&sh[2]) as f32,
    ];

    // ---- sea distance-to-land LUT
    let parch_mask = Arr2::from_vec(
        hh,
        hw,
        (0..hh * hw)
            .map(|k| al.get(r0 + 2 * (k / hw), c0 + 2 * (k % hw)) < 128 && !land.d[k])
            .collect(),
    );
    let d2_parch = prims::edt(&parch_mask.map(|v| !v));
    drop(parch_mask);
    let (bx0, bx1) = ((px_y(240.0) - r0) >> 1, (px_y(40.0) - r0) >> 1);
    let (by0, by1) = ((px_x(-680.0) - c0) >> 1, (px_x(-480.0) - c0) >> 1);
    let edges: [f32; 20] = [
        0.0, 2.0, 4.0, 6.0, 9.0, 13.0, 18.0, 24.0, 32.0, 44.0, 60.0, 80.0, 110.0, 150.0, 200.0,
        280.0, 400.0, 600.0, 900.0, 4000.0,
    ];
    let edges64: Vec<f64> = edges.iter().map(|&v| v as f64).collect();
    let nk = edges.len() - 1;
    let sea_lut_d: Vec<f32> = (0..nk).map(|k| 0.5 * (edges[k] + edges[k + 1])).collect();
    let mut shist = vec![0u32; nk * 3 * 256];
    let mut scnt = vec![0u64; nk];
    let (mut ssum, mut sn) = ([0f64; 3], 0u64);
    for i in bx0..bx1.min(hh) {
        for j in by0..by1.min(hw) {
            let k = i * hw + j;
            if land.d[k] || al.get(r0 + 2 * i, c0 + 2 * j) != 255 {
                continue;
            }
            let p = sub(i, j);
            for ch in 0..3 {
                ssum[ch] += p[ch] as f64;
            }
            sn += 1;
            let di = prims::digitize((d2_land.d[k] * 2.0) as f64, &edges64) as i64 - 1;
            if di >= 0 && (di as usize) < nk {
                let di = di as usize;
                scnt[di] += 1;
                for ch in 0..3 {
                    shist[(di * 3 + ch) * 256 + p[ch] as usize] += 1;
                }
            }
        }
    }
    ensure!(
        sn > 0,
        "no clean sea samples in the vanilla America Caribbean box"
    );
    let mut sea_lut: Vec<Rgbf> = vec![[0f32; 3]; nk];
    for k in 0..nk {
        sea_lut[k] = if scnt[k] > 250 {
            let h =
                |ch: usize| hist_median(&shist[(k * 3 + ch) * 256..(k * 3 + ch + 1) * 256]) as f32;
            [h(0), h(1), h(2)]
        } else if k > 0 {
            sea_lut[k - 1]
        } else {
            [
                (ssum[0] / sn as f64) as f32,
                (ssum[1] / sn as f64) as f32,
                (ssum[2] / sn as f64) as f32,
            ]
        };
    }

    // ---- open-sea clone patch: cleanest 1280x1280 full-res box near world (-580, 130)
    const P: usize = 1280;
    let (qh, qw) = (hh.div_ceil(8), hw.div_ceil(8));
    let mut good = Arr2::new(qh, qw, 0f32);
    for qi in 0..qh {
        for qj in 0..qw {
            let k = (qi * 8) * hw + qj * 8;
            let ok = al.get(r0 + 16 * qi, c0 + 16 * qj) == 255
                && !land.d[k]
                && d2_parch.d[k] * 2.0 > 110.0
                && d2_land.d[k] * 2.0 > 300.0;
            good.d[qi * qw + qj] = if ok { 1.0 } else { 0.0 };
        }
    }
    drop(d2_parch);
    let frac = prims::uniform_filter(&good, P / 16, P / 16, Mode::Constant);
    let (ccx, ccy) = (
        (-580.0 - CANVAS_X0) / 2.0 / 8.0,
        (CANVAS_YT - 130.0) / 2.0 / 8.0,
    );
    let mut best: Option<(f64, usize, usize)> = None;
    for ys in 0..qh {
        for xs in 0..qw {
            if frac.d[ys * qw + xs] > 0.999f32 {
                let d = (xs as f64 - ccx).powi(2) + (ys as f64 - ccy).powi(2);
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, ys, xs));
                }
            }
        }
    }
    let Some((_, ys, xs)) = best else {
        bail!("no clean open-sea box in the vanilla America interior")
    };
    let pr = (r0 as i64 + ys as i64 * 16 - (P / 2) as i64 + 8).clamp(r0 as i64, (r1 - P) as i64)
        as usize;
    let pc = (c0 as i64 + xs as i64 * 16 - (P / 2) as i64 + 8).clamp(c0 as i64, (c1 - P) as i64)
        as usize;
    let sea_patch = rgb.crop(pr, pr + P, pc, pc + P);
    let sea_patch_world_xyxy = [
        CANVAS_X0 + pc as f64 / S0,
        CANVAS_YT - pr as f64 / S0,
        CANVAS_X0 + (pc + P) as f64 / S0,
        CANVAS_YT - (pr + P) as f64 / S0,
    ];
    let mut fh = [[0u32; 256]; 3];
    for p in &sea_patch.d {
        for ch in 0..3 {
            fh[ch][p[ch] as usize] += 1;
        }
    }
    let sea_far = [
        hist_median(&fh[0]) as f32,
        hist_median(&fh[1]) as f32,
        hist_median(&fh[2]) as f32,
    ];
    drop(d2_land);

    // ---- land high-pass texture patches (two clean full-res 1024 windows)
    const Q: usize = 1024;
    let (th, tw) = (hh.div_ceil(16), hw.div_ceil(16));
    let mut gl = Arr2::new(th, tw, 0f32);
    for ti in 0..th {
        for tj in 0..tw {
            let k = (ti * 16) * hw + tj * 16;
            if valid.d[k] && d2_sea.d[k] * 2.0 > 40.0 {
                gl.d[ti * tw + tj] = 1.0;
            }
        }
    }
    drop(d2_sea);
    let fr = prims::uniform_filter(&gl, Q / 32, Q / 32, Mode::Constant);
    // descending by value; ties in descending index order (reverse of a stable ascending sort)
    let mut order: Vec<usize> = (0..fr.d.len()).collect();
    order.sort_by(|&a, &b| {
        fr.d[a]
            .partial_cmp(&fr.d[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.reverse();
    let n_gt = fr.d.iter().filter(|&&v| v > 0.30f32).count();
    let nsel = 2000usize.min(if n_gt == 0 { 1 } else { n_gt });
    let sel = &order[..nsel];
    let mut rng = Pcg64::new(5);
    let perm = rng.permutation(sel.len());
    let pos: Vec<(usize, usize)> = perm[..400usize.min(sel.len())]
        .iter()
        .map(|&k| {
            let (ys, xs) = (sel[k] / tw, sel[k] % tw);
            let rr = (r0 as i64 + ys as i64 * 32 - (Q / 2) as i64 + 16)
                .clamp(r0 as i64, (r1 - Q) as i64) as usize;
            let cc = (c0 as i64 + xs as i64 * 32 - (Q / 2) as i64 + 16)
                .clamp(c0 as i64, (c1 - Q) as i64) as usize;
            (rr, cc)
        })
        .collect();
    let valid_ref = &valid;
    let stds: Vec<f64> = {
        let nt = 6usize.min(pos.len().max(1));
        let per = pos.len().div_ceil(nt).max(1);
        let parts: Vec<Vec<f64>> = std::thread::scope(|s| {
            let hs: Vec<_> = pos
                .chunks(per)
                .map(|ch| {
                    s.spawn(move || {
                        ch.iter()
                            .map(|&(rr, cc)| {
                                let hp = hp_window(rgb, rr, cc, Q);
                                let (mut sm, mut sq, mut n) = (0f64, 0f64, 0f64);
                                for y in 0..Q / 2 {
                                    for x in 0..Q / 2 {
                                        if valid_ref.get(((rr - r0) >> 1) + y, ((cc - c0) >> 1) + x)
                                        {
                                            let v = hp.get(2 * y, 2 * x);
                                            for ch in 0..3 {
                                                sm += v[ch] as f64;
                                                sq += v[ch] as f64 * v[ch] as f64;
                                            }
                                            n += 3.0;
                                        }
                                    }
                                }
                                if n > 0.0 { std_of(sm, sq, n) } else { f64::NAN }
                            })
                            .collect::<Vec<f64>>()
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("texture candidate worker"))
                .collect()
        });
        parts.into_iter().flatten().collect()
    };
    ensure!(
        !stds.is_empty() && stds.iter().all(|v| v.is_finite()),
        "no valid land texture candidate window in the vanilla America interior"
    );
    let mut tmp = stds.clone();
    let med = prims::median(&mut tmp);
    let mut cands: Vec<(f64, usize, usize)> = stds
        .iter()
        .zip(&pos)
        .map(|(&s, &(rr, cc))| (s, rr, cc))
        .collect();
    cands.sort_by(|a, b| {
        (a.0 - med)
            .abs()
            .partial_cmp(&(b.0 - med).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let first = cands[0];
    let second = cands[1..]
        .iter()
        .find(|c| c.1.abs_diff(first.1) > Q || c.2.abs_diff(first.2) > Q)
        .copied();
    let zero_mean = |mut a: Arr2<Rgbf>| -> Arr2<Rgbf> {
        let mut m = [0f64; 3];
        for p in &a.d {
            for ch in 0..3 {
                m[ch] += p[ch] as f64;
            }
        }
        let n = a.d.len() as f64;
        let m = [(m[0] / n) as f32, (m[1] / n) as f32, (m[2] / n) as f32];
        for p in a.d.iter_mut() {
            for ch in 0..3 {
                p[ch] -= m[ch];
            }
        }
        a
    };
    let hp0 = hp_window(rgb, first.1, first.2, Q);
    let (hp1, std1) = match second {
        Some(c) => (hp_window(rgb, c.1, c.2, Q), c.0),
        None => {
            // single clean window: its transpose is the second patch
            let mut t = Arr2::new(Q, Q, [0f32; 3]);
            for r in 0..Q {
                for c in 0..Q {
                    t.set(c, r, hp0.get(r, c));
                }
            }
            (t, first.0)
        }
    };
    let hp = [zero_mean(hp0), zero_mean(hp1)];

    let info = LearnInfo {
        p0,
        land_lut_samples_total: lut_cnt_total,
        h_max_with_samples,
        hp_std: [first.0, std1],
        sea_patch_world_xyxy,
        land_lum_std,
    };
    Ok((
        Learned {
            parch_src,
            land_lut,
            sand,
            sea_lut_d,
            sea_lut,
            sea_patch,
            sea_far,
            hp,
        },
        info,
    ))
}

// ---------------------------------------------------------------------------------------------- frame relocation
fn build_frames(rgb: &mut Arr2<Rgb8>, al: &mut Arr2<u8>, rep: &mut serde_json::Map<String, Value>) {
    // 1) copy the west frame column SHIFT_W west (rows 1:1, latitude graticule continues)
    for r in 0..H_C {
        rgb.d
            .copy_within(r * W_C + C_FB0..r * W_C + C_OLD_W, r * W_C + C_FB1);
        al.d.copy_within(r * W_C + C_FB0..r * W_C + C_OLD_W, r * W_C + C_FB1);
    }
    rep.insert(
        "west_frame_copy".into(),
        json!({"src_cols": [C_FB0, C_OLD_W], "dst_cols": [C_FB1, C_NEW_W], "world_src_x": [-782.5, -760.0],
               "world_dst_x": [-910.47, -888.0], "shift_px": SHIFT_W}),
    );
    // 2) extend the north and south frame bands west to the relocated column with seam-optimised chunks
    let mut joins: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for (r0, r1, name) in [
        (0usize, R_BAND_N, "north_band_fill_chunks"),
        (R_BAND_S, H_C, "south_band_fill_chunks"),
    ] {
        let old_rgb = rgb.crop(r0, r1, 0, W_C);
        let old_al = al.crop(r0, r1, 0, W_C);
        let (j, chunks) = fill_cols(
            rgb,
            al,
            r0,
            &old_rgb,
            &old_al,
            0,
            r1 - r0,
            C_NEW_W,
            C_BAND_SEAM,
            C_BAND_SEAM,
            C_BAND_SRC_END,
            1536,
            true,
        );
        rep.insert(
            name.into(),
            Value::Array(
                chunks
                    .iter()
                    .map(|c| json!({"dst": c.dst, "src": c.src, "cost": c.cost}))
                    .collect(),
            ),
        );
        joins.push((r0, r1, j));
    }
    for (r0, r1, j) in joins {
        let mut blk = rgb.crop(r0, r1, 0, W_C);
        let mut seams = vec![C_NEW_W];
        seams.extend(j);
        for s in seams {
            blk = fade_seam(&blk, s, 1, 1, 96);
        }
        blk = fade_seam(&blk, C_BAND_SEAM, 1, -1, 96);
        rgb.paste(r0, 0, &blk);
    }
}

// ---------------------------------------------------------------------------------------------- void detection
const F_LAND_LIKE: u8 = 1;
const F_INK: u8 = 2;
const F_STRICT: u8 = 4;
const F_RAWP: u8 = 8;

/// land_like / ink / strict-parchment / raw-parchment flags over the interior window (filters in row chunks).
fn detect_void_filters(rgb: &Arr2<Rgb8>, al: &Arr2<u8>, hm: &[u8]) -> Arr2<u8> {
    let (h, w) = (R_WIN1 - R_WIN0, C_WIN1 - C_WIN0);
    let mut flags = vec![0u8; h * w];
    const CH: usize = 1024;
    let p0 = P0_AMER;
    for a in (0..h).step_by(CH) {
        let b = (a + CH).min(h);
        let (aa, bb) = (a.saturating_sub(8), (b + 8).min(h));
        let mut f = Arr2::new(bb - aa, w, [0f32; 3]);
        for r in aa..bb {
            let src = &rgb.row(R_WIN0 + r)[C_WIN0..C_WIN1];
            for (d, s) in f.row_mut(r - aa).iter_mut().zip(src) {
                *d = [s[0] as f32, s[1] as f32, s[2] as f32];
            }
        }
        let sm = prims::uniform_filter_rgb(&f, 5, 5, Mode::Reflect);
        let (f, sm) = (&f, &sm);
        par_rows(&mut flags[a * w..b * w], w, |i, row| {
            let rr = a + i;
            let fr = rr - aa;
            for (j, out) in row.iter_mut().enumerate() {
                let s = sm.get(fr, j);
                let mx = s[0].max(s[1]).max(s[2]);
                let mn = s[0].min(s[1]).min(s[2]);
                let sat = (mx - mn) / mx.max(1.0);
                let lum = ((s[0] + s[1]) + s[2]) / 3.0;
                let ll = sat > 0.335f32 || (lum < 110.0 && sat > 0.28f32);
                let px = f.get(fr, j);
                let mx0 = px[0].max(px[1]).max(px[2]);
                let mn0 = px[0].min(px[1]).min(px[2]);
                let sat0 = (mx0 - mn0) / mx0.max(1.0);
                let lum0 = ((px[0] + px[1]) + px[2]) / 3.0;
                let ik = lum0 < 125.0 && sat0 < 0.30f32;
                let dsm = (s[0] - p0[0]) * (s[0] - p0[0])
                    + (s[1] - p0[1]) * (s[1] - p0[1])
                    + (s[2] - p0[2]) * (s[2] - p0[2]);
                let dpm = dsm.sqrt();
                let hv = hm_at(hm, R_WIN0 + rr, C_WIN0 + j);
                let st =
                    al.get(R_WIN0 + rr, C_WIN0 + j) < 8 && hv <= 40 && dpm < 26.0 && sat < 0.29f32;
                let dpx = (px[0] - p0[0]) * (px[0] - p0[0])
                    + (px[1] - p0[1]) * (px[1] - p0[1])
                    + (px[2] - p0[2]) * (px[2] - p0[2]);
                let rp = dpx.sqrt() < 75.0 && sat0 < 0.33f32 && lum0 > 100.0;
                *out = (ll as u8) * F_LAND_LIKE
                    | (ik as u8) * F_INK
                    | (st as u8) * F_STRICT
                    | (rp as u8) * F_RAWP;
            }
        });
    }
    Arr2::from_vec(h, w, flags)
}

/// Components (4-connectivity) of `op` that touch an `edge`-px border band of the array or hold at least
/// `min_size` px, as a mask. Run-length union-find: memory proportional to the number of runs.
fn component_keep(op: &Arr2<bool>, edge: usize, min_size: usize) -> Arr2<bool> {
    let (h, w) = (op.h, op.w);
    let mut starts: Vec<u32> = Vec::new();
    let mut ends: Vec<u32> = Vec::new();
    let mut off = vec![0usize; h + 1];
    for r in 0..h {
        let row = op.row(r);
        let mut c = 0;
        while c < w {
            if row[c] {
                let s = c;
                while c < w && row[c] {
                    c += 1;
                }
                starts.push(s as u32);
                ends.push(c as u32);
            } else {
                c += 1;
            }
        }
        off[r + 1] = starts.len();
    }
    let n = starts.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let g = p[p[x as usize] as usize];
            p[x as usize] = g;
            x = g;
        }
        x
    }
    for r in 1..h {
        let (mut i, mut j) = (off[r - 1], off[r]);
        let (ie, je) = (off[r], off[r + 1]);
        while i < ie && j < je {
            if starts[i] < ends[j] && starts[j] < ends[i] {
                let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j as u32));
                if a != b {
                    parent[b as usize] = a;
                }
            }
            if ends[i] <= ends[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    let mut size = vec![0usize; n];
    let mut touch = vec![false; n];
    for r in 0..h {
        for k in off[r]..off[r + 1] {
            let root = find(&mut parent, k as u32) as usize;
            size[root] += (ends[k] - starts[k]) as usize;
            if r < edge
                || r + edge >= h
                || (starts[k] as usize) < edge
                || ends[k] as usize + edge > w
            {
                touch[root] = true;
            }
        }
    }
    let mut out = Arr2::new(h, w, false);
    for r in 0..h {
        for k in off[r]..off[r + 1] {
            let root = find(&mut parent, k as u32) as usize;
            if touch[root] || size[root] >= min_size {
                out.row_mut(r)[starts[k] as usize..ends[k] as usize].fill(true);
            }
        }
    }
    out
}

fn flag_mask(flags: &Arr2<u8>, bit: u8) -> Arr2<bool> {
    flags.map(|v| v & bit != 0)
}

/// Parchment void inside the old rect interior, decided by ART only (alpha 0 & not land-coloured, connected after a
/// 9x9 opening to a window edge or >= 2000 px; creep over ink / anti-aliasing / strict parchment).
fn detect_void(al: &Arr2<u8>, flags: &Arr2<u8>) -> Arr2<bool> {
    let (h, w) = (flags.h, flags.w);
    let aw = |r: usize, c: usize| al.get(R_WIN0 + r, C_WIN0 + c);
    // only land-coloured regions >= 9 px thick count as land here (graticule strokes read as land after smoothing)
    let mut dom = prims::opening_box(&flag_mask(flags, F_LAND_LIKE), 9);
    for r in 0..h {
        let row = dom.row_mut(r);
        for (c, v) in row.iter_mut().enumerate() {
            *v = aw(r, c) < 8 && !*v;
        }
    }
    let op = prims::opening_box(&dom, 9);
    let mut v = component_keep(&op, 6, 2000);
    drop(op);
    prims::grow(&mut v, &dom, 10);
    let closed = prims::closing_box(&v, 15);
    for (x, d) in v.d.iter_mut().zip(&closed.d) {
        *x = *d;
    }
    drop(closed);
    for (x, d) in v.d.iter_mut().zip(&dom.d) {
        *x &= *d;
    }
    let filled = prims::fill_holes(&v);
    for ((x, f), d) in v.d.iter_mut().zip(&filled.d).zip(&dom.d) {
        *x |= *f & *d;
    }
    drop(filled);
    let partial = Arr2::from_vec(
        h,
        w,
        (0..h * w)
            .map(|k| {
                let a = aw(k / w, k % w);
                a >= 8 && a < 255
            })
            .collect(),
    );
    // cand = (dom & ink) | partial
    let cand = Arr2::from_vec(
        h,
        w,
        (0..h * w)
            .map(|k| (dom.d[k] && flags.d[k] & F_INK != 0) || partial.d[k])
            .collect(),
    );
    prims::grow(&mut v, &cand, 8);
    drop(cand);
    prims::grow(&mut v, &dom, 1);
    prims::grow(&mut v, &partial, 4);
    drop(partial);
    drop(dom);
    prims::grow(&mut v, &flag_mask(flags, F_STRICT), 12);
    // torn-edge highlights along the void/land border: short creep over un-smoothed parchment-family px only
    let rawp = Arr2::from_vec(
        h,
        w,
        (0..h * w)
            .map(|k| flags.d[k] & F_RAWP != 0 && aw(k / w, k % w) < 8)
            .collect(),
    );
    prims::grow(&mut v, &rawp, 8);
    v
}

fn compute_masks(
    rgb: &Arr2<Rgb8>,
    al: &Arr2<u8>,
    hm: &[u8],
    rep: &mut serde_json::Map<String, Value>,
) -> Result<Masks> {
    let (h, w) = (R_WIN1 - R_WIN0, C_WIN1 - C_WIN0);
    let flags = detect_void_filters(rgb, al, hm);
    let v = detect_void(al, &flags);
    let aw = |r: usize, c: usize| al.get(R_WIN0 + r, C_WIN0 + c);

    // torn-parchment remnants east of the strip edge (x -757.5..-752)
    let wr = REM_PX + 16;
    let p0 = P0_AMER;
    let mut pc = Arr2::new(h, wr, false);
    for r in 0..h {
        for c in 0..wr {
            let px = rgb.get(R_WIN0 + r, C_WIN0 + c);
            let f = [px[0] as f32, px[1] as f32, px[2] as f32];
            let mx = f[0].max(f[1]).max(f[2]);
            let mn = f[0].min(f[1]).min(f[2]);
            let sat0 = (mx - mn) / mx.max(1.0);
            let d = (f[0] - p0[0]) * (f[0] - p0[0])
                + (f[1] - p0[1]) * (f[1] - p0[1])
                + (f[2] - p0[2]) * (f[2] - p0[2]);
            let a = aw(r, c);
            let fl = flags.d[r * w + c];
            let raw_parch = a < 8 && d.sqrt() < 52.0 && sat0 < 0.31f32;
            pc.d[r * wr + c] = c < REM_PX
                && (fl & F_STRICT != 0
                    || raw_parch
                    || (a < 8 && fl & F_LAND_LIKE == 0 && fl & F_INK != 0)
                    || (a >= 8 && a < 255));
        }
    }
    let mut rem = Arr2::new(h, wr, false);
    for r in 0..h {
        for c in 0..wr {
            rem.d[r * wr + c] = v.d[r * w + c];
        }
        rem.d[r * wr] = true;
    }
    prims::grow(&mut rem, &pc, REM_PX);
    drop(pc);
    // parchment slivers fully enclosed by owned px (holes <= 600 px, incl. against the strip edge) are owned too
    let mut cover = Arr2::new(h, wr + 1, true);
    for r in 0..h {
        for c in 0..wr {
            cover.d[r * (wr + 1) + c + 1] = rem.d[r * wr + c] || v.d[r * w + c];
        }
    }
    let (lab, nh) = prims::label(&cover.map(|x| !x));
    if nh > 0 {
        let mut hs = vec![0usize; nh + 1];
        for &l in &lab.d {
            hs[l as usize] += 1;
        }
        let mut filled = 0usize;
        for r in 0..h {
            for c in 0..wr {
                let l = lab.d[r * (wr + 1) + c + 1] as usize;
                if l > 0 && hs[l] <= 600 {
                    rem.d[r * wr + c] = true;
                    filled += 1;
                }
            }
        }
        rep.insert("remnant_enclosed_hole_px_filled".into(), json!(filled));
    }
    drop(lab);
    for r in 0..h {
        for c in 0..wr {
            if v.d[r * w + c] {
                rem.d[r * wr + c] = false;
            }
        }
    }
    let rm_px = rem.d.iter().filter(|&&x| x).count();

    let mut vc = Bits::new();
    vc.or_window(R_WIN0, C_WIN0, &v);
    let mut vland = Bits::new();
    for r in 0..h {
        for c in 0..w {
            let k = r * w + c;
            let in_rm = c < wr && rem.d[r * wr + c];
            if aw(r, c) < 8
                && !v.d[k]
                && (hm_at(hm, R_WIN0 + r, C_WIN0 + c) > 8 || flags.d[k] & F_LAND_LIKE != 0)
                && !in_rm
            {
                vland.set(R_WIN0 + r, C_WIN0 + c);
            }
        }
    }
    let mut mp = Bits::new();
    mp.or_window(R_WIN0, C_WIN0, &v);
    mp.or_window(R_WIN0, C_WIN0, &rem);
    let void_px = v.d.iter().filter(|&&x| x).count();
    drop(rem);
    drop(v);
    drop(flags);
    mp.set_rect(R_WIN0, R_WIN1, C_NEW_W, C_WIN0); // new strip (incl. erased old frame + margin)
    // vignette heal along the old playable edge
    let sub = mp.unpack(R_WIN0, R_WIN1, C_NEW_W, C_HEAL1);
    let dd = prims::edt(&sub.map(|x| !x));
    let mut heal = Arr2::new(sub.h, sub.w, false);
    let mut heal_px = 0usize;
    for r in 0..sub.h {
        for c in 0..sub.w {
            let k = r * sub.w + c;
            if dd.d[k] <= HEAL_PX && !sub.d[k] && al.get(R_WIN0 + r, C_NEW_W + c) >= 8 {
                heal.d[k] = true;
                heal_px += 1;
            }
        }
    }
    drop(dd);
    drop(sub);
    mp.or_window(R_WIN0, C_NEW_W, &heal);
    rep.insert("void_px_total_in_old_rect".into(), json!(void_px));
    rep.insert(
        "strip_px".into(),
        json!((R_WIN1 - R_WIN0) * (C_WIN0 - C_NEW_W)),
    );
    rep.insert("parchment_remnant_px_repainted".into(), json!(rm_px));
    rep.insert("heal_px_vanilla_sea_repainted".into(), json!(heal_px));
    Ok(Masks { mp, vc, vland })
}

struct LandJoin {
    original: Arr2<f32>,
    correction: Arr2<Rgbf>,
}

impl LandJoin {
    fn new(
        rgb: &Arr2<Rgb8>,
        al: &Arr2<u8>,
        masks: &Masks,
        terr: &DemTerrain,
        painter: &Painter,
    ) -> Result<Self> {
        let z = terr.z_coarse_grid();
        let mut generated = Arr2::new(z.h, z.w, false);
        let mut stock = Arr2::new(z.h, z.w, false);
        let mut references = Arr2::new(z.h, z.w, false);
        let mut residual = Arr2::new(z.h, z.w, [0.0; 3]);
        for r in 0..z.h {
            for c in 0..z.w {
                let k = r * z.w + c;
                if z.d[k] <= 0.0 {
                    continue;
                }
                let (mut painted, mut land, mut count, mut sum) =
                    (0usize, 0usize, 0usize, [0f32; 3]);
                for rr in r * 8..r * 8 + 8 {
                    for cc in c * 8..c * 8 + 8 {
                        if masks.mp.get(rr, cc) {
                            painted += 1;
                        } else if masks.vland.get(rr, cc) && al.get(rr, cc) < 8 {
                            land += 1;
                            let colour = rgb.get(rr, cc);
                            // Same usable-ground criterion as palette learning: black rock/underlay patches
                            // must not become a colour correction smeared out over the new plains.
                            if colour.iter().map(|&v| v as u16).sum::<u16>() <= 135 {
                                continue;
                            }
                            count += 1;
                            for ch in 0..3 {
                                sum[ch] += colour[ch] as f32;
                            }
                        }
                    }
                }
                generated.d[k] = painted >= 32;
                stock.d[k] = !generated.d[k] && land >= 32;
                references.d[k] = stock.d[k] && count >= 32;
                if references.d[k] {
                    let predicted =
                        painter.land_colour(z_to_h(z.d[k]), CANVAS_YT - (r * 8 + 4) as f64 / S0);
                    for ch in 0..3 {
                        residual.d[k][ch] = sum[ch] / count as f32 - predicted[ch];
                    }
                }
            }
        }
        let transition = seam::Transition::new(&generated, &stock, &references, HM_PX as f32)?;
        let mut correction = transition.colour_residual(|k| residual.d[k]);
        for (rgb, &weight) in correction.d.iter_mut().zip(&transition.correction.d) {
            for channel in rgb {
                *channel *= weight;
            }
        }
        Ok(Self {
            original: transition.original,
            correction,
        })
    }

    fn sample(&self, r: usize, c: usize) -> seam::Sample {
        seam::Sample::texture(r, c, self.original.h, self.original.w)
    }

    fn affects(&self, r: usize, c: usize) -> bool {
        self.sample(r, c).scalar(&self.original.d) < 1.0
    }

    fn any_in(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> bool {
        (r0.saturating_sub(8) / 8..(r1.div_ceil(8) + 1).min(self.original.h)).any(|r| {
            (c0.saturating_sub(8) / 8..(c1.div_ceil(8) + 1).min(self.original.w))
                .any(|c| self.original.get(r, c) < 1.0)
        })
    }
}

// ---------------------------------------------------------------------------------------------- painting
fn paint_canvas(
    rgb: &mut Arr2<Rgb8>,
    al: &mut Arr2<u8>,
    masks: &Masks,
    learned: &Learned,
    terr: &DemTerrain,
    rep: &mut serde_json::Map<String, Value>,
    progress: &dyn Fn(&str),
) -> Result<LandJoin> {
    let canvas = Canvas {
        h: H_C,
        w: W_C,
        x0: CANVAS_X0,
        yt: CANVAS_YT,
    };
    let painter = Painter::new(Variant::America, canvas, learned);
    progress("America: matching land across the former parchment boundary");
    let join = LandJoin::new(rgb, al, masks, terr, &painter)?;
    let edges = [
        Edge {
            side: b'W',
            pos: C_NEW_W as f64,
            base_m: EDGE_BASE,
            flag: None,
        },
        Edge {
            side: b'N',
            pos: R_WIN0 as f64,
            base_m: EDGE_BASE,
            flag: None,
        },
        Edge {
            side: b'S',
            pos: R_WIN1 as f64,
            base_m: EDGE_BASE,
            flag: None,
        },
        Edge {
            side: b'E',
            pos: C_WIN1 as f64,
            base_m: EDGE_BASE,
            flag: None,
        },
    ];
    let (mut void_land, mut void_sea, mut strip_land, mut strip_sea) = (0u64, 0u64, 0u64, 0u64);
    ensure!(
        masks.mp.bounds().is_some(),
        "America painter ownership mask is empty"
    );
    let (r_a, r_b) = (R_WIN0, R_WIN1);
    let mut col_chunks = vec![(C_NEW_W, STRIP_CHUNK_END)];
    let mut c = STRIP_CHUNK_END;
    while c < C_WIN1 {
        col_chunks.push((c, (c + 3072).min(C_WIN1)));
        c += 3072;
    }
    let mut a = r_a;
    while a < r_b {
        let b = (a + 1024).min(r_b);
        for &(c0, c1) in &col_chunks {
            if !masks.mp.any_in(a, b, c0, c1) && !join.any_in(a, b, c0, c1) {
                continue;
            }
            let mut res = painter.paint(terr, a, b, c0, c1, &edges);
            let (wr0, wr1, wc0, wc1) = (res.r0, res.r1, res.c0, res.c1);
            ensure!(
                wr0 == a.saturating_sub(200) && wr1 == (b + 200).min(H_C),
                "painter window rows differ from the America slab"
            );
            let m = masks.mp.unpack(wr0, wr1, wc0, wc1);
            let vl = Arr2::new(0, 0, false); // land matching is global; colour_match handles sea only
            let cur_rgb = rgb.crop(wr0, wr1, wc0, wc1);
            let cur_al = al.crop(wr0, wr1, wc0, wc1);
            let sg_sea: Vec<f32> = (wc0..wc1)
                .map(|col| {
                    SEA_SIGMA_NEAR
                        + (SEA_SIGMA_STRIP - SEA_SIGMA_NEAR)
                            * smoothstep((C_HEAL1 as f32 + 300.0 - col as f32) / 300.0)
                })
                .collect();
            let opts = MatchOpts {
                sea_sigma: SeaSigma::PerColumn(sg_sea),
                land_sigma: 0.0, // global contour field replaces slab-local dark-rim anchors
                inwin: Some([R_WIN0, R_WIN1, C_NEW_W, C_WIN1]),
            };
            colour_match(&mut res, &m, &vl, &cur_rgb, &cur_al, &opts);
            drop((vl, cur_rgb, cur_al));
            for r in a..b {
                let wr = r - wr0;
                for col in c0..c1 {
                    let wc = col - wc0;
                    let k = wr * res.rgb.w + wc;
                    let owned = m.d[k];
                    if !owned && (al.get(r, col) >= 8 || !res.land.d[k] || res.cov.d[k] <= 0.5) {
                        continue;
                    }
                    let mut p = res.rgb.d[k];
                    if res.land.d[k] && res.cov.d[k] > 0.5 {
                        let sample = join.sample(r, col);
                        let old_weight = if owned {
                            0.0
                        } else {
                            sample.scalar(&join.original.d)
                        };
                        if !owned && old_weight >= 1.0 {
                            continue;
                        }
                        let correction = sample.rgb(&join.correction.d);
                        let old = rgb.get(r, col);
                        for ch in 0..3 {
                            p[ch] = (p[ch] + correction[ch]) * (1.0 - old_weight)
                                + old[ch] as f32 * old_weight;
                        }
                    }
                    rgb.set(
                        r,
                        col,
                        [
                            rnd(p[0]).clamp(0.0, 255.0) as u8,
                            rnd(p[1]).clamp(0.0, 255.0) as u8,
                            rnd(p[2]).clamp(0.0, 255.0) as u8,
                        ],
                    );
                    if !owned {
                        continue;
                    } // colour only; original coastline alpha is immutable
                    al.set(r, col, res.alpha.d[k]);
                    let lnd = res.land.d[k] && res.cov.d[k] > 0.5;
                    if masks.vc.get(r, col) {
                        if lnd { void_land += 1 } else { void_sea += 1 }
                    } else if lnd {
                        strip_land += 1;
                    } else {
                        strip_sea += 1;
                    }
                }
            }
        }
        progress(&format!("America: painted rows up to {b} of {r_b}"));
        a = b;
    }
    rep.insert("painted_px".into(), json!({"void_land": void_land, "void_sea": void_sea, "strip_land": strip_land, "strip_sea": strip_sea}));
    rep.insert("land_join".into(), json!({"ignore_original_rim_units": seam::RIM_UNITS, "original_blend_units": seam::ORIGINAL_UNITS, "generated_correction_units": seam::GENERATED_UNITS}));
    Ok(join)
}

// ---------------------------------------------------------------------------------------------- heightmap
/// Per heightmap px (rows i0..i1 south-first, cols j0..j1): number of set canvas px of its 8x8 block in `bits`.
fn block_counts(bits: &Bits, i0: usize, i1: usize, j0: usize, j1: usize) -> Vec<u8> {
    let w = j1 - j0;
    let mut out = vec![0u8; (i1 - i0) * w];
    for i in i0..i1 {
        let r0 = (HM_ROW_TOP - 1 - i) * 8;
        for j in j0..j1 {
            let byte = j - HM_OFF / 8;
            let mut n = 0u32;
            for r in r0..r0 + 8 {
                n += bits.d[r * bits.stride + byte].count_ones();
            }
            out[(i - i0) * w + (j - j0)] = n as u8;
        }
    }
    out
}

fn coast_stats(agree: &[bool], landhm: &[bool], sel: &[bool]) -> Value {
    let (mut n, mut ok, mut sea_on_land, mut land_on_sea) = (0usize, 0usize, 0usize, 0usize);
    for k in 0..sel.len() {
        if !sel[k] {
            continue;
        }
        n += 1;
        if agree[k] {
            ok += 1;
        } else if landhm[k] {
            sea_on_land += 1;
        } else {
            land_on_sea += 1;
        }
    }
    json!({"px": n, "agreement": if n > 0 { jr(ok as f64 / n as f64, 5) } else { Value::Null },
           "art_sea_height_land_px": sea_on_land, "art_land_height_sea_px": land_on_sea})
}

/// Raw heights retain the written land/sea outline. A source-derived residual field joins DEM relief to sound
/// original relief across the whole painted contour, including a bounded strip of the old land. Adding a
/// residual preserves mountain slopes instead of pulling them onto a nearest-height shelf.
fn build_heightmap(
    hm: &[u8],
    terr: &DemTerrain,
    masks: &Masks,
    al_out: &Arr2<u8>,
    rep: &mut serde_json::Map<String, Value>,
) -> Result<Vec<u8>> {
    let Some((rmin, rmax, cmin, cmax)) = masks.mp.bounds() else {
        bail!("America painter ownership mask is empty")
    };
    let margin = (seam::ORIGINAL_UNITS as f64 * HM_PX).ceil() as usize + 1;
    let i0 = HM_ROW_TOP
        .saturating_sub((rmax + 8) / 8)
        .saturating_sub(margin)
        .max(HM_ROW_TOP - R_WIN1.div_ceil(8));
    let i1 = (HM_ROW_TOP - rmin.saturating_sub(8) / 8 + margin).min(HM_ROW_TOP - R_WIN0 / 8);
    let j0 = ((cmin + HM_OFF) / 8)
        .saturating_sub(margin)
        .max((C_NEW_W + HM_OFF) / 8);
    let j1 = ((cmax + HM_OFF) / 8 + 1 + margin).min((C_WIN1 + HM_OFF).div_ceil(8));
    let (hh, ww) = (i1 - i0, j1 - j0);
    let n = hh * ww;
    let van: Vec<u8> = (i0..i1)
        .flat_map(|i| hm[i * HM_W + j0..i * HM_W + j1].iter().copied())
        .collect();
    let fm = block_counts(&masks.mp, i0, i1, j0, j1);
    let fv = block_counts(&masks.vc, i0, i1, j0, j1);
    // art land fraction per block from the written alpha
    let fa: Vec<u8> = {
        let mut out = vec![0u8; n];
        for i in i0..i1 {
            let r0 = (HM_ROW_TOP - 1 - i) * 8;
            for j in j0..j1 {
                let c0 = j * 8 - HM_OFF;
                let mut k = 0u32;
                for r in r0..r0 + 8 {
                    k += al_out.row(r)[c0..c0 + 8].iter().filter(|&&a| a < 8).count() as u32;
                }
                out[(i - i0) * ww + (j - j0)] = k as u8;
            }
        }
        out
    };
    let art_land: Vec<bool> = fa.iter().map(|&k| k >= 32).collect();
    let keep_void: Vec<bool> = (0..n)
        .map(|k| fv[k] >= 32 && van[k] > VOID_HM_STRAY)
        .collect();
    let partial_fix: Vec<bool> = (0..n)
        .map(|k| {
            fm[k] > 0
                && fm[k] < 32
                && !keep_void[k]
                && art_land[k] != (van[k] > 0)
                && van[k] <= VOID_HM_STRAY
        })
        .collect();
    let own: Vec<bool> = (0..n)
        .map(|k| (fm[k] >= 32 || partial_fix[k]) && !keep_void[k])
        .collect();
    // DEM/coarse heights for the window (south-ascending rows): one coarse cell per heightmap px
    let zc = terr.z_coarse_grid();
    let hb: Vec<f32> = (0..n)
        .map(|k| {
            let (i, j) = (i0 + k / ww, j0 + k % ww);
            let z = zc.get(HM_ROW_TOP - 1 - i, j - HM_OFF / 8);
            if z > 0.0 {
                rnd(z_to_h(z)).clamp(1.0, 255.0)
            } else {
                1.0
            }
        })
        .collect();
    let zland: Vec<bool> = (0..n)
        .map(|k| zc.get(HM_ROW_TOP - 1 - (i0 + k / ww), (j0 + k % ww) - HM_OFF / 8) > 0.0)
        .collect();
    let generated = Arr2::from_vec(hh, ww, (0..n).map(|k| own[k] && art_land[k]).collect());
    let stock = Arr2::from_vec(hh, ww, (0..n).map(|k| !own[k] && van[k] > 0).collect());
    let transition = seam::Transition::new(&generated, &stock, &stock, HM_PX as f32)?;
    let residual = transition.residual(|k| van[k] as f32 - hb[k]);
    let mut out = hm.to_vec();
    for k in 0..n {
        let land = if own[k] { art_land[k] } else { van[k] > 0 };
        if !own[k] && (!land || transition.original.d[k] >= 1.0) {
            continue;
        }
        let ht = if land {
            transition
                .mix(k, hb[k], van[k] as f32, residual.d[k])
                .round_ties_even()
                .clamp(1.0, 255.0) as u8
        } else {
            0
        };
        out[(i0 + k / ww) * HM_W + j0 + k % ww] = ht;
    }
    // ---- statistics
    let (mut changed, mut to_zero, mut to_land) = (0usize, 0usize, 0usize);
    let (mut bb_r0, mut bb_r1, mut bb_c0, mut bb_c1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for k in 0..n {
        let (i, j) = (i0 + k / ww, j0 + k % ww);
        let (o, v) = (out[i * HM_W + j], van[k]);
        if o != v {
            changed += 1;
            if o == 0 {
                to_zero += 1;
            } else {
                to_land += 1;
            }
            bb_r0 = bb_r0.min(i);
            bb_r1 = bb_r1.max(i);
            bb_c0 = bb_c0.min(j);
            bb_c1 = bb_c1.max(j);
        }
    }
    let identical_outside = (0..n).all(|k| {
        own[k]
            || transition.original.d[k] < 1.0
            || out[(i0 + k / ww) * HM_W + j0 + k % ww] == van[k]
    });
    let cnt = |f: &dyn Fn(usize) -> bool| (0..n).filter(|&k| f(k)).count();
    let mut r = serde_json::Map::new();
    r.insert("window_hm_rows_cols".into(), json!([i0, i1, j0, j1]));
    r.insert("changed_px".into(), json!(changed));
    r.insert(
        "changed_bbox_rows_cols".into(),
        if changed > 0 {
            json!([bb_r0, bb_r1, bb_c0, bb_c1])
        } else {
            Value::Null
        },
    );
    r.insert("owned_px".into(), json!(cnt(&|k| own[k])));
    r.insert(
        "owned_px_void".into(),
        json!(cnt(&|k| own[k] && fv[k] >= 32)),
    );
    r.insert(
        "owned_px_with_vanilla_h_gt0".into(),
        json!(cnt(&|k| own[k] && van[k] > 0)),
    );
    r.insert(
        "owned_px_with_vanilla_h_gt13".into(),
        json!(cnt(&|k| own[k] && van[k] > VOID_HM_STRAY)),
    );
    r.insert(
        "original_height_void_px".into(),
        json!(cnt(&|k| keep_void[k])),
    );
    r.insert(
        "partially_owned_blocks_owned".into(),
        json!(cnt(&|k| partial_fix[k])),
    );
    r.insert(
        "outside_transition_identical".into(),
        json!(identical_outside),
    );
    r.insert(
        "original_land_blend_px".into(),
        json!(cnt(&|k| !own[k] && transition.original.d[k] < 1.0)),
    );
    r.insert("changed_px_to_zero".into(), json!(to_zero));
    r.insert("changed_px_to_land".into(), json!(to_land));
    let winval = |k: usize| out[(i0 + k / ww) * HM_W + j0 + k % ww];
    let mut vals: Vec<f64> = (0..n)
        .filter(|&k| own[k] && winval(k) > 0)
        .map(|k| winval(k) as f64)
        .collect();
    if !vals.is_empty() {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mx = (0..n).filter(|&k| own[k]).map(winval).max().unwrap_or(0);
        r.insert(
            "owned_land_h_percentiles_p10_p50_p90_max".into(),
            json!([
                prims::percentile(&vals, 10.0) as i64,
                prims::percentile(&vals, 50.0) as i64,
                prims::percentile(&vals, 90.0) as i64,
                mx
            ]),
        );
    }
    // coast proof: written art alpha vs written heights over the whole footprint
    let landhm: Vec<bool> = (0..n).map(|k| winval(k) > 0).collect();
    let agree: Vec<bool> = (0..n).map(|k| art_land[k] == landhm[k]).collect();
    let sel = |f: &dyn Fn(usize) -> bool| -> Vec<bool> { (0..n).map(f).collect() };
    let ring = |k: usize| own[k] && transition.correction.d[k] > 0.0;
    let partial = |k: usize| (fm[k] as u32 + fv[k] as u32) > 0 && !own[k] && !keep_void[k];
    let dem_agree = {
        let (mut a, mut t) = (0usize, 0usize);
        for k in 0..n {
            if own[k] {
                t += 1;
                if zland[k] == art_land[k] {
                    a += 1;
                }
            }
        }
        if t > 0 {
            jr(a as f64 / t as f64, 5)
        } else {
            Value::Null
        }
    };
    r.insert(
        "coast".into(),
        json!({
            "owned_all": coast_stats(&agree, &landhm, &own),
            "owned_transition_ring": coast_stats(&agree, &landhm, &sel(&|k| ring(k))),
            "owned_outside_transition": coast_stats(&agree, &landhm, &sel(&|k| own[k] && !ring(k))),
            "owned_void": coast_stats(&agree, &landhm, &sel(&|k| own[k] && fv[k] >= 32)),
            "owned_strip_heal_remnants": coast_stats(&agree, &landhm, &sel(&|k| own[k] && fv[k] < 32)),
            "original_height_void": coast_stats(&agree, &landhm, &keep_void),
            "partially_owned_kept": coast_stats(&agree, &landhm, &sel(&|k| partial(k))),
            "dem_vs_art_land_owned_agreement": dem_agree,
            "window_any_painted": coast_stats(&agree, &landhm, &sel(&|k| (fm[k] as u32 + fv[k] as u32) > 0)),
        }),
    );
    r.insert("same_size_as_baseline".into(), json!(out.len() == hm.len()));
    rep.insert("heightmap".into(), Value::Object(r));
    Ok(out)
}

// ---------------------------------------------------------------------------------------------- verification
fn psnr(a: &Arr2<Rgb8>, b: &Arr2<Rgb8>) -> f64 {
    let mut s = 0f64;
    for (p, q) in a.d.iter().zip(&b.d) {
        for ch in 0..3 {
            let d = p[ch] as f64 - q[ch] as f64;
            s += d * d;
        }
    }
    let mse = s / (a.d.len() * 3) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// Where did the written level-0 art differ from the baseline? Expected zones: painter ownership, the relocated
/// frame columns, the north / south band fills + seam fades, and the bounded original-land join (whole 4x4 blocks
/// because a changed DXT5 block is re-encoded whole). Also the encoded-vs-painted PSNR / alpha error of every
/// changed level-0 tile.
fn verify_written(
    src: &dyn TileSource,
    tiles: &BTreeMap<usize, Vec<u8>>,
    rgb: &Arr2<Rgb8>,
    al: &Arr2<u8>,
    masks: &Masks,
    join: &LandJoin,
    rep: &mut serde_json::Map<String, Value>,
) -> Result<()> {
    let expected = |r: usize, c: usize| -> bool {
        masks.mp.get(r, c)
            || join.affects(r, c)
            || c < C_NEW_W + 8
            || (r < R_BAND_N + 8 && c < C_BAND_SEAM + 100)
            || (r >= R_BAND_S - 8 && c < C_BAND_SEAM + 100)
    };
    struct TileOut {
        changed: u64,
        outside: u64,
        east_of_win1: u64,
        south_east: u64,
        north_east: u64,
        bbox: Option<[usize; 4]>,
        ubbox: Option<[usize; 4]>,
        psnr: f64,
        alpha_err: i32,
    }
    let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
    for ty in 0..CANVAS_NTY {
        for tx in 0..CANVAS_NTX {
            let idx = stpx::tile_index(0, CANVAS_TX0 + tx, CANVAS_TY0 + ty);
            if tiles.contains_key(&idx) {
                jobs.push((idx, tx, ty));
            }
        }
    }
    let per = jobs.len().div_ceil(threads()).max(1);
    let results: Vec<Result<Vec<TileOut>>> = std::thread::scope(|s| {
        let hs: Vec<_> = jobs
            .chunks(per)
            .map(|ch| {
                let expected = &expected;
                s.spawn(move || -> Result<Vec<TileOut>> {
                    let mut v = Vec::new();
                    for &(idx, tx, ty) in ch {
                        let (orgb, oal) = stpx::dxt5_decode_rgba(&src.raw_tile(idx)?);
                        let (nrgb, nal) = stpx::dxt5_decode_rgba(&tiles[&idx]);
                        let mut t = TileOut {
                            changed: 0,
                            outside: 0,
                            east_of_win1: 0,
                            south_east: 0,
                            north_east: 0,
                            bbox: None,
                            ubbox: None,
                            psnr: 0.0,
                            alpha_err: 0,
                        };
                        let painted = rgb.crop(ty * 512, ty * 512 + 512, tx * 512, tx * 512 + 512);
                        t.psnr = psnr(&painted, &nrgb);
                        for r in 0..512 {
                            for c in 0..512 {
                                let k = r * 512 + c;
                                t.alpha_err = t.alpha_err.max(
                                    (al.get(ty * 512 + r, tx * 512 + c) as i32 - nal.d[k] as i32)
                                        .abs(),
                                );
                                if orgb.d[k] == nrgb.d[k] && oal.d[k] == nal.d[k] {
                                    continue;
                                }
                                let (gr, gc) = (ty * 512 + r, tx * 512 + c);
                                t.changed += 1;
                                let bb = t.bbox.get_or_insert([gr, gr, gc, gc]);
                                *bb = [bb[0].min(gr), bb[1].max(gr), bb[2].min(gc), bb[3].max(gc)];
                                // expected zone, expanded to the 4x4 block
                                let (br, bc) = (gr / 4 * 4, gc / 4 * 4);
                                let exp =
                                    (br..br + 4).any(|rr| (bc..bc + 4).any(|cc| expected(rr, cc)));
                                if !exp {
                                    t.outside += 1;
                                    let ub = t.ubbox.get_or_insert([gr, gr, gc, gc]);
                                    *ub = [
                                        ub[0].min(gr),
                                        ub[1].max(gr),
                                        ub[2].min(gc),
                                        ub[3].max(gc),
                                    ];
                                }
                                if gc >= C_WIN1 {
                                    t.east_of_win1 += 1;
                                }
                                if gr >= R_WIN1 && gc >= C_BAND_SEAM + 100 {
                                    t.south_east += 1;
                                }
                                if gr < R_WIN0 && gc >= C_BAND_SEAM + 100 {
                                    t.north_east += 1;
                                }
                            }
                        }
                        v.push(t);
                    }
                    Ok(v)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("verification worker"))
            .collect()
    });
    let mut all: Vec<TileOut> = Vec::new();
    for r in results {
        all.extend(r?);
    }
    let merge = |f: &dyn Fn(&TileOut) -> Option<[usize; 4]>| -> Value {
        let mut m: Option<[usize; 4]> = None;
        for t in &all {
            if let Some(b) = f(t) {
                m = Some(match m {
                    None => b,
                    Some(a) => [
                        a[0].min(b[0]),
                        a[1].max(b[1]),
                        a[2].min(b[2]),
                        a[3].max(b[3]),
                    ],
                });
            }
        }
        m.map(|b| json!(b)).unwrap_or(Value::Null)
    };
    let total: u64 = all.iter().map(|t| t.changed).sum();
    let outside: u64 = all.iter().map(|t| t.outside).sum();
    rep.insert(
        "preservation".into(),
        json!({
            "changed_px_total": total, "changed_in_expected_zones": total - outside, "changed_outside_expected_zones": outside,
            "changed_bbox_rows_cols": merge(&|t| t.bbox), "unexpected_bbox_rows_cols": merge(&|t| t.ubbox),
            "changed_east_of_C_WIN1_px": all.iter().map(|t| t.east_of_win1).sum::<u64>(),
            "changed_south_of_R_WIN1_east_of_band_px": all.iter().map(|t| t.south_east).sum::<u64>(),
            "changed_north_of_R_WIN0_east_of_band_px": all.iter().map(|t| t.north_east).sum::<u64>(),
        }),
    );
    if !all.is_empty() {
        let mut ps: Vec<f64> = all.iter().map(|t| t.psnr).collect();
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if ps.len() % 2 == 1 {
            ps[ps.len() / 2]
        } else {
            (ps[ps.len() / 2 - 1] + ps[ps.len() / 2]) / 2.0
        };
        rep.insert(
            "level0_encoded_vs_painted_psnr_db_min_median".into(),
            json!([jr(ps[0], 2), jr(med, 2)]),
        );
        rep.insert(
            "level0_alpha_max_abs_err".into(),
            json!(all.iter().map(|t| t.alpha_err).max().unwrap_or(0)),
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------- driver
/// Generate the fixed America theatre: changed supertexture tiles (raw DXT5, all 7 levels), the complete new raw
/// heightmap (8192 x 4096, bottom-first) and the self-measured report.
pub fn generate(
    src: &dyn TileSource,
    baseline_heights: &[u8],
    dem: &Dem,
    progress: &dyn Fn(&str),
) -> Result<TheatreResult> {
    ensure!(
        baseline_heights.len() == HM_W * HM_H,
        "America: baseline heightmap must be {HM_W}x{HM_H} raw bytes, got {}",
        baseline_heights.len()
    );
    let t0 = std::time::Instant::now();
    let hm = baseline_heights;
    let mut rep = serde_json::Map::new();
    progress("America: loading baseline canvas");
    let (mut rgb, mut al) = stpx::load_canvas(src, CANVAS_WIN)?;
    ensure!(
        rgb.h == H_C && rgb.w == W_C,
        "America: unexpected canvas size {}x{}",
        rgb.h,
        rgb.w
    );
    rep.insert(
        "measured".into(),
        json!({
            "old_rect": [[-760.0, 0.0], [-340.0, 460.0]], "new_rect": [[-888.0, 0.0], [-340.0, 460.0]],
            "west_frame_band_world_x": [-781.6, -772.2], "north_frame_band_world_y": [469.9, 480.0],
            "south_frame_band_world_y": [-22.2, -12.5], "west_of_frame": "black void to world edge",
            "interior_parchment_rgb": P0_AMER, "latitude_graticule_step_deg": 2.32,
            "canvas_tiles": [CANVAS_TX0, CANVAS_TY0, CANVAS_NTX, CANVAS_NTY],
        }),
    );
    progress("America: detecting the unexplored-parchment void");
    let masks = compute_masks(&rgb, &al, hm, &mut rep)?;
    progress("America: learning palette and textures from the baseline interior");
    let (learned, info) = learn(&rgb, &al, hm)?;
    rep.insert(
        "learned".into(),
        json!({
            "sea_patch_world_xyxy": info.sea_patch_world_xyxy.iter().map(|&v| jr(v, 1)).collect::<Vec<_>>(),
            "sand_rgb": learned.sand.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            "parchment_rgb": info.p0.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            "hp_std": info.hp_std.iter().map(|&v| jr(v, 2)).collect::<Vec<_>>(),
            "land_lut_samples_total": info.land_lut_samples_total,
            "max_h_with_samples": info.h_max_with_samples,
            "land_lum_std": jr(info.land_lum_std, 3),
        }),
    );
    progress("America: relocating the frame");
    build_frames(&mut rgb, &mut al, &mut rep);
    let canvas = Canvas {
        h: H_C,
        w: W_C,
        x0: CANVAS_X0,
        yt: CANVAS_YT,
    };
    let terr = DemTerrain::new(dem, canvas);
    let join = paint_canvas(
        &mut rgb, &mut al, &masks, &learned, &terr, &mut rep, progress,
    )?;
    progress("America: encoding tiles");
    let (tiles, stats) = stpx::build_tiles(
        src,
        CANVAS_WIN,
        &rgb,
        &al,
        stpx::ChannelUpdates::ColourRepair,
    )?;
    rep.insert(
        "tiles_changed_per_level".into(),
        Value::Array(stats.iter().map(|s| json!({"level": s.level, "tiles": s.tiles, "changed_4x4_blocks": s.changed_blocks})).collect()),
    );
    rep.insert("tiles_changed_total".into(), json!(tiles.len()));
    verify_written(src, &tiles, &rgb, &al, &masks, &join, &mut rep)?;
    drop((rgb, join));
    progress("America: writing the heightmap");
    let heights = build_heightmap(hm, &terr, &masks, &al, &mut rep)?;
    rep.insert("seconds".into(), json!(jr(t0.elapsed().as_secs_f64(), 1)));
    Ok(TheatreResult {
        tiles,
        heights,
        report: Value::Object(rep),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_roundtrip_and_bounds() {
        let mut b = Bits::new();
        let mut m = Arr2::new(3, 20, false);
        m.set(0, 3, true);
        m.set(2, 19, true);
        b.or_window(10, 5, &m);
        assert!(b.get(10, 8) && b.get(12, 24) && !b.get(10, 9));
        assert_eq!(b.bounds(), Some((10, 12, 8, 24)));
        assert_eq!(b.unpack(10, 13, 5, 25), m);
    }

    #[test]
    fn component_keep_edges_and_size() {
        let mut m = Arr2::new(30, 30, false);
        for r in 10..14 {
            for c in 10..14 {
                m.set(r, c, true); // small interior blob: dropped
            }
        }
        for c in 0..4 {
            m.set(20, c, true); // touches the left edge: kept
        }
        let k = component_keep(&m, 6, 2000);
        assert!(!k.get(10, 10) && k.get(20, 0));
        let k = component_keep(&m, 6, 16);
        assert!(k.get(10, 10) && k.get(13, 13));
    }

    #[test]
    fn median_of_histogram() {
        let mut h = [0u32; 256];
        h[10] = 2;
        h[20] = 2;
        assert_eq!(hist_median(&h), 15.0);
        h[30] = 1;
        assert_eq!(hist_median(&h), 20.0);
    }
}
