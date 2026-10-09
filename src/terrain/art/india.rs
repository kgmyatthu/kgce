//! Native shipping India theatre (port of `fixed_india.py --heal` = `paint_india_east.py` with HEAL +
//! `heightmap_india_east.py`): enlarged India rectangle (440,-80)..(790,300) painted over the old rectangle
//! (440,20)..(680,280) of the user's own ETW supertexture / heightmap, from a public DEM.
//!
//! Pipeline: learn palette / land LUT / sea clone patch / high-pass texture from the baseline art ->
//! void + painter-owned masks (incl. the heal zones) -> frame relocation -> window painting
//! ([`super::painter::Painter`]) -> changed tiles (+ mips) -> heightmap with taper.
//! Nothing is read except `src`, `baseline_heights` and the DEM.

use std::collections::BTreeMap;

use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value, json};

use super::TheatreResult;
use super::dem::{Dem, z_to_h};
use super::grid::{Arr2, Rgb8, Rgbf};
use super::painter::{
    Canvas, DemTerrain, Edge, FillChunk, Learned, MatchOpts, Painter, S0, SeaSigma, Variant,
    colour_match, fade_seam, fill_cols,
};
use super::prims::{self, Mode};
use super::resample::{Pcg64, PilFilter, pil_resize_f32};
use super::stpx::{self, CanvasWindow, TileSource};

// ------------------------------------------------------------------------------------------------ settings.py

const WORLD_X0: f64 = -1280.0;
const WORLD_Y0: f64 = -640.0;
const HM_PX: f64 = 3.2;
const FRAME: f64 = 20.0;
const TILE_UNITS: f64 = 512.0 / S0;
/// Old / new theatre rectangles `((x0, y0), (x1, y1))`.
const OLD_RECT: ((f64, f64), (f64, f64)) = ((440.0, 20.0), (680.0, 280.0));
const NEW_RECT: ((f64, f64), (f64, f64)) = ((440.0, -80.0), (790.0, 300.0));
const CANVAS_TILES: (usize, usize, usize, usize) = (85, 15, 21, 23);

const HM_W: usize = 8192;
const HM_H: usize = 4096;

// paint_india_east.py constants
const IND_R0: usize = 8250;
const SPL_C: usize = 640;
const WEST_KEEP_ROW0: usize = 5400;
const VOID_HM_MAX: u8 = 8;
const HEAL_PX: f32 = 110.0;
const TAPER_UNITS: f64 = 8.0;

/// Canvas constants derived from the rectangles exactly like `settings.py`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Geo {
    h_c: usize,
    w_c: usize,
    canvas_x0: f64,
    canvas_yt: f64,
    old_n0: usize,
    old_n1: usize,
    new_n0: usize,
    new_n1: usize,
    c_old_e0: usize,
    c_old_e1: usize,
    c_new_e0: usize,
    c_new_e1: usize,
    c_w: usize,
    c_junc: usize,
    rs_old: usize,
    drow: usize,
    rs_new: usize,
    hm_col0: usize,
    hm_rows_top: usize,
}

impl Geo {
    fn derive() -> Result<Geo> {
        let (tx0, ty0, ntx, nty) = CANVAS_TILES;
        let canvas_x0 = WORLD_X0 + tx0 as f64 * TILE_UNITS;
        let canvas_yt = -WORLD_Y0 - ty0 as f64 * TILE_UNITS;
        let canvas_x1 = canvas_x0 + ntx as f64 * TILE_UNITS;
        let canvas_yb = canvas_yt - nty as f64 * TILE_UNITS;
        ensure!(
            canvas_x0 <= NEW_RECT.0.0 - FRAME
                && canvas_x1 >= NEW_RECT.1.0 + FRAME
                && canvas_yt >= NEW_RECT.1.1 + FRAME
                && canvas_yb <= NEW_RECT.0.1 - FRAME,
            "India canvas does not contain the new rectangle plus its frame"
        );
        let px_x = |x: f64| ((x - canvas_x0) * S0).round_ties_even() as i64;
        let px_y = |y: f64| ((canvas_yt - y) * S0).round_ties_even() as i64;
        let old_n0 = px_y(OLD_RECT.1.1 + FRAME);
        let old_n1 = old_n0 + 548;
        let shift_n = ((NEW_RECT.1.1 - OLD_RECT.1.1) * S0).round_ties_even() as i64;
        let c_old_e0 = px_x(OLD_RECT.1.0);
        let c_new_e0 = px_x(NEW_RECT.1.0);
        let c_w = px_x(OLD_RECT.0.0);
        let rs_old = px_y(OLD_RECT.0.1 - 6.25);
        let drow = ((OLD_RECT.0.1 - NEW_RECT.0.1) * S0).round_ties_even() as i64;
        let g = Geo {
            h_c: nty * 512,
            w_c: ntx * 512,
            canvas_x0,
            canvas_yt,
            old_n0: old_n0 as usize,
            old_n1: old_n1 as usize,
            new_n0: (old_n0 - shift_n) as usize,
            new_n1: (old_n1 - shift_n) as usize,
            c_old_e0: c_old_e0 as usize,
            c_old_e1: (c_old_e0 + (FRAME * S0) as i64) as usize,
            c_new_e0: c_new_e0 as usize,
            c_new_e1: (c_new_e0 + (FRAME * S0) as i64) as usize,
            c_w: c_w as usize,
            c_junc: (c_w + 8) as usize,
            rs_old: rs_old as usize,
            drow: drow as usize,
            rs_new: (rs_old + drow) as usize,
            hm_col0: ((canvas_x0 - WORLD_X0) * HM_PX).round_ties_even() as usize,
            hm_rows_top: ((canvas_yt - WORLD_Y0) * HM_PX).round_ties_even() as usize,
        };
        Ok(g)
    }
    fn canvas(&self) -> Canvas {
        Canvas {
            h: self.h_c,
            w: self.w_c,
            x0: self.canvas_x0,
            yt: self.canvas_yt,
        }
    }
    fn window(&self) -> CanvasWindow {
        let (tx0, ty0, ntx, nty) = CANVAS_TILES;
        CanvasWindow { tx0, ty0, ntx, nty }
    }
    /// vanilla old-rect interior (void rule applies): rows `r_v0..r_v1`, cols `c_v0..c_v1`
    fn r_v0(&self) -> usize {
        self.old_n1
    }
    fn r_v1(&self) -> usize {
        self.rs_old
    }
    fn c_v0(&self) -> usize {
        self.c_w
    }
    fn c_v1(&self) -> usize {
        self.c_old_e0
    }
}

// ------------------------------------------------------------------------------------------------ parallel helpers

fn nthreads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16)
}

/// `f(first_row, rows)` over disjoint contiguous row blocks of `data` (row length `w`) on all cores.
fn par_row_blocks<T: Send>(data: &mut [T], w: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if data.is_empty() || w == 0 {
        return;
    }
    let h = data.len() / w;
    let per = h.div_ceil(nthreads().min(h));
    std::thread::scope(|s| {
        for (k, blk) in data.chunks_mut(per * w).enumerate() {
            let f = &f;
            s.spawn(move || f(k * per, blk));
        }
    });
}

/// Split `0..n` into contiguous ranges, run `f(a, b)` on each in parallel, results in range order.
fn par_ranges<R: Send>(n: usize, f: impl Fn(usize, usize) -> R + Sync) -> Vec<R> {
    if n == 0 {
        return Vec::new();
    }
    let per = n.div_ceil(nthreads().min(n));
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .step_by(per)
            .map(|a| {
                let f = &f;
                s.spawn(move || f(a, (a + per).min(n)))
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("India worker thread panicked"))
            .collect()
    })
}

fn rint(v: f32) -> f32 {
    v.round_ties_even()
}

fn round_to(v: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    (v * m).round_ties_even() / m
}

/// `np.median` of values held in a 256-bin histogram of `n` samples.
fn hist_median(h: &[u32], n: u64) -> f64 {
    debug_assert!(n > 0);
    let at = |k: u64| -> f64 {
        let mut acc = 0u64;
        for (v, &c) in h.iter().enumerate() {
            acc += c as u64;
            if acc > k {
                return v as f64;
            }
        }
        255.0
    };
    if n % 2 == 1 {
        at(n / 2)
    } else {
        0.5 * (at(n / 2 - 1) + at(n / 2))
    }
}

fn px_f32(p: Rgb8) -> Rgbf {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

// ------------------------------------------------------------------------------------------------ learn()

struct LearnOut {
    learned: Learned,
    p0: [f64; 3],
    hp_std: [f64; 2],
    sea_patch_world: [f64; 4],
    land_samples: u64,
    h_max_with_samples: usize,
}

/// Std (population) of the clipped high-pass of the `q` x `q` window at canvas (`r`, `c`) over the `valid` pixels.
fn hp_window(rgb: &Arr2<Rgb8>, r: usize, c: usize, q: usize) -> Arr2<Rgbf> {
    let mut w = Arr2::new(q, q, [0f32; 3]);
    for i in 0..q {
        let src = &rgb.d[(r + i) * rgb.w + c..][..q];
        for (d, s) in w.row_mut(i).iter_mut().zip(src) {
            *d = px_f32(*s);
        }
    }
    let g = prims::gaussian_filter_rgb(&w, 5.0);
    for (d, s) in w.d.iter_mut().zip(&g.d) {
        for k in 0..3 {
            d[k] = (d[k] - s[k]).clamp(-26.0, 26.0);
        }
    }
    w
}

fn masked_std(hp: &Arr2<Rgbf>, valid: &Arr2<u8>, r: usize, c: usize) -> f64 {
    let q = hp.h;
    let (mut n, mut s) = (0u64, 0f64);
    for i in 0..q {
        let vr = &valid.d[(r + i) * valid.w + c..][..q];
        for (j, p) in hp.row(i).iter().enumerate() {
            if vr[j] & 1 != 0 {
                n += 1;
                s += p[0] as f64 + p[1] as f64 + p[2] as f64;
            }
        }
    }
    if n == 0 {
        return 0.0;
    }
    let mean = s / (3 * n) as f64;
    let mut v = 0f64;
    for i in 0..q {
        let vr = &valid.d[(r + i) * valid.w + c..][..q];
        for (j, p) in hp.row(i).iter().enumerate() {
            if vr[j] & 1 != 0 {
                for k in 0..3 {
                    let d = p[k] as f64 - mean;
                    v += d * d;
                }
            }
        }
    }
    (v / (3 * n) as f64).sqrt()
}

fn learn(rgb: &Arr2<Rgb8>, al: &Arr2<u8>, hm: &[u8], g: &Geo) -> Result<LearnOut> {
    const R0: usize = 1536;
    const R1: usize = 8192;
    const C0: usize = 512;
    const C1: usize = 6656;
    let (hh, ww) = (R1 - R0, C1 - C0);

    // vanilla heightmap of the old rect, north up, bilinearly upsampled 8x onto the canvas grid
    let mut sub = Arr2::new(832, 768, 0f32);
    for i in 0..832 {
        let row = 2943 - i;
        for j in 0..768 {
            sub.set(i, j, hm[row * HM_W + 5504 + j] as f32);
        }
    }
    let hm_up = pil_resize_f32(&sub, ww, hh, PilFilter::Bilinear, None);
    ensure!(
        hm_up.h == hh && hm_up.w == ww,
        "unexpected upsampled heightmap size"
    );
    let land = hm_up.map(|v| v > 0.5);
    let hbin: Arr2<u8> = hm_up.map(|v| rint(v).clamp(0.0, 255.0) as u8);
    drop(hm_up);

    // parchment (blank map) reference from the clean strip between the east ornament and the red band
    let (pr0, pr1, pc0, pc1) = (1792usize, 7680usize, 6736usize, 6778usize);
    let mut parch_src = Arr2::new(pr1 - pr0, pc1 - pc0, [0u8; 3]);
    for r in pr0..pr1 {
        for c in pc0..pc1 {
            ensure!(
                al.get(r, c) == 0,
                "parchment reference strip is not transparent at ({r},{c})"
            );
            parch_src.set(r - pr0, c - pc0, rgb.get(r, c));
        }
    }
    let mut p0 = [0f64; 3];
    for (ch, p) in p0.iter_mut().enumerate() {
        let mut v: Vec<f64> = parch_src.d.iter().map(|px| px[ch] as f64).collect();
        *p = prims::median(&mut v);
    }

    // flags: bit0 valid land sample, bit1 sand-rim sample, bit2 valid & > 40 px inland
    let d_sea = prims::edt(&land);
    let mut flags = Arr2::new(hh, ww, 0u8);
    par_row_blocks(&mut flags.d, ww, |ra, blk| {
        for (k, row) in blk.chunks_mut(ww).enumerate() {
            let r = ra + k;
            let cr = R0 + r;
            let px = &rgb.d[cr * rgb.w + C0..][..ww];
            let ap = &al.d[cr * al.w + C0..][..ww];
            let lr = land.row(r);
            let dr = d_sea.row(r);
            for c in 0..ww {
                if !lr[c] || ap[c] != 0 {
                    continue;
                }
                let p = px[c];
                let ylum = ((p[0] as f32 + p[1] as f32) + p[2] as f32) / 3.0;
                let d = dr[c];
                let mut f = 0u8;
                if d > 14.0 && ylum > 45.0 {
                    f |= 1;
                    if d > 40.0 {
                        f |= 4;
                    }
                }
                if d >= 1.0 && d <= 5.0 && ylum > 60.0 {
                    f |= 2;
                }
                row[c] = f;
            }
        }
    });
    drop(d_sea);

    // ---- land colour LUT: heightmap bin x latitude band, plus sand rim colour
    let mut hist = vec![0u32; 768 * 3 * 256];
    let mut kcount = vec![0u64; 768];
    let mut sand_hist = vec![0u32; 3 * 256];
    let mut sand_n = 0u64;
    for r in 0..hh {
        let y = g.canvas_yt - ((R0 + r) as f64 + 0.5) / S0;
        let band = if y < 110.0 {
            0
        } else if y < 200.0 {
            1
        } else {
            2
        };
        let fr = flags.row(r);
        let px = &rgb.d[(R0 + r) * rgb.w + C0..][..ww];
        let hr = hbin.row(r);
        for c in 0..ww {
            let f = fr[c];
            if f == 0 {
                continue;
            }
            let p = px[c];
            if f & 1 != 0 {
                let key = hr[c] as usize * 3 + band;
                kcount[key] += 1;
                for ch in 0..3 {
                    hist[(key * 3 + ch) * 256 + p[ch] as usize] += 1;
                }
            }
            if f & 2 != 0 {
                sand_n += 1;
                for ch in 0..3 {
                    sand_hist[ch * 256 + p[ch] as usize] += 1;
                }
            }
        }
    }
    drop(hbin);
    const NH: usize = 256;
    let mut lut = vec![vec![[f32::NAN; 3]; NH]; 3];
    let mut lut_all = vec![[f32::NAN; 3]; NH];
    let mut cnt_total = 0u64;
    let mut h_max = 0usize;
    for h in 0..NH {
        let mut all = vec![0u32; 3 * 256];
        let mut n_all = 0u64;
        for b in 0..3 {
            let k = h * 3 + b;
            let n = kcount[k];
            n_all += n;
            for ch in 0..3 {
                for v in 0..256 {
                    all[ch * 256 + v] += hist[(k * 3 + ch) * 256 + v];
                }
            }
            if n >= 2500 {
                for ch in 0..3 {
                    lut[b][h][ch] = hist_median(&hist[(k * 3 + ch) * 256..][..256], n) as f32;
                }
                cnt_total += n;
            }
        }
        if n_all >= 2500 {
            for ch in 0..3 {
                lut_all[h][ch] = hist_median(&all[ch * 256..][..256], n_all) as f32;
            }
            h_max = h;
        }
    }
    drop(hist);
    let hs: Vec<f64> = (0..NH).map(|h| h as f64).collect();
    let good_all: Vec<usize> = (0..NH).filter(|&h| !lut_all[h][0].is_nan()).collect();
    ensure!(
        !good_all.is_empty(),
        "no land colour samples in the baseline India art"
    );
    let xp: Vec<f64> = good_all.iter().map(|&h| h as f64).collect();
    let mut all_interp = vec![[0f32; 3]; NH];
    for ch in 0..3 {
        let fp: Vec<f64> = good_all.iter().map(|&h| lut_all[h][ch] as f64).collect();
        for h in 0..NH {
            all_interp[h][ch] = prims::interp(hs[h], &xp, &fp) as f32;
        }
    }
    let mut land_lut: Vec<Vec<Rgbf>> = Vec::with_capacity(3);
    for b in 0..3 {
        let band: Vec<Rgbf> = (0..NH)
            .map(|h| {
                if lut[b][h][0].is_nan() {
                    all_interp[h]
                } else {
                    lut[b][h]
                }
            })
            .collect();
        land_lut.push(prims::gaussian_filter1d_lut(&band, 1.5, Mode::Nearest));
    }
    ensure!(sand_n > 0, "no sand-rim samples in the baseline India art");
    let mut sand = [0f32; 3];
    for ch in 0..3 {
        sand[ch] = hist_median(&sand_hist[ch * 256..][..256], sand_n) as f32;
    }

    // ---- sea: distance-to-land LUT (alpha 255, away from the parchment vignette)
    let not_land = land.map(|v| !v);
    let d_land = prims::edt(&not_land);
    drop(not_land);
    let parch_inv = Arr2::from_vec(
        hh,
        ww,
        (0..hh * ww)
            .map(|i| !((al.d[(R0 + i / ww) * al.w + C0 + i % ww] < 128) && !land.d[i]))
            .collect(),
    );
    let d_parch = prims::edt(&parch_inv);
    drop(parch_inv);
    const EDGES: [f32; 20] = [
        0.0, 2.0, 4.0, 6.0, 9.0, 13.0, 18.0, 24.0, 32.0, 44.0, 60.0, 80.0, 110.0, 150.0, 200.0,
        280.0, 400.0, 600.0, 900.0, 4000.0,
    ];
    const NB: usize = 19;
    struct SeaAcc {
        n: [u64; NB],
        h: Vec<u32>,
        tot_n: u64,
        tot_sum: [f64; 3],
    }
    let accs = par_ranges(hh, |ra, rb| {
        let mut a = SeaAcc {
            n: [0; NB],
            h: vec![0u32; NB * 3 * 256],
            tot_n: 0,
            tot_sum: [0.0; 3],
        };
        for r in ra..rb {
            if r >= 6564 {
                continue;
            }
            let px = &rgb.d[(R0 + r) * rgb.w + C0..][..ww];
            let ap = &al.d[(R0 + r) * al.w + C0..][..ww];
            let lr = land.row(r);
            let dl = d_land.row(r);
            for c in 288..5938usize {
                if r < 4700 && c >= 5300 {
                    continue;
                }
                if ap[c] != 255 || lr[c] {
                    continue;
                }
                let p = px[c];
                a.tot_n += 1;
                for ch in 0..3 {
                    a.tot_sum[ch] += p[ch] as f64;
                }
                let di = EDGES.partition_point(|&e| e <= dl[c]);
                if di >= 1 && di - 1 < NB {
                    let k = di - 1;
                    a.n[k] += 1;
                    for ch in 0..3 {
                        a.h[(k * 3 + ch) * 256 + p[ch] as usize] += 1;
                    }
                }
            }
        }
        a
    });
    let mut sea_n = [0u64; NB];
    let mut sea_h = vec![0u32; NB * 3 * 256];
    let (mut tot_n, mut tot_sum) = (0u64, [0f64; 3]);
    for a in &accs {
        tot_n += a.tot_n;
        for ch in 0..3 {
            tot_sum[ch] += a.tot_sum[ch];
        }
        for k in 0..NB {
            sea_n[k] += a.n[k];
        }
        for (d, s) in sea_h.iter_mut().zip(&a.h) {
            *d += s;
        }
    }
    ensure!(tot_n > 0, "no open-sea samples in the baseline India art");
    let sea_lut_d: Vec<f32> = (0..NB).map(|k| 0.5 * (EDGES[k] + EDGES[k + 1])).collect();
    let mut sea_lut = vec![[0f32; 3]; NB];
    for k in 0..NB {
        if sea_n[k] > 500 {
            for ch in 0..3 {
                sea_lut[k][ch] = hist_median(&sea_h[(k * 3 + ch) * 256..][..256], sea_n[k]) as f32;
            }
        } else if k > 0 {
            sea_lut[k] = sea_lut[k - 1];
        } else {
            for ch in 0..3 {
                sea_lut[k][ch] = (tot_sum[ch] / tot_n as f64) as f32;
            }
        }
    }

    // ---- open-sea clone patch: biggest clean 1280 box near world (600,60)
    const P: usize = 1280;
    let (sh, sw) = (hh / 16, ww / 16);
    let mut good = Arr2::new(sh, sw, 0f32);
    for i in 0..sh {
        for j in 0..sw {
            let (r, c) = (i * 16, j * 16);
            let al_v = al.d[(R0 + r) * al.w + C0 + c];
            let ok_patch = al_v == 255 && !land.get(r, c) && d_parch.get(r, c) > 110.0;
            if ok_patch && d_land.get(r, c) > 300.0 && d_parch.get(r, c) > 160.0 {
                good.set(i, j, 1.0);
            }
        }
    }
    drop(d_parch);
    drop(d_land);
    drop(land);
    let frac = prims::uniform_filter(&good, P / 16, P / 16, Mode::Constant);
    let ccx = (600.0 - OLD_RECT.0.0) * S0 / 16.0;
    let ccy = (OLD_RECT.1.1 - 60.0) * S0 / 16.0;
    let mut best: Option<(f64, usize, usize)> = None;
    for i in 0..sh {
        for j in 0..sw {
            if frac.get(i, j) > 0.999f32 {
                let d = (j as f64 - ccx).powi(2) + (i as f64 - ccy).powi(2);
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, i, j));
                }
            }
        }
    }
    let Some((_, yi, xi)) = best else {
        bail!("no clean open-sea box in the baseline India art")
    };
    let pr = (yi * 16 + 8).saturating_sub(P / 2).min(hh - P);
    let pc = (xi * 16 + 8).saturating_sub(P / 2).min(ww - P);
    let sea_patch = rgb.crop(R0 + pr, R0 + pr + P, C0 + pc, C0 + pc + P);
    let sea_patch_world = [
        OLD_RECT.0.0 + pc as f64 / S0,
        OLD_RECT.1.1 - pr as f64 / S0,
        OLD_RECT.0.0 + (pc + P) as f64 / S0,
        OLD_RECT.1.1 - (pr + P) as f64 / S0,
    ];
    let mut sea_far = [0f32; 3];
    for ch in 0..3 {
        let mut h = vec![0u32; 256];
        for p in &sea_patch.d {
            h[p[ch] as usize] += 1;
        }
        sea_far[ch] = hist_median(&h, (P * P) as u64) as f32;
    }

    // ---- land high-pass texture patches (two clean 1024 windows)
    const Q: usize = 1024;
    let mut gl = Arr2::new(sh, sw, 0f32);
    for i in 0..sh {
        for j in 0..sw {
            if flags.get(i * 16, j * 16) & 4 != 0 {
                gl.set(i, j, 1.0);
            }
        }
    }
    let fr = prims::uniform_filter(&gl, Q / 16, Q / 16, Mode::Constant);
    let mut cand_pos: Vec<(usize, usize)> = Vec::new();
    for i in 0..sh {
        for j in 0..sw {
            if fr.get(i, j) > 0.88f32 {
                cand_pos.push((i, j));
            }
        }
    }
    ensure!(
        !cand_pos.is_empty(),
        "no clean land texture window in the baseline India art"
    );
    let order = Pcg64::new(5).permutation(cand_pos.len());
    let picks_k: Vec<usize> = order.into_iter().take(200).collect();
    let cpos = |k: usize| -> (usize, usize) {
        let (i, j) = cand_pos[k];
        (
            (i * 16 + 8).saturating_sub(Q / 2).min(hh - Q),
            (j * 16 + 8).saturating_sub(Q / 2).min(ww - Q),
        )
    };
    // flags viewed in canvas coordinates would need an offset: evaluate on the flags window directly
    let stds: Vec<f64> = {
        let nt = nthreads().min(6).min(picks_k.len());
        let results: Vec<Vec<(usize, f64)>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..nt)
                .map(|t| {
                    let (picks_k, flags, cpos) = (&picks_k, &flags, &cpos);
                    s.spawn(move || {
                        let mut out = Vec::new();
                        for idx in (t..picks_k.len()).step_by(nt) {
                            let (rr, cc) = cpos(picks_k[idx]);
                            let hp = hp_window(rgb, R0 + rr, C0 + cc, Q);
                            out.push((idx, masked_std(&hp, flags, rr, cc)));
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("India texture worker panicked"))
                .collect()
        });
        let mut v = vec![0f64; picks_k.len()];
        for (idx, s) in results.into_iter().flatten() {
            v[idx] = s;
        }
        v
    };
    let med = {
        let mut v = stds.clone();
        prims::median(&mut v)
    };
    let mut order2: Vec<usize> = (0..stds.len()).collect();
    order2.sort_by(|&a, &b| {
        (stds[a] - med)
            .abs()
            .partial_cmp(&(stds[b] - med).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (r_a, c_a) = cpos(picks_k[order2[0]]);
    let mut second: Option<usize> = None;
    for &k in &order2[1..] {
        let (r, c) = cpos(picks_k[k]);
        if r.abs_diff(r_a) > Q || c.abs_diff(c_a) > Q {
            second = Some(k);
            break;
        }
    }
    let centre = |mut hp: Arr2<Rgbf>| -> Arr2<Rgbf> {
        let mut m = [0f64; 3];
        for p in &hp.d {
            for ch in 0..3 {
                m[ch] += p[ch] as f64;
            }
        }
        let n = hp.d.len() as f64;
        for p in hp.d.iter_mut() {
            for ch in 0..3 {
                p[ch] -= (m[ch] / n) as f32;
            }
        }
        hp
    };
    let hp0 = hp_window(rgb, R0 + r_a, C0 + c_a, Q);
    let (hp1, std1) = match second {
        Some(k) => {
            let (r, c) = cpos(picks_k[k]);
            (hp_window(rgb, R0 + r, C0 + c, Q), stds[k])
        }
        None => {
            let mut t = Arr2::new(Q, Q, [0f32; 3]);
            for i in 0..Q {
                for j in 0..Q {
                    t.set(i, j, hp0.get(j, i));
                }
            }
            (t, stds[order2[0]])
        }
    };
    let hp_std = [stds[order2[0]], std1];
    let hp = [centre(hp0), centre(hp1)];

    Ok(LearnOut {
        learned: Learned {
            parch_src,
            land_lut: [
                land_lut[0].clone(),
                land_lut[1].clone(),
                land_lut[2].clone(),
            ],
            sand,
            sea_lut_d,
            sea_lut,
            sea_patch,
            sea_far,
            hp,
        },
        p0,
        hp_std,
        sea_patch_world,
        land_samples: cnt_total,
        h_max_with_samples: h_max,
    })
}

// ------------------------------------------------------------------------------------------------ void detection / masks

/// Vanilla heightmap value under every canvas pixel of the old-rect interior window (8x8 px per heightmap px).
fn hm_at_canvas(hm: &[u8], g: &Geo) -> Arr2<u8> {
    let (h, w) = (g.r_v1() - g.r_v0(), g.c_v1() - g.c_v0());
    let mut out = Arr2::new(h, w, 0u8);
    par_row_blocks(&mut out.d, w, |ra, blk| {
        for (k, row) in blk.chunks_mut(w).enumerate() {
            let r = g.r_v0() + ra + k;
            let hr = g.hm_rows_top - 1 - r / 8;
            for (c, v) in row.iter_mut().enumerate() {
                *v = hm[hr * HM_W + g.hm_col0 + (g.c_v0() + c) / 8];
            }
        }
    });
    out
}

/// Vanilla parchment ('unexplored' void) inside the old rect interior -> `(void, land_like)` (window arrays).
fn detect_void(
    rgb: &Arr2<Rgb8>,
    al: &Arr2<u8>,
    hm_v: &Arr2<u8>,
    g: &Geo,
) -> (Arr2<bool>, Arr2<bool>) {
    let (h, w) = (hm_v.h, hm_v.w);
    let (r_off, c_off) = (g.r_v0(), g.c_v0());
    // flags: bit0 land_like, bit1 ink, bit2 strict (feathered parchment-coloured pixels). uniform filters run in row
    // chunks with a 4 px halo (a 5x5 box needs 2); the halo is clipped at the window so scipy's reflect mode applies at
    // the true window edges only.
    let mut flags = Arr2::new(h, w, 0u8);
    const PM: [f32; 3] = [179.5, 168.5, 138.0];
    par_row_blocks(&mut flags.d, w, |ra, blk| {
        let nrows = blk.len() / w;
        let mut s = 0;
        while s < nrows {
            let e = (s + 128).min(nrows);
            let (gs, ge) = (ra + s, ra + e);
            let hs = gs.saturating_sub(4);
            let he = (ge + 4).min(h);
            let mut crop = Arr2::new(he - hs, w, [0f32; 3]);
            for r in hs..he {
                let src = &rgb.d[(r_off + r) * rgb.w + c_off..][..w];
                for (d, p) in crop.row_mut(r - hs).iter_mut().zip(src) {
                    *d = px_f32(*p);
                }
            }
            let sm = prims::uniform_filter_rgb(&crop, 5, 5, Mode::Reflect);
            for r in gs..ge {
                let ap = &al.d[(r_off + r) * al.w + c_off..][..w];
                let hv = hm_v.row(r);
                let out = &mut blk[(r - ra) * w..][..w];
                for c in 0..w {
                    let q = sm.get(r - hs, c);
                    let mx = q[0].max(q[1]).max(q[2]);
                    let mn = q[0].min(q[1]).min(q[2]);
                    let sat = (mx - mn) / mx.max(1.0);
                    let lum = ((q[0] + q[1]) + q[2]) / 3.0;
                    let mut f = 0u8;
                    if sat > 0.335 || (lum < 110.0 && sat > 0.28) {
                        f |= 1;
                    }
                    let p = px_f32(rgb.d[(r_off + r) * rgb.w + c_off + c]);
                    let mx0 = p[0].max(p[1]).max(p[2]);
                    let mn0 = p[0].min(p[1]).min(p[2]);
                    let sat0 = (mx0 - mn0) / mx0.max(1.0);
                    let lum0 = ((p[0] + p[1]) + p[2]) / 3.0;
                    if lum0 < 125.0 && sat0 < 0.30 {
                        f |= 2;
                    }
                    if ap[c] < 8 && hv[c] <= 40 {
                        let d0 = q[0] - PM[0];
                        let d1 = q[1] - PM[1];
                        let d2 = q[2] - PM[2];
                        let dist = (((d0 * d0) + (d1 * d1)) + (d2 * d2)).sqrt();
                        if dist < 22.0 && sat < 0.29 {
                            f |= 4;
                        }
                    }
                    out[c] = f;
                }
            }
            s = e;
        }
    });
    let bits = |bit: u8| -> Arr2<bool> { flags.map(|f| f & bit != 0) };
    let land_like = bits(1);
    let strict = bits(4);
    let mut dom = Arr2::new(h, w, false);
    let mut frag = Arr2::new(h, w, false);
    let mut cand = Arr2::new(h, w, false);
    for r in 0..h {
        let ap = &al.d[(r_off + r) * al.w + c_off..][..w];
        let hv = hm_v.row(r);
        for c in 0..w {
            let d = ap[c] < 8 && hv[c] <= VOID_HM_MAX;
            let fr = ap[c] >= 8 && ap[c] < 255 && hv[c] <= VOID_HM_MAX;
            let f = flags.get(r, c);
            let ink = f & 2 != 0;
            dom.set(r, c, d);
            frag.set(r, c, fr);
            cand.set(r, c, (d && (ink || f & 1 == 0)) || fr);
        }
    }
    drop(flags);
    let cmask = Arr2::from_vec(
        h,
        w,
        dom.d
            .iter()
            .zip(&land_like.d)
            .map(|(&d, &l)| d && !l)
            .collect(),
    );
    let op = prims::opening_box(&cmask, 9);
    let (lab, n) = prims::label(&op);
    drop(op);
    let mut seed = vec![false; n + 1];
    for r in (0..6.min(h)).chain(h.saturating_sub(6)..h) {
        for &l in lab.row(r) {
            seed[l as usize] = true;
        }
    }
    for r in 0..h {
        for &l in &lab.row(r)[w - 6..] {
            seed[l as usize] = true;
        }
    }
    seed[0] = false;
    let mut v = lab.map(|l| seed[l as usize]);
    drop(lab);
    prims::grow(&mut v, &cmask, 10); // restore the ragged edge cut by the opening
    drop(cmask);
    let mut v = prims::closing_box(&v, 15);
    for (x, &d) in v.d.iter_mut().zip(&dom.d) {
        *x &= d;
    }
    let holes = prims::fill_holes(&v);
    for ((x, &hl), &d) in v.d.iter_mut().zip(&holes.d).zip(&dom.d) {
        *x |= hl && d;
    }
    drop(holes);
    prims::grow(&mut v, &cand, 8); // ink coast outlines / anti-aliased fringe
    drop(cand);
    prims::grow(&mut v, &dom, 1);
    prims::grow(&mut v, &frag, 4);
    drop((frag, dom));
    prims::grow(&mut v, &strict, 12); // feathered land edge fading into parchment
    drop(strict);
    for r in 0..h {
        if r_off + r >= WEST_KEEP_ROW0 {
            let n = SPL_C.saturating_sub(c_off).min(w);
            v.row_mut(r)[..n].fill(false);
        }
    }
    (v, land_like)
}

/// Painter-owned canvas mask, void mask / vanilla-land mask of the old-rect window.
struct Masks {
    /// every pixel the painter owns (full canvas)
    mp: Arr2<bool>,
    /// void pixels inside the old rect (window `r_v0..r_v1 x c_v0..c_v1`)
    v: Arr2<bool>,
    /// vanilla land pixels bordering the painted area (colour anchors; same window)
    vland: Arr2<bool>,
}

/// Rectangle `r0..r1 x c0..c1` of a window-anchored mask, false outside the window.
fn crop_win(win: &Arr2<bool>, g: &Geo, r0: usize, r1: usize, c0: usize, c1: usize) -> Arr2<bool> {
    let mut out = Arr2::new(r1 - r0, c1 - c0, false);
    let (wr0, wc0) = (g.r_v0(), g.c_v0());
    let (a0, a1) = (r0.max(wr0), r1.min(wr0 + win.h));
    let (b0, b1) = (c0.max(wc0), c1.min(wc0 + win.w));
    if a0 < a1 && b0 < b1 {
        for r in a0..a1 {
            out.row_mut(r - r0)[b0 - c0..b1 - c0]
                .copy_from_slice(&win.row(r - wr0)[b0 - wc0..b1 - wc0]);
        }
    }
    out
}

fn fill_mask(m: &mut Arr2<bool>, r0: usize, r1: usize, c0: usize, c1: usize) {
    for r in r0..r1 {
        m.row_mut(r)[c0..c1].fill(true);
    }
}

fn compute_masks(
    old_al: &Arr2<u8>,
    old_rgb: &Arr2<Rgb8>,
    hm: &[u8],
    g: &Geo,
    rep: &mut Map<String, Value>,
) -> Result<Masks> {
    let hm_v = hm_at_canvas(hm, g);
    let (v, land_like) = detect_void(old_rgb, old_al, &hm_v, g);
    let (h, w) = (v.h, v.w);
    let mut vland = land_like;
    for r in 0..h {
        let ap = &old_al.d[(g.r_v0() + r) * old_al.w + g.c_v0()..][..w];
        let hv = hm_v.row(r);
        let vr = v.row(r);
        let lr = vland.row_mut(r);
        for c in 0..w {
            lr[c] = ap[c] < 8 && !vr[c] && (hv[c] > VOID_HM_MAX || lr[c]);
        }
    }
    let mut mp = Arr2::new(g.h_c, g.w_c, false);
    for r in 0..h {
        mp.row_mut(g.r_v0() + r)[g.c_v0()..g.c_v0() + w].copy_from_slice(v.row(r));
    }
    fill_mask(&mut mp, g.new_n1, g.old_n1, g.c_w, g.c_new_e0); // north band interior
    fill_mask(&mut mp, g.old_n1, g.rs_new, g.c_old_e0, g.c_new_e0); // east strip down to the new south frame
    fill_mask(&mut mp, g.rs_old, g.rs_new, SPL_C, g.c_old_e0); // south strip
    // HEAL: repaint the dark vignette ring of the old sea's east / south margin (open sea only)
    let mut hz: Vec<usize> = Vec::new();
    for (ra, rb, ca, cb) in [
        (5950usize, g.rs_old, 6000usize, g.c_v1()),
        (7800, g.rs_old, 700, g.c_v1()),
    ] {
        let sub = mp.crop(ra, rb, ca, cb);
        let inv = sub.map(|b| !b);
        let dd = prims::edt(&inv);
        for r in 0..sub.h {
            for c in 0..sub.w {
                if dd.get(r, c) <= HEAL_PX && !sub.get(r, c) && old_al.get(ra + r, ca + c) >= 8 {
                    hz.push((ra + r) * g.w_c + ca + c);
                }
            }
        }
    }
    hz.sort_unstable();
    hz.dedup();
    rep.insert("heal_vanilla_sea_px_repainted".into(), json!(hz.len()));
    for &i in &hz {
        mp.d[i] = true;
    }
    rep.insert(
        "void_px_total_in_old_rect".into(),
        json!(v.d.iter().filter(|&&b| b).count()),
    );
    let stray =
        v.d.iter()
            .zip(&hm_v.d)
            .filter(|&(&b, &h)| b && h > 0)
            .count();
    rep.insert("void_px_stray_hm_le8_nonzero".into(), json!(stray));
    Ok(Masks { mp, v, vland })
}

// ------------------------------------------------------------------------------------------------ frames

fn copy_block(
    new_rgb: &mut Arr2<Rgb8>,
    new_al: &mut Arr2<u8>,
    nr0: usize,
    nc0: usize,
    old_rgb: &Arr2<Rgb8>,
    old_al: &Arr2<u8>,
    or0: usize,
    oc0: usize,
    nrows: usize,
    ncols: usize,
) {
    for r in 0..nrows {
        let d = (nr0 + r) * new_rgb.w + nc0;
        let s = (or0 + r) * old_rgb.w + oc0;
        new_rgb.d[d..d + ncols].copy_from_slice(&old_rgb.d[s..s + ncols]);
        new_al.d[d..d + ncols].copy_from_slice(&old_al.d[s..s + ncols]);
    }
}

fn chunks_json(c: &[FillChunk]) -> Value {
    Value::Array(
        c.iter().map(|k| json!({"dst": k.dst, "src": k.src, "cost": [round_to(k.cost[0], 2), round_to(k.cost[1], 2)]})).collect(),
    )
}

/// f32 mean of three consecutive canvas rows over `c0..c1`, three channels interleaved.
fn mean3_rows(rgb: &Arr2<Rgb8>, r: usize, c0: usize, c1: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((c1 - c0) * 3);
    for c in c0..c1 {
        let (a, b, d) = (rgb.get(r, c), rgb.get(r + 1, c), rgb.get(r + 2, c));
        for ch in 0..3 {
            out.push(((a[ch] as f32 + b[ch] as f32) + d[ch] as f32) / 3.0);
        }
    }
    out
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum();
    (s / a.len() as f64) as f32
}

/// Sum over four rows `r..r+4` of columns `0..SPL_C`, /4 (exact: integer data), channels interleaved.
fn mean4_rows(rgb: &Arr2<Rgb8>, r: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(SPL_C * 3);
    for c in 0..SPL_C {
        for ch in 0..3 {
            let s: u32 = (0..4).map(|k| rgb.get(r + k, c)[ch] as u32).sum();
            out.push(s as f64 / 4.0);
        }
    }
    out
}

struct Joins {
    north: Vec<usize>,
    south: Vec<usize>,
    west_p0: usize,
}

fn build_frames(
    old_rgb: &Arr2<Rgb8>,
    old_al: &Arr2<u8>,
    new_rgb: &mut Arr2<Rgb8>,
    new_al: &mut Arr2<u8>,
    g: &Geo,
    rep: &mut Map<String, Value>,
) -> Result<Joins> {
    let h_c = g.h_c;
    // 1. north band: west part and corner copied by +20 units, x680..790 filled from the band west of x=680
    let nb = g.old_n1 - g.old_n0;
    copy_block(
        new_rgb,
        new_al,
        g.new_n0,
        g.c_junc,
        old_rgb,
        old_al,
        g.old_n0,
        g.c_junc,
        nb,
        g.c_old_e0 - g.c_junc,
    );
    copy_block(
        new_rgb,
        new_al,
        g.new_n0,
        g.c_new_e0,
        old_rgb,
        old_al,
        g.old_n0,
        g.c_old_e0,
        nb,
        g.c_new_e1 - g.c_new_e0,
    );
    let (north, nchunks) = fill_cols(
        new_rgb,
        new_al,
        g.new_n0,
        old_rgb,
        old_al,
        g.old_n0,
        nb,
        g.c_old_e0,
        g.c_new_e0,
        700,
        g.c_old_e0 - 100,
        1536,
        true,
    );
    rep.insert("north_frame_fill_chunks".into(), chunks_json(&nchunks));
    // 2. east frame column for rows OLD_N1..IND_R0; stretch under the new corner
    copy_block(
        new_rgb,
        new_al,
        g.old_n1,
        g.c_new_e0,
        old_rgb,
        old_al,
        g.old_n1,
        g.c_old_e0,
        IND_R0 - g.old_n1,
        g.c_new_e1 - g.c_new_e0,
    );
    let (ce0, ce1) = (g.c_old_e0, g.c_old_e1);
    let corner_last = mean3_rows(old_rgb, g.old_n1 - 3, ce0, ce1);
    let top_ref = mean3_rows(old_rgb, g.old_n1, ce0, ce1);
    let mut best = (f32::INFINITY, 0usize, 0f32, 0f32);
    for s in 520usize..=5200 {
        let cl = mean_abs_diff(&corner_last, &mean3_rows(old_rgb, g.new_n1 + s, ce0, ce1));
        let cr = mean_abs_diff(&top_ref, &mean3_rows(old_rgb, g.old_n1 + s - 3, ce0, ce1));
        let tot = cl + cr;
        if tot < best.0 {
            best = (tot, s, cl, cr);
        }
    }
    let sv = best.1;
    rep.insert("east_frame_fill_shift_px".into(), json!(sv));
    rep.insert(
        "east_frame_seam_cost".into(),
        json!([round_to(best.2 as f64, 2), round_to(best.3 as f64, 2)]),
    );
    copy_block(
        new_rgb,
        new_al,
        g.new_n1,
        g.c_new_e0,
        old_rgb,
        old_al,
        g.new_n1 + sv,
        g.c_old_e0,
        g.old_n1 - g.new_n1,
        g.c_new_e1 - g.c_new_e0,
    );
    // 3. south frame: vanilla rows RS_OLD.. move down DROW (x <= 680); x680..790 filled from the band west of x=680
    let h_old_end = h_c - g.drow;
    copy_block(
        new_rgb,
        new_al,
        g.rs_new,
        SPL_C,
        old_rgb,
        old_al,
        g.rs_old,
        SPL_C,
        h_c - g.rs_new,
        g.c_old_e0 - SPL_C,
    );
    let (south, schunks) = fill_cols(
        new_rgb,
        new_al,
        g.rs_new,
        old_rgb,
        old_al,
        g.rs_old,
        h_c - g.rs_new,
        g.c_old_e0,
        g.c_new_e0,
        700,
        g.c_old_e0 - 100,
        1536,
        false,
    );
    rep.insert("south_frame_fill_chunks".into(), chunks_json(&schunks));
    // 4. west frame column (cols 0..SPL_C): insert DROW rows at p0; search (p0, a) for the best seam
    let p_list: Vec<usize> = (5900..8200).step_by(2).collect();
    let pl: Vec<Vec<f64>> = p_list.iter().map(|&p| mean4_rows(old_rgb, p - 4)).collect();
    let pr: Vec<Vec<f64>> = p_list.iter().map(|&p| mean4_rows(old_rgb, p)).collect();
    let mad = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
    };
    let mut best_w: (f64, usize, usize, f64, f64) = (f64::INFINITY, 0, 0, 0.0, 0.0);
    for a in (5600usize..5736).step_by(2) {
        let ma = mean4_rows(old_rgb, a - 4);
        let mb = mean4_rows(old_rgb, a + g.drow);
        let mut jb = (f64::INFINITY, 0usize, 0f64, 0f64);
        for (j, (l, r)) in pl.iter().zip(&pr).enumerate() {
            let (cl, cr) = (mad(l, &ma), mad(r, &mb));
            if cl + cr < jb.0 {
                jb = (cl + cr, j, cl, cr);
            }
        }
        if jb.0 < best_w.0 {
            best_w = (jb.0, a, p_list[jb.1], jb.2, jb.3);
        }
    }
    let (_, a, p0, cl, cr) = best_w;
    ensure!(best_w.0.is_finite(), "west column splice search failed");
    rep.insert(
        "west_column_splice".into(),
        json!({"insert_at_row": p0, "src_rows": [a, a + g.drow], "cost": [round_to(cl, 2), round_to(cr, 2)]}),
    );
    copy_block(new_rgb, new_al, p0, 0, old_rgb, old_al, a, 0, g.drow, SPL_C);
    copy_block(
        new_rgb,
        new_al,
        p0 + g.drow,
        0,
        old_rgb,
        old_al,
        p0,
        0,
        h_old_end - p0,
        SPL_C,
    );
    Ok(Joins {
        north,
        south,
        west_p0: p0,
    })
}

fn build_frame_canvas(
    old_rgb: &Arr2<Rgb8>,
    old_al: &Arr2<u8>,
    g: &Geo,
    rep: &mut Map<String, Value>,
) -> Result<(Arr2<Rgb8>, Arr2<u8>)> {
    let mut new_rgb = old_rgb.clone();
    let mut new_al = old_al.clone();
    let joins = build_frames(old_rgb, old_al, &mut new_rgb, &mut new_al, g, rep)?;
    // soften the colour steps at the frame-fill seams
    for (r0, r1, seams, end_seam) in [
        (g.new_n0, g.new_n1, &joins.north, true),
        (g.rs_new, g.h_c, &joins.south, false),
    ] {
        let mut blk = new_rgb.crop(r0, r1, 0, g.w_c);
        blk = fade_seam(&blk, g.c_old_e0, 1, 1, 96);
        for &s in seams.iter() {
            blk = fade_seam(&blk, s, 1, 1, 96);
        }
        if end_seam {
            blk = fade_seam(&blk, g.c_new_e0, 1, -1, 96);
        }
        new_rgb.paste(r0, 0, &blk);
    }
    // east frame column: seams at the top corner and the stretch
    let top = g.old_n1 + 600;
    let mut col = new_rgb.crop(0, top, g.c_new_e0, g.c_new_e1);
    col = fade_seam(&col, g.new_n1, 0, 1, 64);
    col = fade_seam(&col, g.old_n1, 0, -1, 64);
    new_rgb.paste(0, g.c_new_e0, &col);
    // west column splice
    let p0 = joins.west_p0;
    let mut wc = new_rgb.crop(0, g.h_c, 0, SPL_C);
    wc = fade_seam(&wc, p0, 0, 1, 64);
    wc = fade_seam(&wc, p0 + g.drow, 0, -1, 64);
    new_rgb.paste(0, 0, &wc);
    // west fill: x435..440.3 below the new north frame = mirror of the Europe-frame margin parchment just above
    const WC0: usize = 384;
    let n = g.old_n1 - g.new_n1;
    let wlen = g.c_junc - WC0;
    let mut keep = Arr2::new(n, wlen, false);
    for r in 0..n {
        for c in 0..wlen {
            let p = old_rgb.get(g.new_n1 + r, WC0 + c);
            keep.set(
                r,
                c,
                (p[1] as i32 - p[2] as i32) > 50 && (p[0] as i32 - p[2] as i32) > 55,
            );
        }
    }
    let keep = prims::dilate(&prims::dilate(&keep));
    for r in 0..n {
        for c in 0..wlen {
            let (pr_, pa) = if keep.get(r, c) {
                (
                    old_rgb.get(g.new_n1 + r, WC0 + c),
                    old_al.get(g.new_n1 + r, WC0 + c),
                )
            } else {
                (old_rgb.get(g.new_n1 - 1 - r, WC0 + c), 0)
            };
            new_rgb.set(g.new_n1 + r, WC0 + c, pr_);
            new_al.set(g.new_n1 + r, WC0 + c, pa);
        }
    }
    Ok((new_rgb, new_al))
}

// ------------------------------------------------------------------------------------------------ painting

#[derive(Default)]
struct PaintStats {
    void_land: u64,
    void_sea: u64,
    strip_land: u64,
    strip_sea: u64,
}

fn paint_canvas(
    new_rgb: &mut Arr2<Rgb8>,
    new_al: &mut Arr2<u8>,
    masks: &Masks,
    painter: &Painter,
    terr: &DemTerrain,
    g: &Geo,
    progress: &dyn Fn(&str),
) -> PaintStats {
    let edges = [
        Edge {
            side: b'N',
            pos: g.new_n1 as f64,
            base_m: 2.0,
            flag: None,
        },
        Edge {
            side: b'W',
            pos: g.c_w as f64,
            base_m: 2.0,
            flag: Some((0..g.h_c).map(|r| r < WEST_KEEP_ROW0).collect()),
        },
        Edge {
            side: b'E',
            pos: g.c_new_e0 as f64,
            base_m: 3.2,
            flag: Some((0..g.h_c).map(|r| r < IND_R0).collect()),
        },
        Edge {
            side: b'S',
            pos: g.rs_new as f64,
            base_m: 2.2,
            flag: None,
        },
    ];
    let opts = MatchOpts {
        sea_sigma: SeaSigma::Const(45.0),
        land_sigma: 30.0,
        inwin: None,
    };
    let mp = &masks.mp;
    let r_a = (0..g.h_c)
        .find(|&r| mp.row(r).iter().any(|&b| b))
        .unwrap_or(0);
    let r_b = (0..g.h_c)
        .rfind(|&r| mp.row(r).iter().any(|&b| b))
        .map_or(0, |r| r + 1);
    let mut st = PaintStats::default();
    let mut a = r_a;
    while a < r_b {
        let b = (a + 1024).min(r_b);
        let mut c0 = g.c_w;
        while c0 < g.c_new_e0 {
            let c1 = (c0 + 2048).min(g.c_new_e0);
            if (a..b).any(|r| mp.row(r)[c0..c1].iter().any(|&m| m)) {
                composite_window(
                    painter, terr, new_rgb, new_al, masks, a, b, c0, c1, &edges, &opts, &mut st, g,
                );
            }
            c0 = c1;
        }
        progress(&format!("India: painted rows up to {b} of {r_b}"));
        a = b;
    }
    st
}

#[allow(clippy::too_many_arguments)]
fn composite_window(
    painter: &Painter,
    terr: &DemTerrain,
    new_rgb: &mut Arr2<Rgb8>,
    new_al: &mut Arr2<u8>,
    masks: &Masks,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    edges: &[Edge],
    opts: &MatchOpts,
    st: &mut PaintStats,
    g: &Geo,
) {
    let mut win = painter.paint(terr, r0, r1, c0, c1, edges);
    let (wr0, wr1, wc0, wc1) = (win.r0, win.r1, win.c0, win.c1);
    let m = masks.mp.crop(wr0, wr1, wc0, wc1);
    let vland = crop_win(&masks.vland, g, wr0, wr1, wc0, wc1);
    let cur_rgb = new_rgb.crop(wr0, wr1, wc0, wc1);
    let cur_al = new_al.crop(wr0, wr1, wc0, wc1);
    colour_match(&mut win, &m, &vland, &cur_rgb, &cur_al, opts);
    drop((cur_rgb, cur_al, vland));
    let vc = crop_win(&masks.v, g, r0, r1, c0, c1);
    for r in r0..r1 {
        for c in c0..c1 {
            let (i, j) = (r - wr0, c - wc0);
            if !m.get(i, j) {
                continue;
            }
            let p = win.rgb.get(i, j);
            new_rgb.set(
                r,
                c,
                [
                    rint(p[0]).clamp(0.0, 255.0) as u8,
                    rint(p[1]).clamp(0.0, 255.0) as u8,
                    rint(p[2]).clamp(0.0, 255.0) as u8,
                ],
            );
            new_al.set(r, c, win.alpha.get(i, j));
            let lnd = win.land.get(i, j) && win.cov.get(i, j) > 0.5;
            let void = vc.get(r - r0, c - c0);
            match (void, lnd) {
                (true, true) => st.void_land += 1,
                (true, false) => st.void_sea += 1,
                (false, true) => st.strip_land += 1,
                (false, false) => st.strip_sea += 1,
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ heightmap

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Port of `heightmap_india_east.main()`: DEM-derived heights where the painter owns the canvas, tapered toward the
/// nearest kept vanilla height; everything else is the baseline. Returns the complete new heightmap + its report.
fn build_heightmap(
    hm: &[u8],
    masks: &Masks,
    terr: &DemTerrain,
    g: &Geo,
) -> Result<(Vec<u8>, Value)> {
    let top = g.hm_rows_top;
    let c_off = 8 * g.hm_col0;
    let mp = &masks.mp;
    let (mut rmin, mut rmax, mut cmin, mut cmax) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for r in 0..g.h_c {
        let row = mp.row(r);
        if let (Some(a), Some(b)) = (row.iter().position(|&x| x), row.iter().rposition(|&x| x)) {
            rmin = rmin.min(r);
            rmax = rmax.max(r);
            cmin = cmin.min(a);
            cmax = cmax.max(b);
        }
    }
    ensure!(rmin != usize::MAX, "painter-owned mask is empty");
    let i0 = top - (rmax / 8 + 1);
    let i1 = top - rmin / 8;
    let j0 = (cmin + c_off) / 8;
    let j1 = (cmax + c_off) / 8 + 1;
    let (nh, nw) = (i1 - i0, j1 - j0);
    ensure!(j0 >= 64, "heightmap window too close to the west edge");

    // ownership: >= 50 % painter-owned heightmap px (outside the void, or void px with stray heights <= 8)
    let mut van = Arr2::new(nh, nw, 0u8);
    let mut own = Arr2::new(nh, nw, false);
    let mut fv_half = Arr2::new(nh, nw, false);
    for i in 0..nh {
        let hi = i0 + i;
        let r0 = 8 * (top - 1 - hi);
        for j in 0..nw {
            let c0 = 8 * (j0 + j) - c_off;
            let mut fm = 0u32;
            let mut fv = 0u32;
            for r in r0..r0 + 8 {
                let row = &mp.row(r)[c0..c0 + 8];
                fm += row.iter().filter(|&&b| b).count() as u32;
            }
            let (vr0, vc0) = (g.r_v0(), g.c_v0());
            for r in r0.max(vr0)..(r0 + 8).min(vr0 + masks.v.h) {
                let lo = c0.max(vc0);
                let hi_c = (c0 + 8).min(vc0 + masks.v.w);
                if lo < hi_c {
                    fv += masks.v.row(r - vr0)[lo - vc0..hi_c - vc0]
                        .iter()
                        .filter(|&&b| b)
                        .count() as u32;
                }
            }
            let vh = hm[hi * HM_W + j0 + j];
            van.set(i, j, vh);
            fv_half.set(i, j, fv >= 32);
            own.set(i, j, fm >= 32 && (fv < 32 || vh <= VOID_HM_MAX));
        }
    }
    // DEM heights for the window, chunked in 128 heightmap rows
    let mut hb = Arr2::new(nh, nw, 0u8);
    let mut a = 0;
    while a < nh {
        let b = (a + 128).min(nh);
        let (r0, r1) = (8 * (top - (i0 + b)), 8 * (top - (i0 + a)));
        let (c0, c1) = (8 * j0 - c_off, 8 * j1 - c_off);
        let z = terr.z(r0, r1, c0, c1);
        for i in a..b {
            let zr0 = 8 * (top - 1 - (i0 + i)) - r0;
            for j in 0..nw {
                let (mut cnt, mut sum) = (0u32, 0f64);
                for dr in 0..8 {
                    for dc in 0..8 {
                        let v = z.get(zr0 + dr, 8 * j + dc);
                        if v > 0.0 {
                            cnt += 1;
                            sum += v as f64;
                        }
                    }
                }
                let h = if cnt >= 32 {
                    z_to_h((sum / cnt.max(1) as f64) as f32)
                } else {
                    0.0
                };
                hb.set(i, j, rint(h).clamp(0.0, 255.0) as u8);
            }
        }
        a = b;
    }
    // taper against the nearest kept vanilla pixel (old-rect interior kept px of any height + Europe land to the west)
    let jm = j0 - 64;
    let kw = nw + 64;
    let mut keep = Arr2::new(nh, kw, false);
    let mut vv = Arr2::new(nh, kw, 0f32);
    let row_lo = top - 1 - (g.r_v1() - 1) / 8;
    let row_hi = top - 1 - g.r_v0() / 8;
    let col_lo = g.hm_col0 + g.c_w / 8;
    let col_hi = g.hm_col0 + g.c_old_e0 / 8;
    for i in 0..nh {
        let hi = i0 + i;
        for j in 0..64 {
            let v = hm[hi * HM_W + jm + j];
            keep.set(i, j, v > 0);
            vv.set(i, j, v as f32);
        }
        for j in 0..nw {
            let hj = j0 + j;
            let inside = hi >= row_lo && hi <= row_hi && hj >= col_lo && hj < col_hi;
            keep.set(i, 64 + j, !own.get(i, j) && inside);
            vv.set(i, 64 + j, van.get(i, j) as f32);
        }
    }
    let nokeep = keep.map(|k| !k);
    let (dist, idx) = prims::edt_with_indices(&nokeep);
    drop(nokeep);
    let mut win_new = Arr2::new(nh, nw, 0u8);
    for i in 0..nh {
        for j in 0..nw {
            if !own.get(i, j) {
                win_new.set(i, j, van.get(i, j));
                continue;
            }
            let h = hb.get(i, j);
            let v = if h > 0 {
                let d_units = dist.get(i, 64 + j) as f64 / HM_PX;
                let nb = vv.d[idx.get(i, 64 + j) as usize] as f64;
                let s = smooth(d_units / TAPER_UNITS);
                (s * h as f64 + (1.0 - s) * nb).round_ties_even().max(1.0)
            } else {
                0.0
            };
            win_new.set(i, j, v.clamp(0.0, 255.0) as u8);
        }
    }
    let mut out = hm.to_vec();
    for i in 0..nh {
        out[(i0 + i) * HM_W + j0..(i0 + i) * HM_W + j1].copy_from_slice(win_new.row(i));
    }

    // report (numbers heightmap_india_east.py prints)
    let (mut changed, mut to_zero, mut to_land) = (0usize, 0usize, 0usize);
    let (mut bb_r0, mut bb_r1, mut bb_c0, mut bb_c1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    let (mut outside_ok, mut gt8_changed) = (true, 0usize);
    for r in 0..HM_H {
        let inside_rows = r >= i0 && r < i1;
        for c in 0..HM_W {
            let k = r * HM_W + c;
            let ch = out[k] != hm[k];
            let in_own = inside_rows && c >= j0 && c < j1 && own.get(r - i0, c - j0);
            if ch {
                changed += 1;
                bb_r0 = bb_r0.min(r);
                bb_r1 = bb_r1.max(r);
                bb_c0 = bb_c0.min(c);
                bb_c1 = bb_c1.max(c);
                if out[k] == 0 {
                    to_zero += 1;
                } else {
                    to_land += 1;
                }
                if !in_own {
                    outside_ok = false;
                }
            }
            if hm[k] > VOID_HM_MAX && ch && !in_own {
                gt8_changed += 1;
            }
        }
    }
    let (mut owned, mut owned_void, mut owned_strip, mut void_stray, mut strip_over) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut v_land, mut v_sea, mut s_land, mut s_sea) = (0usize, 0usize, 0usize, 0usize);
    let mut land_vals: Vec<f64> = Vec::new();
    for i in 0..nh {
        for j in 0..nw {
            if !own.get(i, j) {
                continue;
            }
            owned += 1;
            let nv = win_new.get(i, j);
            let void = fv_half.get(i, j);
            if void {
                owned_void += 1;
                if van.get(i, j) > 0 {
                    void_stray += 1;
                }
                if nv > 0 { v_land += 1 } else { v_sea += 1 }
            } else {
                owned_strip += 1;
                if van.get(i, j) > 0 {
                    strip_over += 1;
                }
                if nv > 0 { s_land += 1 } else { s_sea += 1 }
            }
            if nv > 0 {
                land_vals.push(nv as f64);
            }
        }
    }
    land_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |p: f64| -> i64 { prims::percentile(&land_vals, p) as i64 };
    let mut big = 0i32;
    for (dy, dx) in [(0usize, 1usize), (1, 0)] {
        for i in 0..nh - dy {
            for j in 0..nw - dx {
                if own.get(i, j) != own.get(i + dy, j + dx) {
                    big = big
                        .max((win_new.get(i, j) as i32 - win_new.get(i + dy, j + dx) as i32).abs());
                }
            }
        }
    }
    let report = json!({
        "window_hm_rows_cols": [i0, i1, j0, j1],
        "changed_px": changed,
        "changed_bbox_rows_cols": [bb_r0, bb_r1, bb_c0, bb_c1],
        "owned_px": owned, "owned_void_px": owned_void, "owned_strip_px": owned_strip,
        "void_px_vanilla_nonzero_stray_overwritten": void_stray,
        "strip_px_vanilla_nonzero_overwritten": strip_over,
        "outside_owned_identical": outside_ok,
        "vanilla_nonzero_gt8_changed": gt8_changed,
        "changed_px_to_zero": to_zero, "changed_px_to_land": to_land,
        "void_px_land_after": v_land, "void_px_sea_after": v_sea,
        "strip_px_land_after": s_land, "strip_px_sea_after": s_sea,
        "owned_land_h_percentiles_p10_p50_p90_max": [pct(10.0), pct(50.0), pct(90.0), land_vals.last().copied().unwrap_or(0.0) as i64],
        "max_height_step_across_owned_boundary": big,
    });
    Ok((out, report))
}

// ------------------------------------------------------------------------------------------------ driver

/// Generate the shipping India theatre (`fixed_india.py --heal`): changed supertexture tiles (all mip levels,
/// RAW DXT5) and the complete new raw heightmap. All baseline inputs are read-only.
pub fn generate(
    src: &dyn TileSource,
    baseline_heights: &[u8],
    dem: &Dem,
    progress: &dyn Fn(&str),
) -> Result<TheatreResult> {
    ensure!(
        baseline_heights.len() == HM_W * HM_H,
        "baseline heightmap has {} bytes, expected {}",
        baseline_heights.len(),
        HM_W * HM_H
    );
    let g = Geo::derive()?;
    let mut rep = Map::new();
    progress("India: loading baseline canvas");
    let (old_rgb, old_al) = stpx::load_canvas(src, g.window())?;
    ensure!(
        old_rgb.h == g.h_c && old_rgb.w == g.w_c,
        "unexpected India canvas size"
    );

    progress("India: detecting void and painter-owned areas");
    let masks = compute_masks(&old_al, &old_rgb, baseline_heights, &g, &mut rep)?;

    progress("India: learning palette, coast and texture from the baseline art");
    let lo = learn(&old_rgb, &old_al, baseline_heights, &g)?;
    rep.insert(
        "learned".into(),
        json!({
            "sea_patch_world_xyxy": lo.sea_patch_world.iter().map(|&v| round_to(v, 1)).collect::<Vec<_>>(),
            "sand_rgb": lo.learned.sand.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            "parchment_rgb": lo.p0.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            "hp_std": lo.hp_std.iter().map(|&v| round_to(v, 2)).collect::<Vec<_>>(),
            "land_lut_samples_total": lo.land_samples,
            "max_h_with_samples": lo.h_max_with_samples,
        }),
    );

    progress("India: relocating frames");
    let (mut new_rgb, mut new_al) = build_frame_canvas(&old_rgb, &old_al, &g, &mut rep)?;
    drop((old_rgb, old_al));

    progress("India: preparing DEM terrain");
    let terr = DemTerrain::new(dem, g.canvas());
    let painter = Painter::new(Variant::India, g.canvas(), &lo.learned);
    let st = paint_canvas(
        &mut new_rgb,
        &mut new_al,
        &masks,
        &painter,
        &terr,
        &g,
        progress,
    );
    rep.insert(
        "painted_px".into(),
        json!({"void_land": st.void_land, "void_sea": st.void_sea, "strip_land": st.strip_land, "strip_sea": st.strip_sea}),
    );

    progress("India: encoding changed tiles");
    let (tiles, stats): (BTreeMap<usize, Vec<u8>>, _) = stpx::build_tiles(
        src,
        g.window(),
        &new_rgb,
        &new_al,
        stpx::ChannelUpdates::Repaint,
    )?;
    drop((new_rgb, new_al));
    rep.insert(
        "tiles_changed_per_level".into(),
        Value::Array(
            stats.iter().map(|s| json!({"level": s.level, "tiles": s.tiles, "changed_4x4_blocks": s.changed_blocks})).collect(),
        ),
    );
    rep.insert("tiles_changed_total".into(), json!(tiles.len()));

    progress("India: painting the heightmap");
    let (heights, hrep) = build_heightmap(baseline_heights, &masks, &terr, &g)?;
    rep.insert("heightmap".into(), hrep);
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
    fn derived_constants_match_settings_py() {
        let g = Geo::derive().unwrap();
        assert_eq!((g.h_c, g.w_c), (11776, 10752));
        assert_eq!(
            (g.old_n0, g.old_n1, g.new_n0, g.new_n1),
            (1024, 1572, 512, 1060)
        );
        assert_eq!(
            (g.c_old_e0, g.c_old_e1, g.c_new_e0, g.c_new_e1),
            (6656, 7168, 9472, 9984)
        );
        assert_eq!(
            (g.c_w, g.c_junc, g.rs_old, g.drow, g.rs_new),
            (512, 520, 8352, 2560, 10912)
        );
        assert_eq!((g.hm_col0, g.hm_rows_top), (5440, 3136));
    }
}
