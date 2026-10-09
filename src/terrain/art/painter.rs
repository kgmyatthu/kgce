//! Shared native painter of the fixed theatres (port of `Painter.paint` / colour matching of
//! `tools/terrain_paint/paint_india_east.py` and `fixed_america.py`, plus the frame helpers they share).
//!
//! Everything painted here is sampled from the user's own baseline art (the [`Learned`] statistics are
//! produced by `india.rs` / `america.rs` from the baseline tiles at run time) and a public DEM.

use super::dem::{Dem, z_to_h};
use super::grid::{Arr2, Rgb8, Rgbf};
use super::prims::{self, Mode};
use super::resample::Noise;

/// Level-0 supertexture pixels per world unit.
pub const S0: f64 = 25.6;

/// World placement of a canvas: world x of its left edge and world y of its top edge.
#[derive(Clone, Copy, Debug)]
pub struct Canvas {
    pub h: usize,
    pub w: usize,
    pub x0: f64,
    pub yt: f64,
}

/// Statistics learned from the baseline art of one theatre.
pub struct Learned {
    /// Clean parchment strip sampled next to the old frame.
    pub parch_src: Arr2<Rgb8>,
    /// Land colour by height byte for three latitude bands (3 x 256).
    pub land_lut: [Vec<Rgbf>; 3],
    pub sand: Rgbf,
    pub sea_lut_d: Vec<f32>,
    pub sea_lut: Vec<Rgbf>,
    /// 1280 x 1280 clean open-sea patch.
    pub sea_patch: Arr2<Rgb8>,
    pub sea_far: Rgbf,
    /// Two 1024 x 1024 zero-mean land high-pass texture patches.
    pub hp: [Arr2<Rgbf>; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    India,
    America,
}

/// One paper edge of the painted region.
pub struct Edge {
    /// `b'N'`, `b'S'`, `b'E'` or `b'W'`.
    pub side: u8,
    /// Canvas pixel position of the edge.
    pub pos: f64,
    /// Fade width in world units.
    pub base_m: f64,
    /// India: only where true along the cross axis (canvas columns for N/S, rows for E/W).
    pub flag: Option<Vec<bool>>,
}

pub struct PaintWin {
    /// Padded window (pad 200 px, clipped to the canvas).
    pub r0: usize,
    pub r1: usize,
    pub c0: usize,
    pub c1: usize,
    pub rgb: Arr2<Rgbf>,
    pub alpha: Arr2<u8>,
    pub land: Arr2<bool>,
    pub cov: Arr2<f32>,
}

pub enum SeaSigma {
    Const(f32),
    /// Indexed by window column (`win.c1 - win.c0` entries).
    PerColumn(Vec<f32>),
}

pub struct MatchOpts {
    pub sea_sigma: SeaSigma,
    pub land_sigma: f32,
    /// `[r0, r1, c0, c1]` in canvas pixels: colour anchors only inside (America).
    pub inwin: Option<[usize; 4]>,
}

pub struct FillChunk {
    pub dst: [usize; 2],
    pub src: [usize; 2],
    pub cost: [f64; 2],
}

// ------------------------------------------------------------------------------------------ small helpers

/// Triangle wave of period `2p` (`tri` of the Python painters): mirrors coordinates into `0..p`.
pub fn tri(c: i64, p: i64) -> i64 {
    let m = c.rem_euclid(2 * p);
    if m < p { m } else { 2 * p - 1 - m }
}

pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16)
}

/// Run `f(row, out_row)` over every row of `out` (row length `row_len`) on all cores.
fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    let per = rows.div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * per + k, row);
                }
            });
        }
    });
}

// ------------------------------------------------------------------------------------------ terrain

/// DEM elevation at canvas pixels with the coast jitter shared by texture and heightmap (`Terrain0` /
/// `Terrain0A`), plus the coarse (8 px) land mask distance field used far away from coasts.
pub struct DemTerrain<'a> {
    dem: &'a Dem,
    canvas: Canvas,
    njx: Noise,
    njy: Noise,
    /// Elevation (m) at 8 px cell centres over the whole canvas, no jitter, row 0 = north.
    pub z_coarse: Arr2<f32>,
    /// Distance (px) to the nearest land of the coarse mask (specks < 5 cells dropped), times 8.
    dlg: Arr2<f32>,
}

impl<'a> DemTerrain<'a> {
    pub fn new(dem: &'a Dem, canvas: Canvas) -> Self {
        let step = 8usize;
        let (gh, gw) = (canvas.h / step, canvas.w / step);
        let mut zc = vec![0f32; gh * gw];
        par_rows(&mut zc, gw, |r, row| {
            let y = canvas.yt - ((r * step + step / 2) as f64) / S0;
            for (c, v) in row.iter_mut().enumerate() {
                let x = canvas.x0 + ((c * step + step / 2) as f64) / S0;
                *v = dem.sample_plain(x, y);
            }
        });
        let z_coarse = Arr2::from_vec(gh, gw, zc);
        let land = z_coarse.map(|v| v > 0.0);
        let (lab, n) = prims::label(&land);
        let land_big = if n > 0 {
            let mut sz = vec![0usize; n + 1];
            for &l in &lab.d {
                sz[l as usize] += 1;
            }
            sz[0] = 0;
            lab.map(|l| sz[l as usize] >= 5)
        } else {
            land
        };
        let sea = land_big.map(|v| !v);
        let mut dlg = prims::edt(&sea);
        for v in dlg.d.iter_mut() {
            *v *= step as f32;
        }
        DemTerrain {
            dem,
            canvas,
            njx: Noise::new(canvas.h, canvas.w, 20.0, 101, 2),
            njy: Noise::new(canvas.h, canvas.w, 20.0, 102, 2),
            z_coarse,
            dlg,
        }
    }

    pub fn z_coarse_grid(&self) -> &Arr2<f32> {
        &self.z_coarse
    }

    /// Jittered level-0 elevation (m) of canvas rows `r0..r1`, columns `c0..c1`.
    pub fn z(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> Arr2<f32> {
        let jx = self.njx.sample(r0, r1, c0, c1);
        let jy = self.njy.sample(r0, r1, c0, c1);
        let (h, w) = (r1 - r0, c1 - c0);
        let mut out = vec![0f32; h * w];
        let cv = self.canvas;
        par_rows(&mut out, w, |i, row| {
            let y = cv.yt - ((r0 + i) as f64 + 0.5) / S0;
            for (j, v) in row.iter_mut().enumerate() {
                let x = cv.x0 + ((c0 + j) as f64 + 0.5) / S0;
                let k = i * w + j;
                *v = self
                    .dem
                    .sample(x, y, (0.45f32 * jx.d[k]) as f64, (0.45f32 * jy.d[k]) as f64);
            }
        });
        Arr2::from_vec(h, w, out)
    }

    /// Distance (px) sea -> nearest land from the coarse global mask, bilinearly resampled onto the window.
    pub fn dland_global(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> Arr2<f32> {
        let (h, w) = (r1 - r0, c1 - c0);
        let mut out = vec![0f32; h * w];
        par_rows(&mut out, w, |i, row| {
            let y = ((r0 + i) as f64 + 0.5) / 8.0 - 0.5;
            for (j, v) in row.iter_mut().enumerate() {
                let x = ((c0 + j) as f64 + 0.5) / 8.0 - 0.5;
                *v = super::resample::bilinear_nearest(&self.dlg, y, x);
            }
        });
        Arr2::from_vec(h, w, out)
    }
}

// ------------------------------------------------------------------------------------------ painter

pub struct Painter {
    variant: Variant,
    canvas: Canvas,
    nmargin: Noise,
    nmid: Noise,
    nz: Noise,
    nsea: Noise,
    parch: Arr2<Rgb8>,
    parch_row0: i64,
    parch_col0: i64,
    hp: [Arr2<Rgbf>; 2],
    lut: [Vec<Rgbf>; 3],
    sand: Rgbf,
    sea_patch: Arr2<Rgb8>,
    sea_lut_d: Vec<f64>,
    sea_lut_ch: [Vec<f64>; 3],
    sea_far: Rgbf,
}

impl Painter {
    pub fn new(v: Variant, canvas: Canvas, l: &Learned) -> Painter {
        let (h, w) = (canvas.h, canvas.w);
        let (parch_row0, parch_col0) = match v {
            Variant::India => (1792, 0),
            Variant::America => (
                ((canvas.yt - 350.0) * S0).round() as i64,
                ((-765.0 - canvas.x0) * S0).round() as i64,
            ),
        };
        let ch = |k: usize| l.sea_lut.iter().map(|c| c[k] as f64).collect::<Vec<f64>>();
        Painter {
            variant: v,
            canvas,
            nmargin: Noise::new(h, w, 70.0, 201, 3),
            nmid: Noise::new(h, w, 200.0, 202, 3),
            nz: Noise::new(h, w, 7.0, 203, 2),
            nsea: Noise::new(h, w, 300.0, 204, 2),
            parch: l.parch_src.clone(),
            parch_row0,
            parch_col0,
            hp: l.hp.clone(),
            lut: l.land_lut.clone(),
            sand: l.sand,
            sea_patch: l.sea_patch.clone(),
            sea_lut_d: l.sea_lut_d.iter().map(|&v| v as f64).collect(),
            sea_lut_ch: [ch(0), ch(1), ch(2)],
            sea_far: l.sea_far,
        }
    }

    pub(super) fn land_colour(&self, h: f32, yw: f64) -> Rgbf {
        let hi = h.clamp(0.0, 255.0);
        let h0 = hi.floor() as usize;
        let h1 = (h0 + 1).min(255);
        let fh = hi - h0 as f32;
        let (w, ws): ([f64; 3], f64) = match self.variant {
            Variant::India => {
                let yc = yw.clamp(65.0, 240.0);
                let w0 = (1.0 - (yc - 65.0).abs() / 90.0).clamp(0.0, 1.0);
                let w1 = (1.0 - (yc - 155.0).abs() / 90.0).clamp(0.0, 1.0);
                let w2 = (1.0 - (yc - 240.0).abs() / 90.0).clamp(0.0, 1.0);
                ([w0, w1, w2], w0 + w1 + w2 + 1e-6)
            }
            Variant::America => {
                // hat functions over the learned band centres 75 / 220 / 375: a partition of unity
                let yc = yw.clamp(50.0, 400.0);
                let w0 = ((220.0 - yc) / 145.0).clamp(0.0, 1.0);
                let w2 = ((yc - 220.0) / 155.0).clamp(0.0, 1.0);
                ([w0, 1.0 - w0 - w2, w2], 1.0)
            }
        };
        let mut out = [0f32; 3];
        for (b, wb) in w.iter().enumerate() {
            let a = self.lut[b][h0];
            let c = self.lut[b][h1];
            let k = (*wb / ws) as f32;
            for ch in 0..3 {
                out[ch] += (a[ch] * (1.0 - fh) + c[ch] * fh) * k;
            }
        }
        out
    }

    /// Paint the window (rows `r0..r1`, columns `c0..c1`) of the canvas. The returned window is padded by
    /// 200 px on every side (clipped to the canvas); its colours are the unmatched painter output.
    pub fn paint(
        &self,
        terr: &DemTerrain,
        r0: usize,
        r1: usize,
        c0: usize,
        c1: usize,
        edges: &[Edge],
    ) -> PaintWin {
        let pad = 200usize;
        let (rr0, rr1) = (r0.saturating_sub(pad), (r1 + pad).min(self.canvas.h));
        let (cc0, cc1) = (c0.saturating_sub(pad), (c1 + pad).min(self.canvas.w));
        let (h, w) = (rr1 - rr0, cc1 - cc0);
        let z = terr.z(rr0, rr1, cc0, cc1);
        let nz = self.nz.sample(rr0, rr1, cc0, cc1);
        let land = z.map(|v| v > 0.0);
        let zs_in = Arr2::from_vec(
            h,
            w,
            z.d.iter()
                .zip(&nz.d)
                .map(|(&zv, &n)| {
                    let zf = if zv > 0.0 {
                        zv + n * (25.0 + 0.045 * zv.max(0.0))
                    } else {
                        zv
                    };
                    zf.max(0.0)
                })
                .collect(),
        );
        drop(nz);
        let any_sea = land.d.iter().any(|&l| !l);
        let d_sea = if any_sea {
            prims::edt(&land)
        } else {
            Arr2::new(h, w, 1.0e4f32)
        };
        let dl_g = terr.dland_global(rr0, rr1, cc0, cc1);
        let (lab, n) = prims::label(&land);
        let land_big = if n > 0 {
            let mut sz = vec![0usize; n + 1];
            for &l in &lab.d {
                sz[l as usize] += 1;
            }
            sz[0] = 0;
            lab.map(|l| sz[l as usize] >= 300)
        } else {
            land.clone()
        };
        drop(lab);
        let d_land = if land_big.d.iter().any(|&v| v) {
            let sea_big = land_big.map(|v| !v);
            let d_loc = prims::edt(&sea_big);
            Arr2::from_vec(
                h,
                w,
                d_loc
                    .d
                    .iter()
                    .zip(&dl_g.d)
                    .map(|(&dl, &dg)| {
                        let wb = smoothstep((dl - 100.0) / 50.0);
                        (1.0 - wb) * dl + wb * dg
                    })
                    .collect(),
            )
        } else {
            dl_g
        };
        // hillshade from the smoothed, noise-roughened elevation
        let zs = prims::gaussian_filter(&zs_in, 1.5);
        drop(zs_in);
        let (gy, gx) = prims::gradient(&zs);
        drop(zs);
        let (mut lx, mut ly, mut lz) = (-0.55f32, 0.55f32, 0.63f32);
        let ln = (lx * lx + ly * ly + lz * lz).sqrt();
        lx /= ln;
        ly /= ln;
        lz /= ln;
        let k = 3.5f32;
        let nm = self.nmid.sample(rr0, rr1, cc0, cc1);
        let nmg = self.nmargin.sample(rr0, rr1, cc0, cc1);
        let ns = self.nsea.sample(rr0, rr1, cc0, cc1);
        let sea_soft = prims::gaussian_filter(&land.map(|l| if l { 1.0f32 } else { 0.0 }), 0.9);

        let pp = self.sea_patch.h as i64;
        let nedge = edges.len();
        let mut rgb = vec![[0f32; 3]; h * w];
        let mut alpha = vec![0u8; h * w];
        let mut cov_out = vec![0f32; h * w];
        let cv = self.canvas;
        // one pass per row; the three output rows are filled together
        let mut rows: Vec<(&mut [Rgbf], &mut [u8], &mut [f32])> = rgb
            .chunks_mut(w)
            .zip(alpha.chunks_mut(w))
            .zip(cov_out.chunks_mut(w))
            .map(|((a, b), c)| (a, b, c))
            .collect();
        let per = h.div_ceil(threads()).max(1);
        std::thread::scope(|sc| {
            for (ci, chunk) in rows.chunks_mut(per).enumerate() {
                let (z, d_sea, d_land, gy, gx, nm, nmg, ns, sea_soft) =
                    (&z, &d_sea, &d_land, &gy, &gx, &nm, &nmg, &ns, &sea_soft);
                sc.spawn(move || {
                    for (kk, (orgb, oal, ocov)) in chunk.iter_mut().enumerate() {
                        let i = ci * per + kk;
                        let r = rr0 + i;
                        let yw = cv.yt - (r as f64 + 0.5) / S0;
                        for j in 0..w {
                            let c = cc0 + j;
                            let p = i * w + j;
                            let zv = z.d[p];
                            let hh = z_to_h(zv.max(0.0));
                            let mut col = self.land_colour(hh, yw);
                            // hillshade
                            let dzdx = gx.d[p] / 610.0;
                            let dzdy_n = -gy.d[p] / 610.0;
                            let (nx, ny) = (-dzdx * k, -dzdy_n * k);
                            let nrm = (nx * nx + ny * ny + 1.0).sqrt();
                            let hs = (nx * lx + ny * ly + lz) / nrm;
                            let shade = (1.0 + 1.15 * (hs - lz)).clamp(0.55, 1.45);
                            let t0 = self.hp[0].get(
                                tri(r as i64 + 123, 1024) as usize,
                                tri(c as i64 + 457, 1024) as usize,
                            );
                            let t1 = self.hp[1].get(
                                tri(r as i64 + 701, 1024) as usize,
                                tri(c as i64 + 29, 1024) as usize,
                            );
                            let modn = 1.0 + 0.055 * nm.d[p];
                            let wr = ((1.0 - d_sea.d[p] / 7.0).clamp(0.0, 1.0).powf(1.3)) * 0.78;
                            for ch in 0..3 {
                                let v = col[ch] * shade + (0.45 * t0[ch] + 0.45 * t1[ch]);
                                let v = v * modn;
                                col[ch] = v * (1.0 - wr) + self.sand[ch] * wr;
                            }
                            // sea
                            let clone = self.sea_patch.get(
                                tri(r as i64 + 311, pp) as usize,
                                tri(c as i64 + 97, pp) as usize,
                            );
                            let dd = (d_land.d[p] as f64).min(900.0);
                            let mut sea = [0f32; 3];
                            for ch in 0..3 {
                                let lut_d =
                                    prims::interp(dd, &self.sea_lut_d, &self.sea_lut_ch[ch]) as f32;
                                sea[ch] =
                                    clone[ch] as f32 + (lut_d - self.sea_far[ch]) + 3.0 * ns.d[p];
                            }
                            // distance (world units) to the nearest paper edge, > 0 inside the painted area
                            let jit = 0.9 * nmg.d[p] as f64;
                            let mut s = 1.0e3f64;
                            for e in edges.iter().take(nedge) {
                                let mut d = match e.side {
                                    b'E' => (e.pos - (c as f64 + 0.5)) / S0,
                                    b'W' => ((c as f64 + 0.5) - e.pos) / S0,
                                    b'S' => (e.pos - (r as f64 + 0.5)) / S0,
                                    _ => ((r as f64 + 0.5) - e.pos) / S0,
                                };
                                d -= e.base_m + jit;
                                if let Some(flag) = &e.flag {
                                    let on = if e.side == b'N' || e.side == b'S' {
                                        flag[c]
                                    } else {
                                        flag[r]
                                    };
                                    if !on {
                                        d = 1.0e3;
                                    }
                                }
                                s = s.min(d);
                            }
                            let cov = (s * S0 + 0.5).clamp(0.0, 1.0) as f32;
                            let sf = s as f32;
                            let rim = (1.0 - sf / 2.2).clamp(0.0, 1.0).powf(1.6);
                            let vig = (1.0 - sf / 4.5).clamp(0.0, 1.0).powf(1.4);
                            let soft = sea_soft.d[p];
                            let parch = self.parchment(r, c, yw);
                            let mut out = [0f32; 3];
                            for ch in 0..3 {
                                let land_rgb = col[ch] * (1.0 - 0.34 * rim);
                                let sea_rgb = sea[ch] * (1.0 - 0.40 * vig);
                                let terrain = land_rgb * soft + sea_rgb * (1.0 - soft);
                                out[ch] =
                                    (terrain * cov + parch[ch] * (1.0 - cov)).clamp(0.0, 255.0);
                            }
                            orgb[j] = out;
                            oal[j] = (255.0 * (1.0 - soft) * cov)
                                .round_ties_even()
                                .clamp(0.0, 255.0) as u8;
                            ocov[j] = cov;
                        }
                    }
                });
            }
        });
        drop(rows);
        PaintWin {
            r0: rr0,
            r1: rr1,
            c0: cc0,
            c1: cc1,
            rgb: Arr2::from_vec(h, w, rgb),
            alpha: Arr2::from_vec(h, w, alpha),
            land,
            cov: Arr2::from_vec(h, w, cov_out),
        }
    }

    /// Replicated clean parchment under the painted area (India also darkens the south ornament shadow band).
    fn parchment(&self, r: usize, c: usize, yw: f64) -> Rgbf {
        let (ph, pw) = (self.parch.h as i64, self.parch.w as i64);
        let ri = tri(r as i64 - self.parch_row0, ph);
        let ci = match self.variant {
            Variant::India => (c as i64 + (ri / 97) * 13).rem_euclid(pw),
            Variant::America => tri(c as i64 - self.parch_col0, pw),
        };
        let p = self.parch.get(ri as usize, ci as usize);
        let mut out = [p[0] as f32, p[1] as f32, p[2] as f32];
        if self.variant == Variant::India {
            let y0 = 13.75 - 100.0;
            if yw >= y0 {
                let f = 1.0 - 0.22 * smoothstep((1.0 - (yw - y0) / 2.2) as f32);
                for v in out.iter_mut() {
                    *v *= f;
                }
            }
        }
        out
    }
}

// ------------------------------------------------------------------------------------------ colour matching

/// Colour-match the painted window to the baseline pixels it borders: the painted pixels (`m`) get a smooth
/// per-class (sea / land) additive correction learned from the nearest unowned anchors, fading with distance.
/// All arrays cover the padded window `win.r0..r1 x win.c0..c1`. `win.rgb` is corrected in place.
pub fn colour_match(
    win: &mut PaintWin,
    m: &Arr2<bool>,
    vland: &Arr2<bool>,
    cur_rgb: &Arr2<Rgb8>,
    cur_al: &Arr2<u8>,
    o: &MatchOpts,
) {
    let (h, w) = (win.rgb.h, win.rgb.w);
    let inwin = |r: usize, c: usize| -> bool {
        match o.inwin {
            None => true,
            Some([a, b, cc0, cc1]) => {
                let (gr, gc) = (win.r0 + r, win.c0 + c);
                gr >= a && gr < b && gc >= cc0 && gc < cc1
            }
        }
    };
    for sea_class in [true, false] {
        if !sea_class && o.land_sigma <= 0.0 {
            continue;
        }
        let anchors: Arr2<bool> = if sea_class {
            let mut a = Arr2::new(h, w, false);
            for r in 0..h {
                for c in 0..w {
                    let i = r * w + c;
                    a.d[i] = cur_al.d[i] == 255 && !m.d[i] && inwin(r, c);
                }
            }
            a
        } else {
            let mut a = prims::erode(vland, 10);
            for r in 0..h {
                for c in 0..w {
                    let i = r * w + c;
                    a.d[i] = a.d[i] && inwin(r, c);
                }
            }
            a
        };
        if !anchors.d.iter().any(|&v| v) {
            continue;
        }
        let not_a = anchors.map(|v| !v);
        let (dist, near) = prims::edt_with_indices(&not_a);
        drop(not_a);
        let af = anchors.map(|v| if v { 1.0f32 } else { 0.0 });
        let den = prims::gaussian_filter(&af, 8.0);
        let mut ds: [Arr2<f32>; 3] = [
            Arr2::new(0, 0, 0.0),
            Arr2::new(0, 0, 0.0),
            Arr2::new(0, 0, 0.0),
        ];
        for ch in 0..3 {
            let d = Arr2::from_vec(
                h,
                w,
                (0..h * w)
                    .map(|i| {
                        if af.d[i] > 0.0 {
                            cur_rgb.d[i][ch] as f32 - win.rgb.d[i][ch]
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
            let mut g = prims::gaussian_filter(&d, 8.0);
            for (v, dn) in g.d.iter_mut().zip(&den.d) {
                *v /= dn.max(1e-3);
            }
            ds[ch] = g;
        }
        for r in 0..h {
            for c in 0..w {
                let i = r * w + c;
                if !m.d[i] || win.land.d[i] == sea_class {
                    continue;
                }
                let sg = if sea_class {
                    match &o.sea_sigma {
                        SeaSigma::Const(v) => *v,
                        SeaSigma::PerColumn(v) => v[c],
                    }
                } else {
                    o.land_sigma
                };
                let d = dist.d[i];
                if d >= 3.0 * sg {
                    continue;
                }
                let wgt = (-(d / sg) * (d / sg)).exp() * win.cov.d[i];
                let ni = near.d[i] as usize;
                for ch in 0..3 {
                    let corr = ds[ch].d[ni].clamp(-45.0, 45.0);
                    win.rgb.d[i][ch] += corr * wgt;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------ frames

fn mean_cols(rgb: &Arr2<Rgb8>, r0: usize, nrows: usize, c0: usize, c1: usize) -> Vec<Rgbf> {
    let n = (c1 - c0) as f32;
    (0..nrows)
        .map(|r| {
            let mut s = [0f32; 3];
            for c in c0..c1 {
                let p = rgb.get(r0 + r, c);
                for ch in 0..3 {
                    s[ch] += p[ch] as f32;
                }
            }
            [s[0] / n, s[1] / n, s[2] / n]
        })
        .collect()
}

/// Pick the source column `a` in `[a_min, a_max]` whose `n`-column run continues the destination line `left`
/// (and `right`, when given) most seamlessly. `band` is the baseline band (rows `r0..r0+nrows`); source
/// "lines" are its columns. Returns `(a, cost_left, cost_right)`.
pub fn best_offset(
    band: &Arr2<Rgb8>,
    r0: usize,
    nrows: usize,
    a_min: usize,
    a_max: usize,
    n: usize,
    left: &[Rgbf],
    right: Option<&[Rgbf]>,
) -> (usize, f64, f64) {
    const NREF: usize = 4;
    let cost = |a: usize| -> (f64, f64) {
        let ml = mean_cols(band, r0, nrows, a - NREF, a);
        let mut cl = 0f64;
        for (x, y) in left.iter().zip(&ml) {
            for ch in 0..3 {
                cl += (x[ch] as f64 - y[ch] as f64).abs();
            }
        }
        cl /= (nrows * 3) as f64;
        let cr = match right {
            Some(rt) => {
                let mr = mean_cols(band, r0, nrows, a + n, a + n + NREF);
                let mut s = 0f64;
                for (x, y) in rt.iter().zip(&mr) {
                    for ch in 0..3 {
                        s += (x[ch] as f64 - y[ch] as f64).abs();
                    }
                }
                s / (nrows * 3) as f64
            }
            None => 0.0,
        };
        (cl, cr)
    };
    let cands: Vec<usize> = (a_min..=a_max).collect();
    let per = cands.len().div_ceil(threads()).max(1);
    let parts: Vec<(f64, usize, f64, f64)> = std::thread::scope(|s| {
        let hs: Vec<_> = cands
            .chunks(per)
            .map(|ch| {
                let cost = &cost;
                s.spawn(move || {
                    let mut best = (f64::INFINITY, ch[0], 0.0, 0.0);
                    for &a in ch {
                        let (cl, cr) = cost(a);
                        if cl + cr < best.0 {
                            best = (cl + cr, a, cl, cr);
                        }
                    }
                    best
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("best_offset worker"))
            .collect()
    });
    // chunks are ascending, so the first strict minimum of the sequential scan is the first chunk-best
    let mut best = parts[0];
    for p in &parts[1..] {
        if p.0 < best.0 {
            best = *p;
        }
    }
    (best.1, best.2, best.3)
}

/// Fill destination columns `p0..p1` of the band `new_*[new_r0 .. new_r0+nrows]` with seam-optimised chunks
/// of the baseline band `old_*[old_r0 ..]` (source columns `[src_lo, src_hi)`). Every chunk is matched to the
/// destination columns just before it; the last one also to the columns after `p1` when `match_right`.
/// `old_*` must not alias `new_*` (pass a copy of the band when both are the same canvas).
#[allow(clippy::too_many_arguments)]
pub fn fill_cols(
    new_rgb: &mut Arr2<Rgb8>,
    new_al: &mut Arr2<u8>,
    new_r0: usize,
    old_rgb: &Arr2<Rgb8>,
    old_al: &Arr2<u8>,
    old_r0: usize,
    nrows: usize,
    p0: usize,
    p1: usize,
    src_lo: usize,
    src_hi: usize,
    chunk: usize,
    match_right: bool,
) -> (Vec<usize>, Vec<FillChunk>) {
    let mut pos = p0;
    let mut chunks: Vec<FillChunk> = Vec::new();
    while pos < p1 {
        let n = if ((p1 - pos) as f64) < 1.5 * chunk as f64 {
            p1 - pos
        } else {
            chunk
        };
        let left = mean_cols(new_rgb, new_r0, nrows, pos - 4, pos);
        let right = if match_right && pos + n == p1 {
            Some(mean_cols(new_rgb, new_r0, nrows, p1, p1 + 4))
        } else {
            None
        };
        let (a, cl, cr) = best_offset(
            old_rgb,
            old_r0,
            nrows,
            src_lo + 4,
            src_hi - n - 4,
            n,
            &left,
            right.as_deref(),
        );
        for r in 0..nrows {
            for k in 0..n {
                new_rgb.set(new_r0 + r, pos + k, old_rgb.get(old_r0 + r, a + k));
                new_al.set(new_r0 + r, pos + k, old_al.get(old_r0 + r, a + k));
            }
        }
        chunks.push(FillChunk {
            dst: [pos, pos + n],
            src: [a, a + n],
            cost: [(cl * 100.0).round() / 100.0, (cr * 100.0).round() / 100.0],
        });
        pos += n;
    }
    let joins = chunks.iter().skip(1).map(|c| c.dst[0]).collect();
    (joins, chunks)
}

/// Remove the low-frequency colour step across a frame-fill seam with a linear fade over `width` px (fill side
/// only). `axis` 1: `seam` is a column index; 0: a row index. `side` > 0: the filled stretch starts at `seam`;
/// < 0: it ends at `seam`.
pub fn fade_seam(img: &Arr2<Rgb8>, seam: usize, axis: u8, side: i32, width: usize) -> Arr2<Rgb8> {
    let (lines, plen) = if axis == 1 {
        (img.h, img.w)
    } else {
        (img.w, img.h)
    };
    let at = |line: usize, pos: usize| -> (usize, usize) {
        if axis == 1 { (line, pos) } else { (pos, line) }
    };
    assert!(
        seam >= 6 && seam + 6 <= plen,
        "fade_seam: seam {seam} too close to the border"
    );
    let mean6 = |line: usize, p0: usize| -> Rgbf {
        let mut s = [0f32; 3];
        for p in p0..p0 + 6 {
            let (r, c) = at(line, p);
            let v = img.get(r, c);
            for ch in 0..3 {
                s[ch] += v[ch] as f32;
            }
        }
        [s[0] / 6.0, s[1] / 6.0, s[2] / 6.0]
    };
    let diff: Vec<[f32; 3]> = (0..lines)
        .map(|l| {
            let a = mean6(l, seam - 6);
            let b = mean6(l, seam);
            if side > 0 {
                [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
            } else {
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
            }
        })
        .collect();
    let d = prims::uniform_filter1d_lut(&diff, 48, Mode::Nearest);
    let wts: Vec<f32> = (0..width)
        .map(|i| {
            let t = if width > 1 {
                i as f64 / (width - 1) as f64
            } else {
                0.0
            };
            if side > 0 { (1.0 - t) as f32 } else { t as f32 }
        })
        .collect();
    let mut out = img.clone();
    let (start, end) = if side > 0 {
        (seam, seam + width)
    } else {
        (seam - width, seam)
    };
    assert!(
        end <= plen,
        "fade_seam: fade width {width} leaves the image"
    );
    for l in 0..lines {
        for (k, p) in (start..end).enumerate() {
            let (r, c) = at(l, p);
            let v = img.get(r, c);
            let mut o = [0u8; 3];
            for ch in 0..3 {
                o[ch] = (v[ch] as f32 + d[l][ch] * wts[k])
                    .round_ties_even()
                    .clamp(0.0, 255.0) as u8;
            }
            out.set(r, c, o);
        }
    }
    out
}
