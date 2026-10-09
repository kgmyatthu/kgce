//! Numerical parity layer for the fixed-theatre painters: numpy `default_rng` (PCG64) bit-exact RNG,
//! `geo.Noise`, Pillow `Image.resize` for mode `'F'`, and scipy cubic-spline DEM sampling.
//!
//! Provenance of every algorithm: numpy `random` (SeedSequence, PCG64 XSL-RR 128/64, ziggurat normal,
//! `random_interval`), Pillow `Resample.c` (`precompute_coeffs`, float horizontal/vertical passes) and
//! scipy `ndimage` (`ni_splines.c` mirror prefilter, `ni_interpolation.c` geometric transform with mode
//! `nearest`).  The RNG is bit exact against numpy 2.x; the image resamplers evaluate the same double
//! precision recurrences in the same order as the reference implementations.

use super::grid::Arr2;
use std::sync::LazyLock;

// ======================================================================================= numpy PCG64

const SS_INIT_A: u32 = 0x43b0_d7e5;
const SS_MULT_A: u32 = 0x931e_8875;
const SS_INIT_B: u32 = 0x8b51_f9dd;
const SS_MULT_B: u32 = 0x58f3_8ded;
const SS_MIX_MULT_L: u32 = 0xca01_f9dd;
const SS_MIX_MULT_R: u32 = 0x4973_f715;
const SS_XSHIFT: u32 = 16;
const SS_POOL: usize = 4;

fn ss_hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(SS_MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> SS_XSHIFT)
}

fn ss_mix(x: u32, y: u32) -> u32 {
    let r = SS_MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(SS_MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> SS_XSHIFT)
}

/// `SeedSequence(seed).generate_state(4, np.uint64)`.
fn seed_sequence_state(seed: u64) -> [u64; 4] {
    // entropy words, little endian; 0 -> [0]
    let mut entropy = vec![seed as u32];
    if seed >> 32 != 0 {
        entropy.push((seed >> 32) as u32);
    }
    let mut pool = [0u32; SS_POOL];
    let mut hc = SS_INIT_A;
    for (i, p) in pool.iter_mut().enumerate() {
        *p = ss_hashmix(entropy.get(i).copied().unwrap_or(0), &mut hc);
    }
    for i_src in 0..SS_POOL {
        for i_dst in 0..SS_POOL {
            if i_src != i_dst {
                let h = ss_hashmix(pool[i_src], &mut hc);
                pool[i_dst] = ss_mix(pool[i_dst], h);
            }
        }
    }
    for &e in entropy.iter().skip(SS_POOL) {
        for p in pool.iter_mut() {
            let h = ss_hashmix(e, &mut hc);
            *p = ss_mix(*p, h);
        }
    }
    let mut hc = SS_INIT_B;
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut d = pool[i % SS_POOL] ^ hc;
        hc = hc.wrapping_mul(SS_MULT_B);
        d = d.wrapping_mul(hc);
        *w = d ^ (d >> SS_XSHIFT);
    }
    let mut out = [0u64; 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32;
    }
    out
}

const PCG_MULT: u128 = ((2549297995355413924u128) << 64) | 4865540595714422341u128;

pub(crate) struct Ziggurat {
    pub(crate) ki: [u64; 256],
    pub(crate) wi: [f64; 256],
    pub(crate) fi: [f64; 256],
}

const ZIG_R: f64 = 3.654_152_885_361_008_8;
const ZIG_INV_R: f64 = 0.273_661_237_329_758_28;

/// Layer area used by numpy's table generator (inferred from `wi_double[0] = area / f(r) / 2^52`; it
/// deviates from the analytic area `r f(r) + integral_r^inf f` by ~5e-15 relative).
const ZIG_SECTION_AREA: f64 = 0.004_928_673_233_974_652;

/// numpy's 52-bit double ziggurat tables for the normal distribution, rebuilt from the construction
/// formula (Marsaglia-Tsang, 256 layers, r = 3.6541528853610088, layer area `ZIG_SECTION_AREA`,
/// integer thresholds rounded half-to-even); verified entry-for-entry against numpy's constants.
pub(crate) static ZIGGURAT: LazyLock<Ziggurat> = LazyLock::new(|| {
    let m = 4503599627370496.0f64; // 2^52
    let f = |x: f64| (-0.5 * x * x).exp();
    let mut ki = [0u64; 256];
    let mut wi = [0f64; 256];
    let mut fi = [0f64; 256];
    let mut dn = ZIG_R;
    let mut tn = dn;
    let vn = ZIG_SECTION_AREA;
    let q = vn / f(dn);
    ki[0] = ((dn / q) * m).round_ties_even() as u64;
    ki[1] = 0;
    wi[0] = q / m;
    wi[255] = dn / m;
    fi[0] = 1.0;
    fi[255] = f(dn);
    for i in (1..=254).rev() {
        dn = (-2.0 * (vn / dn + f(dn)).ln()).sqrt();
        ki[i + 1] = ((dn / tn) * m).round_ties_even() as u64;
        tn = dn;
        fi[i] = f(dn);
        wi[i] = dn / m;
    }
    Ziggurat { ki, wi, fi }
});

/// numpy `np.random.default_rng(seed)` (PCG64, XSL-RR 128/64), bit exact.
#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    pub fn new(seed: u64) -> Self {
        let s = seed_sequence_state(seed);
        let initstate = (s[0] as u128) << 64 | s[1] as u128;
        let initseq = (s[2] as u128) << 64 | s[3] as u128;
        let mut r = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    /// numpy `next_uint32`: buffered halves of a 64-bit draw (low half first, then high half).
    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let n = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (n >> 32) as u32;
        n as u32
    }

    /// numpy `next_double`: 53-bit uniform in [0,1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// `Generator.standard_normal()` (ziggurat, one double).
    pub fn standard_normal(&mut self) -> f64 {
        let z = &*ZIGGURAT;
        loop {
            let mut r = self.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * z.wi[idx];
            if sign != 0 {
                x = -x;
            }
            if rabs < z.ki[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIG_INV_R * (-self.next_f64()).ln_1p();
                    let yy = -(-self.next_f64()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 1 != 0 {
                            -(ZIG_R + xx)
                        } else {
                            ZIG_R + xx
                        };
                    }
                }
            } else if (z.fi[idx - 1] - z.fi[idx]) * self.next_f64() + z.fi[idx]
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    /// numpy `random_interval(max)`: uniform integer in `[0, max]` by masked rejection.
    fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= 0xffff_ffff {
            loop {
                let v = (self.next_u32() as u64) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `Generator.permutation(n)`: `arange(n)` shuffled in place (Fisher-Yates from the end).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut a: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.random_interval(i as u64) as usize;
            a.swap(i, j);
        }
        a
    }
}

// ======================================================================================= PIL resize

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PilFilter {
    Bilinear,
    Bicubic,
}

impl PilFilter {
    fn support(self) -> f64 {
        match self {
            PilFilter::Bilinear => 1.0,
            PilFilter::Bicubic => 2.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            PilFilter::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            PilFilter::Bicubic => {
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
    ksize: usize,
    /// per output index: (xmin, count)
    bounds: Vec<(usize, usize)>,
    kk: Vec<f64>,
}

/// Pillow `precompute_coeffs`; `in0`/`in1` are the float32 box edges Pillow receives.
fn precompute_coeffs(in_size: usize, in0: f32, in1: f32, out_size: usize, f: PilFilter) -> Coeffs {
    let scale = (in1 - in0) as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = f.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut kk = vec![0f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    let in0 = in0 as f64;
    for xx in 0..out_size {
        let center = in0 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        xmax -= xmin;
        let xmax = xmax.max(0) as usize;
        let k = &mut kk[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0f64;
        for x in 0..xmax {
            let w = f.eval((x as f64 + xmin as f64 - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        if ww != 0.0 {
            for v in k.iter_mut().take(xmax) {
                *v /= ww;
            }
        }
        bounds.push((xmin as usize, xmax));
    }
    Coeffs { ksize, bounds, kk }
}

fn workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 32)
}

/// Pillow `ImagingResampleHorizontal` (mode 'F'): rows `first..first+rows` of `src` -> `rows x ow`.
fn resample_horizontal(
    src: &Arr2<f32>,
    first: usize,
    rows: usize,
    ow: usize,
    c: &Coeffs,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * ow];
    if rows == 0 || ow == 0 {
        return out;
    }
    let nt = workers().min(rows);
    let per = rows.div_ceil(nt);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * ow).enumerate() {
            s.spawn(move || {
                for (ri, orow) in chunk.chunks_mut(ow).enumerate() {
                    let srow = src.row(first + ci * per + ri);
                    for (xx, o) in orow.iter_mut().enumerate() {
                        let (xmin, n) = c.bounds[xx];
                        let k = &c.kk[xx * c.ksize..xx * c.ksize + n];
                        let px = &srow[xmin..xmin + n];
                        let mut ss = 0.0f64;
                        for (p, kv) in px.iter().zip(k) {
                            ss += *p as f64 * *kv;
                        }
                        *o = ss as f32;
                    }
                }
            });
        }
    });
    out
}

/// Core of `Image.resize((ow, oh), filter, box)` for mode 'F'. `out` has `oh*ow` entries; with
/// `add = Some(w)` the result is accumulated as `out += w * resized` in float32 (numpy `acc += w * arr`).
fn resize_into(
    src: &Arr2<f32>,
    ow: usize,
    oh: usize,
    f: PilFilter,
    bx: [f32; 4],
    out: &mut [f32],
    add: Option<f32>,
) {
    assert!(ow > 0 && oh > 0, "pil resize: empty output size");
    assert!(out.len() == ow * oh);
    let need_h = ow != src.w || bx[0] != 0.0 || bx[2] != ow as f32;
    let need_v = oh != src.h || bx[1] != 0.0 || bx[3] != oh as f32;
    let ch = precompute_coeffs(src.w, bx[0], bx[2], ow, f);
    let cv = precompute_coeffs(src.h, bx[1], bx[3], oh, f);
    let ybox_first = cv.bounds[0].0;
    let last = cv.bounds[oh - 1];
    let ybox_last = last.0 + last.1;
    let rows = ybox_last.saturating_sub(ybox_first);
    let temp_store;
    let (temp, tw): (&[f32], usize) = if need_h {
        temp_store = resample_horizontal(src, ybox_first, rows, ow, &ch);
        (&temp_store, ow)
    } else {
        (&src.d[ybox_first * src.w..], src.w)
    };
    let put = |o: &mut f32, v: f32| match add {
        Some(w) => *o += w * v,
        None => *o = v,
    };
    if !need_v {
        for yy in 0..oh {
            for xx in 0..ow {
                put(&mut out[yy * ow + xx], temp[yy * tw + xx]);
            }
        }
        return;
    }
    let nt = workers().min(oh);
    let per = oh.div_ceil(nt);
    let cvr = &cv;
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * ow).enumerate() {
            s.spawn(move || {
                let mut acc = vec![0f64; ow];
                for (ri, orow) in chunk.chunks_mut(ow).enumerate() {
                    let yy = ci * per + ri;
                    let (ymin, n) = cvr.bounds[yy];
                    let k = &cvr.kk[yy * cvr.ksize..yy * cvr.ksize + n];
                    acc.iter_mut().for_each(|a| *a = 0.0);
                    for (j, kv) in k.iter().enumerate() {
                        let trow =
                            &temp[(ymin - ybox_first + j) * tw..(ymin - ybox_first + j) * tw + ow];
                        for (a, t) in acc.iter_mut().zip(trow) {
                            *a += *t as f64 * *kv;
                        }
                    }
                    for (o, a) in orow.iter_mut().zip(&acc) {
                        put(o, *a as f32);
                    }
                }
            });
        }
    });
}

fn box_f32(src: &Arr2<f32>, boxr: Option<[f64; 4]>) -> [f32; 4] {
    match boxr {
        Some(b) => {
            let bx = [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32];
            assert!(
                bx[0] >= 0.0
                    && bx[1] >= 0.0
                    && bx[2] >= bx[0]
                    && bx[3] >= bx[1]
                    && bx[2] <= src.w as f32
                    && bx[3] <= src.h as f32,
                "pil resize: box {b:?} outside source {}x{}",
                src.w,
                src.h
            );
            bx
        }
        None => [0.0, 0.0, src.w as f32, src.h as f32],
    }
}

/// `Image.fromarray(a, 'F').resize((out_w, out_h), f, box=boxr)` as Pillow's `ImagingResample`:
/// horizontal pass then vertical pass, double precision coefficients/accumulators, float32 between passes.
/// `boxr = (left, top, right, bottom)` in source pixels (converted to float32 like Pillow does).
pub fn pil_resize_f32(
    src: &Arr2<f32>,
    out_w: usize,
    out_h: usize,
    f: PilFilter,
    boxr: Option<[f64; 4]>,
) -> Arr2<f32> {
    let bx = box_f32(src, boxr);
    let mut out = vec![0f32; out_w * out_h];
    resize_into(src, out_w, out_h, f, bx, &mut out, None);
    Arr2 {
        h: out_h,
        w: out_w,
        d: out,
    }
}

// ======================================================================================= geo.Noise

/// Port of `geo.Noise`: window-addressable fractal value noise (float32 lattices, PIL bicubic).
pub struct Noise {
    s: f64,
    lattices: Vec<Arr2<f32>>,
}

impl Noise {
    /// `Noise((h, w), scale_px, seed, octaves)`: the lattice of every octave `o` has
    /// `(int(h/sc)+6, int(w/sc)+6)` standard normals, `sc = scale/2**o`, drawn from one shared RNG.
    pub fn new(h: usize, w: usize, scale_px: f64, seed: u64, octaves: usize) -> Self {
        let mut rng = Pcg64::new(seed);
        let mut lattices = Vec::with_capacity(octaves);
        for o in 0..octaves {
            let sc = scale_px / (1u64 << o) as f64;
            let lh = (h as f64 / sc) as usize + 6;
            let lw = (w as f64 / sc) as usize + 6;
            let d: Vec<f32> = (0..lh * lw).map(|_| rng.standard_normal() as f32).collect();
            lattices.push(Arr2 { h: lh, w: lw, d });
        }
        Noise {
            s: scale_px,
            lattices,
        }
    }

    /// `geo.Noise.sample(r0, r1, c0, c1)`.
    pub fn sample(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> Arr2<f32> {
        let (oh, ow) = (r1 - r0, c1 - c0);
        let mut acc = vec![0f32; oh * ow];
        let mut wsum = 0.0f64;
        for (o, lat) in self.lattices.iter().enumerate() {
            let sc = self.s / (1u64 << o) as f64;
            let w = 0.55f64.powi(o as i32);
            let bx = box_f32(
                lat,
                Some([
                    c0 as f64 / sc + 1.0,
                    r0 as f64 / sc + 1.0,
                    c1 as f64 / sc + 1.0,
                    r1 as f64 / sc + 1.0,
                ]),
            );
            resize_into(
                lat,
                ow,
                oh,
                PilFilter::Bicubic,
                bx,
                &mut acc,
                Some(w as f32),
            );
            wsum += w * w;
        }
        let norm = wsum.sqrt();
        let nt = workers().min(oh.max(1));
        let per = (oh.div_ceil(nt.max(1)) * ow).max(1);
        std::thread::scope(|s| {
            for chunk in acc.chunks_mut(per) {
                s.spawn(move || {
                    chunk
                        .iter_mut()
                        .for_each(|v| *v = (*v as f64 / norm) as f32)
                });
            }
        });
        Arr2 {
            h: oh,
            w: ow,
            d: acc,
        }
    }
}

// ======================================================================================= scipy spline

const SPLINE_POLE: f64 = -0.267_949_192_431_122_706_472_553_658_494_127_633; // sqrt(3) - 2

/// scipy `apply_filter` for one 1-D line, order 3, mode 'mirror' (gain, mirror causal init, causal and
/// anticausal recursions), same operation order as `ni_splines.c`.
fn spline_line_mirror(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = SPLINE_POLE;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= gain;
    }
    let z_n_1 = z.powf((n - 1) as f64);
    c[0] += z_n_1 * c[n - 1];
    let mut z_i = z;
    for i in 1..n - 1 {
        c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] /= 1.0 - z_n_1 * z_n_1;
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// `ndi.spline_filter(Z, order=3)` (mode 'mirror' prefilter, rows then columns... scipy filters axis 0
/// first, then axis 1), float64 output.
pub fn spline_filter3(z: &Arr2<f32>) -> Arr2<f64> {
    let (h, w) = (z.h, z.w);
    let mut d: Vec<f64> = z.d.iter().map(|&v| v as f64).collect();
    if h == 0 || w == 0 {
        return Arr2 { h, w, d };
    }
    // axis 0 (down the columns): process column blocks to keep the gather cache friendly
    let nt = workers().min(w);
    {
        // transpose-free: every thread owns a disjoint set of columns, accessed through a raw split
        let per = w.div_ceil(nt);
        let base = d.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for t in 0..nt {
                let c_lo = t * per;
                let c_hi = ((t + 1) * per).min(w);
                if c_lo >= c_hi {
                    continue;
                }
                s.spawn(move || {
                    let p = base as *mut f64;
                    let mut line = vec![0f64; h];
                    for col in c_lo..c_hi {
                        // SAFETY: threads touch disjoint columns of the h*w buffer, which outlives the scope.
                        unsafe {
                            for r in 0..h {
                                line[r] = *p.add(r * w + col);
                            }
                            spline_line_mirror(&mut line);
                            for r in 0..h {
                                *p.add(r * w + col) = line[r];
                            }
                        }
                    }
                });
            }
        });
    }
    // axis 1 (along rows)
    let nt = workers().min(h);
    let per = h.div_ceil(nt);
    std::thread::scope(|s| {
        for chunk in d.chunks_mut(per * w) {
            s.spawn(move || {
                for row in chunk.chunks_mut(w) {
                    spline_line_mirror(row);
                }
            });
        }
    });
    Arr2 { h, w, d }
}

#[inline]
fn near(i: i64, len: usize) -> usize {
    // scipy map_coordinate(idx, len, NI_EXTEND_NEAREST)
    i.clamp(0, len as i64 - 1) as usize
}

/// scipy `get_spline_interpolation_weights`, order 3 (`x` is the unclamped coordinate).
#[inline]
fn cubic_weights(x: f64) -> [f64; 4] {
    let x = x - x.floor();
    let y = x;
    let z = 1.0 - x;
    let w1 = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    let w2 = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    let w0 = z * z * z / 6.0;
    let w3 = 1.0 - w0 - w1 - w2;
    [w0, w1, w2, w3]
}

/// `ndi.map_coordinates(zf, [[y], [x]], order=3, mode='nearest', prefilter=False)`; `zf` must already be
/// spline filtered ([`spline_filter3`]). `(y, x)` are (row, col) in grid cells. Like scipy, the
/// coordinate is NOT clamped for the weights (it only saturates through the clamped tap indices).
pub fn spline_sample_nearest(zf: &Arr2<f64>, y: f64, x: f64) -> f64 {
    let wy = cubic_weights(y);
    let wx = cubic_weights(x);
    let sy = y.floor() as i64 - 1;
    let sx = x.floor() as i64 - 1;
    let cols = [
        near(sx, zf.w),
        near(sx + 1, zf.w),
        near(sx + 2, zf.w),
        near(sx + 3, zf.w),
    ];
    let mut t = 0.0f64;
    for (j, wyj) in wy.iter().enumerate() {
        let row = zf.row(near(sy + j as i64, zf.h));
        for (i, &c) in cols.iter().enumerate() {
            t += row[c] * wyj * wx[i];
        }
    }
    t
}

/// `ndi.map_coordinates(g, [[y], [x]], order=1, mode='nearest')`, float32 result.
pub fn bilinear_nearest(g: &Arr2<f32>, y: f64, x: f64) -> f32 {
    let wy0 = 1.0 - (y - y.floor());
    let wx0 = 1.0 - (x - x.floor());
    let wy = [wy0, 1.0 - wy0];
    let wx = [wx0, 1.0 - wx0];
    let sy = y.floor() as i64;
    let sx = x.floor() as i64;
    let cols = [near(sx, g.w), near(sx + 1, g.w)];
    let mut t = 0.0f64;
    for (j, wyj) in wy.iter().enumerate() {
        let row = g.row(near(sy + j as i64, g.h));
        for (i, &c) in cols.iter().enumerate() {
            t += row[c] as f64 * wyj * wx[i];
        }
    }
    t as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_seed0_first_raw_matches_numpy() {
        // numpy: np.random.default_rng(0).bit_generator.random_raw() == 0x..; first standard_normal 0.1257302210933933
        let mut r = Pcg64::new(0);
        let v = r.standard_normal();
        assert!((v - 0.125_730_221_093_393_3).abs() < 1e-15, "{v}");
    }

    #[test]
    fn permutation_is_a_permutation() {
        let mut r = Pcg64::new(5);
        let mut p = r.permutation(1000);
        p.sort_unstable();
        assert!(p.iter().enumerate().all(|(i, &v)| i == v));
    }
}
