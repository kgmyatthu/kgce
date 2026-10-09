//! scipy.ndimage / numpy parity primitives for the native fixed-theatre painters.
//!
//! Everything here reproduces a specific scipy/numpy call used by the Python painters
//! (`tools/terrain_paint/*.py`); each function documents the call it replaces and every place where the
//! result is not bit-identical.
//!
//! Edge-mode semantics (identical to scipy.ndimage):
//! * `Mode::Reflect`  — "half-sample symmetric" `d c b a | a b c d | d c b a` (scipy default for the filters),
//!   repeated periodically (period 2n) when a kernel reaches further than the array is long;
//! * `Mode::Nearest`  — edge pixel repeated;
//! * `Mode::Constant` — cval = 0.
//!
//! Binary morphology uses `border_value = 0` everywhere (pixels outside the array count as False), exactly like
//! the default of `ndi.binary_*` and like the explicit `border_value=0` of `binary_closing` in the painters.
//! Float filters accumulate in `f64` (as scipy does) and store `f32` between the two axes (scipy stores the
//! intermediate axis result in the float32 output array); axis 0 is filtered first.
//!
//! Distance transform: distances are exact Euclidean (integer squared distances, `sqrt` in f64 then cast to f32,
//! the same as `distance_transform_edt(...).astype(np.float32)`). Nearest-feature *indices* are exact too, but
//! when several feature pixels are equally near the choice may differ from scipy's (see `edt_with_indices`).

use super::grid::Arr2;
use std::thread;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Reflect,
    Nearest,
    Constant,
}

// ---------------------------------------------------------------------------------------------------------------
// threading helpers
// ---------------------------------------------------------------------------------------------------------------

fn nthreads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16)
}

/// Split `data` (rows of `row_len` elements) into contiguous row slabs and run `f(first_row, slab)` on each in a
/// scoped thread. `work` is a rough operation count used to stay single-threaded for tiny inputs.
fn par_chunks<T: Send>(
    data: &mut [T],
    row_len: usize,
    nrows: usize,
    work: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    if data.is_empty() || row_len == 0 || nrows == 0 {
        return;
    }
    let nt = if work < (1 << 17) {
        1
    } else {
        nthreads().min(nrows)
    };
    if nt <= 1 {
        f(0, data);
        return;
    }
    let per = nrows.div_ceil(nt);
    let f = &f;
    thread::scope(|s| {
        for (i, ch) in data.chunks_mut(per * row_len).enumerate() {
            s.spawn(move || f(i * per, ch));
        }
    });
}

/// Raw pointer shared between scoped threads that write provably disjoint elements.
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline]
    fn ptr(&self) -> *mut T {
        self.0
    }
}

#[inline]
fn map_idx(i: isize, n: usize, mode: Mode) -> Option<usize> {
    let ni = n as isize;
    if i >= 0 && i < ni {
        return Some(i as usize);
    }
    match mode {
        Mode::Nearest => Some(if i < 0 { 0 } else { n - 1 }),
        Mode::Reflect => {
            let p = 2 * ni;
            let m = i.rem_euclid(p);
            Some(if m >= ni {
                (p - 1 - m) as usize
            } else {
                m as usize
            })
        }
        Mode::Constant => None,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Euclidean distance transform (Meijster et al. exact algorithm)
// ---------------------------------------------------------------------------------------------------------------

const NONE: u32 = u32::MAX;

/// `ndi.distance_transform_edt(mask)`: Euclidean distance of every True pixel to the nearest False pixel
/// (False pixels get 0). A mask without any False pixel yields `1.0e9` everywhere.
pub fn edt(mask: &Arr2<bool>) -> Arr2<f32> {
    edt_core(mask, false).0
}

/// `ndi.distance_transform_edt(mask, return_indices=True)`: also the flat index (`row*w+col`) of the nearest False
/// pixel (the pixel itself for False pixels). Distances are exact; where several False pixels are exactly equally
/// near, the one reported is chosen deterministically (smallest source row in the column pass, smallest column in
/// the row pass) and may differ from scipy's Maurer-algorithm choice. Without any False pixel the indices are
/// `u32::MAX`.
pub fn edt_with_indices(mask: &Arr2<bool>) -> (Arr2<f32>, Arr2<u32>) {
    let (d, i) = edt_core(mask, true);
    (d, i.expect("indices requested"))
}

fn edt_core(mask: &Arr2<bool>, keep_idx: bool) -> (Arr2<f32>, Option<Arr2<u32>>) {
    let (h, w) = (mask.h, mask.w);
    assert!(
        (h as u64) * (w as u64) < u32::MAX as u64,
        "edt: image too large for u32 indices"
    );
    let mut dist = vec![0f32; h * w];
    let mut idx = vec![NONE; h * w];
    if h == 0 || w == 0 {
        return (
            Arr2::from_vec(h, w, dist),
            keep_idx.then(|| Arr2::from_vec(h, w, idx)),
        );
    }
    if !mask.d.contains(&false) {
        dist.fill(1.0e9);
        return (
            Arr2::from_vec(h, w, dist),
            keep_idx.then(|| Arr2::from_vec(h, w, idx)),
        );
    }
    // pass 1: for every pixel the nearest False row inside its own column (NONE when the column has none)
    {
        let nt = if h * w < (1 << 18) { 1 } else { nthreads() };
        let cw = w.div_ceil(nt).max(32);
        let ptr = Shared(idx.as_mut_ptr());
        let m = &mask.d;
        thread::scope(|s| {
            for c0 in (0..w).step_by(cw) {
                let c1 = (c0 + cw).min(w);
                let p = &ptr;
                s.spawn(move || {
                    let base = p.ptr();
                    // SAFETY: every thread only touches columns c0..c1 of `idx`, ranges are disjoint.
                    unsafe {
                        for r in 0..h {
                            for c in c0..c1 {
                                let i = r * w + c;
                                *base.add(i) = if !m[i] {
                                    r as u32
                                } else if r > 0 {
                                    *base.add(i - w)
                                } else {
                                    NONE
                                };
                            }
                        }
                        let mut nxt = vec![NONE; c1 - c0];
                        for r in (0..h).rev() {
                            for c in c0..c1 {
                                let i = r * w + c;
                                let n = &mut nxt[c - c0];
                                if !m[i] {
                                    *n = r as u32;
                                }
                                let up = *base.add(i);
                                let dn = *n;
                                let best = if up == NONE {
                                    dn
                                } else if dn == NONE {
                                    up
                                } else if (r as u32 - up) <= (dn - r as u32) {
                                    up
                                } else {
                                    dn
                                };
                                *base.add(i) = best;
                            }
                        }
                    }
                });
            }
        });
    }
    // pass 2: per row lower envelope of parabolas over the columns
    let per = h.div_ceil(if h * w < (1 << 18) {
        1
    } else {
        nthreads().min(h)
    });
    thread::scope(|s| {
        for (ci, (ich, dch)) in idx
            .chunks_mut(per * w)
            .zip(dist.chunks_mut(per * w))
            .enumerate()
        {
            s.spawn(move || {
                let mut nr = vec![0u32; w];
                let mut g2 = vec![0i64; w];
                let mut sv = vec![0usize; w];
                let mut tv = vec![0i64; w];
                for (li, (irow, drow)) in ich.chunks_mut(w).zip(dch.chunks_mut(w)).enumerate() {
                    let r = (ci * per + li) as i64;
                    nr.copy_from_slice(irow);
                    for c in 0..w {
                        g2[c] = if nr[c] == NONE {
                            -1
                        } else {
                            let d = r - nr[c] as i64;
                            d * d
                        };
                    }
                    let wi = w as i64;
                    let f = |x: i64, i: usize| -> i64 {
                        let dx = x - i as i64;
                        dx * dx + g2[i]
                    };
                    let mut q: isize = -1;
                    for u in 0..w {
                        if g2[u] < 0 {
                            continue;
                        }
                        if q < 0 {
                            q = 0;
                            sv[0] = u;
                            tv[0] = 0;
                            continue;
                        }
                        while q >= 0 && f(tv[q as usize], sv[q as usize]) > f(tv[q as usize], u) {
                            q -= 1;
                        }
                        if q < 0 {
                            q = 0;
                            sv[0] = u;
                            tv[0] = 0;
                        } else {
                            let i = sv[q as usize] as i64;
                            let uu = u as i64;
                            let num = uu * uu - i * i + g2[u] - g2[i as usize];
                            let wv = 1 + num.div_euclid(2 * (uu - i));
                            if wv < wi {
                                q += 1;
                                sv[q as usize] = u;
                                tv[q as usize] = wv;
                            }
                        }
                    }
                    if q < 0 {
                        drow.fill(1.0e9);
                        continue;
                    }
                    for u in (0..w).rev() {
                        let site = sv[q as usize];
                        let dx = u as i64 - site as i64;
                        let d2 = dx * dx + g2[site];
                        drow[u] = (d2 as f64).sqrt() as f32;
                        irow[u] = nr[site] * w as u32 + site as u32;
                        if u as i64 == tv[q as usize] {
                            q -= 1;
                            if q < 0 {
                                break;
                            }
                        }
                    }
                }
            });
        }
    });
    (
        Arr2::from_vec(h, w, dist),
        keep_idx.then(|| Arr2::from_vec(h, w, idx)),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// separable float filters
// ---------------------------------------------------------------------------------------------------------------

/// Half kernel of `scipy.ndimage.gaussian_filter1d` (truncate = 4): `k[0]` centre, `k[j]` weight at offset +-j.
fn gaussian_half_kernel(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as usize;
    let s2 = sigma * sigma;
    let phi: Vec<f64> = (-(radius as i64)..=radius as i64)
        .map(|x| (-0.5 / s2 * ((x * x) as f64)).exp())
        .collect();
    let sum: f64 = phi.iter().sum();
    (0..=radius).map(|j| phi[radius + j] / sum).collect()
}

/// Symmetric correlation along axis 1 (within rows) of interleaved `nch`-channel rows.
fn sym_axis1(src: &[f32], dst: &mut [f32], h: usize, w: usize, nch: usize, k: &[f64], mode: Mode) {
    let r = k.len() - 1;
    par_chunks(dst, w * nch, h, h * w * nch * (r + 1), |row0, chunk| {
        let mut ext = vec![0f64; w + 2 * r];
        let mut acc = vec![0f64; w];
        for (i, drow) in chunk.chunks_mut(w * nch).enumerate() {
            let srow = &src[(row0 + i) * w * nch..][..w * nch];
            for ch in 0..nch {
                for (j, e) in ext.iter_mut().enumerate() {
                    *e = match map_idx(j as isize - r as isize, w, mode) {
                        Some(q) => srow[q * nch + ch] as f64,
                        None => 0.0,
                    };
                }
                for c in 0..w {
                    acc[c] = ext[c + r] * k[0];
                }
                for (j, &kj) in k.iter().enumerate().skip(1) {
                    let lo = &ext[r - j..r - j + w];
                    let hi = &ext[r + j..r + j + w];
                    for c in 0..w {
                        acc[c] += (lo[c] + hi[c]) * kj;
                    }
                }
                for c in 0..w {
                    drow[c * nch + ch] = acc[c] as f32;
                }
            }
        }
    });
}

/// Symmetric correlation along axis 0 (across rows).
fn sym_axis0(src: &[f32], dst: &mut [f32], h: usize, w: usize, nch: usize, k: &[f64], mode: Mode) {
    let rw = w * nch;
    let r = k.len() - 1;
    par_chunks(dst, rw, h, h * rw * (r + 1), |row0, chunk| {
        let mut acc = vec![0f64; rw];
        for (i, drow) in chunk.chunks_mut(rw).enumerate() {
            let rr = row0 + i;
            let c0 = &src[rr * rw..][..rw];
            for x in 0..rw {
                acc[x] = c0[x] as f64 * k[0];
            }
            for (j, &kj) in k.iter().enumerate().skip(1) {
                let a = map_idx(rr as isize - j as isize, h, mode);
                let b = map_idx(rr as isize + j as isize, h, mode);
                match (a, b) {
                    (Some(a), Some(b)) => {
                        let sa = &src[a * rw..][..rw];
                        let sb = &src[b * rw..][..rw];
                        for x in 0..rw {
                            acc[x] += (sa[x] as f64 + sb[x] as f64) * kj;
                        }
                    }
                    (Some(a), None) | (None, Some(a)) => {
                        let sa = &src[a * rw..][..rw];
                        for x in 0..rw {
                            acc[x] += sa[x] as f64 * kj;
                        }
                    }
                    (None, None) => {}
                }
            }
            for x in 0..rw {
                drow[x] = acc[x] as f32;
            }
        }
    });
}

/// Uniform (running-sum) filter along axis 1; window `[c - size/2, c + size - size/2 - 1]`.
fn uni_axis1(
    src: &[f32],
    dst: &mut [f32],
    h: usize,
    w: usize,
    nch: usize,
    size: usize,
    mode: Mode,
) {
    let lo = (size / 2) as isize;
    let inv = size as f64;
    par_chunks(dst, w * nch, h, h * w * nch * 2, |row0, chunk| {
        let mut ext = vec![0f64; w + size - 1];
        for (i, drow) in chunk.chunks_mut(w * nch).enumerate() {
            let srow = &src[(row0 + i) * w * nch..][..w * nch];
            for ch in 0..nch {
                for (j, e) in ext.iter_mut().enumerate() {
                    *e = match map_idx(j as isize - lo, w, mode) {
                        Some(q) => srow[q * nch + ch] as f64,
                        None => 0.0,
                    };
                }
                let mut tmp = 0.0f64;
                for e in &ext[..size] {
                    tmp += *e;
                }
                tmp /= inv;
                drow[ch] = tmp as f32;
                for c in 1..w {
                    tmp += (ext[c + size - 1] - ext[c - 1]) / inv;
                    drow[c * nch + ch] = tmp as f32;
                }
            }
        }
    });
}

/// Uniform (running-sum) filter along axis 0.
fn uni_axis0(
    src: &[f32],
    dst: &mut [f32],
    h: usize,
    w: usize,
    nch: usize,
    size: usize,
    mode: Mode,
) {
    let rw = w * nch;
    let lo = (size / 2) as isize;
    let inv = size as f64;
    par_chunks(dst, rw, h, h * rw * 2, |row0, chunk| {
        let nr = chunk.len() / rw;
        let mut acc = vec![0f64; rw];
        let row_at =
            |p: isize| -> Option<&[f32]> { map_idx(p, h, mode).map(|q| &src[q * rw..][..rw]) };
        for j in 0..size {
            if let Some(s) = row_at(row0 as isize - lo + j as isize) {
                for x in 0..rw {
                    acc[x] += s[x] as f64;
                }
            }
        }
        for a in acc.iter_mut() {
            *a /= inv;
        }
        for i in 0..nr {
            let drow = &mut chunk[i * rw..][..rw];
            for x in 0..rw {
                drow[x] = acc[x] as f32;
            }
            if i + 1 < nr {
                let newp = (row0 + i + 1) as isize - lo + size as isize - 1;
                let oldp = (row0 + i) as isize - lo;
                match (row_at(newp), row_at(oldp)) {
                    (Some(n), Some(o)) => {
                        for x in 0..rw {
                            acc[x] += (n[x] as f64 - o[x] as f64) / inv;
                        }
                    }
                    (Some(n), None) => {
                        for x in 0..rw {
                            acc[x] += (n[x] as f64) / inv;
                        }
                    }
                    (None, Some(o)) => {
                        for x in 0..rw {
                            acc[x] += (-(o[x] as f64)) / inv;
                        }
                    }
                    (None, None) => {}
                }
            }
        }
    });
}

fn gaussian2d(src: &[f32], h: usize, w: usize, nch: usize, sigma: f64) -> Vec<f32> {
    if !(sigma > 1e-15) || h * w == 0 {
        return src.to_vec();
    }
    let k = gaussian_half_kernel(sigma);
    let mut t = vec![0f32; src.len()];
    sym_axis0(src, &mut t, h, w, nch, &k, Mode::Reflect);
    let mut o = vec![0f32; src.len()];
    sym_axis1(&t, &mut o, h, w, nch, &k, Mode::Reflect);
    o
}

/// `ndi.gaussian_filter(a, sigma)` (mode 'reflect', truncate 4, float32 in/out, float64 accumulation).
pub fn gaussian_filter(a: &Arr2<f32>, sigma: f64) -> Arr2<f32> {
    Arr2::from_vec(a.h, a.w, gaussian2d(&a.d, a.h, a.w, 1, sigma))
}

/// `ndi.gaussian_filter(a, (sigma, sigma, 0))` of an (h, w, 3) float32 image: spatial smoothing only.
pub fn gaussian_filter_rgb(a: &Arr2<[f32; 3]>, sigma: f64) -> Arr2<[f32; 3]> {
    let flat = flatten3(&a.d);
    unflatten3(a.h, a.w, &gaussian2d(&flat, a.h, a.w, 3, sigma))
}

/// `ndi.gaussian_filter1d(v, sigma, axis=0, mode=mode)` of an (n, 3) float32 array.
pub fn gaussian_filter1d_lut(v: &[[f32; 3]], sigma: f64, mode: Mode) -> Vec<[f32; 3]> {
    if !(sigma > 1e-15) || v.is_empty() {
        return v.to_vec();
    }
    let k = gaussian_half_kernel(sigma);
    let flat = flatten3(v);
    let mut o = vec![0f32; flat.len()];
    sym_axis0(&flat, &mut o, v.len(), 1, 3, &k, mode);
    o.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn uniform2d(
    src: &[f32],
    h: usize,
    w: usize,
    nch: usize,
    sr: usize,
    sc: usize,
    mode: Mode,
) -> Vec<f32> {
    assert!(sr >= 1 && sc >= 1, "uniform filter sizes must be >= 1");
    if h * w == 0 || (sr == 1 && sc == 1) {
        return src.to_vec();
    }
    let mut cur: Option<Vec<f32>> = None;
    if sr > 1 {
        let mut o = vec![0f32; src.len()];
        uni_axis0(src, &mut o, h, w, nch, sr, mode);
        cur = Some(o);
    }
    if sc > 1 {
        let mut o = vec![0f32; src.len()];
        uni_axis1(cur.as_deref().unwrap_or(src), &mut o, h, w, nch, sc, mode);
        cur = Some(o);
    }
    cur.expect("at least one axis filtered")
}

/// `ndi.uniform_filter(a, (size_r, size_c), mode=mode)`; window `[i - size/2, i + size - size/2 - 1]`.
pub fn uniform_filter(a: &Arr2<f32>, size_r: usize, size_c: usize, mode: Mode) -> Arr2<f32> {
    Arr2::from_vec(a.h, a.w, uniform2d(&a.d, a.h, a.w, 1, size_r, size_c, mode))
}

/// `ndi.uniform_filter(a, (size_r, size_c, 1), mode=mode)` of an (h, w, 3) float32 image.
pub fn uniform_filter_rgb(
    a: &Arr2<[f32; 3]>,
    size_r: usize,
    size_c: usize,
    mode: Mode,
) -> Arr2<[f32; 3]> {
    let flat = flatten3(&a.d);
    unflatten3(
        a.h,
        a.w,
        &uniform2d(&flat, a.h, a.w, 3, size_r, size_c, mode),
    )
}

/// `ndi.uniform_filter1d(v, size, axis=0, mode=mode)` of an (n, 3) float32 array.
pub fn uniform_filter1d_lut(v: &[[f32; 3]], size: usize, mode: Mode) -> Vec<[f32; 3]> {
    assert!(size >= 1, "uniform filter size must be >= 1");
    if v.is_empty() || size == 1 {
        return v.to_vec();
    }
    let flat = flatten3(v);
    let mut o = vec![0f32; flat.len()];
    uni_axis0(&flat, &mut o, v.len(), 1, 3, size, mode);
    o.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn flatten3(v: &[[f32; 3]]) -> Vec<f32> {
    let mut o = Vec::with_capacity(v.len() * 3);
    for p in v {
        o.extend_from_slice(p);
    }
    o
}

fn unflatten3(h: usize, w: usize, f: &[f32]) -> Arr2<[f32; 3]> {
    Arr2::from_vec(
        h,
        w,
        f.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// binary morphology (border value 0)
// ---------------------------------------------------------------------------------------------------------------

/// Horizontal box pass over every row; `dilate` selects OR (else AND) over `[c-a, c+a]`, outside = False.
fn box_h(src: &[bool], dst: &mut [bool], h: usize, w: usize, k: usize, dilate: bool) {
    let a = k / 2;
    par_chunks(dst, w, h, h * w * 2, |row0, chunk| {
        for (i, drow) in chunk.chunks_mut(w).enumerate() {
            let srow = &src[(row0 + i) * w..][..w];
            let mut cnt = 0usize;
            if dilate {
                cnt = a; // `a` virtual False cells left of the row
                for (x, &v) in srow.iter().enumerate() {
                    cnt = if v { 0 } else { cnt + 1 };
                    if x >= a {
                        drow[x - a] = cnt < k;
                    }
                }
                for x in w..w + a {
                    cnt += 1;
                    if x >= a {
                        drow[x - a] = cnt < k;
                    }
                }
            } else {
                for (x, &v) in srow.iter().enumerate() {
                    cnt = if v { cnt + 1 } else { 0 };
                    if x >= a {
                        drow[x - a] = cnt >= k;
                    }
                }
                for c in w.saturating_sub(a)..w {
                    drow[c] = false;
                }
            }
        }
    });
}

/// Vertical box pass; same semantics as `box_h` along axis 0.
fn box_v(src: &[bool], dst: &mut [bool], h: usize, w: usize, k: usize, dilate: bool) {
    let a = k / 2;
    let k32 = k as u32;
    par_chunks(dst, w, h, h * w * 2, |row0, chunk| {
        let nr = chunk.len() / w;
        let mut cnt = vec![0u32; w];
        let start = row0 as isize - a as isize;
        let end = (row0 + nr + a) as isize;
        for s in start..end {
            let real = s >= 0 && (s as usize) < h;
            if real {
                let row = &src[s as usize * w..][..w];
                if dilate {
                    for x in 0..w {
                        cnt[x] = if row[x] { 0 } else { cnt[x] + 1 };
                    }
                } else {
                    for x in 0..w {
                        cnt[x] = if row[x] { cnt[x] + 1 } else { 0 };
                    }
                }
            } else if dilate {
                for c in cnt.iter_mut() {
                    *c += 1;
                }
            } else {
                cnt.fill(0);
            }
            let out = s - a as isize;
            if out >= row0 as isize {
                let drow = &mut chunk[(out as usize - row0) * w..][..w];
                if dilate {
                    for x in 0..w {
                        drow[x] = cnt[x] < k32;
                    }
                } else {
                    for x in 0..w {
                        drow[x] = cnt[x] >= k32;
                    }
                }
            }
        }
    });
}

fn box_morph(m: &Arr2<bool>, k: usize, dilate: bool) -> Arr2<bool> {
    assert!(k % 2 == 1, "box structure size must be odd (got {k})");
    if k == 1 || m.d.is_empty() {
        return m.clone();
    }
    let (h, w) = (m.h, m.w);
    let mut tmp = vec![false; h * w];
    box_h(&m.d, &mut tmp, h, w, k, dilate);
    let mut out = vec![false; h * w];
    box_v(&tmp, &mut out, h, w, k, dilate);
    Arr2::from_vec(h, w, out)
}

/// `ndi.binary_dilation(m, np.ones((k, k)))` (border 0). `k` must be odd.
pub fn dilate_box(m: &Arr2<bool>, k: usize) -> Arr2<bool> {
    box_morph(m, k, true)
}

/// `ndi.binary_erosion(m, np.ones((k, k)), border_value=0)`. `k` must be odd.
pub fn erode_box(m: &Arr2<bool>, k: usize) -> Arr2<bool> {
    box_morph(m, k, false)
}

/// `ndi.binary_opening(m, structure=np.ones((k, k)))`: erosion then dilation, border 0.
pub fn opening_box(m: &Arr2<bool>, k: usize) -> Arr2<bool> {
    dilate_box(&erode_box(m, k), k)
}

/// `ndi.binary_closing(m, structure=np.ones((k, k)), border_value=0)`: dilation then erosion; the erosion sees the
/// outside as False, so the result is eroded within `k/2` pixels of the array border (scipy semantics).
pub fn closing_box(m: &Arr2<bool>, k: usize) -> Arr2<bool> {
    erode_box(&dilate_box(m, k), k)
}

fn cross_once(m: &Arr2<bool>, dilate: bool) -> Arr2<bool> {
    let (h, w) = (m.h, m.w);
    let mut out = vec![false; h * w];
    par_chunks(&mut out, w, h, h * w * 2, |row0, chunk| {
        for (i, drow) in chunk.chunks_mut(w).enumerate() {
            let r = row0 + i;
            let cur = m.row(r);
            let up = if r > 0 { Some(m.row(r - 1)) } else { None };
            let dn = if r + 1 < h { Some(m.row(r + 1)) } else { None };
            if dilate {
                for c in 0..w {
                    let mut v = cur[c];
                    if c > 0 {
                        v |= cur[c - 1];
                    }
                    if c + 1 < w {
                        v |= cur[c + 1];
                    }
                    drow[c] = v;
                }
                if let Some(u) = up {
                    for c in 0..w {
                        drow[c] |= u[c];
                    }
                }
                if let Some(d) = dn {
                    for c in 0..w {
                        drow[c] |= d[c];
                    }
                }
            } else if let (Some(u), Some(d)) = (up, dn) {
                if w >= 3 {
                    for c in 1..w - 1 {
                        drow[c] = cur[c] & cur[c - 1] & cur[c + 1] & u[c] & d[c];
                    }
                }
            }
        }
    });
    Arr2::from_vec(h, w, out)
}

/// `ndi.binary_dilation(m)`: 4-connected cross structure, one iteration, border 0.
pub fn dilate(m: &Arr2<bool>) -> Arr2<bool> {
    cross_once(m, true)
}

/// `ndi.binary_erosion(m, iterations=iterations)`: cross structure, `border_value=0`. `iterations == 0` repeats
/// until nothing changes (scipy semantics).
pub fn erode(m: &Arr2<bool>, iterations: usize) -> Arr2<bool> {
    if iterations == 0 {
        let mut cur = m.clone();
        loop {
            let nxt = cross_once(&cur, false);
            if nxt == cur {
                return cur;
            }
            cur = nxt;
        }
    }
    let mut cur = cross_once(m, false);
    for _ in 1..iterations {
        cur = cross_once(&cur, false);
    }
    cur
}

/// `for _ in range(iterations): v |= ndi.binary_dilation(v) & allowed` (cross structure), computed with a frontier:
/// the first iteration scans the whole array in parallel, later ones only touch the pixels added by the previous one.
/// Result is identical to the repeated full-array formulation (including `v` pixels outside `allowed`).
pub fn grow(v: &mut Arr2<bool>, allowed: &Arr2<bool>, iterations: usize) {
    assert!(v.h == allowed.h && v.w == allowed.w, "grow: shape mismatch");
    let (h, w) = (v.h, v.w);
    if iterations == 0 || h * w == 0 {
        return;
    }
    assert!(
        (h as u64) * (w as u64) < u32::MAX as u64,
        "grow: image too large for u32 indices"
    );
    // iteration 1: candidates from the whole array (v is only read here)
    let nt = if h * w < (1 << 18) {
        1
    } else {
        nthreads().min(h)
    };
    let per = h.div_ceil(nt);
    let lists: Vec<Vec<u32>> = {
        let vd = &v.d;
        let al = &allowed.d;
        thread::scope(|s| {
            let hs: Vec<_> = (0..nt)
                .map(|t| {
                    let (r0, r1) = (t * per, ((t + 1) * per).min(h));
                    s.spawn(move || {
                        let mut out = Vec::new();
                        for r in r0..r1 {
                            for c in 0..w {
                                let i = r * w + c;
                                if vd[i] || !al[i] {
                                    continue;
                                }
                                let hit = (r > 0 && vd[i - w])
                                    || (r + 1 < h && vd[i + w])
                                    || (c > 0 && vd[i - 1])
                                    || (c + 1 < w && vd[i + 1]);
                                if hit {
                                    out.push(i as u32);
                                }
                            }
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter()
                .map(|j| j.join().expect("grow worker"))
                .collect()
        })
    };
    let mut frontier: Vec<u32> = Vec::new();
    for l in lists {
        for i in l {
            if !v.d[i as usize] {
                v.d[i as usize] = true;
                frontier.push(i);
            }
        }
    }
    for _ in 1..iterations {
        if frontier.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for &p in &frontier {
            let p = p as usize;
            let (r, c) = (p / w, p % w);
            let mut try_add = |q: usize| {
                if !v.d[q] && allowed.d[q] {
                    v.d[q] = true;
                    next.push(q as u32);
                }
            };
            if r > 0 {
                try_add(p - w);
            }
            if r + 1 < h {
                try_add(p + w);
            }
            if c > 0 {
                try_add(p - 1);
            }
            if c + 1 < w {
                try_add(p + 1);
            }
        }
        frontier = next;
    }
}

/// `ndi.binary_fill_holes(m)`: background (False) pixels not 4-connected to the array border become True.
pub fn fill_holes(m: &Arr2<bool>) -> Arr2<bool> {
    let (h, w) = (m.h, m.w);
    if h == 0 || w == 0 {
        return m.clone();
    }
    let mut bg = vec![false; h * w];
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for c in 0..w {
        stack.push((0, c as u32));
        stack.push((h as u32 - 1, c as u32));
    }
    for r in 0..h {
        stack.push((r as u32, 0));
        stack.push((r as u32, w as u32 - 1));
    }
    while let Some((r, c)) = stack.pop() {
        let (r, c) = (r as usize, c as usize);
        let i = r * w + c;
        if m.d[i] || bg[i] {
            continue;
        }
        let row = r * w;
        let mut l = c;
        while l > 0 && !m.d[row + l - 1] {
            l -= 1;
        }
        let mut rr = c;
        while rr + 1 < w && !m.d[row + rr + 1] {
            rr += 1;
        }
        bg[row + l..=row + rr].fill(true);
        for nr in [r.wrapping_sub(1), r + 1] {
            if nr >= h {
                continue;
            }
            let nrow = nr * w;
            let mut in_run = false;
            for x in l..=rr {
                let ok = !m.d[nrow + x] && !bg[nrow + x];
                if ok && !in_run {
                    stack.push((nr as u32, x as u32));
                }
                in_run = ok;
            }
        }
    }
    Arr2::from_vec(h, w, bg.into_iter().map(|b| !b).collect())
}

// ---------------------------------------------------------------------------------------------------------------
// connected components
// ---------------------------------------------------------------------------------------------------------------

fn uf_find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        let g = p[p[x as usize] as usize];
        p[x as usize] = g;
        x = g;
    }
    x
}

/// `ndi.label(m)` with the default 4-connected structure. Labels `1..=n` are numbered in raster order of each
/// component's first pixel (identical to scipy); 0 is background. Run-length union-find: memory is proportional to
/// the number of horizontal runs, not pixels.
pub fn label(m: &Arr2<bool>) -> (Arr2<i32>, usize) {
    let (h, w) = (m.h, m.w);
    let mut out = vec![0i32; h * w];
    if h * w == 0 {
        return (Arr2::from_vec(h, w, out), 0);
    }
    assert!(
        (h as u64) * (w as u64) < i32::MAX as u64,
        "label: image too large"
    );
    let mut parent: Vec<u32> = Vec::new();
    let mut prev: Vec<(u32, u32, u32)> = Vec::new(); // (start, end exclusive, run id)
    let mut cur: Vec<(u32, u32, u32)> = Vec::new();
    for r in 0..h {
        cur.clear();
        let row = m.row(r);
        let orow = &mut out[r * w..(r + 1) * w];
        let mut pj = 0usize;
        let mut c = 0usize;
        while c < w {
            if !row[c] {
                c += 1;
                continue;
            }
            let s = c;
            while c < w && row[c] {
                c += 1;
            }
            let e = c;
            let id = parent.len() as u32;
            parent.push(id);
            orow[s..e].fill(id as i32 + 1);
            while pj < prev.len() && (prev[pj].1 as usize) <= s {
                pj += 1;
            }
            let mut k = pj;
            while k < prev.len() && (prev[k].0 as usize) < e {
                let a = uf_find(&mut parent, id);
                let b = uf_find(&mut parent, prev[k].2);
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    parent[hi as usize] = lo;
                }
                k += 1;
            }
            cur.push((s as u32, e as u32, id));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let nruns = parent.len();
    let mut lut = vec![0i32; nruns];
    let mut n = 0i32;
    for id in 0..nruns {
        let root = uf_find(&mut parent, id as u32) as usize;
        if root == id {
            n += 1;
            lut[id] = n;
        } else {
            lut[id] = lut[root];
        }
    }
    drop(parent);
    par_chunks(&mut out, w, h, h * w, |_, chunk| {
        for v in chunk.iter_mut() {
            if *v != 0 {
                *v = lut[*v as usize - 1];
            }
        }
    });
    (Arr2::from_vec(h, w, out), n as usize)
}

// ---------------------------------------------------------------------------------------------------------------
// numpy helpers
// ---------------------------------------------------------------------------------------------------------------

/// `np.interp(x, xp, fp)`: piecewise-linear, clamped to `fp[0]` / `fp[last]` outside `xp`; `xp` ascending.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    assert!(
        !xp.is_empty() && xp.len() == fp.len(),
        "interp: xp/fp must be non-empty and equal length"
    );
    if x.is_nan() {
        return x;
    }
    let n = xp.len();
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    // j: largest index with xp[j] <= x, clamped so that j + 1 < n
    let j = xp.partition_point(|&v| v <= x).saturating_sub(1);
    if j >= n - 1 {
        return fp[n - 1];
    }
    if x == xp[j] {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if r.is_nan() && fp[j] == fp[j + 1] {
            r = fp[j];
        }
    }
    r
}

/// `np.median(v)`: mean of the two middle values for even lengths; NaN for empty input or if any value is NaN.
/// Reorders `v`.
pub fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 || v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let mid = n / 2;
    let (left, hi, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *hi;
    if n % 2 == 1 {
        hi
    } else {
        let lo = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo + hi) / 2.0
    }
}

/// `np.percentile(sorted, p)` (linear interpolation) on an ascending slice, `p` in 0..=100.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let virt = (n - 1) as f64 * (p / 100.0);
    let prev = virt.floor().max(0.0);
    let gamma = virt - prev;
    let pi = (prev as usize).min(n - 1);
    let ni = (pi + 1).min(n - 1);
    let (a, b) = (sorted[pi], sorted[ni]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// `np.gradient(a)` of a float32 2-D array: `(d/d row, d/d col)`, spacing 1; central differences inside, one-sided
/// first-order differences on the borders; float32 arithmetic like numpy. Each axis needs at least 2 samples.
pub fn gradient(a: &Arr2<f32>) -> (Arr2<f32>, Arr2<f32>) {
    let (h, w) = (a.h, a.w);
    assert!(
        h >= 2 && w >= 2,
        "gradient: need at least 2 samples along each axis"
    );
    let mut gr = vec![0f32; h * w];
    let mut gc = vec![0f32; h * w];
    par_chunks(&mut gr, w, h, h * w, |row0, chunk| {
        for (i, drow) in chunk.chunks_mut(w).enumerate() {
            let r = row0 + i;
            if r == 0 {
                for c in 0..w {
                    drow[c] = a.d[w + c] - a.d[c];
                }
            } else if r == h - 1 {
                for c in 0..w {
                    drow[c] = a.d[r * w + c] - a.d[(r - 1) * w + c];
                }
            } else {
                for c in 0..w {
                    drow[c] = (a.d[(r + 1) * w + c] - a.d[(r - 1) * w + c]) / 2.0;
                }
            }
        }
    });
    par_chunks(&mut gc, w, h, h * w, |row0, chunk| {
        for (i, drow) in chunk.chunks_mut(w).enumerate() {
            let row = a.row(row0 + i);
            drow[0] = row[1] - row[0];
            drow[w - 1] = row[w - 1] - row[w - 2];
            for c in 1..w - 1 {
                drow[c] = (row[c + 1] - row[c - 1]) / 2.0;
            }
        }
    });
    (Arr2::from_vec(h, w, gr), Arr2::from_vec(h, w, gc))
}

/// `np.digitize(x, bins)` (right=False) for ascending `bins`: number of bins `<= x` (NaN sorts last -> `bins.len()`).
pub fn digitize(x: f64, bins: &[f64]) -> usize {
    if x.is_nan() {
        return bins.len();
    }
    bins.partition_point(|&b| b <= x)
}
