//! Public-DEM sampler of the fixed-theatre painters (port of `geo.Dem` / `fixed_america.Dem`, `geo.z_to_h`, `fixed_america.h_to_z`).
//!
//! The elevation data are crops of NOAA NCEI ETOPO 2022 (60 arc-second surface elevation, doi:10.25921/fd45-gt74, public domain) stored under
//! `assets/policy/terrain/` in the lossless format written by `tools/terrain_paint/prepare_dem.py` (layout documented there; every float32
//! cell is bit-exact) and embedded with `include_bytes!`. Nothing game-derived is stored. `calibration.json` carries the authored world <->
//! lon/lat affines of theatres 1 and 3 and the 41-point pooled metres -> height-byte quantile map.
//!
//! Sampling is exactly the Python one: world (x, y) -> (lon, lat) through the theatre's affine inverse, -> DEM cell units
//! `f = (lon - lon_of_first_cell) / (1/60)` (+ jitter in cells), cubic B-spline (prefiltered once, `mode='nearest'`) -> `f32`.

use super::grid::Arr2;
use super::resample::{spline_filter3, spline_sample_nearest};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::LazyLock;

static GRID_INDIA: &[u8] = include_bytes!("../../../assets/policy/terrain/dem_india.bin");
static GRID_AMERICA: &[u8] = include_bytes!("../../../assets/policy/terrain/dem_america.bin");
static CALIBRATION_JSON: &[u8] = include_bytes!("../../../assets/policy/terrain/calibration.json");

const MAGIC: &[u8; 8] = b"KGCDEM02";
const RES: f64 = 1.0 / 60.0;
/// Lattice of the integer layer (`q = rint(16 v)`) and the number of layer-2 groups (24 bit-length classes x sign x two neighbour flags).
const LATTICE: f32 = 16.0;
const NBL: usize = 24;
const NGROUPS: usize = NBL * 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Theatre {
    India,
    America,
}

impl Theatre {
    fn key(self) -> &'static str {
        match self {
            Theatre::India => "3",
            Theatre::America => "1",
        }
    }
    fn grid(self) -> &'static [u8] {
        match self {
            Theatre::India => GRID_INDIA,
            Theatre::America => GRID_AMERICA,
        }
    }
}

#[derive(Deserialize)]
struct TheatreCal {
    inv_lon: [f64; 3],
    inv_lat: [f64; 3],
}

#[derive(Deserialize)]
struct QuantileMap {
    z_m: Vec<f64>,
    h: Vec<f64>,
}

#[derive(Deserialize)]
struct Calibration {
    theatres: std::collections::BTreeMap<String, TheatreCal>,
    pooled_quantile_map: QuantileMap,
}

static CALIBRATION: LazyLock<std::result::Result<Calibration, String>> = LazyLock::new(|| {
    let c: Calibration = serde_json::from_slice(CALIBRATION_JSON)
        .map_err(|e| format!("embedded calibration.json: {e}"))?;
    let q = &c.pooled_quantile_map;
    if q.z_m.len() < 2
        || q.z_m.len() != q.h.len()
        || q.z_m.windows(2).any(|w| !(w[0] < w[1]))
        || q.h.windows(2).any(|w| !(w[0] < w[1]))
    {
        return Err("embedded calibration.json: invalid pooled_quantile_map".into());
    }
    Ok(c)
});

fn calibration() -> Result<&'static Calibration> {
    CALIBRATION.as_ref().map_err(|e| anyhow::anyhow!("{e}"))
}

/// `np.interp(x, xp, fp)` (ascending `xp`, clamped ends).
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // last j with xp[j] <= x
    let j = xp.partition_point(|&v| v <= x) - 1;
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `geo.z_to_h`: `max(interp(z, qz, qh), 1.0)` (metres -> ETW height byte, not rounded).
pub fn z_to_h(z: f32) -> f32 {
    let q = &calibration()
        .expect("embedded terrain calibration")
        .pooled_quantile_map;
    np_interp(z as f64, &q.z_m, &q.h).max(1.0) as f32
}

/// `fixed_america.h_to_z`: inverse of the pooled quantile map (h 0..255 -> metres; 0 -> 0).
#[cfg(test)]
pub fn h_to_z(h: f32) -> f32 {
    static CURVE: LazyLock<(Vec<f64>, Vec<f64>)> = LazyLock::new(|| {
        let q = &calibration()
            .expect("embedded terrain calibration")
            .pooled_quantile_map;
        let mut hp = vec![0.0];
        hp.extend_from_slice(&q.h);
        let mut zp = vec![0.0];
        zp.extend_from_slice(&q.z_m);
        (hp, zp)
    });
    np_interp(h as f64, &CURVE.0, &CURVE.1) as f32
}

/// SHA-256 over all shipped terrain data (both grids + calibration): part of the cache recipe binding.
pub fn recipe_fingerprint() -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"kgc-terrain-recipe-v2");
    for part in [GRID_INDIA, GRID_AMERICA, CALIBRATION_JSON] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().into()
}

pub struct Dem {
    zf: Arr2<f64>,
    lon0: f64,
    lat0: f64,
    inv_lon: [f64; 3],
    inv_lat: [f64; 3],
}

impl Dem {
    /// Decodes the embedded grid of `t`, verifies its SHA-256 trailer and builds the spline-filtered grid once.
    pub fn load(t: Theatre) -> Result<Dem> {
        let cal = calibration()?;
        let tc = cal
            .theatres
            .get(t.key())
            .with_context(|| format!("calibration.json has no theatre {}", t.key()))?;
        let g =
            decode_grid(t.grid()).with_context(|| format!("decoding embedded DEM for {t:?}"))?;
        let lon0 = (g.ci0 as f64 + 0.5) * RES - 180.0;
        let lat0 = (g.rj0 as f64 + 0.5) * RES - 90.0;
        let zf = spline_filter3(&g.z);
        drop(g);
        Ok(Dem {
            zf,
            lon0,
            lat0,
            inv_lon: tc.inv_lon,
            inv_lat: tc.inv_lat,
        })
    }

    /// Elevation (m) at world `(x, y)`; `(jx, jy)` is the jitter in DEM cells added to the cell coordinates.
    pub fn sample(&self, x: f64, y: f64, jx: f64, jy: f64) -> f32 {
        let lon = self.inv_lon[0] + self.inv_lon[1] * x + self.inv_lon[2] * y;
        let lat = self.inv_lat[0] + self.inv_lat[1] * x + self.inv_lat[2] * y;
        let fx = (lon - self.lon0) / RES + jx;
        let fy = (lat - self.lat0) / RES + jy;
        spline_sample_nearest(&self.zf, fy, fx) as f32
    }

    pub fn sample_plain(&self, x: f64, y: f64) -> f32 {
        let lon = self.inv_lon[0] + self.inv_lon[1] * x + self.inv_lon[2] * y;
        let lat = self.inv_lat[0] + self.inv_lat[1] * x + self.inv_lat[2] * y;
        let fx = (lon - self.lon0) / RES;
        let fy = (lat - self.lat0) / RES;
        spline_sample_nearest(&self.zf, fy, fx) as f32
    }
}

// ------------------------------------------------------------------------------------------------ grid codec
struct Grid {
    z: Arr2<f32>,
    ci0: i64,
    rj0: i64,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "truncated DEM data");
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Context of a layer-1 residual: number of bins `<= |r_west| + |r_north|` (`np.digitize`).
#[inline]
fn context(bins: &[i32], act: i32) -> usize {
    bins.iter().take_while(|&&b| b <= act).count()
}

/// One context/group of byte-plane streams: `count` little-endian values, inflated plane by plane.
struct Group {
    vals: Vec<u32>,
    pos: usize,
}

impl Group {
    #[inline]
    fn next(&mut self) -> Result<u32> {
        let v = *self.vals.get(self.pos).context("DEM stream exhausted")?;
        self.pos += 1;
        Ok(v)
    }
}

/// Reads the table of `n` groups (`u32 count, u8 nplanes, u32 comp_len[nplanes]`), then inflates their planes from `body`.
fn read_groups(c: &mut Cursor, n: usize) -> Result<(Vec<(usize, Vec<usize>)>, usize)> {
    let mut table = Vec::with_capacity(n);
    let mut total = 0usize;
    for _ in 0..n {
        let count = c.u32()? as usize;
        let npl = c.take(1)?[0] as usize;
        ensure!(npl <= 4, "bad DEM plane count");
        let lens = (0..npl)
            .map(|_| c.u32().map(|v| v as usize))
            .collect::<Result<Vec<_>>>()?;
        total += count;
        table.push((count, lens));
    }
    Ok((table, total))
}

fn inflate_groups(c: &mut Cursor, table: &[(usize, Vec<usize>)]) -> Result<Vec<Group>> {
    let mut groups = Vec::with_capacity(table.len());
    for (count, lens) in table {
        let mut vals = vec![0u32; *count];
        for (p, &clen) in lens.iter().enumerate() {
            let comp = c.take(clen)?;
            let mut raw = Vec::with_capacity(*count);
            flate2::read::ZlibDecoder::new(comp)
                .read_to_end(&mut raw)
                .context("DEM plane inflate")?;
            ensure!(raw.len() == *count, "DEM plane length mismatch");
            for (v, b) in vals.iter_mut().zip(&raw) {
                *v |= (*b as u32) << (8 * p);
            }
        }
        groups.push(Group { vals, pos: 0 });
    }
    Ok(groups)
}

#[inline]
fn unzigzag(z: u32) -> i32 {
    (z >> 1) as i32 ^ -((z & 1) as i32)
}

/// `ord(x)`: order-preserving signed integer of the IEEE-754 bits (sign-magnitude -> two's complement).
#[inline]
fn ordered(x: f32) -> i32 {
    let u = x.to_bits();
    let mag = (u & 0x7fff_ffff) as i32;
    if u >> 31 == 1 { -mag } else { mag }
}

#[inline]
fn unordered(o: i32) -> f32 {
    if o < 0 {
        f32::from_bits(0x8000_0000 | o.unsigned_abs())
    } else {
        f32::from_bits(o as u32)
    }
}

fn decode_grid(data: &[u8]) -> Result<Grid> {
    ensure!(data.len() > 32 + 8, "DEM data too short");
    let (body, tail) = data.split_at(data.len() - 32);
    let digest = Sha256::digest(body);
    ensure!(
        digest.as_slice() == tail,
        "embedded DEM grid fails its SHA-256 check"
    );
    let mut c = Cursor { b: body, p: 0 };
    ensure!(c.take(8)? == MAGIC, "bad DEM magic");
    let (ncols, nrows) = (c.u32()? as usize, c.u32()? as usize);
    let (ci0, rj0) = (c.i32()? as i64, c.i32()? as i64);
    ensure!(c.u32()? == 60, "DEM grid is not on the 1/60 degree lattice");
    ensure!(
        (8..=20000).contains(&ncols) && (8..=20000).contains(&nrows),
        "implausible DEM size {ncols}x{nrows}"
    );
    let nb = c.u32()? as usize;
    ensure!(nb <= 64, "bad DEM context bin count");
    let bins: Vec<i32> = (0..nb).map(|_| c.i32()).collect::<Result<_>>()?;
    let nctx = nb + 1;
    let (t1, n1) = read_groups(&mut c, nctx)?;
    let (t2, n2) = read_groups(&mut c, NGROUPS)?;
    let cells = ncols * nrows;
    ensure!(
        n1 == cells && n2 == cells,
        "DEM stream cell counts {n1}/{n2} != {cells}"
    );
    let mut g1 = inflate_groups(&mut c, &t1)?;
    let mut g2 = inflate_groups(&mut c, &t2)?;
    ensure!(c.p == body.len(), "trailing bytes in DEM data");

    // layer 1: MED-predicted residuals -> lattice integers q = rint(16 v) (rows ascend northwards; out of grid = 0)
    let mut q = vec![0i32; cells];
    let mut rprev = vec![0i32; ncols];
    let mut rcur = vec![0i32; ncols];
    for r in 0..nrows {
        for col in 0..ncols {
            let rw = if col > 0 { rcur[col - 1] } else { 0 };
            let rn = rprev[col];
            let k = context(&bins, rw.abs() + rn.abs());
            let res = unzigzag(g1[k].next()?);
            let i = r * ncols + col;
            let w = if col > 0 { q[i - 1] } else { 0 };
            let n = if r > 0 { q[i - ncols] } else { 0 };
            let nw = if r > 0 && col > 0 {
                q[i - ncols - 1]
            } else {
                0
            };
            let (mx, mn) = (w.max(n), w.min(n));
            let pred = if nw >= mx {
                mn
            } else if nw <= mn {
                mx
            } else {
                w + n - nw
            };
            q[i] = pred + res;
            rcur[col] = res;
        }
        std::mem::swap(&mut rprev, &mut rcur);
    }
    drop(g1);

    // layer 2: float32 ulp offset d = ord(v) - ord(q / 16), grouped by (bit length of |q|, sign q, d_W != 0, d_N != 0)
    let mut z = vec![0f32; cells];
    let mut nz_prev = vec![false; ncols];
    let mut nz_cur = vec![false; ncols];
    for r in 0..nrows {
        for col in 0..ncols {
            let i = r * ncols + col;
            let qv = q[i];
            ensure!(
                qv.unsigned_abs() < 1 << (NBL - 1),
                "DEM lattice value out of range"
            );
            let bl = (32 - qv.unsigned_abs().leading_zeros()) as usize;
            let dw = col > 0 && nz_cur[col - 1];
            let g = (((bl * 2 + (qv < 0) as usize) * 2 + dw as usize) * 2) + nz_prev[col] as usize;
            let d = unzigzag(g2[g].next()?);
            let w = qv as f32 / LATTICE;
            let o = ordered(w) as i64 + d as i64;
            ensure!(
                o >= i32::MIN as i64 && o <= i32::MAX as i64,
                "DEM ulp offset out of range"
            );
            z[i] = unordered(o as i32);
            nz_cur[col] = d != 0;
        }
        std::mem::swap(&mut nz_prev, &mut nz_cur);
    }
    ensure!(
        z.iter().all(|v| v.is_finite()),
        "DEM grid holds a non-finite value"
    );
    Ok(Grid {
        z: Arr2::from_vec(nrows, ncols, z),
        ci0,
        rj0,
    })
}

/// Test access to the decoded quantised grid of a theatre (no spline filter): `(z, first column index, first row index)`.
#[cfg(test)]
pub fn decode_embedded_grid(t: Theatre) -> Result<(Arr2<f32>, i64, i64)> {
    let g = decode_grid(t.grid())?;
    Ok((g.z, g.ci0, g.rj0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_curves() {
        assert!((z_to_h(0.0) - 1.0).abs() < 1e-6);
        assert!((z_to_h(7159.73295015625) - 254.0).abs() < 1e-3);
        assert!(z_to_h(500.0) > z_to_h(100.0));
        assert_eq!(h_to_z(0.0), 0.0);
        assert!((h_to_z(254.0) - 7159.733).abs() < 0.01);
        assert!((h_to_z(z_to_h(300.0)) - 300.0).abs() < 0.01);
    }

    #[test]
    fn embedded_grids_decode() {
        for t in [Theatre::India, Theatre::America] {
            let (z, _, _) = decode_embedded_grid(t).unwrap();
            let land = z.d.iter().filter(|&&v| v > 0.0).count() as f64 / z.d.len() as f64;
            assert!((0.2..0.8).contains(&land), "{t:?} land fraction {land}");
        }
        assert_ne!(recipe_fingerprint(), [0u8; 32]);
    }
}
