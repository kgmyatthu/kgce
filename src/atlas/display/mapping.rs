//! World-registered raster mapping of the connected atlas: `out_px = scale * in_px + k` per chart with half-open
//! crop coverage. Port of `Mapping` from the accepted `raster_scale.py`; every operation is evaluated in the same
//! float64 order so output pixel coverage and inverse sample positions are identical.

use crate::atlas::Plan;
use anyhow::{Result, bail, ensure};

#[derive(Clone, Debug)]
pub struct PxRect {
    pub chart: String,
    pub sx0: i64,
    pub sy0: i64,
    pub sx1: i64,
    pub sy1: i64,
    pub ox0: i64,
    pub oy0: i64,
    pub ox1: i64,
    pub oy1: i64,
    pub kx: f64,
    pub ky: f64,
}

#[derive(Clone, Debug)]
pub struct Mapping {
    pub s: f64,
    pub density: f64,
    pub src_w: i64,
    pub src_h: i64,
    pub out_w: i64,
    pub out_h: i64,
    pub rects: Vec<PxRect>,
}

fn integral(v: f64, what: &str) -> Result<i64> {
    let r = v.round_ties_even();
    ensure!((v - r).abs() <= 1e-6, "{what} is not an integral pixel value: {v}");
    Ok(r as i64)
}

pub fn ceildiv(a: i64, b: i64) -> i64 {
    -((-a).div_euclid(b))
}

impl Mapping {
    pub fn new(plan: &Plan, src_w: usize, src_h: usize, name: &str) -> Result<Self> {
        let s = plan.scale;
        ensure!(s.is_finite() && s >= 1.0, "{name}: enlargement must be finite and >= 1, got {s}");
        let (sw, w) = (plan.source_world, plan.world);
        let swx = sw[1][0] - sw[0][0];
        let swy = sw[1][1] - sw[0][1];
        let dx = src_w as f64 / swx;
        let dy = src_h as f64 / swy;
        ensure!(
            (dx - dy).abs() <= 1e-9 * dx.max(dy),
            "{name}: source {src_w}x{src_h} is not uniform over source world {swx}x{swy}"
        );
        let d = dx;
        let dims = [(w[1][0] - w[0][0]) * d, (w[1][1] - w[0][1]) * d];
        let mut out = [0i64; 2];
        for i in 0..2 {
            let r = dims[i].round_ties_even();
            ensure!((dims[i] - r).abs() <= 1e-6, "{name}: nonintegral raster dimensions: {dims:?}");
            out[i] = r as i64;
        }
        let (out_w, out_h) = (out[0], out[1]);
        let mut rects: Vec<PxRect> = Vec::new();
        for chart in &plan.charts {
            let off = chart.offset;
            let mut kx = (s * sw[0][0] + off[0] - w[0][0]) * d;
            let mut ky = (w[1][1] - s * sw[1][1] - off[1]) * d;
            if (kx - kx.round_ties_even()).abs() < 1e-7 {
                kx = kx.round_ties_even();
            }
            if (ky - ky.round_ties_even()).abs() < 1e-7 {
                ky = ky.round_ties_even();
            }
            for rr in &chart.crop_rects {
                let sx0 = integral((rr[0][0] - sw[0][0]) * d, &format!("{name} {} crop x0", chart.key))?;
                let sx1 = integral((rr[1][0] - sw[0][0]) * d, &format!("{name} {} crop x1", chart.key))?;
                let sy0 = integral((sw[1][1] - rr[1][1]) * d, &format!("{name} {} crop y0", chart.key))?;
                let sy1 = integral((sw[1][1] - rr[0][1]) * d, &format!("{name} {} crop y1", chart.key))?;
                ensure!(
                    0 <= sx0 && sx0 < sx1 && sx1 <= src_w as i64 && 0 <= sy0 && sy0 < sy1 && sy1 <= src_h as i64,
                    "{name}: crop rect of {} outside the source raster",
                    chart.key
                );
                let ce = |v: i64, k: f64| (s * v as f64 + k - 1e-7).ceil() as i64;
                let r = PxRect {
                    chart: chart.key.clone(),
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    ox0: ce(sx0, kx),
                    ox1: ce(sx1, kx),
                    oy0: ce(sy0, ky),
                    oy1: ce(sy1, ky),
                    kx,
                    ky,
                };
                ensure!(
                    0 <= r.ox0 && r.ox0 < r.ox1 && r.ox1 <= out_w && 0 <= r.oy0 && r.oy0 < r.oy1 && r.oy1 <= out_h,
                    "{name}: output rect of {} {:?} leaves the output raster {out_w}x{out_h}",
                    chart.key,
                    (r.ox0, r.oy0, r.ox1, r.oy1)
                );
                rects.push(r);
            }
        }
        for (i, a) in rects.iter().enumerate() {
            for b in &rects[i + 1..] {
                if a.chart != b.chart && a.ox0 < b.ox1 && b.ox0 < a.ox1 && a.oy0 < b.oy1 && b.oy0 < a.oy1 {
                    bail!("{name}: charts {} and {} overlap in the output raster", a.chart, b.chart);
                }
            }
        }
        Ok(Self { s, density: d, src_w: src_w as i64, src_h: src_h as i64, out_w, out_h, rects })
    }

    /// Nearest inverse sample row/column of an output pixel inside `r` (clamped to the source crop).
    #[inline]
    pub fn src_row(&self, r: &PxRect, oy: i64) -> i64 {
        (((oy as f64 - r.ky) / self.s + 1e-9).floor() as i64).clamp(r.sy0, r.sy1 - 1)
    }

    #[inline]
    pub fn src_col(&self, r: &PxRect, ox: i64) -> i64 {
        (((ox as f64 - r.kx) / self.s + 1e-9).floor() as i64).clamp(r.sx0, r.sx1 - 1)
    }

    /// True when the union of all output crops covers the whole output raster.
    pub fn covers_output(&self) -> bool {
        let mut xs = vec![0, self.out_w];
        let mut ys = vec![0, self.out_h];
        for r in &self.rects {
            xs.extend([r.ox0, r.ox1]);
            ys.extend([r.oy0, r.oy1]);
        }
        xs.sort_unstable();
        xs.dedup();
        ys.sort_unstable();
        ys.dedup();
        let mut area = 0i64;
        for xw in xs.windows(2) {
            for yw in ys.windows(2) {
                if self.rects.iter().any(|r| r.ox0 <= xw[0] && xw[1] <= r.ox1 && r.oy0 <= yw[0] && yw[1] <= r.oy1) {
                    area += (xw[1] - xw[0]) * (yw[1] - yw[0]);
                }
            }
        }
        area == self.out_w * self.out_h
    }

    /// Source-space boxes `(x0, y0, x1, y1)` of every crop.
    pub fn source_boxes(&self) -> Vec<[i64; 4]> {
        self.rects.iter().map(|r| [r.sx0, r.sy0, r.sx1, r.sy1]).collect()
    }

    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "density_px_per_unit": self.density,
            "source_px": [self.src_w, self.src_h],
            "output_px": [self.out_w, self.out_h],
            "scale": self.s,
            "rects": self.rects.iter().map(|r| serde_json::json!({
                "chart": r.chart,
                "source_px": [r.sx0, r.sy0, r.sx1, r.sy1],
                "output_px": [r.ox0, r.oy0, r.ox1, r.oy1],
                "offset_px": [r.kx, r.ky],
            })).collect::<Vec<_>>(),
        })
    }
}

/// Boolean `bh x bw` grid (row-major) of `unit`-sized cells (origin `x0,y0`) intersecting any `[x0,y0,x1,y1]` box.
pub fn touch_grid(boxes: &[[i64; 4]], y0: i64, x0: i64, bh: usize, bw: usize, unit: i64) -> Vec<bool> {
    let mut g = vec![false; bh * bw];
    for b in boxes {
        let cx0 = ((b[0] - x0).div_euclid(unit)).max(0);
        let cx1 = ceildiv(b[2] - x0, unit).min(bw as i64);
        let cy0 = ((b[1] - y0).div_euclid(unit)).max(0);
        let cy1 = ceildiv(b[3] - y0, unit).min(bh as i64);
        if cx0 < cx1 && cy0 < cy1 {
            for y in cy0..cy1 {
                for x in cx0..cx1 {
                    g[y as usize * bw + x as usize] = true;
                }
            }
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proven_plan_maps_every_level_without_chart_overlap_and_keeps_source_size() {
        let plan = Plan::proven();
        for (w, h) in [(8192usize, 4096usize), (4096, 2048), (2048, 1024), (65536, 32768), (1024, 512)] {
            let m = Mapping::new(&plan, w, h, "t").unwrap();
            assert_eq!((m.out_w, m.out_h), (w as i64, h as i64));
            assert_eq!(m.rects.len(), plan.charts.iter().map(|c| c.crop_rects.len()).sum::<usize>());
        }
    }

    #[test]
    fn inverse_sample_stays_inside_the_source_crop() {
        let plan = Plan::proven();
        let m = Mapping::new(&plan, 4096, 2048, "t").unwrap();
        for r in &m.rects {
            for oy in [r.oy0, r.oy1 - 1] {
                let sy = m.src_row(r, oy);
                assert!(r.sy0 <= sy && sy < r.sy1);
            }
            for ox in [r.ox0, r.ox1 - 1] {
                let sx = m.src_col(r, ox);
                assert!(r.sx0 <= sx && sx < r.sx1);
            }
        }
    }
}
