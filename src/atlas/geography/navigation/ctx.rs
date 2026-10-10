//! Shared state of one navigation migration: plan-derived exact constants, chart ownership, quantization maps,
//! the pathfinding area table and collected cross-file references.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value as Json, json};

use super::util::{UNIT, rects_cover, to_f32};
use super::window::Envelope;
use crate::atlas::Plan;

/// One pathfinding area: its owning chart and (after `do_pathfinding`) its new grid.
pub(super) struct AreaRow {
    pub area: usize,
    pub chart: usize,
    pub origin: [i64; 2],
    pub cell: i64,
    pub cols: i64,
    pub rows: i64,
    /// source grid rectangle in world units: `[origin, origin + cols/rows * cell]`
    pub src_rect_world: [[f64; 2]; 2],
    pub align_remainder_raw: Option<[i64; 2]>,
    pub new: Option<NewGrid>,
}

pub(super) struct NewGrid {
    pub origin: [i64; 2],
    pub cell: i64,
    pub start: [i64; 2],
}

/// A trade spline that cannot be mapped without inventing geometry.
pub(super) struct Blocked {
    pub label: String,
    pub spline_index: usize,
    pub kind: &'static str,
    pub charts: Vec<String>,
    pub owners: Vec<Option<String>>,
    pub uncovered: Vec<usize>,
    pub points: Vec<[f64; 2]>,
    pub need: Vec<(String, [[f64; 2]; 2])>,
}

/// Region keys / area ids the tables refer to (validated against the regions contract afterwards).
#[derive(Default)]
pub(super) struct References {
    /// region keys named by POI neighbour/border lists and region pairs
    pub poi_regions: BTreeSet<String>,
    /// (region key, area id) pairs of POI region pairs
    pub poi_areas: BTreeSet<(String, u32)>,
    pub sea_regions: BTreeSet<String>,
    pub sea_ports: BTreeSet<String>,
    pub trade_ports: BTreeSet<String>,
    pub trade_settlements: BTreeSet<String>,
}

pub(super) struct Ctx<'p> {
    pub plan: &'p Plan,
    pub scale: f64,
    /// `N = scale * 2^20`
    pub scale_n: i64,
    /// per chart index: raw Q20 offset
    pub off_raw: Vec<[i64; 2]>,
    pub areas: Vec<AreaRow>,
    /// per chart index: x/y quantization maps
    pub q: Vec<Option<[HashMap<i64, i64>; 2]>>,
    /// spline index -> charts of its control points
    pub spline_charts: BTreeMap<usize, BTreeSet<usize>>,
    pub blocked: Vec<Blocked>,
    pub env: Envelope,
    pub tiebreaks: usize,
    pub refs: References,
}

/// Exact `v * 2^bits` of a finite f64, or an error when it is not an integer.
fn exact_bits(v: f64, bits: i32) -> Result<i128> {
    let s = v * 2f64.powi(bits);
    if !s.is_finite() || s.fract() != 0.0 || s.abs() > 1.0e30 {
        bail!("{v} is not exactly representable at 2^-{bits}");
    }
    Ok(s as i128)
}

impl<'p> Ctx<'p> {
    fn build(plan: &'p Plan) -> Result<Self> {
        if plan.fixed_bits != 20 || plan.source_fixed_bits != 20 {
            bail!(
                "atlas plan fixed bits {}->{}: the stock engine needs Q20->Q20",
                plan.source_fixed_bits,
                plan.fixed_bits
            );
        }
        if plan.world != plan.source_world {
            bail!("atlas plan must keep world == source_world: the stock executable keeps its world minimum");
        }
        let scale = plan.scale;
        if scale < 1.0 {
            bail!("atlas scale {scale} < 1: strictly increasing Q20 rounding is only guaranteed for scale >= 1");
        }
        let n = scale * UNIT as f64;
        if n != n.floor() {
            bail!(
                "plan scale {scale:?} is not a multiple of 2^-20 (grid cell 2.0 and integer-world grid lines would not \
                 map to exact raw values); nearest valid scale {:?}",
                n.round() / UNIT as f64
            );
        }
        let mut off_raw = Vec::new();
        for c in &plan.charts {
            let mut raws = [0i64; 2];
            for (k, v) in c.offset.iter().enumerate() {
                let r = v * UNIT as f64;
                if !r.is_finite() || r.fract() != 0.0 {
                    bail!("chart {}: offset {:?} is not an integral raw Q20 value", c.key, c.offset);
                }
                raws[k] = r as i64;
            }
            off_raw.push(raws);
        }
        let ctx = Self {
            plan,
            scale,
            scale_n: n as i64,
            q: (0..plan.charts.len()).map(|_| None).collect(),
            off_raw,
            areas: Vec::new(),
            spline_charts: BTreeMap::new(),
            blocked: Vec::new(),
            env: Envelope::default(),
            tiebreaks: 0,
            refs: References::default(),
        };
        Ok(ctx)
    }

    pub fn new(plan: &'p Plan) -> Result<Self> {
        let ctx = Self::build(plan)?;
        ctx.check_chart_geometry()?;
        Ok(ctx)
    }

    pub fn chart_key(&self, chart: usize) -> &str {
        &self.plan.charts[chart].key
    }

    /// Index of the chart named `key` (a chart key or one of its theatre keys).
    pub fn chart_index(&self, key: &str) -> Result<usize> {
        self.plan
            .charts
            .iter()
            .position(|c| c.key == key || c.theatres.iter().any(|t| t == key))
            .ok_or_else(|| anyhow!("Atlas has no chart/theatre {key}"))
    }

    /// Output crop rectangle scaled by 2^60 (exact): `(rect * N + off * 2^20) / 2^20` times 2^60.
    fn out_rect_60(&self, rect: [[f64; 2]; 2], chart: usize) -> Result<[[i128; 2]; 2]> {
        let mut out = [[0i128; 2]; 2];
        for a in 0..2 {
            for k in 0..2 {
                let r40 = exact_bits(rect[a][k], 40)?;
                out[a][k] = r40 * self.scale_n as i128 + ((self.off_raw[chart][k] as i128) << 40);
            }
        }
        Ok(out)
    }

    fn check_chart_geometry(&self) -> Result<()> {
        let p = self.plan;
        let mut world = [[0i128; 2]; 2];
        for a in 0..2 {
            for k in 0..2 {
                world[a][k] = exact_bits(p.world[a][k], 60)?;
            }
        }
        let mut outs: Vec<(usize, [[i128; 2]; 2])> = Vec::new();
        for (ci, c) in p.charts.iter().enumerate() {
            if c.crop_rects.is_empty() {
                bail!("chart {} has no crop rectangle", c.key);
            }
            for r in &c.crop_rects {
                let o = self.out_rect_60(*r, ci)?;
                if (0..2).any(|k| o[0][k] < world[0][k] || o[1][k] > world[1][k]) {
                    bail!("chart {}: crop {r:?} lands outside the world", c.key);
                }
                outs.push((ci, o));
            }
            for (t, r) in &c.theatre_rects {
                if !rects_cover(&c.crop_rects, r[0], r[1]) {
                    bail!("chart {}: theatre {t} rectangle {r:?} is not inside its crop union", c.key);
                }
            }
        }
        for (i, (ka, a)) in outs.iter().enumerate() {
            for (kb, b) in &outs[i + 1..] {
                if ka != kb && (0..2).all(|k| a[0][k] < b[1][k] && b[0][k] < a[1][k]) {
                    bail!("output crops of charts {} and {} overlap", p.charts[*ka].key, p.charts[*kb].key);
                }
            }
        }
        Ok(())
    }

    /// Unique chart owning a source world point (theatre rectangles first, then crop rectangles); `None` if
    /// uncovered. A point on a shared theatre boundary belongs to the pathfinding grid containing it half-open
    /// `[origin, far)`; anything else ambiguous is an error.
    pub fn owner(&mut self, point: [f64; 2]) -> Result<Option<usize>> {
        let p = self.plan;
        let mut hit: Vec<usize> = (0..p.charts.len())
            .filter(|&i| p.charts[i].theatre_rects.values().any(|r| Plan::contains(*r, point)))
            .collect();
        if hit.is_empty() {
            hit = (0..p.charts.len())
                .filter(|&i| p.charts[i].crop_rects.iter().any(|r| Plan::contains(*r, point)))
                .collect();
        }
        if hit.len() > 1 {
            let owners: Vec<usize> = hit
                .iter()
                .copied()
                .filter(|&c| {
                    self.areas.iter().any(|row| {
                        row.chart == c && (0..2).all(|a| row.src_rect_world[0][a] <= point[a] && point[a] < row.src_rect_world[1][a])
                    })
                })
                .collect();
            if owners.len() == 1 {
                self.tiebreaks += 1;
                return Ok(Some(owners[0]));
            }
            let mut keys: Vec<&str> = hit.iter().map(|&i| p.charts[i].key.as_str()).collect();
            keys.sort_unstable();
            bail!("point {point:?} is owned by several charts {keys:?}");
        }
        Ok(hit.first().copied())
    }

    pub fn poi_chart(&mut self, x_raw: i32, y_raw: i32) -> Result<usize> {
        let pt = [x_raw as f64 / UNIT as f64, y_raw as f64 / UNIT as f64];
        self.owner(pt)?
            .ok_or_else(|| anyhow!("poi at ({}, {}) is outside every chart", pt[0], pt[1]))
    }

    pub fn area_chart(&self, k: usize) -> Result<usize> {
        self.areas
            .get(k)
            .map(|r| r.chart)
            .ok_or_else(|| anyhow!("pathfinding area {k} has no chart assignment"))
    }

    pub fn axes(&self, chart: usize) -> Result<&[HashMap<i64, i64>; 2]> {
        self.q[chart]
            .as_ref()
            .ok_or_else(|| anyhow!("chart {}: no quantization map (no pathfinding grid on this chart)", self.chart_key(chart)))
    }

    pub fn point_q(&self, x: i32, y: i32, chart: usize) -> Result<(i32, i32)> {
        let q = self.axes(chart)?;
        let get = |a: usize, v: i32| -> Result<i32> {
            let t = q[a]
                .get(&(v as i64))
                .ok_or_else(|| anyhow!("chart {}: raw value {v} missing from the quantization map", self.chart_key(chart)))?;
            i32::try_from(*t).map_err(|_| anyhow!("chart {}: quantized value {t} of {v} leaves i32", self.chart_key(chart)))
        };
        Ok((get(0, x)?, get(1, y)?))
    }

    /// New cell size is exactly `old * scale`.
    pub fn cell_exact(&self, cs: i64, ncs: i64) -> bool {
        ncs as i128 * UNIT as i128 == cs as i128 * self.scale_n as i128
    }

    pub fn note(&mut self, tag: &str, x: i64, y: i64) {
        self.env.add(tag, x, y);
    }

    pub fn note_f(&mut self, tag: &str, p: [f32; 2]) {
        self.env.add_f32(tag, p);
    }

    pub fn check_theatre_in_crop(&self, name: &str, chart: usize, mn: [f32; 2], mx: [f32; 2]) -> Result<()> {
        let c = &self.plan.charts[chart];
        let lo = [mn[0] as f64, mn[1] as f64];
        let hi = [mx[0] as f64, mx[1] as f64];
        if !rects_cover(&c.crop_rects, lo, hi) {
            bail!("sea grid {name}: [{lo:?}, {hi:?}] is not inside the crop union of chart {}", c.key);
        }
        Ok(())
    }

    /// One chart per xy point. A spline must lie wholly inside the crops of ONE chart; otherwise it is recorded in
    /// `self.blocked` (raised by `raise_if_blocked`) and never mapped. Non-spline arrays must be fully owned.
    /// Returns the chart of every point and the set of charts present.
    pub fn point_charts(
        &mut self,
        pts: &[[f64; 2]],
        label: &str,
        spline: Option<usize>,
    ) -> Result<(Vec<usize>, BTreeSet<usize>)> {
        let mut owners: Vec<Option<usize>> = Vec::with_capacity(pts.len());
        for pt in pts {
            owners.push(self.owner(*pt)?);
        }
        let present: BTreeSet<usize> = owners.iter().flatten().copied().collect();
        let mut keys: Vec<String> = present.iter().map(|&c| self.chart_key(c).to_string()).collect();
        keys.sort();
        let missing: Vec<usize> = owners.iter().enumerate().filter(|(_, c)| c.is_none()).map(|(i, _)| i).collect();
        let Some(spline_index) = spline else {
            if !missing.is_empty() || keys.is_empty() {
                let shown: Vec<usize> = missing.iter().copied().take(5).collect();
                let first = missing.first().map(|&i| format!("{:?}", pts[i])).unwrap_or_else(|| "(no points)".into());
                bail!("{label}: points {shown:?} such as {first} are outside every chart");
            }
            return Ok((owners.into_iter().flatten().collect(), present));
        };
        if keys.len() == 1 && missing.is_empty() {
            return Ok((owners.into_iter().flatten().collect(), present));
        }
        let mut need: Vec<(String, [[f64; 2]; 2])> = Vec::new();
        for (pt, c) in pts.iter().zip(&owners) {
            let k = match c {
                Some(c) => self.chart_key(*c).to_string(),
                None if keys.len() == 1 => keys[0].clone(),
                None => "(no chart)".to_string(),
            };
            match need.iter_mut().find(|(n, _)| *n == k) {
                Some((_, b)) => {
                    for a in 0..2 {
                        b[0][a] = b[0][a].min(pt[a]);
                        b[1][a] = b[1][a].max(pt[a]);
                    }
                }
                None => need.push((k, [*pt, *pt])),
            }
        }
        let kind = if keys.is_empty() {
            "no_chart"
        } else if keys.len() > 1 {
            "cross_chart"
        } else {
            "outside_crop"
        };
        let fallback = match keys.first() {
            Some(k) => self.chart_index(k)?,
            None => 0,
        };
        let owner_keys: Vec<Option<String>> = owners.iter().map(|c| c.map(|c| self.chart_key(c).to_string())).collect();
        self.blocked.push(Blocked {
            label: label.to_string(),
            spline_index,
            kind,
            charts: keys,
            owners: owner_keys,
            uncovered: missing,
            points: pts.to_vec(),
            need,
        });
        Ok((vec![fallback; pts.len()], present))
    }

    // ---- scalar float helpers (f32 in, f32 out)
    pub fn f32_scaled(&self, v: f32) -> Result<f32> {
        if !v.is_finite() {
            bail!("non-finite float {v}");
        }
        to_f32(v as f64 * self.scale)
    }

    /// f32 distance: `>= 0` scaled, the sentinel (exactly `allow_sentinel`) unchanged.
    pub fn distance(&self, v: f32, allow_sentinel: Option<f32>) -> Result<f32> {
        if v < 0.0 {
            if allow_sentinel != Some(v) {
                bail!("unexpected negative distance {v}");
            }
            return Ok(v);
        }
        self.f32_scaled(v)
    }

    pub fn xy_f64(&self, p: [f64; 2], chart: usize) -> [f64; 2] {
        self.plan.xy(p, &self.plan.charts[chart])
    }

    pub fn xy_f32(&self, p: [f32; 2], chart: usize) -> Result<[f32; 2]> {
        let m = self.xy_f64([p[0] as f64, p[1] as f64], chart);
        if !m[0].is_finite() || !m[1].is_finite() {
            bail!("non-finite scaled point {p:?}");
        }
        Ok([m[0] as f32, m[1] as f32])
    }

    pub fn fixed_len(&self, raw: i32) -> Result<i32> {
        self.plan.fixed_distance(raw)
    }

    /// Fail with the full inventory when any trade spline was blocked.
    pub fn raise_if_blocked(&mut self, route_ids: &[Vec<u32>], node_route_ids: &[Vec<u32>]) -> Result<()> {
        if self.blocked.is_empty() {
            return Ok(());
        }
        let mut use_count: HashMap<usize, usize> = HashMap::new();
        for ids in route_ids {
            for j in ids {
                *use_count.entry(*j as usize / 2).or_default() += 1;
            }
        }
        let mut node_use: HashMap<usize, usize> = HashMap::new();
        for ids in node_route_ids {
            for j in ids {
                *node_use.entry(*j as usize / 2).or_default() += 1;
            }
        }
        let inventory: Vec<Json> = self
            .blocked
            .iter()
            .map(|b| {
                let mut need = serde_json::Map::new();
                for (k, r) in &b.need {
                    need.insert(k.clone(), json!(r));
                }
                json!({
                    "label": b.label, "kind": b.kind, "charts": b.charts, "owners": b.owners,
                    "uncovered_point_indices": b.uncovered, "points": b.points,
                    "required_source_bbox_per_chart": need, "spline_index": b.spline_index,
                    "routes_using": use_count.get(&b.spline_index).copied().unwrap_or(0),
                    "node_routes_using": node_use.get(&b.spline_index).copied().unwrap_or(0),
                })
            })
            .collect();
        let first = self
            .blocked
            .iter()
            .take(6)
            .map(|b| format!("{} {} charts={:?} uncovered={}", b.label, b.kind, b.charts, b.uncovered.len()))
            .collect::<Vec<_>>()
            .join("; ");
        bail!(
            "{} trade splines cannot be mapped without inventing geometry (span charts or lie outside every crop): \
             {first} ... full inventory: {}",
            self.blocked.len(),
            Json::Array(inventory)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_plan_constants_are_exact() {
        let plan = Plan::proven();
        let ctx = Ctx::new(&plan).unwrap();
        assert_eq!(ctx.scale_n, 1_417_358);
        assert_eq!(ctx.off_raw[0], [-446_407_024, -121_153_736]);
        assert_eq!(ctx.off_raw[5], [-1_177_763_752, -254_385_388]);
        // round_half_even(x) + offset == round_half_even(x + offset) (the plan's float rule) for every x only when the
        // offset is even; the quantizer verifies the equality per value, the stock offsets all satisfy it.
        assert!(ctx.off_raw.iter().flatten().all(|v| v % 2 == 0));
        assert!(ctx.cell_exact(2_097_152, 2_834_716));
        assert!(!ctx.cell_exact(2_097_152, 2_834_717));
    }

    #[test]
    fn plans_that_break_the_stock_contract_are_rejected() {
        let mut p = Plan::proven();
        p.fixed_bits = 18;
        assert!(Ctx::new(&p).is_err());
        let mut p = Plan::proven();
        p.scale = 1.3516979217529297 + 1.0e-9;
        assert!(Ctx::new(&p).is_err(), "not a multiple of 2^-20");
        let mut p = Plan::proven();
        p.scale = 0.5;
        assert!(Ctx::new(&p).is_err());
        let mut p = Plan::proven();
        p.world[0][0] -= 1.0;
        assert!(Ctx::new(&p).is_err(), "world != source_world");
        let mut p = Plan::proven();
        p.charts[0].offset[0] += 1.0e-9;
        assert!(Ctx::new(&p).is_err(), "offset not integral raw");
    }

    #[test]
    fn overlapping_output_crops_are_rejected() {
        // put america's output crop on top of Europe's: output x from about 0, y from about -50
        let mut p = Plan::proven();
        p.charts[1].offset = [1040.0, -36.0];
        let err = Ctx::new(&p).err().expect("overlap must be rejected").to_string();
        assert!(err.contains("overlap"), "{err}");
        let mut p = Plan::proven();
        p.charts[0].crop_rects.clear();
        assert!(Ctx::new(&p).is_err());
    }

    #[test]
    fn owner_resolves_unique_charts_and_uncovered_points() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let europe = ctx.chart_index("2").unwrap();
        assert_eq!(ctx.owner([0.0, 300.0]).unwrap(), Some(europe));
        // x = 440 lies on the Europe/India theatre seam, but both belong to ONE chart (eur_ind): no tiebreak needed
        assert_eq!(ctx.owner([440.0, 200.0]).unwrap(), Some(europe));
        assert_eq!(ctx.tiebreaks, 0);
        assert_eq!(ctx.owner([1000.0, 600.0]).unwrap(), None);
        assert!(ctx.poi_chart((1000.0 * 1048576.0) as i32, 0).is_err());
    }

    #[test]
    fn half_open_grid_tiebreak_resolves_a_shared_boundary_and_ambiguity_is_an_error() {
        // Two charts whose theatre rectangles share the boundary x = 100.
        let mut plan = Plan::proven();
        plan.charts[0].theatre_rects.clear();
        plan.charts[0].theatre_rects.insert("2".into(), [[-10.0, 0.0], [100.0, 10.0]]);
        plan.charts[1].theatre_rects.clear();
        plan.charts[1].theatre_rects.insert("1".into(), [[100.0, 0.0], [200.0, 10.0]]);
        let (a, b) = (0usize, 1usize);
        let mut ctx = Ctx::build(&plan).unwrap();
        let row = |chart: usize, lo: f64, hi: f64| AreaRow {
            area: 0,
            chart,
            origin: [0, 0],
            cell: 1,
            cols: 1,
            rows: 1,
            src_rect_world: [[lo, 0.0], [hi, 10.0]],
            align_remainder_raw: None,
            new: None,
        };
        // no grid -> ambiguous
        assert!(ctx.owner([100.0, 5.0]).is_err());
        ctx.areas = vec![row(b, 100.0, 200.0), row(a, -10.0, 100.0)];
        // [-10, 100) and [100, 200): x = 100 belongs to the second grid only (half-open), i.e. chart b
        assert_eq!(ctx.owner([100.0, 5.0]).unwrap(), Some(b));
        assert_eq!(ctx.tiebreaks, 1);
        // both grids claim the point -> still ambiguous
        ctx.areas = vec![row(a, -10.0, 100.5), row(b, 100.0, 200.0)];
        assert!(ctx.owner([100.0, 5.0]).is_err());
    }

    #[test]
    fn spline_spanning_charts_is_blocked_and_reported_with_inventory() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let pts = [[0.0, 300.0], [-500.0, 200.0], [-500.0, 210.0], [-500.0, 220.0]];
        let (charts, present) = ctx.point_charts(&pts, "SPLINES[3]", Some(3)).unwrap();
        assert_eq!(charts.len(), 4);
        assert_eq!(present.len(), 2);
        assert_eq!(ctx.blocked.len(), 1);
        assert_eq!(ctx.blocked[0].kind, "cross_chart");
        let err = ctx.raise_if_blocked(&[vec![6, 7]], &[vec![6]]).unwrap_err().to_string();
        assert!(err.contains("SPLINES[3]") && err.contains("cross_chart") && err.contains("\"routes_using\":2"), "{err}");
        // a non-spline array with an uncovered point is an error immediately
        assert!(ctx.point_charts(&[[1000.0, 600.0]], "TRADE_NODE_ROUTES[0]", None).is_err());
    }
}
