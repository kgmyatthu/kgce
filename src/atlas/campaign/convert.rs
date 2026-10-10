//! Stock-Q20 atlas migration of `startpos.esf` (native port of the accepted `campaign_atlas.py`).
//!
//! Every spatial value moves with the plan transform of its OWNING chart: `out = src * scale + chart.offset`. Fixed
//! points stay Q20 (`Plan::fixed_xy`, plus the per-chart point overrides of the navigation builder). Nothing is
//! guessed: any record whose owner is ambiguous / uncovered, any region that straddles charts, any HLCI group without
//! surviving geometry, any LOS that would vanish and any unclassified spatial field is collected as a problem and the
//! run fails without writing output.
//!
//! Not changed (and why): object ids/handles, budgets, angles, model heights, CAI weights, economy, pathfinder
//! topology (grid node maps, OBSTACLE_*, clearance floats), sea-grid cell ids, faction/region ids, theatre ids, shroud
//! trees (verified empty) and every `v1`/`v4` id field of [`KNOWN_I32_IDS`].

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result, ensure};
use serde_json::{Value as Json, json};

use super::contract::{Contract, PointOverrides};
use super::nodes::*;
use super::visibility::{Cell, Tree, Visibility, decode_window, encode_window, tree_bbox};
use crate::atlas::{Chart, Plan, Rect};
use crate::esf::{Node, Value};
use crate::regionmap::geometry::{P, point_in_ring, point_segment_distance};
use crate::regionmap::regions::RegionData;

pub const Q: f64 = (1u64 << 20) as f64;
const THEATRE_NAMES: [(&str, &str); 3] = [("america", "1"), ("europe", "2"), ("india", "3")];
const CAI_REGION_KEY: usize = 10;

/// Records whose first fixed-point pair(s) start at the listed child indexes.
pub const FIXED_PAIRS: [(&str, &[usize]); 6] = [
    ("LOCOMOTABLE", &[0, 2]),
    ("REGION_SLOT", &[6]),
    ("SIEGEABLE_GARRISON_RESIDENCE", &[10]),
    ("CAI_SITUATED", &[0]),
    ("CAI_TRADING_POST", &[1]),
    ("CAI_REGION_HLCI", &[3]),
];
/// Records / arrays that hold float world points and are converted by a dedicated handler.
const POINT_RECORDS: [&str; 5] = ["CAMPAIGN_CAMERA", "LINE_OF_SIGHT", "PENDING_BATTLE", "TRADE_SEGMENTS[]", "TRADE_NODES[]"];
/// i32 fields >= 2^22 in magnitude that are object handles / ids (census of the stock start position), not coordinates.
const KNOWN_I32_IDS: [(&str, usize); 20] = [
    ("ARMY", 4),
    ("CHARACTER", 2),
    ("CHARACTER_POST", 0),
    ("CHARACTER_POST", 4),
    ("CHARACTER_POST", 5),
    ("DIPLOMACY_RELATIONSHIP", 0),
    ("FACTION", 4),
    ("FACTION", 7),
    ("FACTION", 8),
    ("FACTION", 35),
    ("FACTION", 36),
    ("FACTION", 38),
    ("FACTION", 39),
    ("FACTION_TECHNOLOGY_MANAGER", 1),
    ("GOVERNMENT", 0),
    ("GOVERNORSHIP", 1),
    ("REGION", 4),
    ("REGION_RECRUITMENT_MANAGER", 2),
    ("SETTLEMENT", 4),
    ("UNIT", 4),
];

/// Point that no chart claims (or two charts claim ambiguously).
#[derive(Debug)]
pub struct Unmapped(pub String);

#[derive(Clone, Debug)]
pub struct Problem {
    pub kind: &'static str,
    pub at: String,
    pub detail: String,
}

pub type Counter = BTreeMap<String, u64>;

fn bump(c: &mut Counter, key: impl Into<String>, by: u64) {
    *c.entry(key.into()).or_default() += by;
}

/// Political region geometry with a key index.
pub struct RegionSet {
    pub data: RegionData,
    pub index: HashMap<String, usize>,
}

impl RegionSet {
    pub fn new(data: RegionData) -> Self {
        let index = data.regions.iter().enumerate().map(|(i, r)| (r.key.clone(), i)).collect();
        RegionSet { data, index }
    }

    /// True when `pt` lies inside area `ai` of region `ri` or within `tol` of its outline (shapely
    /// `buffer(tol).contains(pt)` of the accepted POC).
    pub fn near(&self, ri: usize, ai: usize, pt: P, tol: f64) -> bool {
        let Some(area) = self.data.regions[ri].areas.get(ai) else { return false };
        let (mut inside, mut in_hole) = (false, false);
        for o in &area.outlines {
            let ring: Vec<P> = o.ring.iter().map(|&i| self.data.v(i)).collect();
            if point_in_ring(pt, &ring) {
                if o.outer {
                    inside = true;
                } else {
                    in_hole = true;
                }
            }
        }
        if inside && !in_hole {
            return true;
        }
        area.outlines.iter().any(|o| {
            let n = o.ring.len();
            (0..n).any(|k| point_segment_distance(pt, self.data.v(o.ring[k]), self.data.v(o.ring[(k + 1) % n])) < tol)
        })
    }
}

fn rect_overlap(a: &Rect, b: &Rect) -> bool {
    let w = a[1][0].min(b[1][0]) - a[0][0].max(b[0][0]);
    let h = a[1][1].min(b[1][1]) - a[0][1].max(b[0][1]);
    w > 1e-9 && h > 1e-9
}

/// True when the union of `rects` covers every point of `target` (rectilinear coordinate-compression test).
pub fn covers(rects: &[Rect], target: &Rect) -> bool {
    let mut xs = vec![target[0][0], target[1][0]];
    let mut ys = vec![target[0][1], target[1][1]];
    for r in rects {
        xs.extend([r[0][0], r[1][0]]);
        ys.extend([r[0][1], r[1][1]]);
    }
    for v in [&mut xs, &mut ys] {
        v.retain(|x| x.is_finite());
        v.sort_by(f64::total_cmp);
        v.dedup();
    }
    for wx in xs.windows(2) {
        for wy in ys.windows(2) {
            let c = [(wx[0] + wx[1]) / 2.0, (wy[0] + wy[1]) / 2.0];
            if Plan::contains(*target, c) && !rects.iter().any(|r| Plan::contains(*r, c)) {
                return false;
            }
        }
    }
    true
}

/// Validate the plan beyond `Plan::validate`; returns the output crops (crop rectangles moved by their chart).
pub fn check_plan(plan: &Plan) -> Result<BTreeMap<String, Vec<Rect>>> {
    plan.validate()?;
    ensure!(plan.fixed_bits == 20 && plan.source_fixed_bits == 20, "atlas campaign migration requires an atlas plan with Q20 -> Q20");
    ensure!(plan.world == plan.source_world, "stock atlas keeps the world rectangle (world != source_world)");
    ensure!(plan.scale >= 1.0, "scale < 1 is unsupported (visibility cells would be dropped)");
    let mut outs: BTreeMap<String, Vec<Rect>> = BTreeMap::new();
    for c in &plan.charts {
        let rects: Vec<Rect> = c.crop_rects.iter().map(|r| [plan.xy(r[0], c), plan.xy(r[1], c)]).collect();
        for (t, tr) in &c.theatre_rects {
            ensure!(covers(&c.crop_rects, tr), "chart {}: theatre {t} rect not covered by its crop union", c.key);
        }
        for r in &rects {
            ensure!(
                (0..2).all(|i| r[0][i] >= plan.world[0][i] - 1e-9 && r[1][i] <= plan.world[1][i] + 1e-9),
                "chart {}: output crop outside the world",
                c.key
            );
        }
        outs.insert(c.key.clone(), rects);
    }
    let keys: Vec<&String> = outs.keys().collect();
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            ensure!(!outs[*a].iter().any(|ra| outs[*b].iter().any(|rb| rect_overlap(ra, rb))), "output crops of charts {a} and {b} overlap");
        }
    }
    Ok(outs)
}

/// Camera target clamp `(xmin, ymin, xmax, ymax)`: output bounding box of all theatre rects plus the source margins,
/// clamped to the world. (The accepted POC moves the start-position bounds with the plain affine; only the regions /
/// sea-grid theatre rectangle copies are Q12-snapped.)
pub fn camera_bounds(plan: &Plan, src: [f32; 4]) -> Result<[f32; 4]> {
    let rects: Vec<(&Chart, &Rect)> = plan.charts.iter().flat_map(|c| c.theatre_rects.values().map(move |r| (c, r))).collect();
    ensure!(!rects.is_empty(), "plan has no theatre rectangles");
    let t0: [f64; 2] = std::array::from_fn(|i| rects.iter().map(|(_, r)| r[0][i]).fold(f64::INFINITY, f64::min));
    let t1: [f64; 2] = std::array::from_fn(|i| rects.iter().map(|(_, r)| r[1][i]).fold(f64::NEG_INFINITY, f64::max));
    let s = src.map(f64::from);
    let margins = [t0[0] - s[0], t0[1] - s[1], s[2] - t1[0], s[3] - t1[1]];
    ensure!(
        margins.iter().all(|m| *m >= 0.0),
        "source camera bounds {src:?} do not cover the theatre rects {t0:?} {t1:?}: the source world does not match the accepted atlas source"
    );
    let mut outs: Vec<[f64; 2]> = Vec::new();
    for (c, r) in &rects {
        outs.push(plan.xy(r[0], c));
        outs.push(plan.xy(r[1], c));
    }
    let o0: [f64; 2] = std::array::from_fn(|i| outs.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min));
    let o1: [f64; 2] = std::array::from_fn(|i| outs.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max));
    let nb = [o0[0] - margins[0], o0[1] - margins[1], o1[0] + margins[2], o1[1] + margins[3]];
    let [lo, hi] = plan.world;
    Ok([nb[0].max(lo[0]) as f32, nb[1].max(lo[1]) as f32, nb[2].min(hi[0]) as f32, nb[3].min(hi[1]) as f32])
}

/// `(xmin, ymin, xmax, ymax)` of the single `EPISODIC_RESTRICTIONS` record.
pub fn camera_bounds_of(root: &Node) -> Result<[f32; 4]> {
    let mut found: Vec<&Node> = Vec::new();
    for_each_record(root, "EPISODIC_RESTRICTIONS", &mut |n| found.push(n));
    ensure!(found.len() == 1, "expected one EPISODIC_RESTRICTIONS, found {}", found.len());
    let ch = record_children(found[0], "EPISODIC_RESTRICTIONS")?;
    Ok([f32_at(ch, 9)?, f32_at(ch, 10)?, f32_at(ch, 11)?, f32_at(ch, 12)?])
}

/// Byte offsets and raw values of every vertex of a PATHFINDING_GRID cached-polygon payload.
pub fn cached_point_offsets(count: u32, raw: &[u8]) -> Result<Vec<(usize, i32, i32)>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    let rd = |at: usize| -> Result<[u8; 4]> {
        raw.get(at..at + 4).map(|b| [b[0], b[1], b[2], b[3]]).context("truncated cached polygon")
    };
    for _ in 0..count {
        let n = u32::from_le_bytes(rd(cursor)?) as usize;
        cursor += 4;
        ensure!(cursor + n * 8 + 4 <= raw.len(), "truncated cached polygon");
        for _ in 0..n {
            out.push((cursor, i32::from_le_bytes(rd(cursor)?), i32::from_le_bytes(rd(cursor + 4)?)));
            cursor += 8;
        }
        cursor += 4;
    }
    ensure!(cursor == raw.len(), "unconsumed cached polygon payload");
    Ok(out)
}

/// Region currently converted: chart evidence and the CAI theatre hint of its `REGION` record.
pub struct RegionCtx {
    pub key: String,
    pub hint: Option<&'static str>,
    pub charts: BTreeSet<String>,
}

impl RegionCtx {
    fn new(children: &[Node]) -> Result<Self> {
        let hint = match children.get(23) {
            Some(Node::Value(Value::String16(s) | Value::String8(s))) => THEATRE_NAMES.iter().find(|(k, _)| *k == s.as_str()).map(|(_, v)| *v),
            _ => None,
        };
        Ok(RegionCtx { key: str_at(children, 0)?.to_string(), hint, charts: BTreeSet::new() })
    }
}

/// Dispatch class of a record (`name[]` for record arrays).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Hlci,
    Fixed(&'static [usize]),
    Character,
    CaiRegion,
    Boundary,
    Camera,
    Episodic,
    Los,
    Pending,
    TradeNodes,
    TradeSegments,
    TradeRoutes,
    Grid,
    Shroud,
    Obstacles,
    Other,
}

impl Kind {
    fn of(name: &str) -> Kind {
        match name {
            "CAI_REGION_HLCI" => Kind::Hlci,
            "CHARACTER" => Kind::Character,
            "CAI_REGION" => Kind::CaiRegion,
            "CAI_REGION_BOUNDARY" => Kind::Boundary,
            "CAMPAIGN_CAMERA" => Kind::Camera,
            "EPISODIC_RESTRICTIONS" => Kind::Episodic,
            "LINE_OF_SIGHT" => Kind::Los,
            "PENDING_BATTLE" => Kind::Pending,
            "TRADE_NODES[]" => Kind::TradeNodes,
            "TRADE_SEGMENTS[]" => Kind::TradeSegments,
            "TRADE_ROUTES[]" => Kind::TradeRoutes,
            "PATHFINDING_GRID[]" => Kind::Grid,
            "QUAD_TREE_BIT_ARRAY" => Kind::Shroud,
            "CHARACTER_OBSTACLE[]" | "FORT_OBSTACLE[]" | "BARRIER_OBSTACLE[]" => Kind::Obstacles,
            other => match FIXED_PAIRS.iter().find(|(n, _)| *n == other) {
                Some((_, at)) => Kind::Fixed(*at),
                None => Kind::Other,
            },
        }
    }
}

fn where_of(path: &[String]) -> String {
    path[path.len().saturating_sub(4)..].join("/")
}

pub struct Conv<'a> {
    pub plan: &'a Plan,
    pub vis: Visibility,
    pub overrides: PointOverrides,
    pub contract: &'a Contract,
    pub src_regions: RegionSet,
    pub out_regions: RegionSet,
    pub counts: Counter,
    pub changed: Counter,
    pub sentinels: Counter,
    pub problems: Vec<Problem>,
    point_cache: HashMap<(i32, i32), String>,
    pub cached_grids: Vec<Json>,
    pub los_stats: BTreeMap<String, Counter>,
    pub hlci_relocated: Vec<Json>,
    pub deleted_hlci_ids: BTreeSet<u32>,
    pub deleted_hlci: Vec<Json>,
    pub zero_center_regions: Vec<String>,
    pub grid_charts: Vec<String>,
    pub cai_region_key: HashMap<u32, String>,
    pub q_lo: [i64; 2],
    pub q_hi: [i64; 2],
    pub out_crops: BTreeMap<String, Vec<Rect>>,
    pub src_bounds: [f32; 4],
    pub new_bounds: [f32; 4],
}

impl<'a> Conv<'a> {
    /// `root` is the pristine source start position (read-only here): it supplies the grid ownership, the CAI region
    /// keys and the source camera bounds.
    pub fn new(
        plan: &'a Plan,
        overrides: PointOverrides,
        contract: &'a Contract,
        src_regions: RegionSet,
        out_regions: RegionSet,
        root: &Node,
        grid_charts: Vec<String>,
    ) -> Result<Self> {
        let out_crops = check_plan(plan)?;
        let cai_region_key = collect_region_keys(root)?;
        ensure!(
            cai_region_key.len() == contract.regions.len(),
            "start position has {} CAI regions but the regions contract lists {}: the start position was not built from the source regions the atlas geography was derived from",
            cai_region_key.len(),
            contract.regions.len()
        );
        let src_bounds = camera_bounds_of(root)?;
        let new_bounds = camera_bounds(plan, src_bounds)?;
        let q = |v: f64| (v * Q).round_ties_even() as i64;
        Ok(Conv {
            plan,
            vis: Visibility::new(plan)?,
            overrides,
            contract,
            src_regions,
            out_regions,
            counts: Counter::new(),
            changed: Counter::new(),
            sentinels: Counter::new(),
            problems: Vec::new(),
            point_cache: HashMap::new(),
            cached_grids: Vec::new(),
            los_stats: BTreeMap::new(),
            hlci_relocated: Vec::new(),
            deleted_hlci_ids: BTreeSet::new(),
            deleted_hlci: Vec::new(),
            zero_center_regions: Vec::new(),
            grid_charts,
            cai_region_key,
            q_lo: [q(plan.world[0][0]), q(plan.world[0][1])],
            q_hi: [q(plan.world[1][0]), q(plan.world[1][1])],
            out_crops,
            src_bounds,
            new_bounds,
        })
    }

    pub fn problem(&mut self, kind: &'static str, at: impl Into<String>, detail: impl Into<String>) {
        self.problems.push(Problem { kind, at: at.into(), detail: detail.into() });
    }

    // ------------------------------------------------------------------------------------------ ownership
    fn candidates(&self, pt: [f64; 2]) -> Vec<&'a Chart> {
        let plan: &'a Plan = self.plan;
        let th: Vec<&Chart> = plan.charts.iter().filter(|c| c.theatre_rects.values().any(|r| Plan::contains(*r, pt))).collect();
        if !th.is_empty() {
            return th;
        }
        plan.charts.iter().filter(|c| c.crop_rects.iter().any(|r| Plan::contains(*r, pt))).collect()
    }

    pub fn owner(&self, pt: [f64; 2], hint: Option<&str>, cache_key: Option<(i32, i32)>) -> Result<&'a Chart, Unmapped> {
        let cands = self.candidates(pt);
        match cands.len() {
            1 => return Ok(cands[0]),
            0 => return Err(Unmapped(format!("point {pt:?} lies in no theatre rect or crop"))),
            _ => {}
        }
        if let Some(k) = cache_key.and_then(|k| self.point_cache.get(&k)) {
            if let Some(c) = cands.iter().copied().find(|c| &c.key == k) {
                return Ok(c);
            }
        }
        if let Some(h) = hint.and_then(|h| self.plan.chart(h).ok()) {
            if let Some(c) = cands.iter().copied().find(|c| c.key == h.key) {
                return Ok(c);
            }
        }
        Err(Unmapped(format!("point {pt:?} is ambiguous between charts {:?}", cands.iter().map(|c| &c.key).collect::<Vec<_>>())))
    }

    pub fn fixed_out(&self, chart: &Chart, rx: i32, ry: i32) -> Result<(i32, i32), Unmapped> {
        let [mut x, mut y] = self.plan.fixed_xy([rx, ry], chart).map_err(|e| Unmapped(e.to_string()))?;
        if let Some(v) = self.overrides.get(&chart.key, 0, rx) {
            x = v;
        }
        if let Some(v) = self.overrides.get(&chart.key, 1, ry) {
            y = v;
        }
        if !((self.q_lo[0]..=self.q_hi[0]).contains(&i64::from(x)) && (self.q_lo[1]..=self.q_hi[1]).contains(&i64::from(y))) {
            return Err(Unmapped(format!("output point ({x},{y}) outside the stock world")));
        }
        Ok((x, y))
    }

    pub fn out_xy(&self, chart: &Chart, pt: [f64; 2]) -> Result<[f64; 2], Unmapped> {
        let p = self.plan.xy(pt, chart);
        let w = self.plan.world;
        if !(w[0][0] <= p[0] && p[0] <= w[1][0] && w[0][1] <= p[1] && p[1] <= w[1][1]) {
            return Err(Unmapped(format!("output point ({},{}) outside the stock world", p[0], p[1])));
        }
        Ok(p)
    }

    // ------------------------------------------------------------------------------------------ record handlers
    #[allow(clippy::too_many_arguments)]
    fn fixed_pair(
        &mut self,
        name: &str,
        ch: &mut [Node],
        at: usize,
        ctx: &mut Option<RegionCtx>,
        path: &[String],
        sentinel_ok: bool,
    ) -> Result<()> {
        let (rx, ry) = (i32_at(ch, at)?, i32_at(ch, at + 1)?);
        let where_ = path.join("/");
        if rx == SENT || ry == SENT {
            self.problem("sentinel", where_, format!("unexpected i32 sentinel at fields {at},{}", at + 1));
            return Ok(());
        }
        if (rx, ry) == (0, 0) && sentinel_ok {
            bump(&mut self.sentinels, format!("{name}.unplaced_(0,0)"), 1);
            return Ok(());
        }
        let hint = ctx.as_ref().and_then(|c| c.hint);
        let moved = self.owner([f64::from(rx) / Q, f64::from(ry) / Q], hint, Some((rx, ry))).and_then(|c| self.fixed_out(c, rx, ry).map(|p| (c, p)));
        let (chart, (nx, ny)) = match moved {
            Ok(v) => v,
            Err(e) => {
                self.problem("unmapped_point", where_, e.0);
                return Ok(());
            }
        };
        self.point_cache.entry((rx, ry)).or_insert_with(|| chart.key.clone());
        if let Some(c) = ctx.as_mut() {
            if name == "REGION_SLOT" || name == "SIEGEABLE_GARRISON_RESIDENCE" {
                c.charts.insert(chart.key.clone());
            }
        }
        set_i32(ch, at, nx)?;
        set_i32(ch, at + 1, ny)?;
        bump(&mut self.counts, format!("{name}.fixed_points"), 1);
        bump(&mut self.counts, format!("chart.{}", chart.key), 1);
        bump(&mut self.changed, format!("{name}.fixed_points"), u64::from((nx, ny) != (rx, ry)));
        Ok(())
    }

    fn xy_value(&mut self, name: &str, ch: &mut [Node], at: usize, chart: &Chart, where_: &str) -> Result<()> {
        let before = xy_at(ch, at)?;
        match self.out_xy(chart, before.map(f64::from)) {
            Ok(p) => {
                set_xy(ch, at, [p[0] as f32, p[1] as f32])?;
                bump(&mut self.counts, format!("{name}.xy_points"), 1);
            }
            Err(e) => self.problem("unmapped_point", where_, e.0),
        }
        Ok(())
    }

    fn point12(&mut self, name: &str, ch: &mut [Node], at: usize, hint: Option<&str>, where_: &str) -> Result<()> {
        let before = xy_at(ch, at)?;
        match self.owner(before.map(f64::from), hint, None) {
            Ok(chart) => self.xy_value(name, ch, at, chart, where_),
            Err(e) => {
                self.problem("unmapped_point", where_, e.0);
                Ok(())
            }
        }
    }

    fn distance(&mut self, ch: &mut [Node], at: usize, name: &str) -> Result<()> {
        let old = f32_at(ch, at)?;
        if old < 0.0 {
            bump(&mut self.counts, format!("{name}.negative_sentinels"), 1);
        } else {
            set_f32(ch, at, (f64::from(old) * self.plan.scale) as f32)?;
        }
        bump(&mut self.counts, format!("{name}.distances"), 1);
        Ok(())
    }

    fn distance_array(&mut self, ch: &mut [Node], at: usize, name: &str, expected: usize) -> Result<()> {
        let raw = array_at(ch, at, 0x4A)?;
        ensure!(raw.len() == expected * 4, "unexpected {name} distance-array layout");
        let scale = self.plan.scale;
        let new: Vec<u8> = raw
            .chunks_exact(4)
            .flat_map(|b| {
                let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                (if v < 0.0 { v } else { (f64::from(v) * scale) as f32 }).to_le_bytes()
            })
            .collect();
        set_array(ch, at, 0x4A, new)?;
        bump(&mut self.counts, format!("{name}.distance_array_values"), expected as u64);
        Ok(())
    }

    fn cached_polygons(&mut self, grids: &mut [Vec<Node>]) -> Result<()> {
        ensure!(
            grids.len() == self.grid_charts.len(),
            "grid count changed ({} grids, {} owned)",
            grids.len(),
            self.grid_charts.len()
        );
        for (gi, grid) in grids.iter_mut().enumerate() {
            let chart: &'a Chart = {
                let plan: &'a Plan = self.plan;
                plan.chart(&self.grid_charts[gi])?
            };
            let count = u32_at(grid, 0)?;
            let raw = array_at(grid, 1, 0x48)?.to_vec();
            let mut out = raw.clone();
            let mut points = 0u64;
            for (offset, rx, ry) in cached_point_offsets(count, &raw)? {
                if rx == SENT || ry == SENT {
                    self.problem("sentinel", format!("PATHFINDING_GRID[{gi}]"), "sentinel vertex in cached polygon");
                    continue;
                }
                match self.fixed_out(chart, rx, ry) {
                    Ok((nx, ny)) => {
                        out[offset..offset + 4].copy_from_slice(&nx.to_le_bytes());
                        out[offset + 4..offset + 8].copy_from_slice(&ny.to_le_bytes());
                        points += 1;
                    }
                    Err(e) => self.problem("unmapped_point", format!("PATHFINDING_GRID[{gi}]@{offset}"), e.0),
                }
            }
            set_array(grid, 1, 0x48, out)?;
            self.cached_grids.push(json!({"grid": gi, "chart": chart.key, "polygons": count, "points": points, "bytes": raw.len()}));
            bump(&mut self.counts, "PATHFINDING_GRID.cached_points", points);
        }
        Ok(())
    }

    fn trade_segment(&mut self, item: &mut [Node], idx: usize) -> Result<()> {
        let where_ = format!("TRADE_SEGMENTS[{idx}]");
        let count = u32_at(item, 1)? as usize;
        let end = 2 + count * 4;
        ensure!(item.len() == end + 11, "unexpected trade segment tail");
        let pts: Vec<[f64; 2]> = (2..end).map(|i| xy_at(item, i).map(|p| p.map(f64::from))).collect::<Result<_>>()?;
        let in_crops = |c: &Chart, p: &[f64; 2]| c.crop_rects.iter().any(|r| Plan::contains(*r, *p));
        let plan: &'a Plan = self.plan;
        let mut cands: Vec<&'a Chart> = plan.charts.iter().filter(|c| pts.iter().all(|p| in_crops(c, p))).collect();
        if cands.len() > 1 {
            let ends = [pts[0], pts[pts.len() - 1]];
            cands.retain(|c| ends.iter().all(|p| c.theatre_rects.values().any(|r| Plan::contains(*r, *p))));
        }
        if cands.len() != 1 {
            self.problem(
                "trade_segment",
                where_,
                format!("{} owning charts; points outside crops or across charts (first {:?}, last {:?})", cands.len(), pts[0], pts[pts.len() - 1]),
            );
            return Ok(());
        }
        let chart = cands[0];
        for i in 2..end {
            self.xy_value("TRADE_SEGMENTS[]", item, i, chart, &where_)?;
        }
        self.distance_array(item, end, "TRADE_SEGMENTS[]", count)?;
        self.distance_array(item, end + 1, "TRADE_SEGMENTS[]", count)?;
        self.distance(item, end + 2, "TRADE_SEGMENTS[]")?;
        self.distance(item, end + 7, "TRADE_SEGMENTS[]")?;
        self.line_of_sight(&mut item[end + 8], &(where_ + "/LINE_OF_SIGHT"), "TRADE_SEGMENTS[]")
    }

    fn line_of_sight(&mut self, node: &mut Node, where_: &str, parent: &str) -> Result<()> {
        let ch = record_children_mut(node, "LINE_OF_SIGHT")?;
        if !bool_at(ch, 0)? {
            if ch.len() != 1 {
                self.problem("los", where_, "inactive LOS with payload");
            }
            return Ok(());
        }
        if ch.len() != 4 || xy_at(ch, 1).is_err() || xy_at(ch, 2).is_err() {
            self.problem("los", where_, "unexpected active LOS layout");
            return Ok(());
        }
        let tree = Tree::parse(&ch[3])?;
        let (w, h, span) = (tree.w, tree.h, tree.span);
        let (bmin, bmax) = (xy_at(ch, 1)?, xy_at(ch, 2)?);
        let cs = self.vis.cell_size(w, h);
        let [c0x, c0y, c1x, c1y] = self.vis.cell_range(bmin, bmax, w, h);
        for (i, v) in [bmin[0], bmin[1], bmax[0], bmax[1]].into_iter().enumerate() {
            let cell = (f64::from(v) - self.vis.min[i & 1]) / cs[i & 1];
            if (cell - cell.round_ties_even()).abs() > 1e-6 {
                self.problem("los", where_, "source bbox not cell aligned");
                return Ok(());
            }
        }
        let win = [c0x.max(0), c0y.max(0), c1x.min(w), c1y.min(h)];
        let tb = tree_bbox(&tree.root, span)?;
        if let Some(t) = tb {
            if t[0] < win[0] || t[1] < win[1] || t[2] >= win[2] || t[3] >= win[3] {
                self.problem("los", where_, format!("set cells {t:?} outside bbox cells {win:?}"));
                return Ok(());
            }
        }
        bump(self.los_stats.entry(parent.to_string()).or_default(), "records", 1);
        let nb: [i64; 4];
        let mut new_tree: Option<Node> = None;
        match tb {
            None => {
                let mid = [(f64::from(bmin[0]) + f64::from(bmax[0])) / 2.0, (f64::from(bmin[1]) + f64::from(bmax[1])) / 2.0];
                let (lo, hi) = match self.owner(mid, None, None) {
                    Ok(chart) => (self.plan.xy(bmin.map(f64::from), chart), self.plan.xy(bmax.map(f64::from), chart)),
                    Err(e) => {
                        self.problem("los", where_, format!("empty LOS bbox unowned: {}", e.0));
                        return Ok(());
                    }
                };
                let n0: [i64; 2] = std::array::from_fn(|i| (((lo[i] - self.vis.min[i]) / cs[i] + 1e-9).floor() as i64).max(0));
                let lim = [w, h];
                let n1: [i64; 2] = std::array::from_fn(|i| (((hi[i] - self.vis.min[i]) / cs[i] - 1e-9).ceil() as i64).min(lim[i]));
                nb = [n0[0], n0[1], n1[0], n1[1]];
                bump(self.los_stats.entry(parent.to_string()).or_default(), "empty_records", 1);
            }
            Some(t) => {
                let pad = [t[0] - win[0], t[1] - win[1], win[2] - 1 - t[2], win[3] - 1 - t[3]];
                let res = self.vis.resample(&tree.root, span, w, h, win)?;
                let st = self.los_stats.entry(parent.to_string()).or_default();
                bump(st, "source_cells", res.stats.source_cells);
                bump(st, "source_cells_in_crops", res.stats.source_cells_in_crops);
                bump(st, "source_cells_represented", res.stats.source_cells_represented);
                bump(st, "dest_cells", res.stats.dest_cells);
                if res.stats.source_cells_in_crops < res.stats.source_cells {
                    bump(st, "records_with_clipped_cells", 1);
                }
                if res.stats.source_cells_represented < res.stats.source_cells_in_crops {
                    bump(st, "records_with_unrepresented_cells", 1);
                }
                let Some(dest) = res.dest.filter(|d| d.count() > 0) else {
                    self.problem("los", where_, format!("LOS has {} source cells and none survive the crops", res.stats.source_cells));
                    return Ok(());
                };
                let bb = dest.bbox().expect("non-empty destination");
                let [ox, oy] = res.origin;
                let t2 = [bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy];
                nb = [(t2[0] - pad[0]).max(0), (t2[1] - pad[1]).max(0), (t2[2] + pad[2] + 1).min(w), (t2[3] + pad[3] + 1).min(h)];
                let out = Tree { root: encode_window(&dest, ox, oy, span), ..tree.clone() };
                new_tree = Some(out.to_node());
            }
        }
        let lo = self.vis.cell_world(nb[0], nb[1], w, h);
        let hi = self.vis.cell_world(nb[2], nb[3], w, h);
        set_xy(ch, 1, [lo[0] as f32, lo[1] as f32])?;
        set_xy(ch, 2, [hi[0] as f32, hi[1] as f32])?;
        if let Some(t) = new_tree {
            ch[3] = t;
        }
        bump(&mut self.counts, "LINE_OF_SIGHT.active", 1);
        Ok(())
    }

    /// `encode(decode(tree))` must reproduce every source tree exactly (proves the canonical stock tree shape).
    fn codec_check(&mut self, root: &Node) -> Result<()> {
        let mut checked = 0u64;
        let mut failures = 0u64;
        let mut first_error: Option<anyhow::Error> = None;
        for_each_record(root, "LINE_OF_SIGHT", &mut |n| {
            let Node::Record { children, .. } = n else { return };
            if !matches!(children.first(), Some(Node::Value(Value::Bool(true)))) || children.len() != 4 {
                return;
            }
            let run = || -> Result<bool> {
                let tree = Tree::parse(&children[3])?;
                let (b0, b1) = (xy_at(children, 1)?, xy_at(children, 2)?);
                let c = self.vis.cell_range(b0, b1, tree.w, tree.h);
                let win = [c[0].max(0), c[1].max(0), c[2].min(tree.w), c[3].min(tree.h)];
                let arr = decode_window(&tree.root, tree.span, win)?;
                Ok(encode_window(&arr, win[0], win[1], tree.span) == tree.root)
            };
            match run() {
                Ok(same) => {
                    checked += 1;
                    failures += u64::from(!same);
                }
                Err(e) => {
                    first_error.get_or_insert(e);
                }
            }
        });
        if let Some(e) = first_error {
            return Err(e.context("LINE_OF_SIGHT tree cannot be decoded"));
        }
        self.counts.insert("codec_check.trees".into(), checked);
        if failures > 0 {
            self.problem("codec", "LINE_OF_SIGHT", format!("{failures}/{checked} source trees do not round-trip through the encoder"));
        }
        Ok(())
    }

    fn hlci(&mut self, ch: &mut [Node], where_: &str) -> Result<()> {
        let rid = u32_at(ch, 0)?;
        let Some(key) = self.cai_region_key.get(&rid).filter(|k| self.contract.region(k).is_some()).cloned() else {
            self.problem("hlci", where_, format!("region id {rid} not in contract"));
            return Ok(());
        };
        let old_areas = u32_list(ch, 2)?;
        let new_areas = self.contract.new_areas(&key, &old_areas)?;
        if new_areas.is_empty() {
            self.problem("hlci_group_vanished", where_, format!("region {key}: areas {old_areas:?} have no output geometry"));
            return Ok(());
        }
        let (rx, ry) = (i32_at(ch, 3)?, i32_at(ch, 4)?);
        let pt_src = [f64::from(rx) / Q, f64::from(ry) / Q];
        let sri = *self.src_regions.index.get(&key).with_context(|| format!("source regions lack {key}"))?;
        let ori = *self.out_regions.index.get(&key).with_context(|| format!("output regions lack {key}"))?;
        let src_inside = old_areas.iter().any(|&a| self.src_regions.near(sri, a as usize, pt_src, 1e-3));
        let moved = self.owner(pt_src, None, Some((rx, ry))).and_then(|c| self.fixed_out(c, rx, ry));
        let (nx, ny, mode) = match moved {
            Ok((x, y)) if !src_inside || new_areas.iter().any(|&a| self.out_regions.near(ori, a as usize, [f64::from(x) / Q, f64::from(y) / Q], 0.05)) => {
                (x, y, if src_inside { "transformed" } else { "transformed_source_not_in_group" })
            }
            _ => {
                let Some(ip) = new_areas.iter().find_map(|&a| self.contract.interior_point(&key, a)) else {
                    self.problem("hlci", where_, format!("region {key}: point {pt_src:?} has no owner/geometry and no interior point"));
                    return Ok(());
                };
                let (qx, qy) = ((ip[0] * Q).round_ties_even() as i64, (ip[1] * Q).round_ties_even() as i64);
                if !((self.q_lo[0]..=self.q_hi[0]).contains(&qx) && (self.q_lo[1]..=self.q_hi[1]).contains(&qy)) {
                    self.problem("hlci", where_, "interior point outside world");
                    return Ok(());
                }
                let (nx, ny) = (qx as i32, qy as i32);
                self.hlci_relocated.push(json!({"region": key, "source_point": pt_src, "new_point": [f64::from(nx) / Q, f64::from(ny) / Q], "new_areas": new_areas}));
                (nx, ny, "relocated_to_interior_point")
            }
        };
        set_i32(ch, 3, nx)?;
        set_i32(ch, 4, ny)?;
        set_array(ch, 2, 0x48, u32_list_bytes(&new_areas))?;
        bump(&mut self.counts, format!("CAI_REGION_HLCI.{mode}"), 1);
        bump(&mut self.counts, "CAI_REGION_HLCI.area_list_changed", u64::from(old_areas != new_areas));
        Ok(())
    }

    fn cai_region(&mut self, ch: &mut [Node], where_: &str) -> Result<()> {
        let (x, y) = (f32_at(ch, 4)?, f32_at(ch, 5)?);
        let key = str_at(ch, CAI_REGION_KEY)?.to_string();
        if (x, y) == (0.0, 0.0) {
            // Wilderness centres remain unset even when activating a neighbour adds boundary links.
            // Settlement, world-region and governorship references still require a real centre.
            let listed = |i: usize| array_len(ch, i).map(|n| n > 0);
            if listed(3)? || listed(6)? || listed(13)? || bool_at(ch, 8)?
                || u32_at(ch, 2)? != 0 || u32_at(ch, 11)? != 0 || u32_at(ch, 12)? != 0
            {
                self.problem("cai_region", where_, format!("{key}: (0,0) centre but region has active settlement/world-region/governorship references or flag"));
                return Ok(());
            }
            self.zero_center_regions.push(key);
            bump(&mut self.sentinels, "CAI_REGION.unset_centre", 1);
            return Ok(());
        }
        let moved = self.owner([f64::from(x), f64::from(y)], None, None).and_then(|c| self.out_xy(c, [f64::from(x), f64::from(y)]));
        match moved {
            Ok(p) => {
                set_f32(ch, 4, p[0] as f32)?;
                set_f32(ch, 5, p[1] as f32)?;
                bump(&mut self.counts, "CAI_REGION.xy_points", 1);
            }
            Err(e) => self.problem("unmapped_point", where_, format!("{key}: {}", e.0)),
        }
        Ok(())
    }

    /// Delete HLCI groups whose areas all lack output geometry. Removes only the `CAI_WORLD_REGION_HLCIS` item and the
    /// id inside its CAI region's HLCI id list (the one proven reference); survivors keep ids and order. A region left
    /// without any HLCI, or an id used as a real (non index-valued) trade-route-segment reference, is a blocker.
    fn drop_vanished_hlci(&mut self, root: &mut Node) -> Result<()> {
        let cw = find_named_mut(root, "CAI_WORLD").context("start position has no CAI_WORLD")?;
        let mut doomed: BTreeMap<u32, (u32, String, Vec<u32>)> = BTreeMap::new();
        let mut per_region: BTreeMap<u32, i64> = BTreeMap::new();
        for item in world_items(cw, "CAI_WORLD_REGION_HLCIS")? {
            let rec = find_record(item, "CAI_REGION_HLCI")?;
            let rid = u32_at(rec, 0)?;
            *per_region.entry(rid).or_default() += 1;
            let key = self.cai_region_key.get(&rid).with_context(|| format!("HLCI of unknown CAI region id {rid}"))?;
            let old = u32_list(rec, 2)?;
            if self.contract.region(key).is_some() && self.contract.new_areas(key, &old)?.is_empty() {
                doomed.insert(item_id(item)?, (rid, key.clone(), old));
            }
        }
        for (rid, _, _) in doomed.values() {
            *per_region.entry(*rid).or_default() -= 1;
        }
        let affected: BTreeSet<u32> = doomed.values().map(|d| d.0).collect();
        for rid in affected {
            if per_region[&rid] <= 0 {
                let key = self.cai_region_key[&rid].clone();
                self.problem("hlci_region_empty", key, "all area groups of this region vanish; the region would keep no HLCI");
            }
        }
        for (k, item) in world_items(cw, "CAI_WORLD_TRADE_ROUTE_SEGMENTS")?.iter().enumerate() {
            let rec = find_record(item, "CAI_TRADE_ROUTE_SEGMENT")?;
            let reference = u32_at(rec, 1)?;
            if doomed.contains_key(&reference) && reference as usize != k {
                self.problem("hlci_referenced", format!("CAI_TRADE_ROUTE_SEGMENT[{k}]"), format!("references deleted HLCI {reference}"));
            }
        }
        if !doomed.is_empty() {
            world_items_mut(cw, "CAI_WORLD_REGION_HLCIS")?.retain(|item| item_id(item).is_ok_and(|id| !doomed.contains_key(&id)));
            for item in world_items_mut(cw, "CAI_WORLD_REGIONS")? {
                let reg = find_record_mut(item, "CAI_REGION")?;
                let ids = u32_list(reg, 1)?;
                let keep: Vec<u32> = ids.iter().copied().filter(|i| !doomed.contains_key(i)).collect();
                if keep.len() != ids.len() {
                    set_array(reg, 1, 0x48, u32_list_bytes(&keep))?;
                }
            }
        }
        self.deleted_hlci_ids = doomed.keys().copied().collect();
        self.deleted_hlci = doomed.iter().map(|(h, (_, k, a))| json!({"hlci_id": h, "region": k, "source_areas": a})).collect();
        self.counts.insert("CAI_REGION_HLCI.deleted_groups".into(), doomed.len() as u64);
        Ok(())
    }

    // ------------------------------------------------------------------------------------------ traversal
    fn scan(&mut self, items: &[Node], pname: &str, path: &[String]) -> Result<()> {
        for (at, entry) in items.iter().enumerate() {
            let Node::Value(v) = entry else { continue };
            if matches!(v, Value::XY(_) | Value::XYZ(_) | Value::Array { code: 0x4C | 0x4D, .. }) && !POINT_RECORDS.contains(&pname) {
                self.problem("unclassified_field", path.join("/"), format!("unclassified spatial value (code {:#04x}) at {at}", v.code()));
            }
            if let Value::I32(x) = v {
                let handled = FIXED_PAIRS.iter().any(|(n, _)| *n == pname)
                    || matches!(pname, "CAI_REGION_HLCI" | "CAI_SITUATED" | "CAI_TRADING_POST")
                    || KNOWN_I32_IDS.contains(&(pname, at));
                if x.unsigned_abs() >= (1 << 22) && *x != SENT && !handled {
                    let segment_handle = pname == "TRADE_SEGMENTS[]" && at == 2 + 4 * u32_at(items, 1)? as usize + 9;
                    if !segment_handle {
                        self.problem("unclassified_field", path.join("/"), format!("large i32 field {at} not a known id"));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn visit(&mut self, node: &mut Node, path: &mut Vec<String>, ctx: &mut Option<RegionCtx>) -> Result<()> {
        let (pname, is_arr) = match &*node {
            Node::Value(_) => return Ok(()),
            Node::Record { name, .. } => (name.clone(), false),
            Node::Array { name, .. } => (format!("{name}[]"), true),
        };
        let is_region = pname == "REGION" && path.last().is_some_and(|p| p == "REGIONS_ARRAY[]");
        let mut own: Option<RegionCtx> = None;
        if is_region {
            own = Some(RegionCtx::new(node.children()?)?);
        }
        let ctx: &mut Option<RegionCtx> = if is_region { &mut own } else { ctx };
        path.push(pname.clone());
        let kind = Kind::of(&pname);
        let where_ = where_of(path);
        match &*node {
            Node::Record { children, .. } => self.scan(children, &pname, path)?,
            Node::Array { items, .. } => {
                for item in items {
                    self.scan(item, &pname, path)?;
                }
            }
            Node::Value(_) => {}
        }
        match kind {
            Kind::Hlci => self.hlci(node.children_mut()?, &where_)?,
            Kind::Fixed(ats) => {
                let parent = path.len().checked_sub(2).map(|i| path[i].as_str());
                let sentinel = (pname == "SIEGEABLE_GARRISON_RESIDENCE" && parent == Some("REGION_SLOT")) || (pname == "LOCOMOTABLE" && parent == Some("CHARACTER"));
                let ch = node.children_mut()?;
                for &at in ats {
                    self.fixed_pair(&pname, ch, at, ctx, path, sentinel)?;
                }
                if pname == "LOCOMOTABLE" && !(is_empty_u32_array(ch, 17) && is_empty_u32_array(ch, 18)) {
                    self.problem("locomotable", where_.clone(), "nonempty locomotion path requires an explicit decoder");
                }
            }
            Kind::Character => {
                if !is_empty_u32_array(node.children()?, 12) {
                    self.problem("character", where_.clone(), "nonempty character cached point array requires a decoder");
                }
            }
            Kind::CaiRegion => self.cai_region(node.children_mut()?, &where_)?,
            Kind::Boundary => self.distance(node.children_mut()?, 2, "CAI_REGION_BOUNDARY")?,
            Kind::Camera => {
                let ch = node.children_mut()?;
                let [x, height, y] = xyz_at(ch, 2)?;
                let moved = self.owner([f64::from(x), f64::from(y)], None, None).and_then(|c| self.out_xy(c, [f64::from(x), f64::from(y)]));
                match moved {
                    Ok(p) => {
                        set_xyz(ch, 2, [p[0] as f32, height, p[1] as f32])?;
                        bump(&mut self.counts, "CAMPAIGN_CAMERA.xyz_points", 1);
                    }
                    Err(e) => self.problem("unmapped_point", where_.clone(), e.0),
                }
            }
            Kind::Episodic => {
                let ch = node.children_mut()?;
                for (at, v) in (9..13).zip(self.new_bounds) {
                    set_f32(ch, at, v)?;
                }
                bump(&mut self.counts, "EPISODIC_RESTRICTIONS.bounds", 1);
            }
            Kind::Los => {
                let parent = path.len().checked_sub(2).map(|i| path[i].clone()).unwrap_or_default();
                self.line_of_sight(node, &where_, &parent)?;
                path.pop();
                return Ok(());
            }
            Kind::Pending => {
                let ch = node.children_mut()?;
                for at in 0..ch.len() {
                    if let Ok(p) = xy_at(ch, at) {
                        if p == [0.0, 0.0] && (0..4).all(|i| matches!(ch.get(i), Some(Node::Value(Value::I32(0))))) {
                            bump(&mut self.sentinels, "PENDING_BATTLE.unset_position", 1);
                        } else {
                            self.point12("PENDING_BATTLE", ch, at, None, &where_)?;
                        }
                    }
                }
            }
            Kind::TradeNodes => {
                for item in node.items_mut()? {
                    self.point12("TRADE_NODES[]", item, 0, None, &where_)?;
                }
            }
            Kind::TradeSegments => {
                for (idx, item) in node.items_mut()?.iter_mut().enumerate() {
                    self.trade_segment(item, idx)?;
                }
                path.pop();
                return Ok(());
            }
            Kind::TradeRoutes => {
                for item in node.items_mut()? {
                    ensure!(item.len() == 4, "unexpected trade route shape");
                    self.distance(item, 3, "TRADE_ROUTES[]")?;
                }
            }
            Kind::Grid => self.cached_polygons(node.items_mut()?)?,
            Kind::Shroud => {
                let tree = Tree::parse(node)?;
                let parent = path.len().checked_sub(2).map(|i| path[i].as_str());
                if tree.root != Cell::Leaf(0, 0) || parent != Some("CAMPAIGN_SHROUD") {
                    self.problem("shroud", where_.clone(), "non-empty or unexpected free-standing visibility tree needs an explicit decoder");
                }
                bump(&mut self.counts, "CAMPAIGN_SHROUD.empty_trees", 1);
                path.pop();
                return Ok(());
            }
            Kind::Obstacles => {
                if !node.items()?.is_empty() {
                    self.problem("obstacles", where_.clone(), "unexpected owned obstacle records in this start position");
                }
            }
            Kind::Other => {}
        }
        match node {
            Node::Record { children, .. } => {
                for c in children.iter_mut() {
                    self.visit(c, path, ctx)?;
                }
            }
            Node::Array { items, .. } => {
                for c in items.iter_mut().flatten() {
                    self.visit(c, path, ctx)?;
                }
            }
            Node::Value(_) => {}
        }
        path.pop();
        let _ = is_arr;
        if is_region {
            if let Some(c) = &own {
                match c.charts.len() {
                    0 => {}
                    1 => bump(&mut self.counts, format!("region_chart.{}", c.charts.iter().next().unwrap()), 1),
                    _ => self.problem("region_straddles_charts", c.key.clone(), format!("settlement/slots map to charts {:?}", c.charts)),
                }
            }
        }
        Ok(())
    }

    /// Convert the whole start position in place.
    pub fn transform(&mut self, root: &mut Node) -> Result<()> {
        self.codec_check(root)?;
        self.drop_vanished_hlci(root)?;
        self.visit(root, &mut Vec::new(), &mut None)
    }
}

/// CAI region id -> region key (`CAI_WORLD_REGIONS[]`).
pub fn collect_region_keys(root: &Node) -> Result<HashMap<u32, String>> {
    let world = find_named(root, "CAI_WORLD").context("start position has no CAI_WORLD")?;
    let mut out = HashMap::new();
    for item in world_items(world, "CAI_WORLD_REGIONS")? {
        let id = item_id(item)?;
        let reg = find_record(item, "CAI_REGION")?;
        out.insert(id, str_at(reg, CAI_REGION_KEY)?.to_string());
    }
    Ok(out)
}

/// Problems grouped for an actionable failure message.
pub fn problem_summary(problems: &[Problem], limit: usize) -> String {
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for p in problems {
        *by_kind.entry(p.kind).or_default() += 1;
    }
    let mut s = format!("atlas startpos blocked: {} problems {by_kind:?}", problems.len());
    for p in problems.iter().take(limit) {
        s.push_str(&format!("\n  [{}] {}: {}", p.kind, p.at, p.detail));
    }
    if problems.len() > limit {
        s.push_str(&format!("\n  ... and {} more", problems.len() - limit));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regionmap::regions::{Area, Outline, Region};

    fn u(v: u32) -> Node {
        Node::Value(Value::U32(v))
    }
    fn i(v: i32) -> Node {
        Node::Value(Value::I32(v))
    }
    fn arr(ids: &[u32]) -> Node {
        Node::Value(Value::Array { code: 0x48, data: u32_list_bytes(ids) })
    }

    /// One square single-area region `r` centred on `c` (world units).
    fn square(c: [f64; 2], h: f64) -> RegionSet {
        let p = |dx: f64, dy: f64| [(c[0] + dx) as f32, (c[1] + dy) as f32];
        RegionSet::new(RegionData {
            vertices: vec![p(-h, -h), p(h, -h), p(h, h), p(-h, h)],
            regions: vec![Region {
                key: "r".into(),
                kind: "land".into(),
                areas: vec![Area {
                    land_bridge: false,
                    channel_coast: false,
                    passable: true,
                    adjoining_id: 0,
                    faces: vec![],
                    outlines: vec![Outline { outer: true, ring: vec![0, 1, 2, 3], runs: vec![] }],
                    impassable_id: 65535,
                    island_id: 0,
                }],
                tail: 0,
                settlement: None,
            }],
            theatres: vec![],
        })
    }

    fn contract(fragments: Json) -> Contract {
        Contract::parse(&json!({"format": 1, "region_data": {
            "source_region_count": 1, "output_region_count": 1, "region_order_preserved": true,
            "regions": [{"key": "r", "output_index": 0, "source_areas": 1, "output_areas": 1,
                         "area_fragments": fragments, "area_source": [0], "interior_point": {"0": [1.5, 2.5]}}]}}))
        .unwrap()
    }

    /// slot at `slot` (raw Q20) + one CAI region `r` whose single HLCI sits at `hlci` (raw Q20).
    fn fixture(slot: (i32, i32), hlci: (i32, i32)) -> Node {
        let mut region = vec![u(0); 14];
        region[1] = arr(&[100]);
        for at in [3, 6, 7, 13] {
            region[at] = arr(&[]);
        }
        region[4] = Node::Value(Value::F32(100.0));
        region[5] = Node::Value(Value::F32(300.0));
        region[8] = Node::Value(Value::Bool(false));
        region[10] = Node::Value(Value::String16("r".into()));
        let hlci_item = vec![
            Node::record("PROPSET", 0, vec![]),
            u(100),
            Node::record("CAI_REGION_HLCI", 0, vec![u(7), u(1), arr(&[0]), i(hlci.0), i(hlci.1)]),
        ];
        let world = Node::record(
            "CAI_WORLD",
            0,
            vec![
                Node::array("CAI_WORLD_REGIONS", 0, vec![vec![u(7), Node::record("CAI_REGION", 0, region)]]),
                Node::array("CAI_WORLD_REGION_HLCIS", 0, vec![hlci_item]),
                Node::array("CAI_WORLD_TRADE_ROUTE_SEGMENTS", 0, vec![]),
            ],
        );
        let mut episodic = vec![Node::Value(Value::F32(0.0)); 13];
        for (at, v) in (9..13).zip([-780.0f32, -250.0, 940.0, 540.0]) {
            episodic[at] = Node::Value(Value::F32(v));
        }
        let slot = Node::record("REGION_SLOT", 0, vec![u(0), u(0), u(0), u(0), u(0), u(0), i(slot.0), i(slot.1)]);
        Node::record("ROOT", 0, vec![slot, world, Node::record("EPISODIC_RESTRICTIONS", 0, episodic)])
    }

    const RAW: (i32, i32) = (100 << 20, 300 << 20); // world (100, 300): inside the Europe theatre

    fn run(root: &mut Node, plan: &Plan, c: &Contract, out: RegionSet, overrides: Json) -> Vec<Problem> {
        let ov = PointOverrides::parse(&overrides, plan).unwrap();
        let mut conv = Conv::new(plan, ov, c, square([100.0, 300.0], 20.0), out, root, vec![]).unwrap();
        conv.transform(root).unwrap();
        conv.problems
    }

    fn no_overrides(plan: &Plan) -> Json {
        json!({"format": 1, "source_fixed_bits": 20, "fixed_bits": 20, "scale": plan.scale, "charts": {}})
    }

    fn slot_point(root: &Node) -> (i32, i32) {
        let Node::Record { children, .. } = root else { panic!() };
        let ch = record_children(&children[0], "REGION_SLOT").unwrap();
        (i32_at(ch, 6).unwrap(), i32_at(ch, 7).unwrap())
    }

    #[test]
    fn neighbour_boundaries_preserve_unset_wilderness_centres_but_active_regions_need_centres() {
        let plan = Plan::proven();
        let c = contract(json!([[0]]));
        let out = plan.xy([100.0, 300.0], plan.chart("eur_ind").unwrap());
        for active in [false, true] {
            let mut root = fixture(RAW, RAW);
            let ch = record_children_mut(find_named_mut(&mut root, "CAI_REGION").unwrap(), "CAI_REGION").unwrap();
            ch[4] = Node::Value(Value::F32(0.0));
            ch[5] = Node::Value(Value::F32(0.0));
            ch[7] = arr(&[200]);
            if active {
                ch[11] = u(123);
            }
            let problems = run(&mut root, &plan, &c, square(out, 30.0), no_overrides(&plan));
            assert_eq!(problems.is_empty(), !active);
            if !active {
                let ch = record_children(find_named(&root, "CAI_REGION").unwrap(), "CAI_REGION").unwrap();
                assert_eq!((f32_at(ch, 4).unwrap(), f32_at(ch, 5).unwrap()), (0.0, 0.0));
                assert_eq!(u32_list(ch, 7).unwrap(), vec![200]);
            }
        }
    }

    #[test]
    fn fixed_points_follow_their_owning_chart_and_navigation_overrides_win() {
        let plan = Plan::proven();
        let chart = plan.chart("2").unwrap();
        let expected = plan.fixed_xy([RAW.0, RAW.1], chart).unwrap();
        let c = contract(json!([[0]]));
        let out_centre = [f64::from(expected[0]) / Q, f64::from(expected[1]) / Q];

        let mut root = fixture(RAW, RAW);
        assert!(run(&mut root, &plan, &c, square(out_centre, 20.0), no_overrides(&plan)).is_empty());
        assert_eq!(slot_point(&root), (expected[0], expected[1]));

        // a navigation override replaces the generic rounding of exactly that source coordinate on that chart
        let mut ov = no_overrides(&plan);
        let mut xs = serde_json::Map::new();
        xs.insert(RAW.0.to_string(), json!(expected[0] + 1));
        ov["charts"] = json!({"eur_ind": {"x": xs, "y": {}}});
        let mut root = fixture(RAW, RAW);
        assert!(run(&mut root, &plan, &c, square(out_centre, 20.0), ov).is_empty());
        assert_eq!(slot_point(&root), (expected[0] + 1, expected[1]));
    }

    #[test]
    fn points_in_no_chart_fail_closed_instead_of_being_scaled() {
        let plan = Plan::proven();
        let c = contract(json!([[0]]));
        let mut root = fixture((-1200 << 20, -600 << 20), RAW);
        let problems = run(&mut root, &plan, &c, square([0.0, 0.0], 5.0), no_overrides(&plan));
        assert!(problems.iter().any(|p| p.kind == "unmapped_point" && p.detail.contains("no theatre rect or crop")), "{problems:?}");
    }

    #[test]
    fn hlci_inside_the_output_geometry_is_moved_and_outside_it_goes_to_the_contract_interior_point() {
        let plan = Plan::proven();
        let chart = plan.chart("2").unwrap();
        let expected = plan.fixed_xy([RAW.0, RAW.1], chart).unwrap();
        let c = contract(json!([[0]]));
        let hlci_xy = |root: &Node| {
            let Node::Record { children, .. } = root else { panic!() };
            let world = find_named(&children[1], "CAI_WORLD").unwrap();
            let item = &world_items(world, "CAI_WORLD_REGION_HLCIS").unwrap()[0];
            let rec = find_record(item, "CAI_REGION_HLCI").unwrap();
            (i32_at(rec, 3).unwrap(), i32_at(rec, 4).unwrap())
        };

        let mut root = fixture(RAW, RAW);
        let out = square([f64::from(expected[0]) / Q, f64::from(expected[1]) / Q], 20.0);
        assert!(run(&mut root, &plan, &c, out, no_overrides(&plan)).is_empty());
        assert_eq!(hlci_xy(&root), (expected[0], expected[1]));

        // output geometry elsewhere: the group point must not stay outside its areas
        let mut root = fixture(RAW, RAW);
        assert!(run(&mut root, &plan, &c, square([0.0, 0.0], 5.0), no_overrides(&plan)).is_empty());
        assert_eq!(hlci_xy(&root), ((1.5 * Q) as i32, (2.5 * Q) as i32));
    }

    #[test]
    fn a_region_whose_only_hlci_group_vanishes_is_a_blocker() {
        let plan = Plan::proven();
        let c = contract(json!([[]]));
        let mut root = fixture(RAW, RAW);
        let problems = run(&mut root, &plan, &c, square([0.0, 0.0], 5.0), no_overrides(&plan));
        assert!(problems.iter().any(|p| p.kind == "hlci_region_empty"), "{problems:?}");
    }

    #[test]
    fn camera_bounds_keep_the_original_margins_around_the_moved_theatres() {
        let plan = Plan::proven();
        let src = [-780.0f32, -250.0, 940.0, 540.0];
        let nb = camera_bounds(&plan, src).unwrap();
        // margins: theatres span x -760..920, y -230..520 in the source
        let rects: Vec<(&Chart, &Rect)> = plan.charts.iter().flat_map(|c| c.theatre_rects.values().map(move |r| (c, r))).collect();
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for (c, r) in rects {
            for p in [plan.xy(r[0], c), plan.xy(r[1], c)] {
                for a in 0..2 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
        }
        let w = plan.world;
        let expect = [(lo[0] - 20.0).max(w[0][0]), (lo[1] - 20.0).max(w[0][1]), (hi[0] + 20.0).min(w[1][0]), (hi[1] + 20.0).min(w[1][1])];
        for (got, want) in nb.iter().zip(expect) {
            assert!((f64::from(*got) - want).abs() < 1e-3, "{nb:?} vs {expect:?}");
        }
        assert!(camera_bounds(&plan, [-700.0, -250.0, 940.0, 540.0]).is_err(), "bounds that do not cover the theatres must be refused");
    }

    #[test]
    fn rect_union_coverage_detects_gaps() {
        let a: Rect = [[0.0, 0.0], [10.0, 10.0]];
        let b: Rect = [[10.0, 0.0], [20.0, 10.0]];
        assert!(covers(&[a, b], &[[2.0, 2.0], [18.0, 8.0]]));
        assert!(!covers(&[a], &[[2.0, 2.0], [18.0, 8.0]]));
        let gap: Rect = [[11.0, 0.0], [20.0, 10.0]];
        assert!(!covers(&[a, gap], &[[2.0, 2.0], [18.0, 8.0]]));
    }

    #[test]
    fn cached_polygon_payload_must_be_consumed_exactly() {
        let mut raw = Vec::new();
        for pts in [[(1, 2), (3, 4)].as_slice(), [(5, 6)].as_slice()] {
            raw.extend((pts.len() as u32).to_le_bytes());
            for (x, y) in pts {
                raw.extend((*x as i32).to_le_bytes());
                raw.extend((*y as i32).to_le_bytes());
            }
            raw.extend(0u32.to_le_bytes());
        }
        let got = cached_point_offsets(2, &raw).unwrap();
        assert_eq!(got.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(), vec![(1, 2), (3, 4), (5, 6)]);
        assert!(cached_point_offsets(1, &raw).is_err());
        assert!(cached_point_offsets(2, &raw[..raw.len() - 1]).is_err());
    }
}
