//! Clipping of every ring edge of one region dataset against the crop rectangles of each chart, and the
//! reconstruction of the clipped pieces as polygons (accepted POC `regions_atlas_lib.py`).
//!
//! Every undirected edge is clipped once, so the two neighbours of an edge receive bit-identical new vertices.
//! The pieces are rebuilt per (owning source area, chart) from directed edges.

use super::geom::{P, Rect64, RectUnion, norm_bits, rect_distance, rects_overlap_area, ring_interior_point, widen};
use super::source::Source;
use crate::atlas::Plan;
use crate::regionmap::geometry::point_in_ring;
use anyhow::{Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub const INF_RANK: u64 = 1 << 60;
/// chart index of the world-gap polygon
pub const GAP: usize = usize::MAX;

pub type Key = ((usize, usize), usize);

// ---------------------------------------------------------------------------------------------
// plan context
// ---------------------------------------------------------------------------------------------

pub struct Ctx {
    pub plan: Plan,
    /// crop rectangles per chart (source space)
    pub rects: Vec<Vec<Rect64>>,
    pub unions: Vec<RectUnion>,
    pub world_out: [[f32; 2]; 2],
    pub min_world_margin: f64,
    pub min_chart_separation: Option<f64>,
}

impl Ctx {
    pub fn new(plan: &Plan) -> Result<Self> {
        plan.validate()?;
        Self::build(plan)
    }

    /// Geometry checks and derived tables for any plan (the supported-preset check lives in [`Ctx::new`]).
    pub(super) fn build(plan: &Plan) -> Result<Self> {
        let rects: Vec<Vec<Rect64>> = plan.charts.iter().map(|c| c.crop_rects.clone()).collect();
        let unions: Vec<RectUnion> = rects.iter().map(|r| RectUnion::new(r)).collect();
        let world = plan.world;
        let world_out = [[world[0][0] as f32, world[0][1] as f32], [world[1][0] as f32, world[1][1] as f32]];
        let n = plan.charts.len();
        // source crops of different charts must not overlap
        for i in 0..n {
            for j in i + 1..n {
                for a in &rects[i] {
                    for b in &rects[j] {
                        ensure!(
                            rects_overlap_area(*a, *b) <= 1e-9,
                            "regions atlas: source crops of charts {} and {} overlap",
                            plan.charts[i].key,
                            plan.charts[j].key
                        );
                    }
                }
            }
        }
        let mut out_rects: Vec<Vec<Rect64>> = Vec::new();
        let mut min_margin = f64::INFINITY;
        for (i, chart) in plan.charts.iter().enumerate() {
            let moved: Vec<Rect64> = rects[i].iter().map(|r| [plan.xy(r[0], chart), plan.xy(r[1], chart)]).collect();
            for r in &moved {
                let margin = (r[0][0] - world[0][0]).min(world[1][0] - r[1][0]).min(r[0][1] - world[0][1]).min(world[1][1] - r[1][1]);
                ensure!(margin >= 0.0, "regions atlas: chart {} output crop leaves the world {world:?}", chart.key);
                min_margin = min_margin.min(margin);
            }
            out_rects.push(moved);
        }
        let mut min_sep = f64::INFINITY;
        for i in 0..n {
            for j in i + 1..n {
                for a in &out_rects[i] {
                    for b in &out_rects[j] {
                        min_sep = min_sep.min(rect_distance(*a, *b));
                    }
                }
            }
        }
        ensure!(
            min_margin > 1e-3 && (n <= 1 || min_sep > 1e-3),
            "regions atlas: output crops must be strictly separate (world margin {min_margin}, chart separation {min_sep})"
        );
        for (i, chart) in plan.charts.iter().enumerate() {
            for (th, rect) in &chart.theatre_rects {
                ensure!(
                    unions[i].covers_rect(*rect),
                    "regions atlas: theatre {th} rect is not inside the crops of chart {}",
                    chart.key
                );
            }
        }
        Ok(Self {
            plan: plan.clone(),
            rects,
            unions,
            world_out,
            min_world_margin: min_margin,
            min_chart_separation: if n > 1 { Some(min_sep) } else { None },
        })
    }

    /// Chart output coordinate of a source point, rounded to f32.
    pub fn oxy(&self, ci: usize, p: P) -> [f32; 2] {
        let q = self.plan.xy(p, &self.plan.charts[ci]);
        [q[0] as f32, q[1] as f32]
    }

    pub fn chart_key(&self, ci: usize) -> String {
        if ci == GAP { "gap".to_string() } else { self.plan.charts[ci].key.clone() }
    }

    /// Index of the chart that owns a source point (theatre rectangles first, then crop rectangles).
    pub fn chart_of_point(&self, p: P) -> Result<usize> {
        let chart = self.plan.chart_at(p, true)?;
        self.chart_index(&chart.key)
    }

    pub fn chart_index(&self, key: &str) -> Result<usize> {
        self.plan.charts.iter().position(|c| c.key == key).ok_or_else(|| anyhow::anyhow!("regions atlas: unknown chart {key}"))
    }
}

// ---------------------------------------------------------------------------------------------
// Liang-Barsky
// ---------------------------------------------------------------------------------------------

type Side = (usize, f64);

#[derive(Clone, Copy, Debug)]
struct Interval {
    t0: f64,
    t1: f64,
    s0: Option<Side>,
    s1: Option<Side>,
}

fn lb(a: P, b: P, rect: Rect64) -> Option<Interval> {
    let [[x0, y0], [x1, y1]] = rect;
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let (mut s0, mut s1): (Option<Side>, Option<Side>) = (None, None);
    for (p, q, side) in [
        (-dx, a[0] - x0, (0usize, x0)),
        (dx, x1 - a[0], (0usize, x1)),
        (-dy, a[1] - y0, (1usize, y0)),
        (dy, y1 - a[1], (1usize, y1)),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                if t > t0 {
                    t0 = t;
                    s0 = Some(side);
                }
            } else if t < t1 {
                t1 = t;
                s1 = Some(side);
            }
        }
    }
    if t0 >= t1 {
        return None;
    }
    Some(Interval { t0, t1, s0, s1 })
}

fn edge_intervals(a: P, b: P, rects: &[Rect64]) -> Vec<Interval> {
    let mut iv: Vec<Interval> = rects.iter().filter_map(|r| lb(a, b, *r)).collect();
    iv.sort_by(|x, y| (x.t0, x.t1).partial_cmp(&(y.t0, y.t1)).unwrap());
    let mut merged: Vec<Interval> = Vec::new();
    for cur in iv {
        match merged.last_mut() {
            Some(last) if cur.t0 <= last.t1 => {
                if cur.t1 > last.t1 {
                    last.t1 = cur.t1;
                    last.s1 = cur.s1;
                }
            }
            _ => merged.push(cur),
        }
    }
    merged
}

fn xpoint(a: P, b: P, side: Side) -> P {
    let (k, val) = side;
    let o = 1 - k;
    let t = (val - a[k]) / (b[k] - a[k]);
    let mut p = [0.0; 2];
    p[k] = val;
    p[o] = a[o] + t * (b[o] - a[o]);
    p
}

fn cross2(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ax * by - ay * bx
}

// ---------------------------------------------------------------------------------------------
// clipper
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NodeKey {
    /// source vertex id on chart
    V(u32, usize),
    /// crossing: source edge (lo, hi), chart, interval index, end
    X(u32, u32, usize, usize, u8),
    /// crop-rectangle corner: chart, ring, position
    C(usize, usize, usize),
    /// world corner
    W(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub n0: usize,
    pub n1: usize,
    pub rank: u64,
    pub from_src: bool,
}

#[derive(Clone, Debug)]
pub struct Frag {
    pub ci: usize,
    pub outer: Vec<usize>,
    pub holes: Vec<Vec<usize>>,
    pub area: f64,
    pub unchanged: bool,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Debug)]
pub struct Absorbed {
    pub owner: (usize, usize),
    pub chart: String,
    pub output_area: f64,
    pub into: (usize, usize),
    pub shared_border: f64,
}

pub struct Clipper<'a> {
    pub ctx: &'a Ctx,
    pub src: &'a Source<'a>,
    node_key: HashMap<NodeKey, usize>,
    key_order: Vec<(NodeKey, usize)>,
    pub nsrc: Vec<P>,
    nchart: Vec<usize>,
    pub nout: Vec<[f32; 2]>,
    xinfo: HashMap<usize, (Side, u32, u32)>,
    sub: HashMap<Key, Vec<(usize, usize, u64)>>,
    pieces: Vec<(usize, usize, (usize, usize), usize)>,
    pub stats: BTreeMap<String, u64>,
    corner_at: HashMap<(usize, u64, u64), usize>,
    node_vid: HashMap<usize, u32>,
    coll: HashSet<(usize, u32, u32)>,
    ring_pts: Vec<Vec<Vec<P>>>,
    pub absorbed: Vec<Absorbed>,
}

impl<'a> Clipper<'a> {
    pub fn new(ctx: &'a Ctx, src: &'a Source<'a>) -> Self {
        Self {
            ctx,
            src,
            node_key: HashMap::new(),
            key_order: Vec::new(),
            nsrc: Vec::new(),
            nchart: Vec::new(),
            nout: Vec::new(),
            xinfo: HashMap::new(),
            sub: HashMap::new(),
            pieces: Vec::new(),
            stats: BTreeMap::new(),
            corner_at: HashMap::new(),
            node_vid: HashMap::new(),
            coll: HashSet::new(),
            ring_pts: Vec::new(),
            absorbed: Vec::new(),
        }
    }

    fn bump(&mut self, key: &str) {
        *self.stats.entry(key.to_string()).or_insert(0) += 1;
    }

    // ---- nodes ----
    fn node(&mut self, key: NodeKey, src_xy: P, ci: usize, out_xy: Option<[f32; 2]>) -> usize {
        if let NodeKey::V(vid, _) = key {
            if let Some(&c) = self.corner_at.get(&(ci, norm_bits(src_xy[0]), norm_bits(src_xy[1]))) {
                self.node_vid.insert(c, vid);
                return c;
            }
        }
        if let Some(&n) = self.node_key.get(&key) {
            return n;
        }
        let n = self.nsrc.len();
        self.node_key.insert(key, n);
        self.key_order.push((key, n));
        self.nsrc.push(src_xy);
        self.nchart.push(ci);
        let out = match out_xy {
            Some(o) => o,
            None => self.ctx.oxy(ci, src_xy),
        };
        self.nout.push(out);
        if let NodeKey::V(vid, _) = key {
            self.node_vid.insert(n, vid);
        }
        n
    }

    fn ring_geometry(&mut self) -> Result<()> {
        let ctx = self.ctx;
        for ci in 0..ctx.unions.len() {
            let rings = ctx.unions[ci].rings()?;
            for (ri, ring) in rings.iter().enumerate() {
                for (k, p) in ring.iter().enumerate() {
                    let n = self.node(NodeKey::C(ci, ri, k), *p, ci, None);
                    self.corner_at.insert((ci, norm_bits(p[0]), norm_bits(p[1])), n);
                }
            }
            self.ring_pts.push(rings);
        }
        Ok(())
    }

    fn collinear_on_ring(&self, ci: usize, a: P, b: P) -> bool {
        for ring in &self.ring_pts[ci] {
            let m = ring.len();
            for k in 0..m {
                let (p, q) = (ring[k], ring[(k + 1) % m]);
                if p[1] == q[1] {
                    if a[1] == p[1] && p[1] == b[1] && p[0].min(q[0]) <= a[0].min(b[0]) && a[0].max(b[0]) <= p[0].max(q[0]) {
                        return true;
                    }
                } else if a[0] == p[0] && p[0] == b[0] && p[1].min(q[1]) <= a[1].min(b[1]) && a[1].max(b[1]) <= p[1].max(q[1]) {
                    return true;
                }
            }
        }
        false
    }

    fn xout(&self, ci: usize, side: Side, other_val: f64) -> [f32; 2] {
        let (k, val) = side;
        let mut pt = [0.0; 2];
        pt[k] = val;
        pt[1 - k] = other_val;
        self.ctx.oxy(ci, pt)
    }

    pub fn run(&mut self) -> Result<()> {
        let src = self.src;
        let ctx = self.ctx;
        let v = &src.v;
        self.ring_geometry()?;
        let canon: Vec<(u32, u32)> = src
            .eo
            .keys()
            .map(|&(i, j)| (i.min(j), i.max(j)))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        *self.stats.entry("undirected_edges".into()).or_insert(0) = canon.len() as u64;
        for ci in 0..ctx.rects.len() {
            let rects = &ctx.rects[ci];
            for &(lo, hi) in &canon {
                let (a, b) = (v[lo as usize], v[hi as usize]);
                let (mnx, mxx) = (a[0].min(b[0]), a[0].max(b[0]));
                let (mny, mxy) = (a[1].min(b[1]), a[1].max(b[1]));
                let mut cand = false;
                let mut full = false;
                for r in rects {
                    let [[x0, y0], [x1, y1]] = *r;
                    cand |= mnx <= x1 && mxx >= x0 && mny <= y1 && mxy >= y0;
                    full |= mnx > x0 && mxx < x1 && mny > y0 && mxy < y1;
                }
                if !cand {
                    continue;
                }
                let ivs: Vec<Interval> = if full {
                    self.bump("edges_inside_one_rect");
                    vec![Interval { t0: 0.0, t1: 1.0, s0: None, s1: None }]
                } else {
                    if self.collinear_on_ring(ci, a, b) {
                        // the edge IS a piece of the chart boundary: the area inside owns it as boundary piece
                        self.node(NodeKey::V(lo, ci), a, ci, None);
                        self.node(NodeKey::V(hi, ci), b, ci, None);
                        self.coll.insert((ci, lo, hi));
                        self.bump("edges_on_crop_boundary");
                        continue;
                    }
                    let ivs = edge_intervals(a, b, rects);
                    if ivs.is_empty() {
                        continue;
                    }
                    self.bump("edges_clipped_or_crossing");
                    ivs
                };
                let dirs = [((lo, hi), src.eo.get(&(lo, hi))), ((hi, lo), src.eo.get(&(hi, lo)))];
                for (q, iv) in ivs.iter().enumerate() {
                    if iv.t1 - iv.t0 <= 0.0 {
                        continue;
                    }
                    let n0 = self.end_node(ci, lo, hi, a, b, q, 0, iv.t0, iv.s0)?;
                    let n1 = self.end_node(ci, lo, hi, a, b, q, 1, iv.t1, iv.s1)?;
                    for ((e, d), (u, w)) in dirs.iter().zip([(n0, n1), (n1, n0)]) {
                        let Some(d) = d else { continue };
                        let rank = src.eo[e].rank;
                        self.sub.entry((d.area(), ci)).or_default().push((u, w, rank));
                    }
                    for (nn, side) in [(n0, iv.s0), (n1, iv.s1)] {
                        if let Some(side) = side {
                            self.xinfo.insert(nn, (side, lo, hi));
                        }
                    }
                }
            }
        }
        self.boundaries()
    }

    #[allow(clippy::too_many_arguments)]
    fn end_node(&mut self, ci: usize, lo: u32, hi: u32, a: P, b: P, q: usize, end: u8, t: f64, side: Option<Side>) -> Result<usize> {
        match side {
            None => {
                let vid = if end == 0 { lo } else { hi };
                if (end == 0 && t != 0.0) || (end == 1 && t != 1.0) {
                    bail!("regions atlas: interval end without side that is not an endpoint");
                }
                let xy = self.src.v[vid as usize];
                Ok(self.node(NodeKey::V(vid, ci), xy, ci, None))
            }
            Some(side) => {
                let p = xpoint(a, b, side);
                let out = self.xout(ci, side, p[1 - side.0]);
                Ok(self.node(NodeKey::X(lo, hi, ci, q, end), p, ci, Some(out)))
            }
        }
    }

    // ---- chart boundaries and the owner of each boundary stretch ----
    fn boundaries(&mut self) -> Result<()> {
        let mut by_chart: HashMap<usize, Vec<usize>> = HashMap::new();
        for (key, n) in &self.key_order {
            if matches!(key, NodeKey::V(..) | NodeKey::X(..)) {
                by_chart.entry(self.nchart[*n]).or_default().push(*n);
            }
        }
        let name = self.src.name;
        for ci in 0..self.ring_pts.len() {
            let cand: Vec<usize> = by_chart.get(&ci).cloned().unwrap_or_default();
            let mut remaining_x: BTreeSet<usize> = cand.iter().copied().filter(|n| self.xinfo.contains_key(n)).collect();
            let rings = self.ring_pts[ci].clone();
            let mut built: Vec<Vec<usize>> = Vec::new();
            for ring in &rings {
                let mut seq: Vec<usize> = Vec::new();
                let m = ring.len();
                for k in 0..m {
                    let (p, q) = (ring[k], ring[(k + 1) % m]);
                    seq.push(self.corner_at[&(ci, norm_bits(p[0]), norm_bits(p[1]))]);
                    let axis = if p[1] == q[1] { 0 } else { 1 };
                    let line_axis = 1 - axis;
                    let val = p[line_axis];
                    let (lo_c, hi_c) = (p[axis].min(q[axis]), p[axis].max(q[axis]));
                    let mut on: Vec<(f64, usize)> = Vec::new();
                    for &n in &cand {
                        let s = self.nsrc[n];
                        if s[line_axis] == val && lo_c < s[axis] && s[axis] < hi_c {
                            on.push(((s[axis] - p[axis]).abs(), n));
                        }
                    }
                    on.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    for (_, n) in on {
                        remaining_x.remove(&n);
                        seq.push(n);
                    }
                }
                built.push(seq);
            }
            ensure!(
                remaining_x.is_empty(),
                "regions atlas: {name}: crossings not on any chart boundary segment: {}",
                remaining_x.len()
            );
            for seq in &built {
                self.ring_owners(ci, seq)?;
            }
        }
        Ok(())
    }

    /// (before, after) owners of a crossing on a boundary walked in direction `dvec`.
    fn owner_after_before(&self, n: usize, dvec: P) -> Result<((usize, usize), (usize, usize))> {
        let (_, lo, hi) = self.xinfo[&n];
        let src = self.src;
        let (l, r, ev) = if let Some(e) = src.eo.get(&(lo, hi)) {
            (
                e.area(),
                src.eo.get(&(hi, lo)).map(|x| x.area()),
                [src.v[hi as usize][0] - src.v[lo as usize][0], src.v[hi as usize][1] - src.v[lo as usize][1]],
            )
        } else {
            let e = src.eo.get(&(hi, lo)).ok_or_else(|| anyhow::anyhow!("regions atlas: crossing edge {lo},{hi} is in no ring"))?;
            (
                e.area(),
                None,
                [src.v[lo as usize][0] - src.v[hi as usize][0], src.v[lo as usize][1] - src.v[hi as usize][1]],
            )
        };
        let Some(r) = r else {
            bail!("regions atlas: {}: crossing edge {lo},{hi} has no neighbour (area {l:?}) on a chart boundary", src.name);
        };
        let s = cross2(dvec[0], dvec[1], ev[0], ev[1]);
        ensure!(s != 0.0, "regions atlas: {}: edge {lo},{hi} is collinear with a crop boundary", src.name);
        Ok(if s > 0.0 { (l, r) } else { (r, l) })
    }

    fn ring_owners(&mut self, ci: usize, seq: &[usize]) -> Result<()> {
        let m = seq.len();
        let src = self.src;
        let mut owner: Vec<Option<(usize, usize)>> = vec![None; m];
        for k in 0..m {
            let (a, b) = (seq[k], seq[(k + 1) % m]);
            if let (Some(&va), Some(&vb)) = (self.node_vid.get(&a), self.node_vid.get(&b)) {
                if self.coll.contains(&(ci, va.min(vb), va.max(vb))) {
                    let Some(d) = src.eo.get(&(va, vb)) else {
                        bail!(
                            "regions atlas: {}: boundary edge {va}->{vb} exists only against the ring direction (no area inside chart {} owns it)",
                            src.name,
                            self.ctx.chart_key(ci)
                        );
                    };
                    owner[k] = Some(d.area());
                }
            }
        }
        let mut ba: HashMap<usize, ((usize, usize), (usize, usize))> = HashMap::new();
        for (k, &n) in seq.iter().enumerate() {
            if self.xinfo.contains_key(&n) {
                let nxt = self.nsrc[seq[(k + 1) % m]];
                let cur = self.nsrc[n];
                ba.insert(k, self.owner_after_before(n, [nxt[0] - cur[0], nxt[1] - cur[1]])?);
            }
        }
        for k in 0..m {
            let mut cands: Vec<(usize, usize)> = Vec::new();
            if let Some(x) = ba.get(&k) {
                cands.push(x.1);
            }
            if let Some(x) = ba.get(&((k + 1) % m)) {
                cands.push(x.0);
            }
            if let Some(o) = owner[k] {
                cands.push(o);
            }
            let distinct: BTreeSet<(usize, usize)> = cands.iter().copied().collect();
            if distinct.len() > 1 {
                bail!(
                    "regions atlas: {}: inconsistent owners along chart boundary {}: {cands:?} between {:?} and {:?}",
                    src.name,
                    self.ctx.chart_key(ci),
                    self.nsrc[seq[k]],
                    self.nsrc[seq[(k + 1) % m]]
                );
            }
            if let Some(&c) = cands.first() {
                owner[k] = Some(c);
            }
        }
        let corner_set: HashSet<usize> = self.corner_at.values().copied().collect();
        let mut changed = true;
        while changed {
            changed = false;
            for k in 0..m {
                if owner[k].is_some() {
                    continue;
                }
                if corner_set.contains(&seq[k]) && owner[(k + m - 1) % m].is_some() {
                    owner[k] = owner[(k + m - 1) % m];
                    changed = true;
                } else if corner_set.contains(&seq[(k + 1) % m]) && owner[(k + 1) % m].is_some() {
                    owner[k] = owner[(k + 1) % m];
                    changed = true;
                }
            }
        }
        for k in 0..m {
            if owner[k].is_none() {
                owner[k] = Some(self.locate_inward(seq[k], seq[(k + 1) % m])?);
                self.bump("boundary_pieces_located_by_point");
            }
        }
        for k in 0..m {
            self.pieces.push((seq[k], seq[(k + 1) % m], owner[k].unwrap(), ci));
        }
        Ok(())
    }

    fn locate_inward(&self, na: usize, nb: usize) -> Result<(usize, usize)> {
        let (a, b) = (self.nsrc[na], self.nsrc[nb]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let ln = dx.hypot(dy);
        let d = 1e-3f64.min(ln / 4.0);
        let mid = [(a[0] + b[0]) / 2.0 - dy / ln * d, (a[1] + b[1]) / 2.0 + dx / ln * d];
        self.src.locate(mid)
    }

    // ---- world gap ----
    /// The world rectangle minus every chart crop, as one polygon owned by the backdrop area.
    pub fn build_gap(&mut self) -> Result<Frag> {
        let ww = self.ctx.world_out;
        let corners = [[ww[0][0], ww[0][1]], [ww[1][0], ww[0][1]], [ww[1][0], ww[1][1]], [ww[0][0], ww[1][1]]];
        let mut wn = Vec::new();
        for (k, c) in corners.iter().enumerate() {
            wn.push(self.node(NodeKey::W(k), [f64::NAN; 2], GAP, Some(*c)));
        }
        let mut lst: Vec<Edge> = (0..4).map(|k| Edge { n0: wn[k], n1: wn[(k + 1) % 4], rank: INF_RANK, from_src: false }).collect();
        for &(n0, n1, _, _) in &self.pieces {
            lst.push(Edge { n0: n1, n1: n0, rank: INF_RANK, from_src: false });
        }
        let mut fr = self.trace((usize::MAX, usize::MAX), GAP, &lst)?;
        ensure!(fr.len() == 1, "regions atlas: {}: the gap between the charts is not one polygon ({})", self.src.name, fr.len());
        let mut gap = fr.remove(0);
        gap.unchanged = false;
        Ok(gap)
    }

    // ---- fragments ----
    /// (owner area, chart) -> fragments ordered by decreasing area. Clipped pieces smaller than `min_area_out`
    /// (output units^2) are absorbed into the neighbour sharing the longest border so no tiny polygon is emitted and
    /// the partition stays closed; a piece touching a theatre rectangle is never absorbed (error).
    pub fn fragments(&mut self, min_area_out: f64) -> Result<HashMap<Key, Vec<Frag>>> {
        let mut edges: HashMap<Key, Vec<Edge>> = HashMap::new();
        for (key, lst) in &self.sub {
            let e = edges.entry(*key).or_default();
            for &(n0, n1, rank) in lst {
                e.push(Edge { n0, n1, rank, from_src: true });
            }
        }
        for &(n0, n1, owner, ci) in &self.pieces {
            edges.entry((owner, ci)).or_default().push(Edge { n0, n1, rank: INF_RANK, from_src: false });
        }
        self.absorbed.clear();
        let mut modified: BTreeSet<Key> = BTreeSet::new();
        let mut out: HashMap<Key, Vec<Frag>> = HashMap::new();
        let mut keys: Vec<Key> = edges.keys().copied().collect();
        keys.sort();
        for key in keys {
            let frags = self.trace(key.0, key.1, &edges[&key])?;
            out.insert(key, frags);
        }
        if min_area_out > 0.0 {
            loop {
                let mut small: Vec<(f64, Key, usize)> = Vec::new();
                for (key, frs) in &out {
                    for (i, fr) in frs.iter().enumerate() {
                        if !fr.unchanged && fr.area < min_area_out {
                            small.push((fr.area, *key, i));
                        }
                    }
                }
                if small.is_empty() {
                    break;
                }
                small.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap());
                let (_, key, i) = small[0];
                let fr = out[&key][i].clone();
                self.absorb(&mut edges, &mut out, &mut modified, key, &fr)?;
            }
        }
        for key in &modified {
            if let Some(frs) = out.get_mut(key) {
                for fr in frs {
                    fr.unchanged = false;
                }
            }
        }
        Ok(out)
    }

    fn absorb(
        &mut self,
        edges: &mut HashMap<Key, Vec<Edge>>,
        out: &mut HashMap<Key, Vec<Frag>>,
        modified: &mut BTreeSet<Key>,
        key: Key,
        fr: &Frag,
    ) -> Result<()> {
        let (owner, ci) = key;
        let ctx = self.ctx;
        let pts: Vec<P> = fr.outer.iter().map(|&n| self.nsrc[n]).collect();
        for (th, rect) in &ctx.plan.charts[ci].theatre_rects {
            if polygon_intersects_rect(&pts, *rect) {
                bail!(
                    "regions atlas: {}: clipped piece of area {owner:?} in chart {} (output area {:.6}) touches theatre {th}; refusing to absorb a playable piece",
                    self.src.name,
                    ctx.chart_key(ci),
                    fr.area
                );
            }
        }
        let mut owner_of: HashMap<(usize, usize), Key> = HashMap::new();
        for (k2, lst) in edges.iter() {
            for e in lst {
                owner_of.insert((e.n0, e.n1), *k2);
            }
        }
        let mut share: BTreeMap<Key, f64> = BTreeMap::new();
        for e in &fr.edges {
            if let Some(nb) = owner_of.get(&(e.n1, e.n0)) {
                let (a, b) = (self.nout[e.n0], self.nout[e.n1]);
                let (dx, dy) = (b[0] as f64 - a[0] as f64, b[1] as f64 - a[1] as f64);
                *share.entry(*nb).or_insert(0.0) += dx.hypot(dy);
            }
        }
        share.retain(|k2, _| k2.1 == ci && *k2 != key);
        ensure!(
            !share.is_empty(),
            "regions atlas: {}: clipped piece of area {owner:?} in chart {} has no neighbouring area to absorb it",
            self.src.name,
            ctx.chart_key(ci)
        );
        let mut nb = *share.keys().next().unwrap();
        for (k2, v) in &share {
            let best = share[&nb];
            if *v > best || (*v == best && *k2 > nb) {
                nb = *k2;
            }
        }
        let mine: HashSet<(usize, usize)> = fr.edges.iter().map(|e| (e.n0, e.n1)).collect();
        let mut drop_rev: HashSet<(usize, usize)> = HashSet::new();
        let mut relabel: Vec<Edge> = Vec::new();
        for e in &fr.edges {
            if owner_of.get(&(e.n1, e.n0)) == Some(&nb) {
                drop_rev.insert((e.n1, e.n0));
            } else {
                relabel.push(*e);
            }
        }
        let mut new_nb: Vec<Edge> = edges[&nb].iter().copied().filter(|e| !drop_rev.contains(&(e.n0, e.n1))).collect();
        new_nb.extend(relabel);
        edges.insert(nb, new_nb);
        let rest: Vec<Edge> = edges[&key].iter().copied().filter(|e| !mine.contains(&(e.n0, e.n1))).collect();
        edges.insert(key, rest);
        self.absorbed.push(Absorbed {
            owner,
            chart: ctx.chart_key(ci),
            output_area: fr.area,
            into: nb.0,
            shared_border: share[&nb],
        });
        modified.insert(nb);
        for k2 in [key, nb] {
            if edges[&k2].is_empty() {
                out.remove(&k2);
                edges.remove(&k2);
            } else {
                let frags = self.trace(k2.0, k2.1, &edges[&k2])?;
                out.insert(k2, frags);
            }
        }
        Ok(())
    }

    /// Trace the directed edge list into closed cycles; rebuild outer rings and holes.
    pub fn trace(&mut self, owner: (usize, usize), ci: usize, lst: &[Edge]) -> Result<Vec<Frag>> {
        let nout = &self.nout;
        let name = self.src.name;
        let ck = self.ctx.chart_key(ci);
        let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
        for (k, e) in lst.iter().enumerate() {
            outgoing.entry(e.n0).or_default().push(k);
        }
        let mut used = vec![false; lst.len()];
        let mut cycles: Vec<Vec<usize>> = Vec::new();
        let mut order: Vec<usize> = (0..lst.len()).collect();
        order.sort_by_key(|&k| (lst[k].rank, k));
        let mut pinch = 0u64;
        for start in order {
            if used[start] {
                continue;
            }
            let mut cyc: Vec<usize> = Vec::new();
            let mut k = start;
            while !used[k] {
                used[k] = true;
                cyc.push(k);
                let n1 = lst[k].n1;
                let mut cand: Vec<usize> = outgoing.get(&n1).map(|v| v.iter().copied().filter(|c| !used[*c]).collect()).unwrap_or_default();
                if cand.is_empty() {
                    break;
                }
                if cand.len() > 1 {
                    // rotate clockwise from the reverse of the incoming edge: first edge met
                    let a0 = nout[lst[k].n0];
                    let a1 = nout[n1];
                    let back = (a0[1] as f64 - a1[1] as f64).atan2(a0[0] as f64 - a1[0] as f64);
                    pinch += 1;
                    let cw = |c: usize| -> f64 {
                        let b1 = nout[lst[c].n1];
                        let ang = (b1[1] as f64 - a1[1] as f64).atan2(b1[0] as f64 - a1[0] as f64);
                        let d = (back - ang).rem_euclid(2.0 * std::f64::consts::PI);
                        if d > 1e-12 { d } else { 2.0 * std::f64::consts::PI }
                    };
                    cand.sort_by(|&x, &y| cw(x).partial_cmp(&cw(y)).unwrap());
                }
                k = cand[0];
            }
            if lst[*cyc.last().unwrap()].n1 != lst[cyc[0]].n0 {
                bail!("regions atlas: {name}: open edge chain for area {owner:?} in chart {ck}");
            }
            cycles.push(cyc);
        }
        if pinch > 0 {
            *self.stats.entry("pinch_choices".into()).or_insert(0) += pinch;
        }
        let nout = &self.nout;
        let pt = |n: usize| -> P { widen(nout[n]) };
        struct R {
            cyc: Vec<usize>,
            area: f64,
        }
        let mut rings: Vec<R> = Vec::new();
        for cyc in cycles {
            let pts: Vec<P> = cyc.iter().map(|&k| pt(lst[k].n0)).collect();
            let s2 = crate::regionmap::geometry::signed_area2(&pts);
            rings.push(R { cyc, area: s2 / 2.0 });
        }
        if rings.iter().any(|r| r.area == 0.0) {
            bail!("regions atlas: {name}: zero-area ring for area {owner:?} chart {ck}");
        }
        let outers: Vec<&R> = rings.iter().filter(|r| r.area > 0.0).collect();
        let holes: Vec<&R> = rings.iter().filter(|r| r.area < 0.0).collect();
        let outer_pts: Vec<Vec<P>> = outers.iter().map(|r| r.cyc.iter().map(|&k| pt(lst[k].n0)).collect()).collect();
        let mut hole_for: Vec<Vec<&R>> = vec![Vec::new(); outers.len()];
        for h in &holes {
            let hp: Vec<P> = h.cyc.iter().map(|&k| pt(lst[k].n0)).collect();
            let rp = ring_interior_point(&hp);
            let mut best: Option<(usize, f64)> = None;
            for (i, op) in outer_pts.iter().enumerate() {
                if point_in_ring(rp, op) {
                    let a = outers[i].area;
                    if best.is_none_or(|(_, ba)| a < ba) {
                        best = Some((i, a));
                    }
                }
            }
            let Some((i, _)) = best else {
                bail!("regions atlas: {name}: hole of area {owner:?} in chart {ck} has no containing outer ring");
            };
            hole_for[i].push(*h);
        }
        let rotate = |c: &[usize]| -> (Vec<usize>, u64) {
            let ranks: Vec<u64> = c.iter().map(|&k| lst[k].rank).collect();
            let s = (0..c.len()).min_by_key(|&t| (ranks[t], t)).unwrap();
            ((0..c.len()).map(|t| lst[c[(s + t) % c.len()]].n0).collect(), *ranks.iter().min().unwrap())
        };
        let mut frags: Vec<Frag> = Vec::new();
        for (i, oc) in outers.iter().enumerate() {
            let (outer, _) = rotate(&oc.cyc);
            let mut hl: Vec<(Vec<usize>, u64, f64)> = hole_for[i]
                .iter()
                .map(|h| {
                    let (ids, mr) = rotate(&h.cyc);
                    (ids, mr, h.area)
                })
                .collect();
            hl.sort_by_key(|z| z.1);
            let area = oc.area + hl.iter().map(|z| z.2).sum::<f64>();
            let src_edges = oc.cyc.iter().filter(|&&k| lst[k].from_src).count()
                + hole_for[i].iter().flat_map(|h| h.cyc.iter()).filter(|&&k| lst[k].from_src).count();
            let nedge = oc.cyc.len() + hole_for[i].iter().map(|h| h.cyc.len()).sum::<usize>();
            let mut edges: Vec<Edge> = oc.cyc.iter().map(|&k| lst[k]).collect();
            for h in &hole_for[i] {
                edges.extend(h.cyc.iter().map(|&k| lst[k]));
            }
            frags.push(Frag { ci, outer, holes: hl.into_iter().map(|z| z.0).collect(), area, unchanged: src_edges == nedge, edges });
        }
        frags.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap());
        Ok(frags)
    }
}

/// Closed polygon (ring) versus closed rectangle intersection.
fn polygon_intersects_rect(ring: &[P], r: Rect64) -> bool {
    let inside = |p: P| r[0][0] <= p[0] && p[0] <= r[1][0] && r[0][1] <= p[1] && p[1] <= r[1][1];
    if ring.iter().any(|&p| inside(p)) {
        return true;
    }
    let corners = [r[0], [r[1][0], r[0][1]], r[1], [r[0][0], r[1][1]]];
    if corners.iter().any(|&c| point_in_ring(c, ring)) {
        return true;
    }
    let n = ring.len();
    (0..n).any(|k| lb(ring[k], ring[(k + 1) % n], r).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proven_plan_charts_are_separated_inside_the_world_and_cover_their_theatres() {
        let ctx = Ctx::new(&Plan::proven()).unwrap();
        assert!(ctx.min_world_margin > 1e-3);
        assert!(ctx.min_chart_separation.unwrap() > 1e-3);
        assert_eq!(ctx.unions.len(), 6);
    }

    #[test]
    fn chart_ownership_prefers_theatre_rectangles_and_falls_back_to_crops() {
        let ctx = Ctx::new(&Plan::proven()).unwrap();
        let eur = ctx.chart_index("eur_ind").unwrap();
        assert_eq!(ctx.chart_of_point([0.0, 300.0]).unwrap(), eur);
        // inside the 10-unit crop margin of Europe, outside its theatre rectangle
        assert_eq!(ctx.chart_of_point([-185.0, 300.0]).unwrap(), eur);
        // open ocean between the crops belongs to no chart
        assert!(ctx.chart_of_point([-1000.0, 0.0]).is_err());
    }

    #[test]
    fn liang_barsky_reports_the_crossed_sides() {
        let r = [[0.0, 0.0], [10.0, 10.0]];
        let iv = lb([-5.0, 5.0], [15.0, 5.0], r).unwrap();
        assert_eq!((iv.t0, iv.t1), (0.25, 0.75));
        assert_eq!(iv.s0, Some((0, 0.0)));
        assert_eq!(iv.s1, Some((0, 10.0)));
        // an edge that starts inside has no entering side
        let iv = lb([5.0, 5.0], [15.0, 5.0], r).unwrap();
        assert_eq!(iv.t0, 0.0);
        assert!(iv.s0.is_none());
        assert_eq!(iv.s1, Some((0, 10.0)));
        // parallel and outside
        assert!(lb([-5.0, 20.0], [15.0, 20.0], r).is_none());
        // touching the corner only
        assert!(lb([-5.0, 5.0], [5.0, 15.0], r).is_none());
        assert!(lb([-5.0, 15.0], [5.0, 25.0], r).is_none());
    }

    #[test]
    fn overlapping_crop_rectangles_merge_into_one_interval_keeping_outer_sides() {
        let rs = [[[0.0, 0.0], [10.0, 10.0]], [[8.0, 0.0], [20.0, 10.0]]];
        let m = edge_intervals([-5.0, 5.0], [25.0, 5.0], &rs);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].s0, Some((0, 0.0)));
        assert_eq!(m[0].s1, Some((0, 20.0)));
        assert_eq!(xpoint([-5.0, 5.0], [25.0, 5.0], (0, 20.0)), [20.0, 5.0]);
        // disjoint rectangles stay two intervals
        let rs = [[[0.0, 0.0], [10.0, 10.0]], [[12.0, 0.0], [20.0, 10.0]]];
        assert_eq!(edge_intervals([-5.0, 5.0], [25.0, 5.0], &rs).len(), 2);
    }

    #[test]
    fn polygon_rect_intersection_detects_containment_and_crossing() {
        let ring = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(polygon_intersects_rect(&ring, [[4.0, 4.0], [5.0, 5.0]]));
        assert!(polygon_intersects_rect(&ring, [[-5.0, 4.0], [5.0, 5.0]]));
        assert!(polygon_intersects_rect(&ring, [[-5.0, -5.0], [15.0, 15.0]]));
        assert!(!polygon_intersects_rect(&ring, [[11.0, 0.0], [12.0, 5.0]]));
    }
}
