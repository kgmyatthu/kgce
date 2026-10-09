//! Clipping of regions.esf areas at theatre rectangles (+ margin): the vanilla convention for sea polygons, applied to
//! the enlarged theatres, plus the `unexplorable` backdrop land areas that straddle a rectangle.
//!
//! Measured conventions (vanilla NTW regions.esf, reference implementation `work/exp4/seaclip.py`):
//! * every sea polygon is cut by the theatre rectangle + 5: the inside parts are passable (the sea group id), the
//!   outside parts impassable with fresh component ids;
//! * both halves SHARE the cut vertices, cut lines carry a vertex every 5 units;
//! * across the cut the connectivity runs name the OTHER half; neighbour rings that touch the cut carry the crossing
//!   vertex (their edge is split, runs and faces follow);
//! * outer rings are counter-clockwise, holes clockwise; faces are the per-area earcut (clockwise triangles);
//!   runs are the maximal consecutive edges with one neighbour and a ring starts on a run boundary;
//! * the part that keeps most of the original edges keeps the area index, the others are appended.
//!
//! Boolean operations run on a 2^-16 unit integer grid (`i_overlay`); the rectangle lines and every vertex within
//! 2e-3 of one are snapped onto the line first (vanilla coordinates carry ~1e-3 noise around round numbers).

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, anyhow, bail, ensure};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay::{Overlay, ShapeType};
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::i_float::int::point::IntPoint;

use super::geometry::{P, Rect};
use super::regions::{Area, NONE, Outline, RegionData, code};

// (backdrop land is recognised structurally, see `RegionData::is_backdrop_island`)

const SCALE: f64 = 65536.0;
const SNAP: f64 = 2e-3;
const MIN_PART: f64 = 1e-4;
const CUT_STEP: f64 = 5.0;
/// distance within which a cut vertex counts as lying on an original edge
const ON_EDGE: f64 = 5e-3;

/// One clipped area: the original index (which keeps the largest part) and the indices of all its parts.
#[derive(Clone, Debug, PartialEq)]
pub struct ClippedArea {
    pub region: usize,
    pub area: usize,
    pub parts: Vec<usize>,
}

/// What [`RegionData::clip_at_theatres`] changed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipReport {
    pub areas_clipped: usize,
    /// impassable sea areas fully inside a rectangle that became passable
    pub areas_flagged_passable: usize,
    pub cut_vertices: usize,
    /// neighbour rings that received crossing vertices
    pub neighbour_rings_updated: usize,
    pub clipped: Vec<ClippedArea>,
    /// world rectangles whose `query_info` cells must be rebuilt (`rebuild_query_index_within`)
    pub dirty_rects: Vec<Rect>,
}

impl ClipReport {
    /// [`ClipReport::dirty_rects`] as the `(lo, hi)` pairs [`RegionData::rebuild_query_index_within`] takes.
    pub fn dirty_pairs(&self) -> Vec<(P, P)> {
        self.dirty_rects
            .iter()
            .map(|r| ([r[0][0] as f64, r[0][1] as f64], [r[1][0] as f64, r[1][1] as f64]))
            .collect()
    }
}

// ------------------------------------------------------------------------------------------------- helpers

fn world(p: IntPoint) -> P {
    [p.x as f64 / SCALE, p.y as f64 / SCALE]
}

/// Twice the signed area of an integer contour (> 0 counter-clockwise).
fn area2_int(c: &[IntPoint]) -> i128 {
    let n = c.len();
    (0..n)
        .map(|k| {
            let (a, b) = (c[k], c[(k + 1) % n]);
            a.x as i128 * b.y as i128 - b.x as i128 * a.y as i128
        })
        .sum()
}

/// Vertex pool: appends vertices and reuses an existing vertex with identical f32 coordinates.
struct Pool {
    index: HashMap<[u32; 2], u32>,
}

impl Pool {
    fn key(p: [f32; 2]) -> [u32; 2] {
        [(p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits()]
    }

    fn new(vertices: &[[f32; 2]]) -> Self {
        Self { index: vertices.iter().enumerate().map(|(i, p)| (Self::key(*p), i as u32)).collect() }
    }

    /// (id, true when the vertex was appended)
    fn add(&mut self, vertices: &mut Vec<[f32; 2]>, p: [f32; 2]) -> (u32, bool) {
        if let Some(&i) = self.index.get(&Self::key(p)) {
            return (i, false);
        }
        let i = vertices.len() as u32;
        vertices.push(p);
        self.index.insert(Self::key(p), i);
        (i, true)
    }
}

/// The four lines of a theatre rectangle + margin, as exact f32 values and on the integer grid.
#[derive(Clone, Copy, Debug)]
struct Lines {
    xs: [f32; 2],
    ys: [f32; 2],
    xi: [i32; 2],
    yi: [i32; 2],
}

impl Lines {
    fn new(rect: Rect, margin: f32) -> Self {
        let xs = [rect[0][0] - margin, rect[1][0] + margin];
        let ys = [rect[0][1] - margin, rect[1][1] + margin];
        let grid = |v: f32| (v as f64 * SCALE).round() as i32;
        Self { xs, ys, xi: [grid(xs[0]), grid(xs[1])], yi: [grid(ys[0]), grid(ys[1])] }
    }

    /// Grid point of a vertex, snapped onto a rectangle line when within [`SNAP`] of it.
    fn snap(&self, p: [f32; 2]) -> IntPoint {
        let near = |v: f32, lines: &[f32; 2], grid: &[i32; 2]| {
            let v = v as f64;
            (0..2).find(|&i| (v - lines[i] as f64).abs() < SNAP).map(|i| grid[i])
        };
        IntPoint::new(
            near(p[0], &self.xs, &self.xi).unwrap_or_else(|| (p[0] as f64 * SCALE).round() as i32),
            near(p[1], &self.ys, &self.yi).unwrap_or_else(|| (p[1] as f64 * SCALE).round() as i32),
        )
    }

    /// The rectangle line both points lie on: (0, x) for a vertical line, (1, y) for a horizontal one, with the exact f32 value.
    fn on_line(&self, a: IntPoint, b: IntPoint) -> Option<(usize, f32)> {
        for i in 0..2 {
            if a.x == b.x && a.x == self.xi[i] {
                return Some((0, self.xs[i]));
            }
        }
        for i in 0..2 {
            if a.y == b.y && a.y == self.yi[i] {
                return Some((1, self.ys[i]));
            }
        }
        None
    }

    fn box_contour(&self) -> Vec<IntPoint> {
        vec![
            IntPoint::new(self.xi[0], self.yi[0]),
            IntPoint::new(self.xi[1], self.yi[0]),
            IntPoint::new(self.xi[1], self.yi[1]),
            IntPoint::new(self.xi[0], self.yi[1]),
        ]
    }
}

/// A clipped part: contours (outer counter-clockwise first, holes clockwise) and its area in world units².
struct Part {
    contours: Vec<Vec<IntPoint>>,
    area: f64,
}

fn parts_of(shapes: Vec<Vec<Vec<IntPoint>>>) -> Vec<Part> {
    let mut out = Vec::new();
    for shape in shapes {
        let mut contours = Vec::with_capacity(shape.len());
        let mut area = 0.0;
        for (k, mut c) in shape.into_iter().enumerate() {
            let a2 = area2_int(&c);
            if (k == 0) != (a2 > 0) {
                c.reverse();
            }
            area += a2.abs() as f64 / 2.0 / SCALE / SCALE * if k == 0 { 1.0 } else { -1.0 };
            contours.push(c);
        }
        if area > MIN_PART {
            out.push(Part { contours, area });
        }
    }
    out
}

/// `subject` against the rectangle: (parts inside, parts outside), each sorted by decreasing area.
fn split_by_box(subject: &[Vec<IntPoint>], lines: &Lines) -> (Vec<Part>, Vec<Part>) {
    let run = |rule: OverlayRule| {
        let mut overlay = Overlay::new(subject.iter().map(Vec::len).sum::<usize>() + 4);
        overlay.add_contours(subject, ShapeType::Subject);
        overlay.add_contour(&lines.box_contour(), ShapeType::Clip);
        let mut parts = parts_of(overlay.overlay(rule, FillRule::EvenOdd));
        parts.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap_or(std::cmp::Ordering::Equal));
        parts
    };
    (run(OverlayRule::Intersect), run(OverlayRule::Difference))
}

fn part_covers(part: &Part, q: P) -> bool {
    let ring = |c: &Vec<IntPoint>| c.iter().map(|&p| world(p)).collect::<Vec<P>>();
    let inside = |c: &Vec<IntPoint>| {
        let pts = ring(c);
        // boundary counts as inside (shapely `covers`)
        let n = pts.len();
        (0..n).any(|k| super::geometry::point_segment_distance(q, pts[k], pts[(k + 1) % n]) < 1e-9)
            || super::geometry::point_in_ring(q, &pts)
    };
    inside(&part.contours[0]) && !part.contours[1..].iter().any(|h| super::geometry::point_in_ring(q, &ring(h)))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// does not overlap the rectangle
    Outside,
    In,
    Out,
    Split,
}

struct WorkRing {
    ids: Vec<u32>,
    codes: Vec<u32>,
    outer: bool,
}

// ------------------------------------------------------------------------------------------------- RegionData

impl RegionData {
    /// Backdrop island: an impassable area of an unsettled land region whose connectivity names only sea regions or
    /// areas of its own region (it borders no other land region). These are the unexplorable-style polygons of the
    /// world map that become part of the theatre scenery when a theatre grows; continental backdrop (china, himalayas,
    /// ...) borders other land regions and is left alone.
    pub fn is_backdrop_island(&self, ri: usize, ai: usize) -> bool {
        let region = &self.regions[ri];
        let area = &region.areas[ai];
        region.kind == "land"
            && region.settlement.is_none()
            && !area.passable
            && area.outlines.iter().flat_map(|o| o.runs.iter()).all(|run| {
                run.code == NONE || {
                    let nr = (run.code >> 16) as usize;
                    nr == ri || self.regions.get(nr).is_some_and(|r| r.kind == "sea")
                }
            })
    }

    fn area_contours(&self, ri: usize, ai: usize, lines: &Lines) -> Vec<Vec<IntPoint>> {
        self.regions[ri].areas[ai]
            .outlines
            .iter()
            .map(|o| o.ring.iter().map(|&i| lines.snap(self.vertices[i as usize])).collect())
            .collect()
    }

    fn classify(&self, ri: usize, ai: usize, lines: &Lines) -> Class {
        let area = &self.regions[ri].areas[ai];
        let Some(outer) = area.outlines.first() else { return Class::Outside };
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for &i in &outer.ring {
            for a in 0..2 {
                lo[a] = lo[a].min(self.vertices[i as usize][a]);
                hi[a] = hi[a].max(self.vertices[i as usize][a]);
            }
        }
        if hi[0] < lines.xs[0] || lo[0] > lines.xs[1] || hi[1] < lines.ys[0] || lo[1] > lines.ys[1] {
            return Class::Outside;
        }
        let (ins, out) = split_by_box(&self.area_contours(ri, ai, lines), lines);
        match (ins.is_empty(), out.is_empty()) {
            (true, true) => Class::Outside,
            (true, false) => Class::Out,
            (false, true) => Class::In,
            (false, false) => Class::Split,
        }
    }

    /// Passable sea group of a theatre: the component of the passable sea areas that overlap its rectangle most.
    fn theatre_sea_group(&self, lines: &Lines) -> Option<u16> {
        let mut weight: BTreeMap<u16, f64> = BTreeMap::new();
        for r in self.regions.iter().filter(|r| r.kind == "sea") {
            for a in r.areas.iter().filter(|a| a.passable) {
                let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
                for &i in a.outlines.iter().flat_map(|o| o.ring.iter()) {
                    for k in 0..2 {
                        lo[k] = lo[k].min(self.vertices[i as usize][k]);
                        hi[k] = hi[k].max(self.vertices[i as usize][k]);
                    }
                }
                let w = (hi[0].min(lines.xs[1]) - lo[0].max(lines.xs[0])).max(0.0) as f64;
                let h = (hi[1].min(lines.ys[1]) - lo[1].max(lines.ys[0])).max(0.0) as f64;
                if w * h > 0.0 {
                    *weight.entry(a.adjoining_id).or_insert(0.0) += w * h;
                }
            }
        }
        weight
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(g, _)| g)
    }

    /// Passable sea group an area joins when it becomes passable: the theatre's own group, unless the area only
    /// touches other passable sea (then the one it shares the longest border with); a fresh id as last resort.
    fn sea_group_for(&self, ri: usize, ai: usize, flagged: &HashSet<(usize, usize)>, theatre_group: Option<u16>, fresh: &mut u32) -> Result<u16> {
        let mut touch: BTreeMap<u16, f64> = BTreeMap::new();
        for o in &self.regions[ri].areas[ai].outlines {
            let n = o.ring.len();
            for run in o.runs.iter().filter(|r| r.code != NONE) {
                let (nr, na) = ((run.code >> 16) as usize, (run.code & 0xFFFF) as usize);
                if flagged.contains(&(nr, na)) || (nr, na) == (ri, ai) {
                    continue;
                }
                let Some(neighbour) = self.regions.get(nr).filter(|r| r.kind == "sea").and_then(|r| r.areas.get(na)) else {
                    continue;
                };
                if !neighbour.passable {
                    continue;
                }
                for k in run.from as usize..=run.to as usize {
                    let (p, q) = (self.v(o.ring[k % n]), self.v(o.ring[(k + 1) % n]));
                    *touch.entry(neighbour.adjoining_id).or_insert(0.0) += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                }
            }
        }
        let group = match theatre_group {
            Some(g) if touch.is_empty() || touch.contains_key(&g) => Some(g),
            _ => touch
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(g, _)| g)
                .or(theatre_group),
        };
        match group {
            Some(g) => Ok(g),
            None => {
                let id = *fresh;
                *fresh += 1;
                u16::try_from(id).map_err(|_| anyhow!("component ids exhausted"))
            }
        }
    }

    /// Clip every sea area and every `unexplorable` backdrop land area that straddles the rectangle + `margin`
    /// (vanilla uses 5.0) of a theatre that [`RegionData::extend_theatre`] enlarged, see the module docs. Inside
    /// parts of sea areas become passable (they join the theatre's passable sea group), outside parts impassable;
    /// impassable sea areas fully inside a rectangle become passable; every part of a clipped backdrop area is
    /// impassable. Fresh component ids continue the maximum of the file. Idempotent. Theatres that were not
    /// extended are left alone: the vanilla file is not clipped at all of its theatre rectangles.
    ///
    /// `query_info` is NOT touched (no access to the tree): rebuild it with
    /// [`RegionData::rebuild_query_index_within`] over [`ClipReport::dirty_rects`], or fully.
    pub fn clip_at_theatres(&mut self, margin: f32) -> Result<ClipReport> {
        let rects: Vec<Rect> = self.theatres.iter().filter(|t| t.extended).map(|t| t.rect).collect();
        self.clip_at_rects(&rects, margin, true)
    }

    /// [`RegionData::clip_at_theatres`] for sea areas only (leaves the backdrop land alone).
    pub fn clip_sea_at_theatres(&mut self, margin: f32) -> Result<ClipReport> {
        let rects: Vec<Rect> = self.theatres.iter().filter(|t| t.extended).map(|t| t.rect).collect();
        self.clip_at_rects(&rects, margin, false)
    }

    /// Clip at explicit rectangles (for a file whose theatres were extended earlier); `backdrop_land` also clips the
    /// straddling backdrop islands (see [`RegionData::is_backdrop_island`]).
    pub fn clip_at_rects(&mut self, rects: &[Rect], margin: f32, backdrop_land: bool) -> Result<ClipReport> {
        let mut report = ClipReport::default();
        if rects.is_empty() {
            return Ok(report);
        }
        let mut pool = Pool::new(&self.vertices);
        let mut fresh = self.fresh_component_id();
        for &rect in rects {
            let lines = Lines::new(rect, margin);
            let theatre_group = self.theatre_sea_group(&lines);
            // targets first (the areas appended while clipping are already inside/outside parts)
            let mut targets: Vec<(usize, usize, Class)> = Vec::new();
            for (ri, r) in self.regions.iter().enumerate() {
                let sea = r.kind == "sea";
                if !(sea || (backdrop_land && r.kind == "land" && r.settlement.is_none())) {
                    continue;
                }
                for (ai, a) in r.areas.iter().enumerate() {
                    if a.passable {
                        continue; // vanilla passable areas (and the inside parts of an earlier run) are left alone
                    }
                    if !sea && !self.is_backdrop_island(ri, ai) {
                        continue;
                    }
                    match self.classify(ri, ai, &lines) {
                        Class::Outside => {}
                        c => targets.push((ri, ai, c)),
                    }
                }
            }
            let mut flagged: HashSet<(usize, usize)> = HashSet::new();
            for (ri, ai, class) in targets {
                let sea = self.regions[ri].kind == "sea";
                match class {
                    Class::In if sea => {
                        flagged.insert((ri, ai));
                        report.areas_flagged_passable += 1;
                    }
                    Class::Split => {
                        let (parts, bbox) = self.clip_area(&mut pool, ri, ai, &lines, &mut report)?;
                        report.areas_clipped += 1;
                        let src = self.regions[ri].areas[ai].clone();
                        let (imp, isl) = (
                            if src.impassable_id == 65535 { 286 } else { src.impassable_id },
                            if src.island_id == 65535 { 104 } else { src.island_id },
                        );
                        for &(pi, inside) in &parts {
                            if inside && sea {
                                flagged.insert((ri, pi));
                                continue;
                            }
                            let comp = u16::try_from(fresh).map_err(|_| anyhow!("component ids exhausted"))?;
                            fresh += 1;
                            let a = &mut self.regions[ri].areas[pi];
                            (a.passable, a.adjoining_id, a.impassable_id, a.island_id) = (false, comp, imp, isl);
                        }
                        report.clipped.push(ClippedArea { region: ri, area: ai, parts: parts.iter().map(|p| p.0).collect() });
                        report.dirty_rects.push(bbox);
                    }
                    _ => {}
                }
            }
            // passable sea: group per flagged area (deterministic order)
            let mut order: Vec<(usize, usize)> = flagged.iter().copied().collect();
            order.sort_unstable();
            for &(ri, ai) in &order {
                let group = self.sea_group_for(ri, ai, &flagged, theatre_group, &mut fresh)?;
                let a = &mut self.regions[ri].areas[ai];
                let imp = if a.impassable_id == 65535 { 286 } else { a.impassable_id };
                (a.passable, a.adjoining_id, a.impassable_id, a.island_id) = (true, group, imp, 65535);
            }
        }
        Ok(report)
    }

    /// Split area `ai` of region `ri` (classified `Split`) at the rectangle. Returns [(area index, inside)] per part
    /// (the inside parts first, each group by decreasing area) and the bounding box of the original area.
    #[allow(clippy::type_complexity)]
    fn clip_area(&mut self, pool: &mut Pool, ri: usize, ai: usize, lines: &Lines, report: &mut ClipReport) -> Result<(Vec<(usize, bool)>, Rect)> {
        let key = self.regions[ri].key.clone();
        let who = format!("{key} area {ai}");
        let t_code = code(ri, ai);
        let area = self.regions[ri].areas[ai].clone();
        ensure!(
            area.outlines.iter().filter(|o| o.outer).count() == 1 && area.outlines[0].outer,
            "{who}: clipping needs exactly one outer ring (first)"
        );
        let subject = self.area_contours(ri, ai, lines);
        let (ins, out) = split_by_box(&subject, lines);
        let parts: Vec<(Part, bool)> = ins.into_iter().map(|p| (p, true)).chain(out.into_iter().map(|p| (p, false))).collect();

        // original rings with per-edge codes
        let rings: Vec<WorkRing> = area
            .outlines
            .iter()
            .map(|o| WorkRing { ids: o.ring.clone(), codes: Self::edge_codes(o), outer: o.outer })
            .collect();
        // grid point -> original vertex id
        let mut snap2id: HashMap<IntPoint, u32> = HashMap::new();
        let mut orig_pts: Vec<(IntPoint, u32)> = Vec::new();
        for r in &rings {
            for &i in &r.ids {
                let s = lines.snap(self.vertices[i as usize]);
                match snap2id.insert(s, i) {
                    Some(prev) if prev != i => bail!("{who}: vertices {prev} and {i} snap to the same grid point"),
                    Some(_) => {}
                    None => orig_pts.push((s, i)),
                }
            }
        }
        orig_pts.sort_by_key(|&(p, _)| (p.x, p.y));
        let mut orig_edges: Vec<((u32, u32), u32)> = Vec::new();
        for r in &rings {
            let n = r.ids.len();
            for k in 0..n {
                orig_edges.push(((r.ids[k], r.ids[(k + 1) % n]), r.codes[k]));
            }
        }

        // ---- grid point -> id for every part ring (cut vertices are appended); original vertices that the boolean
        // operation merged away (collinear) are put back so shared borders keep their vertices
        let mut new_ids: Vec<u32> = Vec::new();
        let mut part_rings: Vec<Vec<Vec<(IntPoint, u32)>>> = Vec::new();
        for (part, _) in &parts {
            let mut rs = Vec::new();
            for c in &part.contours {
                let mut ring: Vec<(IntPoint, u32)> = Vec::with_capacity(c.len());
                for &pt in c {
                    let id = match snap2id.get(&pt) {
                        Some(&i) => i,
                        None => {
                            let on_x = (0..2).find(|&i| pt.x == lines.xi[i]);
                            let on_y = (0..2).find(|&i| pt.y == lines.yi[i]);
                            ensure!(
                                on_x.is_some() || on_y.is_some(),
                                "{who}: new vertex ({:.4},{:.4}) is not on a rectangle line",
                                world(pt)[0],
                                world(pt)[1]
                            );
                            let (wx, wy) = (world(pt)[0], world(pt)[1]);
                            let p = [on_x.map_or(wx as f32, |i| lines.xs[i]), on_y.map_or(wy as f32, |i| lines.ys[i])];
                            let (i, fresh_vertex) = pool.add(&mut self.vertices, p);
                            if fresh_vertex {
                                report.cut_vertices += 1;
                            }
                            if !new_ids.contains(&i) {
                                new_ids.push(i);
                            }
                            i
                        }
                    };
                    ring.push((pt, id));
                }
                rs.push(insert_collinear(ring, &orig_pts));
            }
            part_rings.push(rs);
        }

        // ---- crossing vertices on original edges -> chains of sub-edges
        let mut cross: HashMap<(u32, u32), Vec<(f64, u32)>> = HashMap::new();
        for &i in &new_ids {
            let p = self.v(i);
            let mut best: Option<(f64, (u32, u32), f64)> = None;
            for &((u, v), _) in &orig_edges {
                let (a, b) = (self.v(u), self.v(v));
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let l2 = dx * dx + dy * dy;
                let t = if l2 == 0.0 { 0.0 } else { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) };
                let d = ((p[0] - (a[0] + t * dx)).powi(2) + (p[1] - (a[1] + t * dy)).powi(2)).sqrt();
                if best.is_none_or(|(bd, _, _)| d < bd) {
                    best = Some((d, (u, v), t));
                }
            }
            let (d, e, t) = best.ok_or_else(|| anyhow!("{who}: no original edges"))?;
            if d > ON_EDGE {
                let on_x = (0..2).any(|k| p[0] == lines.xs[k] as f64);
                let on_y = (0..2).any(|k| p[1] == lines.ys[k] as f64);
                if on_x && on_y {
                    continue; // a rectangle corner inside the polygon
                }
                bail!("{who}: new vertex ({:.4},{:.4}) is {d:.4} away from every original edge", p[0], p[1]);
            }
            cross.entry(e).or_default().push((t, i));
        }
        let mut sub_code: HashMap<(u32, u32), u32> = HashMap::new();
        let mut chains: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for &((u, v), c) in &orig_edges {
            let mut mids = cross.remove(&(u, v)).unwrap_or_default();
            mids.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut pts = vec![u];
            pts.extend(mids.into_iter().map(|m| m.1));
            pts.push(v);
            for w in pts.windows(2) {
                sub_code.insert((w[0], w[1]), c);
            }
            chains.insert((u, v), pts);
        }

        // ---- part indices: the part with most original edges keeps the area index
        let weight: Vec<usize> = part_rings
            .iter()
            .map(|rs| {
                rs.iter()
                    .map(|r| (0..r.len()).filter(|&k| sub_code.contains_key(&(r[k].1, r[(k + 1) % r.len()].1))).count())
                    .sum()
            })
            .collect();
        let keep = (0..parts.len()).fold(0, |best, j| if weight[j] > weight[best] { j } else { best });
        let mut part_idx = Vec::with_capacity(parts.len());
        let mut next = self.regions[ri].areas.len();
        for j in 0..parts.len() {
            if j == keep {
                part_idx.push(ai);
            } else {
                part_idx.push(next);
                next += 1;
            }
        }
        let part_code: Vec<u32> = part_idx.iter().map(|&i| code(ri, i)).collect();

        // ---- ring codes edge by edge
        let mut owner: HashMap<(u32, u32), u32> = HashMap::new();
        let mut built: Vec<Vec<WorkRing>> = Vec::new();
        let mut selfref: Vec<(usize, usize, usize)> = Vec::new(); // (part, ring, edge)
        for (j, rs) in part_rings.iter().enumerate() {
            let mut outs = Vec::new();
            for (rn, ring) in rs.iter().enumerate() {
                let n = ring.len();
                let mut codes = Vec::with_capacity(n);
                for k in 0..n {
                    let ((pa, a), (pb, b)) = (ring[k], ring[(k + 1) % n]);
                    if let Some(&c) = sub_code.get(&(a, b)) {
                        owner.insert((a, b), part_code[j]);
                        if c == t_code {
                            selfref.push((j, rn, k));
                        }
                        codes.push(c);
                    } else {
                        ensure!(
                            lines.on_line(pa, pb).is_some(),
                            "{who}: ring edge {a}-{b} is neither an original sub-edge nor on a cut line"
                        );
                        // across the cut: a point just right of a->b lies in the other half
                        let (wa, wb) = (world(pa), world(pb));
                        let (dx, dy) = (wb[0] - wa[0], wb[1] - wa[1]);
                        let l = (dx * dx + dy * dy).sqrt();
                        let d = (1e-3f64).min(l * 0.25);
                        let q = [(wa[0] + wb[0]) / 2.0 + dy / l * d, (wa[1] + wb[1]) / 2.0 - dx / l * d];
                        let other = (0..parts.len()).find(|&k| k != j && part_covers(&parts[k].0, q)).map_or(NONE, |k| part_code[k]);
                        codes.push(other);
                    }
                }
                outs.push(WorkRing { ids: ring.iter().map(|p| p.1).collect(), codes, outer: rn == 0 });
            }
            built.push(outs);
        }
        // an original edge naming the clipped area itself points at the part that owns the reverse edge
        for (j, rn, k) in selfref {
            let r = &mut built[j][rn];
            let n = r.ids.len();
            r.codes[k] = owner.get(&(r.ids[(k + 1) % n], r.ids[k])).copied().unwrap_or(NONE);
        }

        // ---- cut edges get a vertex every CUT_STEP units (the halves share them: same ids)
        let mut cut_vertices = 0usize;
        let mut subdivide = |this: &mut Self, ring: &[(IntPoint, u32)], codes: &[u32]| -> (Vec<u32>, Vec<u32>) {
            let n = ring.len();
            let (mut oi, mut oc) = (Vec::new(), Vec::new());
            for k in 0..n {
                let ((pa, a), (pb, b)) = (ring[k], ring[(k + 1) % n]);
                oi.push(a);
                oc.push(codes[k]);
                if sub_code.contains_key(&(a, b)) {
                    continue;
                }
                let Some((axis, line)) = lines.on_line(pa, pb) else { continue };
                // axis 0: vertical line, the free coordinate is y
                let free = 1 - axis;
                let (wa, wb) = (world(pa)[free], world(pb)[free]);
                let (lo, hi) = (wa.min(wb), wa.max(wb));
                let mut vals = Vec::new();
                let mut m = (lo / CUT_STEP).floor() + 1.0;
                while m * CUT_STEP < hi - 1e-3 {
                    if m * CUT_STEP > lo + 1e-3 {
                        vals.push(m * CUT_STEP);
                    }
                    m += 1.0;
                }
                if wa > wb {
                    vals.reverse();
                }
                for v in vals {
                    let p = if axis == 0 { [line, v as f32] } else { [v as f32, line] };
                    let (vi, fresh_vertex) = pool.add(&mut this.vertices, p);
                    if fresh_vertex {
                        cut_vertices += 1;
                    }
                    oi.push(vi);
                    oc.push(codes[k]);
                }
            }
            (oi, oc)
        };

        // ---- assemble the parts (a hole identical to an original hole is carried over verbatim)
        let orig_holes: HashMap<Vec<u32>, usize> = rings
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.outer)
            .map(|(i, r)| {
                let mut s = r.ids.clone();
                s.sort_unstable();
                (s, i)
            })
            .collect();
        let mut new_areas: Vec<Area> = Vec::new();
        for (j, outs) in built.into_iter().enumerate() {
            let mut finished: Vec<(usize, u32, WorkRing)> = Vec::new(); // (original index or max, first id, ring)
            for (rn, r) in outs.into_iter().enumerate() {
                if rn > 0 {
                    let mut s = r.ids.clone();
                    s.sort_unstable();
                    if let Some(&oi) = orig_holes.get(&s) {
                        let o = &rings[oi];
                        let codes = o
                            .codes
                            .iter()
                            .enumerate()
                            .map(|(k, &c)| {
                                if c == t_code {
                                    let n = o.ids.len();
                                    owner.get(&(o.ids[(k + 1) % n], o.ids[k])).copied().unwrap_or(NONE)
                                } else {
                                    c
                                }
                            })
                            .collect();
                        finished.push((oi, o.ids[0], WorkRing { ids: o.ids.clone(), codes, outer: false }));
                        continue;
                    }
                }
                let ring: Vec<(IntPoint, u32)> = part_rings[j][rn].clone();
                let (ids, codes) = subdivide(self, &ring, &r.codes);
                let n = ids.len();
                let start = (0..n).find(|&k| codes[(k + n - 1) % n] != codes[k]).unwrap_or(0);
                let (mut ids, mut codes) = (ids, codes);
                ids.rotate_left(start);
                codes.rotate_left(start);
                finished.push((if rn == 0 { 0 } else { usize::MAX }, ids[0], WorkRing { ids, codes, outer: rn == 0 }));
            }
            // outer ring first, holes in their original order
            finished.sort_by_key(|(oi, first, r)| (!r.outer, *oi, *first));
            let work: Vec<WorkRing> = finished.into_iter().map(|f| f.2).collect();
            let faces = self.triangulate_area(&work.iter().map(|r| r.ids.as_slice()).collect::<Vec<_>>());
            new_areas.push(Area {
                land_bridge: area.land_bridge,
                channel_coast: area.channel_coast,
                passable: area.passable,
                adjoining_id: area.adjoining_id,
                faces,
                outlines: work
                    .into_iter()
                    .map(|r| Outline { outer: r.outer, runs: Self::runs_from_codes(&r.codes), ring: r.ids })
                    .collect(),
                impassable_id: area.impassable_id,
                island_id: area.island_id,
            });
        }
        report.cut_vertices += cut_vertices;

        // ---- install: the keeping part replaces the area, the others are appended
        for (j, a) in new_areas.into_iter().enumerate() {
            if part_idx[j] == ai {
                self.regions[ri].areas[ai] = a;
            } else {
                debug_assert_eq!(part_idx[j], self.regions[ri].areas.len());
                self.regions[ri].areas.push(a);
            }
        }

        // ---- neighbours: every edge that named the clipped area names the owning part and carries crossing vertices
        let skip: HashSet<(usize, usize)> = part_idx.iter().map(|&i| (ri, i)).collect();
        for (rj, region) in self.regions.iter_mut().enumerate() {
            for (aj, a) in region.areas.iter_mut().enumerate() {
                if skip.contains(&(rj, aj)) {
                    continue;
                }
                let mut split_edges: Vec<(u32, u32, Vec<u32>)> = Vec::new();
                let mut touched = false;
                for o in a.outlines.iter_mut() {
                    let codes = Self::edge_codes(o);
                    if !codes.contains(&t_code) {
                        continue;
                    }
                    touched = true;
                    let n = o.ring.len();
                    let (mut ids, mut ncodes) = (Vec::new(), Vec::new());
                    for k in 0..n {
                        let (a_id, b_id) = (o.ring[k], o.ring[(k + 1) % n]);
                        if codes[k] != t_code {
                            ids.push(a_id);
                            ncodes.push(codes[k]);
                            continue;
                        }
                        let chain = chains
                            .get(&(b_id, a_id))
                            .ok_or_else(|| anyhow!("{}: ring edge {a_id}-{b_id} names the clipped {who} but it has no such edge", region.key))?;
                        let pts: Vec<u32> = chain.iter().rev().copied().collect();
                        for w in pts.windows(2) {
                            ids.push(w[0]);
                            ncodes.push(
                                *owner
                                    .get(&(w[1], w[0]))
                                    .ok_or_else(|| anyhow!("{}: sub-edge {}-{} of the clipped {who} has no owner part", region.key, w[1], w[0]))?,
                            );
                        }
                        if pts.len() > 2 {
                            split_edges.push((a_id, b_id, pts));
                        }
                    }
                    o.ring = ids;
                    o.runs = Self::runs_from_codes(&ncodes);
                }
                if touched {
                    report.neighbour_rings_updated += 1;
                }
                for (a_id, b_id, pts) in split_edges {
                    split_faces(&mut a.faces, a_id, b_id, &pts);
                }
            }
        }

        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        for &i in area.outlines.iter().flat_map(|o| o.ring.iter()) {
            for k in 0..2 {
                lo[k] = lo[k].min(self.vertices[i as usize][k]);
                hi[k] = hi[k].max(self.vertices[i as usize][k]);
            }
        }
        Ok((part_idx.iter().zip(&parts).map(|(&i, p)| (i, p.1)).collect(), [lo, hi]))
    }
}

/// Put back the original vertices lying exactly on a ring edge (the boolean operation drops collinear vertices).
/// `pts` is sorted by (x, y).
fn insert_collinear(ring: Vec<(IntPoint, u32)>, pts: &[(IntPoint, u32)]) -> Vec<(IntPoint, u32)> {
    let n = ring.len();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let ((a, ia), (b, ib)) = (ring[k], ring[(k + 1) % n]);
        out.push((a, ia));
        let (x0, x1) = (a.x.min(b.x), a.x.max(b.x));
        let (y0, y1) = (a.y.min(b.y), a.y.max(b.y));
        let first = pts.partition_point(|p| p.0.x < x0);
        let mut found: Vec<(i64, IntPoint, u32)> = Vec::new();
        for &(p, id) in &pts[first..] {
            if p.x > x1 {
                break;
            }
            if p.y < y0 || p.y > y1 || id == ia || id == ib || p == a || p == b {
                continue;
            }
            let cross = (b.x as i64 - a.x as i64) * (p.y as i64 - a.y as i64) - (b.y as i64 - a.y as i64) * (p.x as i64 - a.x as i64);
            if cross == 0 {
                let along = (p.x as i64 - a.x as i64).abs() + (p.y as i64 - a.y as i64).abs();
                found.push((along, p, id));
            }
        }
        found.sort_by_key(|f| f.0);
        out.extend(found.into_iter().map(|f| (f.1, f.2)));
    }
    out
}

/// Insert the crossing vertices `pts[1..len-1]` of the edge `a -> b` into every triangle that has that edge.
fn split_faces(faces: &mut Vec<[u32; 3]>, a: u32, b: u32, pts: &[u32]) {
    let mut out = Vec::with_capacity(faces.len() + pts.len());
    for t in faces.iter() {
        if t.contains(&a) && t.contains(&b) {
            let x = *t.iter().find(|&&v| v != a && v != b).unwrap();
            let ia = t.iter().position(|&v| v == a).unwrap();
            let forward = t[(ia + 1) % 3] == b;
            let chain: Vec<u32> = if forward { pts.to_vec() } else { pts.iter().rev().copied().collect() };
            for w in chain.windows(2) {
                out.push([w[0], w[1], x]);
            }
        } else {
            out.push(*t);
        }
    }
    *faces = out;
}

#[cfg(test)]
mod tests {
    use super::super::geometry::{point_in_ring, polygon_area};
    use super::*;
    use crate::esf;

    const VANILLA: &str = "F:/NTW2/work/exp1/backup/campaign_maps/nap_world/regions.esf";
    const EXP3: &str = "F:/NTW2/work/exp3/out/data/campaign_maps/nap_world/regions.esf";
    const GOLDEN_B: &str = "F:/NTW2/work/exp5/variantB/stage1/regions.esf";
    const GOLDEN_FULL: &str = "F:/NTW2/work/exp5/stage1/regions.esf";

    pub(crate) fn load(path: &str) -> Option<(esf::Esf, RegionData)> {
        let bytes = std::fs::read(path).ok()?;
        let file = esf::read(&bytes).unwrap();
        let rd = RegionData::parse(&file.root).unwrap();
        Some((file, rd))
    }

    fn name(rd: &RegionData, c: u32) -> String {
        if c == NONE { "-".into() } else { format!("{}:{}", rd.regions[(c >> 16) as usize].key, c & 0xFFFF) }
    }

    /// (vertex coordinate, neighbour name of the edge starting there) per ring vertex
    fn ring_view(rd: &RegionData, o: &Outline) -> Vec<(P, String)> {
        let codes = RegionData::edge_codes(o);
        o.ring.iter().zip(codes).map(|(&i, c)| (rd.v(i), name(rd, c))).collect()
    }

    fn rotation_match(a: &[(P, String)], b: &[(P, String)], tol: f64) -> bool {
        let n = a.len();
        if n != b.len() {
            return false;
        }
        (0..n).any(|s| {
            (0..n).all(|k| {
                let (p, q) = (&a[k], &b[(k + s) % n]);
                (p.0[0] - q.0[0]).abs() < tol && (p.0[1] - q.0[1]).abs() < tol && p.1 == q.1
            })
        })
    }

    /// Differences between two region sets (same region keys in the same order): area counts, flags, component ids,
    /// rings equal up to rotation within 1e-3 with equal neighbour names.
    fn differences(a: &RegionData, b: &RegionData, only: &dyn Fn(&str) -> bool, skip_comp: &[(&str, usize)]) -> Vec<String> {
        let mut out = Vec::new();
        let (mut fwd, mut rev): (HashMap<u16, u16>, HashMap<u16, u16>) = (HashMap::new(), HashMap::new());
        if a.regions.len() != b.regions.len() {
            out.push(format!("region counts {} vs {}", a.regions.len(), b.regions.len()));
            return out;
        }
        for (ra, rb) in a.regions.iter().zip(&b.regions) {
            if ra.key != rb.key {
                out.push(format!("region order {} vs {}", ra.key, rb.key));
                return out;
            }
            if !only(&ra.key) {
                continue;
            }
            if ra.areas.len() != rb.areas.len() {
                out.push(format!("{}: {} areas vs {}", ra.key, ra.areas.len(), rb.areas.len()));
                continue;
            }
            for (i, (x, y)) in ra.areas.iter().zip(&rb.areas).enumerate() {
                let fa = (x.passable, x.impassable_id, x.island_id, x.land_bridge, x.channel_coast);
                let fb = (y.passable, y.impassable_id, y.island_id, y.land_bridge, y.channel_coast);
                if fa != fb {
                    out.push(format!("{} area {i}: flags {fa:?} vs {fb:?}", ra.key));
                }
                // components: exact for passable areas, a bijection for impassable ones (fresh ids are numbered from
                // the maximum of the input, which differs from the old golden because Burma keeps component 13)
                if !skip_comp.iter().any(|(k, n)| *k == ra.key && *n == i) {
                    if x.passable != y.passable || x.passable {
                        if x.adjoining_id != y.adjoining_id {
                            out.push(format!("{} area {i}: component {} vs {}", ra.key, x.adjoining_id, y.adjoining_id));
                        }
                    } else if *fwd.entry(x.adjoining_id).or_insert(y.adjoining_id) != y.adjoining_id
                        || *rev.entry(y.adjoining_id).or_insert(x.adjoining_id) != x.adjoining_id
                    {
                        out.push(format!("{} area {i}: impassable component {} vs {} breaks the id bijection", ra.key, x.adjoining_id, y.adjoining_id));
                    }
                }
                if x.outlines.len() != y.outlines.len() {
                    out.push(format!("{} area {i}: {} rings vs {}", ra.key, x.outlines.len(), y.outlines.len()));
                    continue;
                }
                let views_b: Vec<_> = y.outlines.iter().map(|o| ring_view(b, o)).collect();
                let mut used = vec![false; views_b.len()];
                for (k, o) in x.outlines.iter().enumerate() {
                    let va = ring_view(a, o);
                    let hit = (0..views_b.len()).find(|&j| !used[j] && y.outlines[j].outer == o.outer && rotation_match(&va, &views_b[j], 1e-3));
                    match hit {
                        Some(j) => used[j] = true,
                        None => out.push(format!("{} area {i} ring {k} ({} vertices) has no match", ra.key, va.len())),
                    }
                }
            }
        }
        out
    }

    /// Innermost area (smallest outer ring) containing each point, with prefiltering by bounding box.
    struct Locator<'a> {
        rd: &'a RegionData,
        polys: Vec<(usize, usize, bool, [f64; 4], Vec<P>, f64)>,
    }

    impl<'a> Locator<'a> {
        fn new(rd: &'a RegionData) -> Self {
            let mut polys = Vec::new();
            for (ri, r) in rd.regions.iter().enumerate() {
                for (ai, a) in r.areas.iter().enumerate() {
                    for o in &a.outlines {
                        let pts: Vec<P> = o.ring.iter().map(|&i| rd.v(i)).collect();
                        let bb = pts.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])]);
                        let area = polygon_area(&pts);
                        polys.push((ri, ai, o.outer, bb, pts, area));
                    }
                }
            }
            Self { rd, polys }
        }

        /// passable flag of the innermost area at `p` (None outside everything)
        fn passable(&self, p: P) -> Option<bool> {
            let mut best: Option<(f64, usize, usize)> = None;
            let mut holed: HashSet<(usize, usize)> = HashSet::new();
            for (ri, ai, outer, bb, pts, area) in &self.polys {
                if p[0] < bb[0] || p[0] > bb[2] || p[1] < bb[1] || p[1] > bb[3] || !point_in_ring(p, pts) {
                    continue;
                }
                if *outer {
                    if best.is_none_or(|b| *area < b.0) && !holed.contains(&(*ri, *ai)) {
                        best = Some((*area, *ri, *ai));
                    }
                } else {
                    holed.insert((*ri, *ai));
                    if best.is_some_and(|b| (b.1, b.2) == (*ri, *ai)) {
                        best = None;
                    }
                }
            }
            best.map(|b| self.rd.regions[b.1].areas[b.2].passable)
        }
    }

    fn grid_mismatches(a: &RegionData, b: &RegionData, rect: Rect) -> (usize, usize) {
        let (la, lb) = (Locator::new(a), Locator::new(b));
        let (mut n, mut bad) = (0, 0);
        let mut x = rect[0][0] as f64 + 0.37;
        while x < rect[1][0] as f64 {
            let mut y = rect[0][1] as f64 + 0.61;
            while y < rect[1][1] as f64 {
                n += 1;
                if la.passable([x, y]) != lb.passable([x, y]) {
                    bad += 1;
                }
                y += 1.0;
            }
            x += 1.0;
        }
        (n, bad)
    }

    const INDIA: Rect = [[440.0, -80.0], [790.0, 300.0]];

    #[test]
    fn only_extended_theatres_are_clipped() {
        let Some((_, mut vanilla)) = load(VANILLA) else { return };
        // vanilla is NOT clipped at its own theatre rectangles (e.g. big impassable sea areas straddle the old India
        // rectangle), so theatres that were not extended must be left alone
        assert_eq!(vanilla.clip_at_theatres(5.0).unwrap(), ClipReport::default());
        assert_eq!(vanilla.clip_sea_at_theatres(5.0).unwrap(), ClipReport::default());
    }

    #[test]
    fn vanilla_topology_is_clean_and_sumatra_pole_matches() {
        let Some((_, vanilla)) = load(VANILLA) else { return };
        let problems = vanilla.topology_check();
        assert!(problems.is_empty(), "vanilla has {} topology problems, first {:?}", problems.len(), &problems[..problems.len().min(5)]);
        assert!(vanilla.face_problems().len() > 0, "vanilla carries face deviations (documented on face_problems)");
        let Some((_, golden)) = load(GOLDEN_FULL) else { return };
        let sumatra = golden.index("sumatra").unwrap();
        let pole = golden.pole_of_inaccessibility(sumatra, 0);
        eprintln!("sumatra pole {pole:?}");
        assert!((pole[0] - 724.612).abs() < 0.05 && (pole[1] + 1.597).abs() < 0.05, "{pole:?}");
    }

    #[test]
    fn extension_then_clipping_equals_clipping_the_exp3_file() {
        let (Some((_, mut a)), Some((_, mut b))) = (load(VANILLA), load(EXP3)) else { return };
        let change = a.extend_theatre("3", INDIA).unwrap();
        assert_eq!(change.old_rect, [[440.0, 20.0], [680.0, 280.0]]);
        let moved: Vec<_> = change.moved.iter().map(|m| (m.id, m.old, m.new)).collect();
        assert_eq!(moved, vec![(11, [[660.63, 17.73], [678.66, 118.5]], [[770.63, 17.73], [788.66, 118.5]])]);
        assert_eq!(a.theatres.iter().map(|t| (&t.key, t.rect, &t.transitions)).collect::<Vec<_>>(), b.theatres.iter().map(|t| (&t.key, t.rect, &t.transitions)).collect::<Vec<_>>());
        let ra = a.clip_sea_at_theatres(5.0).unwrap();
        let rb = b.clip_at_rects(&[INDIA], 5.0, false).unwrap();
        assert_eq!(ra, rb);
        assert!(ra.areas_clipped > 0);
    }

    /// Structurally sound (the vanilla file is) and no more face deviations than the input had.
    fn assert_sound(rd: &RegionData, faces_before: usize) {
        let problems = rd.topology_check();
        assert!(problems.is_empty(), "{} topology problems, first: {:#?}", problems.len(), &problems[..problems.len().min(8)]);
        let faces = rd.face_problems();
        assert!(faces.len() <= faces_before, "{} face problems (input {faces_before}), first: {:#?}", faces.len(), &faces[..faces.len().min(5)]);
    }

    #[test]
    fn sea_clipping_reproduces_the_golden_variant_b() {
        let (Some((_, mut rd)), Some((_, golden))) = (load(EXP3), load(GOLDEN_B)) else { return };
        let faces_before = rd.face_problems().len();
        assert_sound(&rd, faces_before);
        let ind = rd.index("indochina").unwrap();
        rd.merge_areas(ind, &[0, 1, 2, 6, 7]).unwrap();
        let comp = rd.component_for(ind, 0);
        assert_eq!(comp, 13, "Burma joins the Bengal land component (deliberately not the old golden 857)");
        rd.set_passable(ind, 0, comp, 65535, 104).unwrap();
        let report = rd.clip_at_rects(&[INDIA], 5.0, false).unwrap();
        eprintln!("clipped: {:?}", report.clipped.iter().map(|c| (rd.regions[c.region].key.clone(), c.area, c.parts.clone())).collect::<Vec<_>>());
        eprintln!("report: flagged {} cut vertices {} neighbour rings {}", report.areas_flagged_passable, report.cut_vertices, report.neighbour_rings_updated);
        let clipped_keys: Vec<&str> = report.clipped.iter().map(|c| rd.regions[c.region].key.as_str()).collect();
        for k in ["indian_ocean", "south_china_sea", "arabian_sea"] {
            assert!(clipped_keys.contains(&k), "{k} was not clipped");
        }
        let diff = differences(&rd, &golden, &|_| true, &[("indochina", 0)]);
        assert!(diff.is_empty(), "{} differences, first: {:#?}", diff.len(), &diff[..diff.len().min(12)]);
        let (n, bad) = grid_mismatches(&rd, &golden, [[435.0, -85.0], [795.0, 305.0]]);
        eprintln!("1-unit grid: {n} points, {bad} differ");
        assert_eq!(bad, 0);
        assert_sound(&rd, faces_before);
        // idempotent
        let again = rd.clip_at_rects(&[INDIA], 5.0, false).unwrap();
        assert_eq!((again.areas_clipped, again.areas_flagged_passable, again.cut_vertices), (0, 0, 0));
    }

    #[test]
    fn merge_reproduces_the_golden_union() {
        let Some((_, mut rd)) = load(EXP3) else { return };
        let faces_before = rd.face_problems().len();
        let ind = rd.index("indochina").unwrap();
        let at = rd.merge_areas(ind, &[0, 1, 2, 6, 7]).unwrap();
        assert_eq!(at, 0);
        let union = &rd.regions[ind].areas[0];
        assert_eq!(union.outlines.len(), 1);
        assert_eq!(union.outlines[0].ring.len(), 346);
        let pts: Vec<P> = union.outlines[0].ring.iter().map(|&i| rd.v(i)).collect();
        let area = polygon_area(&pts);
        eprintln!("union: {} vertices, area {area}", pts.len());
        assert!((area - 11868.75).abs() < 0.01, "union area {area}");
        assert_eq!(rd.regions[ind].areas.len(), 4, "union + the three islets");
        assert_sound(&rd, faces_before);
        // component rule: Burma touches Bengal (passable land, component 13) -> 13
        assert_eq!(rd.component_for(ind, 0), 13);
    }

    #[test]
    fn carve_reproduces_the_golden_sumatra() {
        let (Some((_, mut rd)), Some((_, golden))) = (load(EXP3), load(GOLDEN_FULL)) else { return };
        let faces_before = rd.face_problems().len();
        let ind = rd.index("indochina").unwrap();
        rd.merge_areas(ind, &[0, 1, 2, 6, 7]).unwrap();
        let comp = rd.component_for(ind, 0);
        assert_eq!(comp, 13, "Burma joins the Bengal land component (deliberately not the old golden 857)");
        rd.set_passable(ind, 0, comp, 65535, 104).unwrap();
        rd.clip_at_rects(&[INDIA], 5.0, false).unwrap();
        let un = rd.index("unexplorable").unwrap();
        let sumatra = rd.carve_area(un, 31, "sumatra").unwrap();
        assert_eq!(sumatra, golden.regions.len() - 1);
        let comp = rd.component_for(sumatra, 0);
        assert_eq!(comp, rd.fresh_component_id(), "Sumatra only touches sea and the unsettled backdrop: fresh component");
        rd.set_passable(sumatra, 0, comp, 65535, 104).unwrap();
        let report = rd.clip_at_rects(&[INDIA], 5.0, true).unwrap();
        eprintln!("land clips: {:?}", report.clipped);
        let diff = differences(&rd, &golden, &|_| true, &[("indochina", 0), ("sumatra", 0)]);
        assert!(diff.is_empty(), "{} differences, first: {:#?}", diff.len(), &diff[..diff.len().min(12)]);
        assert_sound(&rd, faces_before);
    }

    #[test]
    fn carve_before_clip_gives_the_same_geometry() {
        // pipeline order: clip first (sea + backdrop islands), then carve, then activate; only numbering differs
        let (Some((_, mut rd)), Some((_, golden))) = (load(VANILLA), load(GOLDEN_FULL)) else { return };
        rd.extend_theatre("3", INDIA).unwrap();
        let ind = rd.index("indochina").unwrap();
        rd.merge_areas(ind, &[0, 1, 2, 6, 7]).unwrap();
        rd.clip_at_theatres(5.0).unwrap();
        let un = rd.index("unexplorable").unwrap();
        let sumatra = rd.carve_area(un, 31, "sumatra").unwrap();
        let comp = rd.component_for(sumatra, 0);
        assert_eq!(comp, rd.fresh_component_id(), "Sumatra only touches sea and the unsettled backdrop: fresh component");
        rd.set_passable(sumatra, 0, comp, 65535, 104).unwrap();
        let burma = rd.component_for(ind, 0);
        assert_eq!(burma, 13);
        rd.set_passable(ind, 0, burma, 65535, 104).unwrap();
        let diff = differences(&rd, &golden, &|_| true, &[("indochina", 0), ("sumatra", 0)]);
        assert!(diff.is_empty(), "{} differences, first: {:#?}", diff.len(), &diff[..diff.len().min(10)]);
        assert_eq!(rd.topology_check(), Vec::<String>::new());
    }

    /// canonical quads (v1 < v2, right/left swapped with the vertices) of every leaf cell, in tree order
    fn leaf_quads(node: &esf::Node, out: &mut Vec<Vec<[u32; 4]>>) {
        for ch in node.children().unwrap().iter().skip(2) {
            if ch.name() == Some("cell") {
                let segs = ch.children().unwrap()[2].u32_array().unwrap();
                let mut quads: Vec<[u32; 4]> = segs
                    .chunks_exact(4)
                    .map(|q| if q[0] <= q[1] { [q[0], q[1], q[2], q[3]] } else { [q[1], q[0], q[3], q[2]] })
                    .collect();
                quads.sort_unstable();
                out.push(quads);
            } else {
                leaf_quads(ch, out);
            }
        }
    }

    #[test]
    fn rebuilding_the_query_index_within_the_dirty_rects_equals_a_full_rebuild() {
        let Some((file, mut rd)) = load(EXP3) else { return };
        let report = rd.clip_at_rects(&[INDIA], 5.0, true).unwrap();
        assert!(report.areas_clipped > 0 && !report.dirty_rects.is_empty());
        let (mut within, mut full) = (file.root.clone(), file.root.clone());
        rd.store(&mut within).unwrap();
        rd.store(&mut full).unwrap();
        rd.rebuild_query_index_within(&mut within, Some(&report.dirty_pairs())).unwrap();
        rd.rebuild_query_index(&mut full).unwrap();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        leaf_quads(within.path(&["query_info", "node"]).unwrap(), &mut a);
        leaf_quads(full.path(&["query_info", "node"]).unwrap(), &mut b);
        assert_eq!(a.len(), b.len());
        let stale = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(stale, 0, "{stale} of {} leaf cells outside the dirty rectangles are stale", a.len());
    }
}
