//! Chart lookup and exact clipping helpers of the vector transform (`Geo` of the accepted `vector_scale.py`).

use crate::atlas::Plan;
use anyhow::{Result, bail};
use std::cell::Cell;

pub type R = [[f64; 2]; 2];

pub struct ChartGeo {
    pub key: String,
    pub offset: [f64; 2],
    pub rects: Vec<R>,
    pub disjoint: Vec<R>,
}

pub struct Geo {
    pub scale: f64,
    pub charts: Vec<ChartGeo>,
    /// Points that fell on a boundary shared by two charts (first chart wins).
    pub ties: Cell<usize>,
}

pub struct Piece {
    /// `(segment, t)` of the first vertex in the source polyline (traversal order key).
    pub start: (usize, f64),
    pub xy: Vec<[f64; 2]>,
    pub ex: Vec<f64>,
    tail: bool,
}

pub fn rect_has(r: &R, x: f64, y: f64) -> bool {
    r[0][0] <= x && x <= r[1][0] && r[0][1] <= y && y <= r[1][1]
}

fn rect_subtract(p: &R, q: &R) -> Vec<R> {
    let lo = [p[0][0].max(q[0][0]), p[0][1].max(q[0][1])];
    let hi = [p[1][0].min(q[1][0]), p[1][1].min(q[1][1])];
    if !(hi[0] > lo[0] && hi[1] > lo[1]) {
        return vec![*p];
    }
    let mut res = Vec::new();
    if p[0][0] < q[0][0] {
        res.push([[p[0][0], p[0][1]], [q[0][0], p[1][1]]]);
    }
    if q[1][0] < p[1][0] {
        res.push([[q[1][0], p[0][1]], [p[1][0], p[1][1]]]);
    }
    let (x0, x1) = (p[0][0].max(q[0][0]), p[1][0].min(q[1][0]));
    if p[0][1] < q[0][1] {
        res.push([[x0, p[0][1]], [x1, q[0][1]]]);
    }
    if q[1][1] < p[1][1] {
        res.push([[x0, q[1][1]], [x1, p[1][1]]]);
    }
    res
}

pub fn disjoint_rects(rects: &[R]) -> Vec<R> {
    let mut out: Vec<R> = Vec::new();
    for r in rects {
        let mut pieces = vec![*r];
        for q in &out {
            let mut next = Vec::new();
            for p in &pieces {
                next.extend(rect_subtract(p, q));
            }
            pieces = next;
        }
        out.extend(pieces);
    }
    out
}

impl Geo {
    pub fn new(plan: &Plan) -> Result<Self> {
        let charts: Vec<ChartGeo> = plan
            .charts
            .iter()
            .map(|c| ChartGeo {
                key: c.key.clone(),
                offset: c.offset,
                rects: c.crop_rects.clone(),
                disjoint: disjoint_rects(&c.crop_rects),
            })
            .collect();
        let geo = Self { scale: plan.scale, charts, ties: Cell::new(0) };
        let (src, world) = (plan.source_world, plan.world);
        for c in &geo.charts {
            for r in &c.rects {
                if r[0][0] > r[1][0] || r[0][1] > r[1][1] {
                    bail!("chart {} has an inverted crop rect {r:?}", c.key);
                }
                if (0..2).any(|i| r[0][i] < src[0][i] - 1e-9 || r[1][i] > src[1][i] + 1e-9) {
                    bail!("chart {} crop {r:?} lies outside source_world", c.key);
                }
                let (lo, hi) = (geo.moved(r[0], c), geo.moved(r[1], c));
                if (0..2).any(|i| lo[i] < world[0][i] - 1e-6 || hi[i] > world[1][i] + 1e-6) {
                    bail!("chart {} transformed crop {lo:?}..{hi:?} outside plan world", c.key);
                }
            }
        }
        for (i, a) in geo.charts.iter().enumerate() {
            for b in &geo.charts[i + 1..] {
                for ra in &a.disjoint {
                    for rb in &b.disjoint {
                        let (lo, hi) = (
                            [ra[0][0].max(rb[0][0]), ra[0][1].max(rb[0][1])],
                            [ra[1][0].min(rb[1][0]), ra[1][1].min(rb[1][1])],
                        );
                        if hi[0] - lo[0] > 1e-9 && hi[1] - lo[1] > 1e-9 {
                            bail!("crop rects of {} and {} overlap with positive area", a.key, b.key);
                        }
                        let (la, lb) = (
                            [geo.moved(ra[0], a), geo.moved(ra[1], a)],
                            [geo.moved(rb[0], b), geo.moved(rb[1], b)],
                        );
                        let (lo, hi) = (
                            [la[0][0].max(lb[0][0]), la[0][1].max(lb[0][1])],
                            [la[1][0].min(lb[1][0]), la[1][1].min(lb[1][1])],
                        );
                        if hi[0] - lo[0] > 1e-6 && hi[1] - lo[1] > 1e-6 {
                            bail!("transformed crops of {} and {} overlap in the output world", a.key, b.key);
                        }
                    }
                }
            }
        }
        Ok(geo)
    }

    fn moved(&self, p: [f64; 2], c: &ChartGeo) -> [f64; 2] {
        [p[0] * self.scale + c.offset[0], p[1] * self.scale + c.offset[1]]
    }

    /// Chart whose crop union contains the point (inclusive); the first chart wins boundary ties (counted).
    pub fn chart_of_point(&self, x: f64, y: f64) -> Option<usize> {
        let mut hits = self.charts.iter().enumerate().filter(|(_, c)| c.rects.iter().any(|r| rect_has(r, x, y)));
        let first = hits.next().map(|(i, _)| i);
        if hits.next().is_some() {
            self.ties.set(self.ties.get() + 1);
        }
        first
    }

    pub fn xy(&self, x: f64, y: f64, chart: usize) -> (f64, f64) {
        let c = &self.charts[chart];
        (x * self.scale + c.offset[0], y * self.scale + c.offset[1])
    }

    /// Charts whose TRANSFORMED crop union contains an output-space point.
    pub fn inverse_chart_of_output(&self, x: f64, y: f64, tol: f64) -> Vec<usize> {
        let mut hits = Vec::new();
        for (i, c) in self.charts.iter().enumerate() {
            for r in &c.rects {
                let (lo, hi) = (self.moved(r[0], c), self.moved(r[1], c));
                if lo[0] - tol <= x && x <= hi[0] + tol && lo[1] - tol <= y && y <= hi[1] + tol {
                    hits.push(i);
                    break;
                }
            }
        }
        hits
    }

    /// Exact Liang-Barsky clip of a polyline (`xy`, one attribute `ex` per point) to the chart crop union. Attributes are
    /// interpolated at cut points; a closed ring that is cut is re-joined across its start vertex.
    pub fn clip_polyline(&self, chart: usize, xy: &[[f64; 2]], ex: &[f64]) -> Vec<Piece> {
        let rects = &self.charts[chart].rects;
        let n = xy.len();
        for r in rects {
            if xy.iter().all(|p| p[0] >= r[0][0] && p[1] >= r[0][1] && p[0] <= r[1][0] && p[1] <= r[1][1]) {
                return vec![Piece { start: (0, 0.0), xy: xy.to_vec(), ex: ex.to_vec(), tail: false }];
            }
        }
        let mut pieces: Vec<Piece> = Vec::new();
        let mut cur: Option<Piece> = None;
        for i in 0..n.saturating_sub(1) {
            let (p0, p1, e0, e1) = (xy[i], xy[i + 1], ex[i], ex[i + 1]);
            let d = [p1[0] - p0[0], p1[1] - p0[1]];
            let mut ivs: Vec<(f64, f64)> = Vec::new();
            for r in rects {
                let (mut t0, mut t1, mut ok) = (0.0f64, 1.0f64, true);
                for a in 0..2 {
                    let (lo, hi) = (r[0][a], r[1][a]);
                    if d[a] == 0.0 {
                        if p0[a] < lo || p0[a] > hi {
                            ok = false;
                            break;
                        }
                    } else {
                        let (mut ta, mut tb) = ((lo - p0[a]) / d[a], (hi - p0[a]) / d[a]);
                        if ta > tb {
                            std::mem::swap(&mut ta, &mut tb);
                        }
                        t0 = t0.max(ta);
                        t1 = t1.min(tb);
                        if t0 > t1 {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    ivs.push((t0, t1));
                }
            }
            ivs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut merged: Vec<[f64; 2]> = Vec::new();
            for (a, b) in ivs {
                if let Some(m) = merged.last_mut() {
                    if a <= m[1] + 1e-12 {
                        m[1] = m[1].max(b);
                        continue;
                    }
                }
                merged.push([a, b]);
            }
            merged.retain(|m| m[1] > m[0]);
            if cur.is_some() && (merged.is_empty() || merged[0][0] > 0.0) {
                pieces.push(cur.take().unwrap());
            }
            for m in &merged {
                let (a, b) = (m[0], m[1]);
                let (sx, se) = if a <= 0.0 {
                    (p0, e0)
                } else {
                    ([p0[0] + a * d[0], p0[1] + a * d[1]], e0 + a * (e1 - e0))
                };
                let (px, pe) = if b >= 1.0 {
                    (p1, e1)
                } else {
                    ([p0[0] + b * d[0], p0[1] + b * d[1]], e0 + b * (e1 - e0))
                };
                if a <= 0.0 && cur.is_some() {
                    let c = cur.as_mut().unwrap();
                    c.xy.push(px);
                    c.ex.push(pe);
                } else {
                    cur = Some(Piece { start: (i, a.max(0.0)), xy: vec![sx, px], ex: vec![se, pe], tail: false });
                }
                if b < 1.0 {
                    pieces.push(cur.take().unwrap());
                }
            }
        }
        if let Some(mut c) = cur.take() {
            c.tail = true;
            pieces.push(c);
        }
        let closed = n >= 4 && xy[0] == xy[n - 1];
        if closed && pieces.len() >= 2 && pieces[0].start == (0, 0.0) && pieces[pieces.len() - 1].tail {
            let head = pieces.remove(0);
            let mut last = pieces.pop().unwrap();
            last.xy.extend_from_slice(&head.xy[1..]);
            last.ex.extend_from_slice(&head.ex[1..]);
            pieces.insert(0, last);
        }
        pieces
    }

    /// True when the box `[lo, hi]` is covered by the chart crop union.
    pub fn covered_by_union(&self, chart: usize, lo: [f64; 2], hi: [f64; 2]) -> bool {
        let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
        let c = &self.charts[chart];
        if area <= 0.0 {
            return c.rects.iter().any(|r| rect_has(r, lo[0], lo[1]) && rect_has(r, hi[0], hi[1]));
        }
        let mut acc = 0.0;
        for r in &c.disjoint {
            let (a, b) = ([lo[0].max(r[0][0]), lo[1].max(r[0][1])], [hi[0].min(r[1][0]), hi[1].min(r[1][1])]);
            if b[0] > a[0] && b[1] > a[1] {
                acc += (b[0] - a[0]) * (b[1] - a[1]);
            }
        }
        acc >= area * (1.0 - 1e-12)
    }
}

/// Sutherland-Hodgman clip of a polygon of full attribute vectors against `rect` on `(x, z) = (cols 0, 2)`; winding
/// is preserved and new vertices (id `None`) interpolate every attribute.
pub fn clip_poly_to_rect(mut poly: Vec<[f64; 20]>, mut ids: Vec<Option<usize>>, rect: &R) -> (Vec<[f64; 20]>, Vec<Option<usize>>) {
    for (axis, bound, keep_ge) in [(0usize, rect[0][0], true), (0, rect[1][0], false), (2, rect[0][1], true), (2, rect[1][1], false)] {
        if poly.is_empty() {
            break;
        }
        let mut out = Vec::new();
        let mut oids = Vec::new();
        let n = poly.len();
        for i in 0..n {
            let (cur, prev) = (poly[i], poly[(i + n - 1) % n]);
            let inside = |v: &[f64; 20]| if keep_ge { v[axis] >= bound } else { v[axis] <= bound };
            let (cin, pin) = (inside(&cur), inside(&prev));
            if cin != pin {
                let t = (bound - prev[axis]) / (cur[axis] - prev[axis]);
                let mut v = [0f64; 20];
                for k in 0..20 {
                    v[k] = prev[k] + t * (cur[k] - prev[k]);
                }
                v[axis] = bound;
                out.push(v);
                oids.push(None);
            }
            if cin {
                out.push(cur);
                oids.push(ids[i]);
            }
        }
        poly = out;
        ids = oids;
    }
    (poly, ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo() -> Geo {
        Geo::new(&Plan::proven()).unwrap()
    }

    #[test]
    fn a_polyline_leaving_a_crop_is_cut_exactly_and_interpolates_height() {
        let g = geo();
        let ci = g.charts.iter().position(|c| c.key == "america").unwrap();
        let r = g.charts[ci].rects[0];
        let inside = [r[0][0] + 10.0, r[0][1] + 10.0];
        let outside = [r[0][0] - 10.0, r[0][1] + 10.0];
        let pieces = g.clip_polyline(ci, &[inside, outside], &[0.0, 20.0]);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].xy.len(), 2);
        assert!((pieces[0].xy[1][0] - r[0][0]).abs() < 1e-9);
        assert!((pieces[0].ex[1] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_closed_ring_cut_at_its_start_vertex_is_rejoined_into_one_piece() {
        let g = geo();
        let ci = g.charts.iter().position(|c| c.key == "brazil").unwrap();
        let r = g.charts[ci].rects[0];
        let (xin, xout, y0, y1) = (r[0][0] + 10.0, r[0][0] - 10.0, r[0][1] + 5.0, r[0][1] + 25.0);
        // the start vertex is inside; the ring leaves through the west edge and re-enters at the closing segment
        let ring = [[xin, y0], [xin, y1], [xout, y1], [xout, y0], [xin, y0]];
        let pieces = g.clip_polyline(ci, &ring, &[0.0; 5]);
        assert_eq!(pieces.len(), 1, "the head and tail pieces must be joined across the start vertex");
    }

    #[test]
    fn disjoint_decomposition_preserves_union_area() {
        let a: R = [[0.0, 0.0], [4.0, 4.0]];
        let b: R = [[2.0, 2.0], [6.0, 6.0]];
        let parts = disjoint_rects(&[a, b]);
        let area: f64 = parts.iter().map(|r| (r[1][0] - r[0][0]) * (r[1][1] - r[0][1])).sum();
        assert!((area - 28.0).abs() < 1e-12);
    }
}
