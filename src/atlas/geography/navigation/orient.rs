//! Orientation constraints of a chart's source polygons (pathfinding piece polygons and cached startpos polygons):
//! the sign of the signed area and the sign of every nonzero vertex turn must survive the quantization. The greedy local
//! search of the POC (`repair_orientation`) nudges unpinned shared coordinate values by a few raw units.

use anyhow::{Result, bail};
use serde_json::Value as Json;

use super::axis::Axis;
use super::util::{Counter, sgn};

const AREA_FLIP_COST: i64 = 1000;
/// Shifts up to this many raw units are always tried so exactly-collinear triples (soft terms) get a chance too.
pub(super) const MIN_SOFT_SHIFT: i64 = 3;
pub(super) const TURN_FLIP_COST: i64 = 100;
const REFLEX_COST: i64 = 10;

/// `'p'` pathfinding piece polygon, `'c'` cached startpos polygon.
pub(super) struct Poly {
    pub tag: char,
    pub pts: Vec<[i64; 2]>,
}

pub(super) struct Orient {
    pub polys: Vec<Poly>,
    /// x and y axis
    pub axes: [Axis; 2],
    /// per axis, per polygon: axis value index of every point
    idx: [Vec<Vec<usize>>; 2],
    src_area: Vec<i32>,
    src_turn: Vec<Vec<i32>>,
    /// per axis, per value index: polygons that use the value (ascending, each once)
    inc: [Vec<Vec<usize>>; 2],
}

fn turns(x: &[i64], y: &[i64]) -> Vec<i128> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let (p, q) = ((i + n - 1) % n, (i + 1) % n);
            (x[i] - x[p]) as i128 * (y[q] - y[i]) as i128 - (y[i] - y[p]) as i128 * (x[q] - x[i]) as i128
        })
        .collect()
}

fn area2(x: &[i64], y: &[i64]) -> i128 {
    let n = x.len();
    (0..n)
        .map(|i| x[i] as i128 * y[(i + 1) % n] as i128 - x[(i + 1) % n] as i128 * y[i] as i128)
        .sum()
}

impl Orient {
    pub fn new(polys: Vec<Poly>, ax: Axis, ay: Axis) -> Result<Self> {
        let axes = [ax, ay];
        let mut idx: [Vec<Vec<usize>>; 2] = [Vec::with_capacity(polys.len()), Vec::with_capacity(polys.len())];
        for poly in &polys {
            for a in 0..2 {
                let ids = poly
                    .pts
                    .iter()
                    .map(|p| {
                        axes[a]
                            .index_of(p[a])
                            .ok_or_else(|| anyhow::anyhow!("axis {}: polygon value {} is not in the map", axes[a].name, p[a]))
                    })
                    .collect::<Result<Vec<_>>>()?;
                idx[a].push(ids);
            }
        }
        let mut src_area = Vec::with_capacity(polys.len());
        let mut src_turn = Vec::with_capacity(polys.len());
        for poly in &polys {
            if poly.pts.len() < 3 {
                src_area.push(0);
                src_turn.push(Vec::new());
            } else {
                let x: Vec<i64> = poly.pts.iter().map(|p| p[0]).collect();
                let y: Vec<i64> = poly.pts.iter().map(|p| p[1]).collect();
                src_area.push(sgn(area2(&x, &y)));
                src_turn.push(turns(&x, &y).into_iter().map(sgn).collect());
            }
        }
        let mut inc: [Vec<Vec<usize>>; 2] = [vec![Vec::new(); axes[0].allv.len()], vec![Vec::new(); axes[1].allv.len()]];
        for a in 0..2 {
            for (pi, ids) in idx[a].iter().enumerate() {
                let mut uniq = ids.clone();
                uniq.sort_unstable();
                uniq.dedup();
                for vi in uniq {
                    inc[a][vi].push(pi);
                }
            }
        }
        Ok(Self { polys, axes, idx, src_area, src_turn, inc })
    }

    fn coords(&self, pi: usize) -> (Vec<i64>, Vec<i64>) {
        (
            self.idx[0][pi].iter().map(|&i| self.axes[0].t[i]).collect(),
            self.idx[1][pi].iter().map(|&i| self.axes[1].t[i]).collect(),
        )
    }

    /// `(area_flip, turn_flips, reflex_zero_turns, convex_nonzero_zero_turns)` of polygon `pi` under the current maps.
    /// A zero source turn (collinear triple) is best kept exactly zero; if it cannot be, it must at least not turn
    /// against the polygon's area sign (reflex corner).
    pub fn stats(&self, pi: usize) -> (i64, i64, i64, i64) {
        if self.idx[0][pi].len() < 3 {
            return (0, 0, 0, 0);
        }
        let (x, y) = self.coords(pi);
        let sa = self.src_area[pi];
        let af = (sa != 0 && sgn(area2(&x, &y)) != sa) as i64;
        let (mut tf, mut reflex, mut soft) = (0i64, 0i64, 0i64);
        for (&s, v) in self.src_turn[pi].iter().zip(turns(&x, &y)) {
            if s != 0 {
                tf += (sgn(v) != s) as i64;
            } else if v != 0 && sa != 0 {
                if sgn(v) == sa {
                    soft += 1;
                } else {
                    reflex += 1;
                }
            }
        }
        (af, tf, reflex, soft)
    }

    pub fn cost(&self, pi: usize) -> i64 {
        let (af, tf, reflex, soft) = self.stats(pi);
        af * AREA_FLIP_COST + tf * TURN_FLIP_COST + reflex * REFLEX_COST + soft
    }

    fn unique_ids(&self, a: usize, pi: usize) -> Vec<usize> {
        let mut v = self.idx[a][pi].clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn move_allowed(&self, a: usize, vi: usize, d: i64, max_shift: i64) -> bool {
        let axis = &self.axes[a];
        let new = axis.t[vi] + d;
        !((new - axis.init[vi]).abs() > max_shift
            || (vi > 0 && new <= axis.t[vi - 1])
            || (vi + 1 < axis.t.len() && new >= axis.t[vi + 1]))
    }

    /// Greedy local search: move unpinned shared coordinate values by +-1.. raw units (never beyond `max_shift` from
    /// plain rounding, never onto/past a neighbour value) whenever that strictly lowers the flip cost of all polygons
    /// that use the value. Returns `(moves, polygons still with a sign flip, polygons with only soft terms left)`.
    pub fn repair(&mut self, max_shift: i64, passes: usize) -> (usize, usize, usize) {
        let mut cost: Vec<i64> = (0..self.polys.len()).map(|pi| self.cost(pi)).collect();
        let deltas: Vec<i64> = (1..=max_shift).flat_map(|m| [m, -m]).collect();
        let mut moves = 0usize;
        for _ in 0..passes {
            let bad: Vec<usize> = cost.iter().enumerate().filter(|(_, c)| **c > 0).map(|(i, _)| i).collect();
            if bad.is_empty() {
                break;
            }
            let mut progressed = false;
            for pi in bad {
                if cost[pi] == 0 {
                    continue;
                }
                let mut cands: Vec<(i64, usize, usize, usize, i64)> = Vec::new();
                for a in 0..2 {
                    for vi in self.unique_ids(a, pi) {
                        if !self.axes[a].pinned[vi] {
                            for &d in &deltas {
                                cands.push((d.abs(), self.inc[a][vi].len(), a, vi, d));
                            }
                        }
                    }
                }
                cands.sort_unstable();
                for (_, _, a, vi, d) in cands {
                    if !self.move_allowed(a, vi, d, max_shift) {
                        continue;
                    }
                    let users = std::mem::take(&mut self.inc[a][vi]);
                    let before: i64 = users.iter().map(|&j| cost[j]).sum();
                    let old = self.axes[a].t[vi];
                    self.axes[a].t[vi] = old + d;
                    let after: Vec<i64> = users.iter().map(|&j| self.cost(j)).collect();
                    let improved = after.iter().sum::<i64>() < before;
                    if improved {
                        for (&j, &c) in users.iter().zip(&after) {
                            cost[j] = c;
                        }
                        moves += 1;
                        progressed = true;
                    } else {
                        self.axes[a].t[vi] = old;
                    }
                    self.inc[a][vi] = users;
                    if improved {
                        break;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        // Joint moves: a near-collinear vertex triple may need two shared coordinates moved together.
        let pending: Vec<usize> = cost.iter().enumerate().filter(|(_, c)| **c > 0).map(|(i, _)| i).collect();
        for pi in pending {
            if cost[pi] == 0 {
                continue;
            }
            let mut singles: Vec<(usize, usize, i64)> = Vec::new();
            for a in 0..2 {
                for vi in self.unique_ids(a, pi) {
                    if !self.axes[a].pinned[vi] {
                        for &d in &deltas {
                            singles.push((a, vi, d));
                        }
                    }
                }
            }
            singles.sort_by_key(|m| m.2.abs());
            let mut best: Option<(i64, (usize, usize, i64), (usize, usize, i64))> = None;
            for i in 0..singles.len() {
                let m1 = singles[i];
                for j in i + 1..singles.len() {
                    let m2 = singles[j];
                    if (m1.0, m1.1) == (m2.0, m2.1) {
                        continue;
                    }
                    let total = m1.2.abs() + m2.2.abs();
                    if let Some(b) = &best {
                        if total >= b.0 {
                            continue;
                        }
                    }
                    let mut applied: Vec<(usize, usize, i64)> = Vec::with_capacity(2);
                    let mut ok = true;
                    for &(a, vi, d) in &[m1, m2] {
                        if !self.move_allowed(a, vi, d, max_shift) {
                            ok = false;
                            break;
                        }
                        applied.push((a, vi, self.axes[a].t[vi]));
                        self.axes[a].t[vi] += d;
                    }
                    if ok {
                        let mut users: Vec<usize> = self.inc[m1.0][m1.1].iter().chain(&self.inc[m2.0][m2.1]).copied().collect();
                        users.sort_unstable();
                        users.dedup();
                        let before: i64 = users.iter().map(|&u| cost[u]).sum();
                        let after: i64 = users.iter().map(|&u| self.cost(u)).sum();
                        if after < before && best.as_ref().is_none_or(|b| total < b.0) {
                            best = Some((total, m1, m2));
                        }
                    }
                    for &(a, vi, old) in applied.iter().rev() {
                        self.axes[a].t[vi] = old;
                    }
                }
            }
            if let Some((_, m1, m2)) = best {
                let mut users: Vec<usize> = Vec::new();
                for (a, vi, d) in [m1, m2] {
                    self.axes[a].t[vi] += d;
                    users.extend_from_slice(&self.inc[a][vi]);
                }
                users.sort_unstable();
                users.dedup();
                for j in users {
                    cost[j] = self.cost(j);
                }
                moves += 2;
            }
        }
        let hard = cost.iter().filter(|&&c| c >= TURN_FLIP_COST).count();
        let soft = cost.iter().filter(|&&c| 0 < c && c < TURN_FLIP_COST).count();
        (moves, hard, soft)
    }

    /// Before/after counters per polygon class (`pathfinding_pieces`, `cached_startpos`).
    pub fn report(&self) -> Json {
        let mut out = serde_json::Map::new();
        for (tag, name) in [('p', "pathfinding_pieces"), ('c', "cached_startpos")] {
            let mut c = Counter::new();
            for (pi, poly) in self.polys.iter().enumerate() {
                if poly.tag != tag {
                    continue;
                }
                c.add("polygons", 1);
                let sa = self.src_area[pi];
                if sa == 0 {
                    c.add("source_zero_area", 1);
                    continue;
                }
                let (af, tf, reflex, soft) = self.stats(pi);
                let (x, y) = self.coords(pi);
                c.add("area_sign_flips", af);
                c.add("new_zero_area", (area2(&x, &y) == 0) as i64);
                c.add("polygons_with_turn_flip", (tf > 0) as i64);
                c.add("turn_flips", tf);
                c.add("source_nonzero_turns", self.src_turn[pi].iter().filter(|&&s| s != 0).count() as i64);
                c.add("source_collinear_turns", self.src_turn[pi].iter().filter(|&&s| s == 0).count() as i64);
                c.add("collinear_turns_now_nonzero", reflex + soft);
                c.add("collinear_turns_now_reflex", reflex);
                c.add("collinear_turns_now_convex", soft);
            }
            out.insert(name.to_string(), c.to_json());
        }
        Json::Object(out)
    }

    /// Sum of `area_sign_flips` and `new_zero_area` over every class of a [`report`](Self::report).
    pub fn area_violations(report: &Json) -> i64 {
        report
            .as_object()
            .map(|m| {
                m.values()
                    .map(|v| v.get("area_sign_flips").and_then(Json::as_i64).unwrap_or(0) + v.get("new_zero_area").and_then(Json::as_i64).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn into_axes(self) -> [Axis; 2] {
        self.axes
    }
}

/// The POC's shift schedule: shifts `1..=max_shift` are tried in turn and the smallest sufficient one is used.
/// Returns `(repair moves, hard polygons left, soft polygons left)`; errors when a flipped/zero area remains.
pub(super) fn repair_schedule(orient: &mut Orient, max_shift: i64, label: &str) -> Result<(usize, usize, usize, Json, Json)> {
    let before = orient.report();
    let (mut moves, mut hard_left, mut soft_left) = (0usize, 0usize, 0usize);
    for shift in 1..=max_shift {
        let (m, h, s) = orient.repair(shift, 40);
        moves += m;
        hard_left = h;
        soft_left = s;
        if hard_left == 0 && (shift >= MIN_SOFT_SHIFT || soft_left == 0) {
            break;
        }
    }
    for axis in &orient.axes {
        axis.check()?;
    }
    let after = orient.report();
    let area_left = Orient::area_violations(&after);
    if area_left != 0 {
        bail!("{label}orientation repair left {area_left} polygons with a flipped/zero area: {after}");
    }
    Ok((moves, hard_left, soft_left, before, after))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const N: i64 = 1_417_358;
    const U: i64 = 1 << 20;

    fn axis(name: &str, values: &[i64], pins: &[i64]) -> Axis {
        let plain = |v: i64| -> Result<i64> { Ok(super::super::util::round_div_even(v as i128 * N as i128, U as i128) as i64) };
        Axis::new(
            name,
            &values.iter().copied().collect::<BTreeSet<_>>(),
            &pins.iter().copied().collect::<BTreeSet<_>>(),
            N,
            0,
            U,
            &plain,
        )
        .unwrap()
    }

    /// Small near-degenerate triangles on a pinned axis: after the repair schedule no area/turn sign may differ from
    /// the source and every axis stays strictly increasing.
    #[test]
    fn repair_restores_turn_signs_and_keeps_pins() {
        // pins U and 3U anchor the maps; the polygons' values sit a few raw units after pin U
        let xs = [0, 3, 6, 7];
        let ys = [0, 1, 2];
        let ax = axis("x", &xs.map(|v| v + U), &[U, 3 * U]);
        let ay = axis("y", &ys.map(|v| v + U), &[U, 3 * U]);
        let pts = |a: [i64; 2], b: [i64; 2], c: [i64; 2]| vec![a, b, c].into_iter().map(|p| [p[0] + U, p[1] + U]).collect::<Vec<_>>();
        let polys = vec![Poly { tag: 'p', pts: pts([0, 0], [3, 2], [6, 1]) }, Poly { tag: 'c', pts: pts([0, 0], [7, 1], [3, 2]) }];
        let mut o = Orient::new(polys, ax, ay).unwrap();
        let (_, hard, _, _, after) = repair_schedule(&mut o, 8, "").unwrap();
        assert_eq!(hard, 0);
        assert_eq!(Orient::area_violations(&after), 0);
        for a in &o.axes {
            assert!(a.t.windows(2).all(|w| w[0] < w[1]));
        }
        for pi in 0..o.polys.len() {
            assert_eq!(o.stats(pi).0, 0);
            assert_eq!(o.stats(pi).1, 0);
        }
    }

    #[test]
    fn unknown_polygon_value_is_an_error() {
        let ax = axis("x", &[U], &[U]);
        let ay = axis("y", &[U], &[U]);
        let polys = vec![Poly { tag: 'p', pts: vec![[U, U], [U + 5, U], [U, U]] }];
        assert!(Orient::new(polys, ax, ay).is_err());
    }
}
