//! `pathfinding.esf`: 7 `pathfinding_areas` items (vertices[], u32 polygon index stream, grid_data).
//!
//! * vertices: the first four pairs are `(-2147483647, -2147483647)` dummies referenced by the corner ids 0..3 and are
//!   kept raw; every other pair goes through the per-chart quantization map (no sentinel allowed);
//! * grid_data: origin, cell size and the u16 start indices (rebased onto the stock world minimum, never wrapping);
//!   cols/rows/[7..9]/arr3/arr7/grid_cells (header bytes, packed boundary words, ids) are untouched;
//! * the polygon index stream is untouched.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value as Json, json};

use super::ctx::{Ctx, NewGrid};
use super::diff::Touched;
use super::grid::{areas, cells, checked_head, corner, for_each_piece};
use super::util::{Bounds, Counter, I32_SPAN, SENT_VERTEX, UNIT, floor_div, index_of, sgn, wrap_i32};
use crate::esf::{Node, Value};

/// Global cell index of the grid origin as stored in grid_data[2]/[3]: `round_half_up((origin - world_min) / cell)`,
/// the subtraction done in a wrapping i32 when `wrap` (the 2^32/2^20 = 4096 world-unit overflow of the baseline
/// engine; EastIndies startX 64528 = u16(-1008)).
pub(super) fn start_index(origin_raw: i64, world_min_raw: i64, cell_raw: i64, wrap: bool) -> Result<i64> {
    if cell_raw <= 0 {
        bail!("grid cell size {cell_raw} is not positive");
    }
    let mut num = origin_raw - world_min_raw;
    if wrap {
        num = wrap_i32(num);
    } else if !(-I32_SPAN..I32_SPAN).contains(&num) {
        bail!("world-relative offset outside i32");
    }
    Ok(floor_div(2 * num as i128 + cell_raw as i128, 2 * cell_raw as i128) as i64)
}

/// Every piece polygon (corner ids 0..3 resolved against its cell) before and after: orientation (signed area sign),
/// zero-length edges, containment in the cell rectangle.
#[allow(clippy::too_many_arguments)]
fn check_polygons(
    k: usize,
    item: &[Node],
    old_pairs: &[[i64; 2]],
    new_pairs: &[[i64; 2]],
    old: (&super::grid::GridHead, i64),
    new: (&super::grid::GridHead, i64),
) -> Result<Json> {
    let cols = old.0.cols as i64;
    let rows = old.0.rows as i64;
    let mut c = Counter::new();
    let mut pieces = 0i64;
    let walked = for_each_piece(k, item, cols, |col, row, ids| {
        pieces += 1;
        let mut res = Vec::with_capacity(2);
        for (head, cell, pairs) in [(old.0, old.1, old_pairs), (new.0, new.1, new_pairs)] {
            let mut pts = Vec::with_capacity(ids.len());
            for &i in ids {
                pts.push(if i < 4 {
                    corner(head, col, row, cell, i)
                } else {
                    *pairs
                        .get(i as usize - 4)
                        .ok_or_else(|| anyhow!("area {k}: vertex id {i} outside the vertex table"))?
                });
            }
            let n = pts.len();
            let (cx, cy) = (head.ox as i64 + col * cell, head.oy as i64 + row * cell);
            let area2: i128 = (0..n)
                .map(|a| {
                    let b = (a + 1) % n;
                    pts[a][0] as i128 * pts[b][1] as i128 - pts[b][0] as i128 * pts[a][1] as i128
                })
                .sum();
            let zero = (0..n).filter(|&a| pts[a] == pts[(a + 1) % n]).count() as i64;
            let outside = pts
                .iter()
                .filter(|q| !(cx <= q[0] && q[0] <= cx + cell && cy <= q[1] && q[1] <= cy + cell))
                .count() as i64;
            let turns: Vec<i128> = (0..n)
                .map(|i| {
                    let (p, q) = ((i + n - 1) % n, (i + 1) % n);
                    (pts[i][0] - pts[p][0]) as i128 * (pts[q][1] - pts[i][1]) as i128
                        - (pts[i][1] - pts[p][1]) as i128 * (pts[q][0] - pts[i][0]) as i128
                })
                .collect();
            res.push((area2, zero, outside, turns));
        }
        let (a0, z0, o0, t0) = (&res[0].0, res[0].1, res[0].2, &res[0].3);
        let (a1, z1, o1, t1) = (&res[1].0, res[1].1, res[1].2, &res[1].3);
        c.add("turn_flips", t0.iter().zip(t1).filter(|(u, v)| **u != 0 && sgn(**u) != sgn(**v)).count() as i64);
        c.add("area_zero_old", (*a0 == 0) as i64);
        c.add("orientation_flips", (sgn(*a0) != sgn(*a1)) as i64);
        c.add("zero_length_edges_old", z0);
        c.add("zero_length_edges_new", z1);
        c.add("outside_cell_old", o0);
        c.add("outside_cell_new", o1);
        Ok(())
    })?;
    if walked != cols * rows {
        bail!("area {k}: walked {walked} cells, grid has {}", cols * rows);
    }
    if c.get("orientation_flips") != 0
        || c.get("turn_flips") != 0
        || c.get("zero_length_edges_new") > c.get("zero_length_edges_old")
        || c.get("outside_cell_new") > c.get("outside_cell_old")
    {
        bail!("area {k}: polygon check failed {}", c.to_json());
    }
    let mut m = Map::new();
    m.insert("pieces".into(), json!(pieces));
    for key in [
        "area_zero_old",
        "orientation_flips",
        "turn_flips",
        "zero_length_edges_old",
        "zero_length_edges_new",
        "outside_cell_old",
        "outside_cell_new",
    ] {
        m.insert(key.into(), json!(c.get(key)));
    }
    Ok(Json::Object(m))
}

/// Count of consecutive polygon edges whose two corners were merged onto one position.
fn collapsed_edges(k: usize, stream: &[u32], merged_ids: &HashMap<u32, [i64; 2]>) -> Result<i64> {
    let mut pos = 0usize;
    let mut zero = 0;
    while pos < stream.len() {
        let n = stream[pos] as usize;
        let poly = stream
            .get(pos + 1..pos + 1 + n)
            .ok_or_else(|| anyhow!("area {k}: polygon index stream does not end on a polygon boundary"))?;
        pos += 1 + n;
        for a in 0..poly.len() {
            let b = poly[(a + 1) % poly.len()];
            if poly[a] != b && merged_ids.contains_key(&poly[a]) && merged_ids.get(&b) == merged_ids.get(&poly[a]) {
                zero += 1;
            }
        }
    }
    Ok(zero)
}

pub(super) fn do_pathfinding(ctx: &mut Ctx, root: &mut Node, touched: &mut Touched) -> Result<Json> {
    let plan = ctx.plan;
    let wmin_old = [0, 1].map(|a| (plan.source_world[0][a] * UNIT as f64).round_ties_even() as i64);
    let wmin_new = [0, 1].map(|a| (plan.world[0][a] * UNIT as f64).round_ties_even() as i64);
    let root_areas = index_of(root.children()?, "pathfinding.esf root", "pathfinding_areas")?;
    let n_areas = root.children()?[root_areas].items()?.len();
    if n_areas != 7 {
        bail!("expected 7 pathfinding areas, found {n_areas}");
    }
    let mut report = Vec::new();
    let mut totals = Counter::new();
    let s = UNIT as f64;
    for k in 0..n_areas {
        let item: &mut Vec<Node> = &mut root.children_mut()?[root_areas].items_mut()?[k];
        let head = checked_head(k, item)?;
        let chart = ctx.area_chart(k)?;
        let (qx, qy) = {
            let q = ctx.axes(chart)?;
            (q[0].clone(), q[1].clone())
        };
        let (ox, oy, cs) = (head.ox as i64, head.oy as i64, head.cs as i64);
        let (cols, rows) = (head.cols as i64, head.rows as i64);
        let base = [root_areas, k];
        // ---------------- vertices
        let mut old_pairs: Vec<[i64; 2]> = Vec::new();
        let mut new_pairs: Vec<[i64; 2]> = Vec::new();
        let (mut ob, mut nb) = (Bounds::new(), Bounds::new());
        let mut dummies = 0i64;
        let total_vertices;
        {
            let verts = item[0].items_mut()?;
            total_vertices = verts.len();
            for (vi, vert) in verts.iter_mut().enumerate() {
                if vert.len() != 2 {
                    bail!("area {k} vertex {vi}: {} entries", vert.len());
                }
                let (x, y) = (super::util::node_i32(&vert[0])?, super::util::node_i32(&vert[1])?);
                if vi < 4 {
                    if (x, y) != (SENT_VERTEX, SENT_VERTEX) {
                        bail!("area {k}: vertex {vi} is not a dummy: {:?}", (x, y));
                    }
                    dummies += 1;
                    continue;
                }
                if x == SENT_VERTEX || y == SENT_VERTEX {
                    bail!("area {k}: sentinel inside vertex {vi}");
                }
                let nx = *qx.get(&(x as i64)).ok_or_else(|| anyhow!("area {k}: raw x {x} missing from the quantization map"))?;
                let ny = *qy.get(&(y as i64)).ok_or_else(|| anyhow!("area {k}: raw y {y} missing from the quantization map"))?;
                ctx.note("pathfinding.vertex", nx, ny);
                vert[0] = crate::esf::i32v(i32::try_from(nx).map_err(|_| anyhow!("area {k}: quantized x {nx} leaves i32"))?);
                vert[1] = crate::esf::i32v(i32::try_from(ny).map_err(|_| anyhow!("area {k}: quantized y {ny} leaves i32"))?);
                touched.add(&[base[0], base[1], 0, vi, 0]);
                touched.add(&[base[0], base[1], 0, vi, 1]);
                old_pairs.push([x as i64, y as i64]);
                new_pairs.push([nx, ny]);
                ob.add(x as f64 / s, y as f64 / s);
                nb.add(nx as f64 / s, ny as f64 / s);
            }
        }
        // ---------------- grid_data header
        let [nox, noy] = plan.fixed_xy([head.ox, head.oy], &plan.charts[chart])?.map(|v| v as i64);
        if (nox, noy) != (*qx.get(&ox).unwrap_or(&nox), *qy.get(&oy).unwrap_or(&noy)) {
            bail!("area {k}: grid origin disagrees with the quantization map");
        }
        let ncs = ctx.fixed_len(head.cs)? as i64;
        if !ctx.cell_exact(cs, ncs) {
            bail!("area {k}: cell size {cs} -> {ncs} is not an exact multiple of the scale");
        }
        // start indices: verify the engine model on the baseline, then derive them at Q20 (no wrap possible)
        let mut model = Vec::new();
        for (name, o, w, stored) in [("x", ox, wmin_old[0], head.sx), ("y", oy, wmin_old[1], head.sy)] {
            let nominal = if (-I32_SPAN..I32_SPAN).contains(&(o - w)) { Some(start_index(o, w, cs, false)?) } else { None };
            let wrapped = start_index(o, w, cs, true)? & 0xFFFF;
            if stored as i64 != wrapped {
                bail!("area {k} start{} {stored} != baseline engine model {wrapped}", name.to_uppercase());
            }
            model.push(json!({
                "axis": name, "stored": stored, "model_wrapped_u16": wrapped, "nominal_unwrapped": nominal,
                "overflowed": nominal.is_none() || nominal != Some(wrapped),
            }));
        }
        let mut remainders = [0i64; 2];
        for (a, (name, o, c_, w)) in [("x", nox, cols, wmin_new[0]), ("y", noy, rows, wmin_new[1])].into_iter().enumerate() {
            let far = o + c_ * ncs;
            if o < w || far - w >= I32_SPAN {
                bail!(
                    "area {k}: grid {name} range [{}, {}] leaves the stock grid-index window [world_min, world_min + 2048): \
                     the stock Q20 grid-index subtraction would wrap/clamp",
                    o as f64 / s,
                    far as f64 / s
                );
            }
            remainders[a] = (o - w).rem_euclid(ncs);
        }
        ctx.areas[k].align_remainder_raw = Some(remainders);
        ctx.note("pathfinding.grid", nox, noy);
        ctx.note("pathfinding.grid", nox + cols * ncs, noy + rows * ncs);
        let nsx = start_index(nox, wmin_new[0], ncs, false)?;
        let nsy = start_index(noy, wmin_new[1], ncs, false)?;
        for (name, v) in [("x", nsx), ("y", nsy)] {
            if !(0..=0xFFFF).contains(&v) {
                bail!("area {k} new start{name} {v} outside u16");
            }
        }
        ctx.areas[k].new = Some(NewGrid { origin: [nox, noy], cell: ncs, start: [nsx, nsy] });
        {
            let g = item[2].children_mut()?;
            g[0] = crate::esf::i32v(nox as i32);
            g[1] = crate::esf::i32v(noy as i32);
            g[2] = crate::esf::u16v(nsx as u16);
            g[3] = crate::esf::u16v(nsy as u16);
            g[4] = crate::esf::i32v(ncs as i32);
        }
        for idx in 0..5 {
            touched.add(&[base[0], base[1], 2, idx]);
        }
        // ---------------- geometry checks
        let item: &Vec<Node> = item;
        let ext_old = [ox + cols * cs, oy + rows * cs];
        let ext_new = [nox + cols * ncs, noy + rows * ncs];
        let bounds_ok = ob.lo[0] * s >= ox as f64
            && ob.lo[1] * s >= oy as f64
            && ob.hi[0] * s <= ext_old[0] as f64
            && ob.hi[1] * s <= ext_old[1] as f64;
        let nbounds_ok = nb.lo[0] * s >= nox as f64
            && nb.lo[1] * s >= noy as f64
            && nb.hi[0] * s <= ext_new[0] as f64
            && nb.hi[1] * s <= ext_new[1] as f64;
        if !(bounds_ok && nbounds_ok) {
            bail!("area {k}: vertices outside the grid rectangle (old ok={bounds_ok}, new ok={nbounds_ok})");
        }
        let (mut on_old, mut on_new, mut snapped_in, mut snapped_out) = (0i64, 0i64, 0i64, 0i64);
        for (p, q) in old_pairs.iter().zip(&new_pairs) {
            for (v, o, c, nv, no, nc) in [(p[0], ox, cs, q[0], nox, ncs), (p[1], oy, cs, q[1], noy, ncs)] {
                let a = (v - o).rem_euclid(c) == 0;
                let b = (nv - no).rem_euclid(nc) == 0;
                on_old += a as i64;
                on_new += b as i64;
                snapped_out += (a && !b) as i64;
                snapped_in += (b && !a) as i64;
            }
        }
        let distinct_old = old_pairs.iter().collect::<HashSet<_>>().len();
        let distinct_new = new_pairs.iter().collect::<HashSet<_>>().len();
        let inv = |pairs: &[[i64; 2]]| {
            pairs.windows(2).filter(|w| (w[0][1], w[0][0]) > (w[1][1], w[1][0])).count()
        };
        let (inv_old, inv_new) = (inv(&old_pairs), inv(&new_pairs));
        if snapped_out != 0 || snapped_in != 0 {
            bail!("area {k}: grid-line vertices moved by Q20->Q20 rounding: out={snapped_out} in={snapped_in}");
        }
        // Vertices whose positions coincide would collapse onto one position (ids stay distinct); count the ones that are
        // consecutive corners of a polygon (zero-length edge).
        let mut by_pos: HashMap<[i64; 2], Vec<u32>> = HashMap::new();
        for (off, q) in new_pairs.iter().enumerate() {
            by_pos.entry(*q).or_default().push(off as u32 + 4);
        }
        let merged: Vec<(&[i64; 2], &Vec<u32>)> = by_pos.iter().filter(|(_, ids)| ids.len() > 1).collect();
        let merged_ids: HashMap<u32, [i64; 2]> =
            merged.iter().flat_map(|(q, ids)| ids.iter().map(move |i| (*i, **q))).collect();
        let zero_edges = if merged_ids.is_empty() {
            0
        } else {
            collapsed_edges(k, &super::util::le_u32s(super::util::packed(&item[1], 0x48)?)?, &merged_ids)?
        };
        let rounding = json!({
            "merged_positions": merged.len(),
            "vertices_merged": merged.iter().map(|(_, ids)| ids.len()).sum::<usize>(),
            "polygon_edges_collapsed_to_zero_length": zero_edges,
            "sorted_order_inversions": {"old": inv_old, "new": inv_new},
            "note": "Q20->Q20 with scale >= 1.  The per-chart quantization map is strictly increasing and pins every grid line, so positions never merge and the (y,x) order is kept; vertex ids/topology are unchanged.",
        });
        if !merged.is_empty() || distinct_old != distinct_new || inv_old != inv_new {
            bail!("area {k}: quantization failed to keep vertices distinct/ordered");
        }
        let new_head = checked_head(k, item)?;
        let poly_report = check_polygons(k, item, &old_pairs, &new_pairs, (&head, cs), (&new_head, ncs))?;
        let index_bytes = super::util::packed(&item[1], 0x48)?.len();
        let g = item[2].children()?;
        let n_cells = cells(item)?.len();
        let on_lattice = remainders.iter().all(|r| *r == 0);
        report.push(json!({
            "area": k,
            "vertices": {
                "total": total_vertices, "dummies_kept_raw": dummies, "scaled": old_pairs.len(),
                "old_bounds_world": ob.dump(), "new_bounds_world": nb.dump(),
                "distinct_positions": {"old": distinct_old, "new": distinct_new},
                "sort_inversions": {"old": inv_old, "new": inv_new},
                "on_grid_line_coords": {"old": on_old, "new": on_new, "lost": snapped_out, "gained": snapped_in},
                "q20_to_q18_rounding": rounding,
            },
            "polygon_check": poly_report,
            "polygon_index_stream": {"array_code": 8, "bytes": index_bytes, "unchanged": true},
            "grid": {
                "cols": cols, "rows": rows,
                "origin_world": {"old": [ox as f64 / s, oy as f64 / s], "new": [nox as f64 / s, noy as f64 / s]},
                "origin_raw": {"old": [ox, oy], "new": [nox, noy]},
                "cell_size_world": {"old": cs as f64 / s, "new": ncs as f64 / s},
                "cell_size_raw": {"old": cs, "new": ncs},
                "extent_world": {
                    "old": [ext_old[0] as f64 / s, ext_old[1] as f64 / s],
                    "new": [ext_new[0] as f64 / s, ext_new[1] as f64 / s],
                },
                "start_xy": {"old": [head.sx, head.sy], "new": [nsx, nsy]},
                "start_model": model,
                "untouched": {
                    "cells": n_cells,
                    "header_fields": [header_field(&g[7])?, header_field(&g[8])?, header_field(&g[9])?],
                    "arr3_bytes": raw_len(&g[10])?, "arr7_bytes": raw_len(&g[11])?,
                },
            },
            "atlas": {
                "chart": ctx.chart_key(chart), "offset_raw": ctx.off_raw[chart], "world_min_raw": wmin_new,
                "start_alignment_remainder_raw": remainders, "grid_aligned_to_world_lattice": on_lattice,
            },
        }));
        totals.add("vertices_scaled", old_pairs.len() as i64);
        totals.add("dummy_vertices_kept", dummies);
        totals.add("grid_cells", n_cells as i64);
    }
    Ok(json!({"areas": report, "totals": totals.to_json()}))
}

fn header_field(n: &Node) -> Result<i64> {
    match n {
        Node::Value(Value::U32(v)) => Ok(*v as i64),
        Node::Value(Value::U16(v)) => Ok(*v as i64),
        other => bail!("unexpected grid header field {other:?}"),
    }
}

fn raw_len(n: &Node) -> Result<usize> {
    Ok(n.raw_array()?.len())
}

/// Region ids (`grid_data[10]`, i16 regions.esf indices) of every area: min and max.
pub(super) fn region_id_range(root: &Node) -> Result<Option<(i16, i16)>> {
    let mut out: Option<(i16, i16)> = None;
    for item in areas(root)?.1 {
        let raw = super::util::packed(&item[2].children()?[10], 0x43)?;
        if raw.len() % 2 != 0 {
            bail!("region id array holds {} bytes", raw.len());
        }
        for c in raw.chunks_exact(2) {
            let v = i16::from_le_bytes([c[0], c[1]]);
            out = Some(match out {
                None => (v, v),
                Some((lo, hi)) => (lo.min(v), hi.max(v)),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_index_rounds_half_up_and_wraps_like_the_baseline_engine() {
        let cell = 2 * UNIT; // 2 world units
        let wmin = -1280 * UNIT;
        // exact multiples
        assert_eq!(start_index(wmin + 10 * cell, wmin, cell, false).unwrap(), 10);
        // half a cell rounds UP, just below rounds down
        assert_eq!(start_index(wmin + 10 * cell + UNIT, wmin, cell, false).unwrap(), 11);
        assert_eq!(start_index(wmin + 10 * cell + UNIT - 1, wmin, cell, false).unwrap(), 10);
        // negative offsets floor correctly (origin left of the world minimum)
        assert_eq!(start_index(wmin - cell, wmin, cell, false).unwrap(), -1);
        assert_eq!(start_index(wmin - UNIT, wmin, cell, false).unwrap(), 0);
        // EastIndies: 800 + 1280 world units = 2080 > 2048 overflows i32; wrapped u16 start is 64528 for cell 2
        let origin = 800 * UNIT;
        assert!(start_index(origin, wmin, cell, false).is_err());
        assert_eq!(start_index(origin, wmin, cell, true).unwrap() & 0xFFFF, 64528);
    }

    #[test]
    fn start_index_rejects_non_positive_cells() {
        assert!(start_index(0, 0, 0, false).is_err());
        assert!(start_index(0, 0, -1, true).is_err());
    }

    #[test]
    fn collapsed_edges_count_adjacent_merged_corners_and_reject_truncated_streams() {
        let merged: HashMap<u32, [i64; 2]> = HashMap::from([(4, [1, 1]), (5, [1, 1]), (6, [2, 2])]);
        // triangle 4,5,7 has the edge 4-5 collapsed; polygon 6,7,8 none
        assert_eq!(collapsed_edges(0, &[3, 4, 5, 7, 3, 6, 7, 8], &merged).unwrap(), 1);
        assert!(collapsed_edges(0, &[3, 4, 5], &merged).is_err());
    }

    #[test]
    fn region_id_range_reads_every_area() {
        let root = super::super::grid::test_support::small_root();
        // the fixture's i16 array is empty
        assert_eq!(region_id_range(&root).unwrap(), None);
    }
}
