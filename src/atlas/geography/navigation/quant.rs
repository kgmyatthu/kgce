//! Per-chart Q20 -> Q20 quantization maps (plain rule + grid-line pins + orientation repair) for every pathfinding
//! vertex, POI and campaign position of the source, each owned by exactly one chart. Never a global map.

use std::collections::BTreeSet;

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value as Json, json};

use super::axis::Axis;
use super::ctx::{AreaRow, Ctx};
use super::flat::Reader;
use super::grid::{areas, checked_head, path_polygons, vertex_pairs};
use super::orient::{Orient, Poly, repair_schedule};
use super::util::{SENT_VERTEX, UNIT, rects_cover};
use crate::atlas::geography::startpos_points::{cached_grid_charts, cached_polygons_by_grid, owned_fixed_points};
use crate::esf::Node;
use crate::model::Progress;

/// Upper bound (raw Q20 units) of a coordinate's deviation from plain rounding while repairing polygon orientations;
/// shifts 1..=N are tried in turn and the smallest sufficient one is used.
pub(super) const MAX_SHIFT: i64 = 8;

/// `(x, y)` Q20 raw positions of every POI (same stream walk as `do_poi`, read only).
pub(super) fn poi_positions(root: &Node) -> Result<Vec<(i32, i32)>> {
    let vals = root.children()?;
    let mut r = Reader::new(vals);
    let n = r.u32()?;
    let mut out = Vec::with_capacity((n as usize).min(vals.len()));
    for _ in 0..n {
        r.u32()?;
        r.i32()?;
        r.bool()?;
        let (x, y) = (r.i32()?, r.i32()?);
        if x == SENT_VERTEX || y == SENT_VERTEX {
            bail!("sentinel position in poi");
        }
        out.push((x, y));
        r.string()?;
        r.u32()?;
        r.string()?;
        r.u32()?;
        r.f32()?;
        for _ in 0..r.u32()? {
            r.string()?;
            r.f32()?;
        }
        r.f32()?;
        for _ in 0..r.u32()? {
            r.string()?;
            r.f32()?;
        }
        for _ in 0..r.u32()? {
            r.u32()?;
        }
        r.u32()?;
        r.bool()?;
    }
    r.finish()?;
    Ok(out)
}

/// Pathfinding area -> owning chart: the unique chart whose crop union wholly contains the area's grid.
pub(super) fn area_table(ctx: &Ctx, pf: &Node) -> Result<Vec<AreaRow>> {
    let mut rows = Vec::new();
    for (k, item) in areas(pf)?.1.iter().enumerate() {
        let h = checked_head(k, item)?;
        let (ox, oy, cs) = (h.ox as i64, h.oy as i64, h.cs as i64);
        let (cols, nrows) = (h.cols as i64, h.rows as i64);
        let u = UNIT as f64;
        let lo = [ox as f64 / u, oy as f64 / u];
        let hi = [(ox + cols * cs) as f64 / u, (oy + nrows * cs) as f64 / u];
        let hits: Vec<usize> = (0..ctx.plan.charts.len())
            .filter(|&c| rects_cover(&ctx.plan.charts[c].crop_rects, lo, hi))
            .collect();
        if hits.len() != 1 {
            bail!(
                "area {k}: grid rectangle {lo:?}..{hi:?} lies wholly inside the crop of {} charts {:?} (needs exactly one); \
                 a grid cannot be clipped",
                hits.len(),
                hits.iter().map(|&c| ctx.chart_key(c)).collect::<Vec<_>>()
            );
        }
        rows.push(AreaRow {
            area: k,
            chart: hits[0],
            origin: [ox, oy],
            cell: cs,
            cols,
            rows: nrows,
            src_rect_world: [lo, hi],
            align_remainder_raw: None,
            new: None,
        });
    }
    Ok(rows)
}

pub(super) struct Quantized {
    /// per chart key: x/y axis reports and the orientation report
    pub reports: Map<String, Json>,
    /// per chart key: `{x: {src_raw: dst_raw}, y: {...}}` of the entries that differ from the plain rule
    pub charts_doc: Map<String, Json>,
}

fn orientation_json(max_shift: i64, polygons: usize, moves: usize, hard: usize, soft: usize, before: Json, after: Json) -> Json {
    json!({
        "constraint": "sign of signed area and sign of every nonzero vertex turn of each polygon; strict per-axis order; \
                       pinned grid lines; max deviation from plain rounding per axis = max_shift",
        "max_shift_raw_q20": max_shift,
        "polygons": polygons,
        "repair_moves": moves,
        "polygons_still_violating": {"sign_flip_polygons": hard, "collinear_turn_polygons": soft},
        "before_repair": before,
        "after_repair": after,
    })
}

/// Fills `ctx.areas` and `ctx.q`; see the module doc.
pub(super) fn build_quantization_atlas(
    ctx: &mut Ctx,
    pf: &Node,
    poi: &Node,
    startpos: &Node,
    progress: &Progress<'_>,
) -> Result<Quantized> {
    let plan = ctx.plan;
    ctx.areas = area_table(ctx, pf)?;
    let nch = plan.charts.len();
    let mut vals: Vec<[BTreeSet<i64>; 2]> = vec![[BTreeSet::new(), BTreeSet::new()]; nch];
    let mut pins: Vec<[BTreeSet<i64>; 2]> = vec![[BTreeSet::new(), BTreeSet::new()]; nch];
    let mut counts: Vec<super::util::Counter> = vec![super::util::Counter::new(); nch];
    for (k, item) in areas(pf)?.1.iter().enumerate() {
        let (ci, origin, cell, cols, rows) = {
            let row = &ctx.areas[k];
            (row.chart, row.origin, row.cell, row.cols, row.rows)
        };
        for (vi, p) in vertex_pairs(item)?.into_iter().enumerate() {
            if vi >= 4 {
                if p[0] == SENT_VERTEX || p[1] == SENT_VERTEX {
                    bail!("area {k}: sentinel inside vertex {vi}");
                }
                vals[ci][0].insert(p[0] as i64);
                vals[ci][1].insert(p[1] as i64);
                counts[ci].add("pathfinding_vertices", 1);
            }
        }
        pins[ci][0].extend((0..=cols).map(|i| origin[0] + i * cell));
        pins[ci][1].extend((0..=rows).map(|i| origin[1] + i * cell));
    }
    progress("navigation: POI ownership");
    for (x, y) in poi_positions(poi)? {
        let ci = ctx.poi_chart(x, y)?;
        vals[ci][0].insert(x as i64);
        vals[ci][1].insert(y as i64);
        counts[ci].add("poi_positions", 1);
    }
    progress("navigation: startpos fixed points");
    let got = cached_grid_charts(startpos, plan)?;
    let want: Vec<String> = ctx.areas.iter().map(|r| ctx.chart_key(r.chart).to_string()).collect();
    if got != want {
        bail!("startpos cached grid charts {got:?} != pathfinding area charts {want:?}");
    }
    for point in owned_fixed_points(startpos, plan)? {
        let [x, y] = point.point;
        if x == SENT_VERTEX || y == SENT_VERTEX {
            bail!("sentinel position in startpos fixed point stream ({})", point.record);
        }
        if point.record == "CAI_REGION_HLCI" && point.chart.is_none() {
            // These representatives are regenerated from the retained region-group geometry, not transformed source
            // points; the campaign conversion validates their new area coverage.
            continue;
        }
        let ci = point
            .chart
            .as_deref()
            .and_then(|k| plan.charts.iter().position(|c| c.key == k))
            .ok_or_else(|| anyhow!("startpos point of {} owned by unknown chart {:?}", point.record, point.chart))?;
        vals[ci][0].insert(x as i64);
        vals[ci][1].insert(y as i64);
        counts[ci].add("startpos_points", 1);
    }
    let mut polys: Vec<Vec<Poly>> = (0..nch).map(|_| Vec::new()).collect();
    for (k, pts) in path_polygons(pf)? {
        polys[ctx.areas[k].chart].push(Poly { tag: 'p', pts });
    }
    for (gi, pts) in cached_polygons_by_grid(startpos)? {
        let ci = ctx.areas.get(gi).ok_or_else(|| anyhow!("cached grid {gi} has no pathfinding area"))?.chart;
        for p in &pts {
            vals[ci][0].insert(p[0] as i64);
            vals[ci][1].insert(p[1] as i64);
        }
        polys[ci].push(Poly { tag: 'c', pts: pts.iter().map(|p| [p[0] as i64, p[1] as i64]).collect() });
    }
    let mut reports = Map::new();
    let mut charts_doc = Map::new();
    for ci in 0..nch {
        if !(0..2).any(|a| !vals[ci][a].is_empty() || !pins[ci][a].is_empty()) {
            continue;
        }
        let key = plan.charts[ci].key.clone();
        progress(&format!("navigation: quantizing chart {key}"));
        let offw = plan.charts[ci].offset;
        let off = ctx.off_raw[ci];
        let make = |a: usize, name: &str| -> Result<Axis> {
            let expected = |v: i64| -> Result<i64> {
                Ok(plan.fixed_value(v as f64 / UNIT as f64 * plan.scale + offw[a])? as i64)
            };
            Axis::new(&format!("{key}.{name}"), &vals[ci][a], &pins[ci][a], ctx.scale_n, off[a], UNIT, &expected)
        };
        let (ax, ay) = (make(0, "x")?, make(1, "y")?);
        let chart_polys = std::mem::take(&mut polys[ci]);
        let n_polys = chart_polys.len();
        let mut orient = Orient::new(chart_polys, ax, ay)?;
        let (moves, hard, soft, before, after) = repair_schedule(&mut orient, MAX_SHIFT, &format!("chart {key}: "))?;
        let [ax, ay] = orient.into_axes();
        let mut rep = Map::new();
        let mut devs = Vec::new();
        let mut maps = Vec::new();
        for (axis, name) in [(&ax, "x"), (&ay, "y")] {
            let (mut r, dev) = axis.report();
            for (k, v) in counts[ci].iter() {
                r.insert(k.to_string(), json!(v));
            }
            rep.insert(name.to_string(), Json::Object(r));
            devs.push(dev);
            maps.push(axis.mapping());
        }
        let my = maps.pop().unwrap();
        let mx = maps.pop().unwrap();
        ctx.q[ci] = Some([mx, my]);
        ctx.note("quantized_values", ax.t[0], ay.t[0]);
        ctx.note("quantized_values", *ax.t.last().unwrap(), *ay.t.last().unwrap());
        rep.insert("orientation".to_string(), orientation_json(MAX_SHIFT, n_polys, moves, hard, soft, before, after));
        reports.insert(key.clone(), Json::Object(rep));
        let mut doc = Map::new();
        for (name, dev) in [("x", &devs[0]), ("y", &devs[1])] {
            let mut m = Map::new();
            for (src, dst) in dev {
                m.insert(src.to_string(), json!(dst));
            }
            doc.insert(name.to_string(), Json::Object(m));
        }
        charts_doc.insert(key, Json::Object(doc));
    }
    Ok(Quantized { reports, charts_doc })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf;

    fn poi_node(x: i32, y: i32) -> Vec<Node> {
        vec![
            esf::u32v(0),
            esf::i32v(5),
            esf::b(false),
            esf::i32v(x),
            esf::i32v(y),
            esf::s16("france"),
            esf::u32v(13),
            esf::s16("france"),
            esf::u32v(13),
            esf::f32v(1.0),
            esf::u32v(1),
            esf::s16("spain"),
            esf::f32v(2.0),
            esf::f32v(2.0),
            esf::u32v(0),
            esf::u32v(2),
            esf::u32v(7),
            esf::u32v(8),
            esf::u32v(0),
            esf::b(true),
        ]
    }

    #[test]
    fn poi_positions_walk_the_flat_stream() {
        let mut vals = vec![esf::u32v(2)];
        vals.extend(poi_node(10, 20));
        vals.extend(poi_node(-5, 6));
        let root = Node::record("CAI_POI_ROOT", 0, vals);
        assert_eq!(poi_positions(&root).unwrap(), vec![(10, 20), (-5, 6)]);
    }

    #[test]
    fn poi_stream_errors_are_reported() {
        let mut vals = vec![esf::u32v(1)];
        vals.extend(poi_node(SENT_VERTEX, 0));
        assert!(poi_positions(&Node::record("CAI_POI_ROOT", 0, vals)).is_err());
        let mut vals = vec![esf::u32v(1)];
        vals.extend(poi_node(1, 2));
        vals.push(esf::u32v(9));
        assert!(poi_positions(&Node::record("CAI_POI_ROOT", 0, vals)).is_err());
        let mut vals = vec![esf::u32v(2)];
        vals.extend(poi_node(1, 2));
        assert!(poi_positions(&Node::record("CAI_POI_ROOT", 0, vals)).is_err());
    }
}
