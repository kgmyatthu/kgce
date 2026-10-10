//! `sea_grids.esf`: flat `CAI_SEA_GRID_ROOT` stream, per theatre
//! `s16 name, xy min, xy max, f32 cell size, u32 cols, u32 rows`, `cols*rows * (u32 xi, u32 yi, xy lo, xy hi)`, per
//! cell an area `u32 id, 3 * (u32 n + n*s16), u32 n + n*u32`, and finally `u32 n + n*(u32, u32, f32)` connections.
//!
//! Every position goes through the affine transform of its theatre's chart. The theatre `min`/`max` corners (the third
//! copy of the theatre rectangle next to the two stored in regions.esf) are the camera limits: they are Q12 camera
//! snapped (`Plan::camera_rect`) after the transform was verified to equal the plan's theatre rectangle transform. Cell
//! `lo`/`hi` boxes are NOT snapped. Cell size and connection distances are `* scale` (0 stays 0, -1 stays -1).

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value as Json, json};

use super::ctx::Ctx;
use super::diff::Touched;
use super::flat::Flat;
use super::util::Counter;
use super::util::{Bounds, Range};
use crate::esf::Node;

fn pair(p: [f32; 2]) -> Json {
    json!([p[0], p[1]])
}

pub(super) fn do_sea_grids(ctx: &mut Ctx, vals: &mut Vec<Node>, touched: &mut Touched) -> Result<Json> {
    let plan = ctx.plan;
    let mut ed = Flat::new(vals, touched);
    let count = ed.u32()?;
    let mut theatres = Vec::new();
    let (mut total_cells, mut total_conn) = (0u64, 0u64);
    let mut sent = Counter::new();
    for _ in 0..count {
        let name = ed.string()?;
        let ci = ctx.chart_index(&name)?;
        let chart = &plan.charts[ci];
        let (mn, mx) = (ed.xy()?, ed.xy()?);
        ctx.check_theatre_in_crop(&name, ci, mn, mx)?;
        let rect = *chart
            .theatre_rects
            .get(&name)
            .ok_or_else(|| anyhow!("sea grid {name}: chart {} has no theatre rectangle {name}", chart.key))?;
        let rect_match = [mn, mx]
            .iter()
            .zip(&rect)
            .all(|(m, r)| (0..2).all(|a| m[a] as f64 == r[a]));
        if !rect_match {
            bail!(
                "sea grid {name}: stored rectangle [{mn:?}, {mx:?}] differs from the plan's theatre rectangle {rect:?}; the camera \
                 snap is only defined for the unmodified theatre bounds"
            );
        }
        // unsnapped transform of both corners, verified against the plan's theatre-rectangle transform
        let unsnapped = [ctx.xy_f32(mn, ci)?, ctx.xy_f32(mx, ci)?];
        for (c, corner) in rect.iter().enumerate() {
            let want = plan.xy(*corner, chart);
            let want32 = [want[0] as f32, want[1] as f32];
            if unsnapped[c] != want32 {
                bail!("sea grid {name}: mapped corner {c} {:?} != plan theatre rectangle transform {want32:?}", unsnapped[c]);
            }
        }
        let snapped = plan.camera_rect(chart, &name)?;
        ed.replace_xy(2, snapped[0])?;
        ed.replace_xy(1, snapped[1])?;
        for corner in snapped {
            ctx.note_f("sea_grids", corner);
        }
        let factor = ed.f32()?;
        let new_factor = ctx.f32_scaled(factor)?;
        ed.replace_f32(1, new_factor)?;
        let (xs, ys) = (ed.u32()?, ed.u32()?);
        let (mut cb_old, mut cb_new) = (Bounds::new(), Bounds::new());
        let mut first_cell: Option<Json> = None;
        for yi in 0..ys {
            for xi in 0..xs {
                if (ed.u32()?, ed.u32()?) != (xi, yi) {
                    bail!("{name}: cell out of order at ({xi},{yi})");
                }
                let (lo, hi) = (ed.xy()?, ed.xy()?);
                let (nlo, nhi) = (ctx.xy_f32(lo, ci)?, ctx.xy_f32(hi, ci)?);
                ctx.note_f("sea_grids", nlo);
                ctx.note_f("sea_grids", nhi);
                ed.replace_xy(2, nlo)?;
                ed.replace_xy(1, nhi)?;
                for p in [lo, hi] {
                    cb_old.add(p[0] as f64, p[1] as f64);
                }
                for p in [nlo, nhi] {
                    cb_new.add(p[0] as f64, p[1] as f64);
                }
                if first_cell.is_none() {
                    first_cell = Some(json!({"lo": pair(lo), "hi": pair(hi), "new_lo": pair(nlo), "new_hi": pair(nhi)}));
                }
            }
        }
        let ncells = xs as u64 * ys as u64;
        let mut ids = Vec::new();
        let mut list_counts = [0i64; 4];
        for _ in 0..ncells {
            ids.push(ed.u32()?);
            for q in 0..3 {
                for _ in 0..ed.u32()? {
                    let key = ed.string()?;
                    match q {
                        0 | 1 => {
                            ctx.refs.sea_regions.insert(key);
                        }
                        _ => {
                            ctx.refs.sea_ports.insert(key);
                        }
                    }
                    list_counts[q] += 1;
                }
            }
            for _ in 0..ed.u32()? {
                ed.u32()?;
                list_counts[3] += 1;
            }
        }
        let nconn = ed.u32()?;
        let (mut cr_old, mut cr_new) = (Range::new(), Range::new());
        let (mut zeros, mut neg) = (0i64, 0i64);
        for _ in 0..nconn {
            ed.u32()?;
            ed.u32()?;
            let f = ed.f32()?;
            let out = ctx.distance(f, Some(-1.0))?;
            if f < 0.0 {
                neg += 1;
                continue;
            }
            if f == 0.0 {
                zeros += 1;
            }
            cr_old.add(f as f64);
            cr_new.add(out as f64);
            ed.replace_f32(1, out)?;
        }
        total_cells += ncells;
        total_conn += nconn as u64;
        sent.add("connection_minus_one_kept", neg);
        sent.add("connection_zero_adjacent_kept", zeros);
        let ids_json = json!({
            "first": ids.first(), "last": ids.last(), "count": ids.len(), "unchanged": true,
        });
        theatres.push(json!({
            "name": name,
            "matches_plan_theatre_rect": rect_match,
            "min_max": {
                "old": [pair(mn), pair(mx)],
                "new": [pair(snapped[0]), pair(snapped[1])],
            },
            "camera_snap": {
                "quantum": 1.0 / 4096.0,
                "unsnapped_new": [pair(unsnapped[0]), pair(unsnapped[1])],
                "stored_new": [pair(snapped[0]), pair(snapped[1])],
                "unsnapped_equals_plan_theatre_rect_transform": true,
                "max_coordinate_change": (0..2)
                    .flat_map(|c| (0..2).map(move |a| (c, a)))
                    .map(|(c, a)| (snapped[c][a] as f64 - unsnapped[c][a] as f64).abs())
                    .fold(0.0f64, f64::max),
                "cell_boxes_snapped": false,
            },
            "cell_size": {"old": factor, "new": new_factor},
            "cols": xs, "rows": ys, "cells": ncells,
            "cell_bbox_bounds": {"old": cb_old.dump(), "new": cb_new.dump()},
            "first_cell": first_cell,
            "area_ids": ids_json,
            "list_entries": {
                "lands": list_counts[0], "seas": list_counts[1], "ports": list_counts[2], "numbers": list_counts[3],
                "unchanged": true,
            },
            "connections": {
                "count": nconn, "minus_one_kept": neg, "zero_kept": zeros,
                "positive_old": cr_old.dump(), "positive_new": cr_new.dump(),
            },
            "chart": chart.key,
        }));
    }
    ed.finish()?;
    let mut out = Map::new();
    out.insert("theatres".into(), Json::Array(theatres));
    out.insert("total_cells".into(), json!(total_cells));
    out.insert("total_connections".into(), json!(total_conn));
    out.insert("sentinels".into(), sent.to_json());
    out.insert("modified_values".into(), json!(touched.paths.len()));
    Ok(Json::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::Plan;
    use crate::esf;

    /// One theatre (`2`) with a 1x1 grid and one connection list entry of each kind.
    fn stream(min: [f32; 2], max: [f32; 2]) -> Vec<Node> {
        vec![
            esf::u32v(1),
            esf::s16("2"),
            esf::xyv(min[0], min[1]),
            esf::xyv(max[0], max[1]),
            esf::f32v(10.0),
            esf::u32v(1),
            esf::u32v(1),
            esf::u32v(0),
            esf::u32v(0),
            esf::xyv(-100.0, 200.0),
            esf::xyv(-90.0, 210.0),
            esf::u32v(5),
            esf::u32v(1),
            esf::s16("france"),
            esf::u32v(0),
            esf::u32v(1),
            esf::s16("port:france:brest"),
            esf::u32v(0),
            esf::u32v(3),
            esf::u32v(0),
            esf::u32v(0),
            esf::f32v(-1.0),
            esf::u32v(0),
            esf::u32v(0),
            esf::f32v(0.0),
            esf::u32v(0),
            esf::u32v(0),
            esf::f32v(4.0),
        ]
    }

    #[test]
    fn theatre_corners_are_camera_snapped_and_everything_else_scales_unsnapped() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let mut vals = stream([-180.0, 140.0], [440.0, 520.0]);
        let mut touched = Touched::default();
        let detail = do_sea_grids(&mut ctx, &mut vals, &mut touched).unwrap();
        let europe = plan.chart("2").unwrap();
        let snapped = plan.camera_rect(europe, "2").unwrap();
        assert_eq!(vals[2], esf::xyv(snapped[0][0], snapped[0][1]));
        assert_eq!(vals[3], esf::xyv(snapped[1][0], snapped[1][1]));
        // the unsnapped corner is not a multiple of 1/4096 for this offset, so snapping must have changed it
        let unsnapped = plan.xy([-180.0, 140.0], europe);
        assert_ne!(unsnapped[0] as f32, snapped[0][0]);
        assert_eq!(snapped[0][0] * 4096.0, (snapped[0][0] * 4096.0).round());
        // cell box is the plain f32 transform, not snapped
        let lo = plan.xy([-100.0, 200.0], europe);
        assert_eq!(vals[9], esf::xyv(lo[0] as f32, lo[1] as f32));
        assert_eq!(vals[4], esf::f32v((10.0f64 * plan.scale) as f32));
        // connections: -1 kept, 0 stays 0, 4.0 scaled
        assert_eq!(vals[21], esf::f32v(-1.0));
        assert_eq!(vals[24], esf::f32v(0.0));
        assert_eq!(vals[27], esf::f32v((4.0f64 * plan.scale) as f32));
        assert_eq!(detail["sentinels"]["connection_minus_one_kept"], json!(1));
        assert_eq!(detail["theatres"][0]["camera_snap"]["cell_boxes_snapped"], json!(false));
        assert!(ctx.refs.sea_regions.contains("france") && ctx.refs.sea_ports.contains("port:france:brest"));
    }

    #[test]
    fn modified_theatre_rectangle_is_rejected() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let mut vals = stream([-180.0, 140.0], [441.0, 520.0]);
        assert!(do_sea_grids(&mut ctx, &mut vals, &mut Touched::default()).is_err());
    }

    #[test]
    fn unknown_theatre_and_out_of_order_cells_are_errors() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let mut vals = stream([-180.0, 140.0], [440.0, 520.0]);
        vals[1] = esf::s16("999");
        assert!(do_sea_grids(&mut ctx, &mut vals, &mut Touched::default()).is_err());
        let mut ctx = Ctx::new(&plan).unwrap();
        let mut vals = stream([-180.0, 140.0], [440.0, 520.0]);
        vals[7] = esf::u32v(1);
        assert!(do_sea_grids(&mut ctx, &mut vals, &mut Touched::default()).is_err());
    }
}
