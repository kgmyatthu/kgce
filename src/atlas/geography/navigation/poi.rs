//! `poi.esf`: one flat `CAI_POI_ROOT` stream; Q20 x/y -> per-chart quantized Q20, every distance * scale (the "nearest"
//! distance keeps its `-1` sentinel); counts, strings, ids, types, flags, link lists untouched.
//!
//! Per POI: `u32 index, i32 type, bool flag, i32 x, i32 y, s16 region, u32 area id, s16 region, u32 area id, f32
//! settlement distance, u32 n, n * (s16 region, f32), f32 nearest (-1 = none), u32 m, m * (s16 region, f32), u32 k,
//! k * u32 links, u32, bool`. The area id is the region area's adjoining-passable-area id and is NOT changed here.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde_json::{Map, Value as Json, json};

use super::ctx::Ctx;
use super::diff::Touched;
use super::flat::Flat;
use super::util::{Bounds, Counter, Range, SENT_VERTEX, UNIT};
use crate::esf::Node;

pub(super) fn do_poi(ctx: &mut Ctx, vals: &mut Vec<Node>, touched: &mut Touched) -> Result<Json> {
    let mut ed = Flat::new(vals, touched);
    let n = ed.u32()?;
    let (mut ob, mut nb) = (Bounds::new(), Bounds::new());
    let kinds = ["settlement", "neighbour", "nearest", "border"];
    let mut dist: Vec<Range> = kinds.iter().map(|_| Range::new()).collect();
    let mut dist_new: Vec<Range> = kinds.iter().map(|_| Range::new()).collect();
    let mut sentinels = Counter::new();
    let mut types: BTreeMap<i32, i64> = BTreeMap::new();
    let mut per_chart: BTreeMap<String, i64> = BTreeMap::new();
    let unit = UNIT as f64;

    // Scales the f32 just consumed (1 back) unless it is the allowed negative sentinel.
    fn scale_dist(
        ctx: &Ctx,
        ed: &mut Flat,
        kind: usize,
        names: &[&str],
        v: f32,
        allow: Option<f32>,
        dist: &mut [Range],
        dist_new: &mut [Range],
        sentinels: &mut Counter,
    ) -> Result<()> {
        let out = ctx.distance(v, allow)?;
        if v < 0.0 {
            sentinels.add(names[kind], 1);
            return Ok(());
        }
        dist[kind].add(v as f64);
        dist_new[kind].add(out as f64);
        ed.replace_f32(1, out)
    }

    for _ in 0..n {
        ed.u32()?; // index
        *types.entry(ed.i32()?).or_default() += 1;
        ed.bool()?;
        let (x, y) = (ed.i32()?, ed.i32()?);
        if x == SENT_VERTEX || y == SENT_VERTEX {
            bail!("sentinel position in poi");
        }
        let chart = ctx.poi_chart(x, y)?;
        let (nx, ny) = ctx.point_q(x, y, chart)?;
        ctx.note("poi", nx as i64, ny as i64);
        *per_chart.entry(ctx.chart_key(chart).to_string()).or_default() += 1;
        ed.replace_i32(2, nx)?;
        ed.replace_i32(1, ny)?;
        ob.add(x as f64 / unit, y as f64 / unit);
        nb.add(nx as f64 / unit, ny as f64 / unit);
        let (r1, a1) = (ed.string()?, ed.u32()?);
        let (r2, a2) = (ed.string()?, ed.u32()?);
        if (&r1, a1) != (&r2, a2) {
            bail!("poi region pair mismatch");
        }
        if !r1.is_empty() {
            ctx.refs.poi_regions.insert(r1.clone());
            ctx.refs.poi_areas.insert((r1, a1));
        }
        let v = ed.f32()?;
        scale_dist(ctx, &mut ed, 0, &kinds, v, None, &mut dist, &mut dist_new, &mut sentinels)?;
        for _ in 0..ed.u32()? {
            let name = ed.string()?;
            ctx.refs.poi_regions.insert(name);
            let v = ed.f32()?;
            scale_dist(ctx, &mut ed, 1, &kinds, v, None, &mut dist, &mut dist_new, &mut sentinels)?;
        }
        let v = ed.f32()?;
        scale_dist(ctx, &mut ed, 2, &kinds, v, Some(-1.0), &mut dist, &mut dist_new, &mut sentinels)?;
        for _ in 0..ed.u32()? {
            let name = ed.string()?;
            ctx.refs.poi_regions.insert(name);
            let v = ed.f32()?;
            scale_dist(ctx, &mut ed, 3, &kinds, v, None, &mut dist, &mut dist_new, &mut sentinels)?;
        }
        for _ in 0..ed.u32()? {
            ed.u32()?; // link list
        }
        ed.u32()?;
        ed.bool()?;
    }
    ed.finish()?;
    let modified = touched.paths.len();
    let mut detail = Map::new();
    detail.insert("pois".into(), json!(n));
    detail.insert("types".into(), Json::Object(types.iter().map(|(k, v)| (k.to_string(), json!(v))).collect()));
    detail.insert("position_bounds_world".into(), json!({"old": ob.dump(), "new": nb.dump()}));
    let dump = |ranges: &[Range]| -> Json {
        Json::Object(kinds.iter().zip(ranges).map(|(k, r)| (k.to_string(), r.dump())).collect())
    };
    detail.insert("distances_old".into(), dump(&dist));
    detail.insert("distances_new".into(), dump(&dist_new));
    detail.insert("negative_sentinels_kept".into(), sentinels.to_json());
    detail.insert("modified_values".into(), json!(modified));
    detail.insert("pois_per_chart".into(), json!(per_chart));
    Ok(Json::Object(detail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::Plan;
    use crate::esf;

    fn poi(x: i32, y: i32, settlement: f32, nearest: f32) -> Vec<Node> {
        vec![
            esf::u32v(0),
            esf::i32v(4),
            esf::b(false),
            esf::i32v(x),
            esf::i32v(y),
            esf::s16("france"),
            esf::u32v(13),
            esf::s16("france"),
            esf::u32v(13),
            esf::f32v(settlement),
            esf::u32v(1),
            esf::s16("spain"),
            esf::f32v(3.0),
            esf::f32v(nearest),
            esf::u32v(0),
            esf::u32v(0),
            esf::u32v(9),
            esf::b(true),
        ]
    }

    #[test]
    fn negative_sentinel_stays_other_distances_scale_and_untouched_values_are_not_declared() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let ci = ctx.chart_index("2").unwrap();
        // identity-like map for the two coordinates
        let (x, y) = (10 * 1_048_576, 300 * 1_048_576);
        ctx.q[ci] = Some([[(x as i64, 7i64)].into_iter().collect(), [(y as i64, 9i64)].into_iter().collect()]);
        let mut vals = vec![esf::u32v(1)];
        vals.extend(poi(x, y, 2.0, -1.0));
        let mut touched = Touched::default();
        let detail = do_poi(&mut ctx, &mut vals, &mut touched).unwrap();
        assert_eq!(vals[4], esf::i32v(7));
        assert_eq!(vals[5], esf::i32v(9));
        let scale = plan.scale;
        assert_eq!(vals[10], esf::f32v((2.0f64 * scale) as f32));
        assert_eq!(vals[13], esf::f32v((3.0f64 * scale) as f32));
        assert_eq!(vals[14], esf::f32v(-1.0), "the nearest sentinel is kept");
        // x, y, settlement distance, neighbour distance
        assert_eq!(touched.paths.len(), 4);
        assert_eq!(detail["negative_sentinels_kept"]["nearest"], json!(1));
        assert!(ctx.refs.poi_regions.contains("spain") && ctx.refs.poi_areas.contains(&("france".to_string(), 13)));
    }

    #[test]
    fn unexpected_negative_distance_and_pair_mismatch_are_errors() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let ci = ctx.chart_index("2").unwrap();
        let (x, y) = (10 * 1_048_576, 300 * 1_048_576);
        ctx.q[ci] = Some([[(x as i64, 7i64)].into_iter().collect(), [(y as i64, 9i64)].into_iter().collect()]);
        let mut vals = vec![esf::u32v(1)];
        vals.extend(poi(x, y, -2.0, 1.0));
        assert!(do_poi(&mut ctx, &mut vals, &mut Touched::default()).is_err());
        let mut vals = vec![esf::u32v(1)];
        let mut p = poi(x, y, 2.0, 1.0);
        p[8] = esf::u32v(14);
        vals.extend(p);
        assert!(do_poi(&mut ctx, &mut vals, &mut Touched::default()).is_err());
    }

    #[test]
    fn point_outside_every_chart_is_an_error() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let mut vals = vec![esf::u32v(1)];
        vals.extend(poi(1000 * 1_048_576, 600 * 1_048_576, 2.0, 1.0));
        assert!(do_poi(&mut ctx, &mut vals, &mut Touched::default()).is_err());
    }
}
