//! `regions-index-contract.json`: the mapping from source (region, area) to output areas that the campaign
//! migration consumes, plus one certified interior point per output area.

use super::assemble::{Kind, OutData};
use super::clip::{Ctx, GAP};
use super::geom::{AreaPoly, P, ring_area, widen};
use crate::regionmap::geometry::polylabel;
use anyhow::Result;
use serde_json::{Map, Value, json};

pub struct Interior {
    pub point: [f32; 2],
    pub in_window: bool,
    pub radius: f64,
}

fn perimeter(ring: &[P]) -> f64 {
    let n = ring.len();
    (0..n).map(|k| (ring[(k + 1) % n][0] - ring[k][0]).hypot(ring[(k + 1) % n][1] - ring[k][1])).sum()
}

type Piece = (Vec<P>, Vec<Vec<P>>);

fn piece_area(p: &Piece) -> f64 {
    ring_area(&p.0) - p.1.iter().map(|h| ring_area(h)).sum::<f64>()
}

/// Point of a polygon farthest from its outline (pole of inaccessibility), rounded to f32.
fn pole(p: &Piece) -> ([f32; 2], f64) {
    let mut rings: Vec<Vec<P>> = vec![p.0.clone()];
    rings.extend(p.1.iter().cloned());
    let length: f64 = rings.iter().map(|r| perimeter(r)).sum();
    let (c, d) = polylabel(&rings, (length * 1e-6).max(1e-3));
    ([c[0] as f32, c[1] as f32], d)
}

fn open_ring(c: &geo::LineString<f64>) -> Vec<P> {
    let mut pts: Vec<P> = c.0.iter().map(|k| [k.x, k.y]).collect();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    pts
}

/// The polygon intersected with the axis-aligned window `[x0, y0, x1, y1]`.
fn clip_to_window(poly: &AreaPoly, w: [f64; 4]) -> Vec<Piece> {
    let b = poly.bbox;
    if b[0][0] >= w[0] && b[0][1] >= w[1] && b[1][0] <= w[2] && b[1][1] <= w[3] {
        return vec![(poly.outer.clone(), poly.holes.clone())];
    }
    if b[1][0] <= w[0] || b[0][0] >= w[2] || b[1][1] <= w[1] || b[0][1] >= w[3] {
        return Vec::new();
    }
    use geo::BooleanOps;
    let ls = |r: &[P]| geo::LineString::from(r.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>());
    let g = geo::Polygon::new(ls(&poly.outer), poly.holes.iter().map(|h| ls(h)).collect());
    let win = geo::Rect::new(geo::Coord { x: w[0], y: w[1] }, geo::Coord { x: w[2], y: w[3] }).to_polygon();
    let mp = g.intersection(&win);
    mp.0
        .iter()
        .map(|p| (open_ring(p.exterior()), p.interiors().iter().map(open_ring).collect::<Vec<_>>()))
        .collect()
}

/// Point strictly inside `poly`; inside the certified `window` (output space) when the polygon reaches it, at the
/// pole of inaccessibility of that part.
pub fn interior_point(poly: &AreaPoly, window: [f64; 4]) -> Interior {
    let pieces = clip_to_window(poly, window);
    let total: f64 = pieces.iter().map(piece_area).sum();
    if !pieces.is_empty() && total > 1e-9 {
        let best = pieces.iter().max_by(|a, b| piece_area(a).partial_cmp(&piece_area(b)).unwrap()).unwrap();
        let (p, r) = pole(best);
        let q = widen(p);
        if poly.contains(q) && poly.boundary_distance(q) > 0.0 {
            return Interior { point: p, in_window: true, radius: r };
        }
    }
    let (p, r) = pole(&(poly.outer.clone(), poly.holes.clone()));
    Interior { point: p, in_window: false, radius: r }
}

pub fn build_contract(ctx: &Ctx, ods: &[(Kind, &OutData)], window: [f64; 4]) -> Result<Value> {
    let mut out = Map::new();
    out.insert("format".into(), json!(1));
    out.insert("window".into(), json!(window));
    for (kind, od) in ods {
        let src = od.src;
        let mut regs: Vec<Value> = Vec::new();
        for (ri, areas) in od.regions.iter().enumerate() {
            let n_src = src.counts[ri];
            let mut amap: Vec<Value> = Vec::new();
            let mut frags: Vec<Value> = Vec::new();
            for ai in 0..n_src {
                match od.area_map.get(&(ri, ai)) {
                    Some(loc) if !loc.is_empty() => {
                        amap.push(json!(loc[0]));
                        frags.push(json!(loc));
                    }
                    _ => {
                        amap.push(Value::Null);
                        frags.push(json!([]));
                    }
                }
            }
            let mut ip = Map::new();
            for oa in areas {
                let poly = od.poly_of(oa)?;
                let i = interior_point(&poly, window);
                ip.insert(
                    oa.local.to_string(),
                    json!({"point": [i.point[0], i.point[1]], "in_window": i.in_window, "inscribed_radius": i.radius}),
                );
            }
            regs.push(json!({
                "source_index": ri,
                "key": src.region_key(ri)?,
                "type": src.region_kind(ri)?,
                "output_index": ri,
                "source_areas": n_src,
                "output_areas": areas.len(),
                "area_map": amap,
                "area_fragments": frags,
                "area_source": areas.iter().map(|a| a.src.1).collect::<Vec<_>>(),
                "area_chart": areas.iter().map(|a| if a.ci == GAP { Value::Null } else { json!(ctx.plan.charts[a.ci].key) }).collect::<Vec<_>>(),
                "area_is_world_gap": areas.iter().map(|a| a.is_gap).collect::<Vec<_>>(),
                "geometry_kind": areas.iter().map(|a| if a.is_gap { "atlas_void" } else { "affine_crop" }).collect::<Vec<_>>(),
                "interior_point": Value::Object(ip),
            }));
        }
        let absorbed: Vec<Value> = od
            .absorbed
            .iter()
            .map(|a| {
                json!({
                    "owner": [a.owner.0, a.owner.1],
                    "chart": a.chart,
                    "output_area": a.output_area,
                    "into": [a.into.0, a.into.1],
                    "shared_border": a.shared_border,
                })
            })
            .collect();
        out.insert(
            kind.label().to_string(),
            json!({
                "source_region_count": src.n_regions,
                "output_region_count": od.regions.len(),
                "region_order_preserved": true,
                "backdrop": {
                    "source_area": [od.backdrop.0, od.backdrop.1],
                    "region": src.region_key(od.backdrop.0)?,
                    "why": od.backdrop_why,
                    "output_local": od.gap_local,
                },
                "dropped_areas": od.dropped,
                "absorbed_slivers": absorbed,
                "regions": regs,
            }),
        );
    }
    Ok(Value::Object(out))
}
