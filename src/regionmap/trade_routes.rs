//! `trade_routes.esf`: the campaign trade graph. Root record children, in order:
//! `SPLINES[]` items `[xy array, bool]` (the bool is true for land road splines, false for sea lanes);
//! `PORTS[]` items `[s16 key]`; `SETTLEMENTS[]` items `[s16 region key]`;
//! `ROUTES[]` items `[u32 from, u32 to, u32[] spline ids, f32 length]`; `TRADE_NODE_ROUTES[]` (sea trade nodes to ports).
//!
//! Node index space = ports (`0..PORTS`) then settlements, so a settlement's node is `PORTS + position`. Verified on
//! vanilla `nap_europe`:
//! * a spline id is `spline_index * 2 + reversed` (reversed = walk the points backwards); the concatenated polyline of
//!   a route starts at `from`'s position and ends at `to`'s;
//! * every connected pair appears once, `from < to`; the route list is grouped by `from` but otherwise unordered;
//! * `length` is the summed polyline length in world units; a route with an empty spline list (37 settlement pairs
//!   and one port-settlement pair) is a direct link whose length is the straight distance between the two nodes.
//!
//! Adding a settlement appends its node (index = old count, hence larger than every neighbour) and one route
//! `(neighbour, new)` per neighbour; nothing existing is renumbered.
//!
//! Spline access ([`TradeSplines`], used by the sea-grid `numbers` rule) and the route guarantee
//! ([`ensure_settlement_routes`]: every settlement node has at least one route).

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::esf::{self, Node};
use crate::regionmap::geometry::{P, segment_overlaps_rect};
use crate::regionmap::regions::{RegionData, settlement_center};

/// How a new route is represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    /// Vanilla's convention for neighbours without a road: empty spline list, length = straight distance.
    Direct,
    /// One new straight two-point land spline per route, referenced forwards (`index * 2`).
    Spline,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Node index of the new settlement.
    pub node: u32,
    pub splines_added: usize,
    pub routes_added: usize,
    /// Settlement count after the edit.
    pub settlements: usize,
}

fn count_of(root: &Node, name: &str) -> Result<usize> {
    Ok(root.get(name)?.items()?.len())
}

fn settlement_keys(root: &Node) -> Result<Vec<&str>> {
    root.get("SETTLEMENTS")?
        .items()?
        .iter()
        .map(|item| item.first().ok_or_else(|| anyhow!("empty SETTLEMENTS item"))?.as_str())
        .collect()
}

/// Node index of the settlement of region `key`.
pub fn settlement_node_index(root: &Node, key: &str) -> Result<u32> {
    let ports = count_of(root, "PORTS")?;
    let at = settlement_keys(root)?
        .iter()
        .position(|k| *k == key)
        .ok_or_else(|| anyhow!("trade_routes.esf has no settlement {key}"))?;
    Ok((ports + at) as u32)
}

/// Add settlement `key` at `pos`, linked to each `(neighbour key, neighbour settlement position)` using vanilla's
/// direct-link convention.
pub fn add_settlement(root: &mut Node, key: &str, pos: [f32; 2], neighbours: &[(&str, [f32; 2])]) -> Result<Report> {
    add_settlement_with(root, key, pos, neighbours, Link::Direct)
}

pub fn add_settlement_with(
    root: &mut Node,
    key: &str,
    pos: [f32; 2],
    neighbours: &[(&str, [f32; 2])],
    link: Link,
) -> Result<Report> {
    ensure!(
        settlement_node_index(root, key).is_err(),
        "trade_routes.esf already has settlement {key}"
    );
    let mut from = Vec::with_capacity(neighbours.len());
    for (nb, _) in neighbours {
        let n = settlement_node_index(root, nb)?;
        ensure!(!from.contains(&n), "neighbour {nb} listed twice");
        from.push(n);
    }
    let node = (count_of(root, "PORTS")? + count_of(root, "SETTLEMENTS")?) as u32;
    let mut splines = count_of(root, "SPLINES")?;
    ensure!(
        (splines + neighbours.len()) * 2 <= u32::MAX as usize,
        "spline id overflow"
    );

    let mut new_splines = Vec::new();
    let mut new_routes = Vec::with_capacity(neighbours.len());
    for ((_, npos), &nn) in neighbours.iter().zip(&from) {
        // The route runs neighbour -> new settlement (from < to), so the polyline starts at the neighbour.
        let length = (((pos[0] - npos[0]) as f64).powi(2) + ((pos[1] - npos[1]) as f64).powi(2)).sqrt() as f32;
        let ids: Vec<u32> = match link {
            Link::Direct => Vec::new(),
            Link::Spline => {
                new_splines.push(vec![esf::arr_xy(&[*npos, pos]), esf::b(true)]);
                splines += 1;
                vec![((splines - 1) * 2) as u32]
            }
        };
        new_routes.push(vec![esf::u32v(nn), esf::u32v(node), esf::arr_u32(&ids), esf::f32v(length)]);
    }

    let report = Report {
        node,
        splines_added: new_splines.len(),
        routes_added: new_routes.len(),
        settlements: count_of(root, "SETTLEMENTS")? + 1,
    };
    root.get_mut("SPLINES")?.items_mut()?.extend(new_splines);
    root.get_mut("SETTLEMENTS")?.items_mut()?.push(vec![esf::s16(key)]);
    root.get_mut("ROUTES")?.items_mut()?.extend(new_routes);
    Ok(report)
}

/// One `SPLINES[]` item: a chain of cubic Beziers (`3k + 1` control points) and the land-road flag.
#[derive(Clone, Debug, PartialEq)]
pub struct Spline {
    pub points: Vec<[f32; 2]>,
    /// true for land road splines, false for sea lanes
    pub land: bool,
}

/// Segments per cubic Bezier when a spline is flattened for overlap tests (the sea-grid reference rule).
pub const BEZIER_STEPS: usize = 64;

/// Read-only view of `SPLINES[]` with a bounding box per spline (control points bound the curve).
#[derive(Clone, Debug, Default)]
pub struct TradeSplines {
    splines: Vec<Spline>,
    bboxes: Vec<(P, P)>,
}

impl TradeSplines {
    pub fn parse(root: &Node) -> Result<Self> {
        let mut splines = Vec::new();
        for (i, item) in root.get("SPLINES")?.items()?.iter().enumerate() {
            ensure!(item.len() >= 2, "SPLINES[{i}] has {} children", item.len());
            let points = item[0].xy_array().with_context(|| format!("SPLINES[{i}] points"))?;
            ensure!(
                points.len() >= 4 && (points.len() - 1) % 3 == 0,
                "SPLINES[{i}] has {} control points, not a chain of cubic Beziers",
                points.len()
            );
            splines.push(Spline { points, land: item[1].as_bool().with_context(|| format!("SPLINES[{i}] flag"))? });
        }
        let bboxes = splines
            .iter()
            .map(|s| {
                let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
                for p in &s.points {
                    for k in 0..2 {
                        lo[k] = lo[k].min(p[k] as f64);
                        hi[k] = hi[k].max(p[k] as f64);
                    }
                }
                (lo, hi)
            })
            .collect();
        Ok(Self { splines, bboxes })
    }

    pub fn len(&self) -> usize {
        self.splines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.splines.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&Spline> {
        self.splines.get(index)
    }

    /// Control points of spline `index`.
    pub fn control_points(&self, index: usize) -> &[[f32; 2]] {
        &self.splines[index].points
    }

    /// Land-road flag of spline `index`.
    pub fn flag(&self, index: usize) -> bool {
        self.splines[index].land
    }

    /// Spline `index` flattened to a polyline ([`BEZIER_STEPS`] segments per Bezier).
    pub fn flatten(&self, index: usize) -> Vec<P> {
        let pts = &self.splines[index].points;
        let mut out: Vec<P> = Vec::with_capacity((pts.len() - 1) / 3 * BEZIER_STEPS + 1);
        for s in (0..pts.len() - 1).step_by(3) {
            let q: Vec<P> = pts[s..s + 4].iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
            for i in 0..=BEZIER_STEPS {
                if s > 0 && i == 0 {
                    continue;
                }
                let t = i as f64 / BEZIER_STEPS as f64;
                let u = 1.0 - t;
                let mut p = [0.0; 2];
                for k in 0..2 {
                    p[k] = u.powi(3) * q[0][k] + 3.0 * u * u * t * q[1][k] + 3.0 * u * t * t * q[2][k] + t.powi(3) * q[3][k];
                }
                out.push(p);
            }
        }
        out
    }

    /// Indices (ascending) of the splines whose curve has a positive-length overlap with the interior of the
    /// rectangle `lo`..`hi` (a curve point strictly inside counts too).
    pub fn splines_in(&self, lo: P, hi: P) -> Vec<u32> {
        let mut out = Vec::new();
        for (i, (bl, bh)) in self.bboxes.iter().enumerate() {
            if !(bh[0] > lo[0] && bl[0] < hi[0] && bh[1] > lo[1] && bl[1] < hi[1]) {
                continue;
            }
            let c = self.flatten(i);
            let inside = c.iter().any(|p| p[0] > lo[0] && p[0] < hi[0] && p[1] > lo[1] && p[1] < hi[1]);
            if inside || c.windows(2).any(|w| segment_overlaps_rect(w[0], w[1], lo, hi)) {
                out.push(i as u32);
            }
        }
        out
    }
}

/// `(port count, settlement keys, route count per node)` of `trade_routes.esf`; node index space = ports then
/// settlements.
fn node_degrees(root: &Node) -> Result<(usize, Vec<String>, Vec<usize>)> {
    let ports = count_of(root, "PORTS")?;
    let keys: Vec<String> = settlement_keys(root)?.into_iter().map(str::to_string).collect();
    let mut degree = vec![0usize; ports + keys.len()];
    for (i, r) in root.get("ROUTES")?.items()?.iter().enumerate() {
        ensure!(r.len() >= 4, "ROUTES[{i}] has {} children", r.len());
        for end in &r[..2] {
            let n = end.as_u32()? as usize;
            match degree.get_mut(n) {
                Some(d) => *d += 1,
                None => bail!("ROUTES[{i}] refers to node {n}, only {} exist", degree.len()),
            }
        }
    }
    Ok((ports, keys, degree))
}

/// Keys of the trade nodes without any route: port keys (`PORTS[]`) first, then settlement region keys. Empty when
/// the trade graph has no routeless node.
pub fn routeless_nodes(root: &Node) -> Result<Vec<String>> {
    let (ports, keys, degree) = node_degrees(root)?;
    let mut out = Vec::new();
    for (i, item) in root.get("PORTS")?.items()?.iter().enumerate() {
        if degree[i] == 0 {
            out.push(item.first().and_then(|n| n.as_str().ok()).map(str::to_string).unwrap_or_else(|| format!("port #{i}")));
        }
    }
    out.extend(keys.into_iter().enumerate().filter(|(j, _)| degree[ports + j] == 0).map(|(_, k)| k));
    Ok(out)
}

/// Give every settlement node of `trade_routes.esf` at least one route. A settlement without any route (an island
/// whose settlement has no port node, e.g. a new region) gets one route to the NEAREST settlement that had a route
/// before this call, in the form land-adjacency routes use: empty spline list, length = straight settlement distance,
/// endpoints ascending. Settlement positions come from `rd` (regions without a settlement are skipped).
/// Returns `(isolated settlement key, partner key, distance)` per route added.
pub fn ensure_settlement_routes(root: &mut Node, rd: &RegionData) -> Result<Vec<(String, String, f32)>> {
    let (ports, keys, degree) = node_degrees(root)?;
    let pos: Vec<Option<[f32; 2]>> = keys
        .iter()
        .map(|k| rd.index(k).ok().and_then(|ri| settlement_center(&rd.regions[ri]).ok()))
        .collect();
    let had_route: Vec<bool> = (0..keys.len()).map(|j| degree[ports + j] > 0).collect();
    let mut added = Vec::new();
    let mut routes = Vec::new();
    for (i, name) in keys.iter().enumerate() {
        let node = ports + i;
        let Some(p) = pos[i] else { continue };
        if degree[node] > 0 {
            continue;
        }
        let mut best: Option<(f64, usize)> = None;
        for j in 0..keys.len() {
            let (true, Some(q)) = (j != i && had_route[j], pos[j]) else { continue };
            let d = ((q[0] as f64) - (p[0] as f64)).hypot((q[1] as f64) - (p[1] as f64));
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, j));
            }
        }
        let (d, j) = best.ok_or_else(|| anyhow!("no routed settlement to connect {name} to"))?;
        let other = ports + j;
        let (a, b) = (other.min(node), other.max(node));
        routes.push(vec![esf::u32v(a as u32), esf::u32v(b as u32), esf::arr_u32(&[]), esf::f32v(d as f32)]);
        added.push((name.clone(), keys[j].clone(), d as f32));
    }
    root.get_mut("ROUTES")?.items_mut()?.extend(routes);
    Ok(added)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const EXP5: &str = "F:/NTW2/work/exp5";

    fn read(path: &str) -> Option<esf::Esf> {
        if !Path::new(path).exists() {
            eprintln!("skipped: {path} not available");
            return None;
        }
        Some(esf::read(&std::fs::read(path).unwrap()).unwrap())
    }

    fn spline(points: &[[f32; 2]]) -> Spline {
        Spline { points: points.to_vec(), land: false }
    }

    fn splines_of(list: Vec<Spline>) -> TradeSplines {
        let mut s = TradeSplines { splines: list, bboxes: Vec::new() };
        s.bboxes = s
            .splines
            .iter()
            .map(|sp| {
                let xs = sp.points.iter().map(|p| p[0] as f64);
                let ys = sp.points.iter().map(|p| p[1] as f64);
                (
                    [xs.clone().fold(f64::MAX, f64::min), ys.clone().fold(f64::MAX, f64::min)],
                    [xs.fold(f64::MIN, f64::max), ys.fold(f64::MIN, f64::max)],
                )
            })
            .collect();
        s
    }

    #[test]
    fn spline_rect_overlap_rule() {
        // a straight Bezier along y = 5 from x = -5 to x = 5, one along the box edge y = 10, one far away
        let line = |y: f32, x0: f32, x1: f32| {
            let d = (x1 - x0) / 3.0;
            spline(&[[x0, y], [x0 + d, y], [x0 + 2.0 * d, y], [x1, y]])
        };
        let s = splines_of(vec![line(5.0, -5.0, 5.0), line(10.0, 0.0, 9.0), line(50.0, 0.0, 9.0), line(5.0, -9.0, -1.0)]);
        assert_eq!(s.splines_in([0.0, 0.0], [10.0, 10.0]), vec![0]);
        assert_eq!(s.flatten(0).len(), BEZIER_STEPS + 1);
        assert_eq!(s.control_points(1).len(), 4);
        assert!(!s.flag(1));
    }

    #[test]
    fn routeless_sumatra_gets_one_route() {
        let (Some(mut tr), Some(rg)) = (
            read(&format!("{EXP5}/driver_work/map/trade_routes.esf")),
            read(&format!("{EXP5}/driver_work/map/regions.esf")),
        ) else {
            return;
        };
        let rd = RegionData::parse(&rg.root).unwrap();
        let routes_before = tr.root.get("ROUTES").unwrap().items().unwrap().len();
        assert_eq!(routeless_nodes(&tr.root).unwrap(), vec!["sumatra".to_string()]);
        let added = ensure_settlement_routes(&mut tr.root, &rd).unwrap();
        assert_eq!(added.len(), 1, "{added:?}");
        assert_eq!((added[0].0.as_str(), added[0].1.as_str()), ("sumatra", "indochina"));
        assert!((added[0].2 - 128.04).abs() < 0.005, "{}", added[0].2);
        assert_eq!(tr.root.get("ROUTES").unwrap().items().unwrap().len(), routes_before + 1);
        // idempotent: nothing left to fix
        assert!(ensure_settlement_routes(&mut tr.root, &rd).unwrap().is_empty());
        assert!(routeless_nodes(&tr.root).unwrap().is_empty());
        // byte-identical to the reference script's output (exp5/out)
        if let Some(reference) = read(&format!("{EXP5}/out/data/campaign_maps/nap_world/trade_routes.esf")) {
            let mut fresh = read(&format!("{EXP5}/driver_work/map/trade_routes.esf")).unwrap();
            ensure_settlement_routes(&mut fresh.root, &rd).unwrap();
            assert_eq!(esf::write(&fresh).unwrap(), esf::write(&reference).unwrap());
        }
    }

    #[test]
    fn pristine_trade_routes_have_no_isolated_settlement() {
        let (Some(mut tr), Some(rg)) = (
            read("F:/NTW2/work/exp4/backup/data/campaign_maps/nap_world/trade_routes.esf"),
            read("F:/NTW2/work/exp1/backup/campaign_maps/nap_world/regions.esf"),
        ) else {
            return;
        };
        let rd = RegionData::parse(&rg.root).unwrap();
        assert!(ensure_settlement_routes(&mut tr.root, &rd).unwrap().is_empty());
        assert!(routeless_nodes(&tr.root).unwrap().is_empty());
        let s = TradeSplines::parse(&tr.root).unwrap();
        assert_eq!(s.len(), tr.root.get("SPLINES").unwrap().items().unwrap().len());
    }
}
