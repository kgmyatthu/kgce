//! `trade_routes.esf` (and the mirror of it in the start position): the road network is a graph of nodes (ports, then
//! settlements) joined by routes made of Bezier splines. A settlement's trade node sits near its centre, not on it
//! (measured on the ETW map: 0 .. 1.44 u away, median 0.18 u), so a moved capital carries the end of every road that
//! reaches it by the same offset – the end point and its handle translate, the road keeps its offset and its direction
//! at the settlement – and every route length that depends on those ends changes by the change of the spline lengths
//! (for a route without splines by the change of the straight distance to the other end).
//!
//! Spline lengths use the editor's flattening convention (64 steps per Bezier, `TradeSplines::flatten`); route lengths
//! of the shipped file are not plain sums of them (they carry extra weights), so they are updated by differences.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, anyhow, bail, ensure};

use super::{Plan, target::Kind};
use crate::{
    esf::{self, Node},
    regionmap::regions::RegionData,
};

/// Segments per cubic Bezier when measuring a spline.
const STEPS: usize = 64;
/// A road end must lie this close to the settlement centre for the node to be attributed to it.
const NODE_REACH: f64 = 3.0;

/// Positions of the nodes before and after the moves: capital centres by region key, port slots by slot key.
#[derive(Clone, Debug, Default)]
pub(super) struct Positions {
    old: HashMap<String, [f32; 2]>,
    new: HashMap<String, [f32; 2]>,
    ports: HashMap<String, [f32; 2]>,
}

impl Positions {
    pub(super) fn of(rd: &RegionData) -> Result<Self> {
        let mut out = Self::default();
        for index in 0..rd.regions.len() {
            for target in super::target::targets(rd, index)? {
                match target.kind {
                    Kind::Capital => {
                        out.old.insert(target.region.clone(), target.position);
                    }
                    Kind::Port => {
                        out.ports.insert(target.slot.clone(), target.position);
                    }
                    Kind::Town => {}
                }
            }
        }
        out.new = out.old.clone();
        Ok(out)
    }

    pub(super) fn moved(&mut self, plans: &[Plan]) {
        for plan in plans.iter().filter(|plan| plan.target.kind == Kind::Capital) {
            self.new.insert(plan.target.region.clone(), plan.to);
        }
    }

    fn before(&self, key: &str) -> Option<[f32; 2]> {
        self.old.get(key).or_else(|| self.ports.get(key)).copied()
    }

    fn after(&self, key: &str) -> Option<[f32; 2]> {
        self.new.get(key).or_else(|| self.ports.get(key)).copied()
    }

    /// Change of the straight distance between nodes `a` and `b` (keys as in `PORTS[]` / `SETTLEMENTS[]`).
    pub(super) fn distance_change(&self, a: &str, b: &str) -> Result<f64> {
        let get = |known: Option<[f32; 2]>, key: &str| known.ok_or_else(|| anyhow!("no position for trade node {key}"));
        let flat = |p: [f32; 2], q: [f32; 2]| (f64::from(p[0]) - f64::from(q[0])).hypot(f64::from(p[1]) - f64::from(q[1]));
        let old = flat(get(self.before(a), a)?, get(self.before(b), b)?);
        let new = flat(get(self.after(a), a)?, get(self.after(b), b)?);
        Ok(new - old)
    }
}

/// Length of one cubic Bezier `p[0..4]` as a 64-segment polyline.
pub(super) fn bezier_len(p: &[[f32; 2]]) -> f64 {
    let q: Vec<[f64; 2]> = p.iter().map(|v| [f64::from(v[0]), f64::from(v[1])]).collect();
    let mut last = q[0];
    let mut total = 0.0;
    for i in 1..=STEPS {
        let t = i as f64 / STEPS as f64;
        let u = 1.0 - t;
        let mut at = [0.0; 2];
        for k in 0..2 {
            at[k] = u.powi(3) * q[0][k] + 3.0 * u * u * t * q[1][k] + 3.0 * u * t * t * q[2][k] + t.powi(3) * q[3][k];
        }
        total += (at[0] - last[0]).hypot(at[1] - last[1]);
        last = at;
    }
    total
}

/// Length of a chain of Beziers (`3k + 1` control points).
pub(super) fn chain_len(points: &[[f32; 2]]) -> f64 {
    (0..points.len().saturating_sub(1)).step_by(3).map(|s| bezier_len(&points[s..s + 4])).sum()
}

fn shifted(p: [f32; 2], d: [f64; 2]) -> [f32; 2] {
    [(f64::from(p[0]) + d[0]) as f32, (f64::from(p[1]) + d[1]) as f32]
}

/// One spline whose end moved.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SplineChange {
    pub old: Vec<[f32; 2]>,
    pub new: Vec<[f32; 2]>,
    pub old_len: f64,
    pub new_len: f64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Change {
    /// spline index -> change
    pub splines: BTreeMap<usize, SplineChange>,
    pub routes: usize,
    pub notes: Vec<String>,
}

fn key_list(root: &Node, name: &str) -> Result<Vec<String>> {
    root.get(name)?
        .items()?
        .iter()
        .map(|item| Ok(item.first().ok_or_else(|| anyhow!("empty {name} item"))?.as_str()?.to_string()))
        .collect()
}

/// Key (`PORTS[]` entry or region key of `SETTLEMENTS[]`) of node `index`; nodes are ports first, then settlements.
fn node_key<'a>(ports: &'a [String], settlements: &'a [String], index: usize) -> Result<&'a str> {
    if index < ports.len() {
        Ok(ports[index].as_str())
    } else {
        settlements
            .get(index - ports.len())
            .map(String::as_str)
            .ok_or_else(|| anyhow!("trade_routes.esf refers to node {index}, which does not exist"))
    }
}

/// Moves the road ends of the planned capitals in `root` (`trade_routes.esf`) and updates the route lengths.
pub(super) fn apply(root: &mut Node, plans: &[Plan], positions: &Positions) -> Result<Change> {
    let ports = key_list(root, "PORTS")?;
    let settlements = key_list(root, "SETTLEMENTS")?;
    let mut change = Change::default();
    // node index -> (plan index)
    let mut nodes: BTreeMap<usize, usize> = BTreeMap::new();
    for (at, plan) in plans.iter().enumerate().filter(|(_, plan)| plan.target.kind == Kind::Capital) {
        match settlements.iter().position(|key| *key == plan.target.region) {
            Some(position) => {
                nodes.insert(ports.len() + position, at);
            }
            None => change.notes.push(format!(
                "trade_routes.esf has no node for {}: no road to carry",
                plan.target.region
            )),
        }
    }
    if nodes.is_empty() {
        return Ok(change);
    }
    let delta_of = |plan: &Plan| {
        [f64::from(plan.to[0]) - f64::from(plan.target.position[0]), f64::from(plan.to[1]) - f64::from(plan.target.position[1])]
    };

    // ---- which spline ends belong to the moved nodes: (spline, at the last point, plan)
    let mut ends: BTreeSet<(usize, bool, usize)> = BTreeSet::new();
    let routes = root.get("ROUTES")?.items()?;
    for (route_index, route) in routes.iter().enumerate() {
        ensure!(route.len() >= 4, "ROUTES[{route_index}] has {} children", route.len());
        let (from, to) = (route[0].as_u32()? as usize, route[1].as_u32()? as usize);
        let ids = route[2].u32_array()?;
        let (Some(first), Some(last)) = (ids.first(), ids.last()) else { continue };
        if let Some(&plan) = nodes.get(&from) {
            ends.insert(((first >> 1) as usize, first & 1 == 1, plan));
        }
        if let Some(&plan) = nodes.get(&to) {
            ends.insert(((last >> 1) as usize, last & 1 == 0, plan));
        }
    }

    // ---- translate the ends
    let mut points: BTreeMap<usize, Vec<[f32; 2]>> = BTreeMap::new();
    for &(spline, at_end, plan_index) in &ends {
        if !points.contains_key(&spline) {
            let item = root
                .get("SPLINES")?
                .items()?
                .get(spline)
                .ok_or_else(|| anyhow!("a route refers to spline {spline}, which does not exist"))?;
            points.insert(spline, item[0].xy_array()?);
        }
        let plan = &plans[plan_index];
        let pts = points.get_mut(&spline).ok_or_else(|| anyhow!("spline {spline} vanished"))?;
        ensure!(pts.len() >= 4, "spline {spline} has {} control points", pts.len());
        let (end, handle) = if at_end { (pts.len() - 1, pts.len() - 2) } else { (0, 1) };
        let reach = (f64::from(pts[end][0]) - f64::from(plan.target.position[0]))
            .hypot(f64::from(pts[end][1]) - f64::from(plan.target.position[1]));
        ensure!(
            reach <= NODE_REACH,
            "the road end of spline {spline} is {reach:.2} u from {} (expected a trade node near the settlement)",
            plan.target.region
        );
        let d = delta_of(plan);
        pts[end] = shifted(pts[end], d);
        pts[handle] = shifted(pts[handle], d);
    }
    for (&spline, new) in &points {
        let item = root.get("SPLINES")?.items()?.get(spline).ok_or_else(|| anyhow!("spline {spline} vanished"))?;
        let old = item[0].xy_array()?;
        change.splines.insert(
            spline,
            SplineChange { old_len: chain_len(&old), new_len: chain_len(new), old, new: new.clone() },
        );
    }
    {
        let items = root.get_mut("SPLINES")?.items_mut()?;
        for (&spline, new) in &points {
            items[spline][0] = esf::arr_xy(new);
        }
    }

    // ---- route lengths
    let mut updates: Vec<(usize, f32)> = Vec::new();
    {
        let routes = root.get("ROUTES")?.items()?;
        for (route_index, route) in routes.iter().enumerate() {
            let (from, to) = (route[0].as_u32()? as usize, route[1].as_u32()? as usize);
            let ids = route[2].u32_array()?;
            let mut delta: f64 = ids
                .iter()
                .filter_map(|id| change.splines.get(&((id >> 1) as usize)))
                .map(|c| c.new_len - c.old_len)
                .sum();
            if ids.is_empty() && (nodes.contains_key(&from) || nodes.contains_key(&to)) {
                delta += positions.distance_change(node_key(&ports, &settlements, from)?, node_key(&ports, &settlements, to)?)?;
            }
            if delta != 0.0 {
                let length = route[3].as_f32()?;
                updates.push((route_index, (f64::from(length) + delta).max(0.0) as f32));
            }
        }
    }
    change.routes = updates.len();
    let routes = root.get_mut("ROUTES")?.items_mut()?;
    for (index, length) in updates {
        match routes.get_mut(index).and_then(|route| route.get_mut(3)) {
            Some(slot) => *slot = esf::f32v(length),
            None => bail!("ROUTES[{index}] lost its length"),
        }
    }
    change.notes.push(format!(
        "trade_routes.esf: {} road end(s) carried with {} settlement(s), {} route length(s) updated",
        ends.len(),
        nodes.len(),
        change.routes
    ));
    Ok(change)
}
