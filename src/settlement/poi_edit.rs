//! `poi.esf`: the settlement point (type 4) of a moved capital and every distance that is measured to it.
//!
//! Measured on the ETW map: `settlement_distance` of a region's points is the flat distance to its type-4 point (median
//! error 6e-7 u), the settlement point's neighbour list holds its distance to each bordering region's outline, and the
//! `border` entries of the points of the neighbouring regions hold a distance to this region's settlement that is only
//! approximately the flat distance (median deviation 0.33 u). Moving a capital therefore sets the settlement point and
//! its neighbour distances exactly and shifts every other distance by the change of the flat distance, which keeps the
//! shipped offsets. Towns and ports have no point in `poi.esf`.

use anyhow::{Result, anyhow, ensure};

use super::{Plan, target::Kind};
use crate::{
    esf::Node,
    regionmap::{poi, regions::RegionData},
};

fn p64(p: [f32; 2]) -> [f64; 2] {
    [f64::from(p[0]), f64::from(p[1])]
}

fn flat(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Moves the settlement points of the planned capitals in `root` (`poi.esf`). `rd` supplies the borders.
pub(super) fn apply(root: &mut Node, rd: &RegionData, plans: &[Plan]) -> Result<Vec<String>> {
    let capitals: Vec<&Plan> = plans.iter().filter(|plan| plan.target.kind == Kind::Capital).collect();
    if capitals.is_empty() {
        return Ok(Vec::new());
    }
    let mut pois = poi::parse(root)?;
    let mut shifted = 0usize;
    for plan in &capitals {
        let key = plan.target.region.as_str();
        let (from, to) = (p64(plan.target.position), p64(plan.to));
        let at = pois
            .iter()
            .position(|p| p.kind == poi::SETTLEMENT && p.region == key)
            .ok_or_else(|| anyhow!("poi.esf has no settlement point for {key}"))?;
        ensure!(
            flat([pois[at].x, pois[at].y], from) < 0.01,
            "poi.esf: the settlement point of {key} is at ({}, {}), not at the settlement {from:?}",
            pois[at].x,
            pois[at].y
        );
        for p in pois.iter_mut() {
            let here = [p.x, p.y];
            if p.region == key {
                if p.kind == poi::SETTLEMENT {
                    p.x = to[0];
                    p.y = to[1];
                    p.settlement_distance = 0.0;
                    let names: Vec<String> = p.neighbours.iter().map(|(name, _)| name.clone()).collect();
                    for name in names {
                        if let Ok(index) = rd.index(&name) {
                            p.set_neighbour(&name, rd.distance_to_region(index, to) as f32);
                        }
                    }
                } else {
                    p.settlement_distance =
                        (f64::from(p.settlement_distance) + flat(here, to) - flat(here, from)).max(0.0) as f32;
                }
                shifted += 1;
            }
            for entry in p.border.iter_mut().filter(|entry| entry.0 == key) {
                entry.1 = (f64::from(entry.1) + flat(here, to) - flat(here, from)).max(0.0) as f32;
                shifted += 1;
            }
        }
    }
    poi::store(root, &pois)?;
    Ok(vec![format!(
        "poi.esf: {} settlement point(s) moved, {shifted} distance entries updated",
        capitals.len()
    )])
}
