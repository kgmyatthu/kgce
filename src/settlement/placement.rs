//! Where a settlement may stand: inside its own passable area (outer ring in, hole rings out), clear of every outline,
//! and at the spacing the shipped map keeps between settlements.
//!
//! Measured on the ETW world map (138 capitals, 286 towns, 132 ports): the closest capital / town pair is 3.21 u apart,
//! the closest town pair 3.59 u. The pathfinding node of a capital has a circumradius of about 2.07 u (one-slot capital
//! 1.70 u), that of a town 1.05 u; the node must lie inside the land of the region (see `nav`), so the same margins are
//! required of the point itself.

use anyhow::{Result, bail, ensure};

use super::target::{Kind, Target, slot_xz, targets};
use crate::regionmap::{geometry::point_segment_distance, regions::RegionData};

/// Smallest distance between two settlement points (capital centres, town and port slots).
pub const MIN_SEPARATION: f64 = 3.2;
/// Smallest distance of a capital centre to the outline of its area ([`crate::map::splits::SETTLEMENT_CLEARANCE`]).
const CAPITAL_CLEARANCE: f64 = 2.1;
/// Same for a one-slot capital (diamond node, half-diagonal 1.70 u).
const SMALL_CAPITAL_CLEARANCE: f64 = 1.8;
const TOWN_CLEARANCE: f64 = 1.1;
/// A slot that moves with a capital must stay at least this far from the outline.
const SLOT_CLEARANCE: f64 = 0.2;

fn clearance(target: &Target) -> f64 {
    match target.kind {
        Kind::Capital if target.general_slots > 1 => CAPITAL_CLEARANCE,
        Kind::Capital => SMALL_CAPITAL_CLEARANCE,
        _ => TOWN_CLEARANCE,
    }
}

fn p64(p: [f32; 2]) -> [f64; 2] {
    [f64::from(p[0]), f64::from(p[1])]
}

/// Distance from `at` to the nearest ring (outer or hole) of area `area` of region `ri`.
fn outline_distance(data: &RegionData, ri: usize, area: usize, at: [f64; 2]) -> f64 {
    let mut best = f64::INFINITY;
    for outline in &data.regions[ri].areas[area].outlines {
        let n = outline.ring.len();
        for k in 0..n {
            let (a, b) = (data.v(outline.ring[k]), data.v(outline.ring[(k + 1) % n]));
            best = best.min(point_segment_distance(at, a, b));
        }
    }
    best
}

/// Current positions of the slots that move with `target`.
fn member_positions(data: &RegionData, target: &Target) -> Vec<(String, [f32; 2])> {
    let Some(settlement) = &data.regions[target.region_index].settlement else {
        return Vec::new();
    };
    let Ok(children) = settlement.children() else {
        return Vec::new();
    };
    let Some(slots) = children.get(2).and_then(|node| node.items().ok()) else {
        return Vec::new();
    };
    target
        .members
        .iter()
        .filter_map(|&index| {
            let slot = slots.get(index)?;
            Some((slot.first()?.as_str().ok()?.to_string(), slot_xz(slot.get(2)?)?))
        })
        .collect()
}

/// Checks that moving `target` to `to` is allowed. All distances are world units.
///
/// * ports are refused: their landing crossings (pathfinding), dock point (sea grids, trade routes) and coast
///   geometry would all have to be re-synthesised;
/// * the point must lie in a passable area of the target's own region, in the same area as before (a move never
///   changes the landmass, so the AI's connectivity is untouched), outside every hole ring;
/// * it keeps the node clearance from every outline of that area, and so does every slot that moves with a capital;
/// * it keeps [`MIN_SEPARATION`] from every other settlement point of the map.
pub fn check(data: &RegionData, target: &Target, to: [f32; 2]) -> Result<()> {
    ensure!(to.iter().all(|v| v.is_finite()), "The destination of {} is not a finite position", target.slot);
    if target.kind == Kind::Port {
        bail!(
            "{} is a port and cannot be moved: its landing crossings, dock point and trade-route port node follow the coastline",
            target.slot
        );
    }
    let ri = target.region_index;
    let region = &data.regions[ri];
    ensure!(region.kind == "land", "{} is not a land region", region.key);
    let home = data.area_at(ri, p64(target.position));
    let Some(area) = data.area_at(ri, p64(to)) else {
        bail!("{} would leave its region {}: the destination lies outside it or inside a hole", target.slot, region.key);
    };
    if let Some(home) = home {
        ensure!(
            area == home,
            "{} must stay in its own area of {} (the destination is in another landmass)",
            target.slot,
            region.key
        );
    }
    ensure!(region.areas[area].passable, "The destination lies in an impassable area of {}", region.key);
    let need = clearance(target);
    let clear = outline_distance(data, ri, area, p64(to));
    ensure!(
        clear >= need,
        "{} would stand {clear:.2} u from the border or coast of {}; keep at least {need:.1} u so its pathfinding node fits",
        target.slot,
        region.key
    );
    let delta = [f64::from(to[0]) - f64::from(target.position[0]), f64::from(to[1]) - f64::from(target.position[1])];
    if target.kind == Kind::Capital {
        for (key, at) in member_positions(data, target) {
            let moved = [f64::from(at[0]) + delta[0], f64::from(at[1]) + delta[1]];
            ensure!(
                data.area_at(ri, moved) == Some(area),
                "Slot {key} of {} would fall outside the area of {}",
                target.slot,
                region.key
            );
            ensure!(
                outline_distance(data, ri, area, moved) >= SLOT_CLEARANCE,
                "Slot {key} of {} would touch the border of {}",
                target.slot,
                region.key
            );
        }
    }
    for other_index in 0..data.regions.len() {
        for other in targets(data, other_index)? {
            if other.region_index == ri && other.slot == target.slot {
                continue;
            }
            let (dx, dy) = (f64::from(to[0]) - f64::from(other.position[0]), f64::from(to[1]) - f64::from(other.position[1]));
            let distance = dx.hypot(dy);
            ensure!(
                distance >= MIN_SEPARATION,
                "{} would stand {distance:.2} u from {} ({}); settlements keep at least {MIN_SEPARATION} u apart",
                target.slot,
                other.slot,
                other.region
            );
        }
    }
    Ok(())
}
