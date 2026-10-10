//! `startpos.esf`: every record that stores the position of a moved settlement, found through the ids that link them
//! (never by guessing coordinates), plus the garrisoned characters standing in a moved capital.
//!
//! Capital: `REGION/SETTLEMENT/SIEGEABLE_GARRISON_RESIDENCE` (fixed 12.20 children 10, 11), every `REGION_SLOT` of its
//! `settlement:` cluster (children 6, 7; array slots, road slot, fortification slot), the `CAI_REGION` (f32 children 4,
//! 5), the `CAI_SITUATED` of the `CAI_SETTLEMENT` item whose world id is the settlement's, the settlement
//! `CAI_REGION_HLCI` of the region (child 3, 4), and the road ends in `CAMPAIGN_TRADE_MANAGER` (segments mirror the
//! `trade_routes.esf` splines one to one). Town: its `REGION_SLOT` and the `CAI_REGION_SLOT` item of the same region at
//! the old point.
//!
//! Characters whose `LOCOMOTABLE` and the `CAI_SITUATED` of resource mobiles sit exactly on the old capital point are
//! the garrison of the settlement and move with it; nothing else is touched (an army anywhere else keeps its place).
//! Cached line-of-sight bitmaps of every moved entity are dropped (the inactive form the file already uses for 1375
//! settlements, 2206 entities in all): a cache computed around the old point must not survive the move.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, bail, ensure};

use super::{Plan, target::Kind, trade};
use crate::{
    esf::{self, Node, Value},
    startpos::cai::nodes as nd,
};

const FIXED: f64 = 1_048_576.0;
/// How far a stored position may deviate from the position `regions.esf` gives before the files are called inconsistent.
const NEAR: f64 = 0.01;

fn fixed(v: f32) -> i32 {
    (f64::from(v) * FIXED).round() as i32
}

fn fx(p: [f32; 2]) -> (i32, i32) {
    (fixed(p[0]), fixed(p[1]))
}

fn near(v: (i32, i32), p: [f32; 2]) -> bool {
    (f64::from(v.0) / FIXED - f64::from(p[0])).abs() < NEAR && (f64::from(v.1) / FIXED - f64::from(p[1])).abs() < NEAR
}

fn pair(children: &[Node], at: usize) -> Option<(i32, i32)> {
    match (children.get(at), children.get(at + 1)) {
        (Some(Node::Value(Value::I32(x))), Some(Node::Value(Value::I32(y)))) => Some((*x, *y)),
        _ => None,
    }
}

fn set_pair(children: &mut [Node], at: usize, value: (i32, i32)) -> Result<()> {
    ensure!(
        pair(children, at).is_some(),
        "expected a fixed-point pair at children {at} and {}",
        at + 1
    );
    children[at] = esf::i32v(value.0);
    children[at + 1] = esf::i32v(value.1);
    Ok(())
}

fn int_of(node: &Node) -> Option<u32> {
    match node {
        Node::Value(Value::U32(v)) => Some(*v),
        Node::Value(Value::I32(v)) => Some(*v as u32),
        _ => None,
    }
}

fn spec_string(item: &[Node], at: usize) -> Option<&str> {
    nd::spec_kids(item).ok()?.get(at).and_then(nd::string_of)
}

/// Drops the cached line of sight below `children` (a `LINE_OF_SIGHT` record becomes its inactive one-child form);
/// returns how many caches were active.
fn clear_sight(children: &mut [Node]) -> usize {
    let mut dropped = 0;
    for child in children.iter_mut() {
        if let Node::Record { name, children, .. } = child {
            if name.as_str() == "LINE_OF_SIGHT" && children.len() != 1 {
                *children = vec![esf::b(false)];
                dropped += 1;
            }
        }
    }
    dropped
}

#[derive(Default)]
struct Counts {
    regions: usize,
    slots: usize,
    cai: usize,
    mobiles: usize,
    characters: usize,
    sight: usize,
    segments: usize,
    routes: usize,
}

/// What the `WORLD` part learned about a capital: the settlement's world id and the fixed point the file stored.
struct Anchor {
    settlement_id: Option<u32>,
    old: Option<(i32, i32)>,
}

/// Every `REGION_SLOT` record of a `REGION_SLOT_MANAGER` (array slots, road slot, fortification slot).
fn for_each_slot(manager: &mut Node, f: &mut dyn FnMut(&mut Vec<Node>) -> Result<()>) -> Result<()> {
    for item in manager.get_mut("REGION_SLOT_ARRAY")?.items_mut()? {
        f(item
            .first_mut()
            .ok_or_else(|| anyhow!("empty REGION_SLOT_ARRAY item"))?
            .children_mut()?)?;
    }
    for name in ["ROAD_SLOT", "FORTIFICATION_SLOT"] {
        for child in manager.get_mut(name)?.children_mut()?.iter_mut() {
            if nd::is_record_named(child, "REGION_SLOT") {
                f(child.children_mut()?)?;
            }
        }
    }
    Ok(())
}

fn world_edit(world: &mut Node, plan: &Plan, counts: &mut Counts) -> Result<Anchor> {
    let key = plan.target.region.as_str();
    let items = world.path_mut(&["REGION_MANAGER", "REGIONS_ARRAY"])?.items_mut()?;
    let item = items
        .iter_mut()
        .find(|item| {
            item.first()
                .and_then(|node| node.children().ok())
                .and_then(|children| children.first())
                .and_then(nd::string_of)
                == Some(key)
        })
        .ok_or_else(|| anyhow!("the start position has no region {key}"))?;
    let region = item.first_mut().ok_or_else(|| anyhow!("empty REGIONS_ARRAY item"))?;
    let mut anchor = Anchor { settlement_id: None, old: None };
    if plan.target.kind == Kind::Capital {
        let settlement = region.get_mut("SETTLEMENT")?.children_mut()?;
        ensure!(settlement.len() > 5, "{key}: SETTLEMENT is too short");
        let residence = settlement[0].children_mut()?;
        ensure!(residence.len() > 14, "{key}: the settlement residence is too short");
        let old = pair(residence, 10).ok_or_else(|| anyhow!("{key}: the settlement residence has no position"))?;
        ensure!(
            near(old, plan.target.position),
            "{key}: the start position keeps the settlement at ({}, {}), regions.esf at {:?}",
            f64::from(old.0) / FIXED,
            f64::from(old.1) / FIXED,
            plan.target.position
        );
        anchor.settlement_id = residence.get(1).and_then(int_of);
        anchor.old = Some(old);
        set_pair(residence, 10, fx(plan.to))?;
        counts.sight += clear_sight(residence);
        counts.regions += 1;
    }
    let mut matched = 0usize;
    let mut sight = 0usize;
    let manager = region.get_mut("REGION_SLOT_MANAGER")?;
    for_each_slot(manager, &mut |slot| {
        let Some(slot_key) = slot.get(3).and_then(nd::string_of).map(str::to_owned) else {
            return Ok(());
        };
        let Some(moved) = plan.slots.iter().find(|moved| moved.key == slot_key) else {
            return Ok(());
        };
        let old = pair(slot, 6).ok_or_else(|| anyhow!("slot {slot_key} has no position"))?;
        ensure!(
            near(old, moved.from),
            "slot {slot_key}: the start position keeps it at ({}, {}), regions.esf at {:?}",
            f64::from(old.0) / FIXED,
            f64::from(old.1) / FIXED,
            moved.from
        );
        set_pair(slot, 6, fx(moved.to))?;
        if let Some(Node::Record { children, .. }) = slot.get_mut(0) {
            if let Some(residence) = pair(children, 10) {
                if residence != (0, 0) && near(residence, moved.from) {
                    set_pair(children, 10, fx(moved.to))?;
                }
            }
            sight += clear_sight(children);
        }
        matched += 1;
        Ok(())
    })?;
    ensure!(
        matched == plan.slots.len(),
        "{key}: the start position holds {matched} of the {} slots that move",
        plan.slots.len()
    );
    counts.slots += matched;
    counts.sight += sight;
    Ok(anchor)
}

fn cai_edit(world: &mut Node, plan: &Plan, anchor: &Anchor, counts: &mut Counts, notes: &mut Vec<String>) -> Result<()> {
    let key = plan.target.region.as_str();
    let (from, to) = (plan.target.position, plan.to);
    let region_cai = {
        let items = world.get_mut("CAI_WORLD_REGIONS")?.items_mut()?;
        let item = items
            .iter_mut()
            .find(|item| spec_string(item, 10) == Some(key))
            .ok_or_else(|| anyhow!("the start position has no CAI region {key}"))?;
        let id = nd::item_id(item)?;
        if plan.target.kind == Kind::Capital {
            let kids = nd::spec_kids_mut(item)?;
            let old = [kids.get(4).map(Node::as_f32), kids.get(5).map(Node::as_f32)];
            let (Some(Ok(x)), Some(Ok(y))) = (&old[0], &old[1]) else {
                bail!("CAI region {key} has no settlement position")
            };
            ensure!(
                (f64::from(*x) - f64::from(from[0])).abs() < NEAR && (f64::from(*y) - f64::from(from[1])).abs() < NEAR,
                "CAI region {key} keeps its settlement at ({x}, {y}), regions.esf at {from:?}"
            );
            nd::put(kids, 4, nd::f32n(to[0]))?;
            nd::put(kids, 5, nd::f32n(to[1]))?;
            counts.cai += 1;
        }
        id
    };
    if plan.target.kind == Kind::Capital {
        let settlement_id = anchor.settlement_id.ok_or_else(|| anyhow!("{key}: no settlement world id"))?;
        let old = anchor.old.ok_or_else(|| anyhow!("{key}: no settlement position"))?;
        let mut found = 0usize;
        for item in world.get_mut("CAI_WORLD_SETTLEMENTS")?.items_mut()? {
            if nd::spec_name(item).ok() != Some("CAI_SETTLEMENT") || nd::sc_at(nd::spec_kids(item)?, 2)? != settlement_id {
                continue;
            }
            let situated = nd::kids_mut(nd::rec_by_name_mut(item, "CAI_SITUATED")?)?;
            ensure!(pair(situated, 0) == Some(old), "CAI settlement of {key} is not at the stored settlement position");
            set_pair(situated, 0, fx(to))?;
            found += 1;
        }
        ensure!(found == 1, "{key}: {found} CAI settlements carry world id {settlement_id}");
        counts.cai += 1;
        let mut hlcis = 0usize;
        for item in world.get_mut("CAI_WORLD_REGION_HLCIS")?.items_mut()? {
            let Ok(kids) = nd::spec_kids_mut(item) else { continue };
            if kids.len() > 4 && nd::scalar(&kids[0]) == Some(region_cai) && pair(kids, 3) == Some(old) {
                set_pair(kids, 3, fx(to))?;
                hlcis += 1;
            }
        }
        if hlcis == 0 {
            notes.push(format!("{key}: the start position has no settlement HLCI at the old position"));
        }
        counts.cai += hlcis;
        for item in world.get_mut("CAI_WORLD_RESOURCE_MOBILES")?.items_mut()? {
            if let Ok(situated) = nd::rec_by_name_mut(item, "CAI_SITUATED") {
                let kids = nd::kids_mut(situated)?;
                if pair(kids, 0) == Some(old) {
                    set_pair(kids, 0, fx(to))?;
                    counts.mobiles += 1;
                }
            }
        }
    } else {
        let slot = plan.slots.first().ok_or_else(|| anyhow!("{key}: nothing to move"))?;
        let (old, new) = (fx(slot.from), fx(slot.to));
        let mut found = 0usize;
        for item in world.get_mut("CAI_WORLD_REGION_SLOTS")?.items_mut()? {
            let Ok(situated) = nd::rec_by_name_mut(item, "CAI_SITUATED") else { continue };
            let kids = nd::kids_mut(situated)?;
            if kids.len() > 2 && nd::scalar(&kids[2]) == Some(region_cai) && pair(kids, 0) == Some(old) {
                set_pair(kids, 0, new)?;
                found += 1;
            }
        }
        ensure!(found == 1, "{}: {found} CAI region slots stand at the old position", plan.target.slot);
        counts.cai += 1;
    }
    Ok(())
}

/// Moves every character whose position is exactly on a moved capital (the garrison).
fn move_garrisons(node: &mut Node, moves: &HashMap<(i32, i32), (i32, i32)>, counts: &mut Counts) {
    match node {
        Node::Record { name, children, .. } => {
            if name.as_str() == "CHARACTER" {
                let mut moved = false;
                if let Some(Node::Record { children: locomotable, .. }) =
                    children.iter_mut().find(|child| nd::is_record_named(child, "LOCOMOTABLE"))
                {
                    for at in [0usize, 2] {
                        if let Some(&to) = pair(locomotable, at).as_ref().and_then(|old| moves.get(old)) {
                            if set_pair(locomotable, at, to).is_ok() {
                                moved = true;
                            }
                        }
                    }
                }
                if moved {
                    counts.characters += 1;
                    counts.sight += clear_sight(children);
                }
            }
            for child in children.iter_mut() {
                move_garrisons(child, moves, counts);
            }
        }
        Node::Array { items, .. } => {
            for item in items.iter_mut() {
                for child in item.iter_mut() {
                    move_garrisons(child, moves, counts);
                }
            }
        }
        Node::Value(_) => {}
    }
}

fn trade_edit(
    manager: &mut Node,
    plans: &[Plan],
    change: &trade::Change,
    positions: &trade::Positions,
    counts: &mut Counts,
) -> Result<()> {
    // node index -> key, for ports and settlements
    let mut keys: HashMap<u32, String> = HashMap::new();
    for name in ["PORT_INDICES", "SETTLEMENT_INDICES"] {
        for item in manager.get(name)?.items()? {
            let (Some(key), Some(index)) = (item.first().and_then(|n| n.as_str().ok()), item.get(1).and_then(int_of)) else {
                bail!("{name} holds an item without key and index");
            };
            keys.insert(index, key.to_string());
        }
    }
    let moved_keys: HashSet<&str> = plans
        .iter()
        .filter(|plan| plan.target.kind == Kind::Capital)
        .map(|plan| plan.target.region.as_str())
        .collect();
    let moved_nodes: HashSet<u32> = keys
        .iter()
        .filter(|(_, key)| moved_keys.contains(key.as_str()))
        .map(|(&index, _)| index)
        .collect();

    // segments mirror the splines of trade_routes.esf one to one
    let mut delta: HashMap<usize, f64> = HashMap::new();
    let segments = manager.get_mut("TRADE_SEGMENTS")?.items_mut()?;
    for (&index, spline) in &change.splines {
        let item = segments
            .get_mut(index)
            .ok_or_else(|| anyhow!("the start position has no trade segment {index}"))?;
        let count = item.get(1).and_then(int_of).ok_or_else(|| anyhow!("trade segment {index} has no count"))? as usize;
        let end = 2 + count * 4;
        ensure!(item.len() > end + 8, "trade segment {index} is shorter than its count says");
        let expand = |points: &[[f32; 2]]| -> Vec<[f32; 2]> {
            let mut out = Vec::with_capacity(points.len() + points.len() / 3);
            for start in (0..points.len().saturating_sub(1)).step_by(3) {
                out.extend_from_slice(&points[start..start + 4]);
            }
            out
        };
        let (old_points, new_points) = (expand(&spline.old), expand(&spline.new));
        ensure!(old_points.len() == count * 4, "trade segment {index} does not mirror its spline");
        for (k, old) in old_points.iter().enumerate() {
            let stored = item[2 + k].as_xy()?;
            ensure!(
                (stored[0] - old[0]).abs() < 1e-3 && (stored[1] - old[1]).abs() < 1e-3,
                "trade segment {index} does not mirror its spline at point {k}"
            );
        }
        let old_lengths = item[end].f32_array()?;
        ensure!(old_lengths.len() == count, "trade segment {index}: {} lengths for {count} curves", old_lengths.len());
        let old_total = item[end + 2].as_f32()?;
        let mut lengths = old_lengths.clone();
        for b in 0..count {
            let (a, n) = (&old_points[b * 4..b * 4 + 4], &new_points[b * 4..b * 4 + 4]);
            if a != n {
                let before = trade::bezier_len(a);
                if before > 0.0 {
                    lengths[b] = (f64::from(old_lengths[b]) * trade::bezier_len(n) / before) as f32;
                }
            }
        }
        let mut starts = Vec::with_capacity(count);
        let mut run = 0.0f32;
        for length in &lengths {
            starts.push(run);
            run += *length;
        }
        let total = lengths.iter().map(|&v| f64::from(v)).sum::<f64>() as f32;
        for (k, point) in new_points.iter().enumerate() {
            item[2 + k] = esf::xyv(point[0], point[1]);
        }
        item[end] = esf::arr_f32(&lengths);
        item[end + 1] = esf::arr_f32(&starts);
        item[end + 2] = esf::f32v(total);
        ensure!(item[end + 7].as_f32().is_ok(), "trade segment {index}: unexpected layout after the lengths");
        item[end + 7] = esf::f32v(total);
        if let Node::Record { children, .. } = &mut item[end + 8] {
            if children.len() != 1 {
                *children = vec![esf::b(false)];
                counts.sight += 1;
            }
        }
        delta.insert(index, f64::from(total) - f64::from(old_total));
        counts.segments += 1;
    }

    // routes: segment ids are spline index * 2 + reversed; a route without segments is a direct link
    for route in manager.get_mut("TRADE_ROUTES")?.items_mut()? {
        ensure!(route.len() == 4, "TRADE_ROUTES item with {} children", route.len());
        let (from, to) = (
            route[0].as_u32().map_err(|e| anyhow!("trade route end: {e}"))?,
            route[1].as_u32().map_err(|e| anyhow!("trade route end: {e}"))?,
        );
        let ids = route[2].u32_array()?;
        let mut total: f64 = ids.iter().filter_map(|id| delta.get(&((id >> 1) as usize))).sum();
        if ids.is_empty() && (moved_nodes.contains(&from) || moved_nodes.contains(&to)) {
            let (Some(a), Some(b)) = (keys.get(&from), keys.get(&to)) else {
                bail!("trade route {from}-{to} joins a node without a key");
            };
            total += positions.distance_change(a, b)?;
        }
        if total != 0.0 {
            let length = route[3].as_f32()?;
            route[3] = esf::f32v((f64::from(length) + total).max(0.0) as f32);
            counts.routes += 1;
        }
    }
    Ok(())
}

/// Moves every record of the start position that holds the position of a planned settlement; returns report lines.
pub(super) fn apply(root: &mut Node, plans: &[Plan], change: &trade::Change, positions: &trade::Positions) -> Result<Vec<String>> {
    let mut counts = Counts::default();
    let mut notes: Vec<String> = Vec::new();
    let model = root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])?;
    let mut garrisons: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
    for plan in plans {
        let anchor = world_edit(model.get_mut("WORLD")?, plan, &mut counts)?;
        cai_edit(model.path_mut(&["CAI_INTERFACE", "CAI_WORLD"])?, plan, &anchor, &mut counts, &mut notes)?;
        if let Some(old) = anchor.old {
            garrisons.insert(old, fx(plan.to));
        }
    }
    if !garrisons.is_empty() {
        move_garrisons(model, &garrisons, &mut counts);
    }
    if plans.iter().any(|plan| plan.target.kind == Kind::Capital) {
        trade_edit(model.get_mut("CAMPAIGN_TRADE_MANAGER")?, plans, change, positions, &mut counts)?;
    }
    notes.push(format!(
        "startpos.esf: {} settlement residence(s), {} slot(s), {} CAI record(s), {} garrisoned character(s) and {} CAI mobile(s) moved; {} trade segment(s) / {} route length(s) updated; {} cached line-of-sight map(s) dropped",
        counts.regions, counts.slots, counts.cai, counts.characters, counts.mobiles, counts.segments, counts.routes, counts.sight
    ));
    Ok(notes)
}

/// Every `REGION_SLOT` (array slots, road slot, fortification slot) of a region record, key -> fixed position.
fn slot_positions(region: &Node) -> Result<HashMap<String, (i32, i32)>> {
    let manager = region.get("REGION_SLOT_MANAGER")?;
    let mut out = HashMap::new();
    let mut take = |children: &[Node]| {
        if let (Some(key), Some(at)) = (children.get(3).and_then(nd::string_of), pair(children, 6)) {
            out.insert(key.to_string(), at);
        }
    };
    for item in manager.get("REGION_SLOT_ARRAY")?.items()? {
        if let Some(slot) = item.first() {
            take(slot.children()?);
        }
    }
    for name in ["ROAD_SLOT", "FORTIFICATION_SLOT"] {
        for child in manager.get(name)?.children()? {
            if nd::is_record_named(child, "REGION_SLOT") {
                take(child.children()?);
            }
        }
    }
    Ok(out)
}

/// Checks a serialised and re-read start position: every planned settlement residence and slot stands at the fixed-point
/// image of the position `regions.esf` now has.
pub(super) fn verify(root: &Node, plans: &[Plan]) -> Result<()> {
    let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "WORLD"])?;
    let items = world.path(&["REGION_MANAGER", "REGIONS_ARRAY"])?.items()?;
    for plan in plans {
        let key = plan.target.region.as_str();
        let region = items
            .iter()
            .filter_map(|item| item.first())
            .find(|region| {
                region.children().ok().and_then(|children| children.first()).and_then(nd::string_of) == Some(key)
            })
            .ok_or_else(|| anyhow!("the written start position lost region {key}"))?;
        if plan.target.kind == Kind::Capital {
            let residence = region.get("SETTLEMENT")?.children()?[0].children()?;
            ensure!(
                pair(residence, 10) == Some(fx(plan.to)),
                "{key}: the start position settlement is not at {:?}",
                plan.to
            );
        }
        let slots = slot_positions(region)?;
        for moved in &plan.slots {
            ensure!(
                slots.get(&moved.key) == Some(&fx(moved.to)),
                "{}: the start position slot is not at {:?}",
                moved.key,
                moved.to
            );
        }
    }
    Ok(())
}
