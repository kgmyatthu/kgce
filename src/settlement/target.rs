//! The movable settlements of a [`RegionData`]: every region's capital (one target for the whole `settlement:` slot
//! cluster), its town slots and its port slots, each under a stable key.
//!
//! Slot keys (`slot_descriptions` of `regions.esf`, identical in the ETW source and the converted NTW file):
//! * capital: the settlement id `settlement:<region>:<name>` (the first three parts of every settlement slot key;
//!   the converter re-indexes the tail `:<type>:<n>`, the id never changes),
//! * town: `town:<region>:<name>`, port: `port:<region>:<name>`.
//!
//! Resources (mines, farms, ...) are not settlements and are not listed.

use anyhow::{Result, anyhow, bail, ensure};

use crate::{
    esf::{Node, Value},
    regionmap::regions::RegionData,
};

/// ETW general settlement slot types; the converter keeps at most four of them (`settlement_<n>_slot`).
const GENERAL_SLOT_TYPES: [&str; 6] = [
    "settlement_army-admin",
    "settlement_ordnance",
    "settlement_government",
    "settlement_culture",
    "settlement_minor",
    "settlement_navy-admin",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Capital,
    Town,
    Port,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub region: String,
    pub region_index: usize,
    /// canonical slot id: the settlement id of a capital, the exact slot key of a town / port
    pub slot: String,
    pub kind: Kind,
    /// world position (x, y) of the capital centre / the slot
    pub position: [f32; 2],
    /// number of general settlement slots of a capital (0 for towns and ports); decides the pathfinding node shape
    pub general_slots: usize,
    /// indices into `slot_descriptions` that move with the target (capital: its whole `settlement:` cluster)
    pub members: Vec<usize>,
}

/// World position (x, y) of a slot description's position child: ETW slots hold an XY, converted ones an XYZ whose
/// ground plane is (x, z).
pub(super) fn slot_xz(node: &Node) -> Option<[f32; 2]> {
    match node {
        Node::Value(Value::XY(p)) => Some(*p),
        Node::Value(Value::XYZ(p)) => Some([p[0], p[2]]),
        _ => None,
    }
}

pub(super) fn slot_key(slot: &[Node]) -> Result<&str> {
    slot.first().ok_or_else(|| anyhow!("empty slot description"))?.as_str()
}

/// General slot count announced by a slot type: `settlement_<n>_slot` (converted file) is `Some(n)`.
fn numbered_general(slot_type: &str) -> Option<usize> {
    slot_type.strip_prefix("settlement_")?.strip_suffix("_slot")?.parse().ok()
}

/// Every movable target of region `region_index`, the capital first.
pub fn targets(data: &RegionData, region_index: usize) -> Result<Vec<Target>> {
    let region = &data.regions[region_index];
    let Some(settlement) = &region.settlement else {
        return Ok(Vec::new());
    };
    let children = settlement.children()?;
    ensure!(
        children.len() >= 3,
        "{}: the settlement record has {} children, expected at least 3",
        region.key,
        children.len()
    );
    let center = children[0]
        .as_xy()
        .map_err(|error| anyhow!("{}: settlement centre: {error}", region.key))?;
    let slots = children[2].items()?;
    let mut out: Vec<Target> = Vec::new();
    let mut capital: Option<(String, Vec<usize>)> = None;
    let (mut numbered, mut general_count) = (None::<usize>, 0usize);
    for (index, slot) in slots.iter().enumerate() {
        let key = slot_key(slot)?;
        let parts: Vec<&str> = key.split(':').collect();
        match parts[0] {
            "settlement" => {
                ensure!(parts.len() >= 3, "{}: settlement slot key {key} has no settlement id", region.key);
                let id = parts[..3].join(":");
                match &mut capital {
                    Some((known, members)) => {
                        ensure!(
                            *known == id,
                            "{}: settlement slots of two settlements ({known}, {id})",
                            region.key
                        );
                        members.push(index);
                    }
                    None => capital = Some((id, vec![index])),
                }
                if let Some(Ok(slot_type)) = slot.get(1).map(Node::as_str) {
                    if let Some(n) = numbered_general(slot_type) {
                        numbered = Some(numbered.map_or(n, |known| known.max(n)));
                    } else if GENERAL_SLOT_TYPES.contains(&slot_type) {
                        general_count += 1;
                    }
                }
            }
            "town" | "port" => {
                ensure!(parts.len() >= 3, "{}: slot key {key} is not <kind>:<region>:<name>", region.key);
                let position = slot
                    .get(2)
                    .and_then(slot_xz)
                    .ok_or_else(|| anyhow!("{}: slot {key} has no position", region.key))?;
                out.push(Target {
                    region: region.key.clone(),
                    region_index,
                    slot: key.to_string(),
                    kind: if parts[0] == "town" { Kind::Town } else { Kind::Port },
                    position,
                    general_slots: 0,
                    members: vec![index],
                });
            }
            _ => {}
        }
    }
    if let Some((id, members)) = capital {
        out.insert(
            0,
            Target {
                region: region.key.clone(),
                region_index,
                slot: id,
                kind: Kind::Capital,
                position: center,
                general_slots: numbered.unwrap_or(general_count.min(4)),
                members,
            },
        );
    }
    Ok(out)
}

/// The target `slot` of region `region`. `slot` is the canonical id (see the module docs), a full slot key of the
/// capital's cluster (`settlement:<region>:<name>:<type>:<n>`) or the bare name when that is unambiguous.
pub fn resolve(data: &RegionData, region: &str, slot: &str) -> Result<Target> {
    let index = data.index(region)?;
    let all = targets(data, index)?;
    let wanted = if slot.starts_with("settlement:") {
        slot.split(':').take(3).collect::<Vec<_>>().join(":")
    } else {
        slot.to_string()
    };
    if let Some(found) = all.iter().find(|target| target.slot == wanted) {
        return Ok(found.clone());
    }
    if !slot.contains(':') {
        let named: Vec<&Target> = all
            .iter()
            .filter(|target| target.slot.rsplit(':').next() == Some(slot))
            .collect();
        match named.as_slice() {
            [one] => return Ok((*one).clone()),
            [] => {}
            many => bail!(
                "{slot} is ambiguous in {region}: {}",
                many.iter().map(|target| target.slot.as_str()).collect::<Vec<_>>().join(", ")
            ),
        }
    }
    bail!(
        "{region} has no movable settlement {slot}; available: {}",
        if all.is_empty() {
            "none".to_string()
        } else {
            all.iter().map(|target| target.slot.as_str()).collect::<Vec<_>>().join(", ")
        }
    )
}
