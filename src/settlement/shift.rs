//! Translation of a `settlement_and_slots` record: the one place that changes slot coordinates in `regions.esf`, used by
//! the editor preview and by the exported file alike so that both always agree.

use anyhow::{Result, anyhow, ensure};

use super::target::{Kind, Target, slot_key, slot_xz};
use crate::esf::{Node, Value};

/// Where one slot description went.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotMove {
    pub key: String,
    pub from: [f32; 2],
    pub to: [f32; 2],
}

/// Maps points of the moved target: the old anchor lands exactly on the requested destination (so the exported
/// position equals the preview bit for bit), every other point is translated by the same offset in double precision
/// and rounded once to `f32`.
struct Mover {
    from: [f32; 2],
    to: [f32; 2],
    dx: f64,
    dy: f64,
}

impl Mover {
    fn new(from: [f32; 2], to: [f32; 2]) -> Self {
        Self {
            from,
            to,
            dx: f64::from(to[0]) - f64::from(from[0]),
            dy: f64::from(to[1]) - f64::from(from[1]),
        }
    }

    fn point(&self, p: [f32; 2]) -> [f32; 2] {
        if p == self.from {
            self.to
        } else {
            [(f64::from(p[0]) + self.dx) as f32, (f64::from(p[1]) + self.dy) as f32]
        }
    }

    /// Translates every coordinate below `node`; returns how many coordinate values changed.
    fn node(&self, node: &mut Node) -> usize {
        match node {
            Node::Value(Value::XY(p)) => {
                *p = self.point(*p);
                1
            }
            Node::Value(Value::XYZ(p)) => {
                let q = self.point([p[0], p[2]]);
                p[0] = q[0];
                p[2] = q[1];
                1
            }
            Node::Value(Value::Array { code, data }) if *code == 0x4c => {
                let mut count = 0;
                for chunk in data.chunks_exact_mut(8) {
                    let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let y = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                    let q = self.point([x, y]);
                    chunk[0..4].copy_from_slice(&q[0].to_le_bytes());
                    chunk[4..8].copy_from_slice(&q[1].to_le_bytes());
                    count += 1;
                }
                count
            }
            Node::Value(Value::Array { code, data }) if *code == 0x4d => {
                let mut count = 0;
                for chunk in data.chunks_exact_mut(12) {
                    let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let z = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                    let q = self.point([x, z]);
                    chunk[0..4].copy_from_slice(&q[0].to_le_bytes());
                    chunk[8..12].copy_from_slice(&q[1].to_le_bytes());
                    count += 1;
                }
                count
            }
            Node::Value(_) => 0,
            Node::Record { children, .. } => children.iter_mut().map(|child| self.node(child)).sum(),
            Node::Array { items, .. } => items.iter_mut().flatten().map(|child| self.node(child)).sum(),
        }
    }
}

/// Number of coordinate values (XY, XYZ, points of packed point arrays) below `node`.
fn coordinates(node: &Node) -> usize {
    match node {
        Node::Value(Value::XY(_)) | Node::Value(Value::XYZ(_)) => 1,
        Node::Value(Value::Array { code, data }) if *code == 0x4c => data.len() / 8,
        Node::Value(Value::Array { code, data }) if *code == 0x4d => data.len() / 12,
        Node::Value(_) => 0,
        Node::Record { children, .. } => children.iter().map(coordinates).sum(),
        Node::Array { items, .. } => items.iter().flatten().map(coordinates).sum(),
    }
}

/// Moves `target` to `to` inside its region's `settlement_and_slots` record `settlement`: the capital centre, its
/// footprint and every slot of its cluster (or one town slot) translate by the same offset in x and y; heights stay
/// (slot heights are one constant per slot type). Returns where each slot went.
pub fn move_in(settlement: &mut Node, target: &Target, to: [f32; 2]) -> Result<Vec<SlotMove>> {
    let mover = Mover::new(target.position, to);
    let children = settlement.children_mut()?;
    ensure!(children.len() >= 3, "{}: the settlement record is too short", target.region);
    if target.kind == Kind::Capital {
        for rest in &children[3..] {
            ensure!(
                coordinates(rest) == 0,
                "{}: the settlement carries road, railway or canal geometry; it cannot be moved",
                target.region
            );
        }
        ensure!(
            children[0].as_xy().ok() == Some(target.position),
            "{}: the settlement centre is not at {:?}",
            target.region,
            target.position
        );
        mover.node(&mut children[0]);
        mover.node(&mut children[1]);
    }
    let slots = children[2].items_mut()?;
    let mut moved = Vec::with_capacity(target.members.len());
    for &index in &target.members {
        let slot = slots
            .get_mut(index)
            .ok_or_else(|| anyhow!("{}: slot description {index} vanished", target.region))?;
        let key = slot_key(slot)?.to_string();
        let from = slot.get(2).and_then(slot_xz).ok_or_else(|| anyhow!("{key} has no position"))?;
        if target.kind != Kind::Capital {
            ensure!(from == target.position, "{key}: the slot is not at {:?}", target.position);
        }
        for child in slot.iter_mut() {
            mover.node(child);
        }
        let now = slot.get(2).and_then(slot_xz).ok_or_else(|| anyhow!("{key} lost its position"))?;
        moved.push(SlotMove { key, from, to: now });
    }
    if target.kind != Kind::Capital {
        ensure!(
            moved.first().map(|slot| slot.to) == Some(to),
            "{}: the slot did not land on the destination",
            target.slot
        );
    }
    Ok(moved)
}
