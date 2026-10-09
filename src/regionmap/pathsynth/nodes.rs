//! Settlement node pieces (type 7, region path): the vanilla shapes measured on 17 India settlements
//! (work/exp4/pathsynth.py): `settlement_4_slot` cities get a regular octagon (circumradius 2.0745, vertices at
//! 0/45/.. degrees, centre = point + (0.06842, -0.01508)), `settlement_1_slot` towns an axis-aligned diamond
//! (half-diagonal 1.6973) centred on the point.

use super::geometry::{Contour, Pt, ring_pts};

/// Settlement slot type of the donor settlement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    /// `settlement_4_slot` city
    FourSlot,
    /// `settlement_1_slot` town
    OneSlot,
}

impl SlotKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "settlement_4_slot" => Some(Self::FourSlot),
            "settlement_1_slot" => Some(Self::OneSlot),
            _ => None,
        }
    }
}

/// vertex offsets from the rounded settlement point, fixed point, CCW from W
const OCTAGON: [Pt; 8] = [
    [-2102592, -15808],
    [-1465728, 1521680],
    [71744, 2158528],
    [1609216, 1521680],
    [2246080, -15808],
    [1609216, -1553296],
    [71744, -2190144],
    [-1465728, -1553296],
];
/// CCW from W
const DIAMOND: [Pt; 4] = [[-1779712, 0], [0, -1779712], [1779712, 0], [0, 1779712]];

/// The node polygon around `point` (fixed point, relative coordinates).
pub fn node_ring(kind: SlotKind, point: Pt) -> Contour {
    let offs: &[Pt] = match kind {
        SlotKind::FourSlot => &OCTAGON,
        SlotKind::OneSlot => &DIAMOND,
    };
    let raw: Contour = offs.iter().map(|o| [point[0] + o[0], point[1] + o[1]]).collect();
    ring_pts(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regionmap::pathsynth::geometry::{FIX, area2};

    #[test]
    fn node_areas_match_vanilla() {
        let o = node_ring(SlotKind::FourSlot, [0, 0]);
        assert!((area2(&o) as f64 / 2.0 / (FIX * FIX) - 12.162).abs() < 0.01);
        let d = node_ring(SlotKind::OneSlot, [0, 0]);
        assert!((area2(&d) as f64 / 2.0 / (FIX * FIX) - 5.761).abs() < 0.01);
    }
}
