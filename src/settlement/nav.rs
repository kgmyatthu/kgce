//! `pathfinding.esf`: a settlement is a type-7 "node" piece of its region's single path, carved out of the region's land
//! (type 0, same path) around the settlement point. Moving a settlement therefore
//! 1. turns the old node pieces back into land (merged with the land pieces they touch),
//! 2. carves the node polygon at the new point out of the region's land exactly like a split does for the node of its new
//!    region, and
//! 3. redoes the vertex hygiene and the rule words of every touched cell.
//!
//! Shapes measured on the ETW grids: capitals with four general slots a regular octagon (circumradius 2.07 u), one-slot
//! capitals a diamond (half-diagonal 1.70 u) – both from `pathsynth::nodes` – and towns an octagon whose vertices are at
//! (±0.98996, ±0.35355) / (±0.35355, ±0.98996) around the town point. Settlements the map has no node for (most of
//! America) stay without one. The node must fit completely: a destination whose node would be clipped by a border, coast
//! or another piece is refused, never silently squeezed.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, anyhow, bail, ensure};

use super::{
    Plan,
    target::{Kind, Target},
};
use crate::{
    esf::Node,
    regionmap::{
        pathfinding::Grid,
        pathsynth::{
            finish,
            geometry::{self as g, Contour, FIX, Pt, Shapes},
            land::Frame,
            model::{MPiece, Model},
            nodes::{self, SlotKind},
        },
        regions::RegionData,
        stages,
    },
};

/// Area growth of the old node ring when looking for its pieces (a piece must lie wholly inside the grown ring).
const GROW: f64 = 1.06;
/// Largest area a node may lose to its surroundings (map units²).
const CLIP_TOLERANCE: f64 = 1e-5;

/// Town node: vertex offsets from the town point, fixed 12.20, counter-clockwise from the east.
const TOWN_OFFSETS: [Pt; 8] = [
    [1038048, 370720],
    [370720, 1038048],
    [-370720, 1038048],
    [-1038048, 370720],
    [-1038048, -370720],
    [-370720, -1038048],
    [370720, -1038048],
    [1038048, -370720],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NodeShape {
    City,
    SmallCity,
    Town,
}

pub(super) fn shape_of(target: &Target) -> Result<NodeShape> {
    match target.kind {
        Kind::Town => Ok(NodeShape::Town),
        Kind::Capital => match target.general_slots {
            4 => Ok(NodeShape::City),
            1 => Ok(NodeShape::SmallCity),
            n => bail!("{}: no pathfinding node shape is known for a capital with {n} general slots", target.slot),
        },
        Kind::Port => bail!("{} is a port; ports are not movable", target.slot),
    }
}

fn ring(shape: NodeShape, point: Pt) -> Contour {
    match shape {
        NodeShape::City => nodes::node_ring(SlotKind::FourSlot, point),
        NodeShape::SmallCity => nodes::node_ring(SlotKind::OneSlot, point),
        NodeShape::Town => {
            let raw: Vec<Pt> = TOWN_OFFSETS.iter().map(|o| [point[0] + o[0], point[1] + o[1]]).collect();
            g::ring_pts(&raw)
        }
    }
}

fn corners_of(lo: Pt, hi: Pt) -> BTreeSet<Pt> {
    [[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]].into_iter().collect()
}

fn p64(p: [f32; 2]) -> [f64; 2] {
    [f64::from(p[0]), f64::from(p[1])]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    /// the map has no node at the old position: none is made at the new one either
    Absent,
    Moved { old_pieces: usize, touched: usize },
}

/// Carves the node of `shape` around `to` out of the land (type 0) of path `single`; `touched` collects the cells
/// that changed. The node must fit completely.
pub(super) fn carve(m: &mut Model, touched: &mut BTreeMap<usize, bool>, single: u16, shape: NodeShape, to: [f32; 2], fr: &Frame) -> Result<()> {
    let (cols, rows, cs) = (m.cols, m.rows, m.cs);
    let fixed = |p: [f32; 2]| fr.fixed(fr.rel(p));
    let as_f = |pt: Pt| [f64::from(pt[0]), f64::from(pt[1])];
    let rect_of = |c: usize, r: usize| ([c as i32 * cs, r as i32 * cs], [(c as i32 + 1) * cs, (r as i32 + 1) * cs]);
    let to_pt = fixed(to);
    let ring_new = ring(shape, to_pt);
    let node_q: Shapes = vec![vec![g::node_ring(&ring_new, cs)]];
    let (nl, nh) = g::shapes_bbox(&node_q).ok_or_else(|| anyhow!("empty node"))?;
    ensure!(
        nl[0] >= 0 && nl[1] >= 0 && nh[0] <= cols as i32 * cs && nh[1] <= rows as i32 * cs,
        "the node at {to:?} leaves the pathfinding grid"
    );
    let (nc_lo, nc_hi) = ((nl[0] / cs) as usize, ((nh[0] / cs) as usize).min(cols - 1));
    let (nr_lo, nr_hi) = ((nl[1] / cs) as usize, ((nh[1] / cs) as usize).min(rows - 1));
    let mut land_parts: Vec<Shapes> = Vec::new();
    for r in nr_lo..=nr_hi {
        for c in nc_lo..=nc_hi {
            let idx = r * cols + c;
            let (lo, hi) = rect_of(c, r);
            match m.cells[idx].run {
                Some((a, pid)) if m.cells[idx].pieces.is_empty() => {
                    if a & 15 == 0 && pid == single {
                        land_parts.push(vec![g::rect_shape(lo, hi)]);
                    }
                }
                _ => {
                    for p in &m.cells[idx].pieces {
                        if p.ptype() == 0 && p.pid == single {
                            land_parts.push(vec![vec![p.pts.clone()]]);
                        }
                    }
                }
            }
        }
    }
    let refs: Vec<&Shapes> = land_parts.iter().collect();
    let land_new = g::union_many(&refs);
    let node = g::intersection(&node_q, &land_new);
    ensure!(
        g::contains_point(&node, as_f(to_pt)),
        "the settlement point {to:?} lies outside the land of its region in the pathfinding grid"
    );
    let (full, got) = (g::area(&node_q), g::area(&node));
    ensure!(
        full - got <= CLIP_TOLERANCE,
        "the node at {to:?} does not fit: {:.4} of {full:.3} u² would be clipped by a border, coast, road or another settlement; choose a point further inland",
        full - got
    );
    for r in nr_lo..=nr_hi {
        for c in nc_lo..=nc_hi {
            let idx = r * cols + c;
            let (lo, hi) = rect_of(c, r);
            let node_clip = g::clip_rect(&node, lo, hi);
            if g::area(&node_clip) < 1e-9 {
                continue;
            }
            if m.cells[idx].pieces.is_empty() {
                match m.cells[idx].run {
                    Some((a, pid)) if a & 15 == 0 && pid == single => {
                        m.cells[idx].pieces.push(MPiece {
                            a,
                            pid,
                            gbit: 0,
                            pts: vec![[hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]], [hi[0], lo[1]]],
                            src: None,
                        });
                        m.cells[idx].run = None;
                    }
                    _ => continue,
                }
            }
            let mut out: Vec<MPiece> = Vec::new();
            let mut changed = false;
            for p in std::mem::take(&mut m.cells[idx].pieces) {
                if p.ptype() != 0 || p.pid != single {
                    out.push(p);
                    continue;
                }
                let poly: Shapes = vec![vec![p.pts.clone()]];
                let inside = g::intersection(&poly, &node_clip);
                if inside.is_empty() {
                    out.push(p);
                    continue;
                }
                changed = true;
                for q in inside {
                    for part in g::split_holes(&q) {
                        out.push(MPiece { a: 7, pid: single, gbit: 0, pts: g::ring_pts(&part[0]), src: None });
                    }
                }
                for q in g::difference(&poly, &node_clip) {
                    for part in g::split_holes(&q) {
                        out.push(MPiece { a: 0, pid: single, gbit: p.gbit, pts: g::ring_pts(&part[0]), src: None });
                    }
                }
            }
            if changed {
                out.sort_by_key(|p| *p.pts.iter().min().unwrap_or(&[0, 0]));
                touched.insert(idx, false);
            }
            m.cells[idx].pieces = out;
        }
    }

    Ok(())
}

/// A full-cell land piece is a run cell again.
pub(super) fn normalize_runs(m: &mut Model, touched: &mut BTreeMap<usize, bool>) {
    let (cols, cs) = (m.cols, m.cs);
    let rect_of = |c: usize, r: usize| ([c as i32 * cs, r as i32 * cs], [(c as i32 + 1) * cs, (r as i32 + 1) * cs]);
    let touched_idx: Vec<usize> = touched.keys().copied().collect();
    for idx in touched_idx {
        let (c, r) = (idx % cols, idx / cols);
        let (lo, hi) = rect_of(c, r);
        let cell = &mut m.cells[idx];
        if cell.pieces.len() == 1
            && cell.pieces[0].ptype() == 0
            && cell.pieces[0].pts.len() == 4
            && cell.pieces[0].pts.iter().copied().collect::<BTreeSet<Pt>>() == corners_of(lo, hi)
        {
            if let Some(p) = cell.pieces.pop() {
                cell.run = Some((p.a, p.pid));
                touched.insert(idx, true);
            }
        }
    }
}

/// Moves the node of `region` from `from` to `to` in `grid` (see the module docs).
pub(super) fn relocate(grid: &mut Grid, region: usize, shape: NodeShape, from: [f32; 2], to: [f32; 2]) -> Result<Outcome> {
    let single = grid
        .single_path(region as i16)
        .ok_or_else(|| anyhow!("pathfinding area {} has no single path for region #{region}", grid.area))?;
    let old = Model::from_grid(grid)?;
    let (cols, rows, cs) = (old.cols, old.rows, old.cs);
    let fr = Frame { origin: [f64::from(old.origin[0]) / FIX, f64::from(old.origin[1]) / FIX], cs };
    let fixed = |p: [f32; 2]| fr.fixed(fr.rel(p));
    let cell_of = |pt: Pt| -> (usize, usize) {
        (((pt[0].max(0) / cs) as usize).min(cols - 1), ((pt[1].max(0) / cs) as usize).min(rows - 1))
    };
    let as_f = |pt: Pt| [f64::from(pt[0]), f64::from(pt[1])];

    // ---- is there a node at the old position?
    let from_pt = fixed(from);
    let (fc, fr_) = cell_of(from_pt);
    let mut present = false;
    for r in fr_.saturating_sub(1)..=(fr_ + 1).min(rows - 1) {
        for c in fc.saturating_sub(1)..=(fc + 1).min(cols - 1) {
            for p in &old.cells[r * cols + c].pieces {
                if p.ptype() == 7 && p.pid == single && g::contains_point(&vec![vec![p.pts.clone()]], as_f(from_pt)) {
                    present = true;
                }
            }
        }
    }
    if !present {
        return Ok(Outcome::Absent);
    }

    // ---- the pieces of the old node: type 7 of the region's path, wholly inside the grown node ring
    let ring_old = ring(shape, from_pt);
    let grown: Contour = ring_old
        .iter()
        .map(|p| {
            [
                from_pt[0] + (f64::from(p[0] - from_pt[0]) * GROW).round() as i32,
                from_pt[1] + (f64::from(p[1] - from_pt[1]) * GROW).round() as i32,
            ]
        })
        .collect();
    let grown_shapes: Shapes = vec![vec![grown]];
    let (glo, ghi) = g::shapes_bbox(&grown_shapes).ok_or_else(|| anyhow!("empty node ring"))?;
    let (c_lo, c_hi) = (((glo[0] / cs).max(0) as usize), ((ghi[0] / cs).max(0) as usize).min(cols - 1));
    let (r_lo, r_hi) = (((glo[1] / cs).max(0) as usize), ((ghi[1] / cs).max(0) as usize).min(rows - 1));
    let mut node_pieces: BTreeSet<(usize, usize)> = BTreeSet::new();
    for r in r_lo..=r_hi {
        for c in c_lo..=c_hi {
            let idx = r * cols + c;
            for (k, p) in old.cells[idx].pieces.iter().enumerate() {
                if p.ptype() != 7 || p.pid != single {
                    continue;
                }
                let poly: Shapes = vec![vec![p.pts.clone()]];
                let whole = g::area(&poly);
                if whole > 0.0 && g::area(&g::intersection(&poly, &grown_shapes)) >= whole * (1.0 - 1e-6) {
                    node_pieces.insert((idx, k));
                }
            }
        }
    }
    let parts: Vec<Shapes> = node_pieces
        .iter()
        .map(|&(idx, k)| vec![vec![old.cells[idx].pieces[k].pts.clone()]])
        .collect();
    let refs: Vec<&Shapes> = parts.iter().collect();
    let node_union = g::union_many(&refs);
    ensure!(
        g::contains_point(&node_union, as_f(from_pt)),
        "{}: the node pieces around {from:?} do not hold the settlement point",
        region
    );
    let ring_area = g::area(&vec![vec![ring_old.clone()]]);
    ensure!(
        g::area(&node_union) <= ring_area * 1.1 + 1e-6,
        "the node at {from:?} is larger than a settlement node ({:.3} of {ring_area:.3} u²)",
        g::area(&node_union)
    );

    let mut m = old.clone();
    let mut touched: BTreeMap<usize, bool> = BTreeMap::new();

    // ---- 1. the old node becomes land again, merged with the land pieces of its cells
    let cells_hit: BTreeSet<usize> = node_pieces.iter().map(|&(idx, _)| idx).collect();
    for &idx in &cells_hit {
        let mut land: Vec<Shapes> = Vec::new();
        let mut keep: Vec<MPiece> = Vec::new();
        let mut gbit: Option<u8> = None;
        for p in std::mem::take(&mut m.cells[idx].pieces) {
            let is_node = p.src.is_some_and(|k| node_pieces.contains(&(idx, k)));
            if is_node || (p.ptype() == 0 && p.pid == single) {
                if !is_node && gbit.is_none() {
                    gbit = Some(p.gbit);
                }
                land.push(vec![vec![p.pts.clone()]]);
            } else {
                keep.push(p);
            }
        }
        let refs: Vec<&Shapes> = land.iter().collect();
        for shape in g::union_many(&refs) {
            for part in g::split_holes(&shape) {
                keep.push(MPiece { a: 0, pid: single, gbit: gbit.unwrap_or(0), pts: g::ring_pts(&part[0]), src: None });
            }
        }
        keep.sort_by_key(|p| *p.pts.iter().min().unwrap_or(&[0, 0]));
        m.cells[idx].pieces = keep;
        touched.insert(idx, false);
    }

    // ---- 2. the new node, carved out of the region's land
    carve(&mut m, &mut touched, single, shape, to, &fr)?;
    normalize_runs(&mut m, &mut touched);

    // ---- 3. hygiene and rule words
    let (hygiene, _) = finish::hygiene_and_words(&mut m, &old, &touched);
    if let Some(failure) = hygiene.failures.into_iter().next() {
        bail!("settlement node move: {failure}");
    }
    m.write_into(grid)?;
    Ok(Outcome::Moved { old_pieces: node_pieces.len(), touched: touched.len() })
}

/// Moves the pathfinding node of every planned capital / town in `root` (`pathfinding.esf`). `rd` is the geometry
/// (borders are never changed by a move). Every theatre grid that changed is validated against its state before:
/// any new rule violation aborts. Returns report lines.
pub(super) fn apply(root: &mut Node, rd: &RegionData, plans: &[Plan]) -> Result<Vec<String>> {
    let mut grids: BTreeMap<usize, (Grid, BTreeSet<String>)> = BTreeMap::new();
    let (mut moved, mut absent, mut pieces, mut cells) = (0usize, 0usize, 0usize, 0usize);
    for plan in plans {
        let shape = shape_of(&plan.target)?;
        let (from, to) = (plan.target.position, plan.to);
        let area = Grid::area_at(root, p64(from))?
            .ok_or_else(|| anyhow!("{} lies outside every pathfinding area", plan.target.slot))?;
        ensure!(
            Grid::area_at(root, p64(to))? == Some(area),
            "{} would leave its pathfinding area",
            plan.target.slot
        );
        if !grids.contains_key(&area) {
            let grid = Grid::parse_area(root, area)?;
            let baseline = stages::problem_identities(&grid, rd);
            grids.insert(area, (grid, baseline));
        }
        let (grid, _) = grids.get_mut(&area).ok_or_else(|| anyhow!("pathfinding area {area} vanished"))?;
        match relocate(grid, plan.target.region_index, shape, from, to)
            .with_context(|| format!("Pathfinding node of {}", plan.target.slot))?
        {
            Outcome::Absent => absent += 1,
            Outcome::Moved { old_pieces, touched } => {
                moved += 1;
                pieces += old_pieces;
                cells += touched;
            }
        }
    }
    for (area, (grid, baseline)) in &grids {
        let problems = stages::new_problems(baseline, grid, rd);
        ensure!(
            problems.is_empty(),
            "pathfinding.esf area {area}: the moved settlement nodes break {} rule(s), e.g. {}",
            problems.len(),
            problems.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
        );
        grid.store(root)?;
    }
    Ok(vec![format!(
        "pathfinding.esf: {moved} settlement node(s) relocated ({pieces} old node piece(s) restored to land, {cells} cell(s) rebuilt), {absent} settlement(s) have no node in the stock map and got none"
    )])
}
