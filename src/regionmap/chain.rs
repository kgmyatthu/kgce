//! The format-9 geometry chain on a parsed `regions.esf`: the fixed theatre bounds, clipping at the theatre
//! rectangles and the activation options (`areas`, `merge`, `carve_from`). Both the build pipeline and the
//! editor's preview (the Ownership map, the CLI validation) run exactly these functions, so what the user sees is
//! what gets built.
//!
//! Nothing here touches files; the pipeline stores the result and brings the dependent files (pathfinding, sea
//! grids, canvases, start position) in line, see `pipeline`.

use super::{
    geometry::Rect,
    clip::ClipReport,
    regions::{NONE, RegionData, TheatreChange, align_theatre_rect},
    theatres::FIXED_BOUNDS,
};
use crate::model::{AreaSelection, RegionActivation};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeSet;

/// Margin (world units) by which sea polygons are clipped outside a theatre rectangle (the vanilla convention).
pub const CLIP_MARGIN: f32 = 5.0;

/// A world point in open sea inside `rect`, the pixel colour that fills the new canvas margin. India has a fixed
/// reference point (the experiment's); other theatres use the sea area with the deepest inside point (its pole of
/// inaccessibility) that lies in the rectangle.
pub fn reference_sea_point(rd: &RegionData, stem: &str, rect: Rect) -> Result<[f32; 2]> {
    if let Some(point) = crate::terrain::default_sea_point(stem) {
        return Ok(point);
    }
    let mut best: Option<(f64, [f32; 2])> = None;
    for (ri, region) in rd.regions.iter().enumerate().filter(|(_, r)| r.kind == "sea") {
        for (ai, area) in region.areas.iter().enumerate().filter(|(_, a)| a.passable) {
            let _ = area;
            let p = rd.pole_of_inaccessibility(ri, ai);
            let inside = p[0] > f64::from(rect[0][0])
                && p[0] < f64::from(rect[1][0])
                && p[1] > f64::from(rect[0][1])
                && p[1] < f64::from(rect[1][1]);
            if !inside || rd.area_at(ri, p) != Some(ai) {
                continue;
            }
            let depth = rd.distance_to_region(ri, p);
            if best.is_none_or(|(d, _)| depth > d) {
                best = Some((depth, [p[0] as f32, p[1] as f32]));
            }
        }
    }
    best.map(|(_, p)| p)
        .with_context(|| format!("theatre {stem}: no open-sea point inside {rect:?} to fill the new canvas margin"))
}

/// The activation changes the polygon structure of the map (more than flagging the area of the settlement).
pub fn changes_geometry(activation: &RegionActivation) -> bool {
    activation.merge || activation.carve_from.is_some() || !matches!(activation.areas, AreaSelection::Centre)
}

/// What [`extend_and_clip`] did.
#[derive(Clone, Debug)]
pub struct TheatreStep {
    pub changes: Vec<TheatreChange>,
    pub clip: Option<ClipReport>,
}

/// Applies the fixed bounds before clipping; returns only changed theatres. Reject incompatible source lattices
/// rather than silently changing the fixed policy. Reapplying the policy leaves rectangles and transitions alone.
pub fn extend(rd: &mut RegionData) -> Result<Vec<TheatreChange>> {
    let mut changes = Vec::new();
    for &(theatre, rect) in FIXED_BOUNDS {
        let old = rd.theatre_rect(theatre)?;
        if old == rect {
            continue;
        }
        ensure!(
            align_theatre_rect(old, rect) == rect,
            "The fixed bounds of theatre {theatre} {rect:?} do not align with the original source grid {old:?}"
        );
        changes.push(rd.extend_theatre(theatre, rect)
            .with_context(|| format!("Apply the fixed bounds of theatre {theatre}: {rect:?}"))?);
    }
    Ok(changes)
}

/// Clips the sea and backdrop polygons at the (extended) theatre rectangles + [`CLIP_MARGIN`].
pub fn clip(rd: &mut RegionData) -> Result<ClipReport> {
    rd.clip_at_theatres(CLIP_MARGIN)
        .context("Clip the sea and backdrop polygons at the theatre rectangles")
}

/// [`extend`] then [`clip`]; a theatre already sitting at its fixed bounds is left completely alone.
pub fn extend_and_clip(rd: &mut RegionData) -> Result<TheatreStep> {
    let changes = extend(rd)?;
    let clip = if changes.is_empty() { None } else { Some(clip(rd)?) };
    Ok(TheatreStep { changes, clip })
}

/// Result of [`prepare_activation`].
#[derive(Clone, Debug, Default)]
pub struct ActivationGeometry {
    pub key: String,
    /// index of the region in `regions.esf` (for a carved region: the new last index)
    pub region: usize,
    /// areas of the region that this step flagged passable (indices after a merge)
    pub made_passable: Vec<usize>,
    pub carved: bool,
    pub merged: bool,
    /// human-readable notes for the build report
    pub notes: Vec<String>,
}

/// Groups of `selected` areas of region `ri` that share a border (connectivity codes name the neighbouring area).
fn touching_groups(rd: &RegionData, ri: usize, selected: &[usize]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..selected.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (i, &a) in selected.iter().enumerate() {
        for outline in &rd.regions[ri].areas[a].outlines {
            for run in &outline.runs {
                if run.code == NONE || (run.code >> 16) as usize != ri {
                    continue;
                }
                if let Some(j) = selected.iter().position(|&b| b == (run.code & 0xFFFF) as usize) {
                    let (x, y) = (find(&mut parent, i), find(&mut parent, j));
                    parent[x] = y;
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut roots: Vec<usize> = Vec::new();
    for i in 0..selected.len() {
        let root = find(&mut parent, i);
        match roots.iter().position(|&r| r == root) {
            Some(g) => groups[g].push(selected[i]),
            None => {
                roots.push(root);
                groups.push(vec![selected[i]]);
            }
        }
    }
    groups
}

/// Applies the geometry options of one activation to `rd`, before its settlement is attached:
///
/// 1. `carve_from`: area `Indices([n])` of the source region moves into a new land region `key` (appended last);
/// 2. the selected areas (`areas`; the new region's only area after a carve) are determined;
/// 3. `merge`: the selected areas that touch each other are united into one area (areas touching nothing, e.g.
///    offshore islets, are neither united nor made passable and keep their flags);
/// 4. every selected area still impassable becomes passable, with the component id of a passable land neighbour it
///    borders or a fresh one ([`RegionData::component_for`]).
///
/// An activation with the default options (`Centre`, no merge, no carve) changes nothing: its settlement area must
/// already be passable.
pub fn prepare_activation(rd: &mut RegionData, activation: &RegionActivation) -> Result<ActivationGeometry> {
    activation.validate_options()?;
    let mut out = ActivationGeometry { key: activation.key.clone(), ..Default::default() };
    let ri = match &activation.carve_from {
        Some(source) => {
            let AreaSelection::Indices(list) = &activation.areas else {
                bail!("{}: carving needs explicit area indices", activation.key);
            };
            ensure!(
                list.len() == 1,
                "{}: a carved region is made of exactly one area of {source} (got {})",
                activation.key,
                list.len()
            );
            let source_index = rd.index(source).with_context(|| format!("{}: carve source region", activation.key))?;
            ensure!(rd.index(&activation.key).is_err(), "{}: a region with this key exists already", activation.key);
            let count = rd.regions[source_index].areas.len();
            ensure!(
                list[0] < count,
                "{}: area {} does not exist, {source} has {count} areas",
                activation.key,
                list[0]
            );
            let index = rd.carve_area(source_index, list[0], &activation.key)?;
            out.carved = true;
            out.notes.push(format!("area {} of {source} became the new region {} (index {index})", list[0], activation.key));
            index
        }
        None => rd.index(&activation.key)?,
    };
    out.region = ri;
    ensure!(rd.regions[ri].kind == "land", "{}: only land regions can be activated", activation.key);
    let count = rd.regions[ri].areas.len();
    let mut selected: Vec<usize> = match (&activation.carve_from, &activation.areas) {
        (Some(_), _) => vec![0],
        (None, AreaSelection::Centre) => Vec::new(),
        (None, AreaSelection::All) => (0..count).collect(),
        (None, AreaSelection::Indices(list)) => {
            if let Some(bad) = list.iter().find(|&&a| a >= count) {
                bail!("{}: area {bad} does not exist, the region has {count} areas", activation.key);
            }
            list.clone()
        }
    };

    if activation.merge {
        let mut groups: Vec<Vec<usize>> =
            touching_groups(rd, ri, &selected).into_iter().filter(|g| g.len() >= 2).collect();
        ensure!(
            !groups.is_empty(),
            "{}: merge needs at least two selected areas that share a border",
            activation.key
        );
        ensure!(
            groups.len() == 1,
            "{}: the selected areas form {} separate groups ({}); select one connected group to merge",
            activation.key,
            groups.len(),
            groups.iter().map(|g| format!("{g:?}")).collect::<Vec<_>>().join(", ")
        );
        let group = groups.remove(0);
        let left_out: Vec<usize> = selected.iter().copied().filter(|a| !group.contains(a)).collect();
        if !left_out.is_empty() {
            out.notes.push(format!("areas {left_out:?} touch no other selected area: left as they are"));
        }
        let united = rd.merge_areas(ri, &group)?;
        out.merged = true;
        out.notes.push(format!("areas {group:?} united into area {united}"));
        selected = vec![united];
    }

    for area in selected {
        if rd.regions[ri].areas[area].passable {
            continue;
        }
        let component = rd.component_for(ri, area);
        let island = u32::from(rd.regions[ri].areas[area].island_id);
        rd.set_passable(ri, area, component, 65535, island)?;
        out.made_passable.push(area);
        out.notes.push(format!("area {area} passable (component {component})"));
    }
    Ok(out)
}

/// Keys of regions whose area count or area component ids differ between `before` and `after` (new regions included),
/// plus `always`.
/// These need their start-position HLCI entries rebuilt.
pub fn regions_with_changed_areas(before: &RegionData, after: &RegionData, always: &[String]) -> Vec<String> {
    let mut keys: BTreeSet<String> = always.iter().cloned().collect();
    for region in &after.regions {
        let components = |r: &super::regions::Region| r.areas.iter().map(|a| a.adjoining_id).collect::<Vec<_>>();
        match before.regions.iter().find(|r| r.key == region.key) {
            Some(old) if old.areas.len() == region.areas.len() && components(old) == components(region) => {}
            _ => {
                keys.insert(region.key.clone());
            }
        }
    }
    keys.into_iter().collect()
}
