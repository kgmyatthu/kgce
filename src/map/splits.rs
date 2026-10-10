//! Editor-side helpers for region splits: replaying the project's splits on the parsed `regions.esf`, snapping the
//! user's clicks to ring vertices, validating a drawn cut, previewing both halves, choosing a lookup colour and
//! deriving the editor's rule set. Everything here is pure (no UI); the Ownership map drives it.

use super::TheatreRect;
use crate::{
    model::{Project, RegionActivation, RegionSplit, Rules},
    regionmap::{
        chain,
        geometry::{P, point_in_ring},
        regions::{RegionData, SettlementSpec, Split, settlement_center},
    },
};
use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

type Triangle = [[f32; 2]; 3];

/// `regionmap::regions::Split` with the meaning of the project's [`RegionSplit`].
pub fn engine_split(split: &RegionSplit) -> Split {
    Split {
        key: split.source.clone(),
        area: split.area,
        a: split.a,
        b: split.b,
        interior: split.interior.clone(),
        new_key: split.key.clone(),
        settlement: SettlementSpec {
            name: split.settlement.clone(),
            center: split.center,
        },
    }
}

/// Replays the whole project on the shipped geometry in the order the build uses: fixed theatre bounds and clipping,
/// then every activation in project order (its geometry options, then its settlement, so that the component id of
/// a later activation can follow the settled land of an earlier one), then every split. An activated wilderness
/// region carries a settlement cloned from its donor, so a later split may cut it.
pub fn apply_all(data: &mut RegionData, project: &Project) -> Result<()> {
    chain::extend_for_project(data, project)?;
    chain::clip(data)?;
    for (index, activation) in project.activations.iter().enumerate() {
        activate_with_geometry(data, activation)
            .with_context(|| format!("Activation #{} {} (donor {})", index + 1, activation.key, activation.donor))?;
    }
    for (index, split) in project.splits.iter().enumerate() {
        data.split(&engine_split(split))
            .with_context(|| format!("Split #{} {} (from {})", index + 1, split.key, split.source))?;
    }
    crate::settlement::preview(data, &project.settlement_moves)?;
    Ok(())
}

/// [`apply_all`] on a copy of `base`, plus the theatre rectangles the project ends up with.
pub fn replay(base: &RegionData, project: &Project) -> Result<(RegionData, Vec<TheatreRect>)> {
    let mut data = base.clone();
    apply_all(&mut data, project)?;
    let rects = rects_of(&data);
    Ok((data, rects))
}

/// The theatre rectangles of `data`.
pub fn rects_of(data: &RegionData) -> Vec<TheatreRect> {
    data.theatre_rects()
        .into_iter()
        .map(|(key, [lo, hi])| TheatreRect { key, lo, hi })
        .collect()
}


/// One activation on `data`: its geometry options (areas, merge, carve), then its settlement. The build's geometry
/// stage runs exactly this per activation so that the editor preview and the generated map agree.
pub fn activate_with_geometry(
    data: &mut RegionData,
    activation: &RegionActivation,
) -> Result<Option<chain::ActivationGeometry>> {
    let done = if chain::changes_geometry(activation) {
        Some(chain::prepare_activation(data, activation)?)
    } else {
        None
    };
    activate(data, activation)?;
    Ok(done)
}

/// Gives the wilderness region its settlement: the donor's settlement slots moved to the activation's centre.
fn activate(data: &mut RegionData, activation: &RegionActivation) -> Result<()> {
    let region = data.index(&activation.key)?;
    let donor = data.index(&activation.donor)?;
    ensure!(
        data.regions[region].settlement.is_none(),
        "{} already has a settlement",
        activation.key
    );
    data.attach_settlement(
        region,
        donor,
        &SettlementSpec {
            name: activation.settlement.clone(),
            center: activation.center,
        },
    )
}

/// The editor's rule set: the shipped rules plus the entries derived for the project's split regions. Ownership
/// overrides stay out of it (the editor resolves them through `workflow::effective_owner` on top of these baselines).
pub fn project_rules(rules_root: &Path, project: &Project) -> Result<Rules> {
    let mut rules = Rules::load(rules_root)?;
    let mut probe = project.clone();
    probe.owners.clear();
    rules.apply_project(&probe)?;
    Ok(rules)
}

/// Lower-case ASCII slug usable inside region and settlement keys.
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_string()
}

/// Region and settlement keys end up in `:`-separated slot keys and file names.
pub fn valid_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

fn used_colours(rules: &Rules) -> Vec<[u8; 3]> {
    rules
        .regions
        .as_object()
        .into_iter()
        .flat_map(|regions| regions.values())
        .filter_map(|rule| {
            let rgb = rule["lookup_rgb"].as_array()?;
            Some([
                rgb.first()?.as_u64()? as u8,
                rgb.get(1)?.as_u64()? as u8,
                rgb.get(2)?.as_u64()? as u8,
            ])
        })
        .collect()
}

fn hsv(h: f64, s: f64, v: f64) -> [u8; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [(r * 255.0).round() as u8, (g * 255.0).round() as u8, (b * 255.0).round() as u8]
}

/// A lookup colour that no region of `rules` (which already lists the project's split regions) uses, kept visually
/// apart from the used ones where the palette allows.
pub fn pick_colour(rules: &Rules) -> [u8; 3] {
    let used = used_colours(rules);
    let distance = |a: [u8; 3], b: [u8; 3]| {
        (0..3)
            .map(|k| (i32::from(a[k]) - i32::from(b[k])).pow(2))
            .sum::<i32>()
    };
    for min in [40 * 40, 24 * 24, 12 * 12, 1] {
        for i in 0..4096u32 {
            let h = (f64::from(i) * 0.618_033_988_75).fract();
            let s = [0.9, 0.65, 1.0][(i / 5 % 3) as usize];
            let v = [1.0, 0.8, 0.6][(i / 15 % 3) as usize];
            let candidate = hsv(h, s, v);
            if candidate.iter().all(|c| *c >= 8) && used.iter().all(|u| distance(*u, candidate) >= min) {
                return candidate;
            }
        }
    }
    [1, 1, 1]
}

/// Splits that must go when split `index` is removed: itself and, transitively, every later split cut from a
/// region that disappears (`source` equal to a removed key). Ascending order.
pub fn dependent_splits(splits: &[RegionSplit], index: usize) -> Vec<usize> {
    let mut removed = vec![index];
    let mut keys = vec![splits[index].key.clone()];
    for (i, split) in splits.iter().enumerate().skip(index + 1) {
        if keys.contains(&split.source) {
            removed.push(i);
            keys.push(split.key.clone());
        }
    }
    removed
}

// ---------------------------------------------------------------------------------------------- activation

/// A wilderness region that can become a campaign region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Activatable {
    /// theatre (regions.esf theatre key, equal to the rules' `theatre`) whose rectangle holds the whole region
    pub theatre: String,
    /// number of areas of the region
    pub areas: usize,
    /// first passable area (the one the settlement normally sits in); none while the region is still backdrop
    /// (its `areas` option then decides which areas become passable)
    pub passable_area: Option<usize>,
}

/// `pt` lies inside area `ai` of region `ri`: an outer ring contains it and no hole does.
fn in_area(data: &RegionData, ri: usize, ai: usize, pt: [f32; 2]) -> bool {
    let mut inside = false;
    for outline in &data.regions[ri].areas[ai].outlines {
        let ring: Vec<P> = outline.ring.iter().map(|&id| data.v(id)).collect();
        if point_in_ring(wide(pt), &ring) {
            if outline.outer {
                inside = true;
            } else {
                return false;
            }
        }
    }
    inside
}

/// The passable area of region `ri` that contains `pt`.
pub fn passable_area_at(data: &RegionData, ri: usize, pt: [f32; 2]) -> Option<usize> {
    (0..data.regions[ri].areas.len()).find(|&ai| data.regions[ri].areas[ai].passable && in_area(data, ri, ai, pt))
}

/// Whether `key` is an ETW land polygon without settlement or campaign rules whose every outline vertex lies in one
/// theatre rectangle, and where it can be activated.
pub fn activatable(data: &RegionData, theatres: &[TheatreRect], rules: &Rules, key: &str) -> Result<Activatable> {
    let ri = data.index(key)?;
    let region = &data.regions[ri];
    ensure!(region.kind == "land", "{key} is not a land region");
    ensure!(region.settlement.is_none(), "{key} already has a settlement");
    ensure!(rules.regions.get(key).is_none(), "{key} is already a campaign region");
    let mut theatre: Option<&str> = None;
    for outline in region.areas.iter().flat_map(|area| &area.outlines) {
        for &id in &outline.ring {
            let at = data.vertices[id as usize];
            let mut holding = theatres
                .iter()
                .filter(|t| t.lo[0] <= at[0] && at[0] <= t.hi[0] && t.lo[1] <= at[1] && at[1] <= t.hi[1]);
            let (Some(found), None) = (holding.next(), holding.next()) else {
                bail!("{key} is not inside exactly one theatre (a vertex at {:.1}, {:.1} is not)", at[0], at[1]);
            };
            match theatre {
                None => theatre = Some(&found.key),
                Some(known) => ensure!(known == found.key, "{key} spans two theatres"),
            }
        }
    }
    let theatre = theatre.with_context(|| format!("{key} has no outline"))?;
    let passable_area = region.areas.iter().position(|area| area.passable);
    Ok(Activatable {
        theatre: theatre.to_string(),
        areas: region.areas.len(),
        passable_area,
    })
}

/// `key` is a playable region of `theatre` that exists before the project's splits and has a settlement to copy.
fn usable_donor(data: &RegionData, rules: &Rules, splits: &[RegionSplit], key: &str, theatre: &str) -> bool {
    rules.regions.get(key).is_some_and(|rule| rule["theatre"].as_str() == Some(theatre))
        && !splits.iter().any(|split| split.key == key)
        && data.index(key).is_ok_and(|ri| data.regions[ri].settlement.is_some())
}

/// Playable regions of `theatre` next to `key`, longest shared border first.
pub fn donor_candidates(
    data: &RegionData,
    rules: &Rules,
    splits: &[RegionSplit],
    key: &str,
    theatre: &str,
) -> Vec<String> {
    let Ok(ri) = data.index(key) else {
        return Vec::new();
    };
    let lengths = data.border_lengths(ri);
    let mut found: Vec<(String, f64)> = data
        .adjacent(ri)
        .into_iter()
        .map(|neighbour| (data.regions[neighbour].key.clone(), lengths.get(&neighbour).copied().unwrap_or(0.0)))
        .filter(|(neighbour, _)| usable_donor(data, rules, splits, neighbour, theatre))
        .collect();
    found.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    found.into_iter().map(|(neighbour, _)| neighbour).collect()
}

/// Every acceptable donor of `theatre`: the neighbours of `key` first, then the other playable regions by key.
pub fn donor_choices(
    data: &RegionData,
    rules: &Rules,
    splits: &[RegionSplit],
    key: &str,
    theatre: &str,
) -> Vec<String> {
    let mut choices = donor_candidates(data, rules, splits, key, theatre);
    let rest: Vec<String> = rules
        .regions
        .as_object()
        .into_iter()
        .flat_map(|regions| regions.keys())
        .filter(|candidate| !choices.contains(candidate) && usable_donor(data, rules, splits, candidate, theatre))
        .cloned()
        .collect();
    choices.extend(rest);
    choices
}

/// 'wilderness_great_plains' -> 'Great Plains'.
pub fn default_name(key: &str) -> String {
    key.strip_prefix("wilderness_")
        .unwrap_or(key)
        .split(['_', '-'])
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().chain(chars).collect::<String>())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Everything `Apply` must satisfy before an activation enters the project. `rules` holds the project's current
/// regions (without this activation), `data` its current geometry.
pub fn validate_activation(
    activation: &RegionActivation,
    rules: &Rules,
    data: &RegionData,
    theatres: &[TheatreRect],
    splits: &[RegionSplit],
) -> Result<Activatable> {
    activation.validate_options()?;
    // geometry options (areas, merge, carve) are tried on a copy: the region that results must be activatable
    let trial;
    let data = if chain::changes_geometry(activation) {
        let mut copy = data.clone();
        chain::prepare_activation(&mut copy, activation)?;
        trial = copy;
        &trial
    } else {
        data
    };
    if activation.carve_from.is_some() {
        ensure!(valid_key(&activation.key), "Region key must use only a-z, 0-9, '_' and '-'");
    }
    let found = activatable(data, theatres, rules, &activation.key)?;
    ensure!(!activation.name.trim().is_empty(), "Name the region");
    ensure!(valid_key(&activation.settlement), "Settlement key must use only a-z, 0-9, '_' and '-'");
    ensure!(!activation.settlement_name.trim().is_empty(), "Name the settlement");
    ensure!(
        usable_donor(data, rules, splits, &activation.donor, &found.theatre),
        "{} is not a playable region of this theatre that can serve as the template",
        activation.donor
    );
    ensure!(
        !used_colours(rules).contains(&activation.colour),
        "The map colour is already used by another region"
    );
    let ri = data.index(&activation.key)?;
    ensure!(
        passable_area_at(data, ri, activation.center).is_some(),
        "The settlement must lie inside a passable area of {}",
        activation.key
    );
    check_settlement_clearance(data, ri, activation.center)?;
    Ok(found)
}

/// Smallest distance a settlement point may have to the outline of its area: the settlement node (type 7 piece of
/// the pathfinding grid) has a circumradius of about 2.07 u and must be surrounded by land, vanilla never has it
/// touching a border.
pub const SETTLEMENT_CLEARANCE: f64 = 2.1;

/// Refuses a settlement point closer than [`SETTLEMENT_CLEARANCE`] to the outline of the region.
pub fn check_settlement_clearance(data: &RegionData, ri: usize, center: [f32; 2]) -> Result<()> {
    let distance = data.distance_to_region(ri, [f64::from(center[0]), f64::from(center[1])]);
    ensure!(
        distance >= SETTLEMENT_CLEARANCE,
        "The settlement point is only {distance:.2} u from the outline of {}; keep at least {SETTLEMENT_CLEARANCE} u so its node fits inside the region",
        data.regions[ri].key
    );
    Ok(())
}

/// What goes when an activation is removed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActivationRemoval {
    /// indices into `project.activations`: the activation itself and, transitively, later ones that use a removed
    /// region as their donor
    pub activations: Vec<usize>,
    /// indices into `project.splits`: every split cut from a removed region, transitively
    pub splits: Vec<usize>,
    /// keys of all regions that disappear
    pub keys: Vec<String>,
}

/// Everything that depends on the activation of `key` (ascending indices); `None` when `key` is not activated.
pub fn removal_of_activation(
    activations: &[RegionActivation],
    splits: &[RegionSplit],
    key: &str,
) -> Option<ActivationRemoval> {
    let first = activations.iter().position(|activation| activation.key == key)?;
    let mut removal = ActivationRemoval {
        activations: vec![first],
        splits: Vec::new(),
        keys: vec![key.to_string()],
    };
    for (index, activation) in activations.iter().enumerate().skip(first + 1) {
        if removal.keys.contains(&activation.donor) {
            removal.activations.push(index);
            removal.keys.push(activation.key.clone());
        }
    }
    for (index, split) in splits.iter().enumerate() {
        if removal.keys.contains(&split.source) {
            removal.splits.push(index);
            removal.keys.push(split.key.clone());
        }
    }
    Some(removal)
}

impl ActivationRemoval {
    /// Drops the activations and splits and every owner override of the vanishing regions.
    pub fn remove_from(&self, project: &mut Project) {
        let mut position = 0;
        project.activations.retain(|_| {
            position += 1;
            !self.activations.contains(&(position - 1))
        });
        position = 0;
        project.splits.retain(|_| {
            position += 1;
            !self.splits.contains(&(position - 1))
        });
        for key in &self.keys {
            project.owners.remove(key);
        }
        project.settlement_moves.retain(|movement| !self.keys.contains(&movement.region));
    }
}

// ---------------------------------------------------------------------------------------------- snapping

/// A ring vertex the user's click snapped to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snap {
    pub area: usize,
    /// position in the area's outer ring
    pub pos: usize,
    pub point: [f32; 2],
}

/// Areas of region `ri` that `RegionData::split` accepts: exactly one outline, an outer ring.
pub fn cuttable_areas(data: &RegionData, ri: usize) -> Vec<usize> {
    data.regions[ri]
        .areas
        .iter()
        .enumerate()
        .filter(|(_, area)| area.outlines.len() == 1 && area.outlines[0].outer)
        .map(|(index, _)| index)
        .collect()
}

/// Nearest outer-ring vertex of the cuttable areas of region `ri` (restricted to `only_area` when given) and its
/// distance to `pt`.
pub fn nearest_vertex(
    data: &RegionData,
    ri: usize,
    only_area: Option<usize>,
    pt: [f32; 2],
) -> Option<(Snap, f64)> {
    let mut best: Option<(Snap, f64)> = None;
    for area in cuttable_areas(data, ri) {
        if only_area.is_some_and(|only| only != area) {
            continue;
        }
        for (pos, &id) in data.regions[ri].areas[area].outlines[0].ring.iter().enumerate() {
            let v = data.vertices[id as usize];
            let d = f64::from(v[0] - pt[0]).hypot(f64::from(v[1] - pt[1]));
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((Snap { area, pos, point: v }, d));
            }
        }
    }
    best
}

/// `pt` lies inside the ring.
pub fn inside_ring(ring: &[[f32; 2]], pt: [f32; 2]) -> bool {
    let ring: Vec<P> = ring.iter().copied().map(wide).collect();
    point_in_ring(wide(pt), &ring)
}

/// Outer ring of `area` of region `ri` as points.
pub fn ring_points(data: &RegionData, ri: usize, area: usize) -> Vec<[f32; 2]> {
    data.regions[ri].areas[area].outlines[0]
        .ring
        .iter()
        .map(|&id| data.vertices[id as usize])
        .collect()
}

// ---------------------------------------------------------------------------------------------- cut validation

fn orient(a: P, b: P, c: P) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn on_segment(a: P, b: P, p: P) -> bool {
    p[0] >= a[0].min(b[0]) && p[0] <= a[0].max(b[0]) && p[1] >= a[1].min(b[1]) && p[1] <= a[1].max(b[1])
}

/// Closed segments `p1p2` and `q1q2` share at least one point.
fn segments_touch(p1: P, p2: P, q1: P, q2: P) -> bool {
    let (d1, d2, d3, d4) = (orient(q1, q2, p1), orient(q1, q2, p2), orient(p1, p2, q1), orient(p1, p2, q2));
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    (d1 == 0.0 && on_segment(q1, q2, p1))
        || (d2 == 0.0 && on_segment(q1, q2, p2))
        || (d3 == 0.0 && on_segment(p1, p2, q1))
        || (d4 == 0.0 && on_segment(p1, p2, q2))
}

fn wide(p: [f32; 2]) -> P {
    [f64::from(p[0]), f64::from(p[1])]
}

/// Checks that the cut `ring[a] -> clicks (from a towards b) -> ring[b]` stays inside the ring and neither touches
/// the border except at its two end points nor crosses itself.
pub fn check_cut(ring: &[[f32; 2]], a: usize, b: usize, clicks: &[[f32; 2]]) -> Result<()> {
    ensure!(a < ring.len() && b < ring.len(), "Cut end points are not on the region border");
    ensure!(a != b, "Both cut end points are the same border vertex");
    let ring_pts: Vec<P> = ring.iter().copied().map(wide).collect();
    let mut cut: Vec<P> = vec![ring_pts[a]];
    cut.extend(clicks.iter().copied().map(wide));
    cut.push(ring_pts[b]);
    for point in &cut[1..cut.len() - 1] {
        ensure!(point_in_ring(*point, &ring_pts), "A cut point lies outside the region");
    }
    for (i, pair) in cut.windows(2).enumerate() {
        let mid = [(pair[0][0] + pair[1][0]) / 2.0, (pair[0][1] + pair[1][1]) / 2.0];
        ensure!(point_in_ring(mid, &ring_pts), "The cut leaves the region between two of its points");
        // only the cut's two end points may meet the border, each at its own vertex
        let ends: Vec<P> = [(i == 0).then_some(pair[0]), (i + 2 == cut.len()).then_some(pair[1])]
            .into_iter()
            .flatten()
            .collect();
        for k in 0..ring_pts.len() {
            let (e1, e2) = (ring_pts[k], ring_pts[(k + 1) % ring_pts.len()]);
            if !segments_touch(pair[0], pair[1], e1, e2) {
                continue;
            }
            let Some(&shared) = ends.iter().find(|p| **p == e1 || **p == e2) else {
                bail!("The cut touches or crosses the region border; keep it inside and end it on a border vertex");
            };
            let other_cut = if shared == pair[0] { pair[1] } else { pair[0] };
            let other_edge = if shared == e1 { e2 } else { e1 };
            ensure!(orient(shared, other_cut, other_edge) != 0.0, "The cut runs along the region border");
        }
        for (_, later) in cut.windows(2).enumerate().skip(i + 2) {
            ensure!(!segments_touch(pair[0], pair[1], later[0], later[1]), "The cut crosses itself");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------- preview

/// Both halves a split would produce, as outline and fill triangles.
#[derive(Clone, Debug)]
pub struct Trial {
    pub new_ring: Vec<[f32; 2]>,
    pub new_triangles: Vec<Triangle>,
    pub kept_ring: Vec<[f32; 2]>,
    pub kept_triangles: Vec<Triangle>,
    /// settlement centre of the source region before the split (its settlement stays with the kept half)
    pub source_center: Option<[f32; 2]>,
}

impl Trial {
    /// The source region's own settlement would end up in the new half.
    pub fn source_settlement_in_new_half(&self) -> bool {
        self.source_center.is_some_and(|center| self.new_half_contains(center))
    }

    /// `center` lies inside the new half.
    pub fn new_half_contains(&self, center: [f32; 2]) -> bool {
        let ring: Vec<P> = self.new_ring.iter().copied().map(wide).collect();
        point_in_ring(wide(center), &ring)
    }
}

/// Runs the split on a copy of `data` and reads both halves back from the result.
pub fn trial_split(data: &RegionData, split: &RegionSplit) -> Result<Trial> {
    let source_center = data
        .index(&split.source)
        .ok()
        .and_then(|ri| settlement_center(&data.regions[ri]).ok());
    let mut trial = data.clone();
    let mut probe = engine_split(split);
    probe.new_key = "__split_preview__".into();
    let new_index = trial.split(&probe)?;
    let ri = trial.index(&split.source)?;
    let half = |region: usize, area: usize| {
        let a = &trial.regions[region].areas[area];
        let ring = a.outlines[0].ring.iter().map(|&id| trial.vertices[id as usize]).collect();
        let triangles = a
            .faces
            .iter()
            .map(|t| [0, 1, 2].map(|k| trial.vertices[t[k] as usize]))
            .collect();
        (ring, triangles)
    };
    let (new_ring, new_triangles) = half(new_index, 0);
    let (kept_ring, kept_triangles) = half(ri, split.area);
    Ok(Trial {
        new_ring,
        new_triangles,
        kept_ring,
        kept_triangles,
        source_center,
    })
}

/// The project entry for a finished drawing: `clicks` run from end point `a` towards end point `b`. `RegionData::split`
/// takes the interior points in that same order (its ring N is the arc `a..=b` followed by the points reversed; the
/// doc comments saying "from b towards a" describe the opposite and are wrong; pinned by the test below).
pub fn build_split(
    source: &str,
    start: Snap,
    end: Snap,
    clicks: &[[f32; 2]],
) -> RegionSplit {
    RegionSplit {
        source: source.into(),
        area: start.area,
        a: start.pos,
        b: end.pos,
        interior: clicks.to_vec(),
        key: String::new(),
        name: String::new(),
        settlement: String::new(),
        settlement_name: String::new(),
        center: [0.0, 0.0],
        colour: [0, 0, 0],
    }
}

/// Exchanges the halves: forward arc `b..=a` becomes the new region.
pub fn swapped(split: &RegionSplit) -> RegionSplit {
    RegionSplit {
        a: split.b,
        b: split.a,
        interior: split.interior.iter().rev().copied().collect(),
        ..split.clone()
    }
}

/// Cut points of a split, from `a` towards `b`.
pub fn clicks_of(split: &RegionSplit) -> Vec<[f32; 2]> {
    split.interior.clone()
}

/// Everything `Apply split` must satisfy before touching the project.
pub fn validate_fields(split: &RegionSplit, rules: &Rules) -> Result<()> {
    ensure!(valid_key(&split.key), "Region key must use only a-z, 0-9, '_' and '-'");
    ensure!(
        rules.regions.get(&split.key).is_none(),
        "Region key {} already exists",
        split.key
    );
    ensure!(!split.name.trim().is_empty(), "Name the new region");
    ensure!(valid_key(&split.settlement), "Settlement key must use only a-z, 0-9, '_' and '-'");
    ensure!(!split.settlement_name.trim().is_empty(), "Name the settlement");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<[f32; 2]> {
        vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    }

    fn square_region() -> RegionData {
        use crate::regionmap::regions::{Area, NONE, Outline, Region, Run};
        RegionData {
            vertices: square(),
            regions: vec![Region {
                key: "sq".into(),
                kind: "land".into(),
                areas: vec![Area {
                    land_bridge: false,
                    channel_coast: false,
                    passable: true,
                    adjoining_id: 0,
                    faces: vec![[0, 1, 2], [0, 2, 3]],
                    outlines: vec![Outline {
                        outer: true,
                        ring: vec![0, 1, 2, 3],
                        runs: vec![Run { code: NONE, from: 0, to: 3 }],
                    }],
                    impassable_id: 65535,
                    island_id: 0,
                }],
                tail: 0,
                settlement: None,
            }],
            theatres: Vec::new(),
        }
    }

    /// The forward arc a..=b plus the cut is the new region: a trapezoid of area 35 when the interior points are
    /// stored from a towards b.
    #[test]
    fn a_drawn_cut_produces_the_half_it_shows() {
        let data = square_region();
        let start = Snap { area: 0, pos: 0, point: [0.0, 0.0] };
        let end = Snap { area: 0, pos: 1, point: [10.0, 0.0] };
        // drawn from a (0,0) towards b (10,0) through (3,5) and (7,5)
        let mut split = build_split("sq", start, end, &[[3.0, 5.0], [7.0, 5.0]]);
        split.key = "sq_new".into();
        split.settlement = "town".into();
        check_cut(&ring_points(&data, 0, 0), split.a, split.b, &clicks_of(&split)).unwrap();
        let ring: Vec<P> = square().iter().copied().map(wide).collect();
        assert_eq!(crate::regionmap::geometry::polygon_area(&ring), 100.0);
        let mut copy = data.clone();
        let index = copy.split(&engine_split(&split)).unwrap();
        let new_ring: Vec<P> = copy.regions[index].areas[0].outlines[0]
            .ring
            .iter()
            .map(|&id| wide(copy.vertices[id as usize]))
            .collect();
        assert_eq!(crate::regionmap::geometry::polygon_area(&new_ring), 35.0);
    }

    #[test]
    fn slugs_are_key_safe() {
        assert_eq!(slug("Saint-Petersburg  Oblast"), "saint_petersburg_oblast");
        assert_eq!(slug("  Åre!"), "re");
        assert!(valid_key("alsace_new"));
        assert!(!valid_key("a:b"));
    }

    #[test]
    fn a_straight_chord_is_a_valid_cut_and_a_crossing_one_is_not() {
        let ring = square();
        assert!(check_cut(&ring, 0, 2, &[]).is_ok());
        assert!(check_cut(&ring, 0, 2, &[[5.0, 2.0], [5.0, 8.0]]).is_ok());
        assert!(check_cut(&ring, 0, 2, &[[12.0, 5.0]]).is_err());
        assert!(check_cut(&ring, 0, 1, &[[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 4.0]]).is_err());
        assert!(check_cut(&ring, 0, 0, &[]).is_err());
    }

    #[test]
    fn removal_takes_dependent_splits_along() {
        let split = |source: &str, key: &str| RegionSplit {
            source: source.into(),
            key: key.into(),
            ..build_split(source, Snap { area: 0, pos: 0, point: [0.0; 2] }, Snap { area: 0, pos: 1, point: [0.0; 2] }, &[])
        };
        let splits = vec![split("a", "b"), split("c", "d"), split("b", "e"), split("e", "f")];
        assert_eq!(dependent_splits(&splits, 0), vec![0, 2, 3]);
        assert_eq!(dependent_splits(&splits, 1), vec![1]);
    }

    fn rules_with(regions: serde_json::Value) -> Rules {
        let empty = || serde_json::json!({});
        Rules {
            root: Default::default(),
            regions,
            factions: empty(),
            units: empty(),
            buildings: empty(),
            buildings_by_slot_type: empty(),
            slot_types: empty(),
            armies: empty(),
        }
    }

    fn rect(key: &str, lo: [f32; 2], hi: [f32; 2]) -> TheatreRect {
        TheatreRect { key: key.into(), lo, hi }
    }

    #[test]
    fn a_wilderness_square_inside_one_theatre_can_be_activated() {
        let data = square_region();
        let rules = rules_with(serde_json::json!({}));
        let found = activatable(&data, &[rect("1", [-1.0, -1.0], [11.0, 11.0])], &rules, "sq").unwrap();
        assert_eq!(found, Activatable { theatre: "1".into(), areas: 1, passable_area: Some(0) });
        // straddling the rectangle edge, or inside two rectangles at once, is refused
        assert!(activatable(&data, &[rect("1", [-1.0, -1.0], [5.0, 11.0])], &rules, "sq").is_err());
        let both = [rect("1", [-1.0, -1.0], [11.0, 11.0]), rect("2", [-2.0, -2.0], [12.0, 12.0])];
        assert!(activatable(&data, &both, &rules, "sq").is_err());
        // a region with campaign rules is not wilderness
        let ruled = rules_with(serde_json::json!({"sq": {}}));
        assert!(activatable(&data, &[rect("1", [-1.0, -1.0], [11.0, 11.0])], &ruled, "sq").is_err());
        assert_eq!(passable_area_at(&data, 0, [5.0, 5.0]), Some(0));
        assert_eq!(passable_area_at(&data, 0, [15.0, 5.0]), None);
    }

    #[test]
    fn default_names_drop_the_wilderness_prefix() {
        assert_eq!(default_name("wilderness_great_plains"), "Great Plains");
        assert_eq!(default_name("alps"), "Alps");
        assert_eq!(default_name("central_italy"), "Central Italy");
    }

    #[test]
    fn removing_an_activation_takes_its_splits_and_dependent_activations_along() {
        let activation = |key: &str, donor: &str| RegionActivation {
            key: key.into(),
            donor: donor.into(),
            name: key.into(),
            settlement: "s".into(),
            settlement_name: "S".into(),
            center: [0.0; 2],
            colour: [1, 2, 3],
            ..Default::default()
        };
        let split = |source: &str, key: &str| RegionSplit {
            source: source.into(),
            key: key.into(),
            ..build_split(source, Snap { area: 0, pos: 0, point: [0.0; 2] }, Snap { area: 0, pos: 1, point: [0.0; 2] }, &[])
        };
        let activations = vec![activation("w1", "france"), activation("w2", "spain"), activation("w3", "w1")];
        let splits = vec![split("w1", "w1_east"), split("w2", "w2_east"), split("w1_east", "w1_far")];
        let removal = removal_of_activation(&activations, &splits, "w1").unwrap();
        assert_eq!(removal.activations, vec![0, 2]);
        assert_eq!(removal.splits, vec![0, 2]);
        assert_eq!(removal.keys, vec!["w1", "w3", "w1_east", "w1_far"]);
        let mut project = Project { activations, splits, ..Project::default() };
        project.owners.insert("w1_east".into(), "france".into());
        removal.remove_from(&mut project);
        assert_eq!(project.activations.len(), 1);
        assert_eq!(project.splits.len(), 1);
        assert!(project.owners.is_empty());
        assert!(removal_of_activation(&project.activations, &project.splits, "w1").is_none());
    }

    #[test]
    fn swapping_twice_restores_the_split() {
        let split = build_split(
            "a",
            Snap { area: 1, pos: 3, point: [0.0; 2] },
            Snap { area: 1, pos: 9, point: [0.0; 2] },
            &[[1.0, 1.0], [2.0, 2.0]],
        );
        assert_eq!(split.interior, vec![[1.0, 1.0], [2.0, 2.0]]);
        assert_eq!(swapped(&split).interior, vec![[2.0, 2.0], [1.0, 1.0]]);
        assert_eq!((swapped(&split).a, swapped(&split).b), (9, 3));
        assert_eq!(swapped(&swapped(&split)), split);
        assert_eq!(clicks_of(&split), vec![[1.0, 1.0], [2.0, 2.0]]);
    }
}
