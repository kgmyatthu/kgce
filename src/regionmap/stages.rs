//! File-level stages of the format-8 chain, run by `pipeline::generate` on the converted map files under
//! `data/campaign_maps/nap_world`. The pure geometry lives in [`super::chain`]; these functions bring every file that
//! depends on it in line:
//!
//! 1. [`geometry_stage`] (before the activations): the fixed theatre bounds (rectangles, climate/wind canvases,
//!    TGAs, pathfinding grid, sea-grid boxes), clipping, the activation geometry options, painted terrain;
//! 2. [`synthesis_stage`] (after the activations): pathfinding paths for every activated / carved region;
//! 3. [`sea_stage`] (after the splits): sea-grid lists and connections of every touched theatre, trade-route guarantee;
//! 4. [`validate_world`] (at the end): the cross-file consistency checks.

use super::{
    apply::{AppliedActivation, AppliedSplit},
    chain::{self, ActivationGeometry},
    geometry::Rect,
    pathfinding::{Grid, TransitionArea},
    pathsynth,
    regions::RegionData,
    sea_grids::SeaGrids,
    trade_routes::{self, TradeSplines},
};
use crate::{
    esf::{self, Esf},
    map::MapArtifacts,
    model::{Progress, Project},
    terrain,
};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn read_esf(path: &Path) -> Result<Esf> {
    esf::read(&fs::read(path).with_context(|| format!("Read {}", path.display()))?)
        .with_context(|| format!("Parse {}", path.display()))
}

fn write_esf(path: &Path, file: &Esf) -> Result<()> {
    fs::write(path, esf::write(file)?).with_context(|| format!("Write {}", path.display()))
}

/// What [`geometry_stage`] decided and found; later stages and the validation read it.
#[derive(Clone, Debug)]
pub struct GeometryStage {
    /// the converted regions before any change (start-position HLCI rebuild compares area counts against it)
    pub base: RegionData,
    /// theatres (regions.esf keys) whose rectangle grew
    pub extended: BTreeSet<String>,
    /// ports listed by the untouched `sea_grids.esf` (the set `regenerate_lists` restricts a cell's ports to)
    pub vanilla_ports: BTreeSet<String>,
    /// per theatre key: the rule-checker problems of the untouched ETW pathfinding grid (see [`problem_identities`]).
    /// America and Europe fail the checker already; a build must not ADD to them
    pub baseline_problems: BTreeMap<String, BTreeSet<String>>,
    /// one entry per activation that has geometry options, in project order
    pub geometry: Vec<ActivationGeometry>,
    pub notes: Vec<String>,
}

/// Index of the theatre `key` in `regions.esf` = index of its `pathfinding_areas` item.
fn theatre_index(rd: &RegionData, key: &str) -> Result<usize> {
    rd.theatre_rects()
        .iter()
        .position(|(k, _)| k == key)
        .with_context(|| format!("regions.esf has no theatre {key}"))
}

/// Pathfinding area `index` belongs to a theatre whose rectangle starts at `rect`'s corner.
fn check_grid_matches(grid: &Grid, theatre: &str, rect: Rect) -> Result<()> {
    ensure!(
        (grid.x0 - f64::from(rect[0][0])).abs() < 0.01 && (grid.y0 - f64::from(rect[0][1])).abs() < 0.01,
        "pathfinding area {} starts at ({}, {}) but theatre {theatre} starts at ({}, {})",
        grid.area,
        grid.x0,
        grid.y0,
        rect[0][0],
        rect[0][1]
    );
    Ok(())
}

/// Stage 1, before the activations: applies the fixed theatre bounds and the activation geometry options to the
/// converted map in `map_dir` (`regions.esf`, `pathfinding.esf`, `sea_grids.esf`, the theatre TGAs and, when the
/// project names painted terrain, the `display` assets). The settlement-level work of the activations
/// (`apply_activations`) follows on the result.
pub fn geometry_stage(map_dir: &Path, project: &Project, progress: &Progress<'_>) -> Result<GeometryStage> {
    let regions_path = map_dir.join("regions.esf");
    let pf_path = map_dir.join("pathfinding.esf");
    let sea_path = map_dir.join("sea_grids.esf");
    let mut regions_file = read_esf(&regions_path)?;
    let mut rd = RegionData::parse(&regions_file.root)?;
    let base = rd.clone();
    let mut sea_file = read_esf(&sea_path)?;
    let mut sea = SeaGrids::parse(&sea_file.root)?;
    let vanilla_ports = sea.listed_ports();
    let mut notes = Vec::new();
    let mut extended = BTreeSet::new();
    let mut baseline_problems = BTreeMap::new();
    {
        let pf = read_esf(&pf_path)?;
        for (area, (key, _)) in rd.theatre_rects().iter().enumerate() {
            if area >= Grid::area_count(&pf.root)? {
                break;
            }
            let identities = problem_identities(&Grid::parse_area(&pf.root, area)?, &rd);
            if !identities.is_empty() {
                notes.push(format!(
                    "Theatre {key}: the pathfinding rule checker already reports {} problem(s) on the untouched ETW grid; the build is checked not to add any",
                    identities.len()
                ));
            }
            baseline_problems.insert(key.clone(), identities);
        }
    }

    // Fixed theatre bounds, applied to every build (also to projects without activations). Align once through the
    // same geometry operation the editor preview uses; every dependent asset works from these effective bounds,
    // including the climate/wind canvases. A theatre already sitting at its fixed bounds is skipped entirely.
    let changes = chain::extend(&mut rd)?;
    let mut sea_points = BTreeMap::new();
    if !changes.is_empty() {
        let mut pf_file = read_esf(&pf_path)?;
        for change in &changes {
            let theatre = &change.theatre;
            let old = change.old_rect;
            let stem = terrain::theatre_stem(theatre).context("theatre without map textures")?;
            sea_points.insert(theatre.clone(), chain::reference_sea_point(&base, stem, old)?);
            progress(&format!("Theatre {theatre}: extending climate and wind canvases"));
            let sizes = terrain::extend_canvases_in(&mut regions_file.root, theatre, old, change.new_rect)
                .with_context(|| format!("Theatre {theatre}: climate/wind canvases"))?;
            notes.push(format!(
                "Theatre {theatre} extended {:?} -> {:?}: climate canvas {}x{}, wind canvas {}x{}",
                old, change.new_rect, sizes.climate.0, sizes.climate.1, sizes.wind.0, sizes.wind.1
            ));
        }
        // The pathfinding extension needs the unclipped sea polygons.
        for change in &changes {
            let theatre = &change.theatre;
            progress(&format!("Theatre {theatre}: extending pathfinding grid, sea grid and map canvases"));
            let area = theatre_index(&rd, theatre)?;
            let mut grid = Grid::parse_area(&pf_file.root, area)?;
            check_grid_matches(&grid, theatre, change.old_rect)?;
            let transitions: Vec<TransitionArea> = change
                .moved
                .iter()
                .map(|m| TransitionArea { id: m.id, old: m.old, new: m.new })
                .collect();
            grid.extend(change.new_rect, &rd, &transitions)
                .with_context(|| format!("Theatre {theatre}: extend the pathfinding grid"))?;
            grid.store(&mut pf_file.root)?;
            sea.resize_theatre(theatre, change.new_rect)
                .with_context(|| format!("Theatre {theatre}: resize the sea grid"))?;
            let stem = terrain::theatre_stem(theatre)
                .with_context(|| format!("Theatre {theatre} has no theatre TGAs; only theatres 1, 2 and 3 can be extended"))?;
            let sea_point = *sea_points.get(theatre).with_context(|| format!("Theatre {theatre}: no sea point"))?;
            let sizes = terrain::recanvas_tgas(map_dir, stem, change.old_rect, change.new_rect, sea_point)?;
            notes.push(format!(
                "Theatre {theatre}: {} moved transition area(s); TGAs {}",
                change.moved.len(),
                sizes.iter().map(|(n, (w, h))| format!("{n} {w}x{h}")).collect::<Vec<_>>().join(", ")
            ));
            extended.insert(theatre.clone());
        }
        write_esf(&pf_path, &pf_file)?;
        progress("Clipping sea and backdrop polygons at the theatre rectangles");
        let clip = chain::clip(&mut rd)?;
        notes.push(format!("Clipped at the theatre rectangles + {} u: {clip:?}", chain::CLIP_MARGIN));
    }

    // Per activation, in project order: geometry options, then the settlement. The settlement is attached here
    // (apply_activations only attaches when it is missing) so the component id of a later activation can follow the
    // settled land of an earlier one, exactly as the editor preview does.
    let mut geometry = Vec::new();
    for (index, activation) in project.activations.iter().enumerate() {
        progress(&format!("Activation {}: area selection, merge, carve and settlement", activation.key));
        let done = crate::map::splits::activate_with_geometry(&mut rd, activation)
            .with_context(|| format!("Activation #{} {}", index + 1, activation.key))?;
        if let Some(done) = done {
            notes.extend(done.notes.iter().map(|n| format!("Activation {}: {n}", activation.key)));
            geometry.push(done);
        }
    }

    let problems = rd.topology_check();
    ensure!(
        problems.is_empty(),
        "the regions.esf geometry is inconsistent after the theatre / activation changes ({} problems):\n  {}",
        problems.len(),
        problems.iter().take(20).cloned().collect::<Vec<_>>().join("\n  ")
    );
    rd.store(&mut regions_file.root)?;
    let cells = rd.rebuild_query_index(&mut regions_file.root)?;
    notes.push(format!("regions.esf query index rebuilt ({cells} cells)"));
    write_esf(&regions_path, &regions_file)?;
    sea.store(&mut sea_file.root)?;
    write_esf(&sea_path, &sea_file)?;

    if let Some(dir) = &project.painted_terrain {
        progress("Importing painted terrain over the generated display assets");
        let copied = terrain::import_painted_terrain(dir, map_dir)
            .with_context(|| format!("Import painted terrain from {}", dir.display()))?;
        notes.push(format!("Painted terrain imported from {}: {}", dir.display(), copied.join(", ")));
    }
    Ok(GeometryStage { base, extended, vanilla_ports, baseline_problems, geometry, notes })
}

/// Theatres whose pathfinding grid and sea grid the project changed: extended ones plus those of activations with
/// geometry options.
pub fn touched_theatres(stage: &GeometryStage, applied: &[AppliedActivation], project: &Project) -> BTreeSet<String> {
    let mut out = stage.extended.clone();
    for activation in project.activations.iter().filter(|a| chain::changes_geometry(a)) {
        if let Some(done) = applied.iter().find(|a| a.key == activation.key) {
            out.insert(done.theatre.clone());
        }
    }
    out
}

/// Stage 2, after the activations: gives every activated region whose polygon the chain changed (carved, merged,
/// newly passable) its paths in the theatre's pathfinding grid. A region that already has a path (an ordinary
/// wilderness activation) keeps it, unless its geometry changed, which is refused: its old pieces would not match.
pub fn synthesis_stage(
    map_dir: &Path,
    project: &Project,
    map: &MapArtifacts,
    applied: &[AppliedActivation],
    applied_splits: &[AppliedSplit],
    stage: &GeometryStage,
    progress: &Progress<'_>,
) -> Result<Vec<String>> {
    let mut notes = Vec::new();
    if project.activations.is_empty() && project.splits.is_empty() {
        return Ok(notes);
    }
    let regions_file = read_esf(&map_dir.join("regions.esf"))?;
    let rd = RegionData::parse(&regions_file.root)?;
    let mut pf_file = read_esf(&map_dir.join("pathfinding.esf"))?;
    let mut by_area: BTreeMap<usize, Vec<pathsynth::Target>> = BTreeMap::new();
    for activation in &project.activations {
        let done = applied
            .iter()
            .find(|a| a.key == activation.key)
            .with_context(|| format!("activation {} was not applied", activation.key))?;
        let centre = [f64::from(activation.center[0]), f64::from(activation.center[1])];
        let area = Grid::area_at(&pf_file.root, centre)?
            .with_context(|| format!("activation {}: the settlement lies outside every pathfinding area", activation.key))?;
        let grid = Grid::parse_area(&pf_file.root, area)?;
        let present = grid.region_ids.contains(&(done.index as i16));
        if present {
            ensure!(
                !chain::changes_geometry(activation),
                "activation {}: the region already has paths in the pathfinding grid; selecting areas, merging or \
                 carving would leave them stale. Use `centre` / no merge for regions the theatre already paths",
                activation.key
            );
            continue;
        }
        let kind_name = map.slot_mapping["regions"][&activation.key]["settlement_slot_type"]
            .as_str()
            .with_context(|| format!("activation {}: the settlement slot type is unknown", activation.key))?;
        let kind = pathsynth::SlotKind::from_name(kind_name)
            .with_context(|| format!("activation {}: settlement slot type {kind_name} has no node shape", activation.key))?;
        by_area.entry(area).or_default().push(pathsynth::Target {
            region: done.index,
            settlement_slot_kind: kind,
            options: pathsynth::TargetOptions::default(),
        });
    }
    // split regions whose source had no paths when the split ran (it is synthesised above): the new half is a target too
    for (split, done) in project.splits.iter().zip(applied_splits) {
        let centre = [f64::from(split.center[0]), f64::from(split.center[1])];
        let Some(area) = Grid::area_at(&pf_file.root, centre)? else { continue };
        let grid = Grid::parse_area(&pf_file.root, area)?;
        if grid.region_ids.contains(&(done.new_index as i16)) {
            continue;
        }
        let kind_name = map.slot_mapping["regions"][&split.key]["settlement_slot_type"]
            .as_str()
            .with_context(|| format!("split {}: the settlement slot type is unknown", split.key))?;
        let kind = pathsynth::SlotKind::from_name(kind_name)
            .with_context(|| format!("split {}: settlement slot type {kind_name} has no node shape", split.key))?;
        by_area.entry(area).or_default().push(pathsynth::Target {
            region: done.new_index,
            settlement_slot_kind: kind,
            options: pathsynth::TargetOptions::default(),
        });
    }
    let options = pathsynth::Options::default();
    for (area, targets) in by_area {
        progress(&format!(
            "Pathfinding area {area}: synthesising paths for {}",
            targets.iter().map(|t| rd.regions[t.region].key.as_str()).collect::<Vec<_>>().join(", ")
        ));
        let mut grid = Grid::parse_area(&pf_file.root, area)?;
        let report = pathsynth::synthesize(&mut grid, &rd, &targets, &options)
            .with_context(|| format!("pathfinding area {area}: path synthesis"))?;
        let empty = BTreeSet::new();
        let baseline = rd.theatre_rects().get(area).and_then(|(key, _)| stage.baseline_problems.get(key)).unwrap_or(&empty);
        let problems = new_problems(baseline, &grid, &rd);
        ensure!(
            problems.is_empty(),
            "pathfinding area {area} fails the rule checker after synthesis ({} new problems):\n  {}",
            problems.len(),
            problems.iter().take(20).cloned().collect::<Vec<_>>().join("\n  ")
        );
        grid.store(&mut pf_file.root)?;
        for region in &report.per_region {
            notes.push(format!(
                "Pathfinding {}: path {}, {} landing crossing(s){}",
                rd.regions[region.region].key,
                region.path_id,
                region.landings,
                if region.node_clipped { ", settlement node clipped to the land zone" } else { "" }
            ));
        }
        notes.extend(report.warnings.iter().map(|w| format!("Pathfinding area {area}: {w}")));
    }
    write_esf(&map_dir.join("pathfinding.esf"), &pf_file)?;
    Ok(notes)
}

/// Stage 3, after the splits: regenerates the sea-grid lists and connections of the touched theatres from the final
/// geometry and the trade-route splines, and makes sure every settlement node has a route.
pub fn sea_stage(
    map_dir: &Path,
    project: &Project,
    stage: &GeometryStage,
    touched: &BTreeSet<String>,
    progress: &Progress<'_>,
) -> Result<Vec<String>> {
    let mut notes = Vec::new();
    let regions_file = read_esf(&map_dir.join("regions.esf"))?;
    let rd = RegionData::parse(&regions_file.root)?;
    let mut trade_file = read_esf(&map_dir.join("trade_routes.esf"))?;
    if !touched.is_empty() {
        let splines = TradeSplines::parse(&trade_file.root)?;
        let mut sea_file = read_esf(&map_dir.join("sea_grids.esf"))?;
        let mut sea = SeaGrids::parse(&sea_file.root)?;
        for theatre in touched {
            progress(&format!("Theatre {theatre}: sea grid lists and connections"));
            let changed = sea.regenerate_lists(theatre, &rd, &splines, &stage.vanilla_ports)?;
            let added = sea.complete_conns(theatre)?;
            notes.push(format!("Sea grid {theatre}: {changed} cell list(s) regenerated, {added} connection(s) added"));
        }
        sea.store(&mut sea_file.root)?;
        write_esf(&map_dir.join("sea_grids.esf"), &sea_file)?;
    }
    if !project.activations.is_empty() || !project.splits.is_empty() || !touched.is_empty() {
        let routes = trade_routes::ensure_settlement_routes(&mut trade_file.root, &rd)?;
        for (isolated, partner, distance) in &routes {
            notes.push(format!("Trade routes: {isolated} had no route; linked to {partner} ({distance:.1} u)"));
        }
        if !routes.is_empty() {
            write_esf(&map_dir.join("trade_routes.esf"), &trade_file)?;
        }
    }
    Ok(notes)
}

/// Stage 4: cross-file checks of the finished map files (`map_dir`); the start-position checks are the caller's.
/// Returns every problem found. `touched` are the theatres the chain changed.
pub fn validate_world(
    map_dir: &Path,
    project: &Project,
    stage: &GeometryStage,
    touched: &BTreeSet<String>,
) -> Result<Vec<String>> {
    let mut problems = Vec::new();
    let regions_file = read_esf(&map_dir.join("regions.esf"))?;
    let rd = RegionData::parse(&regions_file.root)?;
    problems.extend(rd.topology_check().into_iter().map(|p| format!("regions.esf: {p}")));
    let pf_file = read_esf(&map_dir.join("pathfinding.esf"))?;
    let trade_file = read_esf(&map_dir.join("trade_routes.esf"))?;
    let sea_file = read_esf(&map_dir.join("sea_grids.esf"))?;

    // every settled region the project adds, and every settled region of a touched theatre, is in its theatre's
    // pathfinding region list with a single path
    let added: BTreeSet<&str> = project
        .activations
        .iter()
        .map(|a| a.key.as_str())
        .chain(project.splits.iter().map(|s| s.key.as_str()))
        .collect();
    let mut grids: BTreeMap<usize, Grid> = BTreeMap::new();
    for (ri, region) in rd.regions.iter().enumerate() {
        let Some(_) = &region.settlement else { continue };
        let Ok(centre) = super::regions::settlement_center(region) else { continue };
        let Some(area) = Grid::area_at(&pf_file.root, [f64::from(centre[0]), f64::from(centre[1])])? else {
            if added.contains(region.key.as_str()) {
                problems.push(format!("pathfinding.esf: the settlement of {} lies outside every pathfinding area", region.key));
            }
            continue;
        };
        let theatre_touched = rd.theatre_rects().get(area).is_some_and(|(key, _)| touched.contains(key));
        if !added.contains(region.key.as_str()) && !theatre_touched {
            continue;
        }
        if let std::collections::btree_map::Entry::Vacant(v) = grids.entry(area) {
            v.insert(Grid::parse_area(&pf_file.root, area)?);
        }
        let grid = &grids[&area];
        if !grid.region_ids.contains(&(ri as i16)) || grid.single_path(ri as i16).is_none() {
            problems.push(format!(
                "pathfinding.esf: area {area} has no single path for {} (the region is not in the region list)",
                region.key
            ));
        }
    }
    // every settlement the project adds sits in a type-7 node of its region's own single path; the halves of a
    // split share a group path (their border band)
    let node_keys: Vec<&str> = project
        .activations
        .iter()
        .filter(|a| chain::changes_geometry(a))
        .map(|a| a.key.as_str())
        .chain(project.splits.iter().map(|s| s.key.as_str()))
        .collect();
    for key in node_keys {
        let Ok(ri) = rd.index(key) else { continue };
        let Ok(centre) = super::regions::settlement_center(&rd.regions[ri]) else { continue };
        let point = [f64::from(centre[0]), f64::from(centre[1])];
        let Some(area) = Grid::area_at(&pf_file.root, point)? else { continue };
        if let std::collections::btree_map::Entry::Vacant(v) = grids.entry(area) {
            v.insert(Grid::parse_area(&pf_file.root, area)?);
        }
        if !settlement_node_at(&grids[&area], ri as i16, point) {
            problems.push(format!("pathfinding.esf: the settlement of {key} at {centre:?} is not in a type-7 node of its own path"));
        }
    }
    for split in &project.splits {
        let (Ok(a), Ok(b)) = (rd.index(&split.source), rd.index(&split.key)) else { continue };
        let centre = [f64::from(split.center[0]), f64::from(split.center[1])];
        let Some(area) = Grid::area_at(&pf_file.root, centre)? else { continue };
        if let std::collections::btree_map::Entry::Vacant(v) = grids.entry(area) {
            v.insert(Grid::parse_area(&pf_file.root, area)?);
        }
        let (a, b) = (a as i16, b as i16);
        if !grids[&area].paths.iter().any(|p| p.len() > 1 && p.contains(&a) && p.contains(&b)) {
            problems.push(format!("pathfinding.esf: no group path joins {} and its split {} (the border band is missing)", split.source, split.key));
        }
    }
    for (area, grid) in &grids {
        if let Some((key, _)) = rd.theatre_rects().get(*area) {
            if touched.contains(key) {
                let empty = BTreeSet::new();
                let baseline = stage.baseline_problems.get(key).unwrap_or(&empty);
                problems.extend(new_problems(baseline, grid, &rd).into_iter().map(|p| format!("pathfinding.esf area {area}: {p}")));
            }
        }
    }

    // sea grid lands/seas consistent with the geometry (regeneration reproduces the stored lists)
    if !touched.is_empty() {
        let splines = TradeSplines::parse(&trade_file.root)?;
        let stored = SeaGrids::parse(&sea_file.root)?;
        let ports = stage.vanilla_ports.clone();
        let mut fresh = stored.clone();
        for theatre in touched {
            fresh.regenerate_lists(theatre, &rd, &splines, &ports)?;
            fresh.complete_conns(theatre)?;
            if fresh.grid(theatre) != stored.grid(theatre) {
                problems.push(format!("sea_grids.esf: the lists or connections of theatre {theatre} do not match the regions"));
            }
        }
    }

    // no trade node without a route
    for node in trade_routes::routeless_nodes(&trade_file.root)? {
        problems.push(format!("trade_routes.esf: node {node} has no route"));
    }
    Ok(problems)
}

/// Per-theatre cell counts of the stored sea grids (`(theatre key, cells)`), for the start-position check that its
/// `CAI_SEA_GRID_CELL` count equals them.
pub fn sea_cell_counts(map_dir: &Path) -> Result<Vec<(String, usize)>> {
    let sea = SeaGrids::parse(&read_esf(&map_dir.join("sea_grids.esf"))?.root)?;
    Ok(sea.grids.iter().map(|g| (g.name.clone(), g.cells.len())).collect())
}

/// A type-7 piece of `region`'s single path holds `point` (the settlement node), looking at the cell of the point
/// and its eight neighbours.
pub fn settlement_node_at(grid: &Grid, region: i16, point: [f64; 2]) -> bool {
    let Some(path) = grid.single_path(region) else { return false };
    let (col, row) = (((point[0] - grid.x0) / grid.size).floor() as i64, ((point[1] - grid.y0) / grid.size).floor() as i64);
    for dr in -1..=1i64 {
        for dc in -1..=1i64 {
            let (c, r) = (col + dc, row + dr);
            if c < 0 || r < 0 || c as usize >= grid.cols || r as usize >= grid.rows {
                continue;
            }
            let index = r as usize * grid.cols + c as usize;
            for piece in &grid.cells[index].pieces {
                if piece.a & 0xF == 7 && piece.path_id() == path {
                    if let Ok(poly) = grid.piece_polygon(index, piece) {
                        if super::geometry::point_in_ring(point, &poly) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// The rule-checker problems of `grid` in a form that survives a theatre extension and a path / vertex renumbering:
/// cell coordinates become world coordinates of the cell origin, path ids and vertex ids are dropped, polygon dumps
/// are cut off. Two problems with the same identity are the same defect.
pub fn problem_identities(grid: &Grid, rd: &RegionData) -> BTreeSet<String> {
    use regex::{Captures, Regex};
    let cell = Regex::new(r"cell \((\d+),(\d+)\)").expect("static pattern");
    let single = Regex::new(r"single path \d+ is").expect("static pattern");
    let group = Regex::new(r"group path \d+ ").expect("static pattern");
    let vertex = Regex::new(r"at id \d+").expect("static pattern");
    pathsynth::validate_all(grid, rd)
        .into_iter()
        .map(|problem| {
            let problem = problem.split(" (area2").next().unwrap_or(&problem).to_string();
            let problem = cell.replace_all(&problem, |c: &Captures| {
                let (col, row): (f64, f64) = (c[1].parse().unwrap_or(0.0), c[2].parse().unwrap_or(0.0));
                format!("cell @({:.2},{:.2})", grid.x0 + col * grid.size, grid.y0 + row * grid.size)
            });
            let problem = single.replace_all(&problem, "single path is");
            let problem = group.replace_all(&problem, "group path ");
            vertex.replace_all(&problem, "at vertex").into_owned()
        })
        .collect()
}

/// Problems of `grid` that the untouched grid (`baseline`, from [`problem_identities`]) did not have.
pub fn new_problems(baseline: &BTreeSet<String>, grid: &Grid, rd: &RegionData) -> Vec<String> {
    problem_identities(grid, rd).into_iter().filter(|p| !baseline.contains(p)).collect()
}
