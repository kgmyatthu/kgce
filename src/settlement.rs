//! Moving existing settlements: a capital (its whole `settlement:` slot cluster) or a town slot to a new point inside its
//! own region.
//!
//! A [`Move`] is stored in the project in source (original game) coordinates, addressed by region key and a stable slot
//! key (see `target`: the settlement id `settlement:<region>:<name>` of a capital, `town:<region>:<name>` of a town). The
//! same code serves the editor and the export:
//!
//! * [`validate`] – may this slot go there? (inside its own passable area, outside holes, node clearance from the
//!   outline, the shipped spacing between settlements; ports are refused)
//! * [`preview`] – the editor's replay: the moves translate the slots of the parsed `regions.esf` geometry, so the map
//!   markers, the next move's validation and the export all see the same positions
//! * [`apply`] – the export: after the source-space start position is built (with activations, splits and HLCIs) and
//!   before the atlas transform, the generated files are changed consistently:
//!   `regions.esf` (slot positions, theatre region list), `pathfinding.esf` (settlement node carved at the new point and
//!   the old one restored to land, rule-checked), `poi.esf` (settlement point and distances), `trade_routes.esf` (road
//!   ends and route lengths), `startpos.esf` (residence, slots, CAI region / settlement / HLCI / region slots, garrisoned
//!   characters, trade segments and routes). The campaign database rows are keyed by id and hold no coordinates; they
//!   are checked to still list every moved slot.
//!
//! Nothing is written unless every move and every consistency check succeeded.

use std::{
    collections::BTreeSet,
    fs,
    path::Path,
};

use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    esf::{self, Esf, Node},
    map::MapArtifacts,
    model::Progress,
    regionmap::regions::RegionData,
};

mod nav;
mod placement;
mod poi_edit;
mod shift;
mod startpos;
mod target;
mod trade;

#[cfg(test)]
mod tests;

pub use placement::{MIN_SEPARATION, check};
pub use shift::SlotMove;
pub use target::{Kind, Target, resolve, targets};

/// One settlement moved to a new position, in source coordinates (before any atlas transform).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Move {
    /// region key (`regions.esf`)
    pub region: String,
    /// stable slot key: the settlement id of a capital (`settlement:<region>:<name>`) or the slot key of a town
    /// (`town:<region>:<name>`); a bare name is accepted when it is unambiguous in the region
    pub slot: String,
    /// destination (x, y)
    pub position: [f32; 2],
}

/// Structure checks that need no map: finite positions, named targets, no target moved twice (under the same spelling;
/// two spellings of one target are caught when the moves are resolved).
pub fn validate_structure(moves: &[Move]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for movement in moves {
        ensure!(
            !movement.region.is_empty() && !movement.slot.is_empty(),
            "A settlement move needs a region and a slot"
        );
        ensure!(
            movement.position.iter().all(|v| v.is_finite()),
            "Settlement {}/{} has a non-finite position",
            movement.region,
            movement.slot
        );
        ensure!(
            seen.insert((movement.region.as_str(), movement.slot.as_str())),
            "Settlement {}/{} is moved more than once",
            movement.region,
            movement.slot
        );
    }
    Ok(())
}

/// One effective move after resolution: the target as it stood before, the destination and where its slots went.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub target: Target,
    pub to: [f32; 2],
    pub slots: Vec<SlotMove>,
}

/// Resolves, validates and applies `moves` to the settlements of `data`, in order (a later move sees the earlier ones).
/// Moves whose destination equals the current position are skipped. On error `data` is exactly as it was.
pub(crate) fn plan(data: &mut RegionData, moves: &[Move]) -> Result<Vec<Plan>> {
    validate_structure(moves)?;
    let mut saved: Vec<(usize, Option<Node>)> = Vec::new();
    match plan_into(data, moves, &mut saved) {
        Ok(plans) => Ok(plans),
        Err(error) => {
            for (index, settlement) in saved.into_iter().rev() {
                data.regions[index].settlement = settlement;
            }
            Err(error)
        }
    }
}

fn plan_into(data: &mut RegionData, moves: &[Move], saved: &mut Vec<(usize, Option<Node>)>) -> Result<Vec<Plan>> {
    let mut seen: BTreeSet<(usize, String)> = BTreeSet::new();
    let mut plans = Vec::new();
    for (number, movement) in moves.iter().enumerate() {
        let context = || format!("Settlement move #{} ({}/{})", number + 1, movement.region, movement.slot);
        let target = resolve(data, &movement.region, &movement.slot).with_context(context)?;
        ensure!(
            seen.insert((target.region_index, target.slot.clone())),
            "{} of {} is moved more than once",
            target.slot,
            target.region
        );
        if movement.position == target.position {
            continue;
        }
        check(data, &target, movement.position).with_context(context)?;
        saved.push((target.region_index, data.regions[target.region_index].settlement.clone()));
        let node = data.regions[target.region_index]
            .settlement
            .as_mut()
            .ok_or_else(|| anyhow!("{} has no settlement", target.region))?;
        let slots = shift::move_in(node, &target, movement.position).with_context(context)?;
        plans.push(Plan { target, to: movement.position, slots });
    }
    Ok(plans)
}

/// May `movement` be applied to `data`? (Existing moves in `data` count as done.)
pub fn validate(data: &RegionData, movement: &Move) -> Result<()> {
    validate_structure(std::slice::from_ref(movement))?;
    let target = resolve(data, &movement.region, &movement.slot)?;
    check(data, &target, movement.position)
}

/// Editor replay: applies `moves` to the slots of `data` (all or nothing).
pub fn preview(data: &mut RegionData, moves: &[Move]) -> Result<()> {
    plan(data, moves).map(|_| ())
}

fn read_esf(path: &Path) -> Result<Esf> {
    esf::read(&fs::read(path).with_context(|| format!("Read {}", path.display()))?)
        .with_context(|| format!("Parse {}", path.display()))
}

/// Replaces the settlement record of every moved region with the moved one of `rd` and moves the region's entry in the
/// theatre region lists.
fn write_regions(file: &mut Esf, rd: &RegionData, plans: &[Plan]) -> Result<()> {
    let mut done: BTreeSet<usize> = BTreeSet::new();
    {
        let items = file.root.path_mut(&["region_data", "regions"])?.items_mut()?;
        for plan in plans {
            if !done.insert(plan.target.region_index) {
                continue;
            }
            let key = plan.target.region.as_str();
            let item = items
                .iter_mut()
                .find(|item| item.first().and_then(|node| node.as_str().ok()) == Some(key))
                .ok_or_else(|| anyhow!("regions.esf has no region {key}"))?;
            let slot = item
                .iter_mut()
                .find(|node| node.name() == Some("settlement_and_slots"))
                .ok_or_else(|| anyhow!("region {key} has no settlement record"))?;
            *slot = rd.regions[plan.target.region_index]
                .settlement
                .clone()
                .ok_or_else(|| anyhow!("region {key} lost its settlement"))?;
        }
    }
    for plan in plans.iter().filter(|plan| plan.target.kind == Kind::Capital) {
        let region = plan.target.region.as_str();
        let mut found = 0usize;
        for item in file.root.get_mut("theatres_and_region_keys")?.items_mut()? {
            for entry in item[0].get_mut("region_keys")?.items_mut()? {
                if entry[0].as_str().ok() == Some(region) {
                    let listed = entry[1].as_xy()?;
                    ensure!(
                        (f64::from(listed[0]) - f64::from(plan.target.position[0])).abs() < 0.01
                            && (f64::from(listed[1]) - f64::from(plan.target.position[1])).abs() < 0.01,
                        "The theatre list places {region} at {listed:?}, its settlement is at {:?}",
                        plan.target.position
                    );
                    entry[1] = esf::xyv(plan.to[0], plan.to[1]);
                    found += 1;
                }
            }
        }
        ensure!(found >= 1, "{region} is not listed in any theatre of regions.esf");
    }
    Ok(())
}

/// The campaign database must still list every moved slot (its rows are keyed by id and hold no coordinates).
fn check_database(map: &MapArtifacts, plan: &Plan) -> Result<()> {
    let table = match plan.target.kind {
        Kind::Capital => "campaign_map_settlements_tables",
        Kind::Town | Kind::Port => "campaign_map_towns_and_ports_tables",
    };
    let rows = map
        .db_rows
        .get(table)
        .and_then(|file| file.get("rows"))
        .and_then(|rows| rows.as_array())
        .ok_or_else(|| anyhow!("The generated database has no {table} rows"))?;
    let listed = rows.iter().any(|row| {
        row.as_object()
            .is_some_and(|fields| fields.values().any(|value| value.as_str() == Some(plan.target.slot.as_str())))
    });
    ensure!(listed, "{table} has no row for {}; the moved slot would be unregistered", plan.target.slot);
    Ok(())
}

/// Reads the serialised `regions.esf` back and checks that every moved target stands exactly where the preview put it.
fn verify_regions(bytes: &[u8], plans: &[Plan]) -> Result<()> {
    let back = esf::read(bytes).context("Re-read the written regions.esf")?;
    let rd = RegionData::parse(&back.root).context("Re-read the written region geometry")?;
    for plan in plans {
        let target = resolve(&rd, &plan.target.region, &plan.target.slot)?;
        ensure!(
            target.position == plan.to,
            "{} was written at {:?} instead of {:?}",
            plan.target.slot,
            target.position,
            plan.to
        );
    }
    Ok(())
}

/// Export step: moves the settlements of `moves` in the generated map and start position under `data_dir` (the NTW
/// `data` folder being built). Returns report lines.
pub fn apply(data_dir: &Path, moves: &[Move], map: &mut MapArtifacts, progress: &Progress<'_>) -> Result<Vec<String>> {
    validate_structure(moves)?;
    if moves.is_empty() {
        return Ok(Vec::new());
    }
    let map_dir = data_dir.join("campaign_maps").join("nap_world");
    let startpos_path = data_dir.join("campaigns").join("wld_napoleon").join("startpos.esf");
    progress(&format!("Moving {} settlement(s) in the generated campaign", moves.len()));

    let mut regions = read_esf(&map_dir.join("regions.esf"))?;
    let mut rd = RegionData::parse(&regions.root).context("Read the generated region geometry")?;
    let mut positions = trade::Positions::of(&rd)?;
    let plans = plan(&mut rd, moves)?;
    if plans.is_empty() {
        return Ok(vec!["Settlement moves: every destination equals the current position, nothing changed".to_string()]);
    }
    positions.moved(&plans);
    for plan in &plans {
        check_database(map, plan)?;
    }
    let mut notes: Vec<String> = Vec::new();

    write_regions(&mut regions, &rd, &plans)?;
    notes.push(format!(
        "regions.esf: {} settlement(s) moved ({})",
        plans.len(),
        plans
            .iter()
            .map(|plan| format!("{} to ({}, {})", plan.target.slot, plan.to[0], plan.to[1]))
            .collect::<Vec<_>>()
            .join(", ")
    ));

    progress("Moving settlement nodes in the pathfinding grid");
    let mut pathfinding = read_esf(&map_dir.join("pathfinding.esf"))?;
    notes.extend(nav::apply(&mut pathfinding.root, &rd, &plans)?);

    let mut poi = read_esf(&map_dir.join("poi.esf"))?;
    notes.extend(poi_edit::apply(&mut poi.root, &rd, &plans)?);

    let mut trade_file = read_esf(&map_dir.join("trade_routes.esf"))?;
    let change = trade::apply(&mut trade_file.root, &plans, &positions)?;
    notes.extend(change.notes.iter().cloned());

    progress("Moving settlement positions in the start position");
    let mut start = read_esf(&startpos_path)?;
    notes.extend(startpos::apply(&mut start.root, &plans, &change, &positions)?);

    let regions_bytes = esf::write(&regions)?;
    verify_regions(&regions_bytes, &plans)?;
    let start_bytes = esf::write(&start)?;
    let start_back = esf::read(&start_bytes).context("Re-read the written startpos.esf")?;
    startpos::verify(&start_back.root, &plans)?;
    let pathfinding_bytes = esf::write(&pathfinding)?;
    let poi_bytes = esf::write(&poi)?;
    let trade_bytes = esf::write(&trade_file)?;

    for (path, bytes) in [
        (map_dir.join("regions.esf"), regions_bytes),
        (map_dir.join("pathfinding.esf"), pathfinding_bytes),
        (map_dir.join("poi.esf"), poi_bytes),
        (map_dir.join("trade_routes.esf"), trade_bytes),
        (startpos_path, start_bytes),
    ] {
        fs::write(&path, bytes).with_context(|| format!("Write {}", path.display()))?;
    }
    notes.push(
        "Campaign database rows are keyed by slot id and hold no coordinates: unchanged (every moved slot is still listed)"
            .to_string(),
    );
    Ok(notes)
}
