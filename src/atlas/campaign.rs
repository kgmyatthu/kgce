//! Stock-Q20 atlas migration of the generated campaign: `startpos.esf`, `scripting.lua` and the coordinate-bearing
//! database rows. Native port of the accepted POC (`campaign_atlas.py`, `atlas_visibility.py`, `startpos_scale.py`,
//! camera bounds of `camera_sync.py`); runs once, AFTER the source-space start position (settlement moves, HLCls) and
//! the geography transform, BEFORE packing. Nothing is written unless every record is classified and the independent
//! verification of the re-read output has no violation.

mod contract;
mod convert;
mod nodes;
mod script;
mod tables;
mod verify;
mod visibility;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use super::geography::{Geography, cached_grid_charts};
use super::Plan;
use crate::esf::{self, Esf};
use crate::model::Progress;
use crate::regionmap::regions::RegionData;

pub use convert::camera_bounds;
pub use tables::{PLAYABLE_AREAS, transform_members, transform_trade_nodes};

use contract::{Contract, PointOverrides};
use convert::{Conv, RegionSet, problem_summary};

const CAMPAIGN_DIR: &str = "campaigns/wld_napoleon";
const OUTPUT_REGIONS: &str = "campaign_maps/nap_world/regions.esf";

/// Region order, area counts and theatre rectangles of the three region sources must agree: the HLCI area lists are
/// source-indexed and the converter checks them against the output geometry.
fn check_regions(plan: &Plan, src: &RegionSet, out: &RegionSet, contract: &Contract) -> Result<()> {
    let keys = |s: &RegionSet| s.data.regions.iter().map(|r| r.key.clone()).collect::<Vec<_>>();
    let wanted: Vec<String> = contract.regions.iter().map(|r| r.key.clone()).collect();
    ensure!(keys(src) == wanted && keys(out) == wanted, "region key order differs between source regions, output regions and the regions contract");
    for r in &contract.regions {
        let (s, o) = (&src.data.regions[src.index[&r.key]], &out.data.regions[out.index[&r.key]]);
        ensure!(
            s.areas.len() == r.source_areas && o.areas.len() == r.output_areas,
            "{}: area counts ({} source, {} output) differ from the contract ({} / {})",
            r.key,
            s.areas.len(),
            o.areas.len(),
            r.source_areas,
            r.output_areas
        );
    }
    for chart in &plan.charts {
        for (theatre, want) in &chart.theatre_rects {
            let have = src
                .data
                .theatres
                .iter()
                .find(|t| &t.key == theatre)
                .with_context(|| format!("source regions have no theatre {theatre}"))?
                .rect;
            ensure!(
                (0..2).all(|a| (0..2).all(|b| f64::from(have[a][b]) == want[a][b])),
                "source theatre {theatre} is {have:?} but the accepted atlas expects {want:?}: the atlas is only defined for the unmodified stock theatre rectangles (undo theatre extensions or disable the atlas)"
            );
        }
    }
    Ok(())
}

/// Convert `data/campaigns/wld_napoleon/{startpos.esf,scripting.lua}` in place to the atlas.
///
/// `data` is the generated data root. The OUTPUT `campaign_maps/nap_world/regions.esf` must already be the atlas
/// version (written by `geography::transform`), `geography.source_regions` the untouched source regions.
pub fn transform(data: &Path, plan: &Plan, geography: &Geography, progress: &Progress<'_>) -> Result<Value> {
    plan.validate()?;
    let dir = data.join(CAMPAIGN_DIR);
    let (startpos_path, script_path) = (dir.join("startpos.esf"), dir.join("scripting.lua"));
    progress("Atlas campaign: reading start position, script and region geometry");
    let start = esf::read(&fs::read(&startpos_path).with_context(|| format!("Read {}", startpos_path.display()))?)
        .with_context(|| format!("Parse {}", startpos_path.display()))?;
    let lua = fs::read_to_string(&script_path).with_context(|| format!("Read {}", script_path.display()))?;
    let regions_path = data.join(OUTPUT_REGIONS);
    let out_regions = esf::read(&fs::read(&regions_path).with_context(|| format!("Read {}", regions_path.display()))?)
        .with_context(|| format!("Parse {}", regions_path.display()))?;

    let contract = Contract::parse(&geography.contract)?;
    let overrides = PointOverrides::parse(&geography.point_overrides, plan)?;
    let src_set = RegionSet::new(RegionData::parse(&geography.source_regions.root).context("Parse source regions")?);
    let out_set = RegionSet::new(RegionData::parse(&out_regions.root).context("Parse atlas regions")?);
    check_regions(plan, &src_set, &out_set, &contract)?;

    let grid_charts = cached_grid_charts(&start.root, plan)?;
    let mut conv = Conv::new(plan, overrides, &contract, src_set, out_set, &start.root, grid_charts)?;
    progress("Atlas campaign: moving start position coordinates, line-of-sight caches and CAI indexes");
    let mut root = start.root.clone();
    conv.transform(&mut root)?;
    if !conv.problems.is_empty() {
        bail!(problem_summary(&conv.problems, 40));
    }
    let (script, script_report) = script::atlas_script(&lua, plan, conv.src_bounds, conv.new_bounds)?;

    progress("Atlas campaign: verifying the serialised start position against the source");
    let bytes = esf::write(&Esf { timestamp: start.timestamp, names: start.names.clone(), root }).context("Serialise atlas start position")?;
    let reread = esf::read(&bytes).context("Re-read atlas start position")?;
    let verification = verify::verify(&conv, &start.root, &reread.root)?;
    if !verification.violations.is_empty() {
        let mut msg = format!("atlas startpos verification failed: {} violations", verification.violations.len());
        for (at, detail) in verification.violations.iter().take(40) {
            msg.push_str(&format!("\n  {at}: {detail}"));
        }
        bail!(msg);
    }

    fs::write(&startpos_path, &bytes).with_context(|| format!("Write {}", startpos_path.display()))?;
    fs::write(&script_path, script).with_context(|| format!("Write {}", script_path.display()))?;
    progress("Atlas campaign: start position and script written");

    let b = |v: [f32; 4]| json!({"xmin": v[0], "ymin": v[1], "xmax": v[2], "ymax": v[3]});
    Ok(json!({
        "mode": "atlas",
        "scale": plan.scale,
        "startpos_bytes": bytes.len(),
        "counts": conv.counts,
        "changed": conv.changed,
        "sentinels_preserved": conv.sentinels,
        "zero_center_cai_regions": conv.zero_center_regions,
        "cached_grids": conv.cached_grids,
        "los": conv.los_stats,
        "hlci_relocated": conv.hlci_relocated,
        "hlci_deleted": conv.deleted_hlci,
        "region_geometry_kinds": contract.geometry_kinds,
        "camera_bounds": {"source": b(conv.src_bounds), "output": b(conv.new_bounds)},
        "script": script_report,
        "scaled_by_plan_scale": [
            "CAI_REGION_BOUNDARY length (not recomputed: ETW boundary lengths are not reproducible from region geometry)",
            "TRADE_ROUTES[] distances (hops between theatres scaled, not re-measured)",
            "TRADE_SEGMENTS curve length arrays/totals (similarity transform of Bezier control points)"
        ],
        "unchanged": [
            "entity and region handles", "faction capitals", "movement budgets", "angles", "model heights", "CAI property weights",
            "economic coefficients", "pathfinder topology, node maps, clearance floats and obstacle ownership", "sea-grid cell ids",
            "camera zoom limits and zoom factor", "camera height", "shroud trees (verified empty)"
        ],
        "verification": verification.to_json(),
    }))
}
