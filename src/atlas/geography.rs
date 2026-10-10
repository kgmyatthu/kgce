//! Atlas geography: `regions.esf` (clipped political + groundtype partitions, regenerated indexes and every
//! chart-owned spatial section) and the campaign navigation files (pathfinding grids, POIs, sea grids, trade
//! routes) migrated together to the connected stock-Q20 atlas, with the camera rectangles synchronised.
//!
//! The pass runs on a source-space world (after region edits, settlement moves and startpos creation) and writes
//! every geography file once everything verified. The original source regions are returned so that the campaign
//! migration can remap area indexes.

mod assemble;
mod camera;
mod clip;
mod contract;
mod geom;
mod navigation;
mod query;
mod regions;
mod scaler;
mod source;
mod startpos_points;
mod verify;

pub use startpos_points::{OwnedPoint, cached_grid_charts, cached_polygons_by_grid, owned_fixed_points};

use crate::atlas::Plan;
use crate::esf::{self, Esf};
use crate::model::Progress;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Result of the geography pass.
pub struct Geography {
    /// The untouched source `regions.esf` (campaign area remapping reads its area/region structure).
    pub source_regions: Esf,
    /// `regions-index-contract.json`: source (region, area) -> output areas, interior points, backdrop.
    pub contract: Value,
    /// `q20-coordinate-overrides.json`: per-chart raw Q20 values that differ from the plain affine rule.
    pub point_overrides: Value,
    /// Factual validation report of the regions and navigation passes.
    pub report: Value,
}

const FILES: [&str; 5] = ["regions.esf", "pathfinding.esf", "poi.esf", "sea_grids.esf", "trade_routes.esf"];

/// Migrate `<data>/campaign_maps/nap_world/{regions,pathfinding,poi,sea_grids,trade_routes}.esf` in place.
pub fn transform(data: &Path, plan: &Plan, progress: &Progress<'_>) -> Result<Geography> {
    plan.validate()?;
    let world_dir = data.join("campaign_maps/nap_world");
    let regions_path = world_dir.join("regions.esf");
    let bytes = std::fs::read(&regions_path).with_context(|| format!("Read {}", regions_path.display()))?;
    let source_regions = esf::read(&bytes).with_context(|| format!("Parse {}", regions_path.display()))?;

    let regions = regions::transform(plan, &source_regions, progress)?;
    progress("Atlas navigation: migrating pathfinding, POI, sea grids and trade routes");
    let nav = navigation::transform(data, plan, &source_regions, &regions.esf, &regions.contract, progress)?;

    // serialise every file first, then replace them, so a failure never leaves a half-migrated world
    let outputs: [(&str, &Esf); 5] = [
        (FILES[0], &regions.esf),
        (FILES[1], &nav.pathfinding),
        (FILES[2], &nav.poi),
        (FILES[3], &nav.sea_grids),
        (FILES[4], &nav.trade_routes),
    ];
    let mut staged: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (name, file) in outputs {
        let bytes = esf::write(file).with_context(|| format!("Serialise the migrated {name}"))?;
        let back = esf::read(&bytes).with_context(|| format!("Re-parse the migrated {name}"))?;
        ensure!(back.root == file.root, "The migrated {name} does not parse back to the migrated tree");
        let target = world_dir.join(name);
        let tmp = world_dir.join(format!("{name}.atlas.tmp"));
        std::fs::write(&tmp, &bytes).with_context(|| format!("Write {}", tmp.display()))?;
        staged.push((tmp, target));
    }
    for (tmp, target) in &staged {
        std::fs::rename(tmp, target).with_context(|| format!("Replace {}", target.display()))?;
    }
    progress("Atlas geography: all five navigation and region files migrated");
    Ok(Geography {
        source_regions,
        contract: regions.contract,
        point_overrides: nav.overrides,
        report: json!({"regions": regions.report, "navigation": nav.report}),
    })
}
