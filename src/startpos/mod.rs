//! `campaigns/wld_napoleon/startpos.esf` + `scripting.lua` generation (native Rust port of the accepted `startpos/*.py`).

pub mod cai;
pub mod context;
pub mod core;
pub mod factions;
pub mod forces;
pub mod helpers;

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::map::MapArtifacts;
use crate::model::{Progress, Rules};
use crate::snapshot::Snapshot;

/// Writes `campaigns/wld_napoleon/{startpos.esf,scripting.lua}` below `output_data`.
pub fn build(
    snapshot: &Snapshot,
    rules: &Rules,
    map: &MapArtifacts,
    added_keys: &BTreeSet<String>,
    output_data: &Path,
    progress: &Progress<'_>,
) -> Result<()> {
    build_with_notes(snapshot, rules, map, added_keys, output_data, progress).map(|_| ())
}

/// Same as `build`, additionally returning report notes (region conversion summary, relocated starting forces).
pub fn build_with_notes(
    snapshot: &Snapshot,
    rules: &Rules,
    map: &MapArtifacts,
    added_keys: &BTreeSet<String>,
    output_data: &Path,
    progress: &Progress<'_>,
) -> Result<Vec<String>> {
    // the pipeline edits the generated regions.esf (theatre extensions) before the start position is built: the camera
    // bounds follow its theatre rectangles
    let regions_path = output_data.join("campaign_maps/nap_world/regions.esf");
    let edited_regions = if regions_path.is_file() {
        let bytes = fs::read(&regions_path).with_context(|| format!("Read {}", regions_path.display()))?;
        Some(crate::esf::read(&bytes).with_context(|| format!("Parse {}", regions_path.display()))?)
    } else {
        None
    };
    let sea_path = output_data.join("campaign_maps/nap_world/sea_grids.esf");
    let sea_grids = crate::esf::read(&fs::read(&sea_path).with_context(|| format!("Read {}", sea_path.display()))?)
        .with_context(|| format!("Parse {}", sea_path.display()))?;
    let generated = self::core::generate(
        snapshot,
        rules,
        map,
        added_keys,
        edited_regions.as_ref().map(|e| &e.root),
        &sea_grids.root,
        progress,
    )?;
    let dir = output_data.join("campaigns").join(self::core::CAMPAIGN_KEY);
    fs::create_dir_all(&dir).with_context(|| format!("Create {}", dir.display()))?;
    fs::write(dir.join("startpos.esf"), &generated.startpos).context("Write startpos.esf")?;
    fs::write(dir.join("scripting.lua"), generated.lua.as_bytes())
        .context("Write scripting.lua")?;
    progress(&format!(
        "Startpos: wrote startpos.esf ({:.1} MB) and scripting.lua",
        generated.startpos.len() as f64 / 1e6
    ));
    Ok(generated.notes)
}
