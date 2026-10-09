use crate::{
    compatibility::{self, Prepared},
    esf, map,
    model::{BuildReport, Progress, Project, RegionActivation, RegionSplit, Rules},
    pack,
    regionmap::{
        apply::{AppliedActivation, AppliedSplit},
        chain,
        pathfinding::Grid,
        regions::RegionData,
        stages, startpos_split,
    },
    startpos,
};
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub fn assets_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    for ancestor in exe.ancestors().skip(1) {
        let assets = ancestor.join("assets");
        if assets.join("rules/schema_emp.ron").is_file() && assets.join("policy").is_dir() {
            return Ok(assets);
        }
    }
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    if local.join("rules/schema_emp.ron").is_file() && local.join("policy").is_dir() {
        return Ok(local);
    }
    bail!("Cannot find conversion recipes beside the editor. Keep the assets directory with the application; original game resources are rebuilt locally after import.")
}

pub fn default_workspace(assets: &Path) -> PathBuf {
    assets.parent().unwrap_or(Path::new(".")).join("workspace")
}

pub fn generate(
    workspace: &Path,
    assets: &Path,
    project: &Project,
    progress: &Progress<'_>,
) -> Result<BuildReport> {
    crate::safety::ensure_writable_path(&workspace.join("generated"))?;
    progress("Verifying source snapshots and target compatibility");
    let prepared = compatibility::prepare(workspace, assets, project, progress)?;
    let generated = workspace.join("generated");
    fs::create_dir_all(&generated)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let pending = generated.join(format!(".building-{nonce}"));
    fs::create_dir(&pending)?;
    let result = generate_into(&prepared, project, &pending, progress);
    let mut report = match result {
        Ok(report) => report,
        Err(error) => {
            // Keep diagnostic metadata, never publish a partial campaign as the finished output.
            let _ = fs::write(pending.join("BUILD_FAILED.txt"), format!("{error:#}\n"));
            return Err(error.context(format!(
                "Conversion did not replace your last output. Diagnostic directory: {}",
                pending.display()
            )));
        }
    };
    let final_dir = generated.join("ntw");
    let previous = generated.join(format!("previous-{nonce}"));
    if final_dir.exists() {
        fs::rename(&final_dir, &previous)?;
    }
    if let Err(error) = fs::rename(&pending, &final_dir) {
        if previous.exists() {
            let _ = fs::rename(&previous, &final_dir);
        }
        return Err(error.into());
    }
    report.output = final_dir.join(crate::export::ZIP_NAME);
    fs::write(
        final_dir.join("build_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    progress(&format!("NTW ZIP ready: {}", report.output.display()));
    Ok(report)
}

fn generate_into(
    prepared: &Prepared,
    project: &Project,
    destination: &Path,
    progress: &Progress<'_>,
) -> Result<BuildReport> {
    let source = &prepared.snapshot;
    if !prepared.pack_name.ends_with(".pack")
        || prepared.pack_name.contains(['/', '\\', ':'])
        || prepared.pack_name.starts_with('.')
    {
        bail!("Compatibility plan produced an unsafe output pack name");
    }
    let rules = &prepared.rules;
    check_added_owners(rules, &project.activations, &project.splits)?;
    let data = destination.join("data");
    fs::create_dir_all(&data)?;
    progress("Converting world geography and terrain from ETW originals");
    let mut map = map::build(source, rules, &data, &project.terrain, progress)?;
    let world_dir = data.join("campaign_maps/nap_world");
    // Format-9 chain: fixed theatre bounds, clipping, area selection / merge / carve. It always runs: the fixed
    // geometry applies to every build, with or without activations or splits.
    let mut notes: Vec<String> = Vec::new();
    progress("Applying the fixed theatre bounds and preparing the activation geometry");
    let stage = stages::geometry_stage(&world_dir, project, progress)?;
    notes.extend(stage.notes.iter().cloned());
    let applied_activations: Vec<AppliedActivation> = if project.activations.is_empty() {
        Vec::new()
    } else {
        progress(&format!(
            "Activating {} wilderness region(s) on the generated world map",
            project.activations.len()
        ));
        // with the regenerated pathfinding / sea grids the per-region grid cross-checks wait for the final validation
        crate::regionmap::apply::apply_activations_staged(
            &world_dir,
            &project.activations,
            rules,
            &mut map,
            false,
        )?
        .0
    };
    let applied: Vec<AppliedSplit> = if project.splits.is_empty() {
        Vec::new()
    } else {
        progress(&format!(
            "Applying {} region split(s) to the generated world map",
            project.splits.len()
        ));
        crate::regionmap::apply::apply_splits(&world_dir, &project.splits, rules, &mut map)?
    };
    // after the splits: a split of a region the synthesis has to path gets both halves synthesised together
    notes.extend(stages::synthesis_stage(
        &world_dir,
        project,
        &map,
        &applied_activations,
        &applied,
        &stage,
        progress,
    )?);
    let touched = stages::touched_theatres(&stage, &applied_activations, project);
    notes.extend(stages::sea_stage(&world_dir, project, &stage, &touched, progress)?);
    progress("Rebuilding Napoleon campaign state and ownership references");
    let added_keys: BTreeSet<String> = project
        .activations
        .iter()
        .map(|a| a.key.clone())
        .chain(project.splits.iter().map(|s| s.key.clone()))
        .collect();
    let mut warnings = startpos::build_with_notes(source, rules, &map, &added_keys, &data, progress)?;
    warnings.extend(notes);
    let startpos_path = data.join("campaigns/wld_napoleon/startpos.esf");
    if !applied_activations.is_empty() {
        warnings.extend(add_activated_regions(
            &startpos_path,
            &world_dir,
            rules,
            &project.activations,
            &applied_activations,
            progress,
        )?);
    }
    if !applied.is_empty() {
        warnings.extend(add_split_regions(
            &startpos_path,
            rules,
            &project.splits,
            &applied,
            progress,
        )?);
    }
    warnings.extend(rebuild_hlcis(&startpos_path, &world_dir, project, &stage, progress)?);
    progress("Checking the generated map and start position against each other");
    check_world(&world_dir, &startpos_path, project, &stage, &touched)?;
    progress("Encoding Napoleon database tables and localisation");
    let mut members = compatibility::build_members(prepared, &map, project, progress)?;
    // The campaign loader prefers loose files. A merged script must replace the
    // generated staging script, not be silently shadowed by it inside the pack.
    if let Some(script) = members
        .remove("campaigns\\wld_napoleon\\scripting.lua")
        .or_else(|| members.remove("campaigns/wld_napoleon/scripting.lua"))
    {
        fs::write(data.join("campaigns/wld_napoleon/scripting.lua"), script)?;
    }
    pack::write_pack(&data.join(&prepared.pack_name), &members, 4)?;
    progress("Inspecting generated campaign ownership and required output files");
    validate_output(&data, rules, &prepared.pack_name, &members)?;
    let metadata = export_metadata(prepared, project)?;
    for (name, bytes) in &metadata {
        fs::write(destination.join(name), bytes)?;
    }
    progress("Packaging and verifying the input-bound NTW ZIP");
    let files =
        crate::export::package(&data, &destination.join(crate::export::ZIP_NAME), &metadata)?;
    fs::remove_dir_all(&data)?;
    warnings.extend(prepared.analysis.warnings.iter().cloned());
    warnings.extend([
        "ZIP export only: the app never installs, copies into, or overwrites game files.",
        "This creates a new campaign start. Existing saves retain their own territory ownership.",
        "Gameplay compatibility is not verified by static validation; test a new campaign with the mod that supplied the imported tables.",
    ].map(str::to_owned));
    Ok(BuildReport {
        output: destination.into(),
        region_count: rules.regions.as_object().map_or(0, |m| m.len()),
        faction_count: rules.factions.as_object().map_or(0, |m| m.len()),
        ownership_changes: project.owners.len(),
        files,
        warnings,
    })
}

/// An activated or split region's owner must also hold a region of the ETW world: the start position builds each
/// faction's governorships and capital from those, and adds the activated and split regions afterwards.
fn check_added_owners(rules: &Rules, activations: &[RegionActivation], splits: &[RegionSplit]) -> Result<()> {
    let regions = rules.regions.as_object().context("Invalid region rules")?;
    let added = |key: &str| activations.iter().any(|a| a.key == key) || splits.iter().any(|s| s.key == key);
    let keys = activations
        .iter()
        .map(|a| ("Activated", a.key.as_str()))
        .chain(splits.iter().map(|s| ("Split", s.key.as_str())));
    for (kind, key) in keys {
        let owner = regions
            .get(key)
            .and_then(|r| r["owner"].as_str())
            .with_context(|| format!("{kind} region {key} has no owner in the rules"))?;
        let holds_other = regions
            .iter()
            .any(|(other, r)| r["owner"].as_str() == Some(owner) && !added(other));
        if !holds_other {
            bail!(
                "{kind} region {key} is owned by {owner}, which holds no other region. Give the new region to a faction that owns land, or give {owner} a region first.",
            );
        }
    }
    Ok(())
}

/// Add one region to the start position in `file` and verify the result: the edited file is re-read from its
/// serialised form and checked against the state before it; any problem aborts the build.
fn add_verified(
    file: &mut esf::Esf,
    spec: &startpos_split::NewRegion,
    label: &str,
) -> Result<startpos_split::Report> {
    let before = file.root.clone();
    let report = startpos_split::add_region(&mut file.root, spec)
        .with_context(|| format!("{label}: adding the region to the start position"))?;
    let bytes = esf::write(file)?;
    *file = esf::read(&bytes)?;
    let problems = startpos_split::check(&before, &file.root, spec)?;
    if !problems.is_empty() {
        bail!(
            "{label}: the edited start position is inconsistent ({} problems):\n  {}",
            problems.len(),
            problems.iter().take(20).cloned().collect::<Vec<_>>().join("\n  ")
        );
    }
    Ok(report)
}

/// Add every applied activation to the generated start position, in project order (before any split, which may cut an
/// activated region).
fn add_activated_regions(
    path: &Path,
    world_dir: &Path,
    rules: &Rules,
    activations: &[RegionActivation],
    applied: &[AppliedActivation],
    progress: &Progress<'_>,
) -> Result<Vec<String>> {
    if activations.len() != applied.len() {
        bail!(
            "{} region activations were requested but the map applied {}",
            activations.len(),
            applied.len()
        );
    }
    let mut file = esf::read(&fs::read(path).with_context(|| format!("Read {}", path.display()))?)?;
    // the carved regions' neighbours and the boundaries of merged ones come from the final geometry
    let regions = esf::read(&fs::read(world_dir.join("regions.esf")).context("Read the generated regions.esf")?)?;
    let rd = RegionData::parse(&regions.root)?;
    let mut notes = Vec::new();
    for (activation, ap) in activations.iter().zip(applied) {
        progress(&format!(
            "Adding activated region {} (template {}) to the start position",
            activation.key, activation.donor
        ));
        let label = format!("Activation {}", activation.key);
        let spec = if activation.carve_from.is_some() {
            startpos_split::new_region_for_carve(&file.root, &rd, activation, ap, rules)
        } else {
            startpos_split::new_region_for_activation(&file.root, activation, ap, rules)
        }
        .with_context(|| label.clone())?;
        let report = add_verified(&mut file, &spec, &label)?;
        notes.push(format!(
            "Region activation {} <- {} ({}): owner {}, settlement index {}, {} CAI ids, {} neighbours; start position consistency check passed",
            activation.key,
            activation.donor,
            ap.settlement_key,
            report.owner,
            report.settlement_index,
            report.cai_range.1 - report.cai_range.0 + 1,
            spec.neighbours.len()
        ));
        notes.extend(report.notes.iter().map(|n| format!("{label}: {n}")));
        let added = startpos_split::ensure_boundaries(&mut file.root, &rd, &activation.key)
            .with_context(|| format!("{label}: CAI boundaries"))?;
        notes.extend(added.into_iter().map(|n| format!("{label}: {n}")));
    }
    fs::write(path, esf::write(&file)?).with_context(|| format!("Write {}", path.display()))?;
    Ok(notes)
}

/// Rebuilds the `CAI_REGION_HLCI` entries of every region whose areas the chain changed (activated, carved, split
/// and clipped regions): the start position needs exactly one per `regions.esf` area.
fn rebuild_hlcis(
    path: &Path,
    world_dir: &Path,
    project: &Project,
    stage: &stages::GeometryStage,
    progress: &Progress<'_>,
) -> Result<Vec<String>> {
    let regions = esf::read(&fs::read(world_dir.join("regions.esf")).context("Read the generated regions.esf")?)?;
    let rd = RegionData::parse(&regions.root)?;
    let always: Vec<String> = project
        .activations
        .iter()
        .map(|a| a.key.clone())
        .chain(project.splits.iter().map(|s| s.key.clone()))
        .collect();
    let keys = chain::regions_with_changed_areas(&stage.base, &rd, &always);
    progress(&format!("Rebuilding the region HLCIs of {} region(s)", keys.len()));
    let mut file = esf::read(&fs::read(path).with_context(|| format!("Read {}", path.display()))?)?;
    let report = startpos_split::rebuild_region_hlcis(&mut file.root, &rd, &keys)?;
    // every other region whose HLCI count still differs from its area count (the converted vanilla start position has
    // some): the engine rejects any mismatch, so they are repaired as well
    let rest = startpos_split::rebuild_region_hlcis(&mut file.root, &rd, &[])?;
    fs::write(path, esf::write(&file)?).with_context(|| format!("Write {}", path.display()))?;
    let mut notes = vec![format!(
        "Region HLCIs rebuilt for {} region(s): {} created, {} removed, {} in total",
        report.regions.len(),
        report.created.len(),
        report.removed.len(),
        report.total
    )];
    if !rest.regions.is_empty() {
        notes.push(format!(
            "Region HLCIs repaired in {} further region(s) whose count differed from their area count (the validated in-game start position tolerated those deficits; the added HLCIs are not game-tested): {}",
            rest.regions.len(),
            rest.regions.iter().map(|r| r.key.as_str()).collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(notes)
}

/// Cross-file consistency of the finished map files and start position (the validation pass of the chain, run for
/// every build): pathfinding lists, sea grid lists, routes, HLCI counts per region, CAI sea cells. Any problem
/// aborts the build.
fn check_world(
    world_dir: &Path,
    startpos_path: &Path,
    project: &Project,
    stage: &stages::GeometryStage,
    touched: &BTreeSet<String>,
) -> Result<()> {
    let mut problems = stages::validate_world(world_dir, project, stage, touched)?;
    let regions = esf::read(&fs::read(world_dir.join("regions.esf")).context("Read the generated regions.esf")?)?;
    let rd = RegionData::parse(&regions.root)?;
    let pf = esf::read(&fs::read(world_dir.join("pathfinding.esf")).context("Read the generated pathfinding.esf")?)?;
    let mut lists: Vec<Vec<i16>> = Vec::new();
    for area in 0..Grid::area_count(&pf.root)? {
        lists.push(Grid::parse_area(&pf.root, area)?.region_ids);
    }
    let cells: usize = stages::sea_cell_counts(world_dir)?.iter().map(|(_, n)| n).sum();
    let checks = startpos_split::MapChecks::new(&rd, &regions.root, &lists, Some(cells))?;
    let start = esf::read(&fs::read(startpos_path).context("Read the generated startpos.esf")?)?;
    problems.extend(startpos_split::check_world(&start.root, &checks)?);
    if problems.is_empty() {
        return Ok(());
    }
    let listing = problems.iter().take(40).cloned().collect::<Vec<_>>().join("\n  ");
    bail!("The generated campaign is inconsistent ({} problems):\n  {listing}", problems.len())
}

/// Add every applied split to the generated start position, in project order. Each addition is re-read from its
/// serialised form and verified against the state before it; any problem aborts the build.
fn add_split_regions(
    path: &Path,
    rules: &Rules,
    splits: &[RegionSplit],
    applied: &[AppliedSplit],
    progress: &Progress<'_>,
) -> Result<Vec<String>> {
    if splits.len() != applied.len() {
        bail!(
            "{} region splits were requested but the map applied {}",
            splits.len(),
            applied.len()
        );
    }
    let mut file = esf::read(&fs::read(path).with_context(|| format!("Read {}", path.display()))?)?;
    let mut notes = Vec::new();
    for (split, ap) in splits.iter().zip(applied) {
        progress(&format!(
            "Adding region {} (split of {}) to the start position",
            split.key, split.source
        ));
        let expected = startpos_split::cai_region_count(&file.root)?;
        if ap.new_index != expected {
            bail!(
                "Split {}: the map region index is {} but the start position's next region index is {expected}",
                split.key,
                ap.new_index
            );
        }
        let spec = startpos_split::new_region_for(&file.root, split, ap, rules)
            .with_context(|| format!("Split {}", split.key))?;
        let report = add_verified(&mut file, &spec, &format!("Split {}", split.key))?;
        notes.push(format!(
            "Region split {} <- {} ({}): owner {}, settlement index {}, {} CAI ids, {} neighbours; start position consistency check passed",
            split.key,
            split.source,
            ap.settlement_key,
            report.owner,
            report.settlement_index,
            report.cai_range.1 - report.cai_range.0 + 1,
            spec.neighbours.len()
        ));
        notes.extend(report.notes.iter().map(|n| format!("Split {}: {n}", split.key)));
    }
    fs::write(path, esf::write(&file)?).with_context(|| format!("Write {}", path.display()))?;
    Ok(notes)
}

fn export_metadata(prepared: &Prepared, project: &Project) -> Result<BTreeMap<String, Vec<u8>>> {
    let baseline_hash = hex::encode(Sha256::digest(serde_json::to_vec(
        prepared.snapshot.manifest(),
    )?));
    let manifest = serde_json::json!({
        "format": "kgc-compatibility-export",
        "version": 2,
        "generator_version": env!("CARGO_PKG_VERSION"),
        "target_mode": project.target_mode,
        "target": project.target,
        "adapter": project.adapter,
        "source_snapshot_sha256": baseline_hash,
        "dependencies": prepared.dependencies,
        "analysis": prepared.analysis,
        "pack_file": prepared.pack_name,
        "manual_installation_only": true,
        "gameplay_tested": false,
    });
    let instructions = format!(
        "KGC WORLD CAMPAIGN — MANUAL INSTALLATION\n\
         Target: {}\nLanguage: {}\nPack: {}\n\n\
         1. For TSV targets, use the mod setup that supplied the imported tables. compatibility.json records the TSV inputs, not an audit of installed mod packs. Models, textures and other mod assets are not bundled.\n\
         2. The generated pack contains the imported target tables and converted campaign data. The world campaign uses only factions that already exist in your target database; no new faction is added. Ensure the pack overrides the corresponding database members in your mod setup. TSV import cannot discover or configure game load order.\n\
         3. Back up any earlier world-campaign export. Do not combine exports built from different target datasets.\n\
         4. Drag only the ZIP's data folder into the Napoleon folder containing Napoleon.exe. Do not put it inside the existing data folder.\n\
         5. Keep kgc-export outside the game; it records this project's inputs and compatibility decisions.\n\
         6. Start a NEW campaign and test the exact mod setup. Static validation is not gameplay verification.\n\n\
         The editor never installs, changes pack activation, or overwrites game files. Any replacement during manual installation is performed by your file manager.\n",
        prepared.analysis.target_name, prepared.analysis.language, prepared.pack_name,
    );
    Ok(BTreeMap::from([
        (
            "compatibility.json".into(),
            serde_json::to_vec_pretty(&manifest)?,
        ),
        ("project.json".into(), serde_json::to_vec_pretty(project)?),
        ("INSTALL.txt".into(), instructions.into_bytes()),
    ]))
}

pub fn inspect_owners(path: &Path) -> Result<BTreeMap<String, String>> {
    let file = esf::read(&fs::read(path)?)?;
    let world = file
        .root
        .path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "WORLD"])?;
    let mut factions = BTreeMap::new();
    for item in world.get("FACTION_ARRAY")?.items()? {
        let record = item
            .iter()
            .find(|n| n.name() == Some("FACTION"))
            .context("Faction record missing")?;
        let values: Vec<_> = record
            .children()?
            .iter()
            .filter(|n| n.code().is_some())
            .collect();
        let id = values.first().context("Faction ID missing")?.as_i32()? as u32;
        let key = values
            .iter()
            .find_map(|n| n.as_str().ok())
            .context("Faction key missing")?;
        factions.insert(id, key.to_string());
    }
    let mut owners = BTreeMap::new();
    for item in world.path(&["REGION_MANAGER", "REGIONS_ARRAY"])?.items()? {
        let record = item
            .iter()
            .find(|n| n.name() == Some("REGION"))
            .context("Region record missing")?;
        let key = record
            .children()?
            .iter()
            .find_map(|n| n.as_str().ok())
            .context("Region key missing")?;
        let owner_id = record
            .children()?
            .get(20)
            .context("Region owner missing")?
            .as_u32()?;
        owners.insert(
            key.to_string(),
            factions
                .get(&owner_id)
                .with_context(|| format!("Region {key} has unresolved owner {owner_id}"))?
                .clone(),
        );
    }
    Ok(owners)
}

fn validate_output(
    data: &Path,
    rules: &Rules,
    pack_name: &str,
    members: &BTreeMap<String, Vec<u8>>,
) -> Result<()> {
    for name in [
        "regions.esf",
        "pathfinding.esf",
        "poi.esf",
        "sea_grids.esf",
        "trade_routes.esf",
        "metadata.dat",
        "display/heightmap/heightmap.tga",
        "display/supertexture/supertexture.stpi",
        "display/supertexture/supertexture.stpd",
    ] {
        if !data.join("campaign_maps/nap_world").join(name).is_file() {
            bail!("Required generated map file is missing: {name}");
        }
    }
    let owners = inspect_owners(&data.join("campaigns/wld_napoleon/startpos.esf"))?;
    let expected = rules.regions.as_object().context("Invalid region rules")?;
    if owners.len() != expected.len() {
        bail!(
            "Generated {} regions; expected {}",
            owners.len(),
            expected.len()
        );
    }
    for (key, info) in expected {
        if owners.get(key).map(String::as_str) != info["owner"].as_str() {
            bail!("Generated ownership of {key} differs from the edited project");
        }
    }
    let pack = pack::Pack::open(&data.join(pack_name))?;
    if pack.entries.len() != members.len() {
        bail!("Generated pack member count differs from the compatibility plan");
    }
    for (name, bytes) in members {
        let entry = pack
            .find(name)
            .with_context(|| format!("Generated pack is missing {name}"))?;
        if entry.size as usize != bytes.len() {
            bail!("Generated pack has an incorrect size for {name}");
        }
    }
    Ok(())
}
