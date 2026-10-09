//! Project edits shared by the native UI and the CLI: dataset selection, mapping and resolution
//! changes are validated against the freshest [`Analysis`] (effective target catalog, issue input
//! hashes) and export gating is decided in one place so neither front end can bypass blocking
//! compatibility or export a vanilla campaign while TSV mode has no dataset.

use crate::{
    compatibility::{Analysis, Issue},
    model::{Mappings, Project, Resolution, Rules, TargetMode},
    target::TargetBinding,
};
use anyhow::{Result, bail};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const MAPPING_FIELDS: [&str; 4] = ["factions", "units", "buildings", "faction_templates"];

pub fn mapping_title(field: &str) -> &'static str {
    match field {
        "factions" => "Factions",
        "units" => "Units",
        "buildings" => "Buildings",
        "faction_templates" => "Faction donors",
        _ => "Mapping",
    }
}

pub fn mapping_ref<'a>(
    mappings: &'a Mappings,
    field: &str,
) -> Option<&'a BTreeMap<String, String>> {
    match field {
        "factions" => Some(&mappings.factions),
        "units" => Some(&mappings.units),
        "buildings" => Some(&mappings.buildings),
        "faction_templates" => Some(&mappings.faction_templates),
        _ => None,
    }
}

pub fn mapping_mut<'a>(
    mappings: &'a mut Mappings,
    field: &str,
) -> Option<&'a mut BTreeMap<String, String>> {
    match field {
        "factions" => Some(&mut mappings.factions),
        "units" => Some(&mut mappings.units),
        "buildings" => Some(&mut mappings.buildings),
        "faction_templates" => Some(&mut mappings.faction_templates),
        _ => None,
    }
}

/// Valid mapping *source* keys: port-side ETW rule keys, or, for donor assignments, only factions that exist in the
/// imported target catalog (absent original faction keys are never offered, so no identity can be introduced).
pub fn source_options(analysis: &Analysis, field: &str) -> Vec<String> {
    let catalog = &analysis.catalog;
    match field {
        "factions" => catalog.source_factions.clone(),
        "units" => catalog.source_units.clone(),
        "buildings" => catalog.source_buildings.clone(),
        "faction_templates" => catalog.factions.clone(),
        _ => Vec::new(),
    }
}

/// Valid mapping *values* from the effective target catalog (the imported TSV tables).
pub fn target_options(analysis: &Analysis, field: &str) -> Vec<String> {
    let catalog = &analysis.catalog;

    match field {
        "factions" => catalog.factions.clone(),
        "units" => catalog.units.clone(),
        "buildings" => catalog.buildings.clone(),
        "faction_templates" => catalog.donor_factions.clone(),
        _ => Vec::new(),
    }
}

/// Validates against the effective catalog and records the mapping.
pub fn set_mapping(
    project: &mut Project,
    analysis: &Analysis,
    field: &str,
    from: &str,
    to: &str,
) -> Result<()> {
    if !MAPPING_FIELDS.contains(&field) {
        bail!(
            "Unknown mapping kind {field}; use one of {}",
            MAPPING_FIELDS.join(", ")
        );
    }
    if from.trim().is_empty() || to.trim().is_empty() {
        bail!("Mapping keys cannot be empty");
    }
    let targets = target_options(analysis, field);
    if targets.is_empty() {
        bail!(
            "The effective target catalog has no {} entries; import and select a complete TSV dataset and run analyze first",
            mapping_title(field).to_lowercase()
        );
    }
    if !targets.iter().any(|key| key == to) {
        bail!(
            "{to} is not a valid {} target in the effective catalog (use `mapping search`)",
            mapping_title(field).to_lowercase()
        );
    }
    let sources = source_options(analysis, field);
    // Donor assignments name factions that exist in the imported target; an empty list must not accept anything.
    let donor_field = field == "faction_templates";
    if (donor_field || !sources.is_empty()) && !sources.iter().any(|key| key == from) {
        bail!(
            "{from} is not a known {} source key (use `mapping search`)",
            mapping_title(field).to_lowercase()
        );
    }
    mapping_mut(&mut project.mappings, field)
        .expect("validated field")
        .insert(from.into(), to.into());
    Ok(())
}

pub fn remove_mapping(project: &mut Project, field: &str, from: &str) -> Result<bool> {
    let Some(map) = mapping_mut(&mut project.mappings, field) else {
        bail!(
            "Unknown mapping kind {field}; use one of {}",
            MAPPING_FIELDS.join(", ")
        );
    };
    Ok(map.remove(from).is_some())
}

pub fn mapping_count(project: &Project) -> usize {
    MAPPING_FIELDS
        .iter()
        .filter_map(|field| mapping_ref(&project.mappings, field))
        .map(BTreeMap::len)
        .sum()
}

/// Switches the project to TSV mode without choosing a dataset: export stays blocked until one is
/// pinned, so selecting the mode can never silently export vanilla data.
pub fn select_tsv_mode(project: &mut Project) {
    project.target_mode = TargetMode::Tsv;
}

/// Pins the project to one immutable dataset (and switches to TSV mode). A merged adapter is only
/// valid for the exact dataset fingerprint it was imported against, so it is dropped on change.
pub fn pin_target(project: &mut Project, binding: TargetBinding) {
    project.target_mode = TargetMode::Tsv;
    if project.target.as_ref() != Some(&binding) {
        if project
            .adapter
            .as_ref()
            .is_some_and(|adapter| adapter.target_fingerprint != binding.fingerprint)
        {
            project.adapter = None;
        }
        project.target = Some(binding);
    }
}

pub fn use_vanilla(project: &mut Project) {
    project.target_mode = TargetMode::Vanilla;
    project.target = None;
    project.adapter = None;
}

pub fn find_issue<'a>(analysis: &'a Analysis, id: &str) -> Option<&'a Issue> {
    analysis.issues.iter().find(|issue| issue.id == id)
}

/// Records a resolution bound to the issue's current input hash.
pub fn apply_resolution(
    project: &mut Project,
    analysis: &Analysis,
    id: &str,
    choice: &str,
    expected_hash: Option<&str>,
) -> Result<()> {
    let Some(issue) = find_issue(analysis, id) else {
        bail!("No issue {id} in the current analysis (it may be stale; re-run analyze)");
    };
    if let Some(expected) = expected_hash {
        if expected != issue.input_hash {
            bail!(
                "Issue {id} inputs changed (expected hash {expected}, current {}); review it again",
                issue.input_hash
            );
        }
    }
    if issue.choices.is_empty() {
        bail!("Issue {id} has no selectable resolution: {}", issue.title);
    }
    if !issue.choices.iter().any(|candidate| candidate == choice) {
        bail!(
            "{choice} is not a valid choice for {id}; valid: {}",
            issue.choices.join(", ")
        );
    }
    project.resolutions.insert(
        id.into(),
        Resolution {
            input_hash: issue.input_hash.clone(),
            choice: choice.into(),
        },
    );
    Ok(())
}

/// Saved resolutions that no longer match any current issue or whose inputs changed: (issue id, reason).
pub fn stale_resolutions(project: &Project, analysis: &Analysis) -> Vec<(String, String)> {
    project
        .resolutions
        .iter()
        .filter_map(|(id, resolution)| match find_issue(analysis, id) {
            None => Some((id.clone(), "issue no longer exists".into())),
            Some(issue) if issue.input_hash != resolution.input_hash => Some((
                id.clone(),
                "issue inputs changed since the choice was made".into(),
            )),
            Some(issue) if !issue.choices.contains(&resolution.choice) => {
                Some((id.clone(), "choice is no longer offered".into()))
            }
            Some(_) => None,
        })
        .collect()
}

pub fn blockers(analysis: &Analysis) -> Vec<&Issue> {
    analysis
        .issues
        .iter()
        .filter(|issue| issue.severity == "blocker" && !issue.resolved)
        .collect()
}

/// Number of unresolved blocking issues, without collecting them.
pub fn blocker_count(analysis: &Analysis) -> usize {
    analysis
        .issues
        .iter()
        .filter(|issue| issue.severity == "blocker" && !issue.resolved)
        .count()
}

/// Content hash of everything that affects analysis; `name` is cosmetic.
pub fn signature(project: &Project) -> String {
    let mut value = serde_json::to_value(project).unwrap_or_default();
    if let Some(object) = value.as_object_mut() {
        object.remove("name");
    }
    hex::encode(Sha256::digest(value.to_string().as_bytes()))
}

/// TSV mode needs an imported dataset; this is checked before any analysis so the user is told what to do.
pub fn dataset_blocker(project: &Project) -> Option<&'static str> {
    (project.target_mode == TargetMode::Tsv && project.target.is_none())
        .then_some("TSV mode has no dataset: import RPFM TSV tables and select the dataset first.")
}

/// Regions whose original owner is unavailable in the imported target and that the project has still not assigned.
/// They are not issues: the analysis stops before the ownership-dependent checks until each has an existing target
/// faction.
pub fn pending_regions<'a>(
    project: &'a Project,
    analysis: &'a Analysis,
) -> impl Iterator<Item = &'a str> + 'a {
    analysis
        .unassigned_regions
        .iter()
        .map(String::as_str)
        .filter(|region| !project.owners.contains_key(*region))
}

pub fn pending_count(project: &Project, analysis: &Analysis) -> usize {
    pending_regions(project, analysis).count()
}

/// Membership query without allocation (`unassigned_regions` is sorted by the analysis).
pub fn is_pending(project: &Project, analysis: &Analysis, region: &str) -> bool {
    !project.owners.contains_key(region)
        && analysis
            .unassigned_regions
            .binary_search_by(|key| key.as_str().cmp(region))
            .is_ok()
}

/// What an analysis says about the project, without ever reading "not ready" as "ready" or as "0 blockers".
#[derive(Debug, PartialEq, Eq)]
pub enum Standing {
    /// Unresolved blocking issues (pending ownership may exist on top of them).
    Blocked(usize),
    /// No blocking issue, but this many regions still wait for an existing target faction.
    PendingOwnership(usize),
    /// Not ready, with neither blocking issues nor pending regions to explain it.
    NotReady,
    Ready,
}

pub fn standing(project: &Project, analysis: &Analysis) -> Standing {
    let open = blocker_count(analysis);
    if open > 0 {
        return Standing::Blocked(open);
    }
    let pending = pending_count(project, analysis);
    if pending > 0 {
        Standing::PendingOwnership(pending)
    } else if analysis.ready {
        Standing::Ready
    } else {
        Standing::NotReady
    }
}

/// Explains regions awaiting a target owner: what is needed, where, and that the remaining checks wait for it.
pub fn pending_message(count: usize) -> String {
    format!(
        "{count} region(s) await a target owner: their original owner is unavailable in the imported target. Assign an existing target faction (Ownership map → Assign faction to selected); ownership-dependent compatibility checks resume after assignment and re-analysis."
    )
}

/// Why the ZIP export is unavailable, or `None` when the analysis is current and unblocked.
pub fn export_blocker(
    project: &Project,
    analysis: Option<&Analysis>,
    analyzed_signature: Option<&str>,
) -> Option<String> {
    if let Some(reason) = dataset_blocker(project) {
        return Some(reason.into());
    }
    let Some(analysis) = analysis else {
        return Some("Run analysis first.".into());
    };
    if analyzed_signature != Some(signature(project).as_str()) {
        return Some("Project changed since the last analysis; re-run analysis.".into());
    }
    match standing(project, analysis) {
        Standing::Ready => None,
        Standing::Blocked(open) => {
            let mut reason = format!(
                "{open} unresolved blocking issue(s); resolve them in Compatibility review."
            );
            let pending = pending_count(project, analysis);
            if pending > 0 {
                reason.push(' ');
                reason.push_str(&pending_message(pending));
            }
            Some(reason)
        }
        Standing::PendingOwnership(pending) => Some(pending_message(pending)),
        Standing::NotReady => Some(
            "The analysis is not ready although it reports no blocking issue or pending ownership; re-run analysis."
                .into(),
        ),
    }
}

/// The current analysis is held back by nothing except regions awaiting an owner: front ends show that as a pending
/// step (amber) rather than as a failure.
pub fn only_pending_ownership(
    project: &Project,
    analysis: Option<&Analysis>,
    analyzed_signature: Option<&str>,
) -> bool {
    dataset_blocker(project).is_none()
        && analyzed_signature == Some(signature(project).as_str())
        && analysis.is_some_and(|analysis| {
            matches!(standing(project, analysis), Standing::PendingOwnership(_))
        })
}

/// Name of the fixed ownership behaviour shown by the GUI and the CLI.
pub const OWNERSHIP_LABEL: &str = "Existing factions only";

/// Shown in the Import TSV and Ownership map panels and by the CLI: what ownership does and, as importantly, what it
/// does not do.
pub const OWNERSHIP_HELP: &str = "Creates no new faction keys: you assign each region to a faction that already exists in the target database. Starting forces of factions that are removed are NOT transferred to the new owner of their territory, and target-only recipients get no invented army.";

/// The 1805 owner of `region` with the project's faction mappings applied (what an unassigned region resolves to).
pub fn baseline_owner<'a>(project: &'a Project, rules: &'a Rules, region: &str) -> &'a str {
    let base = rules
        .regions
        .get(region)
        .and_then(|region| region["owner"].as_str())
        .unwrap_or("");
    project
        .mappings
        .factions
        .get(base)
        .map(String::as_str)
        .unwrap_or(base)
}

/// The owner a region currently shows: its override, an exact target faction key that is never re-translated, or the
/// 1805 owner with the faction mappings applied.
pub fn effective_owner<'a>(project: &'a Project, rules: &'a Rules, region: &str) -> &'a str {
    match project.owners.get(region) {
        Some(owner) => owner,
        None => baseline_owner(project, rules, region),
    }
}

/// Faction keys of the target catalog, sorted: the only owner choices. Every faction of the effective target database
/// is offered (including ones no region uses yet) and nothing at all until that catalog is known: choices are never
/// guessed.
pub fn owner_choices(analysis: Option<&Analysis>) -> Vec<String> {
    let Some(analysis) = analysis else {
        return Vec::new();
    };
    let mut keys = analysis.catalog.factions.clone();
    keys.sort();
    keys.dedup();
    keys
}

/// Identifies the inputs that decide which factions exist in the target (mode, dataset, adapter). An
/// analysis made under a different key has a catalog that must not feed owner choices.
pub fn catalog_key(project: &Project) -> String {
    let value = json!({
        "mode": project.target_mode,
        "target": project.target,
        "adapter": project.adapter,
    });
    hex::encode(Sha256::digest(value.to_string().as_bytes()))
}

/// Owners must exist in the *actual* target database; `analysis` only has to carry the catalog (blocking
/// issues, including missing owners, do not matter).
fn check_existing_target_faction(analysis: Option<&Analysis>, faction: &str) -> Result<()> {
    let Some(analysis) = analysis else {
        bail!(
            "The target faction catalog is not loaded yet; run analysis first (choices appear once the catalog is known)"
        );
    };
    if analysis.catalog.factions.is_empty() {
        bail!(
            "The effective target catalog has no factions; import and select a complete TSV dataset (or fix the reported table problems) and analyze first"
        );
    }
    if !analysis.catalog.factions.iter().any(|key| key == faction) {
        bail!(
            "{faction} is not an existing faction of the target database; no new faction keys are created (use `mapping search --kind factions --side target` to list them)"
        );
    }
    Ok(())
}

/// Assigns (or, when it equals the baseline, un-assigns) one region; shared by the GUI and `assign`. Ownership needs
/// the analysis only for its target catalog, not an export-ready one. The project is untouched on error.
pub fn assign_owner(
    project: &mut Project,
    rules: &Rules,
    analysis: Option<&Analysis>,
    region: &str,
    faction: &str,
) -> Result<()> {
    if rules.regions.get(region).is_none() {
        bail!("Unknown region {region}");
    }
    check_existing_target_faction(analysis, faction)?;
    let mut next = project.clone();
    if baseline_owner(project, rules, region) == faction {
        next.owners.remove(region);
    } else {
        next.owners.insert(region.into(), faction.into());
    }
    validate_owners(&next, rules)?;
    *project = next;
    Ok(())
}

/// The region of an explicit owner override that is not an existing target faction (id
/// `ownership:<region>:invalid_owner`). Regions awaiting an owner are not issues; see [`pending_regions`].
pub fn owner_blocker_region(issue: &Issue) -> Option<&str> {
    issue
        .id
        .strip_prefix("ownership:")?
        .strip_suffix(":invalid_owner")
}

/// Structural check of the owner overrides: regions exist and keys are non-empty. Membership in the target database
/// is only known to the analysis, which reports owners it cannot find as blockers, so saving and loading never
/// depend on the catalog being loaded.
pub fn validate_owners(project: &Project, rules: &Rules) -> Result<()> {
    for (region, faction) in &project.owners {
        if rules.regions.get(region).is_none() {
            bail!("Unknown region {region}");
        }
        if faction.trim().is_empty() {
            bail!("Empty faction assigned to {region}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compatibility::Catalog, target::AdapterBinding};

    fn issue(id: &str, hash: &str, severity: &str, choices: &[&str], resolved: bool) -> Issue {
        Issue {
            id: id.into(),
            input_hash: hash.into(),
            severity: severity.into(),
            choices: choices.iter().map(|c| c.to_string()).collect(),
            resolved,
            ..Default::default()
        }
    }

    #[test]
    fn owner_validation_is_structural_and_accepts_target_only_factions() {
        let rules = world_rules();
        let mut project = Project::default();
        // Membership in the target database is the analysis's job: saving and loading never need the catalog.
        project
            .owners
            .insert("silesia".into(), "target_only_faction".into());
        validate_owners(&project, &rules).unwrap();
        project.owners.insert("nowhere".into(), "france".into());
        assert!(validate_owners(&project, &rules).is_err());
        project.owners.remove("nowhere");
        project.owners.insert("silesia".into(), "  ".into());
        assert!(validate_owners(&project, &rules).is_err());
    }

    #[test]
    fn resolution_is_bound_to_issue_inputs_and_offered_choices() {
        let analysis = Analysis {
            issues: vec![issue(
                "db:a",
                "h1",
                "blocker",
                &["keep_target", "use_port"],
                false,
            )],
            ..Default::default()
        };
        let mut project = Project::default();
        assert!(apply_resolution(&mut project, &analysis, "db:a", "bogus", None).is_err());
        assert!(
            apply_resolution(&mut project, &analysis, "db:a", "use_port", Some("old")).is_err()
        );
        apply_resolution(&mut project, &analysis, "db:a", "use_port", Some("h1")).unwrap();
        assert_eq!(project.resolutions["db:a"].input_hash, "h1");
        let changed = Analysis {
            issues: vec![issue("db:a", "h2", "blocker", &["keep_target"], false)],
            ..Default::default()
        };
        assert_eq!(stale_resolutions(&project, &changed).len(), 1);
        assert!(stale_resolutions(&project, &analysis).is_empty());
    }

    #[test]
    fn mappings_are_checked_against_the_effective_catalog() {
        let analysis = Analysis {
            catalog: Catalog {
                factions: vec!["mod_france".into()],
                source_factions: vec!["france".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut project = Project::default();
        assert!(set_mapping(&mut project, &analysis, "factions", "france", "nowhere").is_err());
        assert!(set_mapping(&mut project, &analysis, "factions", "ghost", "mod_france").is_err());
        assert!(set_mapping(&mut project, &analysis, "units", "x", "y").is_err());
        set_mapping(&mut project, &analysis, "factions", "france", "mod_france").unwrap();
        assert_eq!(project.mappings.factions["france"], "mod_france");
    }

    #[test]
    fn donor_assignments_only_name_factions_imported_with_the_target() {
        let analysis = Analysis {
            catalog: Catalog {
                factions: vec!["mod_france".into()],
                source_factions: vec!["france".into(), "mod_france".into()],
                donor_factions: vec!["mod_france".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(source_options(&analysis, "faction_templates"), ["mod_france"]);
        let mut project = Project::default();
        // An original ETW faction that the target lacks is never a valid donor-assignment source.
        assert!(
            set_mapping(&mut project, &analysis, "faction_templates", "france", "mod_france").is_err()
        );
        assert!(project.mappings.faction_templates.is_empty());
        set_mapping(&mut project, &analysis, "faction_templates", "mod_france", "mod_france").unwrap();
        assert_eq!(project.mappings.faction_templates["mod_france"], "mod_france");
        // An empty imported catalog accepts no donor source at all.
        let empty = Analysis {
            catalog: Catalog {
                source_factions: vec!["france".into()],
                donor_factions: vec!["mod_france".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            set_mapping(&mut project, &empty, "faction_templates", "france", "mod_france").is_err()
        );
    }

    #[test]
    fn export_requires_a_current_unblocked_analysis() {
        let project = Project::default();
        let sig = signature(&project);
        let ready = Analysis {
            ready: true,
            ..Default::default()
        };
        assert!(export_blocker(&project, None, None).is_some());
        assert!(export_blocker(&project, Some(&ready), Some("stale")).is_some());
        assert!(export_blocker(&project, Some(&ready), Some(&sig)).is_none());
        let blocked = Analysis {
            issues: vec![issue("ui", "h", "blocker", &[], false)],
            ready: false,
            ..Default::default()
        };
        assert!(export_blocker(&project, Some(&blocked), Some(&sig)).is_some());
        let mut renamed = project.clone();
        renamed.name = "other".into();
        assert_eq!(signature(&renamed), sig);
    }

    #[test]
    fn tsv_mode_without_a_dataset_never_exports_even_with_a_clean_analysis() {
        let mut project = Project::default();
        select_tsv_mode(&mut project);
        let ready = Analysis {
            ready: true,
            ..Default::default()
        };
        let sig = signature(&project);
        assert!(export_blocker(&project, Some(&ready), Some(&sig)).is_some());
        pin_target(
            &mut project,
            TargetBinding {
                id: "t1".into(),
                fingerprint: "f1".into(),
            },
        );
        let sig = signature(&project);
        assert!(export_blocker(&project, Some(&ready), Some(&sig)).is_none());
        use_vanilla(&mut project);
        assert!(project.target.is_none() && project.target_mode == TargetMode::Vanilla);
    }

    #[test]
    fn adapters_survive_only_a_repin_of_the_same_dataset_fingerprint() {
        let mut project = Project::default();
        let binding = TargetBinding {
            id: "t1".into(),
            fingerprint: "f1".into(),
        };
        pin_target(&mut project, binding.clone());
        project.adapter = Some(AdapterBinding {
            id: "a1".into(),
            fingerprint: "af".into(),
            target_fingerprint: "f1".into(),
        });
        pin_target(&mut project, binding);
        assert!(project.adapter.is_some());
        pin_target(
            &mut project,
            TargetBinding {
                id: "t2".into(),
                fingerprint: "f2".into(),
            },
        );
        assert!(project.adapter.is_none());
        assert_eq!(project.target.as_ref().map(|t| t.id.as_str()), Some("t2"));
    }

    fn world_rules() -> Rules {
        Rules {
            root: Default::default(),
            factions: serde_json::json!({"france": {}, "prussia": {}}),
            regions: serde_json::json!({
                "acadia": {"owner": "france"},
                "silesia": {"owner": "prussia"},
            }),
            units: Default::default(),
            buildings: Default::default(),
            buildings_by_slot_type: Default::default(),
            slot_types: Default::default(),
            armies: Default::default(),
        }
    }

    /// An analysis that is NOT export-ready (open blocker) but knows the target database.
    fn blocked_catalog(factions: &[&str]) -> Analysis {
        Analysis {
            catalog: Catalog {
                factions: factions.iter().map(|f| f.to_string()).collect(),
                ..Default::default()
            },
            issues: vec![issue(
                "ownership:silesia:invalid_owner",
                "h",
                "blocker",
                &[],
                false,
            )],
            ready: false,
            ..Default::default()
        }
    }

    #[test]
    fn assignment_checks_the_actual_target_catalog_without_an_export_ready_analysis() {
        let rules = world_rules();
        let analysis = blocked_catalog(&["france", "target_only_faction"]);
        assert!(!analysis.ready && !blockers(&analysis).is_empty());
        let mut project = Project::default();
        // A target faction that no region of the world campaign uses is a valid recipient.
        assign_owner(
            &mut project,
            &rules,
            Some(&analysis),
            "silesia",
            "target_only_faction",
        )
        .unwrap();
        assert_eq!(project.owners["silesia"], "target_only_faction");
        // Absent factions (including world-campaign ones) cannot be chosen and leave the project untouched.
        let before = project.owners.clone();
        for faction in ["prussia", "invented"] {
            let error =
                assign_owner(&mut project, &rules, Some(&analysis), "acadia", faction).unwrap_err();
            assert!(format!("{error:#}").contains("no new faction keys"));
        }
        assert_eq!(project.owners, before);
        // No catalog yet / an empty catalog: nothing is guessed.
        assert!(assign_owner(&mut project, &rules, None, "acadia", "france").is_err());
        let empty = blocked_catalog(&[]);
        assert!(assign_owner(&mut project, &rules, Some(&empty), "acadia", "france").is_err());
        assert_eq!(project.owners, before);
        // Choosing the baseline owner drops the override instead of storing it.
        assign_owner(&mut project, &rules, Some(&analysis), "silesia", "france").unwrap();
        assign_owner(&mut project, &rules, Some(&analysis), "acadia", "france").unwrap();
        assert_eq!(project.owners["silesia"], "france");
        assert!(!project.owners.contains_key("acadia"));
        assert!(assign_owner(&mut project, &rules, Some(&analysis), "atlantis", "france").is_err());
    }

    #[test]
    fn the_baseline_owner_follows_mappings_and_is_never_stored_as_an_override() {
        let rules = world_rules();
        let analysis = blocked_catalog(&["france", "mod_prussia"]);
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("prussia".into(), "mod_prussia".into());
        assign_owner(&mut project, &rules, Some(&analysis), "silesia", "france").unwrap();
        assert_eq!(project.owners["silesia"], "france");
        assign_owner(
            &mut project,
            &rules,
            Some(&analysis),
            "silesia",
            "mod_prussia",
        )
        .unwrap();
        assert!(project.owners.is_empty());
    }

    #[test]
    fn owner_choices_come_from_the_target_catalog_only() {
        let analysis = blocked_catalog(&["zeta", "alpha", "alpha"]);
        assert!(owner_choices(None).is_empty());
        assert_eq!(owner_choices(Some(&analysis)), ["alpha", "zeta"]);
    }

    #[test]
    fn an_owner_override_is_an_exact_target_key_while_unassigned_regions_follow_mappings() {
        let rules = world_rules();
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("france".into(), "mod_france".into());
        project.owners.insert("silesia".into(), "france".into());
        assert_eq!(effective_owner(&project, &rules, "silesia"), "france");
        assert_eq!(effective_owner(&project, &rules, "acadia"), "mod_france");
    }

    #[test]
    fn catalog_key_follows_the_target_not_ownership_edits() {
        let mut project = Project::default();
        let key = catalog_key(&project);
        // Owner edits and renames do not change which factions the target has, so they keep the catalog usable.
        project.owners.insert("silesia".into(), "france".into());
        project.name = "renamed".into();
        assert_eq!(catalog_key(&project), key);
        pin_target(
            &mut project,
            TargetBinding {
                id: "t1".into(),
                fingerprint: "f1".into(),
            },
        );
        assert_ne!(catalog_key(&project), key);
        // Ownership edits do change the analysis signature: the analysis must be redone.
        let ready = Analysis {
            ready: true,
            ..Default::default()
        };
        let analyzed = signature(&project);
        assert!(export_blocker(&project, Some(&ready), Some(&analyzed)).is_none());
        project.owners.insert("acadia".into(), "france".into());
        assert!(export_blocker(&project, Some(&ready), Some(&analyzed)).is_some());
    }

    #[test]
    fn a_target_only_owner_survives_a_project_file_roundtrip() {
        let mut project = Project::default();
        project
            .owners
            .insert("silesia".into(), "target_only_faction".into());
        let reloaded = Project::parse(&serde_json::to_vec(&project).unwrap()).unwrap();
        assert_eq!(reloaded.owners["silesia"], "target_only_faction");
        validate_owners(&reloaded, &world_rules()).unwrap();
    }

    #[test]
    fn invalid_owner_blockers_name_their_region() {
        let blocker = issue(
            "ownership:silesia:invalid_owner",
            "h",
            "blocker",
            &[],
            false,
        );
        assert_eq!(owner_blocker_region(&blocker), Some("silesia"));
        for other in [
            "mapping:factions:x:missing_target",
            "mapping:owners:silesia:existing_owner_required",
            "ownership:silesia:something_else",
        ] {
            assert_eq!(
                owner_blocker_region(&issue(other, "h", "blocker", &[], false)),
                None,
                "{other}"
            );
        }
    }

    /// Imported dataset whose original data leaves `regions` without an owner: no issue exists, the analysis is not
    /// ready and only the pending list says why.
    fn pending_analysis(regions: &[&str], factions: &[&str]) -> Analysis {
        Analysis {
            catalog: Catalog {
                factions: factions.iter().map(|f| f.to_string()).collect(),
                ..Default::default()
            },
            unassigned_regions: regions.iter().map(|r| r.to_string()).collect(),
            ready: false,
            ..Default::default()
        }
    }

    #[test]
    fn pending_ownership_holds_export_back_without_being_an_issue() {
        let project = Project::default();
        let sig = signature(&project);
        let pending = pending_analysis(&["acadia", "silesia"], &["france"]);
        assert!(blockers(&pending).is_empty());
        assert_eq!(blocker_count(&pending), 0);
        assert_eq!(standing(&project, &pending), Standing::PendingOwnership(2));
        assert!(export_blocker(&project, Some(&pending), Some(&sig)).is_some());
        // Not ready with nothing pending and no blocker is a different, unexplained state.
        let unexplained = pending_analysis(&[], &["france"]);
        assert_eq!(standing(&project, &unexplained), Standing::NotReady);
        assert!(export_blocker(&project, Some(&unexplained), Some(&sig)).is_some());
        assert!(only_pending_ownership(&project, Some(&pending), Some(&sig)));
        assert!(!only_pending_ownership(&project, Some(&pending), Some("stale")));
        assert!(!only_pending_ownership(&project, Some(&unexplained), Some(&sig)));
    }

    #[test]
    fn genuine_blockers_stay_counted_next_to_pending_ownership() {
        let project = Project::default();
        let sig = signature(&project);
        let mut mixed = pending_analysis(&["silesia"], &["france"]);
        mixed.issues = vec![issue("db:a", "h", "blocker", &[], false)];
        assert_eq!(standing(&project, &mixed), Standing::Blocked(1));
        assert_eq!(blocker_count(&mixed), 1);
        assert_eq!(pending_count(&project, &mixed), 1);
        assert!(!only_pending_ownership(&project, Some(&mixed), Some(&sig)));
        assert!(export_blocker(&project, Some(&mixed), Some(&sig)).is_some());
        // A resolved blocker no longer counts, leaving only the pending ownership.
        mixed.issues[0].resolved = true;
        assert_eq!(standing(&project, &mixed), Standing::PendingOwnership(1));
        assert!(only_pending_ownership(&project, Some(&mixed), Some(&sig)));
    }

    #[test]
    fn assigning_pending_regions_clears_them_only_after_reanalysis_and_refuses_invalid_owners() {
        let rules = world_rules();
        let mut project = Project::default();
        let analysis = pending_analysis(&["acadia", "silesia"], &["france", "target_only_faction"]);
        let analyzed = signature(&project);
        assert!(export_blocker(&project, Some(&analysis), Some(&analyzed)).is_some());
        assert_eq!(
            pending_regions(&project, &analysis).collect::<Vec<_>>(),
            ["acadia", "silesia"]
        );

        // An absent faction is refused and leaves both the project and the pending list untouched.
        for faction in ["invented", "prussia", ""] {
            assert!(assign_owner(&mut project, &rules, Some(&analysis), "acadia", faction).is_err());
        }
        assert!(project.owners.is_empty());
        assert_eq!(pending_count(&project, &analysis), 2);
        // No catalog (or a stale/other analysis without one) means no assignment at all.
        assert!(assign_owner(&mut project, &rules, None, "acadia", "france").is_err());

        // Bulk assignment to an existing target faction (even one no region uses) is accepted for every pending region.
        let batch: Vec<String> = pending_regions(&project, &analysis)
            .map(str::to_owned)
            .collect();
        for region in &batch {
            assign_owner(&mut project, &rules, Some(&analysis), region, "target_only_faction")
                .unwrap();
        }
        assert_eq!(pending_count(&project, &analysis), 0);
        // The old verdict is not trusted: the project changed, so export waits for a fresh analysis.
        assert!(export_blocker(&project, Some(&analysis), Some(&analyzed)).is_some());
        // The re-analysis of the assigned project has nothing pending and is ready.
        let reanalyzed = Analysis {
            ready: true,
            catalog: analysis.catalog.clone(),
            ..Default::default()
        };
        let now = signature(&project);
        assert_ne!(now, analyzed);
        assert_eq!(standing(&project, &reanalyzed), Standing::Ready);
        assert!(export_blocker(&project, Some(&reanalyzed), Some(&now)).is_none());
    }

    #[test]
    fn restoring_an_assignment_brings_the_region_back_to_pending() {
        let rules = world_rules();
        let mut project = Project::default();
        let analysis = pending_analysis(&["silesia"], &["france"]);
        assign_owner(&mut project, &rules, Some(&analysis), "silesia", "france").unwrap();
        assert_eq!(pending_count(&project, &analysis), 0);
        assert!(!is_pending(&project, &analysis, "silesia"));
        project.owners.remove("silesia");
        assert_eq!(pending_count(&project, &analysis), 1);
        assert!(is_pending(&project, &analysis, "silesia"));
        assert!(!is_pending(&project, &analysis, "acadia"));
    }
}
