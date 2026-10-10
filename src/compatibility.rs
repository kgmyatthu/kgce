//! TSV-target compatibility analysis and content build.
//!
//! * [`analyze`] reads the original-source snapshot and the pinned immutable TSV dataset and reports everything that
//!   decides whether the Grand Campaign can be built on that target: required and dependent tables, effective
//!   database, key collisions, mapping problems, localisation, resource review and adapter. Ordinary problems are
//!   issues, never panics. No terrain build, no game writes.
//! * [`prepare`] requires a ready analysis and creates the resolved snapshot, rules and resources.
//! * [`build_members`] builds only campaign DB additions/overrides and target DB members changed by explicit
//!   reconciliation, plus the merged localisation and UI resources. Unchanged target tables remain in the
//!   installed mod; they are conversion inputs, not a second database bundled with the port.
//! * [`required_tables`] lists the tables the conversion needs, for the dataset import preview.
//!
//! The TSV dataset is authoritative: there is no fallback to the original game's rows. This is not a validator of the
//! mod: only what the campaign conversion reads or writes can block (campaign inputs, the keys of generated rows,
//! their references, ownership, geometry and resources). Duplicate keys, unreadable or dangling rows and missing
//! records of anything else are warnings; those rows stay untouched and are never exported. Duplicate keys with
//! different content on a key the conversion reads are never concatenated and left to the engine; they require an
//! explicit, input-hash-bound choice that is applied by rewriting only the affected members (same path, in the
//! generated pack). Nothing here claims that the result was played or that the installed mod packs are compatible.

mod effective;
mod resources;

use crate::{
    campaign_profile,
    content::{self, MapRows, Reconciled},
    db::{self, Cell},
    map::MapArtifacts,
    model::{Progress, Project, Rules, TargetMode},
    snapshot::{Game, Snapshot},
    target::{self, Dataset, Requirement},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use walkdir::WalkDir;

/// Pack file of the vanilla conversion.
pub const VANILLA_PACK: &str = "nap_world.pack";
/// Honest status carried by every analysis.
pub const GAMEPLAY_NOTE: &str = "Gameplay was not tested: this analysis only checks files, keys and references; the result has not been run in Napoleon: Total War.";

// ---------------------------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------------------------

/// Which `Mappings` field and source key fixes a mapping issue.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct MappingHint {
    /// `factions` | `units` | `buildings` | `faction_templates`
    pub field: String,
    pub key: String,
}

/// One finding. `severity` is `blocker`, `warning` or `info`. A blocker with `choices` is resolved by storing
/// `Project::resolutions[id] = {input_hash, choice}`; a blocker without choices is resolved by changing the inputs
/// (mappings, TSV dataset, resource review, adapter) and analysing again.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Issue {
    pub id: String,
    /// Digest of every input the finding depends on (reference, target and port content).
    pub input_hash: String,
    pub severity: String,
    /// `target`, `resources`, `language`, `schema`, `reference`, `db_key`, `mapping`, `ui`, `script`, `template`, `adapter`, `port`, `loc`
    pub category: String,
    pub title: String,
    pub detail: String,
    /// Machine values: `keep_target`, `use_port`, `member:<data path>`.
    pub choices: Vec<String>,
    /// A stored resolution with the same `input_hash` and a valid choice exists.
    pub resolved: bool,
    #[serde(default)]
    pub mapping: Option<MappingHint>,
}

/// One operation of the build, for the change report.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Change {
    pub operation: String,
    pub table: String,
    pub key: String,
    pub source: String,
    pub detail: String,
}

/// Searchable key lists for selectors.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Catalog {
    /// Factions of the effective target (valid values of `mappings.factions`).
    pub factions: Vec<String>,
    /// Units of the effective target (valid values of `mappings.units`).
    pub units: Vec<String>,
    /// Building levels of the effective target (valid values of `mappings.buildings`).
    pub buildings: Vec<String>,
    /// Target factions with a live record in the target start position (valid `faction_templates` donors).
    pub donor_factions: Vec<String>,
    /// Faction keys of the Grand Campaign rules (valid keys of `mappings.factions`).
    pub source_factions: Vec<String>,
    /// ETW and vanilla NTW unit keys of the rules (valid keys of `mappings.units`).
    pub source_units: Vec<String>,
    /// ETW building keys and vanilla NTW building levels of the rules (valid keys of `mappings.buildings`).
    pub source_buildings: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Analysis {
    /// Digest of everything the analysis depends on: the TSV dataset, project mappings/owners/resolutions,
    /// recipe assets and application version. A different fingerprint invalidates a cached `ready`.
    pub fingerprint: String,
    pub target_name: String,
    pub language: String,
    pub issues: Vec<Issue>,
    pub changes: Vec<Change>,
    pub catalog: Catalog,
    pub warnings: Vec<String>,
    /// Regions whose original owner is absent from the target and has not been replaced by a user selection.
    /// These are ownership tasks, not missing-faction mappings. Remaining campaign checks wait for their owners.
    #[serde(default)]
    pub unassigned_regions: Vec<String>,
    pub ready: bool,
    /// Always false: nothing here plays the game.
    pub gameplay_tested: bool,
    /// File name of the generated pack (`nap_world.pack` for vanilla, a deterministic collision-free name for a TSV dataset).
    #[serde(default)]
    pub pack_name: String,
}

impl Analysis {
    /// Blockers that still need a decision or a change of inputs.
    pub fn blockers(&self) -> impl Iterator<Item = &Issue> {
        self.issues
            .iter()
            .filter(|i| i.severity == "blocker" && !i.resolved)
    }
}

/// Everything the converter needs after a successful analysis.
pub struct Prepared {
    /// Effective NTW view (the TSV dataset's tables and text, unambiguous DB rows, adapter template) plus the
    /// untouched ETW source and the original game's non-database resources.
    pub snapshot: Snapshot,
    /// Rules resolved against the target (mappings and owners applied).
    pub rules: Rules,
    pub analysis: Analysis,
    /// UI / script resources of the pack: member name (backslashes) -> bytes.
    pub resources: BTreeMap<String, Vec<u8>>,
    /// Generated pack file name (`nap_world.pack` for vanilla).
    pub pack_name: String,
    /// Dependency lock for the export metadata (dataset manifest, adapter, resources review, caveats).
    pub dependencies: Value,
    plan: Plan,
}

impl Prepared {
    /// Pack member names that rewrite (shadow) target DB members, backslash form.
    pub fn shadowed_members(&self) -> impl Iterator<Item = &String> {
        self.plan.shadow.keys()
    }

    /// Whether the target is the original Napoleon snapshot rather than a TSV dataset.
    pub fn is_vanilla(&self) -> bool {
        !self.plan.modded
    }
}

/// Decisions fixed by the analysis and re-checked against the real map output at build time.
#[derive(Clone)]
struct Plan {
    /// A TSV dataset is the target.
    modded: bool,
    /// Tables whose colliding port rows are replaced by the target's rows.
    keep_target_tables: BTreeSet<String>,
    /// Port strings the target's text wins over.
    loc_keep_target: BTreeSet<String>,
    /// Effective LOC member -> keys the port's string replaces in that member.
    loc_strip: BTreeMap<String, BTreeSet<String>>,
    /// Keys of conflicting strings that carry a decision (either way).
    loc_decided: BTreeSet<String>,
    /// Whether a target string equal to the original English string counts as a plain vanilla replacement.
    loc_vanilla_english: bool,
    /// Rewritten target DB members: pack member name -> bytes.
    shadow: BTreeMap<String, Vec<u8>>,
    /// Dataset localisation members (backslash form); DB inputs are never copied wholesale.
    dataset_loc_members: BTreeSet<String>,
    base: Snapshot,
    project_digest: String,
}

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

/// Length-prefixed SHA-256 over text parts (hex, first 128 bits).
pub(crate) struct Hasher(Sha256);

impl Hasher {
    pub(crate) fn new(domain: &str) -> Self {
        let mut h = Sha256::new();
        h.update(domain.as_bytes());
        h.update([0]);
        Self(h)
    }

    pub(crate) fn part(&mut self, text: &str) -> &mut Self {
        self.0.update((text.len() as u64).to_le_bytes());
        self.0.update(text.as_bytes());
        self
    }

    pub(crate) fn finish(self) -> String {
        hex::encode(&self.0.finalize()[..16])
    }
}

/// An issue before resolutions are looked up.
pub(crate) struct Draft {
    id: String,
    hash: String,
    severity: &'static str,
    category: String,
    title: String,
    detail: String,
    choices: Vec<String>,
    mapping: Option<MappingHint>,
    resolved: bool,
}

impl Draft {
    pub(crate) fn blocker(
        id: String,
        hash: String,
        category: &str,
        title: String,
        detail: String,
    ) -> Self {
        Self {
            id,
            hash,
            severity: "blocker",
            category: category.to_owned(),
            title,
            detail,
            choices: Vec::new(),
            mapping: None,
            resolved: false,
        }
    }

    fn choices(mut self, choices: Vec<String>) -> Self {
        self.choices = choices;
        self
    }

    fn mapping(mut self, hint: Option<MappingHint>) -> Self {
        self.mapping = hint;
        self
    }
}

fn shown<I: IntoIterator<Item = String>>(items: I, max: usize) -> String {
    let all: Vec<String> = items.into_iter().collect();
    let mut text = all.iter().take(max).cloned().collect::<Vec<_>>().join(", ");
    if all.len() > max {
        text.push_str(&format!(", … ({} more)", all.len() - max));
    }
    text
}

fn english(language: &str) -> bool {
    language.eq_ignore_ascii_case("en") || language.eq_ignore_ascii_case("english")
}

/// Digest of the recipe assets (curated rules and bundled UI) the conversion reads.
fn recipe_digest(assets: &Path) -> Result<String> {
    let mut files: Vec<(String, std::path::PathBuf)> = Vec::new();
    for sub in ["rules", "ui"] {
        for entry in WalkDir::new(assets.join(sub)) {
            let entry = entry?;
            if entry.file_type().is_file() {
                files.push((
                    entry
                        .path()
                        .strip_prefix(assets)?
                        .to_string_lossy()
                        .replace('\\', "/"),
                    entry.path().to_path_buf(),
                ));
            }
        }
    }
    files.sort();
    let mut h = Sha256::new();
    for (name, path) in files {
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        let bytes = fs::read(&path).with_context(|| format!("Read {}", path.display()))?;
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    Ok(hex::encode(&h.finalize()[..16]))
}

/// Everything of the project the pack depends on (not its name). A dataset binding only counts in TSV mode so a
/// stale binding cannot change the vanilla conversion.
fn project_digest(project: &Project) -> String {
    let tsv = project.target_mode == TargetMode::Tsv;
    let mut h = Hasher::new("project");
    h.part(&serde_json::to_string(&project.target_mode).unwrap_or_default());
    h.part(&serde_json::to_string(&project.target.as_ref().filter(|_| tsv)).unwrap_or_default());
    h.part(&serde_json::to_string(&project.mappings).unwrap_or_default());
    h.part(&serde_json::to_string(&project.owners).unwrap_or_default());
    if !project.activations.is_empty() {
        h.part("activations");
        h.part(&serde_json::to_string(&project.activations).unwrap_or_default());
    }
    if !project.splits.is_empty() {
        h.part(&serde_json::to_string(&project.splits).unwrap_or_default());
    }
    h.part(&serde_json::to_string(&project.atlas).unwrap_or_default());
    h.part(&project.building_scale.to_string());
    h.part(&serde_json::to_string(&project.settlement_moves).unwrap_or_default());
    h.part("fixed-theatres");
    for (key, rect) in crate::regionmap::theatres::FIXED_BOUNDS {
        h.part(key);
        for coordinate in rect.iter().flatten() {
            h.0.update(coordinate.to_le_bytes());
        }
    }
    if let Some(dir) = &project.painted_terrain {
        h.part("painted-terrain");
        h.part(&dir.to_string_lossy());
        // the files' names and sizes bind the digest to what the folder holds now
        for entry in walkdir::WalkDir::new(dir).sort_by_file_name().into_iter().flatten().filter(|e| e.file_type().is_file()) {
            h.part(&entry.path().strip_prefix(dir).unwrap_or(entry.path()).to_string_lossy());
            h.part(&entry.metadata().map(|m| m.len()).unwrap_or(0).to_string());
        }
    }
    h.part(&serde_json::to_string(&project.resolutions).unwrap_or_default());
    h.part(&serde_json::to_string(&project.adapter).unwrap_or_default());
    h.part("existing-factions-only");
    h.finish()
}

/// Deterministic, collision-free name of the generated pack for a TSV dataset (and its adapter, which changes the
/// content).
fn pack_name_for(dataset: &Dataset, adapter: Option<&target::AdapterBinding>) -> String {
    let mut h = Hasher::new("kgc-pack-name");
    h.part(&dataset.manifest.fingerprint);
    h.part(&adapter.map(|a| a.fingerprint.as_str()).unwrap_or_default());
    format!("zz_kgc_world_{}.pack", &h.finish()[..8])
}

fn keys_of(snapshot: &Snapshot, rules: &Rules, table: &str, column: &str) -> BTreeSet<String> {
    db::load_table_opt(snapshot, Game::Ntw, rules, table)
        .ok()
        .flatten()
        .and_then(|t| {
            let c = t.column(column).ok()?;
            Some(
                t.rows
                    .iter()
                    .filter_map(|r| r[c].as_str().map(str::to_owned))
                    .collect(),
            )
        })
        .unwrap_or_default()
}

fn object_keys(value: &Value) -> impl Iterator<Item = &String> {
    value.as_object().into_iter().flat_map(|m| m.keys())
}

fn object_str_values(value: &Value) -> impl Iterator<Item = &str> {
    value
        .as_object()
        .into_iter()
        .flat_map(|m| m.values())
        .filter_map(Value::as_str)
}

fn source_catalog(rules: &Rules) -> (Vec<String>, Vec<String>, Vec<String>) {
    let factions: Vec<String> = object_keys(&rules.factions).cloned().collect();
    // Unit mappings are keyed by the NTW stand-ins the conversion generates forces from: the values of
    // `rules.units` and the units of the curated start armies. ETW-only keys never affect the forces.
    let mut units: BTreeSet<String> = object_str_values(&rules.units).map(str::to_owned).collect();
    for list in rules
        .armies
        .as_object()
        .into_iter()
        .flat_map(|m| m.values())
    {
        for army in list.as_array().into_iter().flatten() {
            units.extend(
                army.get("units")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(str::to_owned),
            );
        }
    }
    let buildings: BTreeSet<String> = object_keys(&rules.buildings)
        .cloned()
        .chain(object_str_values(&rules.buildings).map(str::to_owned))
        .collect();
    (
        factions,
        units.into_iter().collect(),
        buildings.into_iter().collect(),
    )
}

// ---------------------------------------------------------------------------------------------
// Evaluation state
// ---------------------------------------------------------------------------------------------

struct State<'a> {
    project: &'a Project,
    drafts: Vec<Draft>,
    changes: Vec<Change>,
    warnings: Vec<String>,
    unassigned_regions: BTreeSet<String>,
    target_name: String,
    language: String,
    pack_name: String,
    target_fingerprint: String,
    /// Missing-table blockers already emitted (one per table).
    missing_reported: BTreeSet<String>,
    base_digest: String,
    catalog: Catalog,
    catalog_snapshot: Option<(Snapshot, Rules)>,
    /// Faction keys of every decoded `factions_tables` member of the TSV dataset (used for the owner catalog when
    /// the effective database could not be assembled because of unrelated blockers).
    scanned_factions: BTreeSet<String>,
}

impl<'a> State<'a> {
    fn new(project: &'a Project) -> Self {
        Self {
            project,
            drafts: Vec::new(),
            changes: Vec::new(),
            warnings: vec![GAMEPLAY_NOTE.to_owned()],
            unassigned_regions: BTreeSet::new(),
            target_name: String::new(),
            language: String::new(),
            pack_name: VANILLA_PACK.to_owned(),
            target_fingerprint: "vanilla".to_owned(),
            missing_reported: BTreeSet::new(),
            base_digest: String::new(),
            catalog: Catalog::default(),
            catalog_snapshot: None,
            scanned_factions: BTreeSet::new(),
        }
    }

    fn profile_problem(&mut self, problem: campaign_profile::MappingProblem) {
        if problem.field == "owners"
            && problem.kind == "existing_owner_required"
            && !self.project.owners.contains_key(&problem.key)
        {
            self.unassigned_regions.insert(problem.key);
            return;
        }
        let mut h = Hasher::new("mapping");
        h.part(&problem.field).part(&problem.key).part(&problem.kind).part(&problem.message);
        let (id, category, title, hint) = if problem.field == "owners" {
            (
                format!("ownership:{}:invalid_owner", problem.key),
                "ownership",
                format!("Invalid ownership selection for {}", problem.key),
                None,
            )
        } else {
            (
                format!("mapping:{}:{}:{}", problem.field, problem.key, problem.kind),
                "mapping",
                format!("Mapping needed for {} {}", problem.field, problem.key),
                mapping_hint(&problem.field, &problem.key),
            )
        };
        self.drafts.push(Draft::blocker(id, h.finish(), category, title, problem.message).mapping(hint));
    }

    fn block(&mut self, id: &str, category: &str, title: &str, detail: String) {
        let mut h = Hasher::new("block");
        h.part(id).part(&detail);
        self.drafts.push(Draft::blocker(
            id.to_owned(),
            h.finish(),
            category,
            title.to_owned(),
            detail,
        ));
    }

    fn change(&mut self, operation: &str, table: &str, key: &str, source: &str, detail: String) {
        self.changes.push(Change {
            operation: operation.to_owned(),
            table: table.to_owned(),
            key: key.to_owned(),
            source: source.to_owned(),
            detail,
        });
    }

    /// Registers a decision issue and returns the stored choice when it is bound to exactly these inputs.
    fn decide(&mut self, mut draft: Draft) -> Option<String> {
        let mut choice = None;
        if let Some(res) = self.project.resolutions.get(&draft.id) {
            if res.input_hash != draft.hash {
                self.warnings.push(format!(
                    "The stored choice for \"{}\" was ignored: its inputs changed since it was made.",
                    draft.title
                ));
            } else if draft.choices.contains(&res.choice) {
                choice = Some(res.choice.clone());
            } else {
                self.warnings.push(format!(
                    "The stored choice {:?} for \"{}\" is not offered any more.",
                    res.choice, draft.title
                ));
            }
        }
        draft.resolved = choice.is_some();
        self.drafts.push(draft);
        choice
    }

    fn set_catalog(&mut self, snapshot: &Snapshot, rules: &Rules) {
        self.catalog_snapshot = Some((snapshot.clone(), rules.clone()));
    }

    fn conclude(mut self, assets: &Path, built: Option<Built>) -> Result<Evaluated> {
        if let Some((snapshot, rules)) = self.catalog_snapshot.take() {
            self.catalog.factions = keys_of(&snapshot, &rules, "factions", "key")
                .into_iter()
                .collect();
            if self.catalog.factions.is_empty() {
                self.catalog.factions = std::mem::take(&mut self.scanned_factions)
                    .into_iter()
                    .collect();
            }
            self.catalog.units = keys_of(&snapshot, &rules, "units", "key")
                .into_iter()
                .collect();
            self.catalog.buildings = keys_of(&snapshot, &rules, "building_levels", "level_name")
                .into_iter()
                .collect();
            if let Ok(sp) = crate::startpos::core::startpos_structure(&snapshot) {
                let known: BTreeSet<&String> = self.catalog.factions.iter().collect();
                self.catalog.donor_factions = sp
                    .alive
                    .iter()
                    .filter(|(k, alive)| **alive && known.contains(k))
                    .map(|(k, _)| k.clone())
                    .collect();
            }
        }
        let severity_rank = |s: &str| match s {
            "blocker" => 0,
            "warning" => 1,
            _ => 2,
        };
        self.drafts.sort_by(|a, b| {
            (severity_rank(a.severity), &a.category, &a.id).cmp(&(
                severity_rank(b.severity),
                &b.category,
                &b.id,
            ))
        });
        let issues: Vec<Issue> = self
            .drafts
            .into_iter()
            .map(|d| Issue {
                id: d.id,
                input_hash: d.hash,
                severity: d.severity.to_owned(),
                category: d.category,
                title: d.title,
                detail: d.detail,
                choices: d.choices,
                resolved: d.resolved,
                mapping: d.mapping,
            })
            .collect();
        let ready = self.unassigned_regions.is_empty()
            && !issues.iter().any(|i| i.severity == "blocker" && !i.resolved);

        let mut h = Hasher::new("analysis");
        h.part(env!("CARGO_PKG_VERSION"))
            .part(&self.target_fingerprint)
            .part(&self.base_digest)
            .part(&project_digest(self.project))
            .part(&recipe_digest(assets)?)
            .part(&self.pack_name);
        for i in &issues {
            h.part(&i.id)
                .part(&i.input_hash)
                .part(&i.resolved.to_string());
        }
        for region in &self.unassigned_regions {
            h.part("ownership-pending").part(region);
        }
        let analysis = Analysis {
            fingerprint: h.finish(),
            target_name: self.target_name,
            language: self.language,
            issues,
            changes: self.changes,
            catalog: self.catalog,
            warnings: self.warnings,
            unassigned_regions: self.unassigned_regions.into_iter().collect(),
            ready,
            gameplay_tested: false,
            pack_name: self.pack_name,
        };
        Ok(Evaluated { analysis, built })
    }
}

fn reads_campaign_table(table: &str) -> bool {
    db::CampaignTable::ALL
        .iter()
        .any(|input| input.name() == table)
}

/// Keys of the campaign inputs whose rows the conversion reads values from, derived the way the resolver applies the
/// project to the world rules (`campaign_profile::resolve`):
/// * factions: the target key of every world faction the target defines, and every chosen owner;
/// * regions: the keys of the generated region rows (world and added regions);
/// * units: the stand-ins of the starting armies of those factions, after the unit mappings;
/// * building levels: the level of every building slot the converted world places, after the building mappings.
///
/// A key the project maps away (a default replaced by a mapping) is not read and is not listed. Every other key is at
/// most tested for existence, so a conflicting duplicate of it cannot change what is generated.
#[derive(Default)]
struct Consumed {
    factions: BTreeSet<String>,
    regions: BTreeSet<String>,
    units: BTreeSet<String>,
    levels: BTreeSet<String>,
}

impl Consumed {
    /// `existing_factions` are the keys of the target's faction rows: the resolver omits a world faction (and its
    /// armies) the target does not define.
    fn new(rules: &Rules, project: &Project, existing_factions: &BTreeSet<String>) -> Self {
        let mappings = &project.mappings;
        let mut out = Self::default();
        let faction_of =
            |key: &str| mappings.factions.get(key).cloned().unwrap_or_else(|| key.to_owned());

        for key in object_keys(&rules.factions) {
            let target = faction_of(key);
            if existing_factions.contains(&target) {
                out.factions.insert(target);
            }
        }
        out.factions.extend(
            project
                .owners
                .values()
                .filter(|owner| existing_factions.contains(*owner))
                .cloned(),
        );

        out.regions.extend(object_keys(&rules.regions).cloned());
        out.regions
            .extend(content::extra_regions(project).into_iter().map(|r| r.key));

        for (source, list) in rules.armies.as_object().into_iter().flatten() {
            let key = if rules.factions.get(source).is_some() {
                faction_of(source)
            } else {
                source.clone()
            };
            if !out.factions.contains(&key) {
                continue;
            }
            for army in list.as_array().into_iter().flatten() {
                let units = army.get("units").and_then(Value::as_array).into_iter().flatten();
                for unit in units.filter_map(Value::as_str) {
                    out.units
                        .insert(mappings.units.get(unit).cloned().unwrap_or_else(|| unit.to_owned()));
                }
            }
        }

        // Slot types that receive buildings (fortifications never do) and the level of each building in them.
        let used_slots: BTreeSet<&str> = rules
            .slot_types
            .as_object()
            .into_iter()
            .flat_map(|m| m.values())
            .filter_map(Value::as_str)
            .filter(|slot| *slot != "settlement_fortification")
            .collect();
        let by_slot = rules.buildings_by_slot_type.as_object();
        let etw_buildings: BTreeSet<&String> = rules
            .buildings
            .as_object()
            .into_iter()
            .flat_map(|m| m.keys())
            .chain(by_slot.into_iter().flat_map(|m| m.keys()))
            .collect();
        for etw in etw_buildings {
            if let Some(level) = mappings.buildings.get(etw.as_str()) {
                out.levels.insert(level.clone());
                continue;
            }
            let slots = by_slot.and_then(|m| m.get(etw)).and_then(Value::as_object);
            for (slot, level) in slots.into_iter().flatten() {
                if !used_slots.contains(slot.as_str()) {
                    continue;
                }
                if let Some(level) = level.as_str() {
                    out.levels.insert(
                        mappings.buildings.get(level).cloned().unwrap_or_else(|| level.to_owned()),
                    );
                }
            }
        }
        out
    }

    /// Tables of which the conversion reads row values for some key (see [`Consumed::reads`]).
    fn values_read(table: &str) -> bool {
        matches!(
            table,
            "factions_tables"
                | "regions_tables"
                | "unit_stats_land_tables"
                | "unit_stats_naval_tables"
                | "units_to_exclusive_faction_permissions_tables"
                | "building_levels_tables"
                | "slots_art_tables"
        )
    }

    /// Whether the conversion reads the value of the row of `table` with identity `id` (its key columns joined by
    /// U+001F). Tables that are only tested for key existence (units, technologies, regions of the target,
    /// building chains, building levels the rules do not name...) never decide anything by their row content.
    fn reads(&self, table: &str, id: &str) -> bool {
        let first = id.split('\u{1f}').next().unwrap_or_default();
        match table {
            "factions_tables" => self.factions.contains(id),
            "regions_tables" => self.regions.contains(id),
            "unit_stats_land_tables" | "unit_stats_naval_tables" => self.units.contains(id),
            "units_to_exclusive_faction_permissions_tables" => self.units.contains(first),
            "building_levels_tables" => self.levels.contains(id),
            "slots_art_tables" => id
                .split('\u{1f}')
                .any(|part| crate::map::TEMPLATE_SLOT_TYPES.contains(&part)),
            _ => false,
        }
    }
}

/// Conflicting duplicate keys of a table, or of keys, that the campaign does not read: reported, never decided.
/// The members stay exactly as imported and are not exported.
fn note_unread_duplicates(st: &mut State<'_>, table: &str, members: &[effective::MemberRows]) {
    match effective::duplicates(table, members) {
        Ok(dup) => {
            let count = dup.within.len() + dup.cross.values().map(Vec::len).sum::<usize>();
            if count > 0 {
                st.warnings.push(format!(
                    "{table}: {count} key(s) are defined with different content more than once. The campaign does not read this table's values; the rows stay exactly as imported and are not exported."
                ));
            }
        }
        Err(error) => st.warnings.push(format!(
            "{table}: duplicate keys were not checked ({error:#}). The campaign does not read this table's values."
        )),
    }
}

/// Resolves conflicting duplicate keys the conversion depends on (`relevant(identity)`). A conflict of another key
/// is only reported; members are rewritten only for an explicit, input-hash-bound choice between members.
fn reconcile_duplicates(
    st: &mut State<'_>,
    table: &str,
    members: &mut [effective::MemberRows],
    relevant: &dyn Fn(&str) -> bool,
) -> Result<()> {
    const POLICY: &str = "The conversion reads this key, so its conflicting definitions must be resolved.";
    let dup = effective::duplicates(table, members)?;
    let mut unrelated = 0usize;
    for conflict in &dup.within {
        if !relevant(&conflict.id) {
            unrelated += 1;
            continue;
        }
        let (m, _) = conflict.entries[0];
        let hash = {
            let mut h = Hasher::new("db-within");
            h.part(table)
                .part(&members[m].path)
                .part(&members[m].sha256)
                .part(&conflict.id);
            h.finish()
        };
        st.drafts.push(Draft::blocker(
            format!("db_key:{table}:within:{}", &hash[..8]),
            hash,
            "db_key",
            format!("Conflicting rows inside a member of {table}"),
            format!(
                "Key {:?} has several definitions with different content, including conflicting rows inside one member. First occurrence: {} ({}). {POLICY} Repair the conflicting rows in the source TSV; choosing a member cannot resolve conflicts inside it.",
                conflict.id.replace('\u{1f}', " / "),
                members[m].path,
                members[m].origin,
            ),
        ));
    }
    for (set, conflicts) in &dup.cross {
        let (conflicts, ignored): (Vec<&effective::Conflict>, Vec<&effective::Conflict>) =
            conflicts.iter().partition(|c| relevant(&c.id));
        unrelated += ignored.len();
        if conflicts.is_empty() {
            continue;
        }
        let mut set_hash = Hasher::new("db-set");
        let mut h = Hasher::new("db-key");
        h.part(table);
        for &m in set {
            set_hash.part(&members[m].path);
            h.part(&members[m].path).part(&members[m].sha256);
        }
        for c in &conflicts {
            h.part(&c.id);
            for (m, r) in &c.entries {
                h.part(&db::row_signature(&members[*m].table.rows[*r]));
            }
        }
        let mut draft = Draft::blocker(
            format!("db_key:{table}:{}", &set_hash.finish()[..8]),
            h.finish(),
            "db_key",
            format!("Duplicate database keys in {table}"),
            format!(
                "{} key(s) of {table} are defined with different content in {} active files: {}. Keys: {}. {POLICY} Choose one member; conflicting rows in the other members are removed in the generated pack.",
                conflicts.len(),
                set.len(),
                shown(
                    set.iter()
                        .map(|m| format!("{} ({})", members[*m].path, members[*m].origin)),
                    6
                ),
                shown(conflicts.iter().map(|c| c.id.replace('\u{1f}', " / ")), 10),
            ),
        );
        draft.choices = set
            .iter()
            .rev()
            .map(|m| format!("member:{}", members[*m].path))
            .collect();
        if let Some(choice) = st.decide(draft) {
            let path = choice.strip_prefix("member:").unwrap_or_default();
            if let Some(winner) = set.iter().copied().find(|m| members[*m].path == path) {
                let chosen: Vec<effective::Conflict> = conflicts.iter().map(|c| (*c).clone()).collect();
                effective::keep_member(members, &chosen, winner);
            }
        }
    }
    if unrelated > 0 {
        st.warnings.push(format!(
            "{table}: {unrelated} further key(s) are defined with different content more than once. The campaign does not read their values; the rows stay exactly as imported and are not exported."
        ));
    }
    Ok(())
}

struct Built {
    snapshot: Snapshot,
    rules: Rules,
    resources: BTreeMap<String, Vec<u8>>,
    pack_name: String,
    dependencies: Value,
    plan: Plan,
}

struct Evaluated {
    analysis: Analysis,
    built: Option<Built>,
}

// ---------------------------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------------------------

fn mapping_hint(field: &str, key: &str) -> Option<MappingHint> {
    matches!(
        field,
        "factions" | "units" | "buildings" | "faction_templates"
    )
    .then(|| MappingHint {
        field: field.to_owned(),
        key: key.to_owned(),
    })
}

/// Review of existing-faction recycling: omitted world factions and forces,
/// which existing factions hold territory now, and that no army follows a region to its new owner.
fn recycle_report(st: &mut State<'_>, base: &Rules, resolved: &Rules) {
    const SOURCE: &str = "Existing factions only";
    let (Some(base_f), Some(base_r), Some(base_a), Some(res_f), Some(res_r)) = (
        base.factions.as_object(),
        base.regions.as_object(),
        base.armies.as_object(),
        resolved.factions.as_object(),
        resolved.regions.as_object(),
    ) else {
        return;
    };
    let owner_of = |v: &Value| {
        v.get("owner")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
    };
    // world faction key -> the existing faction that stands for it
    let kept: BTreeMap<&str, &str> = res_f
        .iter()
        .map(|(k, v)| {
            (
                v.get("etw_source")
                    .and_then(Value::as_str)
                    .unwrap_or(k.as_str()),
                k.as_str(),
            )
        })
        .collect();
    let (mut omitted, mut armies, mut fleets) = (Vec::new(), 0usize, 0usize);
    for key in base_f.keys().filter(|k| !kept.contains_key(k.as_str())) {
        let held = base_r.values().filter(|r| owner_of(*r) == *key).count();
        let (mut a, mut n) = (0usize, 0usize);
        for army in base_a
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if army.get("kind").and_then(Value::as_str) == Some("navy") {
                n += 1;
            } else {
                a += 1;
            }
        }
        armies += a;
        fleets += n;
        st.change(
            "omit_faction",
            "factions_tables",
            key,
            SOURCE,
            format!(
                "{key} is not defined by the target: no faction row, strings, uniforms, technologies, names or start-position record are created; its {a} starting army(ies) and {n} fleet(s) are omitted and not handed to the owners of its {held} region(s)"
            ),
        );
        omitted.push(key.clone());
    }
    if !omitted.is_empty() {
        st.warnings.push(format!(
            "{SOURCE}: {} world faction(s) are not in the target and are omitted: {}. Their {armies} starting army(ies) and {fleets} fleet(s) are omitted too; forces never move to the new owner of a region.",
            omitted.len(),
            shown(omitted.iter().cloned(), 12)
        ));
    }
    let mut gains: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for (region, info) in res_r {
        let owner = owner_of(info);
        let before = base_r.get(region).map(owner_of).unwrap_or_default();
        if kept.get(before.as_str()) != Some(&owner.as_str()) {
            gains
                .entry(owner)
                .or_default()
                .push((region.clone(), before));
        }
    }
    for (owner, list) in &gains {
        let from: BTreeSet<&str> = list.iter().map(|(_, b)| b.as_str()).collect();
        st.change(
            "assign_owner",
            "regions_tables",
            owner,
            SOURCE,
            format!(
                "{} region(s) go to the existing faction {owner} ({}), previously {}; no starting army or fleet moves with them",
                list.len(),
                shown(list.iter().map(|(r, _)| r.clone()), 6),
                shown(from.iter().map(|s| (*s).to_owned()), 6)
            ),
        );
    }
    let extra: Vec<&String> = res_f
        .iter()
        .filter(|(k, v)| {
            !base_f.contains_key(
                v.get("etw_source")
                    .and_then(Value::as_str)
                    .unwrap_or(k.as_str()),
            )
        })
        .map(|(k, _)| k)
        .collect();
    if !extra.is_empty() {
        st.warnings.push(format!(
            "{SOURCE}: existing target faction(s) outside the Grand Campaign rules receive regions: {}. They get no starting army; a faction without a record in the target start position needs a structural donor (faction templates), otherwise the build is blocked.",
            shown(extra.iter().map(|k| (*k).clone()), 12)
        ));
    }
    let rebels: Vec<String> = res_r
        .iter()
        .filter(|(region, info)| {
            base_r.get(*region).and_then(|b| b.get("rebel_faction")) != info.get("rebel_faction")
        })
        .map(|(region, info)| {
            format!(
                "{region} -> {}",
                info.get("rebel_faction")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
            )
        })
        .collect();
    if !rebels.is_empty() {
        st.warnings.push(format!(
            "{SOURCE}: {} region(s) use a different rebel faction than the world rules ({}); it is an existing target faction, never a new key.",
            rebels.len(),
            shown(rebels.into_iter(), 6)
        ));
    }
    st.warnings.push(format!(
        "{SOURCE}: every generated faction key already exists in the target database. Starting forces of retained factions keep the usual relocation when they lose their region; receiving territory does not create an army."
    ));
}

fn evaluate(
    workspace: &Path,
    assets: &Path,
    project: &Project,
    progress: &Progress<'_>,
    build: bool,
) -> Result<Evaluated> {
    let mut st = State::new(project);
    progress("Opening the original-source snapshot");
    let base = Snapshot::open(&workspace.join("sources"))?;
    if build {
        base.verify(progress)?;
    }
    st.base_digest = {
        let mut h = Hasher::new("base");
        for index in &base.manifest().indexes {
            h.part(&index.sha256);
        }
        h.finish()
    };
    let rules0 = Rules::load(&assets.join("rules"))?;
    let (source_factions, source_units, source_buildings) = source_catalog(&rules0);
    st.catalog.source_factions = source_factions;
    st.catalog.source_units = source_units;
    st.catalog.source_buildings = source_buildings;
    let geometry = project.validate_structure().and_then(|_| crate::map::region_source(&base))
        .and_then(|(mut data, _)| crate::map::splits::apply_all(&mut data, project));
    if let Err(error) = geometry {
        st.block("geometry:project", "geometry", "Project geometry cannot be applied", format!("{error:#}"));
    }
    st.warnings.push(if project.atlas.is_some() {
        "Connected stock-Q20 atlas: 1.3516979217529297× with the proven theatre arrangement and Q12 camera edges. Europe and India remain connected. Arbitrary edited campaigns still require in-game validation.".into()
    } else {
        "Theatre bounds are fixed: Europe unchanged, India through Thailand and Indonesia, America through Western Plains. This export has not been tested in game.".into()
    });
    if let Err(error) = crate::building_scale::preflight(project.building_scale) {
        st.block("resources:building-shaders", "resources", "Campaign building shaders are unavailable or incompatible", format!("{error:#}"));
    }
    if crate::building_scale::is_active(project.building_scale) {
        st.warnings.push(format!("A separate movie pack scales campaign-building shader geometry to {}× in every campaign. Characters are unchanged; remove older building-scale packs before installing it.", project.building_scale));
    }
    if !project.activations.is_empty() {
        st.warnings.push(format!(
            "{} wilderness region(s) will be activated in the generated map and start position: {}. The activated regions are static-checked only and have not been played.",
            project.activations.len(),
            project.activations.iter().map(|a| format!("{} (template {})", a.key, a.donor)).collect::<Vec<_>>().join(", ")
        ));
    }
    if !project.splits.is_empty() {
        st.warnings.push(format!(
            "{} region split(s) will be cut into the generated map and start position: {}. The new regions are static-checked only and have not been played.",
            project.splits.len(),
            project.splits.iter().map(|s| format!("{} (from {})", s.key, s.source)).collect::<Vec<_>>().join(", ")
        ));
    }

    // ---- target dataset
    let mut dataset: Option<Dataset> = None;
    let mut effective = base.clone();
    let mut adapter: Option<BTreeMap<String, Vec<u8>>> = None;
    // Tables the conversion reads or whose records the generated rows refer to (TSV targets only).
    let mut requirements: Vec<Requirement> = Vec::new();
    match project.target_mode {
        TargetMode::Vanilla => {
            st.target_name = "Vanilla Napoleon".to_owned();
            st.language = base.language().to_owned();
            if project.adapter.is_some() {
                st.block(
                    "adapter:unbound",
                    "adapter",
                    "Adapter without a TSV target",
                    "The project holds a merged adapter but is set to the original game. Adapters belong to exactly one imported TSV dataset; select the dataset again or remove the adapter.".into(),
                );
            }
        }
        TargetMode::Tsv => {
            let Some(binding) = &project.target else {
                st.block(
                    "target:unbound",
                    "target",
                    "No TSV dataset is imported",
                    "The project targets modded data but no TSV dataset is selected. Import the RPFM TSV export of the mod's database and full localisation (tables import) and select it; nothing is exported from the original game's tables instead.".into(),
                );
                return st.conclude(assets, None);
            };
            progress("Loading the pinned TSV dataset");
            let loaded = match target::load(workspace, binding) {
                Ok(d) => d,
                Err(error) => {
                    st.block(
                        "target:binding",
                        "target",
                        "TSV dataset cannot be used",
                        format!("{error:#}"),
                    );
                    return st.conclude(assets, None);
                }
            };
            st.target_name = loaded.manifest.name.clone();
            st.language = loaded.manifest.language.clone();
            st.target_fingerprint = loaded.manifest.fingerprint.clone();
            st.pack_name = pack_name_for(&loaded, project.adapter.as_ref());
            st.warnings.push(
                "TSV data cannot establish the engine load order of the installed mod packs. The generated pack name sorts after the plain mod pack names, but you must make sure it loads last and that the mod's own assets (models, textures, UI, scripts) are installed; this was not verified.".into(),
            );
            if build {
                if let Err(error) = loaded.verify(progress) {
                    st.block(
                        "target:verify",
                        "target",
                        "Stored dataset files no longer match their hashes",
                        format!("{error:#}"),
                    );
                    return st.conclude(assets, None);
                }
            } else {
                match loaded.changes() {
                    Ok(list) => {
                        for line in list.into_iter().take(20) {
                            st.warnings.push(format!("Import folder check (the stored dataset stays authoritative): {line}"));
                        }
                    }
                    Err(error) => st.warnings.push(format!(
                        "The imported TSV files could not be compared with their stored copy: {error:#}"
                    )),
                }
            }
            if !loaded.manifest.resources_reviewed {
                // Acknowledging a limit changes no campaign input, so it never gates the export.
                st.warnings.push(
                    "TSV files contain no campaign, UI or script information: resources you do not supply (as an adapter) use the original conversion templates, and the mod's own assets must be installed separately. No mod resource compatibility is claimed.".into(),
                );
            }
            effective = match loaded.effective_snapshot(&base) {
                Ok(s) => s,
                Err(error) => {
                    st.block(
                        "target:effective",
                        "target",
                        "Effective target cannot be built",
                        format!("{error:#}"),
                    );
                    return st.conclude(assets, None);
                }
            };
            if let Some(leaked) = effective
                .names(Game::Ntw, "db/")
                .into_iter()
                .find(|n| effective.member_priority(Game::Ntw, n) == 0)
            {
                st.block(
                    "target:effective",
                    "target",
                    "Effective target still holds original game tables",
                    format!("{leaked} comes from the original game, not from the dataset. A TSV target must not fall back to original tables; this is an internal error of the dataset preparation."),
                );
                return st.conclude(assets, None);
            }
            requirements = required_tables(&base, &rules0)?;
            // A member named like the generated one only matters in a table the conversion reads or adds rows to.
            let generated_tables: BTreeSet<String> = content::port_table_names()
                .into_iter()
                .chain(requirements.iter().map(|r| r.table.clone()))
                .collect();
            if effective.names(Game::Ntw, "db/").iter().any(|n| {
                n.rsplit('/').next() == Some(content::TABLE_FILE)
                    && n.split('/').nth(1).is_some_and(|t| generated_tables.contains(t))
            }) {
                st.block(
                    "target:generated_members",
                    "target",
                    "Dataset already contains generated database members",
                    format!(
                        "The dataset holds database members named {} in tables the campaign uses (an earlier export of this tool). Export the mod's own tables without that earlier export and import the dataset again.",
                        content::TABLE_FILE
                    ),
                );
                return st.conclude(assets, None);
            }
            if let Some(binding) = &project.adapter {
                if binding.target_fingerprint != loaded.manifest.fingerprint {
                    st.block(
                        "adapter:stale",
                        "adapter",
                        "Adapter belongs to another dataset version",
                        format!(
                            "The adapter {} was merged for dataset fingerprint {} but the project now uses {}. Import the merged resources again for this dataset.",
                            binding.id, binding.target_fingerprint, loaded.manifest.fingerprint
                        ),
                    );
                } else {
                    match target::adapter_files(workspace, binding, &loaded.binding()) {
                        Ok(files) => adapter = Some(files),
                        Err(error) => st.block(
                            "adapter:invalid",
                            "adapter",
                            "Adapter cannot be used",
                            format!("{error:#}"),
                        ),
                    }
                }
            }
            dataset = Some(loaded);
        }
    }

    // The target's faction list is what owners are chosen from; keep it available even when unrelated blockers
    // stop the analysis before the rules are resolved.
    st.set_catalog(&effective, &rules0);

    // ---- UI / script / template resources (also yields the optional template donor)
    let res = resources::plan(
        &resources::Input {
            assets,
            tsv: dataset.is_some(),
            adapter: adapter.as_ref(),
        },
        progress,
    )?;
    st.drafts.extend(res.issues);
    st.warnings.extend(res.warnings);
    for (operation, key, detail) in res.changes {
        st.change(&operation, "", &key, &st.target_name.clone(), detail);
    }
    let mut overlay: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    if let Some(donor) = res.donor {
        overlay.insert(resources::DONOR_PATH.to_owned(), donor);
    }
    let modded = dataset.is_some();

    // ---- the dataset must provide every table and text file the conversion reads (no vanilla fallback)
    if modded && !dataset_complete(&mut st, &effective, &requirements) {
        return st.conclude(assets, None);
    }

    // ---- effective target database: duplicate keys across members
    let mut scan = if modded {
        progress("Reading the TSV database");
        effective::scan(&effective, &base, &rules0)?
    } else {
        effective::Scan::default()
    };
    let needed: BTreeSet<String> = requirements
        .iter()
        .map(|r| r.table.clone())
        .chain(content::port_table_names())
        .collect();
    for problem in &scan.schema {
        if !needed.contains(&problem.table) {
            st.warnings.push(format!(
                "Table {} cannot be read and is left out of the analysis; the campaign does not use it. {}",
                problem.table, problem.detail
            ));
            continue;
        }
        if problem.identity_only
            && !Consumed::values_read(&problem.table)
            && !content::port_table_names().contains(&problem.table)
        {
            st.warnings.push(format!(
                "{} The conversion only tests this table for existing keys, so this does not block it.",
                problem.detail
            ));
            continue;
        }
        let mut h = Hasher::new("schema");
        h.part(&problem.table).part(&problem.detail);
        st.drafts.push(Draft::blocker(
            format!("schema:{}", problem.table),
            h.finish(),
            "schema",
            format!("Table {} cannot be read", problem.table),
            format!(
                "{} The converter cannot verify the keys of this table, so it will not build on it.",
                problem.detail
            ),
        ));
    }
    if let Some(members) = scan.tables.get("factions_tables") {
        for m in members {
            if let Ok(c) = m.table.column("key") {
                st.scanned_factions.extend(
                    m.table
                        .rows
                        .iter()
                        .filter_map(|r| r[c].as_str().map(str::to_owned)),
                );
            }
        }
    }
    // Resolve conflicts in campaign inputs before loading mappings or deriving port rows; only keys whose values
    // the conversion reads can block. Other tables are classified after staging, when the generated rows are known.
    let consumed = Consumed::new(&rules0, project, &st.scanned_factions);
    for (table, members) in &mut scan.tables {
        if reads_campaign_table(table) {
            let table: &str = table;
            reconcile_duplicates(&mut st, table, members, &|id| consumed.reads(table, id))?;
        }
    }

    // ---- rules resolved against the target (mappings, donors, owners)
    let overlay_a = {
        let mut files = overlay.clone();
        for members in scan.tables.values() {
            for (path, bytes, _) in effective::rewrite(members)? {
                files.insert(path, bytes);
            }
        }
        files
    };
    let stage_snapshot = if overlay_a.is_empty() {
        effective.clone()
    } else {
        effective.with_ntw_overlay(overlay_a, Vec::new(), Vec::new())?
    };
    // Always resolve against the target: no conversion may create a faction identity.
    let rules = {
        progress("Resolving mappings against the target");
        let catalog = match campaign_profile::TargetCatalog::load(&stage_snapshot, &rules0) {
            Ok(c) => c,
            Err(error) => {
                st.block(
                    "mapping:catalog",
                    "mapping",
                    "Target tables cannot be read",
                    format!("{error:#}"),
                );
                return st.conclude(assets, None);
            }
        };
        st.set_catalog(&stage_snapshot, &rules0);
        let (resolved, problems) = campaign_profile::resolve(&rules0, project, &catalog)?;
        if !problems.is_empty() {
            for p in problems {
                st.profile_problem(p);
            }
            return st.conclude(assets, None);
        }
        resolved
    };
    st.set_catalog(&stage_snapshot, &rules0);
    recycle_report(&mut st, &rules0, &rules);

    // ---- port rows against the target rows
    progress("Staging port rows");
    let mut port = match content::stage_port(&stage_snapshot, &rules, &content::extra_regions(project), MapRows::Curated) {
        Ok(p) => p,
        Err(error) => {
            st.block(
                "port:stage",
                "port",
                "Port rows cannot be adapted to the target",
                format!("{error:#}"),
            );
            return st.conclude(assets, None);
        }
    };
    for (table, members) in &mut scan.tables {
        if reads_campaign_table(table) {
            continue;
        }
        match port.tables.get(table) {
            Some(pt) if !pt.table.rows.is_empty() => {
                let generated: BTreeSet<String> =
                    pt.table.rows.iter().map(|row| pt.identity_of(row)).collect();
                reconcile_duplicates(&mut st, table, members, &|id| generated.contains(id))?;
            }
            _ => note_unread_duplicates(&mut st, table, members),
        }
    }
    let mut keep_target_tables: BTreeSet<String> = BTreeSet::new();
    let mut target_rows: BTreeMap<String, Vec<Vec<Cell>>> = BTreeMap::new();
    if modded {
        let port_tables: Vec<String> = port.tables.keys().cloned().collect();
        for tname in &port_tables {
            let Some(members) = scan.tables.get_mut(tname) else {
                continue;
            };
            if members.is_empty() {
                continue;
            }
            let Some(pt) = port.tables.get_mut(tname) else {
                continue;
            };
            if pt.table.rows.is_empty() {
                continue;
            }
            if !effective::same_layout(pt, members) {
                st.block(
                    &format!("schema:port:{tname}"),
                    "schema",
                    &format!("Table version of {tname} differs"),
                    format!(
                        "The converted rows use {tname} version {} with columns [{}], but the target's table is version {} with columns [{}]. The converter does not migrate table versions.",
                        pt.table.version,
                        pt.table.field_names().join(", "),
                        members[0].table.version,
                        members[0].table.field_names().join(", ")
                    ),
                );
                continue;
            }
            let collisions = effective::port_collisions(pt, members);
            if collisions.is_empty() {
                continue;
            }
            let mut drop_ids: BTreeSet<String> = BTreeSet::new();
            let mut conflicts = Vec::new();
            for c in collisions {
                if c.identical {
                    drop_ids.insert(c.id.clone());
                    if let Some(&(m, r)) = c.target.first() {
                        target_rows
                            .entry(tname.clone())
                            .or_default()
                            .push(members[m].table.rows[r].clone());
                    }
                    st.changes.push(Change {
                        operation: "reuse_row".into(),
                        table: tname.clone(),
                        key: c.id.replace('\u{1f}', " / "),
                        source: members[c.target[0].0].path.clone(),
                        detail: "the target already has an identical row; it is reused".into(),
                    });
                } else {
                    conflicts.push(c);
                }
            }
            if !conflicts.is_empty() && tname == "factions_tables" {
                for c in &conflicts {
                    let key = c.id.clone();
                    let mut h = Hasher::new("faction-collision");
                    h.part(&key).part(&db::row_signature(
                        &members[c.target[0].0].table.rows[c.target[0].1],
                    ));
                    st.drafts.push(
                        Draft::blocker(
                            format!("mapping:factions:{key}:collision"),
                            h.finish(),
                            "mapping",
                            format!("Faction {key} already exists in the target"),
                            format!(
                                "The Grand Campaign faction {key} has the same key as a target faction with different data. Map {key} to a target faction in the faction mapping."
                            ),
                        )
                        .mapping(mapping_hint("factions", &key)),
                    );
                    drop_ids.insert(c.id.clone());
                }
                conflicts.clear();
            }
            if !conflicts.is_empty() {
                let mut h = Hasher::new("port-row");
                h.part(tname);
                for c in &conflicts {
                    h.part(&c.id);
                    if !content::is_map_table(tname) {
                        h.part(&db::row_signature(&c.port_row));
                    }
                    for (m, r) in &c.target {
                        h.part(&members[*m].path)
                            .part(&members[*m].sha256)
                            .part(&db::row_signature(&members[*m].table.rows[*r]));
                    }
                }
                let required = content::is_port_required(tname);
                let choices = if required {
                    vec!["use_port".to_owned()]
                } else {
                    vec!["keep_target".to_owned(), "use_port".to_owned()]
                };
                let origins: BTreeSet<String> = conflicts
                    .iter()
                    .flat_map(|c| c.target.iter().map(|(m, _)| members[*m].path.clone()))
                    .collect();
                let detail = format!(
                    "{} row(s) the Grand Campaign adds to {tname} have the same key as rows of the target ({}): {}. keep_target {} drops the converted row and uses the target's; use_port removes the target's row from its member (rewritten in the generated pack) and uses the converted one.",
                    conflicts.len(),
                    shown(origins.into_iter(), 4),
                    shown(conflicts.iter().map(|c| c.id.replace('\u{1f}', " / ")), 10),
                    if required {
                        "is not offered here because the generated map depends on the converted row;"
                    } else {
                        "keeps the mod's data;"
                    }
                );
                let draft = Draft::blocker(
                    format!("port_row:{tname}"),
                    h.finish(),
                    "port",
                    format!("Converted rows collide with target rows in {tname}"),
                    detail,
                )
                .choices(choices);
                match st.decide(draft).as_deref() {
                    Some("use_port") => {
                        for c in &conflicts {
                            for (m, r) in &c.target {
                                members[*m].removed.insert(*r);
                            }
                        }
                    }
                    Some("keep_target") => {
                        keep_target_tables.insert(tname.clone());
                        for c in &conflicts {
                            drop_ids.insert(c.id.clone());
                            if let Some(&(m, r)) = c.target.first() {
                                target_rows
                                    .entry(tname.clone())
                                    .or_default()
                                    .push(members[m].table.rows[r].clone());
                            }
                        }
                    }
                    _ => {
                        // Undecided: keep analysing as if the target's row were kept; the blocker stays.
                        for c in &conflicts {
                            drop_ids.insert(c.id.clone());
                            if let Some(&(m, r)) = c.target.first() {
                                target_rows
                                    .entry(tname.clone())
                                    .or_default()
                                    .push(members[m].table.rows[r].clone());
                            }
                        }
                    }
                }
            }
            pt.drop_identities(&drop_ids);
        }
    }

    // ---- the effective target after every decision
    let mut files = overlay.clone();
    let mut shadow: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for members in scan.tables.values() {
        for (path, bytes, removed) in effective::rewrite(members)? {
            st.change(
                "shadow_member",
                &path,
                "",
                "generated pack",
                format!(
                    "{removed} row(s) removed from {path} so every key has exactly one definition"
                ),
            );
            shadow.insert(path.replace('/', "\\"), bytes.clone());
            files.insert(path, bytes);
        }
    }
    let snapshot = if files.is_empty() {
        effective.clone()
    } else {
        effective.with_ntw_overlay(files, Vec::new(), Vec::new())?
    };
    st.set_catalog(&snapshot, &rules0);

    // ---- port rows, rules and references against the final target
    match content::check_port(&port, &snapshot, &rules) {
        Ok(()) => {}
        Err(error) => st.block(
            "port:additive",
            "port",
            "Converted rows are not additive to the target",
            format!("{error:#}"),
        ),
    }
    let view = port.view(&target_rows);
    match content::recycle_problems(&snapshot, &rules, &port, &view) {
        Ok(list) if list.is_empty() => {}
        Ok(list) => {
            let mut h = Hasher::new("recycle-identity");
            for p in &list {
                h.part(p);
            }
            st.drafts.push(Draft::blocker(
                    "recycle:identity".into(),
                    h.finish(),
                    "reference",
                    "Recycling would introduce factions the target does not have".into(),
                    format!(
                        "{} problem(s), for example: {}. No faction key may be added to the target database or the campaign; choose existing owners in the Ownership map.",
                        list.len(),
                        list.iter().take(8).cloned().collect::<Vec<_>>().join("; ")
                    ),
                ));
        }
        Err(error) => st.block(
            "recycle:identity",
            "reference",
            "Recycled faction identities cannot be checked",
            format!("{error:#}"),
        ),
    }
    match content::rule_problems(&snapshot, &rules, &view) {
        Ok(report) => {
            let mut other = Vec::new();
            for (field, key, message) in &report.mapping {
                let mut h = Hasher::new("rule-mapping");
                h.part(field).part(key).part(message);
                st.drafts.push(
                    Draft::blocker(
                        format!("mapping:{field}:{key}:missing_target"),
                        h.finish(),
                        "mapping",
                        format!("Mapping needed for {field} {key}"),
                        message.clone(),
                    )
                    .mapping(mapping_hint(field, key)),
                );
            }
            for p in &report.problems {
                if !report.mapping.iter().any(|(_, _, m)| m == p) {
                    other.push(p.clone());
                }
            }
            if !other.is_empty() {
                let mut h = Hasher::new("rules");
                for p in &other {
                    h.part(p);
                }
                st.drafts.push(Draft::blocker(
                    "reference:rules".into(),
                    h.finish(),
                    "reference",
                    "Campaign rules do not match the target database".into(),
                    format!(
                        "{} inconsistencies, for example: {}",
                        other.len(),
                        other.iter().take(8).cloned().collect::<Vec<_>>().join("; ")
                    ),
                ));
            }
        }
        Err(error) => st.block(
            "reference:rules",
            "reference",
            "Campaign rules cannot be checked against the target database",
            format!("{error:#}"),
        ),
    }
    let present: BTreeSet<String> = db::table_directory(&snapshot, Game::Ntw)
        .into_keys()
        .collect();
    let missing_port = if modded {
        missing_dependencies(&view.tables, &present, &base, &rules)
    } else {
        BTreeMap::new()
    };
    st.block_missing_dependencies(&missing_port);
    match content::reference_problems(&snapshot, &rules, &view.tables, false) {
        Ok(report) => {
            let problems: Vec<&String> = report
                .problems
                .iter()
                .filter(|p| {
                    !missing_port
                        .keys()
                        .any(|r| p.contains(&format!(" -> {r}.")))
                })
                .collect();
            if !problems.is_empty() {
                let mut h = Hasher::new("port-refs");
                for p in &problems {
                    h.part(p);
                }
                st.drafts.push(Draft::blocker(
                    "reference:port".into(),
                    h.finish(),
                    "reference",
                    "Converted rows refer to records the target does not have".into(),
                    format!(
                        "{} dangling reference(s), for example: {}. The target removed or renamed records the converted rows need; map the faction/unit or repair the target.",
                        problems.len(),
                        problems.iter().take(8).map(|p| p.as_str()).collect::<Vec<_>>().join("; ")
                    ),
                ));
            }
        }
        Err(error) => st.block(
            "reference:port",
            "reference",
            "References of the converted rows cannot be checked",
            format!("{error:#}"),
        ),
    }
    if modded {
        target_references(&mut st, &snapshot, &base, &rules, &scan);
    }

    // ---- localisation
    progress("Comparing localisation");
    let loc_plan = match content::collect_loc(&rules, &content::extra_regions(project), MapRows::Curated) {
        Ok(p) => p,
        Err(error) => {
            st.block(
                "loc:stage",
                "loc",
                "Port text cannot be prepared",
                format!("{error:#}"),
            );
            return st.conclude(assets, None);
        }
    };
    let mut loc_keep_target = BTreeSet::new();
    let mut loc_strip: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut loc_decided = BTreeSet::new();
    let loc_vanilla_english = english(base.language()) && (!modded || english(&st.language));
    if modded {
        target_loc_members(&mut st, &snapshot, &loc_plan, &mut loc_strip)?;
    }
    if modded || !english(base.language()) {
        let mut conflicts =
            content::loc_conflicts(&loc_plan, &snapshot, &base, loc_vanilla_english)?;
        conflicts.retain(|c| !loc_strip.get(&c.member).is_some_and(|s| s.contains(&c.key)));
        let mut groups: BTreeMap<String, Vec<&content::LocConflict>> = BTreeMap::new();
        for c in &conflicts {
            groups.entry(c.member.clone()).or_default().push(c);
        }
        let mut use_port: Vec<(String, &content::LocConflict)> = Vec::new();
        for (member, list) in &groups {
            let mut h = Hasher::new("loc-conflict");
            h.part(member);
            for c in list {
                h.part(&c.key)
                    .part(&c.target.0)
                    .part(&c.target.1.to_string())
                    .part(&c.port.0)
                    .part(&c.port.1.to_string());
            }
            let detail = format!(
                "{} string(s) of the Grand Campaign share a key with different text in the target's {member}: {}. keep_target keeps the target's text; use_port writes the converted text.",
                list.len(),
                shown(list.iter().map(|c| c.key.clone()), 10)
            );
            let draft = Draft::blocker(
                format!("loc:{member}"),
                h.finish(),
                "loc",
                format!("Text keys collide with {member}"),
                detail,
            )
            .choices(vec!["keep_target".into(), "use_port".into()]);
            match st.decide(draft).as_deref() {
                Some("keep_target") => {
                    for c in list {
                        loc_keep_target.insert(c.key.clone());
                        loc_decided.insert(c.key.clone());
                    }
                }
                Some("use_port") => {
                    for c in list {
                        loc_decided.insert(c.key.clone());
                        use_port.push((member.clone(), *c));
                    }
                }
                _ => {}
            }
        }
        let main = content::loc_main_paths();
        for (member, c) in use_port {
            if loc_keep_target.contains(&c.key) || member == main[c.port_file] {
                continue;
            }
            loc_strip.entry(member).or_default().insert(c.key.clone());
        }
        if !english(&st.language) {
            let known: BTreeSet<String> = content::effective_loc_members(&snapshot)?
                .into_iter()
                .flat_map(|(_, entries)| entries.into_keys())
                .collect();
            let untranslated = loc_plan
                .rows()
                .filter(|(k, ..)| !known.contains(*k))
                .count();
            st.warnings.push(format!(
                "Target language is {}: {untranslated} converted string(s) are English because the port has no translations for them; the campaign is not fully localised.",
                st.language
            ));
        }
    }

    // ---- localisation carried through after reconciliation; unchanged database inputs stay in the target mod
    let mut dataset_loc_members: BTreeSet<String> = BTreeSet::new();
    if modded {
        dataset_loc_members.extend(
            snapshot
                .names(Game::Ntw, "text/")
                .iter()
                .filter(|n| n.ends_with(".loc"))
                .map(|n| n.replace('/', "\\")),
        );
    }

    // ---- change report
    for (table, pt) in &port.tables {
        if !pt.table.rows.is_empty() {
            st.change(
                "add_rows",
                table,
                "",
                content::TABLE_FILE,
                format!(
                    "{} converted row(s) in db/{table}/{}",
                    pt.table.rows.len(),
                    content::TABLE_FILE
                ),
            );
        }
    }
    let (ui_rows, loc_rows): (usize, usize) = loc_plan.rows().fold(
        (0, 0),
        |(u, l), (_, _, _, file)| if file == 0 { (u + 1, l) } else { (u, l + 1) },
    );
    st.change(
        "merge_loc",
        "text/ui.loc",
        "",
        &st.target_name.clone(),
        format!("{ui_rows} converted string(s) merged into the effective target ui.loc"),
    );
    st.change(
        "merge_loc",
        "text/localisation.loc",
        "",
        &st.target_name.clone(),
        format!("{loc_rows} converted string(s) merged into the effective target localisation.loc"),
    );
    if let Some(d) = &dataset {
        st.change(
            "dependency",
            "",
            &st.pack_name.clone(),
            &st.target_name.clone(),
            "Install the target mod, including its unchanged database tables and assets (models, textures, UI, scripts). The generated pack contains only campaign DB additions/overrides, explicitly reconciled DB members and merged text; it is not a standalone copy of the TSV database.".into(),
        );
        st.change(
            "database_delta",
            "",
            "",
            &st.target_name.clone(),
            format!(
                "{} changed target database member(s) and {} localisation member(s) are carried through; unchanged target DB members are omitted (dataset {}). Campaign additions are listed separately.",
                shadow.len(),
                dataset_loc_members.len(),
                d.manifest.fingerprint
            ),
        );
    }

    let dependencies = match &dataset {
        Some(d) => json!({
            "mode": "tsv",
            "dataset": d.export_manifest(),
            "resources_reviewed": d.manifest.resources_reviewed,
            "adapter": project.adapter,
            "mod_assets": "not included: models, textures, UI and scripts of the mod must be installed manually",
            "database": "target mod required: unchanged TSV database members are not bundled; only campaign additions/overrides and explicitly reconciled target members are exported",
            "load_order": "not established: TSV data carries no pack information; the generated pack name sorts last alphabetically but the real engine load order of the installed packs was not verified",
        }),
        None => json!({"mode": "vanilla"}),
    };
    let plan = Plan {
        modded,
        keep_target_tables,
        loc_keep_target,
        loc_strip,
        loc_decided,
        loc_vanilla_english,
        shadow,
        dataset_loc_members,
        base: base.clone(),
        project_digest: project_digest(project),
    };
    let pack_name = st.pack_name.clone();
    st.conclude(
        assets,
        Some(Built {
            snapshot,
            rules,
            resources: res.members,
            pack_name,
            dependencies,
            plan,
        }),
    )
}

/// Whether any row of `t` holds a non-empty value in column `col`.
fn column_used(t: &db::Table, col: usize) -> bool {
    t.rows
        .iter()
        .any(|r| !matches!(&r[col], Cell::Null) && !matches!(&r[col], Cell::Str(s) if s.is_empty()))
}

/// References actually used by generated rows.
fn used_references(table: &db::Table) -> impl Iterator<Item = (String, String, String)> + '_ {
    content::reference_edges(&table.name, &table.fields)
        .into_iter()
        .filter(|(field, _, _)| table.column(field).is_ok_and(|col| column_used(table, col)))
}

/// Missing, schema-readable DB dependencies. Opaque original resource tables cannot be exported as TSV;
/// their references remain explicitly unverified rather than pretending vanilla supplies them.
fn missing_dependencies<'t>(
    tables: impl IntoIterator<Item = (&'t String, &'t db::Table)>,
    present: &BTreeSet<String>,
    base: &Snapshot,
    rules: &Rules,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (tname, t) in tables {
        for (field, rtab, _) in used_references(t) {
            if present.contains(&rtab) {
                continue;
            }
            if !matches!(
                db::load_table_opt(base, Game::Ntw, rules, &rtab),
                Ok(Some(_))
            ) {
                continue;
            }
            out.entry(rtab)
                .or_default()
                .insert(format!("{tname}.{field}"));
        }
    }
    out
}

impl State<'_> {
    /// Blocks on tables that rows refer to but the dataset does not contain; the original game's rows are never
    /// used instead. Each table is reported once.
    fn block_missing_dependencies(&mut self, missing: &BTreeMap<String, BTreeSet<String>>) {
        for (rtab, users) in missing {
            if !self.missing_reported.insert(rtab.clone()) {
                continue;
            }
            self.block(
                &format!("target:missing_table:{rtab}"),
                "target",
                &format!("Table {rtab} is missing from the TSV dataset"),
                format!(
                    "{} refer to {rtab}, but the dataset has no {rtab} TSV. TSV tables are authoritative and the original game's rows are never used instead. Export {rtab} from the mod with RPFM into the dataset folder and import the dataset again.",
                    shown(users.iter().cloned(), 6)
                ),
            );
        }
    }
}

/// Reports dangling references in supplied target rows as warnings. The conversion is not a validator of the mod:
/// pre-existing original inconsistencies are left out, references into omitted tables stay unverified, and nothing
/// here blocks; the references the campaign depends on (generated rows, rules, mappings) are checked separately.
fn target_references(
    st: &mut State<'_>,
    snapshot: &Snapshot,
    base: &Snapshot,
    rules: &Rules,
    scan: &effective::Scan,
) {
    let mut tables: BTreeMap<String, db::Table> = BTreeMap::new();
    for name in scan.tables.keys() {
        match db::load_table_opt(snapshot, Game::Ntw, rules, name) {
            Ok(Some(t)) => {
                tables.insert(name.clone(), (*t).clone());
            }
            Ok(None) => {}
            Err(error) => {
                st.warnings.push(format!(
                    "Table {name} was not checked for dangling references: {error:#}"
                ));
            }
        }
    }
    let report = match content::reference_problems(snapshot, rules, &tables, true) {
        Ok(r) => r,
        Err(error) => {
            st.warnings.push(format!(
                "Target rows were not checked for dangling references: {error:#}"
            ));
            return;
        }
    };
    let baseline: BTreeSet<String> = {
        let mut vanilla_tables = BTreeMap::new();
        for name in tables.keys() {
            if let Ok(Some(t)) = db::load_table_opt(base, Game::Ntw, rules, name) {
                vanilla_tables.insert(name.clone(), (*t).clone());
            }
        }
        // Check only the decoded baseline. An unrelated opaque table (e.g. models_building) must not
        // make a failed lookup discard every pre-existing dangling-reference diagnostic.
        match base
            .with_ntw_members(BTreeMap::new(), base.names(Game::Ntw, "db/"))
            .and_then(|decoded_only| {
                content::reference_problems(&decoded_only, rules, &vanilla_tables, true)
            }) {
            Ok(report) => report.problems.into_iter().collect(),
            Err(error) => {
                st.warnings.push(format!(
                    "Target rows were not checked for dangling references: the original reference data cannot be compared ({error:#})"
                ));
                return;
            }
        }
    };
    let mut by_table: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for problem in report
        .problems
        .into_iter()
        .filter(|p| !baseline.contains(p))
    {
        let table = problem.split('.').next().unwrap_or_default().to_owned();
        by_table.entry(table).or_default().push(problem);
    }
    for (table, problems) in by_table {
        st.warnings.push(format!(
            "Rows of {table} in the dataset refer to missing records ({} reference(s), for example: {}). The campaign does not read them; the rows stay exactly as imported and are not exported.",
            problems.len(),
            problems.iter().take(5).cloned().collect::<Vec<_>>().join("; ")
        ));
    }
    if report.unverified > 0 {
        st.warnings.push(format!(
            "{} reference value(s) in supplied target rows point into omitted or schema-unsupported tables and were not verified. These do not expand the campaign's required imports; the installed game/mod must provide the referenced records. Original rows are not substituted.",
            report.unverified
        ));
    }
}

// ---------------------------------------------------------------------------------------------
// Required tables and dataset completeness
// ---------------------------------------------------------------------------------------------

/// Minimum target inputs for existing-faction campaign conversion: direct campaign reads, generated-table
/// collision/schema checks, and references actually used by the generated rows. Unchanged target-row
/// references do not expand this set. The two full localisation files are required separately.
pub fn required_tables(base: &Snapshot, rules: &Rules) -> Result<Vec<Requirement>> {
    let mut reasons: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for input in db::CampaignTable::ALL {
        reasons
            .entry(input.name().to_owned())
            .or_default()
            .insert(input.reason().to_owned());
    }
    let generated = content::requirement_rows(rules)
        .context("Read the campaign rows to derive the required imports")?;
    for (table, rows) in &generated {
        reasons
            .entry(table.clone())
            .or_default()
            .insert("the conversion adds converted rows to this table".to_owned());
        for (field, rtab, _) in content::reference_edges(table, &rows.fields) {
            let Ok(col) = rows.column(&field) else {
                continue;
            };
            if !column_used(rows, col) || db::table_files(base, Game::Ntw, &rtab).is_empty() {
                continue;
            }
            reasons.entry(rtab).or_default().insert(format!(
                "referenced by {table}.{field} of the converted rows"
            ));
        }
    }
    Ok(reasons
        .into_iter()
        .map(|(table, reason)| Requirement {
            table,
            reason: reason.into_iter().collect::<Vec<_>>().join("; "),
        })
        .collect())
}

/// Blocks on every required table or text file the dataset does not contain. Returns whether everything is there.
fn dataset_complete(st: &mut State<'_>, effective: &Snapshot, requirements: &[Requirement]) -> bool {
    let present: BTreeSet<String> = db::table_directory(effective, Game::Ntw)
        .into_keys()
        .collect();
    let mut complete = true;
    for req in requirements {
        if present.contains(&req.table) {
            continue;
        }
        complete = false;
        st.missing_reported.insert(req.table.clone());
        st.block(
            &format!("target:missing_table:{}", req.table),
            "target",
            &format!("Required table {} is missing from the TSV dataset", req.table),
            format!(
                "The conversion needs {} ({}). TSV tables are authoritative and the original game's rows are never used instead. Export {} from the mod with RPFM into the dataset folder and import the dataset again.",
                req.table, req.reason, req.table
            ),
        );
    }
    for path in content::loc_main_paths() {
        if !effective.exists(Game::Ntw, &path) {
            complete = false;
            st.block(
                &format!("target:missing_loc:{path}"),
                "target",
                &format!("Required text file {path} is missing from the TSV dataset"),
                format!("The target needs its own full {path}; the original game's text is never used instead. Export it as TSV with RPFM and import the dataset again."),
            );
        }
    }
    complete
}

/// One text key with the definitions the dataset's text members give it: `(member, (text, tooltip))`.
type LocDefs = Vec<(String, (String, u8))>;

/// Keys defined differently by several members, grouped by the (ordered) set of members that define them. Keys
/// whose definitions are identical everywhere are harmless and left out.
fn loc_groups(by_key: BTreeMap<String, LocDefs>) -> BTreeMap<Vec<String>, Vec<(String, LocDefs)>> {
    let mut groups: BTreeMap<Vec<String>, Vec<(String, LocDefs)>> = BTreeMap::new();
    for (key, list) in by_key {
        if list.len() < 2 || list.iter().all(|(_, v)| *v == list[0].1) {
            continue;
        }
        let members: Vec<String> = list.iter().map(|(m, _)| m.clone()).collect();
        groups.entry(members).or_default().push((key, list));
    }
    groups
}

/// Text members of the dataset: a key the conversion writes that is repeated inside one member, or defined
/// differently by several members, needs an explicit decision (the other members are rewritten without the key at
/// their original path). Repeated or differing definitions of any other key are the mod's own and stay untouched.
fn target_loc_members(
    st: &mut State<'_>,
    snapshot: &Snapshot,
    plan: &content::LocPlan,
    strip: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    let main = content::loc_main_paths();
    let mut by_key: BTreeMap<String, Vec<(String, (String, u8))>> = BTreeMap::new();
    for name in snapshot.names(Game::Ntw, "text/") {
        if !name.ends_with(".loc") {
            continue;
        }
        let bytes = db::read_member(snapshot, Game::Ntw, &name)?;
        let entries = match db::decode_loc(&bytes) {
            Ok((_, entries)) => entries,
            Err(error) => {
                st.block(
                    &format!("loc:decode:{name}"),
                    "loc",
                    &format!("Text member {name} cannot be read"),
                    format!("{error:#}"),
                );
                continue;
            }
        };
        let mut seen: BTreeMap<String, (String, u8)> = BTreeMap::new();
        let mut repeated: BTreeSet<String> = BTreeSet::new();
        let mut differing: BTreeSet<String> = BTreeSet::new();
        for (key, text, tip) in entries {
            match seen.get(&key) {
                Some(have) => {
                    repeated.insert(key.clone());
                    if *have != (text.clone(), tip) {
                        differing.insert(key.clone());
                    }
                }
                None => {
                    seen.insert(key, (text, tip));
                }
            }
        }
        // the two main files are merged into full files, which cannot hold a written key twice
        let repeated_here = if main.contains(&name) { &repeated } else { &differing };
        let bad: BTreeSet<&String> = repeated_here
            .iter()
            .filter(|key| plan.contains(key))
            .collect();
        if !bad.is_empty() {
            let mut h = Hasher::new("loc-within");
            h.part(&name);
            for key in &bad {
                h.part(key);
            }
            st.drafts.push(Draft::blocker(
                format!("loc:within:{name}"),
                h.finish(),
                "loc",
                format!("Text keys repeated inside {name}"),
                format!(
                    "{name} defines {} key(s) more than once: {}. The engine's choice between repeated keys is unknown; remove the repeated rows in the mod, export the TSV again and import it.",
                    bad.len(),
                    shown(bad.iter().map(|k| (*k).clone()), 10)
                ),
            ));
        }
        for (key, value) in seen {
            if plan.contains(&key) {
                by_key.entry(key).or_default().push((name.clone(), value));
            }
        }
    }
    for (members, items) in loc_groups(by_key) {
        let mut set_hash = Hasher::new("loc-dup-set");
        for m in &members {
            set_hash.part(m);
        }
        let mut h = Hasher::new("loc-dup");
        for (key, list) in &items {
            h.part(key);
            for (m, (text, tip)) in list {
                h.part(m).part(text).part(&tip.to_string());
            }
        }
        let draft = Draft::blocker(
            format!("loc_dup:{}", &set_hash.finish()[..8]),
            h.finish(),
            "loc",
            "Text keys defined differently by several dataset files".into(),
            format!(
                "{} text key(s) have different text in {}: {}. The engine's behaviour for repeated keys is not verified, so exactly one definition must be chosen; the other files are rewritten without the key (same path, in the generated pack).",
                items.len(),
                shown(members.iter().cloned(), 6),
                shown(items.iter().map(|(k, _)| k.clone()), 10)
            ),
        )
        .choices(members.iter().map(|m| format!("member:{m}")).collect());
        if let Some(choice) = st.decide(draft) {
            let winner = choice.strip_prefix("member:").unwrap_or_default();
            for (key, list) in &items {
                for (m, _) in list {
                    if m != winner {
                        strip.entry(m.clone()).or_default().insert(key.clone());
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------------------------

/// Compatibility report of the project's target. Reads the original-source snapshot and the pinned TSV dataset;
/// builds no terrain and writes nothing.
pub fn analyze(
    workspace: &Path,
    assets: &Path,
    project: &Project,
    progress: &Progress<'_>,
) -> Result<Analysis> {
    Ok(evaluate(workspace, assets, project, progress, false)?.analysis)
}

/// Resolved inputs for the converter. Requires completed ownership selection and no unresolved compatibility blockers.
pub fn prepare(
    workspace: &Path,
    assets: &Path,
    project: &Project,
    progress: &Progress<'_>,
) -> Result<Prepared> {
    let evaluated = evaluate(workspace, assets, project, progress, true)?;
    if !evaluated.analysis.ready {
        let mut lines: Vec<String> = evaluated
            .analysis
            .blockers()
            .take(12)
            .map(|i| format!("  - [{}] {}: {}", i.category, i.title, i.detail))
            .collect();
        if !evaluated.analysis.unassigned_regions.is_empty() {
            lines.insert(0, format!(
                "  - Assign existing target factions in the Ownership map before export; no missing original faction needs to be imported or created. Regions: {}",
                shown(evaluated.analysis.unassigned_regions.iter().cloned(), 12)
            ));
        }
        bail!(
            "The campaign is not ready for conversion ({} compatibility blocker(s), {} region(s) awaiting ownership):\n{}",
            evaluated.analysis.blockers().count(),
            evaluated.analysis.unassigned_regions.len(),
            lines.join("\n")
        );
    }
    let built = evaluated
        .built
        .context("analysis was ready but produced no resolved inputs")?;
    Ok(Prepared {
        snapshot: built.snapshot,
        rules: built.rules,
        analysis: evaluated.analysis,
        resources: built.resources,
        pack_name: built.pack_name,
        dependencies: built.dependencies,
        plan: built.plan,
    })
}

/// Builds the DB/LOC/UI members of the generated pack: the converted tables (minus rows the target already has),
/// rewritten target members, the full merged text files and the UI/script resources. Rechecks everything against
/// the real map output; a row or string that the analysis did not decide aborts the build.
pub fn build_members(
    prepared: &Prepared,
    map: &MapArtifacts,
    project: &Project,
    progress: &Progress<'_>,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let plan = &prepared.plan;
    if project_digest(project) != plan.project_digest {
        bail!("The project changed after it was prepared; analyse and build again.");
    }
    let snapshot = &prepared.snapshot;
    let rules = &prepared.rules;
    progress("Adapting converted rows to the target");
    let mut port = content::stage_port(snapshot, rules, &content::extra_regions(project), MapRows::Artifacts(map))?;
    let mut target_rows: BTreeMap<String, Vec<Vec<Cell>>> = BTreeMap::new();
    if plan.modded {
        let scan = effective::scan_tables(
            snapshot,
            &plan.base,
            rules,
            Some(&content::port_table_names()),
        )?;
        if let Some(problem) = scan
            .schema
            .iter()
            .find(|p| port.tables.contains_key(&p.table))
        {
            bail!("{}: {}", problem.table, problem.detail);
        }
        for (tname, pt) in port.tables.iter_mut() {
            let Some(members) = scan.tables.get(tname) else {
                continue;
            };
            if members.is_empty() {
                continue;
            }
            if pt.table.rows.is_empty() {
                continue;
            }
            if !effective::same_layout(pt, members) {
                bail!(
                    "{tname}: the converted rows use table version {} but the target's table differs",
                    pt.table.version
                );
            }
            let mut drop_ids = BTreeSet::new();
            for c in effective::port_collisions(pt, members) {
                if !c.identical && !plan.keep_target_tables.contains(tname) {
                    bail!(
                        "A converted row of {tname} with key {:?} collides with a target row that the analysis did not decide on. Analyse again and resolve the collision.",
                        c.id.replace('\u{1f}', " / ")
                    );
                }
                if let Some(&(m, r)) = c.target.first() {
                    target_rows
                        .entry(tname.clone())
                        .or_default()
                        .push(members[m].table.rows[r].clone());
                }
                drop_ids.insert(c.id);
            }
            pt.drop_identities(&drop_ids);
        }
    }
    let loc = content::collect_loc(rules, &content::extra_regions(project), MapRows::Artifacts(map))?;
    if plan.modded || !english(plan.base.language()) {
        for c in content::loc_conflicts(&loc, snapshot, &plan.base, plan.loc_vanilla_english)? {
            let stripped = plan
                .loc_strip
                .get(&c.member)
                .is_some_and(|s| s.contains(&c.key));
            if !stripped && !plan.loc_decided.contains(&c.key) {
                bail!(
                    "The string {} collides with the target's {} but the analysis did not decide on it. Analyse again and resolve the collision.",
                    c.key,
                    c.member
                );
            }
        }
    }
    let members = content::build_reconciled(
        snapshot,
        rules,
        &Reconciled {
            port: &port,
            loc: &loc,
            target_rows: &target_rows,
            loc_keep_target: &plan.loc_keep_target,
            loc_strip: &plan.loc_strip,
        },
        progress,
    )?;
    complete_members(prepared, members)
}

/// Complete the campaign delta with localisation, intentional DB replacements and resources.
/// The effective snapshot remains complete for conversion and validation; it is not an export list.
fn complete_members(
    prepared: &Prepared,
    mut members: BTreeMap<String, Vec<u8>>,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let plan = &prepared.plan;
    if plan.modded {
        // Full localisation is deliberately retained; unchanged DB tables come from the installed target.
        for name in &plan.dataset_loc_members {
            if members.contains_key(name) {
                continue; // regenerated text member
            }
            let bytes = db::read_member(&prepared.snapshot, Game::Ntw, name)?;
            members.insert(name.clone(), bytes);
        }
    }
    for (name, bytes) in plan.shadow.iter().chain(prepared.resources.iter()) {
        if members.insert(name.clone(), bytes.clone()).is_some() {
            bail!("Duplicate output pack member {name}");
        }
    }
    validate_members(prepared, &members)?;
    Ok(members)
}

/// Structural check of the members of a build: unique names (case-insensitive), the full text files present,
/// every rewritten target member and resource carried through, nothing from the generated map folders.
pub fn validate_members(prepared: &Prepared, members: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    let mut seen = BTreeSet::new();
    let generated_db: BTreeSet<String> = content::port_table_names()
        .iter()
        .map(|table| format!("db/{table}/{}", content::TABLE_FILE))
        .collect();
    for name in members.keys() {
        if !seen.insert(name.replace('\\', "/").to_lowercase()) {
            bail!("pack member {name} appears twice (names are case-insensitive)");
        }
        let canon = name.replace('\\', "/").to_lowercase();
        if canon.starts_with("campaign_maps/nap_world/") || canon == resources::DONOR_PATH {
            bail!("pack member {name} must not be shipped inside the pack");
        }
        if canon.starts_with("db/") && !generated_db.contains(&canon) && !prepared.plan.shadow.contains_key(name) {
            bail!("pack member {name} is neither a generated campaign table nor an explicitly changed target member");
        }
    }
    for required in ["text\\ui.loc", "text\\localisation.loc"] {
        if !members.contains_key(required) {
            bail!("pack member {required} is missing");
        }
    }
    for name in prepared
        .plan
        .shadow
        .keys()
        .chain(prepared.resources.keys())
        .chain(prepared.plan.dataset_loc_members.iter())
    {
        if !members.contains_key(name) {
            bail!("pack member {name} was planned but not produced");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
