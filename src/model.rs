use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub type Progress<'a> = dyn Fn(&str) + Send + Sync + 'a;

/// Current project file format.
///
/// * v1 (ownership edits only) migrates on read.
/// * v2 (pack-profile targets) migrates on read, see `Project::parse`.
/// * v3 adds `target_mode` and binds `target`/`adapter` to immutable TSV datasets (`crate::target`).
/// * v4 added manual ownership restricted to existing target factions behind a `recycle_existing_factions` option.
/// * v5 makes existing-faction ownership the only behaviour: the option is gone (see `Project::parse`).
/// * v6 adds `splits`: user-drawn region splits applied to the generated world map (`crate::regionmap`).
/// * v7 adds `activations` (wilderness regions made playable); v6 editors would silently drop them, so they refuse v7.
/// * v8 added `theatres` (extended theatre rectangles), `painted_terrain`, and the activation options `areas`,
///   `merge` and `carve_from`; v7 projects parse unchanged (every new field has its neutral default).
/// * v9 fixes the theatre bounds (see `regionmap::theatres`): the `theatres` option is gone. A v8 project's custom
///   rectangles are dropped in memory when it is read — every other decision survives — and the file itself is
///   rewritten without the obsolete option only when the user saves.
pub const PROJECT_FORMAT: u32 = 9;

/// Which database the converter ports into.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetMode {
    /// The original Napoleon: Total War data of the source snapshot.
    #[default]
    Vanilla,
    /// A modded database supplied as an imported TSV dataset. Without a bound dataset nothing can be exported:
    /// this mode never falls back to the vanilla tables.
    Tsv,
}

/// Source -> target-mod key mappings chosen by the user.
///
/// * `factions`: key of a faction of the world rules (the ETW/vanilla faction) -> faction key of the target data
/// * `units`: NTW stand-in unit key (the NTW unit keys the conversion uses for starting forces; ETW unit keys are
///   not accepted) -> unit key of the target data
/// * `buildings`: ETW building key (`rules.buildings` key) or vanilla NTW building level -> target `level_name`
/// * `faction_templates`: *resolved* faction key -> live faction of the target start position used as structural donor
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Mappings {
    #[serde(default)]
    pub factions: BTreeMap<String, String>,
    #[serde(default)]
    pub units: BTreeMap<String, String>,
    #[serde(default)]
    pub buildings: BTreeMap<String, String>,
    #[serde(default)]
    pub faction_templates: BTreeMap<String, String>,
}

impl Mappings {
    pub fn is_empty(&self) -> bool {
        self.factions.is_empty()
            && self.units.is_empty()
            && self.buildings.is_empty()
            && self.faction_templates.is_empty()
    }
}

/// A user decision bound to the exact inputs of the issue it answers: when the reference, target or port inputs change,
/// `input_hash` no longer matches and the choice is not applied.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Resolution {
    pub input_hash: String,
    pub choice: String,
}

/// A region split drawn on the Ownership map. Coordinates are ETW world units (the geometry the generated map
/// keeps verbatim). Splits apply in project order; `source` may itself be the `key` of an earlier split.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RegionSplit {
    /// region being cut
    pub source: String,
    /// index of the source region's area (regions.esf `areas` item) whose outer ring is cut
    pub area: usize,
    /// ring positions (outline vertex order at the time this split applies) where the cut starts and ends; the
    /// forward arc `a..=b` plus the cut line becomes the new region
    pub a: usize,
    pub b: usize,
    /// interior cut points ordered from `a` towards `b` (the order they are drawn)
    pub interior: Vec<[f32; 2]>,
    /// key of the new region (`<source>_<slug>` by default)
    pub key: String,
    /// display name of the new region
    pub name: String,
    /// settlement slug used in slot keys (`settlement:<key>:<settlement>`)
    pub settlement: String,
    pub settlement_name: String,
    /// settlement centre; the source settlement's slots are copied around it
    pub center: [f32; 2],
    /// unique lookup colour of the new region inside its theatre
    pub colour: [u8; 3],
}

impl RegionSplit {
    pub fn settlement_key(&self) -> String {
        format!("settlement:{}:{}", self.key, self.settlement)
    }
}

/// Which areas of an activated region become passable (regions.esf `areas`; a region may consist of several
/// polygons, e.g. the Burma coast and its offshore islets).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AreaSelection {
    /// the area containing the settlement `center` (the only behaviour before format 8)
    #[default]
    #[serde(alias = "Centre")]
    Centre,
    /// every area of the region
    #[serde(alias = "All")]
    All,
    /// exactly these area indices (positions in the region's `areas` as the generated map has them at this point
    /// of the project: after theatre extension and clipping, before any merge)
    #[serde(alias = "Indices")]
    Indices(Vec<usize>),
}

impl AreaSelection {
    /// CLI/GUI spelling: `centre`, `all` or a comma list of area indices (`0,1,2`).
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "centre" | "center" => Ok(Self::Centre),
            "all" => Ok(Self::All),
            list => {
                let indices = list
                    .split(',')
                    .map(|part| part.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .with_context(|| format!("areas must be `centre`, `all` or a comma list of indices, got `{text}`"))?;
                Ok(Self::Indices(indices))
            }
        }
    }

    /// Inverse of [`AreaSelection::parse`].
    pub fn describe(&self) -> String {
        match self {
            Self::Centre => "centre".into(),
            Self::All => "all".into(),
            Self::Indices(list) => list.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
        }
    }
}

/// Activation of a wilderness region: an ETW world land polygon inside a playable theatre that has no settlement
/// and no campaign data (`wilderness_*`, `alps`, ...). Activations apply before splits, in project order; a split
/// may then cut an activated region. The region keeps its regions.esf key; its campaign data (settlement slots,
/// rules, start-position REGION, AI entries) is cloned from `donor`, a playable region of the same theatre.
///
/// Format 8 options: `areas` selects which areas of `key` become passable, `merge` unions the selected areas into
/// one polygon, `carve_from` makes `key` a NEW region carved out of areas of a backdrop region (e.g. an island
/// of `unexplorable`).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RegionActivation {
    /// existing regions.esf key of the wilderness region (with `carve_from`: the key of the new region)
    pub key: String,
    /// playable region whose settlement slots, rules (owner, culture, religion, rebels, resources) and
    /// start-position record serve as the template
    pub donor: String,
    pub name: String,
    pub settlement: String,
    pub settlement_name: String,
    pub center: [f32; 2],
    pub colour: [u8; 3],
    /// which areas of `key` become passable
    #[serde(default)]
    pub areas: AreaSelection,
    /// union the selected areas into one area before the settlement is attached
    #[serde(default)]
    pub merge: bool,
    /// `Some(source)`: `key` does not exist yet; it is created from the `areas` (`Indices`) of region `source`
    #[serde(default)]
    pub carve_from: Option<String>,
}

impl RegionActivation {
    pub fn settlement_key(&self) -> String {
        format!("settlement:{}:{}", self.key, self.settlement)
    }

    /// Structural checks of the format-8 options that need no map: `carve_from` needs exactly one explicit index (the
    /// new region is made of that one area of the source), `merge` needs at least two areas to unite (`All` is
    /// checked against the region by the geometry chain, which knows its areas), indices are unique.
    pub fn validate_options(&self) -> Result<()> {
        if let AreaSelection::Indices(list) = &self.areas {
            if list.is_empty() {
                bail!("Activation {}: the area selection is empty", self.key);
            }
            let mut sorted = list.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != list.len() {
                bail!("Activation {}: the area selection lists an area twice", self.key);
            }
        }
        if let Some(source) = &self.carve_from {
            if source.is_empty() || *source == self.key {
                bail!("Activation {}: carve_from must name another region", self.key);
            }
            if !matches!(&self.areas, AreaSelection::Indices(list) if list.len() == 1) {
                bail!(
                    "Activation {}: a region carved out of {source} is made of exactly one area; give its index (`areas` = one index)",
                    self.key
                );
            }
        }
        if self.merge {
            match &self.areas {
                AreaSelection::Centre => bail!(
                    "Activation {}: merge unites several areas; select `all` or at least two area indices",
                    self.key
                ),
                AreaSelection::Indices(list) if list.len() < 2 => bail!(
                    "Activation {}: merge needs at least two areas, {} selected",
                    self.key,
                    list.len()
                ),
                _ => {}
            }
        }
        Ok(())
    }
}

/// Campaign-map relief of the generated world. The ETW world heightmap squeezes continents into far fewer world
/// units than NTW's own maps and paints uplands about three times higher, which the NTW renderer shows as noisy,
/// bulging terrain. The curve keeps lowlands (`h <= knee`) and compresses everything above by `upland_scale`;
/// `smooth_sigma` is a land-only Gaussian blur in heightmap pixels (3.2 px per world unit). Sea (0) is never changed
/// and land never drops to 0.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TerrainRelief {
    pub knee: u8,
    pub upland_scale: f32,
    pub smooth_sigma: f32,
}

impl Default for TerrainRelief {
    /// Matches NTW's own Europe map statistics (lowland mean ~18, upland p90 ~50, pixel roughness ~3).
    fn default() -> Self {
        Self {
            knee: 24,
            upland_scale: 0.4,
            smooth_sigma: 2.0,
        }
    }
}

impl TerrainRelief {
    /// ETW relief kept as is.
    pub fn original() -> Self {
        Self {
            knee: 255,
            upland_scale: 1.0,
            smooth_sigma: 0.0,
        }
    }

    pub fn is_original(&self) -> bool {
        (self.knee == 255 || self.upland_scale == 1.0) && self.smooth_sigma <= 0.0
    }

    pub fn validate(&self) -> Result<()> {
        if !(0.05..=1.0).contains(&self.upland_scale) {
            bail!("terrain upland_scale must be between 0.05 and 1");
        }
        if !(0.0..=8.0).contains(&self.smooth_sigma) || self.smooth_sigma.is_nan() {
            bail!("terrain smooth_sigma must be between 0 and 8 pixels");
        }
        Ok(())
    }

    /// Height curve for one land pixel (sea pixels are never passed through it).
    pub fn curve(&self, h: u8) -> f32 {
        let h = h as f32;
        let knee = self.knee as f32;
        if h <= knee {
            h
        } else {
            knee + (h - knee) * self.upland_scale
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub format_version: u32,
    pub name: String,
    /// region key -> faction key; faction keys are the *resolved* keys (after `mappings.factions`)
    pub owners: BTreeMap<String, String>,
    #[serde(default)]
    pub target: Option<crate::target::TargetBinding>,
    /// `Tsv` requires `target`: a project in this mode without a dataset blocks every export.
    #[serde(default)]
    pub target_mode: TargetMode,
    #[serde(default)]
    pub mappings: Mappings,
    #[serde(default)]
    pub resolutions: BTreeMap<String, Resolution>,
    #[serde(default)]
    pub adapter: Option<crate::target::AdapterBinding>,
    #[serde(default)]
    pub activations: Vec<RegionActivation>,
    #[serde(default)]
    pub splits: Vec<RegionSplit>,
    #[serde(default)]
    pub terrain: TerrainRelief,
    /// directory holding externally painted terrain for the extended area (supertexture, heightmap, ...; produced
    /// by `tools/terrain_paint`), imported over the generated canvases
    #[serde(default)]
    pub painted_terrain: Option<PathBuf>,
}

impl Default for Project {
    fn default() -> Self {
        Self {
            format_version: PROJECT_FORMAT,
            name: "Grand Campaign".into(),
            owners: BTreeMap::new(),
            target: None,
            target_mode: TargetMode::Vanilla,
            mappings: Mappings::default(),
            resolutions: BTreeMap::new(),
            adapter: None,
            activations: Vec::new(),
            splits: Vec::new(),
            terrain: TerrainRelief::default(),
            painted_terrain: None,
        }
    }
}

impl Project {
    /// Parses a project file, migrating older formats explicitly (owners, name and mappings always survive):
    ///
    /// * v1 carried only name + owners; every later field has its empty default.
    /// * v2 bound a target to an imported *pack profile* (layers of mod packs). Those profiles no longer exist, so
    ///   the binding is never carried over as a valid target: a v2 project that had a target becomes
    ///   `target_mode = Tsv` with no dataset (analysis then asks for a TSV import instead of silently exporting
    ///   vanilla), and its adapter and resolutions are dropped because both were bound to the old profile's exact
    ///   inputs. A v2 project without a target becomes `Vanilla`; its resolutions are kept, its (already unusable)
    ///   unbound adapter is dropped.
    /// * v3-v8 retain all TSV bindings, mappings, owners and resolutions, and get the sole existing-faction
    ///   behaviour whatever their old `recycle_existing_factions` value was. v5-v8 additionally lose only the
    ///   format-8 `theatres` option: custom theatre rectangles are discarded in memory (the fixed bounds of
    ///   `regionmap::theatres` replace them), and the file on disk is not rewritten until the user saves.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut value: Value =
            serde_json::from_slice(bytes).context("Project file is not a valid project")?;
        let version = value
            .get("format_version")
            .and_then(Value::as_u64)
            .context("Project file has no format_version")?;
        match version {
            1 | 3 | 4 | 5 | 6 | 7 | 8 => {}
            2 => migrate_v2(&mut value)?,
            v if v == u64::from(PROJECT_FORMAT) => {}
            other => bail!(
                "Unsupported project version {other} (this editor reads versions 1 to {PROJECT_FORMAT})"
            ),
        }
        if let Some(object) = value.as_object_mut() {
            object.remove("recycle_existing_factions");
            // the fixed theatre bounds replaced the format-8 `theatres` option: the rectangles are dropped here,
            // in memory, and only a user save writes the project back without them
            object.remove("theatres");
        }
        value["format_version"] = Value::from(PROJECT_FORMAT);
        let project: Self = serde_json::from_value(value).context("Project file is not a valid project")?;
        project.validate_structure()?;
        Ok(project)
    }

    /// Checks of the project's own fields that need no map data: activation keys are unique and their options
    /// combine (see [`RegionActivation::validate_options`]). That an area index exists is checked against the map
    /// when the project is built.
    pub fn validate_structure(&self) -> Result<()> {
        for (index, activation) in self.activations.iter().enumerate() {
            if self.activations[..index].iter().any(|other| other.key == activation.key) {
                bail!("Region {} is activated twice", activation.key);
            }
            activation.validate_options()?;
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::parse(&fs::read(path).with_context(|| format!("Read {}", path.display()))?)
            .with_context(|| format!("Load project {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        // Saved files always carry the current format (a project still tagged with an older format in memory is
        // migrated here).
        let mut current = self.clone();
        current.format_version = PROJECT_FORMAT;
        crate::safety::ensure_writable_path(path)?;
        crate::safety::ensure_writable_path(&path.with_extension("json.pending"))?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let pending = path.with_extension("json.pending");
        fs::write(&pending, serde_json::to_vec_pretty(&current)?)?;
        // Same-directory rename atomically replaces the project; failures leave the old file intact.
        fs::rename(&pending, path).with_context(|| format!("Publish project {}", path.display()))
    }
}

/// v2 -> v3 on the JSON value, so no pack-profile type has to survive (see `Project::parse`).
fn migrate_v2(value: &mut Value) -> Result<()> {
    let object = value
        .as_object_mut()
        .context("Project file is not a valid project")?;
    let had_target = object.get("target").is_some_and(|t| !t.is_null());
    object.remove("target");
    object.remove("adapter");
    if had_target {
        object.remove("resolutions");
        object.insert("target_mode".into(), Value::from("tsv"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Rules {
    pub root: PathBuf,
    pub regions: Value,
    pub factions: Value,
    pub units: Value,
    pub buildings: Value,
    pub buildings_by_slot_type: Value,
    pub slot_types: Value,
    pub armies: Value,
}

impl Rules {
    pub fn load(root: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Value> {
            let path = root.join("mapping").join(format!("{name}.json"));
            Ok(serde_json::from_slice(&fs::read(&path).with_context(
                || format!("Read rules {}", path.display()),
            )?)?)
        };
        Ok(Self {
            root: root.into(),
            regions: read("regions")?,
            factions: read("factions")?,
            units: read("units")?,
            buildings: read("buildings")?,
            buildings_by_slot_type: read("buildings_by_slot_type")?,
            slot_types: read("slot_types")?,
            armies: read("armies")?,
        })
    }

    /// Fold the project into the world rules: split regions get a rule entry derived from their source region
    /// (same owner, theatre, culture, religion and rebels; own name, settlement and lookup colour), then the
    /// ownership overrides apply and every faction's territory summary is recomputed.
    pub fn apply_project(&mut self, project: &Project) -> Result<()> {
        let regions = self
            .regions
            .as_object_mut()
            .context("Region rules must be an object")?;
        for activation in &project.activations {
            let mut rule = regions
                .get(&activation.donor)
                .cloned()
                .with_context(|| format!("Activation donor region {} is unknown", activation.donor))?;
            if regions.contains_key(&activation.key) {
                bail!("Region {} is already a campaign region", activation.key);
            }
            rule["name"] = Value::String(activation.name.clone());
            rule["capital_settlement"] = Value::String(activation.settlement_key());
            rule["capital_settlement_loc_key"] = Value::String(format!(
                "start_pos_settlements_onscreen_name_{}",
                activation.settlement_key()
            ));
            rule["capital_settlement_name"] = Value::String(activation.settlement_name.clone());
            rule["lookup_rgb"] = Value::from(activation.colour.to_vec());
            rule["ntw_analog"] = Value::Null;
            rule["notes"] = Value::String(format!("activated wilderness, template {}", activation.donor));
            regions.insert(activation.key.clone(), rule);
        }
        for split in &project.splits {
            let mut rule = regions
                .get(&split.source)
                .cloned()
                .with_context(|| format!("Split source region {} is unknown", split.source))?;
            if regions.contains_key(&split.key) {
                bail!("Split region key {} already exists", split.key);
            }
            rule["name"] = Value::String(split.name.clone());
            rule["capital_settlement"] = Value::String(split.settlement_key());
            rule["capital_settlement_loc_key"] = Value::String(format!(
                "start_pos_settlements_onscreen_name_{}",
                split.settlement_key()
            ));
            rule["capital_settlement_name"] = Value::String(split.settlement_name.clone());
            rule["lookup_rgb"] = Value::from(split.colour.to_vec());
            rule["ntw_analog"] = Value::Null;
            rule["notes"] = Value::String(format!("split from {}", split.source));
            regions.insert(split.key.clone(), rule);
        }
        let factions = self
            .factions
            .as_object_mut()
            .context("Faction rules must be an object")?;
        for (region, faction) in &project.owners {
            if !factions.contains_key(faction) {
                bail!("Unknown faction {faction} assigned to {region}");
            }
            if !regions.contains_key(region) {
                bail!("Unknown region {region}");
            }
        }
        for (region, faction) in &project.owners {
            regions.get_mut(region).expect("validated region")["owner"] =
                Value::String(faction.clone());
        }
        for (key, faction) in factions {
            let held: Vec<String> = regions
                .iter()
                .filter(|(_, r)| r["owner"].as_str() == Some(key.as_str()))
                .map(|(key, _)| key.clone())
                .collect();
            let old_capital = faction["capital"].as_str().map(str::to_owned);
            let capital = old_capital
                .filter(|old| held.contains(old))
                .or_else(|| held.first().cloned());
            faction["regions"] = Value::from(held.len());
            faction["alive"] = Value::Bool(!held.is_empty());
            faction["capital"] = capital.map(Value::String).unwrap_or(Value::Null);
            // A landless faction cannot be offered as a playable campaign start.
            if held.is_empty() {
                faction["playable"] = Value::Bool(false);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildReport {
    pub output: PathBuf,
    pub region_count: usize,
    pub faction_count: usize,
    pub ownership_changes: usize,
    pub files: usize,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rules() -> Rules {
        Rules {
            root: PathBuf::new(),
            regions: json!({"france":{"owner":"france"},"corsica":{"owner":"france"},"england":{"owner":"britain"}}),
            factions: json!({
                "france":{"capital":"france","alive":true,"playable":true,"regions":2},
                "britain":{"capital":"england","alive":true,"playable":true,"regions":1},
                "belgium":{"capital":null,"alive":false,"playable":false,"regions":0}
            }),
            units: Value::Null,
            buildings: Value::Null,
            buildings_by_slot_type: Value::Null,
            slot_types: Value::Null,
            armies: Value::Null,
        }
    }

    #[test]
    fn losing_a_capital_and_reviving_a_faction_updates_holdings() {
        let mut rules = rules();
        let mut project = Project::default();
        project.owners.insert("france".into(), "belgium".into());
        rules.apply_project(&project).unwrap();
        assert_eq!(rules.factions["france"]["capital"], "corsica");
        assert_eq!(rules.factions["france"]["regions"], 1);
        assert_eq!(rules.factions["belgium"]["capital"], "france");
        assert_eq!(rules.factions["belgium"]["alive"], true);
        assert_eq!(rules.factions["belgium"]["regions"], 1);
        assert_eq!(rules.factions["britain"]["capital"], "england");
    }

    #[test]
    fn losing_all_land_removes_the_playable_start() {
        let mut rules = rules();
        let mut project = Project::default();
        project.owners.insert("france".into(), "britain".into());
        project.owners.insert("corsica".into(), "britain".into());
        rules.apply_project(&project).unwrap();
        assert_eq!(rules.factions["france"]["alive"], false);
        assert_eq!(rules.factions["france"]["playable"], false);
        assert_eq!(rules.factions["france"]["capital"], Value::Null);
        assert_eq!(rules.factions["france"]["regions"], 0);
        assert_eq!(rules.factions["britain"]["regions"], 3);
    }

    #[test]
    fn invalid_override_does_not_partially_change_ownership() {
        let mut rules = rules();
        let before = rules.regions.clone();
        let mut project = Project::default();
        project.owners.insert("corsica".into(), "britain".into());
        project
            .owners
            .insert("unknown_region".into(), "britain".into());
        assert!(rules.apply_project(&project).is_err());
        assert_eq!(rules.regions, before);
    }

    #[test]
    fn version_one_project_migrates_without_losing_owners() {
        let v1 = br#"{"format_version":1,"name":"Old","owners":{"corsica":"britain","france":"belgium"}}"#;
        let project = Project::parse(v1).unwrap();
        assert_eq!(project.format_version, PROJECT_FORMAT);
        assert_eq!(project.name, "Old");
        assert_eq!(project.owners.len(), 2);
        assert_eq!(project.owners["corsica"], "britain");
        assert!(project.target.is_none() && project.adapter.is_none());
        assert!(project.mappings.is_empty() && project.resolutions.is_empty());
        assert_eq!(project.target_mode, TargetMode::Vanilla);
    }

    const V2_WITH_PROFILE: &str = r#"{
        "format_version": 2, "name": "Modded", "owners": {"corsica": "mod_france"},
        "target": {"id": "p0123456789abcdef01234567", "fingerprint": "ff"},
        "mappings": {"factions": {"france": "mod_france"}, "units": {"Inf_Line_French": "Mod_Line"}},
        "resolutions": {"issue-1": {"input_hash": "abc", "choice": "use_port"}},
        "adapter": {"id": "a0123", "fingerprint": "ee", "profile_fingerprint": "ff"}
    }"#;

    #[test]
    fn version_two_with_a_pack_profile_requires_a_tsv_import_and_keeps_owners_and_mappings() {
        let project = Project::parse(V2_WITH_PROFILE.as_bytes()).unwrap();
        assert_eq!(project.format_version, PROJECT_FORMAT);
        // Never downgraded to vanilla and never keeps the obsolete profile binding.
        assert_eq!(project.target_mode, TargetMode::Tsv);
        assert!(project.target.is_none() && project.adapter.is_none());
        assert!(project.resolutions.is_empty());
        assert_eq!(project.name, "Modded");
        assert_eq!(project.owners["corsica"], "mod_france");
        assert_eq!(project.mappings.factions["france"], "mod_france");
        assert_eq!(project.mappings.units["Inf_Line_French"], "Mod_Line");
    }

    #[test]
    fn version_two_without_a_target_is_vanilla_and_keeps_resolutions() {
        let mut value: Value = serde_json::from_str(V2_WITH_PROFILE).unwrap();
        value["target"] = Value::Null;
        let project = Project::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(project.target_mode, TargetMode::Vanilla);
        assert!(project.target.is_none() && project.adapter.is_none());
        assert_eq!(project.resolutions["issue-1"].choice, "use_port");
        assert_eq!(project.mappings.factions["france"], "mod_france");
    }

    #[test]
    fn current_project_preserves_dataset_binding_and_mode() {
        let mut project = Project::default();
        project.target_mode = TargetMode::Tsv;
        project.target = Some(crate::target::TargetBinding {
            id: format!("t{}", "a".repeat(24)),
            fingerprint: "f".repeat(64),
        });
        let bytes = serde_json::to_vec(&project).unwrap();
        let back = Project::parse(&bytes).unwrap();
        assert_eq!(back.target_mode, TargetMode::Tsv);
        assert_eq!(back.target, project.target);
        // The removed option is never written again.
        let saved: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(saved.get("recycle_existing_factions").is_none());
    }

    #[test]
    fn current_project_roundtrips_mappings_and_resolutions_and_refuses_future_versions() {
        let mut project = Project::default();
        project.owners.insert("corsica".into(), "mod_france".into());
        project
            .mappings
            .factions
            .insert("france".into(), "mod_france".into());
        project
            .mappings
            .faction_templates
            .insert("mod_france".into(), "hessen".into());
        project.resolutions.insert(
            "issue-1".into(),
            Resolution {
                input_hash: "abc".into(),
                choice: "use_port".into(),
            },
        );
        let bytes = serde_json::to_vec(&project).unwrap();
        let back = Project::parse(&bytes).unwrap();
        assert_eq!(back.mappings, project.mappings);
        assert_eq!(back.resolutions["issue-1"].choice, "use_port");
        assert_eq!(back.owners, project.owners);

        let mut future: Value = serde_json::from_slice(&bytes).unwrap();
        future["format_version"] = Value::from(PROJECT_FORMAT + 1);
        let err = Project::parse(&serde_json::to_vec(&future).unwrap()).unwrap_err();
        assert!(format!("{err:#}").contains("Unsupported project version"));
    }

    #[test]
    fn version_three_migrates_without_losing_tsv_decisions() {
        let old = serde_json::json!({
            "format_version": 3, "name": "TSV project", "owners": {"corsica": "mod_france"},
            "target_mode": "tsv",
            "target": {"id": "told", "fingerprint": "target-fingerprint"},
            "adapter": {"id": "aold", "fingerprint": "adapter-fingerprint", "target_fingerprint": "target-fingerprint"},
            "mappings": {"factions": {"france": "mod_france"}},
            "resolutions": {"collision": {"input_hash": "hash", "choice": "keep_target"}}
        });
        let project = Project::parse(&serde_json::to_vec(&old).unwrap()).unwrap();
        assert_eq!(project.format_version, PROJECT_FORMAT);
        assert_eq!(project.owners["corsica"], "mod_france");
        assert_eq!(project.target_mode, TargetMode::Tsv);
        assert_eq!(project.target.unwrap().fingerprint, "target-fingerprint");
        assert_eq!(
            project.adapter.unwrap().target_fingerprint,
            "target-fingerprint"
        );
        assert_eq!(project.mappings.factions["france"], "mod_france");
        assert_eq!(project.resolutions["collision"].choice, "keep_target");
    }

    /// Both values of the removed option migrate to the one behaviour, keeping every decision untouched.
    #[test]
    fn version_four_migrates_either_recycling_value_without_reassigning_owners() {
        for old_value in [true, false] {
            let old = serde_json::json!({
                "format_version": 4, "name": "Old", "owners": {"corsica": "target_only", "silesia": "prussia"},
                "target_mode": "tsv", "recycle_existing_factions": old_value,
                "target": {"id": "told", "fingerprint": "target-fingerprint"},
                "adapter": {"id": "aold", "fingerprint": "adapter-fingerprint", "target_fingerprint": "target-fingerprint"},
                "mappings": {"factions": {"france": "mod_france"}, "units": {"Inf_Line_French": "Mod_Line"}},
                "resolutions": {"collision": {"input_hash": "hash", "choice": "keep_target"}}
            });
            let project = Project::parse(&serde_json::to_vec(&old).unwrap()).unwrap();
            assert_eq!(project.format_version, PROJECT_FORMAT);
            assert_eq!(project.owners["corsica"], "target_only");
            assert_eq!(project.owners["silesia"], "prussia");
            assert_eq!(project.target_mode, TargetMode::Tsv);
            assert_eq!(project.target.unwrap().fingerprint, "target-fingerprint");
            assert_eq!(project.adapter.unwrap().id, "aold");
            assert_eq!(project.mappings.factions["france"], "mod_france");
            assert_eq!(project.mappings.units["Inf_Line_French"], "Mod_Line");
            assert_eq!(project.resolutions["collision"].choice, "keep_target");
        }
    }

    #[test]
    fn v5_project_loads_with_empty_splits() {
        let v5 = br#"{"format_version":5,"name":"Five","owners":{"corsica":"britain"}}"#;
        let project = Project::parse(v5).unwrap();
        assert_eq!(project.format_version, PROJECT_FORMAT);
        assert!(project.splits.is_empty());
        assert_eq!(project.owners["corsica"], "britain");
    }

    fn corsica_split() -> RegionSplit {
        RegionSplit {
            source: "corsica".into(),
            area: 0,
            a: 3,
            b: 9,
            interior: vec![[1.0, 2.0]],
            key: "corsica_north".into(),
            name: "North Corsica".into(),
            settlement: "bastia".into(),
            settlement_name: "Bastia".into(),
            center: [10.0, 20.0],
            colour: [1, 2, 3],
        }
    }

    #[test]
    fn apply_project_derives_the_split_rule_from_its_source() {
        let mut rules = rules();
        let mut project = Project::default();
        project.splits.push(corsica_split());
        rules.apply_project(&project).unwrap();
        let rule = &rules.regions["corsica_north"];
        assert_eq!(rule["owner"], "france");
        assert_eq!(rule["name"], "North Corsica");
        assert_eq!(rule["capital_settlement"], "settlement:corsica_north:bastia");
        assert_eq!(
            rule["capital_settlement_loc_key"],
            "start_pos_settlements_onscreen_name_settlement:corsica_north:bastia"
        );
        assert_eq!(rule["capital_settlement_name"], "Bastia");
        assert_eq!(rule["lookup_rgb"], json!([1, 2, 3]));
        assert_eq!(rule["ntw_analog"], Value::Null);
        assert_eq!(rules.regions["corsica"]["owner"], "france");
        assert_eq!(rules.factions["france"]["regions"], 3);
    }

    #[test]
    fn split_region_can_be_given_to_another_owner_and_keys_must_be_new() {
        let mut rules = rules();
        let mut project = Project::default();
        project.splits.push(corsica_split());
        project.owners.insert("corsica_north".into(), "britain".into());
        rules.apply_project(&project).unwrap();
        assert_eq!(rules.regions["corsica_north"]["owner"], "britain");
        assert_eq!(rules.regions["corsica"]["owner"], "france");
        assert_eq!(rules.factions["britain"]["regions"], 2);
        assert_eq!(rules.factions["france"]["regions"], 2);

        let mut clash = Project::default();
        let mut split = corsica_split();
        split.key = "england".into();
        clash.splits.push(split);
        assert!(rules_clone_error(&clash));
        let mut unknown = Project::default();
        let mut split = corsica_split();
        split.source = "atlantis".into();
        unknown.splits.push(split);
        assert!(rules_clone_error(&unknown));
    }

    #[test]
    fn v6_project_without_activations_loads() {
        let v6 = br#"{"format_version":6,"name":"Six","owners":{"corsica":"britain"}}"#;
        let project = Project::parse(v6).unwrap();
        assert_eq!(project.format_version, PROJECT_FORMAT);
        assert!(project.activations.is_empty() && project.splits.is_empty());
        let back = Project::parse(&serde_json::to_vec(&project).unwrap()).unwrap();
        assert!(back.activations.is_empty());
    }

    fn plains_activation() -> RegionActivation {
        RegionActivation {
            key: "wilderness_plains".into(),
            donor: "england".into(),
            name: "Plains".into(),
            settlement: "fort_plains".into(),
            settlement_name: "Fort Plains".into(),
            center: [5.0, 6.0],
            colour: [7, 8, 9],
            ..Default::default()
        }
    }

    #[test]
    fn activation_round_trips_and_apply_project_derives_its_rule_from_the_donor() {
        let mut project = Project::default();
        project.activations.push(plains_activation());
        let back = Project::parse(&serde_json::to_vec(&project).unwrap()).unwrap();
        assert_eq!(back.activations, project.activations);

        let mut rules = rules();
        rules.apply_project(&project).unwrap();
        let rule = &rules.regions["wilderness_plains"];
        assert_eq!(rule["owner"], "britain");
        assert_eq!(rule["name"], "Plains");
        assert_eq!(rule["capital_settlement"], "settlement:wilderness_plains:fort_plains");
        assert_eq!(
            rule["capital_settlement_loc_key"],
            "start_pos_settlements_onscreen_name_settlement:wilderness_plains:fort_plains"
        );
        assert_eq!(rule["capital_settlement_name"], "Fort Plains");
        assert_eq!(rule["lookup_rgb"], json!([7, 8, 9]));
        assert_eq!(rule["ntw_analog"], Value::Null);
        assert_eq!(rules.regions["england"]["owner"], "britain");
        assert_eq!(rules.factions["britain"]["regions"], 2);

        // an ownership override applies to the activated key like to any region
        project.owners.insert("wilderness_plains".into(), "france".into());
        let mut rules = self::rules();
        rules.apply_project(&project).unwrap();
        assert_eq!(rules.regions["wilderness_plains"]["owner"], "france");
        assert_eq!(rules.factions["france"]["regions"], 3);
    }

    #[test]
    fn activation_rejects_campaign_regions_and_unknown_donors() {
        let mut existing = Project::default();
        let mut activation = plains_activation();
        activation.key = "corsica".into();
        existing.activations.push(activation);
        assert!(rules_clone_error(&existing));
        let mut unknown = Project::default();
        let mut activation = plains_activation();
        activation.donor = "atlantis".into();
        unknown.activations.push(activation);
        assert!(rules_clone_error(&unknown));
        let mut twice = Project::default();
        twice.activations.push(plains_activation());
        twice.activations.push(plains_activation());
        assert!(rules_clone_error(&twice));
    }

    #[test]
    fn activations_apply_before_splits_so_a_split_may_cut_an_activated_region() {
        let mut cut = corsica_split();
        cut.source = "wilderness_plains".into();
        cut.key = "plains_north".into();

        let mut project = Project::default();
        project.activations.push(plains_activation());
        project.splits.push(cut.clone());
        let mut rules = rules();
        rules.apply_project(&project).unwrap();
        assert_eq!(rules.regions["plains_north"]["owner"], "britain");
        assert_eq!(rules.regions["plains_north"]["notes"], "split from wilderness_plains");
        assert_eq!(rules.factions["britain"]["regions"], 3);

        // a split must not target a wilderness key that was not activated
        let mut inactive = Project::default();
        inactive.splits.push(cut);
        assert!(rules_clone_error(&inactive));
    }

    #[test]
    fn v7_project_parses_with_neutral_format_8_defaults() {
        let v7 = br#"{"format_version":7,"name":"Seven","owners":{},"activations":[
            {"key":"w","donor":"england","name":"W","settlement":"s","settlement_name":"S","center":[1.0,2.0],"colour":[1,2,3]}]}"#;
        let project = Project::parse(v7).unwrap();
        assert_eq!(project.format_version, PROJECT_FORMAT);
        assert!(project.painted_terrain.is_none());
        let activation = &project.activations[0];
        assert_eq!(activation.areas, AreaSelection::Centre);
        assert!(!activation.merge && activation.carve_from.is_none());
    }

    #[test]
    fn v8_theatre_rectangles_are_dropped_but_every_other_decision_survives() {
        let v8 = br#"{"format_version":8,"name":"Eight","owners":{"corsica":"britain"},
            "theatres":[{"theatre":"3","rect":[[440.0,-84.0],[800.0,280.0]]},
                        {"theatre":"1","rect":[[-190.0,-190.0],[190.0,250.0]]}],
            "splits":[{"source":"corsica","area":0,"a":3,"b":9,"interior":[[1.0,2.0]],"key":"corsica_north",
                       "name":"North Corsica","settlement":"bastia","settlement_name":"Bastia",
                       "center":[10.0,20.0],"colour":[1,2,3]}],
            "terrain":{"knee":30,"upland_scale":0.5,"smooth_sigma":3.0},
            "painted_terrain":"painted",
            "activations":[
                {"key":"burma","donor":"england","name":"Burma","settlement":"fort","settlement_name":"Fort",
                 "center":[1.0,2.0],"colour":[1,2,3],"areas":"all","merge":true},
                {"key":"sumatra","donor":"england","name":"Sumatra","settlement":"fort","settlement_name":"Fort",
                 "center":[3.0,4.0],"colour":[4,5,6],"carve_from":"unexplorable","areas":{"indices":[31]}}]}"#;
        let project = Project::parse(v8).unwrap();
        assert_eq!(project.format_version, PROJECT_FORMAT);
        assert_eq!(project.name, "Eight");
        assert_eq!(project.owners["corsica"], "britain");
        assert_eq!(project.splits.len(), 1);
        assert_eq!(project.splits[0].key, "corsica_north");
        assert_eq!(project.terrain.upland_scale, 0.5);
        assert_eq!(project.painted_terrain, Some("painted".into()));
        assert_eq!(project.activations.len(), 2);
        assert_eq!(project.activations[0].areas, AreaSelection::All);
        assert!(project.activations[0].merge);
        assert_eq!(project.activations[1].carve_from.as_deref(), Some("unexplorable"));
        assert_eq!(project.activations[1].areas, AreaSelection::Indices(vec![31]));

        // the current format round-trips everything that survived and never writes the removed option
        let bytes = serde_json::to_vec_pretty(&project).unwrap();
        let stored: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(stored.get("theatres").is_none());
        let back = Project::parse(&bytes).unwrap();
        assert_eq!(back.splits, project.splits);
        assert_eq!(back.activations, project.activations);
        assert_eq!(back.painted_terrain, project.painted_terrain);
        assert_eq!(back.terrain, project.terrain);
        assert_eq!(back.format_version, PROJECT_FORMAT);

        // the area selection is plain JSON a user can write by hand
        let by_hand = format!(
            r#"{{"format_version":{},"name":"x","owners":{{}},"activations":[
                {{"key":"k","donor":"d","name":"K","settlement":"s","settlement_name":"S","center":[0,0],"colour":[1,2,3],
                 "areas":{{"indices":[0,1,2]}},"merge":true}}]}}"#,
            PROJECT_FORMAT
        );
        assert_eq!(Project::parse(by_hand.as_bytes()).unwrap().activations[0].areas, AreaSelection::Indices(vec![0, 1, 2]));
    }

    #[test]
    fn structure_validation_rejects_bad_option_combinations() {
        let parse_err = |project: &Project| Project::parse(&serde_json::to_vec(project).unwrap()).is_err();
        let with = |edit: &dyn Fn(&mut RegionActivation)| {
            let mut project = Project::default();
            let mut activation = plains_activation();
            edit(&mut activation);
            project.activations.push(activation);
            project
        };
        assert!(parse_err(&with(&|a| a.merge = true))); // merge of the centre area only
        assert!(parse_err(&with(&|a| {
            a.merge = true;
            a.areas = AreaSelection::Indices(vec![3]);
        })));
        assert!(!parse_err(&with(&|a| {
            a.merge = true;
            a.areas = AreaSelection::All;
        })));
        assert!(parse_err(&with(&|a| a.areas = AreaSelection::Indices(vec![])))); // empty selection
        assert!(parse_err(&with(&|a| a.areas = AreaSelection::Indices(vec![1, 1])))); // duplicate area
        assert!(parse_err(&with(&|a| a.carve_from = Some("unexplorable".into())))); // carve needs indices
        assert!(parse_err(&with(&|a| {
            a.carve_from = Some(a.key.clone());
            a.areas = AreaSelection::Indices(vec![1]);
        })));
        assert!(!parse_err(&with(&|a| {
            a.carve_from = Some("unexplorable".into());
            a.areas = AreaSelection::Indices(vec![31]);
        })));
    }

    #[test]
    fn unknown_future_format_is_refused() {
        let future = format!(
            r#"{{"format_version":{},"name":"x","owners":{{}}}}"#,
            PROJECT_FORMAT + 1
        );
        let err = Project::parse(future.as_bytes()).unwrap_err();
        assert!(format!("{err:#}").contains("Unsupported project version"));
    }

    fn rules_clone_error(project: &Project) -> bool {
        rules().apply_project(project).is_err()
    }
}
