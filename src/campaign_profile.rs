//! Resolution of the curated world rules against a target mod: the user's faction / unit / building mappings and
//! structural donors are applied to the base [`Rules`], every reference is checked against the *effective* target
//! (database rows of the resolved snapshot and the faction records of its `eur_napoleon` start position), and ownership
//! edits are applied on top. Nothing falls back silently: a reference that does not exist in the target and is not mapped
//! is a [`MappingProblem`] with the exact mapping that fixes it.
//!
//! Mapping semantics (`Project::mappings`):
//! * `factions`: key of a faction of the world rules (ETW / vanilla key, or a rebel faction used by the regions) ->
//!   existing faction key of the target. A mapped faction takes the target's identity: its rows come from the target
//!   database. Region owners are exact keys of the target's factions table; no faction key is ever created.
//! * `units`: NTW stand-in unit key (the keys of the curated starting armies / `rules.units` values) -> unit key of the
//!   target. ETW-only keys are rejected: the conversion generates forces from the stand-ins only.
//! * `buildings`: ETW building key (`rules.buildings` key) or vanilla NTW building level -> target `level_name`; the
//!   level's chain must be buildable (`building_chain_to_slots`) in the slot types the building occupies.
//! * `faction_templates`: resolved faction key -> live faction of the target start position used as structural donor.
//!
//! Resolved `rules.factions` entries carry the original fields plus: `etw_source` (key in the base rules), `source`
//! (always `"ntw"`: every faction is a row of the target's factions table), `ntw_template` and `donor` (explicit
//! structural donor). A world faction the target does not define is omitted, never created.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::db::{self, CampaignTable};
use crate::model::{Mappings, Project, Rules};
use crate::snapshot::Snapshot;
use crate::startpos::core::{faction_infos, startpos_structure};
use crate::startpos::factions::{StartposStructure, identity_from_table, structure_for};

/// A reference that cannot be resolved against the target, with the mapping field and source key that fixes it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MappingProblem {
    /// `factions` | `units` | `buildings` | `faction_templates` | `owners` | `startpos` | `rules`
    pub field: String,
    /// source key (mapping key; region key for owners)
    pub key: String,
    pub message: String,
    /// `missing_target` | `unmapped_missing` | `collision` | `wrong_kind` | `no_template` | `no_donor` |
    /// `invalid_mapping` | `invalid_rules` | `unsupported_startpos` | `existing_owner_required` | `existing_rebel_required`
    pub kind: String,
}

/// How one resolved faction of the campaign is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactionPlan {
    pub key: String,
    pub etw_source: String,
    pub donor: Option<String>,
}

/// Everything of the target the rules are checked against.
#[derive(Clone, Debug)]
pub struct TargetCatalog {
    pub factions: BTreeSet<String>,
    /// Existing faction -> configured target rebellion identity, never a guessed recipient.
    pub rebel_factions: BTreeMap<String, String>,
    pub units: BTreeSet<String>,
    pub land_units: BTreeSet<String>,
    pub naval_units: BTreeSet<String>,
    pub building_levels: BTreeSet<String>,
    /// building level -> chain (`building_levels.chain`)
    pub level_chain: BTreeMap<String, String>,
    /// chain -> slot types it can be built in (`building_chain_to_slots`)
    pub chain_slots: BTreeMap<String, BTreeSet<String>>,
    /// faction -> whether its row can be carried into the start position (name, flags, 8-bit colours), else why not
    pub identity: BTreeMap<String, std::result::Result<(), String>>,
    /// faction records / unit templates of the target `campaigns/eur_napoleon/startpos.esf`, or why it cannot be used
    pub startpos: std::result::Result<Arc<StartposStructure>, String>,
}

fn key_set(
    snapshot: &Snapshot,
    rules: &Rules,
    table: CampaignTable,
    column: &str,
) -> Result<BTreeSet<String>> {
    let t = db::load_campaign_table(snapshot, rules, table)
        .with_context(|| format!("Read target table {}", table.name()))?;
    let c = t.column(column)?;
    Ok(t.rows
        .iter()
        .filter_map(|r| r[c].as_str().map(str::to_owned))
        .collect())
}

impl TargetCatalog {
    /// Reads the effective target (`snapshot` must be the resolved snapshot: its NTW tables are the TSV dataset's).
    pub fn load(snapshot: &Snapshot, rules: &Rules) -> Result<Self> {
        let levels = db::load_campaign_table(snapshot, rules, CampaignTable::BuildingLevels)
            .context("Read target table building_levels")?;
        let (lc, cc) = (levels.column("level_name")?, levels.column("chain")?);
        let level_chain = levels
            .rows
            .iter()
            .filter_map(|r| Some((r[lc].as_str()?.to_owned(), r[cc].as_str()?.to_owned())))
            .collect();
        let slots = db::load_campaign_table(snapshot, rules, CampaignTable::BuildingSlots)
            .context("Read target table building_chain_to_slots")?;
        let (sc, sl) = (slots.column("chain")?, slots.column("slot")?);
        let mut chain_slots: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for r in &slots.rows {
            if let (Some(c), Some(s)) = (r[sc].as_str(), r[sl].as_str()) {
                chain_slots
                    .entry(c.to_owned())
                    .or_default()
                    .insert(s.to_owned());
            }
        }
        let fac = db::load_campaign_table(snapshot, rules, CampaignTable::Factions)
            .context("Read target table factions")?;
        let fk = fac.column("key")?;
        let identity = fac
            .rows
            .iter()
            .filter_map(|r| r[fk].as_str())
            .map(|k| {
                (
                    k.to_owned(),
                    identity_from_table(&fac, k)
                        .map(|_| ())
                        .map_err(|e| format!("{e:#}")),
                )
            })
            .collect();
        let rebel_factions = if let Ok(rebel) = fac.column("settler_rebellion_faction") {
            fac.rows
                .iter()
                .filter_map(|row| {
                    Some((
                        row[fk].as_str()?.to_owned(),
                        row[rebel].as_str()?.to_owned(),
                    ))
                })
                .collect()
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            factions: fac
                .rows
                .iter()
                .filter_map(|r| r[fk].as_str().map(str::to_owned))
                .collect(),
            rebel_factions,
            units: key_set(snapshot, rules, CampaignTable::Units, "key")?,
            land_units: key_set(snapshot, rules, CampaignTable::LandStats, "key")?,
            naval_units: key_set(snapshot, rules, CampaignTable::NavalStats, "key")?,
            building_levels: key_set(snapshot, rules, CampaignTable::BuildingLevels, "level_name")?,
            level_chain,
            chain_slots,
            identity,
            startpos: startpos_structure(snapshot).map_err(|e| format!("{e:#}")),
        })
    }
}

#[derive(Default)]
struct Problems {
    list: Vec<MappingProblem>,
    seen: BTreeSet<(String, String, String)>,
}

impl Problems {
    fn add(&mut self, field: &str, key: &str, kind: &str, message: String) {
        if self
            .seen
            .insert((field.to_owned(), key.to_owned(), kind.to_owned()))
        {
            self.list.push(MappingProblem {
                field: field.to_owned(),
                key: key.to_owned(),
                message,
                kind: kind.to_owned(),
            });
        }
    }
}

fn obj<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object()
        .with_context(|| format!("{what} must be a JSON object"))
}

fn sample<'a>(keys: impl Iterator<Item = &'a str>) -> String {
    let all: Vec<&str> = keys.collect();
    let shown: Vec<&str> = all.iter().copied().take(12).collect();
    format!(
        "{}{}",
        shown.join(", "),
        if all.len() > shown.len() { ", ..." } else { "" }
    )
}

fn check_unit_exists(
    p: &mut Problems,
    cat: &TargetCatalog,
    m: &Mappings,
    src: &str,
    to: &str,
    context: &str,
) {
    // a mapping to a missing unit is reported once with the mapping itself
    if cat.units.contains(to) || m.units.contains_key(src) {
        return;
    }
    p.add(
        "units",
        src,
        "unmapped_missing",
        format!(
            "unit `{src}` ({context}) does not exist in the target's units table; map it to a target unit with mappings.units[\"{src}\"]"
        ),
    );
}

/// The unit must exist in the unit tables of the target as a unit of the force's kind (or be a unit of the target start
/// position), and the start position must hold a template instance of that kind to clone it from.
fn check_unit_kind(
    p: &mut Problems,
    cat: &TargetCatalog,
    src: &str,
    to: &str,
    navy: bool,
    faction: &str,
) {
    if !cat.units.contains(to) {
        return;
    }
    let land = cat.land_units.contains(to);
    let naval = cat.naval_units.contains(to);
    let templated = cat.startpos.as_ref().is_ok_and(|s| s.units.contains(to));
    let ok = if navy {
        naval || (!land && templated)
    } else {
        land || (!naval && templated)
    };
    let (what, want) = if navy {
        ("fleet", "naval")
    } else {
        ("army", "land")
    };
    if !ok {
        let found = match (land, naval) {
            (true, _) => "a land unit",
            (_, true) => "a naval unit",
            _ => "defined in neither unit_stats_land nor unit_stats_naval of the target",
        };
        p.add(
            "units",
            src,
            "wrong_kind",
            format!(
                "unit `{to}` is used in the starting {what} of {faction} but is {found}; map `{src}` to a {want} unit of the target with mappings.units[\"{src}\"]"
            ),
        );
        return;
    }
    if !templated {
        if let Ok(sp) = &cat.startpos {
            let has_template = sp.units.iter().any(|u| {
                if navy {
                    !cat.land_units.contains(u)
                } else {
                    cat.land_units.contains(u)
                }
            });
            if !has_template {
                p.add(
                    "units",
                    src,
                    "no_template",
                    format!(
                        "unit `{to}` ({what} of {faction}) has no {want} unit instance in the target start position to be cloned from; the start position cannot host new {want} units"
                    ),
                );
            }
        }
    }
}

fn check_level(
    p: &mut Problems,
    cat: &TargetCatalog,
    m: &Mappings,
    src: &str,
    to: &str,
    user: &str,
) {
    if cat.building_levels.contains(to) || m.buildings.contains_key(src) {
        return;
    }
    p.add(
        "buildings",
        src,
        "unmapped_missing",
        format!(
            "building level `{src}` (used for `{user}`) does not exist in the target's building_levels; map it to a target level with mappings.buildings[\"{src}\"]"
        ),
    );
}

/// A building level placed in slot types its chain does not allow.
struct Misfit {
    level: String,
    chain: String,
    slots: BTreeSet<String>,
    user: String,
}

/// Records `level` (mapped from `src`) in slot type `slot` when the target's chain of the level cannot be built there.
/// Levels the target does not define are reported elsewhere.
fn note_misfit(
    misfits: &mut BTreeMap<String, Misfit>,
    cat: &TargetCatalog,
    src: &str,
    level: &str,
    slot: &str,
    user: &str,
) {
    let Some(chain) = cat.level_chain.get(level) else {
        return;
    };
    if cat.chain_slots.get(chain).is_some_and(|s| s.contains(slot)) {
        return;
    }
    misfits
        .entry(src.to_owned())
        .or_insert_with(|| Misfit {
            level: level.to_owned(),
            chain: chain.clone(),
            slots: BTreeSet::new(),
            user: user.to_owned(),
        })
        .slots
        .insert(slot.to_owned());
}

/// An explicit level for an ETW building whose slot types are not listed by the rules: its chain must at least be
/// buildable in one of the slot types that receive buildings.
fn check_any_slot(
    p: &mut Problems,
    cat: &TargetCatalog,
    src: &str,
    level: &str,
    used: &BTreeSet<&str>,
) {
    let Some(chain) = cat.level_chain.get(level) else {
        return;
    };
    let fits = cat
        .chain_slots
        .get(chain)
        .is_some_and(|s| s.iter().any(|slot| used.contains(slot.as_str())));
    if !fits {
        p.add(
            "buildings",
            src,
            "wrong_slot",
            format!(
                "building level `{level}` (chain `{chain}`) cannot be built in any slot type of the converted world; map `{src}` to a level of a chain that fits with mappings.buildings[\"{src}\"]"
            ),
        );
    }
}

fn str_values(v: &Value) -> impl Iterator<Item = &str> {
    v.as_object()
        .into_iter()
        .flat_map(|o| o.values())
        .filter_map(Value::as_str)
}

/// Pure resolution step: applies the mappings and ownership edits to `base` and checks everything against `cat`.
/// Returns the resolved rules (only meaningful when the problem list is empty) and all problems found.
pub fn resolve(
    base: &Rules,
    project: &Project,
    cat: &TargetCatalog,
) -> Result<(Rules, Vec<MappingProblem>)> {
    let m = &project.mappings;
    let mut p = Problems::default();
    let base_f = obj(&base.factions, "faction rules")?;
    let base_regions = obj(&base.regions, "region rules")?;
    let base_units = obj(&base.units, "unit rules")?;
    let base_buildings = obj(&base.buildings, "building rules")?;
    let base_by_slot = obj(&base.buildings_by_slot_type, "building slot rules")?;
    let base_armies = obj(&base.armies, "army rules")?;

    // ------------------------------------------------------------------------------------------ factions
    let mut rebel_keys: BTreeSet<String> = BTreeSet::new();
    for v in base_regions.values().chain(base_f.values()) {
        if let Some(r) = v.get("rebel_faction").and_then(Value::as_str) {
            if !base_f.contains_key(r) {
                rebel_keys.insert(r.to_owned());
            }
        }
    }
    let target_of =
        |s: &str| -> String { m.factions.get(s).cloned().unwrap_or_else(|| s.to_owned()) };
    let rebel_target = |k: &str| -> String { target_of(k) };

    for (src, tgt) in &m.factions {
        if !base_f.contains_key(src) && !rebel_keys.contains(src) {
            p.add(
                "factions",
                src,
                "invalid_mapping",
                format!(
                    "faction mapping source `{src}` is not a faction of the world rules (factions: {})",
                    sample(base_f.keys().map(String::as_str))
                ),
            );
        }
        if !cat.factions.contains(tgt) {
            p.add(
                "factions",
                src,
                "missing_target",
                format!("faction `{src}` is mapped to `{tgt}`, which the target's factions table does not define"),
            );
        }
    }
    let mut by_target: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for s in base_f.keys() {
        by_target.entry(target_of(s)).or_default().push(s);
    }
    for (t, srcs) in &by_target {
        if srcs.len() > 1 {
            let key = srcs
                .iter()
                .find(|s| m.factions.contains_key(**s))
                .copied()
                .unwrap_or(srcs[0]);
            p.add(
                "factions",
                key,
                "collision",
                format!(
                    "factions {} all resolve to `{t}`; every world faction needs its own target faction",
                    srcs.join(", ")
                ),
            );
        }
    }

    let mut factions_out = Map::new();
    for (s, f) in base_f {
        let t = target_of(s);
        // A world faction the target does not define is omitted, never created.
        if !cat.factions.contains(&t) {
            continue;
        }
        let mut e = f.clone();
        let eo = e
            .as_object_mut()
            .with_context(|| format!("faction rules {s} must be an object"))?;
        eo.insert("etw_source".into(), Value::String(s.clone()));
        eo.insert("source".into(), Value::String("ntw".into()));
        eo.insert("ntw_template".into(), Value::String(t.clone()));
        eo.remove("donor");
        if let Some(r) = eo
            .get("rebel_faction")
            .and_then(Value::as_str)
            .map(rebel_target)
        {
            eo.insert("rebel_faction".into(), Value::String(r));
        }
        if let Some(d) = m.faction_templates.get(&t) {
            eo.insert("donor".into(), Value::String(d.clone()));
        }
        factions_out.insert(t, e);
    }
    // Ownership is chosen explicitly by the user. A target DB faction need not already be
    // represented in the curated world rules to receive territory.
    for key in project.owners.values() {
        if !cat.factions.contains(key) || factions_out.contains_key(key) {
            continue;
        }
        let mut info = base_f.get(key).cloned().unwrap_or_else(|| {
            serde_json::json!({
                "alive": false, "capital": null, "playable": false, "regions": 0
            })
        });
        let entry = info
            .as_object_mut()
            .with_context(|| format!("faction rules {key} must be an object"))?;
        entry.insert("source".into(), Value::String("ntw".into()));
        entry.insert("ntw_template".into(), Value::String(key.clone()));
        entry.insert("etw_source".into(), Value::String(key.clone()));
        entry.remove("donor");
        if let Some(donor) = m.faction_templates.get(key) {
            entry.insert("donor".into(), Value::String(donor.clone()));
        }
        factions_out.insert(key.clone(), info);
    }
    for (k, d) in &m.faction_templates {
        if !factions_out.contains_key(k) {
            p.add(
                "faction_templates",
                k,
                "invalid_mapping",
                format!(
                    "`{k}` is not a resolved faction of the campaign (factions: {})",
                    sample(factions_out.keys().map(String::as_str))
                ),
            );
        }
        if !cat.factions.contains(d) {
            p.add(
                "faction_templates",
                k,
                "missing_target",
                format!("the donor `{d}` chosen for `{k}` is not a faction of the target's factions table"),
            );
        }
    }
    for (k, f) in &factions_out {
        let template = f.get("ntw_template").and_then(Value::as_str).unwrap_or("");
        if !cat.factions.contains(template) {
            p.add(
                "faction_templates",
                k,
                "missing_target",
                format!(
                    "faction `{k}` is based on the template faction `{template}`, which the target's factions table does not define; choose another with mappings.faction_templates[\"{k}\"]"
                ),
            );
        }
    }

    let mut regions_out = base_regions.clone();
    for r in regions_out.values_mut() {
        if let Some(ro) = r.as_object_mut() {
            if let Some(o) = ro.get("owner").and_then(Value::as_str).map(&target_of) {
                ro.insert("owner".into(), Value::String(o));
            }
            if let Some(rb) = ro
                .get("rebel_faction")
                .and_then(Value::as_str)
                .map(&rebel_target)
            {
                ro.insert("rebel_faction".into(), Value::String(rb));
            }
        }
    }

    // ------------------------------------------------------------------------------------------ units
    // The curated rules replace every ETW unit by an NTW stand-in; starting forces are generated from the stand-in keys.
    // So a unit mapping is applicable only for a stand-in key: it rewrites the forces and the unit rules alike. A mapping
    // for an ETW key alone would change nothing that is generated and is rejected.
    let mut unit_sources: BTreeSet<&str> = str_values(&base.units).collect();
    for list in base_armies.values() {
        for army in list.as_array().into_iter().flatten() {
            unit_sources.extend(
                army.get("units")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str),
            );
        }
    }
    for (src, tgt) in &m.units {
        if !unit_sources.contains(src.as_str()) {
            let message = match base_units.get(src).and_then(Value::as_str) {
                Some(stand_in) => format!(
                    "`{src}` is an ETW unit key: the conversion uses its NTW stand-in `{stand_in}` for starting forces, so a mapping for `{src}` would change nothing; map the stand-in with mappings.units[\"{stand_in}\"] instead"
                ),
                None => format!(
                    "unit mapping source `{src}` is not an NTW unit used by the starting armies or the unit rules; map the NTW stand-in key of the force you want to change"
                ),
            };
            p.add("units", src, "invalid_mapping", message);
        }
        if !cat.units.contains(tgt) {
            p.add(
                "units",
                src,
                "missing_target",
                format!("unit `{src}` is mapped to `{tgt}`, which the target's units table does not define"),
            );
        }
    }
    let mut units_out = Map::new();
    for (etw, v) in base_units {
        let nv = match v {
            Value::String(ntw) => {
                let to = m.units.get(ntw).cloned().unwrap_or_else(|| ntw.clone());
                check_unit_exists(
                    &mut p,
                    cat,
                    m,
                    ntw,
                    &to,
                    &format!("stand-in of ETW unit {etw}"),
                );
                Value::String(to)
            }
            Value::Null => Value::Null,
            other => {
                p.add(
                    "rules",
                    etw,
                    "invalid_rules",
                    format!("unit rules {etw}: expected a unit key or null, found {other}"),
                );
                other.clone()
            }
        };
        units_out.insert(etw.clone(), nv);
    }
    let mut armies_out = Map::new();
    for (s, list) in base_armies {
        let key = if base_f.contains_key(s) {
            target_of(s)
        } else {
            s.clone()
        };
        if !factions_out.contains_key(&key) {
            // A territorial reassignment does not transfer the omitted faction's starting forces.
            continue;
        }
        let mut new_list = list.clone();
        for army in new_list.as_array_mut().into_iter().flatten() {
            let navy = army.get("kind").and_then(Value::as_str) == Some("navy");
            for u in army
                .get_mut("units")
                .and_then(Value::as_array_mut)
                .into_iter()
                .flatten()
            {
                let Some(src) = u.as_str().map(str::to_owned) else {
                    continue;
                };
                let to = m.units.get(&src).cloned().unwrap_or_else(|| src.clone());
                check_unit_exists(
                    &mut p,
                    cat,
                    m,
                    &src,
                    &to,
                    &format!("starting {} of {key}", if navy { "fleet" } else { "army" }),
                );
                check_unit_kind(&mut p, cat, &src, &to, navy, &key);
                *u = Value::String(to);
            }
        }
        armies_out.insert(key, new_list);
    }

    // ------------------------------------------------------------------------------------------ buildings
    let mut building_sources: BTreeSet<&str> = base_buildings.keys().map(String::as_str).collect();
    building_sources.extend(base_by_slot.keys().map(String::as_str));
    building_sources.extend(str_values(&base.buildings));
    for slots in base_by_slot.values() {
        building_sources.extend(str_values(slots));
    }
    for (src, tgt) in &m.buildings {
        if !building_sources.contains(src.as_str()) {
            p.add(
                "buildings",
                src,
                "invalid_mapping",
                format!("building mapping source `{src}` is neither an ETW building nor a vanilla building level of the building rules"),
            );
        }
        if !cat.building_levels.contains(tgt) {
            p.add(
                "buildings",
                src,
                "missing_target",
                format!("building `{src}` is mapped to `{tgt}`, which the target's building_levels table does not define"),
            );
        }
    }
    let mut buildings_out = Map::new();
    for (etw, v) in base_buildings {
        let nv = match v {
            Value::String(level) => {
                let (to, src) = if let Some(t) = m.buildings.get(etw) {
                    (t.clone(), etw.as_str())
                } else if let Some(t) = m.buildings.get(level) {
                    (t.clone(), level.as_str())
                } else {
                    (level.clone(), level.as_str())
                };
                check_level(&mut p, cat, m, src, &to, etw);
                Value::String(to)
            }
            Value::Null => match m.buildings.get(etw) {
                Some(t) => Value::String(t.clone()),
                None => Value::Null,
            },
            other => {
                p.add(
                    "rules",
                    etw,
                    "invalid_rules",
                    format!(
                        "building rules {etw}: expected a building level or null, found {other}"
                    ),
                );
                other.clone()
            }
        };
        buildings_out.insert(etw.clone(), nv);
    }
    let mut by_slot_out = Map::new();
    for (chain, slots) in base_by_slot {
        if m.buildings.contains_key(chain) {
            // an explicit choice for the ETW building replaces its slot-type specific levels
            by_slot_out.insert(chain.clone(), Value::Object(Map::new()));
            continue;
        }
        let mut out = slots.clone();
        for (slot, lvl) in out.as_object_mut().into_iter().flatten() {
            if let Some(level) = lvl.as_str().map(str::to_owned) {
                let to = m
                    .buildings
                    .get(&level)
                    .cloned()
                    .unwrap_or_else(|| level.clone());
                check_level(&mut p, cat, m, &level, &to, &format!("{chain} on {slot}"));
                *lvl = Value::String(to);
            }
        }
        by_slot_out.insert(chain.clone(), out);
    }

    // A level can only be built in slot types its chain is allowed in (`building_chain_to_slots`). The slot types of an ETW
    // building are the ones the curated per-slot-type rules list for it (the converter looks levels up by slot type);
    // fortification slots never receive buildings.
    {
        let used_slots: BTreeSet<&str> = str_values(&base.slot_types)
            .filter(|s| *s != "settlement_fortification")
            .collect();
        let mut misfits: BTreeMap<String, Misfit> = BTreeMap::new();
        let etw_keys: BTreeSet<&String> =
            base_buildings.keys().chain(base_by_slot.keys()).collect();
        for etw in etw_keys {
            let original = base_by_slot.get(etw).and_then(Value::as_object);
            let slots: Vec<(&str, &str)> = original
                .into_iter()
                .flatten()
                .filter_map(|(s, l)| Some((s.as_str(), l.as_str()?)))
                .filter(|(s, _)| used_slots.contains(s))
                .collect();
            if let Some(target) = m.buildings.get(etw) {
                // explicit choice: one level for every slot type of the ETW building
                if slots.is_empty() {
                    check_any_slot(&mut p, cat, etw, target, &used_slots);
                }
                for (slot, _) in &slots {
                    note_misfit(&mut misfits, cat, etw, target, slot, etw);
                }
            } else {
                for (slot, level) in &slots {
                    let target = m.buildings.get(*level).map_or(*level, String::as_str);
                    note_misfit(&mut misfits, cat, level, target, slot, etw);
                }
            }
        }
        for (src, f) in misfits {
            let allowed: Vec<&str> = cat
                .chain_slots
                .get(&f.chain)
                .into_iter()
                .flatten()
                .map(String::as_str)
                .collect();
            let slots: Vec<&str> = f.slots.iter().map(String::as_str).collect();
            p.add(
                "buildings",
                &src,
                "wrong_slot",
                format!(
                    "building level `{}` (chain `{}`, used for `{}`) cannot be built in {} slots: the target allows that chain only in {}; map `{src}` to a level whose chain fits with mappings.buildings[\"{src}\"]",
                    f.level,
                    f.chain,
                    f.user,
                    slots.join(", "),
                    if allowed.is_empty() { "no slot type".to_owned() } else { allowed.join(", ") }
                ),
            );
        }
    }

    let mut out = Rules {
        root: base.root.clone(),
        regions: Value::Object(regions_out),
        factions: Value::Object(factions_out),
        units: Value::Object(units_out),
        buildings: Value::Object(buildings_out),
        buildings_by_slot_type: Value::Object(by_slot_out),
        slot_types: base.slot_types.clone(),
        armies: Value::Object(armies_out),
    };

    // ------------------------------------------------------------------------------------------ ownership
    {
        let mut owners = BTreeMap::new();
        let added_keys: BTreeSet<&str> = project
            .activations
            .iter()
            .map(|a| a.key.as_str())
            .chain(project.splits.iter().map(|s| s.key.as_str()))
            .collect();
        for (region, value) in &project.owners {
            if !base_regions.contains_key(region) && !added_keys.contains(region.as_str()) {
                p.add(
                    "owners",
                    region,
                    "invalid_mapping",
                    format!("project assigns unknown region `{region}`"),
                );
                continue;
            }
            // Owners are exact keys of the target's factions table: a mapped source key is not translated.
            if !cat.factions.contains(value) {
                p.add(
                    "owners",
                    region,
                    "existing_owner_required",
                    format!(
                        "Region `{region}` is assigned to `{value}`, which is not an existing target faction. Choose an existing owner in the Ownership map."
                    ),
                );
                continue;
            }
            owners.insert(region.clone(), value.clone());
        }
        out.apply_project(&Project {
            owners,
            activations: project.activations.clone(),
            splits: project.splits.clone(),
            ..Project::default()
        })?;
    }
    for (region, info) in out
        .regions
        .as_object_mut()
        .context("resolved region rules must be an object")?
    {
        let owner = info.get("owner").and_then(Value::as_str).unwrap_or("");
        if !cat.factions.contains(owner) {
            p.add(
                "owners",
                region,
                "existing_owner_required",
                format!(
                    "Region `{region}` still belongs to `{owner}`, which is absent from the target database. Choose an existing owner in the Ownership map; factions are never created or automatically replaced."
                ),
            );
            continue;
        }
        let rebel = info
            .get("rebel_faction")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !cat.factions.contains(rebel) {
            if let Some(existing) = cat
                .rebel_factions
                .get(owner)
                .filter(|k| cat.factions.contains(*k))
            {
                // Territory still goes only to the user's chosen owner. Revolts follow that
                // existing faction's target DB configuration instead of creating an ETW key.
                info["rebel_faction"] = Value::String(existing.clone());
            } else {
                p.add(
                    "factions",
                    rebel,
                    "existing_rebel_required",
                    format!(
                        "Region `{region}` uses absent rebellion faction `{rebel}` and owner `{owner}` has no valid target rebellion identity. Map `{rebel}` to an existing faction or correct the target's settler_rebellion_faction."
                    ),
                );
            }
        }
    }

    // ------------------------------------------------------------------------------------------ structure
    match &cat.startpos {
        Ok(sp) => {
            for (k, info) in faction_infos(&out)? {
                if let Err(e) = structure_for(&info, &k, sp, &out) {
                    p.add("faction_templates", &k, "no_donor", format!("{e:#}"));
                    continue;
                }
                // A database faction without a record in the start position is built from a donor's record structure
                // and must carry its own row's name, flags and colours; a row the record cannot represent blocks.
                if !sp.alive.contains_key(&k) {
                    match cat.identity.get(&k) {
                        Some(Ok(())) => {}
                        Some(Err(why)) => p.add(
                            "factions",
                            &info.etw_source,
                            "unrepresentable_identity",
                            format!("faction `{k}` has no record in the target start position, and its database identity cannot be written into a donor record: {why}"),
                        ),
                        None => {}
                    }
                }
            }
        }
        Err(why) => p.add(
            "startpos",
            "campaigns/eur_napoleon/startpos.esf",
            "unsupported_startpos",
            format!(
                "the target's campaigns/eur_napoleon/startpos.esf cannot serve as structural donor: {why}"
            ),
        ),
    }
    Ok((out, p.list))
}

fn describe(problems: &[MappingProblem]) -> String {
    let mut lines: Vec<String> = problems
        .iter()
        .take(25)
        .map(|p| format!("  - [{} {}] {}", p.field, p.key, p.message))
        .collect();
    if problems.len() > 25 {
        lines.push(format!("  ... and {} more", problems.len() - 25));
    }
    lines.join("\n")
}

/// Problems of mapping `base` onto the target; empty when [`prepare_rules`] succeeds.
pub fn validate(
    snapshot: &Snapshot,
    base: &Rules,
    project: &Project,
) -> Result<Vec<MappingProblem>> {
    let cat = TargetCatalog::load(snapshot, base)?;
    Ok(resolve(base, project, &cat)?.1)
}

/// Resolved rules for the project: mappings and donors applied, ownership edits applied (`project.owners` holds resolved
/// faction keys; a mapped source key is translated), every reference checked against the effective target `snapshot`.
/// Fails with all unresolved references and the mapping that fixes each.
pub fn prepare_rules(snapshot: &Snapshot, base: &Rules, project: &Project) -> Result<Rules> {
    let cat = TargetCatalog::load(snapshot, base)?;
    let (rules, problems) = resolve(base, project, &cat)?;
    if !problems.is_empty() {
        bail!(
            "The target cannot host the Grand Campaign yet ({} problem{}):\n{}",
            problems.len(),
            if problems.len() == 1 { "" } else { "s" },
            describe(&problems)
        );
    }
    Ok(rules)
}

/// Faction plans of resolved rules (see the module documentation for the fields).
pub fn faction_plan(rules: &Rules) -> Vec<FactionPlan> {
    rules
        .factions
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, v)| FactionPlan {
            key: k.clone(),
            etw_source: v
                .get("etw_source")
                .and_then(Value::as_str)
                .unwrap_or(k)
                .to_owned(),
            donor: v.get("donor").and_then(Value::as_str).map(str::to_owned),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn rules() -> Rules {
        Rules {
            root: PathBuf::new(),
            regions: json!({
                "france":  {"owner": "france",  "rebel_faction": "french_rebels"},
                "corsica": {"owner": "france",  "rebel_faction": "french_rebels"},
                "england": {"owner": "britain", "rebel_faction": "british_rebels"},
                "delhi":   {"owner": "hessen",  "rebel_faction": "british_rebels"}
            }),
            factions: json!({
                "france":  {"alive": true,  "capital": "france",  "ntw_template": "france",  "playable": true,  "regions": 2, "source": "ntw"},
                "britain": {"alive": true,  "capital": "england", "ntw_template": "britain", "playable": true,  "regions": 1, "source": "ntw"},
                "belgium": {"alive": false, "capital": null,      "ntw_template": "belgium", "playable": false, "regions": 0, "source": "ntw"},
                "mughal":  {"alive": false, "capital": null,      "ntw_template": "ottomans", "playable": false, "regions": 0, "source": "etw"},
                "hessen":  {"alive": true,  "capital": "delhi",   "ntw_template": "hessen",  "playable": false, "regions": 1, "source": "ntw"}
            }),
            units: json!({
                "etw_line": "Inf_Line_French",
                "etw_ship": "Frigate_French",
                "etw_dropped": null
            }),
            buildings: json!({
                "etw_barracks": "sArmy2_barracks",
                "etw_market": "tCommerce1_market",
                "etw_unmapped": null
            }),
            buildings_by_slot_type: json!({
                "etw_barracks": {"settlement": "sArmy2_barracks", "settlement_road": "sArmy3_depot"},
                "etw_market": {"town-commercial": "tCommerce1_market"}
            }),
            slot_types: json!({"slot_a": "settlement", "slot_b": "settlement_road", "slot_c": "town-commercial"}),
            armies: json!({
                "france": [
                    {"general": true, "kind": "army", "region": "france", "units": ["Gen_Generals_Staff", "Inf_Line_French"]},
                    {"general": false, "kind": "navy", "region": "france", "units": ["Frigate_French"]}
                ],
                "britain": [
                    {"general": true, "kind": "army", "region": "england", "units": ["Inf_Line_British"]}
                ]
            }),
        }
    }

    /// The curated world as the legacy rules describe it: `mughal` is an ETW-only faction (`source: etw`) the target
    /// does not define, and it owns `delhi`.
    fn omitted_owner_rules() -> Rules {
        let mut r = rules();
        r.regions["delhi"] = json!({"owner": "mughal", "rebel_faction": "mughal"});
        r.factions["mughal"] = json!({"alive": true, "capital": "delhi", "ntw_template": "ottomans", "playable": false, "regions": 1, "source": "etw"});
        r.factions.as_object_mut().unwrap().remove("hessen");
        r
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn startpos() -> StartposStructure {
        let order: Vec<String> = [
            "france", "britain", "belgium", "hungary", "ottomans", "hessen",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let alive = order
            .iter()
            .map(|k| (k.clone(), !matches!(k.as_str(), "belgium" | "hungary")))
            .collect();
        StartposStructure {
            order,
            alive,
            units: set(&["Gen_Generals_Staff", "Inf_Line_French", "Frigate_French"]),
        }
    }

    fn catalog() -> TargetCatalog {
        TargetCatalog {
            factions: set(&[
                "france",
                "britain",
                "belgium",
                "hungary",
                "ottomans",
                "hessen",
                "french_rebels",
                "british_rebels",
                "mod_france",
                "mod_empire",
            ]),
            rebel_factions: [
                ("france", "french_rebels"),
                ("britain", "british_rebels"),
                ("belgium", "french_rebels"),
                ("hungary", "british_rebels"),
                ("ottomans", "british_rebels"),
                ("hessen", "british_rebels"),
                ("mod_france", "french_rebels"),
                ("mod_empire", "british_rebels"),
            ]
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect(),
            units: set(&[
                "Gen_Generals_Staff",
                "Inf_Line_French",
                "Frigate_French",
                "Inf_Line_British",
                "Mod_Line_Inf",
                "Mod_Galleon",
            ]),
            land_units: set(&[
                "Gen_Generals_Staff",
                "Inf_Line_French",
                "Inf_Line_British",
                "Mod_Line_Inf",
            ]),
            naval_units: set(&["Frigate_French", "Mod_Galleon"]),
            building_levels: set(&[
                "sArmy2_barracks",
                "sArmy3_depot",
                "tCommerce1_market",
                "mod_barracks",
                "mod_market",
            ]),
            level_chain: [
                ("sArmy2_barracks", "sArmy"),
                ("sArmy3_depot", "sArmy"),
                ("tCommerce1_market", "tCommerce"),
                ("mod_barracks", "mArmy"),
                ("mod_market", "tCommerce"),
            ]
            .iter()
            .map(|(l, c)| (l.to_string(), c.to_string()))
            .collect(),
            chain_slots: [
                ("sArmy", vec!["settlement", "settlement_road"]),
                ("mArmy", vec!["settlement", "settlement_road"]),
                ("tCommerce", vec!["town-commercial"]),
            ]
            .iter()
            .map(|(c, s)| (c.to_string(), s.iter().map(|x| x.to_string()).collect()))
            .collect(),
            identity: [
                "france",
                "britain",
                "belgium",
                "hungary",
                "ottomans",
                "hessen",
                "french_rebels",
                "british_rebels",
                "mod_france",
                "mod_empire",
            ]
            .iter()
            .map(|k| (k.to_string(), Ok(())))
            .collect(),
            startpos: Ok(Arc::new(startpos())),
        }
    }

    fn problems(project: &Project, cat: &TargetCatalog) -> Vec<MappingProblem> {
        resolve(&rules(), project, cat).unwrap().1
    }

    fn kinds(p: &[MappingProblem]) -> Vec<(String, String, String)> {
        p.iter()
            .map(|p| (p.field.clone(), p.key.clone(), p.kind.clone()))
            .collect()
    }

    #[test]
    fn target_defined_world_resolves_unchanged_and_omits_what_the_target_lacks() {
        let base = rules();
        let (out, probs) = resolve(&base, &Project::default(), &catalog()).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.regions, base.regions);
        assert_eq!(out.units, base.units);
        assert_eq!(out.buildings, base.buildings);
        assert_eq!(out.buildings_by_slot_type, base.buildings_by_slot_type);
        assert_eq!(out.armies, base.armies);
        // the ETW-only faction the target does not define is never created
        let factions = out.factions.as_object().unwrap();
        assert_eq!(
            factions.keys().cloned().collect::<BTreeSet<_>>(),
            set(&["belgium", "britain", "france", "hessen"])
        );
        for (k, f) in factions {
            let mut resolved = f.clone();
            assert_eq!(resolved["etw_source"], k.as_str());
            resolved.as_object_mut().unwrap().remove("etw_source");
            assert_eq!(resolved, base.factions[k], "faction {k}");
        }
    }

    #[test]
    fn faction_mapping_renames_ownership_armies_rebels_and_edits_follow() {
        let mut project = Project::default();
        project
            .mappings
            .faction_templates
            .insert("mod_france".into(), "hessen".into());
        project
            .mappings
            .factions
            .insert("france".into(), "mod_france".into());
        // owners are exact target keys
        project.owners.insert("corsica".into(), "britain".into());
        project.owners.insert("england".into(), "mod_france".into());
        let (out, probs) = resolve(&rules(), &project, &catalog()).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        let f = out.factions.as_object().unwrap();
        assert!(!f.contains_key("france") && f.contains_key("mod_france"));
        assert_eq!(out.factions["mod_france"]["etw_source"], "france");
        assert_eq!(out.factions["mod_france"]["source"], "ntw");
        assert_eq!(out.factions["mod_france"]["ntw_template"], "mod_france");
        assert_eq!(out.factions["mod_france"]["donor"], "hessen");
        // regions: the mapped owner replaced everywhere, then the user edits applied
        assert_eq!(out.regions["france"]["owner"], "mod_france");
        assert_eq!(out.regions["england"]["owner"], "mod_france");
        assert_eq!(out.regions["corsica"]["owner"], "britain");
        assert_eq!(out.regions["france"]["rebel_faction"], "french_rebels");
        assert_eq!(out.factions["mod_france"]["regions"], 2);
        assert_eq!(out.factions["britain"]["regions"], 1);
        // britain lost its capital england to mod_france: the remaining region becomes the capital
        assert_eq!(out.factions["britain"]["capital"], "corsica");
        // armies follow the faction key
        let armies = out.armies.as_object().unwrap();
        assert!(!armies.contains_key("france") && armies.contains_key("mod_france"));
        let plan = faction_plan(&out);
        let mf = plan.iter().find(|p| p.key == "mod_france").unwrap();
        assert_eq!(
            (mf.etw_source.as_str(), mf.donor.as_deref()),
            ("france", Some("hessen"))
        );
    }

    #[test]
    fn faction_without_a_record_in_the_target_startpos_needs_an_explicit_donor() {
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("france".into(), "mod_france".into());
        let probs = problems(&project, &catalog());
        assert_eq!(
            kinds(&probs),
            vec![(
                "faction_templates".into(),
                "mod_france".into(),
                "no_donor".into()
            )]
        );
        assert!(
            probs[0]
                .message
                .contains("mappings.faction_templates[\"mod_france\"]")
        );
        // a dead donor is refused too
        project
            .mappings
            .faction_templates
            .insert("mod_france".into(), "belgium".into());
        let probs = problems(&project, &catalog());
        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0].kind, "no_donor");
        assert!(probs[0].message.contains("dead record"));
    }

    #[test]
    fn faction_mapping_collisions_and_missing_targets_are_reported() {
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("britain".into(), "france".into());
        let probs = problems(&project, &catalog());
        assert!(
            probs
                .iter()
                .any(|p| p.kind == "collision" && p.message.contains("france"))
        );
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("britain".into(), "nowhere".into());
        let probs = problems(&project, &catalog());
        assert!(
            probs
                .iter()
                .any(|p| p.kind == "missing_target" && p.key == "britain")
        );
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("atlantis".into(), "mod_empire".into());
        assert!(
            problems(&project, &catalog())
                .iter()
                .any(|p| p.kind == "invalid_mapping")
        );
    }

    #[test]
    fn a_faction_missing_from_the_target_is_omitted_and_its_territory_needs_a_manual_owner() {
        // landless: omitted without a problem
        let mut cat = catalog();
        cat.factions.remove("belgium");
        let (out, probs) = resolve(&rules(), &Project::default(), &cat).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert!(out.factions.get("belgium").is_none());

        // with territory: the owner must be chosen, forces are not transferred
        let mut cat = catalog();
        cat.factions.remove("britain");
        let (out, probs) = resolve(&rules(), &Project::default(), &cat).unwrap();
        assert_eq!(
            kinds(&probs),
            vec![(
                "owners".into(),
                "england".into(),
                "existing_owner_required".into()
            )]
        );
        assert!(out.armies.get("britain").is_none());
    }

    #[test]
    fn a_rebel_faction_missing_from_the_target_must_be_mapped() {
        let mut cat = catalog();
        cat.factions.remove("british_rebels");
        let probs = problems(&Project::default(), &cat);
        assert!(
            probs
                .iter()
                .any(|p| p.key == "british_rebels" && p.kind == "existing_rebel_required")
        );
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("british_rebels".into(), "french_rebels".into());
        let (out, probs) = resolve(&rules(), &project, &cat).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.regions["england"]["rebel_faction"], "french_rebels");
    }

    #[test]
    fn unit_missing_in_the_target_blocks_until_mapped_to_a_unit_of_the_same_kind() {
        let mut cat = catalog();
        cat.units.remove("Inf_Line_British");
        cat.land_units.remove("Inf_Line_British");
        let probs = problems(&Project::default(), &cat);
        assert_eq!(
            kinds(&probs),
            vec![(
                "units".into(),
                "Inf_Line_British".into(),
                "unmapped_missing".into()
            )]
        );

        let mut project = Project::default();
        project
            .mappings
            .units
            .insert("Inf_Line_British".into(), "Mod_Galleon".into());
        let probs = problems(&project, &cat);
        assert_eq!(
            kinds(&probs),
            vec![(
                "units".into(),
                "Inf_Line_British".into(),
                "wrong_kind".into()
            )]
        );

        project
            .mappings
            .units
            .insert("Inf_Line_British".into(), "Mod_Line_Inf".into());
        let (out, probs) = resolve(&rules(), &project, &cat).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.armies["britain"][0]["units"], json!(["Mod_Line_Inf"]));
        // untouched armies keep their units
        assert_eq!(
            out.armies["france"][0]["units"],
            json!(["Gen_Generals_Staff", "Inf_Line_French"])
        );
    }

    #[test]
    fn unit_mapping_rewrites_stand_in_keys_and_rejects_etw_only_keys() {
        // the stand-in key of a starting force: rewrites forces and the unit rules
        let mut project = Project::default();
        project
            .mappings
            .units
            .insert("Frigate_French".into(), "Mod_Galleon".into());
        project
            .mappings
            .units
            .insert("Inf_Line_French".into(), "Mod_Line_Inf".into());
        let (out, probs) = resolve(&rules(), &project, &catalog()).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.units["etw_line"], "Mod_Line_Inf");
        assert_eq!(out.units["etw_ship"], "Mod_Galleon");
        assert_eq!(out.units["etw_dropped"], Value::Null);
        assert_eq!(
            out.armies["france"][0]["units"],
            json!(["Gen_Generals_Staff", "Mod_Line_Inf"])
        );
        assert_eq!(out.armies["france"][1]["units"], json!(["Mod_Galleon"]));

        // an ETW key would only change the unused unit rules: rejected with the stand-in to map instead
        let mut etw = Project::default();
        etw.mappings
            .units
            .insert("etw_line".into(), "Mod_Line_Inf".into());
        let probs = problems(&etw, &catalog());
        assert_eq!(
            kinds(&probs),
            vec![("units".into(), "etw_line".into(), "invalid_mapping".into())]
        );
        assert!(
            probs[0]
                .message
                .contains("mappings.units[\"Inf_Line_French\"]"),
            "{}",
            probs[0].message
        );

        let mut bad = Project::default();
        bad.mappings
            .units
            .insert("not_a_unit".into(), "Mod_Line_Inf".into());
        bad.mappings
            .units
            .insert("Inf_Line_French".into(), "Missing".into());
        let kinds_found: BTreeSet<String> = problems(&bad, &catalog())
            .into_iter()
            .map(|p| p.kind)
            .collect();
        assert_eq!(kinds_found, set(&["invalid_mapping", "missing_target"]));
    }

    #[test]
    fn navy_cannot_use_land_units_and_template_less_targets_block() {
        let mut cat = catalog();
        cat.startpos = Ok(Arc::new(StartposStructure {
            units: set(&["Inf_Line_French"]),
            ..startpos()
        }));
        // naval units exist in the tables but the start position can only clone land units
        let probs = problems(&Project::default(), &cat);
        assert!(
            probs
                .iter()
                .any(|p| p.kind == "no_template" && p.key == "Frigate_French"),
            "{probs:?}"
        );
        let mut project = Project::default();
        project
            .mappings
            .units
            .insert("Frigate_French".into(), "Inf_Line_French".into());
        let probs = problems(&project, &catalog());
        assert!(
            probs
                .iter()
                .any(|p| p.kind == "wrong_kind" && p.key == "Frigate_French")
        );
    }

    #[test]
    fn building_mappings_replace_levels_and_slot_overrides() {
        let mut project = Project::default();
        project
            .mappings
            .buildings
            .insert("etw_barracks".into(), "mod_barracks".into());
        project
            .mappings
            .buildings
            .insert("tCommerce1_market".into(), "mod_market".into());
        project
            .mappings
            .buildings
            .insert("etw_unmapped".into(), "sArmy3_depot".into());
        let (out, probs) = resolve(&rules(), &project, &catalog()).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.buildings["etw_barracks"], "mod_barracks");
        assert_eq!(out.buildings_by_slot_type["etw_barracks"], json!({}));
        assert_eq!(out.buildings["etw_market"], "mod_market");
        assert_eq!(
            out.buildings_by_slot_type["etw_market"]["town-commercial"],
            "mod_market"
        );
        assert_eq!(out.buildings["etw_unmapped"], "sArmy3_depot");
    }

    #[test]
    fn building_level_whose_chain_cannot_be_built_in_the_slot_type_is_refused() {
        // by level: the market's chain is a town chain, mArmy only fits settlements
        let mut project = Project::default();
        project
            .mappings
            .buildings
            .insert("tCommerce1_market".into(), "mod_barracks".into());
        let probs = problems(&project, &catalog());
        assert_eq!(
            kinds(&probs),
            vec![(
                "buildings".into(),
                "tCommerce1_market".into(),
                "wrong_slot".into()
            )]
        );
        assert!(
            probs[0].message.contains("town-commercial") && probs[0].message.contains("mArmy"),
            "{}",
            probs[0].message
        );

        // by ETW building: the explicit level applies to every slot type of the building
        let mut project = Project::default();
        project
            .mappings
            .buildings
            .insert("etw_market".into(), "mod_barracks".into());
        let probs = problems(&project, &catalog());
        assert_eq!(
            kinds(&probs),
            vec![("buildings".into(), "etw_market".into(), "wrong_slot".into())]
        );

        // a building without listed slot types still needs a chain that fits somewhere
        let mut cat = catalog();
        cat.chain_slots.insert("mArmy".into(), set(&["port"]));
        let mut project = Project::default();
        project
            .mappings
            .buildings
            .insert("etw_unmapped".into(), "mod_barracks".into());
        let probs = problems(&project, &cat);
        assert!(
            probs
                .iter()
                .any(|p| p.key == "etw_unmapped" && p.kind == "wrong_slot"),
            "{probs:?}"
        );

        // the target moving a chain out of a slot type breaks the unmapped vanilla level too
        let mut cat = catalog();
        cat.chain_slots
            .insert("tCommerce".into(), set(&["town-industrial"]));
        let probs = problems(&Project::default(), &cat);
        assert_eq!(
            kinds(&probs),
            vec![(
                "buildings".into(),
                "tCommerce1_market".into(),
                "wrong_slot".into()
            )]
        );
        let mut project = Project::default();
        project
            .mappings
            .buildings
            .insert("tCommerce1_market".into(), "mod_market".into());
        let (_, probs) = resolve(&rules(), &project, &cat).unwrap();
        assert_eq!(
            kinds(&probs),
            vec![(
                "buildings".into(),
                "tCommerce1_market".into(),
                "wrong_slot".into()
            )]
        );
    }

    #[test]
    fn faction_without_a_start_position_record_needs_an_identity_the_record_can_hold() {
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("france".into(), "mod_france".into());
        project
            .mappings
            .faction_templates
            .insert("mod_france".into(), "hessen".into());
        let mut cat = catalog();
        cat.identity.insert(
            "mod_france".into(),
            Err("target database column primary_colour_r = 12.5 cannot be stored".into()),
        );
        let probs = problems(&project, &cat);
        assert_eq!(
            kinds(&probs),
            vec![(
                "factions".into(),
                "france".into(),
                "unrepresentable_identity".into()
            )]
        );
        assert!(probs[0].message.contains("primary_colour_r"));
        // a faction with its own record keeps the record's identity: nothing to represent
        cat.identity.insert("france".into(), Err("unused".into()));
        assert!(problems(&Project::default(), &cat).is_empty());
    }

    #[test]
    fn building_level_missing_in_the_target_is_reported_by_its_vanilla_key() {
        let mut cat = catalog();
        cat.building_levels.remove("sArmy3_depot");
        let probs = problems(&Project::default(), &cat);
        assert_eq!(
            kinds(&probs),
            vec![(
                "buildings".into(),
                "sArmy3_depot".into(),
                "unmapped_missing".into()
            )]
        );
        let mut project = Project::default();
        project
            .mappings
            .buildings
            .insert("sArmy3_depot".into(), "mod_barracks".into());
        let (out, probs) = resolve(&rules(), &project, &cat).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(
            out.buildings_by_slot_type["etw_barracks"]["settlement_road"],
            "mod_barracks"
        );
    }

    #[test]
    fn owners_must_name_existing_target_factions_and_known_regions() {
        let mut project = Project::default();
        project.owners.insert("corsica".into(), "nowhere".into());
        project.owners.insert("atlantis".into(), "britain".into());
        let probs = problems(&project, &catalog());
        assert_eq!(probs.iter().filter(|p| p.field == "owners").count(), 2);

        // owners are exact target keys: a mapped faction is chosen by the key the target gives it
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("britain".into(), "mod_empire".into());
        project
            .mappings
            .faction_templates
            .insert("mod_empire".into(), "hessen".into());
        project.owners.insert("corsica".into(), "mod_empire".into());
        let (out, probs) = resolve(&rules(), &project, &catalog()).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.regions["corsica"]["owner"], "mod_empire");
        assert_eq!(out.factions["france"]["regions"], 1);
        assert_eq!(out.factions["mod_empire"]["regions"], 2);
    }

    #[test]
    fn unusable_target_startpos_blocks_actionably() {
        let mut cat = catalog();
        cat.startpos = Err("no CAMPAIGN_ENV".into());
        let probs = problems(&Project::default(), &cat);
        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0].kind, "unsupported_startpos");
        assert!(probs[0].message.contains("no CAMPAIGN_ENV"));
    }

    #[test]
    fn territory_of_a_faction_the_target_lacks_requires_a_manual_owner_instead_of_assigning_the_template()
     {
        let (out, probs) =
            resolve(&omitted_owner_rules(), &Project::default(), &catalog()).unwrap();
        assert_eq!(
            kinds(&probs),
            vec![(
                "owners".into(),
                "delhi".into(),
                "existing_owner_required".into()
            )]
        );
        assert_eq!(out.regions["delhi"]["owner"], "mughal");
        assert!(out.factions.get("mughal").is_none());
        assert!(out.factions.get("ottomans").is_none());
    }

    #[test]
    fn reassigned_territory_does_not_transfer_the_omitted_factions_forces() {
        let mut base = omitted_owner_rules();
        base.armies["mughal"] = json!([{
            "general": true, "kind": "army", "region": "delhi", "units": ["Inf_Line_French"]
        }]);
        let mut project = Project::default();
        project.owners.insert("delhi".into(), "britain".into());
        let cat = catalog();
        let (out, probs) = resolve(&base, &project, &cat).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.regions["delhi"]["owner"], "britain");
        assert_eq!(out.regions["delhi"]["rebel_faction"], "british_rebels");
        assert_eq!(out.regions["france"], base.regions["france"]);
        assert_eq!(out.factions["britain"]["regions"], 2);
        assert_eq!(out.factions["britain"]["capital"], "england");
        assert!(
            out.factions
                .as_object()
                .unwrap()
                .keys()
                .all(|k| cat.factions.contains(k))
        );
        assert!(
            out.factions
                .as_object()
                .unwrap()
                .values()
                .all(|v| v["source"] == "ntw")
        );
        assert!(out.armies.get("mughal").is_none());
        assert_eq!(out.armies["britain"], base.armies["britain"]);
    }

    #[test]
    fn a_target_only_owner_keeps_its_identity_but_still_requires_a_structural_donor() {
        let mut project = Project::default();
        project.owners.insert("delhi".into(), "mod_empire".into());
        let (_, probs) = resolve(&omitted_owner_rules(), &project, &catalog()).unwrap();
        assert!(
            probs
                .iter()
                .any(|p| p.key == "mod_empire" && p.field == "faction_templates")
        );
        project
            .mappings
            .faction_templates
            .insert("mod_empire".into(), "hessen".into());
        let (out, probs) = resolve(&omitted_owner_rules(), &project, &catalog()).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.regions["delhi"]["owner"], "mod_empire");
        assert_eq!(out.factions["mod_empire"]["source"], "ntw");
        assert_eq!(out.factions["mod_empire"]["ntw_template"], "mod_empire");
        assert_eq!(out.factions["mod_empire"]["donor"], "hessen");
        assert_eq!(out.factions["mod_empire"]["capital"], "delhi");
        assert_eq!(out.factions["mod_empire"]["alive"], true);
        assert!(out.armies.get("mod_empire").is_none());
    }

    #[test]
    fn a_legacy_etw_faction_the_target_defines_is_reused_as_a_target_faction() {
        let mut cat = catalog();
        cat.factions.insert("mughal".into());
        cat.identity.insert("mughal".into(), Ok(()));
        let mut project = Project::default();
        project
            .mappings
            .faction_templates
            .insert("mughal".into(), "hessen".into());
        let base = omitted_owner_rules();
        assert_eq!(base.factions["mughal"]["source"], "etw");
        let (out, probs) = resolve(&base, &project, &cat).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.regions["delhi"]["owner"], "mughal");
        assert_eq!(out.factions["mughal"]["source"], "ntw");
        assert_eq!(out.factions["mughal"]["ntw_template"], "mughal");
        assert_eq!(out.factions["mughal"]["donor"], "hessen");
    }

    #[test]
    fn owner_choices_are_exact_target_keys_not_legacy_source_aliases() {
        let mut project = Project::default();
        project
            .mappings
            .factions
            .insert("france".into(), "mod_france".into());
        project
            .mappings
            .faction_templates
            .insert("mod_france".into(), "hessen".into());
        project.owners.insert("france".into(), "france".into());
        project.owners.insert("delhi".into(), "britain".into());
        let (out, probs) = resolve(&omitted_owner_rules(), &project, &catalog()).unwrap();
        assert!(probs.is_empty(), "{probs:?}");
        assert_eq!(out.regions["france"]["owner"], "france");
        assert_eq!(out.regions["corsica"]["owner"], "mod_france");
        assert_eq!(out.factions["france"]["regions"], 1);
        assert_eq!(out.factions["mod_france"]["regions"], 1);
        project
            .owners
            .insert("delhi".into(), "invented_faction".into());
        let (out, probs) = resolve(&omitted_owner_rules(), &project, &catalog()).unwrap();
        assert!(
            probs
                .iter()
                .any(|p| p.field == "owners" && p.key == "delhi")
        );
        assert!(out.factions.get("invented_faction").is_none());
    }

    #[test]
    fn no_rebellion_identity_is_invented_when_the_targets_configuration_is_missing() {
        let mut project = Project::default();
        project.owners.insert("delhi".into(), "britain".into());
        let mut cat = catalog();
        cat.rebel_factions.remove("britain");
        let (out, probs) = resolve(&omitted_owner_rules(), &project, &cat).unwrap();
        assert_eq!(out.regions["delhi"]["owner"], "britain");
        assert_eq!(out.regions["delhi"]["rebel_faction"], "mughal");
        assert!(probs.iter().any(|p| p.kind == "existing_rebel_required"));
    }
}
