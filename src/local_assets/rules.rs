//! Native rebuild of `rules/{mapping,db_rows,loc}` from the user's own ETW/NTW source snapshot.
//!
//! Everything about the games (region names, colours, religions, settlements, slots, unit and building keys, start
//! forces of the European powers, localisation strings, table layouts) is read from the snapshot's start positions,
//! `regions.esf`, database tables and `.loc` files. The only shipped input is the compact authored conversion policy
//! (`policy/rules/*.json`: who owns which region in 1805, which NTW unit stands in for which ETW unit, how ETW building
//! levels fold into NTW chains, the start stacks of the added colonial powers, ...) and the schema metadata
//! (`rules/schema_{emp,nap}.ron`), which is copied next to the generated rules because the table codec reads it from
//! the rules root.
//!
//! The generated tree has the shape `Rules::load` and the content/start-position code consume.

use crate::{
    db::{self, Cell, Table},
    esf::{self, Node},
    map,
    model::{Progress, Rules},
    snapshot::{Game, Snapshot},
};
use anyhow::{Context, Result, bail, ensure};
use regex::Regex;
use serde::Deserialize;
use serde_json::{Map, Value, json, ser::PrettyFormatter};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

const ETW_STARTPOS: &str = "campaigns/main/startpos.esf";
const ETW_REGIONS: &str = "campaign_maps/global_map/regions.esf";
const NTW_STARTPOS: &str = "campaigns/eur_napoleon/startpos.esf";
const SLOT_TOWN_PREFIX: &str = "town:";
const SLOT_PORT_PREFIX: &str = "port:";
const SETTLEMENT_LOC_PREFIX: &str = "start_pos_settlements_onscreen_name_settlement:";

// ------------------------------------------------------------------------------------------------ policy

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegionsPolicy {
    owner: BTreeMap<String, String>,
    notes: BTreeMap<String, String>,
    analog: BTreeMap<String, String>,
    extra_unit_resources: BTreeMap<String, Vec<String>>,
    display_names: BTreeMap<String, String>,
    rebel_remap: BTreeMap<String, String>,
    rebel_by_owner: BTreeMap<String, String>,
    rebel_fallback: String,
    subculture_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NtwFaction {
    capital: String,
    playable: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EtwFaction {
    capital: String,
    playable: bool,
    template: String,
    rebel_faction: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FactionsPolicy {
    ntw: BTreeMap<String, NtwFaction>,
    etw: BTreeMap<String, EtwFaction>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnitsPolicy {
    rules: Vec<(String, String)>,
    class_default: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrestigeMuseum {
    prefixes: Vec<String>,
    words: Vec<String>,
    chain: String,
    level: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TownReligion {
    prefix: String,
    religions: Vec<String>,
    levels: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildingsPolicy {
    resource_slot: BTreeMap<String, String>,
    chain_slot: BTreeMap<String, String>,
    settlement_sub: BTreeMap<String, String>,
    simple: BTreeMap<String, BTreeMap<String, i64>>,
    plantation: BTreeMap<String, i64>,
    pioneer: Vec<String>,
    prestige_culture: BTreeMap<String, String>,
    prestige_museum: PrestigeMuseum,
    town: BTreeMap<String, (String, i64)>,
    town_religion: TownReligion,
    town_natural: BTreeMap<String, String>,
    town_chain: BTreeMap<String, BTreeMap<String, String>>,
}

type Stack = (String, String, Vec<String>);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmiesPolicy {
    place: BTreeMap<String, Vec<String>>,
    extra: BTreeMap<String, Vec<Stack>>,
    colonial: BTreeMap<String, Vec<Stack>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignPolicy {
    campaign: String,
    variables: BTreeMap<String, f64>,
    loc: Vec<(String, String, String)>,
}

struct Policy {
    regions: RegionsPolicy,
    factions: FactionsPolicy,
    units: UnitsPolicy,
    buildings: BuildingsPolicy,
    armies: ArmiesPolicy,
    campaign: CampaignPolicy,
}

fn read_policy<T: for<'de> Deserialize<'de>>(dir: &Path, name: &str) -> Result<T> {
    let path = dir.join(name);
    let bytes =
        fs::read(&path).with_context(|| format!("Read conversion policy {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("Parse conversion policy {}", path.display()))
}

impl Policy {
    fn load(shipped: &Path) -> Result<Self> {
        let dir = shipped.join("policy").join("rules");
        Ok(Self {
            regions: read_policy(&dir, "regions.json")?,
            factions: read_policy(&dir, "factions.json")?,
            units: read_policy(&dir, "units.json")?,
            buildings: read_policy(&dir, "buildings.json")?,
            armies: read_policy(&dir, "armies.json")?,
            campaign: read_policy(&dir, "campaign.json")?,
        })
    }
}

// ------------------------------------------------------------------------------------------------ source access

struct Source {
    snapshot: Snapshot,
    /// Bootstrap rules: only `root` (where the schemas live) is meaningful.
    boot: Rules,
}

impl Source {
    fn table(&self, game: Game, name: &str) -> Result<Arc<Table>> {
        db::load_table(&self.snapshot, game, &self.boot, name)
    }

    fn esf(&self, game: Game, path: &str) -> Result<esf::Esf> {
        esf::read(&self.snapshot.read(game, path)?)
            .with_context(|| format!("Parse {} {path}", game.dir()))
    }

    /// Localisation of the game as the engine sees it: `localisation.loc` overlaid by `ui.loc`.
    fn loc(&self, game: Game) -> Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        for name in ["text/localisation.loc", "text/ui.loc"] {
            let parsed = db::parse_loc(&self.snapshot.read(game, name)?)
                .with_context(|| format!("Decode {} {name}", game.dir()))?;
            out.extend(parsed.into_iter().map(|(k, (t, _))| (k, t)));
        }
        Ok(out)
    }
}

fn str_cell<'a>(table: &Table, row: &'a [Cell], column: &str) -> Result<&'a str> {
    match &row[table.column(column)?] {
        Cell::Str(s) => Ok(s),
        other => bail!("{}.{column} is not a string ({other:?})", table.name),
    }
}

fn int_cell(table: &Table, row: &[Cell], column: &str) -> Result<i64> {
    Ok(match &row[table.column(column)?] {
        Cell::I16(v) => i64::from(*v),
        Cell::I32(v) => i64::from(*v),
        Cell::I64(v) => *v,
        other => bail!("{}.{column} is not an integer ({other:?})", table.name),
    })
}

fn keys_of(table: &Arc<Table>, column: &str) -> Result<BTreeSet<String>> {
    table
        .rows
        .iter()
        .map(|r| Ok(str_cell(table, r, column)?.to_owned()))
        .collect()
}

fn vtext(v: &esf::Value) -> Result<String> {
    match v {
        esf::Value::String16(s) | esf::Value::String8(s) => Ok(s.clone()),
        other => bail!("expected a string value, found {other:?}"),
    }
}

fn vfloat(v: &esf::Value) -> Result<f64> {
    match v {
        esf::Value::F32(x) => Ok(f64::from(*x)),
        esf::Value::F64(x) => Ok(*x),
        other => bail!("expected a number, found {other:?}"),
    }
}

fn scalar_text(node: &Node, index: usize) -> Result<String> {
    vtext(node.value_at(index)?)
}

/// The record of an array item (Python `item_rec`).
fn item_record(item: &[Node]) -> Result<&Node> {
    item.iter()
        .find(|n| matches!(n, Node::Record { .. }))
        .context("array item holds no record")
}

/// Every record / record array named `name` below (and including) `node`, depth first.
fn find_all<'a>(node: &'a Node, name: &str) -> Vec<&'a Node> {
    let mut out = Vec::new();
    node.walk(&mut |n, _| {
        if n.name() == Some(name) {
            out.push(n);
        }
    });
    out
}

fn world(root: &Node) -> Result<&Node> {
    root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "WORLD"])
}

fn regions_array(root: &Node) -> Result<&Vec<Vec<Node>>> {
    world(root)?
        .path(&["REGION_MANAGER", "REGIONS_ARRAY"])?
        .items()
}

fn title_case(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_letter = false;
    for c in text.chars() {
        if c.is_alphabetic() {
            if prev_letter {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_letter = true;
        } else {
            out.push(c);
            prev_letter = false;
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ output helpers

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let ordered: BTreeMap<&String, Value> =
                map.iter().map(|(k, v)| (k, sorted(v))).collect();
            Value::Object(ordered.into_iter().map(|(k, v)| (k.clone(), v)).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

fn write_json(path: &Path, value: &Value, indent: &'static [u8], sort_keys: bool) -> Result<()> {
    use serde::Serialize;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    let ordered;
    let value = if sort_keys {
        ordered = sorted(value);
        &ordered
    } else {
        value
    };
    let mut out = std::io::BufWriter::new(fs::File::create(path)?);
    let mut ser =
        serde_json::Serializer::with_formatter(&mut out, PrettyFormatter::with_indent(indent));
    value.serialize(&mut ser)?;
    use std::io::Write;
    out.write_all(b"\n")?;
    out.flush()
        .with_context(|| format!("Write {}", path.display()))
}

fn write_tsv(path: &Path, rows: &[(String, String, u8, String)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    use std::io::Write;
    let mut out = std::io::BufWriter::new(fs::File::create(path)?);
    for (key, text, tip, target) in rows {
        ensure!(
            !key.contains(['\t', '\n']) && !text.contains(['\t', '\n']),
            "localisation row {key:?} contains a tab or line break"
        );
        writeln!(out, "{key}\t{text}\t{tip}\t{target}")?;
    }
    out.flush()
        .with_context(|| format!("Write {}", path.display()))
}

// ------------------------------------------------------------------------------------------------ source catalogs read from the start positions

/// What the ETW world start position and `regions.esf` say about one world region.
struct EtwRegion {
    theatre: String,
    settlement: String,
    settlement_loc: String,
    breakdown: BTreeMap<String, f64>,
    theatre_name: String,
    rebel: (String, String, String),
}

struct EtwWorld {
    regions: BTreeMap<String, EtwRegion>,
    /// Theatre key -> region keys (`regions.esf` `theatres_and_region_keys`).
    theatres: BTreeMap<String, Vec<String>>,
    /// Every region key of `regions.esf` (land, sea, impassable).
    esf_region_keys: Vec<String>,
    /// Slot name -> (region, building levels); insertion ordered like the start position lists them.
    slots: Vec<(String, (String, Vec<String>))>,
}

fn read_etw_world(src: &Source) -> Result<EtwWorld> {
    let regions_esf = src.esf(Game::Etw, ETW_REGIONS)?;
    let mut theatres: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for item in regions_esf.root.get("theatres_and_region_keys")?.items()? {
        let rec = item.first().context("empty theatre item")?;
        let key = scalar_text(rec, 0)?;
        let mut keys = Vec::new();
        for entry in rec.get("region_keys")?.items()? {
            keys.push(
                entry
                    .first()
                    .context("empty region key item")?
                    .as_str()?
                    .to_owned(),
            );
        }
        theatres.insert(key, keys);
    }
    let mut esf_region_keys = Vec::new();
    for item in regions_esf
        .root
        .path(&["region_data", "regions"])?
        .items()?
    {
        esf_region_keys.push(
            item.first()
                .context("empty region record")?
                .as_str()?
                .to_owned(),
        );
    }
    let theatre_of: HashMap<&str, &str> = theatres
        .iter()
        .flat_map(|(t, keys)| keys.iter().map(move |k| (k.as_str(), t.as_str())))
        .collect();

    let startpos = src.esf(Game::Etw, ETW_STARTPOS)?;
    let mut regions = BTreeMap::new();
    let mut slots: Vec<(String, (String, Vec<String>))> = Vec::new();
    let mut slot_index: HashMap<String, usize> = HashMap::new();
    for item in regions_array(&startpos.root)? {
        let rec = item_record(item)?;
        let key = scalar_text(rec, 0)?;
        let settlement = rec.get("SETTLEMENT")?;
        let breakdown_node = find_all(rec, "RELIGION_BREAKDOWN")
            .into_iter()
            .next()
            .with_context(|| format!("region {key}: no RELIGION_BREAKDOWN"))?;
        let mut breakdown = BTreeMap::new();
        for entry in breakdown_node.items()? {
            let vals: Vec<&esf::Value> = entry
                .iter()
                .filter_map(|n| {
                    if let Node::Value(v) = n {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect();
            ensure!(
                vals.len() >= 2,
                "region {key}: short religion breakdown entry"
            );
            let share = (vfloat(vals[1])? * 10_000.0).round() / 10_000.0;
            breakdown.insert(vtext(vals[0])?, share);
        }
        for slot in find_all(rec, "REGION_SLOT") {
            let name = scalar_text(slot, 1)?;
            let mut buildings = Vec::new();
            if let Ok(manager) = slot.get("BUILDING_MANAGER") {
                for building in find_all(manager, "BUILDING") {
                    buildings.push(scalar_text(building, 1)?);
                }
            }
            match slot_index.get(&name) {
                Some(&at) => slots[at].1 = (key.clone(), buildings),
                None => {
                    slot_index.insert(name.clone(), slots.len());
                    slots.push((name, (key.clone(), buildings)));
                }
            }
        }
        let theatre = theatre_of
            .get(key.as_str())
            .with_context(|| format!("region {key} is in no theatre of regions.esf"))?
            .to_string();
        let region = EtwRegion {
            theatre,
            settlement: scalar_text(settlement, 1)?,
            settlement_loc: scalar_text(settlement.get("CAMPAIGN_LOCALISATION")?, 0)?,
            breakdown,
            theatre_name: scalar_text(rec, 17)?,
            rebel: (
                scalar_text(rec, 18)?,
                scalar_text(rec, 19)?,
                scalar_text(rec, 20)?,
            ),
        };
        regions.insert(key, region);
    }
    Ok(EtwWorld {
        regions,
        theatres,
        esf_region_keys,
        slots,
    })
}

/// What the NTW 1805 start position (`eur_napoleon`) provides: unit resources and rebel triple of each region, the
/// stacks of every faction and the faction keys it lists.
struct NtwStart {
    resources: BTreeMap<String, Vec<String>>,
    rebels: BTreeMap<String, (String, String, String)>,
    factions: Vec<String>,
    stacks: BTreeMap<String, Vec<Vec<(String, String)>>>,
}

fn read_ntw_start(src: &Source) -> Result<NtwStart> {
    let startpos = src.esf(Game::Ntw, NTW_STARTPOS)?;
    let mut resources = BTreeMap::new();
    let mut rebels = BTreeMap::new();
    for item in regions_array(&startpos.root)? {
        let rec = item_record(item)?;
        let key = scalar_text(rec, 0)?;
        let mut list = Vec::new();
        for entry in rec.get("RESOURCES_ARRAY")?.items()? {
            for node in entry {
                if let Node::Value(v) = node {
                    list.push(vtext(v)?);
                }
            }
        }
        resources.insert(key.clone(), list);
        rebels.insert(
            key,
            (
                scalar_text(rec, 19)?,
                scalar_text(rec, 20)?,
                scalar_text(rec, 21)?,
            ),
        );
    }
    let mut factions = Vec::new();
    let mut stacks = BTreeMap::new();
    for item in world(&startpos.root)?.get("FACTION_ARRAY")?.items()? {
        let rec = item_record(item)?;
        let key = scalar_text(rec, 1)?;
        let mut faction_stacks = Vec::new();
        for army_item in rec.get("ARMY_ARRAY")?.items()? {
            let army = item_record(army_item)?;
            let mut units = Vec::new();
            for unit_item in army.get("UNITS_ARRAY")?.items()? {
                for node in unit_item {
                    let Some(name) = node.name() else { continue };
                    if name != "LAND_UNIT" && name != "NAVAL_UNIT" {
                        continue;
                    }
                    let record_key = node
                        .children()?
                        .iter()
                        .find(|c| matches!(c.name(), Some("LAND_RECORD_KEY" | "NAVAL_RECORD_KEY")))
                        .with_context(|| format!("{key}: unit without record key"))?;
                    units.push((name.to_owned(), scalar_text(record_key, 0)?));
                }
            }
            faction_stacks.push(units);
        }
        factions.push(key.clone());
        stacks.insert(key, faction_stacks);
    }
    Ok(NtwStart {
        resources,
        rebels,
        factions,
        stacks,
    })
}

// ------------------------------------------------------------------------------------------------ mapping/regions + factions

struct Mapping {
    regions: Value,
    factions: Value,
}

fn build_regions_and_factions(
    src: &Source,
    policy: &Policy,
    etw: &EtwWorld,
    ntw: &NtwStart,
    etw_loc: &BTreeMap<String, String>,
) -> Result<Mapping> {
    let ntw_factions_table = src.table(Game::Ntw, "factions")?;
    let ntw_factions = keys_of(&ntw_factions_table, "key")?;
    let etw_factions = keys_of(&src.table(Game::Etw, "factions")?, "key")?;
    let known_factions: BTreeSet<&String> = ntw_factions
        .iter()
        .chain(policy.factions.etw.keys())
        .collect();
    let subcultures = keys_of(&src.table(Game::Ntw, "cultures_subcultures")?, "subculture")?;
    let unit_resources = keys_of(&src.table(Game::Ntw, "region_unit_resources")?, "key")?;
    let etw_regions_table = src.table(Game::Etw, "regions")?;
    let ntw_regions = keys_of(&src.table(Game::Ntw, "regions")?, "key")?;
    let mut etw_region_rows: HashMap<&str, &Vec<Cell>> = HashMap::new();
    for row in &etw_regions_table.rows {
        etw_region_rows.insert(str_cell(&etw_regions_table, row, "key")?, row);
    }

    // the playable world regions are the three land theatres of regions.esf
    let mut world_regions: Vec<&String> = Vec::new();
    for theatre in ["1", "2", "3"] {
        world_regions.extend(
            etw.theatres
                .get(theatre)
                .with_context(|| format!("regions.esf has no theatre {theatre}"))?,
        );
    }
    let distinct: BTreeSet<&String> = world_regions.iter().copied().collect();
    ensure!(
        distinct.len() == world_regions.len(),
        "a region is listed in two theatres"
    );
    let missing: Vec<&&String> = world_regions
        .iter()
        .filter(|k| !policy.regions.owner.contains_key(**k))
        .collect();
    let extra: Vec<&String> = policy
        .regions
        .owner
        .keys()
        .filter(|k| !distinct.contains(k))
        .collect();
    ensure!(
        missing.is_empty() && extra.is_empty(),
        "ownership policy does not match the world map: no owner for {missing:?}, owners for unknown regions {extra:?}"
    );

    let mut regions = Map::new();
    for key in world_regions.iter().copied() {
        let source = etw
            .regions
            .get(key)
            .with_context(|| format!("region {key}: absent from the ETW start position"))?;
        let dominant = {
            let mut best: Option<(&String, f64)> = None;
            for (religion, share) in &source.breakdown {
                if best.is_none_or(|(_, top)| *share > top) {
                    best = Some((religion, *share));
                }
            }
            best.map(|(r, _)| r.clone())
                .with_context(|| format!("region {key}: empty religion breakdown"))?
        };
        let analog = policy.regions.analog.get(key);
        let rebel = match analog {
            Some(analog) => ntw.rebels.get(analog).cloned().with_context(|| {
                format!("region {key}: analogue {analog} is not in the NTW start position")
            })?,
            None => {
                let (faction, name, culture) = &source.rebel;
                let mut faction = policy
                    .regions
                    .rebel_remap
                    .get(faction)
                    .unwrap_or(faction)
                    .clone();
                if !known_factions.contains(&faction) {
                    faction = policy
                        .regions
                        .rebel_by_owner
                        .get(&policy.regions.owner[key])
                        .unwrap_or(&policy.regions.rebel_fallback)
                        .clone();
                }
                let culture = policy
                    .regions
                    .subculture_map
                    .get(culture)
                    .unwrap_or(culture)
                    .clone();
                ensure!(
                    subcultures.contains(&culture),
                    "{key}: subculture {culture} is not in NTW"
                );
                (faction, name.clone(), culture)
            }
        };
        let mut resources = vec!["global".to_owned()];
        if let Some(analog) = analog {
            let list = ntw.resources.get(analog).with_context(|| {
                format!("region {key}: analogue {analog} is not in the NTW start position")
            })?;
            for r in list {
                if !resources.contains(r) {
                    resources.push(r.clone());
                }
            }
        }
        for r in policy
            .regions
            .extra_unit_resources
            .get(key)
            .into_iter()
            .flatten()
        {
            if !resources.contains(r) {
                resources.push(r.clone());
            }
        }
        if let Some(bad) = resources.iter().find(|r| !unit_resources.contains(*r)) {
            bail!("region {key}: unit resource {bad} is not an NTW region_unit_resources key");
        }
        ensure!(
            !ntw_regions.contains(key),
            "region {key} already exists in the NTW regions table"
        );
        let row = etw_region_rows
            .get(key.as_str())
            .with_context(|| format!("region {key}: absent from the ETW regions table"))?;
        let name = etw_loc
            .get(&format!("regions_onscreen_{key}"))
            .with_context(|| format!("region {key}: no ETW display name"))?;
        let capital_name = etw_loc.get(&source.settlement_loc).with_context(|| {
            format!(
                "region {key}: no ETW name for settlement string {}",
                source.settlement_loc
            )
        })?;
        regions.insert(
            key.clone(),
            json!({
                "owner": policy.regions.owner[key],
                "theatre": source.theatre,
                "religion": dominant,
                "religion_breakdown": source.breakdown,
                "continent": str_cell(&etw_regions_table, row, "continent")?,
                "lookup_rgb": [
                    int_cell(&etw_regions_table, row, "r")?,
                    int_cell(&etw_regions_table, row, "g")?,
                    int_cell(&etw_regions_table, row, "b")?
                ],
                "name": name,
                "capital_settlement": source.settlement,
                "capital_settlement_loc_key": source.settlement_loc,
                "capital_settlement_name": capital_name,
                "ntw_analog": analog,
                "unit_resources": resources,
                "theatre_name": source.theatre_name,
                "rebel_faction": rebel.0,
                "rebel_name": rebel.1,
                "sc_culture": rebel.2,
                "notes": policy.regions.notes.get(key).cloned().unwrap_or_default(),
            }),
        );
    }

    let mut owned: BTreeMap<&str, usize> = BTreeMap::new();
    for rule in regions.values() {
        *owned
            .entry(rule["owner"].as_str().unwrap_or_default())
            .or_default() += 1;
    }
    let held = |faction: &str| owned.get(faction).copied().unwrap_or(0);
    let mut factions = Map::new();
    let mut alive_ntw: BTreeSet<&str> = BTreeSet::new();
    for (key, info) in &policy.factions.ntw {
        ensure!(
            ntw_factions.contains(key),
            "faction {key} is not an NTW faction"
        );
        ensure!(
            regions.contains_key(&info.capital),
            "faction {key}: unknown capital {}",
            info.capital
        );
        alive_ntw.insert(key);
        factions.insert(
            key.clone(),
            json!({"source": "ntw", "ntw_template": key, "playable": info.playable, "capital": info.capital, "alive": true, "regions": held(key)}),
        );
    }
    // every other faction of the NTW 1805 start position is present but owns nothing on the world map
    for key in &ntw.factions {
        if alive_ntw.contains(key.as_str()) || factions.contains_key(key) {
            continue;
        }
        ensure!(
            ntw_factions.contains(key),
            "faction {key} of the NTW start position is not in its factions table"
        );
        ensure!(
            held(key) == 0,
            "faction {key} owns regions but has no capital in the faction policy"
        );
        factions.insert(
            key.clone(),
            json!({"source": "ntw", "ntw_template": key, "playable": false, "capital": null, "alive": false, "regions": 0}),
        );
    }
    for (key, info) in &policy.factions.etw {
        ensure!(
            etw_factions.contains(key) && !ntw_factions.contains(key),
            "faction {key} must be an ETW-only faction"
        );
        ensure!(
            regions.contains_key(&info.capital),
            "faction {key}: unknown capital {}",
            info.capital
        );
        ensure!(
            ntw_factions.contains(&info.template) && ntw_factions.contains(&info.rebel_faction),
            "faction {key}: template {} / rebel faction {} are not NTW factions",
            info.template,
            info.rebel_faction
        );
        factions.insert(
            key.clone(),
            json!({
                "source": "etw", "ntw_template": info.template, "playable": info.playable, "capital": info.capital,
                "alive": true, "regions": held(key), "rebel_faction": info.rebel_faction
            }),
        );
    }
    for (owner, _) in &owned {
        let info = factions
            .get(*owner)
            .with_context(|| format!("owner {owner} is not a faction of the faction policy"))?;
        ensure!(info["alive"] == true, "owner {owner} is marked dead");
        let capital = info["capital"].as_str().unwrap_or_default();
        ensure!(
            regions[capital]["owner"].as_str() == Some(*owner),
            "capital {capital} of {owner} is owned by someone else"
        );
    }
    Ok(Mapping {
        regions: Value::Object(regions),
        factions: Value::Object(factions),
    })
}

// ------------------------------------------------------------------------------------------------ mapping/units

fn build_units(src: &Source, policy: &UnitsPolicy) -> Result<Value> {
    let etw_units = src.table(Game::Etw, "units")?;
    let ntw_units = src.table(Game::Ntw, "units")?;
    let ntw_keys = keys_of(&ntw_units, "key")?;
    let rules: Vec<(Regex, &String)> = policy
        .rules
        .iter()
        .map(|(pattern, target)| {
            ensure!(
                ntw_keys.contains(target),
                "unit rule {pattern}: {target} is not an NTW unit"
            );
            Ok((
                Regex::new(pattern).with_context(|| format!("unit rule {pattern}"))?,
                target,
            ))
        })
        .collect::<Result<_>>()?;
    for target in policy.class_default.values() {
        ensure!(
            ntw_keys.contains(target),
            "class default {target} is not an NTW unit"
        );
    }
    let mut out = Map::new();
    for row in &etw_units.rows {
        let key = str_cell(&etw_units, row, "key")?;
        let class = str_cell(&etw_units, row, "class")?;
        let target = match rules.iter().find(|(rx, _)| rx.is_match(key)) {
            Some((_, target)) => (*target).clone(),
            None => policy
                .class_default
                .get(class)
                .cloned()
                .with_context(|| format!("no unit mapping for {key} (class {class})"))?,
        };
        out.insert(key.to_owned(), Value::String(target));
    }
    Ok(Value::Object(out))
}

// ------------------------------------------------------------------------------------------------ mapping/buildings

struct BuildingMaps {
    default: Value,
    by_slot: Value,
    slot_types: Value,
    slot_type_of: HashMap<String, String>,
}

fn build_buildings(src: &Source, policy: &BuildingsPolicy, etw: &EtwWorld) -> Result<BuildingMaps> {
    let ntw_levels = src.table(Game::Ntw, "building_levels")?;
    let mut chain_of: HashMap<String, String> = HashMap::new();
    let mut levels_of_chain: HashMap<String, Vec<String>> = HashMap::new();
    for row in &ntw_levels.rows {
        let level = str_cell(&ntw_levels, row, "level_name")?.to_owned();
        let chain = str_cell(&ntw_levels, row, "chain")?.to_owned();
        levels_of_chain
            .entry(chain.clone())
            .or_default()
            .push(level.clone());
        chain_of.insert(level, chain);
    }
    let chain_to_slots = src.table(Game::Ntw, "building_chain_to_slots")?;
    let mut chain_slots: HashMap<String, BTreeSet<String>> = HashMap::new();
    for row in &chain_to_slots.rows {
        chain_slots
            .entry(str_cell(&chain_to_slots, row, "chain")?.to_owned())
            .or_default()
            .insert(str_cell(&chain_to_slots, row, "slot")?.to_owned());
    }
    let etw_levels_table = src.table(Game::Etw, "building_levels")?;
    let etw_levels: Vec<String> = etw_levels_table
        .rows
        .iter()
        .map(|r| Ok(str_cell(&etw_levels_table, r, "level_name")?.to_owned()))
        .collect::<Result<_>>()?;

    // NTW level `n` of a chain family: the level of the chain named `<chain><n>_...`
    let ntw_level = |chain: &str, n: i64| -> Result<String> {
        let prefix = format!("{chain}{n}_");
        let found: Vec<&String> = levels_of_chain
            .get(chain)
            .into_iter()
            .flatten()
            .filter(|l| l.starts_with(&prefix))
            .collect();
        match found.as_slice() {
            [one] => Ok((*one).clone()),
            other => bail!(
                "NTW chain {chain} has {} levels named {prefix}*, expected exactly one",
                other.len()
            ),
        }
    };

    let slots = &etw.slots;
    let slot_lookup: HashMap<&str, &(String, Vec<String>)> =
        slots.iter().map(|(n, v)| (n.as_str(), v)).collect();
    let town_intent = |level: &str| -> Option<(String, i64)> {
        if let Some((intent, n)) = policy.town.get(level) {
            return Some((intent.clone(), *n));
        }
        let rel = &policy.town_religion;
        let rest = level.strip_prefix(&rel.prefix)?;
        let (religion, index) = rest.rsplit_once('_')?;
        let index: i64 = index.parse().ok()?;
        (rel.religions.iter().any(|r| r == religion) && (0..rel.levels).contains(&index))
            .then(|| ("religion".to_owned(), index + 1))
    };

    // ---- slot types
    let towns_table = src.table(Game::Etw, "campaign_map_towns_and_ports")?;
    let mut towns: Vec<String> = Vec::new();
    let mut ports: Vec<String> = Vec::new();
    let mut kind_of_town: HashMap<String, String> = HashMap::new();
    for row in &towns_table.rows {
        let id = str_cell(&towns_table, row, "town_id")?.to_owned();
        kind_of_town.insert(
            id.clone(),
            str_cell(&towns_table, row, "slot_type")?.to_owned(),
        );
        if id.starts_with(SLOT_TOWN_PREFIX) {
            towns.push(id);
        } else if id.starts_with(SLOT_PORT_PREFIX) {
            ports.push(id);
        }
    }
    let mut stypes: HashMap<String, String> = HashMap::new();
    for town in &towns {
        let buildings: &[String] = slot_lookup
            .get(town.as_str())
            .map_or(&[], |v| v.1.as_slice());
        let intents: Vec<Option<String>> = buildings
            .iter()
            .map(|b| town_intent(b).map(|(i, _)| i))
            .collect();
        let has = |set: &[&str]| intents.iter().flatten().any(|i| set.contains(&i.as_str()));
        let kind = if has(&["edu", "secret", "religion"]) {
            "town-intellectual"
        } else if has(&["commerce", "supply"]) {
            "town-commercial"
        } else if has(&["industry", "gun"]) {
            "town-industrial"
        } else if kind_of_town[town] == "town-textile" {
            "town-commercial"
        } else {
            "town-industrial"
        };
        stypes.insert(town.clone(), kind.to_owned());
    }
    // flip at most one unbuilt town per region to intellectual
    let mut by_region: BTreeMap<&str, Vec<&String>> = BTreeMap::new();
    for town in &towns {
        by_region
            .entry(town.split(':').nth(1).unwrap_or_default())
            .or_default()
            .push(town);
    }
    let world_regions: BTreeSet<&str> = slots.iter().map(|(_, v)| v.0.as_str()).collect();
    for (region, region_towns) in &by_region {
        if !world_regions.contains(region) {
            continue;
        }
        if region_towns
            .iter()
            .any(|t| stypes[*t] == "town-intellectual")
        {
            continue;
        }
        let mut unbuilt: Vec<&&String> = region_towns
            .iter()
            .filter(|t| slot_lookup.get(t.as_str()).is_none_or(|v| v.1.is_empty()))
            .collect();
        unbuilt.sort();
        let ordinal_sum: u32 = region.chars().map(u32::from).sum();
        let eligible = region_towns.len() >= 3 || (region_towns.len() == 2 && ordinal_sum % 3 == 0);
        if eligible {
            if let Some(last) = unbuilt.last() {
                stypes.insert((***last).clone(), "town-intellectual".to_owned());
            }
        }
    }
    for port in &ports {
        stypes.insert(port.clone(), "port".to_owned());
    }
    let slots_table = src.table(Game::Etw, "campaign_map_slots")?;
    for row in &slots_table.rows {
        let id = str_cell(&slots_table, row, "slot_id")?;
        let kind = str_cell(&slots_table, row, "slot_type")?;
        let mapped = policy
            .resource_slot
            .get(kind)
            .with_context(|| format!("unmapped ETW slot type {kind} ({id})"))?;
        stypes.insert(id.to_owned(), mapped.clone());
    }
    let settlements_table = src.table(Game::Etw, "campaign_map_settlements")?;
    for row in &settlements_table.rows {
        stypes.insert(
            str_cell(&settlements_table, row, "settlement_id")?.to_owned(),
            "settlement".to_owned(),
        );
    }
    for (name, _) in slots {
        if stypes.contains_key(name) {
            continue;
        }
        let parts: Vec<&str> = name.split(':').collect();
        let kind = if parts[0] == "settlement" && parts.len() == 4 {
            policy
                .settlement_sub
                .get(parts[3])
                .map_or("settlement", String::as_str)
                .to_owned()
        } else if let Some(mapped) = policy.resource_slot.get(parts[0]) {
            mapped.clone()
        } else if parts[0] == "port" {
            "port".to_owned()
        } else if parts[0] == "town" {
            "town-commercial".to_owned()
        } else {
            bail!("unclassified start-position slot {name}");
        };
        stypes.insert(name.clone(), kind);
    }

    // ---- levels
    let mut usage: HashMap<&str, Vec<(String, u32)>> = HashMap::new();
    for (name, (_, buildings)) in slots {
        for building in buildings {
            let counts = usage.entry(building.as_str()).or_default();
            let kind = &stypes[name];
            match counts.iter_mut().find(|(k, _)| k == kind) {
                Some(entry) => entry.1 += 1,
                None => counts.push((kind.clone(), 1)),
            }
        }
    }
    let museum = &policy.prestige_museum;
    let mut by_slot: BTreeMap<&str, Vec<(String, String)>> = BTreeMap::new();
    for level in &etw_levels {
        let mut options: Vec<(String, String)> = Vec::new();
        if let Some((intent, n)) = town_intent(level) {
            let chains = policy
                .town_chain
                .get(&intent)
                .with_context(|| format!("no town chain policy for intent {intent}"))?;
            for (slot, chain) in chains {
                options.push((slot.clone(), ntw_level(chain, n)?));
            }
        } else if let Some((chain, n)) = policy
            .simple
            .iter()
            .find_map(|(chain, levels)| levels.get(level).map(|n| (chain, *n)))
        {
            let target = ntw_level(chain, n)?;
            let slot = policy
                .chain_slot
                .get(chain)
                .with_context(|| format!("no slot type for chain {chain}"))?;
            options.push((slot.clone(), target.clone()));
            if slot == "settlement" {
                for extra in [
                    "settlement_1_slot",
                    "settlement_2_slot",
                    "settlement_3_slot",
                    "settlement_4_slot",
                ] {
                    options.push((extra.to_owned(), target.clone()));
                }
            }
        } else if let Some(n) = policy.plantation.get(level) {
            if !policy.pioneer.contains(level) {
                options.push(("wine".to_owned(), ntw_level("rWine", *n)?));
                options.push(("wheat".to_owned(), ntw_level("rFarm", (*n).min(3))?));
            }
        } else if let Some(target) = policy.prestige_culture.get(level) {
            options.push(("settlement".to_owned(), target.clone()));
        } else if museum
            .prefixes
            .iter()
            .any(|p| level.starts_with(p.as_str()))
            && museum.words.iter().any(|w| level.contains(w.as_str()))
        {
            options.push((
                "settlement".to_owned(),
                ntw_level(&museum.chain, museum.level)?,
            ));
        }
        by_slot.insert(level, options);
    }
    // default: the NTW slot type the level is used on most in the ETW start position, else its natural type
    let mut default: BTreeMap<&str, Option<String>> = BTreeMap::new();
    for level in &etw_levels {
        let options = &by_slot[level.as_str()];
        if options.is_empty() {
            default.insert(level, None);
            continue;
        }
        let pick = |slot: &str| {
            options
                .iter()
                .find(|(s, _)| s == slot)
                .map(|(_, l)| l.clone())
        };
        let mut choice = None;
        if let Some(counts) = usage.get(level.as_str()) {
            let mut ranked: Vec<&(String, u32)> = counts.iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1)); // stable: first seen wins ties
            choice = ranked.iter().find_map(|(slot, _)| pick(slot));
        }
        let choice = match choice {
            Some(c) => c,
            None => match town_intent(level) {
                Some((intent, _)) => {
                    let natural = policy
                        .town_natural
                        .get(&intent)
                        .with_context(|| format!("no natural town slot for intent {intent}"))?;
                    pick(natural)
                        .with_context(|| format!("level {level}: no option for slot {natural}"))?
                }
                None => pick("settlement").unwrap_or_else(|| options[0].1.clone()),
            },
        };
        default.insert(level, Some(choice));
    }
    // ---- verification: every building of the ETW start position stands on a slot its mapped chain allows
    let mut violations: Vec<String> = Vec::new();
    let allowed = |slot: &str, target: &str| -> bool {
        chain_of
            .get(target)
            .and_then(|chain| chain_slots.get(chain))
            .is_some_and(|slots| slots.contains(slot))
    };
    for (name, (_, buildings)) in slots {
        for building in buildings {
            let options = by_slot.get(building.as_str()).with_context(|| {
                format!("start-position building {building} is not an ETW building level")
            })?;
            let slot = &stypes[name];
            match options.iter().find(|(s, _)| s == slot) {
                None => {
                    if default[building.as_str()].is_some() {
                        violations.push(format!("no per-slot mapping for {building} on {slot}"));
                    }
                }
                Some((_, target)) => {
                    if !ntw_levels
                        .rows
                        .iter()
                        .any(|r| matches!(&r[0], Cell::Str(s) if s == target))
                    {
                        violations.push(format!(
                            "{building} maps to {target}, which is not an NTW level"
                        ));
                    } else if !allowed(slot, target) {
                        violations.push(format!(
                            "{building} on {slot} -> {target}: chain does not allow the slot"
                        ));
                    }
                }
            }
        }
    }
    for (level, options) in &by_slot {
        for (slot, target) in options {
            let chain = chain_of
                .get(target)
                .with_context(|| format!("{level}: {target} is not an NTW level"))?;
            if !allowed(slot, target) {
                violations.push(format!(
                    "{level}: chain {chain} of {target} does not allow slot {slot}"
                ));
            }
            if chain.ends_with("Tutorial") || chain.ends_with("Spain") {
                violations.push(format!(
                    "{level}: maps into the tutorial / Spain chain {chain}"
                ));
            }
        }
    }
    if !violations.is_empty() {
        violations.truncate(20);
        bail!(
            "building mapping violates the NTW chain/slot rules:\n{}",
            violations.join("\n")
        );
    }

    let default_json: Map<String, Value> = default
        .iter()
        .map(|(k, v)| {
            (
                (*k).to_owned(),
                v.clone().map_or(Value::Null, Value::String),
            )
        })
        .collect();
    let by_slot_json: Map<String, Value> = by_slot
        .iter()
        .map(|(k, options)| {
            let inner: Map<String, Value> = options
                .iter()
                .map(|(s, l)| (s.clone(), Value::String(l.clone())))
                .collect();
            ((*k).to_owned(), Value::Object(inner))
        })
        .collect();
    let types_json: Map<String, Value> = stypes
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    Ok(BuildingMaps {
        default: Value::Object(default_json),
        by_slot: Value::Object(by_slot_json),
        slot_types: Value::Object(types_json),
        slot_type_of: stypes,
    })
}

// ------------------------------------------------------------------------------------------------ mapping/armies

fn build_armies(
    src: &Source,
    policy: &ArmiesPolicy,
    ntw: &NtwStart,
    etw: &EtwWorld,
    regions: &Value,
    factions: &Value,
) -> Result<Value> {
    let ntw_units = keys_of(&src.table(Game::Ntw, "units")?, "key")?;
    let ports: BTreeSet<&str> = etw
        .slots
        .iter()
        .filter(|(n, _)| n.starts_with(SLOT_PORT_PREFIX))
        .filter_map(|(n, _)| n.split(':').nth(1))
        .collect();
    let mut out: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (faction, places) in &policy.place {
        let stacks = ntw.stacks.get(faction).with_context(|| {
            format!("faction {faction} has no stacks in the NTW start position")
        })?;
        ensure!(
            stacks.len() == places.len(),
            "faction {faction}: {} NTW start stacks but {} placements in the policy",
            stacks.len(),
            places.len()
        );
        for (units, region) in stacks.iter().zip(places) {
            let kind = if units.iter().any(|(t, _)| t == "NAVAL_UNIT") {
                "navy"
            } else {
                "army"
            };
            let keys: Vec<&String> = units.iter().map(|(_, k)| k).collect();
            let general = keys.first().is_some_and(|k| k.starts_with("Gen_"));
            out.entry(faction.clone())
                .or_default()
                .push(json!({"region": region, "kind": kind, "units": keys, "general": general}));
        }
    }
    for list in [&policy.extra, &policy.colonial] {
        for (faction, stacks) in list {
            for (region, kind, units) in stacks {
                let general = units.first().is_some_and(|k| k.starts_with("Gen_"));
                out.entry(faction.clone()).or_default().push(
                    json!({"region": region, "kind": kind, "units": units, "general": general}),
                );
            }
        }
    }
    for (faction, list) in &out {
        let info = factions
            .get(faction)
            .with_context(|| format!("armies: unknown faction {faction}"))?;
        ensure!(
            info["alive"] == true,
            "armies: faction {faction} is not alive"
        );
        for army in list {
            let region = army["region"].as_str().unwrap_or_default();
            let owner = regions.get(region).with_context(|| {
                format!("armies: {faction} is stationed in unknown region {region}")
            })?["owner"]
                .as_str();
            ensure!(
                owner == Some(faction.as_str()),
                "armies: {faction} stack in {region}, owned by {owner:?}"
            );
            let units: Vec<&str> = army["units"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            if let Some(bad) = units.iter().find(|u| !ntw_units.contains(**u)) {
                bail!("armies: {faction} stack in {region} uses {bad}, which is not an NTW unit");
            }
            ensure!(
                (1..=20).contains(&units.len()),
                "armies: {faction} stack in {region} has {} units",
                units.len()
            );
            if army["kind"] == "navy" {
                ensure!(
                    ports.contains(region),
                    "armies: {faction} fleet in {region}, which has no port"
                );
                ensure!(
                    army["general"] == false,
                    "armies: {faction} fleet in {region} has a general"
                );
            } else {
                const SHIPS: [&str; 6] = [
                    "2_Decker", "3_Decker", "Frigate", "Small_", "Trade_", "Special_",
                ];
                ensure!(
                    !units
                        .iter()
                        .any(|u| SHIPS.iter().any(|p| u.starts_with(p)) || u.starts_with("Steam_")),
                    "armies: {faction} army in {region} contains ships"
                );
            }
        }
    }
    let missing: Vec<&String> = factions
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(f, info)| info["alive"] == true && !out.contains_key(*f))
        .map(|(f, _)| f)
        .collect();
    ensure!(
        missing.is_empty(),
        "armies: alive factions without forces: {missing:?}"
    );
    Ok(json!(out))
}

// ------------------------------------------------------------------------------------------------ db_rows + loc

fn rows_doc(table: &str, version: u32, rows: Vec<Value>) -> Value {
    json!({"table": table, "version": version, "rows": rows})
}

fn content_regions_rows(
    src: &Source,
    etw: &EtwWorld,
    regions: &Value,
    etw_loc: &BTreeMap<String, String>,
) -> Result<(Value, Vec<(String, String)>)> {
    let ntw_regions = src.table(Game::Ntw, "regions")?;
    let etw_regions = src.table(Game::Etw, "regions")?;
    let ntw_keys = keys_of(&ntw_regions, "key")?;
    let regions_obj = regions
        .as_object()
        .context("regions mapping is not an object")?;
    let mut rows: Vec<(String, String, i64, i64, i64, String)> = Vec::new();
    let mut seen: BTreeMap<String, BTreeSet<[i64; 3]>> = BTreeMap::new();
    let mut keys: Vec<&String> = regions_obj.keys().collect();
    keys.sort();
    for key in keys {
        let rule = &regions_obj[key];
        let rgb: Vec<i64> = rule["lookup_rgb"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_i64)
            .collect();
        ensure!(rgb.len() == 3, "region {key}: lookup colour is not RGB");
        ensure!(
            !ntw_keys.contains(key),
            "region {key} already exists in NTW"
        );
        // lookup TGAs are per theatre: colours must be unique inside a theatre
        let theatre = rule["theatre"].as_str().unwrap_or_default().to_owned();
        ensure!(
            seen.entry(theatre)
                .or_default()
                .insert([rgb[0], rgb[1], rgb[2]]),
            "duplicate lookup colour in {key}"
        );
        rows.push((
            key.clone(),
            rule["continent"].as_str().unwrap_or_default().to_owned(),
            rgb[0],
            rgb[1],
            rgb[2],
            rule["capital_settlement_name"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        ));
    }
    // impassable land regions that regions.esf mentions and the ETW regions table lists (sahara, himalayas, ...)
    let mut etw_rows: HashMap<&str, &Vec<Cell>> = HashMap::new();
    for row in &etw_regions.rows {
        etw_rows.insert(str_cell(&etw_regions, row, "key")?, row);
    }
    let mut extras: Vec<&String> = etw
        .esf_region_keys
        .iter()
        .filter(|k| etw_rows.contains_key(k.as_str()) && !regions_obj.contains_key(*k))
        .collect();
    extras.sort();
    let mut extra_names = Vec::new();
    for key in extras {
        ensure!(
            !ntw_keys.contains(key),
            "region {key} already exists in NTW"
        );
        let row = etw_rows[key.as_str()];
        let name = etw_loc
            .get(&format!("regions_onscreen_{key}"))
            .or_else(|| etw_loc.get(&format!("regions_sea_onscreen_name_{key}")))
            .cloned()
            .unwrap_or_else(|| title_case(&key.replace('_', " ")));
        rows.push((
            key.clone(),
            str_cell(&etw_regions, row, "continent")?.to_owned(),
            int_cell(&etw_regions, row, "r")?,
            int_cell(&etw_regions, row, "g")?,
            int_cell(&etw_regions, row, "b")?,
            name.clone(),
        ));
        extra_names.push((key.clone(), name));
    }
    let fields = ntw_regions.field_names();
    ensure!(
        fields == ["key", "continent", "r", "g", "b", "battle_name"],
        "NTW regions table has unexpected columns {fields:?}"
    );
    let docs: Vec<Value> = rows
        .into_iter()
        .map(|(key, continent, r, g, b, battle)| {
            json!({"key": key, "continent": continent, "r": r, "g": g, "b": b, "battle_name": battle})
        })
        .collect();
    Ok((
        rows_doc("regions_tables", ntw_regions.version, docs),
        extra_names,
    ))
}

fn campaign_rows(src: &Source, policy: &CampaignPolicy) -> Result<[(String, Value); 2]> {
    let variables = src.table(Game::Ntw, "campaigns_campaign_variables_junctions")?;
    ensure!(
        variables.field_names() == ["variable_key", "campaign_name", "value"],
        "unexpected columns of campaigns_campaign_variables_junctions: {:?}",
        variables.field_names()
    );
    let junctions = src.table(Game::Ntw, "cdir_campaign_junctions")?;
    ensure!(
        junctions.field_names() == ["key", "campaign"],
        "unexpected columns of cdir_campaign_junctions: {:?}",
        junctions.field_names()
    );
    let variable_rows: Vec<Value> = policy
        .variables
        .iter()
        .map(|(key, value)| json!({"variable_key": key, "campaign_name": policy.campaign, "value": value}))
        .collect();
    Ok([
        (
            "frontend__campaigns_campaign_variables_junctions.json".to_owned(),
            rows_doc(
                "campaigns_campaign_variables_junctions_tables",
                variables.version,
                variable_rows,
            ),
        ),
        (
            "frontend__cdir_campaign_junctions.json".to_owned(),
            rows_doc(
                "cdir_campaign_junctions_tables",
                junctions.version,
                vec![json!({"key": policy.campaign, "campaign": policy.campaign})],
            ),
        ),
    ])
}

/// `loc/content.tsv`: the region, settlement, town/port, slot and theatre strings of the world map.
fn content_loc(
    src: &Source,
    policy: &Policy,
    etw: &EtwWorld,
    regions: &Value,
    extra_names: &[(String, String)],
    etw_loc: &BTreeMap<String, String>,
) -> Result<Vec<(String, String, u8, String)>> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    let mut put = |key: String, text: String| -> Result<()> {
        ensure!(
            !key.contains(['\t', '\n']) && !text.contains(['\t', '\n']),
            "loc row {key:?} has a tab or line break"
        );
        if let Some(old) = out.get(&key) {
            ensure!(
                *old == text,
                "conflicting localisation for {key}: {old:?} vs {text:?}"
            );
        }
        out.insert(key, text);
        Ok(())
    };
    let regions_obj = regions
        .as_object()
        .context("regions mapping is not an object")?;
    for (key, rule) in regions_obj {
        let name = policy
            .regions
            .display_names
            .get(key)
            .map(String::as_str)
            .or_else(|| rule["name"].as_str())
            .unwrap_or_default();
        put(format!("regions_onscreen_{key}"), name.to_owned())?;
        put(
            format!("regions_battle_name_{key}"),
            rule["capital_settlement_name"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        )?;
        put(
            format!(
                "start_pos_regions_rebel_faction_name_{key}{}",
                policy.campaign.campaign
            ),
            rule["rebel_name"].as_str().unwrap_or_default().to_owned(),
        )?;
    }
    // sea regions and the impassable land regions of regions.esf
    for key in &etw.esf_region_keys {
        if let Some(text) = etw_loc.get(&format!("regions_sea_onscreen_name_{key}")) {
            put(format!("regions_sea_onscreen_name_{key}"), text.clone())?;
        }
    }
    for (key, name) in extra_names {
        put(format!("regions_onscreen_{key}"), name.clone())?;
        put(format!("regions_battle_name_{key}"), name.clone())?;
    }
    // settlement names exactly as the ETW start position embeds them
    for (key, text) in etw_loc {
        if key.starts_with(SETTLEMENT_LOC_PREFIX) {
            let region = key["start_pos_settlements_onscreen_name_settlement:".len()..]
                .split(':')
                .next();
            if region.is_some_and(|r| regions_obj.contains_key(r)) {
                put(key.clone(), text.clone())?;
            }
        }
    }
    let towns = src.table(Game::Etw, "campaign_map_towns_and_ports")?;
    for row in &towns.rows {
        put(
            format!(
                "campaign_map_towns_and_ports_onscreen_name_{}",
                str_cell(&towns, row, "town_id")?
            ),
            str_cell(&towns, row, "onscreen_name")?.to_owned(),
        )?;
    }
    let slots = src.table(Game::Etw, "campaign_map_slots")?;
    for row in &slots.rows {
        let key = format!(
            "campaign_map_slots_onscreen_{}",
            str_cell(&slots, row, "slot_id")?
        );
        if let Some(text) = etw_loc.get(&key) {
            put(key, text.clone())?;
        }
    }
    let areas = src.table(Game::Etw, "campaign_map_playable_areas")?;
    for row in &areas.rows {
        let key = format!(
            "campaign_map_playable_areas_onscreen_name_{}",
            str_cell(&areas, row, "index")?
        );
        if let Some(text) = etw_loc.get(&key) {
            put(key, text.clone())?;
        }
    }
    Ok(out
        .into_iter()
        .map(|(k, t)| (k, t, 0, "localisation".to_owned()))
        .collect())
}

fn campaign_loc(policy: &CampaignPolicy) -> Result<Vec<(String, String, u8, String)>> {
    let mut rows: Vec<(String, String, u8, String)> = policy
        .loc
        .iter()
        .map(|(k, t, target)| (k.clone(), t.clone(), 0, target.clone()))
        .collect();
    for (key, _, _, target) in &rows {
        ensure!(
            target == "ui" || target == "localisation",
            "campaign string {key} targets unknown file {target}"
        );
    }
    rows.sort();
    Ok(rows)
}

// ------------------------------------------------------------------------------------------------ entry point

/// Rebuilds `<output_assets>/rules` from the verified source `snapshot` and the shipped policy under
/// `shipped_policy` (the shipped `assets` folder: `policy/rules/*.json`, `rules/schema_{emp,nap}.ron`).
pub(super) fn generate(
    snapshot: &Snapshot,
    shipped_policy: &Path,
    output_assets: &Path,
    progress: &Progress<'_>,
) -> Result<()> {
    let rules_root: PathBuf = output_assets.join("rules");
    let mapping_dir = rules_root.join("mapping");
    let rows_dir = rules_root.join("db_rows");
    let loc_dir = rules_root.join("loc");
    for dir in [&mapping_dir, &rows_dir, &loc_dir] {
        fs::create_dir_all(dir).with_context(|| format!("Create {}", dir.display()))?;
    }
    // vanilla data only: converter overlays never feed the rules
    let snapshot = snapshot.original();

    progress("Rules: reading conversion policy and schemas");
    let policy = Policy::load(shipped_policy)?;
    for schema in ["schema_emp.ron", "schema_nap.ron"] {
        let from = shipped_policy.join("rules").join(schema);
        fs::copy(&from, rules_root.join(schema))
            .with_context(|| format!("Copy schema {}", from.display()))?;
    }
    let src = Source {
        snapshot,
        boot: Rules {
            root: rules_root.clone(),
            regions: Value::Null,
            factions: Value::Null,
            units: Value::Null,
            buildings: Value::Null,
            buildings_by_slot_type: Value::Null,
            slot_types: Value::Null,
            armies: Value::Null,
        },
    };

    progress("Rules: reading the Empire world map and start position");
    let etw = read_etw_world(&src)?;
    progress("Rules: reading the Napoleon 1805 start position");
    let ntw = read_ntw_start(&src)?;
    let etw_loc = src.loc(Game::Etw)?;

    progress("Rules: regions and factions");
    let Mapping { regions, factions } =
        build_regions_and_factions(&src, &policy, &etw, &ntw, &etw_loc)?;
    progress("Rules: unit mapping");
    let units = build_units(&src, &policy.units)?;
    progress("Rules: building and slot mapping");
    let buildings = build_buildings(&src, &policy.buildings, &etw)?;
    progress("Rules: starting forces");
    let armies = build_armies(&src, &policy.armies, &ntw, &etw, &regions, &factions)?;

    for (name, value) in [
        ("regions", &regions),
        ("factions", &factions),
        ("units", &units),
        ("buildings", &buildings.default),
        ("buildings_by_slot_type", &buildings.by_slot),
        ("slot_types", &buildings.slot_types),
        ("armies", &armies),
    ] {
        write_json(&mapping_dir.join(format!("{name}.json")), value, b" ", true)?;
    }
    ensure!(
        buildings.slot_type_of.len() == buildings.slot_types.as_object().map_or(0, Map::len),
        "slot type count drifted"
    );

    progress("Rules: database rows");
    let (regions_doc, extra_names) = content_regions_rows(&src, &etw, &regions, &etw_loc)?;
    write_json(
        &rows_dir.join("content__regions.json"),
        &regions_doc,
        b"",
        false,
    )?;
    for (name, doc) in campaign_rows(&src, &policy.campaign)? {
        write_json(&rows_dir.join(name), &doc, b" ", false)?;
    }

    progress("Rules: localisation");
    write_tsv(
        &loc_dir.join("content.tsv"),
        &content_loc(&src, &policy, &etw, &regions, &extra_names, &etw_loc)?,
    )?;
    write_tsv(
        &loc_dir.join("frontend.tsv"),
        &campaign_loc(&policy.campaign)?,
    )?;

    // the map-owned rows are a function of the source map and the rules just written
    progress("Rules: map tables and strings");
    let rules = Rules::load(&rules_root).context("Load the generated mapping rules")?;
    let (map_rows, map_loc) = map::logic_rows(&src.snapshot, &rules)?;
    for (table, doc) in &map_rows {
        write_json(
            &rows_dir.join(format!("map_logic__{table}.json")),
            doc,
            b" ",
            false,
        )?;
    }
    write_tsv(&loc_dir.join("map_logic.tsv"), &map_loc)?;
    progress("Rules: done");
    Ok(())
}
