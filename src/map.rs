//! Native ETW `global_map` -> NTW `campaign_maps/nap_world` conversion: regions.esf (slot and trade-node schema
//! upgrade), verbatim AI/pathfinding ESFs, theatre TGAs, MTGI metadata, DB rows / localisation rows and the display
//! tree (see `terrain`). Also provides the original ETW region geometry for the editor.

mod triangulate;
pub mod splits;

use crate::{
    db::{self, CampaignTable},
    esf::{self, Node},
    model::{Progress, Project, Rules, TerrainRelief},
    regionmap::regions::{Region, RegionData},
    snapshot::{Game, Snapshot},
    terrain,
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::Path,
};

const ETW_MAP: &str = "campaign_maps/global_map";
const NTW_REF_MAPS: [&str; 5] = [
    "nap_europe",
    "nap_egypt",
    "nap_italy",
    "nap_spain",
    "nap_tut",
];

/// ETW world theatres (keys of campaign_map_playable_areas / regions.esf).
const THEATRES: [&str; 7] = [
    "1",
    "2",
    "3",
    "-1133129049",
    "836795134",
    "1197997136",
    "2113354257",
];

/// Sea-trade theatre -> (trade node group key, node key prefix, display name).
static SEA_THEATRE_GROUP: [(&str, &str, &str, &str); 4] = [
    ("1197997136", "wld_south_america", "wld_brazil", "Brazil"),
    (
        "2113354257",
        "wld_west_africa",
        "wld_ivory_coast",
        "Ivory Coast",
    ),
    (
        "836795134",
        "wld_east_africa",
        "wld_madagascar",
        "Madagascar",
    ),
    (
        "-1133129049",
        "wld_east_indies",
        "wld_east_indies",
        "East Indies",
    ),
];

const GENERAL_SLOTS: [&str; 6] = [
    "settlement_army-admin",
    "settlement_ordnance",
    "settlement_government",
    "settlement_culture",
    "settlement_minor",
    "settlement_navy-admin",
];
const FIXED_SETTLEMENT_SLOTS: [&str; 2] = ["settlement_road", "settlement_fortification"];
/// NTW has settlement_1_slot .. settlement_4_slot.
const MAX_GENERAL_SLOTS: usize = 4;

const SC_PREFIX_TO_CULTURE: [(&str, &str); 4] = [
    ("sc_european", "european"),
    ("sc_mideast", "middle_east"),
    ("sc_indian", "indian"),
    ("sc_tribal", "tribal"),
];
pub(crate) const TEMPLATE_SLOT_TYPES: [&str; 4] = [
    "port",
    "town-commercial",
    "town-industrial",
    "town-intellectual",
];

pub struct MapArtifacts {
    /// region -> exact slot keys/types written to regions.esf (`settlement_slots` contract, `slotv1`).
    pub slot_mapping: Value,
    /// regions.esf trade node index/key/theatre/pos/group (`tradenodesv1`).
    pub trade_nodes: Value,
    /// Table name (with `_tables`) -> `{"table","version","rows":[{field:value}]}`.
    pub db_rows: BTreeMap<String, Value>,
    /// (key, text, tooltip, target) rows for the `localisation` LOC.
    pub loc_rows: Vec<(String, String, u8, String)>,
}

#[derive(Clone, Debug)]
pub struct RegionPolygon {
    pub key: String,
    pub kind: String,
    pub triangles: Vec<[[f32; 2]; 3]>,
}

// ---------------------------------------------------------------------------------------------- ESF helpers

fn read_etw_regions(snapshot: &Snapshot) -> Result<esf::Esf> {
    esf::read(&snapshot.read(Game::Etw, &format!("{ETW_MAP}/regions.esf"))?)
        .context("Parse ETW global_map/regions.esf")
}

/// Children of an array item: the item itself, or the children of its single record (theatre arrays).
fn item_children(item: &[Node]) -> &[Node] {
    if let [Node::Record { children, .. }] = item {
        children
    } else {
        item
    }
}

fn scalars(nodes: &[Node]) -> Vec<&Node> {
    nodes
        .iter()
        .filter(|n| matches!(n, Node::Value(_)))
        .collect()
}

fn slot_array<'a>(item: &'a [Node]) -> Option<&'a Node> {
    item.iter()
        .find(|n| n.name() == Some("settlement_and_slots"))
        .and_then(|sas| sas.get("slot_descriptions").ok())
}

struct Theatre {
    key: String,
    lo: [f32; 2],
    hi: [f32; 2],
    transition_ids: Vec<u32>,
}

fn theatres(root: &Node) -> Result<Vec<Theatre>> {
    let mut out = Vec::new();
    for item in root.get("theatres")?.items()? {
        let ch = item_children(item);
        let v = scalars(ch);
        ensure!(v.len() >= 4, "Theatre record has fewer than 4 scalars");
        let mut transition_ids = Vec::new();
        for n in ch.iter().filter(|n| n.name() == Some("transition_areas")) {
            for a in n.items()? {
                transition_ids.push(a.first().context("Empty transition area")?.as_u32()?);
            }
        }
        out.push(Theatre {
            key: v[0].as_str()?.to_string(),
            lo: v[1].as_xy()?,
            hi: v[2].as_xy()?,
            transition_ids,
        });
    }
    Ok(out)
}

fn theatre_of_point(theatres: &[Theatre], p: [f32; 2]) -> Result<&Theatre> {
    theatres
        .iter()
        .find(|t| t.lo[0] <= p[0] && p[0] <= t.hi[0] && t.lo[1] <= p[1] && p[1] <= t.hi[1])
        .with_context(|| format!("Trade node {p:?} lies outside every theatre"))
}

// ---------------------------------------------------------------------------------------------- region geometry

/// A theatre's playable rectangle in world units (regions.esf `theatres`).
#[derive(Clone, Debug)]
pub struct TheatreRect {
    pub key: String,
    pub lo: [f32; 2],
    pub hi: [f32; 2],
}

/// Original ETW geometry and the fixed campaign geometry with the project's activations and splits applied.
pub struct RegionMap {
    /// regions.esf as shipped (needed to replay the activations and splits after any change)
    pub base: RegionData,
    /// Theatre rectangles of the displayed geometry.
    pub theatres: Vec<TheatreRect>,
    /// Geometry with the fixed bounds and project edits applied.
    pub data: RegionData,
    pub polygons: Vec<RegionPolygon>,
}

/// Parses the original ETW regions.esf: region geometry and theatre rectangles.
pub fn region_source(snapshot: &Snapshot) -> Result<(RegionData, Vec<TheatreRect>)> {
    let esf = read_etw_regions(snapshot)?;
    let data = RegionData::parse(&esf.root).context("Read ETW region geometry")?;
    let rects = theatres(&esf.root)?
        .into_iter()
        .map(|theatre| TheatreRect {
            key: theatre.key,
            lo: theatre.lo,
            hi: theatre.hi,
        })
        .collect();
    Ok((data, rects))
}
/// Triangulated editor preview (x east, y north), using the same fixed bounds and project replay as the build.
pub fn region_polygons(snapshot: &Snapshot, project: &Project) -> Result<RegionMap> {
    let (base, _) = region_source(snapshot)?;
    let (data, theatres) = splits::replay(&base, project)?;
    let polygons = polygons_of(&data, None)?;
    Ok(RegionMap {
        base,
        theatres,
        data,
        polygons,
    })
}

/// Triangulates every region of `data`. Regions whose areas and rings equal those of the same-keyed region in
/// `previous` (data + its polygons) reuse the earlier triangles.
pub fn polygons_of(
    data: &RegionData,
    previous: Option<(&RegionData, &[RegionPolygon])>,
) -> Result<Vec<RegionPolygon>> {
    let same_rings = |a: &Region, b: &Region| {
        a.areas.len() == b.areas.len()
            && a.areas.iter().zip(&b.areas).all(|(x, y)| {
                x.outlines.len() == y.outlines.len()
                    && x.outlines
                        .iter()
                        .zip(&y.outlines)
                        .all(|(p, q)| p.outer == q.outer && p.ring == q.ring)
            })
    };
    let mut out = Vec::with_capacity(data.regions.len());
    for region in &data.regions {
        if let Some((old_data, old_polygons)) = previous {
            let reusable = old_data
                .regions
                .iter()
                .find(|old| old.key == region.key)
                .filter(|old| same_rings(old, region))
                .and_then(|old| old_polygons.iter().find(|p| p.key == old.key));
            if let Some(polygon) = reusable {
                out.push(polygon.clone());
                continue;
            }
        }
        let mut triangles = Vec::new();
        for area in &region.areas {
            let (mut outers, mut holes) = (Vec::new(), Vec::new());
            for outline in &area.outlines {
                let ring = outline.ring.iter().map(|&i| data.v(i)).collect();
                if outline.outer {
                    outers.push(ring)
                } else {
                    holes.push(ring)
                }
            }
            triangulate::triangulate_area(outers, holes, &mut triangles);
        }
        out.push(RegionPolygon {
            key: region.key.clone(),
            kind: region.kind.clone(),
            triangles,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------- NTW reference statistics

/// Median `xyz.y` per NTW slot type over the five NTW reference maps.
fn ntw_slot_heights(snapshot: &Snapshot) -> Result<BTreeMap<String, f64>> {
    let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for map in NTW_REF_MAPS {
        let file =
            esf::read(&snapshot.read(Game::Ntw, &format!("campaign_maps/{map}/regions.esf"))?)
                .with_context(|| format!("Parse NTW {map} regions.esf"))?;
        for item in file.root.path(&["region_data", "regions"])?.items()? {
            let Some(arr) = slot_array(item) else {
                continue;
            };
            for s in arr.items()? {
                let ty = s.get(1).context("Slot without type")?.as_str()?;
                let y = s
                    .get(2)
                    .context("Slot without position")?
                    .as_xyz()
                    .with_context(|| format!("{map}: slot positions are not xyz"))?[1];
                values.entry(ty.to_string()).or_default().push(y as f64);
            }
        }
    }
    Ok(values
        .into_iter()
        .map(|(ty, mut v)| {
            v.sort_by(|a, b| a.total_cmp(b));
            let n = v.len();
            let median = if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            };
            (ty, median)
        })
        .collect())
}

fn text(row: &Value, field: &str) -> String {
    row.get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// (culture, slot type) -> template model, from the NTW `slots_art` table.
fn template_models(
    snapshot: &Snapshot,
    rules: &Rules,
) -> Result<HashMap<(String, String), String>> {
    let mut out = HashMap::new();
    for r in db::load_campaign_table(snapshot, rules, CampaignTable::SlotArt)?.json_rows()? {
        let slot = text(&r, "slot");
        if !TEMPLATE_SLOT_TYPES.contains(&slot.as_str()) {
            continue;
        }
        let model = text(&r, "template_model");
        // Culture rows without a template_model (indian / tribal) use the art model name.
        let model = if model.is_empty() {
            text(&r, "template_model_art_quality_zero")
        } else {
            model
        };
        out.insert((text(&r, "culture"), slot), model);
    }
    Ok(out)
}

fn region_cultures(rules: &Rules, region_keys: &[String]) -> HashMap<String, String> {
    region_keys
        .iter()
        .map(|k| {
            let sc = rules
                .regions
                .get(k)
                .and_then(|r| r.get("sc_culture"))
                .and_then(Value::as_str);
            let mut culture = "european";
            if let Some(sc) = sc {
                for (prefix, c) in SC_PREFIX_TO_CULTURE {
                    if sc.starts_with(prefix) {
                        culture = c;
                    }
                }
            }
            (k.clone(), culture.to_string())
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------- regions.esf conversion

struct SlotEntry {
    etw_key: String,
    new: Option<(String, String)>,
}

struct RegionReport {
    region: String,
    settlement_id: String,
    settlement_slot_type: String,
    general_slots: usize,
    slots: Vec<SlotEntry>,
}

/// NTW's template-backed pick prisms use outward normals from CCW file polygons.
/// ETW footprints are CW; legacy slots without a template still use the CW path.
fn convert_slot_footprint(source: &Node, has_template: bool) -> Result<Node> {
    let mut footprint = source.clone();
    if has_template {
        let Node::Value(esf::Value::Array { code: 0x4C, data }) = &mut footprint else {
            bail!("slot footprint is not a packed XY array");
        };
        ensure!(data.len() % 8 == 0, "slot footprint contains an incomplete XY vertex");
        let mut vertices = data.chunks_exact_mut(8);
        while let (Some(first), Some(last)) = (vertices.next(), vertices.next_back()) {
            first.swap_with_slice(last);
        }
    }
    Ok(footprint)
}

/// Rewrite every region's `slot_descriptions` from the ETW v0 slot model to the NTW v1 model.
fn convert_regions(
    root: &mut Node,
    slot_types: &Map<String, Value>,
    heights: &BTreeMap<String, f64>,
    templates: &HashMap<(String, String), String>,
    cultures: &HashMap<String, String>,
) -> Result<Vec<RegionReport>> {
    let mut report = Vec::new();
    for item in root
        .path_mut(&["region_data", "regions"])?
        .items_mut()?
        .iter_mut()
    {
        let region_key = item
            .first()
            .context("Empty region record")?
            .as_str()?
            .to_string();
        if slot_array(item).is_none() {
            continue;
        }
        let arr = item
            .iter_mut()
            .find(|n| n.name() == Some("settlement_and_slots"))
            .and_then(|sas| sas.get_mut("slot_descriptions").ok())
            .context("slot_descriptions vanished")?;
        ensure!(
            arr.version() == Some(0),
            "{region_key}: slot_descriptions is not ETW version 0"
        );
        let old = std::mem::take(arr.items_mut()?);

        let general: Vec<usize> = old
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s[1].as_str()
                    .map(|t| GENERAL_SLOTS.contains(&t))
                    .unwrap_or(false)
            })
            .map(|(i, _)| i)
            .collect();
        let n_general = general.len().min(MAX_GENERAL_SLOTS);
        let kept: HashSet<usize> = general.iter().copied().take(n_general).collect();
        let settlement_type = format!("settlement_{n_general}_slot");

        let mut new_items: Vec<Vec<Node>> = Vec::new();
        let mut entries = Vec::new();
        let mut index = 0usize;
        let mut settlement_id: Option<String> = None;
        for (i, s) in old.iter().enumerate() {
            ensure!(
                s.len() == 9,
                "{region_key}: slot description has {} fields, expected 9",
                s.len()
            );
            let key = s[0].as_str()?.to_string();
            let etype = s[1].as_str()?.to_string();
            let pos = s[2].as_xy()?;
            let is_general = GENERAL_SLOTS.contains(&etype.as_str());
            let nkey: String;
            let ntype: String;
            let template: String;
            if is_general || FIXED_SETTLEMENT_SLOTS.contains(&etype.as_str()) {
                if is_general && !kept.contains(&i) {
                    entries.push(SlotEntry {
                        etw_key: key,
                        new: None,
                    });
                    continue;
                }
                let parts: Vec<&str> = key.split(':').collect();
                ensure!(
                    parts.len() >= 3 && parts[0] == "settlement",
                    "Unexpected settlement slot key {key}"
                );
                let sid = parts[..3].join(":");
                ntype = if is_general {
                    settlement_type.clone()
                } else {
                    etype.clone()
                };
                nkey = format!("{sid}:{ntype}:{index}");
                index += 1;
                settlement_id = Some(sid);
                template = String::new();
            } else {
                ntype = slot_types
                    .get(&key)
                    .and_then(Value::as_str)
                    .with_context(|| format!("No NTW slot type for {key}"))?
                    .to_string();
                nkey = key.clone();
                template = if TEMPLATE_SLOT_TYPES.contains(&ntype.as_str()) {
                    let culture = cultures
                        .get(&region_key)
                        .context("Region culture missing")?;
                    templates
                        .get(&(culture.clone(), ntype.clone()))
                        .with_context(|| format!("No template model for {culture}/{ntype}"))?
                        .clone()
                } else {
                    String::new()
                };
            }
            let y = ((heights.get(&ntype).copied().unwrap_or(0.0)) * 1e6).round() / 1e6;
            let has_template = !template.is_empty();
            new_items.push(vec![
                esf::s16(nkey.clone()),
                esf::s16(ntype.clone()),
                esf::xyzv(pos[0], y as f32, pos[1]),
                s[3].clone(),
                s[4].clone(),
                s[5].clone(),
                esf::s16(template),
                convert_slot_footprint(&s[6], has_template)?,
                convert_slot_footprint(&s[7], has_template)?,
                convert_slot_footprint(&s[8], has_template)?,
            ]);
            entries.push(SlotEntry {
                etw_key: key,
                new: Some((nkey, ntype)),
            });
        }
        arr.set_version(1)?;
        *arr.items_mut()? = new_items;
        report.push(RegionReport {
            region: region_key.clone(),
            settlement_id: settlement_id
                .with_context(|| format!("{region_key}: region has slots but no settlement"))?,
            settlement_slot_type: settlement_type,
            general_slots: n_general,
            slots: entries,
        });
    }
    Ok(report)
}

/// Trade nodes v0 (arr_xy) -> v1 (array of `{utf16 key, xy}`).
fn convert_trade_nodes(root: &mut Node, keys: &[String]) -> Result<()> {
    let tn = root.get_mut("trade_nodes")?;
    ensure!(tn.version() == Some(0), "trade_nodes is not ETW version 0");
    let pts = tn
        .children()?
        .first()
        .context("trade_nodes has no data")?
        .xy_array()?;
    ensure!(
        pts.len() == keys.len(),
        "trade_nodes has {} nodes, expected {}",
        pts.len(),
        keys.len()
    );
    tn.set_version(1)?;
    let items = keys
        .iter()
        .zip(&pts)
        .map(|(k, p)| vec![esf::s16(k.clone()), esf::xyv(p[0], p[1])])
        .collect();
    *tn.children_mut()? = vec![Node::array("trade_nodes", 0, items)];
    Ok(())
}

// ---------------------------------------------------------------------------------------------- DB helpers

fn field_string(row: &Value, field: &str) -> Result<String> {
    match row.get(field) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Number(n)) => Ok(n.to_string()),
        other => bail!("Row field {field} is not a string: {other:?}"),
    }
}

fn cell(row: &Value, fields: &[String], i: usize) -> Result<Value> {
    let name = fields
        .get(i)
        .with_context(|| format!("Table has only {} fields, wanted #{i}", fields.len()))?;
    row.get(name)
        .cloned()
        .with_context(|| format!("Row lacks field {name}"))
}

fn rows_file(table: &str, version: u32, fields: &[String], rows: Vec<Vec<Value>>) -> Result<Value> {
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        ensure!(
            r.len() == fields.len(),
            "{table}: row has {} values, schema has {} fields",
            r.len(),
            fields.len()
        );
        out.push(Value::Object(fields.iter().cloned().zip(r).collect()));
    }
    Ok(json!({ "table": table, "version": version, "rows": out }))
}

fn ensure_same_fields(table: &str, etw: &[String], ntw: &[String]) -> Result<()> {
    ensure!(
        etw == ntw,
        "{table}: ETW and NTW schemas differ ({etw:?} vs {ntw:?})"
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------- build

/// Everything the ETW -> NTW map conversion derives before any file is written: the schema-converted
/// regions.esf (without the trade-node conversion), its slot report, the trade nodes and the owned rows.
struct MapLogic {
    etw: esf::Esf,
    region_keys: Vec<String>,
    report: Vec<RegionReport>,
    trade_keys: Vec<String>,
    node_theatre: Vec<&'static (&'static str, &'static str, &'static str, &'static str)>,
    node_pts: Vec<[f32; 2]>,
    pa_rows: Vec<Value>,
    pa_fields: Vec<String>,
    db_rows: BTreeMap<String, Value>,
    loc_rows: Vec<(String, String, u8, String)>,
}

fn convert_logic(snapshot: &Snapshot, rules: &Rules, progress: &Progress<'_>) -> Result<MapLogic> {
    // ---- regions.esf ---------------------------------------------------------------------------------
    progress("Map: converting ETW regions.esf to the NTW slot/trade-node schema");
    let mut etw = read_etw_regions(snapshot)?;
    let region_keys: Vec<String> = etw
        .root
        .path(&["region_data", "regions"])?
        .items()?
        .iter()
        .map(|i| {
            Ok(i.first()
                .context("Empty region record")?
                .as_str()?
                .to_string())
        })
        .collect::<Result<_>>()?;
    let cultures = region_cultures(rules, &region_keys);
    let heights = ntw_slot_heights(snapshot)?;
    let templates = template_models(snapshot, rules)?;
    let slot_types = rules
        .slot_types
        .as_object()
        .context("slot_types rules must be an object")?;

    // Trade nodes: 5 per sea-trade theatre in regions.esf order (index i == trade_node_i in trade_routes.esf).
    let node_pts = etw
        .root
        .get("trade_nodes")?
        .children()?
        .first()
        .context("trade_nodes has no data")?
        .xy_array()?;
    let all_theatres = theatres(&etw.root)?;
    let mut counters: HashMap<String, usize> = HashMap::new();
    let mut trade_keys = Vec::new();
    let mut node_theatre: Vec<&'static (&'static str, &'static str, &'static str, &'static str)> =
        Vec::new();
    for p in &node_pts {
        let th = theatre_of_point(&all_theatres, *p)?;
        let group = SEA_THEATRE_GROUP
            .iter()
            .find(|g| g.0 == th.key)
            .with_context(|| format!("Trade node in non sea-trade theatre {}", th.key))?;
        let n = counters.entry(th.key.clone()).or_insert(0);
        *n += 1;
        trade_keys.push(format!("{}_{:02}", group.2, n));
        node_theatre.push(group);
    }

    let report = convert_regions(&mut etw.root, slot_types, &heights, &templates, &cultures)?;

    // ---- DB rows -------------------------------------------------------------------------------------------
    let mut db_rows = BTreeMap::new();
    let (pa_version, pa_fields) = db::table_meta(
        snapshot,
        Game::Etw,
        rules,
        "campaign_map_playable_areas_tables",
    )?;
    let all_pa = db::table(
        snapshot,
        Game::Etw,
        rules,
        "campaign_map_playable_areas_tables",
    )?;
    let mut pa_rows: Vec<Value> = Vec::new();
    for theatre in THEATRES {
        let mut found = Vec::new();
        for r in &all_pa {
            if field_string(r, &pa_fields[0])? == theatre {
                found.push(r.clone());
            }
        }
        ensure!(
            found.len() == 1,
            "Expected one playable-area row for theatre {theatre}, found {}",
            found.len()
        );
        pa_rows.extend(found);
    }
    let mut pa_cells = Vec::new();
    for r in &pa_rows {
        pa_cells.push(
            (0..pa_fields.len())
                .map(|i| cell(r, &pa_fields, i))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    db_rows.insert(
        "campaign_map_playable_areas_tables".to_string(),
        rows_file(
            "campaign_map_playable_areas_tables",
            pa_version,
            &pa_fields,
            pa_cells,
        )?,
    );

    let (st_ver, st_fields) = db::table_meta(
        snapshot,
        Game::Ntw,
        rules,
        "campaign_map_settlements_tables",
    )?;
    let (sl_ver, sl_fields) =
        db::table_meta(snapshot, Game::Ntw, rules, "campaign_map_slots_tables")?;
    let (tp_ver, tp_fields) = db::table_meta(
        snapshot,
        Game::Ntw,
        rules,
        "campaign_map_towns_and_ports_tables",
    )?;
    ensure_same_fields(
        "campaign_map_settlements_tables",
        &db::table_meta(
            snapshot,
            Game::Etw,
            rules,
            "campaign_map_settlements_tables",
        )?
        .1,
        &st_fields,
    )?;
    ensure_same_fields(
        "campaign_map_slots_tables",
        &db::table_meta(snapshot, Game::Etw, rules, "campaign_map_slots_tables")?.1,
        &sl_fields,
    )?;
    ensure_same_fields(
        "campaign_map_towns_and_ports_tables",
        &db::table_meta(
            snapshot,
            Game::Etw,
            rules,
            "campaign_map_towns_and_ports_tables",
        )?
        .1,
        &tp_fields,
    )?;

    let by_first = |table: &str, fields: &[String]| -> Result<HashMap<String, Value>> {
        db::table(snapshot, Game::Etw, rules, table)?
            .into_iter()
            .map(|r| Ok((field_string(&r, &fields[0])?, r)))
            .collect()
    };
    let st_by_id = by_first("campaign_map_settlements_tables", &st_fields)?;
    let sl_by_id = by_first("campaign_map_slots_tables", &sl_fields)?;
    let tp_by_id = by_first("campaign_map_towns_and_ports_tables", &tp_fields)?;

    let mut new_slots: BTreeMap<&str, (&str, &str)> = BTreeMap::new();
    for rep in &report {
        for e in &rep.slots {
            if let Some((key, ty)) = &e.new {
                new_slots.insert(key.as_str(), (rep.region.as_str(), ty.as_str()));
            }
        }
    }
    let mut settlements: Vec<Vec<Value>> = Vec::new();
    for rep in &report {
        let r = st_by_id
            .get(&rep.settlement_id)
            .with_context(|| format!("No ETW settlement row {}", rep.settlement_id))?;
        ensure!(
            field_string(r, &st_fields[1])? == rep.region,
            "Settlement {} belongs to another region",
            rep.settlement_id
        );
        settlements.push(vec![
            json!(rep.settlement_id),
            json!(rep.region),
            cell(r, &st_fields, 2)?,
            json!(rep.general_slots),
            json!("settlement"),
        ]);
    }
    let (mut slots, mut towns): (Vec<Vec<Value>>, Vec<Vec<Value>>) = (Vec::new(), Vec::new());
    for (key, (region, ty)) in &new_slots {
        if key.starts_with("settlement:") {
            continue;
        }
        if ty.starts_with("town") || *ty == "port" {
            let r = tp_by_id
                .get(*key)
                .with_context(|| format!("No ETW towns_and_ports row {key}"))?;
            towns.push(vec![json!(key), json!(ty), cell(r, &tp_fields, 2)?]);
        } else {
            let r = sl_by_id
                .get(*key)
                .with_context(|| format!("No ETW slots row {key}"))?;
            slots.push(vec![
                json!(key),
                json!(region),
                json!(ty),
                cell(r, &sl_fields, 3)?,
                cell(r, &sl_fields, 4)?,
            ]);
        }
    }
    let key_of = |r: &Vec<Value>| r[0].as_str().unwrap_or_default().to_string();
    settlements.sort_by_key(key_of);
    slots.sort_by_key(key_of);
    towns.sort_by_key(key_of);
    progress(&format!(
        "Map: DB rows settlements {}, slots {}, towns/ports {}",
        settlements.len(),
        slots.len(),
        towns.len()
    ));
    let town_loc: Vec<(String, String)> = towns.iter().map(|r| (key_of(r), key_of(r))).collect();
    let slot_loc: Vec<String> = slots.iter().map(key_of).collect();
    db_rows.insert(
        "campaign_map_settlements_tables".into(),
        rows_file(
            "campaign_map_settlements_tables",
            st_ver,
            &st_fields,
            settlements,
        )?,
    );
    db_rows.insert(
        "campaign_map_slots_tables".into(),
        rows_file("campaign_map_slots_tables", sl_ver, &sl_fields, slots)?,
    );
    db_rows.insert(
        "campaign_map_towns_and_ports_tables".into(),
        rows_file(
            "campaign_map_towns_and_ports_tables",
            tp_ver,
            &tp_fields,
            towns,
        )?,
    );

    // ---- trade nodes -------------------------------------------------------------------------------------------
    let (_, ttc_fields) = db::table_meta(
        snapshot,
        Game::Etw,
        rules,
        "trade_theatre_commodities_tables",
    )?;
    let mut ttc: HashMap<String, Value> = HashMap::new();
    for r in db::table(
        snapshot,
        Game::Etw,
        rules,
        "trade_theatre_commodities_tables",
    )? {
        ttc.insert(field_string(&r, &ttc_fields[0])?, r);
    }
    let (tn_ver, tn_fields) = db::table_meta(snapshot, Game::Ntw, rules, "trade_nodes_tables")?;
    let (tg_ver, tg_fields) =
        db::table_meta(snapshot, Game::Ntw, rules, "trade_node_groups_tables")?;
    ensure!(
        tn_fields.len() == 6 && tg_fields.len() == 1,
        "Unexpected NTW trade node table shapes"
    );
    let mut tn_rows = Vec::new();
    for (key, group) in trade_keys.iter().zip(&node_theatre) {
        let theatre = SEA_THEATRE_GROUP
            .iter()
            .find(|g| g.1 == group.1)
            .map(|g| g.0)
            .context("Unknown trade node group")?;
        let r = ttc
            .get(theatre)
            .with_context(|| format!("No trade_theatre_commodities row for theatre {theatre}"))?;
        tn_rows.push(vec![
            json!(key),
            cell(r, &ttc_fields, 1)?,
            cell(r, &ttc_fields, 2)?,
            cell(r, &ttc_fields, 3)?,
            cell(r, &ttc_fields, 4)?,
            json!(group.1),
        ]);
    }
    let groups: BTreeSet<&str> = node_theatre.iter().map(|g| g.1).collect();
    db_rows.insert(
        "trade_nodes_tables".into(),
        rows_file("trade_nodes_tables", tn_ver, &tn_fields, tn_rows)?,
    );
    db_rows.insert(
        "trade_node_groups_tables".into(),
        rows_file(
            "trade_node_groups_tables",
            tg_ver,
            &tg_fields,
            groups.iter().map(|g| vec![json!(g)]).collect(),
        )?,
    );

    // ---- localisation --------------------------------------------------------------------------------------------
    let etw_loc = db::parse_loc(&snapshot.read(Game::Etw, "text/localisation.loc")?)?;
    let mut loc_rows: Vec<(String, String, u8, String)> = Vec::new();
    let mut add = |key: String, text: Option<String>| -> Result<()> {
        let (text, tooltip) = match text {
            Some(t) => (t, 0u8),
            None => etw_loc
                .get(&key)
                .cloned()
                .with_context(|| format!("No ETW loc text for {key}"))?,
        };
        ensure!(
            !text.contains(['\t', '\n', '\r']),
            "Loc text of {key} contains a control character"
        );
        loc_rows.push((key, text, tooltip, "localisation".to_string()));
        Ok(())
    };
    for theatre in THEATRES {
        add(
            format!("campaign_map_playable_areas_onscreen_name_{theatre}"),
            None,
        )?;
    }
    let transition_ids: BTreeSet<u32> = all_theatres
        .iter()
        .flat_map(|t| t.transition_ids.iter().copied())
        .collect();
    for id in &transition_ids {
        add(
            format!("campaign_map_transition_areas_onscreen_name_{id}"),
            None,
        )?;
    }
    for (key, _) in &town_loc {
        add(
            format!("campaign_map_towns_and_ports_onscreen_name_{key}"),
            None,
        )?;
    }
    for key in &slot_loc {
        add(format!("campaign_map_slots_onscreen_{key}"), None)?;
    }
    let mut counters: HashMap<&str, usize> = HashMap::new();
    for (key, group) in trade_keys.iter().zip(&node_theatre) {
        let n = counters.entry(group.1).or_insert(0);
        *n += 1;
        add(
            format!("trade_nodes_display_name_{key}"),
            Some(format!("{} Trade Route {}", group.3, n)),
        )?;
    }
    for g in &groups {
        let name = SEA_THEATRE_GROUP
            .iter()
            .find(|v| v.1 == *g)
            .map(|v| v.3)
            .context("Unknown trade node group")?;
        add(
            format!("trade_node_groups_display_name_{g}"),
            Some(name.to_string()),
        )?;
    }
    progress(&format!("Map: {} localisation rows", loc_rows.len()));
    Ok(MapLogic {
        etw,
        region_keys,
        report,
        trade_keys,
        node_theatre,
        node_pts,
        pa_rows,
        pa_fields,
        db_rows,
        loc_rows,
    })
}

/// The database rows and localisation rows the converted map owns (`MapArtifacts.db_rows` / `loc_rows`),
/// computed from the source snapshot and the rules alone: no files are written and no terrain is painted.
/// The rows are exactly those `build` returns.
pub(crate) fn logic_rows(
    snapshot: &Snapshot,
    rules: &Rules,
) -> Result<(BTreeMap<String, Value>, Vec<(String, String, u8, String)>)> {
    let logic = convert_logic(snapshot, rules, &|_: &str| {})?;
    Ok((logic.db_rows, logic.loc_rows))
}

pub fn build(
    snapshot: &Snapshot,
    rules: &Rules,
    output_data: &Path,
    relief: &TerrainRelief,
    extend_america: bool,
    progress: &Progress<'_>,
) -> Result<MapArtifacts> {
    let map_dir = output_data.join("campaign_maps").join("nap_world");
    fs::create_dir_all(&map_dir).with_context(|| format!("Create {}", map_dir.display()))?;
    let MapLogic {
        mut etw,
        region_keys,
        report,
        trade_keys,
        node_theatre,
        node_pts,
        pa_rows,
        pa_fields,
        db_rows,
        loc_rows,
    } = convert_logic(snapshot, rules, progress)?;

    convert_trade_nodes(&mut etw.root, &trade_keys)?;
    let regions_esf = esf::write(&etw)?;
    write_file(&map_dir.join("regions.esf"), &regions_esf)?;
    progress(&format!(
        "Map: regions.esf {} bytes, {} settlement regions",
        regions_esf.len(),
        report.len()
    ));

    // ---- verbatim AI / pathfinding ESFs ------------------------------------------------------------------
    for name in ["pathfinding", "poi", "sea_grids", "trade_routes"] {
        let bytes = snapshot.read(Game::Etw, &format!("{ETW_MAP}/{name}.esf"))?;
        write_file(&map_dir.join(format!("{name}.esf")), &bytes)?;
    }

    // ---- theatre TGAs, metadata and display tree ---------------------------------------------------------------------
    let art_cache = rules.root.parent().context("Rules have no local asset root")?.join("terrain");
    let tgas = terrain::build_display(
        snapshot,
        &map_dir,
        &region_keys.iter().cloned().collect(),
        relief,
        &art_cache,
        extend_america,
        progress,
    )?;
    for r in &pa_rows {
        for f in [2usize, 3, 4] {
            let name = field_string(r, &pa_fields[f])?;
            ensure!(
                tgas.iter().any(|t| t.eq_ignore_ascii_case(&name)),
                "Playable area references {name}, which is not in ETW global_map"
            );
        }
    }
    terrain::build_metadata(snapshot, &map_dir, progress)?;

    // ---- mapping contracts ----------------------------------------------------------------------------------------------
    let mut regions = Map::new();
    for rep in &report {
        let slots: Vec<Value> = rep
            .slots
            .iter()
            .map(|e| match &e.new {
                Some((k, t)) => {
                    json!({"etw_key": e.etw_key, "key": k, "type": t, "dropped": false})
                }
                None => json!({"etw_key": e.etw_key, "key": null, "type": null, "dropped": true}),
            })
            .collect();
        regions.insert(
            rep.region.clone(),
            json!({"settlement_id": rep.settlement_id, "settlement_slot_type": rep.settlement_slot_type, "general_slots": rep.general_slots, "slots": slots}),
        );
    }
    let slot_mapping = json!({
        "_doc": "region -> settlement slot keys exactly as written to regions.esf slot_descriptions. Startpos slot ids for settlement slots MUST use these keys. 'dropped' ETW slots do not exist in the new map.",
        "regions": regions,
    });
    let nodes: Vec<Value> = trade_keys
        .iter()
        .zip(&node_theatre)
        .zip(&node_pts)
        .enumerate()
        .map(|(i, ((key, group), p))| {
            let theatre = SEA_THEATRE_GROUP.iter().find(|g| g.1 == group.1).map(|g| g.0).unwrap_or_default();
            json!({"index": i, "key": key, "theatre": theatre, "pos": [p[0], p[1]], "group": group.1})
        })
        .collect();
    let trade_nodes = json!({
        "_doc": "index i == regions.esf trade_nodes[i] == trade_routes.esf TRADE_NODE_ROUTES 'trade_node_i'",
        "nodes": nodes,
    });
    progress("Map: ETW world map converted");
    Ok(MapArtifacts {
        slot_mapping,
        trade_nodes,
        db_rows,
        loc_rows,
    })
}

fn write_file(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, data).with_context(|| format!("Write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Napoleon's 0x909d00 prism builder uses (dz, -dx) for template-backed
    // polygons, and negates it for legacy slots. 0x8fca30 accepts n.p + d <= 0.
    fn pick_contains(slot: &[Node], polygon: usize, point: [f32; 2]) -> bool {
        let points = slot[polygon].xy_array().unwrap();
        if points.len() < 3 {
            return false;
        }
        let sign = if slot[6].as_str().unwrap().is_empty() { -1.0 } else { 1.0 };
        points.iter().zip(points.iter().cycle().skip(1)).all(|(a, b)| {
            let normal = [(b[1] - a[1]) * sign, (a[0] - b[0]) * sign];
            normal[0] * (point[0] - a[0]) + normal[1] * (point[1] - a[1]) <= 0.0
        })
    }

    fn etw_slot(key: &str, kind: &str, sea: bool) -> Vec<Node> {
        vec![
            esf::s16(key),
            esf::s16(kind),
            esf::xyv(10.0, -2.0),
            esf::xyv(10.0, 0.0),
            Node::Value(esf::Value::Angle(0)),
            Node::Value(esf::Value::Bool(false)),
            esf::arr_xy(&[[9.0, -3.0], [9.0, -1.0], [11.0, -1.0], [11.0, -3.0]]),
            esf::arr_xy(if sea { &[[9.0, -1.0], [9.0, 1.0], [11.0, 1.0], [11.0, -1.0]] } else { &[] }),
            esf::arr_xy(&[[9.0, -3.0], [9.0, 1.0], [11.0, 1.0], [11.0, -3.0]]),
        ]
    }

    #[test]
    fn converted_ports_and_towns_have_pickable_prisms_without_breaking_legacy_slots() {
        let mut root = Node::record("root", 0, vec![
            Node::record("region_data", 0, vec![
                Node::array("regions", 0, vec![vec![
                    esf::s16("coast"),
                    Node::record("settlement_and_slots", 0, vec![
                        Node::array("slot_descriptions", 0, vec![
                            etw_slot("settlement:coast:capital:settlement_road", "settlement_road", false),
                            etw_slot("port:coast:harbour", "port", true),
                            etw_slot("town:coast:works", "town-metal", false),
                            etw_slot("wheat:coast:farm", "wheat", false),
                        ]),
                    ]),
                ]]),
            ]),
        ]);
        let types = json!({
            "port:coast:harbour": "port",
            "town:coast:works": "town-industrial",
            "wheat:coast:farm": "wheat",
        });
        let templates = HashMap::from([
            (("european".into(), "port".into()), "nap_eur_port".into()),
            (("european".into(), "town-industrial".into()), "town_model".into()),
        ]);
        convert_regions(
            &mut root,
            types.as_object().unwrap(),
            &BTreeMap::new(),
            &templates,
            &HashMap::from([("coast".into(), "european".into())]),
        ).unwrap();
        let region = &root.path(&["region_data", "regions"]).unwrap().items().unwrap()[0];
        let slots = slot_array(region).unwrap().items().unwrap();
        for slot in slots {
            assert!(pick_contains(slot, 7, [10.0, -2.0]), "land footprint: {}", slot[0].as_str().unwrap());
            assert!(pick_contains(slot, 9, [10.0, -2.0]), "selection hull: {}", slot[0].as_str().unwrap());
            assert!(!pick_contains(slot, 9, [20.0, -2.0]));
        }
        let port = &slots[1];
        assert!(pick_contains(port, 8, [10.0, 0.0]));
        assert!(pick_contains(port, 9, [10.0, 0.0]));
        assert!(!pick_contains(port, 7, [10.0, 0.0]));
        assert!(!pick_contains(port, 8, [10.0, -2.0]));
        assert!(!pick_contains(&slots[2], 8, [10.0, 0.0]));
    }
}
