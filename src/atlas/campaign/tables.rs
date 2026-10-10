//! Database side of the atlas migration.
//!
//! Audit of the shipped ETW/NTW schemas (every table the packs can contain): the only columns that hold campaign-world
//! coordinates are `campaign_map_playable_areas` `minx/maxx/miny/maxy` (older table versions). The version the editor
//! generates (and the accepted POC pack) has none of them; the other map-owned tables (settlements, slots, towns and
//! ports, trade nodes/groups) carry keys, slot types and names only. [`transform_members`] therefore moves the
//! playable-area rectangles when the columns exist and reports the table as coordinate-free when they do not; a
//! partial column set or an unexpected old rectangle fails closed.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value as Json, json};

use crate::atlas::Plan;
use crate::db::{self, Cell, Schema};
use crate::map::MapArtifacts;

pub const PLAYABLE_AREAS: &str = "campaign_map_playable_areas_tables";
const BOUNDS: [&str; 4] = ["minx", "miny", "maxx", "maxy"];

/// Member path `db/<table>/<member>` -> table name.
fn table_of(path: &str) -> Option<&str> {
    let mut parts = path.split(['/', '\\']);
    parts.next()?.eq_ignore_ascii_case("db").then_some(())?;
    let table = parts.next()?;
    parts.next()?;
    Some(table)
}

/// Move the world coordinates of the final encoded DB members. Members are keyed by pack path
/// (`db/<table>/<member>`); only changed members are re-encoded.
pub fn transform_members(members: &mut BTreeMap<String, Vec<u8>>, schema: &Schema, plan: &Plan) -> Result<Json> {
    plan.validate()?;
    let mut report = Vec::new();
    let paths: Vec<String> = members.keys().filter(|p| table_of(p).is_some_and(|t| t.eq_ignore_ascii_case(PLAYABLE_AREAS))).cloned().collect();
    for path in paths {
        let mut table = db::decode(PLAYABLE_AREAS, &members[&path], schema).with_context(|| format!("decode {path}"))?;
        let present: Vec<&str> = BOUNDS.iter().copied().filter(|b| table.column(b).is_ok()).collect();
        if present.is_empty() {
            report.push(json!({"member": path, "rows": table.rows.len(), "status": "no coordinate columns (unchanged)"}));
            continue;
        }
        ensure!(
            present.len() == BOUNDS.len(),
            "{path}: playable areas carry only {present:?} of {BOUNDS:?}; refusing to guess the missing world coordinates"
        );
        let index = table.column("index")?;
        let cols: Vec<usize> = BOUNDS.iter().map(|b| table.column(b)).collect::<Result<_>>()?;
        let mut moved = Vec::new();
        let mut kept = Vec::new();
        for row in &mut table.rows {
            let key = row[index].as_str().with_context(|| format!("{path}: playable area index is not text"))?.to_string();
            let Ok(chart) = plan.chart(&key) else {
                kept.push(key);
                continue;
            };
            let src = chart.theatre_rects.get(&key).with_context(|| format!("{path}: {key} is a chart key, not a theatre"))?;
            let old: Vec<f32> = cols
                .iter()
                .map(|&c| match &row[c] {
                    Cell::F32(v) => Ok(*v),
                    other => bail!("{path}: playable area {key} bound is {other:?}, expected f32"),
                })
                .collect::<Result<_>>()?;
            let expected = [src[0][0] as f32, src[0][1] as f32, src[1][0] as f32, src[1][1] as f32];
            ensure!(
                old == expected,
                "{path}: playable area of theatre {key} is {old:?} (minx, miny, maxx, maxy) but the accepted atlas source theatre is {expected:?}"
            );
            let r = plan.camera_rect(chart, &key)?;
            let new = [r[0][0], r[0][1], r[1][0], r[1][1]];
            for (&c, v) in cols.iter().zip(new) {
                row[c] = Cell::F32(v);
            }
            moved.push(json!({"theatre": key, "chart": chart.key, "source": old, "output": new}));
        }
        members.insert(path.clone(), db::encode(&table)?);
        report.push(json!({"member": path, "rows": table.rows.len(), "moved": moved, "left_unchanged_not_atlas_theatres": kept}));
    }
    Ok(json!({"watched_table": PLAYABLE_AREAS, "members": report}))
}

/// Move the sea-trade node positions recorded as map metadata (`MapArtifacts.trade_nodes`).
pub fn transform_trade_nodes(map: &mut MapArtifacts, plan: &Plan) -> Result<()> {
    plan.validate()?;
    let nodes = map.trade_nodes.get_mut("nodes").and_then(Json::as_array_mut).context("trade_nodes artifact has no node list")?;
    for node in nodes {
        let theatre = node["theatre"].as_str().context("trade node without theatre")?.to_string();
        let pos = node["pos"].as_array().filter(|a| a.len() == 2).and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?])).context("trade node without pos")?;
        let owner = plan.chart_at(pos, false).with_context(|| format!("trade node {} lies in no theatre rectangle", node["key"]))?;
        ensure!(
            plan.chart(&theatre)?.key == owner.key,
            "trade node {} belongs to theatre {theatre} but lies in chart {}",
            node["key"],
            owner.key
        );
        let moved = plan.xy(pos, owner);
        node["pos"] = json!([f64::from(moved[0] as f32), f64::from(moved[1] as f32)]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_db_members_of_a_table_directory_are_recognised() {
        assert_eq!(table_of("db/campaign_map_playable_areas_tables/x"), Some("campaign_map_playable_areas_tables"));
        assert_eq!(table_of("db\\campaign_map_playable_areas_tables\\x"), Some("campaign_map_playable_areas_tables"));
        assert_eq!(table_of("db/campaign_map_playable_areas_tables"), None);
        assert_eq!(table_of("text/localisation.loc"), None);
    }

    #[test]
    fn trade_node_positions_follow_their_theatre_chart_and_foreign_nodes_fail() {
        let plan = Plan::proven();
        let mut map = MapArtifacts {
            slot_mapping: Json::Null,
            trade_nodes: json!({"nodes": [{"index": 0, "key": "n", "theatre": "2", "pos": [100.0, 300.0], "group": "g"}]}),
            db_rows: BTreeMap::new(),
            loc_rows: Vec::new(),
        };
        transform_trade_nodes(&mut map, &plan).unwrap();
        let want = plan.xy([100.0, 300.0], plan.chart("2").unwrap());
        assert_eq!(map.trade_nodes["nodes"][0]["pos"], json!([f64::from(want[0] as f32), f64::from(want[1] as f32)]));
        map.trade_nodes["nodes"][0]["theatre"] = json!("1");
        map.trade_nodes["nodes"][0]["pos"] = json!([100.0, 300.0]);
        assert!(transform_trade_nodes(&mut map, &plan).is_err());
    }
}
