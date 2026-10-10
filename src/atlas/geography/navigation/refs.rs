//! Cross-file references of the navigation tables into the (migrated) region data.
//!
//! The migration keeps the region list: same count, same order, same keys (so the pathfinding region indices, the
//! sea-grid region lists, the trade-route keys and the POI region names stay valid). POI area ids are the region
//! areas' adjoining-passable-area ids, which the geography migration keeps; here we prove that every (region, id) pair
//! that exists in the source regions still exists in the new regions.

use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value as Json, json};

use super::ctx::References;
use super::util::node_u16;
use crate::esf::{Esf, Node};

/// Region keys in file order and the adjoining ids of their areas.
pub(super) struct RegionTable {
    pub keys: Vec<String>,
    pub adjoining: HashMap<String, BTreeSet<u16>>,
}

pub(super) struct RegionRefs {
    pub source: RegionTable,
    pub new: RegionTable,
    pub source_strings: HashSet<String>,
    pub new_strings: HashSet<String>,
}

/// Every string value of the tree (Python `string_set`).
pub(super) fn string_set(root: &Node) -> HashSet<String> {
    let mut out = HashSet::new();
    fn go(n: &Node, out: &mut HashSet<String>) {
        match n {
            Node::Value(crate::esf::Value::String16(s) | crate::esf::Value::String8(s)) => {
                out.insert(s.clone());
            }
            Node::Value(_) => {}
            Node::Record { children, .. } => children.iter().for_each(|c| go(c, out)),
            Node::Array { items, .. } => items.iter().flatten().for_each(|c| go(c, out)),
        }
    }
    go(root, &mut out);
    out
}

pub(super) fn region_table(esf: &Esf) -> Result<RegionTable> {
    let mut keys = Vec::new();
    let mut adjoining: HashMap<String, BTreeSet<u16>> = HashMap::new();
    for item in esf.root.get("region_data")?.get("regions")?.items()? {
        let key = item.first().ok_or_else(|| anyhow!("empty region item"))?.as_str()?.to_string();
        let areas = item
            .iter()
            .find(|n| matches!(n, Node::Array { name, .. } if name == "areas"))
            .ok_or_else(|| anyhow!("region {key} has no areas array"))?
            .items()?;
        let ids = adjoining.entry(key.clone()).or_default();
        for a in areas {
            ids.insert(node_u16(a.get(5).ok_or_else(|| anyhow!("region {key}: area without adjoining id"))?)?);
        }
        keys.push(key);
    }
    Ok(RegionTable { keys, adjoining })
}

impl RegionRefs {
    /// Parses both region files and asserts the contract (`region_data` of regions-index-contract.json): region count
    /// and order preserved, keys and indices identical.
    pub fn load(source_regions: &Esf, new_regions: &Esf, contract: &Json) -> Result<Self> {
        let source = region_table(source_regions).context("source regions.esf")?;
        let new = region_table(new_regions).context("new regions.esf")?;
        let rd = contract.get("region_data").ok_or_else(|| anyhow!("regions contract has no region_data"))?;
        let count = |k: &str| -> Result<usize> {
            rd.get(k)
                .and_then(Json::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| anyhow!("regions contract region_data.{k} is missing"))
        };
        let (src_n, out_n) = (count("source_region_count")?, count("output_region_count")?);
        if src_n != out_n || src_n != source.keys.len() || out_n != new.keys.len() {
            bail!(
                "region count changed: contract {src_n} -> {out_n}, source regions.esf {}, new regions.esf {}",
                source.keys.len(),
                new.keys.len()
            );
        }
        if rd.get("region_order_preserved").and_then(Json::as_bool) != Some(true) {
            bail!("regions contract does not state region_order_preserved");
        }
        if source.keys != new.keys {
            let at = source.keys.iter().zip(&new.keys).position(|(a, b)| a != b).unwrap_or(0);
            bail!("region keys/order differ between source and new regions.esf (first difference at index {at})");
        }
        let entries = rd
            .get("regions")
            .and_then(Json::as_array)
            .ok_or_else(|| anyhow!("regions contract has no regions list"))?;
        if entries.len() != src_n {
            bail!("regions contract lists {} regions, expected {src_n}", entries.len());
        }
        for (i, e) in entries.iter().enumerate() {
            let key = e.get("key").and_then(Json::as_str);
            let idx = |k: &str| e.get(k).and_then(Json::as_u64);
            if key != Some(source.keys[i].as_str()) || idx("source_index") != Some(i as u64) || idx("output_index") != Some(i as u64) {
                bail!("regions contract entry {i} ({key:?}) does not match region {i} {:?} of the region files", source.keys[i]);
            }
        }
        Ok(Self {
            source_strings: string_set(&source_regions.root),
            new_strings: string_set(&new_regions.root),
            source,
            new,
        })
    }

    fn knows(&self, key: &str) -> bool {
        self.source.adjoining.contains_key(key)
    }

    /// Region part of a `port:<region>:<name>` key.
    fn port_region(port: &str) -> Option<&str> {
        let mut parts = port.splitn(3, ':');
        match (parts.next(), parts.next(), parts.next()) {
            (Some("port"), Some(region), Some(name)) if !region.is_empty() && !name.is_empty() => Some(region),
            _ => None,
        }
    }

    fn unknown<'a>(&self, names: impl Iterator<Item = &'a String>) -> Vec<String> {
        names.filter(|n| !self.knows(n)).cloned().collect()
    }

    /// Validates everything the tables referenced. Returns `(report, failed gate names)`.
    pub fn check(&self, refs: &References, path_ids: Option<(i16, i16)>) -> (Json, Vec<String>) {
        let mut failed = Vec::new();
        // ---- the POC's POI region-name gate
        let not_in_source: Vec<&String> = refs.poi_regions.iter().filter(|n| !self.source_strings.contains(*n)).collect();
        let lost: Vec<&String> = refs
            .poi_regions
            .iter()
            .filter(|n| self.source_strings.contains(*n) && !self.new_strings.contains(*n))
            .collect();
        if !lost.is_empty() {
            failed.push("poi_region_names".to_string());
        }
        // ---- every referenced region key exists in the contract
        // (names unknown to the region list: see `unknown`)
        let poi_names: BTreeSet<&String> = refs.poi_regions.iter().chain(refs.poi_areas.iter().map(|(r, _)| r)).collect();
        let unknown_poi = self.unknown(poi_names.iter().copied());
        let unknown_sea = self.unknown(refs.sea_regions.iter());
        let ports_unknown = |set: &BTreeSet<String>| -> Vec<String> {
            set.iter()
                .filter(|p| !Self::port_region(p).is_some_and(|r| self.knows(r)))
                .cloned()
                .collect()
        };
        let unknown_sea_ports = ports_unknown(&refs.sea_ports);
        let unknown_trade_ports = ports_unknown(&refs.trade_ports);
        let unknown_settlements = self.unknown(refs.trade_settlements.iter());
        if [&unknown_poi, &unknown_sea, &unknown_sea_ports, &unknown_trade_ports, &unknown_settlements]
            .iter()
            .any(|v| !v.is_empty())
        {
            failed.push("region_references".to_string());
        }
        // ---- pathfinding region ids are region indices
        let count = self.source.keys.len() as i64;
        let ids_ok = path_ids.is_none_or(|(lo, hi)| lo >= 0 && (hi as i64) < count);
        if !ids_ok {
            failed.push("pathfinding_region_ids".to_string());
        }
        // ---- POI area ids (adjoining ids) survive
        let (mut in_source, mut lost_pairs, mut not_in_source_pairs) = (0usize, Vec::new(), Vec::new());
        for (region, id) in &refs.poi_areas {
            let Ok(id16) = u16::try_from(*id) else {
                not_in_source_pairs.push(json!([region, id]));
                continue;
            };
            if self.source.adjoining.get(region).is_some_and(|s| s.contains(&id16)) {
                in_source += 1;
                if !self.new.adjoining.get(region).is_some_and(|s| s.contains(&id16)) {
                    lost_pairs.push(json!([region, id]));
                }
            } else {
                not_in_source_pairs.push(json!([region, id]));
            }
        }
        if !lost_pairs.is_empty() {
            failed.push("poi_area_ids".to_string());
        }
        let report = json!({
            "poi_region_names": refs.poi_regions.len(),
            "not_in_source_regions": not_in_source,
            "lost_in_new_regions": lost,
            "contract": {
                "source_region_count": self.source.keys.len(),
                "output_region_count": self.new.keys.len(),
                "region_order_preserved": true,
                "keys_identical_in_source_and_new_regions": true,
            },
            "unknown_region_keys": {
                "poi": unknown_poi, "sea_grid_region_lists": unknown_sea, "sea_grid_ports": unknown_sea_ports,
                "trade_ports": unknown_trade_ports, "trade_settlements": unknown_settlements,
            },
            "pathfinding_region_ids": {
                "min": path_ids.map(|p| p.0), "max": path_ids.map(|p| p.1), "region_count": count, "ok": ids_ok,
            },
            "poi_area_ids": {
                "pairs": refs.poi_areas.len(), "present_in_source": in_source, "lost_in_new_regions": lost_pairs,
                "not_in_source_regions": not_in_source_pairs,
                "note": "POI area ids are the regions' adjoining-passable-area ids and are not rewritten; the migration keeps them",
            },
        });
        (report, failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf;

    fn area(adjoining: u16) -> Vec<Node> {
        vec![
            esf::b(false),
            esf::b(false),
            esf::b(true),
            esf::xyv(0.0, 0.0),
            esf::xyv(1.0, 1.0),
            esf::u16v(adjoining),
        ]
    }

    fn region(key: &str, ids: &[u16]) -> Vec<Node> {
        vec![
            esf::s16(key),
            esf::s8("land"),
            esf::xyv(0.0, 0.0),
            esf::xyv(1.0, 1.0),
            Node::array("areas", 0, ids.iter().map(|&i| area(i)).collect()),
            esf::i32v(0),
        ]
    }

    fn regions_esf(regions: Vec<Vec<Node>>) -> Esf {
        Esf {
            timestamp: 0,
            names: vec![],
            root: Node::record("root", 0, vec![Node::record("region_data", 0, vec![Node::array("regions", 0, regions)])]),
        }
    }

    fn contract(keys: &[&str]) -> Json {
        json!({"region_data": {
            "source_region_count": keys.len(), "output_region_count": keys.len(), "region_order_preserved": true,
            "regions": keys.iter().enumerate().map(|(i, k)| json!({"key": k, "source_index": i, "output_index": i})).collect::<Vec<_>>(),
        }})
    }

    fn refs() -> References {
        let mut r = References::default();
        r.poi_regions.insert("france".into());
        r.poi_areas.insert(("france".into(), 13));
        r.sea_regions.insert("atlantic".into());
        r.sea_ports.insert("port:france:brest".into());
        r.trade_ports.insert("port:france:brest".into());
        r.trade_settlements.insert("france".into());
        r
    }

    #[test]
    fn consistent_regions_and_references_pass() {
        let src = regions_esf(vec![region("france", &[13, 14]), region("atlantic", &[1])]);
        let new = regions_esf(vec![region("france", &[13, 14, 14]), region("atlantic", &[1])]);
        let rr = RegionRefs::load(&src, &new, &contract(&["france", "atlantic"])).unwrap();
        let (report, failed) = rr.check(&refs(), Some((0, 1)));
        assert!(failed.is_empty(), "{report}");
        assert_eq!(report["poi_area_ids"]["present_in_source"], json!(1));
    }

    #[test]
    fn contract_mismatches_are_errors() {
        let src = regions_esf(vec![region("france", &[13]), region("atlantic", &[1])]);
        let swapped = regions_esf(vec![region("atlantic", &[1]), region("france", &[13])]);
        assert!(RegionRefs::load(&src, &swapped, &contract(&["france", "atlantic"])).is_err(), "order changed");
        let fewer = regions_esf(vec![region("france", &[13])]);
        assert!(RegionRefs::load(&src, &fewer, &contract(&["france", "atlantic"])).is_err(), "count changed");
        assert!(RegionRefs::load(&src, &src, &contract(&["france", "spain"])).is_err(), "contract keys differ");
        let mut c = contract(&["france", "atlantic"]);
        c["region_data"]["region_order_preserved"] = json!(false);
        assert!(RegionRefs::load(&src, &src, &c).is_err());
    }

    #[test]
    fn lost_area_id_unknown_keys_and_bad_region_indices_are_reported() {
        let src = regions_esf(vec![region("france", &[13]), region("atlantic", &[1])]);
        let new = regions_esf(vec![region("france", &[99]), region("atlantic", &[1])]);
        let rr = RegionRefs::load(&src, &new, &contract(&["france", "atlantic"])).unwrap();
        let (_, failed) = rr.check(&refs(), Some((0, 1)));
        assert_eq!(failed, vec!["poi_area_ids".to_string()]);
        let mut r = refs();
        r.trade_ports.insert("port:atlantis:x".into());
        r.sea_regions.insert("nowhere".into());
        let (_, failed) = rr.check(&r, Some((0, 2)));
        assert!(failed.contains(&"region_references".to_string()) && failed.contains(&"pathfinding_region_ids".to_string()));
        let mut r = refs();
        r.poi_areas.insert(("france".into(), 70_000));
        let (report, failed) = rr.check(&r, None);
        assert_eq!(failed, vec!["poi_area_ids".to_string()]);
        assert_eq!(report["poi_area_ids"]["not_in_source_regions"], json!([["france", 70000]]));
    }
}
