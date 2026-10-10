//! Readers of the two Geography products the campaign needs: the regions index contract (`regions-index-contract.json`
//! `region_data`) and the Q20 point overrides (`q20-coordinate-overrides.json`).

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use crate::atlas::Plan;

#[derive(Clone, Debug)]
pub struct ContractRegion {
    pub key: String,
    pub source_areas: usize,
    pub output_areas: usize,
    /// per source area: the output areas it became
    pub area_fragments: Vec<Vec<u32>>,
    /// per output area: the source area it came from
    pub area_source: Vec<Option<u32>>,
    /// output area -> interior point (world units, output space)
    pub interior: BTreeMap<u32, [f64; 2]>,
}

#[derive(Clone, Debug)]
pub struct Contract {
    pub regions: Vec<ContractRegion>,
    by_key: HashMap<String, usize>,
    /// `geometry_kind` of every region that carries one (report only)
    pub geometry_kinds: BTreeMap<String, Value>,
}

fn usize_of(v: &Value, what: &str) -> Result<usize> {
    v.as_u64().map(|n| n as usize).with_context(|| format!("regions contract: {what} is not a count"))
}

fn u32_list(v: &Value, what: &str) -> Result<Vec<u32>> {
    v.as_array()
        .with_context(|| format!("regions contract: {what} is not a list"))?
        .iter()
        .map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok()).with_context(|| format!("regions contract: {what} holds a non-index")))
        .collect()
}

impl Contract {
    pub fn parse(top: &Value) -> Result<Self> {
        ensure!(
            top.get("format").and_then(Value::as_u64) == Some(1) && top.get("region_data").is_some(),
            "unsupported regions contract (expected format 1 with a region_data section)"
        );
        let d = &top["region_data"];
        if d.get("dropped_regions").and_then(Value::as_array).is_some_and(|a| !a.is_empty()) {
            bail!("regions contract drops regions {}: CAI region deletion is not supported", d["dropped_regions"]);
        }
        let listed = d["regions"].as_array().context("regions contract has no region list")?;
        let mut regions = Vec::with_capacity(listed.len());
        let mut by_key = HashMap::new();
        let mut geometry_kinds = BTreeMap::new();
        for r in listed {
            let key = r["key"].as_str().context("regions contract: region without key")?.to_string();
            ensure!(!r["output_index"].is_null(), "region {key} has no output");
            let fragments = r["area_fragments"]
                .as_array()
                .with_context(|| format!("region {key}: no area_fragments"))?
                .iter()
                .map(|f| u32_list(f, "area_fragments"))
                .collect::<Result<Vec<_>>>()?;
            let area_source = match r.get("area_source").and_then(Value::as_array) {
                Some(list) => list.iter().map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok())).collect(),
                None => Vec::new(),
            };
            let mut interior = BTreeMap::new();
            if let Some(map) = r.get("interior_point").and_then(Value::as_object) {
                for (area, ip) in map {
                    let p = if ip.is_object() { &ip["point"] } else { ip };
                    let xy = p.as_array().filter(|a| a.len() == 2).and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?]));
                    let Some(xy) = xy else { bail!("region {key}: interior point of area {area} is malformed") };
                    interior.insert(area.parse::<u32>().with_context(|| format!("region {key}: interior point key {area}"))?, xy);
                }
            }
            if let Some(kind) = r.get("geometry_kind").filter(|k| !k.is_null()) {
                geometry_kinds.insert(key.clone(), kind.clone());
            }
            ensure!(by_key.insert(key.clone(), regions.len()).is_none(), "regions contract lists region {key} twice");
            regions.push(ContractRegion {
                key,
                source_areas: usize_of(&r["source_areas"], "source_areas")?,
                output_areas: usize_of(&r["output_areas"], "output_areas")?,
                area_fragments: fragments,
                area_source,
                interior,
            });
        }
        ensure!(
            usize_of(&d["source_region_count"], "source_region_count")? == usize_of(&d["output_region_count"], "output_region_count")?
                && regions.len() == usize_of(&d["source_region_count"], "source_region_count")?,
            "regions contract changes the region count"
        );
        ensure!(d["region_order_preserved"].as_bool() == Some(true), "regions contract does not preserve region order");
        Ok(Contract { regions, by_key, geometry_kinds })
    }

    pub fn region(&self, key: &str) -> Option<&ContractRegion> {
        self.by_key.get(key).map(|&i| &self.regions[i])
    }

    /// Output areas that replace the source areas `old`, in first-seen order without repeats.
    pub fn new_areas(&self, key: &str, old: &[u32]) -> Result<Vec<u32>> {
        let r = self.region(key).with_context(|| format!("region {key} is not in the regions contract"))?;
        let mut out: Vec<u32> = Vec::new();
        for &a in old {
            let frags = r.area_fragments.get(a as usize).with_context(|| format!("{key}: source area {a} out of contract range"))?;
            for &f in frags {
                if !out.contains(&f) {
                    out.push(f);
                }
            }
        }
        Ok(out)
    }

    pub fn interior_point(&self, key: &str, new_area: u32) -> Option<[f64; 2]> {
        self.region(key)?.interior.get(&new_area).copied()
    }
}

/// Per-chart Q20 destination overrides: `charts.<chart>.{x,y}.{"src_raw": dst_raw}`.
#[derive(Clone, Debug, Default)]
pub struct PointOverrides {
    charts: HashMap<String, [HashMap<i32, i32>; 2]>,
}

impl PointOverrides {
    pub fn parse(v: &Value, plan: &Plan) -> Result<Self> {
        ensure!(v.get("format").and_then(Value::as_u64) == Some(1), "point overrides: unsupported format");
        ensure!(
            v["source_fixed_bits"].as_u64() == Some(u64::from(plan.source_fixed_bits)) && v["fixed_bits"].as_u64() == Some(u64::from(plan.fixed_bits)),
            "point overrides do not match the plan's fixed-point precision"
        );
        ensure!(v["scale"].as_f64() == Some(plan.scale), "point overrides were built for scale {} but the plan scale is {}", v["scale"], plan.scale);
        let mut charts = HashMap::new();
        for (key, entry) in v["charts"].as_object().context("point overrides have no charts")? {
            ensure!(plan.charts.iter().any(|c| &c.key == key), "point overrides for unknown chart {key}");
            let axis = |name: &str| -> Result<HashMap<i32, i32>> {
                let mut m = HashMap::new();
                for (src, dst) in entry[name].as_object().with_context(|| format!("chart {key}: no {name} overrides"))? {
                    let s: i32 = src.parse().with_context(|| format!("chart {key}: source {src}"))?;
                    let d = dst.as_i64().and_then(|n| i32::try_from(n).ok()).with_context(|| format!("chart {key}: destination of {src}"))?;
                    m.insert(s, d);
                }
                Ok(m)
            };
            charts.insert(key.clone(), [axis("x")?, axis("y")?]);
        }
        Ok(PointOverrides { charts })
    }

    /// Destination of a source raw coordinate on `axis` (0 = x, 1 = y), if overridden.
    pub fn get(&self, chart: &str, axis: usize, raw: i32) -> Option<i32> {
        self.charts.get(chart)?[axis].get(&raw).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn contract() -> Contract {
        Contract::parse(&json!({
            "format": 1,
            "region_data": {
                "source_region_count": 1, "output_region_count": 1, "region_order_preserved": true,
                "regions": [{
                    "key": "r", "output_index": 0, "source_areas": 3, "output_areas": 4,
                    "area_fragments": [[0, 1], [1, 2], []], "area_source": [0, 0, 1, null],
                    "interior_point": {"0": {"point": [1.5, 2.5]}, "2": [3.0, 4.0]}
                }]
            }
        }))
        .unwrap()
    }

    #[test]
    fn area_lists_follow_fragments_in_first_seen_order_without_repeats() {
        let c = contract();
        assert_eq!(c.new_areas("r", &[1, 0]).unwrap(), vec![1, 2, 0]);
        assert!(c.new_areas("r", &[2]).unwrap().is_empty());
        assert!(c.new_areas("r", &[3]).is_err());
        assert_eq!(c.interior_point("r", 0), Some([1.5, 2.5]));
        assert_eq!(c.interior_point("r", 2), Some([3.0, 4.0]));
        assert_eq!(c.interior_point("r", 1), None);
    }

    #[test]
    fn dropped_regions_and_count_changes_are_rejected() {
        let bad = json!({"format": 1, "region_data": {"source_region_count": 2, "output_region_count": 1, "region_order_preserved": true, "regions": []}});
        assert!(Contract::parse(&bad).is_err());
        let dropped = json!({"format": 1, "region_data": {"dropped_regions": ["x"], "regions": []}});
        assert!(Contract::parse(&dropped).is_err());
    }

    #[test]
    fn overrides_must_match_the_plan_and_name_known_charts() {
        let plan = Plan::proven();
        let ok = json!({"format": 1, "source_fixed_bits": 20, "fixed_bits": 20, "scale": plan.scale,
                        "charts": {"america": {"x": {"-790874569": -684858748}, "y": {}}}});
        let o = PointOverrides::parse(&ok, &plan).unwrap();
        assert_eq!(o.get("america", 0, -790874569), Some(-684858748));
        assert_eq!(o.get("america", 1, -790874569), None);
        let mut wrong = ok.clone();
        wrong["scale"] = json!(1.25);
        assert!(PointOverrides::parse(&wrong, &plan).is_err());
        let mut unknown = ok;
        unknown["charts"] = json!({"atlantis": {"x": {}, "y": {}}});
        assert!(PointOverrides::parse(&unknown, &plan).is_err());
    }
}
