//! Camera bound synchronisation (accepted POC `camera_sync.py`): only the three stored copies of every theatre
//! rectangle (regions `theatres[]`, regions `theatres_and_region_keys[]`, sea_grids theatre min/max) are snapped to
//! the 1/4096 grid the stock camera polygon builder and pan limiters compare against. No other point is quantised.

use crate::atlas::Plan;
use crate::esf::{self, Node};
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub const QUANTUM: f64 = 1.0 / 4096.0;

/// `(before, after)` stored f32 corners of one theatre rectangle: the plain atlas transform and its Q12 snap.
pub fn camera_corners(plan: &Plan, theatre: &str) -> Result<([[f32; 2]; 2], [[f32; 2]; 2])> {
    let chart = plan.chart(theatre)?;
    let rect = chart.theatre_rects.get(theatre).ok_or_else(|| anyhow::anyhow!("atlas chart {} has no theatre {theatre}", chart.key))?;
    let mut before = [[0f32; 2]; 2];
    let mut after = [[0f32; 2]; 2];
    for (k, pt) in rect.iter().enumerate() {
        let moved = plan.xy(*pt, chart);
        for a in 0..2 {
            let b = moved[a] as f32;
            before[k][a] = b;
            after[k][a] = ((b as f64 / QUANTUM).round_ties_even() * QUANTUM) as f32;
        }
    }
    let snapped = plan.camera_rect(chart, theatre)?;
    ensure!(snapped == after, "camera snap of theatre {theatre} disagrees with Plan::camera_rect: {after:?} != {snapped:?}");
    Ok((before, after))
}

/// Snap the theatre rectangles of `regions.esf` (both stored copies) in place.
pub fn snap_regions(plan: &Plan, root: &mut Node) -> Result<Value> {
    let mut report = serde_json::Map::new();
    let mut max_change = 0f64;
    let expected: BTreeSet<String> = plan.charts.iter().flat_map(|c| c.theatres.iter().cloned()).collect();
    for array in ["theatres_and_region_keys", "theatres"] {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let items = root.get_mut(array)?.items_mut()?;
        for item in items.iter_mut() {
            let Some(Node::Record { children, .. }) = item.iter_mut().find(|n| n.name() == Some("theatre")) else {
                bail!("{array}: item without a theatre record");
            };
            let key = children.first().ok_or_else(|| anyhow::anyhow!("{array}: empty theatre record"))?.as_str()?.to_string();
            let (before, after) = camera_corners(plan, &key)?;
            for k in 0..2 {
                let old = children[1 + k].as_xy()?;
                ensure!(
                    old == before[k],
                    "{array}: theatre {key} bounds {old:?} differ from the verified atlas transform {:?}",
                    before[k]
                );
                for a in 0..2 {
                    max_change = max_change.max((after[k][a] as f64 - before[k][a] as f64).abs());
                }
                children[1 + k] = esf::xyv(after[k][0], after[k][1]);
            }
            ensure!(seen.insert(key.clone()), "{array}: theatre {key} listed twice");
        }
        ensure!(seen == expected, "{array}: theatre coverage {seen:?} differs from the atlas plan {expected:?}");
        report.insert(array.to_string(), json!({"xy_fields": seen.len() * 2}));
    }
    report.insert("quantum".into(), json!(QUANTUM));
    report.insert("max_coordinate_change".into(), json!(max_change));
    Ok(Value::Object(report))
}
