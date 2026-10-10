//! Ownership of the startpos fixed-point positions (Q20 raw `i32` pairs) and of the cached pathfinding polygons.
//!
//! Shared by the navigation quantizer (every campaign position joins its chart's per-axis map) and the campaign
//! converter. Port of `startpos_scale.py` `FIXED_PAIRS`, `cached_point_offsets`, `iter_campaign_fixed_points`,
//! `_geometry_chart`, `cached_grid_charts`, `iter_owned_fixed_points` and `navigation_scale.cached_polygons_by_grid`.

use anyhow::{Context, Result, anyhow, bail};

use crate::atlas::Plan;
use crate::esf::{Node, Value};

/// Records that carry fixed-point `(x, y)` pairs and the child index of the x of every pair (y follows x).
const FIXED_PAIRS: [(&str, &[usize]); 6] = [
    ("LOCOMOTABLE", &[0, 2]),
    ("REGION_SLOT", &[6]),
    ("SIEGEABLE_GARRISON_RESIDENCE", &[10]),
    ("CAI_SITUATED", &[0]),
    ("CAI_TRADING_POST", &[1]),
    ("CAI_REGION_HLCI", &[3]),
];

/// One fixed point of a startpos tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedPoint {
    /// Record name (`LOCOMOTABLE`, `REGION_SLOT`, ..., or `PATHFINDING_GRID[]` for cached polygon points).
    pub record: String,
    /// Owning chart key; `None` when the geometry is ambiguous or unmapped.
    pub chart: Option<String>,
    pub point: [i32; 2],
}

/// Short description of a node for error messages (never dumps a subtree or a packed array).
fn brief(n: &Node) -> String {
    match n {
        Node::Record { name, children, .. } => format!("record {name} with {} children", children.len()),
        Node::Array { name, items, .. } => format!("record array {name} with {} items", items.len()),
        Node::Value(Value::Array { code, data }) => format!("packed array {code:#x} ({} bytes)", data.len()),
        Node::Value(v) => format!("{v:?}"),
    }
}

fn fixed_i32(n: &Node) -> Result<i32> {
    match n {
        Node::Value(Value::I32(v)) => Ok(*v),
        other => bail!("expected i32 scalar, found {}", brief(other)),
    }
}

/// Every polygon of one `PATHFINDING_GRID[]` item `[u32 polygon count, u32[] payload, ...]`: per polygon `u32 n`,
/// `n` pairs of i32, one u32 (skipped). The payload must be consumed exactly.
fn grid_polygons(grid: &[Node]) -> Result<Vec<Vec<[i32; 2]>>> {
    let count = match grid.first() {
        Some(Node::Value(Value::U32(v))) => *v as usize,
        Some(other) => bail!("cached grid: expected u32 polygon count, found {}", brief(other)),
        None => bail!("cached grid: empty item"),
    };
    let raw = match grid.get(1) {
        Some(Node::Value(Value::Array { code: 0x48, data })) => data.as_slice(),
        _ => bail!("unexpected cached polygon payload"),
    };
    let mut cursor = 0usize;
    let mut polygons = Vec::with_capacity(count.min(raw.len() / 8 + 1));
    for _ in 0..count {
        let head = raw
            .get(cursor..cursor + 4)
            .ok_or_else(|| anyhow!("truncated cached polygon"))?;
        let n = u32::from_le_bytes(head.try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + n * 8 + 4 > raw.len() {
            bail!("truncated cached polygon");
        }
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let at = cursor + 8 * i;
            pts.push([
                i32::from_le_bytes(raw[at..at + 4].try_into().unwrap()),
                i32::from_le_bytes(raw[at + 4..at + 8].try_into().unwrap()),
            ]);
        }
        cursor += n * 8 + 4;
        polygons.push(pts);
    }
    if cursor != raw.len() {
        bail!("unconsumed cached polygon payload");
    }
    Ok(polygons)
}

/// Every `PATHFINDING_GRID[]` record array of the tree, in file order.
fn grid_arrays(root: &Node) -> Vec<&Node> {
    let mut out = Vec::new();
    root.walk(&mut |n, _| {
        if matches!(n, Node::Array { name, .. } if name == "PATHFINDING_GRID") {
            out.push(n);
        }
    });
    out
}

/// `(grid index, Q20 points)` of every cached `PATHFINDING_GRID[]` polygon of a startpos root, in file order.
/// The grid index counts the items of each `PATHFINDING_GRID[]` array separately.
pub fn cached_polygons_by_grid(root: &Node) -> Result<Vec<(usize, Vec<[i32; 2]>)>> {
    let mut out = Vec::new();
    for array in grid_arrays(root) {
        for (gi, grid) in array.items()?.iter().enumerate() {
            for pts in grid_polygons(grid).with_context(|| format!("cached grid {gi}"))? {
                out.push((gi, pts));
            }
        }
    }
    Ok(out)
}

fn world_point(plan: &Plan, p: [i32; 2]) -> [f64; 2] {
    let unit = (1u64 << plan.source_fixed_bits) as f64;
    [p[0] as f64 / unit, p[1] as f64 / unit]
}

/// Chart key owning a source Q20 point by geometry (theatre rect, else crop); `None` if unmapped or ambiguous.
fn geometry_chart(plan: &Plan, p: [i32; 2]) -> Option<String> {
    let pt = world_point(plan, p);
    let mut cands: Vec<&str> = plan
        .charts
        .iter()
        .filter(|c| c.theatre_rects.values().any(|r| Plan::contains(*r, pt)))
        .map(|c| c.key.as_str())
        .collect();
    if cands.is_empty() {
        cands = plan
            .charts
            .iter()
            .filter(|c| c.crop_rects.iter().any(|r| Plan::contains(*r, pt)))
            .map(|c| c.key.as_str())
            .collect();
    }
    (cands.len() == 1).then(|| cands[0].to_string())
}

/// Owning chart key of every `PATHFINDING_GRID[]` item (cached polygons), by index.
///
/// Evidence rule: all points of a grid lie inside the crops of exactly one chart (ties resolved by the chart whose
/// theatre rects contain every point). Side-effect free; used by the atlas startpos writer and by navigation.
pub fn cached_grid_charts(root: &Node, plan: &Plan) -> Result<Vec<String>> {
    let arrays = grid_arrays(root);
    if arrays.len() != 1 {
        bail!("expected exactly one PATHFINDING_GRID[] array, found {}", arrays.len());
    }
    let mut result = Vec::new();
    for (gi, grid) in arrays[0].items()?.iter().enumerate() {
        let pts: Vec<[f64; 2]> = grid_polygons(grid)
            .with_context(|| format!("cached grid {gi}"))?
            .into_iter()
            .flatten()
            .map(|p| world_point(plan, p))
            .collect();
        if pts.is_empty() {
            bail!("cached grid {gi} has no points");
        }
        let mut cands: Vec<&crate::atlas::Chart> = plan
            .charts
            .iter()
            .filter(|c| pts.iter().all(|p| c.crop_rects.iter().any(|r| Plan::contains(*r, *p))))
            .collect();
        if cands.len() > 1 {
            cands.retain(|c| pts.iter().all(|p| c.theatre_rects.values().any(|r| Plan::contains(*r, *p))));
        }
        if cands.len() != 1 {
            bail!("cached grid {gi}: {} owning charts for its points", cands.len());
        }
        result.push(cands[0].key.clone());
    }
    Ok(result)
}

/// Every fixed point of the tree with its owning chart, in file order: the `FIXED_PAIRS` pairs of every record and
/// every cached `PATHFINDING_GRID[]` polygon point (owned by the grid's chart). Unset `(0, 0)` sentinels (slot garrison
/// residences, unplaced characters) are skipped; `chart` is `None` when the geometry is ambiguous/unmapped.
pub fn owned_fixed_points(root: &Node, plan: &Plan) -> Result<Vec<OwnedPoint>> {
    let grid_charts = cached_grid_charts(root, plan)?;
    let mut out: Vec<OwnedPoint> = Vec::new();
    let mut error: Option<anyhow::Error> = None;
    root.walk(&mut |node, path| {
        if error.is_some() {
            return;
        }
        let mut visit = || -> Result<()> {
            match node {
                Node::Record { name, children, .. } => {
                    let Some((_, ats)) = FIXED_PAIRS.iter().find(|(n, _)| *n == name.as_str()) else {
                        return Ok(());
                    };
                    let parent = path.last().map(String::as_str);
                    let sentinel = (name == "SIEGEABLE_GARRISON_RESIDENCE" && parent == Some("REGION_SLOT"))
                        || (name == "LOCOMOTABLE" && parent == Some("CHARACTER"));
                    for &at in *ats {
                        let get = |i: usize| -> Result<i32> {
                            fixed_i32(children.get(i).ok_or_else(|| anyhow!("{name} has no child {i}"))?)
                                .with_context(|| format!("{name}[{i}]"))
                        };
                        let point = [get(at)?, get(at + 1)?];
                        if point == [0, 0] && sentinel {
                            continue;
                        }
                        out.push(OwnedPoint { record: name.clone(), chart: geometry_chart(plan, point), point });
                    }
                }
                Node::Array { name, items, .. } if name == "PATHFINDING_GRID" => {
                    for (gi, grid) in items.iter().enumerate() {
                        for pts in grid_polygons(grid).with_context(|| format!("cached grid {gi}"))? {
                            for point in pts {
                                out.push(OwnedPoint {
                                    record: "PATHFINDING_GRID[]".to_string(),
                                    chart: Some(grid_charts[gi].clone()),
                                    point,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
            Ok(())
        };
        if let Err(e) = visit() {
            error = Some(e);
        }
    });
    match error {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf;

    fn i(v: i32) -> Node {
        esf::i32v(v)
    }

    fn q20(v: f64) -> i32 {
        (v * 1048576.0) as i32
    }

    /// payload of polygons given as point lists
    fn payload(polys: &[&[[i32; 2]]]) -> Node {
        let mut words: Vec<u32> = Vec::new();
        for poly in polys {
            words.push(poly.len() as u32);
            for p in *poly {
                words.push(p[0] as u32);
                words.push(p[1] as u32);
            }
            words.push(0);
        }
        esf::arr_u32(&words)
    }

    fn grid(polys: &[&[[i32; 2]]]) -> Vec<Node> {
        vec![esf::u32v(polys.len() as u32), payload(polys)]
    }

    fn root(grids: Vec<Vec<Node>>, other: Vec<Node>) -> Node {
        let mut children = other;
        children.push(Node::record("CAMPAIGN_PATHFINDER", 0, vec![Node::array("PATHFINDING_GRID", 0, grids)]));
        Node::record("ROOT", 0, children)
    }

    fn europe_square() -> [[i32; 2]; 3] {
        [[q20(0.0), q20(200.0)], [q20(10.0), q20(200.0)], [q20(10.0), q20(210.0)]]
    }

    #[test]
    fn cached_polygons_parse_in_order_and_reject_trailing_or_truncated_payloads() {
        let tri = europe_square();
        let tri2 = [[1, 2], [3, 4], [5, 6]];
        let r = root(vec![grid(&[&tri, &tri2])], vec![]);
        let got = cached_polygons_by_grid(&r).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], (0, tri.to_vec()));
        assert_eq!(got[1].1, tri2.to_vec());
        // trailing word
        let mut bad = grid(&[&tri]);
        bad[1] = esf::arr_u32(&[3, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(cached_polygons_by_grid(&root(vec![bad], vec![])).is_err());
        // truncated polygon
        let mut bad = grid(&[&tri]);
        bad[1] = esf::arr_u32(&[3, 0, 0]);
        assert!(cached_polygons_by_grid(&root(vec![bad], vec![])).is_err());
    }

    #[test]
    fn grid_chart_follows_crop_evidence_and_rejects_empty_or_unmapped_grids() {
        let plan = Plan::proven();
        let eur = europe_square();
        let r = root(vec![grid(&[&eur])], vec![]);
        assert_eq!(cached_grid_charts(&r, &plan).unwrap(), vec!["eur_ind".to_string()]);
        // the same grid in the south pole is covered by no crop
        let far = [[q20(0.0), q20(-630.0)], [q20(1.0), q20(-630.0)], [q20(1.0), q20(-629.0)]];
        assert!(cached_grid_charts(&root(vec![grid(&[&far])], vec![]), &plan).is_err());
        // a grid without points
        assert!(cached_grid_charts(&root(vec![grid(&[])], vec![]), &plan).is_err());
        // two PATHFINDING_GRID arrays are rejected
        let two = Node::record(
            "ROOT",
            0,
            vec![
                Node::array("PATHFINDING_GRID", 0, vec![grid(&[&eur])]),
                Node::array("PATHFINDING_GRID", 0, vec![grid(&[&eur])]),
            ],
        );
        assert!(cached_grid_charts(&two, &plan).is_err());
    }

    #[test]
    fn fixed_points_follow_pair_positions_skip_unset_sentinels_and_keep_order() {
        let plan = Plan::proven();
        let (x, y) = (q20(-500.0), q20(250.0)); // america theatre
        let slot_residence = Node::record("SIEGEABLE_GARRISON_RESIDENCE", 0, {
            let mut c: Vec<Node> = (0..10).map(|_| esf::u32v(0)).collect();
            c.push(i(0));
            c.push(i(0));
            c
        });
        let slot = Node::record(
            "REGION_SLOT",
            0,
            vec![esf::u32v(0), esf::u32v(0), esf::u32v(0), esf::u32v(0), esf::u32v(0), esf::u32v(0), i(x), i(y), slot_residence],
        );
        // an unset (0, 0) LOCOMOTABLE inside CHARACTER is skipped; the same inside another record is a real point
        let loc = |px: i32, py: i32| Node::record("LOCOMOTABLE", 0, vec![i(px), i(py), i(px), i(py)]);
        let character = Node::record("CHARACTER", 0, vec![loc(0, 0)]);
        let army = Node::record("ARMY", 0, vec![loc(x, y)]);
        let r = root(vec![grid(&[&europe_square()])], vec![slot, character, army]);
        let pts = owned_fixed_points(&r, &plan).unwrap();
        let summary: Vec<(&str, Option<&str>, [i32; 2])> =
            pts.iter().map(|p| (p.record.as_str(), p.chart.as_deref(), p.point)).collect();
        assert_eq!(summary[0], ("REGION_SLOT", Some("america"), [x, y]));
        // sentinel garrison residence (REGION_SLOT parent) skipped, CHARACTER locomotable skipped, ARMY kept twice
        assert_eq!(summary[1], ("LOCOMOTABLE", Some("america"), [x, y]));
        assert_eq!(summary[2], ("LOCOMOTABLE", Some("america"), [x, y]));
        assert_eq!(pts.iter().filter(|p| p.record == "SIEGEABLE_GARRISON_RESIDENCE").count(), 0);
        assert_eq!(pts.iter().filter(|p| p.record == "PATHFINDING_GRID[]").count(), 3);
        assert!(pts.iter().filter(|p| p.record == "PATHFINDING_GRID[]").all(|p| p.chart.as_deref() == Some("eur_ind")));
    }

    #[test]
    fn non_i32_pair_children_are_errors() {
        let plan = Plan::proven();
        let bad = Node::record("CAI_SITUATED", 0, vec![esf::u32v(1), i(2)]);
        let r = root(vec![grid(&[&europe_square()])], vec![bad]);
        assert!(owned_fixed_points(&r, &plan).is_err());
    }
}
