//! `regions.esf` atlas migration: clipped/regenerated `region_data` and `groundtypes`, regenerated `query_info`,
//! chart-owned scaling of every other spatial section (accepted POC `run_atlas` / `transform_rest`).

use super::assemble::{Kind, OutData, assemble};
use super::camera;
use super::clip::Ctx;
use super::contract::build_contract;
use super::geom::widen;
use super::query::{QueryStats, build_query};
use super::scaler::{Rule, Scaler, enumerate_float_fields, classify_float_fields, scale_bridges, scale_mountains, scale_theatres, scale_trade_nodes, scaler_for};
use super::source::Source;
use super::verify::{SectionPairs, SettlementPair, verify_written};
use crate::atlas::Plan;
use crate::esf::{self, Esf, Node, Value};
use crate::model::Progress;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;

/// random query/polygon agreement samples per chart and over the world
pub const VERIFY_SAMPLES: usize = 4000;

pub struct RegionsOutput {
    /// the migrated regions.esf (camera rectangles already Q12-synchronised)
    pub esf: Esf,
    pub contract: Json,
    pub report: Json,
}

struct Rest<'a> {
    scalers: BTreeMap<String, Scaler<'a>>,
    pairs: SectionPairs,
    settlement_chart: BTreeMap<String, String>,
    bridge_counts: BTreeMap<String, u64>,
}

fn region_data_node(od: &OutData, proto: &Node, settlement: &mut dyn FnMut(usize, &Node) -> Result<Node>) -> Result<Node> {
    let src = od.src;
    let Node::Record { name, version, children } = proto else {
        bail!("regions atlas: region_data is not a record");
    };
    let w = od.clip.ctx.world_out;
    let mut regions_items: Vec<Vec<Node>> = Vec::with_capacity(od.regions.len());
    for (ri, areas) in od.regions.iter().enumerate() {
        let r = src.regs[ri];
        let mut a_items: Vec<Vec<Node>> = Vec::with_capacity(areas.len());
        for oa in areas {
            let proto_area = src.area(oa.src).item;
            let mut item: Vec<Node> = Vec::with_capacity(proto_area.len());
            let mut vi = 0usize;
            for x in proto_area.iter() {
                match x {
                    Node::Value(Value::XY(_)) => {
                        item.push(esf::xyv(oa.bbox[vi.min(1)][0], oa.bbox[vi.min(1)][1]));
                        vi += 1;
                    }
                    Node::Record { name, version, .. } if name == "faces" => {
                        item.push(Node::record(name.clone(), *version, vec![esf::arr_u32(&oa.faces)]));
                    }
                    Node::Array { name, version, items } if name == "outlines" => {
                        let proto_o = items.first().context("regions atlas: area without outline prototype")?;
                        let mut outs: Vec<Vec<Node>> = Vec::with_capacity(oa.rings.len());
                        for (((flag, ids), bb), runs) in oa.rings.iter().zip(&oa.ring_bbox).zip(&oa.conn) {
                            let mut o: Vec<Node> = Vec::with_capacity(proto_o.len());
                            let mut xy_seen = 0usize;
                            for y in proto_o {
                                match y {
                                    Node::Value(Value::Bool(_)) => o.push(esf::b(*flag)),
                                    Node::Value(Value::XY(_)) => {
                                        let p = bb[xy_seen.min(1)];
                                        o.push(esf::xyv(p[0], p[1]));
                                        xy_seen += 1;
                                    }
                                    Node::Value(Value::Array { .. }) => o.push(esf::arr_u32(ids)),
                                    Node::Array { name, version, .. } if name == "connectivity" => {
                                        o.push(Node::array(
                                            name.clone(),
                                            *version,
                                            runs.iter().map(|r| vec![esf::u32v(r[0]), esf::u32v(r[1]), esf::u32v(r[2])]).collect(),
                                        ));
                                    }
                                    other => o.push(other.clone()),
                                }
                            }
                            outs.push(o);
                        }
                        item.push(Node::array(name.clone(), *version, outs));
                    }
                    other => item.push(other.clone()),
                }
            }
            a_items.push(item);
        }
        let lo = [
            areas.iter().map(|a| a.bbox[0][0]).fold(f32::INFINITY, f32::min),
            areas.iter().map(|a| a.bbox[0][1]).fold(f32::INFINITY, f32::min),
        ];
        let hi = [
            areas.iter().map(|a| a.bbox[1][0]).fold(f32::NEG_INFINITY, f32::max),
            areas.iter().map(|a| a.bbox[1][1]).fold(f32::NEG_INFINITY, f32::max),
        ];
        let mut new: Vec<Node> = Vec::with_capacity(r.len());
        let mut vi = 0;
        for x in r.iter() {
            match x {
                Node::Value(Value::XY(_)) => {
                    let p = if vi == 0 { lo } else { hi };
                    new.push(esf::xyv(p[0], p[1]));
                    vi += 1;
                }
                Node::Array { name, version, .. } if name == "areas" => {
                    new.push(Node::array(name.clone(), *version, std::mem::take(&mut a_items)));
                }
                Node::Record { name, .. } if name == "settlement_and_slots" => new.push(settlement(ri, x)?),
                other => new.push(other.clone()),
            }
        }
        regions_items.push(new);
    }
    let vb = esf::arr_xy(&od.coords);
    let mut kids: Vec<Node> = Vec::with_capacity(children.len());
    let mut vi = 0;
    for x in children {
        match x {
            Node::Record { name, version, .. } if name == "vertices" => kids.push(Node::record(name.clone(), *version, vec![vb.clone()])),
            Node::Value(Value::XY(_)) => {
                let p = if vi == 0 { w[0] } else { w[1] };
                kids.push(esf::xyv(p[0], p[1]));
                vi += 1;
            }
            Node::Array { name, version, .. } if name == "regions" => {
                kids.push(Node::array(name.clone(), *version, std::mem::take(&mut regions_items)));
            }
            other => kids.push(other.clone()),
        }
    }
    Ok(Node::record(name.clone(), *version, kids))
}

/// Chart-owned scaling of one political settlement record; its label point, polygon and every slot point must lie
/// inside the crop of the chart owning the label point (and in its theatre's chart).
fn settlement<'a>(ctx: &'a Ctx, main_src: &Source, rest: &mut Rest<'a>, ri: usize, node: &Node) -> Result<Node> {
    let kids = node.children()?;
    let pt = kids.first().context("regions atlas: empty settlement record")?.as_xy()?;
    let chart_i = ctx.chart_of_point(widen(pt))?;
    let chart = &ctx.plan.charts[chart_i];
    let rkey = main_src.region_key(ri)?;
    let tk = main_src.regs[ri]
        .iter()
        .find_map(|n| match n {
            Node::Value(Value::I32(v)) => Some(*v),
            _ => None,
        })
        .with_context(|| format!("regions atlas: region {rkey} has no theatre id"))?;
    if tk != -1 {
        let t = tk.to_string();
        match ctx.plan.charts.iter().find(|c| c.theatres.contains(&t)) {
            Some(tc) if tc.key == chart.key => {}
            _ => bail!("regions atlas: settlement of region {rkey}: label point chart {} != theatre {tk} chart", chart.key),
        }
    }
    let mut pts: Vec<[f32; 2]> = vec![pt];
    for s in kids {
        match s {
            Node::Value(Value::Array { code: 0x4C, .. }) => pts.extend(s.xy_array()?),
            Node::Array { name, items, .. } if name == "slot_descriptions" => {
                for slot in items {
                    for z in slot {
                        match z {
                            Node::Value(Value::XYZ(v)) => pts.push([v[0], v[2]]),
                            Node::Value(Value::XY(v)) => pts.push(*v),
                            Node::Value(Value::Array { code: 0x4C, .. }) => pts.extend(z.xy_array()?),
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    for p in &pts {
        ensure!(
            ctx.unions[chart_i].contains_point(widen(*p)),
            "regions atlas: settlement of region {rkey}: point {p:?} outside crop of chart {}",
            chart.key
        );
    }
    rest.settlement_chart.insert(rkey.to_string(), chart.key.clone());
    let new = scaler_for(&ctx.plan, &mut rest.scalers, chart).node(node, "/root/region_data/regions[]")?;
    rest.pairs.settlement.push(SettlementPair { region: ri, chart: chart_i, src: node.clone(), dst: new.clone() });
    Ok(new)
}

#[allow(clippy::too_many_arguments)]
fn transform_rest<'a>(
    ctx: &'a Ctx,
    main_src: &Source,
    root: &Node,
    od_main: &OutData,
    od_gt: &OutData,
    q_main: &Node,
    q_gt: &Node,
    rest: &mut Rest<'a>,
) -> Result<Node> {
    let plan = &ctx.plan;
    let Node::Record { name: root_name, version: root_version, children } = root else {
        bail!("regions atlas: ESF root is not a record");
    };
    let mut kids: Vec<Node> = Vec::with_capacity(children.len());
    for c in children {
        let Some(n) = c.name() else {
            bail!("regions atlas: unhandled scalar child of the regions root");
        };
        match n {
            "theatres_and_region_keys" | "theatres" => {
                kids.push(scale_theatres(plan, n, c, &mut rest.scalers, &mut rest.pairs.theatre)?);
            }
            "black_shroud_outlines" => {
                ensure!(c.items()?.is_empty(), "regions atlas: black_shroud_outlines is not empty (unsupported)");
                kids.push(c.clone());
            }
            "region_data" => {
                let mut f = |ri: usize, node: &Node| settlement(ctx, main_src, rest, ri, node);
                kids.push(region_data_node(od_main, c, &mut f)?);
            }
            "mountain_data" => kids.push(scale_mountains(plan, c, &mut rest.scalers, &mut rest.pairs.mountain)?),
            "climate_indices" => kids.push(c.clone()),
            "query_info" => kids.push(q_main.clone()),
            "groundtypes" => {
                let mut gk: Vec<Node> = Vec::new();
                for g in c.children()? {
                    match g.name() {
                        Some("region_data") => {
                            let mut f = |_: usize, _: &Node| -> Result<Node> { bail!("regions atlas: groundtypes with settlements") };
                            gk.push(region_data_node(od_gt, g, &mut f)?);
                        }
                        Some("query_info") => gk.push(q_gt.clone()),
                        _ => gk.push(g.clone()),
                    }
                }
                kids.push(Node::record("groundtypes", c.version().unwrap_or(0), gk));
            }
            "map_heights" => {
                for m in c.children()? {
                    if m.name() == Some("MAP_HEIGHT_AREA") && !m.items()?.is_empty() {
                        bail!("regions atlas: map_heights has areas (unsupported)");
                    }
                }
                kids.push(c.clone());
            }
            "trade_nodes" => kids.push(scale_trade_nodes(plan, c, &mut rest.scalers, &mut rest.pairs.trade)?),
            "bridges" => kids.push(scale_bridges(plan, c, &mut rest.bridge_counts)?),
            other => bail!("regions atlas: unhandled root child {other}"),
        }
    }
    Ok(Node::record(root_name.clone(), *root_version, kids))
}

fn dataset_report(od: &OutData, qs: &QueryStats) -> Json {
    let src = od.src;
    let extra = od.regions.iter().flatten().filter(|a| !a.primary && !a.is_gap).count();
    json!({
        "source": {"regions": src.n_regions, "areas": src.areas.len(), "vertices": src.v.len()},
        "output": {"regions": od.regions.len(), "areas": od.regions.iter().map(|r| r.len()).sum::<usize>(), "vertices": od.coords.len()},
        "extra_fragment_areas": extra,
        "backdrop": {
            "area": [od.backdrop.0, od.backdrop.1],
            "region": src.region_key(od.backdrop.0).unwrap_or(""),
            "why": od.backdrop_why,
            "output_local": od.gap_local,
            "gap_vertices": od.gap.outer.len(),
            "gap_holes": od.gap.holes.len(),
        },
        "dropped_areas": od.dropped,
        "absorbed_slivers": od.absorbed.len(),
        "clip_stats": od.clip.stats,
        "assemble_stats": od.stats,
        "faces": od.faces_stats,
        "unmatched_world_border_edges": od.unmatched.len(),
        "query": {
            "nodes": qs.nodes,
            "cells": qs.cells,
            "max_leaf_segments": qs.max_leaf,
            "leaves_capped_by_depth": qs.depth_cap_leaves,
            "probe_fallbacks": qs.probe_fallbacks,
            "rules": {
                "split_when_segments_gt": super::query::SPLIT_THRESHOLD,
                "max_depth": super::query::MAX_DEPTH,
                "probe_line_eps": super::query::PROBE_LINE_EPS,
            },
        },
    })
}

pub fn transform(plan: &Plan, source: &Esf, progress: &Progress<'_>) -> Result<RegionsOutput> {
    let ctx = Ctx::new(plan)?;
    let root = &source.root;
    let rd_main = root.get("region_data")?;
    let rd_gt = root.path(&["groundtypes", "region_data"])?;
    for (label, rd) in [("region_data", rd_main), ("groundtypes.region_data", rd_gt)] {
        let ch = rd.children()?;
        let lo = ch.get(1).context("world bounds missing")?.as_xy()?;
        let hi = ch.get(2).context("world bounds missing")?.as_xy()?;
        ensure!(
            [widen(lo), widen(hi)] == plan.source_world,
            "regions atlas: {label} bounds {:?} != plan source_world {:?}",
            [lo, hi],
            plan.source_world
        );
    }
    // every float leaf of the source is either scaled by a rule or explicitly preserved with a reason
    let fields = enumerate_float_fields(root);
    let (covered, preserved, unknown) = classify_float_fields(&fields);
    ensure!(unknown.is_empty(), "regions atlas: unclassified float fields: {}", serde_json::to_string(&unknown)?);
    let window = [
        plan.interior_point_window[0][0],
        plan.interior_point_window[0][1],
        plan.interior_point_window[1][0],
        plan.interior_point_window[1][1],
    ];
    let bg = &plan.background_region;
    let spec = match (bg.get("key").and_then(|k| k.as_str()), bg.get("source_area").and_then(|a| a.as_u64())) {
        (Some(k), Some(a)) => format!("{k}:{a}"),
        _ => bail!("regions atlas: plan background_region needs key and source_area"),
    };
    progress("Atlas regions: reading source regions");
    let main_src = Source::new(rd_main, "region_data", true)?;
    let gt_src = Source::new(rd_gt, "groundtypes.region_data", false)?;
    progress(&format!(
        "Atlas regions: clipping {} areas / {} vertices (groundtypes {} areas)",
        main_src.areas.len(),
        main_src.v.len(),
        gt_src.areas.len()
    ));
    let od_main = assemble(&ctx, &main_src, Some(&spec), Kind::Main)?;
    let od_gt = assemble(&ctx, &gt_src, None, Kind::Groundtypes)?;
    progress("Atlas regions: rebuilding query_info");
    let (q_main, qs_main) = build_query(&od_main, true, root.get("query_info")?)?;
    let (q_gt, qs_gt) = build_query(&od_gt, false, root.path(&["groundtypes", "query_info"])?)?;

    progress("Atlas regions: scaling theatres, settlements, mountains, trade nodes and bridges");
    let mut rest = Rest {
        scalers: BTreeMap::new(),
        pairs: SectionPairs { theatre: Vec::new(), trade: Vec::new(), mountain: Vec::new(), settlement: Vec::new() },
        settlement_chart: BTreeMap::new(),
        bridge_counts: BTreeMap::new(),
    };
    let new_root = transform_rest(&ctx, &main_src, root, &od_main, &od_gt, &q_main, &q_gt, &mut rest)?;
    let migrated = Esf { timestamp: source.timestamp, names: source.names.clone(), root: new_root };

    progress("Atlas regions: verifying the serialised file");
    let bytes = esf::write(&migrated).context("Serialise the migrated regions.esf")?;
    let reread = esf::read(&bytes).context("Re-parse the migrated regions.esf")?;
    ensure!(reread.root == migrated.root, "regions atlas: serialised regions.esf does not parse back to the migrated tree");
    let verification = verify_written(&ctx, &reread, &main_src, &od_main, &od_gt, &rest.pairs, VERIFY_SAMPLES)?;

    progress("Atlas regions: synchronising camera rectangles");
    let mut final_esf = migrated;
    let camera_report = camera::snap_regions(plan, &mut final_esf.root)?;

    let contract = build_contract(&ctx, &[(Kind::Main, &od_main), (Kind::Groundtypes, &od_gt)], window)?;

    let mut transformed: BTreeMap<String, (Rule, u64)> = BTreeMap::new();
    for s in rest.scalers.values() {
        for (k, st) in s.stats() {
            let e = transformed.entry(k.clone()).or_insert((st.rule, 0));
            e.1 += st.points;
        }
    }
    let mut by_chart: BTreeMap<String, u64> = BTreeMap::new();
    for chart in rest.settlement_chart.values() {
        *by_chart.entry(chart.clone()).or_insert(0) += 1;
    }
    let report = json!({
        "tool": "native atlas regions (accepted POC regions_atlas_build)",
        "plan": {
            "scale": plan.scale,
            "charts": plan.charts.iter().map(|c| json!({"key": c.key, "offset": c.offset, "theatres": c.theatres})).collect::<Vec<_>>(),
            "world": plan.world,
            "source_world": plan.source_world,
            "geometry_checks": {"min_world_margin": ctx.min_world_margin, "min_chart_separation": ctx.min_chart_separation},
        },
        "options": {"min_fragment_area_source_units2": super::assemble::MIN_FRAGMENT_AREA, "allow_drop_areas": true,
                    "interior_window": window, "verify_samples": VERIFY_SAMPLES},
        "region_data": dataset_report(&od_main, &qs_main),
        "groundtypes": dataset_report(&od_gt, &qs_gt),
        "transformed_fields": transformed.iter().map(|(k, (r, p))| (k.clone(), json!({"rule": r.as_str(), "points": p}))).collect::<serde_json::Map<String, Json>>(),
        "settlement_chart": by_chart,
        "bridge_charts": rest.bridge_counts,
        "float_field_audit": {"scaled": covered.len(), "preserved_not_scaled": preserved, "unclassified": unknown},
        "camera": camera_report,
        "verification": verification,
        "unverified_engine_assumptions": [
            "query segment order within a cell is by output edge order (baseline order follows border chains, not reproduced)",
            "faces of re-triangulated groups have no Steiner vertices (baseline groups contain interior Steiner vertices)",
            "fragments of one source area share its v7 ids/flags (copied); new fragment areas are appended at the end of the region",
            "the world gap is one polygon owned by the backdrop area",
        ],
    });
    Ok(RegionsOutput { esf: final_esf, contract, report })
}
