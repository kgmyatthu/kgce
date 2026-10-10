//! Navigation migration of the stock-Q20 connected atlas: native port of `navigation_scale.py` (atlas mode).
//!
//! Reads the SOURCE-SPACE `pathfinding.esf`, `poi.esf`, `sea_grids.esf`, `trade_routes.esf` of
//! `<data>/campaign_maps/nap_world` and the startpos of `<data>/campaigns/wld_napoleon`, maps every position through
//! its OWNING chart (never a global affine) and returns the four new ESF trees plus the Q20 coordinate overrides and a
//! factual validation report.
//!
//! Quantization: `dst = round_half_even(src_raw * N / 2^20) + chart_offset_raw` per axis (`N = scale * 2^20`, exact
//! integer arithmetic; equal to `Plan::fixed_xy` bit for bit: for every in-window value the f64 products/sums of that
//! function are exact, and the quantizer additionally verifies the equality for every value it maps). Every pathfinding
//! grid line `origin + k * cell` is pinned to its exact image; unpinned shared values may be nudged by up to 8 raw units
//! by the greedy orientation repair so that no polygon area/turn sign flips; every deviation from the plain rule is
//! written to the overrides document the campaign conversion consumes.
//!
//! Gates (all hard errors): scale/offset/crop geometry, grid ownership, ambiguous or uncovered POI/spline ownership
//! (with the half-open grid tiebreak), blocked trade splines (cross-chart or outside every crop), orientation repair
//! leaving flipped/zero polygons, the stock window checks (`diff_window`, `grid_index_window`, `sum_window`), start
//! index u16 range, region/area references against the regions contract, and a parallel tree diff proving that ONLY the
//! deliberately touched values changed.

mod axis;
mod ctx;
mod diff;
mod flat;
mod grid;
mod orient;
mod pathfinding;
mod poi;
mod quant;
mod refs;
mod sea_grids;
mod trade;
mod util;
mod window;

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};

use crate::atlas::Plan;
use crate::esf::{self, Esf};
use crate::model::Progress;
use ctx::Ctx;
use diff::{Touched, count_nodes, diff, ensure_only_touched};
use refs::RegionRefs;
use util::UNIT;

pub(super) struct NavigationOutput {
    pub pathfinding: Esf,
    pub poi: Esf,
    pub sea_grids: Esf,
    pub trade_routes: Esf,
    /// `q20-coordinate-overrides.json`: the entries that differ from the plain rule, per chart.
    pub overrides: Json,
    /// Factual validation report (mirrors `navigation-manifest.json`).
    pub report: Json,
}

const FILES: [&str; 4] = ["pathfinding.esf", "poi.esf", "sea_grids.esf", "trade_routes.esf"];

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn load(path: &Path) -> Result<(Vec<u8>, Esf)> {
    let raw = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let esf = esf::read(&raw).with_context(|| format!("parsing {}", path.display()))?;
    Ok((raw, esf))
}

struct Staged {
    name: &'static str,
    raw_len: usize,
    raw_sha: String,
    new: Esf,
    out_len: usize,
    out_sha: String,
    nodes_before: usize,
    nodes_after: usize,
    values_changed: usize,
    modified_sites: usize,
    detail: Json,
}

/// See the module documentation. `data` is the SOURCE-SPACE data directory; `contract` is the regions index contract
/// (`region_data.regions[i].{key, ..}`) of the geography migration.
pub(super) fn transform(
    data: &Path,
    plan: &Plan,
    source_regions: &Esf,
    new_regions: &Esf,
    contract: &Json,
    progress: &Progress<'_>,
) -> Result<NavigationOutput> {
    let started = Instant::now();
    let plan_sha = sha256_hex(&serde_json::to_vec(plan)?);
    let mut ctx = Ctx::new(plan)?;
    let regions = RegionRefs::load(source_regions, new_regions, contract)?;
    let map_dir = data.join("campaign_maps").join("nap_world");

    progress("navigation: reading source files");
    let mut sources: Vec<(&'static str, Vec<u8>, Esf)> = Vec::new();
    for name in FILES {
        let (raw, old) = load(&map_dir.join(name))?;
        // exact roundtrip of the untouched source tree through the writer (so every diff is ours)
        if esf::write(&old)? != raw {
            bail!("{name}: writer does not reproduce the source bytes");
        }
        sources.push((name, raw, old));
    }
    let (_, startpos) = load(&data.join("campaigns").join("wld_napoleon").join("startpos.esf"))?;

    let quantized = quant::build_quantization_atlas(&mut ctx, &sources[0].2.root, &sources[1].2.root, &startpos.root, progress)?;
    let path_ids = pathfinding::region_id_range(&sources[0].2.root)?;

    // ---- phase 1: transform every file in memory and prove that only intended values changed
    let mut staged: Vec<Staged> = Vec::new();
    for (name, raw, old) in &sources {
        progress(&format!("navigation: {name}"));
        let mut new = old.clone();
        let mut touched = Touched::default();
        let detail = match *name {
            "pathfinding.esf" => pathfinding::do_pathfinding(&mut ctx, &mut new.root, &mut touched)?,
            "poi.esf" => poi::do_poi(&mut ctx, new.root.children_mut()?, &mut touched)?,
            "sea_grids.esf" => sea_grids::do_sea_grids(&mut ctx, new.root.children_mut()?, &mut touched)?,
            _ => trade::do_trade_routes(&mut ctx, &mut new.root, &mut touched)?,
        };
        let changed = diff(&old.root, &new.root).with_context(|| format!("{name}: shape check"))?;
        ensure_only_touched(name, &changed, &touched)?;
        let out = esf::write(&new).with_context(|| format!("{name}: writing the transformed tree"))?;
        let reparsed = esf::read(&out).with_context(|| format!("{name}: re-parsing the written bytes"))?;
        if reparsed.timestamp != new.timestamp || reparsed.names != new.names {
            bail!("{name}: re-parsed output changed the header/name table");
        }
        let again = diff(&new.root, &reparsed.root)?;
        if let Some(first) = again.first() {
            bail!("{name}: re-parsed output differs from the transformed tree at {first:?}");
        }
        staged.push(Staged {
            name: *name,
            raw_len: raw.len(),
            raw_sha: sha256_hex(raw),
            out_len: out.len(),
            out_sha: sha256_hex(&out),
            nodes_before: count_nodes(&old.root),
            nodes_after: count_nodes(&new.root),
            values_changed: changed.len(),
            modified_sites: touched.paths.len(),
            detail,
            new,
        });
    }

    // ---- atlas gates over the final positions of all four files
    progress("navigation: stock window gates");
    let wmin_raw = [0, 1].map(|a| (plan.source_world[0][a] * UNIT as f64).round_ties_even() as i64);
    let (window, mut failed) = window::check_stock_window(&ctx.env, wmin_raw)?;
    let (regions_report, region_failed) = regions.check(&ctx.refs, path_ids);
    failed.extend(region_failed);
    if !failed.is_empty() {
        bail!("atlas gates failed: {failed:?}\nstock window: {window}\nregion references: {regions_report}");
    }
    let unaligned: Vec<usize> = ctx
        .areas
        .iter()
        .filter(|r| r.align_remainder_raw.is_some_and(|rem| rem.iter().any(|v| *v != 0)))
        .map(|r| r.area)
        .collect();

    // ---- documents
    let chart_keys: Vec<&str> = plan.charts.iter().map(|c| c.key.as_str()).collect();
    let offset_raw: Map<String, Json> = chart_keys.iter().zip(&ctx.off_raw).map(|(k, o)| (k.to_string(), json!(o))).collect();
    let override_entries: Map<String, Json> = quantized
        .charts_doc
        .iter()
        .map(|(k, v)| {
            let n = |a: &str| v.get(a).and_then(Json::as_object).map_or(0, Map::len);
            (k.clone(), json!({"x": n("x"), "y": n("y")}))
        })
        .collect();
    let mut area_rows = Vec::new();
    let mut grid_rows = Vec::new();
    for row in &ctx.areas {
        let new = row.new.as_ref().with_context(|| format!("pathfinding area {} was not transformed", row.area))?;
        let rem = row.align_remainder_raw.with_context(|| format!("pathfinding area {} has no alignment", row.area))?;
        let new_json = json!({"origin": new.origin, "cell": new.cell, "start": new.start});
        area_rows.push(json!({
            "area": row.area, "chart": ctx.chart_key(row.chart), "source_origin_raw": row.origin,
            "source_cell_raw": row.cell, "cols": row.cols, "rows": row.rows, "new": new_json,
            "align_remainder_raw": rem,
        }));
        grid_rows.push(json!({
            "area": row.area, "chart": ctx.chart_key(row.chart), "origin": row.origin, "cell": row.cell,
            "cols": row.cols, "rows": row.rows, "align_remainder_raw": rem, "new": new_json,
        }));
    }
    let overrides = json!({
        "format": 1,
        "source_fixed_bits": 20,
        "fixed_bits": 20,
        "scale": plan.scale,
        "scale_n_over_2_20": ctx.scale_n,
        "plan_sha256": plan_sha,
        "rule": "dst = round_half_even(src_raw * N / 2^20) + offset_raw[chart][axis]  (identical to Plan::fixed_xy); 'charts' lists ONLY the entries that differ from that rule",
        "offset_raw": offset_raw,
        "areas": area_rows,
        "charts": quantized.charts_doc,
    });

    let mut files = Map::new();
    let mut theatre_snaps = Vec::new();
    for st in &staged {
        if st.name == "sea_grids.esf" {
            if let Some(list) = st.detail.get("theatres").and_then(Json::as_array) {
                for t in list {
                    theatre_snaps.push(json!({"name": t["name"], "chart": t["chart"], "camera_snap": t["camera_snap"]}));
                }
            }
        }
        files.insert(
            st.name.to_string(),
            json!({
                "source": {"bytes": st.raw_len, "sha256": st.raw_sha},
                "output": {"bytes": st.out_len, "sha256": st.out_sha},
                "size_unchanged": st.raw_len == st.out_len,
                "writer_roundtrip_exact": true,
                "structure": {
                    "nodes": st.nodes_before, "nodes_after": st.nodes_after, "shape_identical": true,
                    "values_changed": st.values_changed, "modified_sites": st.modified_sites,
                    "modified_but_equal": st.modified_sites - st.values_changed,
                    "changes_outside_transform": 0, "reparsed_equals_transform": true,
                },
                "detail": st.detail,
            }),
        );
    }
    let report = json!({
        "tool": "kgc_port_editor atlas::geography::navigation (native port of navigation_scale.py, atlas mode)",
        "plan": {
            "sha256": plan_sha, "mode": "atlas", "scale": plan.scale, "world": plan.world,
            "source_world": plan.source_world,
            "fixed_bits": {"source": plan.source_fixed_bits, "output": plan.fixed_bits},
            "scale_n_over_2_20": ctx.scale_n, "offset_raw": offset_raw,
        },
        "source": "campaign_maps/nap_world",
        "files": files,
        "quantization": {
            "rule": "dst = round_half_even(src_raw * N / 2^20) + chart_offset_raw per axis (== Plan::fixed_xy), except where a polygon orientation repair moved an unpinned shared value of that chart; every pathfinding grid line origin+k*cell is pinned to its exact image; ownership per chart, never global",
            "charts": quantized.reports,
            "override_entries": override_entries,
        },
        "atlas": {
            "stock_window": window,
            "envelope_by_tag": ctx.env.dump(),
            "failed": failed,
            "grids": grid_rows,
            "grids_not_aligned_to_world_lattice": unaligned,
            "regions": regions_report,
            "stock_world_min_raw": wmin_raw,
            "owner_tiebreaks_by_half_open_grid": ctx.tiebreaks,
            "camera_snapped_theatre_bounds": {
                "rule": "the theatre min/max corners of sea_grids.esf (third copy next to the two in regions.esf) are the transformed plan theatre rectangle snapped to 1/4096 (Plan::camera_rect); cell boxes are not snapped",
                "theatres": theatre_snaps,
            },
        },
        "seconds": (started.elapsed().as_secs_f64() * 100.0).round() / 100.0,
    });

    let mut by_name = staged.into_iter();
    let mut next = |want: &str| -> Result<Esf> {
        let st = by_name.next().with_context(|| format!("missing staged {want}"))?;
        if st.name != want {
            bail!("staged file order changed: {} != {want}", st.name);
        }
        Ok(st.new)
    };
    Ok(NavigationOutput {
        pathfinding: next("pathfinding.esf")?,
        poi: next("poi.esf")?,
        sea_grids: next("sea_grids.esf")?,
        trade_routes: next("trade_routes.esf")?,
        overrides,
        report,
    })
}
