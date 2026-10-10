//! Verification of the written atlas `regions.esf` (accepted POC `verify_dataset`, `verify_unchanged_faces`,
//! `verify_sections`). Every check runs on the freshly parsed bytes: topology, ring orientation, outline /
//! area / region bounding boxes, connectivity runs, mesh faces, the `query_info` quadtree (structure and
//! point lookups against the polygons) and the scaled sections including exact settlement alignment.

use super::assemble::{NONE_AB, OutData};
use super::clip::Ctx;
use super::geom::{AreaPoly, GridIndex, P, Rect64, widen};
use super::scaler::{MountainPair, TheatrePair, check_scaled, check_scaled_items};
use super::source::Source;
use crate::atlas::Chart;
use crate::esf::{Esf, Node, Value};
use crate::regionmap::geometry::point_segment_distance;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value as Json, json};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Seed of the deterministic sample generator (the POC seed).
const SEED: u64 = 20260101;
/// Samples closer than this to any polygon boundary are not compared (POC `1e-3`).
const BOUNDARY_EPS: f64 = 1e-3;

/// One `settlement_and_slots` record before and after the transform.
pub(super) struct SettlementPair {
    pub region: usize,
    /// index into `ctx.plan.charts`
    pub chart: usize,
    pub src: Node,
    pub dst: Node,
}

/// Before/after pairs of the scaled sections, recorded while transforming.
pub(super) struct SectionPairs {
    pub theatre: Vec<TheatrePair>,
    /// (chart key, item children before, item children after)
    pub trade: Vec<(String, Vec<Node>, Vec<Node>)>,
    pub mountain: Vec<MountainPair>,
    pub settlement: Vec<SettlementPair>,
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

/// splitmix64 generator: fixed seed, deterministic across platforms.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Twice the signed shoelace area of a closed ring (> 0 counter-clockwise).
fn shoelace2(pts: &[P]) -> f64 {
    let n = pts.len();
    let mut s = 0.0;
    for k in 0..n {
        let (a, b) = (pts[k], pts[(k + 1) % n]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    s
}

fn ring_label(outer: bool, s2: f64) -> String {
    format!("{}/{}", if outer { "outer" } else { "hole" }, if s2 > 0.0 { "ccw" } else { "cw" })
}

/// Run-length encoding of a ring's reverse-edge owners: `[owner code, first position, last position]`.
/// `owner(from, to)` is the owner code of the reversed edge `to -> from` (or `NONE_AB`).
fn connectivity_runs(ids: &[u32], owner: impl Fn(u32, u32) -> u32) -> Vec<[u32; 3]> {
    let n = ids.len();
    let mut runs: Vec<[u32; 3]> = Vec::new();
    for p in 0..n {
        let ab = owner(ids[(p + 1) % n], ids[p]);
        if runs.last().is_some_and(|l| l[0] == ab) {
            if let Some(last) = runs.last_mut() {
                last[2] = p as u32;
            }
        } else {
            runs.push([ab, p as u32, p as u32]);
        }
    }
    runs
}

fn child_named<'x>(items: &'x [Node], name: &str) -> Result<&'x Node> {
    items.iter().find(|c| c.name() == Some(name)).with_context(|| format!("missing {name}"))
}

fn xy_at(items: &[Node], i: usize, what: &str) -> Result<[f32; 2]> {
    items.get(i).with_context(|| format!("{what}: missing child {i}"))?.as_xy().with_context(|| format!("{what}: child {i}"))
}

fn rect_of(a: [f32; 2], b: [f32; 2]) -> Rect64 {
    [widen(a), widen(b)]
}

/// Chart output bounds of one chart: its crop rectangles mapped through the chart transform.
fn out_bounds(ctx: &Ctx, ci: usize) -> Rect64 {
    let chart = &ctx.plan.charts[ci];
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for r in &ctx.rects[ci] {
        for corner in [r[0], r[1]] {
            let q = ctx.plan.xy(corner, chart);
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
    }
    [lo, hi]
}

/// True when a ring edge of `poly` (outer ring or kept hole) lies closer than `eps` to `p`.
fn near_boundary(poly: &AreaPoly, p: P, eps: f64) -> bool {
    for ring in std::iter::once(&poly.outer).chain(poly.holes.iter()) {
        let n = ring.len();
        for k in 0..n {
            let (a, b) = (ring[k], ring[(k + 1) % n]);
            if a[0].min(b[0]) > p[0] + eps || a[0].max(b[0]) < p[0] - eps || a[1].min(b[1]) > p[1] + eps || a[1].max(b[1]) < p[1] - eps {
                continue;
            }
            if point_segment_distance(p, a, b) < eps {
                return true;
            }
        }
    }
    false
}

fn orient(a: P, b: P, c: P) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

// ---------------------------------------------------------------------------------------------
// query_info
// ---------------------------------------------------------------------------------------------

/// Read-only view of one written `query_info/node` quadtree with its vertex pool.
struct QueryTree<'b> {
    name: &'static str,
    root: &'b Node,
    v: &'b [P],
    with_left: bool,
}

#[derive(Default)]
struct WalkStats {
    nodes: u64,
    cells: u64,
    bad_partition: u64,
}

fn walk(n: &Node, st: &mut WalkStats) -> Result<()> {
    st.nodes += 1;
    let c = n.children()?;
    ensure!(c.len() > 2, "query node has {} children", c.len());
    let lo = xy_at(c, 0, "query node")?;
    let hi = xy_at(c, 1, "query node")?;
    let kids = &c[2..];
    if kids[0].name() == Some("cell") {
        st.cells += 1;
        return Ok(());
    }
    let m = xy_at(kids[0].children()?, 1, "query quadrant")?;
    let quad = |nm: &str| -> Option<([f32; 2], [f32; 2])> {
        let k = kids.iter().find(|k| k.name() == Some(nm))?;
        let c = k.children().ok()?;
        Some((c.first()?.as_xy().ok()?, c.get(1)?.as_xy().ok()?))
    };
    let ok = kids.len() == 4
        && (|| -> Option<bool> {
            let bl = quad("bottom_left")?;
            let tl = quad("top_left")?;
            let br = quad("bottom_right")?;
            let tr = quad("top_right")?;
            Some(
                bl.0 == lo
                    && tr.1 == hi
                    && tl.0 == [lo[0], m[1]]
                    && tl.1 == [m[0], hi[1]]
                    && br.0 == [m[0], lo[1]]
                    && br.1 == [hi[0], m[1]]
                    && tr.0 == m,
            )
        })()
        .unwrap_or(false);
    if !ok {
        st.bad_partition += 1;
    }
    for k in kids {
        walk(k, st)?;
    }
    Ok(())
}

impl<'b> QueryTree<'b> {
    fn new(src: &'b Source<'_>, qi: &'b Node, with_left: bool) -> Result<Self> {
        Ok(Self { name: src.name, root: qi.get("node")?, v: &src.v, with_left })
    }

    /// The leaf `cell` record containing `p` (top/bottom, left/right of the first child's hi corner).
    fn find_cell(&self, p: P) -> Result<&'b Node> {
        let mut n = self.root;
        loop {
            let c = n.children()?;
            ensure!(c.len() > 2, "regions atlas: verify {}: query node has {} children", self.name, c.len());
            let kids = &c[2..];
            if kids[0].name() == Some("cell") {
                return Ok(&kids[0]);
            }
            let m = xy_at(kids[0].children()?, 1, "query quadrant")?;
            let want = match (p[1] >= m[1] as f64, p[0] >= m[0] as f64) {
                (true, true) => "top_right",
                (true, false) => "top_left",
                (false, true) => "bottom_right",
                (false, false) => "bottom_left",
            };
            n = kids
                .iter()
                .find(|k| k.name() == Some(want))
                .with_context(|| format!("regions atlas: verify {}: query node without {want}", self.name))?;
        }
    }

    /// Port of the POC `lookup`: walk the segments crossed by the ray probe -> `p`, ordered along the ray,
    /// toggling the owner (groundtypes: the crossing always enters the right owner). `None` when a crossed
    /// segment does not touch the current owner.
    fn lookup(&self, p: P) -> Result<Option<u32>> {
        let cell = self.find_cell(p)?;
        let c = cell.children()?;
        ensure!(c.len() >= 3, "regions atlas: verify {}: query cell has {} children", self.name, c.len());
        let pr = widen(xy_at(c, 0, "query cell")?);
        let mut cur = c[1].as_u32().context("query cell owner")?;
        let segs = c[2].u32_array().context("query cell segments")?;
        ensure!(segs.len() % 4 == 0, "regions atlas: verify {}: query cell segment array of {} values", self.name, segs.len());
        let mut hits: Vec<(f64, u32, u32)> = Vec::new();
        for s in segs.chunks_exact(4) {
            let (v1, v2, ab1, ab2) = (s[0], s[1], s[2], s[3]);
            let a = *self.v.get(v1 as usize).with_context(|| format!("regions atlas: verify {}: segment vertex {v1} out of range", self.name))?;
            let b = *self.v.get(v2 as usize).with_context(|| format!("regions atlas: verify {}: segment vertex {v2} out of range", self.name))?;
            if orient(a, b, pr) <= 0.0 {
                bail!("regions atlas: verify {}: segment not oriented with the probe on its left", self.name);
            }
            let (d1, d2, d3, d4) = (orient(pr, p, a), orient(pr, p, b), orient(a, b, pr), orient(a, b, p));
            if (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0) && d1 * d2 * d3 * d4 != 0.0 {
                hits.push((d3 / (d3 - d4), ab1, ab2));
            }
        }
        // Python sorts (t, ab1, ab2) with signed integers
        hits.sort_by(|x, y| x.0.total_cmp(&y.0).then((x.1 as i32).cmp(&(y.1 as i32))).then((x.2 as i32).cmp(&(y.2 as i32))));
        for (_, ab1, ab2) in hits {
            if self.with_left {
                cur = if cur == ab1 {
                    ab2
                } else if cur == ab2 {
                    ab1
                } else {
                    return Ok(None);
                };
            } else {
                cur = ab1;
            }
        }
        Ok(Some(cur))
    }
}

// ---------------------------------------------------------------------------------------------
// verify_dataset
// ---------------------------------------------------------------------------------------------

struct Spec {
    name: &'static str,
    with_faces: bool,
    with_left: bool,
}

fn verify_dataset<'a>(ctx: &Ctx, rd: &'a Node, qi: &'a Node, spec: &Spec, rng: &mut Rng, samples: usize) -> Result<(Source<'a>, Json)> {
    let name = spec.name;
    let src = Source::new(rd, name, spec.with_faces)?;
    let w = rect_of(ctx.world_out[0], ctx.world_out[1]);
    let wa = (w[1][0] - w[0][0]) * (w[1][1] - w[0][1]);
    let mut rep: Map<String, Json> = Map::new();

    // ---- topology: every directed edge has its reverse unless it runs along the world border
    let mut unmatched: Vec<((u32, u32), u64)> =
        src.eo.iter().filter(|(k, _)| !src.eo.contains_key(&(k.1, k.0))).map(|(&e, info)| (e, info.rank)).collect();
    unmatched.sort_by_key(|u| u.1);
    for &((a, b), _) in &unmatched {
        let (pa, pb) = (src.v[a as usize], src.v[b as usize]);
        let on_border = (pa[0] == pb[0] && (pa[0] == w[0][0] || pa[0] == w[1][0])) || (pa[1] == pb[1] && (pa[1] == w[0][1] || pa[1] == w[1][1]));
        if !on_border {
            bail!("regions atlas: verify {name}: unmatched edge off the world border {pa:?}->{pb:?}");
        }
    }
    rep.insert("unmatched_edges_all_on_world_border".into(), json!(unmatched.len()));

    // ---- rings: orientation, outline bbox, connectivity
    let mut signed = 0.0f64;
    let mut orient_count: BTreeMap<String, u64> = BTreeMap::new();
    let (mut bbox_bad, mut conn_bad) = (0u64, 0u64);
    for a in &src.areas {
        let outlines = child_named(a.item, "outlines")?.items()?;
        ensure!(outlines.len() == a.rings.len(), "regions atlas: verify {name}: area {:?} outline count changed", a.key);
        let mut n_outer = 0usize;
        for (ring, o) in a.rings.iter().zip(outlines) {
            let pts = src.ring_points(&ring.ids);
            let s2 = shoelace2(&pts);
            signed += s2 / 2.0;
            *orient_count.entry(ring_label(ring.outer, s2)).or_insert(0) += 1;
            n_outer += ring.outer as usize;
            let mut bb: Rect64 = [[f64::INFINITY; 2], [f64::NEG_INFINITY; 2]];
            for p in &pts {
                for k in 0..2 {
                    bb[0][k] = bb[0][k].min(p[k]);
                    bb[1][k] = bb[1][k].max(p[k]);
                }
            }
            let (olo, ohi) = (xy_at(o, 1, "outline bbox")?, xy_at(o, 2, "outline bbox")?);
            if rect_of(olo, ohi) != bb {
                bbox_bad += 1;
            }
            let derived = connectivity_runs(&ring.ids, |from, to| match src.eo.get(&(from, to)) {
                Some(rev) => (rev.ri << 16) | rev.ai,
                None => NONE_AB,
            });
            let conn = child_named(o, "connectivity")?.items()?;
            let mut stored: Vec<[u32; 3]> = Vec::with_capacity(conn.len());
            for item in conn {
                ensure!(item.len() >= 3, "regions atlas: verify {name}: connectivity item with {} values", item.len());
                stored.push([item[0].as_u32()?, item[1].as_u32()?, item[2].as_u32()?]);
            }
            if stored != derived {
                conn_bad += 1;
            }
        }
        if n_outer != 1 {
            bail!("regions atlas: verify {name}: area {:?} has {n_outer} outer rings", a.key);
        }
    }
    rep.insert("ring_orientation".into(), json!(orient_count));
    if orient_count.keys().any(|k| k != "outer/ccw" && k != "hole/cw") {
        bail!("regions atlas: verify {name}: wrong ring orientation {orient_count:?}");
    }
    rep.insert("outline_bbox_mismatch".into(), json!(bbox_bad));
    rep.insert("connectivity_mismatch".into(), json!(conn_bad));
    rep.insert("sum_signed_ring_area".into(), json!(signed));
    rep.insert("world_area".into(), json!(wa));
    if bbox_bad > 0 || conn_bad > 0 || (signed - wa).abs() > 1e-6 * wa {
        bail!("regions atlas: verify {name}: bbox/connectivity/area-sum check failed {}", Json::Object(rep));
    }

    // ---- area bboxes and region bboxes
    let mut by_region: Vec<Vec<usize>> = vec![Vec::new(); src.n_regions];
    for (i, a) in src.areas.iter().enumerate() {
        by_region[a.key.0].push(i);
    }
    for (ri, lst) in by_region.iter().enumerate() {
        ensure!(!lst.is_empty(), "regions atlas: verify {name}: region {ri} has no areas");
        let mut lo = [f32::INFINITY; 2];
        let mut hi = [f32::NEG_INFINITY; 2];
        for &ix in lst {
            let a = &src.areas[ix];
            let (alo, ahi) = (xy_at(a.item, 3, "area bbox")?, xy_at(a.item, 4, "area bbox")?);
            let outlines = child_named(a.item, "outlines")?.items()?;
            let first = outlines.first().with_context(|| format!("regions atlas: verify {name}: area {:?} has no outline", a.key))?;
            let (olo, ohi) = (xy_at(first, 1, "outline bbox")?, xy_at(first, 2, "outline bbox")?);
            if alo != olo || ahi != ohi {
                bail!("regions atlas: verify {name}: area bbox differs from its outer outline bbox {:?}", a.key);
            }
            for k in 0..2 {
                lo[k] = lo[k].min(alo[k]);
                hi[k] = hi[k].max(ahi[k]);
            }
        }
        let (rlo, rhi) = (xy_at(src.regs[ri], 2, "region bbox")?, xy_at(src.regs[ri], 3, "region bbox")?);
        if rlo != lo || rhi != hi {
            bail!("regions atlas: verify {name}: region bbox differs from union of areas {}", src.region_key(ri)?);
        }
    }

    // ---- faces
    if spec.with_faces {
        let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (i, a) in src.areas.iter().enumerate() {
            let fid = a.faces_id.with_context(|| format!("regions atlas: verify {name}: area {:?} has no faces", a.key))?;
            groups.entry((a.key.0, fid)).or_default().push(i);
        }
        let (mut n_groups, mut n_tris, mut n_nested) = (0u64, 0u64, 0u64);
        for (&(ri, _), members) in &groups {
            let t = src.areas[members[0]].faces.as_ref().with_context(|| format!("regions atlas: verify {name}: region {ri} without faces"))?;
            if t.len() % 3 != 0 || t.iter().any(|&i| i as usize >= src.v.len()) {
                bail!("regions atlas: verify {name}: bad face indices in region {ri}");
            }
            let mut tri_area = 0.0f64;
            for k in (0..t.len()).step_by(3) {
                let (a, b, c) = (src.v[t[k] as usize], src.v[t[k + 1] as usize], src.v[t[k + 2] as usize]);
                let d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                if d >= 0.0 {
                    bail!("regions atlas: verify {name}: non-CW or degenerate face in region {ri}");
                }
                tri_area += -d / 2.0;
            }
            let mut poly_area = 0.0f64;
            for &m in members {
                poly_area += src.poly(src.areas[m].key)?.area;
            }
            n_groups += 1;
            n_tris += (t.len() / 3) as u64;
            // outer minus every hole, nested or not (baseline mesh semantics for the nested-hole defect)
            let mut plain = 0.0f64;
            for &m in members {
                for ring in &src.areas[m].rings {
                    plain += shoelace2(&src.ring_points(&ring.ids)) / 2.0;
                }
            }
            let mut ok = (tri_area - poly_area).abs() <= 1e-4 * poly_area.max(1.0);
            if !ok && (tri_area - plain).abs() <= 1e-4 * plain.max(1.0) {
                n_nested += 1;
                ok = true;
            }
            if !ok {
                let keys: Vec<(usize, usize)> = members.iter().map(|&m| src.areas[m].key).collect();
                bail!("regions atlas: verify {name}: face area {tri_area} != area polygons {poly_area} / {plain} (region {ri}, areas {keys:?})");
            }
        }
        let mut faces: Map<String, Json> = Map::new();
        faces.insert("groups".into(), json!(n_groups));
        faces.insert("triangles".into(), json!(n_tris));
        if n_nested > 0 {
            faces.insert("groups_matching_stock_nested_hole_semantics".into(), json!(n_nested));
        }
        rep.insert("faces".into(), Json::Object(faces));
    }

    // ---- query structure
    let tree = QueryTree::new(&src, qi, spec.with_left)?;
    let mut st = WalkStats::default();
    walk(tree.root, &mut st)?;
    let qc = qi.children()?;
    ensure!(qc.len() >= 3, "regions atlas: verify {name}: query_info has {} children", qc.len());
    let header = [qc[0].as_u32().context("query_info node count")?, qc[1].as_u32().context("query_info cell count")?];
    let query = json!({"nodes": st.nodes, "cells": st.cells, "header": header, "bad_partition": st.bad_partition});
    rep.insert("query".into(), query.clone());
    if [st.nodes, st.cells] != [header[0] as u64, header[1] as u64] || st.bad_partition > 0 {
        bail!("regions atlas: verify {name}: query header/partition mismatch {query}");
    }
    let rc = tree.root.children()?;
    if rect_of(xy_at(rc, 0, "query root")?, xy_at(rc, 1, "query root")?) != w {
        bail!("regions atlas: verify {name}: query root bbox is not the world");
    }

    // ---- query lookups against the polygons
    let mut polys: Vec<Rc<AreaPoly>> = Vec::with_capacity(src.areas.len());
    for a in &src.areas {
        polys.push(src.poly(a.key)?);
    }
    let boxes: Vec<Rect64> = polys
        .iter()
        .map(|p| [[p.bbox[0][0] - BOUNDARY_EPS, p.bbox[0][1] - BOUNDARY_EPS], [p.bbox[1][0] + BOUNDARY_EPS, p.bbox[1][1] + BOUNDARY_EPS]])
        .collect();
    let index = GridIndex::new(w, boxes);
    let mut points: Vec<P> = Vec::with_capacity((ctx.plan.charts.len() + 1) * samples);
    for ci in 0..ctx.plan.charts.len() {
        let b = out_bounds(ctx, ci);
        for _ in 0..samples {
            let x = rng.uniform(b[0][0], b[1][0]);
            let y = rng.uniform(b[0][1], b[1][1]);
            points.push([x, y]);
        }
    }
    for _ in 0..samples {
        let x = rng.uniform(w[0][0], w[1][0]);
        let y = rng.uniform(w[0][1], w[1][1]);
        points.push([x, y]);
    }
    let (mut tested, mut bad, mut skipped, mut incons) = (0u64, 0u64, 0u64, 0u64);
    let mut examples: Vec<String> = Vec::new();
    for p in points {
        let cands = index.at(p);
        if cands.iter().any(|&i| near_boundary(&polys[i as usize], p, BOUNDARY_EPS)) {
            skipped += 1;
            continue;
        }
        let Some(got) = tree.lookup(p)? else {
            incons += 1;
            continue;
        };
        tested += 1;
        let want = ((got >> 16) as usize, (got & 0xFFFF) as usize);
        let hit = cands.iter().any(|&i| src.areas[i as usize].key == want && polys[i as usize].contains(p));
        if !hit {
            bad += 1;
            if examples.len() < 4 {
                let ms: Vec<(usize, usize)> =
                    cands.iter().filter(|&&i| polys[i as usize].contains(p)).map(|&i| src.areas[i as usize].key).collect();
                examples.push(format!("{p:?}: lookup {want:?}, polygons {ms:?}"));
            }
        }
    }
    let lookup = json!({"tested": tested, "mismatches": bad, "inconsistent": incons, "skipped_near_border": skipped});
    rep.insert("query_lookup".into(), lookup.clone());
    if bad > 0 || incons > 2u64.max(tested / 1000) {
        bail!("regions atlas: verify {name}: query lookup disagrees with polygons {lookup} {examples:?}");
    }
    drop(tree);
    drop(polys);
    Ok((src, Json::Object(rep)))
}

// ---------------------------------------------------------------------------------------------
// verify_unchanged_faces
// ---------------------------------------------------------------------------------------------

/// Groups copied verbatim: every output triangle equals the chart-mapped source triangle.
fn verify_unchanged_faces(ctx: &Ctx, od: &OutData, out_src: &Source) -> Result<Json> {
    let src = od.src;
    let (mut verbatim, mut retriangulated) = (0u64, 0u64);
    for (ri, areas) in od.regions.iter().enumerate() {
        for oa in areas {
            if oa.is_gap || !(oa.primary && od.nfrag.get(&oa.src).copied() == Some(1) && oa.frag.unchanged) {
                continue;
            }
            let sa = src.area(oa.src);
            let ix = out_src
                .area_ix
                .get(&(ri, oa.local))
                .with_context(|| format!("regions atlas: verify unchanged faces: output area ({ri}, {}) not in the written file", oa.local))?;
            let ob = &out_src.areas[*ix];
            let (Some(t), Some(u)) = (&sa.faces, &ob.faces) else {
                continue;
            };
            if t.len() != u.len() {
                continue; // re-triangulated group
            }
            let same = t.iter().zip(u).all(|(&s, &d)| match src.v.get(s as usize) {
                Some(&sp) => out_src.v.get(d as usize) == Some(&widen(ctx.oxy(oa.ci, sp))),
                None => false,
            });
            if same {
                verbatim += 1;
            } else {
                retriangulated += 1;
            }
        }
    }
    let mut c: Map<String, Json> = Map::new();
    if verbatim > 0 {
        c.insert("verbatim_groups_members_checked".into(), json!(verbatim));
    }
    if retriangulated > 0 {
        c.insert("retriangulated_members".into(), json!(retriangulated));
    }
    Ok(Json::Object(c))
}

// ---------------------------------------------------------------------------------------------
// verify_sections
// ---------------------------------------------------------------------------------------------

fn chart_by_key<'c>(ctx: &'c Ctx, key: &str) -> Result<&'c Chart> {
    Ok(&ctx.plan.charts[ctx.chart_index(key)?])
}

fn add_counts(total: &mut BTreeMap<String, u64>, prefix: &str, counts: BTreeMap<String, u64>) {
    for (k, v) in counts {
        *total.entry(format!("{prefix}.{k}")).or_insert(0) += v;
    }
}

/// Label point and every slot XY point of a `settlement_and_slots` record, in storage order.
fn settlement_points(node: &Node) -> Result<Vec<[f32; 2]>> {
    let kids = node.children()?;
    let mut out = vec![kids.first().context("empty settlement_and_slots record")?.as_xy().context("settlement label point")?];
    for s in kids {
        if let Node::Array { name, items, .. } = s {
            if name == "slot_descriptions" {
                for item in items {
                    for z in item {
                        if let Node::Value(Value::XY(p)) = z {
                            out.push(*p);
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn verify_sections(ctx: &Ctx, pairs: &SectionPairs, main_src: &Source, tree: &QueryTree) -> Result<Json> {
    let plan = &ctx.plan;
    let mut c: BTreeMap<String, u64> = BTreeMap::new();
    for p in &pairs.theatre {
        let chart = chart_by_key(ctx, &p.chart)?;
        let r = check_scaled_items(&p.name, &p.src, &p.dst, plan, chart, "/root")
            .with_context(|| format!("regions atlas: verify sections: {} of chart {}", p.name, p.chart))?;
        add_counts(&mut c, "theatres", r);
    }
    for (key, before, after) in &pairs.trade {
        let chart = chart_by_key(ctx, key)?;
        let r = check_scaled_items("trade_nodes", before, after, plan, chart, "/root/trade_nodes")
            .with_context(|| format!("regions atlas: verify sections: trade node of chart {key}"))?;
        add_counts(&mut c, "trade", r);
    }
    for p in &pairs.mountain {
        let chart = chart_by_key(ctx, &p.chart)?;
        let parent = "/root/mountain_data/MOUNTAIN_OUTLINES";
        let r = check_scaled_items("OUTLINE_BLOCK", &p.outline_src, &p.outline_dst, plan, chart, parent)
            .with_context(|| format!("regions atlas: verify sections: mountain outline of chart {}", p.chart))?;
        let r2 = check_scaled_items("BOUNDS_BLOCK", &p.bounds_src, &p.bounds_dst, plan, chart, parent)
            .with_context(|| format!("regions atlas: verify sections: mountain bounds of chart {}", p.chart))?;
        add_counts(&mut c, "mountains", r);
        add_counts(&mut c, "mountains", r2);
    }
    for p in &pairs.settlement {
        let chart = plan.charts.get(p.chart).with_context(|| format!("regions atlas: verify sections: settlement chart index {}", p.chart))?;
        let r = check_scaled(&p.src, &p.dst, plan, chart, "/root/region_data/regions[]")
            .with_context(|| format!("regions atlas: verify sections: settlement of region {}", p.region))?;
        add_counts(&mut c, "settlements", r);
    }

    // exact alignment: every settlement/slot point sits in the same region id before and after, inside its theatre
    let (mut mism, mut tested, mut outside_theatre, mut src_outside, mut changed_status, mut not_in_own_region) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for p in &pairs.settlement {
        let ri = p.region;
        ensure!(ri < main_src.regs.len(), "regions atlas: verify sections: settlement region {ri} out of range");
        let ci = p.chart;
        let pts_old = settlement_points(&p.src)?;
        let pts_new = settlement_points(&p.dst)?;
        ensure!(
            pts_old.len() == pts_new.len(),
            "regions atlas: verify sections: settlement of region {ri} changed its point count {} -> {}",
            pts_old.len(),
            pts_new.len()
        );
        let tk = main_src.regs[ri]
            .iter()
            .find_map(|x| match x {
                Node::Value(Value::I32(v)) => Some(*v),
                _ => None,
            })
            .with_context(|| format!("regions atlas: verify sections: region {ri} has no theatre key"))?
            .to_string();
        let rect = plan.charts.iter().find(|ch| ch.theatres.iter().any(|t| *t == tk)).and_then(|ch| ch.theatre_rects.get(&tk)).copied();
        for (po, pn) in pts_old.iter().zip(&pts_new) {
            tested += 1;
            let pw = widen(*po);
            let ro = main_src.locate(pw)?.0;
            let got = tree.lookup(widen(*pn))?;
            if got.is_none_or(|g| (g >> 16) as usize != ro) {
                mism += 1;
            }
            not_in_own_region += (ro != ri) as u64;
            if let Some(rect) = rect {
                let lo = ctx.oxy(ci, rect[0]);
                let hi = ctx.oxy(ci, rect[1]);
                let out_o = !(lo[0] <= pn[0] && pn[0] <= hi[0] && lo[1] <= pn[1] && pn[1] <= hi[1]);
                let src_o = !(rect[0][0] <= pw[0] && pw[0] <= rect[1][0] && rect[0][1] <= pw[1] && pw[1] <= rect[1][1]);
                outside_theatre += out_o as u64;
                src_outside += src_o as u64;
                changed_status += (out_o != src_o) as u64;
            }
        }
    }
    c.insert("settlement_points_tested".into(), tested);
    c.insert("settlement_points_region_mismatch".into(), mism);
    c.insert("settlement_points_outside_owner_theatre".into(), outside_theatre);
    c.insert("settlement_points_outside_owner_theatre_in_source".into(), src_outside);
    c.insert("settlement_points_theatre_membership_changed".into(), changed_status);
    c.insert("settlement_points_not_in_own_region_in_source".into(), not_in_own_region);
    if changed_status > 0 {
        bail!("regions atlas: verify sections: {changed_status} settlement/slot points changed their inside/outside-theatre status");
    }
    if mism > 0 {
        bail!("regions atlas: verify sections: {mism}/{tested} settlement/slot points resolve to a different region after the migration");
    }
    Ok(json!(c))
}

// ---------------------------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------------------------

/// Verify the freshly parsed written file. `main_src` is the original source `region_data` (source space);
/// `od_main` / `od_gt` are the assembled output datasets. `samples` random lookups per chart and over the
/// world are compared with the polygons of each dataset. Returns the POC `checks` report.
pub(super) fn verify_written(
    ctx: &Ctx,
    written: &Esf,
    main_src: &Source,
    od_main: &OutData,
    od_gt: &OutData,
    pairs: &SectionPairs,
    samples: usize,
) -> Result<Json> {
    let mut rng = Rng(SEED);
    let root = &written.root;
    let main_spec = Spec { name: "region_data", with_faces: true, with_left: true };
    let gt_spec = Spec { name: "groundtypes.region_data", with_faces: false, with_left: false };
    let main_qi = root.get("query_info")?;
    let (main_out, mut vmain) = verify_dataset(ctx, root.get("region_data")?, main_qi, &main_spec, &mut rng, samples)?;
    let gt = root.get("groundtypes")?;
    let (_gt_out, mut vgt) = verify_dataset(ctx, gt.get("region_data")?, gt.get("query_info")?, &gt_spec, &mut rng, samples)?;
    let tree = QueryTree::new(&main_out, main_qi, true)?;
    let sections = verify_sections(ctx, pairs, main_src, &tree)?;
    let unchanged = verify_unchanged_faces(ctx, od_main, &main_out)?;
    vmain["dropped_areas"] = json!(od_main.dropped.len());
    vmain["absorbed_slivers"] = json!(od_main.absorbed.len());
    vgt["dropped_areas"] = json!(od_gt.dropped.len());
    vgt["absorbed_slivers"] = json!(od_gt.absorbed.len());
    Ok(json!({
        "region_data": vmain,
        "groundtypes": vgt,
        "sections": sections,
        "unchanged_group_faces": unchanged,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf::{arr_u32, u32v, xyv};

    fn cell(probe: [f32; 2], owner: u32, segs: &[u32]) -> Node {
        Node::record("cell", 0, vec![xyv(probe[0], probe[1]), u32v(owner), arr_u32(segs)])
    }

    /// One vertical edge x = 5 splits the 10x10 world: region 0 on the left, region 1 on the right.
    fn tiny(with_left: bool) -> (Node, Vec<P>) {
        let left = if with_left { 0 } else { u32::MAX };
        let root = Node::record("node", 0, vec![xyv(0.0, 0.0), xyv(10.0, 10.0), cell([0.0, 0.0], 0, &[0, 1, 1 << 16, left])]);
        (root, vec![[5.0, 0.0], [5.0, 10.0]])
    }

    #[test]
    fn lookup_toggles_owner_across_crossed_segment() {
        let (root, v) = tiny(true);
        let tree = QueryTree { name: "t", root: &root, v: &v, with_left: true };
        assert_eq!(tree.lookup([2.0, 5.0]).unwrap(), Some(0));
        assert_eq!(tree.lookup([7.0, 5.0]).unwrap(), Some(1 << 16));
    }

    #[test]
    fn groundtypes_lookup_enters_the_right_owner() {
        let (root, v) = tiny(false);
        let tree = QueryTree { name: "t", root: &root, v: &v, with_left: false };
        assert_eq!(tree.lookup([2.0, 5.0]).unwrap(), Some(0));
        assert_eq!(tree.lookup([7.0, 5.0]).unwrap(), Some(1 << 16));
    }

    #[test]
    fn lookup_reports_inconsistent_owner_and_misoriented_segment() {
        let (_, v) = tiny(true);
        // the probe owner is neither side of the crossed segment
        let odd = Node::record("node", 0, vec![xyv(0.0, 0.0), xyv(10.0, 10.0), cell([0.0, 0.0], 9, &[0, 1, 1 << 16, 0])]);
        let t2 = QueryTree { name: "t", root: &odd, v: &v, with_left: true };
        assert_eq!(t2.lookup([7.0, 5.0]).unwrap(), None);
        // probe on the right of the stored segment direction
        let flipped = Node::record("node", 0, vec![xyv(0.0, 0.0), xyv(10.0, 10.0), cell([0.0, 0.0], 0, &[1, 0, 1 << 16, 0])]);
        let t3 = QueryTree { name: "t", root: &flipped, v: &v, with_left: true };
        assert!(t3.lookup([7.0, 5.0]).unwrap_err().to_string().contains("not oriented"));
    }

    #[test]
    fn lookup_descends_by_the_first_childs_hi_corner() {
        let quad = |name: &str, lo: [f32; 2], hi: [f32; 2], owner: u32| {
            Node::record(name, 0, vec![xyv(lo[0], lo[1]), xyv(hi[0], hi[1]), cell(lo, owner, &[])])
        };
        let root = Node::record(
            "node",
            0,
            vec![
                xyv(0.0, 0.0),
                xyv(10.0, 10.0),
                quad("bottom_left", [0.0, 0.0], [5.0, 5.0], 1),
                quad("top_left", [0.0, 5.0], [5.0, 10.0], 2),
                quad("bottom_right", [5.0, 0.0], [10.0, 5.0], 3),
                quad("top_right", [5.0, 5.0], [10.0, 10.0], 4),
            ],
        );
        let v: Vec<P> = Vec::new();
        let tree = QueryTree { name: "t", root: &root, v: &v, with_left: true };
        assert_eq!(tree.lookup([1.0, 1.0]).unwrap(), Some(1));
        assert_eq!(tree.lookup([1.0, 9.0]).unwrap(), Some(2));
        assert_eq!(tree.lookup([9.0, 1.0]).unwrap(), Some(3));
        assert_eq!(tree.lookup([5.0, 5.0]).unwrap(), Some(4));
        let mut st = WalkStats::default();
        walk(&root, &mut st).unwrap();
        assert_eq!((st.nodes, st.cells, st.bad_partition), (5, 4, 0));
    }

    #[test]
    fn walk_flags_a_broken_quadrant_partition() {
        let quad = |name: &str, lo: [f32; 2], hi: [f32; 2]| Node::record(name, 0, vec![xyv(lo[0], lo[1]), xyv(hi[0], hi[1]), cell(lo, 0, &[])]);
        let root = Node::record(
            "node",
            0,
            vec![
                xyv(0.0, 0.0),
                xyv(10.0, 10.0),
                quad("bottom_left", [0.0, 0.0], [5.0, 5.0]),
                quad("top_left", [0.0, 5.0], [5.0, 10.0]),
                quad("bottom_right", [5.0, 0.0], [10.0, 5.0]),
                // does not start at the split point
                quad("top_right", [6.0, 5.0], [10.0, 10.0]),
            ],
        );
        let mut st = WalkStats::default();
        walk(&root, &mut st).unwrap();
        assert_eq!(st.bad_partition, 1);
    }

    #[test]
    fn connectivity_runs_merge_consecutive_owners_without_wrapping() {
        let owners = [5u32, 5, 7, 5];
        let ids = [10u32, 11, 12, 13];
        let runs = connectivity_runs(&ids, |from, to| {
            // reverse edge of position p goes from ids[p+1] to ids[p]
            let p = ids.iter().position(|&i| i == to).unwrap();
            assert_eq!(from, ids[(p + 1) % 4]);
            owners[p]
        });
        assert_eq!(runs, vec![[5, 0, 1], [7, 2, 2], [5, 3, 3]]);
    }

    #[test]
    fn orientation_labels_and_shoelace_sign() {
        let ccw = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let cw: Vec<P> = ccw.iter().rev().copied().collect();
        assert_eq!(shoelace2(&ccw), 2.0);
        assert_eq!(ring_label(true, shoelace2(&ccw)), "outer/ccw");
        assert_eq!(ring_label(false, shoelace2(&cw)), "hole/cw");
        assert_eq!(ring_label(true, shoelace2(&cw)), "outer/cw");
    }

    #[test]
    fn near_boundary_detects_close_edges_only() {
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let poly = AreaPoly::new(&[outer], &[]).unwrap();
        assert!(near_boundary(&poly, [5.0, 0.0005], 1e-3));
        assert!(near_boundary(&poly, [10.0005, 5.0], 1e-3));
        assert!(!near_boundary(&poly, [5.0, 5.0], 1e-3));
        assert!(!near_boundary(&poly, [5.0, 0.01], 1e-3));
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng(SEED);
        let mut b = Rng(SEED);
        for _ in 0..100 {
            let x = a.uniform(-3.0, 4.0);
            assert_eq!(x, b.uniform(-3.0, 4.0));
            assert!((-3.0..4.0).contains(&x));
        }
    }
}
