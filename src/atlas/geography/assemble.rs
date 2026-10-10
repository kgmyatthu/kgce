//! Assembly of the migrated region dataset from the clipped fragments: area/region ordering, the shared vertex
//! pool, rings, connectivity runs, faces and bounding boxes (accepted POC `regions_atlas_build.py`).

use super::clip::{Absorbed, Clipper, Ctx, Frag, GAP};
use super::geom::{AreaPoly, P, kept_holes, ring_area, widen};
use super::source::Source;
use crate::regionmap::geometry::earcut_with_holes;
use anyhow::{Context, Result, bail, ensure};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const NONE_AB: u32 = 0xFFFF_FFFF;
/// clipped pieces below this many SOURCE units^2 are absorbed into the neighbour with the longest border
pub const MIN_FRAGMENT_AREA: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Main,
    Groundtypes,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Main => "region_data",
            Kind::Groundtypes => "groundtypes",
        }
    }
}

pub struct OArea {
    /// source (region, area) whose record is the prototype
    pub src: (usize, usize),
    pub frag: Frag,
    pub primary: bool,
    pub is_gap: bool,
    pub local: usize,
    pub ci: usize,
    /// (outer flag, vertex ids)
    pub rings: Vec<(bool, Vec<u32>)>,
    /// per ring: (neighbour code, from, to) runs
    pub conn: Vec<Vec<[u32; 3]>>,
    pub faces: Vec<u32>,
    pub bbox: [[f32; 2]; 2],
    pub ring_bbox: Vec<[[f32; 2]; 2]>,
}

pub struct OutData<'a> {
    pub src: &'a Source<'a>,
    pub clip: Clipper<'a>,
    pub gap: Frag,
    pub regions: Vec<Vec<OArea>>,
    /// source (region, area) -> local output indices of its pieces (primary first)
    pub area_map: HashMap<(usize, usize), Vec<usize>>,
    pub nfrag: HashMap<(usize, usize), usize>,
    pub backdrop: (usize, usize),
    pub backdrop_why: String,
    pub gap_local: usize,
    pub dropped: Vec<serde_json::Value>,
    pub absorbed: Vec<Absorbed>,
    pub coords: Vec<[f32; 2]>,
    pool: HashMap<(u32, u32), u32>,
    pub directed: HashMap<(u32, u32), (u32, u32)>,
    /// directed edges in insertion (file) order
    pub directed_order: Vec<(u32, u32)>,
    pub unmatched: Vec<(u32, u32)>,
    pub stats: BTreeMap<String, u64>,
    pub faces_stats: BTreeMap<String, u64>,
}

fn fbits(v: f32) -> u32 {
    if v == 0.0 { 0 } else { v.to_bits() }
}

impl<'a> OutData<'a> {
    pub fn vid(&mut self, xy: [f32; 2]) -> u32 {
        let key = (fbits(xy[0]), fbits(xy[1]));
        if let Some(&i) = self.pool.get(&key) {
            return i;
        }
        let i = self.coords.len() as u32;
        self.pool.insert(key, i);
        self.coords.push(xy);
        i
    }

    pub fn pts(&self, ids: &[u32]) -> Vec<P> {
        ids.iter().map(|&i| widen(self.coords[i as usize])).collect()
    }

    /// Polygon of one output area (single outer ring, nested holes dropped).
    pub fn poly_of(&self, oa: &OArea) -> Result<AreaPoly> {
        let outers: Vec<Vec<P>> = oa.rings.iter().filter(|r| r.0).map(|r| self.pts(&r.1)).collect();
        let holes: Vec<Vec<P>> = oa.rings.iter().filter(|r| !r.0).map(|r| self.pts(&r.1)).collect();
        AreaPoly::new(&outers, &holes).with_context(|| format!("{}: output area {:?}", self.src.name, oa.src))
    }
}

/// Area that owns the world gap. `spec` is `region[:area]` (the plan's background region).
fn choose_backdrop(src: &Source, spec: Option<&str>, kind: Kind, dead: &HashSet<(usize, usize)>) -> Result<((usize, usize), String)> {
    let largest = |ri: usize| -> Result<(usize, usize)> {
        let mut best: Option<(f64, (usize, usize))> = None;
        for ai in 0..src.counts[ri] {
            let a = src.poly((ri, ai))?.area;
            if best.is_none_or(|(s, _)| a > s) {
                best = Some((a, (ri, ai)));
            }
        }
        best.map(|b| b.1).with_context(|| format!("region {ri} has no areas"))
    };
    if let Some(spec) = spec {
        let (name, ai) = spec.split_once(':').unwrap_or((spec, ""));
        let mut found = Vec::new();
        for ri in 0..src.n_regions {
            if src.region_key(ri)? == name {
                found.push(ri);
            }
        }
        ensure!(found.len() == 1, "regions atlas: backdrop region {name:?} not found");
        let ri = found[0];
        if !ai.is_empty() {
            let ai: usize = ai.parse().with_context(|| format!("backdrop area {ai:?}"))?;
            ensure!(ai < src.counts[ri], "regions atlas: backdrop area {name}:{ai} does not exist");
            return Ok(((ri, ai), "plan background_region".to_string()));
        }
        return Ok((largest(ri)?, "plan background_region".to_string()));
    }
    if kind == Kind::Groundtypes {
        for ri in 0..src.n_regions {
            if src.region_key(ri)? == "grassland" {
                return Ok((largest(ri)?, "evidence: baseline sea is 99.8% grassland area 0".to_string()));
            }
        }
        bail!("regions atlas: groundtypes: no grassland region");
    }
    let mut sea: Vec<(usize, usize)> = Vec::new();
    for a in &src.areas {
        if src.region_kind(a.key.0)? == "sea" {
            sea.push(a.key);
        }
    }
    ensure!(!sea.is_empty(), "regions atlas: {}: no sea areas", src.name);
    if let Some(ri) = (0..src.n_regions).find(|&ri| src.region_key(ri).map(|k| k == "southern_ocean").unwrap_or(false)) {
        if sea.contains(&(ri, 0)) {
            return Ok(((ri, 0), "southern_ocean area 0".to_string()));
        }
    }
    let pick = |set: Vec<(usize, usize)>| -> Result<(usize, usize)> {
        let mut best: Option<(f64, (usize, usize))> = None;
        for k in set {
            let a = src.poly(k)?.area;
            if best.is_none_or(|(s, _)| a > s) {
                best = Some((a, k));
            }
        }
        best.map(|b| b.1).context("no candidate area")
    };
    let dd: Vec<(usize, usize)> = sea.iter().copied().filter(|k| dead.contains(k)).collect();
    if !dd.is_empty() {
        return Ok((pick(dd)?, "largest sea area that no chart keeps".to_string()));
    }
    Ok((pick(sea)?, "largest sea area".to_string()))
}

pub fn assemble<'a>(ctx: &'a Ctx, src: &'a Source<'a>, backdrop_spec: Option<&str>, kind: Kind) -> Result<OutData<'a>> {
    let scale2 = ctx.plan.scale * ctx.plan.scale;
    let mut clip = Clipper::new(ctx, src);
    clip.run()?;
    let mut frags = clip.fragments(MIN_FRAGMENT_AREA * scale2)?;
    let gap = clip.build_gap()?;
    let nch = ctx.plan.charts.len();
    let mut pieces_of: HashMap<(usize, usize), Vec<Frag>> = HashMap::new();
    for ri in 0..src.n_regions {
        for ai in 0..src.counts[ri] {
            let mut list = Vec::new();
            for ci in 0..nch {
                if let Some(v) = frags.remove(&((ri, ai), ci)) {
                    list.extend(v);
                }
            }
            pieces_of.insert((ri, ai), list);
        }
    }
    ensure!(frags.is_empty(), "regions atlas: {}: fragments of unknown areas remain", src.name);
    let dead: HashSet<(usize, usize)> = pieces_of.iter().filter(|(_, v)| v.is_empty()).map(|(k, _)| *k).collect();
    let (backdrop, backdrop_why) = choose_backdrop(src, backdrop_spec, kind, &dead)?;
    let mut blockers: Vec<serde_json::Value> = Vec::new();
    let mut dropped: Vec<serde_json::Value> = Vec::new();
    let mut regions: Vec<Vec<OArea>> = Vec::new();
    let mut nfrag: HashMap<(usize, usize), usize> = HashMap::new();
    let mut gap_local: Option<usize> = None;
    for ri in 0..src.n_regions {
        let mut prim: Vec<OArea> = Vec::new();
        let mut extra: Vec<OArea> = Vec::new();
        for ai in 0..src.counts[ri] {
            let fl = pieces_of.remove(&(ri, ai)).unwrap_or_default();
            nfrag.insert((ri, ai), fl.len());
            if fl.is_empty() {
                if (ri, ai) == backdrop {
                    prim.push(new_area(backdrop, gap.clone(), true, true, GAP));
                    nfrag.insert((ri, ai), 1);
                    continue;
                }
                let item = src.area((ri, ai)).item;
                let lo = item.get(3).context("area without bbox")?.as_xy()?;
                let hi = item.get(4).context("area without bbox")?.as_xy()?;
                dropped.push(json!({
                    "kind": "area_without_geometry",
                    "region_index": ri,
                    "region": src.region_key(ri)?,
                    "area": ai,
                    "source_bbox": [[lo[0], lo[1]], [hi[0], hi[1]]],
                }));
                continue;
            }
            let best_ix = (0..fl.len()).fold(0, |b, i| if fl[i].area > fl[b].area { i } else { b });
            let mut rest: Vec<Frag> = Vec::new();
            let mut best: Option<Frag> = None;
            for (i, f) in fl.into_iter().enumerate() {
                if i == best_ix {
                    best = Some(f);
                } else {
                    rest.push(f);
                }
            }
            rest.sort_by(|a, b| (a.ci, -a.area).partial_cmp(&(b.ci, -b.area)).unwrap());
            let best = best.unwrap();
            let ci = best.ci;
            prim.push(new_area((ri, ai), best, true, false, ci));
            for f in rest {
                let ci = f.ci;
                extra.push(new_area((ri, ai), f, false, false, ci));
            }
        }
        let mut areas: Vec<OArea> = prim;
        areas.extend(extra);
        if ri == backdrop.0 && !areas.iter().any(|a| a.is_gap) {
            areas.push(new_area(backdrop, gap.clone(), false, true, GAP));
        }
        if areas.is_empty() {
            blockers.push(json!({
                "kind": "region_without_geometry",
                "region_index": ri,
                "region": src.region_key(ri)?,
                "note": "region id would have no real polygon; a crop must keep real geometry of one of its areas",
            }));
        }
        for (k, oa) in areas.iter_mut().enumerate() {
            oa.local = k;
            if oa.is_gap {
                gap_local = Some(k);
            }
        }
        regions.push(areas);
    }
    if !blockers.is_empty() {
        bail!(
            "regions atlas: {}: {} blockers (regions without geometry): {}",
            src.name,
            blockers.len(),
            serde_json::to_string(&blockers).unwrap_or_default()
        );
    }
    let mut area_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for areas in &regions {
        for oa in areas {
            area_map.entry(oa.src).or_default().push(oa.local);
        }
    }
    for areas in &regions {
        for oa in areas {
            ensure!(
                oa.frag.area >= 1e-6,
                "regions atlas: {}: piece of {:?} chart {} has output area {}",
                src.name,
                oa.src,
                oa.ci,
                oa.frag.area
            );
        }
    }
    let absorbed = clip.absorbed.clone();
    let mut stats = BTreeMap::new();
    stats.insert("absorbed_slivers".to_string(), absorbed.len() as u64);
    let mut od = OutData {
        src,
        clip,
        gap,
        regions,
        area_map,
        nfrag,
        backdrop,
        backdrop_why,
        gap_local: gap_local.context("the backdrop area produced no world gap polygon")?,
        dropped,
        absorbed,
        coords: Vec::new(),
        pool: HashMap::new(),
        directed: HashMap::new(),
        directed_order: Vec::new(),
        unmatched: Vec::new(),
        stats,
        faces_stats: BTreeMap::new(),
    };
    vertices_and_rings(&mut od)?;
    connectivity(&mut od)?;
    faces(ctx, &mut od)?;
    bboxes(&mut od);
    Ok(od)
}

fn new_area(src: (usize, usize), frag: Frag, primary: bool, is_gap: bool, ci: usize) -> OArea {
    OArea {
        src,
        frag,
        primary,
        is_gap,
        local: 0,
        ci,
        rings: Vec::new(),
        conn: Vec::new(),
        faces: Vec::new(),
        bbox: [[0.0; 2]; 2],
        ring_bbox: Vec::new(),
    }
}

fn vertices_and_rings(od: &mut OutData) -> Result<()> {
    let name = od.src.name;
    let mut regions = std::mem::take(&mut od.regions);
    for (ri, areas) in regions.iter_mut().enumerate() {
        for oa in areas.iter_mut() {
            let mut rings: Vec<(bool, Vec<usize>)> = vec![(true, oa.frag.outer.clone())];
            rings.extend(oa.frag.holes.iter().map(|h| (false, h.clone())));
            for (flag, nodes) in rings {
                let ids: Vec<u32> = nodes.iter().map(|&n| od.clip.nout[n]).collect::<Vec<_>>().into_iter().map(|xy| od.vid(xy)).collect();
                ensure!(ids.len() >= 3, "regions atlas: {name}: ring with {} vertices in {:?}", ids.len(), oa.src);
                for k in 0..ids.len() {
                    ensure!(
                        ids[k] != ids[(k + 1) % ids.len()],
                        "regions atlas: {name}: zero-length ring edge in {:?} chart {} at {:?} (two nodes collapsed to one f32 point)",
                        oa.src,
                        oa.ci,
                        od.coords[ids[k] as usize]
                    );
                }
                for k in 0..ids.len() {
                    let e = (ids[k], ids[(k + 1) % ids.len()]);
                    if let Some(prev) = od.directed.insert(e, (ri as u32, oa.local as u32)) {
                        bail!(
                            "regions atlas: {name}: directed edge {e:?} claimed twice ({prev:?} and {:?})",
                            (ri, oa.local)
                        );
                    }
                    od.directed_order.push(e);
                }
                oa.rings.push((flag, ids));
            }
        }
    }
    od.regions = regions;
    Ok(())
}

fn connectivity(od: &mut OutData) -> Result<()> {
    let name = od.src.name;
    let mut unmatched: Vec<(u32, u32)> = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for areas in od.regions.iter_mut() {
        for oa in areas.iter_mut() {
            oa.conn.clear();
            for (_, ids) in &oa.rings {
                let n = ids.len();
                let mut runs: Vec<[u32; 3]> = Vec::new();
                for p in 0..n {
                    let rev = od.directed.get(&(ids[(p + 1) % n], ids[p]));
                    let ab = match rev {
                        Some(r) => (r.0 << 16) | r.1,
                        None => {
                            let e = (ids[p], ids[(p + 1) % n]);
                            if seen.insert(e) {
                                unmatched.push(e);
                            }
                            NONE_AB
                        }
                    };
                    match runs.last_mut() {
                        Some(last) if last[0] == ab => last[2] = p as u32,
                        _ => runs.push([ab, p as u32, p as u32]),
                    }
                }
                oa.conn.push(runs);
            }
        }
    }
    let w = od.clip.ctx.world_out;
    for &(a, b) in &unmatched {
        let (pa, pb) = (od.coords[a as usize], od.coords[b as usize]);
        let on_border = (pa[0] == pb[0] && (pa[0] == w[0][0] || pa[0] == w[1][0])) || (pa[1] == pb[1] && (pa[1] == w[0][1] || pa[1] == w[1][1]));
        ensure!(on_border, "regions atlas: {name}: ring edge {pa:?}->{pb:?} has no neighbour and is not on the world border");
    }
    od.unmatched = unmatched;
    Ok(())
}

/// A stock face array is copied verbatim only when it triangulates exactly its group's polygons (relative 1e-4,
/// plain or nested-hole semantics) and every vertex lies inside the chart's crop.
fn stock_group_ok(ctx: &Ctx, src: &Source, faces: &[u32], sset: &[(usize, usize)], ci: usize) -> Result<bool> {
    if faces.len() % 3 != 0 {
        return Ok(false);
    }
    let v = &src.v;
    let mut ta = 0.0;
    for t in faces.chunks_exact(3) {
        let (a, b, c) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
        ta += -((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0;
    }
    let mut pa = 0.0;
    for m in sset {
        pa += src.poly(*m)?.area;
    }
    let mut plain = 0.0;
    for m in sset {
        for r in &src.area(*m).rings {
            let pts = src.ring_points(&r.ids);
            plain += crate::regionmap::geometry::signed_area2(&pts) / 2.0;
        }
    }
    if !((ta - pa).abs() <= 1e-4 * pa.max(1.0) || (ta - plain).abs() <= 1e-4 * plain.max(1.0)) {
        return Ok(false);
    }
    let u = &ctx.unions[ci];
    let verts: HashSet<u32> = faces.iter().copied().collect();
    Ok(verts.iter().all(|&i| u.contains_point(v[i as usize])))
}

/// Triangulate one output area (clockwise triangles of vertex ids).
fn triangulate(od: &OutData, oa: &OArea) -> Result<(Vec<u32>, u64)> {
    let name = od.src.name;
    let outer_ids: Vec<u32> = oa.rings.iter().filter(|r| r.0).flat_map(|r| r.1.iter().copied()).collect();
    let hole_ids: Vec<&Vec<u32>> = oa.rings.iter().filter(|r| !r.0).map(|r| &r.1).collect();
    let outer_pts = od.pts(&outer_ids);
    let hole_pts: Vec<Vec<P>> = hole_ids.iter().map(|h| od.pts(h)).collect();
    let kept = kept_holes(&hole_pts);
    let polygon_area = ring_area(&outer_pts) - kept.iter().map(|&i| ring_area(&hole_pts[i])).sum::<f64>();
    let mut rings: Vec<Vec<P>> = vec![outer_pts.clone()];
    let mut flat: Vec<u32> = outer_ids.clone();
    for &i in &kept {
        rings.push(hole_pts[i].clone());
        flat.extend(hole_ids[i].iter().copied());
    }
    let finish = |tris: Vec<[u32; 3]>| -> (Vec<u32>, f64, u64) {
        let mut out = Vec::new();
        let mut area = 0.0;
        let mut nz = 0u64;
        for mut t in tris {
            let (a, b, c) = (widen(od.coords[t[0] as usize]), widen(od.coords[t[1] as usize]), widen(od.coords[t[2] as usize]));
            let d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if d == 0.0 {
                nz += 1;
                continue;
            }
            if d > 0.0 {
                t.swap(1, 2);
            }
            out.extend(t);
            area += d.abs() / 2.0;
        }
        (out, area, nz)
    };
    let tris: Vec<[u32; 3]> = earcut_with_holes(&rings).into_iter().map(|t| [flat[t[0]], flat[t[1]], flat[t[2]]]).collect();
    let (ids, area, nz) = finish(tris);
    let tol = 1e-6 * polygon_area.max(1.0);
    if (area - polygon_area).abs() <= tol {
        return Ok((ids, nz));
    }
    bail!(
        "regions atlas: {name}: triangulation area {area} != polygon area {polygon_area} for {:?}",
        oa.src
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum GroupKey {
    Gap,
    Faces(Option<usize>, usize),
}

fn faces(ctx: &Ctx, od: &mut OutData) -> Result<()> {
    let src = od.src;
    if !src.with_faces {
        return Ok(());
    }
    let mut src_groups: HashMap<(usize, Option<usize>), Vec<(usize, usize)>> = HashMap::new();
    for a in &src.areas {
        src_groups.entry((a.key.0, a.faces_id)).or_default().push(a.key);
    }
    let mut regions = std::mem::take(&mut od.regions);
    let mut stats: BTreeMap<String, u64> = BTreeMap::new();
    let bump = |s: &mut BTreeMap<String, u64>, k: &str, n: u64| *s.entry(k.to_string()).or_insert(0) += n;
    for (ri, areas) in regions.iter_mut().enumerate() {
        let mut order: Vec<GroupKey> = Vec::new();
        let mut groups: HashMap<GroupKey, Vec<usize>> = HashMap::new();
        for (i, oa) in areas.iter().enumerate() {
            let key = if oa.is_gap {
                GroupKey::Gap
            } else {
                GroupKey::Faces(src.area(oa.src).faces_id, oa.ci)
            };
            if !groups.contains_key(&key) {
                order.push(key);
            }
            groups.entry(key).or_default().push(i);
        }
        for key in order {
            let members = &groups[&key];
            let mut copy = false;
            if let GroupKey::Faces(fid, ci) = key {
                let sset = &src_groups[&(ri, fid)];
                let all_plain = members.iter().all(|&i| {
                    let m = &areas[i];
                    m.primary && od.nfrag[&m.src] == 1 && m.frag.unchanged
                });
                let covers: HashSet<(usize, usize)> = members.iter().map(|&i| areas[i].src).collect();
                let same_set = covers.len() == sset.len() && sset.iter().all(|k| covers.contains(k));
                if let (true, Some(faces)) = (all_plain && same_set, src.area(areas[members[0]].src).faces.as_ref()) {
                    copy = stock_group_ok(ctx, src, faces, sset, ci)?;
                }
                if !copy {
                    bump(&mut stats, "groups_retriangulated_changed_or_baseline_inconsistent", 1);
                }
            }
            let ids: Vec<u32> = if copy {
                let GroupKey::Faces(_, ci) = key else { unreachable!() };
                let t = src.area(areas[members[0]].src).faces.as_ref().unwrap();
                let ids: Vec<u32> = t.iter().map(|&v| od.vid(ctx.oxy(ci, src.v[v as usize]))).collect();
                bump(&mut stats, "groups_copied_verbatim", 1);
                bump(&mut stats, "triangles_copied", (t.len() / 3) as u64);
                ids
            } else {
                let mut ids = Vec::new();
                for &i in members {
                    let (tri, nz) = triangulate(od, &areas[i])?;
                    ids.extend(tri);
                    if nz > 0 {
                        *od.stats.entry("degenerate_triangles_dropped".to_string()).or_insert(0) += nz;
                    }
                }
                bump(&mut stats, "groups_retriangulated", 1);
                bump(&mut stats, "triangles_new", (ids.len() / 3) as u64);
                ids
            };
            for &i in members {
                areas[i].faces = ids.clone();
            }
        }
    }
    od.regions = regions;
    for (k, n) in &stats {
        *od.stats.entry(k.clone()).or_insert(0) += n;
    }
    od.faces_stats = stats;
    Ok(())
}

fn bboxes(od: &mut OutData) {
    let coords = &od.coords;
    let bb = |ids: &[u32]| -> [[f32; 2]; 2] {
        let pts: Vec<[f32; 2]> = ids.iter().map(|&i| coords[i as usize]).collect();
        let mut lo = [f32::INFINITY; 2];
        let mut hi = [f32::NEG_INFINITY; 2];
        for p in pts {
            for a in 0..2 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        [lo, hi]
    };
    for areas in od.regions.iter_mut() {
        for oa in areas.iter_mut() {
            oa.bbox = bb(&oa.rings[0].1);
            oa.ring_bbox = oa.rings.iter().map(|(_, ids)| bb(ids)).collect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::{Chart, Plan};
    use crate::esf::{self, Node};

    /// One chart (identity transform) cropping [10,30]^2 out of a [-100,100]x[-50,50] world that two regions
    /// (a: x < 20, b: x > 20) partition; the crop cuts the shared border x = 20 between y = 10 and y = 30.
    fn tiny_plan() -> Plan {
        let mut proven = Plan::proven();
        proven.scale = 1.0;
        proven.world = [[-100.0, -50.0], [100.0, 50.0]];
        proven.source_world = proven.world;
        proven.charts = vec![Chart {
            key: "c".into(),
            theatres: vec!["t".into()],
            theatre_rects: [("t".to_string(), [[12.0, 12.0], [28.0, 28.0]])].into_iter().collect(),
            crop_rects: vec![[[10.0, 10.0], [30.0, 30.0]]],
            offset: [0.0, 0.0],
        }];
        proven.background_region = serde_json::json!({"key": "a", "source_area": 0});
        proven
    }

    fn area_item(lo: [f32; 2], hi: [f32; 2], ring: &[u32], faces: &[u32]) -> Vec<Node> {
        vec![
            esf::b(false),
            esf::b(false),
            esf::b(true),
            esf::xyv(lo[0], lo[1]),
            esf::xyv(hi[0], hi[1]),
            esf::u16v(0),
            Node::record("faces", 0, vec![esf::arr_u32(faces)]),
            Node::array(
                "outlines",
                0,
                vec![vec![esf::b(true), esf::xyv(lo[0], lo[1]), esf::xyv(hi[0], hi[1]), esf::arr_u32(ring), Node::array("connectivity", 0, vec![])]],
            ),
            esf::u16v(65535),
            esf::u16v(0),
        ]
    }

    fn region(key: &str, lo: [f32; 2], hi: [f32; 2], area: Vec<Node>) -> Vec<Node> {
        vec![esf::s16(key), esf::s8("sea"), esf::xyv(lo[0], lo[1]), esf::xyv(hi[0], hi[1]), Node::array("areas", 0, vec![area]), esf::i32v(-1)]
    }

    fn tiny_region_data() -> Node {
        let verts = [[-100.0f32, -50.0], [20.0, -50.0], [20.0, 50.0], [-100.0, 50.0], [100.0, -50.0], [100.0, 50.0]];
        let a = region("a", [-100.0, -50.0], [20.0, 50.0], area_item([-100.0, -50.0], [20.0, 50.0], &[0, 1, 2, 3], &[0, 3, 2, 0, 2, 1]));
        let b = region("b", [20.0, -50.0], [100.0, 50.0], area_item([20.0, -50.0], [100.0, 50.0], &[1, 4, 5, 2], &[1, 2, 5, 1, 5, 4]));
        Node::record(
            "region_data",
            0,
            vec![
                Node::record("vertices", 0, vec![esf::arr_xy(&verts)]),
                esf::xyv(-100.0, -50.0),
                esf::xyv(100.0, 50.0),
                Node::array("regions", 0, vec![a, b]),
            ],
        )
    }

    fn twice_area(od: &OutData, oa: &OArea) -> f64 {
        oa.faces
            .chunks_exact(3)
            .map(|t| {
                let (a, b, c) = (widen(od.coords[t[0] as usize]), widen(od.coords[t[1] as usize]), widen(od.coords[t[2] as usize]));
                let d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                assert!(d < 0.0, "faces must be clockwise and non-degenerate");
                -d
            })
            .sum()
    }

    #[test]
    fn crop_pieces_and_the_world_gap_partition_the_world_with_matched_edges() {
        let ctx = Ctx::build(&tiny_plan()).unwrap();
        let rd = tiny_region_data();
        let src = Source::new(&rd, "region_data", true).unwrap();
        let od = assemble(&ctx, &src, Some("a:0"), Kind::Main).unwrap();

        // region a keeps its piece first and owns the world gap as an extra area; region b keeps its single piece
        assert_eq!(od.regions.len(), 2);
        assert_eq!(od.regions[0].len(), 2);
        assert_eq!(od.regions[1].len(), 1);
        assert!(od.regions[0][1].is_gap && od.gap_local == 1);
        assert_eq!(od.area_map[&(0, 0)], vec![0, 1]);
        assert_eq!(od.area_map[&(1, 0)], vec![0]);
        assert!(od.dropped.is_empty());

        let world_area = 200.0 * 100.0;
        assert!((od.regions[0][0].frag.area - 200.0).abs() < 1e-9);
        assert!((od.regions[1][0].frag.area - 200.0).abs() < 1e-9);
        assert!((od.gap.area - (world_area - 400.0)).abs() < 1e-6);
        // the faces of every area triangulate exactly its polygon
        assert!((twice_area(&od, &od.regions[0][0]) / 2.0 - 200.0).abs() < 1e-6);
        assert!((twice_area(&od, &od.regions[1][0]) / 2.0 - 200.0).abs() < 1e-6);
        assert!((twice_area(&od, &od.regions[0][1]) / 2.0 - (world_area - 400.0)).abs() < 1e-3);

        // only the four world-border edges of the gap have no neighbour
        assert_eq!(od.unmatched.len(), 4);
        // piece A: the shared border x = 20 borders B, the other three sides border the gap (region 0, area 1)
        let runs = &od.regions[0][0].conn[0];
        assert_eq!(runs.as_slice(), &[[1 << 16, 0, 0], [1, 1, 3]]);
        // the two neighbours share the very same two crossing vertices
        let ring_a = &od.regions[0][0].rings[0].1;
        let ring_b = &od.regions[1][0].rings[0].1;
        let shared = ring_a.iter().filter(|v| ring_b.contains(v)).count();
        assert_eq!(shared, 2);
        let crossings: Vec<[f32; 2]> = ring_a.iter().filter(|v| ring_b.contains(v)).map(|&v| od.coords[v as usize]).collect();
        assert!(crossings.contains(&[20.0, 10.0]) && crossings.contains(&[20.0, 30.0]));
    }

    #[test]
    fn query_index_of_the_tiny_atlas_lists_every_matched_edge_once() {
        let ctx = Ctx::build(&tiny_plan()).unwrap();
        let rd = tiny_region_data();
        let src = Source::new(&rd, "region_data", true).unwrap();
        let od = assemble(&ctx, &src, Some("a:0"), Kind::Main).unwrap();
        let (q, stats) = super::super::query::build_query(&od, true, &Node::record("query_info", 0, vec![])).unwrap();
        assert_eq!((stats.nodes, stats.cells), (1, 1));
        // the lower-left corner and the border midpoints lie on the world border: the first strictly interior probe is the centre
        assert_eq!(stats.probe_fallbacks.get("3"), Some(&1));
        let cell = q.get("node").unwrap().children().unwrap().iter().find(|c| c.name() == Some("cell")).unwrap();
        let kids = cell.children().unwrap();
        assert_eq!(kids[0].as_xy().unwrap(), [0.0, 0.0]);
        // the centre of the world lies in the gap polygon = region a, area 1
        assert_eq!(kids[1].as_u32().unwrap(), 1);
        // 6 crop-boundary edges (gap | piece) + the shared border = 7 matched edges, 4 numbers each
        assert_eq!(kids[2].u32_array().unwrap().len(), 7 * 4);
    }
}
