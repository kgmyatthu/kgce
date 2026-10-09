//! Pathfinding for a region split (a new region cut out of a region that already has paths in the grid): after
//! `Grid::relabel(source, new, ..)` has given the new region its path and relabelled the pieces by centroid, this
//! pass makes the grid agree with the cut exactly the way the Burma|Bengal border of the experiments does:
//! * every piece of the two regions (and of groups containing either) is assigned to the side of the cut it lies on
//!   (by area, not by centroid);
//! * the region border band `buffer(cut, 0.5, round caps)` ∩ type-0 land becomes a type-0 piece with the group path
//!   (source, new) (plus the other members of a group that already covered the spot);
//! * the settlement node of the new region (type 7, its single path, shape by slot kind) is carved out of its land.
//! Roads (type 6), landings (type 7), coast bands, sea and every other cell are left as they are.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, bail, ensure};

use super::finish;
use super::geometry::{self as g, FIX, Pt, Shapes};
use super::land::{self, Frame, scale};
use super::model::{MPiece, Model};
use super::nodes::{self, SlotKind};
use super::{Options, RegionReport, Report};
use crate::regionmap::pathfinding::{Grid, TRANSITION};
use crate::regionmap::regions::{RegionData, settlement_center};

const BAND_HALF: f64 = 0.5;
const TMP_GROUP: u16 = 5000;
const AREA_EPS: f64 = 1e-9;
const LAND_CLASS: u8 = 0x1f;

/// Group-path bookkeeping: member lists are kept in ascending region-list position.
struct Paths {
    pos: BTreeMap<i16, usize>,
    singles: BTreeMap<i16, u16>,
    groups: BTreeMap<u16, Vec<i16>>,
    by_members: BTreeMap<Vec<i16>, u16>,
    next_tmp: u16,
}

impl Paths {
    fn sorted(&self, mut v: Vec<i16>) -> Vec<i16> {
        v.sort_by_key(|r| self.pos[r]);
        v.dedup();
        v
    }

    fn pid_for(&mut self, members: Vec<i16>) -> Result<u16> {
        let members = self.sorted(members);
        if members.len() == 1 {
            return self.singles.get(&members[0]).copied().ok_or_else(|| anyhow!("region {} has no single path", members[0]));
        }
        if let Some(&p) = self.by_members.get(&members) {
            return Ok(p);
        }
        let p = self.next_tmp;
        self.next_tmp += 1;
        self.groups.insert(p, members.clone());
        self.by_members.insert(members, p);
        Ok(p)
    }

    /// Member regions of path id `pid` (empty for the sea path / transition id).
    fn members(&self, grid: &Grid, pid: u16) -> Vec<i16> {
        if pid == TRANSITION {
            return Vec::new();
        }
        if let Some(g) = self.groups.get(&pid) {
            return g.clone();
        }
        grid.paths.get(pid as usize).cloned().unwrap_or_default()
    }
}

fn corners_of(lo: Pt, hi: Pt) -> BTreeSet<Pt> {
    [[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]].into_iter().collect()
}

/// Give the split `new` (carved out of `source`) its node, border band and group path; see the module docs.
/// `grid` must already contain both regions (`Grid::relabel` ran for this split) and `rd` is the geometry after the
/// split. Call it for every split after all of them were relabelled.
pub fn synthesize_split(grid: &mut Grid, rd: &RegionData, source: usize, new: usize, kind: SlotKind, opts: &Options) -> Result<Report> {
    let mut report = Report::default();
    let (src_id, new_id) = (source as i16, new as i16);
    ensure!(grid.region_ids.contains(&src_id), "split source {} is not in the pathfinding grid", rd.regions[source].key);
    ensure!(grid.region_ids.contains(&new_id), "split region {} is not in the pathfinding grid (relabel first)", rd.regions[new].key);
    for r in [source, new] {
        ensure!(!opts.no_group_regions.iter().any(|n| *n == rd.regions[r].key), "{} never forms group paths", rd.regions[r].key);
    }
    let sea_idx = grid.paths.iter().position(|p| p.is_empty()).ok_or_else(|| anyhow!("pathfinding grid has no sea path"))?;
    let old = Model::from_grid(grid)?;
    let mut m = old.clone();
    let (cols, rows, cs) = (m.cols, m.rows, m.cs);
    let fr = Frame { origin: [m.origin[0] as f64 / FIX, m.origin[1] as f64 / FIX], cs };

    // ---- path tables (group member lists sorted by region-list position; relabel appends twins unsorted)
    let mut paths = Paths {
        pos: grid.region_ids.iter().enumerate().map(|(i, &r)| (r, i)).collect(),
        singles: (0..sea_idx).map(|p| (grid.paths[p][0], p as u16)).collect(),
        groups: BTreeMap::new(),
        by_members: BTreeMap::new(),
        next_tmp: TMP_GROUP,
    };
    for pid in (sea_idx + 1)..grid.paths.len() {
        let members = paths.sorted(grid.paths[pid].clone());
        ensure!(!paths.by_members.contains_key(&members), "duplicate group path {members:?}");
        paths.by_members.insert(members.clone(), pid as u16);
        paths.groups.insert(pid as u16, members);
    }
    let sing_new = paths.singles[&new_id];
    let relevant = |paths: &Paths, grid: &Grid, pid: u16| paths.members(grid, pid).iter().any(|&r| r == src_id || r == new_id);

    // ---- geometry of the cut
    let in_grid = |ri: usize| grid.region_ids.contains(&(ri as i16));
    let no_group: Vec<usize> = rd.regions.iter().enumerate().filter(|(_, r)| opts.no_group_regions.iter().any(|n| *n == r.key)).map(|(i, _)| i).collect();
    let mut new_poly_parts: Vec<Shapes> = Vec::new();
    let mut cut_lines: Vec<Vec<[f64; 2]>> = Vec::new();
    for (ai, a) in rd.regions[new].areas.iter().enumerate() {
        if !a.passable {
            continue;
        }
        let edges = land::area_edges(rd, &fr, new, ai, &no_group, &in_grid);
        new_poly_parts.push(edges.poly);
        for (key, lines) in edges.borders {
            if key.0 == source {
                cut_lines.extend(lines);
            }
        }
    }
    ensure!(!new_poly_parts.is_empty(), "region {} has no passable area", rd.regions[new].key);
    let refs: Vec<&Shapes> = new_poly_parts.iter().collect();
    let new_poly = g::union_many(&refs);
    if cut_lines.is_empty() {
        report.warnings.push(format!("{} and {} share no passable border: no border band created", rd.regions[new].key, rd.regions[source].key));
    }
    let bufs: Vec<Shapes> = cut_lines.iter().map(|ln| g::stroke(&scale(ln), BAND_HALF * FIX, true, 4)).collect();
    let refs: Vec<&Shapes> = bufs.iter().collect();
    let band = g::node_shapes(&g::union_many(&refs), cs);
    let mut regions_bbox: Vec<&Shapes> = vec![&new_poly];
    regions_bbox.push(&band);
    let area_union = g::union_many(&regions_bbox);
    let (bl, bh) = g::shapes_bbox(&area_union).ok_or_else(|| anyhow!("empty split geometry"))?;
    let c_lo = (bl[0] / cs - 1).max(0) as usize;
    let c_hi = ((bh[0] / cs + 1).max(0) as usize).min(cols - 1);
    let r_lo = (bl[1] / cs - 1).max(0) as usize;
    let r_hi = ((bh[1] / cs + 1).max(0) as usize).min(rows - 1);

    let side_pid = |paths: &mut Paths, base: &[i16], new_side: bool| -> Result<u16> {
        let mut v: Vec<i16> = base.iter().copied().filter(|&r| r != src_id && r != new_id).collect();
        v.push(if new_side { new_id } else { src_id });
        paths.pid_for(v)
    };
    let new_side = |q: &Shapes, new_clip: &Shapes| -> bool { g::area(&g::intersection(q, new_clip)) * 2.0 > g::area(q) };

    // ---- cells crossed by the cut or by the new region: side assignment and the band
    let mut touched: BTreeMap<usize, bool> = BTreeMap::new();
    let mut pieces_reassigned = 0usize;
    let mut band_pieces = 0usize;
    for r in r_lo..=r_hi {
        for c in c_lo..=c_hi {
            let idx = r * cols + c;
            let (lo, hi) = ([c as i32 * cs, r as i32 * cs], [(c as i32 + 1) * cs, (r as i32 + 1) * cs]);
            let new_clip = g::clip_rect(&new_poly, lo, hi);
            let band_clip = g::clip_rect(&band, lo, hi);
            let band_here = g::area(&band_clip) > AREA_EPS;
            let cell = m.cells[idx].clone();
            if cell.pieces.is_empty() {
                let Some((a, pid)) = cell.run else { continue };
                if a & 15 != 0 || !relevant(&paths, grid, pid) {
                    continue;
                }
                let base = paths.members(grid, pid);
                if !band_here {
                    let bx: Shapes = vec![g::rect_shape(lo, hi)];
                    let np = side_pid(&mut paths, &base, new_side(&bx, &new_clip))?;
                    if np != pid {
                        m.cells[idx].run = Some((a, np));
                        pieces_reassigned += 1;
                    }
                    continue;
                }
                // a run cell the band crosses: expand it, then fall through to the piece handling
                m.cells[idx].pieces.push(MPiece {
                    a,
                    pid,
                    gbit: 0,
                    pts: vec![[hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]], [hi[0], lo[1]]],
                    src: None,
                });
                m.cells[idx].run = None;
            }
            let mut changed = false;
            let mut out_pieces: Vec<MPiece> = Vec::with_capacity(m.cells[idx].pieces.len() + 4);
            for p in std::mem::take(&mut m.cells[idx].pieces) {
                if !matches!(p.ptype(), 0 | 6 | 7) || !relevant(&paths, grid, p.pid) {
                    out_pieces.push(p);
                    continue;
                }
                let base = paths.members(grid, p.pid);
                let poly: Shapes = vec![vec![p.pts.clone()]];
                if p.ptype() == 0 && band_here && g::area(&g::intersection(&poly, &band_clip)) > AREA_EPS {
                    changed = true;
                    let mut bm: Vec<i16> = base.iter().copied().filter(|&r| r != src_id && r != new_id).collect();
                    bm.extend([src_id, new_id]);
                    let band_pid = paths.pid_for(bm)?;
                    for q in g::intersection(&poly, &band_clip) {
                        for part in g::split_holes(&q) {
                            out_pieces.push(MPiece { a: 0, pid: band_pid, gbit: p.gbit, pts: g::ring_pts(&part[0]), src: None });
                            band_pieces += 1;
                        }
                    }
                    for q in g::difference(&poly, &band_clip) {
                        let ns = new_side(&vec![q.clone()], &new_clip);
                        let np = side_pid(&mut paths, &base, ns)?;
                        for part in g::split_holes(&q) {
                            out_pieces.push(MPiece { a: 0, pid: np, gbit: p.gbit, pts: g::ring_pts(&part[0]), src: None });
                        }
                    }
                } else {
                    let np = side_pid(&mut paths, &base, new_side(&poly, &new_clip))?;
                    if np != p.pid {
                        pieces_reassigned += 1;
                    }
                    out_pieces.push(MPiece { pid: np, ..p });
                }
            }
            if changed {
                out_pieces.sort_by_key(|p| *p.pts.iter().min().unwrap());
                touched.insert(idx, false);
            }
            m.cells[idx].pieces = out_pieces;
        }
    }

    // ---- the settlement node of the new region
    let spt = settlement_center(&rd.regions[new])?;
    let sp = fr.fixed(fr.rel(spt));
    let sp_f = [sp[0] as f64, sp[1] as f64];
    ensure!(g::contains_point(&new_poly, sp_f), "the settlement of {} is not inside its region", rd.regions[new].key);
    let ring = nodes::node_ring(kind, sp);
    let node_q: Shapes = vec![vec![g::node_ring(&ring, cs)]];
    let (nl, nh) = g::shapes_bbox(&node_q).ok_or_else(|| anyhow!("empty node"))?;
    let (nc_lo, nc_hi) = ((nl[0] / cs).max(0) as usize, ((nh[0] / cs).max(0) as usize).min(cols - 1));
    let (nr_lo, nr_hi) = ((nl[1] / cs).max(0) as usize, ((nh[1] / cs).max(0) as usize).min(rows - 1));
    let mut land_parts: Vec<Shapes> = Vec::new();
    for r in nr_lo..=nr_hi {
        for c in nc_lo..=nc_hi {
            let idx = r * cols + c;
            let (lo, hi) = ([c as i32 * cs, r as i32 * cs], [(c as i32 + 1) * cs, (r as i32 + 1) * cs]);
            match m.cells[idx].run {
                Some((a, pid)) if m.cells[idx].pieces.is_empty() => {
                    if a & 15 == 0 && pid == sing_new {
                        land_parts.push(vec![g::rect_shape(lo, hi)]);
                    }
                }
                _ => {
                    for p in &m.cells[idx].pieces {
                        if p.ptype() == 0 && p.pid == sing_new {
                            land_parts.push(vec![vec![p.pts.clone()]]);
                        }
                    }
                }
            }
        }
    }
    let refs: Vec<&Shapes> = land_parts.iter().collect();
    let l_new = g::union_many(&refs);
    let mut node = g::intersection(&node_q, &l_new);
    if node.len() > 1 {
        if let Some(at) = node.iter().position(|sh| g::contains_point(&vec![sh.clone()], sp_f)) {
            node = vec![node[at].clone()];
            report.warnings.push(format!("{}: settlement node split by the land zone, only the part at the settlement is kept", rd.regions[new].key));
        }
    }
    ensure!(
        g::contains_point(&node, sp_f),
        "the settlement point of {} lies within {:.2} units of a border, coast or other piece: its node does not fit; move it into the interior of the region",
        rd.regions[new].key,
        2.1
    );
    let full = g::area(&node_q);
    let clipped = full - g::area(&node);
    let node_clipped = clipped > 1e-5;
    if node_clipped {
        report.warnings.push(format!(
            "{}: settlement node clipped by {:.4} of {:.3} units² (the settlement is closer than the node radius to a border or coast)",
            rd.regions[new].key, clipped, full
        ));
    }
    for r in nr_lo..=nr_hi {
        for c in nc_lo..=nc_hi {
            let idx = r * cols + c;
            let (lo, hi) = ([c as i32 * cs, r as i32 * cs], [(c as i32 + 1) * cs, (r as i32 + 1) * cs]);
            let node_clip = g::clip_rect(&node, lo, hi);
            if g::area(&node_clip) < AREA_EPS {
                continue;
            }
            if m.cells[idx].pieces.is_empty() {
                match m.cells[idx].run {
                    Some((a, pid)) if a & 15 == 0 && pid == sing_new => {
                        m.cells[idx].pieces.push(MPiece {
                            a,
                            pid,
                            gbit: 0,
                            pts: vec![[hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]], [hi[0], lo[1]]],
                            src: None,
                        });
                        m.cells[idx].run = None;
                    }
                    _ => continue,
                }
            }
            let mut out: Vec<MPiece> = Vec::new();
            let mut changed = false;
            for p in std::mem::take(&mut m.cells[idx].pieces) {
                if p.ptype() != 0 || p.pid != sing_new {
                    out.push(p);
                    continue;
                }
                let poly: Shapes = vec![vec![p.pts.clone()]];
                let inside = g::intersection(&poly, &node_clip);
                if inside.is_empty() {
                    out.push(p);
                    continue;
                }
                changed = true;
                for q in inside {
                    for part in g::split_holes(&q) {
                        out.push(MPiece { a: 7, pid: sing_new, gbit: 0, pts: g::ring_pts(&part[0]), src: None });
                    }
                }
                for q in g::difference(&poly, &node_clip) {
                    for part in g::split_holes(&q) {
                        out.push(MPiece { a: 0, pid: sing_new, gbit: p.gbit, pts: g::ring_pts(&part[0]), src: None });
                    }
                }
            }
            if changed {
                out.sort_by_key(|p| *p.pts.iter().min().unwrap());
                touched.insert(idx, false);
            }
            m.cells[idx].pieces = out;
        }
    }

    // ---- a full-cell land piece is a run cell again
    let touched_idx: Vec<usize> = touched.keys().copied().collect();
    for idx in touched_idx {
        let (c, r) = (idx % cols, idx / cols);
        let (lo, hi) = ([c as i32 * cs, r as i32 * cs], [(c as i32 + 1) * cs, (r as i32 + 1) * cs]);
        let cell = &mut m.cells[idx];
        if cell.pieces.len() == 1 && cell.pieces[0].ptype() == 0 && cell.pieces[0].pts.iter().copied().collect::<BTreeSet<Pt>>() == corners_of(lo, hi) && cell.pieces[0].pts.len() == 4 {
            let p = cell.pieces.pop().unwrap();
            cell.run = Some((p.a, p.pid));
            touched.insert(idx, true);
        }
    }
    report.touched_cells = touched.len();

    // ---- hygiene, a-words
    let (hyg, _) = finish::hygiene_and_words(&mut m, &old, &touched);
    report.vertices_merged = hyg.merged;
    report.degenerate_dropped = hyg.dropped;
    report.tjunctions = hyg.tjunctions;
    report.edge_vertices_synced = hyg.synced;
    for f in hyg.failures {
        bail!("split path synthesis: {f}");
    }

    // ---- group numbering
    let band_members = paths.sorted(vec![src_id, new_id]);
    let (newid, _unused) = finish::renumber_groups(&mut m, grid, sea_idx, &paths.groups)?;
    if let Some(&tmp) = paths.by_members.get(&band_members) {
        match newid.get(&tmp) {
            Some(&id) => report.new_groups.push((band_members, id)),
            None if !cut_lines.is_empty() => report.warnings.push("the border band produced no pieces".into()),
            None => {}
        }
    }
    let _ = (pieces_reassigned, band_pieces, LAND_CLASS);
    let landings = m.cells.iter().flat_map(|c| c.pieces.iter()).filter(|p| p.ptype() == 7 && p.pid == sing_new).count().saturating_sub(1);
    report.per_region.push(RegionReport { region: new, path_id: sing_new, landings, node_clipped });
    m.write_into(grid)?;
    Ok(report)
}
