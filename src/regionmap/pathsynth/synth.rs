//! The synthesis driver: port of work/exp5/pathsynth5.py (which builds on exp4/pathsynth.py) for any number of new
//! land regions of one theatre grid.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, anyhow, bail, ensure};

use super::crossings::{self, Landing, MIN_LINE, MIN_SEP, Polyline, SLIDE};
use super::geometry::{self as g, FIX, Pt, Shape, Shapes};
use super::finish;
use super::land::{self, AreaEdges, Frame, scale};
use super::model::{MPiece, Model, OUTSIDE};
use super::nodes;
use super::{Options, RegionReport, Report, Target};
use crate::regionmap::pathfinding::Grid;
use crate::regionmap::regions::{RegionData, settlement_center};

const BORDER_INSET: f64 = 0.05;
const COAST_INSET: f64 = 0.5;
const BAND_HALF: f64 = 0.5;
const SMALL_LAND: f64 = 0.25;
/// temporary pids of new group paths (renumbered at the end)
const TMP_GROUP: u16 = 5000;
const AREA_EPS: f64 = 1e-9;
const SLIVER_MIN: f64 = 1e-3;
const SLIVER_W: f64 = 0.02;
const LAND_CLASS: u8 = 0x1f;
/// area tolerance of the landing zone tests (the hexagons are rounded to fixed point)
const ZONE_TOL: f64 = 5e-6;

struct Unit {
    region: usize,
    area: usize,
    pid: u16,
    pos: usize,
    name: String,
    edges: AreaEdges,
    coast_zone: Shapes,
    buf: Shapes,
    lg: Shapes,
    zg: Shapes,
    node: Shapes,
    lhu: Shapes,
    lland: Shapes,
    sh: Vec<Shape>,
    lh: Vec<Shape>,
    crosses: Vec<Landing>,
    settle: Option<[f64; 2]>,
    kind: nodes::SlotKind,
}

struct Pair {
    unit: usize,
    buf: Shapes,
    pnb: Shapes,
    nbpos: usize,
    nbname: String,
    coast_nb: Shapes,
}

struct Band {
    pid: u16,
    geom: Shapes,
    members: Vec<usize>,
    name: String,
}

struct Layer {
    geom: Shapes,
    bbox: (Pt, Pt),
    ptype: u8,
    pid: u16,
}

fn near_bbox(a: &(Pt, Pt), lo: Pt, hi: Pt) -> bool {
    !(a.1[0] < lo[0] || a.0[0] > hi[0] || a.1[1] < lo[1] || a.0[1] > hi[1])
}

fn flat(s: &Shapes) -> Vec<&Shape> {
    s.iter().collect()
}

fn is_sliver(q: &Shape) -> bool {
    let a = g::shape_area(q);
    a < SLIVER_MIN || (a < 0.5 && g::is_thin(q, SLIVER_W))
}

fn touches_cell_edge(q: &Shape, lo: Pt, hi: Pt) -> bool {
    q.iter().flatten().any(|p| p[0] == lo[0] || p[0] == hi[0] || p[1] == lo[1] || p[1] == hi[1])
}

/// Unite the area-pair strips of one group path before layering, so shared endpoints cannot be covered twice.
/// Different member sets must retain their own geometry; choosing a winner would change region accessibility.
fn unite_band_strips(mut strips: Vec<Shapes>) -> Shapes {
    if strips.len() == 1 {
        strips.pop().unwrap()
    } else {
        let refs: Vec<&Shapes> = strips.iter().collect();
        g::union_many(&refs)
    }
}

pub fn synthesize(grid: &mut Grid, rd: &RegionData, targets: &[Target], opts: &Options) -> Result<Report> {
    let mut report = Report::default();
    let old = Model::from_grid(grid)?;
    let mut m = old.clone();
    let (cols, rows, cs) = (m.cols, m.rows, m.cs);
    let fr = Frame {
        origin: [m.origin[0] as f64 / FIX, m.origin[1] as f64 / FIX],
        cs,
    };
    let cs_f = cs as f64;

    // ---- targets, region list and path ids
    let mut tg: Vec<&Target> = targets.iter().collect();
    tg.sort_by_key(|t| t.region);
    for w in tg.windows(2) {
        ensure!(w[0].region != w[1].region, "region {} is listed twice", w[0].region);
    }
    for t in &tg {
        ensure!(t.region < rd.regions.len(), "target region {} is not in regions.esf", t.region);
        ensure!(!grid.region_ids.contains(&(t.region as i16)), "region {} ({}) is already in the pathfinding grid", t.region, rd.regions[t.region].key);
        ensure!(rd.regions[t.region].kind != "sea", "region {} is a sea region", rd.regions[t.region].key);
    }
    let singles_idx = grid.paths.iter().position(|p| p.is_empty()).ok_or_else(|| anyhow!("pathfinding grid has no sea path"))?;
    let n_new = tg.len();
    let mut pids: BTreeMap<usize, (u16, usize)> = BTreeMap::new();
    for (k, t) in tg.iter().enumerate() {
        grid.region_ids.push(t.region as i16);
        grid.paths.insert(singles_idx + k, vec![t.region as i16]);
        pids.insert(t.region, ((singles_idx + k) as u16, grid.region_ids.len()));
    }
    let shift = |pid: u16| if pid == OUTSIDE || (pid as usize) < singles_idx { pid } else { pid + n_new as u16 };
    for cell in m.cells.iter_mut() {
        for p in cell.pieces.iter_mut() {
            p.pid = shift(p.pid);
        }
        if cell.pieces.is_empty() {
            if let Some(r) = cell.run.as_mut() {
                r.1 = shift(r.1);
            }
        }
    }
    let sea_pid = (singles_idx + n_new) as u16;
    let region_ids: Vec<i16> = grid.region_ids.clone();
    let pos_of = |ri: usize| region_ids.iter().position(|&r| r as usize == ri).map(|p| p + 1);
    let no_group: Vec<usize> = rd
        .regions
        .iter()
        .enumerate()
        .filter(|(_, r)| opts.no_group_regions.iter().any(|n| *n == r.key))
        .map(|(i, _)| i)
        .collect();
    let in_grid = |ri: usize| pos_of(ri).is_some();

    // ---- units: every passable area of every target region
    let mut units: Vec<Unit> = Vec::new();
    for t in &tg {
        let region = &rd.regions[t.region];
        let (pid, pos) = pids[&t.region];
        let settle = settlement_center(region).ok().map(|c| fr.rel(c));
        let mut any = false;
        for (ai, a) in region.areas.iter().enumerate() {
            if !a.passable {
                continue;
            }
            any = true;
            let edges = land::area_edges(rd, &fr, t.region, ai, &no_group, &in_grid);
            units.push(Unit {
                region: t.region,
                area: ai,
                pid,
                pos,
                name: region.key.clone(),
                edges,
                coast_zone: Shapes::new(),
                buf: Shapes::new(),
                lg: Shapes::new(),
                zg: Shapes::new(),
                node: Shapes::new(),
                lhu: Shapes::new(),
                lland: Shapes::new(),
                sh: Vec::new(),
                lh: Vec::new(),
                crosses: Vec::new(),
                settle,
                kind: t.settlement_slot_kind,
            });
        }
        ensure!(any, "region {} has no passable area", region.key);
    }
    let unit_of: HashMap<(usize, usize), usize> = units.iter().enumerate().map(|(i, u)| ((u.region, u.area), i)).collect();

    // ---- land zone, coast zone, border buffers
    for u in units.iter_mut() {
        let coast_bands: Vec<Shapes> = u.edges.coast.iter().map(|ln| g::stroke(&scale(ln), COAST_INSET * FIX, false, 2)).collect();
        let refs: Vec<&Shapes> = coast_bands.iter().collect();
        u.coast_zone = g::union_many(&refs);
        let pin = g::inset(&u.edges.poly, BORDER_INSET * FIX);
        let mut bufs: Vec<Shapes> = Vec::new();
        for lines in u.edges.borders.values() {
            for ln in lines {
                bufs.push(g::stroke(&scale(ln), BAND_HALF * FIX, true, 4));
            }
        }
        let refs: Vec<&Shapes> = bufs.iter().collect();
        u.buf = g::union_many(&refs);
        let mut l = g::difference(&pin, &u.coast_zone);
        l = g::difference(&l, &u.buf);
        u.lg = g::node_shapes(&l, cs);
        u.zg = g::node_shapes(&g::union(&u.edges.poly, &u.coast_zone), cs);
        report.warnings.extend(
            u.edges
                .borders
                .keys()
                .filter(|k| !in_grid(k.0) && !unit_of.contains_key(k))
                .map(|k| format!("{}: border with region {} ignored (not in the pathfinding grid)", u.name, rd.regions[k.0].key)),
        );
    }

    // ---- border pairs and bands
    let mut pairs: BTreeMap<((usize, usize), (usize, usize)), Pair> = BTreeMap::new();
    for (ui, u) in units.iter().enumerate() {
        for (nbk, lines) in &u.edges.borders {
            let key = {
                let a = (u.region, u.area);
                if a < *nbk { (a, *nbk) } else { (*nbk, a) }
            };
            if pairs.contains_key(&key) {
                continue;
            }
            let bufs: Vec<Shapes> = lines.iter().map(|ln| g::stroke(&scale(ln), BAND_HALF * FIX, true, 4)).collect();
            let refs: Vec<&Shapes> = bufs.iter().collect();
            let bf = g::union_many(&refs);
            let (pnb, cnb_lines, nbpos, nbname): (Shapes, Vec<Vec<[f64; 2]>>, usize, String) = match unit_of.get(nbk) {
                Some(&nu) => (units[nu].edges.poly.clone(), units[nu].edges.coast.clone(), units[nu].pos, units[nu].name.clone()),
                None => {
                    let e = land::area_edges(rd, &fr, nbk.0, nbk.1, &no_group, &in_grid);
                    (e.poly, e.coast, pos_of(nbk.0).ok_or_else(|| anyhow!("neighbour {} not in the grid", rd.regions[nbk.0].key))?, rd.regions[nbk.0].key.clone())
                }
            };
            // neighbour coast lines near the border
            let near_lines: Vec<Shapes> = cnb_lines
                .iter()
                .filter(|ln| g::distance_polyline(&scale(ln), &bf) < 3.0 * FIX)
                .map(|ln| g::stroke(&scale(ln), COAST_INSET * FIX, false, 2))
                .collect();
            let refs: Vec<&Shapes> = near_lines.iter().collect();
            pairs.insert(key, Pair { unit: ui, buf: bf, pnb, nbpos, nbname, coast_nb: g::union_many(&refs) });
        }
    }
    // One band (= one group path) per member set: a region pair can border along several (area, area) pairs -- several
    // areas of one region touching a neighbour, or the halves of a split touching the same neighbour areas -- and each
    // pair has its own buffered strip. The strips of one group path overlap where the borders meet (the neighbouring
    // areas share a vertex), so the pieces of one path id would double-cover the cell; they are united instead.
    let mut bands: Vec<Band> = Vec::new();
    let mut band_raws: Vec<Vec<Shapes>> = Vec::new();
    let mut band_of: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut zones_old: Vec<Shapes> = Vec::new();
    let mut group_members: BTreeMap<u16, Vec<usize>> = BTreeMap::new();
    for pr in pairs.values() {
        let u = &units[pr.unit];
        let both = g::union(&u.edges.poly, &pr.pnb);
        let uu = g::inset(&both, BORDER_INSET * FIX);
        let cz = g::union(&u.coast_zone, &pr.coast_nb);
        let band = g::intersection(&g::difference(&uu, &cz), &pr.buf);
        let zone = g::intersection(&pr.buf, &both);
        let mut members = vec![pr.nbpos, u.pos];
        members.sort();
        members.dedup();
        let bi = match band_of.get(&members) {
            Some(&bi) => bi,
            None => {
                let tmp = TMP_GROUP + bands.len() as u16;
                group_members.insert(tmp, members.clone());
                bands.push(Band {
                    pid: tmp,
                    geom: Shapes::new(),
                    members: members.clone(),
                    name: format!("{}|{}", pr.nbname, u.name),
                });
                band_raws.push(Vec::new());
                band_of.insert(members, bands.len() - 1);
                bands.len() - 1
            }
        };
        band_raws[bi].push(band);
        zones_old.push(g::node_shapes(&zone, cs));
    }
    for (b, raw) in bands.iter_mut().zip(band_raws.into_iter().map(unite_band_strips)) {
        b.geom = g::node_shapes(&raw, cs);
    }
    let zall = {
        let mut list: Vec<&Shapes> = units.iter().map(|u| &u.zg).collect();
        list.extend(zones_old.iter());
        g::union_many(&list)
    };
    let zall_bbox = g::shapes_bbox(&zall).ok_or_else(|| anyhow!("empty land zone"))?;

    // ---- old content of the neighbourhood: blocked (land of other regions) and sea cover
    let margin = (4.0 * FIX) as i32;
    let (mut blocked_parts, mut sea_parts): (Vec<Shapes>, Vec<Shapes>) = (Vec::new(), Vec::new());
    for (idx, cell) in m.cells.iter().enumerate() {
        let (c, r) = ((idx % cols) as i32, (idx / cols) as i32);
        let (lo, hi) = ([c * cs, r * cs], [(c + 1) * cs, (r + 1) * cs]);
        if lo[0] > zall_bbox.1[0] + margin || hi[0] < zall_bbox.0[0] - margin || lo[1] > zall_bbox.1[1] + margin || hi[1] < zall_bbox.0[1] - margin {
            continue;
        }
        if cell.pieces.is_empty() {
            let t = cell.run.map(|x| (x.0 & 15) as u8).unwrap_or(2);
            if t != 1 && t != 2 {
                blocked_parts.push(vec![g::rect_shape(lo, hi)]);
            } else if t == 1 {
                sea_parts.push(vec![g::rect_shape(lo, hi)]);
            }
        } else {
            for p in &cell.pieces {
                match p.ptype() {
                    1 => sea_parts.push(vec![vec![p.pts.clone()]]),
                    2 => {}
                    _ => blocked_parts.push(vec![vec![p.pts.clone()]]),
                }
            }
        }
    }
    let refs: Vec<&Shapes> = blocked_parts.iter().collect();
    let blocked = g::difference(&g::union_many(&refs), &zall);
    let refs: Vec<&Shapes> = sea_parts.iter().collect();
    let seacover_old = g::union_many(&refs);
    let (wlo, whi) = (
        [(zall_bbox.0[0] - (6.0 * FIX) as i32).max(0), (zall_bbox.0[1] - (6.0 * FIX) as i32).max(0)],
        [(zall_bbox.1[0] + (6.0 * FIX) as i32).min(cols as i32 * cs), (zall_bbox.1[1] + (6.0 * FIX) as i32).min(rows as i32 * cs)],
    );
    let (sea_pf, _sea_raw) = land::sea_geometry(rd, &fr, wlo, whi, &no_group, &in_grid);
    let sea_q = g::node_shapes(&sea_pf, cs);
    let seacover = g::union(&seacover_old, &sea_q);

    // ---- settlement nodes (before the landings so the landings keep their distance)
    for u in units.iter_mut() {
        let Some(spt) = u.settle else { continue };
        let p = fr.fixed(spt);
        if !g::contains_point(&u.edges.poly, [p[0] as f64, p[1] as f64]) {
            continue;
        }
        let ring = nodes::node_ring(u.kind, p);
        let node_q = vec![vec![g::node_ring(&ring, cs)]];
        let mut node = g::intersection(&u.lg, &node_q);
        let full = g::area(&node_q);
        if node.len() > 1 {
            // keep the part around the settlement point (else the largest)
            let at = node.iter().position(|sh| g::contains_point(&vec![sh.clone()], [p[0] as f64, p[1] as f64]));
            let keep = at.unwrap_or_else(|| (0..node.len()).max_by(|&a, &b| g::shape_area(&node[a]).partial_cmp(&g::shape_area(&node[b])).unwrap()).unwrap());
            node = vec![node[keep].clone()];
            report.warnings.push(format!("{}: settlement node split by the land zone, only the part at the settlement is kept", u.name));
        }
        ensure!(!node.is_empty(), "{}: the settlement point lies outside the land zone", u.name);
        let clipped = full - g::area(&node);
        if clipped > 1e-5 {
            report.warnings.push(format!(
                "{}: settlement node clipped by {:.4} of {:.3} units² (the settlement is closer than the node radius to a border or coast)",
                u.name, clipped, full
            ));
        }
        u.node = node;
    }

    // ---- landing crossings
    let mut placed_all: Vec<Shapes> = Vec::new();
    for u in &units {
        if !u.node.is_empty() {
            placed_all.push(u.node.clone());
        }
    }
    let band_geoms: Vec<&Shapes> = bands.iter().map(|b| &b.geom).collect();
    for ui in 0..units.len() {
        let others_land = {
            let mut list: Vec<&Shapes> = units.iter().enumerate().filter(|(j, _)| *j != ui).map(|(_, u)| &u.lg).collect();
            list.extend(band_geoms.iter().copied());
            g::union_many(&list)
        };
        let sea_ok = g::union(&seacover, &zall);
        let pzone = g::union(&units[ui].lg, &units[ui].coast_zone);
        let lg = units[ui].lg.clone();
        let mut crs: Vec<Landing> = Vec::new();
        let mut gaps: Vec<f64> = Vec::new();
        let coast: Vec<Polyline> = {
            let mut v: Vec<Polyline> = units[ui].edges.coast.iter().map(|l| Polyline::new(l.clone())).collect();
            v.sort_by(|a, b| b.length().partial_cmp(&a.length()).unwrap());
            v
        };
        for line in &coast {
            let ln = line.length();
            if ln < MIN_LINE {
                continue;
            }
            let mut line_cr: Vec<Landing> = Vec::new();
            let try_at = |s: f64, line_cr: &Vec<Landing>, placed_all: &Vec<Shapes>| -> Option<Landing> {
                if s < 1.0 || s > ln - 1.0 {
                    return None;
                }
                if line_cr.iter().any(|o| (s - o.s).abs() < MIN_SEP) {
                    return None;
                }
                let cand = crossings::candidate(line, s);
                let sea: Shapes = vec![vec![cand.sea.clone()]];
                let land_: Shapes = vec![vec![cand.land.clone()]];
                if g::area(&g::difference(&land_, &pzone)) > ZONE_TOL || g::area(&g::intersection(&sea, &lg)) > ZONE_TOL {
                    return None;
                }
                if g::distance(&sea, &others_land) < 1.0 || g::distance(&land_, &others_land) < 1.0 {
                    return None;
                }
                if g::area(&g::difference(&sea, &sea_ok)) > ZONE_TOL {
                    return None;
                }
                if !blocked.is_empty() && (g::distance(&sea, &blocked) < 1.0 || g::distance(&land_, &blocked) < 1.0) {
                    return None;
                }
                let lim = 0.3 * FIX;
                if placed_all.iter().any(|o| g::distance(&sea, o) < lim || g::distance(&land_, o) < lim) {
                    return None;
                }
                Some(cand)
            };
            for want in crossings::wanted_positions(ln, opts.crossing_spacing) {
                let mut done = false;
                for d in SLIDE {
                    if let Some(mut cd) = try_at(want + d, &line_cr, &placed_all) {
                        cd.want = want;
                        placed_all.push(vec![vec![cd.sea.clone()]]);
                        placed_all.push(vec![vec![cd.land.clone()]]);
                        line_cr.push(cd);
                        done = true;
                        break;
                    }
                }
                if !done {
                    let at = line.interpolate(want);
                    report.warnings.push(format!(
                        "{}: no landing near s={:.1} of a {:.1} unit coast line at ({:.1}, {:.1})",
                        units[ui].name, want, ln, at[0] + fr.origin[0], at[1] + fr.origin[1]
                    ));
                }
            }
            // fill pass: any stretch longer than the allowed gap gets extra landings scanned over its whole length
            for _ in 0..20 {
                let mut ss: Vec<f64> = vec![0.0];
                let mut mids: Vec<f64> = line_cr.iter().map(|c| c.s).collect();
                mids.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ss.extend(mids);
                ss.push(ln);
                let big = ss.windows(2).filter(|w| w[1] - w[0] > opts.min_per).max_by(|a, b| (a[1] - a[0]).partial_cmp(&(b[1] - b[0])).unwrap()).map(|w| (w[0], w[1]));
                let Some((a2, b2)) = big else { break };
                let mut found = false;
                for s in crossings::fill_positions(a2, b2) {
                    if let Some(mut cd) = try_at(s, &line_cr, &placed_all) {
                        cd.want = s;
                        placed_all.push(vec![vec![cd.sea.clone()]]);
                        placed_all.push(vec![vec![cd.land.clone()]]);
                        line_cr.push(cd);
                        found = true;
                        break;
                    }
                }
                if !found {
                    report.warnings.push(format!("{}: a stretch of {:.1} units ({:.1}..{:.1}) of a coast line cannot take a landing", units[ui].name, b2 - a2, a2, b2));
                    break;
                }
            }
            let mut ss: Vec<f64> = line_cr.iter().map(|c| c.s).collect();
            ss.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if let (Some(first), Some(last)) = (ss.first(), ss.last()) {
                gaps.push(*first);
                gaps.extend(ss.windows(2).map(|w| w[1] - w[0]));
                gaps.push(ln - last);
            } else {
                gaps.push(ln);
            }
            crs.extend(line_cr);
        }
        let _ = gaps;
        units[ui].crosses = crs;
    }
    // quantise the halves (noded at the grid lines)
    for u in units.iter_mut() {
        u.sh = u.crosses.iter().map(|c| vec![g::node_ring(&c.sea, cs)]).collect();
        u.lh = u.crosses.iter().map(|c| vec![g::node_ring(&c.land, cs)]).collect();
    }
    let sh_all: Shapes = units.iter().flat_map(|u| u.sh.iter().cloned()).collect();
    let shu = g::node_shapes(&g::clean(&sh_all), cs);
    for u in units.iter_mut() {
        let mut parts: Shapes = u.lh.clone();
        parts.extend(u.node.iter().cloned());
        u.lhu = g::node_shapes(&g::clean(&parts), cs);
        let ll = g::difference(&u.lg, &u.lhu);
        let (small, large): (Vec<Shape>, Vec<Shape>) = ll.into_iter().partition(|q| g::shape_area(q) < SMALL_LAND);
        if !small.is_empty() {
            report.warnings.push(format!("{}: {} land fragments < {SMALL_LAND} left by the landings dropped into type 2", u.name, small.len()));
        }
        u.lland = g::node_shapes(&large, cs);
    }
    // closeness of crossing halves and nodes
    {
        let mut all: Vec<Shapes> = Vec::new();
        for u in &units {
            all.extend(u.lh.iter().map(|s| vec![s.clone()]));
            if !u.node.is_empty() {
                all.push(u.node.clone());
            }
        }
        all.extend(shu.iter().map(|s| vec![s.clone()]));
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                if g::distance(&all[i], &all[j]) < 0.05 * FIX && g::area(&g::intersection(&all[i], &all[j])) > 1e-9 {
                    report.warnings.push("a landing half and a settlement node or another landing overlap".into());
                }
            }
        }
    }

    // ---- layers
    let mut layers: Vec<Layer> = Vec::new();
    let mut push_layer = |geom: Shapes, ptype: u8, pid: u16| {
        if let Some(bbox) = g::shapes_bbox(&geom) {
            layers.push(Layer { geom, bbox, ptype, pid });
        }
    };
    for u in &units {
        push_layer(u.lland.clone(), 0, u.pid);
        push_layer(u.lhu.clone(), 7, u.pid);
    }
    push_layer(shu.clone(), 7, sea_pid);
    for b in &bands {
        push_layer(b.geom.clone(), 0, b.pid);
    }
    let cz = {
        let mut list: Vec<&Shapes> = vec![&zall, &shu];
        list.extend(units.iter().map(|u| &u.lhu));
        g::node_shapes(&g::union_many(&list), cs)
    };
    let cz_bbox = g::shapes_bbox(&cz).ok_or_else(|| anyhow!("empty zone"))?;
    let c_lo = ((cz_bbox.0[0] / cs) - 1).max(0) as usize;
    let c_hi = (((cz_bbox.1[0] / cs) + 1).max(0) as usize).min(cols - 1);
    let r_lo = ((cz_bbox.0[1] / cs) - 1).max(0) as usize;
    let r_hi = (((cz_bbox.1[1] / cs) + 1).max(0) as usize).min(rows - 1);
    let cs2x2 = 2 * (cs as i128) * (cs as i128);

    // ---- per cell clipping
    let mut touched: BTreeMap<usize, bool> = BTreeMap::new(); // idx -> became a run cell
    let mut gain_sea: BTreeSet<usize> = BTreeSet::new();
    for r in r_lo..=r_hi {
        for c in c_lo..=c_hi {
            let idx = r * cols + c;
            let (lo, hi) = ([c as i32 * cs, r as i32 * cs], [(c as i32 + 1) * cs, (r as i32 + 1) * cs]);
            let bx: Shapes = vec![g::rect_shape(lo, hi)];
            if !near_bbox(&cz_bbox, lo, hi) {
                continue;
            }
            let cz_frag = g::clip_rect(&cz, lo, hi);
            if g::area(&cz_frag) < AREA_EPS {
                continue;
            }
            let cell = &m.cells[idx];
            let old_t: BTreeSet<u8> = cell.types().into_iter().collect();
            let frags: Vec<Shapes> = layers
                .iter()
                .map(|l| if near_bbox(&l.bbox, lo, hi) { g::clip_rect(&l.geom, lo, hi) } else { Shapes::new() })
                .collect();
            if old_t.iter().all(|t| *t == 1 || *t == 2) {
                let hit = layers.iter().zip(&frags).find(|(l, f)| {
                    l.ptype == 0 && l.pid < TMP_GROUP && f.len() == 1 && f[0].len() == 1 && g::area2(&f[0][0]) == cs2x2
                });
                if let Some((l, _)) = hit {
                    let cell = &mut m.cells[idx];
                    cell.pieces.clear();
                    cell.run = Some((0, l.pid));
                    cell.hdr = [LAND_CLASS; 8];
                    touched.insert(idx, true);
                    continue;
                }
            }
            let mut news: Vec<(Shape, u8, u16)> = Vec::new();
            for (l, f) in layers.iter().zip(&frags) {
                for q in g::polys_of(f, AREA_EPS) {
                    news.push((q, l.ptype, l.pid));
                }
            }
            let mut gained: Vec<Shape> = Vec::new();
            let cell = m.cells[idx].clone();
            let mut seas: Vec<Option<MPiece>> = Vec::new();
            let mut kept: Vec<MPiece> = Vec::new();
            let mut cell_sea_pid = sea_pid;
            if !cell.pieces.is_empty() {
                for p in &cell.pieces {
                    match p.ptype() {
                        1 => {
                            cell_sea_pid = p.pid;
                            seas.push(Some(p.clone()));
                        }
                        2 => {}
                        _ => kept.push(p.clone()),
                    }
                }
            } else {
                let (a0, pid0) = cell.run.ok_or_else(|| anyhow!("cell ({c},{r}) has neither pieces nor a run"))?;
                match (a0 & 15) as u8 {
                    2 => {
                        // whole outside cell of the old grid: the sea next to the new coast has to be created here
                        let sea_here = g::difference(&g::intersection(&sea_q, &bx), &cz);
                        for q in sea_here {
                            if is_sliver(&q) {
                                continue;
                            }
                            if g::shape_area(&q) < 1.0 && !touches_cell_edge(&q, lo, hi) {
                                continue;
                            }
                            gained.push(q);
                        }
                    }
                    1 => {
                        cell_sea_pid = pid0;
                        seas.push(None);
                    }
                    0 => kept.push(MPiece {
                        a: a0,
                        pid: pid0,
                        gbit: 0,
                        pts: vec![[hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]], [hi[0], lo[1]]],
                        src: None,
                    }),
                    t => bail!("the zone reaches a run cell of type {t} at ({c},{r})"),
                }
            }
            // kept land pieces: clipped against the zone
            let mut kept_out: Vec<MPiece> = Vec::new();
            let mut kept_changed = false;
            for p in kept {
                let poly: Shapes = vec![vec![p.pts.clone()]];
                if g::area(&g::intersection(&poly, &cz)) <= AREA_EPS {
                    kept_out.push(p);
                    continue;
                }
                if p.ptype() != 0 {
                    bail!("a type-{} piece lies inside the new zone in cell ({c},{r})", p.ptype());
                }
                kept_changed = true;
                for q in g::polys_of(&g::difference(&poly, &cz), AREA_EPS) {
                    if is_sliver(&q) {
                        continue;
                    }
                    for part in g::split_holes(&q) {
                        kept_out.push(MPiece { a: 0, pid: p.pid, gbit: p.gbit, pts: g::ring_pts(&part[0]), src: None });
                    }
                }
            }
            let kept = kept_out;
            let mut sea_keep: Vec<(MPiece, Shapes)> = Vec::new();
            let mut sea_changed = false;
            for s in seas {
                let (sp, spid, orig): (Shapes, u16, Option<MPiece>) = match s {
                    Some(p) => (vec![vec![p.pts.clone()]], p.pid, Some(p)),
                    None => (bx.clone(), cell_sea_pid, None),
                };
                if g::area(&g::intersection(&sp, &cz)) <= AREA_EPS {
                    match orig {
                        Some(p) => sea_keep.push((p, sp)),
                        None => bail!("an unchanged sea run is inside a touched cell ({c},{r})"),
                    }
                    continue;
                }
                sea_changed = true;
                for q in g::polys_of(&g::difference(&sp, &cz), AREA_EPS) {
                    if is_sliver(&q) {
                        continue;
                    }
                    if g::shape_area(&q) < 1.0 && !touches_cell_edge(&q, lo, hi) {
                        continue;
                    }
                    for part in g::split_holes(&q) {
                        let pts = g::ring_pts(&part[0]);
                        sea_keep.push((MPiece { a: 1, pid: spid, gbit: 0, pts: pts.clone(), src: None }, vec![vec![pts]]));
                    }
                }
            }
            for q in gained {
                for part in g::split_holes(&q) {
                    let pts = g::ring_pts(&part[0]);
                    sea_keep.push((MPiece { a: 1, pid: sea_pid, gbit: 0, pts: pts.clone(), src: None }, vec![vec![pts]]));
                }
                sea_changed = true;
                gain_sea.insert(idx);
            }
            if news.is_empty() && !sea_changed && !kept_changed {
                continue;
            }
            let mut solid: Shapes = news.iter().map(|(q, _, _)| q.clone()).collect();
            for (_, s) in &sea_keep {
                solid.extend(s.iter().cloned());
            }
            for p in &kept {
                solid.push(vec![p.pts.clone()]);
            }
            for (q, t, pid) in &news {
                let qs: Shapes = vec![q.clone()];
                for (_, s) in &sea_keep {
                    if g::area(&g::intersection(&qs, s)) > 1e-5 {
                        bail!("new piece (type {t} pid {pid}) overlaps a sea piece in cell ({c},{r})");
                    }
                }
                for p in &kept {
                    if g::area(&g::intersection(&qs, &vec![vec![p.pts.clone()]])) > 1e-5 {
                        bail!("new piece (type {t} pid {pid}) overlaps kept land in cell ({c},{r})");
                    }
                }
            }
            let rests = g::difference(&bx, &g::clean(&solid));
            if rests.iter().all(|q| g::shape_area(q) <= AREA_EPS) && sea_keep.is_empty() && kept.is_empty() && news.len() == 1 && news[0].1 == 0 {
                let cell = &mut m.cells[idx];
                cell.pieces.clear();
                cell.run = Some((0, news[0].2));
                cell.hdr = [LAND_CLASS; 8];
                touched.insert(idx, true);
                continue;
            }
            if news.is_empty() && sea_keep.is_empty() && kept.is_empty() {
                bail!("cell ({c},{r}) becomes a pure type-2 cell");
            }
            let mut newp: Vec<MPiece> = Vec::new();
            for (q, t, pid) in &news {
                for part in g::split_holes(q) {
                    newp.push(MPiece { a: *t as u32, pid: *pid, gbit: 0, pts: g::ring_pts(&part[0]), src: None });
                }
            }
            for q in g::polys_of(&rests, AREA_EPS) {
                for part in g::split_holes(&q) {
                    newp.push(MPiece { a: 2, pid: OUTSIDE, gbit: 0, pts: g::ring_pts(&part[0]), src: None });
                }
            }
            let mut allp: Vec<MPiece> = kept;
            allp.extend(sea_keep.into_iter().map(|(p, _)| p));
            allp.extend(newp);
            allp.sort_by_key(|p| *p.pts.iter().min().unwrap());
            let cell = &mut m.cells[idx];
            cell.pieces = allp;
            cell.run = None;
            touched.insert(idx, false);
        }
    }
    report.touched_cells = touched.len();

    // ---- vertex hygiene + a-words
    let touched_set: BTreeSet<usize> = touched.keys().copied().collect();
    let (hyg, _) = finish::hygiene_and_words(&mut m, &old, &touched);
    report.vertices_merged = hyg.merged;
    report.degenerate_dropped = hyg.dropped;
    report.tjunctions = hyg.tjunctions;
    report.edge_vertices_synced = hyg.synced;
    report.warnings.extend(hyg.failures);
    let _ = &touched_set;

    // ---- headers of touched piece cells
    for (&idx, &is_run) in &touched {
        if is_run && m.cells[idx].pieces.is_empty() {
            continue;
        }
        let (cc, rr) = (idx % cols, idx / cols);
        let corners = m.corner(cc, rr);
        let types: BTreeSet<u8> = m.cells[idx].pieces.iter().map(|p| p.ptype()).collect();
        let old_types: BTreeSet<u8> = old.cells[idx].types().into_iter().collect();
        let mut hdr: [u8; 8];
        if gain_sea.contains(&idx) {
            // new sea piece in a former outside cell: take the nearest old cell that is entirely sea
            let mut found = None;
            'rad: for rad in 1..=5i32 {
                let mut cands: Vec<(i32, usize)> = Vec::new();
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        if dx.abs().max(dy.abs()) != rad {
                            continue;
                        }
                        let (nc, nr) = (cc as i32 + dx, rr as i32 + dy);
                        if nc < 0 || nr < 0 || nc >= cols as i32 || nr >= rows as i32 {
                            continue;
                        }
                        let j = nr as usize * cols + nc as usize;
                        let t: BTreeSet<u8> = old.cells[j].types().into_iter().collect();
                        if t.len() == 1 && t.contains(&1) {
                            cands.push((dx * dx + dy * dy, j));
                        }
                    }
                }
                if let Some(best) = cands.into_iter().min() {
                    found = Some(old.cells[best.1].hdr);
                    break 'rad;
                }
            }
            hdr = found.ok_or_else(|| anyhow!("no sea neighbour for the header of cell {idx}"))?;
        } else if types.contains(&1) || old_types.iter().any(|t| matches!(t, 0 | 6 | 7)) {
            hdr = old.cells[idx].hdr;
        } else {
            hdr = [LAND_CLASS; 8];
        }
        let mut counts: BTreeMap<u8, usize> = BTreeMap::new();
        for &b in hdr.iter().filter(|&&b| b != 0xff) {
            *counts.entry(b).or_insert(0) += 1;
        }
        let cls = counts.iter().max_by_key(|(b, n)| (**n, **b)).map(|(b, _)| *b).unwrap_or(LAND_CLASS);
        // (direction byte index, corner id): NW, NE, SW, SE
        for (bi, cid) in [(1usize, 1usize), (3, 3), (4, 0), (6, 2)] {
            let pt = corners[cid];
            let at_bad = m.cells[idx].pieces.iter().any(|p| matches!(p.ptype(), 2 | 3) && p.pts.contains(&pt));
            if at_bad {
                hdr[bi] = 0xff;
            } else if hdr[bi] == 0xff {
                hdr[bi] = cls;
            }
        }
        m.cells[idx].hdr = hdr;
    }

    // ---- group paths: final raster-order numbering
    let sea_idx = singles_idx + n_new;
    let mut groups: BTreeMap<u16, Vec<usize>> = BTreeMap::new();
    for pid in (sea_idx + 1)..grid.paths.len() {
        let members: Vec<usize> = grid.paths[pid].iter().map(|&r| pos_of(r as usize).unwrap_or(0)).collect();
        groups.insert(pid as u16, members);
    }
    groups.extend(group_members.iter().map(|(k, v)| (*k, v.clone())));
    for (pid, mem) in &groups {
        ensure!(
            !mem.iter().any(|&p| p > 0 && no_group.contains(&(region_ids[p - 1] as usize))),
            "group {pid} {mem:?} contains a no-group region"
        );
    }
    let groups_ids: BTreeMap<u16, Vec<i16>> = groups.iter().map(|(k, v)| (*k, v.iter().map(|&p| region_ids[p - 1]).collect())).collect();
    let (newid, unused) = finish::renumber_groups(&mut m, grid, sea_idx, &groups_ids)?;
    for b in &bands {
        match newid.get(&b.pid) {
            Some(&id) => report.new_groups.push((b.members.iter().map(|&p| region_ids[p - 1]).collect(), id)),
            None => report.warnings.push(format!("border band {} produced no pieces", b.name)),
        }
    }
    if unused > 0 {
        report.warnings.push(format!("{unused} group path(s) not used by any cell were dropped"));
    }

    for u in &units {
        if let Some(r) = report.per_region.iter_mut().find(|r| r.region == u.region) {
            r.landings += u.crosses.len();
            r.node_clipped |= u.node_clipped_flag(&report.warnings);
            continue;
        }
        report.per_region.push(RegionReport {
            region: u.region,
            path_id: u.pid,
            landings: u.crosses.len(),
            node_clipped: u.node_clipped_flag(&report.warnings),
        });
    }
    report.cells_to_runs = touched.values().filter(|&&v| v).count();
    let _ = (flat as fn(&Shapes) -> Vec<&Shape>, cs_f);
    m.write_into(grid)?;
    Ok(report)
}

impl Unit {
    fn node_clipped_flag(&self, warnings: &[String]) -> bool {
        warnings.iter().any(|w| w.starts_with(&format!("{}: settlement node clipped", self.name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f64, x1: f64) -> Shapes {
        let f = |v: f64| (v * FIX) as i32;
        vec![g::rect_shape([f(x0), 0], [f(x1), f(1.0)])]
    }

    /// Strips of one group path that overlap (two areas of a region touching the same neighbour next to each other, the
    /// halves of a split meeting at a border vertex) must come out as one union, not as two overlapping pieces.
    #[test]
    fn strips_of_one_group_are_united() {
        let out = unite_band_strips(vec![rect(0.0, 2.0), rect(1.0, 3.0)]);
        assert!((g::area(&out) - 3.0).abs() < 1e-9, "united area {}", g::area(&out));
    }

}
