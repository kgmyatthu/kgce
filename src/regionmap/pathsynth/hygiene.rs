//! Vertex hygiene of the touched cells (port of canonicalize_vertices / fix_tjunctions / edge-vertex consistency of
//! work/exp4/pathsynth.py): merge rounding-noise duplicates (within 2 fixed units), insert T-junction vertices into
//! the edges of the same cell, and make the vertices on every shared cell edge identical on both sides.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::geometry::{Pt, area2};
use super::model::Model;

/// vertices closer than this (fixed-point units, per axis) are the same vertex
pub const SNAP: i32 = 2;

fn dedup_ring(pts: impl Iterator<Item = Pt>) -> Vec<Pt> {
    let mut out: Vec<Pt> = Vec::new();
    for p in pts {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    if out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

/// Merge vertices of the cells `idxs` that differ by <= SNAP; the representative is the vertex of the original grid,
/// else one on a grid line, else the most used. Returns (vertices merged, degenerate pieces dropped).
pub fn canonicalize(m: &mut Model, idxs: &[usize], orig: &HashSet<Pt>) -> (usize, usize) {
    let mut cnt: HashMap<Pt, usize> = HashMap::new();
    for &i in idxs {
        for p in &m.cells[i].pieces {
            for &pt in &p.pts {
                *cnt.entry(pt).or_insert(0) += 1;
            }
        }
    }
    let mut pts: Vec<Pt> = cnt.keys().copied().collect();
    pts.sort();
    let index: HashMap<Pt, usize> = pts.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let mut par: Vec<usize> = (0..pts.len()).collect();
    fn find(par: &mut [usize], mut x: usize) -> usize {
        while par[x] != x {
            par[x] = par[par[x]];
            x = par[x];
        }
        x
    }
    let b = 4 * SNAP;
    let mut bucket: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        bucket.entry((p[0].div_euclid(b), p[1].div_euclid(b))).or_default().push(i);
    }
    for (i, p) in pts.iter().enumerate() {
        let (bx, by) = (p[0].div_euclid(b), p[1].div_euclid(b));
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(list) = bucket.get(&(bx + dx, by + dy)) {
                    for &j in list {
                        let q = pts[j];
                        if j != i && (q[0] - p[0]).abs() <= SNAP && (q[1] - p[1]).abs() <= SNAP {
                            let (a, c) = (find(&mut par, i), find(&mut par, j));
                            if a != c {
                                par[a] = c;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<Pt>> = HashMap::new();
    for i in 0..pts.len() {
        let r = find(&mut par, i);
        groups.entry(r).or_default().push(pts[i]);
    }
    let cs = m.cs;
    let on_grid = |p: Pt| p[0].rem_euclid(cs) == 0 || p[1].rem_euclid(cs) == 0;
    let mut rep: HashMap<Pt, Pt> = HashMap::new();
    let mut merged = 0;
    for (_, grp) in groups {
        if grp.len() == 1 {
            continue;
        }
        let best = *grp.iter().max_by_key(|p| (orig.contains(*p), on_grid(**p), cnt[*p], **p)).unwrap();
        for p in &grp {
            rep.insert(*p, best);
        }
        merged += grp.len() - 1;
    }
    let _ = index;
    let mut dropped = 0;
    for &i in idxs {
        let cell = &mut m.cells[i];
        let mut keep = Vec::with_capacity(cell.pieces.len());
        for mut p in std::mem::take(&mut cell.pieces) {
            if p.pts.iter().any(|q| rep.contains_key(q)) {
                let ring = dedup_ring(p.pts.iter().map(|q| *rep.get(q).unwrap_or(q)));
                if ring.len() < 3 || area2(&ring) <= 0 {
                    dropped += 1;
                    continue;
                }
                p.pts = ring;
            }
            keep.push(p);
        }
        cell.pieces = keep;
    }
    (merged, dropped)
}

/// Insert every vertex of a cell that lies on the interior of an edge of another piece of the same cell into that
/// edge. Returns the number of vertices inserted.
pub fn fix_tjunctions(m: &mut Model, idxs: &[usize]) -> usize {
    let mut n_ins = 0;
    for &i in idxs {
        let cell = &mut m.cells[i];
        if cell.pieces.len() < 2 {
            continue;
        }
        let all: BTreeSet<Pt> = cell.pieces.iter().flat_map(|p| p.pts.iter().copied()).collect();
        for p in cell.pieces.iter_mut() {
            let n = p.pts.len();
            let mut out: Vec<Pt> = Vec::with_capacity(n + 2);
            for k in 0..n {
                let (a, b) = (p.pts[k], p.pts[(k + 1) % n]);
                out.push(a);
                let (dx, dy) = ((b[0] - a[0]) as f64, (b[1] - a[1]) as f64);
                let l2 = dx * dx + dy * dy;
                if l2 == 0.0 {
                    continue;
                }
                let l = l2.sqrt();
                let mut ins: Vec<(f64, Pt)> = Vec::new();
                for &v in &all {
                    if v == a || v == b {
                        continue;
                    }
                    let (vx, vy) = ((v[0] - a[0]) as f64, (v[1] - a[1]) as f64);
                    let t = (vx * dx + vy * dy) / l2;
                    if !(t > 0.0 && t < 1.0) {
                        continue;
                    }
                    if (dx * vy - dy * vx).abs() / l <= 1.5 {
                        ins.push((t, v));
                    }
                }
                ins.sort_by(|x, y| x.partial_cmp(y).unwrap());
                n_ins += ins.len();
                out.extend(ins.into_iter().map(|x| x.1));
            }
            p.pts = out;
        }
    }
    n_ins
}

#[derive(Clone, Copy)]
enum Side {
    W,
    E,
    S,
    N,
}

fn side_pts(m: &Model, idx: usize, side: Side) -> BTreeSet<Pt> {
    let (c, r) = (idx % m.cols, idx / m.cols);
    let (ox, oy) = (c as i32 * m.cs, r as i32 * m.cs);
    let cs = m.cs;
    let mut out = BTreeSet::new();
    for p in &m.cells[idx].pieces {
        for &pt in &p.pts {
            let hit = match side {
                Side::W => pt[0] == ox && oy < pt[1] && pt[1] < oy + cs,
                Side::E => pt[0] == ox + cs && oy < pt[1] && pt[1] < oy + cs,
                Side::S => pt[1] == oy && ox < pt[0] && pt[0] < ox + cs,
                Side::N => pt[1] == oy + cs && ox < pt[0] && pt[0] < ox + cs,
            };
            if hit {
                out.insert(pt);
            }
        }
    }
    out
}

fn insert_pt(m: &mut Model, idx: usize, side: Side, pt: Pt) -> bool {
    let (c, r) = (idx % m.cols, idx / m.cols);
    let (ox, oy) = (c as i32 * m.cs, r as i32 * m.cs);
    let cs = m.cs;
    if m.cells[idx].pieces.is_empty() {
        // a run cell that has to carry an edge vertex becomes a one-piece cell
        if let Some((a, pid)) = m.cells[idx].run {
            let (lo, hi) = ([ox, oy], [ox + cs, oy + cs]);
            m.cells[idx].pieces.push(super::model::MPiece {
                a,
                pid,
                gbit: 0,
                pts: vec![[hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]], [hi[0], lo[1]]],
                src: None,
            });
            m.cells[idx].run = None;
        }
    }
    for p in m.cells[idx].pieces.iter_mut() {
        let n = p.pts.len();
        for i in 0..n {
            let (a, b) = (p.pts[i], p.pts[(i + 1) % n]);
            let ok = match side {
                Side::W | Side::E => {
                    let x = if matches!(side, Side::W) { ox } else { ox + cs };
                    a[0] == x && b[0] == x && a[1].min(b[1]) < pt[1] && pt[1] < a[1].max(b[1])
                }
                Side::S | Side::N => {
                    let y = if matches!(side, Side::S) { oy } else { oy + cs };
                    a[1] == y && b[1] == y && a[0].min(b[0]) < pt[0] && pt[0] < a[0].max(b[0])
                }
            };
            if ok {
                p.pts.insert(i + 1, pt);
                return true;
            }
        }
    }
    false
}

/// Make the edge vertices of every touched cell identical to those of its neighbours (inserting the missing ones on
/// either side). Returns (vertices inserted, cells modified outside `touched`, failures).
pub fn sync_edges(m: &mut Model, touched: &[usize]) -> (usize, BTreeSet<usize>, Vec<String>) {
    const SIDES: [(Side, Side, i32, i32); 4] = [(Side::W, Side::E, -1, 0), (Side::E, Side::W, 1, 0), (Side::S, Side::N, 0, -1), (Side::N, Side::S, 0, 1)];
    let tset: BTreeSet<usize> = touched.iter().copied().collect();
    let mut n = 0;
    let mut modified = BTreeSet::new();
    let mut failures = Vec::new();
    for &idx in touched {
        let (c, r) = ((idx % m.cols) as i32, (idx / m.cols) as i32);
        for (side, opp, dx, dy) in SIDES {
            let (nc, nr) = (c + dx, r + dy);
            if nc < 0 || nr < 0 || nc >= m.cols as i32 || nr >= m.rows as i32 {
                continue;
            }
            let j = nr as usize * m.cols + nc as usize;
            let (a, b) = (side_pts(m, idx, side), side_pts(m, j, opp));
            for &pt in b.difference(&a) {
                if insert_pt(m, idx, side, pt) {
                    n += 1;
                } else {
                    failures.push(format!("cell {idx}: cannot insert edge vertex {pt:?} from neighbour {j}"));
                }
            }
            for &pt in a.difference(&b) {
                if insert_pt(m, j, opp, pt) {
                    n += 1;
                    if !tset.contains(&j) {
                        modified.insert(j);
                    }
                } else {
                    failures.push(format!("cell {j}: cannot insert edge vertex {pt:?} from neighbour {idx}"));
                }
            }
        }
    }
    (n, modified, failures)
}

/// Remove spikes (a, b, a) and repeated vertices from the pieces of the cells `idxs` and drop pieces without area
/// (they cover nothing, so the tiling is unchanged). Returns the number of pieces dropped.
pub fn drop_degenerate(m: &mut Model, idxs: &[usize]) -> usize {
    let mut dropped = 0;
    for &i in idxs {
        let cell = &mut m.cells[i];
        if cell.pieces.len() < 2 {
            continue;
        }
        let mut keep = Vec::with_capacity(cell.pieces.len());
        for mut p in std::mem::take(&mut cell.pieces) {
            let mut pts = dedup_ring(p.pts.iter().copied());
            loop {
                let n = pts.len();
                if n < 3 {
                    break;
                }
                match (0..n).find(|&k| pts[(k + n - 1) % n] == pts[(k + 1) % n]) {
                    Some(k) => {
                        let next = (k + 1) % n;
                        let mut out: Vec<Pt> = Vec::with_capacity(n - 2);
                        for (j, q) in pts.iter().enumerate() {
                            if j != k && j != next {
                                out.push(*q);
                            }
                        }
                        pts = dedup_ring(out.into_iter());
                    }
                    None => break,
                }
            }
            if pts.len() < 3 || area2(&pts) <= 0 {
                dropped += 1;
                continue;
            }
            p.pts = pts;
            keep.push(p);
        }
        cell.pieces = keep;
    }
    dropped
}
