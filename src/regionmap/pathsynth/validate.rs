//! Rule checker for a pathfinding grid (port of work/research/pf_decode.py run_checks + the format facts listed there).
//! A piece word is accepted when it equals the rule prediction under either length tolerance (see `rules::PY_LEN_EPS`).

use std::collections::BTreeMap;

use super::geometry::{FIX, area2};
use super::model::{Model, OUTSIDE};
use super::rules::{GAME_LEN_EPS, PY_LEN_EPS, View};
use crate::regionmap::pathfinding::Grid;
use crate::regionmap::regions::RegionData;

/// Tolerance of the per-cell tiling test (map units²).
const TILING_TOL: f64 = 1e-4;
const MAX_LISTED: usize = 12;

struct Problems {
    limit: usize,
    out: Vec<String>,
    counts: BTreeMap<&'static str, usize>,
}

impl Problems {
    fn add(&mut self, kind: &'static str, msg: impl FnOnce() -> String) {
        let n = self.counts.entry(kind).or_insert(0);
        *n += 1;
        if *n <= self.limit {
            self.out.push(msg());
        }
    }
}

/// Check `grid` against the decoded format rules; every returned string is one problem (an empty list = valid).
/// The number of problems of one kind past the first dozen is summarised in a trailing line.
pub fn validate(grid: &Grid, rd: &RegionData) -> Vec<String> {
    validate_limited(grid, rd, MAX_LISTED)
}

/// [`validate`] without the listing cap: every problem is one string (for comparing against a baseline).
pub fn validate_all(grid: &Grid, rd: &RegionData) -> Vec<String> {
    validate_limited(grid, rd, usize::MAX)
}

fn validate_limited(grid: &Grid, rd: &RegionData, limit: usize) -> Vec<String> {
    let mut pr = Problems { limit, out: Vec::new(), counts: BTreeMap::new() };
    let model = match Model::from_grid(grid) {
        Ok(m) => m,
        Err(e) => return vec![format!("grid cannot be decoded: {e}")],
    };
    check_tables(grid, rd, &mut pr);
    check_vertices(grid, &mut pr);
    check_cells(&model, grid, &mut pr);
    check_words(&model, &mut pr);
    let mut out = pr.out;
    for (k, n) in &pr.counts {
        if *n > limit {
            out.push(format!("{k}: {n} problems in total ({} not listed)", n - limit));
        }
    }
    out
}

fn check_tables(g: &Grid, rd: &RegionData, pr: &mut Problems) {
    let n = rd.regions.len();
    let mut seen = std::collections::BTreeSet::new();
    for &r in &g.region_ids {
        if r < 0 || r as usize >= n {
            pr.add("region list", || format!("region list entry {r} is not a regions.esf index (< {n})"));
        } else if !seen.insert(r) {
            pr.add("region list", || format!("region {r} listed twice"));
        }
    }
    let singles = g.paths.iter().position(|p| p.is_empty());
    let Some(sea) = singles else {
        pr.add("path table", || "no sea path".into());
        return;
    };
    if sea != g.region_ids.len() {
        pr.add("path table", || format!("{} single paths for {} listed regions", sea, g.region_ids.len()));
    }
    for (i, p) in g.paths[..sea].iter().enumerate() {
        if p.len() != 1 || g.region_ids.get(i) != p.first() {
            pr.add("path table", || format!("single path {i} is {p:?}, expected the region list entry"));
        }
    }
    let pos = |r: i16| g.region_ids.iter().position(|&x| x == r);
    let no_group: Vec<usize> = rd.regions.iter().enumerate().filter(|(_, r)| r.key == "unexplorable").map(|(i, _)| i).collect();
    let mut groups = std::collections::BTreeSet::new();
    for (pid, p) in g.paths.iter().enumerate().skip(sea + 1) {
        if p.len() < 2 {
            pr.add("path table", || format!("group path {pid} has {} members", p.len()));
        }
        let ps: Vec<Option<usize>> = p.iter().map(|&r| pos(r)).collect();
        if ps.iter().any(|x| x.is_none()) {
            pr.add("path table", || format!("group path {pid} {p:?} names a region outside the region list"));
        } else if ps.windows(2).any(|w| w[0] >= w[1]) {
            pr.add("path table", || format!("group path {pid} {p:?} is not in ascending region-list order"));
        }
        if p.iter().any(|&r| r >= 0 && no_group.contains(&(r as usize))) {
            pr.add("path table", || format!("group path {pid} {p:?} contains unexplorable"));
        }
        if !groups.insert(p.clone()) {
            pr.add("path table", || format!("group path {pid} {p:?} is listed twice"));
        }
    }
    // group numbering = raster order of the first cell (and piece index) holding the pid
    let mut first: BTreeMap<u16, (usize, usize)> = BTreeMap::new();
    for (idx, c) in g.cells.iter().enumerate() {
        match c.empty {
            Some((_, id)) if c.pieces.is_empty() => {
                first.entry(id).or_insert((idx, 0));
            }
            _ => {
                for (k, p) in c.pieces.iter().enumerate() {
                    first.entry(p.path_id()).or_insert((idx, k));
                }
            }
        }
    }
    for (&pid, _) in &first {
        if pid != OUTSIDE && pid as usize >= g.paths.len() {
            pr.add("path ids", || format!("path id {pid} is outside the path table ({} paths)", g.paths.len()));
        }
    }
    let order: Vec<u16> = {
        let mut v: Vec<(u16, (usize, usize))> = first.iter().filter(|&(&p, _)| (p as usize) > sea && (p as usize) < g.paths.len()).map(|(a, b)| (*a, *b)).collect();
        v.sort_by_key(|x| x.1);
        v.into_iter().map(|x| x.0).collect()
    };
    if order.windows(2).any(|w| w[0] > w[1]) {
        pr.add("group order", || "group paths are not numbered in raster order of first appearance".into());
    }
    for pid in (sea + 1)..g.paths.len() {
        if !first.contains_key(&(pid as u16)) {
            pr.add("group order", || format!("group path {pid} {:?} is not used by any cell", g.paths[pid]));
        }
    }
}

fn check_vertices(g: &Grid, pr: &mut Problems) {
    let v = &g.vertices;
    for i in 5..v.len() {
        if (v[i][1], v[i][0]) < (v[i - 1][1], v[i - 1][0]) {
            pr.add("vertex order", || format!("vertex table is not sorted by (y, x) at id {i}"));
        }
    }
    let mut next = 0usize;
    for (&off, ids) in &g.polylines {
        if off != next {
            pr.add("polylines", || format!("polyline table gap at offset {next}"));
        }
        next = off + 1 + ids.len();
    }
}

fn check_cells(m: &Model, g: &Grid, pr: &mut Problems) {
    let cs2 = m.cs as f64 * m.cs as f64 / (FIX * FIX);
    for (idx, cell) in m.cells.iter().enumerate() {
        let (c, r) = (idx % m.cols, idx / m.cols);
        if cell.pieces.is_empty() {
            if cell.run.is_none() {
                pr.add("cells", || format!("cell ({c},{r}) has neither pieces nor a run word"));
            }
            continue;
        }
        let mut sum = 0.0;
        for (k, p) in cell.pieces.iter().enumerate() {
            let a = area2(&p.pts);
            if p.pts.len() < 3 || a <= 0 {
                pr.add("ccw", || format!("cell ({c},{r}) piece {k} (type {}) is not a counter-clockwise polygon (area2 {a}, ring {:?}, cell pieces {:?})", p.ptype(), p.pts, cell.pieces.iter().map(|q| (q.ptype(), q.pts.len())).collect::<Vec<_>>()));
            }
            if p.pts.first() == p.pts.last() && p.pts.len() > 1 {
                pr.add("closed ring", || format!("cell ({c},{r}) piece {k} repeats its first vertex"));
            }
            sum += a as f64 / 2.0 / (FIX * FIX);
        }
        if (sum - cs2).abs() > TILING_TOL {
            pr.add("tiling", || format!("cell ({c},{r}) pieces cover {sum:.6} of {cs2:.6} map units²"));
        }
        let _ = g;
    }
}

fn check_words(m: &Model, pr: &mut Problems) {
    let py = View::build_with(m, PY_LEN_EPS);
    let game = View::build_with(m, GAME_LEN_EPS);
    for (idx, cell) in m.cells.iter().enumerate() {
        let (c, r) = (idx % m.cols, idx / m.cols);
        if cell.pieces.is_empty() {
            let Some((a, _)) = cell.run else { continue };
            let want = |v: &View| {
                let (pass, unk) = v.predict(idx, 0);
                (pass << 24) | (unk << 4) | (a & 15)
            };
            if want(&py) != a && want(&game) != a {
                pr.add("run word", || format!("run cell ({c},{r}) word {a:#010x}, rule gives {:#010x}", want(&py)));
            }
        } else {
            for (k, p) in cell.pieces.iter().enumerate() {
                if py.word(idx, k) != p.a && game.word(idx, k) != p.a {
                    pr.add("piece word", || format!("cell ({c},{r}) piece {k} (type {}) word {:#010x}, rule gives {:#010x}", p.ptype(), p.a, py.word(idx, k)));
                }
            }
        }
    }
}
