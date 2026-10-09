//! Passes shared by `synthesize` and `synthesize_split` after the cell contents changed: vertex hygiene, a-words and
//! the raster-order renumbering of the group paths.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Result, ensure};

use super::geometry::Pt;
use super::hygiene;
use super::model::Model;
use super::rules::View;
use crate::regionmap::pathfinding::Grid;

/// The cells of `set` and their 8 neighbours.
pub fn neighbours(m: &Model, set: &BTreeSet<usize>) -> BTreeSet<usize> {
    let (cols, rows) = (m.cols as i32, m.rows as i32);
    let mut out = set.clone();
    for &idx in set {
        let (cc, rr) = ((idx % m.cols) as i32, (idx / m.cols) as i32);
        for dx in -1..=1 {
            for dy in -1..=1 {
                let (nc, nr) = (cc + dx, rr + dy);
                if nc >= 0 && nr >= 0 && nc < cols && nr < rows {
                    out.insert(nr as usize * m.cols + nc as usize);
                }
            }
        }
    }
    out
}

#[derive(Default)]
pub struct HygieneReport {
    pub merged: usize,
    pub dropped: usize,
    pub tjunctions: usize,
    pub synced: usize,
    pub failures: Vec<String>,
}

/// Vertex hygiene + a-words of the touched cells. `touched` maps a cell index to "became a run cell" (such cells get
/// the rule word of their single full-cell piece); `old` is the model before the change. Returns the report and the
/// cells outside `touched` whose edge vertices had to change.
pub fn hygiene_and_words(m: &mut Model, old: &Model, touched: &BTreeMap<usize, bool>) -> (HygieneReport, BTreeSet<usize>) {
    let touched_set: BTreeSet<usize> = touched.keys().copied().collect();
    let around = neighbours(m, &touched_set);
    let orig: HashSet<Pt> = around.iter().flat_map(|&i| old.cells[i].pieces.iter().flat_map(|p| p.pts.iter().copied())).collect();
    let around_v: Vec<usize> = around.iter().copied().collect();
    let (merged, dropped) = hygiene::canonicalize(m, &around_v, &orig);
    let touched_v: Vec<usize> = touched_set.iter().copied().collect();
    let tj = hygiene::fix_tjunctions(m, &touched_v);
    let (synced, nb_mod, failures) = hygiene::sync_edges(m, &touched_v);
    let clean_v: Vec<usize> = around.iter().chain(nb_mod.iter()).copied().collect();
    let degenerate = hygiene::drop_degenerate(m, &clean_v);
    let report = HygieneReport { merged, dropped: dropped + degenerate, tjunctions: tj, synced, failures };

    let new_view = View::build(m);
    let old_view = View::build(old);
    let mut seed = touched_set.clone();
    seed.extend(nb_mod.iter().copied());
    let nbr = neighbours(m, &seed);
    for &idx in &nbr {
        let was_run = touched.get(&idx) == Some(&true) && m.cells[idx].pieces.is_empty();
        let cell = &mut m.cells[idx];
        if was_run {
            let w = new_view.word(idx, 0);
            let pid = cell.run.unwrap().1;
            cell.run = Some((w, pid));
            continue;
        }
        if cell.pieces.is_empty() {
            let pn = new_view.word(idx, 0);
            let po = old_view.word(idx, 0);
            if pn != po {
                let pid = cell.run.unwrap().1;
                cell.run = Some((pn, pid));
            }
            continue;
        }
        let oc = &old.cells[idx];
        for i in 0..cell.pieces.len() {
            let pn = new_view.word(idx, i);
            match cell.pieces[i].src {
                Some(k) if !oc.pieces.is_empty() && k < oc.pieces.len() => {
                    let po = old_view.word(idx, k);
                    cell.pieces[i].a = if po != pn { pn } else { oc.pieces[k].a };
                }
                _ => cell.pieces[i].a = pn,
            }
        }
    }
    (report, nb_mod)
}

/// Final group numbering: the groups of `groups` (path id -> member region ids, ascending region-list position) that
/// some cell uses are listed after the sea path in raster order of their first cell (then piece index); pieces, runs
/// and `grid.paths` are rewritten. `sea_idx` is the sea path id.
///
/// A group path is identified by its member *set*: the validator rejects two paths with the same members, yet
/// several input ids can name one set (e.g. one temporary band id per area pair when two areas share the same two
/// regions, or a temporary id next to an existing path). All ids of one set are merged into a single new path whose
/// number comes from the earliest raster use of any of them, and every one of those ids is mapped to it.
///
/// Returns the old -> new id map (it holds every input id whose member set is used by some cell, even an id no cell
/// references itself) and the number of input ids not in that map, i.e. dropped because their set is unused.
pub fn renumber_groups(m: &mut Model, grid: &mut Grid, sea_idx: usize, groups: &BTreeMap<u16, Vec<i16>>) -> Result<(HashMap<u16, u16>, usize)> {
    // Every caller orders members by region-list position, so equal sets already have equal slices.
    // Borrow those keys once; `rep[s]` is the lowest input id of set `s`. No allocation per cell.
    let mut sets: BTreeMap<&[i16], usize> = BTreeMap::new();
    let mut set_of: HashMap<u16, usize> = HashMap::with_capacity(groups.len());
    let mut rep: Vec<u16> = Vec::new();
    for (pid, key) in groups {
        let next = sets.len();
        let s = *sets.entry(key.as_slice()).or_insert(next);
        if s == rep.len() {
            rep.push(*pid);
        }
        set_of.insert(*pid, s);
    }
    // earliest raster use (cell, piece) of each set
    let mut first: Vec<Option<(usize, usize)>> = vec![None; rep.len()];
    for (idx, cell) in m.cells.iter().enumerate() {
        if cell.pieces.is_empty() {
            if let Some(&s) = cell.run.and_then(|(_, pid)| set_of.get(&pid)) {
                first[s].get_or_insert((idx, 0));
            }
        } else {
            for (k, p) in cell.pieces.iter().enumerate() {
                if let Some(&s) = set_of.get(&p.pid) {
                    first[s].get_or_insert((idx, k));
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..rep.len()).filter(|&s| first[s].is_some()).collect();
    order.sort_by_key(|&s| first[s]);
    ensure!(sea_idx + 1 + order.len() < 1023, "too many group paths");
    let mut number: Vec<Option<u16>> = vec![None; rep.len()];
    for (k, &s) in order.iter().enumerate() {
        number[s] = Some((sea_idx + 1 + k) as u16);
    }
    let newid: HashMap<u16, u16> = set_of.iter().filter_map(|(&pid, &s)| number[s].map(|n| (pid, n))).collect();
    for cell in m.cells.iter_mut() {
        for p in cell.pieces.iter_mut() {
            if let Some(&n) = newid.get(&p.pid) {
                p.pid = n;
            }
        }
        if cell.pieces.is_empty() {
            if let Some(r) = cell.run.as_mut() {
                if let Some(&n) = newid.get(&r.1) {
                    r.1 = n;
                }
            }
        }
    }
    grid.paths.truncate(sea_idx + 1);
    for &s in &order {
        grid.paths.push(groups[&rep[s]].clone());
    }
    let dropped = groups.len() - newid.len();
    Ok((newid, dropped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regionmap::pathfinding::{Cell, Grid};
    use crate::regionmap::pathsynth::model::{MCell, MPiece};

    fn piece(pid: u16) -> MPiece {
        MPiece { a: 0, pid, gbit: 0, pts: vec![[0, 0], [1, 0], [1, 1]], src: None }
    }

    fn model(cells: Vec<MCell>) -> Model {
        Model { cols: cells.len(), rows: 1, cs: 1 << 20, origin: [0, 0], cells }
    }

    fn grid(paths: Vec<Vec<i16>>) -> Grid {
        Grid {
            area: 0,
            x0: 0.0,
            y0: 0.0,
            size: 1.0,
            cols: 0,
            rows: 0,
            origin_index: [0, 0],
            region_ids: Vec::new(),
            paths,
            cells: Vec::<Cell>::new(),
            vertices: Vec::new(),
            polylines: BTreeMap::new(),
        }
    }

    /// Several temporary ids naming one member set (one band per area pair) must collapse into one path numbered by
    /// the earliest raster use of any of them; run cells and pieces of every id are remapped, unused ids dropped.
    #[test]
    fn ids_with_equal_members_share_one_path() {
        let sea = 4usize;
        let (a1, a2, a3, other, unused, twin_unused) = (900u16, 901u16, 902u16, 903u16, 904u16, 905u16);
        let hdr = [0u8; 8];
        let mut m = model(vec![
            // raster order: other (cell 0), a2 (cell 1 piece 0, then a3), a1 (cell 2 run), a1/a3 again
            MCell { hdr, pieces: vec![piece(other)], run: None },
            MCell { hdr, pieces: vec![piece(a2), piece(a3)], run: None },
            MCell { hdr, pieces: Vec::new(), run: Some((0, a1)) },
            MCell { hdr, pieces: vec![piece(a3), piece(a1)], run: None },
        ]);
        let mut groups: BTreeMap<u16, Vec<i16>> = BTreeMap::new();
        for pid in [a1, a2, a3] {
            groups.insert(pid, vec![34, 205]);
        }
        groups.insert(other, vec![7, 34]);
        groups.insert(unused, vec![8, 9]);
        // same set as `unused`, still unreferenced: dropped together with it
        groups.insert(twin_unused, vec![8, 9]);
        let mut g = grid(vec![Vec::new(); sea + 3]);

        let (newid, dropped) = renumber_groups(&mut m, &mut g, sea, &groups).unwrap();

        // `other` is used first (cell 0), the shared set second
        let (n_other, n_shared) = (sea as u16 + 1, sea as u16 + 2);
        assert_eq!(newid.len(), 4);
        assert_eq!(newid[&other], n_other);
        for pid in [a1, a2, a3] {
            assert_eq!(newid[&pid], n_shared, "id {pid}");
        }
        assert_eq!(dropped, 2);
        assert_eq!(g.paths.len(), sea + 3);
        assert_eq!(g.paths[sea + 1], vec![7, 34]);
        assert_eq!(g.paths[sea + 2], vec![34, 205]);
        let pids: Vec<Vec<u16>> = m.cells.iter().map(|c| c.pieces.iter().map(|p| p.pid).collect()).collect();
        assert_eq!(pids, vec![vec![n_other], vec![n_shared, n_shared], Vec::new(), vec![n_shared, n_shared]]);
        assert_eq!(m.cells[2].run, Some((0, n_shared)));
    }

    /// An id named in `groups` but used only through an equal-set sibling is still mapped, so callers looking up a
    /// band id find the shared path; a set nobody uses maps nothing.
    #[test]
    fn unreferenced_id_follows_used_sibling() {
        let sea = 2usize;
        let hdr = [0u8; 8];
        let mut m = model(vec![MCell { hdr, pieces: vec![piece(500)], run: None }]);
        let groups: BTreeMap<u16, Vec<i16>> = [(500u16, vec![1i16, 2]), (501, vec![1, 2]), (502, vec![3, 4])].into_iter().collect();
        let mut g = grid(vec![Vec::new(); sea + 1]);
        let (newid, dropped) = renumber_groups(&mut m, &mut g, sea, &groups).unwrap();
        assert_eq!(newid[&500], sea as u16 + 1);
        assert_eq!(newid[&501], sea as u16 + 1);
        assert!(!newid.contains_key(&502));
        assert_eq!(dropped, 1);
        assert_eq!(g.paths, vec![Vec::new(), Vec::new(), Vec::new(), vec![1, 2]]);
    }
}
