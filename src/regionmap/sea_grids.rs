//! `sea_grids.esf`: the campaign AI's coarse grid over the map. The file is one flat `CAI_SEA_GRID_ROOT` record:
//! `u32 theatre count`, then per theatre `s16 name, xy min, xy max, f32 factor, u32 xsize, u32 ysize`,
//! `xsize*ysize * (u32 xi, u32 yi, xy lo, xy hi)` (row-major, `yi` outer) with the cell rectangles, then for every
//! cell in the same order an area `u32 id, u32 n + n*s16 land keys, u32 n + n*s16 sea keys, u32 n + n*s16 port keys,
//! u32 n + n*u32 numbers`, and finally `u32 n + n*(u32, u32, f32)` connections.
//!
//! Vanilla rule (reproduced exactly for all 66 cells of `nap_europe`): a cell that touches any sea region lists, in
//! regions.esf index order, every non-sea region (including `eur_map_west`/`eur_map_east`/`all`) whose outline
//! overlaps the cell rectangle in `lands`, and every sea region in `seas`; cells touching no sea have all four lists
//! empty. `ports` are port slot keys and `numbers` are ids of an unrelated table (not poi.esf indices); neither
//! depends on region polygons and both are left untouched.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};

use crate::esf::{self, Node, Value};
use crate::regionmap::geometry::{P, Rect, point_in_ring, segment_overlaps_rect};
use crate::regionmap::regions::RegionData;
use crate::regionmap::trade_routes::TradeSplines;

#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub xi: u32,
    pub yi: u32,
    pub lo: [f32; 2],
    pub hi: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Area {
    pub id: u32,
    pub lands: Vec<String>,
    pub seas: Vec<String>,
    pub ports: Vec<String>,
    pub numbers: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TheatreGrid {
    pub name: String,
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub factor: f32,
    pub xsize: u32,
    pub ysize: u32,
    pub cells: Vec<Cell>,
    pub areas: Vec<Area>,
    pub connections: Vec<(u32, u32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SeaGrids {
    pub grids: Vec<TheatreGrid>,
}

struct Cursor<'a> {
    vals: &'a [Node],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn next(&mut self) -> Result<&'a Value> {
        let n = self.vals.get(self.at).context("sea_grids ended early")?;
        self.at += 1;
        match n {
            Node::Value(v) => Ok(v),
            other => bail!("unexpected structured node {:?} in sea_grids", other.name()),
        }
    }
    fn u32(&mut self) -> Result<u32> {
        match self.next()? {
            Value::U32(v) => Ok(*v),
            other => bail!("expected u32, found {other:?}"),
        }
    }
    fn f32(&mut self) -> Result<f32> {
        match self.next()? {
            Value::F32(v) => Ok(*v),
            other => bail!("expected f32, found {other:?}"),
        }
    }
    fn xy(&mut self) -> Result<[f32; 2]> {
        match self.next()? {
            Value::XY(v) => Ok(*v),
            other => bail!("expected xy, found {other:?}"),
        }
    }
    fn s(&mut self) -> Result<String> {
        match self.next()? {
            Value::String16(v) => Ok(v.clone()),
            other => bail!("expected string, found {other:?}"),
        }
    }
    fn strings(&mut self) -> Result<Vec<String>> {
        let n = self.u32()?;
        (0..n).map(|_| self.s()).collect()
    }
    fn numbers(&mut self) -> Result<Vec<u32>> {
        let n = self.u32()?;
        (0..n).map(|_| self.u32()).collect()
    }
}

impl SeaGrids {
    pub fn parse(root: &Node) -> Result<Self> {
        ensure!(root.name() == Some("CAI_SEA_GRID_ROOT"), "not a sea_grids root");
        let mut c = Cursor { vals: root.children()?, at: 0 };
        let count = c.u32()?;
        let mut grids = Vec::new();
        for _ in 0..count {
            let name = c.s()?;
            let min = c.xy()?;
            let max = c.xy()?;
            let factor = c.f32()?;
            let xsize = c.u32()?;
            let ysize = c.u32()?;
            let mut cells = Vec::new();
            for yi in 0..ysize {
                for xi in 0..xsize {
                    let (cx, cy) = (c.u32()?, c.u32()?);
                    ensure!((cx, cy) == (xi, yi), "cell ({cx},{cy}) out of order, expected ({xi},{yi})");
                    cells.push(Cell { xi, yi, lo: c.xy()?, hi: c.xy()? });
                }
            }
            let mut areas = Vec::new();
            for _ in 0..cells.len() {
                areas.push(Area {
                    id: c.u32()?,
                    lands: c.strings()?,
                    seas: c.strings()?,
                    ports: c.strings()?,
                    numbers: c.numbers()?,
                });
            }
            let n = c.u32()?;
            let mut connections = Vec::new();
            for _ in 0..n {
                connections.push((c.u32()?, c.u32()?, c.f32()?));
            }
            grids.push(TheatreGrid { name, min, max, factor, xsize, ysize, cells, areas, connections });
        }
        ensure!(c.at == c.vals.len(), "{} trailing nodes after the last theatre grid", c.vals.len() - c.at);
        Ok(Self { grids })
    }

    pub fn store(&self, root: &mut Node) -> Result<()> {
        let mut out: Vec<Node> = vec![esf::u32v(self.grids.len() as u32)];
        let strings = |out: &mut Vec<Node>, list: &[String]| {
            out.push(esf::u32v(list.len() as u32));
            out.extend(list.iter().map(|s| esf::s16(s.clone())));
        };
        for g in &self.grids {
            out.push(esf::s16(g.name.clone()));
            out.push(esf::xyv(g.min[0], g.min[1]));
            out.push(esf::xyv(g.max[0], g.max[1]));
            out.push(esf::f32v(g.factor));
            out.push(esf::u32v(g.xsize));
            out.push(esf::u32v(g.ysize));
            for c in &g.cells {
                out.push(esf::u32v(c.xi));
                out.push(esf::u32v(c.yi));
                out.push(esf::xyv(c.lo[0], c.lo[1]));
                out.push(esf::xyv(c.hi[0], c.hi[1]));
            }
            for a in &g.areas {
                out.push(esf::u32v(a.id));
                strings(&mut out, &a.lands);
                strings(&mut out, &a.seas);
                strings(&mut out, &a.ports);
                out.push(esf::u32v(a.numbers.len() as u32));
                out.extend(a.numbers.iter().map(|&n| esf::u32v(n)));
            }
            out.push(esf::u32v(g.connections.len() as u32));
            for &(a, b, f) in &g.connections {
                out.push(esf::u32v(a));
                out.push(esf::u32v(b));
                out.push(esf::f32v(f));
            }
        }
        *root.children_mut()? = out;
        Ok(())
    }

    pub fn grid(&self, theatre: &str) -> Option<&TheatreGrid> {
        self.grids.iter().find(|g| g.name == theatre)
    }

    fn grid_index(&self, theatre: &str) -> Result<usize> {
        self.grids.iter().position(|g| g.name == theatre).with_context(|| format!("sea_grids has no theatre {theatre}"))
    }

    /// Every port slot key listed in some cell of any theatre (the "vanilla listed" set when called on the
    /// snapshot's untouched sea_grids.esf).
    pub fn listed_ports(&self) -> BTreeSet<String> {
        self.grids.iter().flat_map(|g| g.areas.iter()).flat_map(|a| a.ports.iter().cloned()).collect()
    }

    /// Move `theatre`'s grid onto `new_rect` keeping the cell count and the cell ids (the start position carries one
    /// `CAI_SEA_GRID_CELL` per cell): the cell size becomes `factor = ceil(max(w / nx, h / ny))` and the boxes are laid
    /// out from the new minimum corner. Region/port/number lists and connections are left as they are; call
    /// [`SeaGrids::regenerate_lists`] and [`SeaGrids::complete_conns`] afterwards.
    pub fn resize_theatre(&mut self, theatre: &str, new_rect: Rect) -> Result<()> {
        let gi = self.grid_index(theatre)?;
        let g = &mut self.grids[gi];
        ensure!(g.xsize > 0 && g.ysize > 0, "theatre {theatre} has an empty sea grid");
        let [lo, hi] = new_rect;
        let (w, h) = ((hi[0] - lo[0]) as f64, (hi[1] - lo[1]) as f64);
        ensure!(w > 0.0 && h > 0.0, "theatre {theatre}: empty rectangle {new_rect:?}");
        let factor = (w / g.xsize as f64).max(h / g.ysize as f64).ceil();
        g.min = lo;
        g.max = hi;
        g.factor = factor as f32;
        for c in &mut g.cells {
            let blo = [lo[0] as f64 + c.xi as f64 * factor, lo[1] as f64 + c.yi as f64 * factor];
            c.lo = [blo[0] as f32, blo[1] as f32];
            c.hi = [(blo[0] + factor) as f32, (blo[1] + factor) as f32];
        }
        Ok(())
    }

    /// Recompute the lists of every cell of `theatre` from the current `rd` and trade splines. Cells without a sea
    /// region get all four lists empty; others get
    /// * `seas`/`lands`: regions (regions.esf order) with an outer ring that contains the box centre or whose edge
    ///   touches or crosses the box (kind `sea` -> seas, every other kind -> lands),
    /// * `ports`: port slots (`settlement_and_slots` slot type `port`) of the `vanilla_ports` set whose dock point lies
    ///   in the box (`lo <= p < hi`), sorted by key; a port the cell lists now that no longer lies in any box of the
    ///   grid stays listed in the cell nearest to its dock point (vanilla's India grid lists bandar_abbas that way),
    /// * `numbers`: `trade_routes.esf` spline indices whose Bezier chain overlaps the box interior.
    /// Returns the number of cells whose lists changed.
    pub fn regenerate_lists(
        &mut self,
        theatre: &str,
        rd: &RegionData,
        splines: &TradeSplines,
        vanilla_ports: &BTreeSet<String>,
    ) -> Result<usize> {
        let gi = self.grid_index(theatre)?;
        let rings = RegionRings::new(rd);
        let docks = port_docks(rd)?;
        let g = &mut self.grids[gi];
        let mut lists: Vec<Area> = Vec::with_capacity(g.cells.len());
        for (c, old) in g.cells.iter().zip(&g.areas) {
            let (lo, hi) = ([c.lo[0] as f64, c.lo[1] as f64], [c.hi[0] as f64, c.hi[1] as f64]);
            let (seas, lands) = rings.overlapping(rd, lo, hi);
            let mut a = Area { id: old.id, lands: Vec::new(), seas: Vec::new(), ports: Vec::new(), numbers: Vec::new() };
            if !seas.is_empty() {
                a.lands = lands.iter().map(|&i| rd.regions[i].key.clone()).collect();
                a.seas = seas.iter().map(|&i| rd.regions[i].key.clone()).collect();
                a.ports = docks
                    .iter()
                    .filter(|(k, d)| vanilla_ports.contains(*k) && in_box(**d, lo, hi))
                    .map(|(k, _)| k.clone())
                    .collect();
                a.numbers = splines.splines_in(lo, hi);
            }
            lists.push(a);
        }
        // listed ports that fell off the grid: nearest cell to the dock point
        let boxes: Vec<(P, P)> = g.cells.iter().map(|c| ([c.lo[0] as f64, c.lo[1] as f64], [c.hi[0] as f64, c.hi[1] as f64])).collect();
        if !boxes.is_empty() {
            let min = [boxes.iter().map(|b| b.0[0]).fold(f64::MAX, f64::min), boxes.iter().map(|b| b.0[1]).fold(f64::MAX, f64::min)];
            let max = [boxes.iter().map(|b| b.1[0]).fold(f64::MIN, f64::max), boxes.iter().map(|b| b.1[1]).fold(f64::MIN, f64::max)];
            let previously: BTreeSet<&String> = g.areas.iter().flat_map(|a| a.ports.iter()).collect();
            for k in previously {
                let (Some(&q), true) = (docks.get(k), vanilla_ports.contains(k)) else { continue };
                if boxes.iter().any(|b| in_box(q, b.0, b.1)) {
                    continue;
                }
                let c = [q[0].clamp(min[0], max[0] - 1e-3), q[1].clamp(min[1], max[1] - 1e-3)];
                for (i, b) in boxes.iter().enumerate() {
                    if in_box(c, b.0, b.1) && !lists[i].seas.is_empty() {
                        lists[i].ports.push(k.clone());
                        lists[i].ports.sort();
                    }
                }
            }
        }
        let mut changed = 0;
        for (old, new) in g.areas.iter_mut().zip(lists) {
            if *old != new {
                *old = new;
                changed += 1;
            }
        }
        Ok(changed)
    }

    /// Complete the connection graph of `theatre` over its sea-bearing cells: existing values are kept, every missing
    /// pair `(id_a < id_b)` is added (8-adjacent cells 0.0, others the distance between the cell centres) and the list
    /// is sorted. Returns the number of connections added.
    pub fn complete_conns(&mut self, theatre: &str) -> Result<usize> {
        let gi = self.grid_index(theatre)?;
        let g = &mut self.grids[gi];
        let centres: BTreeMap<u32, [f64; 2]> = g
            .cells
            .iter()
            .zip(&g.areas)
            .map(|(c, a)| (a.id, [(c.lo[0] as f64 + c.hi[0] as f64) / 2.0, (c.lo[1] as f64 + c.hi[1] as f64) / 2.0]))
            .collect();
        let sea: Vec<u32> = g.areas.iter().filter(|a| !a.seas.is_empty()).map(|a| a.id).collect();
        let have: BTreeSet<(u32, u32)> = g.connections.iter().map(|&(a, b, _)| (a, b)).collect();
        let factor = g.factor as f64;
        let mut added = 0;
        for &ia in &sea {
            for &ib in &sea {
                if ia < ib && !have.contains(&(ia, ib)) {
                    let (a, b) = (centres[&ia], centres[&ib]);
                    let adjacent = (a[0] - b[0]).abs() <= factor + 1e-3 && (a[1] - b[1]).abs() <= factor + 1e-3;
                    let d = if adjacent { 0.0 } else { (a[0] - b[0]).hypot(a[1] - b[1]) };
                    g.connections.push((ia, ib, d as f32));
                    added += 1;
                }
            }
        }
        g.connections.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)).then(x.2.total_cmp(&y.2)));
        Ok(added)
    }
}

/// `lo <= p < hi` on both axes (the cell membership rule of port dock points).
fn in_box(p: P, lo: P, hi: P) -> bool {
    lo[0] <= p[0] && p[0] < hi[0] && lo[1] <= p[1] && p[1] < hi[1]
}

/// Dock point (second xy of the slot entry) of every port slot of `rd`, keyed by slot key.
fn port_docks(rd: &RegionData) -> Result<BTreeMap<String, P>> {
    let mut out = BTreeMap::new();
    for r in &rd.regions {
        let Some(s) = &r.settlement else { continue };
        let ch = s.children()?;
        let Some(slots) = ch.get(2) else { continue };
        for item in slots.items()? {
            if item.len() > 3 && item[1].as_str().ok() == Some("port") {
                let d = item[3].as_xy().with_context(|| format!("dock point of {}", r.key))?;
                out.insert(item[0].as_str()?.to_string(), [d[0] as f64, d[1] as f64]);
            }
        }
    }
    Ok(out)
}

/// The segment `p`-`q` meets the closed rectangle over a stretch of positive parameter length (running along an
/// edge counts, a single touch point does not).
fn segment_touches_rect(p: P, q: P, lo: P, hi: P) -> bool {
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (den, num) in [(-dx, p[0] - lo[0]), (dx, hi[0] - p[0]), (-dy, p[1] - lo[1]), (dy, hi[1] - p[1])] {
        if den == 0.0 {
            if num < 0.0 {
                return false;
            }
        } else {
            let t = num / den;
            if den < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    t0 < t1
}

/// Outer rings of every region with their bounding boxes, for repeated box queries.
struct RegionRings {
    /// (region index, ring, (min, max))
    rings: Vec<(usize, Vec<P>, (P, P))>,
}

impl RegionRings {
    fn new(rd: &RegionData) -> Self {
        let mut rings = Vec::new();
        for (ri, r) in rd.regions.iter().enumerate() {
            for a in &r.areas {
                for o in a.outlines.iter().filter(|o| o.outer) {
                    let ring: Vec<P> = o.ring.iter().map(|&i| rd.v(i)).collect();
                    if ring.is_empty() {
                        continue;
                    }
                    let mut bb = ([f64::MAX; 2], [f64::MIN; 2]);
                    for p in &ring {
                        for k in 0..2 {
                            bb.0[k] = bb.0[k].min(p[k]);
                            bb.1[k] = bb.1[k].max(p[k]);
                        }
                    }
                    rings.push((ri, ring, bb));
                }
            }
        }
        Self { rings }
    }

    /// `(sea regions, other regions)` (ascending indices) with an outer ring that contains the box centre or has an
    /// edge touching or crossing the box.
    fn overlapping(&self, rd: &RegionData, lo: P, hi: P) -> (Vec<usize>, Vec<usize>) {
        let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
        let (mut sea, mut land) = (BTreeSet::new(), BTreeSet::new());
        for (ri, ring, bb) in &self.rings {
            let target = if rd.regions[*ri].kind == "sea" { &mut sea } else { &mut land };
            if target.contains(ri) || bb.1[0] < lo[0] || bb.0[0] > hi[0] || bb.1[1] < lo[1] || bb.0[1] > hi[1] {
                continue;
            }
            let n = ring.len();
            if point_in_ring(centre, ring) || (0..n).any(|i| segment_touches_rect(ring[i], ring[(i + 1) % n], lo, hi)) {
                target.insert(*ri);
            }
        }
        (sea.into_iter().collect(), land.into_iter().collect())
    }
}

/// Indices (ascending, = regions.esf order) of the regions of `rd` whose outline overlaps the rectangle `lo`..`hi`
/// and that are (`sea == true`) or are not (`sea == false`) of kind `sea`. A region counts when a boundary segment
/// of one of its outer outlines crosses the rectangle interior or the rectangle centre lies inside the polygon.
pub fn covering_regions(rd: &RegionData, lo: [f32; 2], hi: [f32; 2], sea: bool) -> Vec<usize> {
    let (lo, hi) = ([lo[0] as f64, lo[1] as f64], [hi[0] as f64, hi[1] as f64]);
    (0..rd.regions.len())
        .filter(|&ri| (rd.regions[ri].kind == "sea") == sea && rect_touches_region(rd, ri, lo, hi))
        .collect()
}

fn rect_touches_region(rd: &RegionData, ri: usize, lo: P, hi: P) -> bool {
    let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    for a in &rd.regions[ri].areas {
        for o in a.outlines.iter().filter(|o| o.outer) {
            let ring: Vec<P> = o.ring.iter().map(|&i| rd.v(i)).collect();
            if ring.is_empty() {
                continue;
            }
            for i in 0..ring.len() {
                if segment_overlaps_rect(ring[i], ring[(i + 1) % ring.len()], lo, hi) {
                    return true;
                }
            }
            if point_in_ring(centre, &ring) {
                return true;
            }
        }
    }
    false
}

/// Recompute the land lists of `root` against `rd` after region `new_index` was added (and its parent shrunk):
/// every cell that lists any region (i.e. every cell that touches the sea) gets exactly the non-sea regions
/// overlapping its rectangle, in regions.esf order (this reproduces all 39 populated cells of vanilla). Cells without
/// any sea stay empty. Returns the number of cells whose land list changed.
pub fn add_region(root: &mut Node, rd: &RegionData, new_index: usize) -> Result<usize> {
    add_region_within(root, rd, new_index, None)
}

/// [`add_region`] restricted to the cells that overlap one of `rects` (`(lo, hi)` world rectangles, the area the
/// split changed); every other cell is left as it is.
pub fn add_region_within(
    root: &mut Node,
    rd: &RegionData,
    new_index: usize,
    rects: Option<&[([f64; 2], [f64; 2])]>,
) -> Result<usize> {
    ensure!(new_index < rd.regions.len(), "region index {new_index} out of range");
    let mut sg = SeaGrids::parse(root)?;
    let mut changed = 0;
    for g in &mut sg.grids {
        for (cell, area) in g.cells.iter().zip(g.areas.iter_mut()) {
            if area.seas.is_empty() && area.lands.is_empty() {
                continue;
            }
            if let Some(rects) = rects {
                let (clo, chi) = ([cell.lo[0] as f64, cell.lo[1] as f64], [cell.hi[0] as f64, cell.hi[1] as f64]);
                if !rects.iter().any(|(lo, hi)| clo[0] <= hi[0] && chi[0] >= lo[0] && clo[1] <= hi[1] && chi[1] >= lo[1]) {
                    continue;
                }
            }
            let lands: Vec<String> = covering_regions(rd, cell.lo, cell.hi, false)
                .into_iter()
                .map(|i| rd.regions[i].key.clone())
                .collect();
            if lands != area.lands {
                area.lands = lands;
                changed += 1;
            }
        }
    }
    sg.store(root)?;
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const WORK: &str = "F:/NTW2/work";

    fn read(path: &str) -> Option<esf::Esf> {
        if !Path::new(path).exists() {
            eprintln!("skipped: {path} not available");
            return None;
        }
        Some(esf::read(&std::fs::read(path).unwrap()).unwrap())
    }

    fn same_set<T: Ord + Clone>(a: &[T], b: &[T]) -> bool {
        a.iter().collect::<BTreeSet<_>>() == b.iter().collect::<BTreeSet<_>>()
    }

    #[test]
    fn regenerate_on_vanilla_matches_stored_lists() {
        let (Some(sg), Some(rg), Some(tr)) = (
            read(&format!("{WORK}/exp1/backup/campaign_maps/nap_world/sea_grids.esf")),
            read(&format!("{WORK}/exp1/backup/campaign_maps/nap_world/regions.esf")),
            read(&format!("{WORK}/exp4/backup/data/campaign_maps/nap_world/trade_routes.esf")),
        ) else {
            return;
        };
        let stored = SeaGrids::parse(&sg.root).unwrap();
        let rd = RegionData::parse(&rg.root).unwrap();
        let splines = TradeSplines::parse(&tr.root).unwrap();
        let listed = stored.listed_ports();
        let mut regenerated = stored.clone();
        let (mut cells, mut ok) = (0, 0);
        let mut per_list = [0usize; 4];
        let mut misses = Vec::new();
        for gi in 0..stored.grids.len() {
            let name = stored.grids[gi].name.clone();
            regenerated.regenerate_lists(&name, &rd, &splines, &listed).unwrap();
            for (ci, (a, b)) in stored.grids[gi].areas.iter().zip(&regenerated.grids[gi].areas).enumerate() {
                cells += 1;
                let each = [
                    same_set(&a.lands, &b.lands),
                    same_set(&a.seas, &b.seas),
                    same_set(&a.ports, &b.ports),
                    same_set(&a.numbers, &b.numbers),
                ];
                for (n, e) in per_list.iter_mut().zip(each) {
                    *n += e as usize;
                }
                if each.iter().all(|&e| e) {
                    ok += 1;
                } else {
                    misses.push(format!(
                        "{name}#{ci}: lands {}/{} seas {}/{} ports {}/{} nums {}/{}",
                        a.lands.len(), b.lands.len(), a.seas.len(), b.seas.len(), a.ports.len(), b.ports.len(),
                        a.numbers.len(), b.numbers.len()
                    ));
                }
            }
        }
        eprintln!("vanilla: {ok}/{cells} cells identical, per list (lands, seas, ports, numbers) {per_list:?}; misses:\n  {}", misses.join("\n  "));
        assert_eq!(cells, 128);
        // The reference script (exp4/seagrid_step.py --vanilla) measures exactly these rates: 95 cells with all four
        // lists equal, lands 111, seas 108, ports 127, nums 119 of 128. The misses are cells at the edge of the
        // theatres whose stored lists reach outside their box (not derivable from a rule) plus 28 splines.
        assert!(ok >= 95, "{ok}/{cells}");
        assert!(per_list[0] >= 111 && per_list[1] >= 108 && per_list[2] >= 127 && per_list[3] >= 119, "{per_list:?}");
    }

    #[test]
    fn variant_b_lists_are_reproduced_exactly() {
        let dir = format!("{WORK}/exp5/variantB/out/data/campaign_maps/nap_world");
        let (Some(van), Some(golden), Some(rg), Some(tr)) = (
            read(&format!("{WORK}/exp1/backup/campaign_maps/nap_world/sea_grids.esf")),
            read(&format!("{dir}/sea_grids.esf")),
            read(&format!("{dir}/regions.esf")),
            read(&format!("{dir}/trade_routes.esf")),
        ) else {
            return;
        };
        let vanilla = SeaGrids::parse(&van.root).unwrap();
        let golden = SeaGrids::parse(&golden.root).unwrap();
        let rd = RegionData::parse(&rg.root).unwrap();
        let splines = TradeSplines::parse(&tr.root).unwrap();
        let mut sg = vanilla.clone();
        sg.resize_theatre("3", [[440.0, -80.0], [790.0, 300.0]]).unwrap();
        sg.regenerate_lists("3", &rd, &splines, &vanilla.listed_ports()).unwrap();
        let before = sg.grid("3").unwrap().connections.len();
        sg.complete_conns("3").unwrap();
        let (got, want) = (sg.grid("3").unwrap(), golden.grid("3").unwrap());
        assert_eq!((got.factor, got.xsize, got.ysize, got.min, got.max), (want.factor, want.xsize, want.ysize, want.min, want.max));
        assert_eq!(got.cells, want.cells);
        for (i, (a, b)) in got.areas.iter().zip(&want.areas).enumerate() {
            assert_eq!(a, b, "cell {i}");
        }
        eprintln!("conns {before} -> {}", got.connections.len());
        assert_eq!(got.connections.len(), 66);
        assert_eq!(got.connections, want.connections);
        // idempotent
        let again = sg.clone();
        sg.complete_conns("3").unwrap();
        assert_eq!(sg, again);
        // other theatres are untouched
        for (a, b) in vanilla.grids.iter().zip(&sg.grids) {
            if a.name != "3" {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn resize_keeps_cell_count_and_ids() {
        let Some(van) = read(&format!("{WORK}/exp1/backup/campaign_maps/nap_world/sea_grids.esf")) else { return };
        let mut sg = SeaGrids::parse(&van.root).unwrap();
        let ids: Vec<u32> = sg.grid("3").unwrap().areas.iter().map(|a| a.id).collect();
        sg.resize_theatre("3", [[440.0, -80.0], [790.0, 300.0]]).unwrap();
        let g = sg.grid("3").unwrap();
        assert_eq!((g.xsize, g.ysize, g.factor), (4, 4, 95.0));
        assert_eq!(g.areas.iter().map(|a| a.id).collect::<Vec<_>>(), ids);
        assert_eq!(g.cells[0].lo, [440.0, -80.0]);
        assert_eq!(g.cells[15].hi, [820.0, 300.0]);
        assert!(sg.resize_theatre("nope", [[0.0, 0.0], [1.0, 1.0]]).is_err());
    }
}
