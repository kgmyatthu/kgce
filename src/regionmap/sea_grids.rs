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

    /// Total number of cells of all theatre grids (the length of the start position's `CAI_WORLD_SEA_GRID_CELLS`).
    pub fn cell_count(&self) -> usize {
        self.grids.iter().map(|g| g.cells.len()).sum()
    }

    /// Index, among all cells of all theatres in file order, of the cell the campaign AI files a fleet at `(x, y)` in:
    /// the first theatre whose rectangle `[min, max)` holds the point, then the cell column/row `floor((p - min) /
    /// factor)` clamped to the grid. This is the rule that holds for every fleet of the stock start positions (the
    /// cell lists of `CAI_WORLD_SEA_GRID_CELLS` and the cell id of each fleet's `CAI_MOBILE`), so the start position has
    /// to be built against the grid of the *finished* map: [`SeaGrids::resize_theatre`] moves the cell boxes.
    pub fn cell_index_at(&self, x: f64, y: f64) -> Option<usize> {
        let mut offset = 0;
        for g in &self.grids {
            let (x0, y0, x1, y1) = (g.min[0] as f64, g.min[1] as f64, g.max[0] as f64, g.max[1] as f64);
            if x0 <= x && x < x1 && y0 <= y && y < y1 && g.xsize > 0 && g.ysize > 0 {
                let col = (((x - x0) / g.factor as f64) as usize).min(g.xsize as usize - 1);
                let row = (((y - y0) / g.factor as f64) as usize).min(g.ysize as usize - 1);
                return Some(offset + row * g.xsize as usize + col);
            }
            offset += g.cells.len();
        }
        None
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

    /// Make the connection graph of `theatre` the clique of its sea-bearing cells, which is the shape of every stock
    /// theatre (one `(id_a < id_b)` pair per two sea cells, none for a cell without sea): existing values of pairs
    /// between sea cells are kept, every missing pair is added (8-adjacent cells 0.0, others the distance between the
    /// cell centres), connections that name a cell without any sea are removed (a cell that lost its sea when the
    /// grid moved keeps no stale edges) and the list is sorted. Returns `(added, removed)`.
    pub fn complete_conns(&mut self, theatre: &str) -> Result<(usize, usize)> {
        let gi = self.grid_index(theatre)?;
        let g = &mut self.grids[gi];
        let centres: BTreeMap<u32, [f64; 2]> = g
            .cells
            .iter()
            .zip(&g.areas)
            .map(|(c, a)| (a.id, [(c.lo[0] as f64 + c.hi[0] as f64) / 2.0, (c.lo[1] as f64 + c.hi[1] as f64) / 2.0]))
            .collect();
        let sea: BTreeSet<u32> = g.areas.iter().filter(|a| !a.seas.is_empty()).map(|a| a.id).collect();
        let before = g.connections.len();
        g.connections.retain(|&(a, b, _)| sea.contains(&a) && sea.contains(&b));
        let removed = before - g.connections.len();
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
        Ok((added, removed))
    }

    /// Check that every pair of sea-bearing cells has exactly one ordered connection, with no edges into non-sea cells.
    pub fn graph_problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        for g in &self.grids {
            let sea: BTreeSet<u32> = g.areas.iter().filter(|a| !a.seas.is_empty()).map(|a| a.id).collect();
            let mut seen: BTreeSet<(u32, u32)> = BTreeSet::new();
            for &(a, b, _) in &g.connections {
                if !seen.insert((a, b)) {
                    out.push(format!("theatre {}: connection {a}-{b} is listed twice", g.name));
                }
                if a >= b {
                    out.push(format!("theatre {}: connection {a}-{b} is not ordered", g.name));
                }
                for id in [a, b] {
                    if !sea.contains(&id) {
                        out.push(format!("theatre {}: connection {a}-{b} names cell {id} which has no sea", g.name));
                    }
                }
            }
            let missing = sea.iter().flat_map(|&a| sea.iter().filter(move |&&b| a < b).map(move |&b| (a, b))).filter(|p| !seen.contains(p)).count();
            if missing > 0 {
                out.push(format!("theatre {}: {missing} pair(s) of sea cells are not connected", g.name));
            }
        }
        out
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

    fn synthetic(sea: [bool; 4], conns: Vec<(u32, u32, f32)>) -> SeaGrids {
        let mut cells = Vec::new();
        let mut areas = Vec::new();
        for i in 0..4u32 {
            let (xi, yi) = (i % 2, i / 2);
            let lo = [100.0 * xi as f32, 100.0 * yi as f32];
            cells.push(Cell { xi, yi, lo, hi: [lo[0] + 100.0, lo[1] + 100.0] });
            let seas = if sea[i as usize] { vec!["sea".to_string()] } else { vec![] };
            areas.push(Area { id: 10 + i, lands: vec![], seas, ports: vec![], numbers: vec![] });
        }
        SeaGrids {
            grids: vec![TheatreGrid {
                name: "t".into(),
                min: [0.0, 0.0],
                max: [200.0, 200.0],
                factor: 100.0,
                xsize: 2,
                ysize: 2,
                cells,
                areas,
                connections: conns,
            }],
        }
    }

    #[test]
    fn complete_conns_makes_the_graph_the_clique_of_the_sea_cells() {
        // cell 11 lost its sea when the grid moved but kept its edges; cell 13 gained sea and has none
        let mut sg = synthetic([true, false, true, true], vec![(10, 11, -1.0), (10, 12, 7.0), (11, 12, -1.0)]);
        assert_eq!(sg.graph_problems().len(), 3, "{:?}", sg.graph_problems());
        assert_eq!(sg.complete_conns("t").unwrap(), (2, 2));
        assert_eq!(sg.graph_problems(), Vec::<String>::new());
        // kept value, centre distance of the diagonal-adjacent pair is 0.0 (8-adjacent), all sorted
        assert_eq!(sg.grids[0].connections, vec![(10, 12, 7.0), (10, 13, 0.0), (12, 13, 0.0)]);
        // idempotent
        assert_eq!(sg.complete_conns("t").unwrap(), (0, 0));
    }

    #[test]
    fn graph_problems_names_a_sea_cell_without_connections() {
        let sg = synthetic([true, true, true, false], vec![(10, 11, 0.0)]);
        let problems = sg.graph_problems();
        assert_eq!(problems.len(), 1, "{problems:?}");
    }

    #[test]
    fn stock_sea_graphs_have_no_problems() {
        let Some(van) = read(&format!("{WORK}/exp1/backup/campaign_maps/nap_world/sea_grids.esf")) else { return };
        assert_eq!(SeaGrids::parse(&van.root).unwrap().graph_problems(), Vec::<String>::new());
    }

    /// The stock India grid of this editor's output: theatre "3", 4x4 cells of 95 from (440, -80) with ids 96..=111 of
    /// which `sea` carry sea; `edges` are the stock connections.
    fn india(sea: std::ops::RangeInclusive<u32>, edges: Vec<(u32, u32, f32)>) -> SeaGrids {
        let (mut cells, mut areas) = (Vec::new(), Vec::new());
        for id in 96..112u32 {
            let (xi, yi) = ((id - 96) % 4, (id - 96) / 4);
            let lo = [440.0 + 95.0 * xi as f32, -80.0 + 95.0 * yi as f32];
            cells.push(Cell { xi, yi, lo, hi: [lo[0] + 95.0, lo[1] + 95.0] });
            let seas = if sea.contains(&id) { vec!["bay_of_bengal".to_string()] } else { vec![] };
            areas.push(Area { id, lands: vec![], seas, ports: vec![], numbers: vec![] });
        }
        SeaGrids {
            grids: vec![TheatreGrid {
                name: "3".into(),
                min: [440.0, -80.0],
                max: [790.0, 300.0],
                factor: 95.0,
                xsize: 4,
                ysize: 4,
                cells,
                areas,
                connections: edges,
            }],
        }
    }

    /// Stock India had sea in cells 96..=104 (9 cells, 36 pairs); the extended bounds add sea to 105, 106 and 107
    /// (12 cells, 66 pairs). Every new cell must be connected to every other sea cell, the stock edges keep their
    /// costs, and no cell without sea gets an edge.
    #[test]
    fn india_nine_to_twelve_sea_cells_gives_the_full_66_pair_graph() {
        let mut stock_edges = Vec::new();
        for a in 96..=104u32 {
            for b in a + 1..=104 {
                stock_edges.push((a, b, (a * 1000 + b) as f32));
            }
        }
        assert_eq!(stock_edges.len(), 36);
        let mut sg = india(96..=104, stock_edges.clone());
        assert_eq!(sg.graph_problems(), Vec::<String>::new());
        // the regenerated lists give 105..=107 sea
        for area in &mut sg.grids[0].areas[9..12] {
            area.seas = vec!["bay_of_bengal".to_string()];
        }
        assert!(!sg.graph_problems().is_empty());
        assert_eq!(sg.complete_conns("3").unwrap(), (30, 0));
        assert_eq!(sg.graph_problems(), Vec::<String>::new());
        let conns = &sg.grids[0].connections;
        assert_eq!(conns.len(), 66);
        assert!(conns.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)));
        for new in [105u32, 106, 107] {
            let mut others: Vec<u32> = conns
                .iter()
                .filter_map(|&(a, b, _)| if a == new { Some(b) } else if b == new { Some(a) } else { None })
                .collect();
            others.sort_unstable();
            let want: Vec<u32> = (96..=107).filter(|&id| id != new).collect();
            assert_eq!(others, want, "cell {new}");
        }
        for &(a, b, cost) in &stock_edges {
            assert_eq!(conns.iter().find(|c| (c.0, c.1) == (a, b)).map(|c| c.2), Some(cost), "stock edge {a}-{b}");
        }
        // new pairs: 0.0 between 8-adjacent cells, else the distance between the cell centres
        let centre = |id: u32| {
            let (xi, yi) = ((id - 96) % 4, (id - 96) / 4);
            [440.0 + 95.0 * (xi as f64 + 0.5), -80.0 + 95.0 * (yi as f64 + 0.5)]
        };
        for &(a, b, cost) in conns.iter().filter(|c| !stock_edges.iter().any(|s| (s.0, s.1) == (c.0, c.1))) {
            let (ca, cb) = (centre(a), centre(b));
            let (dx, dy) = ((ca[0] - cb[0]).abs(), (ca[1] - cb[1]).abs());
            let want = if dx <= 95.0 && dy <= 95.0 { 0.0 } else { dx.hypot(dy) };
            assert_eq!(cost, want as f32, "new edge {a}-{b}");
        }
        let cost_of = |a: u32, b: u32| conns.iter().find(|c| (c.0, c.1) == (a, b)).unwrap().2;
        assert_eq!(cost_of(104, 105), 0.0);
        assert_eq!(cost_of(105, 106), 0.0);
        assert_eq!(cost_of(96, 105), 95.0f64.hypot(190.0) as f32);
        assert_eq!(cost_of(96, 107), 285.0f64.hypot(190.0) as f32);
        // cells 108..=111 have no sea and no edge
        assert!(conns.iter().all(|c| c.0 < 108 && c.1 < 108));
        // a second pass changes nothing
        let again = sg.clone();
        assert_eq!(sg.complete_conns("3").unwrap(), (0, 0));
        assert_eq!(sg, again);
    }

    /// 75-unit stock-like grid of `nx` x `ny` cells from `min`, cell ids from 0.
    fn stock_like(min: [f32; 2], nx: u32, ny: u32) -> SeaGrids {
        let (mut cells, mut areas) = (Vec::new(), Vec::new());
        for i in 0..nx * ny {
            let (xi, yi) = (i % nx, i / nx);
            let lo = [min[0] + 75.0 * xi as f32, min[1] + 75.0 * yi as f32];
            cells.push(Cell { xi, yi, lo, hi: [lo[0] + 75.0, lo[1] + 75.0] });
            areas.push(Area { id: i, lands: vec![], seas: vec![], ports: vec![], numbers: vec![] });
        }
        let max = [min[0] + 75.0 * nx as f32, min[1] + 75.0 * ny as f32];
        SeaGrids { grids: vec![TheatreGrid { name: "t".into(), min, max, factor: 75.0, xsize: nx, ysize: ny, cells, areas, connections: vec![] }] }
    }

    /// After a theatre is moved by `resize_theatre` (new origin, new cell size, negative coordinates, atlas-scaled
    /// rectangles) the fleet lookup must still agree with the cell rectangles the same file stores.
    #[test]
    fn fleet_cell_lookup_agrees_with_the_cell_boxes_after_resizing() {
        let cases: [(&str, [f32; 2], u32, u32, Rect); 3] = [
            ("India extended", [440.0, 20.0], 4, 4, [[440.0, -80.0], [790.0, 300.0]]),
            ("America extended west, negative x", [-760.0, 0.0], 6, 7, [[-888.0, 0.0], [-340.0, 460.0]]),
            (
                "atlas-scaled India",
                [440.0, 20.0],
                4,
                4,
                [[169.020263671875, -223.677001953125], [642.114501953125, 289.96826171875]],
            ),
        ];
        for (name, min, nx, ny, rect) in cases {
            let mut sg = stock_like(min, nx, ny);
            sg.grids[0].name = "t".into();
            sg.resize_theatre("t", rect).unwrap();
            let g = &sg.grids[0];
            for (i, cell) in g.cells.iter().enumerate() {
                let centre = [(cell.lo[0] as f64 + cell.hi[0] as f64) / 2.0, (cell.lo[1] as f64 + cell.hi[1] as f64) / 2.0];
                if centre[0] < rect[1][0] as f64 && centre[1] < rect[1][1] as f64 {
                    assert_eq!(sg.cell_index_at(centre[0], centre[1]), Some(i), "{name}: centre of cell {i}");
                }
            }
            // a lattice of points strictly inside the rectangle lands in the cell whose stored box holds it
            for iy in 0..23 {
                for ix in 0..23 {
                    let p = [
                        rect[0][0] as f64 + (rect[1][0] - rect[0][0]) as f64 * (ix as f64 + 0.37) / 23.0,
                        rect[0][1] as f64 + (rect[1][1] - rect[0][1]) as f64 * (iy as f64 + 0.37) / 23.0,
                    ];
                    let i = sg.cell_index_at(p[0], p[1]).unwrap_or_else(|| panic!("{name}: {p:?} is in no cell"));
                    let c = &g.cells[i];
                    assert!(
                        c.lo[0] as f64 <= p[0] && p[0] < c.hi[0] as f64 && c.lo[1] as f64 <= p[1] && p[1] < c.hi[1] as f64,
                        "{name}: {p:?} filed in cell {i} {:?}-{:?}",
                        c.lo,
                        c.hi
                    );
                }
            }
        }
    }

    /// Cell borders of integer grids that cross zero: the lower border belongs to the higher cell, just below it to
    /// the lower one, also for negative coordinates; the theatre rectangle's upper edges are outside.
    #[test]
    fn fleet_cell_lookup_borders_with_negative_origin() {
        // rectangle [-160, 290) x [-75, 450), 6 columns of 75
        let sg = stock_like([-160.0, -75.0], 6, 7);
        assert_eq!(sg.cell_index_at(-160.0, -75.0), Some(0));
        assert_eq!(sg.cell_index_at(-85.0, -75.0), Some(1));
        assert_eq!(sg.cell_index_at(-85.001, -75.0), Some(0));
        assert_eq!(sg.cell_index_at(-160.0, 0.0), Some(6));
        assert_eq!(sg.cell_index_at(-160.0, -0.001), Some(0));
        // either side of zero in the same cell, one cell row up
        assert_eq!(sg.cell_index_at(-0.5, 10.0), Some(8));
        assert_eq!(sg.cell_index_at(0.0, 0.0), Some(8));
        assert_eq!(sg.cell_index_at(289.999, 449.999), Some(6 * 7 - 1));
        // outside the rectangle
        assert_eq!(sg.cell_index_at(290.0, 0.0), None);
        assert_eq!(sg.cell_index_at(-160.1, 0.0), None);
        assert_eq!(sg.cell_index_at(-100.0, 450.0), None);
        assert_eq!(sg.cell_index_at(-100.0, -75.1), None);
    }
}
