//! `QUAD_TREE_BIT_ARRAY` line-of-sight caches and their stock-Q20 atlas resampling (native port of the accepted
//! `atlas_visibility.py`).
//!
//! Layout facts (reverse-engineered, verified by the codec round-trip check of the converter): a tree is `W x H`
//! cells over the campaign world (cell = world extent / W,H); `span` is the smallest power of two that is >= 8 and
//! >= max(W,H). A node of span `s` covers `s x s` cells. A leaf pair `(lo,hi)` is uniform (`0,0` / all ones) for span
//! > 8 and an 8x8 bit mask (`bit = y*8 + x`) for span 8. Children are ordered `(xlo,yhi) (xhi,yhi) (xlo,ylo)
//! (xhi,ylo)`; only all-empty / all-full quadruples are collapsed.
//!
//! Atlas rule: the world rectangle is unchanged, so `W`, `H`, `span` and the cell size stay. Every destination cell
//! centre is mapped back through the OWNING chart (the chart whose output crop contains the centre) to a source
//! point; the bit is the source bit when that point lies inside the chart's crop, 0 otherwise. Nothing else is
//! invented.

use anyhow::{Result, bail, ensure};

use super::nodes::u32_at;
use crate::atlas::{Plan, Rect};
use crate::esf::{Node, Value};

const TREE: &str = "QUAD_TREE_BIT_ARRAY";
const NODE: &str = "QUAD_TREE_BIT_ARRAY_NODE";
const FULL: u32 = u32::MAX;
const CHILD_ORDER: [(i64, i64); 4] = [(0, 1), (1, 1), (0, 0), (1, 0)];

pub fn span_for(w: i64, h: i64) -> i64 {
    let mut s = 8;
    while s < w || s < h {
        s *= 2;
    }
    s
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cell {
    Leaf(u32, u32),
    Split(Box<[Cell; 4]>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree {
    /// version of the `QUAD_TREE_BIT_ARRAY` record
    pub version: u8,
    /// version shared by every `QUAD_TREE_BIT_ARRAY_NODE`
    pub node_version: u8,
    pub w: i64,
    pub h: i64,
    pub span: i64,
    pub root: Cell,
}

fn parse_cell(n: &Node, version: u8) -> Result<Cell> {
    let Node::Record { name, version: v, children } = n else {
        bail!("expected {NODE}");
    };
    ensure!(name == NODE, "expected {NODE}, found {name}");
    ensure!(*v == version, "{NODE} versions differ ({v} vs {version}): the codec cannot reproduce mixed versions");
    if children.len() == 2 && children.iter().all(|c| matches!(c, Node::Value(Value::U32(_)))) {
        return Ok(Cell::Leaf(u32_at(children, 0)?, u32_at(children, 1)?));
    }
    if children.len() == 4 {
        let kids = children.iter().map(|c| parse_cell(c, version)).collect::<Result<Vec<_>>>()?;
        let Ok(kids) = <[Cell; 4]>::try_from(kids) else { unreachable!() };
        return Ok(Cell::Split(Box::new(kids)));
    }
    bail!("unsupported {NODE} shape ({} children)", children.len())
}

fn cell_node(cell: &Cell, version: u8) -> Node {
    match cell {
        Cell::Leaf(a, b) => Node::record(NODE, version, vec![Node::Value(Value::U32(*a)), Node::Value(Value::U32(*b))]),
        Cell::Split(kids) => Node::record(NODE, version, kids.iter().map(|k| cell_node(k, version)).collect()),
    }
}

impl Tree {
    /// Parse a `QUAD_TREE_BIT_ARRAY` record.
    pub fn parse(node: &Node) -> Result<Self> {
        let Node::Record { name, version, children } = node else {
            bail!("expected {TREE}");
        };
        ensure!(name == TREE, "expected {TREE}, found {name}");
        ensure!(children.len() == 4, "unexpected {TREE} layout ({} children)", children.len());
        let (w, h, span) = (i64::from(u32_at(children, 0)?), i64::from(u32_at(children, 1)?), i64::from(u32_at(children, 2)?));
        ensure!(span == span_for(w, h), "{TREE} span {span} != engine rule {}", span_for(w, h));
        let node_version = children[3].version().ok_or_else(|| anyhow::anyhow!("{TREE} has no root node"))?;
        let root = parse_cell(&children[3], node_version)?;
        Ok(Tree { version: *version, node_version, w, h, span, root })
    }

    pub fn to_node(&self) -> Node {
        Node::record(
            TREE,
            self.version,
            vec![
                Node::Value(Value::U32(self.w as u32)),
                Node::Value(Value::U32(self.h as u32)),
                Node::Value(Value::U32(self.span as u32)),
                cell_node(&self.root, self.node_version),
            ],
        )
    }
}

// ------------------------------------------------------------------------------------------------ dense windows

/// Dense bit window, row-major `[y][x]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub bits: Vec<bool>,
}

impl Grid {
    pub fn new(w: usize, h: usize) -> Self {
        Grid { w, h, bits: vec![false; w * h] }
    }
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.bits[y * self.w + x]
    }
    pub fn set(&mut self, x: usize, y: usize) {
        self.bits[y * self.w + x] = true;
    }
    pub fn count(&self) -> u64 {
        self.bits.iter().filter(|b| **b).count() as u64
    }
    /// Inclusive bounds of the set cells.
    pub fn bbox(&self) -> Option<[i64; 4]> {
        let mut best: Option<[i64; 4]> = None;
        for y in 0..self.h {
            for x in 0..self.w {
                if self.get(x, y) {
                    let (x, y) = (x as i64, y as i64);
                    best = Some(match best {
                        None => [x, y, x, y],
                        Some(b) => [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
                    });
                }
            }
        }
        best
    }
}

/// Dense array of the cells `win = (x0, y0, x1, y1)` (half-open, inside the span).
pub fn decode_window(root: &Cell, span: i64, win: [i64; 4]) -> Result<Grid> {
    let [x0, y0, x1, y1] = win;
    let mut out = Grid::new((x1 - x0).max(0) as usize, (y1 - y0).max(0) as usize);
    if out.bits.is_empty() {
        return Ok(out);
    }
    fn fill(cell: &Cell, s: i64, ox: i64, oy: i64, win: [i64; 4], out: &mut Grid) -> Result<()> {
        let [x0, y0, x1, y1] = win;
        if ox >= x1 || oy >= y1 || ox + s <= x0 || oy + s <= y0 {
            return Ok(());
        }
        match cell {
            Cell::Leaf(a, b) => {
                if s == 8 {
                    let m = u64::from(*a) | (u64::from(*b) << 32);
                    if m != 0 {
                        for i in 0..64i64 {
                            if m >> i & 1 == 1 {
                                let (x, y) = (ox + (i & 7), oy + (i >> 3));
                                if x0 <= x && x < x1 && y0 <= y && y < y1 {
                                    out.set((x - x0) as usize, (y - y0) as usize);
                                }
                            }
                        }
                    }
                } else if (*a, *b) == (FULL, FULL) {
                    for y in oy.max(y0)..(oy + s).min(y1) {
                        for x in ox.max(x0)..(ox + s).min(x1) {
                            out.set((x - x0) as usize, (y - y0) as usize);
                        }
                    }
                } else if (*a, *b) != (0, 0) {
                    bail!("non-uniform leaf pair at span {s}");
                }
                Ok(())
            }
            Cell::Split(kids) => {
                ensure!(s > 8, "children below leaf span");
                let h = s / 2;
                for (k, (dx, dy)) in CHILD_ORDER.iter().enumerate() {
                    fill(&kids[k], h, ox + dx * h, oy + dy * h, win, out)?;
                }
                Ok(())
            }
        }
    }
    fill(root, span, 0, 0, win, &mut out)?;
    Ok(out)
}

/// Tight inclusive `(x0, y0, x1, y1)` bounds of all set cells.
pub fn tree_bbox(root: &Cell, span: i64) -> Result<Option<[i64; 4]>> {
    fn rec(cell: &Cell, s: i64, ox: i64, oy: i64, best: &mut Option<[i64; 4]>) -> Result<()> {
        let mut take = |x0: i64, y0: i64, x1: i64, y1: i64| {
            *best = Some(match *best {
                None => [x0, y0, x1, y1],
                Some(b) => [b[0].min(x0), b[1].min(y0), b[2].max(x1), b[3].max(y1)],
            });
        };
        match cell {
            Cell::Leaf(a, b) => {
                if s == 8 {
                    let m = u64::from(*a) | (u64::from(*b) << 32);
                    for i in 0..64i64 {
                        if m >> i & 1 == 1 {
                            take(ox + (i & 7), oy + (i >> 3), ox + (i & 7), oy + (i >> 3));
                        }
                    }
                } else if (*a, *b) == (FULL, FULL) {
                    take(ox, oy, ox + s - 1, oy + s - 1);
                } else if (*a, *b) != (0, 0) {
                    bail!("non-uniform leaf pair");
                }
                Ok(())
            }
            Cell::Split(kids) => {
                let h = s / 2;
                for (k, (dx, dy)) in CHILD_ORDER.iter().enumerate() {
                    rec(&kids[k], h, ox + dx * h, oy + dy * h, best)?;
                }
                Ok(())
            }
        }
    }
    let mut best = None;
    rec(root, span, 0, 0, &mut best)?;
    Ok(best)
}

fn canon(kids: [Cell; 4]) -> Cell {
    if kids.iter().all(|k| matches!(k, Cell::Leaf(0, 0))) {
        return Cell::Leaf(0, 0);
    }
    if kids.iter().all(|k| matches!(k, Cell::Leaf(FULL, FULL))) {
        return Cell::Leaf(FULL, FULL);
    }
    Cell::Split(Box::new(kids))
}

/// Root cell of a `span x span` tree whose set cells are `arr[y][x]` placed at cell origin `(ox, oy)`.
pub fn encode_window(arr: &Grid, ox: i64, oy: i64, span: i64) -> Cell {
    let (w, h) = (arr.w, arr.h);
    let stride = w + 1;
    let mut ii = vec![0i64; (h + 1) * stride];
    for y in 0..h {
        let mut row = 0i64;
        for x in 0..w {
            row += i64::from(arr.get(x, y));
            ii[(y + 1) * stride + x + 1] = ii[y * stride + x + 1] + row;
        }
    }
    let count = |x0: i64, y0: i64, x1: i64, y1: i64| -> i64 {
        let ax0 = (x0 - ox).max(0) as usize;
        let ay0 = (y0 - oy).max(0) as usize;
        let ax1 = (x1 - ox).min(w as i64).max(0) as usize;
        let ay1 = (y1 - oy).min(h as i64).max(0) as usize;
        if ax0 >= ax1 || ay0 >= ay1 {
            return 0;
        }
        ii[ay1 * stride + ax1] - ii[ay0 * stride + ax1] - ii[ay1 * stride + ax0] + ii[ay0 * stride + ax0]
    };
    fn build(cx: i64, cy: i64, s: i64, arr: &Grid, ox: i64, oy: i64, count: &dyn Fn(i64, i64, i64, i64) -> i64) -> Cell {
        let c = count(cx, cy, cx + s, cy + s);
        if c == 0 {
            return Cell::Leaf(0, 0);
        }
        if c == s * s {
            return Cell::Leaf(FULL, FULL);
        }
        if s == 8 {
            let mut m = 0u64;
            for yy in 0..8i64 {
                for xx in 0..8i64 {
                    let (gx, gy) = (cx + xx - ox, cy + yy - oy);
                    if gx >= 0 && gy >= 0 && (gx as usize) < arr.w && (gy as usize) < arr.h && arr.get(gx as usize, gy as usize) {
                        m |= 1 << (yy * 8 + xx);
                    }
                }
            }
            return Cell::Leaf(m as u32, (m >> 32) as u32);
        }
        let h = s / 2;
        let kids: Vec<Cell> = CHILD_ORDER.iter().map(|(dx, dy)| build(cx + dx * h, cy + dy * h, h, arr, ox, oy, count)).collect();
        let Ok(kids) = <[Cell; 4]>::try_from(kids) else { unreachable!() };
        canon(kids)
    }
    build(0, 0, span, arr, ox, oy, &count)
}

// ------------------------------------------------------------------------------------------------ atlas resampler

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResampleStats {
    pub source_cells: u64,
    pub source_cells_in_crops: u64,
    pub source_cells_represented: u64,
    pub dest_cells: u64,
}

pub struct Resampled {
    /// None when no source cell exists or no destination cell survives
    pub dest: Option<Grid>,
    pub origin: [i64; 2],
    pub stats: ResampleStats,
}

fn inside(r: &Rect, x: f64, y: f64) -> bool {
    r[0][0] <= x && x <= r[1][0] && r[0][1] <= y && y <= r[1][1]
}

struct ChartCrops {
    offset: [f64; 2],
    crops: Vec<Rect>,
}

/// Chart-aware nearest-neighbour resampler over the plan's charts.
pub struct Visibility {
    pub min: [f64; 2],
    ext: [f64; 2],
    scale: f64,
    charts: Vec<ChartCrops>,
}

impl Visibility {
    pub fn new(plan: &Plan) -> Result<Self> {
        ensure!(plan.world == plan.source_world, "atlas visibility requires world == source_world (cell grid unchanged)");
        ensure!(plan.scale >= 1.0, "atlas visibility requires scale >= 1 (nearest sampling must not drop source cells)");
        let min = plan.world[0];
        Ok(Visibility {
            min,
            ext: [plan.world[1][0] - min[0], plan.world[1][1] - min[1]],
            scale: plan.scale,
            charts: plan.charts.iter().map(|c| ChartCrops { offset: c.offset, crops: c.crop_rects.clone() }).collect(),
        })
    }

    pub fn cell_size(&self, w: i64, h: i64) -> [f64; 2] {
        [self.ext[0] / w as f64, self.ext[1] / h as f64]
    }

    /// Source LOS convention: set cells lie inside `[c0, c1)`.
    pub fn cell_range(&self, bmin: [f32; 2], bmax: [f32; 2], w: i64, h: i64) -> [i64; 4] {
        let cs = self.cell_size(w, h);
        let c = |v: f32, i: usize| ((f64::from(v) - self.min[i]) / cs[i]).floor() as i64;
        [c(bmin[0], 0), c(bmin[1], 1), c(bmax[0], 0), c(bmax[1], 1)]
    }

    pub fn cell_world(&self, cx: i64, cy: i64, w: i64, h: i64) -> [f64; 2] {
        let cs = self.cell_size(w, h);
        [self.min[0] + cx as f64 * cs[0], self.min[1] + cy as f64 * cs[1]]
    }

    /// Resample the set cells of the source window `win` into the atlas.
    pub fn resample(&self, root: &Cell, span: i64, w: i64, h: i64, win: [i64; 4]) -> Result<Resampled> {
        let [sx0, sy0, sx1, sy1] = win;
        let src = decode_window(root, span, win)?;
        let mut stats = ResampleStats { source_cells: src.count(), ..Default::default() };
        let none = |stats| Resampled { dest: None, origin: [0, 0], stats };
        if stats.source_cells == 0 {
            return Ok(none(stats));
        }
        let cs = self.cell_size(w, h);
        // source cell centres inside any chart crop (loss / representation accounting)
        for y in 0..src.h {
            for x in 0..src.w {
                if src.get(x, y) {
                    let cx = self.min[0] + ((x as i64 + sx0) as f64 + 0.5) * cs[0];
                    let cy = self.min[1] + ((y as i64 + sy0) as f64 + 0.5) * cs[1];
                    if self.charts.iter().any(|c| c.crops.iter().any(|r| inside(r, cx, cy))) {
                        stats.source_cells_in_crops += 1;
                    }
                }
            }
        }
        // world rectangle of the source window
        let (wx0, wy0) = (self.min[0] + sx0 as f64 * cs[0], self.min[1] + sy0 as f64 * cs[1]);
        let (wx1, wy1) = (self.min[0] + sx1 as f64 * cs[0], self.min[1] + sy1 as f64 * cs[1]);
        let mut windows: Vec<(usize, i64, i64, i64, i64)> = Vec::new();
        for (ci, chart) in self.charts.iter().enumerate() {
            let [ox, oy] = chart.offset;
            for r in &chart.crops {
                let (ix0, iy0) = (r[0][0].max(wx0), r[0][1].max(wy0));
                let (ix1, iy1) = (r[1][0].min(wx1), r[1][1].min(wy1));
                if ix0 >= ix1 || iy0 >= iy1 {
                    continue;
                }
                let (dx0, dx1) = (ix0 * self.scale + ox, ix1 * self.scale + ox);
                let (dy0, dy1) = (iy0 * self.scale + oy, iy1 * self.scale + oy);
                windows.push((
                    ci,
                    ((dx0 - self.min[0]) / cs[0]).floor() as i64 - 1,
                    ((dy0 - self.min[1]) / cs[1]).floor() as i64 - 1,
                    ((dx1 - self.min[0]) / cs[0]).ceil() as i64 + 1,
                    ((dy1 - self.min[1]) / cs[1]).ceil() as i64 + 1,
                ));
            }
        }
        if windows.is_empty() {
            return Ok(none(stats));
        }
        let gx0 = windows.iter().map(|k| k.1).min().unwrap().max(0);
        let gy0 = windows.iter().map(|k| k.2).min().unwrap().max(0);
        let gx1 = windows.iter().map(|k| k.3).max().unwrap().min(w);
        let gy1 = windows.iter().map(|k| k.4).max().unwrap().min(h);
        if gx1 <= gx0 || gy1 <= gy0 {
            return Ok(none(stats));
        }
        let mut dest = Grid::new((gx1 - gx0) as usize, (gy1 - gy0) as usize);
        let mut represented = Grid::new(src.w, src.h);
        for (ci, x0, y0, x1, y1) in windows {
            let (x0, y0, x1, y1) = (x0.max(gx0), y0.max(gy0), x1.min(gx1), y1.min(gy1));
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            let chart = &self.charts[ci];
            let [ox, oy] = chart.offset;
            let px: Vec<f64> = (x0..x1).map(|i| (self.min[0] + (i as f64 + 0.5) * cs[0] - ox) / self.scale).collect();
            let py: Vec<f64> = (y0..y1).map(|j| (self.min[1] + (j as f64 + 0.5) * cs[1] - oy) / self.scale).collect();
            for (jj, &y) in py.iter().enumerate() {
                let sy = ((y - self.min[1]) / cs[1]).floor() as i64 - sy0;
                if sy < 0 || sy >= src.h as i64 {
                    continue;
                }
                for (ii, &x) in px.iter().enumerate() {
                    let sx = ((x - self.min[0]) / cs[0]).floor() as i64 - sx0;
                    if sx < 0 || sx >= src.w as i64 || !chart.crops.iter().any(|r| inside(r, x, y)) {
                        continue;
                    }
                    if src.get(sx as usize, sy as usize) {
                        dest.set((x0 - gx0) as usize + ii, (y0 - gy0) as usize + jj);
                        represented.set(sx as usize, sy as usize);
                    }
                }
            }
        }
        stats.dest_cells = dest.count();
        stats.source_cells_represented = represented.count();
        Ok(Resampled { dest: Some(dest), origin: [gx0, gy0], stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_with(cells: &[(usize, usize)], w: usize, h: usize) -> Grid {
        let mut g = Grid::new(w, h);
        for &(x, y) in cells {
            g.set(x, y);
        }
        g
    }

    #[test]
    fn encode_decode_round_trips_masks_full_blocks_and_negative_space() {
        let span = 64;
        let mut cells = vec![(3, 5), (4, 5), (12, 20), (63, 63), (0, 0)];
        // a full 16x16 block collapses to a uniform leaf above span 8
        for y in 16..32 {
            for x in 32..48 {
                cells.push((x, y));
            }
        }
        let grid = tree_with(&cells, 64, 64);
        let root = encode_window(&grid, 0, 0, span);
        assert_eq!(decode_window(&root, span, [0, 0, 64, 64]).unwrap(), grid);
        assert_eq!(tree_bbox(&root, span).unwrap(), Some([0, 0, 63, 63]));
        // canonical: re-encoding the decoded window reproduces the tree exactly
        let again = encode_window(&decode_window(&root, span, [0, 0, 64, 64]).unwrap(), 0, 0, span);
        assert_eq!(again, root);
        // an offset window reaches the same cells
        let sub = decode_window(&root, span, [32, 16, 48, 32]).unwrap();
        assert_eq!(sub.count(), 256);
        let moved = encode_window(&sub, 32, 16, span);
        assert_eq!(decode_window(&moved, span, [0, 0, 64, 64]).unwrap().count(), 256);
    }

    #[test]
    fn empty_window_is_an_empty_leaf() {
        let root = encode_window(&Grid::new(0, 0), 0, 0, 64);
        assert_eq!(root, Cell::Leaf(0, 0));
        assert_eq!(tree_bbox(&root, 64).unwrap(), None);
    }

    #[test]
    fn tree_node_round_trip_keeps_versions() {
        let grid = tree_with(&[(1, 1), (40, 40)], 64, 64);
        let tree = Tree { version: 3, node_version: 2, w: 64, h: 64, span: 64, root: encode_window(&grid, 0, 0, 64) };
        assert_eq!(Tree::parse(&tree.to_node()).unwrap(), tree);
    }

    #[test]
    fn every_destination_cell_maps_back_to_a_set_source_cell_inside_its_chart_crop() {
        let plan = Plan::proven();
        let vis = Visibility::new(&plan).unwrap();
        // 2560 x 1280 world at one cell per world unit; the span is 4096.
        let (w, h, span) = (2560i64, 1280i64, 4096i64);
        let cs = vis.cell_size(w, h);
        // a source patch well inside Europe (theatre rect [-180,140]..[440,520])
        let (x0, y0) = (((100.0 - vis.min[0]) / cs[0]) as usize, ((300.0 - vis.min[1]) / cs[1]) as usize);
        let patch: Vec<(usize, usize)> = (0..6).flat_map(|dy| (0..6).map(move |dx| (x0 + dx, y0 + dy))).collect();
        let grid = tree_with(&patch, w as usize, h as usize);
        let root = encode_window(&grid, 0, 0, span);
        let win = [x0 as i64 - 2, y0 as i64 - 2, x0 as i64 + 8, y0 as i64 + 8];
        let res = vis.resample(&root, span, w, h, win).unwrap();
        let dest = res.dest.expect("surviving cells");
        assert_eq!(res.stats.source_cells, 36);
        assert_eq!(res.stats.source_cells_represented, 36);
        assert!(res.stats.dest_cells >= 36, "scale >= 1 may not lose cells");
        let chart = plan.chart("2").unwrap();
        for y in 0..dest.h {
            for x in 0..dest.w {
                if dest.get(x, y) {
                    let c = [vis.min[0] + ((x as i64 + res.origin[0]) as f64 + 0.5) * cs[0], vis.min[1] + ((y as i64 + res.origin[1]) as f64 + 0.5) * cs[1]];
                    let p = plan.inverse_xy(c, chart);
                    let (sx, sy) = (((p[0] - vis.min[0]) / cs[0]).floor() as usize, ((p[1] - vis.min[1]) / cs[1]).floor() as usize);
                    assert!(grid.get(sx, sy), "destination cell {c:?} has no set source cell");
                }
            }
        }
    }
}
