//! Typed view of the `grid_data` of an ETW/NTW `pathfinding.esf` and the region re-labelling a region split needs.
//!
//! Format facts (vanilla NTW `nap_europe`, see work/research/pfgrid.py):
//! * `pathfinding_areas[0]` = [vertices[] (i32 x, i32 y fixed 12.20), u32[] polylines (count, ids...; ids 0..=3 are
//!   the cell corners BL, TL, BR, TR), grid_data].
//! * grid_data children: [0] i32 x0, [1] i32 y0 (fixed 12.20), [2] u16, [3] u16, [4] i32 cell size (fixed),
//!   [5] u32 cols, [6] u32 rows, [7] u32 pieces + empty cells, [8] u16 region count, [9] u16 listed region count,
//!   [10] i16[] region ids = regions.esf indices, [11] u16[] path list, [12] grid_cells[].
//! * path list = 1-based positions into [10] for every single region, then 0 (= sea path), then groups
//!   `n, pos...`; a path id is the position of its entry in that sequence (1023 = transition / none).
//! * grid_cells are in raster order (row-major) with run-length coding of cells without pieces: a cell item is
//!   [u8[8] header, boundaries[] pieces] or [header, boundaries[], u32 a, u16 path id, u8[] 12-byte lines] where every
//!   line is (header, u32 a) of one following cell sharing the run's path id.
//! * piece = (u32 a, u32 b): a = passable << 24 | flags << 4 | type; b = path id << 22 | grid << 21 | polyline index.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail, ensure};

use super::geometry::{P, Rect, point_in_ring};
use super::pathsynth::{model::Model, rules::View};
use super::regions::RegionData;
use crate::esf::{self, Node, Value};

pub const TRANSITION: u16 = 1023;
const FIXED: f64 = 1.0 / (1u64 << 20) as f64;

#[derive(Clone, Debug)]
pub struct Piece {
    pub a: u32,
    pub b: u32,
}

impl Piece {
    pub fn path_id(&self) -> u16 {
        (self.b >> 22) as u16
    }
    pub fn polyline(&self) -> usize {
        (self.b & 0x1F_FFFF) as usize
    }
    pub fn with_path_id(&self, id: u16) -> Piece {
        Piece {
            a: self.a,
            b: ((id as u32) << 22) | (self.b & 0x3F_FFFF),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub header: [u8; 8],
    pub pieces: Vec<Piece>,
    /// (a, path id) of a cell without pieces
    pub empty: Option<(u32, u16)>,
}

#[derive(Clone, Debug)]
pub struct Grid {
    /// theatre area index this grid was parsed from
    pub area: usize,
    pub x0: f64,
    pub y0: f64,
    pub size: f64,
    pub cols: usize,
    pub rows: usize,
    /// `grid_data` children 2 and 3 (u16): (x0 + 1280) / size and (y0 + 640) / size in the shipped files
    pub origin_index: [u16; 2],
    /// regions.esf indices
    pub region_ids: Vec<i16>,
    /// path id -> region ids (empty = sea)
    pub paths: Vec<Vec<i16>>,
    pub cells: Vec<Cell>,
    /// vertex table in 12.20 fixed point (entries 0..4 are dummies; corner ids 0..=3 never index it)
    pub vertices: Vec<[i32; 2]>,
    /// polyline start offset -> vertex ids
    pub polylines: BTreeMap<usize, Vec<u32>>,
}

/// Fixed-point (12.20) value of a coordinate that is a multiple of 2^-20.
pub fn fixed_of(v: f64) -> i32 {
    (v / FIXED).round() as i32
}

fn u16_of(n: &Node) -> Result<u16> {
    match n {
        Node::Value(Value::U16(v)) => Ok(*v),
        other => bail!("expected u16, found {other:?}"),
    }
}

fn raw(n: &Node) -> Result<&[u8]> {
    n.raw_array()
}

impl Grid {
    /// Number of theatre areas (`pathfinding_areas` items) in the file.
    pub fn area_count(root: &Node) -> Result<usize> {
        Ok(root.get("pathfinding_areas")?.items()?.len())
    }

    /// Index of the theatre area whose grid rectangle contains `pt`, if any.
    pub fn area_at(root: &Node, pt: P) -> Result<Option<usize>> {
        for (i, area) in root.get("pathfinding_areas")?.items()?.iter().enumerate() {
            let gd = area[2].children()?;
            let x0 = gd[0].as_i32()? as f64 * FIXED;
            let y0 = gd[1].as_i32()? as f64 * FIXED;
            let size = gd[4].as_i32()? as f64 * FIXED;
            let (cols, rows) = (gd[5].as_u32()? as f64, gd[6].as_u32()? as f64);
            if pt[0] >= x0 && pt[1] >= y0 && pt[0] < x0 + cols * size && pt[1] < y0 + rows * size {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    pub fn parse(root: &Node) -> Result<Self> {
        Self::parse_area(root, 0)
    }

    /// Parse theatre area `index` of the file.
    pub fn parse_area(root: &Node, index: usize) -> Result<Self> {
        let areas = root.get("pathfinding_areas")?.items()?;
        let area = areas.get(index).ok_or_else(|| anyhow!("pathfinding.esf has no area {index}"))?;
        let vertices = area[0]
            .items()?
            .iter()
            .map(|v| Ok([v[0].as_i32()?, v[1].as_i32()?]))
            .collect::<Result<Vec<_>>>()?;
        let poly = area[1].u32_array()?;
        let mut polylines = BTreeMap::new();
        let mut i = 0;
        while i < poly.len() {
            let n = poly[i] as usize;
            polylines.insert(i, poly[i + 1..i + 1 + n].to_vec());
            i += 1 + n;
        }
        let gd = area[2].children()?;
        ensure!(gd.len() == 13, "grid_data has {} children", gd.len());
        let x0 = gd[0].as_i32()? as f64 * FIXED;
        let y0 = gd[1].as_i32()? as f64 * FIXED;
        let size = gd[4].as_i32()? as f64 * FIXED;
        let cols = gd[5].as_u32()? as usize;
        let rows = gd[6].as_u32()? as usize;
        let origin_index = [u16_of(&gd[2])?, u16_of(&gd[3])?];
        let region_ids: Vec<i16> = raw(&gd[10])?
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let list: Vec<u16> = raw(&gd[11])?
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let rid = |pos: u16| -> Result<i16> {
            region_ids
                .get(pos as usize - 1)
                .copied()
                .ok_or_else(|| anyhow!("path list position {pos} outside the region list"))
        };
        let mut paths = Vec::new();
        let mut k = 0;
        while list[k] != 0 {
            paths.push(vec![rid(list[k])?]);
            k += 1;
        }
        k += 1;
        paths.push(Vec::new());
        while k < list.len() {
            let n = list[k] as usize;
            paths.push((0..n).map(|j| rid(list[k + 1 + j])).collect::<Result<_>>()?);
            k += 1 + n;
        }
        let mut cells = Vec::with_capacity(cols * rows);
        for item in gd[12].items()? {
            let header: [u8; 8] = raw(&item[0])?.try_into().map_err(|_| anyhow!("cell header size"))?;
            let pieces = item[1]
                .items()?
                .iter()
                .map(|p| {
                    Ok(Piece {
                        a: p[0].as_u32()?,
                        b: p[1].as_u32()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            match item.len() {
                2 => cells.push(Cell {
                    header,
                    pieces,
                    empty: None,
                }),
                5 => {
                    ensure!(pieces.is_empty(), "run descriptor with pieces");
                    let a = item[2].as_u32()?;
                    let id = match &item[3] {
                        Node::Value(Value::U16(v)) => *v,
                        other => bail!("run path id {other:?}"),
                    };
                    cells.push(Cell {
                        header,
                        pieces: Vec::new(),
                        empty: Some((a, id)),
                    });
                    for line in raw(&item[4])?.chunks_exact(12) {
                        cells.push(Cell {
                            header: line[..8].try_into().unwrap(),
                            pieces: Vec::new(),
                            empty: Some((u32::from_le_bytes(line[8..12].try_into().unwrap()), id)),
                        });
                    }
                }
                n => bail!("grid cell with {n} children"),
            }
        }
        ensure!(cells.len() == cols * rows, "{} cells decoded for a {cols}x{rows} grid", cells.len());
        Ok(Self {
            area: index,
            x0,
            y0,
            size,
            cols,
            rows,
            origin_index,
            region_ids,
            paths,
            cells,
            vertices,
            polylines,
        })
    }

    pub fn store(&self, root: &mut Node) -> Result<()> {
        let singles = self.paths.iter().position(|p| p.is_empty()).ok_or_else(|| anyhow!("no sea path"))?;
        let pos = |id: i16| -> Result<u16> {
            self.region_ids
                .iter()
                .position(|&r| r == id)
                .map(|p| p as u16 + 1)
                .ok_or_else(|| anyhow!("region {id} is not in the pathfinding region list"))
        };
        let mut list: Vec<u16> = Vec::new();
        for p in &self.paths[..singles] {
            ensure!(p.len() == 1, "single path with {} regions", p.len());
            list.push(pos(p[0])?);
        }
        list.push(0);
        for p in &self.paths[singles + 1..] {
            list.push(p.len() as u16);
            for &r in p {
                list.push(pos(r)?);
            }
        }
        let mut items: Vec<Vec<Node>> = Vec::new();
        let mut counted = 0usize;
        let mut i = 0;
        while i < self.cells.len() {
            let c = &self.cells[i];
            let header = Node::Value(Value::Array {
                code: 0x46,
                data: c.header.to_vec(),
            });
            let pieces = Node::array(
                "boundaries",
                0,
                c.pieces.iter().map(|p| vec![esf::u32v(p.a), esf::u32v(p.b)]).collect(),
            );
            counted += c.pieces.len();
            match c.empty {
                None => {
                    items.push(vec![header, pieces]);
                    i += 1;
                }
                Some((a, id)) => {
                    let mut lines = Vec::new();
                    let mut j = i + 1;
                    while j < self.cells.len() {
                        match self.cells[j].empty {
                            Some((la, lid)) if lid == id => {
                                lines.extend_from_slice(&self.cells[j].header);
                                lines.extend_from_slice(&la.to_le_bytes());
                                j += 1;
                            }
                            _ => break,
                        }
                    }
                    counted += j - i;
                    items.push(vec![
                        header,
                        pieces,
                        esf::u32v(a),
                        esf::u16v(id),
                        Node::Value(Value::Array {
                            code: 0x46,
                            data: lines,
                        }),
                    ]);
                    i = j;
                }
            }
        }
        let area = &mut root.get_mut("pathfinding_areas")?.items_mut()?[self.area];
        {
            let verts = area[0].items_mut()?;
            verts.clear();
            verts.extend(self.vertices.iter().map(|v| vec![esf::i32v(v[0]), esf::i32v(v[1])]));
        }
        let mut poly: Vec<u32> = Vec::new();
        let mut expect = 0usize;
        for (&off, ids) in &self.polylines {
            ensure!(off == expect, "polyline table has a gap at offset {expect}");
            poly.push(ids.len() as u32);
            poly.extend_from_slice(ids);
            expect = poly.len();
        }
        area[1] = esf::arr_u32(&poly);
        let gd = area[2].children_mut()?;
        gd[0] = esf::i32v(fixed_of(self.x0));
        gd[1] = esf::i32v(fixed_of(self.y0));
        gd[2] = esf::u16v(self.origin_index[0]);
        gd[3] = esf::u16v(self.origin_index[1]);
        gd[4] = esf::i32v(fixed_of(self.size));
        gd[5] = esf::u32v(self.cols as u32);
        gd[6] = esf::u32v(self.rows as u32);
        gd[7] = esf::u32v(counted as u32);
        gd[8] = esf::u16v(self.region_ids.len() as u16);
        gd[9] = esf::u16v(self.region_ids.len() as u16);
        gd[10] = Node::Value(Value::Array {
            code: 0x43,
            data: self.region_ids.iter().flat_map(|r| r.to_le_bytes()).collect(),
        });
        gd[11] = Node::Value(Value::Array {
            code: 0x47,
            data: list.iter().flat_map(|r| r.to_le_bytes()).collect(),
        });
        gd[12] = Node::array("grid_cells", 0, items);
        Ok(())
    }

    pub fn cell_origin(&self, index: usize) -> P {
        let (col, row) = (index % self.cols, index / self.cols);
        [self.x0 + col as f64 * self.size, self.y0 + row as f64 * self.size]
    }

    /// Closed polygon of a piece (cell corners resolved).
    pub fn piece_polygon(&self, cell: usize, piece: &Piece) -> Result<Vec<P>> {
        let o = self.cell_origin(cell);
        let s = self.size;
        let ids = self
            .polylines
            .get(&piece.polyline())
            .ok_or_else(|| anyhow!("piece references polyline offset {}", piece.polyline()))?;
        ids.iter()
            .map(|&v| {
                Ok(match v {
                    0 => [o[0], o[1]],
                    1 => [o[0], o[1] + s],
                    2 => [o[0] + s, o[1]],
                    3 => [o[0] + s, o[1] + s],
                    v => {
                        let p = self
                            .vertices
                            .get(v as usize)
                            .ok_or_else(|| anyhow!("piece vertex {v} outside the vertex table"))?;
                        [p[0] as f64 * FIXED, p[1] as f64 * FIXED]
                    }
                })
            })
            .collect()
    }

    /// Index of the single-region path of `region`, if any.
    pub fn single_path(&self, region: i16) -> Option<u16> {
        self.paths
            .iter()
            .position(|p| p.len() == 1 && p[0] == region)
            .map(|p| p as u16)
    }

    /// Re-label every piece and empty cell of `old` whose polygon centroid / cell centre satisfies `in_new` as
    /// belonging to the new region `new` (regions.esf index). Group paths get a twin with `old` replaced.
    /// Returns (pieces relabelled, empty cells relabelled).
    pub fn relabel(&mut self, old: i16, new: i16, in_new: &dyn Fn(P) -> bool) -> Result<(usize, usize)> {
        ensure!(!self.region_ids.contains(&new), "region {new} is already in the pathfinding grid");
        let singles = self.paths.iter().position(|p| p.is_empty()).ok_or_else(|| anyhow!("no sea path"))?;
        // insert the new single path right after the existing singles: every later path id shifts by one
        self.region_ids.push(new);
        self.paths.insert(singles, vec![new]);
        let shift = |id: u16| if id == TRANSITION || (id as usize) < singles { id } else { id + 1 };
        for c in self.cells.iter_mut() {
            for p in c.pieces.iter_mut() {
                *p = p.with_path_id(shift(p.path_id()));
            }
            if let Some((_, id)) = c.empty.as_mut() {
                *id = shift(*id);
            }
        }
        let old_single = self.single_path(old).ok_or_else(|| anyhow!("region {old} has no path"))?;
        let new_single = singles as u16;
        let mut twins: BTreeMap<u16, u16> = BTreeMap::new();
        twins.insert(old_single, new_single);
        let mut relabelled = (0usize, 0usize);
        for i in 0..self.cells.len() {
            let centre = {
                let o = self.cell_origin(i);
                [o[0] + self.size / 2.0, o[1] + self.size / 2.0]
            };
            if let Some((_, id)) = self.cells[i].empty {
                if id == old_single && in_new(centre) {
                    self.cells[i].empty = Some((self.cells[i].empty.unwrap().0, new_single));
                    relabelled.1 += 1;
                }
                continue;
            }
            let pieces = self.cells[i].pieces.clone();
            let mut changed = false;
            let mut out = Vec::with_capacity(pieces.len());
            for p in pieces {
                let id = p.path_id();
                let members = if (id as usize) < self.paths.len() { &self.paths[id as usize] } else { &Vec::new() };
                if id != TRANSITION && members.contains(&old) {
                    let poly = self.piece_polygon(i, &p)?;
                    if in_new(centroid(&poly)) {
                        let twin = match twins.get(&id) {
                            Some(&t) => t,
                            None => {
                                let group: Vec<i16> = members.iter().map(|&r| if r == old { new } else { r }).collect();
                                let t = match self.paths.iter().position(|g| *g == group) {
                                    Some(t) => t as u16,
                                    None => {
                                        self.paths.push(group);
                                        (self.paths.len() - 1) as u16
                                    }
                                };
                                twins.insert(id, t);
                                t
                            }
                        };
                        out.push(p.with_path_id(twin));
                        changed = true;
                        relabelled.0 += 1;
                        continue;
                    }
                }
                out.push(p);
            }
            if changed {
                self.cells[i].pieces = out;
            }
        }
        Ok(relabelled)
    }
}

/// Area-weighted centroid of a closed polygon (mean of the points for degenerate ones).
pub fn centroid(poly: &[P]) -> P {
    let n = poly.len();
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        let w = p[0] * q[1] - q[0] * p[1];
        a += w;
        cx += (p[0] + q[0]) * w;
        cy += (p[1] + q[1]) * w;
    }
    if a.abs() < 1e-9 {
        let m = n.max(1) as f64;
        return [poly.iter().map(|p| p[0]).sum::<f64>() / m, poly.iter().map(|p| p[1]).sum::<f64>() / m];
    }
    [cx / (3.0 * a), cy / (3.0 * a)]
}

/// A transition area of a theatre before and after the theatre rectangle moved (map units, `[min, max]`).
#[derive(Clone, Copy, Debug)]
pub struct TransitionArea {
    pub id: u32,
    pub old: Rect,
    pub new: Rect,
}

#[derive(Clone, Debug, Default)]
pub struct ExtendReport {
    pub cols: usize,
    pub rows: usize,
    /// old grid origin in the new grid (cells)
    pub offset: [usize; 2],
    pub new_sea_cells: usize,
    pub new_outside_cells: usize,
    /// old type-2 pieces inside a moved transition area that became sea pieces
    pub transition_pieces_to_sea: usize,
    pub words_changed: usize,
}

const OUTSIDE_HEADER: [u8; 8] = [0x1f, 0xff, 0x1f, 0xff, 0xff, 0x1f, 0xff, 0x1f];
/// piece / run word of a whole passable sea cell before the neighbour rules are applied
const SEA_RUN_WORD: u32 = 0xff11_0111;
const OUTSIDE_RUN_WORD: u32 = 0x0000_0002;

/// Outline of regions.esf with the bounding box used to find the innermost polygon covering a point.
struct Locator {
    rings: Vec<(usize, usize, bool, Vec<P>, [f64; 4], f64)>,
    sea: Vec<bool>,
}

impl Locator {
    fn new(rd: &RegionData, window: [f64; 4]) -> Self {
        let mut rings = Vec::new();
        for (ri, r) in rd.regions.iter().enumerate() {
            for (ai, a) in r.areas.iter().enumerate() {
                for o in &a.outlines {
                    let pts: Vec<P> = o.ring.iter().map(|&v| rd.v(v)).collect();
                    if pts.is_empty() {
                        continue;
                    }
                    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
                    for p in &pts {
                        for k in 0..2 {
                            lo[k] = lo[k].min(p[k]);
                            hi[k] = hi[k].max(p[k]);
                        }
                    }
                    if hi[0] < window[0] || lo[0] > window[2] || hi[1] < window[1] || lo[1] > window[3] {
                        continue;
                    }
                    let barea = (hi[0] - lo[0]) * (hi[1] - lo[1]);
                    rings.push((ri, ai, o.outer, pts, [lo[0], lo[1], hi[0], hi[1]], barea));
                }
            }
        }
        Self { rings, sea: rd.regions.iter().map(|r| r.kind == "sea").collect() }
    }

    /// Region covering `pt` (innermost outer ring by bounding-box area, holes of that area excluded).
    fn locate(&self, pt: P) -> Option<usize> {
        let mut best: Option<(usize, f64, usize)> = None;
        for (ri, ai, outer, pts, bb, barea) in &self.rings {
            if *outer
                && bb[0] <= pt[0]
                && pt[0] <= bb[2]
                && bb[1] <= pt[1]
                && pt[1] <= bb[3]
                && best.is_none_or(|b| *barea < b.1)
                && point_in_ring(pt, pts)
            {
                best = Some((*ri, *barea, *ai));
            }
        }
        let (ri, _, ai) = best?;
        for (r2, a2, outer, pts, bb, _) in &self.rings {
            if *r2 == ri && *a2 == ai && !*outer && bb[0] <= pt[0] && pt[0] <= bb[2] && bb[1] <= pt[1] && pt[1] <= bb[3] && point_in_ring(pt, pts) {
                return None;
            }
        }
        Some(ri)
    }

    fn is_sea(&self, pt: P) -> bool {
        self.locate(pt).is_some_and(|ri| self.sea[ri])
    }
}

impl Grid {
    /// Enlarge the grid to the theatre rectangle `new_rect` (port of the pathfinding part of work/exp3/exp3_india_big.py):
    /// old cells keep their content and move to their new position; new cells become sea runs where a sea polygon of
    /// `rd` covers the cell centre and outside runs elsewhere (header of the nearest old sea cell / the outside header,
    /// 0x01 bytes towards off-grid neighbours); old type-2 pieces inside a moved transition area become sea pieces;
    /// the a-words near the old border, the new cells and the moved transition areas are recomputed with the rules.
    pub fn extend(&mut self, new_rect: Rect, rd: &RegionData, transitions: &[TransitionArea]) -> Result<ExtendReport> {
        let dim = self.size;
        let (nmin, nmax) = ([new_rect[0][0] as f64, new_rect[0][1] as f64], [new_rect[1][0] as f64, new_rect[1][1] as f64]);
        // Corner ids resolve relative to their cell, while other vertices are absolute.
        // Rounding a fractional cell offset moves only the former and corrupts old polygons.
        ensure!(
            dim.is_finite() && dim > 0.0
                && [nmin[0], nmax[0]].iter().all(|x| {
                    let cells = (x - self.x0) / dim;
                    cells.is_finite() && (cells - cells.round()).abs() < 1e-9
                })
                && [nmin[1], nmax[1]].iter().all(|y| {
                    let cells = (y - self.y0) / dim;
                    cells.is_finite() && (cells - cells.round()).abs() < 1e-9
                }),
            "theatre rectangle {new_rect:?} is not aligned to the {dim}-unit pathfinding grid at ({}, {}); align the theatre bounds before extending",
            self.x0, self.y0
        );
        let ncols = ((nmax[0] - nmin[0]) / dim).round() as i64;
        let nrows = ((nmax[1] - nmin[1]) / dim).round() as i64;
        let offc = ((self.x0 - nmin[0]) / dim).round() as i64;
        let offr = ((self.y0 - nmin[1]) / dim).round() as i64;
        ensure!(
            offc >= 0 && offr >= 0 && offc + self.cols as i64 <= ncols && offr + self.rows as i64 <= nrows,
            "the new rectangle {new_rect:?} does not contain the current grid"
        );
        let (ncols, nrows, offc, offr) = (ncols as usize, nrows as usize, offc as usize, offr as usize);
        let sea_pid = self.paths.iter().position(|p| p.is_empty()).ok_or_else(|| anyhow!("no sea path"))? as u16;
        let sea_hdrs: Vec<(usize, usize, [u8; 8])> = (0..self.cells.len())
            .filter_map(|i| {
                let c = &self.cells[i];
                match c.empty {
                    Some((a, id)) if id == sea_pid && a & 15 == 1 && c.pieces.is_empty() => Some((i % self.cols, i / self.cols, c.header)),
                    _ => None,
                }
            })
            .collect();
        ensure!(!sea_hdrs.is_empty(), "the grid has no sea run cell to copy headers from");
        let nearest_sea_hdr = |c: usize, r: usize| -> [u8; 8] {
            let mut best = (i64::MAX, 0usize);
            for (k, (sc, sr, _)) in sea_hdrs.iter().enumerate() {
                let (dx, dy) = ((*sc + offc) as i64 - c as i64, (*sr + offr) as i64 - r as i64);
                let d = dx * dx + dy * dy;
                if d < best.0 {
                    best = (d, k);
                }
            }
            sea_hdrs[best.1].2
        };
        let loc = Locator::new(rd, [nmin[0], nmin[1], nmax[0], nmax[1]]);
        let moved: Vec<&TransitionArea> = transitions.iter().filter(|t| t.old != t.new).collect();
        let in_rect = |cx: f64, cy: f64, r: &Rect| r[0][0] as f64 <= cx && cx <= r[1][0] as f64 && r[0][1] as f64 <= cy && cy <= r[1][1] as f64;
        let mut report = ExtendReport { cols: ncols, rows: nrows, offset: [offc, offr], ..Default::default() };
        let mut cells: Vec<Option<Cell>> = vec![None; ncols * nrows];
        let old_cells = std::mem::take(&mut self.cells);
        for r in 0..nrows {
            for c in 0..ncols {
                let (cx, cy) = (nmin[0] + (c as f64 + 0.5) * dim, nmin[1] + (r as f64 + 0.5) * dim);
                let (oc, orow) = (c as i64 - offc as i64, r as i64 - offr as i64);
                let cell = if oc >= 0 && (oc as usize) < self.cols && orow >= 0 && (orow as usize) < self.rows {
                    let mut cell = old_cells[orow as usize * self.cols + oc as usize].clone();
                    if cell.pieces.iter().any(|p| p.a & 15 == 2) && moved.iter().any(|t| in_rect(cx, cy, &t.old)) {
                        for p in cell.pieces.iter_mut().filter(|p| p.a & 15 == 2) {
                            *p = Piece { a: (p.a & !0xf) | 1, b: ((sea_pid as u32) << 22) | (p.b & 0x3f_ffff) };
                            report.transition_pieces_to_sea += 1;
                        }
                    }
                    cell
                } else if loc.is_sea([cx, cy]) {
                    report.new_sea_cells += 1;
                    Cell { header: nearest_sea_hdr(c, r), pieces: Vec::new(), empty: Some((SEA_RUN_WORD, sea_pid)) }
                } else {
                    report.new_outside_cells += 1;
                    Cell { header: OUTSIDE_HEADER, pieces: Vec::new(), empty: Some((OUTSIDE_RUN_WORD, TRANSITION)) }
                };
                cells[r * ncols + c] = Some(cell);
            }
        }
        let mut cells: Vec<Cell> = cells.into_iter().map(|c| c.expect("every new cell is filled")).collect();
        // headers of new cells: 0x01 towards off-grid neighbours
        for r in 0..nrows {
            for c in 0..ncols {
                let (oc, orow) = (c as i64 - offc as i64, r as i64 - offr as i64);
                if oc >= 0 && (oc as usize) < self.cols && orow >= 0 && (orow as usize) < self.rows {
                    continue;
                }
                let cell = &mut cells[r * ncols + c];
                for (i, (dx, dy)) in crate::regionmap::pathsynth::rules::DIRV.iter().enumerate() {
                    let (nc, nr) = (c as i32 + dx, r as i32 + dy);
                    if nc < 0 || nr < 0 || nc >= ncols as i32 || nr >= nrows as i32 {
                        cell.header[i] = 0x01;
                    }
                }
            }
        }
        let (old_cols, old_rows) = (self.cols, self.rows);
        self.cells = cells;
        self.cols = ncols;
        self.rows = nrows;
        self.x0 = nmin[0];
        self.y0 = nmin[1];
        self.origin_index = [((nmin[0] + 1280.0) / dim) as u16, ((nmin[1] + 640.0) / dim) as u16];
        // a-words near the old border / the new cells / the moved transition areas
        let model = Model::from_grid(self)?;
        let view = View::build(&model);
        let near_change = |c: usize, r: usize| -> bool {
            let (oc, orow) = (c as i64 - offc as i64, r as i64 - offr as i64);
            if oc < 0 || orow < 0 || oc >= old_cols as i64 || orow >= old_rows as i64 {
                return true;
            }
            if oc <= 1 || orow <= 1 || oc >= old_cols as i64 - 2 || orow >= old_rows as i64 - 2 {
                return true;
            }
            let (cx, cy) = (nmin[0] + (c as f64 + 0.5) * dim, nmin[1] + (r as f64 + 0.5) * dim);
            moved.iter().any(|t| {
                t.old[0][0] as f64 - 3.0 <= cx && cx <= t.old[1][0] as f64 + 3.0 && t.old[0][1] as f64 - 3.0 <= cy && cy <= t.old[1][1] as f64 + 3.0
            })
        };
        for r in 0..nrows {
            for c in 0..ncols {
                if !near_change(c, r) {
                    continue;
                }
                let idx = r * ncols + c;
                let cell = &mut self.cells[idx];
                if let Some((a, pid)) = cell.empty {
                    if cell.pieces.is_empty() {
                        if a & 15 == 2 {
                            continue;
                        }
                        let na = view.word(idx, 0);
                        if na != a {
                            report.words_changed += 1;
                        }
                        cell.empty = Some((na, pid));
                        continue;
                    }
                }
                for (i, p) in cell.pieces.iter_mut().enumerate() {
                    if matches!(p.a & 15, 2 | 3) {
                        continue;
                    }
                    let na = view.word(idx, i);
                    if na != p.a {
                        report.words_changed += 1;
                    }
                    p.a = na;
                }
            }
        }
        Ok(report)
    }
}
