//! Decoded, editable form of a pathfinding grid: every cell with its pieces as fixed-point polygons (coordinates
//! relative to the grid origin) and the conversion to/from `Grid` (vertex table, polyline table, run-length cells).

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, anyhow, ensure};

use super::geometry::Pt;
use crate::regionmap::pathfinding::{Cell, Grid, Piece, TRANSITION, fixed_of};

pub const OUTSIDE: u16 = TRANSITION;
/// vertex-table entries 0..4 (corner ids) are placeholders
pub const DUMMY: [i32; 2] = [-2147483647, -2147483647];

#[derive(Clone, Debug)]
pub struct MPiece {
    pub a: u32,
    pub pid: u16,
    pub gbit: u8,
    /// open ring, CCW, coordinates relative to the grid origin (fixed 12.20)
    pub pts: Vec<Pt>,
    /// index of this piece in the cell before the synthesis (None for new pieces)
    pub src: Option<usize>,
}

impl MPiece {
    pub fn ptype(&self) -> u8 {
        (self.a & 15) as u8
    }
}

#[derive(Clone, Debug)]
pub struct MCell {
    pub hdr: [u8; 8],
    pub pieces: Vec<MPiece>,
    /// `(a, pid)` of a cell without pieces
    pub run: Option<(u32, u16)>,
}

impl MCell {
    /// Piece types of the cell (run cells: the run's type).
    pub fn types(&self) -> Vec<u8> {
        if self.pieces.is_empty() {
            self.run.map(|(a, _)| vec![(a & 15) as u8]).unwrap_or_default()
        } else {
            self.pieces.iter().map(|p| p.ptype()).collect()
        }
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub cols: usize,
    pub rows: usize,
    /// cell size, fixed point
    pub cs: i32,
    /// grid origin, fixed point (absolute)
    pub origin: [i32; 2],
    pub cells: Vec<MCell>,
}

impl Model {
    pub fn corner(&self, c: usize, r: usize) -> [Pt; 4] {
        let (x, y) = (c as i32 * self.cs, r as i32 * self.cs);
        // ids 0..=3 = BL, TL, BR, TR
        [[x, y], [x, y + self.cs], [x + self.cs, y], [x + self.cs, y + self.cs]]
    }

    pub fn from_grid(g: &Grid) -> Result<Self> {
        let cs = fixed_of(g.size);
        let origin = [fixed_of(g.x0), fixed_of(g.y0)];
        let mut cells = Vec::with_capacity(g.cells.len());
        for (idx, cell) in g.cells.iter().enumerate() {
            let (c, r) = (idx % g.cols, idx / g.cols);
            let corners = [
                [c as i32 * cs, r as i32 * cs],
                [c as i32 * cs, (r as i32 + 1) * cs],
                [(c as i32 + 1) * cs, r as i32 * cs],
                [(c as i32 + 1) * cs, (r as i32 + 1) * cs],
            ];
            let mut pieces = Vec::with_capacity(cell.pieces.len());
            for (k, p) in cell.pieces.iter().enumerate() {
                let ids = g
                    .polylines
                    .get(&p.polyline())
                    .ok_or_else(|| anyhow!("cell {idx}: polyline offset {} missing", p.polyline()))?;
                let mut pts = Vec::with_capacity(ids.len());
                for &v in ids {
                    pts.push(if v < 4 {
                        corners[v as usize]
                    } else {
                        let a = g.vertices.get(v as usize).ok_or_else(|| anyhow!("cell {idx}: vertex {v} outside the table"))?;
                        [a[0].wrapping_sub(origin[0]), a[1].wrapping_sub(origin[1])]
                    });
                }
                pieces.push(MPiece {
                    a: p.a,
                    pid: p.path_id(),
                    gbit: ((p.b >> 21) & 1) as u8,
                    pts,
                    src: Some(k),
                });
            }
            cells.push(MCell {
                hdr: cell.header,
                pieces,
                run: cell.empty,
            });
        }
        Ok(Self {
            cols: g.cols,
            rows: g.rows,
            cs,
            origin,
            cells,
        })
    }

    /// Rebuild the vertex / polyline tables and the cells of `g` from this model (the table layout of the shipped
    /// files: vertices = exactly the used non-corner points sorted by (y, x); polylines = the unit square followed by
    /// one entry per piece in raster order).
    pub fn write_into(&self, g: &mut Grid) -> Result<()> {
        ensure!(self.cols == g.cols && self.rows == g.rows, "model and grid dimensions differ");
        let mut used: Vec<Pt> = Vec::new();
        for (idx, cell) in self.cells.iter().enumerate() {
            let (c, r) = (idx % self.cols, idx / self.cols);
            let corners = self.corner(c, r);
            for p in &cell.pieces {
                for pt in &p.pts {
                    if !corners.contains(pt) {
                        used.push(*pt);
                    }
                }
            }
        }
        used.sort_by_key(|p| (p[1], p[0]));
        used.dedup();
        let vid: HashMap<Pt, u32> = used.iter().enumerate().map(|(i, p)| (*p, 4 + i as u32)).collect();
        let mut vertices: Vec<[i32; 2]> = Vec::with_capacity(4 + used.len());
        for k in 0..4 {
            vertices.push(g.vertices.get(k).copied().unwrap_or(DUMMY));
        }
        vertices.extend(used.iter().map(|p| [p[0].wrapping_add(self.origin[0]), p[1].wrapping_add(self.origin[1])]));
        let mut polylines: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        polylines.insert(0, vec![0, 2, 3, 1]);
        let mut next = 5usize;
        let mut cells = Vec::with_capacity(self.cells.len());
        for (idx, cell) in self.cells.iter().enumerate() {
            let (c, r) = (idx % self.cols, idx / self.cols);
            let corners = self.corner(c, r);
            let mut pieces = Vec::with_capacity(cell.pieces.len());
            for p in &cell.pieces {
                let ids: Vec<u32> = p
                    .pts
                    .iter()
                    .map(|pt| corners.iter().position(|q| q == pt).map(|k| k as u32).unwrap_or_else(|| vid[pt]))
                    .collect();
                let off = next;
                next += 1 + ids.len();
                polylines.insert(off, ids);
                pieces.push(Piece {
                    a: p.a,
                    b: ((p.pid as u32) << 22) | ((p.gbit as u32) << 21) | off as u32,
                });
            }
            cells.push(Cell {
                header: cell.hdr,
                pieces,
                empty: if cell.pieces.is_empty() { cell.run } else { None },
            });
        }
        g.vertices = vertices;
        g.polylines = polylines;
        g.cells = cells;
        Ok(())
    }
}

/// True for the run `a` words of a land / sea / outside run cell.
pub fn is_land_run(a: u32) -> bool {
    a & 15 == 0
}
