//! The decoded flag-word rules of pathfinding grid pieces (port of work/research/pf_decode.py):
//! `a = passable << 24 | unk << 4 | ptype`, with
//! * passable = 8-bit touch mask (bits E, NW, N, NE, SW, S, SE, W), cleared towards off-grid neighbours, 0 for types 2/3;
//! * unk = 5 nibbles S, W, intra, E, N of compatible-neighbour ranks.

use super::geometry::Pt;
use super::model::{MCell, Model};

/// Tolerance of the touch tests: 1e-4 map units in fixed point.
pub const EPS: f64 = 1e-4 * 1048576.0;
/// Minimum length of a shared boundary segment that counts as an overlap, in the two measured conventions: the
/// experiment scripts (and therefore the golden files) use 1e-4; the shipped vanilla file needs a finer one (a 4.3e-5
/// segment counts there, 1e-5 matches every vanilla India word).
pub const PY_LEN_EPS: f64 = 1e-4 * 1048576.0;
pub const GAME_LEN_EPS: f64 = 1e-5 * 1048576.0;

/// Piece types a piece of type `t` can link to (pf_decode.COMPAT).
pub fn compat(t: u8, q: u8) -> bool {
    match t {
        0 | 6 => matches!(q, 0 | 5 | 6 | 7),
        1 => matches!(q, 1 | 4 | 7),
        7 => matches!(q, 0 | 1 | 6 | 7),
        4 => matches!(q, 1 | 4 | 5),
        5 => matches!(q, 0 | 4 | 5 | 6),
        _ => false,
    }
}

pub fn has_compat(t: u8) -> bool {
    (0..16u8).any(|q| compat(t, q))
}

/// direction bit order of `passable` / the header: E, NW, N, NE, SW, S, SE, W
pub const DIRV: [(i32, i32); 8] = [(1, 0), (-1, 1), (0, 1), (1, 1), (-1, -1), (0, -1), (1, -1), (-1, 0)];

/// edge index order used by the interval tables
pub const S: usize = 0;
pub const W: usize = 1;
pub const E: usize = 2;
pub const N: usize = 3;
const NIB: [u32; 4] = [0, 1, 3, 4];
const OPP: [usize; 4] = [N, E, W, S];
const OFF: [(i32, i32); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];

#[derive(Clone, Debug)]
pub struct VPiece {
    pub ptype: u8,
    pub poly: Vec<Pt>,
    /// maximal boundary intervals on the cell edges S, W, E, N (coordinate along the edge, relative fixed)
    pub iv: [Vec<(i32, i32)>; 4],
    /// touch flags in the `passable` bit order
    pub f: [bool; 8],
}

pub struct View {
    pub cols: usize,
    pub rows: usize,
    pub len_eps: f64,
    pub cells: Vec<Vec<VPiece>>,
}

fn near(a: i32, b: i32) -> bool {
    ((a as f64) - (b as f64)).abs() < EPS
}

fn edge_intervals(poly: &[Pt], e: usize, cx: i32, cy: i32, cs: i32, len: f64) -> Vec<(i32, i32)> {
    let n = poly.len();
    let mut out: Vec<(i32, i32)> = Vec::new();
    for i in 0..n {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        let (ok, a, b) = match e {
            S => (near(p[1], cy) && near(q[1], cy), p[0], q[0]),
            N => (near(p[1], cy + cs) && near(q[1], cy + cs), p[0], q[0]),
            W => (near(p[0], cx) && near(q[0], cx), p[1], q[1]),
            _ => (near(p[0], cx + cs) && near(q[0], cx + cs), p[1], q[1]),
        };
        if ok && ((a as f64) - (b as f64)).abs() > len {
            out.push((a.min(b), a.max(b)));
        }
    }
    out.sort();
    let mut m: Vec<(i32, i32)> = Vec::new();
    for (a, b) in out {
        match m.last_mut() {
            Some(l) if (a as f64) <= l.1 as f64 + len => l.1 = l.1.max(b),
            _ => m.push((a, b)),
        }
    }
    m
}

fn edge_flags(poly: &[Pt], cx: i32, cy: i32, cs: i32) -> [bool; 8] {
    let n = poly.len();
    let on_x = |v: i32| (0..n).any(|i| near(poly[i][0], v) && near(poly[(i + 1) % n][0], v));
    let on_y = |v: i32| (0..n).any(|i| near(poly[i][1], v) && near(poly[(i + 1) % n][1], v));
    let corner = |x: i32, y: i32| poly.iter().any(|p| near(p[0], x) && near(p[1], y));
    let (w, e, s, nn) = (on_x(cx), on_x(cx + cs), on_y(cy), on_y(cy + cs));
    // order: E, NW, N, NE, SW, S, SE, W
    [e, corner(cx, cy + cs), nn, corner(cx + cs, cy + cs), corner(cx, cy), s, corner(cx + cs, cy), w]
}

fn vpiece(poly: Vec<Pt>, ptype: u8, cx: i32, cy: i32, cs: i32, len: f64) -> VPiece {
    let iv = [S, W, E, N].map(|e| edge_intervals(&poly, e, cx, cy, cs, len));
    let f = edge_flags(&poly, cx, cy, cs);
    VPiece { ptype, poly, iv, f }
}

fn run_piece(a: u32, cx: i32, cy: i32, cs: i32) -> VPiece {
    VPiece {
        ptype: (a & 15) as u8,
        poly: vec![[cx, cy], [cx + cs, cy], [cx + cs, cy + cs], [cx, cy + cs]],
        iv: [vec![(cx, cx + cs)], vec![(cy, cy + cs)], vec![(cy, cy + cs)], vec![(cx, cx + cs)]],
        f: [true; 8],
    }
}

pub fn cell_view(m: &Model, idx: usize, len: f64) -> Vec<VPiece> {
    let (c, r) = (idx % m.cols, idx / m.cols);
    let (cx, cy) = (c as i32 * m.cs, r as i32 * m.cs);
    let cell: &MCell = &m.cells[idx];
    if cell.pieces.is_empty() {
        let a = cell.run.map(|x| x.0).unwrap_or(0);
        vec![run_piece(a, cx, cy, m.cs)]
    } else {
        cell.pieces.iter().map(|p| vpiece(p.pts.clone(), p.ptype(), cx, cy, m.cs, len)).collect()
    }
}

impl View {
    /// View with the experiment scripts' tolerance (what the golden files were generated with).
    pub fn build(m: &Model) -> Self {
        Self::build_with(m, PY_LEN_EPS)
    }

    pub fn build_with(m: &Model, len_eps: f64) -> Self {
        View {
            cols: m.cols,
            rows: m.rows,
            len_eps,
            cells: (0..m.cells.len()).map(|i| cell_view(m, i, len_eps)).collect(),
        }
    }

    fn passable(&self, c: usize, r: usize, b: &VPiece) -> u32 {
        if matches!(b.ptype, 2 | 3) {
            return 0;
        }
        let mut m = 0u32;
        for (i, &(dx, dy)) in DIRV.iter().enumerate() {
            let (nc, nr) = (c as i32 + dx, r as i32 + dy);
            if b.f[i] && nc >= 0 && nc < self.cols as i32 && nr >= 0 && nr < self.rows as i32 {
                m |= 1 << i;
            }
        }
        m
    }

    fn edge_nibble(&self, c: usize, r: usize, b: &VPiece, e: usize) -> u32 {
        if !has_compat(b.ptype) || b.iv[e].is_empty() {
            return 0;
        }
        let (nc, nr) = (c as i32 + OFF[e].0, r as i32 + OFF[e].1);
        if nc < 0 || nc >= self.cols as i32 || nr < 0 || nr >= self.rows as i32 {
            return 0;
        }
        let w = &self.cells[nr as usize * self.cols + nc as usize];
        let mut m = 0u32;
        let mut rk = 0;
        for q in w {
            if !compat(b.ptype, q.ptype) {
                continue;
            }
            if overlap(&b.iv[e], &q.iv[OPP[e]], self.len_eps) && rk < 4 {
                m |= 1 << rk;
            }
            rk += 1;
        }
        m
    }

    fn intra(&self, idx: usize, i: usize) -> u32 {
        let ps = &self.cells[idx];
        let b = &ps[i];
        if !has_compat(b.ptype) {
            return 0;
        }
        let mut m = 0u32;
        let mut rk = 0;
        for (j, q) in ps.iter().enumerate() {
            if !compat(b.ptype, q.ptype) {
                continue;
            }
            if j != i && shares_boundary(&b.poly, &q.poly, self.len_eps) && rk < 4 {
                m |= 1 << rk;
            }
            rk += 1;
        }
        m
    }

    /// `(passable, unk)` predicted for piece `i` of cell `idx`.
    pub fn predict(&self, idx: usize, i: usize) -> (u32, u32) {
        let (c, r) = (idx % self.cols, idx / self.cols);
        let b = &self.cells[idx][i];
        let pass = self.passable(c, r, b);
        let mut unk = 0u32;
        for e in [S, W, E, N] {
            unk |= self.edge_nibble(c, r, b, e) << (4 * NIB[e]);
        }
        unk |= self.intra(idx, i) << 8;
        (pass, unk)
    }

    /// Predicted `a` word of piece `i` of cell `idx`.
    pub fn word(&self, idx: usize, i: usize) -> u32 {
        let (pass, unk) = self.predict(idx, i);
        (pass << 24) | (unk << 4) | self.cells[idx][i].ptype as u32
    }
}

fn overlap(a: &[(i32, i32)], b: &[(i32, i32)], len: f64) -> bool {
    a.iter().any(|x| b.iter().any(|y| (x.1.min(y.1) as f64) - (x.0.max(y.0) as f64) > len))
}

/// True when the two closed rings share a boundary segment longer than `len` (fixed units).
pub fn shares_boundary(p: &[Pt], q: &[Pt], len: f64) -> bool {
    let (n, m) = (p.len(), q.len());
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        let (ax, ay, bx, by) = (a[0] as f64, a[1] as f64, b[0] as f64, b[1] as f64);
        let l = (bx - ax).hypot(by - ay);
        if l < len {
            continue;
        }
        let (ux, uy) = ((bx - ax) / l, (by - ay) / l);
        for j in 0..m {
            let (c, d) = (q[j], q[(j + 1) % m]);
            let (cx, cy, dx, dy) = (c[0] as f64, c[1] as f64, d[0] as f64, d[1] as f64);
            if ((cx - ax) * uy - (cy - ay) * ux).abs() > EPS || ((dx - ax) * uy - (dy - ay) * ux).abs() > EPS {
                continue;
            }
            let t1 = (cx - ax) * ux + (cy - ay) * uy;
            let t2 = (dx - ax) * ux + (dy - ay) * uy;
            if l.min(t1.max(t2)) - 0f64.max(t1.min(t2)) > len {
                return true;
            }
        }
    }
    false
}
