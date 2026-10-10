//! Plane helpers for the atlas regions migration: exact rectilinear unions, polygon containment with the
//! stock "nested hole" semantics and a uniform-grid point index. Coordinates are campaign world units.

use crate::regionmap::geometry::{point_in_ring, point_segment_distance, polylabel, signed_area2};
use anyhow::{Result, bail, ensure};
use std::collections::BTreeSet;

pub type P = [f64; 2];
pub type Rect64 = [P; 2];

/// Hashable key of a point; `-0.0` and `0.0` are the same key (as they are equal floats).
pub fn norm_bits(v: f64) -> u64 {
    if v == 0.0 { 0 } else { v.to_bits() }
}

pub fn point_key(p: P) -> (u64, u64) {
    (norm_bits(p[0]), norm_bits(p[1]))
}

pub fn widen(p: [f32; 2]) -> P {
    [p[0] as f64, p[1] as f64]
}

pub fn ring_area(ring: &[P]) -> f64 {
    signed_area2(ring).abs() / 2.0
}

pub fn bbox_of(ring: &[P]) -> Rect64 {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in ring {
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    [lo, hi]
}

pub fn rect_contains(r: Rect64, p: P) -> bool {
    r[0][0] <= p[0] && p[0] <= r[1][0] && r[0][1] <= p[1] && p[1] <= r[1][1]
}

pub fn rects_overlap_area(a: Rect64, b: Rect64) -> f64 {
    let w = (a[1][0].min(b[1][0]) - a[0][0].max(b[0][0])).max(0.0);
    let h = (a[1][1].min(b[1][1]) - a[0][1].max(b[0][1])).max(0.0);
    w * h
}

pub fn rect_distance(a: Rect64, b: Rect64) -> f64 {
    let dx = (a[0][0] - b[1][0]).max(b[0][0] - a[1][0]).max(0.0);
    let dy = (a[0][1] - b[1][1]).max(b[0][1] - a[1][1]).max(0.0);
    dx.hypot(dy)
}

/// Pole of inaccessibility of one ring (a point strictly inside it).
pub fn ring_interior_point(ring: &[P]) -> P {
    let bb = bbox_of(ring);
    let diag = (bb[1][0] - bb[0][0]).hypot(bb[1][1] - bb[0][1]);
    polylabel(&[ring.to_vec()], (diag * 1e-4).max(1e-9)).0
}

// ---------------------------------------------------------------------------------------------
// Rectilinear union of axis-aligned rectangles
// ---------------------------------------------------------------------------------------------

/// Union of axis-aligned rectangles on the compressed coordinate grid (exact, no floating point arithmetic on
/// the coordinates themselves).
#[derive(Clone, Debug)]
pub struct RectUnion {
    xs: Vec<f64>,
    ys: Vec<f64>,
    covered: Vec<Vec<bool>>,
}

fn sorted_unique(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

impl RectUnion {
    pub fn new(rects: &[Rect64]) -> Self {
        let xs = sorted_unique(rects.iter().flat_map(|r| [r[0][0], r[1][0]]).collect());
        let ys = sorted_unique(rects.iter().flat_map(|r| [r[0][1], r[1][1]]).collect());
        let mut covered = vec![vec![false; ys.len().saturating_sub(1)]; xs.len().saturating_sub(1)];
        for r in rects {
            for (i, row) in covered.iter_mut().enumerate() {
                if xs[i] >= r[0][0] && xs[i + 1] <= r[1][0] {
                    for (j, c) in row.iter_mut().enumerate() {
                        if ys[j] >= r[0][1] && ys[j + 1] <= r[1][1] {
                            *c = true;
                        }
                    }
                }
            }
        }
        Self { xs, ys, covered }
    }

    #[cfg(test)]
    pub fn area(&self) -> f64 {
        let mut a = 0.0;
        for (i, row) in self.covered.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                if *c {
                    a += (self.xs[i + 1] - self.xs[i]) * (self.ys[j + 1] - self.ys[j]);
                }
            }
        }
        a
    }

    fn cell(&self, i: isize, j: isize) -> bool {
        i >= 0
            && j >= 0
            && (i as usize) < self.covered.len()
            && (j as usize) < self.covered.first().map_or(0, |r| r.len())
            && self.covered[i as usize][j as usize]
    }

    /// True when the closed point lies in the union.
    pub fn contains_point(&self, p: P) -> bool {
        for i in 0..self.covered.len() {
            if p[0] < self.xs[i] || p[0] > self.xs[i + 1] {
                continue;
            }
            for j in 0..self.covered[i].len() {
                if self.covered[i][j] && p[1] >= self.ys[j] && p[1] <= self.ys[j + 1] {
                    return true;
                }
            }
        }
        false
    }

    /// True when the whole closed rectangle `b` lies inside the union.
    pub fn covers_rect(&self, b: Rect64) -> bool {
        let mut xs: Vec<f64> = self.xs.iter().copied().filter(|x| *x > b[0][0] && *x < b[1][0]).collect();
        xs.extend([b[0][0], b[1][0]]);
        let mut ys: Vec<f64> = self.ys.iter().copied().filter(|y| *y > b[0][1] && *y < b[1][1]).collect();
        ys.extend([b[0][1], b[1][1]]);
        let xs = sorted_unique(xs);
        let ys = sorted_unique(ys);
        for i in 0..xs.len().saturating_sub(1) {
            for j in 0..ys.len().saturating_sub(1) {
                let c = [(xs[i] + xs[i + 1]) / 2.0, (ys[j] + ys[j + 1]) / 2.0];
                if !self.contains_point(c) {
                    return false;
                }
            }
        }
        true
    }

    /// Boundary rings: exterior rings counter-clockwise, holes clockwise, interior on the left, collinear
    /// vertices removed. Errors on a pinch point (two rectangles touching only at a corner).
    pub fn rings(&self) -> Result<Vec<Vec<P>>> {
        use std::collections::BTreeMap;
        ensure!(!self.covered.is_empty() && !self.covered[0].is_empty(), "empty rectangle union");
        let key = |p: P| point_key(p);
        let mut next: BTreeMap<(u64, u64), (P, P)> = BTreeMap::new();
        let mut add = |a: P, b: P| -> Result<()> {
            if next.insert(key(a), (a, b)).is_some() {
                bail!("rectangle union has a pinch point at {a:?}");
            }
            Ok(())
        };
        for i in 0..self.covered.len() as isize {
            for j in 0..self.covered[0].len() as isize {
                if !self.cell(i, j) {
                    continue;
                }
                let (x0, x1) = (self.xs[i as usize], self.xs[i as usize + 1]);
                let (y0, y1) = (self.ys[j as usize], self.ys[j as usize + 1]);
                if !self.cell(i, j - 1) {
                    add([x0, y0], [x1, y0])?;
                }
                if !self.cell(i + 1, j) {
                    add([x1, y0], [x1, y1])?;
                }
                if !self.cell(i, j + 1) {
                    add([x1, y1], [x0, y1])?;
                }
                if !self.cell(i - 1, j) {
                    add([x0, y1], [x0, y0])?;
                }
            }
        }
        // boundary edges of one cell run per unit cell side: merge consecutive collinear edges while chaining
        let mut used: BTreeSet<(u64, u64)> = BTreeSet::new();
        let mut rings: Vec<Vec<P>> = Vec::new();
        let starts: Vec<(u64, u64)> = next.keys().copied().collect();
        for s in starts {
            if used.contains(&s) {
                continue;
            }
            let mut ring: Vec<P> = Vec::new();
            let mut cur = s;
            loop {
                if !used.insert(cur) {
                    bail!("rectangle union boundary is not a simple cycle");
                }
                let (a, b) = next[&cur];
                ring.push(a);
                cur = key(b);
                if cur == s {
                    break;
                }
                ensure!(next.contains_key(&cur), "rectangle union boundary is open at {b:?}");
            }
            let ring = simplify_ring(&ring);
            ensure!(ring.len() >= 4, "degenerate rectangle union ring");
            rings.push(ring);
        }
        // deterministic order: by lowest (y, x) vertex; each ring starts at its own lowest (y, x) vertex
        let mut out: Vec<Vec<P>> = Vec::new();
        for ring in rings {
            let start = (0..ring.len())
                .min_by(|&a, &b| {
                    (ring[a][1], ring[a][0]).partial_cmp(&(ring[b][1], ring[b][0])).unwrap()
                })
                .unwrap();
            let rotated: Vec<P> = (0..ring.len()).map(|k| ring[(start + k) % ring.len()]).collect();
            out.push(rotated);
        }
        out.sort_by(|a, b| (a[0][1], a[0][0]).partial_cmp(&(b[0][1], b[0][0])).unwrap());
        Ok(out)
    }
}

/// Drop vertices whose turn is exactly zero (collinear triples).
pub fn simplify_ring(pts: &[P]) -> Vec<P> {
    let n = pts.len();
    let mut out = Vec::new();
    for i in 0..n {
        let (p0, p1, p2) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
        let cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]);
        if cross != 0.0 {
            out.push(p1);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Polygons
// ---------------------------------------------------------------------------------------------

/// Indices of the holes that are kept: a hole nested inside a larger hole is dropped (the stock data has this
/// defect and the stock meshes treat the outer hole as the void), zero-area holes are dropped.
pub fn kept_holes(holes: &[Vec<P>]) -> Vec<usize> {
    let boxes: Vec<Rect64> = holes.iter().map(|h| bbox_of(h)).collect();
    let areas: Vec<f64> = holes.iter().map(|h| ring_area(h)).collect();
    let mut keep = Vec::new();
    for i in 0..holes.len() {
        if areas[i] <= 0.0 {
            continue;
        }
        let mut rp: Option<P> = None;
        let mut nested = false;
        for j in 0..holes.len() {
            if j == i || areas[j] <= areas[i] {
                continue;
            }
            // representative point of hole `i` inside the larger hole `j`
            let p = *rp.get_or_insert_with(|| ring_interior_point(&holes[i]));
            if rect_contains(boxes[j], p) && point_in_ring(p, &holes[j]) {
                nested = true;
                break;
            }
        }
        if !nested {
            keep.push(i);
        }
    }
    keep
}

/// One area polygon: a single outer ring and its kept holes.
#[derive(Clone, Debug)]
pub struct AreaPoly {
    pub outer: Vec<P>,
    pub holes: Vec<Vec<P>>,
    pub bbox: Rect64,
    pub area: f64,
}

impl AreaPoly {
    pub fn new(outers: &[Vec<P>], holes: &[Vec<P>]) -> Result<Self> {
        ensure!(outers.len() == 1, "area with {} outer rings", outers.len());
        let kept_rings: Vec<Vec<P>> = kept_holes(holes).into_iter().map(|i| holes[i].clone()).collect();
        let area = ring_area(&outers[0]) - kept_rings.iter().map(|h| ring_area(h)).sum::<f64>();
        Ok(Self { bbox: bbox_of(&outers[0]), outer: outers[0].clone(), holes: kept_rings, area })
    }

    /// Inside the outer ring and in none of the kept holes (points on the boundary are arbitrary).
    pub fn contains(&self, p: P) -> bool {
        rect_contains(self.bbox, p)
            && point_in_ring(p, &self.outer)
            && !self.holes.iter().any(|h| point_in_ring(p, h))
    }

    pub fn boundary_distance(&self, p: P) -> f64 {
        let mut best = f64::INFINITY;
        for ring in std::iter::once(&self.outer).chain(self.holes.iter()) {
            let n = ring.len();
            for k in 0..n {
                best = best.min(point_segment_distance(p, ring[k], ring[(k + 1) % n]));
            }
        }
        best
    }

    /// Strictly inside (shapely `within` for a point), with a 1e-9 boundary tolerance.
    pub fn strictly_contains(&self, p: P) -> bool {
        self.contains(p) && self.boundary_distance(p) > 1e-9
    }
}

// ---------------------------------------------------------------------------------------------
// Uniform grid point index over rectangles
// ---------------------------------------------------------------------------------------------

pub struct GridIndex {
    lo: P,
    cell: P,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<u32>>,
    boxes: Vec<Rect64>,
}

impl GridIndex {
    pub fn new(world: Rect64, boxes: Vec<Rect64>) -> Self {
        let (nx, ny) = (64usize, 32usize);
        let cell = [(world[1][0] - world[0][0]) / nx as f64, (world[1][1] - world[0][1]) / ny as f64];
        let mut idx = Self { lo: world[0], cell, nx, ny, buckets: vec![Vec::new(); nx * ny], boxes: Vec::new() };
        for (id, b) in boxes.iter().enumerate() {
            let (x0, x1) = (idx.col(b[0][0]), idx.col(b[1][0]));
            let (y0, y1) = (idx.row(b[0][1]), idx.row(b[1][1]));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    idx.buckets[y * nx + x].push(id as u32);
                }
            }
        }
        idx.boxes = boxes;
        idx
    }

    fn col(&self, x: f64) -> usize {
        (((x - self.lo[0]) / self.cell[0]).floor().max(0.0) as usize).min(self.nx - 1)
    }

    fn row(&self, y: f64) -> usize {
        (((y - self.lo[1]) / self.cell[1]).floor().max(0.0) as usize).min(self.ny - 1)
    }

    /// Ids whose bounding box contains `p` (closed), ascending.
    pub fn at(&self, p: P) -> Vec<u32> {
        let b = &self.buckets[self.row(p[1]) * self.nx + self.col(p[0])];
        b.iter().copied().filter(|&id| rect_contains(self.boxes[id as usize], p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<P> {
        vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }

    #[test]
    fn overlapping_rectangles_union_to_one_ccw_ring_without_collinear_vertices() {
        let u = RectUnion::new(&[[[0.0, 0.0], [4.0, 2.0]], [[3.0, 1.0], [6.0, 5.0]]]);
        let rings = u.rings().unwrap();
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 8);
        assert!(signed_area2(&rings[0]) > 0.0);
        assert_eq!(ring_area(&rings[0]), 19.0);
        assert_eq!(u.area(), 19.0);
        assert!(u.covers_rect([[3.0, 1.0], [4.0, 2.0]]));
        assert!(u.covers_rect([[0.0, 0.0], [4.0, 2.0]]));
        // the notch between the two rectangles is not covered
        assert!(!u.covers_rect([[0.0, 0.0], [6.0, 5.0]]));
        assert!(u.contains_point([4.0, 1.0]));
        assert!(!u.contains_point([5.0, 0.5]));
    }

    #[test]
    fn rectangles_touching_only_at_a_corner_are_rejected() {
        let u = RectUnion::new(&[[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]]);
        assert!(u.rings().is_err());
    }

    #[test]
    fn a_hole_nested_in_a_larger_hole_is_dropped() {
        let holes = vec![square(0.0, 0.0, 10.0, 10.0), square(2.0, 2.0, 4.0, 4.0), square(20.0, 20.0, 22.0, 22.0)];
        assert_eq!(kept_holes(&holes), vec![0, 2]);
        let poly = AreaPoly::new(&[square(-5.0, -5.0, 30.0, 30.0)], &holes).unwrap();
        assert_eq!(poly.area, 35.0 * 35.0 - 100.0 - 4.0);
        assert!(!poly.contains([3.0, 3.0]), "points of the dropped nested hole stay inside the void of the large hole");
        assert!(poly.contains([15.0, 15.0]));
        assert!(!poly.strictly_contains([-5.0, 0.0]));
    }

    #[test]
    fn area_with_two_outer_rings_is_rejected() {
        assert!(AreaPoly::new(&[square(0.0, 0.0, 1.0, 1.0), square(2.0, 2.0, 3.0, 3.0)], &[]).is_err());
    }

    #[test]
    fn grid_index_only_returns_boxes_containing_the_point() {
        let world = [[-100.0, -50.0], [100.0, 50.0]];
        let idx = GridIndex::new(world, vec![[[0.0, 0.0], [10.0, 10.0]], [[-100.0, -50.0], [100.0, 50.0]], [[20.0, 20.0], [30.0, 30.0]]]);
        assert_eq!(idx.at([5.0, 5.0]), vec![0, 1]);
        assert_eq!(idx.at([25.0, 25.0]), vec![1, 2]);
        assert_eq!(idx.at([100.0, 50.0]), vec![1]);
    }
}
