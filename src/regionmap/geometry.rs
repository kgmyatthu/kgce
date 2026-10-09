//! Plane geometry used by the campaign-map region editor: segment/rect overlap, point-in-ring tests and an
//! index-preserving ear-clipping triangulation. Coordinates are campaign world units, y pointing north.

/// Axis-aligned rectangle `[[min_x, min_y], [max_x, max_y]]` in campaign world units (shared by the region,
/// pathfinding, sea-grid and startpos slices).
pub type Rect = [[f32; 2]; 2];

pub type P = [f64; 2];

/// True when the segment `p`-`q` has a positive-length overlap with the interior of the rectangle `lo`..`hi`
/// (a segment that only touches a corner or runs along one rectangle edge does not count). This is the membership
/// rule of the regions.esf `query_info` cells.
pub fn segment_overlaps_rect(p: P, q: P, lo: P, hi: P) -> bool {
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (den, num) in [
        (-dx, p[0] - lo[0]),
        (dx, hi[0] - p[0]),
        (-dy, p[1] - lo[1]),
        (dy, hi[1] - p[1]),
    ] {
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
    if t0 >= t1 {
        return false;
    }
    let c0 = [p[0] + dx * t0, p[1] + dy * t0];
    let c1 = [p[0] + dx * t1, p[1] + dy * t1];
    let on_same_edge = (c0[0] == lo[0] && c1[0] == lo[0])
        || (c0[0] == hi[0] && c1[0] == hi[0])
        || (c0[1] == lo[1] && c1[1] == lo[1])
        || (c0[1] == hi[1] && c1[1] == hi[1]);
    !on_same_edge
}

/// Even-odd point-in-polygon test.
pub fn point_in_ring(pt: P, ring: &[P]) -> bool {
    let mut inside = false;
    let n = ring.len();
    for i in 0..n {
        let [x1, y1] = ring[i];
        let [x2, y2] = ring[(i + 1) % n];
        if (y1 > pt[1]) != (y2 > pt[1]) {
            let xi = x1 + (pt[1] - y1) * (x2 - x1) / (y2 - y1);
            if xi > pt[0] {
                inside = !inside;
            }
        }
    }
    inside
}

/// Distance from `pt` to the segment `a`-`b`.
pub fn point_segment_distance(pt: P, a: P, b: P) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    };
    let (px, py) = (a[0] + t * dx - pt[0], a[1] + t * dy - pt[1]);
    (px * px + py * py).sqrt()
}

/// Twice the signed area (> 0 for counter-clockwise rings, y up).
pub fn signed_area2(ring: &[P]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum()
}

pub fn polygon_area(ring: &[P]) -> f64 {
    signed_area2(ring).abs() / 2.0
}

fn cross(o: P, a: P, b: P) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn point_in_triangle(p: P, a: P, b: P, c: P) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(b, c, p);
    let d3 = cross(c, a, p);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

fn segments_cross(p: P, q: P, r: P, s: P) -> bool {
    let d1 = cross(r, s, p);
    let d2 = cross(r, s, q);
    let d3 = cross(p, q, r);
    let d4 = cross(p, q, s);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// True when two non-adjacent edges of the closed ring properly cross (O(n²); rings here have < 1000 vertices).
pub fn ring_self_intersects(ring: &[P]) -> bool {
    let n = ring.len();
    for i in 0..n {
        for j in i + 2..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            if segments_cross(ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n]) {
                return true;
            }
        }
    }
    false
}

/// Ear-clipping triangulation of a simple ring. Returns triangles as indices into `ring`, wound clockwise
/// (the winding regions.esf uses for its `faces`). Degenerate rings yield no triangles.
pub fn earcut_indices(ring: &[P]) -> Vec<[usize; 3]> {
    let n = ring.len();
    if n < 3 {
        return Vec::new();
    }
    // process in clockwise order so that convex corners have a negative cross product
    let ccw = signed_area2(ring) > 0.0;
    let mut order: Vec<usize> = (0..n).collect();
    if ccw {
        order.reverse();
    }
    let mut tris = Vec::with_capacity(n - 2);
    let mut guard = 0usize;
    while order.len() > 3 && guard < 4 * n * n {
        guard += 1;
        let m = order.len();
        let mut clipped = false;
        for i in 0..m {
            let (ia, ib, ic) = (order[(i + m - 1) % m], order[i], order[(i + 1) % m]);
            let (a, b, c) = (ring[ia], ring[ib], ring[ic]);
            let cr = cross(a, b, c);
            if cr >= 0.0 {
                continue; // reflex or collinear corner
            }
            let blocked = order.iter().any(|&j| {
                j != ia
                    && j != ib
                    && j != ic
                    && ring[j] != a
                    && ring[j] != b
                    && ring[j] != c
                    && point_in_triangle(ring[j], a, b, c)
            });
            if blocked {
                continue;
            }
            tris.push([ia, ib, ic]);
            order.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            // no strict ear left (collinear runs): drop the flattest corner to make progress
            let m = order.len();
            let worst = (0..m)
                .min_by(|&i, &j| {
                    let ci = cross(ring[order[(i + m - 1) % m]], ring[order[i]], ring[order[(i + 1) % m]]).abs();
                    let cj = cross(ring[order[(j + m - 1) % m]], ring[order[j]], ring[order[(j + 1) % m]]).abs();
                    ci.partial_cmp(&cj).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            order.remove(worst);
        }
    }
    if order.len() == 3 {
        let (a, b, c) = (ring[order[0]], ring[order[1]], ring[order[2]]);
        if cross(a, b, c) != 0.0 {
            tris.push([order[0], order[1], order[2]]);
        }
    }
    tris
}

/// Ear-clipping triangulation of a polygon with holes (`rings[0]` is the outer ring, the rest are holes; any
/// winding). Returns clockwise triangles as indices into the concatenation of `rings` (the winding of regions.esf
/// `faces`); zero-area triangles are dropped. Uses the mapbox-earcut port of `geo`, like the vanilla tools.
pub fn earcut_with_holes(rings: &[Vec<P>]) -> Vec<[usize; 3]> {
    use geo::TriangulateEarcut;
    if rings.is_empty() || rings[0].len() < 3 {
        return Vec::new();
    }
    let line = |r: &Vec<P>| geo::LineString::from(r.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>());
    let polygon = geo::Polygon::new(line(&rings[0]), rings[1..].iter().filter(|r| r.len() >= 3).map(line).collect());
    // `geo` flattens the CLOSED rings (a repeated first vertex ends each open ring): map every flat slot back to
    // its position in the concatenation of the rings
    let kept: Vec<&Vec<P>> = std::iter::once(&rings[0]).chain(rings[1..].iter().filter(|r| r.len() >= 3)).collect();
    let mut slot: Vec<usize> = Vec::new();
    let mut base = 0usize;
    for (j, r) in kept.iter().enumerate() {
        for k in 0..polygon_ring_len(&polygon, j) {
            slot.push(base + k % r.len());
        }
        base += r.len();
    }
    let raw = polygon.earcut_triangles_raw();
    let flat: Vec<P> = rings.iter().filter(|r| r.len() >= 3).flat_map(|r| r.iter().copied()).collect();
    let mut out = Vec::new();
    for t in raw.triangle_indices.chunks_exact(3) {
        let tri = [slot[t[0]], slot[t[1]], slot[t[2]]];
        let (a, b, c) = (flat[tri[0]], flat[tri[1]], flat[tri[2]]);
        let s = cross(a, b, c);
        if s == 0.0 {
            continue;
        }
        out.push(if s > 0.0 { [tri[0], tri[2], tri[1]] } else { tri });
    }
    out
}

fn polygon_ring_len(polygon: &geo::Polygon<f64>, j: usize) -> usize {
    if j == 0 {
        polygon.exterior().0.len()
    } else {
        polygon.interiors()[j - 1].0.len()
    }
}

/// Pole of inaccessibility (the point inside the polygon farthest from its outline) by the polylabel cell
/// subdivision. `rings[0]` is the outer ring, the rest holes. Returns the point and its distance to the outline.
pub fn polylabel(rings: &[Vec<P>], precision: f64) -> (P, f64) {
    use std::collections::BinaryHeap;

    struct Cell {
        c: P,
        h: f64,
        d: f64,
        max: f64,
    }
    impl PartialEq for Cell {
        fn eq(&self, o: &Self) -> bool {
            self.max == o.max
        }
    }
    impl Eq for Cell {}
    impl PartialOrd for Cell {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Cell {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.max.partial_cmp(&o.max).unwrap_or(std::cmp::Ordering::Equal)
        }
    }
    // signed distance: positive inside (even-odd over all rings), negative outside
    let dist = |p: P| -> f64 {
        let mut inside = false;
        let mut best = f64::MAX;
        for ring in rings {
            let n = ring.len();
            for k in 0..n {
                let (a, b) = (ring[k], ring[(k + 1) % n]);
                if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
                    inside = !inside;
                }
                best = best.min(point_segment_distance(p, a, b));
            }
        }
        if inside { best } else { -best }
    };
    let cell = |c: P, h: f64| -> Cell {
        let d = dist(c);
        Cell { c, h, d, max: d + h * std::f64::consts::SQRT_2 }
    };
    let Some(outer) = rings.first().filter(|r| r.len() >= 3) else {
        return ([0.0, 0.0], 0.0);
    };
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in outer {
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
    let size = w.min(h);
    if size == 0.0 {
        return (lo, 0.0);
    }
    let mut queue = BinaryHeap::new();
    let mut x = lo[0];
    while x < hi[0] {
        let mut y = lo[1];
        while y < hi[1] {
            queue.push(cell([x + size / 2.0, y + size / 2.0], size / 2.0));
            y += size;
        }
        x += size;
    }
    // centroid of the outer ring as the first guess
    let a2 = signed_area2(outer);
    let mut best = cell([lo[0] + w / 2.0, lo[1] + h / 2.0], 0.0);
    if a2 != 0.0 {
        let n = outer.len();
        let (mut cx, mut cy) = (0.0, 0.0);
        for k in 0..n {
            let (p, q) = (outer[k], outer[(k + 1) % n]);
            let f = p[0] * q[1] - q[0] * p[1];
            cx += (p[0] + q[0]) * f;
            cy += (p[1] + q[1]) * f;
        }
        let centroid = cell([cx / (3.0 * a2), cy / (3.0 * a2)], 0.0);
        if centroid.d > best.d {
            best = centroid;
        }
    }
    while let Some(top) = queue.pop() {
        if top.d > best.d {
            best = Cell { c: top.c, h: top.h, d: top.d, max: top.max };
        }
        if top.max - best.d <= precision {
            continue;
        }
        let h2 = top.h / 2.0;
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            queue.push(cell([top.c[0] + dx * h2, top.c[1] + dy * h2], h2));
        }
    }
    (best.c, best.d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_overlap_excludes_touches_and_edge_runs() {
        let (lo, hi) = ([0.0, 0.0], [10.0, 10.0]);
        assert!(segment_overlaps_rect([-1.0, 5.0], [5.0, 5.0], lo, hi));
        assert!(segment_overlaps_rect([2.0, 2.0], [3.0, 3.0], lo, hi));
        assert!(!segment_overlaps_rect([0.0, -5.0], [0.0, 20.0], lo, hi), "runs along the left edge");
        assert!(segment_overlaps_rect([-5.0, 15.0], [15.0, -5.0], lo, hi), "diagonal through two corners");
        assert!(!segment_overlaps_rect([10.0, 10.0], [20.0, 20.0], lo, hi), "touches a corner only");
        assert!(!segment_overlaps_rect([11.0, 0.0], [11.0, 10.0], lo, hi));
    }

    #[test]
    fn crossed_rings_are_detected() {
        let square: Vec<P> = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let bow_tie: Vec<P> = vec![[0.0, 0.0], [4.0, 4.0], [4.0, 0.0], [0.0, 4.0]];
        assert!(!ring_self_intersects(&square));
        assert!(ring_self_intersects(&bow_tie));
    }

    #[test]
    fn earcut_covers_a_concave_ring_clockwise() {
        // an L shape, counter-clockwise input
        let ring: Vec<P> = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [1.0, 1.0], [1.0, 4.0], [0.0, 4.0]];
        let tris = earcut_indices(&ring);
        assert_eq!(tris.len(), ring.len() - 2);
        let mut area = 0.0;
        for t in &tris {
            let c = cross(ring[t[0]], ring[t[1]], ring[t[2]]);
            assert!(c < 0.0, "triangle must be clockwise");
            area += -c / 2.0;
        }
        assert!((area - polygon_area(&ring)).abs() < 1e-9);
    }
    #[test]
    fn earcut_with_a_hole_covers_exactly_the_ring_minus_the_hole() {
        let outer: Vec<P> = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole: Vec<P> = vec![[3.0, 3.0], [3.0, 6.0], [6.0, 6.0], [6.0, 3.0]]; // clockwise
        let rings = vec![outer.clone(), hole.clone()];
        let flat: Vec<P> = outer.iter().chain(hole.iter()).copied().collect();
        let tris = earcut_with_holes(&rings);
        let mut area = 0.0;
        for t in &tris {
            let c = cross(flat[t[0]], flat[t[1]], flat[t[2]]);
            assert!(c < 0.0, "triangles are clockwise");
            area += -c / 2.0;
        }
        assert!((area - 91.0).abs() < 1e-9, "{area}");
    }

    #[test]
    fn polylabel_finds_the_middle_of_a_ring_with_a_hole() {
        let outer: Vec<P> = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]];
        let (p, d) = polylabel(&[outer.clone()], 1e-4);
        assert!((d - 5.0).abs() < 1e-3 && (p[1] - 5.0).abs() < 1e-3, "{p:?} {d}");
        // a hole in the middle pushes the pole to one of the free halves
        let hole: Vec<P> = vec![[8.0, 3.0], [8.0, 7.0], [12.0, 7.0], [12.0, 3.0]];
        let (p, d) = polylabel(&[outer, hole], 1e-4);
        assert!(d < 5.0 && d > 3.5 && (p[0] < 8.0 || p[0] > 12.0), "{p:?} {d}");
    }
}
