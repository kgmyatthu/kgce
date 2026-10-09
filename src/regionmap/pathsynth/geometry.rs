//! Integer (12.20 fixed point, coordinates relative to the grid origin) polygon geometry for the pathfinding synthesis:
//! boolean operations on `i_overlay`'s integer overlay, buffering of polylines, rectangle clipping and noding.
//!
//! A `Shape` is `[outer ring (CCW), hole rings (CW)..]`; every ring is open (the closing edge is implicit).

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay::{Overlay, ShapeType};
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::i_float::int::point::IntPoint;

pub type Pt = [i32; 2];
pub type Contour = Vec<Pt>;
pub type Shape = Vec<Contour>;
pub type Shapes = Vec<Shape>;

/// Fixed-point units per map unit.
pub const FIX: f64 = 1048576.0;

/// Twice the signed area of a ring in fixed units² (> 0 for CCW).
pub fn area2(c: &[Pt]) -> i128 {
    let n = c.len();
    let mut s = 0i128;
    for i in 0..n {
        let p = c[i];
        let q = c[(i + 1) % n];
        s += p[0] as i128 * q[1] as i128 - q[0] as i128 * p[1] as i128;
    }
    s
}

/// Area of a shape set in map units².
pub fn area(shapes: &Shapes) -> f64 {
    let mut s = 0i128;
    for sh in shapes {
        for (k, c) in sh.iter().enumerate() {
            let a = area2(c).abs();
            if k == 0 { s += a } else { s -= a }
        }
    }
    s as f64 / 2.0 / (FIX * FIX)
}

pub fn shape_area(sh: &Shape) -> f64 {
    area(&vec![sh.clone()])
}

pub fn bbox_of(c: &[Pt]) -> (Pt, Pt) {
    let mut lo = [i32::MAX; 2];
    let mut hi = [i32::MIN; 2];
    for p in c {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

pub fn shapes_bbox(shapes: &Shapes) -> Option<(Pt, Pt)> {
    let mut lo = [i32::MAX; 2];
    let mut hi = [i32::MIN; 2];
    let mut any = false;
    for sh in shapes {
        if let Some(c) = sh.first() {
            let (l, h) = bbox_of(c);
            for k in 0..2 {
                lo[k] = lo[k].min(l[k]);
                hi[k] = hi[k].max(h[k]);
            }
            any = true;
        }
    }
    any.then_some((lo, hi))
}

fn to_int(c: &[Pt]) -> impl Iterator<Item = IntPoint> + '_ {
    c.iter().map(|p| IntPoint::new(p[0], p[1]))
}

fn from_int(shapes: Vec<Vec<Vec<IntPoint>>>) -> Shapes {
    let mut out = Shapes::new();
    for sh in shapes {
        let mut contours: Shape = Vec::with_capacity(sh.len());
        for (k, c) in sh.into_iter().enumerate() {
            let mut c: Contour = c.into_iter().map(|p| [p.x, p.y]).collect();
            if c.len() < 3 {
                if k == 0 {
                    contours.clear();
                    break;
                }
                continue;
            }
            let a = area2(&c);
            if (k == 0 && a < 0) || (k > 0 && a > 0) {
                c.reverse();
            }
            contours.push(c);
        }
        if !contours.is_empty() {
            out.push(contours);
        }
    }
    out
}

fn run(subj: &[&Shapes], clip: &[&Shapes], rule: OverlayRule) -> Shapes {
    let cap: usize = subj.iter().chain(clip.iter()).map(|s| s.iter().map(|sh| sh.iter().map(|c| c.len()).sum::<usize>()).sum::<usize>()).sum();
    let mut ov = Overlay::new(cap);
    for s in subj {
        for sh in s.iter() {
            for c in sh {
                ov.add_path_iter(to_int(c), ShapeType::Subject);
            }
        }
    }
    for s in clip {
        for sh in s.iter() {
            for c in sh {
                ov.add_path_iter(to_int(c), ShapeType::Clip);
            }
        }
    }
    from_int(ov.overlay(rule, FillRule::NonZero))
}

pub fn union_many(list: &[&Shapes]) -> Shapes {
    run(list, &[], OverlayRule::Subject)
}

pub fn union(a: &Shapes, b: &Shapes) -> Shapes {
    run(&[a, b], &[], OverlayRule::Subject)
}

pub fn difference(a: &Shapes, b: &Shapes) -> Shapes {
    if a.is_empty() || b.is_empty() {
        return a.clone();
    }
    run(&[a], &[b], OverlayRule::Difference)
}

pub fn intersection(a: &Shapes, b: &Shapes) -> Shapes {
    if a.is_empty() || b.is_empty() {
        return Shapes::new();
    }
    run(&[a], &[b], OverlayRule::Intersect)
}

/// Normalise: union of the shapes with themselves (resolves overlaps, orients, simplifies).
pub fn clean(a: &Shapes) -> Shapes {
    run(&[a], &[], OverlayRule::Subject)
}

/// Shapes with an area above `min_area` (map units²), each as a one-shape set.
pub fn polys_of(shapes: &Shapes, min_area: f64) -> Vec<Shape> {
    shapes.iter().filter(|s| shape_area(s) > min_area).cloned().collect()
}

pub fn rect_shape(lo: Pt, hi: Pt) -> Shape {
    vec![vec![[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]]
}

/// Ring of `q` oriented CCW, deduplicated, rotated to start at its lexicographically largest (x, y) point
/// (the order vanilla pieces use).
pub fn ring_pts(c: &[Pt]) -> Contour {
    let mut out: Contour = Vec::with_capacity(c.len());
    for &p in c {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    while out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    if area2(&out) < 0 {
        out.reverse();
    }
    let k = (0..out.len()).max_by_key(|&i| (out[i][0], out[i][1])).unwrap_or(0);
    out.rotate_left(k);
    out
}

// ------------------------------------------------------------------------------------------------ buffering
fn round_pt(x: f64, y: f64) -> Pt {
    [x.round() as i32, y.round() as i32]
}

/// Disc as an inscribed regular polygon with `4 * quad_segs` sides (CCW).
fn disc(cx: f64, cy: f64, r: f64, quad_segs: usize) -> Contour {
    let n = 4 * quad_segs;
    (0..n)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            round_pt(cx + r * a.cos(), cy + r * a.sin())
        })
        .collect()
}

/// Buffer of an open polyline (points in fixed units, relative) with half-width `r` fixed units: flat caps unless
/// `round_caps`, round joins approximated with `4 * quad_segs`-gon discs.
pub fn stroke(path: &[[f64; 2]], r: f64, round_caps: bool, quad_segs: usize) -> Shapes {
    stroke_impl(path, r, round_caps, quad_segs, false)
}

/// Buffer of a closed ring (all vertices are joins).
pub fn stroke_ring(ring: &[[f64; 2]], r: f64, quad_segs: usize) -> Shapes {
    stroke_impl(ring, r, true, quad_segs, true)
}

fn stroke_impl(path: &[[f64; 2]], r: f64, round_caps: bool, quad_segs: usize, closed: bool) -> Shapes {
    let n = path.len();
    let mut parts: Shapes = Vec::new();
    if n == 0 {
        return parts;
    }
    let segs = if closed { n } else { n.saturating_sub(1) };
    for i in 0..segs {
        let a = path[i];
        let b = path[(i + 1) % n];
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = dx.hypot(dy);
        if l < 1e-9 {
            continue;
        }
        let (nx, ny) = (-dy / l * r, dx / l * r);
        let c = vec![
            round_pt(a[0] - nx, a[1] - ny),
            round_pt(b[0] - nx, b[1] - ny),
            round_pt(b[0] + nx, b[1] + ny),
            round_pt(a[0] + nx, a[1] + ny),
        ];
        if area2(&c) > 0 {
            parts.push(vec![c]);
        }
    }
    for i in 0..n {
        let interior = closed || (i > 0 && i + 1 < n);
        if interior || round_caps {
            parts.push(vec![disc(path[i][0], path[i][1], r, quad_segs)]);
        }
    }
    clean(&parts)
}

/// `shape` minus the strip of half-width `r` (fixed units) around its whole boundary (an inset by `r`).
pub fn inset(shapes: &Shapes, r: f64) -> Shapes {
    let mut band: Shapes = Vec::new();
    for sh in shapes {
        for c in sh {
            let ring: Vec<[f64; 2]> = c.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
            band.extend(stroke_ring(&ring, r, 2));
        }
    }
    difference(shapes, &band)
}

/// True when every point of `sh` lies within `w` map units of its boundary (a negative buffer by `w` is empty).
pub fn is_thin(sh: &Shape, w: f64) -> bool {
    let s = vec![sh.clone()];
    area(&inset(&s, w * FIX)) < 1e-12
}

// ------------------------------------------------------------------------------------------------ clipping
fn mul_div_round(a: i64, b: i64, d: i64) -> i64 {
    // round-half-even(a * b / d)
    round_div(a as i128 * b as i128, d as i128) as i64
}

/// round-half-even of n / d (d != 0).
pub fn round_div(mut n: i128, mut d: i128) -> i128 {
    if d < 0 {
        n = -n;
        d = -d;
    }
    let q = n.div_euclid(d);
    let r = n.rem_euclid(d);
    let twice = 2 * r;
    if twice > d || (twice == d && q % 2 != 0) { q + 1 } else { q }
}

/// Sutherland–Hodgman clip of a ring to an axis-aligned window (result may carry zero-area bridges on the window
/// boundary when the ring is concave; callers re-overlay it).
fn clip_ring_window(ring: &[Pt], lo: Pt, hi: Pt) -> Contour {
    let mut cur: Contour = ring.to_vec();
    for side in 0..4 {
        if cur.is_empty() {
            break;
        }
        let inside = |p: Pt| match side {
            0 => p[0] >= lo[0],
            1 => p[0] <= hi[0],
            2 => p[1] >= lo[1],
            _ => p[1] <= hi[1],
        };
        let cross = |a: Pt, b: Pt| -> Pt {
            match side {
                0 | 1 => {
                    let x = if side == 0 { lo[0] } else { hi[0] };
                    let y = a[1] as i64 + mul_div_round(b[1] as i64 - a[1] as i64, x as i64 - a[0] as i64, b[0] as i64 - a[0] as i64);
                    [x, y as i32]
                }
                _ => {
                    let y = if side == 2 { lo[1] } else { hi[1] };
                    let x = a[0] as i64 + mul_div_round(b[0] as i64 - a[0] as i64, y as i64 - a[1] as i64, b[1] as i64 - a[1] as i64);
                    [x as i32, y]
                }
            }
        };
        let mut next: Contour = Vec::with_capacity(cur.len() + 4);
        let m = cur.len();
        for i in 0..m {
            let a = cur[i];
            let b = cur[(i + 1) % m];
            match (inside(a), inside(b)) {
                (true, true) => next.push(b),
                (true, false) => next.push(cross(a, b)),
                (false, true) => {
                    next.push(cross(a, b));
                    next.push(b);
                }
                _ => {}
            }
        }
        cur = next;
    }
    cur
}

/// Intersection of `shapes` with the rectangle `lo..hi` (inclusive box).
pub fn clip_rect(shapes: &Shapes, lo: Pt, hi: Pt) -> Shapes {
    let m = (FIX as i32).max(1);
    let (wlo, whi) = ([lo[0] - m, lo[1] - m], [hi[0] + m, hi[1] + m]);
    let mut parts: Shapes = Vec::new();
    for sh in shapes {
        let Some(outer) = sh.first() else { continue };
        let (bl, bh) = bbox_of(outer);
        if bh[0] < wlo[0] || bl[0] > whi[0] || bh[1] < wlo[1] || bl[1] > whi[1] {
            continue;
        }
        let mut contours: Shape = Vec::new();
        for c in sh {
            let (cl, ch) = bbox_of(c);
            if ch[0] < wlo[0] || cl[0] > whi[0] || ch[1] < wlo[1] || cl[1] > whi[1] {
                continue;
            }
            let cc = clip_ring_window(c, wlo, whi);
            if cc.len() >= 3 {
                contours.push(cc);
            }
        }
        if !contours.is_empty() {
            parts.push(contours);
        }
    }
    if parts.is_empty() {
        return Shapes::new();
    }
    intersection(&parts, &vec![rect_shape(lo, hi)])
}

/// Cut holes out of a shape set by splitting along horizontal lines through the holes (every result piece is a
/// simple polygon without holes).
pub fn split_holes(sh: &Shape) -> Vec<Shape> {
    if sh.len() < 2 {
        return vec![sh.clone()];
    }
    let (lo, hi) = bbox_of(&sh[0]);
    let (hl, hh) = bbox_of(&sh[1]);
    let ycut = ((hl[1] as i64 + hh[1] as i64) / 2) as i32;
    let s = vec![sh.clone()];
    let mut out = Vec::new();
    for half in [rect_shape([lo[0] - 1, lo[1] - 1], [hi[0] + 1, ycut]), rect_shape([lo[0] - 1, ycut], [hi[0] + 1, hi[1] + 1])] {
        for piece in intersection(&s, &vec![half]) {
            out.extend(split_holes(&piece));
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ noding
/// Insert the crossing of every edge with the grid lines (multiples of `cs`, relative coordinates) into the ring,
/// rounding the crossing to fixed point; consecutive duplicates are dropped.
pub fn node_ring(ring: &[Pt], cs: i32) -> Contour {
    let n = ring.len();
    let cs64 = cs as i64;
    let mut out: Contour = Vec::with_capacity(n * 2);
    for i in 0..n {
        let p = ring[i];
        let q = ring[(i + 1) % n];
        out.push(p);
        // (t numerator, t denominator (>0), point)
        let mut cr: Vec<(i128, i128, Pt)> = Vec::new();
        if p[0] != q[0] {
            let (a, b) = (p[0].min(q[0]) as i64, p[0].max(q[0]) as i64);
            let k0 = a.div_euclid(cs64) + 1;
            let mut x = k0 * cs64;
            while x < b {
                let num = x - p[0] as i64;
                let den = q[0] as i64 - p[0] as i64;
                let y = p[1] as i64 + mul_div_round(q[1] as i64 - p[1] as i64, num, den);
                let (tn, td) = if den < 0 { (-(num as i128), -(den as i128)) } else { (num as i128, den as i128) };
                cr.push((tn, td, [x as i32, y as i32]));
                x += cs64;
            }
        }
        if p[1] != q[1] {
            let (a, b) = (p[1].min(q[1]) as i64, p[1].max(q[1]) as i64);
            let k0 = a.div_euclid(cs64) + 1;
            let mut y = k0 * cs64;
            while y < b {
                let num = y - p[1] as i64;
                let den = q[1] as i64 - p[1] as i64;
                let x = p[0] as i64 + mul_div_round(q[0] as i64 - p[0] as i64, num, den);
                let (tn, td) = if den < 0 { (-(num as i128), -(den as i128)) } else { (num as i128, den as i128) };
                cr.push((tn, td, [x as i32, y as i32]));
                y += cs64;
            }
        }
        cr.sort_by(|a, b| (a.0 * b.1).cmp(&(b.0 * a.1)));
        out.extend(cr.into_iter().map(|c| c.2));
    }
    let mut ded: Contour = Vec::with_capacity(out.len());
    for p in out {
        if ded.last() != Some(&p) {
            ded.push(p);
        }
    }
    while ded.len() > 1 && ded[0] == ded[ded.len() - 1] {
        ded.pop();
    }
    ded
}

/// Node every ring of `shapes` at the grid lines (see `node_ring`).
pub fn node_shapes(shapes: &Shapes, cs: i32) -> Shapes {
    shapes.iter().map(|sh| sh.iter().map(|c| node_ring(c, cs)).collect()).collect()
}

// ------------------------------------------------------------------------------------------------ distances
fn seg_dist2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 == 0.0 { 0.0 } else { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) };
    let (qx, qy) = (a[0] + t * dx, a[1] + t * dy);
    (p[0] - qx).powi(2) + (p[1] - qy).powi(2)
}

fn segs_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let o = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    let (d1, d2, d3, d4) = (o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b));
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0)) && d1 != 0.0 && d2 != 0.0 && d3 != 0.0 && d4 != 0.0
}

fn f(p: Pt) -> [f64; 2] {
    [p[0] as f64, p[1] as f64]
}

/// Point-in-shape-set test (even-odd over all rings).
pub fn contains_point(shapes: &Shapes, p: [f64; 2]) -> bool {
    let mut inside = false;
    for sh in shapes {
        for c in sh {
            let n = c.len();
            let mut j = n - 1;
            for i in 0..n {
                let (a, b) = (f(c[i]), f(c[j]));
                if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
                    inside = !inside;
                }
                j = i;
            }
        }
    }
    inside
}

/// Minimum distance between two shape sets in fixed units (0 when they overlap).
pub fn distance(a: &Shapes, b: &Shapes) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for sa in a {
        for ca in sa {
            let (al, ah) = bbox_of(ca);
            for sb in b {
                for cb in sb {
                    let (bl, bh) = bbox_of(cb);
                    let gap = |l1: i32, h1: i32, l2: i32, h2: i32| ((l2 as f64 - h1 as f64).max(l1 as f64 - h2 as f64)).max(0.0);
                    let g = gap(al[0], ah[0], bl[0], bh[0]).hypot(gap(al[1], ah[1], bl[1], bh[1]));
                    if g >= best {
                        continue;
                    }
                    for i in 0..ca.len() {
                        let (p, q) = (f(ca[i]), f(ca[(i + 1) % ca.len()]));
                        for j in 0..cb.len() {
                            let (r, s) = (f(cb[j]), f(cb[(j + 1) % cb.len()]));
                            if segs_cross(p, q, r, s) {
                                return 0.0;
                            }
                            let d = seg_dist2(p, r, s).min(seg_dist2(q, r, s)).min(seg_dist2(r, p, q)).min(seg_dist2(s, p, q));
                            if d < best * best {
                                best = d.sqrt();
                            }
                        }
                    }
                }
            }
        }
    }
    if best > 0.0 {
        // containment (one set entirely inside a filled part of the other)
        if let Some(c) = a.first().and_then(|s| s.first()).and_then(|c| c.first()) {
            if contains_point(b, f(*c)) {
                return 0.0;
            }
        }
        if let Some(c) = b.first().and_then(|s| s.first()).and_then(|c| c.first()) {
            if contains_point(a, f(*c)) {
                return 0.0;
            }
        }
    }
    best
}

/// Minimum distance between an open polyline (fixed units, relative) and a shape set (0 when it touches or crosses).
pub fn distance_polyline(line: &[[f64; 2]], b: &Shapes) -> f64 {
    let mut best = f64::INFINITY;
    for sh in b {
        for c in sh {
            let n = c.len();
            let (cl, ch) = bbox_of(c);
            for w in line.windows(2) {
                let (lo, hi) = ([w[0][0].min(w[1][0]), w[0][1].min(w[1][1])], [w[0][0].max(w[1][0]), w[0][1].max(w[1][1])]);
                let gx = ((cl[0] as f64 - hi[0]).max(lo[0] - ch[0] as f64)).max(0.0);
                let gy = ((cl[1] as f64 - hi[1]).max(lo[1] - ch[1] as f64)).max(0.0);
                if gx.hypot(gy) >= best {
                    continue;
                }
                for i in 0..n {
                    let (r, s) = (f(c[i]), f(c[(i + 1) % n]));
                    if segs_cross(w[0], w[1], r, s) {
                        return 0.0;
                    }
                    let d = seg_dist2(w[0], r, s).min(seg_dist2(w[1], r, s)).min(seg_dist2(r, w[0], w[1])).min(seg_dist2(s, w[0], w[1]));
                    best = best.min(d.sqrt());
                }
            }
        }
    }
    if best > 0.0 {
        if let Some(p) = line.first() {
            if contains_point(b, *p) {
                return 0.0;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(x0: i32, y0: i32, x1: i32, y1: i32) -> Shapes {
        vec![rect_shape([x0, y0], [x1, y1])]
    }

    #[test]
    fn boolean_ops_keep_integer_coordinates() {
        let a = sq(0, 0, 100, 100);
        let b = sq(50, 50, 150, 150);
        assert_eq!(area2(&union(&a, &b)[0][0]).abs(), 2 * (100 * 100 + 100 * 100 - 50 * 50));
        let i = intersection(&a, &b);
        assert_eq!(area2(&i[0][0]), 2 * 50 * 50);
        let d = difference(&a, &b);
        assert_eq!(area2(&d[0][0]), 2 * (100 * 100 - 50 * 50));
    }

    #[test]
    fn rect_clip_matches_overlay() {
        // L-shaped polygon clipped by a box
        let l = vec![vec![vec![[0, 0], [300, 0], [300, 100], [100, 100], [100, 300], [0, 300]]]];
        let c = clip_rect(&l, [50, 50], [250, 250]);
        let want = intersection(&l, &sq(50, 50, 250, 250));
        assert_eq!(c.len(), want.len());
        assert_eq!(area2(&c[0][0]), area2(&want[0][0]));
    }

    #[test]
    fn noding_inserts_grid_crossings() {
        let ring = vec![[10, 10], [250, 10], [250, 250], [10, 250]];
        let n = node_ring(&ring, 100);
        assert!(n.contains(&[100, 10]) && n.contains(&[200, 10]) && n.contains(&[250, 100]) && n.contains(&[100, 250]));
    }

    #[test]
    fn stroke_covers_width() {
        let line = vec![[0.0, 0.0], [1000.0, 0.0]];
        let s = stroke(&line, 50.0, false, 2);
        assert!((area(&s) * FIX * FIX - 100.0 * 1000.0).abs() < 1.0);
        let thin = vec![vec![[0, 0], [10000, 0], [10000, 100], [0, 100]]];
        assert!(is_thin(&thin, 100.0 / FIX));
    }

    #[test]
    fn holes_are_split() {
        let sh = vec![
            vec![[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            vec![[400, 400], [400, 600], [600, 600], [600, 400]],
        ];
        let parts = split_holes(&sh);
        assert_eq!(parts.len(), 2);
        let total: f64 = parts.iter().map(shape_area).sum();
        assert!((total * FIX * FIX - (1_000_000.0 - 40_000.0)).abs() < 1.0);
    }
}
