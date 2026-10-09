//! Polygon-with-holes triangulation (ear clipping with hole bridging, after the earcut algorithm).
//! Input rings are in world coordinates, y pointing north.

pub type Point = [f64; 2];
pub type Triangle = [[f32; 2]; 3];

struct Vertex {
    id: usize,
    x: f64,
    y: f64,
    prev: usize,
    next: usize,
    steiner: bool,
}

struct List {
    v: Vec<Vertex>,
}

fn signed_area(ring: &[Point]) -> f64 {
    // > 0 for counter-clockwise rings (y up).
    let mut sum = 0.0;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        sum += (ring[j][0] - ring[i][0]) * (ring[i][1] + ring[j][1]);
        j = i;
    }
    sum
}

impl List {
    fn insert(&mut self, id: usize, p: Point, last: Option<usize>) -> usize {
        let at = self.v.len();
        match last {
            None => self.v.push(Vertex {
                id,
                x: p[0],
                y: p[1],
                prev: at,
                next: at,
                steiner: false,
            }),
            Some(l) => {
                let next = self.v[l].next;
                self.v.push(Vertex {
                    id,
                    x: p[0],
                    y: p[1],
                    prev: l,
                    next,
                    steiner: false,
                });
                self.v[next].prev = at;
                self.v[l].next = at;
            }
        }
        at
    }

    fn remove(&mut self, p: usize) {
        let (prev, next) = (self.v[p].prev, self.v[p].next);
        self.v[next].prev = prev;
        self.v[prev].next = next;
    }

    fn equals(&self, a: usize, b: usize) -> bool {
        self.v[a].x == self.v[b].x && self.v[a].y == self.v[b].y
    }

    fn area(&self, p: usize, q: usize, r: usize) -> f64 {
        let (p, q, r) = (&self.v[p], &self.v[q], &self.v[r]);
        (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    }

    /// Build a circular list. `ccw` selects the winding the list ends up in (outer rings CCW, holes CW).
    fn linked_list(&mut self, ids: &[usize], pts: &[Point], ccw: bool) -> usize {
        let ring: Vec<Point> = ids.iter().map(|&i| pts[i]).collect();
        let forward = ccw == (signed_area(&ring) > 0.0);
        let mut last: Option<usize> = None;
        if forward {
            for (k, &id) in ids.iter().enumerate() {
                last = Some(self.insert(id, ring[k], last));
            }
        } else {
            for (k, &id) in ids.iter().enumerate().rev() {
                last = Some(self.insert(id, ring[k], last));
            }
        }
        let mut last = last.expect("ring has points");
        let next = self.v[last].next;
        if self.equals(last, next) {
            self.remove(last);
            last = next;
        }
        last
    }

    fn filter_points(&mut self, start: usize, end: Option<usize>) -> usize {
        let mut end = end.unwrap_or(start);
        let mut p = start;
        loop {
            let mut again = false;
            if !self.v[p].steiner
                && (self.equals(p, self.v[p].next)
                    || self.area(self.v[p].prev, p, self.v[p].next) == 0.0)
            {
                self.remove(p);
                p = self.v[p].prev;
                end = p;
                if p == self.v[p].next {
                    break;
                }
                again = true;
            } else {
                p = self.v[p].next;
            }
            if !again && p == end {
                break;
            }
        }
        end
    }

    fn point_in_triangle(
        ax: f64,
        ay: f64,
        bx: f64,
        by: f64,
        cx: f64,
        cy: f64,
        px: f64,
        py: f64,
    ) -> bool {
        (cx - px) * (ay - py) >= (ax - px) * (cy - py)
            && (ax - px) * (by - py) >= (bx - px) * (ay - py)
            && (bx - px) * (cy - py) >= (cx - px) * (by - py)
    }

    fn is_ear(&self, ear: usize) -> bool {
        let (a, b, c) = (self.v[ear].prev, ear, self.v[ear].next);
        if self.area(a, b, c) >= 0.0 {
            return false;
        }
        let (ax, ay, bx, by, cx, cy) = (
            self.v[a].x,
            self.v[a].y,
            self.v[b].x,
            self.v[b].y,
            self.v[c].x,
            self.v[c].y,
        );
        let (x0, x1) = (ax.min(bx).min(cx), ax.max(bx).max(cx));
        let (y0, y1) = (ay.min(by).min(cy), ay.max(by).max(cy));
        let mut p = self.v[c].next;
        while p != a {
            let q = &self.v[p];
            if q.x >= x0
                && q.x <= x1
                && q.y >= y0
                && q.y <= y1
                && !(ax == q.x && ay == q.y)
                && Self::point_in_triangle(ax, ay, bx, by, cx, cy, q.x, q.y)
                && self.area(q.prev, p, q.next) >= 0.0
            {
                return false;
            }
            p = q.next;
        }
        true
    }

    fn earcut_linked(&mut self, mut ear: usize, tri: &mut Vec<[usize; 3]>, pass: u8) {
        let mut stop = ear;
        while self.v[ear].prev != self.v[ear].next {
            let (prev, next) = (self.v[ear].prev, self.v[ear].next);
            if self.is_ear(ear) {
                tri.push([self.v[prev].id, self.v[ear].id, self.v[next].id]);
                self.remove(ear);
                ear = self.v[next].next;
                stop = ear;
                continue;
            }
            ear = next;
            if ear == stop {
                match pass {
                    0 => {
                        let f = self.filter_points(ear, None);
                        self.earcut_linked(f, tri, 1);
                    }
                    1 => {
                        let f = self.filter_points(ear, None);
                        let c = self.cure_local_intersections(f, tri);
                        self.earcut_linked(c, tri, 2);
                    }
                    _ => self.split_earcut(ear, tri),
                }
                break;
            }
        }
    }

    fn on_segment(&self, p: usize, q: usize, r: usize) -> bool {
        let (p, q, r) = (&self.v[p], &self.v[q], &self.v[r]);
        q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) && q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
    }

    fn sign(v: f64) -> i32 {
        if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        }
    }

    fn intersects(&self, p1: usize, q1: usize, p2: usize, q2: usize) -> bool {
        let o1 = Self::sign(self.area(p1, q1, p2));
        let o2 = Self::sign(self.area(p1, q1, q2));
        let o3 = Self::sign(self.area(p2, q2, p1));
        let o4 = Self::sign(self.area(p2, q2, q1));
        (o1 != o2 && o3 != o4)
            || (o1 == 0 && self.on_segment(p1, p2, q1))
            || (o2 == 0 && self.on_segment(p1, q2, q1))
            || (o3 == 0 && self.on_segment(p2, p1, q2))
            || (o4 == 0 && self.on_segment(p2, q1, q2))
    }

    fn cure_local_intersections(&mut self, mut start: usize, tri: &mut Vec<[usize; 3]>) -> usize {
        let mut p = start;
        loop {
            let a = self.v[p].prev;
            let pn = self.v[p].next;
            let b = self.v[pn].next;
            if !self.equals(a, b)
                && self.intersects(a, p, pn, b)
                && self.locally_inside(a, b)
                && self.locally_inside(b, a)
            {
                tri.push([self.v[a].id, self.v[p].id, self.v[b].id]);
                self.remove(p);
                self.remove(pn);
                start = b;
                p = b;
            }
            p = self.v[p].next;
            if p == start {
                break;
            }
        }
        self.filter_points(p, None)
    }

    fn locally_inside(&self, a: usize, b: usize) -> bool {
        let (an, ap) = (self.v[a].next, self.v[a].prev);
        if self.area(ap, a, an) < 0.0 {
            self.area(a, b, an) >= 0.0 && self.area(a, ap, b) >= 0.0
        } else {
            self.area(a, b, ap) < 0.0 || self.area(a, an, b) < 0.0
        }
    }

    fn middle_inside(&self, a: usize, b: usize) -> bool {
        let px = (self.v[a].x + self.v[b].x) / 2.0;
        let py = (self.v[a].y + self.v[b].y) / 2.0;
        let mut inside = false;
        let mut p = a;
        loop {
            let (cur, nxt) = (&self.v[p], &self.v[self.v[p].next]);
            if (cur.y > py) != (nxt.y > py)
                && nxt.y != cur.y
                && px < (nxt.x - cur.x) * (py - cur.y) / (nxt.y - cur.y) + cur.x
            {
                inside = !inside;
            }
            p = self.v[p].next;
            if p == a {
                break;
            }
        }
        inside
    }

    fn intersects_polygon(&self, a: usize, b: usize) -> bool {
        let mut p = a;
        loop {
            let n = self.v[p].next;
            if self.v[p].id != self.v[a].id
                && self.v[n].id != self.v[a].id
                && self.v[p].id != self.v[b].id
                && self.v[n].id != self.v[b].id
                && self.intersects(p, n, a, b)
            {
                return true;
            }
            p = n;
            if p == a {
                return false;
            }
        }
    }

    fn is_valid_diagonal(&self, a: usize, b: usize) -> bool {
        let (an, ap, bn, bp) = (
            self.v[a].next,
            self.v[a].prev,
            self.v[b].next,
            self.v[b].prev,
        );
        self.v[an].id != self.v[b].id
            && self.v[ap].id != self.v[b].id
            && !self.intersects_polygon(a, b)
            && ((self.locally_inside(a, b)
                && self.locally_inside(b, a)
                && self.middle_inside(a, b)
                && (self.area(ap, a, bp) != 0.0 || self.area(a, bp, b) != 0.0))
                || (self.equals(a, b) && self.area(ap, a, an) > 0.0 && self.area(bp, b, bn) > 0.0))
    }

    fn split_polygon(&mut self, a: usize, b: usize) -> usize {
        let a2 = self.v.len();
        let b2 = a2 + 1;
        let (an, bp) = (self.v[a].next, self.v[b].prev);
        let (ai, ax, ay) = (self.v[a].id, self.v[a].x, self.v[a].y);
        let (bi, bx, by) = (self.v[b].id, self.v[b].x, self.v[b].y);
        self.v.push(Vertex {
            id: ai,
            x: ax,
            y: ay,
            prev: b2,
            next: an,
            steiner: false,
        });
        self.v.push(Vertex {
            id: bi,
            x: bx,
            y: by,
            prev: bp,
            next: a2,
            steiner: false,
        });
        self.v[a].next = b;
        self.v[b].prev = a;
        self.v[an].prev = a2;
        self.v[bp].next = b2;
        b2
    }

    fn split_earcut(&mut self, start: usize, tri: &mut Vec<[usize; 3]>) {
        let mut a = start;
        loop {
            let mut b = self.v[self.v[a].next].next;
            while b != self.v[a].prev {
                if self.v[a].id != self.v[b].id && self.is_valid_diagonal(a, b) {
                    let c = self.split_polygon(a, b);
                    let a_next = self.v[a].next;
                    let a = self.filter_points(a, Some(a_next));
                    let c_next = self.v[c].next;
                    let c = self.filter_points(c, Some(c_next));
                    self.earcut_linked(a, tri, 0);
                    self.earcut_linked(c, tri, 0);
                    return;
                }
                b = self.v[b].next;
            }
            a = self.v[a].next;
            if a == start {
                break;
            }
        }
    }

    fn leftmost(&self, start: usize) -> usize {
        let mut p = start;
        let mut left = start;
        loop {
            let (q, l) = (&self.v[p], &self.v[left]);
            if q.x < l.x || (q.x == l.x && q.y < l.y) {
                left = p;
            }
            p = q.next;
            if p == start {
                return left;
            }
        }
    }

    fn sector_contains_sector(&self, m: usize, p: usize) -> bool {
        self.area(self.v[m].prev, m, self.v[p].prev) < 0.0
            && self.area(self.v[p].next, m, self.v[m].next) < 0.0
    }

    fn find_hole_bridge(&self, hole: usize, outer: usize) -> Option<usize> {
        let (hx, hy) = (self.v[hole].x, self.v[hole].y);
        let mut qx = f64::NEG_INFINITY;
        let mut m: Option<usize> = None;
        let mut p = outer;
        loop {
            if self.equals(hole, p) {
                return Some(p);
            }
            let n = self.v[p].next;
            let (pp, nn) = (&self.v[p], &self.v[n]);
            if hy <= pp.y && hy >= nn.y && nn.y != pp.y {
                let x = pp.x + (hy - pp.y) * (nn.x - pp.x) / (nn.y - pp.y);
                if x <= hx && x > qx {
                    qx = x;
                    m = Some(if pp.x < nn.x { p } else { n });
                    if x == hx {
                        return m;
                    }
                }
            }
            p = n;
            if p == outer {
                break;
            }
        }
        let mut m = m?;
        let stop = m;
        let (mx, my) = (self.v[m].x, self.v[m].y);
        let mut tan_min = f64::INFINITY;
        p = m;
        loop {
            let q = &self.v[p];
            let (t1x, t2x) = if hy < my { (hx, qx) } else { (qx, hx) };
            if hx >= q.x
                && q.x >= mx
                && hx != q.x
                && Self::point_in_triangle(t1x, hy, mx, my, t2x, hy, q.x, q.y)
            {
                let tan = (hy - q.y).abs() / (hx - q.x);
                if self.locally_inside(p, hole)
                    && (tan < tan_min
                        || (tan == tan_min
                            && (q.x > self.v[m].x
                                || (q.x == self.v[m].x && self.sector_contains_sector(m, p)))))
                {
                    m = p;
                    tan_min = tan;
                }
            }
            p = q.next;
            if p == stop {
                break;
            }
        }
        Some(m)
    }

    fn eliminate_holes(&mut self, holes: &[Vec<usize>], pts: &[Point], mut outer: usize) -> usize {
        let mut queue: Vec<usize> = Vec::new();
        for h in holes {
            let list = self.linked_list(h, pts, false);
            if list == self.v[list].next {
                self.v[list].steiner = true;
            }
            queue.push(self.leftmost(list));
        }
        queue.sort_by(|&a, &b| {
            self.v[a]
                .x
                .partial_cmp(&self.v[b].x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for hole in queue {
            outer = self.eliminate_hole(hole, outer);
        }
        outer
    }

    fn eliminate_hole(&mut self, hole: usize, outer: usize) -> usize {
        let Some(bridge) = self.find_hole_bridge(hole, outer) else {
            return outer;
        };
        let reverse = self.split_polygon(bridge, hole);
        let rn = self.v[reverse].next;
        self.filter_points(reverse, Some(rn));
        let bn = self.v[bridge].next;
        self.filter_points(bridge, Some(bn))
    }
}

fn polygon_area(ring: &[Point]) -> f64 {
    signed_area(ring).abs()
}

fn point_in_ring(p: Point, ring: &[Point]) -> bool {
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let (a, b) = (ring[i], ring[j]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn clean(mut ring: Vec<Point>) -> Vec<Point> {
    ring.dedup();
    while ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    ring
}

/// Triangulate one area made of outer rings and hole rings (holes are assigned to the smallest outer ring that
/// contains most of their vertices).
pub fn triangulate_area(outers: Vec<Vec<Point>>, holes: Vec<Vec<Point>>, out: &mut Vec<Triangle>) {
    let outers: Vec<Vec<Point>> = outers
        .into_iter()
        .map(clean)
        .filter(|r| r.len() >= 3)
        .collect();
    let holes: Vec<Vec<Point>> = holes
        .into_iter()
        .map(clean)
        .filter(|r| r.len() >= 3)
        .collect();
    let mut assigned: Vec<Vec<usize>> = vec![Vec::new(); outers.len()];
    for (hi, hole) in holes.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (oi, outer) in outers.iter().enumerate() {
            let inside = hole.iter().filter(|&&p| point_in_ring(p, outer)).count();
            if inside * 2 >= hole.len() {
                let area = polygon_area(outer);
                if best.map_or(true, |(_, a)| area < a) {
                    best = Some((oi, area));
                }
            }
        }
        if let Some((oi, _)) = best {
            assigned[oi].push(hi);
        }
    }
    for (oi, outer) in outers.iter().enumerate() {
        let mut pts: Vec<Point> = outer.clone();
        let outer_ids: Vec<usize> = (0..pts.len()).collect();
        let mut hole_ids: Vec<Vec<usize>> = Vec::new();
        for &hi in &assigned[oi] {
            let start = pts.len();
            pts.extend_from_slice(&holes[hi]);
            hole_ids.push((start..pts.len()).collect());
        }
        let mut list = List { v: Vec::new() };
        let start = list.linked_list(&outer_ids, &pts, true);
        if list.v[start].next == list.v[start].prev {
            continue;
        }
        let start = if hole_ids.is_empty() {
            start
        } else {
            list.eliminate_holes(&hole_ids, &pts, start)
        };
        let mut tri = Vec::new();
        list.earcut_linked(start, &mut tri, 0);
        for t in tri {
            out.push([0, 1, 2].map(|k| [pts[t[k]][0] as f32, pts[t[k]][1] as f32]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_mesh_preserves_lakes_and_disconnected_islands() {
        let mut mesh = Vec::new();
        triangulate_area(
            vec![
                vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                vec![[20.0, 0.0], [22.0, 0.0], [22.0, 2.0], [20.0, 2.0]],
            ],
            vec![vec![[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]]],
            &mut mesh,
        );
        let triangles: Vec<Vec<Point>> = mesh
            .iter()
            .map(|t| t.iter().map(|p| [p[0] as f64, p[1] as f64]).collect())
            .collect();
        let area: f64 = triangles
            .iter()
            .map(|t| {
                ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1])
                    - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0]))
                    .abs()
                    / 2.0
            })
            .sum();
        assert!((area - 88.0).abs() < 1e-6, "Land area changed: {area}");
        assert!(
            !triangles.iter().any(|t| point_in_ring([5.2, 5.4], t)),
            "Lake filled with land"
        );
        assert!(
            !triangles.iter().any(|t| point_in_ring([15.0, 1.0], t)),
            "Disconnected islands joined"
        );
    }
}
