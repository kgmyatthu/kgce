//! Vanilla-style landing crossings (port of work/exp4/crossings.py): two congruent hexagons ("half octagons")
//! separated by a 0.1 wide type-2 gap; the sea half (type 7, sea path) lies seaward of the coast outline, the land half
//! (type 7, region path) is its mirror image landward. The gap centre lies 0.1 landward of the outline and the gap runs
//! along the coast tangent rounded to 5 degrees.

use super::geometry::{Contour, FIX, Pt, ring_pts};

const HEX: [(f64, f64); 6] = [(0.0, 0.95), (0.9, 0.95), (1.4, 0.45), (1.4, -0.45), (0.9, -0.95), (0.0, -0.95)];
const GAP: f64 = 0.1;
const OFFSET: f64 = 0.1;
const ANGLE_STEP: f64 = 5.0;

/// offsets tried (map units) around the wanted arc position
pub const SLIDE: [f64; 19] = [0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0];
/// minimum arc distance between two landings on one coast line
pub const MIN_SEP: f64 = 4.0;
/// coast fragments shorter than this get no landing
pub const MIN_LINE: f64 = 4.0;

/// Polyline in map units (coordinates relative to the grid origin).
#[derive(Clone, Debug)]
pub struct Polyline {
    pub pts: Vec<[f64; 2]>,
    cum: Vec<f64>,
}

impl Polyline {
    pub fn new(pts: Vec<[f64; 2]>) -> Self {
        let mut cum = Vec::with_capacity(pts.len());
        let mut s = 0.0;
        for (i, p) in pts.iter().enumerate() {
            if i > 0 {
                s += (p[0] - pts[i - 1][0]).hypot(p[1] - pts[i - 1][1]);
            }
            cum.push(s);
        }
        Self { pts, cum }
    }

    pub fn length(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    pub fn interpolate(&self, s: f64) -> [f64; 2] {
        let s = s.clamp(0.0, self.length());
        let i = self.cum.partition_point(|&c| c < s).clamp(1, self.pts.len() - 1);
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let l = self.cum[i] - self.cum[i - 1];
        let t = if l == 0.0 { 0.0 } else { (s - self.cum[i - 1]) / l };
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    }

    /// Direction (degrees) of the chord through the line `half` either side of `s`.
    pub fn chord_angle(&self, s: f64, half: f64) -> f64 {
        let a = self.interpolate((s - half).max(0.0));
        let b = self.interpolate((s + half).min(self.length()));
        (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees()
    }
}

#[derive(Clone, Debug)]
pub struct Landing {
    pub s: f64,
    pub want: f64,
    pub sea: Contour,
    pub land: Contour,
    pub gap_centre: [f64; 2],
    pub angle: f64,
}

fn fixed(p: [f64; 2]) -> Pt {
    [(p[0] * FIX).round() as i32, (p[1] * FIX).round() as i32]
}

/// Candidate landing at arc position `s` of `line` (land on the left of the travel direction).
pub fn candidate(line: &Polyline, s: f64) -> Landing {
    let q = line.interpolate(s);
    let a = (line.chord_angle(s, 0.95) / ANGLE_STEP).round_ties_even() * ANGLE_STEP;
    let r = a.to_radians();
    let t = (r.cos(), r.sin());
    let n_land = (-t.1, t.0);
    let o = [q[0] + OFFSET * n_land.0, q[1] + OFFSET * n_land.1];
    let half = |sign: f64| -> Contour {
        let (nx, ny) = (sign * n_land.0, sign * n_land.1);
        let raw: Contour = HEX
            .iter()
            .map(|&(u, v)| fixed([o[0] + nx * (GAP / 2.0 + u) + t.0 * v, o[1] + ny * (GAP / 2.0 + u) + t.1 * v]))
            .collect();
        ring_pts(&raw)
    };
    Landing { s, want: s, sea: half(-1.0), land: half(1.0), gap_centre: o, angle: a }
}

/// Wanted arc positions along a coast line of length `ln`.
pub fn wanted_positions(ln: f64, spacing: f64) -> Vec<f64> {
    let n_w = if ln > 6.0 { ((ln - 6.0) / spacing).ceil() as usize + 1 } else { 1 };
    if n_w <= 1 {
        vec![ln / 2.0]
    } else {
        (0..n_w).map(|k| 3.0 + k as f64 * (ln - 6.0) / (n_w - 1) as f64).collect()
    }
}

/// Positions scanned (closest to the middle first) to fill the stretch `a..b` that is longer than the allowed gap.
pub fn fill_positions(a: f64, b: f64) -> Vec<f64> {
    let mid = (a + b) / 2.0;
    let n = ((b - a - 8.0) / 0.25) as i64 + 1;
    let mut v: Vec<f64> = (0..n.max(0)).map(|k| a + 4.0 + 0.25 * k as f64).collect();
    v.sort_by(|x, y| (x - mid).abs().partial_cmp(&(y - mid).abs()).unwrap());
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regionmap::pathsynth::geometry::area2;

    #[test]
    fn hexagon_area_and_orientation() {
        let line = Polyline::new(vec![[0.0, 0.0], [20.0, 0.0]]);
        let c = candidate(&line, 10.0);
        let a = area2(&c.sea) as f64 / 2.0 / (FIX * FIX);
        assert!((a - 2.41).abs() < 0.01, "{a}");
        assert!(area2(&c.land) > 0 && area2(&c.sea) > 0, "halves must be CCW");
        // land on the left of +x travel: land half above the line, sea half below
        let oy = (c.gap_centre[1] * FIX) as i32;
        assert!(c.land.iter().all(|p| p[1] > oy) && c.sea.iter().all(|p| p[1] < oy));
    }
}
