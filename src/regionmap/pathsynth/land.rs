//! Land / coast / border geometry of regions.esf areas in the grid frame (port of the region reading half of
//! work/exp5/pathsynth5.py).

use std::collections::BTreeMap;

use super::geometry::{self as g, FIX, Pt, Shapes};
use crate::regionmap::regions::{NONE, RegionData};

/// Classification of one outline edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cls {
    /// impassable neighbour, no neighbour, a region that never forms group paths, or another area of the same region
    Imp,
    /// edge shared with a sea region
    Sea,
    /// edge shared with a passable land area `(region, area)`
    Pass(usize, usize),
}

/// Grid frame: converts regions.esf coordinates (map units) to coordinates relative to the grid origin.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub origin: [f64; 2],
    pub cs: i32,
}

impl Frame {
    /// relative map units
    pub fn rel(&self, p: [f32; 2]) -> [f64; 2] {
        [p[0] as f64 - self.origin[0], p[1] as f64 - self.origin[1]]
    }
    pub fn fixed(&self, p: [f64; 2]) -> Pt {
        [(p[0] * FIX).round() as i32, (p[1] * FIX).round() as i32]
    }
}

pub fn scale(line: &[[f64; 2]]) -> Vec<[f64; 2]> {
    line.iter().map(|p| [p[0] * FIX, p[1] * FIX]).collect()
}

/// Per ring edge class of outline `oi` of area (`ri`, `ai`).
pub fn edge_classes(rd: &RegionData, ri: usize, ai: usize, oi: usize, no_group: &[usize], in_grid: &dyn Fn(usize) -> bool) -> Vec<Cls> {
    let o = &rd.regions[ri].areas[ai].outlines[oi];
    let n = o.ring.len();
    let mut cls = vec![Cls::Imp; n];
    for run in &o.runs {
        if run.code == NONE {
            continue;
        }
        let (nr, na) = ((run.code >> 16) as usize, (run.code & 0xFFFF) as usize);
        let Some(region) = rd.regions.get(nr) else { continue };
        let c = if region.kind == "sea" {
            Cls::Sea
        } else if (nr, na) != (ri, ai) && nr != ri && !no_group.contains(&nr) && in_grid(nr) && region.areas.get(na).is_some_and(|a| a.passable) {
            Cls::Pass(nr, na)
        } else {
            Cls::Imp
        };
        for k in run.from..=run.to {
            if (k as usize) < n {
                cls[k as usize] = c;
            }
        }
    }
    cls
}

/// Maximal polylines of consecutive ring edges whose class satisfies `pred` (same class per polyline); the ring is
/// rotated to start at a class change.
pub fn class_runs(ring: &[[f64; 2]], cls: &[Cls], pred: &dyn Fn(Cls) -> bool) -> Vec<(Cls, Vec<[f64; 2]>)> {
    let n = ring.len();
    let mut start = 0;
    for k in 0..n {
        if cls[(k + n - 1) % n] != cls[k] {
            start = k;
            break;
        }
    }
    let mut runs = Vec::new();
    let mut cur: Option<(Cls, Vec<[f64; 2]>)> = None;
    for j in 0..n {
        let k = (start + j) % n;
        let c = cls[k];
        if pred(c) {
            match &mut cur {
                Some((cc, pts)) if *cc == c => pts.push(ring[(k + 1) % n]),
                _ => {
                    if let Some(done) = cur.take() {
                        runs.push(done);
                    }
                    cur = Some((c, vec![ring[k], ring[(k + 1) % n]]));
                }
            }
        } else if let Some(done) = cur.take() {
            runs.push(done);
        }
    }
    if let Some(done) = cur.take() {
        runs.push(done);
    }
    runs
}

/// Polygon of one area (outer rings united, holes removed) in fixed relative coordinates.
pub fn area_polygon(rd: &RegionData, fr: &Frame, ri: usize, ai: usize) -> Shapes {
    let a = &rd.regions[ri].areas[ai];
    let mut outers: Shapes = Vec::new();
    let mut holes: Shapes = Vec::new();
    for o in &a.outlines {
        let ring: Vec<Pt> = o.ring.iter().map(|&v| fr.fixed(fr.rel(rd.vertices[v as usize]))).collect();
        let ring = g::ring_pts(&ring);
        if ring.len() < 3 {
            continue;
        }
        if o.outer { outers.push(vec![ring]) } else { holes.push(vec![ring]) }
    }
    let u = g::clean(&outers);
    g::difference(&u, &g::clean(&holes))
}

/// Region ring in relative map units.
pub fn ring_rel(rd: &RegionData, fr: &Frame, ri: usize, ai: usize, oi: usize) -> Vec<[f64; 2]> {
    rd.regions[ri].areas[ai].outlines[oi].ring.iter().map(|&v| fr.rel(rd.vertices[v as usize])).collect()
}

/// Land polygon, coast lines (`Sea` class runs) and border lines per passable-land neighbour of one area.
pub struct AreaEdges {
    pub poly: Shapes,
    pub coast: Vec<Vec<[f64; 2]>>,
    pub borders: BTreeMap<(usize, usize), Vec<Vec<[f64; 2]>>>,
}

pub fn area_edges(rd: &RegionData, fr: &Frame, ri: usize, ai: usize, no_group: &[usize], in_grid: &dyn Fn(usize) -> bool) -> AreaEdges {
    let mut coast = Vec::new();
    let mut borders: BTreeMap<(usize, usize), Vec<Vec<[f64; 2]>>> = BTreeMap::new();
    for (oi, _) in rd.regions[ri].areas[ai].outlines.iter().enumerate() {
        let ring = ring_rel(rd, fr, ri, ai, oi);
        if ring.is_empty() {
            continue;
        }
        let cls = edge_classes(rd, ri, ai, oi, no_group, in_grid);
        for (_, ln) in class_runs(&ring, &cls, &|c| c == Cls::Sea) {
            coast.push(ln);
        }
        for (c, ln) in class_runs(&ring, &cls, &|c| matches!(c, Cls::Pass(..))) {
            if let Cls::Pass(nr, na) = c {
                borders.entry((nr, na)).or_default().push(ln);
            }
        }
    }
    AreaEdges { poly: area_polygon(rd, fr, ri, ai), coast, borders }
}

/// Passable sea of the sea regions inside the window `lo..hi` (fixed relative) minus the 0.5 band along every
/// non-sea edge. Returns (sea minus bands, sea).
pub fn sea_geometry(rd: &RegionData, fr: &Frame, lo: Pt, hi: Pt, no_group: &[usize], in_grid: &dyn Fn(usize) -> bool) -> (Shapes, Shapes) {
    let win = vec![g::rect_shape(lo, hi)];
    let mut polys: Vec<Shapes> = Vec::new();
    let mut band: Shapes = Vec::new();
    let m = FIX;
    for (ri, r) in rd.regions.iter().enumerate() {
        if r.kind != "sea" {
            continue;
        }
        for (ai, a) in r.areas.iter().enumerate() {
            if !a.passable {
                continue;
            }
            // bbox test on the outer rings
            let mut any = false;
            for o in a.outlines.iter().filter(|o| o.outer) {
                let pts: Vec<Pt> = o.ring.iter().map(|&v| fr.fixed(fr.rel(rd.vertices[v as usize]))).collect();
                let (bl, bh) = g::bbox_of(&pts);
                if !(bh[0] < lo[0] || bl[0] > hi[0] || bh[1] < lo[1] || bl[1] > hi[1]) {
                    any = true;
                }
            }
            if !any {
                continue;
            }
            let poly = area_polygon(rd, fr, ri, ai);
            polys.push(g::intersection(&poly, &win));
            for (oi, _) in a.outlines.iter().enumerate() {
                let ring = ring_rel(rd, fr, ri, ai, oi);
                if ring.is_empty() {
                    continue;
                }
                let cls = edge_classes(rd, ri, ai, oi, no_group, in_grid);
                for (_, ln) in class_runs(&ring, &cls, &|c| c != Cls::Sea) {
                    let s = scale(&ln);
                    let (mut l, mut h) = ([f64::MAX; 2], [f64::MIN; 2]);
                    for p in &s {
                        for k in 0..2 {
                            l[k] = l[k].min(p[k]);
                            h[k] = h[k].max(p[k]);
                        }
                    }
                    if h[0] < lo[0] as f64 - m || l[0] > hi[0] as f64 + m || h[1] < lo[1] as f64 - m || l[1] > hi[1] as f64 + m {
                        continue;
                    }
                    band.extend(g::stroke(&s, 0.5 * FIX, false, 2));
                }
            }
        }
    }
    let refs: Vec<&Shapes> = polys.iter().collect();
    let sea = g::union_many(&refs);
    let band = g::clean(&band);
    (g::difference(&sea, &band), sea)
}
