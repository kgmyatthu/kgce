//! Regeneration of `query_info`, the quadtree point-location index of regions.esf (accepted POC rules):
//! a node is split iff more than 50 matched undirected ring edges strictly cross the open interior of its box
//! (depth cap 12); a leaf holds a probe point (the box lower-left corner unless within 0.0031 of the line of a
//! listed segment), the area at the probe and every strictly crossing matched edge as
//! (v1, v2, right owner, left owner) oriented so that the probe lies on its left (groundtypes: left owner -1).

use super::assemble::OutData;
use super::geom::{AreaPoly, GridIndex, P, Rect64, widen};
use crate::esf::{self, Node};
use anyhow::{Result, bail};
use std::collections::{BTreeMap, HashSet};

pub const SPLIT_THRESHOLD: usize = 50;
pub const MAX_DEPTH: usize = 12;
pub const PROBE_LINE_EPS: f64 = 0.0031;

pub struct QueryStats {
    pub nodes: u64,
    pub cells: u64,
    pub max_leaf: usize,
    pub depth_cap_leaves: u64,
    pub probe_fallbacks: BTreeMap<String, u64>,
}

struct Builder<'a, 'b> {
    od: &'b OutData<'a>,
    with_left: bool,
    edges: Vec<(u32, u32)>,
    a: Vec<P>,
    b: Vec<P>,
    polys: Vec<AreaPoly>,
    keys: Vec<(usize, usize)>,
    index: GridIndex,
    nodes: u64,
    cells: u64,
    max_leaf: usize,
    depth_cap_leaves: u64,
    probe_fallbacks: BTreeMap<String, u64>,
}

fn f32r(v: f64) -> f64 {
    v as f32 as f64
}

fn v12(p: P) -> Node {
    esf::xyv(p[0] as f32, p[1] as f32)
}

/// True when the segment has a positive-length overlap with the OPEN interior of the box.
fn crosses_open(a: P, b: P, lo: P, hi: P) -> bool {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [(-dx, a[0] - lo[0]), (dx, hi[0] - a[0]), (-dy, a[1] - lo[1]), (dy, hi[1] - a[1])] {
        if p == 0.0 {
            if q <= 0.0 {
                return false;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    t0 < t1
}

impl<'a, 'b> Builder<'a, 'b> {
    fn code(&self, a: u32, b: u32) -> u32 {
        let r = self.od.directed[&(a, b)];
        (r.0 << 16) | r.1
    }

    fn intersects(poly: &AreaPoly, p: P) -> bool {
        poly.contains(p) || poly.boundary_distance(p) <= 1e-9
    }

    fn strict_hits(&self, p: P) -> Vec<usize> {
        self.index.at(p).into_iter().map(|i| i as usize).filter(|&i| self.polys[i].strictly_contains(p)).collect()
    }

    fn region_at(&self, pt: P, prefer: &HashSet<u32>) -> Result<u32> {
        let strict = self.strict_hits(pt);
        if strict.len() == 1 {
            let k = self.keys[strict[0]];
            return Ok(((k.0 as u32) << 16) | k.1 as u32);
        }
        let mut hits: Vec<usize> =
            self.index.at(pt).into_iter().map(|i| i as usize).filter(|&i| Self::intersects(&self.polys[i], pt)).collect();
        if hits.is_empty() {
            bail!("regions atlas: {}: probe {pt:?} is in no output area", self.od.src.name);
        }
        if hits.len() > 1 {
            let code_of = |i: usize| ((self.keys[i].0 as u32) << 16) | self.keys[i].1 as u32;
            let pref: Vec<usize> = hits.iter().copied().filter(|&i| prefer.contains(&code_of(i))).collect();
            if !pref.is_empty() {
                hits = pref;
            }
        }
        let best = if hits.len() > 1 {
            *hits.iter().min_by(|&&x, &&y| self.polys[x].area.partial_cmp(&self.polys[y].area).unwrap()).unwrap()
        } else {
            hits[0]
        };
        Ok(((self.keys[best].0 as u32) << 16) | self.keys[best].1 as u32)
    }

    fn node(&mut self, lo: P, hi: P, idx: &[usize], depth: usize) -> Result<Vec<Node>> {
        self.nodes += 1;
        let idx: Vec<usize> = idx.iter().copied().filter(|&e| crosses_open(self.a[e], self.b[e], lo, hi)).collect();
        if idx.len() > SPLIT_THRESHOLD && depth < MAX_DEPTH {
            let mx = f32r((lo[0] + hi[0]) / 2.0);
            let my = f32r((lo[1] + hi[1]) / 2.0);
            let quads: [(&str, P, P); 4] = [
                ("bottom_left", [lo[0], lo[1]], [mx, my]),
                ("top_left", [lo[0], my], [mx, hi[1]]),
                ("bottom_right", [mx, lo[1]], [hi[0], my]),
                ("top_right", [mx, my], [hi[0], hi[1]]),
            ];
            let mut kids = Vec::new();
            for (name, clo, chi) in quads {
                let sub = self.node(clo, chi, &idx, depth + 1)?;
                let mut children = vec![v12(clo), v12(chi)];
                children.extend(sub);
                kids.push(Node::record(name, 0, children));
            }
            return Ok(kids);
        }
        if idx.len() > SPLIT_THRESHOLD {
            self.depth_cap_leaves += 1;
        }
        self.cells += 1;
        self.max_leaf = self.max_leaf.max(idx.len());
        Ok(vec![self.cell(lo, hi, &idx)?])
    }

    fn cell(&mut self, lo: P, hi: P, idx: &[usize]) -> Result<Node> {
        let (mx, my) = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0);
        let mut cands: Vec<P> = vec![lo, [lo[0], my], [mx, lo[1]], [mx, my], [lo[0], lo[1] + (hi[1] - lo[1]) / 3.0]];
        for i in 1..8 {
            for j in 1..8 {
                cands.push([lo[0] + (hi[0] - lo[0]) * i as f64 / 8.0, lo[1] + (hi[1] - lo[1]) * j as f64 / 8.0]);
            }
        }
        let mut probe: Option<P> = None;
        let mut fallback: Option<P> = None;
        for (k, c) in cands.iter().enumerate() {
            let c = [f32r(c[0]), f32r(c[1])];
            let far_enough = idx.iter().all(|&e| {
                let (a, b) = (self.a[e], self.b[e]);
                let d = [b[0] - a[0], b[1] - a[1]];
                let l2 = d[0].hypot(d[1]);
                let cr = d[0] * (c[1] - a[1]) - d[1] * (c[0] - a[0]);
                cr.abs() / l2 > PROBE_LINE_EPS
            });
            if !far_enough {
                continue;
            }
            if self.strict_hits(c).len() != 1 {
                if fallback.is_none() {
                    fallback = Some(c);
                }
                continue;
            }
            probe = Some(c);
            if k != 0 {
                *self.probe_fallbacks.entry(k.to_string()).or_insert(0) += 1;
            }
            break;
        }
        if probe.is_none() {
            if let Some(f) = fallback {
                probe = Some(f);
                *self.probe_fallbacks.entry("not_strictly_inside_one_area".to_string()).or_insert(0) += 1;
            }
        }
        let Some(probe) = probe else {
            bail!("regions atlas: {}: no usable probe point in cell {lo:?}-{hi:?}", self.od.src.name);
        };
        let mut segs: Vec<u32> = Vec::with_capacity(idx.len() * 4);
        let mut prefer: HashSet<u32> = HashSet::new();
        for &e in idx {
            let (mut a, mut b) = self.edges[e];
            let (pa, pb) = (self.a[e], self.b[e]);
            let d = [pb[0] - pa[0], pb[1] - pa[1]];
            let cr = d[0] * (probe[1] - pa[1]) - d[1] * (probe[0] - pa[0]);
            if cr < 0.0 {
                std::mem::swap(&mut a, &mut b);
            }
            let left = self.code(a, b);
            let right = self.code(b, a);
            prefer.insert(left);
            prefer.insert(right);
            segs.extend([a, b, right, if self.with_left { left } else { u32::MAX }]);
        }
        let ab0 = self.region_at(probe, &prefer)?;
        Ok(Node::record("cell", 0, vec![v12(probe), esf::u32v(ab0), esf::arr_u32(&segs)]))
    }
}

/// Build the new `query_info` record. `proto` is the source record (its version is kept).
pub fn build_query(od: &OutData, with_left: bool, proto: &Node) -> Result<(Node, QueryStats)> {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for &(a, b) in &od.directed_order {
        if od.directed.contains_key(&(b, a)) && !seen.contains(&(a, b)) {
            edges.push((a, b));
            seen.insert((a, b));
            seen.insert((b, a));
        }
    }
    let a: Vec<P> = edges.iter().map(|e| widen(od.coords[e.0 as usize])).collect();
    let b: Vec<P> = edges.iter().map(|e| widen(od.coords[e.1 as usize])).collect();
    let mut polys = Vec::new();
    let mut keys = Vec::new();
    for (ri, areas) in od.regions.iter().enumerate() {
        for oa in areas {
            polys.push(od.poly_of(oa)?);
            keys.push((ri, oa.local));
        }
    }
    let w = od.clip.ctx.world_out;
    let world: Rect64 = [widen(w[0]), widen(w[1])];
    let index = GridIndex::new(world, polys.iter().map(|p| p.bbox).collect());
    let mut bld = Builder {
        od,
        with_left,
        edges,
        a,
        b,
        polys,
        keys,
        index,
        nodes: 0,
        cells: 0,
        max_leaf: 0,
        depth_cap_leaves: 0,
        probe_fallbacks: BTreeMap::new(),
    };
    let all: Vec<usize> = (0..bld.edges.len()).collect();
    let kids = bld.node(world[0], world[1], &all, 0)?;
    let mut children = vec![v12(world[0]), v12(world[1])];
    children.extend(kids);
    let node = Node::record("node", 0, children);
    let version = proto.version().unwrap_or(0);
    let q = Node::record("query_info", version, vec![esf::u32v(bld.nodes as u32), esf::u32v(bld.cells as u32), node]);
    Ok((
        q,
        QueryStats {
            nodes: bld.nodes,
            cells: bld.cells,
            max_leaf: bld.max_leaf,
            depth_cap_leaves: bld.depth_cap_leaves,
            probe_fallbacks: bld.probe_fallbacks,
        },
    ))
}
