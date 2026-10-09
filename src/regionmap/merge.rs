//! Union of areas of one region and carving an area out into a new region (reference implementation
//! `work/exp5/regions_step.py`, stages A and B).
//!
//! * Union: the rings of the selected areas are joined by cancelling the edge pairs they share (an internal edge is
//!   the same vertex pair in opposite directions); the remaining directed edges form one counter-clockwise outer ring
//!   plus clockwise hole rings. Vertex ids are reused, the union becomes area 0, the other areas keep their order
//!   behind it, and every connectivity code of the file is re-pointed.
//! * Carve: an area of a backdrop region (e.g. `unexplorable` area 31) moves verbatim into a NEW land region that is
//!   appended LAST (the node layout of an unsettled region: no settlement, tail -1). The source's later areas shift
//!   down by one and every connectivity code of the file is re-pointed.
//!
//! Neither operation touches `query_info` (no access to the tree): rebuild it afterwards with
//! [`RegionData::rebuild_query_index`].

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, anyhow, bail, ensure};

use super::geometry::{P, ring_self_intersects, signed_area2};
use super::regions::{Area, NONE, Outline, Region, RegionData, code};

impl RegionData {
    /// Re-point every connectivity code of the file with `map` (old code -> new code).
    fn remap_codes(&mut self, map: impl Fn(u32) -> u32) {
        for r in &mut self.regions {
            for a in &mut r.areas {
                for o in &mut a.outlines {
                    let codes = Self::edge_codes(o);
                    let new: Vec<u32> = codes.iter().map(|&c| if c == NONE { c } else { map(c) }).collect();
                    if new != codes {
                        o.runs = Self::runs_from_codes(&new);
                    }
                }
            }
        }
    }

    /// Union the areas `areas` of region `region` into ONE area placed first (index 0) in the region; the other areas
    /// follow in their old order. Flags and component are those of `areas[0]` (call [`RegionData::set_passable`]
    /// afterwards). Holes of the selected areas are carried. Fails when the union is not one simple polygon (pinch
    /// vertices, several outer parts, self-intersection). Returns the new area index (always 0).
    pub fn merge_areas(&mut self, region: usize, areas: &[usize]) -> Result<usize> {
        let reg = self.regions.get(region).ok_or_else(|| anyhow!("no region {region}"))?;
        let key = reg.key.clone();
        ensure!(areas.len() >= 2, "{key}: merging needs at least two areas");
        let selected: HashSet<usize> = areas.iter().copied().collect();
        ensure!(selected.len() == areas.len(), "{key}: duplicate areas in {areas:?}");
        ensure!(areas.iter().all(|&a| a < reg.areas.len()), "{key}: area index out of range in {areas:?}");

        // directed edges of the selected areas
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        let mut order: Vec<(u32, u32)> = Vec::new();
        for &ai in areas {
            for o in &reg.areas[ai].outlines {
                let n = o.ring.len();
                for (k, c) in Self::edge_codes(o).into_iter().enumerate() {
                    let e = (o.ring[k], o.ring[(k + 1) % n]);
                    ensure!(edges.insert(e, c).is_none(), "{key}: directed edge {}->{} occurs twice", e.0, e.1);
                    order.push(e);
                }
            }
        }
        let is_merged = |c: u32| c != NONE && (c >> 16) as usize == region && selected.contains(&((c & 0xFFFF) as usize));
        let boundary: Vec<(u32, u32)> = order.iter().copied().filter(|&(a, b)| !edges.contains_key(&(b, a))).collect();
        for e in &boundary {
            ensure!(!is_merged(edges[e]), "{key}: boundary edge {}->{} names a merged area", e.0, e.1);
        }
        // the internal edges must name each other's areas, otherwise the file is inconsistent already
        let mut next: BTreeMap<u32, u32> = BTreeMap::new();
        for &(a, b) in &boundary {
            ensure!(next.insert(a, b).is_none(), "{key}: pinch vertex {a} in the union");
        }

        // new index of every old area of the region: union 0, the rest behind it in order
        let mut new_index: HashMap<usize, usize> = HashMap::new();
        let mut rank = 1;
        for ai in 0..reg.areas.len() {
            if selected.contains(&ai) {
                new_index.insert(ai, 0);
            } else {
                new_index.insert(ai, rank);
                rank += 1;
            }
        }
        let remap = |c: u32| -> u32 {
            if c != NONE && (c >> 16) as usize == region {
                code(region, new_index[&((c & 0xFFFF) as usize)])
            } else {
                c
            }
        };

        // boundary cycles, each started at its smallest vertex id
        let mut cycles: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        let mut seen: HashSet<u32> = HashSet::new();
        for (&start, _) in next.iter() {
            if seen.contains(&start) {
                continue;
            }
            let (mut ring, mut codes) = (vec![start], Vec::new());
            seen.insert(start);
            let mut v = start;
            loop {
                let w = next[&v];
                codes.push(remap(edges[&(v, w)]));
                if w == start {
                    break;
                }
                ensure!(seen.insert(w), "{key}: the union boundary revisits vertex {w}");
                ring.push(w);
                v = w;
            }
            cycles.push((ring, codes));
        }
        let mut outer: Option<(Vec<u32>, Vec<u32>)> = None;
        let mut holes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for (ring, codes) in cycles {
            let pts: Vec<P> = ring.iter().map(|&i| self.v(i)).collect();
            ensure!(!ring_self_intersects(&pts), "{key}: the union ring starting at vertex {} intersects itself", ring[0]);
            if signed_area2(&pts) > 0.0 {
                ensure!(outer.is_none(), "{key}: the selected areas do not join into one area (several outer rings)");
                outer = Some((ring, codes));
            } else {
                holes.push((ring, codes));
            }
        }
        let Some(outer) = outer else { bail!("{key}: the union has no outer ring") };

        let build = |(mut ring, mut codes): (Vec<u32>, Vec<u32>), is_outer: bool| -> Outline {
            // a run always starts at edge 0
            let n = ring.len();
            let start = (0..n).find(|&k| codes[(k + n - 1) % n] != codes[k]).unwrap_or(0);
            ring.rotate_left(start);
            codes.rotate_left(start);
            Outline { outer: is_outer, runs: Self::runs_from_codes(&codes), ring }
        };
        let mut outlines = vec![build(outer, true)];
        outlines.extend(holes.into_iter().map(|h| build(h, false)));
        let faces = self.triangulate_area(&outlines.iter().map(|o| o.ring.as_slice()).collect::<Vec<_>>());
        let template = &reg.areas[areas[0]];
        let union = Area {
            land_bridge: template.land_bridge,
            channel_coast: template.channel_coast,
            passable: template.passable,
            adjoining_id: template.adjoining_id,
            faces,
            outlines,
            impassable_id: template.impassable_id,
            island_id: template.island_id,
        };

        // restructure the region, then re-point the whole file
        let old: Vec<Area> = std::mem::take(&mut self.regions[region].areas);
        let mut restructured = vec![union];
        restructured.extend(old.into_iter().enumerate().filter(|(i, _)| !selected.contains(i)).map(|(_, a)| a));
        self.regions[region].areas = restructured;
        self.remap_codes(remap);
        Ok(0)
    }

    /// Move area `area` of `source` verbatim into a NEW land region `new_key` appended last (index = old region
    /// count): no settlement, tail -1 like every unsettled region. The source's later areas shift down by one and every
    /// connectivity code is re-pointed. Returns the new region index.
    pub fn carve_area(&mut self, source: usize, area: usize, new_key: &str) -> Result<usize> {
        let src = self.regions.get(source).ok_or_else(|| anyhow!("no region {source}"))?;
        let key = src.key.clone();
        ensure!(src.kind == "land", "{key} is a {} region: only land areas can be carved out", src.kind);
        ensure!(area < src.areas.len(), "{key}: no area {area}");
        ensure!(src.areas.len() > 1, "{key}: carving its only area would leave the region empty");
        ensure!(self.index(new_key).is_err(), "region {new_key} already exists");
        let new_index = self.regions.len();
        ensure!(new_index < 0xFFFF, "regions.esf cannot hold more than 65535 regions");
        let remap = |c: u32| -> u32 {
            if (c >> 16) as usize != source {
                return c;
            }
            match (c & 0xFFFF) as usize {
                a if a == area => code(new_index, 0),
                a if a > area => code(source, a - 1),
                _ => c,
            }
        };
        self.remap_codes(remap);
        let moved = self.regions[source].areas.remove(area);
        self.regions.push(Region { key: new_key.to_string(), kind: "land".to_string(), areas: vec![moved], tail: -1, settlement: None });
        Ok(new_index)
    }
}
