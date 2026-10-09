//! Typed view of the political layer of an ETW/NTW `regions.esf` (`region_data`, `theatres_and_region_keys`,
//! `query_info`) with the operations the region editor needs: splitting a region along a polyline and rebuilding the
//! point-location index. Everything else in the file (climate, groundtypes, mountains, heights, bridges, theatres)
//! is left untouched in the ESF tree.
//!
//! Format facts established against the vanilla NTW `nap_europe` map (see work/research/qcheck.py):
//! * `region_data.vertices` is one shared XY pool; neighbouring regions reference the same vertex indices.
//! * region item = [key s16, kind s8 ("land"|"sea"|"river"), bbox min XY, bbox max XY, areas[], i32,
//!   optional settlement_and_slots]; area item = [land bridge, channel coast, passable, bbox min, bbox max,
//!   u16 adjoining-passable-area id, faces { u32[] triangles, clockwise }, outlines[], u16 impassable-area id
//!   (65535 = passable), u16 island id]; outline item = [bool outer, bbox min, bbox max, u32[] ring, connectivity[]].
//! * connectivity item = [u32 (neighbour region << 16 | neighbour area), u32 from, u32 to]: ring edges k->k+1 for
//!   k in from..=to border that neighbour; the closing edge belongs to the run ending at n-1; the edge joining two
//!   runs belongs to the preceding run. The owner is on the LEFT of every ring edge.
//! * `query_info` is a quadtree of cells [XY probe, u32 region<<16|area at the probe, u32[] (v1, v2, right, left)...]
//!   listing every ring edge with a positive-length overlap with the cell interior (each physical edge once).

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail, ensure};

use super::geometry::{
    P, Rect, earcut_indices, point_in_ring, point_segment_distance, polygon_area, ring_self_intersects,
    segment_overlaps_rect, signed_area2,
};
use crate::esf::{self, Node, Value};

pub const NONE: u32 = 0xFFFF_FFFF;

#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// neighbour `region << 16 | area`, or NONE
    pub code: u32,
    pub from: u32,
    pub to: u32,
}

#[derive(Clone, Debug)]
pub struct Outline {
    pub outer: bool,
    pub ring: Vec<u32>,
    pub runs: Vec<Run>,
}

#[derive(Clone, Debug)]
pub struct Area {
    pub land_bridge: bool,
    pub channel_coast: bool,
    pub passable: bool,
    pub adjoining_id: u16,
    pub faces: Vec<[u32; 3]>,
    pub outlines: Vec<Outline>,
    pub impassable_id: u16,
    pub island_id: u16,
}

#[derive(Clone, Debug)]
pub struct Region {
    pub key: String,
    pub kind: String,
    pub areas: Vec<Area>,
    pub tail: i32,
    /// raw `settlement_and_slots` record (absent for sea / river / background regions)
    pub settlement: Option<Node>,
}

/// A transition area of a theatre (`theatres[].transition_areas[]`): the id and the world rectangle of the window
/// through which armies leave the theatre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionArea {
    pub id: u32,
    pub rect: Rect,
}

/// A theatre of regions.esf: key, playable rectangle (stored twice in the file: `theatres[]` and
/// `theatres_and_region_keys[]`) and transition areas.
#[derive(Clone, Debug, PartialEq)]
pub struct Theatre {
    pub key: String,
    pub rect: Rect,
    pub transitions: Vec<TransitionArea>,
    /// set by [`RegionData::extend_theatre`] (not stored): the rectangle grew in this session, so the sea and
    /// backdrop areas around it must be clipped ([`RegionData::clip_at_theatres`])
    pub extended: bool,
}

/// One transition area that [`RegionData::extend_theatre`] moved with its edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionMove {
    pub id: u32,
    pub old: Rect,
    pub new: Rect,
}

/// Result of [`RegionData::extend_theatre`].
#[derive(Clone, Debug, PartialEq)]
pub struct TheatreChange {
    pub theatre: String,
    pub old_rect: Rect,
    pub new_rect: Rect,
    pub moved: Vec<TransitionMove>,
}

#[derive(Clone, Debug)]
pub struct RegionData {
    pub vertices: Vec<[f32; 2]>,
    pub regions: Vec<Region>,
    /// theatre rectangles and transition areas (empty when the file has no `theatres`); written back by `store`
    pub theatres: Vec<Theatre>,
}

pub fn code(region: usize, area: usize) -> u32 {
    ((region as u32) << 16) | area as u32
}

fn u16_of(n: &Node) -> Result<u16> {
    match n {
        Node::Value(Value::U16(v)) => Ok(*v),
        other => bail!("expected u16, found {other:?}"),
    }
}

fn is_named(n: &Node, name: &str) -> bool {
    n.name() == Some(name)
}

fn child<'a>(items: &'a [Node], name: &str) -> Result<&'a Node> {
    items
        .iter()
        .find(|c| is_named(c, name))
        .ok_or_else(|| anyhow!("child {name} missing"))
}

impl RegionData {
    pub fn parse(root: &Node) -> Result<Self> {
        let rd = root.get("region_data")?;
        let vertices = rd.get("vertices")?.children()?[0].xy_array()?;
        let mut regions = Vec::new();
        for item in rd.get("regions")?.items()? {
            let key = item[0].as_str()?.to_string();
            let kind = item[1].as_str()?.to_string();
            let mut areas = Vec::new();
            for a in child(item, "areas")?.items()? {
                ensure!(a.len() == 10, "region {key}: area with {} children", a.len());
                let faces = child(a, "faces")?.children()?[0]
                    .u32_array()?
                    .chunks_exact(3)
                    .map(|t| [t[0], t[1], t[2]])
                    .collect();
                let mut outlines = Vec::new();
                for o in child(a, "outlines")?.items()? {
                    ensure!(o.len() == 5, "region {key}: outline with {} children", o.len());
                    let runs = o[4]
                        .items()?
                        .iter()
                        .map(|c| {
                            Ok(Run {
                                code: c[0].as_u32()?,
                                from: c[1].as_u32()?,
                                to: c[2].as_u32()?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    outlines.push(Outline {
                        outer: o[0].as_bool()?,
                        ring: o[3].u32_array()?,
                        runs,
                    });
                }
                areas.push(Area {
                    land_bridge: a[0].as_bool()?,
                    channel_coast: a[1].as_bool()?,
                    passable: a[2].as_bool()?,
                    adjoining_id: u16_of(&a[5])?,
                    faces,
                    outlines,
                    impassable_id: u16_of(&a[8])?,
                    island_id: u16_of(&a[9])?,
                });
            }
            let tail_at = 1 + item.iter().position(|c| is_named(c, "areas")).unwrap();
            let tail = item[tail_at].as_i32()?;
            let settlement = item.iter().find(|c| is_named(c, "settlement_and_slots")).cloned();
            regions.push(Region {
                key,
                kind,
                areas,
                tail,
                settlement,
            });
        }
        Ok(Self { vertices, regions, theatres: parse_theatres(root)? })
    }

    pub fn index(&self, key: &str) -> Result<usize> {
        self.regions
            .iter()
            .position(|r| r.key == key)
            .ok_or_else(|| anyhow!("region {key} is not in regions.esf"))
    }

    pub fn v(&self, i: u32) -> P {
        let p = self.vertices[i as usize];
        [p[0] as f64, p[1] as f64]
    }

    fn ring_points(&self, ring: &[u32]) -> Vec<P> {
        ring.iter().map(|&i| self.v(i)).collect()
    }

    fn bbox_of(&self, rings: impl Iterator<Item = u32>) -> ([f32; 2], [f32; 2]) {
        let mut lo = [f32::MAX, f32::MAX];
        let mut hi = [f32::MIN, f32::MIN];
        for i in rings {
            let p = self.vertices[i as usize];
            lo[0] = lo[0].min(p[0]);
            lo[1] = lo[1].min(p[1]);
            hi[0] = hi[0].max(p[0]);
            hi[1] = hi[1].max(p[1]);
        }
        (lo, hi)
    }

    /// Serialise `region_data` back into `root` (vertices + regions; the bounds scalars are kept).
    pub fn store(&self, root: &mut Node) -> Result<()> {
        let rd = root.get_mut("region_data")?;
        rd.get_mut("vertices")?.children_mut()?[0] = esf::arr_xy(&self.vertices);
        let mut items = Vec::with_capacity(self.regions.len());
        for r in &self.regions {
            let mut areas = Vec::with_capacity(r.areas.len());
            for a in &r.areas {
                let (alo, ahi) = self.bbox_of(a.outlines.iter().flat_map(|o| o.ring.iter().copied()));
                let faces: Vec<u32> = a.faces.iter().flat_map(|t| t.iter().copied()).collect();
                let outlines = a
                    .outlines
                    .iter()
                    .map(|o| {
                        let (olo, ohi) = self.bbox_of(o.ring.iter().copied());
                        let runs = o
                            .runs
                            .iter()
                            .map(|r| vec![esf::u32v(r.code), esf::u32v(r.from), esf::u32v(r.to)])
                            .collect();
                        vec![
                            esf::b(o.outer),
                            esf::xyv(olo[0], olo[1]),
                            esf::xyv(ohi[0], ohi[1]),
                            esf::arr_u32(&o.ring),
                            Node::array("connectivity", 0, runs),
                        ]
                    })
                    .collect();
                areas.push(vec![
                    esf::b(a.land_bridge),
                    esf::b(a.channel_coast),
                    esf::b(a.passable),
                    esf::xyv(alo[0], alo[1]),
                    esf::xyv(ahi[0], ahi[1]),
                    esf::u16v(a.adjoining_id),
                    Node::record("faces", 0, vec![esf::arr_u32(&faces)]),
                    Node::array("outlines", 0, outlines),
                    esf::u16v(a.impassable_id),
                    esf::u16v(a.island_id),
                ]);
            }
            let (lo, hi) = self.bbox_of(
                r.areas
                    .iter()
                    .flat_map(|a| a.outlines.iter().flat_map(|o| o.ring.iter().copied())),
            );
            let mut item = vec![
                esf::s16(r.key.clone()),
                esf::s8(r.kind.clone()),
                esf::xyv(lo[0], lo[1]),
                esf::xyv(hi[0], hi[1]),
                Node::array("areas", 0, areas),
                esf::i32v(r.tail),
            ];
            if let Some(s) = &r.settlement {
                item.push(s.clone());
            }
            items.push(item);
        }
        *rd.get_mut("regions")? = Node::array("regions", 0, items);
        self.store_theatres(root)
    }

    /// Write the theatre rectangles (both copies) and transition areas back into `theatres[]` /
    /// `theatres_and_region_keys[]`. Files without theatres are left alone.
    fn store_theatres(&self, root: &mut Node) -> Result<()> {
        if self.theatres.is_empty() {
            return Ok(());
        }
        for th in &self.theatres {
            let mut found = 0;
            for item in root.get_mut("theatres")?.items_mut()? {
                let children = theatre_children_mut(item);
                if children.iter().find_map(|n| theatre_key(n)) != Some(th.key.as_str()) {
                    continue;
                }
                found += 1;
                set_rect_scalars(children, th.rect)?;
                for ch in children.iter_mut().filter(|n| is_named(n, "transition_areas")) {
                    for ta in ch.items_mut()? {
                        let id = ta[0].as_u32()?;
                        let t = th
                            .transitions
                            .iter()
                            .find(|t| t.id == id)
                            .ok_or_else(|| anyhow!("theatre {}: transition area {id} vanished", th.key))?;
                        ta[1] = esf::xyv(t.rect[0][0], t.rect[0][1]);
                        ta[2] = esf::xyv(t.rect[1][0], t.rect[1][1]);
                    }
                }
            }
            ensure!(found == 1, "theatre {} is listed {found} times in theatres[]", th.key);
            let mut found = 0;
            for item in root.get_mut("theatres_and_region_keys")?.items_mut()? {
                let children = theatre_children_mut(item);
                if children.iter().find_map(|n| theatre_key(n)) != Some(th.key.as_str()) {
                    continue;
                }
                found += 1;
                set_rect_scalars(children, th.rect)?;
            }
            ensure!(found == 1, "theatre {} is listed {found} times in theatres_and_region_keys[]", th.key);
        }
        Ok(())
    }

    /// Every ring edge once: (v1, v2, owner code, neighbour code) in ring direction (owner on the left).
    fn edges(&self) -> Vec<(u32, u32, u32, u32)> {
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut out = Vec::new();
        for (ri, r) in self.regions.iter().enumerate() {
            for (ai, a) in r.areas.iter().enumerate() {
                for o in &a.outlines {
                    let n = o.ring.len();
                    for run in &o.runs {
                        for k in run.from..=run.to {
                            let v1 = o.ring[k as usize];
                            let v2 = o.ring[(k as usize + 1) % n];
                            let key = (v1.min(v2), v1.max(v2));
                            if seen.insert(key) {
                                out.push((v1, v2, code(ri, ai), run.code));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Summed outline length shared with each neighbouring region (by region index).
    pub fn border_lengths(&self, ri: usize) -> std::collections::BTreeMap<usize, f64> {
        let mut out = std::collections::BTreeMap::new();
        for a in &self.regions[ri].areas {
            for o in &a.outlines {
                let n = o.ring.len();
                for run in o.runs.iter().filter(|r| r.code != NONE) {
                    let nb = (run.code >> 16) as usize;
                    if nb == ri {
                        continue;
                    }
                    for k in run.from as usize..=run.to as usize {
                        let (p, q) = (self.v(o.ring[k % n]), self.v(o.ring[(k + 1) % n]));
                        *out.entry(nb).or_insert(0.0) += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                    }
                }
            }
        }
        out
    }

    /// Regions sharing a border with region `ri` (from connectivity codes, excluding `ri` itself).
    pub fn adjacent(&self, ri: usize) -> std::collections::BTreeSet<usize> {
        self.regions[ri]
            .areas
            .iter()
            .flat_map(|a| a.outlines.iter().flat_map(|o| o.runs.iter()))
            .filter(|r| r.code != NONE)
            .map(|r| (r.code >> 16) as usize)
            .filter(|&n| n != ri)
            .collect()
    }

    /// Euclidean distance from `pt` to the nearest outline edge of region `ri`.
    pub fn distance_to_region(&self, ri: usize, pt: P) -> f64 {
        let mut best = f64::MAX;
        for a in &self.regions[ri].areas {
            for o in &a.outlines {
                let n = o.ring.len();
                for k in 0..n {
                    let d = point_segment_distance(pt, self.v(o.ring[k]), self.v(o.ring[(k + 1) % n]));
                    best = best.min(d);
                }
            }
        }
        best
    }

    /// Index of the area of region `ri` that contains `pt` (an outer ring contains it and no hole ring does).
    pub fn area_at(&self, ri: usize, pt: P) -> Option<usize> {
        self.regions[ri].areas.iter().position(|a| {
            let mut inside = false;
            for o in &a.outlines {
                if point_in_ring(pt, &self.ring_points(&o.ring)) {
                    if o.outer {
                        inside = true;
                    } else {
                        return false;
                    }
                }
            }
            inside
        })
    }

    /// True when `pt` lies inside region `ri` (an outer ring of one of its areas contains it and no hole ring does).
    /// Single-region version of [`RegionData::locate`], without allocating per point beyond the ring being tested.
    pub fn in_region(&self, ri: usize, pt: P) -> bool {
        self.area_at(ri, pt).is_some()
    }

    /// Gives the settlement-less land region `ri` (a wilderness region) a settlement: the settlement record of
    /// `donor` translated to `spec.center` with the slot keys renamed to `settlement:<key>:<spec.name>:<type>:<n>`.
    /// `tail` becomes the donor's: it is the id of the theatre that lists the region (1 America, 2 Europe, 3 India;
    /// measured on all 137 settled regions of the ETW world map), -1 for wilderness. The centre must lie in a
    /// passable area. Geometry is untouched. Does not touch the theatre region list (see [`add_theatre_region_key`]).
    pub fn attach_settlement(&mut self, ri: usize, donor: usize, spec: &SettlementSpec) -> Result<()> {
        ensure!(ri != donor, "region {} cannot be its own donor", self.regions[ri].key);
        let region = &self.regions[ri];
        ensure!(region.kind == "land", "region {} is a {} region, only land regions can get a settlement", region.key, region.kind);
        ensure!(region.settlement.is_none(), "region {} already has a settlement", region.key);
        let template = self.regions[donor]
            .settlement
            .as_ref()
            .ok_or_else(|| anyhow!("donor region {} has no settlement", self.regions[donor].key))?;
        let settlement = relocate_settlement(template, &self.regions[donor].key, &region.key, spec)?;
        let tail = self.regions[donor].tail;
        ensure!(tail >= 0, "donor region {} has no theatre id in its tail", self.regions[donor].key);
        let area = self
            .area_at(ri, [spec.center[0] as f64, spec.center[1] as f64])
            .ok_or_else(|| anyhow!("settlement position {:?} is not inside region {}", spec.center, region.key))?;
        ensure!(
            region.areas[area].passable,
            "settlement position {:?} lies in impassable area {area} of region {}; choose a point in a passable area",
            spec.center,
            region.key
        );
        let region = &mut self.regions[ri];
        region.settlement = Some(settlement);
        region.tail = tail;
        Ok(())
    }

    /// Innermost (region, area) whose outer rings contain `pt` and whose hole rings do not.
    pub fn locate(&self, pt: P) -> Option<u32> {
        let mut best: Option<(f64, u32)> = None;
        for (ri, r) in self.regions.iter().enumerate() {
            for (ai, a) in r.areas.iter().enumerate() {
                let mut inside = false;
                let mut size = f64::MAX;
                for o in &a.outlines {
                    let pts = self.ring_points(&o.ring);
                    if point_in_ring(pt, &pts) {
                        if o.outer {
                            inside = true;
                            size = size.min(polygon_area(&pts));
                        } else {
                            inside = false;
                            break;
                        }
                    }
                }
                if inside && best.is_none_or(|(s, _)| size < s) {
                    best = Some((size, code(ri, ai)));
                }
            }
        }
        best.map(|(_, c)| c)
    }

    /// Recompute every leaf cell of `query_info` from the current geometry (the quadtree shape is kept).
    pub fn rebuild_query_index(&self, root: &mut Node) -> Result<usize> {
        self.rebuild_query_index_within(root, None)
    }

    /// [`RegionData::rebuild_query_index`] restricted to the leaf cells that overlap one of `rects` (`(lo, hi)`
    /// world rectangles, the area a split changed); all other cells keep their stored content.
    pub fn rebuild_query_index_within(&self, root: &mut Node, rects: Option<&[(P, P)]>) -> Result<usize> {
        let edges = self.edges();
        let edge_pts: Vec<(P, P)> = edges.iter().map(|e| (self.v(e.0), self.v(e.1))).collect();
        let node = root.path_mut(&["query_info", "node"])?;
        let mut cells = 0usize;
        self.rebuild_node(node, &edges, &edge_pts, rects, &mut cells)?;
        Ok(cells)
    }

    fn rebuild_node(
        &self,
        node: &mut Node,
        edges: &[(u32, u32, u32, u32)],
        edge_pts: &[(P, P)],
        rects: Option<&[(P, P)]>,
        cells: &mut usize,
    ) -> Result<()> {
        let lo = node.children()?[0].as_xy()?;
        let hi = node.children()?[1].as_xy()?;
        let lo = [lo[0] as f64, lo[1] as f64];
        let hi = [hi[0] as f64, hi[1] as f64];
        for ch in node.children_mut()?.iter_mut().skip(2) {
            if is_named(ch, "cell") {
                if rects.is_some_and(|rs| !rs.iter().any(|(rlo, rhi)| lo[0] <= rhi[0] && hi[0] >= rlo[0] && lo[1] <= rhi[1] && hi[1] >= rlo[1])) {
                    continue;
                }
                let mut segs: Vec<u32> = Vec::new();
                let mut local: Vec<(P, P)> = Vec::new();
                for (e, (p, q)) in edges.iter().zip(edge_pts) {
                    if segment_overlaps_rect(*p, *q, lo, hi) {
                        segs.extend([e.0, e.1, e.3, e.2]);
                        local.push((*p, *q));
                    }
                }
                // probe: the lower-left corner unless an edge passes through it; then climb in halves
                let mut probe = lo;
                let mut step = (hi[1] - lo[1]) / 2.0;
                while local
                    .iter()
                    .any(|(p, q)| point_segment_distance(probe, *p, *q) < 1e-4)
                    && step > 1e-3
                {
                    probe = [lo[0], probe[1] + step];
                    step /= 2.0;
                }
                let at = self.locate(probe).unwrap_or(NONE);
                *ch = Node::record(
                    "cell",
                    0,
                    vec![
                        esf::xyv(probe[0] as f32, probe[1] as f32),
                        esf::u32v(at),
                        esf::arr_u32(&segs),
                    ],
                );
                *cells += 1;
            } else {
                self.rebuild_node(ch, edges, edge_pts, rects, cells)?;
            }
        }
        Ok(())
    }

    /// Per-edge neighbour codes of a ring (index k = edge k -> k+1), from its runs.
    pub(crate) fn edge_codes(o: &Outline) -> Vec<u32> {
        let n = o.ring.len();
        let mut codes = vec![NONE; n];
        for run in &o.runs {
            for k in run.from..=run.to {
                codes[k as usize % n] = run.code;
            }
        }
        codes
    }

    /// Runs from per-edge neighbour codes (maximal consecutive edges with one neighbour, no wrap merge: the
    /// vanilla data always starts a run at edge 0).
    pub(crate) fn runs_from_codes(codes: &[u32]) -> Vec<Run> {
        let mut runs: Vec<Run> = Vec::new();
        for (k, &c) in codes.iter().enumerate() {
            match runs.last_mut() {
                Some(r) if r.code == c && r.to + 1 == k as u32 => r.to = k as u32,
                _ => runs.push(Run {
                    code: c,
                    from: k as u32,
                    to: k as u32,
                }),
            }
        }
        runs
    }

    /// Faces of an area (outer ring first, then holes) as clockwise triangles of vertex ids.
    pub(crate) fn triangulate_area(&self, rings: &[&[u32]]) -> Vec<[u32; 3]> {
        let pts: Vec<Vec<P>> = rings.iter().map(|r| self.ring_points(r)).collect();
        let flat: Vec<u32> = rings.iter().flat_map(|r| r.iter().copied()).collect();
        super::geometry::earcut_with_holes(&pts)
            .into_iter()
            .map(|t| [flat[t[0]], flat[t[1]], flat[t[2]]])
            .collect()
    }

    pub(crate) fn triangulate_ring(&self, ring: &[u32]) -> Vec<[u32; 3]> {
        let pts = self.ring_points(ring);
        earcut_indices(&pts)
            .into_iter()
            .map(|t| [ring[t[0]], ring[t[1]], ring[t[2]]])
            .collect()
    }

    /// Split area `area` of region `key` along the polyline `a` -> interior -> `b` (positions in the area's outer
    /// ring). The forward arc `a..=b` plus the polyline becomes the new region `new_key`; the rest keeps `key`.
    /// Neighbour connectivity is rewritten, faces are re-triangulated. Returns the new region index.
    pub fn split(&mut self, split: &Split) -> Result<usize> {
        let ri = self.index(&split.key)?;
        ensure!(self.index(&split.new_key).is_err(), "region {} already exists", split.new_key);
        let region = &self.regions[ri];
        let area = region
            .areas
            .get(split.area)
            .ok_or_else(|| anyhow!("{}: no area {}", split.key, split.area))?;
        ensure!(
            area.outlines.len() == 1 && area.outlines[0].outer,
            "{}: area {} must have exactly one outer ring",
            split.key,
            split.area
        );
        let ring = area.outlines[0].ring.clone();
        let codes = Self::edge_codes(&area.outlines[0]);
        let n = ring.len();
        let (a, b) = (split.a, split.b);
        ensure!(a < n && b < n && a != b, "split positions must be distinct ring positions");

        // new vertices for the interior points, p[0] next to a, p[last] next to b
        let base = self.vertices.len() as u32;
        self.vertices.extend(split.interior.iter().copied());
        let interior: Vec<u32> = (0..split.interior.len() as u32).map(|i| base + i).collect();

        // ring N (new region): forward arc a..=b then interior reversed; ring S: b..=a (wrapping) then interior
        let mut arc_n: Vec<usize> = Vec::new();
        let mut k = a;
        loop {
            arc_n.push(k);
            if k == b {
                break;
            }
            k = (k + 1) % n;
        }
        let mut arc_s: Vec<usize> = Vec::new();
        k = b;
        loop {
            arc_s.push(k);
            if k == a {
                break;
            }
            k = (k + 1) % n;
        }
        let new_index = self.regions.len();
        let orig_code = code(ri, split.area);
        let new_code = code(new_index, 0);

        let mut ring_n: Vec<u32> = arc_n.iter().map(|&k| ring[k]).collect();
        let mut codes_n: Vec<u32> = arc_n[..arc_n.len() - 1].iter().map(|&k| codes[k]).collect();
        ring_n.extend(interior.iter().rev());
        codes_n.extend(std::iter::repeat_n(orig_code, interior.len() + 1));

        let mut ring_s: Vec<u32> = arc_s.iter().map(|&k| ring[k]).collect();
        let mut codes_s: Vec<u32> = arc_s[..arc_s.len() - 1].iter().map(|&k| codes[k]).collect();
        ring_s.extend(interior.iter());
        codes_s.extend(std::iter::repeat_n(new_code, interior.len() + 1));
        ensure!(ring_n.len() == codes_n.len() && ring_s.len() == codes_s.len());
        // Both halves must stay simple rings with the source's orientation (a crossed cut would silently produce
        // bow-ties that every downstream index accepts).
        let orientation = signed_area2(&self.ring_points(&ring));
        for (label, r) in [("new", &ring_n), ("remaining", &ring_s)] {
            let pts = self.ring_points(r);
            ensure!(
                signed_area2(&pts) * orientation > 0.0,
                "{}: the cut leaves the {label} half with reversed orientation (cut points must run from the start vertex towards the end vertex)",
                split.key
            );
            ensure!(
                !ring_self_intersects(&pts),
                "{}: the cut makes the {label} half self-intersecting",
                split.key
            );
        }

        // neighbours: every edge of ring N that bordered (ri, area) now borders the new region
        let n_edges: HashSet<(u32, u32)> = (0..ring_n.len())
            .map(|k| {
                let (v1, v2) = (ring_n[k], ring_n[(k + 1) % ring_n.len()]);
                (v1.min(v2), v1.max(v2))
            })
            .collect();
        for (oi, other) in self.regions.iter_mut().enumerate() {
            for (ai, oa) in other.areas.iter_mut().enumerate() {
                if (oi, ai) == (ri, split.area) {
                    continue;
                }
                for o in oa.outlines.iter_mut() {
                    let mut oc = Self::edge_codes(o);
                    let m = o.ring.len();
                    let mut changed = false;
                    for k in 0..m {
                        if oc[k] == orig_code {
                            let (v1, v2) = (o.ring[k], o.ring[(k + 1) % m]);
                            if n_edges.contains(&(v1.min(v2), v1.max(v2))) {
                                oc[k] = new_code;
                                changed = true;
                            }
                        }
                    }
                    if changed {
                        o.runs = Self::runs_from_codes(&oc);
                    }
                }
            }
        }

        // the source area keeps ring S
        let faces_s = self.triangulate_ring(&ring_s);
        let faces_n = self.triangulate_ring(&ring_n);
        let src = &mut self.regions[ri].areas[split.area];
        src.outlines[0].ring = ring_s;
        src.outlines[0].runs = Self::runs_from_codes(&codes_s);
        src.faces = faces_s;
        let template = src.clone();

        let settlement = match &self.regions[ri].settlement {
            Some(s) => Some(relocate_settlement(s, &split.key, &split.new_key, &split.settlement)?),
            None => None,
        };
        let kind = self.regions[ri].kind.clone();
        let tail = self.regions[ri].tail;
        self.regions.push(Region {
            key: split.new_key.clone(),
            kind,
            areas: vec![Area {
                faces: faces_n,
                outlines: vec![Outline {
                    outer: true,
                    ring: ring_n,
                    runs: Self::runs_from_codes(&codes_n),
                }],
                ..template
            }],
            tail,
            settlement,
        });
        Ok(new_index)
    }
}

/// Parameters of a region split.
pub struct Split {
    pub key: String,
    pub area: usize,
    /// ring positions of the split end points (existing vertices)
    pub a: usize,
    pub b: usize,
    /// interior split points ordered from `a` towards `b`
    pub interior: Vec<[f32; 2]>,
    pub new_key: String,
    pub settlement: SettlementSpec,
}

/// Where the new region's settlement goes; slots are copied from the source settlement by translation.
pub struct SettlementSpec {
    pub name: String,
    pub center: [f32; 2],
}

fn rename_key(key: &str, old_region: &str, new_region: &str, new_name: &str) -> String {
    // "<kind>:<region>:<name>[:<rest>]"
    let mut parts: Vec<&str> = key.split(':').collect();
    if parts.len() >= 3 && parts[1] == old_region {
        parts[1] = new_region;
        parts[2] = new_name;
    }
    parts.join(":")
}

/// Clone a `settlement_and_slots` record to a new centre: keeps only the settlement slots, translates every
/// coordinate by the centre delta and renames the slot keys.
fn relocate_settlement(
    src: &Node,
    old_region: &str,
    new_region: &str,
    spec: &SettlementSpec,
) -> Result<Node> {
    let ch = src.children()?;
    let old = ch[0].as_xy()?;
    let d = [spec.center[0] - old[0], spec.center[1] - old[1]];
    let shift_xy = |n: &Node| -> Result<Node> {
        match n {
            Node::Value(Value::XY(v)) => Ok(esf::xyv(v[0] + d[0], v[1] + d[1])),
            Node::Value(Value::XYZ(v)) => Ok(esf::xyzv(v[0] + d[0], v[1], v[2] + d[1])),
            Node::Value(Value::Array { code: 0x4c, .. }) => {
                let pts: Vec<[f32; 2]> = n.xy_array()?.into_iter().map(|p| [p[0] + d[0], p[1] + d[1]]).collect();
                Ok(esf::arr_xy(&pts))
            }
            other => Ok(other.clone()),
        }
    };
    let mut out = Vec::with_capacity(ch.len());
    out.push(shift_xy(&ch[0])?);
    out.push(shift_xy(&ch[1])?);
    let slots = ch[2].items()?;
    let mut kept = Vec::new();
    for s in slots {
        if !s[0].as_str()?.starts_with("settlement:") {
            continue;
        }
        let mut item: Vec<Node> = s.iter().map(shift_xy).collect::<Result<_>>()?;
        item[0] = esf::s16(rename_key(s[0].as_str()?, old_region, new_region, &spec.name));
        kept.push(item);
    }
    ensure!(!kept.is_empty(), "source settlement has no settlement slots");
    out.push(Node::array("slot_descriptions", ch[2].version().unwrap_or(1), kept));
    out.extend(ch[3..].iter().cloned());
    Ok(Node::record("settlement_and_slots", src.version().unwrap_or(0), out))
}

/// Append `(key, settlement position)` to the theatre region-key list that already lists `like`.
pub fn add_theatre_region_key(root: &mut Node, like: &str, key: &str, pos: [f32; 2]) -> Result<()> {
    for item in root.get_mut("theatres_and_region_keys")?.items_mut()? {
        let theatre = &mut item[0];
        let keys = theatre.get_mut("region_keys")?;
        if keys.items()?.iter().any(|it| it[0].as_str().ok() == Some(like)) {
            keys.items_mut()?.push(vec![esf::s16(key), esf::xyv(pos[0], pos[1])]);
            return Ok(());
        }
    }
    bail!("no theatre lists region {like}")
}

/// Settlement centre of a region (`settlement_and_slots` child 0).
pub fn settlement_center(region: &Region) -> Result<[f32; 2]> {
    region
        .settlement
        .as_ref()
        .ok_or_else(|| anyhow!("{} has no settlement", region.key))?
        .children()?[0]
        .as_xy()
        .context("settlement centre")
}

/// Region keys in file order (what pathfinding.esf indexes by position).
pub fn keys(rd: &RegionData) -> HashMap<String, usize> {
    rd.regions.iter().enumerate().map(|(i, r)| (r.key.clone(), i)).collect()
}

// ------------------------------------------------------------------------------------------------- theatres

fn theatre_children(item: &[Node]) -> &[Node] {
    match item {
        [Node::Record { children, .. }] => children,
        _ => item,
    }
}

fn theatre_children_mut(item: &mut Vec<Node>) -> &mut Vec<Node> {
    if item.len() == 1 && matches!(item[0], Node::Record { .. }) {
        match &mut item[0] {
            Node::Record { children, .. } => children,
            _ => unreachable!(),
        }
    } else {
        item
    }
}

fn theatre_key(n: &Node) -> Option<&str> {
    match n {
        Node::Value(Value::String16(s)) | Node::Value(Value::String8(s)) => Some(s),
        _ => None,
    }
}

fn xy_value(n: &Node) -> Option<[f32; 2]> {
    match n {
        Node::Value(Value::XY(p)) => Some(*p),
        _ => None,
    }
}

/// Replace the first two XY scalars (min, max) of a theatre record.
fn set_rect_scalars(children: &mut [Node], rect: Rect) -> Result<()> {
    let mut at = children.iter_mut().filter(|n| xy_value(n).is_some());
    let lo = at.next().ok_or_else(|| anyhow!("theatre record has no min XY"))?;
    *lo = esf::xyv(rect[0][0], rect[0][1]);
    let hi = at.next().ok_or_else(|| anyhow!("theatre record has no max XY"))?;
    *hi = esf::xyv(rect[1][0], rect[1][1]);
    Ok(())
}

fn parse_theatres(root: &Node) -> Result<Vec<Theatre>> {
    let Some(list) = root.children()?.iter().find(|c| is_named(c, "theatres")) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for item in list.items()? {
        let children = theatre_children(item);
        let key = children
            .iter()
            .find_map(theatre_key)
            .ok_or_else(|| anyhow!("theatre record has no key"))?
            .to_string();
        let mut xy = children.iter().filter_map(xy_value);
        let (Some(lo), Some(hi)) = (xy.next(), xy.next()) else {
            bail!("theatre {key}: min/max XY missing");
        };
        let mut transitions = Vec::new();
        for ch in children.iter().filter(|n| is_named(n, "transition_areas")) {
            for ta in ch.items()? {
                ensure!(ta.len() >= 3, "theatre {key}: transition area with {} children", ta.len());
                let (Some(tlo), Some(thi)) = (xy_value(&ta[1]), xy_value(&ta[2])) else {
                    bail!("theatre {key}: transition area without min/max XY");
                };
                transitions.push(TransitionArea { id: ta[0].as_u32()?, rect: [tlo, thi] });
            }
        }
        out.push(Theatre { key, rect: [lo, hi], transitions, extended: false });
    }
    Ok(out)
}

/// A transition area counts as sitting on a theatre edge when it is within this many units of it.
const TRANSITION_EDGE_REACH: f32 = 5.0;

/// Which edge of `rect` (0 west, 1 east, 2 south, 3 north) the transition strip `t` lies along: a strip taller than
/// wide runs along a west/east edge, a wide one along a south/north edge; the nearest edge within reach wins.
fn transition_edge(rect: Rect, t: Rect) -> Option<usize> {
    let (w, h) = (t[1][0] - t[0][0], t[1][1] - t[0][1]);
    let gap = |axis: usize, at: f32| (t[0][axis] - at).max(at - t[1][axis]).max(0.0);
    let mut best: Option<(f32, usize)> = None;
    for (edge, axis, at) in [(0, 0, rect[0][0]), (1, 0, rect[1][0]), (2, 1, rect[0][1]), (3, 1, rect[1][1])] {
        let allowed = if axis == 0 { w <= h } else { h <= w };
        let d = gap(axis, at);
        if allowed && d <= TRANSITION_EDGE_REACH && best.is_none_or(|(b, _)| d < b) {
            best = Some((d, edge));
        }
    }
    best.map(|(_, e)| e)
}

/// Round a theatre outwards to the shipped map's two-unit pathfinding lattice.
/// Keeping the original cell boundaries preserves both corner-relative and fixed world vertices.
pub fn align_theatre_rect(old: Rect, requested: Rect) -> Rect {
    let mut aligned = requested;
    for axis in 0..2 {
        let origin = f64::from(old[0][axis]);
        aligned[0][axis] = (origin + ((f64::from(requested[0][axis]) - origin) / 2.0).floor() * 2.0) as f32;
        aligned[1][axis] = (origin + ((f64::from(requested[1][axis]) - origin) / 2.0).ceil() * 2.0) as f32;
    }
    aligned
}

impl RegionData {
    fn theatre(&self, key: &str) -> Result<&Theatre> {
        self.theatres
            .iter()
            .find(|t| t.key == key)
            .ok_or_else(|| anyhow!("theatre {key} is not in regions.esf"))
    }

    /// Playable rectangle of a theatre.
    pub fn theatre_rect(&self, theatre: &str) -> Result<Rect> {
        Ok(self.theatre(theatre)?.rect)
    }

    /// All theatre rectangles in file order.
    pub fn theatre_rects(&self) -> Vec<(String, Rect)> {
        self.theatres.iter().map(|t| (t.key.clone(), t.rect)).collect()
    }

    /// Transition areas of a theatre.
    pub fn transition_areas(&self, theatre: &str) -> Result<Vec<TransitionArea>> {
        Ok(self.theatre(theatre)?.transitions.clone())
    }

    /// Grow the playable rectangle of `theatre` to `rect` (both stored copies change on `store`). Transition areas
    /// that sit on an edge that moves slide with it, keeping their size. Edges snap outwards to whole pathfinding cells.
    pub fn extend_theatre(&mut self, theatre: &str, rect: Rect) -> Result<TheatreChange> {
        let at = self
            .theatres
            .iter()
            .position(|t| t.key == theatre)
            .ok_or_else(|| anyhow!("theatre {theatre} is not in regions.esf"))?;
        let old = self.theatres[at].rect;
        ensure!(
            rect[0][0] < rect[1][0] && rect[0][1] < rect[1][1],
            "theatre {theatre}: the new rectangle {rect:?} is empty"
        );
        ensure!(
            rect[0][0] <= old[0][0] && rect[0][1] <= old[0][1] && rect[1][0] >= old[1][0] && rect[1][1] >= old[1][1],
            "theatre {theatre}: the new rectangle {rect:?} must contain the old one {old:?} (extensions only grow)"
        );
        let rect = align_theatre_rect(old, rect);
        // Existing shared edges (Europe/India) are intentional. New contacts can make Napoleon's camera
        // polygon builder collapse a boundary and loop while loading. Check the snapped result before any mutation.
        let meets = |a: Rect, b: Rect| (0..2).all(|axis| a[0][axis] <= b[1][axis] && b[0][axis] <= a[1][axis]);
        for other in self.theatres.iter().filter(|t| t.key != theatre) {
            let overlaps = (0..2).all(|axis| rect[0][axis] < other.rect[1][axis] && other.rect[0][axis] < rect[1][axis]);
            ensure!(
                !overlaps,
                "Theatre {theatre} snapped rectangle {rect:?} overlaps theatre {} {:?}; theatre interiors must not overlap",
                other.key, other.rect
            );
            ensure!(
                !meets(rect, other.rect) || meets(old, other.rect),
                "Theatre {theatre} snapped rectangle {rect:?} newly touches theatre {} {:?}. \
                 Leave a positive gap after 2-unit snapping: new theatre contacts can collapse Napoleon's camera bounds and hang campaign loading",
                other.key, other.rect
            );
        }
        // signed shift of each edge: west, east, south, north
        let delta = [rect[0][0] - old[0][0], rect[1][0] - old[1][0], rect[0][1] - old[0][1], rect[1][1] - old[1][1]];
        let mut moved = Vec::new();
        for ta in &mut self.theatres[at].transitions {
            let Some(edge) = transition_edge(old, ta.rect) else { continue };
            let d = delta[edge];
            if d == 0.0 {
                continue;
            }
            let axis = if edge < 2 { 0 } else { 1 };
            let before = ta.rect;
            ta.rect[0][axis] += d;
            ta.rect[1][axis] += d;
            moved.push(TransitionMove { id: ta.id, old: before, new: ta.rect });
        }
        self.theatres[at].extended |= rect != old;
        self.theatres[at].rect = rect;
        Ok(TheatreChange { theatre: theatre.to_string(), old_rect: old, new_rect: rect, moved })
    }

    /// Next unused area component id (one more than the largest id of any area).
    pub fn fresh_component_id(&self) -> u32 {
        self.regions
            .iter()
            .flat_map(|r| r.areas.iter())
            .map(|a| a.adjoining_id as u32)
            .max()
            .map_or(0, |m| m + 1)
    }

    /// Component id an area takes when it becomes passable: that of the passable LAND neighbour (of a settled region)
    /// it shares the longest border with, else a fresh id. Attach the settlements of earlier activations first.
    pub fn component_for(&self, region: usize, area: usize) -> u32 {
        let mut length: std::collections::BTreeMap<u16, f64> = std::collections::BTreeMap::new();
        for o in &self.regions[region].areas[area].outlines {
            let n = o.ring.len();
            for run in o.runs.iter().filter(|r| r.code != NONE) {
                let (nr, na) = ((run.code >> 16) as usize, (run.code & 0xFFFF) as usize);
                if (nr, na) == (region, area) {
                    continue;
                }
                // only settled land regions carry campaign land components; the passable pieces of the unsettled
                // backdrop (e.g. unexplorable area 8, component 2) are scenery islands of their own
                let Some(neighbour) = self.regions.get(nr).filter(|r| r.kind == "land" && r.settlement.is_some()).and_then(|r| r.areas.get(na)) else {
                    continue;
                };
                if !neighbour.passable {
                    continue;
                }
                for k in run.from as usize..=run.to as usize {
                    let (p, q) = (self.v(o.ring[k % n]), self.v(o.ring[(k + 1) % n]));
                    *length.entry(neighbour.adjoining_id).or_insert(0.0) += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                }
            }
        }
        length
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or_else(|| self.fresh_component_id(), |(c, _)| c as u32)
    }

    /// Make an area passable with the given component id (`impassable_id` 65535 for passable land, 286 for sea,
    /// `island` as in the source area).
    pub fn set_passable(&mut self, region: usize, area: usize, component: u32, impassable_id: u32, island: u32) -> Result<()> {
        let key = &self.regions.get(region).ok_or_else(|| anyhow!("no region {region}"))?.key;
        let (component, impassable_id, island) = (
            u16::try_from(component).map_err(|_| anyhow!("{key}: component id {component} exceeds u16"))?,
            u16::try_from(impassable_id).map_err(|_| anyhow!("{key}: impassable id {impassable_id} exceeds u16"))?,
            u16::try_from(island).map_err(|_| anyhow!("{key}: island id {island} exceeds u16"))?,
        );
        let key = key.clone();
        let a = self.regions[region]
            .areas
            .get_mut(area)
            .ok_or_else(|| anyhow!("{key}: no area {area}"))?;
        a.passable = true;
        a.adjoining_id = component;
        a.impassable_id = impassable_id;
        a.island_id = island;
        Ok(())
    }

    /// Pole of inaccessibility of an area (the interior point farthest from every ring edge, polylabel with
    /// tolerance 0.001).
    pub fn pole_of_inaccessibility(&self, region: usize, area: usize) -> P {
        let a = &self.regions[region].areas[area];
        let rings: Vec<Vec<P>> = a.outlines.iter().map(|o| self.ring_points(&o.ring)).collect();
        super::geometry::polylabel(&rings, 0.001).0
    }

    /// Consistency problems of the geometry (empty when sound): ring vertices in range, runs partitioning every
    /// ring, outer rings counter-clockwise and holes clockwise, every neighbour code resolving to an existing area
    /// that carries the reverse edge and names back. (Face sums: see [`RegionData::face_problems`].)
    pub fn topology_check(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let nv = self.vertices.len() as u32;
        // directed ring edge -> codes naming it (owner, neighbour)
        let mut directed: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::new();
        for (ri, r) in self.regions.iter().enumerate() {
            for (ai, a) in r.areas.iter().enumerate() {
                for o in &a.outlines {
                    let n = o.ring.len();
                    for (k, c) in Self::edge_codes(o).into_iter().enumerate() {
                        if o.ring[k] < nv && o.ring[(k + 1) % n] < nv {
                            directed.entry((o.ring[k], o.ring[(k + 1) % n])).or_default().push((code(ri, ai), c));
                        }
                    }
                }
            }
        }
        for (ri, r) in self.regions.iter().enumerate() {
            for (ai, a) in r.areas.iter().enumerate() {
                let who = format!("{} area {ai}", r.key);
                if a.outlines.first().is_none_or(|o| !o.outer) {
                    problems.push(format!("{who}: first ring is not an outer ring"));
                }
                for (oi, o) in a.outlines.iter().enumerate() {
                    let n = o.ring.len();
                    if n < 3 || o.ring.iter().any(|&v| v >= nv) {
                        problems.push(format!("{who} ring {oi}: {n} vertices, ids out of range or too short"));
                        continue;
                    }
                    let pts = self.ring_points(&o.ring);
                    let s = signed_area2(&pts);
                    if o.outer != (s > 0.0) {
                        problems.push(format!("{who} ring {oi}: {} ring is {}", if o.outer { "outer" } else { "hole" }, if s > 0.0 { "counter-clockwise" } else { "clockwise" }));
                    }
                    let mut cover = vec![0u8; n];
                    for run in &o.runs {
                        for k in run.from..=run.to {
                            match cover.get_mut(k as usize) {
                                Some(c) => *c += 1,
                                None => problems.push(format!("{who} ring {oi}: run edge {k} beyond {n} edges")),
                            }
                        }
                    }
                    if cover.iter().any(|&c| c != 1) {
                        problems.push(format!("{who} ring {oi}: runs do not cover each edge exactly once"));
                    }
                    for (k, c) in Self::edge_codes(o).into_iter().enumerate() {
                        if c == NONE {
                            continue;
                        }
                        let (nr, na) = ((c >> 16) as usize, (c & 0xFFFF) as usize);
                        if self.regions.get(nr).is_none_or(|r| na >= r.areas.len()) {
                            problems.push(format!("{who} ring {oi} edge {k}: dangling code {nr}:{na}"));
                            continue;
                        }
                        let (a_id, b_id) = (o.ring[k], o.ring[(k + 1) % n]);
                        let paired = directed
                            .get(&(b_id, a_id))
                            .is_some_and(|v| v.iter().any(|&(owner, back)| owner == c && back == code(ri, ai)));
                        if !paired {
                            problems.push(format!(
                                "{who} ring {oi} edge {k}: neighbour {}:{na} has no reverse edge {b_id}->{a_id} naming back",
                                self.regions[nr].key
                            ));
                        }
                    }
                }
            }
        }
        problems
    }

    /// Areas whose `faces` do not cover exactly the area their rings enclose (summed triangle area vs outer area
    /// minus holes, relative tolerance 1e-3). Kept apart from [`RegionData::topology_check`] because the vanilla file
    /// itself carries 533 such areas (the faces of nested polygons repeat their parent's); everything this module
    /// builds is sound, so compare counts before and after an operation.
    pub fn face_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        for r in &self.regions {
            for (ai, a) in r.areas.iter().enumerate() {
                let enclosed: f64 = a
                    .outlines
                    .iter()
                    .map(|o| signed_area2(&self.ring_points(&o.ring)) / 2.0)
                    .sum();
                let faces: f64 = a
                    .faces
                    .iter()
                    .map(|t| signed_area2(&[self.v(t[0]), self.v(t[1]), self.v(t[2])]).abs() / 2.0)
                    .sum();
                if (faces - enclosed).abs() > 1e-3 * enclosed.abs().max(1.0) {
                    problems.push(format!("{} area {ai}: faces cover {faces:.4} but the rings enclose {enclosed:.4}", r.key));
                }
            }
        }
        problems
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theatre_fixture(rects: &[(&str, Rect)]) -> RegionData {
        RegionData {
            vertices: vec![],
            regions: vec![],
            theatres: rects.iter().map(|(key, rect)| Theatre {
                key: (*key).into(), rect: *rect, transitions: vec![], extended: false,
            }).collect(),
        }
    }

    #[test]
    fn snapped_contact_is_rejected_before_transition_or_rectangle_mutation() {
        let india = [[440.0, 20.0], [680.0, 280.0]];
        let mut rd = theatre_fixture(&[
            ("3", india),
            ("-1133129049", [[800.0, -80.0], [920.0, -5.0]]),
        ]);
        let transition = [[660.63, 17.73], [678.66, 118.5]];
        rd.theatres[0].transitions.push(TransitionArea { id: 11, rect: transition });
        assert!(rd.extend_theatre("3", [[440.0, -84.0], [799.0, 280.0]]).is_err());
        assert_eq!(rd.theatres[0].rect, india);
        assert_eq!(rd.theatres[0].transitions[0].rect, transition);
        assert!(!rd.theatres[0].extended);
    }

    #[test]
    fn original_shared_edge_and_positive_gap_are_preserved_but_overlap_is_rejected() {
        let mut rd = theatre_fixture(&[
            ("2", [[-180.0, 140.0], [440.0, 520.0]]),
            ("3", [[440.0, 20.0], [680.0, 280.0]]),
            ("-1133129049", [[800.0, -80.0], [920.0, -5.0]]),
        ]);
        let safe = [[440.0, -84.0], [798.0, 280.0]];
        assert_eq!(rd.extend_theatre("3", safe).unwrap().new_rect, safe);
        assert!(rd.extend_theatre("3", [[439.5, -84.0], [798.0, 280.0]]).is_err());
        assert_eq!(rd.theatre_rect("3").unwrap(), safe);
    }

    #[test]
    fn new_corner_contact_is_rejected() {
        let mut rd = theatre_fixture(&[
            ("1", [[0.0, 0.0], [4.0, 4.0]]),
            ("2", [[8.0, 8.0], [12.0, 12.0]]),
        ]);
        assert!(rd.extend_theatre("1", [[0.0, 0.0], [8.0, 8.0]]).is_err());
    }

    #[test]
    fn jointly_created_contact_is_rejected_in_either_extension_order() {
        let base = theatre_fixture(&[
            ("1", [[0.0, 0.0], [4.0, 4.0]]),
            ("2", [[12.0, 0.0], [16.0, 4.0]]),
        ]);
        let extensions = [
            ("1", [[0.0, 0.0], [8.0, 4.0]]),
            ("2", [[8.0, 0.0], [16.0, 4.0]]),
        ];
        for first in 0..2 {
            let mut rd = base.clone();
            let (key, rect) = extensions[first];
            rd.extend_theatre(key, rect).unwrap();
            let (key, rect) = extensions[1 - first];
            assert!(rd.extend_theatre(key, rect).is_err());
        }
    }

    #[test]
    fn fractional_extension_snaps_outward_and_moves_transition_with_edge() {
        let old = [[440.0, 0.0], [680.0, 280.0]];
        let mut rd = RegionData {
            vertices: vec![], regions: vec![],
            theatres: vec![Theatre {
                key: "3".into(), rect: old, extended: false,
                transitions: vec![TransitionArea { id: 11, rect: [[660.63, 17.73], [678.66, 118.5]] }],
            }],
        };
        let change = rd.extend_theatre("3", [[439.75, -83.25], [801.25, 280.25]]).unwrap();
        assert_eq!(change.new_rect, [[438.0, -84.0], [802.0, 282.0]]);
        assert_eq!(rd.theatre_rect("3").unwrap(), change.new_rect);
        assert_eq!(change.moved[0].new, [[782.63, 17.73], [800.66, 118.5]]);
        let again = rd.extend_theatre("3", change.new_rect).unwrap();
        assert_eq!(again.new_rect, change.new_rect);
        assert!(again.moved.is_empty());
        assert!(rd.extend_theatre("3", [[438.1, -84.0], [802.0, 282.0]]).is_err(), "snapping must not accept a shrinking request");
    }

    /// Square region [x0, x0+10] x [0, 10] (counter-clockwise, one vertex block per call).
    fn square(rd: &mut RegionData, key: &str, x0: f32, passable: bool, settlement: Option<Node>, tail: i32) {
        let base = rd.vertices.len() as u32;
        rd.vertices.extend([[x0, 0.0], [x0 + 10.0, 0.0], [x0 + 10.0, 10.0], [x0, 10.0]]);
        rd.regions.push(Region {
            key: key.into(),
            kind: "land".into(),
            areas: vec![Area {
                land_bridge: false,
                channel_coast: false,
                passable,
                adjoining_id: 20,
                faces: vec![[base, base + 1, base + 2], [base, base + 2, base + 3]],
                outlines: vec![Outline { outer: true, ring: (base..base + 4).collect(), runs: vec![] }],
                impassable_id: if passable { 65535 } else { 7 },
                island_id: 104,
            }],
            tail,
            settlement,
        });
    }

    fn donor_settlement() -> Node {
        Node::record(
            "settlement_and_slots",
            0,
            vec![
                esf::xyv(5.0, 5.0),
                esf::xyv(5.0, 5.0),
                Node::array(
                    "slot_descriptions",
                    1,
                    vec![
                        vec![esf::s16("settlement:donor:town:port:1"), esf::u32v(0), esf::xyzv(6.0, 0.0, 4.0)],
                        vec![esf::s16("region:donor:general:1"), esf::u32v(0), esf::xyzv(9.0, 0.0, 9.0)],
                    ],
                ),
            ],
        )
    }

    fn world() -> RegionData {
        let mut rd = RegionData { vertices: vec![], regions: vec![], theatres: vec![] };
        square(&mut rd, "donor", 0.0, true, Some(donor_settlement()), 2);
        square(&mut rd, "wild", 20.0, true, None, -1);
        square(&mut rd, "rock", 40.0, false, None, -1);
        rd
    }

    #[test]
    fn attach_copies_the_donor_settlement_and_theatre_id() {
        let mut rd = world();
        let spec = SettlementSpec { name: "fort".into(), center: [26.0, 4.0] };
        rd.attach_settlement(1, 0, &spec).unwrap();
        let wild = &rd.regions[1];
        assert_eq!(wild.tail, 2);
        assert_eq!(settlement_center(wild).unwrap(), [26.0, 4.0]);
        let slots = wild.settlement.as_ref().unwrap().children().unwrap()[2].items().unwrap();
        assert_eq!(slots.len(), 1, "only settlement slots are kept");
        assert_eq!(slots[0][0].as_str().unwrap(), "settlement:wild:fort:port:1");
        assert_eq!(slots[0][2].as_xyz().unwrap(), [27.0, 0.0, 3.0]);
        // the donor is untouched
        assert_eq!(rd.regions[0].tail, 2);
        assert_eq!(rd.area_at(1, [26.0, 4.0]), Some(0));
        assert_eq!(rd.area_at(1, [5.0, 4.0]), None);
    }

    #[test]
    fn attach_rejects_bad_targets() {
        let spec = |x: f32| SettlementSpec { name: "fort".into(), center: [x, 4.0] };
        let mut rd = world();
        assert!(rd.attach_settlement(1, 0, &spec(5.0)).unwrap_err().to_string().contains("not inside"));
        assert!(rd.attach_settlement(2, 0, &spec(46.0)).unwrap_err().to_string().contains("impassable"));
        assert!(rd.attach_settlement(0, 1, &spec(5.0)).unwrap_err().to_string().contains("already has a settlement"));
        assert!(rd.attach_settlement(1, 2, &spec(26.0)).unwrap_err().to_string().contains("has no settlement"));
        assert!(rd.attach_settlement(1, 1, &spec(26.0)).unwrap_err().to_string().contains("own donor"));
        assert!(rd.regions[1].settlement.is_none() && rd.regions[1].tail == -1, "failed attaches change nothing");
    }

    const VANILLA: &str = "F:/NTW2/work/exp1/backup/campaign_maps/nap_world/regions.esf";
    const EXP3: &str = "F:/NTW2/work/exp3/out/data/campaign_maps/nap_world/regions.esf";

    fn load_file(path: &str) -> Option<(esf::Esf, RegionData)> {
        let bytes = std::fs::read(path).ok()?;
        let file = esf::read(&bytes).unwrap();
        let rd = RegionData::parse(&file.root).unwrap();
        Some((file, rd))
    }

    #[test]
    fn vanilla_regions_round_trip_byte_exact() {
        let Some((mut file, rd)) = load_file(VANILLA) else { return };
        assert_eq!(rd.theatres.len(), 7);
        assert_eq!(rd.theatre_rect("3").unwrap(), [[440.0, 20.0], [680.0, 280.0]]);
        let original = std::fs::read(VANILLA).unwrap();
        rd.store(&mut file.root).unwrap();
        assert_eq!(esf::write(&file).unwrap(), original, "store must not change an untouched file");
    }

    #[test]
    fn extended_theatre_is_stored_in_both_copies_with_its_transition_area() {
        let (Some((mut file, mut rd)), Some((_, exp3))) = (load_file(VANILLA), load_file(EXP3)) else { return };
        let new = [[440.0, -80.0], [790.0, 300.0]];
        let change = rd.extend_theatre("3", new).unwrap();
        assert_eq!(change.moved.len(), 1);
        assert_eq!(change.moved[0].id, 11);
        assert_eq!(change.moved[0].old[0][0], 660.63);
        assert_eq!(change.moved[0].new[0][0], 770.63);
        assert!(rd.extend_theatre("3", [[450.0, -80.0], [790.0, 300.0]]).is_err(), "extensions only grow");
        assert!(rd.extend_theatre("nope", new).is_err());
        rd.store(&mut file.root).unwrap();
        let reread = RegionData::parse(&file.root).unwrap();
        assert_eq!(reread.theatres.iter().map(|t| (&t.key, t.rect, &t.transitions)).collect::<Vec<_>>(), exp3.theatres.iter().map(|t| (&t.key, t.rect, &t.transitions)).collect::<Vec<_>>());
        // the second copy (theatres_and_region_keys) carries the new rectangle too
        let item = file
            .root
            .get("theatres_and_region_keys")
            .unwrap()
            .items()
            .unwrap()
            .iter()
            .map(|it| &it[0])
            .find(|t| t.children().unwrap().iter().find_map(theatre_key) == Some("3"))
            .unwrap();
        let mut xy = item.children().unwrap().iter().filter_map(xy_value);
        assert_eq!((xy.next().unwrap(), xy.next().unwrap()), (new[0], new[1]));
    }

    #[test]
    fn every_vanilla_transition_area_sits_on_its_edge() {
        let Some((_, rd)) = load_file(VANILLA) else { return };
        // (theatre key order in the file) -> expected edge per transition id: 0 west, 1 east, 2 south, 3 north
        let expected: [(u32, usize); 23] = [
            (22, 1), (23, 0), (1, 0), (2, 2), (3, 1), (4, 1), (5, 1), (6, 0), (7, 0), (8, 0), (9, 2), (10, 0), (11, 1),
            (20, 2), (21, 3), (13, 2), (14, 3), (15, 1), (16, 1), (24, 0), (17, 0), (18, 0), (19, 2),
        ];
        let mut seen = 0;
        for t in &rd.theatres {
            for ta in &t.transitions {
                let want = expected.iter().find(|e| e.0 == ta.id).unwrap().1;
                assert_eq!(transition_edge(t.rect, ta.rect), Some(want), "theatre {} transition {}", t.key, ta.id);
                seen += 1;
            }
        }
        assert_eq!(seen, 23);
    }

    #[test]
    fn components_and_passability_helpers() {
        let mut rd = world();
        let fresh = rd.fresh_component_id();
        assert_eq!(fresh, rd.regions.iter().flat_map(|r| r.areas.iter()).map(|a| a.adjoining_id as u32).max().unwrap() + 1);
        assert!(rd.set_passable(2, 0, 70000, 65535, 0).is_err());
        rd.set_passable(2, 0, fresh, 65535, 104).unwrap();
        let a = &rd.regions[2].areas[0];
        assert_eq!((a.passable, a.adjoining_id as u32, a.impassable_id, a.island_id), (true, fresh, 65535, 104));
    }
}
