//! One `region_data` tree of regions.esf (political or groundtypes) with its derived directed-edge graph.

use super::geom::{AreaPoly, P, Rect64, bbox_of, widen};
use crate::esf::{Node, Value};
use anyhow::{Context, Result, bail, ensure};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Ring {
    pub outer: bool,
    pub ids: Vec<u32>,
}

pub struct SrcArea<'a> {
    pub key: (usize, usize),
    /// the area item as stored (prototype of every output fragment)
    pub item: &'a Vec<Node>,
    pub rings: Vec<Ring>,
    /// raw triangle index array of the area's `faces` node (shared by the areas of a connected group)
    pub faces: Option<Vec<u32>>,
    pub faces_id: Option<usize>,
    pub bbox: Rect64,
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeInfo {
    pub ri: u32,
    pub ai: u32,
    /// global rank of the directed edge in file order
    pub rank: u64,
}

impl EdgeInfo {
    pub fn area(&self) -> (usize, usize) {
        (self.ri as usize, self.ai as usize)
    }
}

pub struct Source<'a> {
    pub name: &'static str,
    pub with_faces: bool,
    pub v: Vec<P>,
    pub regs: Vec<&'a Vec<Node>>,
    pub areas: Vec<SrcArea<'a>>,
    pub area_ix: HashMap<(usize, usize), usize>,
    pub eo: HashMap<(u32, u32), EdgeInfo>,
    pub n_regions: usize,
    /// number of areas of every region
    pub counts: Vec<usize>,
    polys: RefCell<HashMap<usize, Rc<AreaPoly>>>,
}

fn find_one<'x>(items: &'x [Node], name: &str) -> Result<&'x Node> {
    let mut hits = items.iter().filter(|c| c.name() == Some(name));
    let first = hits.next().with_context(|| format!("expected one {name}, found none"))?;
    ensure!(hits.next().is_none(), "expected one {name}, found several");
    Ok(first)
}

impl<'a> Source<'a> {
    pub fn new(rd: &'a Node, name: &'static str, with_faces: bool) -> Result<Self> {
        let v: Vec<P> = rd.get("vertices")?.children()?.first().context("vertices record is empty")?.xy_array()?.into_iter().map(widen).collect();
        let regs: Vec<&'a Vec<Node>> = rd.get("regions")?.items()?.iter().collect();
        let mut areas = Vec::new();
        let mut area_ix = HashMap::new();
        let mut eo: HashMap<(u32, u32), EdgeInfo> = HashMap::new();
        let mut counts = Vec::with_capacity(regs.len());
        let mut rank = 0u64;
        let mut faces_intern: HashMap<Vec<u32>, usize> = HashMap::new();
        for (ri, r) in regs.iter().copied().enumerate() {
            let items = find_one(r, "areas").with_context(|| format!("{name}: region {ri}"))?.items()?;
            counts.push(items.len());
            for (ai, a) in items.iter().enumerate() {
                let outs = find_one(a, "outlines").with_context(|| format!("{name}: region {ri} area {ai}"))?.items()?;
                let mut rings = Vec::new();
                for (oi, o) in outs.iter().enumerate() {
                    ensure!(o.len() >= 5, "{name}: region {ri} area {ai} outline {oi} has {} children", o.len());
                    let flag = o[0].as_bool()?;
                    let ring_node = o
                        .iter()
                        .find(|c| matches!(c, Node::Value(Value::Array { .. })))
                        .with_context(|| format!("{name}: outline without ring array"))?;
                    let ids = ring_node.u32_array()?;
                    let n = ids.len();
                    ensure!(n >= 3, "{name}: region {ri} area {ai} outline {oi} has {n} vertices");
                    for &i in &ids {
                        ensure!((i as usize) < v.len(), "{name}: ring vertex {i} outside the {} entry vertex pool", v.len());
                    }
                    for p in 0..n {
                        let e = (ids[p], ids[(p + 1) % n]);
                        let info = EdgeInfo { ri: ri as u32, ai: ai as u32, rank };
                        rank += 1;
                        if eo.insert(e, info).is_some() {
                            bail!("regions atlas: {name}: duplicate directed edge {e:?}");
                        }
                    }
                    rings.push(Ring { outer: flag, ids });
                }
                let faces = match a.iter().find(|c| c.name() == Some("faces")) {
                    Some(f) => Some(f.children()?.first().context("empty faces record")?.u32_array()?),
                    None => None,
                };
                let faces_id = faces.as_ref().map(|f| {
                    let next = faces_intern.len();
                    *faces_intern.entry(f.clone()).or_insert(next)
                });
                let outer_pts: Vec<P> = rings.iter().filter(|r| r.outer).flat_map(|r| r.ids.iter().map(|&i| v[i as usize])).collect();
                let bbox = if outer_pts.is_empty() { [[0.0; 2]; 2] } else { bbox_of(&outer_pts) };
                area_ix.insert((ri, ai), areas.len());
                areas.push(SrcArea { key: (ri, ai), item: a, rings, faces, faces_id, bbox });
            }
        }
        Ok(Self { name, with_faces, v, n_regions: regs.len(), regs, areas, area_ix, eo, counts, polys: RefCell::new(HashMap::new()) })
    }

    pub fn region_key(&self, ri: usize) -> Result<&str> {
        self.regs[ri].first().context("empty region")?.as_str()
    }

    pub fn region_kind(&self, ri: usize) -> Result<&str> {
        self.regs[ri].get(1).context("region without kind")?.as_str()
    }

    pub fn area(&self, key: (usize, usize)) -> &SrcArea<'a> {
        &self.areas[self.area_ix[&key]]
    }

    pub fn ring_points(&self, ids: &[u32]) -> Vec<P> {
        ids.iter().map(|&i| self.v[i as usize]).collect()
    }

    /// Polygon of one area (single outer ring, nested holes dropped); cached.
    pub fn poly(&self, key: (usize, usize)) -> Result<Rc<AreaPoly>> {
        let ix = self.area_ix[&key];
        if let Some(p) = self.polys.borrow().get(&ix) {
            return Ok(p.clone());
        }
        let a = &self.areas[ix];
        let outers: Vec<Vec<P>> = a.rings.iter().filter(|r| r.outer).map(|r| self.ring_points(&r.ids)).collect();
        let holes: Vec<Vec<P>> = a.rings.iter().filter(|r| !r.outer).map(|r| self.ring_points(&r.ids)).collect();
        let poly = Rc::new(AreaPoly::new(&outers, &holes).with_context(|| format!("{}: area {key:?}", self.name))?);
        self.polys.borrow_mut().insert(ix, poly.clone());
        Ok(poly)
    }

    /// The smallest area containing `pt`.
    pub fn locate(&self, pt: P) -> Result<(usize, usize)> {
        let mut best: Option<(f64, (usize, usize))> = None;
        for a in &self.areas {
            if !(a.bbox[0][0] <= pt[0] && pt[0] <= a.bbox[1][0] && a.bbox[0][1] <= pt[1] && pt[1] <= a.bbox[1][1]) {
                continue;
            }
            let poly = self.poly(a.key)?;
            if poly.contains(pt) && best.is_none_or(|(s, _)| poly.area < s) {
                best = Some((poly.area, a.key));
            }
        }
        match best {
            Some((_, k)) => Ok(k),
            None => bail!("regions atlas: {}: point {pt:?} is in no source area", self.name),
        }
    }
}
