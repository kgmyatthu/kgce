//! `poi.esf`: the campaign AI's points of interest. The file is one flat `CAI_POI_ROOT` record:
//! `u32 count` then per point `[u32 index, i32 type, bool, i32 x, i32 y (fixed 12.20), s16 region, u32 area id,
//! s16 region, u32 area id, f32 distance to the region's settlement, u32 n, n * (s16 neighbour region, f32 distance
//! to it), f32 minimum of those (-1 when none), u32 m, m * (s16 region, f32), u32 k, k * u32, u32, bool]`.
//! Types seen: 1/2/5 grid samples, 3 border points, 4 the settlement, 6 sea samples, 7 coast/background, 8 theatre.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Result, anyhow, bail, ensure};

use super::geometry::P;
use super::regions::RegionData;
use crate::esf::{self, Node, Value};

const FIXED: f64 = 1.0 / (1u64 << 20) as f64;
pub const SETTLEMENT: i32 = 4;

#[derive(Clone, Debug)]
pub struct Poi {
    pub index: u32,
    pub kind: i32,
    pub flag: bool,
    pub x: f64,
    pub y: f64,
    pub region: String,
    pub area_id: u32,
    pub settlement_distance: f32,
    pub neighbours: Vec<(String, f32)>,
    pub nearest: f32,
    pub border: Vec<(String, f32)>,
    pub links: Vec<u32>,
    pub u: u32,
    pub b: bool,
}

struct Cursor<'a> {
    vals: &'a [Node],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn next(&mut self) -> Result<&'a Value> {
        let v = self
            .vals
            .get(self.at)
            .ok_or_else(|| anyhow!("poi.esf ends early at value {}", self.at))?;
        self.at += 1;
        match v {
            Node::Value(v) => Ok(v),
            other => bail!("poi.esf: record where a value was expected: {other:?}"),
        }
    }
    fn u32(&mut self) -> Result<u32> {
        match self.next()? {
            Value::U32(v) => Ok(*v),
            v => bail!("poi.esf: expected u32, found {v:?}"),
        }
    }
    fn i32(&mut self) -> Result<i32> {
        match self.next()? {
            Value::I32(v) => Ok(*v),
            v => bail!("poi.esf: expected i32, found {v:?}"),
        }
    }
    fn f32(&mut self) -> Result<f32> {
        match self.next()? {
            Value::F32(v) => Ok(*v),
            v => bail!("poi.esf: expected f32, found {v:?}"),
        }
    }
    fn bool(&mut self) -> Result<bool> {
        match self.next()? {
            Value::Bool(v) => Ok(*v),
            v => bail!("poi.esf: expected bool, found {v:?}"),
        }
    }
    fn s16(&mut self) -> Result<String> {
        match self.next()? {
            Value::String16(v) => Ok(v.clone()),
            v => bail!("poi.esf: expected string, found {v:?}"),
        }
    }
    fn pairs(&mut self) -> Result<Vec<(String, f32)>> {
        let n = self.u32()?;
        (0..n).map(|_| Ok((self.s16()?, self.f32()?))).collect()
    }
}

pub fn parse(root: &Node) -> Result<Vec<Poi>> {
    let vals = root.children()?;
    let mut c = Cursor { vals, at: 0 };
    let n = c.u32()?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let index = c.u32()?;
        let kind = c.i32()?;
        let flag = c.bool()?;
        let x = c.i32()? as f64 * FIXED;
        let y = c.i32()? as f64 * FIXED;
        let region = c.s16()?;
        let area_id = c.u32()?;
        let region2 = c.s16()?;
        let area2 = c.u32()?;
        ensure!(region == region2 && area_id == area2, "poi {index}: mismatched region pair");
        let settlement_distance = c.f32()?;
        let neighbours = c.pairs()?;
        let nearest = c.f32()?;
        let border = c.pairs()?;
        let k = c.u32()?;
        let links = (0..k).map(|_| c.u32()).collect::<Result<_>>()?;
        let u = c.u32()?;
        let b = c.bool()?;
        out.push(Poi {
            index,
            kind,
            flag,
            x,
            y,
            region,
            area_id,
            settlement_distance,
            neighbours,
            nearest,
            border,
            links,
            u,
            b,
        });
    }
    ensure!(c.at == vals.len(), "poi.esf: {} trailing values", vals.len() - c.at);
    Ok(out)
}

fn fixed(v: f64) -> i32 {
    (v / FIXED).round() as i32
}

pub fn store(root: &mut Node, pois: &[Poi]) -> Result<()> {
    let mut out: Vec<Node> = Vec::with_capacity(pois.len() * 20);
    out.push(esf::u32v(pois.len() as u32));
    for (i, p) in pois.iter().enumerate() {
        ensure!(p.index as usize == i, "poi indices must be sequential");
        out.push(esf::u32v(p.index));
        out.push(esf::i32v(p.kind));
        out.push(esf::b(p.flag));
        out.push(esf::i32v(fixed(p.x)));
        out.push(esf::i32v(fixed(p.y)));
        for _ in 0..2 {
            out.push(esf::s16(p.region.clone()));
            out.push(esf::u32v(p.area_id));
        }
        out.push(esf::f32v(p.settlement_distance));
        let pairs = |out: &mut Vec<Node>, list: &Vec<(String, f32)>| {
            out.push(esf::u32v(list.len() as u32));
            for (name, d) in list {
                out.push(esf::s16(name.clone()));
                out.push(esf::f32v(*d));
            }
        };
        pairs(&mut out, &p.neighbours);
        out.push(esf::f32v(p.nearest));
        pairs(&mut out, &p.border);
        out.push(esf::u32v(p.links.len() as u32));
        out.extend(p.links.iter().map(|&l| esf::u32v(l)));
        out.push(esf::u32v(p.u));
        out.push(esf::b(p.b));
    }
    *root.children_mut()? = out;
    Ok(())
}

impl Poi {
    pub fn set_neighbour(&mut self, region: &str, distance: f32) {
        match self.neighbours.iter_mut().find(|(n, _)| n == region) {
            Some(e) => e.1 = distance,
            None => self.neighbours.push((region.to_string(), distance)),
        }
        self.refresh_nearest();
    }
    pub fn remove_neighbour(&mut self, region: &str) {
        self.neighbours.retain(|(n, _)| n != region);
        self.refresh_nearest();
    }
    fn refresh_nearest(&mut self) {
        self.nearest = self
            .neighbours
            .iter()
            .map(|(_, d)| *d)
            .fold(f32::MAX, f32::min);
        if self.neighbours.is_empty() {
            self.nearest = -1.0;
        }
    }
}

/// What [`apply_split`] changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitReport {
    /// points of the source region that now belong to the new region
    pub moved: usize,
    /// neighbour / border lists of other points that were updated
    pub touched: usize,
    /// index of the appended settlement point of the new region
    pub settlement_point: usize,
}

fn flat_distance(p: &Poi, at: [f64; 2]) -> f32 {
    ((p.x - at[0]).powi(2) + (p.y - at[1]).powi(2)).sqrt() as f32
}

/// Updates the points for a region split. `rd` is the geometry AFTER the split (region `new_index` was carved out of
/// `source_index`), `center` is the new region's settlement position and `new_adjacent` the regions bordering the new
/// region (`rd.adjacent(new_index)`, includes the source).
///  * points of the source that now lie inside the new region are re-tagged (settlement distance to the new
///    settlement, neighbour list rebuilt from `new_adjacent`);
///  * the remaining source points gain the new region as a neighbour;
///  * points of regions bordering the new region re-point their source neighbour / border entries;
///  * a settlement point for the new region is appended (cloned from the source's settlement point).
pub fn apply_split(
    pois: &mut Vec<Poi>,
    rd: &RegionData,
    source_index: usize,
    new_index: usize,
    center: [f64; 2],
    new_adjacent: &BTreeSet<usize>,
) -> Result<SplitReport> {
    let src = rd.regions[source_index].key.clone();
    let new = rd.regions[new_index].key.clone();
    let index_of: HashMap<&str, usize> = rd.regions.iter().enumerate().map(|(i, r)| (r.key.as_str(), i)).collect();
    let dist = |i: usize, p: &Poi| rd.distance_to_region(i, [p.x, p.y]) as f32;
    let settlements: HashMap<String, [f64; 2]> = pois
        .iter()
        .filter(|p| p.kind == SETTLEMENT)
        .map(|p| (p.region.clone(), [p.x, p.y]))
        .chain(std::iter::once((new.clone(), center)))
        .collect();
    let mut report = SplitReport::default();
    let mut template: Option<Poi> = None;
    for p in pois.iter_mut() {
        if p.region == src {
            if rd.in_region(new_index, [p.x, p.y]) {
                p.region = new.clone();
                p.settlement_distance = flat_distance(p, center);
                if !p.neighbours.is_empty() {
                    let list: Vec<(String, f32)> =
                        new_adjacent.iter().map(|&i| (rd.regions[i].key.clone(), dist(i, p))).collect();
                    p.neighbours.clear();
                    for (k, d) in list {
                        p.set_neighbour(&k, d);
                    }
                }
                report.moved += 1;
            } else {
                if p.kind == SETTLEMENT {
                    template = Some(p.clone());
                }
                if !p.neighbours.is_empty() {
                    let d = dist(new_index, p);
                    p.set_neighbour(&new, d);
                }
                report.touched += 1;
            }
        } else if index_of.get(p.region.as_str()).is_some_and(|ri| new_adjacent.contains(ri)) {
            if p.neighbours.iter().any(|(n, _)| *n == src) {
                let (ds, dn) = (dist(source_index, p), dist(new_index, p));
                p.set_neighbour(&src, ds);
                p.set_neighbour(&new, dn);
                report.touched += 1;
            }
            if p.border.iter().any(|(n, _)| *n == src) && dist(new_index, p) < dist(source_index, p) {
                let d = flat_distance(p, center);
                for entry in p.border.iter_mut().filter(|e| e.0 == src) {
                    entry.0 = new.clone();
                    entry.1 = d;
                    report.touched += 1;
                }
            }
        }
    }
    // border points of the new region that name a neighbouring region: distance to that region's settlement
    for p in pois.iter_mut().filter(|p| p.region == new) {
        let at = [p.x, p.y];
        for entry in p.border.iter_mut() {
            if let Some(s) = settlements.get(&entry.0) {
                entry.1 = ((at[0] - s[0]).powi(2) + (at[1] - s[1]).powi(2)).sqrt() as f32;
            }
        }
    }
    let mut sett = template.ok_or_else(|| anyhow!("region {src} has no settlement point in poi.esf"))?;
    sett.index = pois.len() as u32;
    sett.region = new.clone();
    sett.x = center[0];
    sett.y = center[1];
    sett.settlement_distance = 0.0;
    sett.neighbours.clear();
    for &i in new_adjacent {
        let d = rd.distance_to_region(i, center) as f32;
        sett.set_neighbour(&rd.regions[i].key, d);
    }
    report.settlement_point = pois.len();
    pois.push(sett);
    Ok(report)
}

// ------------------------------------------------------------------------------------------------ activation

/// Interior lattice spacing (world units) of an activated region is `sqrt(passable area / interior points)` of its
/// donor (vanilla measurements: 1.5 .. 7 units, big plains 6 .. 7), rounded to 0.5 and clamped to this range.
const LATTICE_SPACING: (f64, f64) = (2.0, 8.0);
/// Interior points keep this distance from every outline of the region (vanilla: down to 0.5).
const LATTICE_MARGIN: f64 = 1.0;
/// Border points sit this far inside their region, measured from the shared border (vanilla: 0.5).
const BORDER_OFFSET: f64 = 0.5;
/// One border point per this many world units of a shared border run (at least one per run).
const BORDER_POINT_SPACING: f64 = 6.0;

/// What [`apply_activation`] generated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActivationReport {
    /// lattice spacing taken from the donor (world units)
    pub spacing: f64,
    /// index of the settlement point (type 4)
    pub settlement_point: usize,
    /// type-5 interior points of the activated region
    pub interior: usize,
    /// type-3 border points inside the activated region / mirrored ones inside its playable neighbours
    pub border: usize,
    pub mirrored: usize,
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Passable area of a region (holes subtracted).
fn passable_area(rd: &RegionData, ri: usize) -> f64 {
    let mut total = 0.0;
    for a in rd.regions[ri].areas.iter().filter(|a| a.passable) {
        for o in &a.outlines {
            let ring: Vec<[f64; 2]> = o.ring.iter().map(|&i| rd.v(i)).collect();
            let s = super::geometry::polygon_area(&ring);
            total += if o.outer { s } else { -s };
        }
    }
    total
}

/// `b` flag of the point of `kind` nearest to `at` (the flag varies smoothly over the map, vanilla has no rule for it).
fn nearest_flag(pois: &[Poi], kind: i32, at: [f64; 2]) -> Option<bool> {
    pois.iter()
        .filter(|p| p.kind == kind)
        .min_by(|a, b| dist([a.x, a.y], at).total_cmp(&dist([b.x, b.y], at)))
        .map(|p| p.b)
}

/// Gives the activated wilderness region `index` (no settlement, only coast points so far) the points of a playable
/// region. `center` is its settlement position, `donor` the template region and `playable(i)` tells whether region
/// `i` is a playable one (it must also own a settlement point). Appends, never renumbers:
///  * the settlement point (type 4), cloned from the donor's, neighbours = bordering non-river regions with the
///    distance of the point to each;
///  * a square lattice of type-5 points (flag / `b` of the donor's type-5 points) in the passable areas at the donor's
///    density, on the half-unit grid of vanilla points (x, y = k/2 + 0.25), at least [`LATTICE_MARGIN`] from every
///    outline and half a spacing from the settlement; each carries the distance to the settlement and to every
///    bordering non-river region;
///  * type-3 border points along every shared run with a playable neighbour (one per [`BORDER_POINT_SPACING`] of run,
///    at least one, spread evenly), [`BORDER_OFFSET`] inside the region, border list = [(neighbour, distance to the
///    neighbour's settlement)], and the vanilla mirror of each inside the neighbour naming the new region.
/// Points of other regions already list the wilderness region as a neighbour (the converter keeps ETW's lists), so
/// nothing existing is touched.
pub fn apply_activation(
    pois: &mut Vec<Poi>,
    rd: &RegionData,
    donor: usize,
    index: usize,
    center: [f64; 2],
    playable: &dyn Fn(usize) -> bool,
) -> Result<ActivationReport> {
    let key = rd.regions[index].key.clone();
    let donor_key = rd.regions[donor].key.clone();
    ensure!(
        !pois.iter().any(|p| p.region == key && matches!(p.kind, 1 | 2 | 3 | 4 | 5)),
        "region {key} already has settlement or grid points in poi.esf"
    );
    let area = rd
        .area_at(index, center)
        .ok_or_else(|| anyhow!("settlement position {center:?} is not inside region {key}"))?;
    let area_id = rd.regions[index].areas[area].adjoining_id as u32;
    let adjacent: BTreeSet<usize> = rd.adjacent(index).into_iter().filter(|&i| rd.regions[i].kind != "river").collect();
    let neighbour_list = |at: [f64; 2]| -> Vec<(String, f32)> {
        adjacent.iter().map(|&i| (rd.regions[i].key.clone(), rd.distance_to_region(i, at) as f32)).collect()
    };
    let mut settlements: HashMap<String, [f64; 2]> =
        pois.iter().filter(|p| p.kind == SETTLEMENT).map(|p| (p.region.clone(), [p.x, p.y])).collect();
    let donor_settlement = pois
        .iter()
        .find(|p| p.kind == SETTLEMENT && p.region == donor_key)
        .cloned()
        .ok_or_else(|| anyhow!("donor region {donor_key} has no settlement point in poi.esf"))?;
    let template5 = pois
        .iter()
        .find(|p| p.kind == 5 && p.region == donor_key)
        .cloned()
        .ok_or_else(|| anyhow!("donor region {donor_key} has no type-5 point in poi.esf"))?;
    let donor_points = pois.iter().filter(|p| p.region == donor_key && matches!(p.kind, 1 | 2 | 5)).count();
    let donor_area = passable_area(rd, donor);
    ensure!(donor_points > 0 && donor_area > 0.0, "cannot measure the point density of donor {donor_key}");
    let spacing = ((donor_area / donor_points as f64).sqrt() * 2.0).round() / 2.0;
    let spacing = spacing.clamp(LATTICE_SPACING.0, LATTICE_SPACING.1);

    let mut report = ActivationReport { spacing, ..Default::default() };
    let push = |pois: &mut Vec<Poi>, mut p: Poi| -> usize {
        p.index = pois.len() as u32;
        pois.push(p);
        pois.len() - 1
    };

    // ---- settlement point
    let mut sett = donor_settlement;
    sett.region = key.clone();
    sett.area_id = area_id;
    sett.x = center[0];
    sett.y = center[1];
    sett.settlement_distance = 0.0;
    sett.neighbours.clear();
    for (k, d) in neighbour_list(center) {
        sett.set_neighbour(&k, d);
    }
    report.settlement_point = push(pois, sett);
    settlements.insert(key.clone(), center);

    // ---- interior lattice
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for a in rd.regions[index].areas.iter().filter(|a| a.passable) {
        for o in a.outlines.iter().filter(|o| o.outer) {
            for &v in &o.ring {
                let p = rd.v(v);
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
    }
    ensure!(lo[0] <= hi[0], "region {key} has no passable outline");
    let first = |v: f64| (v / spacing).floor() * spacing + 0.25;
    let mut y = first(lo[1]);
    while y <= hi[1] {
        let mut x = first(lo[0]);
        while x <= hi[0] {
            let at = [x, y];
            if let Some(a) = rd.area_at(index, at).filter(|&a| rd.regions[index].areas[a].passable) {
                if dist(at, center) >= spacing / 2.0 && rd.distance_to_region(index, at) >= LATTICE_MARGIN {
                    let mut p = template5.clone();
                    p.region = key.clone();
                    p.area_id = rd.regions[index].areas[a].adjoining_id as u32;
                    p.x = x;
                    p.y = y;
                    p.settlement_distance = dist(at, center) as f32;
                    p.neighbours.clear();
                    for (k, d) in neighbour_list(at) {
                        p.set_neighbour(&k, d);
                    }
                    push(pois, p);
                    report.interior += 1;
                }
            }
            x += spacing;
        }
        y += spacing;
    }

    // ---- border points with playable neighbours
    let neighbours: Vec<usize> = adjacent
        .iter()
        .copied()
        .filter(|&i| playable(i) && settlements.contains_key(&rd.regions[i].key))
        .collect();
    for ni in neighbours {
        let nkey = rd.regions[ni].key.clone();
        let nsett = settlements[&nkey];
        // (region, position, own settlement, the other region, its settlement)
        for (mine, theirs) in border_sites(rd, index, ni) {
            let sides = [(index, mine, center, &nkey, nsett), (ni, theirs, nsett, &key, center)];
            for (region, at, own_settlement, other, other_settlement) in sides {
                let owner = &rd.regions[region].key;
                let mut p = pois
                    .iter()
                    .find(|p| p.kind == 3 && p.region == *owner)
                    .or_else(|| pois.iter().find(|p| p.kind == 3 && p.region == donor_key))
                    .cloned()
                    .ok_or_else(|| anyhow!("no type-3 point to clone for region {owner}"))?;
                p.region = owner.clone();
                if let Some(a) = rd.area_at(region, at) {
                    p.area_id = rd.regions[region].areas[a].adjoining_id as u32;
                }
                p.x = at[0];
                p.y = at[1];
                p.settlement_distance = dist(at, own_settlement) as f32;
                p.neighbours.clear();
                p.nearest = -1.0;
                p.border = vec![(other.clone(), dist(at, other_settlement) as f32)];
                p.b = nearest_flag(pois, 3, at).unwrap_or(p.b);
                push(pois, p);
                if region == index {
                    report.border += 1;
                } else {
                    report.mirrored += 1;
                }
            }
        }
    }
    Ok(report)
}

/// Positions for the border points between region `index` and its neighbour `ni`: per shared run one site per
/// [`BORDER_POINT_SPACING`] of run length (at least one), evenly spread along the run. Each site is the pair
/// (position [`BORDER_OFFSET`] inside `index`, mirrored position inside `ni`); sites whose offsets do not land in the
/// respective region are dropped.
fn border_sites(rd: &RegionData, index: usize, ni: usize) -> Vec<([f64; 2], [f64; 2])> {
    let mut sites = Vec::new();
    for a in &rd.regions[index].areas {
        for o in a.outlines.iter().filter(|o| o.outer) {
            let n = o.ring.len();
            for run in o.runs.iter().filter(|r| r.code != super::regions::NONE && (r.code >> 16) as usize == ni) {
                let edges: Vec<(P, P, f64)> = (run.from as usize..=run.to as usize)
                    .map(|k| {
                        let (p, q) = (rd.v(o.ring[k % n]), rd.v(o.ring[(k + 1) % n]));
                        (p, q, dist(p, q))
                    })
                    .filter(|e| e.2 > 0.0)
                    .collect();
                let total: f64 = edges.iter().map(|e| e.2).sum();
                if edges.is_empty() {
                    continue;
                }
                let count = ((total / BORDER_POINT_SPACING).round() as usize).max(1);
                for j in 0..count {
                    let mut want = (j as f64 + 0.5) * total / count as f64;
                    let mut edge = &edges[edges.len() - 1];
                    for e in &edges {
                        if want <= e.2 {
                            edge = e;
                            break;
                        }
                        want -= e.2;
                    }
                    let (p, q, len) = *edge;
                    let t = (want / len).clamp(0.0, 1.0);
                    let on = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
                    let normal = [-(q[1] - p[1]) / len, (q[0] - p[0]) / len];
                    let side = |s: f64| [on[0] + s * BORDER_OFFSET * normal[0], on[1] + s * BORDER_OFFSET * normal[1]];
                    // the owner is on the left of a ring edge; check instead of trusting the winding
                    let s = if rd.in_region(index, side(1.0)) { 1.0 } else { -1.0 };
                    let (mine, theirs) = (side(s), side(-s));
                    if rd.in_region(index, mine) && rd.in_region(ni, theirs) {
                        sites.push((mine, theirs));
                    }
                }
            }
        }
    }
    sites
}
