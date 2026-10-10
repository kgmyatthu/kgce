//! Consistency check of the generated CAI graph (native port of `handle_check_cai.py`).
//!
//! The reference-carrying fields are *learned* from the pristine NTW `eur_napoleon` CAI_INTERFACE (fields whose values resolve to CAI
//! objects in >= 98 % of the cases) and then enforced on the generated interface:
//! * no dangling CAI id in a learned reference field,
//! * every non-zero value of the pool item header lists resolves (post-load 0xcfd790 fails on NULL),
//! * header #10 `[(provider, position)]` pair invariant (position < len(provider BLOCK_OWNS), BLOCK_OWNS[position] is the item),
//! * no duplicate CAI ids,
//! * CAI -> WORLD references of the learned world-id fields resolve,
//! * WORLD REGION child 22 and CAI_GOVERNORSHIP #0 resolve to world governorships.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};

use crate::esf::{Node, Value};
use crate::regionmap::sea_grids::SeaGrids;
use crate::startpos::context::BuildContext;

use super::nodes::*;

const LEARN_RATE: f64 = 0.98;
const MAX_LISTED: usize = 12;
const QUAD: [&str; 4] = ["bottom_left", "top_left", "bottom_right", "top_right"];

pub struct CheckReport {
    lines: Vec<String>,
    bad: usize,
    universe: usize,
    learned: usize,
    world_learned: usize,
}

impl CheckReport {
    pub fn bad(&self) -> usize {
        self.bad
    }
    pub fn problems(&self) -> &[String] {
        &self.lines
    }
    pub fn summary(&self) -> String {
        format!(
            "consistency check: {} CAI ids, {} learned reference fields, {} learned world-reference fields, {} problem(s)",
            self.universe, self.learned, self.world_learned, self.bad
        )
    }

    fn add(&mut self, count: usize, line: String) {
        if count == 0 {
            return;
        }
        self.bad += count;
        if self.lines.len() < 4 * MAX_LISTED {
            self.lines.push(line);
        }
    }
}

type Key = (u32, u32, u8); // (path id, child index, 0 = u32 / 1 = u32 array element)

/// path interner: every distinct record path gets a small id so keys of different files compare
#[derive(Default)]
struct Paths {
    kids: Vec<HashMap<String, u32>>,
    parent: Vec<(u32, String)>,
}

impl Paths {
    fn new() -> Self {
        Self {
            kids: vec![HashMap::new()],
            parent: vec![(0, "/CAI_INTERFACE".to_string())],
        }
    }

    fn child(&mut self, pid: u32, name: &str) -> u32 {
        if let Some(id) = self.kids[pid as usize].get(name) {
            return *id;
        }
        let id = self.parent.len() as u32;
        self.parent.push((pid, name.to_string()));
        self.kids.push(HashMap::new());
        self.kids[pid as usize].insert(name.to_string(), id);
        id
    }

    fn render(&self, mut pid: u32) -> String {
        let mut parts = Vec::new();
        loop {
            let (p, n) = &self.parent[pid as usize];
            parts.push(n.clone());
            if pid == 0 {
                break;
            }
            pid = *p;
        }
        parts.reverse();
        parts.join("/")
    }
}

fn walk(ch: &[Node], pid: u32, paths: &mut Paths, out: &mut HashMap<Key, Vec<u32>>) {
    for (i, c) in ch.iter().enumerate() {
        match c {
            Node::Record { name, children, .. } => {
                let nm = if QUAD.contains(&name.as_str()) {
                    "Q"
                } else {
                    name.as_str()
                };
                let cp = paths.child(pid, nm);
                walk(children, cp, paths, out);
            }
            Node::Array { name, items, .. } => {
                let cp = paths.child(pid, &format!("{name}[]"));
                for it in items {
                    walk(it, cp, paths, out);
                }
            }
            Node::Value(Value::U32(v)) => out.entry((pid, i as u32, 0)).or_default().push(*v),
            Node::Value(Value::Array { code: 0x48, data }) if !data.is_empty() => {
                let e = out.entry((pid, i as u32, 1)).or_default();
                for b in data.chunks_exact(4) {
                    e.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
            }
            _ => {}
        }
    }
}

fn collect(ci: &Node, paths: &mut Paths) -> Result<HashMap<Key, Vec<u32>>> {
    let mut out = HashMap::new();
    walk(kids(ci)?, 0, paths, &mut out);
    Ok(out)
}

/// header id of a CAI item: first scalar after its PROPSET record
fn item_hdr(it: &[Node]) -> Option<u32> {
    let i = it.iter().position(|c| is_record_named(c, PROPSET))?;
    it[i + 1..].iter().find(|c| is_value(c)).and_then(scalar)
}

fn pool_arrays<'a>(
    pool: &'a Node,
) -> Result<impl Iterator<Item = (&'a str, &'a Vec<Vec<Node>>)> + 'a> {
    Ok(kids(pool)?.iter().filter_map(|a| match a {
        Node::Array { name, items, .. } if name.starts_with("CAI_BDI_POOL_") => {
            Some((name.as_str(), items))
        }
        _ => None,
    }))
}

/// every CAI object id (world item headers, pool items, faction second headers, interface / world headers)
fn universe(ci: &Node) -> Result<HashSet<u32>> {
    let mut u = HashSet::new();
    let world = get(ci, &["CAI_WORLD"])?;
    for a in kids(world)? {
        if let Node::Array { items, .. } = a {
            for it in items {
                if let Some(h) = item_hdr(it) {
                    u.insert(h);
                }
            }
        }
    }
    for it in get_items(world, &["CAI_WORLD_FACTIONS"])? {
        let r_ = spec_kids(it)?;
        let pi = find_idx(r_, PROPSET).ok_or_else(|| anyhow!("CAI_FACTION without PROPSET"))?;
        u.insert(sc_at(r_, pi + 1)?);
    }
    // interface header: first scalar after PROPSET among the non-array children
    let flat: Vec<&Node> = kids(ci)?
        .iter()
        .filter(|c| !matches!(c, Node::Array { .. }))
        .collect();
    if let Some(i) = flat.iter().position(|c| is_record_named(c, PROPSET)) {
        if let Some(h) = flat[i + 1..]
            .iter()
            .find(|c| is_value(c))
            .and_then(|c| scalar(c))
        {
            u.insert(h);
        }
    }
    let wh = kids(world)?
        .iter()
        .find(|c| is_u32(c))
        .and_then(scalar)
        .ok_or_else(|| anyhow!("CAI_WORLD without header id"))?;
    u.insert(wh);
    let mut add_pool = |p: &Node| -> Result<()> {
        for (_, items) in pool_arrays(p)? {
            for it in items {
                if let Some(h) = item_hdr(it) {
                    u.insert(h);
                }
            }
        }
        Ok(())
    };
    add_pool(get(
        ci,
        &["CAI_INTERFACE_DIRECTOR_BDI_POOL", "CAI_BDI_POOL"],
    )?)?;
    add_pool(get(ci, &["CAI_BDI_POOL"])?)?;
    for m in get_items(ci, &["CAI_INTERFACE_MANAGERS"])? {
        add_pool(m.first().ok_or_else(|| anyhow!("empty CAI manager item"))?)?;
    }
    Ok(u)
}

fn is_ref(v: u32, floor: u32) -> bool {
    v >= floor && v != 0xFFFF_FFFF
}

fn learn(
    fields: &HashMap<Key, Vec<u32>>,
    accept: impl Fn(u32) -> bool,
    floor: u32,
) -> HashSet<Key> {
    let mut refs = HashSet::new();
    for (key, vals) in fields {
        let nv: Vec<u32> = vals.iter().copied().filter(|v| is_ref(*v, floor)).collect();
        if nv.len() < 3 {
            continue;
        }
        let distinct: HashSet<u32> = nv.iter().copied().collect();
        if distinct.len() < 2 {
            continue;
        }
        let hit = nv.iter().filter(|v| accept(**v)).count();
        if hit as f64 / nv.len() as f64 >= LEARN_RATE {
            refs.insert(*key);
        }
    }
    refs
}

/// world object ids (>= 1_000_000) below `node`; arrays named in `skip` are not entered
fn world_ids(node: &Node, skip: &[&str], out: &mut HashSet<u32>) {
    match node {
        Node::Record { children, .. } => {
            for c in children {
                world_ids(c, skip, out);
            }
        }
        Node::Array { name, items, .. } => {
            if skip.contains(&name.as_str()) {
                return;
            }
            for it in items {
                for c in it {
                    world_ids(c, skip, out);
                }
            }
        }
        Node::Value(Value::U32(v)) => {
            if *v >= 1_000_000 {
                out.insert(*v);
            }
        }
        Node::Value(Value::I32(v)) => {
            if (*v as u32) >= 1_000_000 {
                out.insert(*v as u32);
            }
        }
        Node::Value(_) => {}
    }
}

fn pool_items<'a>(p: &'a Node) -> Result<Vec<&'a Vec<Node>>> {
    let mut v = Vec::new();
    for (name, items) in pool_arrays(p)? {
        if name != "CAI_BDI_POOL_FAILED" {
            v.extend(items.iter());
        }
    }
    Ok(v)
}

pub fn verify(ctx: &BuildContext, ntw: &Node, ci: &Node) -> Result<CheckReport> {
    let mut rep = CheckReport {
        lines: Vec::new(),
        bad: 0,
        universe: 0,
        learned: 0,
        world_learned: 0,
    };
    let mut paths = Paths::new();

    // ---- learn the reference fields from the NTW baseline
    let nci = get(ntw, &["CAI_INTERFACE"])?;
    let un = universe(nci)?;
    let nf = collect(nci, &mut paths)?;
    let refs = learn(&nf, |v| un.contains(&v), 100);
    let uc = universe(ci)?;
    let cf = collect(ci, &mut paths)?;
    rep.universe = uc.len();
    rep.learned = refs.len();

    // ---- dangling CAI ids in learned reference fields
    let mut dangling: Vec<(Key, usize, Vec<u32>)> = Vec::new();
    for key in &refs {
        if let Some(vals) = cf.get(key) {
            let miss: Vec<u32> = vals
                .iter()
                .copied()
                .filter(|v| is_ref(*v, 100) && !uc.contains(v))
                .collect();
            if !miss.is_empty() {
                dangling.push((*key, vals.len(), miss));
            }
        }
    }
    dangling.sort_by_key(|d| d.0);
    for (key, n, miss) in dangling {
        rep.add(
            miss.len(),
            format!(
                "DANGLING CAI id in {}#{}: {}/{} e.g. {:?}",
                paths.render(key.0),
                key.1,
                miss.len(),
                n,
                &miss[..miss.len().min(3)]
            ),
        );
    }

    // ---- pool items: header lists must resolve; header #10 pairs; duplicate ids
    let dir_pool = get(ci, &["CAI_INTERFACE_DIRECTOR_BDI_POOL", "CAI_BDI_POOL"])?;
    let root_pool = get(ci, &["CAI_BDI_POOL"])?;
    let mgr_items = get_items(ci, &["CAI_INTERFACE_MANAGERS"])?;
    let mut pools: Vec<(String, &Node)> = vec![
        ("root".to_string(), root_pool),
        ("director".to_string(), dir_pool),
    ];
    for (k, m) in mgr_items.iter().enumerate() {
        pools.push((
            format!("manager {k}"),
            m.first().ok_or_else(|| anyhow!("empty CAI manager item"))?,
        ));
    }
    let mut ids_seen: HashMap<u32, usize> = HashMap::new();
    let mut allit: HashMap<u32, &Vec<Node>> = HashMap::new();
    let mut strict: HashMap<(String, usize), (usize, u32, u32)> = HashMap::new();
    const HDR_LISTS: [usize; 8] = [9, 10, 13, 14, 18, 19, 20, 21];
    for (tag, p) in &pools {
        let tag_class = if tag == "root" || tag == "director" {
            tag.clone()
        } else {
            "manager".to_string()
        };
        for it in pool_items(p)? {
            let id = sc_at(it, 2)?;
            *ids_seen.entry(id).or_insert(0) += 1;
            allit.insert(id, it);
            for i in HDR_LISTS {
                if let Some(c) = it.get(i) {
                    if is_u32_arr(c) {
                        for v in arr_vals(c) {
                            if v >= 100 && !uc.contains(&v) {
                                let e = strict.entry((tag_class.clone(), i)).or_insert((0, v, id));
                                e.0 += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut strict_keys: Vec<_> = strict.keys().cloned().collect();
    strict_keys.sort();
    for k in strict_keys {
        let (n, v, id) = strict[&k];
        rep.add(
            n,
            format!(
                "UNRESOLVED pool header list #{} of {} pools: {} values, e.g. {} in item {}",
                k.1, k.0, n, v, id
            ),
        );
    }
    for a in kids(get(ci, &["CAI_WORLD"])?)? {
        if let Node::Array { items, .. } = a {
            for it in items {
                if let Some(h) = item_hdr(it) {
                    *ids_seen.entry(h).or_insert(0) += 1;
                    allit.entry(h).or_insert(it);
                }
            }
        }
    }
    let mut dups: Vec<u32> = ids_seen
        .iter()
        .filter(|(_, n)| **n > 1)
        .map(|(k, _)| *k)
        .collect();
    dups.sort_unstable();
    rep.add(
        dups.len(),
        format!(
            "DUPLICATE CAI ids: {} e.g. {:?}",
            dups.len(),
            &dups[..dups.len().min(6)]
        ),
    );

    // header #10 = [(provider id, position)] pairs: position < len(provider.BLOCK_OWNS) and BLOCK_OWNS[position] is the item
    let mut pair_bad: std::collections::BTreeMap<&str, (usize, String)> =
        std::collections::BTreeMap::new();
    let mut xids: Vec<u32> = allit.keys().copied().collect();
    xids.sort_unstable();
    for xid in xids {
        let it = allit[&xid];
        if it.len() <= 10 || !is_u32_arr(&it[10]) {
            continue;
        }
        if is_record_named(&it[0], PROPSET) {
            continue; // world items carry no #10 list in this position
        }
        let l = arr_vals(&it[10]);
        for pair in l.chunks_exact(2) {
            let (t_, ix) = (pair[0], pair[1] as usize);
            let mut flag = |kind: &'static str, detail: String| {
                let e = pair_bad.entry(kind).or_insert((0, detail));
                e.0 += 1;
            };
            let Some(target) = allit.get(&t_) else {
                flag("target is not a CAI item", format!("item {xid} -> {t_}"));
                continue;
            };
            let Some(ow) = target.iter().find(|x| is_array_named(x, BLOCK_OWNS)) else {
                flag("target has no BLOCK_OWNS", format!("item {xid} -> {t_}"));
                continue;
            };
            let owned = arr_items(ow)?;
            if ix >= owned.len() {
                flag(
                    "position >= len(BLOCK_OWNS)",
                    format!("item {xid} -> {t_} position {ix} of {}", owned.len()),
                );
            } else if owned[ix].first().and_then(scalar) != Some(xid) {
                flag(
                    "BLOCK_OWNS[position] is another item",
                    format!("item {xid} -> {t_} position {ix}"),
                );
            }
        }
    }
    for (kind, (n, detail)) in pair_bad {
        rep.add(n, format!("PAIR-INVARIANT {kind}: {n} e.g. {detail}"));
    }

    // ---- CAI -> WORLD references
    let nm_world = get(ntw, &["WORLD"])?;
    let mut wn = HashSet::new();
    world_ids(nm_world, &[], &mut wn);
    let mut wc = HashSet::new();
    world_ids(nm_world, &["FACTION_ARRAY", "REGIONS_ARRAY"], &mut wc);
    for f in ctx.frec.values() {
        world_ids(f, &[], &mut wc);
    }
    for r in &ctx.regions {
        world_ids(r, &[], &mut wc);
    }
    let mut wrefs: Vec<Key> = learn(&nf, |v| wn.contains(&v), 1_000_000)
        .into_iter()
        .filter(|k| !refs.contains(k))
        .collect();
    wrefs.sort_unstable();
    rep.world_learned = wrefs.len();
    for key in &wrefs {
        if let Some(vals) = cf.get(key) {
            let miss: Vec<u32> = vals
                .iter()
                .copied()
                .filter(|v| is_ref(*v, 1_000_000) && !wc.contains(v))
                .collect();
            rep.add(
                miss.len(),
                format!(
                    "WORLD-DANGLING {}#{}: {} e.g. {:?}",
                    paths.render(key.0),
                    key.1,
                    miss.len(),
                    &miss[..miss.len().min(3)]
                ),
            );
        }
    }

    // ---- REGION child 22 and CAI governorships -> world governorships
    let mut gov_ids: HashSet<u32> = HashSet::new();
    for f in ctx.frec.values() {
        for p in get_items(f, &["GOVERNMENT", "POSTS_ARRAY"])? {
            if let Some(post) = p.first() {
                for c in kids(post)? {
                    if is_record_named(c, "GOVERNORSHIP") {
                        if let Some(g) = kids(c)?
                            .iter()
                            .find(|x| matches!(x, Node::Value(Value::I32(_))))
                            .and_then(scalar)
                        {
                            gov_ids.insert(g);
                        }
                    }
                }
            }
        }
    }
    let mut rbad: Vec<(u32, u32)> = Vec::new();
    for r in &ctx.regions {
        let ch = kids(r)?;
        let g = sc_at(ch, 22)?;
        if !gov_ids.contains(&g) {
            rbad.push((sc_at(ch, 0).unwrap_or(0), g));
        }
    }
    rep.add(
        rbad.len(),
        format!(
            "REGION#22 -> world governorship dangling: {} of {} e.g. {:?}",
            rbad.len(),
            ctx.regions.len(),
            &rbad[..rbad.len().min(3)]
        ),
    );
    let mut gov0_bad = 0;
    for it in get_items(ci, &["CAI_WORLD", "CAI_WORLD_GOVERNORSHIPS"])? {
        if !gov_ids.contains(&sc_at(spec_kids(it)?, 0)?) {
            gov0_bad += 1;
        }
    }
    rep.add(
        gov0_bad,
        format!("CAI_GOVERNORSHIP#0 is not a world governorship: {gov0_bad}"),
    );
    Ok(rep)
}

/// Every fleet of the start position stands in the sea grid cell its position falls into under the grid of the
/// finished map, and the cell and the fleet agree on it: the fleet's `CAI_MOBILE` names that cell (spec child 15) and
/// the cell lists the fleet. The stock start positions satisfy this for every fleet. Returns one line per violation.
pub fn fleet_cell_problems(ci: &Node, sea: &SeaGrids) -> Result<Vec<String>> {
    let world = get(ci, &["CAI_WORLD"])?;
    let cells = get_items(world, &["CAI_WORLD_SEA_GRID_CELLS"])?;
    let mut problems = Vec::new();
    if cells.len() != sea.cell_count() {
        problems.push(format!(
            "the start position holds {} CAI sea grid cells, sea_grids.esf {}",
            cells.len(),
            sea.cell_count()
        ));
        return Ok(problems);
    }
    let mut cell_of_id: HashMap<u32, usize> = HashMap::new();
    let mut listed: Vec<Vec<u32>> = Vec::with_capacity(cells.len());
    for (gi, it) in cells.iter().enumerate() {
        cell_of_id.insert(item_id(it)?, gi);
        let list = spec_kids(it)?
            .iter()
            .find(|c| matches!(c, Node::Array { .. }))
            .ok_or_else(|| anyhow!("CAI sea grid cell {gi} has no mobile array"))?;
        let mut ids = Vec::new();
        for m in arr_items(list)? {
            ids.push(sc_at(m, 0)?);
        }
        ids.sort_unstable();
        listed.push(ids);
    }
    let mut expected: Vec<Vec<u32>> = vec![Vec::new(); cells.len()];
    for it in get_items(world, &["CAI_WORLD_RESOURCE_MOBILES"])? {
        let sit = kids(rec_by_name(it, "CAI_SITUATED")?)?;
        if sc_at(sit, 4)? != 6 {
            continue;
        }
        let id = item_id(it)?;
        let (x, y) = (f64::from(sc_at(sit, 0)? as i32) / 1_048_576.0, f64::from(sc_at(sit, 1)? as i32) / 1_048_576.0);
        let want = sea.cell_index_at(x, y);
        let raw = arr_at(spec_kids(it)?, 15)?;
        let matches = match want {
            Some(w) => raw.len() == 1 && cell_of_id.get(&raw[0]) == Some(&w),
            None => raw.is_empty(),
        };
        if !matches {
            let named: Vec<usize> = raw.iter().filter_map(|c| cell_of_id.get(c).copied()).collect();
            problems.push(format!(
                "fleet {id} at ({x:.1}, {y:.1}) belongs to sea grid cell {want:?} but names cell(s) {named:?}"
            ));
        }
        if let Some(w) = want {
            expected[w].push(id);
        }
    }
    for (gi, want) in expected.iter_mut().enumerate() {
        want.sort_unstable();
        if *want != listed[gi] {
            problems.push(format!("sea grid cell {gi} lists fleets {:?} but {:?} stand in it", listed[gi], want));
        }
    }
    Ok(problems)
}

#[cfg(test)]
mod fleet_cell_tests {
    use super::*;
    use crate::regionmap::sea_grids::{Area, Cell, TheatreGrid};

    fn grid(name: &str, min: [f32; 2], max: [f32; 2], factor: f32, nx: u32, ny: u32) -> TheatreGrid {
        let mut cells = Vec::new();
        let mut areas = Vec::new();
        for yi in 0..ny {
            for xi in 0..nx {
                let lo = [min[0] + xi as f32 * factor, min[1] + yi as f32 * factor];
                cells.push(Cell { xi, yi, lo, hi: [lo[0] + factor, lo[1] + factor] });
                areas.push(Area { id: areas.len() as u32, lands: vec![], seas: vec![], ports: vec![], numbers: vec![] });
            }
        }
        TheatreGrid { name: name.into(), min, max, factor, xsize: nx, ysize: ny, cells, areas, connections: vec![] }
    }

    fn rec(name: &str, children: Vec<Node>) -> Node {
        Node::Record { name: name.into(), version: 0, children }
    }

    fn arr(name: &str, items: Vec<Vec<Node>>) -> Node {
        Node::Array { name: name.into(), version: 0, items }
    }

    /// CAI id of the cell item `gi`
    fn cell_id(gi: usize) -> u32 {
        7000 + gi as u32
    }

    fn fixed(v: f64) -> i32 {
        (v * 1_048_576.0) as i32
    }

    /// interface with `cells` cells (lists as given) and one mobile per `(id, kind, x, y, named cell)`
    fn interface(lists: &[&[u32]], mobiles: &[(u32, u32, f64, f64, Option<usize>)]) -> Node {
        let cells = lists
            .iter()
            .enumerate()
            .map(|(gi, l)| {
                vec![
                    u32n(cell_id(gi)),
                    rec(
                        "CAI_SEA_GRID_CELL",
                        vec![u32n(gi as u32), arr("CAI_SEA_GRID_CELL_MOBILES", l.iter().map(|m| vec![u32n(*m)]).collect())],
                    ),
                ]
            })
            .collect();
        let mobs = mobiles
            .iter()
            .map(|&(id, kind, x, y, named)| {
                let mut spec = vec![u32n(0); 16];
                spec[15] = arrn(&named.map(|c| vec![cell_id(c)]).unwrap_or_default());
                vec![
                    u32n(id),
                    rec("CAI_SITUATED", vec![i32n(fixed(x)), i32n(fixed(y)), u32n(1), arrn(&[1]), u32n(kind)]),
                    rec("CAI_MOBILE", spec),
                ]
            })
            .collect();
        rec(
            "CAI_INTERFACE",
            vec![rec(
                "CAI_WORLD",
                vec![arr("CAI_WORLD_SEA_GRID_CELLS", cells), arr("CAI_WORLD_RESOURCE_MOBILES", mobs)],
            )],
        )
    }

    /// two theatres (4 + 4 cells): `3` is the vanilla India grid, `2` lies elsewhere
    fn two_theatres(india_min: [f32; 2], india_max: [f32; 2], factor: f32) -> SeaGrids {
        SeaGrids { grids: vec![grid("2", [-180.0, 140.0], [440.0, 520.0], 310.0, 2, 2), grid("3", india_min, india_max, factor, 2, 2)] }
    }

    #[test]
    fn cell_rule_follows_the_grid_that_was_moved() {
        let vanilla = two_theatres([440.0, 20.0], [680.0, 280.0], 75.0);
        let moved = two_theatres([440.0, -80.0], [790.0, 300.0], 95.0);
        // a fleet of the stock India start position
        assert_eq!(vanilla.cell_index_at(565.3, 50.7), Some(4 + 1));
        assert_eq!(moved.cell_index_at(565.3, 50.7), Some(4 + 3));
        // outside every theatre rectangle
        assert_eq!(moved.cell_index_at(900.0, 50.0), None);
        assert_eq!(moved.cell_count(), 8);
    }

    #[test]
    fn fleet_filed_by_the_stock_grid_is_reported_on_the_moved_grid() {
        let moved = two_theatres([440.0, -80.0], [790.0, 300.0], 95.0);
        // position (565.3, 50.7) is in cell 7 of the moved grid; the start position built against the stock grid
        // files it in cell 5
        let stale = interface(&[&[], &[], &[], &[], &[], &[9], &[], &[], ], &[(9, 6, 565.3, 50.7, Some(5))]);
        let problems = fleet_cell_problems(&stale, &moved).unwrap();
        assert_eq!(problems.len(), 3, "{problems:?}");
    }

    #[test]
    fn consistent_fleets_pass_and_land_forces_are_ignored() {
        let moved = two_theatres([440.0, -80.0], [790.0, 300.0], 95.0);
        let ok = interface(
            &[&[], &[], &[], &[], &[], &[], &[], &[9, 11]],
            &[
                (9, 6, 565.3, 50.7, Some(7)),
                (11, 6, 600.0, 60.0, Some(7)),
                // an army somewhere in the same cell is not a fleet
                (12, 4, 600.0, 60.0, None),
                // a fleet outside every grid is in no cell
                (13, 6, 900.0, 50.0, None),
            ],
        );
        assert_eq!(fleet_cell_problems(&ok, &moved).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn cell_count_mismatch_is_reported() {
        let moved = two_theatres([440.0, -80.0], [790.0, 300.0], 95.0);
        let few = interface(&[&[], &[]], &[]);
        assert_eq!(fleet_cell_problems(&few, &moved).unwrap().len(), 1);
    }

    /// The stock start position satisfies the rule against the stock sea grid (the rule the generator now builds to).
    #[test]
    fn stock_start_position_follows_the_rule() {
        const WORK: &str = "F:/NTW2/work";
        let (start, sea) = (
            format!("{WORK}/exp4/backup/data/campaigns/wld_napoleon/startpos.esf"),
            format!("{WORK}/exp1/backup/campaign_maps/nap_world/sea_grids.esf"),
        );
        if !std::path::Path::new(&start).exists() || !std::path::Path::new(&sea).exists() {
            eprintln!("skipped: stock start position not available");
            return;
        }
        let start = crate::esf::read(&std::fs::read(start).unwrap()).unwrap();
        let sea = crate::esf::read(&std::fs::read(sea).unwrap()).unwrap();
        let sea = SeaGrids::parse(&sea.root).unwrap();
        let ci = start.root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"]).unwrap();
        assert_eq!(fleet_cell_problems(ci, &sea).unwrap(), Vec::<String>::new());
    }

    /// America (extended west, negative x), Europe and the extended India grid, in file order: 42 + 54 + 16 cells.
    /// `scale`/`shift` map the whole layout the way an atlas chart moves a theatre.
    fn three_theatres(scale: f64, shift: [f64; 2]) -> SeaGrids {
        let place = |p: [f32; 2]| [(p[0] as f64 * scale + shift[0]) as f32, (p[1] as f64 * scale + shift[1]) as f32];
        let theatre = |name: &str, min: [f32; 2], max: [f32; 2], factor: f32, nx: u32, ny: u32| {
            grid(name, place(min), place(max), (factor as f64 * scale) as f32, nx, ny)
        };
        SeaGrids {
            grids: vec![
                theatre("1", [-888.0, 0.0], [-340.0, 460.0], 92.0, 6, 7),
                theatre("2", [-180.0, 140.0], [440.0, 520.0], 75.0, 9, 6),
                theatre("3", [440.0, -80.0], [790.0, 300.0], 95.0, 4, 4),
            ],
        }
    }

    /// start position over `lists` (per cell, the fleets it names) and `fleets` (id, x, y, filed cell)
    fn filed(lists: &[Vec<u32>], fleets: &[(u32, f64, f64, Option<usize>)]) -> Node {
        let refs: Vec<&[u32]> = lists.iter().map(Vec::as_slice).collect();
        let mobiles: Vec<(u32, u32, f64, f64, Option<usize>)> = fleets.iter().map(|&(id, x, y, c)| (id, 6, x, y, c)).collect();
        interface(&refs, &mobiles)
    }

    /// Fleets at hand-computed positions across three theatres with negative coordinates, cell borders (exactly on
    /// them and just below) and the offset of each theatre in the global cell numbering.
    #[test]
    fn fleets_are_filed_by_theatre_offset_and_cell_borders() {
        let sea = three_theatres(1.0, [0.0, 0.0]);
        assert_eq!(sea.cell_count(), 112);
        // (id, x, y, expected global cell)
        let at: [(u32, f64, f64, Option<usize>); 13] = [
            (1, -888.0, 0.0, Some(0)),
            (2, -796.0, 0.0, Some(1)),
            (3, -796.001, 0.0, Some(0)),
            (4, -400.5, 459.9, Some(4 * 6 + 5)),
            (5, -341.0, 100.0, Some(6 + 5)),
            // x == the rectangle's upper edge: in no theatre
            (6, -340.0, 100.0, None),
            (7, -180.0, 140.0, Some(42)),
            (8, 0.0, 300.0, Some(42 + 2 * 9 + 2)),
            (9, 439.9, 519.9, Some(42 + 5 * 9 + 8)),
            (10, 565.3, 50.7, Some(96 + 4 + 1)),
            (11, 535.0, 15.0, Some(96 + 4 + 1)),
            (12, 534.999, 14.999, Some(96)),
            (13, 789.9, 299.9, Some(96 + 15)),
        ];
        let mut lists = vec![Vec::new(); 112];
        for &(id, x, y, cell) in &at {
            assert_eq!(sea.cell_index_at(x, y), cell, "fleet {id}");
            if let Some(c) = cell {
                lists[c].push(id);
            }
        }
        assert_eq!(fleet_cell_problems(&filed(&lists, &at), &sea).unwrap(), Vec::<String>::new());
        // one fleet filed in a neighbouring cell: its own reference, the cell that wrongly lists it and the cell that
        // lacks it are the three findings
        for (k, &(_, _, _, cell)) in at.iter().enumerate() {
            let Some(c) = cell else { continue };
            let wrong = if c + 1 < 112 { c + 1 } else { c - 1 };
            let mut fleets = at;
            fleets[k].3 = Some(wrong);
            let mut bad = lists.clone();
            bad[c].retain(|&id| id != at[k].0);
            bad[wrong].push(at[k].0);
            assert_eq!(fleet_cell_problems(&filed(&bad, &fleets), &sea).unwrap().len(), 3, "fleet {}", at[k].0);
            // the fleet names its cell but the cell does not list it: one finding
            let mut omitted = lists.clone();
            omitted[c].retain(|&id| id != at[k].0);
            assert_eq!(fleet_cell_problems(&filed(&omitted, &at), &sea).unwrap().len(), 1, "fleet {}", at[k].0);
        }
    }

    /// Fleets at the centre of every cell whose centre lies inside its theatre rectangle, in the stock layout and in
    /// an atlas-scaled, shifted layout: each is filed in its own cell. Positions of one layout filed against the
    /// other layout's grid are reported.
    #[test]
    fn every_cell_is_reachable_in_stock_and_atlas_scaled_layouts() {
        let centres = |sea: &SeaGrids| {
            let mut out = Vec::new();
            let mut offset = 0;
            for g in &sea.grids {
                for (i, c) in g.cells.iter().enumerate() {
                    let centre = [(c.lo[0] as f64 + c.hi[0] as f64) / 2.0, (c.lo[1] as f64 + c.hi[1] as f64) / 2.0];
                    if centre[0] < g.max[0] as f64 && centre[1] < g.max[1] as f64 {
                        out.push((centre[0], centre[1], offset + i));
                    }
                }
                offset += g.cells.len();
            }
            out
        };
        let stock = three_theatres(1.0, [0.0, 0.0]);
        let atlas = three_theatres(1.3516979217529297, [-20.5, 33.25]);
        for (name, sea) in [("stock", &stock), ("atlas", &atlas)] {
            let cells = centres(sea);
            // 30 America + 40 Europe + 16 India cells have their centre inside the rectangle
            assert_eq!(cells.len(), 86, "{name}");
            let fleets: Vec<(u32, f64, f64, Option<usize>)> =
                cells.iter().enumerate().map(|(n, &(x, y, c))| (1000 + n as u32, x, y, Some(c))).collect();
            let mut lists = vec![Vec::new(); 112];
            for &(id, _, _, c) in &fleets {
                lists[c.unwrap()].push(id);
            }
            assert_eq!(fleet_cell_problems(&filed(&lists, &fleets), sea).unwrap(), Vec::<String>::new(), "{name}");
        }
        // the atlas positions filed by the stock grid land in other cells or none
        let fleets: Vec<(u32, f64, f64, Option<usize>)> = centres(&atlas)
            .iter()
            .enumerate()
            .map(|(n, &(x, y, _))| (1000 + n as u32, x, y, stock.cell_index_at(x, y)))
            .collect();
        let mut lists = vec![Vec::new(); 112];
        for &(id, _, _, c) in &fleets {
            if let Some(c) = c {
                lists[c].push(id);
            }
        }
        assert!(!fleet_cell_problems(&filed(&lists, &fleets), &atlas).unwrap().is_empty());
    }
}
