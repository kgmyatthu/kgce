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
