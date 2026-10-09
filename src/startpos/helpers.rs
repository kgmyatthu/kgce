//! ESF tree utilities, id allocation, source templates (NTW eur_napoleon), unit database and the ETW region locator
//! used by the startpos builder. Port of `startpos/common.py`, `ids.py`, `entities.py` (non CAI part) and `src.py`.
//!
//! The Python builder addressed children by position in mixed lists (`ch[18]`, `ch[20]`...). The same layout is kept: a
//! `Node::Record`'s `children` mixes scalar `Node::Value`s with nested `Record`/`Array` nodes, and the "k-th scalar"
//! helpers below mirror Python's `vals(node)[k]`.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};
use regex::Regex;

use crate::db::CampaignTable;
use crate::esf::{self, Node, Value};
use crate::snapshot::{Game, Snapshot};

pub use crate::esf::{arr_u32, b, f32v, i32v, s16, u32v};

pub const FIXED_ONE: f64 = 1048576.0;

/// world coordinate -> 12.20 fixed point int32 used by LOCOMOTABLE / CAI_SITUATED
pub fn fixed(v: f64) -> i32 {
    (v * FIXED_ONE).round() as i32
}

pub fn unfixed(v: i32) -> f64 {
    v as f64 / FIXED_ONE
}

// ------------------------------------------------------------------------------------------------ value helpers

/// Python-style ESF type code of a value (arrays normalised to `0x40 | element code`).
pub fn vcode(v: &Value) -> u8 {
    match v {
        Value::Bool(_) => 1,
        Value::I8(_) => 2,
        Value::I16(_) => 3,
        Value::I32(_) => 4,
        Value::I64(_) => 5,
        Value::U8(_) => 6,
        Value::U16(_) => 7,
        Value::U32(_) => 8,
        Value::U64(_) => 9,
        Value::F32(_) => 10,
        Value::F64(_) => 11,
        Value::XY(_) => 12,
        Value::XYZ(_) => 13,
        Value::String16(_) => 14,
        Value::String8(_) => 15,
        Value::Angle(_) => 16,
        Value::Array { code, .. } => 0x40 | (code & 0x3f),
    }
}

/// type code of a node if it is a scalar/array value
pub fn ncode(n: &Node) -> Option<u8> {
    match n {
        Node::Value(v) => Some(vcode(v)),
        _ => None,
    }
}

pub fn is_value(n: &Node) -> bool {
    matches!(n, Node::Value(_))
}

/// `n` is a u32 packed array (Python code 0x48)
pub fn is_u32_arr(n: &Node) -> bool {
    ncode(n) == Some(0x48)
}

pub fn node_name(n: &Node) -> Option<&str> {
    match n {
        Node::Record { name, .. } | Node::Array { name, .. } => Some(name.as_str()),
        Node::Value(_) => None,
    }
}

pub fn is_named(n: &Node, name: &str) -> bool {
    node_name(n) == Some(name)
}

pub fn is_record(n: &Node, name: &str) -> bool {
    matches!(n, Node::Record { name: nm, .. } if nm == name)
}

pub fn is_array(n: &Node, name: &str) -> bool {
    matches!(n, Node::Array { name: nm, .. } if nm == name)
}

/// object id stored as i32 or u32
pub fn as_id(n: &Node) -> Option<u32> {
    match n {
        Node::Value(Value::I32(x)) => Some(*x as u32),
        Node::Value(Value::U32(x)) => Some(*x),
        _ => None,
    }
}

pub fn as_text(n: &Node) -> Option<&str> {
    match n {
        Node::Value(Value::String16(s)) | Node::Value(Value::String8(s)) => Some(s.as_str()),
        _ => None,
    }
}

pub fn as_flag(n: &Node) -> Option<bool> {
    match n {
        Node::Value(Value::Bool(x)) => Some(*x),
        _ => None,
    }
}

pub fn as_number(n: &Node) -> Option<f64> {
    match n {
        Node::Value(v) => match v {
            Value::I8(x) => Some(*x as f64),
            Value::I16(x) => Some(*x as f64),
            Value::I32(x) => Some(*x as f64),
            Value::I64(x) => Some(*x as f64),
            Value::U8(x) => Some(*x as f64),
            Value::U16(x) => Some(*x as f64),
            Value::U32(x) => Some(*x as f64),
            Value::U64(x) => Some(*x as f64),
            Value::F32(x) => Some(*x as f64),
            Value::F64(x) => Some(*x),
            _ => None,
        },
        _ => None,
    }
}

pub fn u8v(v: u8) -> Node {
    Node::Value(Value::U8(v))
}

pub fn xyz(v: [f32; 3]) -> Node {
    Node::Value(Value::XYZ(v))
}

pub fn xy(v: [f32; 2]) -> Node {
    Node::Value(Value::XY(v))
}

pub fn i32_from_id(id: u32) -> Node {
    i32v(id as i32)
}

/// decode the u32 elements of a packed u32/i32 array node
pub fn arr_values(n: &Node) -> Option<Vec<u32>> {
    match n {
        Node::Value(Value::Array { code, data }) if matches!(code & 0x3f, 4 | 8) => Some(
            data.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        _ => None,
    }
}

// ------------------------------------------------------------------------------------------------ navigation

pub fn kids(n: &Node) -> Result<&Vec<Node>> {
    match n {
        Node::Record { children, .. } => Ok(children),
        other => bail!("expected a record, found {}", describe(other)),
    }
}

pub fn kids_mut(n: &mut Node) -> Result<&mut Vec<Node>> {
    match n {
        Node::Record { children, .. } => Ok(children),
        other => bail!("expected a record, found {}", describe(other)),
    }
}

pub fn items(n: &Node) -> Result<&Vec<Vec<Node>>> {
    match n {
        Node::Array { items, .. } => Ok(items),
        other => bail!("expected a record array, found {}", describe(other)),
    }
}

pub fn items_mut(n: &mut Node) -> Result<&mut Vec<Vec<Node>>> {
    match n {
        Node::Array { items, .. } => Ok(items),
        other => bail!("expected a record array, found {}", describe(other)),
    }
}

fn describe(n: &Node) -> String {
    match n {
        Node::Record { name, .. } => format!("record {name}"),
        Node::Array { name, .. } => format!("array {name}"),
        Node::Value(v) => format!("value(code {})", vcode(v)),
    }
}

/// navigate by record/array names (all but the last step must be records)
pub fn get<'a>(n: &'a Node, path: &[&str]) -> Result<&'a Node> {
    let mut cur = n;
    for p in path {
        let ch = kids(cur).with_context(|| format!("while looking for {p}"))?;
        cur = ch
            .iter()
            .find(|c| is_named(c, p))
            .ok_or_else(|| anyhow!("missing ESF node {p}"))?;
    }
    Ok(cur)
}

pub fn get_mut<'a>(n: &'a mut Node, path: &[&str]) -> Result<&'a mut Node> {
    let mut cur = n;
    for p in path {
        let ch = kids_mut(cur).with_context(|| format!("while looking for {p}"))?;
        let i = ch
            .iter()
            .position(|c| is_named(c, p))
            .ok_or_else(|| anyhow!("missing ESF node {p}"))?;
        cur = &mut ch[i];
    }
    Ok(cur)
}

/// same as `get` over a child list
pub fn child<'a>(children: &'a [Node], name: &str) -> Result<&'a Node> {
    children
        .iter()
        .find(|c| is_named(c, name))
        .ok_or_else(|| anyhow!("missing ESF node {name}"))
}

pub fn child_mut<'a>(children: &'a mut [Node], name: &str) -> Result<&'a mut Node> {
    children
        .iter_mut()
        .find(|c| is_named(c, name))
        .ok_or_else(|| anyhow!("missing ESF node {name}"))
}

/// index of the first Record/Array child with `name`
pub fn idx_of(children: &[Node], name: &str) -> Result<usize> {
    children
        .iter()
        .position(|c| is_named(c, name))
        .ok_or_else(|| anyhow!("missing ESF node {name}"))
}

/// positions of the scalar children
pub fn scalar_idx(children: &[Node]) -> Vec<usize> {
    children
        .iter()
        .enumerate()
        .filter(|(_, c)| is_value(c))
        .map(|(i, _)| i)
        .collect()
}

/// k-th scalar child (Python `vals(node)[k]`)
pub fn nth_val(children: &[Node], k: usize) -> Result<&Node> {
    children
        .iter()
        .filter(|c| is_value(c))
        .nth(k)
        .ok_or_else(|| anyhow!("record has no scalar #{k}"))
}

pub fn nth_id(children: &[Node], k: usize) -> Result<u32> {
    as_id(nth_val(children, k)?).ok_or_else(|| anyhow!("scalar #{k} is not an object id"))
}

pub fn nth_text(children: &[Node], k: usize) -> Result<&str> {
    as_text(nth_val(children, k)?).ok_or_else(|| anyhow!("scalar #{k} is not a string"))
}

pub fn set_nth_val(children: &mut [Node], k: usize, value: Node) -> Result<()> {
    let idx = scalar_idx(children);
    let i = *idx
        .get(k)
        .ok_or_else(|| anyhow!("record has no scalar #{k}"))?;
    children[i] = value;
    Ok(())
}

pub fn id_at(children: &[Node], i: usize) -> Result<u32> {
    children
        .get(i)
        .and_then(as_id)
        .ok_or_else(|| anyhow!("child #{i} is not an object id"))
}

pub fn text_at(children: &[Node], i: usize) -> Result<&str> {
    children
        .get(i)
        .and_then(as_text)
        .ok_or_else(|| anyhow!("child #{i} is not a string"))
}

pub fn expect_code(children: &[Node], i: usize, code: u8, what: &str) -> Result<()> {
    match children.get(i).and_then(ncode) {
        Some(c) if c == code => Ok(()),
        other => bail!("{what}: child #{i} has type {other:?}, expected {code}"),
    }
}

/// pre-order visit of every Record / Array below `children`
pub fn walk_mut(children: &mut [Node], f: &mut dyn FnMut(&mut Node)) {
    for c in children.iter_mut() {
        if !is_value(c) {
            f(c);
        }
        match c {
            Node::Record { children, .. } => walk_mut(children, f),
            Node::Array { items, .. } => {
                for it in items.iter_mut() {
                    walk_mut(it, f);
                }
            }
            Node::Value(_) => {}
        }
    }
}

pub fn walk(children: &[Node], f: &mut dyn FnMut(&Node)) {
    for c in children {
        if !is_value(c) {
            f(c);
        }
        match c {
            Node::Record { children, .. } => walk(children, f),
            Node::Array { items, .. } => {
                for it in items {
                    walk(it, f);
                }
            }
            Node::Value(_) => {}
        }
    }
}

// ------------------------------------------------------------------------------------------------ id remapping

fn remap_value(v: &mut Value, map: &BTreeMap<u32, u32>) {
    match v {
        Value::I32(x) => {
            if *x >= 0 {
                if let Some(n) = map.get(&(*x as u32)) {
                    *x = *n as i32;
                }
            }
        }
        Value::U32(x) => {
            if let Some(n) = map.get(x) {
                *x = *n;
            }
        }
        Value::Array { code, data } if matches!(*code & 0x3f, 4 | 8) => {
            for chunk in data.chunks_exact_mut(4) {
                let x = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if let Some(n) = map.get(&x) {
                    chunk.copy_from_slice(&n.to_le_bytes());
                }
            }
        }
        _ => {}
    }
}

/// Replace u32/i32 scalars and packed u32/i32 elements equal to a mapping key, recursively.
pub fn remap_ids(children: &mut [Node], map: &BTreeMap<u32, u32>) {
    if map.is_empty() {
        return;
    }
    for c in children.iter_mut() {
        match c {
            Node::Value(v) => remap_value(v, map),
            Node::Record { children, .. } => remap_ids(children, map),
            Node::Array { items, .. } => {
                for it in items.iter_mut() {
                    remap_ids(it, map);
                }
            }
        }
    }
}

/// Same as `remap_ids`, direct (scalar) children only.
pub fn remap_scalars(children: &mut [Node], map: &BTreeMap<u32, u32>) {
    for c in children.iter_mut() {
        if let Node::Value(v) = c {
            remap_value(v, map);
        }
    }
}

// ------------------------------------------------------------------------------------------------ ids

fn collect_used_into(children: &[Node], used: &mut HashSet<u32>) {
    for c in children {
        match c {
            Node::Value(v) => match v {
                Value::I32(x) => {
                    used.insert(*x as u32);
                }
                Value::U32(x) => {
                    used.insert(*x);
                }
                Value::I64(x) => {
                    used.insert((*x as u64 & 0xFFFF_FFFF) as u32);
                }
                Value::U64(x) => {
                    used.insert((*x & 0xFFFF_FFFF) as u32);
                }
                Value::Array { code, data } if matches!(*code & 0x3f, 4 | 8) => {
                    for ch in data.chunks_exact(4) {
                        used.insert(u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
                    }
                }
                _ => {}
            },
            Node::Record { children, .. } => collect_used_into(children, used),
            Node::Array { items, .. } => {
                for it in items {
                    collect_used_into(it, used);
                }
            }
        }
    }
}

/// every integer value (and packed element) found in the given trees: fresh object ids must avoid all of them
pub fn collect_used(roots: &[&Node]) -> HashSet<u32> {
    let mut used = HashSet::new();
    for r in roots {
        match r {
            Node::Record { children, .. } => collect_used_into(children, &mut used),
            other => collect_used_into(std::slice::from_ref(*other), &mut used),
        }
    }
    used
}

/// World object ids (multiples of four) and CAI ids.
#[derive(Clone, Debug)]
pub struct IdAlloc {
    used: HashSet<u32>,
    w: u32,
    c: u32,
}

impl IdAlloc {
    pub fn new(used: HashSet<u32>, world_start: u32, cai_start: u32) -> Self {
        Self {
            used,
            w: world_start,
            c: cai_start,
        }
    }

    pub fn world(&mut self) -> u32 {
        while self.used.contains(&self.w) {
            self.w += 4;
        }
        let v = self.w;
        self.w += 4;
        self.used.insert(v);
        v
    }

    pub fn cai(&mut self) -> u32 {
        let v = self.c;
        self.c += 1;
        v
    }
}

// ------------------------------------------------------------------------------------------------ model access

/// CAMPAIGN_MODEL record of a startpos root
pub fn model_of(root: &Node) -> Result<&Node> {
    get(root, &["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])
}

pub fn model_of_mut(root: &mut Node) -> Result<&mut Node> {
    get_mut(root, &["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])
}

pub fn read_esf(snapshot: &Snapshot, game: Game, path: &str) -> Result<esf::Esf> {
    let bytes = snapshot
        .read(game, path)
        .with_context(|| format!("Read {path} from the source snapshot"))?;
    esf::read(&bytes).with_context(|| format!("Parse {path}"))
}

/// faction key stored in CAMPAIGN_PLAYER_SETUP child 2
pub fn faction_key(rec: &Node) -> Result<String> {
    let cps = get(rec, &["CAMPAIGN_PLAYER_SETUP"])?;
    Ok(text_at(kids(cps)?, 2)?.to_string())
}

// ------------------------------------------------------------------------------------------------ JSON helpers

pub fn jstr<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("rule field {key} missing or not a string"))
}

pub fn jnum(v: &serde_json::Value) -> Option<i64> {
    match v {
        serde_json::Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        serde_json::Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

// ------------------------------------------------------------------------------------------------ unit database

/// men / crew counts of the effective NTW unit tables (the target's when the snapshot carries a mod)
#[derive(Clone, Debug, Default)]
pub struct UnitDb {
    pub land_men: BTreeMap<String, u32>,
    /// (marines, seamen, gunners)
    pub naval_crew: BTreeMap<String, (u32, u32, u32)>,
    /// land units whose men differ from (or are absent in) the immutable original snapshot: the template instances of
    /// the original start position carry the original counts, these must be rewritten
    pub land_changed: BTreeMap<String, u32>,
    /// same for ship crews
    pub naval_changed: BTreeMap<String, (u32, u32, u32)>,
}

type UnitTables = (BTreeMap<String, u32>, BTreeMap<String, (u32, u32, u32)>);

fn read_unit_tables(snapshot: &Snapshot, rules: &crate::model::Rules) -> Result<UnitTables> {
    let mut land = BTreeMap::new();
    let mut naval = BTreeMap::new();
    for row in
        crate::db::load_campaign_table(snapshot, rules, CampaignTable::LandStats)?.json_rows()?
    {
        let key = jstr(&row, "key")?.to_string();
        let men = row
            .get("men")
            .and_then(jnum)
            .ok_or_else(|| anyhow!("unit_stats_land {key}: no men"))?;
        land.insert(key, men as u32);
    }
    for row in
        crate::db::load_campaign_table(snapshot, rules, CampaignTable::NavalStats)?.json_rows()?
    {
        let key = jstr(&row, "key")?.to_string();
        let n = |c: &str| -> Result<u32> {
            Ok(row
                .get(c)
                .and_then(jnum)
                .ok_or_else(|| anyhow!("unit_stats_naval {key}: no {c}"))? as u32)
        };
        let crew = (n("num_marines")?, n("num_seamen")?, n("num_gunners")?);
        naval.insert(key, crew);
    }
    Ok((land, naval))
}

impl UnitDb {
    pub fn load(snapshot: &Snapshot, rules: &crate::model::Rules) -> Result<Self> {
        let (land_men, naval_crew) = read_unit_tables(snapshot, rules)?;
        if land_men.is_empty() || naval_crew.is_empty() {
            bail!("NTW unit tables are empty");
        }
        // the start position templates were saved against the original tables: only values the target really changed
        // are rewritten, so an unmodified target reproduces the original instances byte for byte
        let (base_land, base_naval) = read_unit_tables(&snapshot.original(), rules)?;
        let land_changed = land_men
            .iter()
            .filter(|(k, v)| base_land.get(*k) != Some(*v))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let naval_changed = naval_crew
            .iter()
            .filter(|(k, v)| base_naval.get(*k) != Some(*v))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        Ok(UnitDb {
            land_men,
            naval_crew,
            land_changed,
            naval_changed,
        })
    }
}

// ------------------------------------------------------------------------------------------------ NTW templates

/// Indexes of NTW eur_napoleon entities used as clone sources (non-CAI part of Python `Templates`).
pub struct Templates {
    /// NTW faction keys in NTW FACTION_ARRAY order
    pub fac_order: Vec<String>,
    pub fac_by_key: BTreeMap<String, Node>,
    /// faction -> role -> [(CHARACTER, ARMY/NAVY)]
    pub force_pool: BTreeMap<String, BTreeMap<String, Vec<(Node, Node)>>>,
    /// unit key in first-seen order
    pub unit_order: Vec<String>,
    /// unit key -> LAND_UNIT / NAVAL_UNIT instances
    pub unit_inst: HashMap<String, Vec<Node>>,
    prefix_pair: HashMap<String, (i32, i32)>,
}

static PREFIX_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(
        r"^(Inf_[A-Za-z]+|Cav_[A-Za-z]+|Art_[A-Za-z]+|Mounted_Inf|Gen|Frigate|2_Decker|3_Decker|1_Decker|Small|Sloop|Corvette|[A-Za-z0-9]+)",
    )
    .expect("static regex")
});

pub fn unit_prefix(key: &str) -> String {
    match PREFIX_RE.find(key) {
        Some(m) => m.as_str().to_string(),
        None => key.to_string(),
    }
}

impl Templates {
    pub fn new(ntw_model: &Node) -> Result<Self> {
        let world = get(ntw_model, &["WORLD"])?;
        let mut t = Templates {
            fac_order: Vec::new(),
            fac_by_key: BTreeMap::new(),
            force_pool: BTreeMap::new(),
            unit_order: Vec::new(),
            unit_inst: HashMap::new(),
            prefix_pair: HashMap::new(),
        };
        for it in items(get(world, &["FACTION_ARRAY"])?)? {
            let rec = it
                .first()
                .ok_or_else(|| anyhow!("empty FACTION_ARRAY item"))?;
            let key = faction_key(rec)?;
            t.fac_order.push(key.clone());
            t.fac_by_key.insert(key, rec.clone());
        }
        for fkey in t.fac_order.clone() {
            let rec = &t.fac_by_key[&fkey];
            let mut chars: HashMap<u32, &Node> = HashMap::new();
            for it in items(get(rec, &["CHARACTER_ARRAY"])?)? {
                chars.insert(nth_id(kids(&it[0])?, 0)?, &it[0]);
            }
            let mut pool: BTreeMap<String, Vec<(Node, Node)>> = BTreeMap::new();
            let mut insts: Vec<(String, Node)> = Vec::new();
            for it in items(get(rec, &["ARMY_ARRAY"])?)? {
                let a = &it[0];
                let mf = kids(get(a, &["MILITARY_FORCE"])?)?;
                let cid = nth_id(mf, 1)?;
                let ch = chars.get(&cid).ok_or_else(|| {
                    anyhow!("{fkey}: army leader {cid} is not a character of the faction")
                })?;
                let role = nth_text(kids(ch)?, 1)?.to_string();
                pool.entry(role)
                    .or_default()
                    .push(((*ch).clone(), a.clone()));
                for ui in items(get(a, &["UNITS_ARRAY"])?)? {
                    let lu = &ui[0];
                    let rk = kids(lu)?
                        .first()
                        .ok_or_else(|| anyhow!("unit without record key"))?;
                    let key = text_at(kids(rk)?, 0)?.to_string();
                    insts.push((key, lu.clone()));
                }
            }
            if !pool.is_empty() {
                t.force_pool.insert(fkey.clone(), pool);
            }
            for (k, lu) in insts {
                if !t.unit_inst.contains_key(&k) {
                    t.unit_order.push(k.clone());
                }
                t.unit_inst.entry(k).or_default().push(lu);
            }
        }
        for key in t.unit_order.clone() {
            let lu = &t.unit_inst[&key][0];
            let un = kids(get(lu, &["UNIT"])?)?;
            let a = as_number(nth_val(un, 3)?)
                .ok_or_else(|| anyhow!("{key}: UNIT scalar 3 not numeric"))?
                as i32;
            let bb = as_number(nth_val(un, 10)?)
                .ok_or_else(|| anyhow!("{key}: UNIT scalar 10 not numeric"))?
                as i32;
            let pr = unit_prefix(&key);
            let replace = t.prefix_pair.get(&pr).map_or(true, |old| a < old.0);
            if replace {
                t.prefix_pair.insert(pr, (a, bb));
            }
        }
        Ok(t)
    }

    pub fn pair_for_prefix(&self, prefix: &str) -> Option<(i32, i32)> {
        self.prefix_pair.get(prefix).copied()
    }

    /// leaders of `role` for a faction's donor; falls back to the first faction that has such a leader
    pub fn pick_pool(&self, donor: &str, role: &str) -> Result<&Vec<(Node, Node)>> {
        if let Some(p) = self.force_pool.get(donor).and_then(|r| r.get(role)) {
            if !p.is_empty() {
                return Ok(p);
            }
        }
        for fk in &self.fac_order {
            if let Some(p) = self.force_pool.get(fk).and_then(|r| r.get(role)) {
                if !p.is_empty() {
                    return Ok(p);
                }
            }
        }
        bail!("no template leaders for {role}")
    }
}

// ------------------------------------------------------------------------------------------------ ETW global map

/// regions.esf of the ETW global map: point location (forces) and theatre rectangles (camera).
pub struct GlobalMap {
    verts: Vec<[f64; 2]>,
    regions: Vec<LocRegion>,
    pub theatres: Vec<([f32; 2], [f32; 2])>,
    pub region_data: ([f32; 2], [f32; 2]),
}

struct LocRegion {
    key: String,
    kind: String,
    tris: Vec<[u32; 3]>,
    bbox: [f64; 4],
}

pub const GLOBAL_REGIONS_PATH: &str = "campaign_maps/global_map/regions.esf";

/// Theatre rectangles (scalars of each `theatres` item are key, min xy, max xy, bool) and the `region_data` rectangle of a
/// `regions.esf` root.
pub fn theatre_rects(root: &Node) -> Result<(Vec<([f32; 2], [f32; 2])>, ([f32; 2], [f32; 2]))> {
    let rd = get(root, &["region_data"])?;
    let mut theatres = Vec::new();
    for it in items(get(root, &["theatres"])?)? {
        let ch: &Vec<Node> = if it.len() == 1 && matches!(it[0], Node::Record { .. }) {
            kids(&it[0])?
        } else {
            it
        };
        let lo = match nth_val(ch, 1)? {
            Node::Value(Value::XY(v)) => *v,
            _ => bail!("theatre minimum is not an XY value"),
        };
        let hi = match nth_val(ch, 2)? {
            Node::Value(Value::XY(v)) => *v,
            _ => bail!("theatre maximum is not an XY value"),
        };
        theatres.push((lo, hi));
    }
    if theatres.is_empty() {
        bail!("regions.esf has no theatres");
    }
    let rdk = kids(rd)?;
    let a = match nth_val(rdk, 0)? {
        Node::Value(Value::XY(v)) => *v,
        _ => bail!("region_data lower bound is not XY"),
    };
    let c = match nth_val(rdk, 1)? {
        Node::Value(Value::XY(v)) => *v,
        _ => bail!("region_data upper bound is not XY"),
    };
    Ok((theatres, (a, c)))
}

impl GlobalMap {
    pub fn load(snapshot: &Snapshot) -> Result<Self> {
        let esf = read_esf(snapshot, Game::Etw, GLOBAL_REGIONS_PATH)?;
        let root = &esf.root;
        let rd = get(root, &["region_data"])?;
        let vnode = get(rd, &["vertices"])?;
        let raw = kids(vnode)?
            .iter()
            .find_map(|c| match c {
                Node::Value(Value::Array { data, .. }) => Some(data),
                _ => None,
            })
            .ok_or_else(|| anyhow!("regions.esf vertices array missing"))?;
        let verts: Vec<[f64; 2]> = raw
            .chunks_exact(8)
            .map(|c| {
                [
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64,
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]) as f64,
                ]
            })
            .collect();
        let mut regions: Vec<LocRegion> = Vec::new();
        let mut by_key: HashMap<String, usize> = HashMap::new();
        for it in items(get(rd, &["regions"])?)? {
            let key = as_text(it.first().ok_or_else(|| anyhow!("empty region item"))?)
                .ok_or_else(|| anyhow!("region key"))?
                .to_string();
            let kind = as_text(it.get(1).ok_or_else(|| anyhow!("region kind"))?)
                .ok_or_else(|| anyhow!("region kind"))?
                .to_string();
            let areas = it
                .iter()
                .find(|c| matches!(c, Node::Array { .. }))
                .ok_or_else(|| anyhow!("region {key}: no areas"))?;
            let mut tris: Vec<[u32; 3]> = Vec::new();
            for area in items(areas)? {
                let faces = area
                    .iter()
                    .find(|c| is_record(c, "faces"))
                    .ok_or_else(|| anyhow!("region {key}: area without faces"))?;
                let data = kids(faces)?
                    .iter()
                    .find_map(|c| match c {
                        Node::Value(Value::Array { data, .. }) => Some(data),
                        _ => None,
                    })
                    .ok_or_else(|| anyhow!("region {key}: faces array missing"))?;
                let idx: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                for t in idx.chunks_exact(3) {
                    if t.iter().any(|&i| i as usize >= verts.len()) {
                        bail!("region {key}: face index outside the vertex pool");
                    }
                    tris.push([t[0], t[1], t[2]]);
                }
            }
            let mut bbox = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
            for t in &tris {
                for &i in t {
                    let v = verts[i as usize];
                    bbox[0] = bbox[0].min(v[0]);
                    bbox[1] = bbox[1].min(v[1]);
                    bbox[2] = bbox[2].max(v[0]);
                    bbox[3] = bbox[3].max(v[1]);
                }
            }
            let lr = LocRegion {
                key: key.clone(),
                kind,
                tris,
                bbox,
            };
            match by_key.get(&key) {
                Some(&i) => regions[i] = lr,
                None => {
                    by_key.insert(key, regions.len());
                    regions.push(lr);
                }
            }
        }
        let (theatres, region_data) = theatre_rects(root)?;
        Ok(Self {
            verts,
            regions,
            theatres,
            region_data,
        })
    }

    /// Replace the theatre rectangles and the `region_data` rectangle by those of the (edited) generated `regions.esf`
    /// in `root`: the camera bounds must follow the theatres the map really has, not the ETW originals.
    pub fn use_theatres_of(&mut self, root: &Node) -> Result<()> {
        (self.theatres, self.region_data) = theatre_rects(root)?;
        Ok(())
    }

    /// (region key, "land"|"sea"|"river") for an ETW world coordinate
    pub fn locate(&self, x: f64, y: f64) -> Option<(&str, &str)> {
        for r in &self.regions {
            if r.tris.is_empty() || x < r.bbox[0] || y < r.bbox[1] || x > r.bbox[2] || y > r.bbox[3]
            {
                continue;
            }
            for t in &r.tris {
                let a = self.verts[t[0] as usize];
                let b = self.verts[t[1] as usize];
                let c = self.verts[t[2] as usize];
                let d1 = (x - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (y - b[1]);
                let d2 = (x - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (y - c[1]);
                let d3 = (x - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (y - a[1]);
                let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
                let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
                if !(neg && pos) {
                    return Some((r.key.as_str(), r.kind.as_str()));
                }
            }
        }
        None
    }

    /// (x_min, y_min, x_max, y_max) the campaign camera target may reach: bounding box of every theatre rectangle
    /// plus `margin`, limited to the region_data rectangle.
    pub fn camera_bounds(&self, margin: f64) -> [f64; 4] {
        let lo = [
            self.theatres
                .iter()
                .map(|t| t.0[0] as f64)
                .fold(f64::MAX, f64::min),
            self.theatres
                .iter()
                .map(|t| t.0[1] as f64)
                .fold(f64::MAX, f64::min),
        ];
        let hi = [
            self.theatres
                .iter()
                .map(|t| t.1[0] as f64)
                .fold(f64::MIN, f64::max),
            self.theatres
                .iter()
                .map(|t| t.1[1] as f64)
                .fold(f64::MIN, f64::max),
        ];
        let (rl, rh) = self.region_data;
        [
            (lo[0] - margin).max(rl[0] as f64),
            (lo[1] - margin).max(rl[1] as f64),
            (hi[0] + margin).min(rh[0] as f64),
            (hi[1] + margin).min(rh[1] as f64),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The camera bounds (EPISODIC_RESTRICTIONS 9..12, `scripting.lua`) must follow the theatre rectangles of the
    /// edited generated regions.esf, not those of the ETW original.
    #[test]
    fn camera_bounds_follow_the_edited_theatre_rects() {
        let path = "F:/NTW2/work/exp5/out/data/campaign_maps/nap_world/regions.esf";
        let Ok(bytes) = std::fs::read(path) else { return };
        let edited = esf::read(&bytes).unwrap();
        let (rects, region_data) = theatre_rects(&edited.root).unwrap();
        let mut gm = GlobalMap {
            verts: Vec::new(),
            regions: Vec::new(),
            theatres: vec![([-10.0, -10.0], [10.0, 10.0])],
            region_data,
        };
        assert_eq!(gm.camera_bounds(60.0)[2], (10.0 + 60.0f64).min(region_data.1[0] as f64));
        gm.use_theatres_of(&edited.root).unwrap();
        assert_eq!(gm.theatres, rects);
        let b = gm.camera_bounds(60.0);
        let want_hi = [
            rects.iter().map(|t| t.1[0]).fold(f32::MIN, f32::max) as f64 + 60.0,
            rects.iter().map(|t| t.1[1]).fold(f32::MIN, f32::max) as f64 + 60.0,
        ];
        let want_lo = [
            rects.iter().map(|t| t.0[0]).fold(f32::MAX, f32::min) as f64 - 60.0,
            rects.iter().map(|t| t.0[1]).fold(f32::MAX, f32::min) as f64 - 60.0,
        ];
        assert_eq!(b[0], want_lo[0].max(region_data.0[0] as f64));
        assert_eq!(b[1], want_lo[1].max(region_data.0[1] as f64));
        assert_eq!(b[2], want_hi[0].min(region_data.1[0] as f64));
        assert_eq!(b[3], want_hi[1].min(region_data.1[1] as f64));
        // the extended India rectangle (440,-80)..(790,300) is inside the bounds
        assert!(b[2] >= 790.0 && b[3] >= 300.0 && b[1] <= -80.0, "camera bounds {b:?} do not cover the extended theatre");
    }
}
