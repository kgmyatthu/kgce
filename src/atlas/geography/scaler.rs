//! Rule-driven atlas scaling of the non-polygon `regions.esf` sections (port of the accepted POC
//! `regions_scale.py` `Scaler`/`RULES`/float audit/`check_scaled` plus the `_mountains`, `_bridges`,
//! trade-node and theatre helpers of `regions_atlas_build.py`).
//!
//! Every absolute horizontal coordinate is mapped through `Plan::xy` (`new = old * scale + chart offset`,
//! evaluated in f64 and rounded once to f32).  Heights, bridge angles and every other leaf are copied
//! verbatim.  The set of scaled fields is the explicit [`rules`] table keyed by (normalised container
//! path, child index); an independent audit ([`enumerate_float_fields`] / [`classify_float_fields`]) lists
//! every float-typed leaf so a new or unknown float field can never be silently skipped or scaled.
//!
//! Paths use the POC spelling: record arrays carry a trailing `[]` (the Rust ESF reader stores array
//! names without it, so it is appended here) and quadtree node names collapse to `Q` ([`norm`]).

use crate::atlas::{Chart, Plan};
use crate::esf::{Node, Value};
use anyhow::{Context, Result, bail, ensure};
use regex::Regex;
use serde_json::{Map, Value as Json};
use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

// ---------------------------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------------------------

/// How one leaf is scaled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    /// scalar v12 (x, y)
    Xy,
    /// scalar v13 stored (x, height, y): components 0 and 2 scaled
    Xyz,
    /// arr12: packed (x, y) f32 pairs
    ArrXy,
    /// arr13: packed (x, y, angle) f32 triples: components 0 and 1 scaled
    ArrXya,
}

impl Rule {
    /// POC spelling (`xy`, `xyz`, `arr_xy`, `arr_xya`).
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Rule::Xy => "xy",
            Rule::Xyz => "xyz",
            Rule::ArrXy => "arr_xy",
            Rule::ArrXya => "arr_xya",
        }
    }
}

type RuleSet = &'static [(usize, Rule)];

const XY_01: RuleSet = &[(0, Rule::Xy), (1, Rule::Xy)];
const XY_12: RuleSet = &[(1, Rule::Xy), (2, Rule::Xy)];
const XY_23: RuleSet = &[(2, Rule::Xy), (3, Rule::Xy)];
const XY_34: RuleSet = &[(3, Rule::Xy), (4, Rule::Xy)];

/// Rules of `/root/region_data` (political, with settlements/slots) or `/root/groundtypes/region_data`,
/// for the part of the path below the prefix.
fn region_data_rules(rest: &str, with_faces: bool) -> RuleSet {
    match rest {
        "" => XY_12,
        "/vertices" => &[(0, Rule::ArrXy)],
        "/regions[]" => XY_23,
        "/regions[]/areas[]" => XY_34,
        "/regions[]/areas[]/outlines[]" => XY_12,
        "/regions[]/settlement_and_slots" if with_faces => &[(0, Rule::Xy), (1, Rule::ArrXy)],
        "/regions[]/settlement_and_slots/slot_descriptions[]" if with_faces => &[
            (2, Rule::Xyz),
            (3, Rule::Xy),
            (7, Rule::ArrXy),
            (8, Rule::ArrXy),
            (9, Rule::ArrXy),
        ],
        _ => &[],
    }
}

fn query_rules(rest: &str) -> RuleSet {
    match rest {
        "/node" | "/node/Q" => XY_01,
        "/node/Q/cell" => &[(0, Rule::Xy)],
        _ => &[],
    }
}

/// The POC `RULES` table: scaled leaf indices of the container at the (normalised) `path`.
pub(super) fn rules(path: &str) -> RuleSet {
    match path {
        "/root/theatres_and_region_keys[]/theatre" => XY_12,
        "/root/theatres_and_region_keys[]/theatre/region_keys[]" => &[(1, Rule::Xy)],
        "/root/theatres[]/theatre" => XY_12,
        "/root/theatres[]/theatre/transition_areas[]" => XY_12,
        "/root/mountain_data/MOUNTAIN_OUTLINES/OUTLINE_BLOCK[]" => &[(0, Rule::ArrXy)],
        "/root/mountain_data/MOUNTAIN_OUTLINES/BOUNDS_BLOCK[]" => XY_01,
        "/root/trade_nodes/trade_nodes[]" => &[(1, Rule::Xy)],
        "/root/bridges" => &[(0, Rule::ArrXya)],
        _ => {
            if let Some(rest) = path.strip_prefix("/root/region_data") {
                region_data_rules(rest, true)
            } else if let Some(rest) = path.strip_prefix("/root/groundtypes/region_data") {
                region_data_rules(rest, false)
            } else if let Some(rest) = path.strip_prefix("/root/query_info") {
                query_rules(rest)
            } else if let Some(rest) = path.strip_prefix("/root/groundtypes/query_info") {
                query_rules(rest)
            } else {
                &[]
            }
        }
    }
}

fn rule_for(path: &str, index: usize) -> Option<Rule> {
    rules(path).iter().find(|(i, _)| *i == index).map(|(_, r)| *r)
}

/// The POC `PRESERVED_FLOATS` table: float leaves deliberately not scaled, with the reason.
fn preserved_reason(path: &str, index: usize) -> Option<&'static str> {
    match (path, index) {
        ("/root/theatres[]/theatre/wind_map", 2) => Some(
            "wind_map f32: per-theatre constant (0.0056-0.0073, uncorrelated with theatre size; grid \
             density is 0.4 cells/unit for every theatre) belonging to the normalized direction table; \
             the grid dims and arr2 bytes are implicit-grid data over the (now larger) theatre rect",
        ),
        ("/root/theatres[]/theatre/transition_areas[]", 3) => Some(
            "transition-area trailing f32: 0.0 in all 23 source records, so x*3 is the identity on the \
             stored value; semantics unknown, not an XY coordinate",
        ),
        ("/root/theatres[]/theatre/transition_areas[]/transition_links[]", 0) => Some(
            "transition-link f32: 0.0 in all 258 source records (identity under scaling); the three \
             following u32 are link/type codes",
        ),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Paths and leaf helpers
// ---------------------------------------------------------------------------------------------

static QUAD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(/(bottom_left|top_left|bottom_right|top_right|Q))+").expect("quadtree path regex")
});

/// Collapse quadtree node names (`bottom_left`, `top_left`, `bottom_right`, `top_right`, `Q`, any depth)
/// to a single `/Q` exactly like the POC.
pub(super) fn norm(path: &str) -> String {
    // Every alternative contains one of these fragments; skip the regex for the common case.
    if !path.contains('Q') && !path.contains("_left") && !path.contains("_right") {
        return path.to_string();
    }
    QUAD.replace_all(path, "/Q").into_owned()
}

/// POC-style short description of a leaf (`a[:2]` in the Python messages).
fn leaf_desc(v: &Value) -> String {
    match v {
        Value::String16(s) | Value::String8(s) => format!("('s', {s:?})"),
        Value::Array { code, .. } => format!("('arr', {})", code & 0x3F),
        other => format!("('v', {})", other.code()),
    }
}

fn node_ident(n: &Node) -> String {
    match n {
        Node::Record { name, version, .. } => format!("('{name}', {version})"),
        Node::Array { name, version, .. } => format!("('{name}[]', {version})"),
        Node::Value(v) => leaf_desc(v),
    }
}

/// Bit-exact leaf equality (the POC compares `leaf_bytes`): floats by bits, strings by kind and text.
fn value_bits_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::F32(x), Value::F32(y)) => x.to_bits() == y.to_bits(),
        (Value::F64(x), Value::F64(y)) => x.to_bits() == y.to_bits(),
        (Value::XY(x), Value::XY(y)) => x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
        (Value::XYZ(x), Value::XYZ(y)) => x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
        _ => a == b,
    }
}

fn bits2(a: [f32; 2], b: [f32; 2]) -> bool {
    a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()
}

/// f64 -> f32 rounding; errors where Python's `struct.pack('<f')` raises OverflowError.
fn f32_of(v: f64) -> Result<f32> {
    let r = v as f32;
    ensure!(!(v.is_finite() && !r.is_finite()), "coordinate {v} overflows f32");
    Ok(r)
}

/// `f32(plan.xy(p as f64, chart))` per axis (overflow checked, NaN/inf not rejected).
fn map_xy(plan: &Plan, chart: &Chart, p: [f32; 2]) -> Result<[f32; 2]> {
    let m = plan.xy([p[0] as f64, p[1] as f64], chart);
    Ok([f32_of(m[0])?, f32_of(m[1])?])
}

/// `map_xy` rejecting non-finite sources and results.
fn scale_xy(plan: &Plan, chart: &Chart, p: [f32; 2]) -> Result<[f32; 2]> {
    ensure!(p[0].is_finite() && p[1].is_finite(), "non-finite coordinate {p:?}");
    let out = map_xy(plan, chart, p)?;
    ensure!(
        out[0].is_finite() && out[1].is_finite(),
        "non-finite coordinate {p:?} -> {out:?}"
    );
    Ok(out)
}

fn read_f32(b: &[u8], k: usize) -> f32 {
    f32::from_le_bytes([b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3]])
}

// ---------------------------------------------------------------------------------------------
// Scaler
// ---------------------------------------------------------------------------------------------

/// Per (container path, child index) statistics of scaled values (POC `Scaler.stats` entry).  The atlas
/// report uses `rule`/`points`; the value ranges are the rest of the POC entry (asserted by the tests).
#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct FieldStat {
    pub rule: Rule,
    pub values: u64,
    pub points: u64,
    pub src_min: [f64; 2],
    pub src_max: [f64; 2],
    pub dst_min: [f64; 2],
    pub dst_max: [f64; 2],
}

impl FieldStat {
    fn new(rule: Rule) -> Self {
        Self {
            rule,
            values: 0,
            points: 0,
            src_min: [f64::INFINITY; 2],
            src_max: [f64::NEG_INFINITY; 2],
            dst_min: [f64::INFINITY; 2],
            dst_max: [f64::NEG_INFINITY; 2],
        }
    }

    /// Both coordinates are finite (checked by `scale_xy`).
    fn note(&mut self, src: [f32; 2], dst: [f32; 2]) {
        self.values += 2;
        self.points += 1;
        for i in 0..2 {
            let (s, d) = (src[i] as f64, dst[i] as f64);
            self.src_min[i] = self.src_min[i].min(s);
            self.src_max[i] = self.src_max[i].max(s);
            self.dst_min[i] = self.dst_min[i].min(d);
            self.dst_max[i] = self.dst_max[i].max(d);
        }
    }
}

/// Rebuilds ESF subtrees applying [`rules`] with one chart's affine map.
pub(super) struct Scaler<'a> {
    plan: &'a Plan,
    chart: &'a Chart,
    stats: BTreeMap<String, FieldStat>,
}

impl<'a> Scaler<'a> {
    pub fn new(plan: &'a Plan, chart: &'a Chart) -> Self {
        Self {
            plan,
            chart,
            stats: BTreeMap::new(),
        }
    }

    /// `f32(plan.xy(p as f64, chart))`; errors on non-finite input/output or f32 overflow.
    #[cfg(test)]
    pub fn xy(&self, p: [f32; 2]) -> Result<[f32; 2]> {
        scale_xy(self.plan, self.chart, p)
    }

    pub fn stats(&self) -> &BTreeMap<String, FieldStat> {
        &self.stats
    }

    fn leaf(&mut self, rule: Rule, value: &Value, path: &str, index: usize) -> Result<Value> {
        let key = format!("{path}[{index}]");
        let (plan, chart) = (self.plan, self.chart);
        let st = self
            .stats
            .entry(key.clone())
            .or_insert_with(|| FieldStat::new(rule));
        match rule {
            Rule::Xy => {
                let Value::XY(p) = value else {
                    bail!("{key}: expected v12, found {}", leaf_desc(value));
                };
                let new = scale_xy(plan, chart, *p)?;
                st.note(*p, new);
                Ok(Value::XY(new))
            }
            Rule::Xyz => {
                let Value::XYZ([x, h, y]) = value else {
                    bail!("{key}: expected v13, found {}", leaf_desc(value));
                };
                let new = scale_xy(plan, chart, [*x, *y])?;
                st.note([*x, *y], new);
                Ok(Value::XYZ([new[0], *h, new[1]]))
            }
            Rule::ArrXy => {
                let Value::Array { code: 0x4C, data } = value else {
                    bail!("{key}: bad arr12 ({})", leaf_desc(value));
                };
                ensure!(
                    data.len() % 8 == 0,
                    "{key}: bad arr12 (('arr', 12), {} bytes)",
                    data.len()
                );
                let mut out = Vec::with_capacity(data.len());
                for k in 0..data.len() / 8 {
                    let p = [read_f32(data, 2 * k), read_f32(data, 2 * k + 1)];
                    let new = scale_xy(plan, chart, p)?;
                    st.note(p, new);
                    out.extend_from_slice(&new[0].to_le_bytes());
                    out.extend_from_slice(&new[1].to_le_bytes());
                }
                Ok(Value::Array {
                    code: 0x4C,
                    data: out,
                })
            }
            Rule::ArrXya => {
                let Value::Array { code: 0x4D, data } = value else {
                    bail!("{key}: bad arr13 ({})", leaf_desc(value));
                };
                ensure!(
                    data.len() % 12 == 0,
                    "{key}: bad arr13 (('arr', 13), {} bytes)",
                    data.len()
                );
                let mut out = Vec::with_capacity(data.len());
                for k in 0..data.len() / 12 {
                    let p = [read_f32(data, 3 * k), read_f32(data, 3 * k + 1)];
                    let new = scale_xy(plan, chart, p)?;
                    st.note(p, new);
                    out.extend_from_slice(&new[0].to_le_bytes());
                    out.extend_from_slice(&new[1].to_le_bytes());
                    out.extend_from_slice(&data[12 * k + 8..12 * k + 12]);
                }
                Ok(Value::Array {
                    code: 0x4D,
                    data: out,
                })
            }
        }
    }

    /// Rebuild the children of the container at `path` (exact, already normalised path): ruled leaf
    /// indices are scaled, nested records/arrays recurse, everything else is cloned.
    pub fn children(&mut self, path: &str, kids: &[Node]) -> Result<Vec<Node>> {
        let rules = rules(path);
        let mut out = Vec::with_capacity(kids.len());
        for (i, k) in kids.iter().enumerate() {
            match k {
                Node::Value(v) => match rules.iter().find(|(r, _)| *r == i) {
                    Some((_, rule)) => out.push(Node::Value(self.leaf(*rule, v, path, i)?)),
                    None => out.push(k.clone()),
                },
                _ => out.push(self.node(k, path)?),
            }
        }
        let stale: Vec<usize> = rules
            .iter()
            .map(|(i, _)| *i)
            .filter(|i| !matches!(kids.get(*i), Some(Node::Value(_))))
            .collect();
        ensure!(stale.is_empty(), "{path}: rule indices {stale:?} do not hit leaves");
        Ok(out)
    }

    /// Rebuild one record or record array found under `parent` (a normalised path).
    pub fn node(&mut self, node: &Node, parent: &str) -> Result<Node> {
        match node {
            Node::Record {
                name,
                version,
                children,
            } => {
                let path = norm(&format!("{parent}/{name}"));
                Ok(Node::Record {
                    name: name.clone(),
                    version: *version,
                    children: self.children(&path, children)?,
                })
            }
            Node::Array {
                name,
                version,
                items,
            } => {
                let path = norm(&format!("{parent}/{name}[]"));
                let mut new = Vec::with_capacity(items.len());
                for item in items {
                    new.push(self.children(&path, item)?);
                }
                Ok(Node::Array {
                    name: name.clone(),
                    version: *version,
                    items: new,
                })
            }
            Node::Value(v) => bail!("cannot scale bare leaf {} under {parent}", leaf_desc(v)),
        }
    }
}

/// The scaler of `chart`, created on first use (POC `sc(chart)`).
pub(super) fn scaler_for<'a, 'b>(
    plan: &'a Plan,
    scalers: &'b mut BTreeMap<String, Scaler<'a>>,
    chart: &'a Chart,
) -> &'b mut Scaler<'a> {
    scalers
        .entry(chart.key.clone())
        .or_insert_with(|| Scaler::new(plan, chart))
}

// ---------------------------------------------------------------------------------------------
// Independent float audit
// ---------------------------------------------------------------------------------------------

/// One float-typed leaf position of the tree (aggregated over every occurrence).
#[derive(Clone, Debug)]
pub(super) struct FloatField {
    pub path: String,
    pub index: usize,
    /// `v10`..`v13` or `arr10`..`arr13`
    pub kind: String,
    pub count: u64,
    pub min: f64,
    pub max: f64,
}

fn float_kind(v: &Value) -> Option<&'static str> {
    match v {
        Value::F32(_) => Some("v10"),
        Value::F64(_) => Some("v11"),
        Value::XY(_) => Some("v12"),
        Value::XYZ(_) => Some("v13"),
        Value::Array { code: 0x4A, .. } => Some("arr10"),
        Value::Array { code: 0x4B, .. } => Some("arr11"),
        Value::Array { code: 0x4C, .. } => Some("arr12"),
        Value::Array { code: 0x4D, .. } => Some("arr13"),
        _ => None,
    }
}

/// Smallest/largest float component of a float leaf (packed arrays are read as f32).
fn float_range(v: &Value) -> Option<(f64, f64)> {
    fn fold(it: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
        let mut r: Option<(f64, f64)> = None;
        for x in it {
            r = Some(match r {
                None => (x, x),
                Some((lo, hi)) => (if x < lo { x } else { lo }, if x > hi { x } else { hi }),
            });
        }
        r
    }
    match v {
        Value::F32(x) => fold(std::iter::once(*x as f64)),
        Value::F64(x) => fold(std::iter::once(*x)),
        Value::XY(p) => fold(p.iter().map(|x| *x as f64)),
        Value::XYZ(p) => fold(p.iter().map(|x| *x as f64)),
        Value::Array { data, .. } => fold(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        ),
        _ => None,
    }
}

#[derive(Default)]
struct Audit {
    paths: HashMap<String, usize>,
    index: HashMap<(usize, usize, &'static str), usize>,
    out: Vec<FloatField>,
}

impl Audit {
    fn path_id(&mut self, path: &str) -> usize {
        if let Some(id) = self.paths.get(path) {
            return *id;
        }
        let id = self.paths.len();
        self.paths.insert(path.to_string(), id);
        id
    }

    /// One container slot (record children or one record-array item).
    fn slot(&mut self, path: &str, kids: &[Node]) {
        let mut pid: Option<usize> = None;
        for (i, k) in kids.iter().enumerate() {
            let Node::Value(v) = k else { continue };
            let Some(kind) = float_kind(v) else { continue };
            let id = match pid {
                Some(id) => id,
                None => {
                    let id = self.path_id(path);
                    pid = Some(id);
                    id
                }
            };
            let at = match self.index.get(&(id, i, kind)) {
                Some(at) => *at,
                None => {
                    self.out.push(FloatField {
                        path: path.to_string(),
                        index: i,
                        kind: kind.to_string(),
                        count: 0,
                        min: f64::INFINITY,
                        max: f64::NEG_INFINITY,
                    });
                    let at = self.out.len() - 1;
                    self.index.insert((id, i, kind), at);
                    at
                }
            };
            let e = &mut self.out[at];
            e.count += 1;
            if let Some((lo, hi)) = float_range(v) {
                if lo < e.min {
                    e.min = lo;
                }
                if hi > e.max {
                    e.max = hi;
                }
            }
        }
    }

    fn rec(&mut self, node: &Node, parent: &str) {
        match node {
            Node::Record { name, children, .. } => {
                let path = norm(&format!("{parent}/{name}"));
                self.slot(&path, children);
                for c in children {
                    if !matches!(c, Node::Value(_)) {
                        self.rec(c, &path);
                    }
                }
            }
            Node::Array { name, items, .. } => {
                let path = norm(&format!("{parent}/{name}[]"));
                for item in items {
                    self.slot(&path, item);
                    for c in item {
                        if !matches!(c, Node::Value(_)) {
                            self.rec(c, &path);
                        }
                    }
                }
            }
            Node::Value(_) => {}
        }
    }
}

/// Every float-typed leaf (f32, f64, xy, xyz, packed 0x4A..0x4D) of the whole tree, aggregated by
/// (normalised container path, child index, kind) in first-occurrence order.  Packed arrays whose byte
/// length is not a multiple of four ignore the trailing bytes.
pub(super) fn enumerate_float_fields(root: &Node) -> Vec<FloatField> {
    let mut audit = Audit::default();
    audit.rec(root, "");
    audit.out
}

/// Split the audit into (covered by [`rules`], deliberately preserved, unknown).  Entries are JSON
/// objects `{path, index, type, count, min, max, action|reason}`; non-finite min/max become null.
pub(super) fn classify_float_fields(fields: &[FloatField]) -> (Vec<Json>, Vec<Json>, Vec<Json>) {
    fn num(v: f64) -> Json {
        if v.is_finite() {
            Json::from(v)
        } else {
            Json::Null
        }
    }
    let (mut covered, mut preserved, mut unknown) = (Vec::new(), Vec::new(), Vec::new());
    for f in fields {
        let mut e = Map::new();
        e.insert("path".into(), Json::from(f.path.clone()));
        e.insert("index".into(), Json::from(f.index as u64));
        e.insert("type".into(), Json::from(f.kind.clone()));
        e.insert("count".into(), Json::from(f.count));
        e.insert("min".into(), num(f.min));
        e.insert("max".into(), num(f.max));
        if let Some(rule) = rule_for(&f.path, f.index) {
            e.insert("action".into(), Json::from(rule.as_str()));
            covered.push(Json::Object(e));
        } else if let Some(reason) = preserved_reason(&f.path, f.index) {
            e.insert("reason".into(), Json::from(reason));
            preserved.push(Json::Object(e));
        } else {
            unknown.push(Json::Object(e));
        }
    }
    (covered, preserved, unknown)
}

// ---------------------------------------------------------------------------------------------
// check_scaled
// ---------------------------------------------------------------------------------------------

struct Checker<'p> {
    plan: &'p Plan,
    chart: &'p Chart,
    counts: BTreeMap<String, u64>,
}

impl Checker<'_> {
    fn bump(&mut self, key: &str, n: u64) {
        if n != 0 {
            *self.counts.entry(key.to_string()).or_insert(0) += n;
        }
    }

    fn cmp_leaf(&mut self, path: &str, i: usize, a: &Value, b: &Value) -> Result<()> {
        let Some(rule) = rule_for(path, i) else {
            ensure!(
                value_bits_eq(a, b),
                "unscaled leaf changed at {path}[{i}]: {} {}",
                leaf_desc(a),
                leaf_desc(b)
            );
            self.bump("identical_leaves", 1);
            return Ok(());
        };
        ensure!(a.code() == b.code(), "type changed at {path}[{i}]");
        let (plan, chart) = (self.plan, self.chart);
        match rule {
            Rule::Xy => {
                let (Value::XY(pa), Value::XY(pb)) = (a, b) else {
                    bail!("bad XY at {path}[{i}]");
                };
                ensure!(bits2(map_xy(plan, chart, *pa)?, *pb), "bad XY at {path}[{i}]");
                self.bump("scaled_xy_points", 1);
            }
            Rule::Xyz => {
                let (Value::XYZ(pa), Value::XYZ(pb)) = (a, b) else {
                    bail!("bad XYZ at {path}[{i}]");
                };
                let e = map_xy(plan, chart, [pa[0], pa[2]])?;
                ensure!(
                    e[0].to_bits() == pb[0].to_bits()
                        && pa[1].to_bits() == pb[1].to_bits()
                        && e[1].to_bits() == pb[2].to_bits(),
                    "bad XYZ at {path}[{i}]"
                );
                self.bump("scaled_xyz_points", 1);
                self.bump("preserved_heights", 1);
            }
            Rule::ArrXy | Rule::ArrXya => {
                let (want, stride, w) = if rule == Rule::ArrXy {
                    (0x4C, 8usize, 2usize)
                } else {
                    (0x4D, 12usize, 3usize)
                };
                let (Value::Array { code: ca, data: da }, Value::Array { data: db, .. }) = (a, b)
                else {
                    bail!("bad array at {path}[{i}]");
                };
                ensure!(
                    *ca == want && da.len() % stride == 0,
                    "bad array at {path}[{i}] ({} bytes)",
                    da.len()
                );
                ensure!(da.len() == db.len(), "array length changed at {path}[{i}]");
                let mut points = 0u64;
                for (j, (pa, pb)) in da
                    .chunks_exact(stride)
                    .zip(db.chunks_exact(stride))
                    .enumerate()
                {
                    let src = [read_f32(pa, 0), read_f32(pa, 1)];
                    let got = [read_f32(pb, 0), read_f32(pb, 1)];
                    ensure!(
                        bits2(map_xy(plan, chart, src)?, got),
                        "bad array point {path}[{i}]#{}",
                        j * w
                    );
                    if w == 3 {
                        ensure!(
                            pa[8..12] == pb[8..12],
                            "angle changed {path}[{i}]#{}",
                            j * w
                        );
                    }
                    points += 1;
                }
                if w == 3 {
                    self.bump("preserved_bridge_angles", points);
                }
                self.bump("scaled_array_points", points);
            }
        }
        Ok(())
    }

    fn cmp_kids(&mut self, path: &str, ka: &[Node], kb: &[Node]) -> Result<()> {
        ensure!(ka.len() == kb.len(), "child count changed at {path}");
        for (i, (a, b)) in ka.iter().zip(kb).enumerate() {
            match (a, b) {
                (Node::Value(x), Node::Value(y)) => self.cmp_leaf(path, i, x, y)?,
                (Node::Value(_), _) | (_, Node::Value(_)) => bail!("shape changed at {path}[{i}]"),
                _ => self.cmp_node(a, b, path)?,
            }
        }
        Ok(())
    }

    fn cmp_node(&mut self, a: &Node, b: &Node, parent: &str) -> Result<()> {
        match (a, b) {
            (
                Node::Record {
                    name: na,
                    version: va,
                    children: ca,
                },
                Node::Record {
                    name: nb,
                    version: vb,
                    children: cb,
                },
            ) if na == nb && va == vb => {
                let path = norm(&format!("{parent}/{na}"));
                self.bump("nodes", 1);
                self.cmp_kids(&path, ca, cb)
            }
            (
                Node::Array {
                    name: na,
                    version: va,
                    items: ia,
                },
                Node::Array {
                    name: nb,
                    version: vb,
                    items: ib,
                },
            ) if na == nb && va == vb => {
                let path = norm(&format!("{parent}/{na}[]"));
                self.bump("nodes", 1);
                ensure!(ia.len() == ib.len(), "item count changed at {path}");
                for (x, y) in ia.iter().zip(ib) {
                    self.cmp_kids(&path, x, y)?;
                }
                Ok(())
            }
            _ => bail!(
                "node name/version changed at {parent}: {} {}",
                node_ident(a),
                node_ident(b)
            ),
        }
    }
}

/// Parallel walk of `src` and `dst` (both found under `parent`): every ruled component equals
/// `f32(plan.xy(src, chart))` bit-exactly (heights and bridge angles bit-identical), everything else is
/// bit-identical.  Returns the POC counters; errors on the first mismatch.
pub(super) fn check_scaled(
    src: &Node,
    dst: &Node,
    plan: &Plan,
    chart: &Chart,
    parent: &str,
) -> Result<BTreeMap<String, u64>> {
    let mut c = Checker {
        plan,
        chart,
        counts: BTreeMap::new(),
    };
    c.cmp_node(src, dst, parent)?;
    Ok(c.counts)
}

/// `check_scaled` for one record-array item pair as stored in the section pairs: the item children are
/// compared as the single item of record array `NAME[]` (array name without `[]`) located under `parent`.
pub(super) fn check_scaled_items(
    array_name: &str,
    src: &[Node],
    dst: &[Node],
    plan: &Plan,
    chart: &Chart,
    parent: &str,
) -> Result<BTreeMap<String, u64>> {
    let path = norm(&format!("{parent}/{array_name}[]"));
    let mut c = Checker {
        plan,
        chart,
        counts: BTreeMap::new(),
    };
    c.bump("nodes", 1);
    c.cmp_kids(&path, src, dst)?;
    Ok(c.counts)
}

// ---------------------------------------------------------------------------------------------
// Section helpers (regions_atlas_build.py transform_rest)
// ---------------------------------------------------------------------------------------------

const OUTLINE_PATH: &str = "/root/mountain_data/MOUNTAIN_OUTLINES/OUTLINE_BLOCK[]";
const BOUNDS_PATH: &str = "/root/mountain_data/MOUNTAIN_OUTLINES/BOUNDS_BLOCK[]";
const TRADE_PATH: &str = "/root/trade_nodes/trade_nodes[]";

/// One scaled theatre item (`theatres_and_region_keys[]` or `theatres[]` entry): its children before/after.
pub(super) struct TheatrePair {
    /// `theatres_and_region_keys` or `theatres`
    pub name: String,
    pub chart: String,
    pub src: Vec<Node>,
    pub dst: Vec<Node>,
}

/// One scaled mountain outline with its bounds item (children before/after).
pub(super) struct MountainPair {
    pub chart: String,
    pub outline_src: Vec<Node>,
    pub outline_dst: Vec<Node>,
    pub bounds_src: Vec<Node>,
    pub bounds_dst: Vec<Node>,
}

fn theatre_key(item: &[Node]) -> Result<String> {
    match item.first() {
        Some(Node::Record { name, children, .. }) if name == "theatre" => Ok(children
            .first()
            .context("theatre record is empty")?
            .as_str()
            .context("theatre key")?
            .to_string()),
        _ => bail!("theatre item does not start with a theatre record"),
    }
}

/// Scale a `theatres_and_region_keys` / `theatres` record array (`name` without `[]`): each item's chart
/// is the one owning its theatre key.
pub(super) fn scale_theatres<'a>(
    plan: &'a Plan,
    name: &str,
    node: &Node,
    scalers: &mut BTreeMap<String, Scaler<'a>>,
    pairs: &mut Vec<TheatrePair>,
) -> Result<Node> {
    let Node::Array {
        name: n,
        version,
        items,
    } = node
    else {
        bail!("{name} is not a record array");
    };
    ensure!(n == name, "expected record array {name}, found {n}");
    let path = format!("/root/{name}[]");
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let key = theatre_key(item)?;
        let chart = plan
            .charts
            .iter()
            .find(|c| c.theatres.iter().any(|t| *t == key))
            .with_context(|| format!("{name}: theatre {key} belongs to no atlas chart"))?;
        let new = scaler_for(plan, scalers, chart).children(&path, item)?;
        pairs.push(TheatrePair {
            name: name.to_string(),
            chart: chart.key.clone(),
            src: item.clone(),
            dst: new.clone(),
        });
        out.push(new);
    }
    Ok(Node::Array {
        name: n.clone(),
        version: *version,
        items: out,
    })
}

/// The single chart owning every point (crop areas allowed).
fn points_chart<'a>(plan: &'a Plan, pts: &[[f32; 2]], what: &str) -> Result<&'a Chart> {
    let mut found: BTreeMap<&'a str, &'a Chart> = BTreeMap::new();
    for p in pts {
        let c = plan.chart_at([p[0] as f64, p[1] as f64], true)?;
        found.entry(c.key.as_str()).or_insert(c);
    }
    if found.len() != 1 {
        bail!(
            "{what}: points fall in charts {:?}",
            found.keys().collect::<Vec<_>>()
        );
    }
    Ok(found.into_values().next().expect("one chart"))
}

fn with_items(template: &Node, items: Vec<Vec<Node>>) -> Node {
    match template {
        Node::Array { name, version, .. } => Node::Array {
            name: name.clone(),
            version: *version,
            items,
        },
        other => other.clone(),
    }
}

/// Scale `mountain_data`: every outline (arr12) and its bounds item use the one chart owning all outline points.
pub(super) fn scale_mountains<'a>(
    plan: &'a Plan,
    node: &Node,
    scalers: &mut BTreeMap<String, Scaler<'a>>,
    pairs: &mut Vec<MountainPair>,
) -> Result<Node> {
    let Node::Record {
        name,
        version,
        children,
    } = node
    else {
        bail!("mountain_data is not a record");
    };
    let mut out = Vec::with_capacity(children.len());
    for mo in children {
        let Node::Record {
            name: mn,
            version: mv,
            children: mk,
        } = mo
        else {
            out.push(mo.clone());
            continue;
        };
        if mn != "MOUNTAIN_OUTLINES" {
            out.push(mo.clone());
            continue;
        }
        let block = |wanted: &str| {
            mk.iter()
                .position(|x| matches!(x, Node::Array { name, .. } if name.as_str() == wanted))
                .with_context(|| format!("MOUNTAIN_OUTLINES has no {wanted}[]"))
        };
        let oi_idx = block("OUTLINE_BLOCK")?;
        let bi_idx = block("BOUNDS_BLOCK")?;
        let (oitems, bitems) = match (&mk[oi_idx], &mk[bi_idx]) {
            (Node::Array { items: o, .. }, Node::Array { items: b, .. }) => (o, b),
            _ => bail!("mountain outline/bounds blocks are not record arrays"),
        };
        ensure!(oitems.len() == bitems.len(), "mountain outline/bounds counts differ");
        let mut o_new = Vec::with_capacity(oitems.len());
        let mut b_new = Vec::with_capacity(bitems.len());
        for (oi, bi) in oitems.iter().zip(bitems) {
            let pts = oi
                .first()
                .context("mountain outline item is empty")?
                .xy_array()
                .context("mountain outline points")?;
            let chart = points_chart(plan, &pts, "mountain outline")?;
            let o = scaler_for(plan, scalers, chart).children(OUTLINE_PATH, oi)?;
            let b = scaler_for(plan, scalers, chart).children(BOUNDS_PATH, bi)?;
            pairs.push(MountainPair {
                chart: chart.key.clone(),
                outline_src: oi.clone(),
                outline_dst: o.clone(),
                bounds_src: bi.clone(),
                bounds_dst: b.clone(),
            });
            o_new.push(o);
            b_new.push(b);
        }
        let kids = mk
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if i == oi_idx {
                    with_items(x, std::mem::take(&mut o_new))
                } else if i == bi_idx {
                    with_items(x, std::mem::take(&mut b_new))
                } else {
                    x.clone()
                }
            })
            .collect();
        out.push(Node::Record {
            name: mn.clone(),
            version: *mv,
            children: kids,
        });
    }
    Ok(Node::Record {
        name: name.clone(),
        version: *version,
        children: out,
    })
}

/// Scale `trade_nodes`: each `trade_nodes[]` item uses the chart owning its xy (item child 1, crop allowed).
/// Pairs are `(chart key, children before, children after)`.
pub(super) fn scale_trade_nodes<'a>(
    plan: &'a Plan,
    node: &Node,
    scalers: &mut BTreeMap<String, Scaler<'a>>,
    pairs: &mut Vec<(String, Vec<Node>, Vec<Node>)>,
) -> Result<Node> {
    let Node::Record {
        name,
        version,
        children,
    } = node
    else {
        bail!("trade_nodes is not a record");
    };
    let mut out = Vec::with_capacity(children.len());
    for t in children {
        let Node::Array {
            name: tn,
            version: tv,
            items,
        } = t
        else {
            out.push(t.clone());
            continue;
        };
        if tn != "trade_nodes" {
            out.push(t.clone());
            continue;
        }
        let mut new_items = Vec::with_capacity(items.len());
        for it in items {
            let p = it
                .get(1)
                .context("trade node has no position child")?
                .as_xy()
                .context("trade node position")?;
            let chart = plan.chart_at([p[0] as f64, p[1] as f64], true)?;
            let new = scaler_for(plan, scalers, chart).children(TRADE_PATH, it)?;
            pairs.push((chart.key.clone(), it.clone(), new.clone()));
            new_items.push(new);
        }
        out.push(Node::Array {
            name: tn.clone(),
            version: *tv,
            items: new_items,
        });
    }
    Ok(Node::Record {
        name: name.clone(),
        version: *version,
        children: out,
    })
}

/// Scale `bridges` (arr13 of (x, y, angle)): x/y through the chart owning each bridge (crop allowed), the
/// angle untouched.  `counts` accumulates bridges per chart key.
pub(super) fn scale_bridges(
    plan: &Plan,
    node: &Node,
    counts: &mut BTreeMap<String, u64>,
) -> Result<Node> {
    let Node::Record {
        name,
        version,
        children,
    } = node
    else {
        bail!("bridges is not a record");
    };
    let pts = children
        .first()
        .context("bridges has no children")?
        .xyz_array()
        .context("bridges point array")?;
    let mut data = Vec::with_capacity(pts.len() * 12);
    for [x, y, a] in pts {
        let chart = plan.chart_at([x as f64, y as f64], true)?;
        *counts.entry(chart.key.clone()).or_insert(0) += 1;
        let new = scale_xy(plan, chart, [x, y])?;
        data.extend_from_slice(&new[0].to_le_bytes());
        data.extend_from_slice(&new[1].to_le_bytes());
        data.extend_from_slice(&a.to_le_bytes());
    }
    let mut kids = Vec::with_capacity(children.len());
    kids.push(Node::Value(Value::Array { code: 0x4D, data }));
    kids.extend(children[1..].iter().cloned());
    Ok(Node::Record {
        name: name.clone(),
        version: *version,
        children: kids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf::{arr_f32, arr_xy, arr_xyz, f32v, s16, u32v, xyv, xyzv};

    fn expect(plan: &Plan, chart: &Chart, p: [f32; 2]) -> [f32; 2] {
        let m = plan.xy([p[0] as f64, p[1] as f64], chart);
        [m[0] as f32, m[1] as f32]
    }

    fn msg<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => format!("{e:#}"),
        }
    }

    fn settlement() -> Node {
        Node::record(
            "settlement_and_slots",
            0,
            vec![
                xyv(10.0, 20.0),
                arr_xy(&[[1.0, 2.0], [3.0, 4.0]]),
                Node::array(
                    "slot_descriptions",
                    0,
                    vec![vec![
                        u32v(1),
                        u32v(2),
                        xyzv(5.0, 77.0, 6.0),
                        xyv(7.0, 8.0),
                        u32v(0),
                        u32v(0),
                        u32v(0),
                        arr_xy(&[[9.0, 10.0]]),
                        arr_xy(&[]),
                        arr_xy(&[[11.0, 12.0], [13.0, 14.0]]),
                    ]],
                ),
            ],
        )
    }

    #[test]
    fn norm_collapses_quadtree_names_only() {
        assert_eq!(
            norm("/root/query_info/node/bottom_left/top_right/Q/Q"),
            "/root/query_info/node/Q"
        );
        assert_eq!(
            norm("/root/groundtypes/query_info/node/top_left/cell"),
            "/root/groundtypes/query_info/node/Q/cell"
        );
        assert_eq!(norm("/root/region_data/regions[]"), "/root/region_data/regions[]");
    }

    #[test]
    fn rule_index_that_misses_a_leaf_is_an_error() {
        let plan = Plan::proven();
        let chart = plan.chart("eur_ind").unwrap();
        let mut sc = Scaler::new(&plan, chart);
        // trade_nodes[] rule index 1 hits a record instead of an xy leaf
        let kids = vec![s16("k"), Node::record("x", 0, vec![])];
        let err = msg(sc.children(TRADE_PATH, &kids));
        assert!(err.contains("do not hit leaves"), "{err}");
        // and a too-short item
        let err = msg(sc.children(TRADE_PATH, &[s16("k")]));
        assert!(err.contains("rule indices [1] do not hit leaves"), "{err}");
    }

    #[test]
    fn xyz_height_preserved_and_stats_recorded() {
        let plan = Plan::proven();
        let chart = plan.chart("eur_ind").unwrap();
        let mut sc = Scaler::new(&plan, chart);
        let src = settlement();
        let dst = sc.node(&src, "/root/region_data/regions[]").unwrap();
        let slots = &dst.get("slot_descriptions").unwrap().items().unwrap()[0];
        let [x, h, y] = slots[2].as_xyz().unwrap();
        let e = expect(&plan, chart, [5.0, 6.0]);
        assert_eq!(x.to_bits(), e[0].to_bits());
        assert_eq!(y.to_bits(), e[1].to_bits());
        assert_eq!(h.to_bits(), 77.0f32.to_bits());
        assert_eq!(slots[0], u32v(1));
        assert_eq!(slots[8].raw_array().unwrap().len(), 0);
        let label = dst.children().unwrap()[0].as_xy().unwrap();
        let e = expect(&plan, chart, [10.0, 20.0]);
        assert_eq!(label[0].to_bits(), e[0].to_bits());
        let st = &sc.stats()["/root/region_data/regions[]/settlement_and_slots[0]"];
        assert_eq!((st.rule, st.points, st.values), (Rule::Xy, 1, 2));
        assert_eq!(st.src_min, [10.0, 20.0]);
        assert_eq!(st.src_max, [10.0, 20.0]);
        assert_eq!(st.dst_min[0], e[0] as f64);
        let st = &sc.stats()["/root/region_data/regions[]/settlement_and_slots[1]"];
        assert_eq!((st.rule, st.points, st.src_max), (Rule::ArrXy, 2, [3.0, 4.0]));
        let c = check_scaled(&src, &dst, &plan, chart, "/root/region_data/regions[]").unwrap();
        assert_eq!(c["scaled_xyz_points"], 1);
        assert_eq!(c["preserved_heights"], 1);
        assert_eq!(c["scaled_array_points"], 2 + 1 + 2);
    }

    #[test]
    fn check_scaled_rejects_tampered_leaves() {
        let plan = Plan::proven();
        let chart = plan.chart("eur_ind").unwrap();
        let mut sc = Scaler::new(&plan, chart);
        let src = settlement();
        let good = sc.node(&src, "/root/region_data/regions[]").unwrap();
        let parent = "/root/region_data/regions[]";
        assert!(check_scaled(&src, &good, &plan, chart, parent).is_ok());

        // unscaled u32 changed
        let mut bad = good.clone();
        if let Node::Array { items, .. } = &mut bad.children_mut().unwrap()[2] {
            items[0][0] = u32v(2);
        }
        let err = msg(check_scaled(&src, &bad, &plan, chart, parent));
        assert!(err.contains("unscaled leaf changed at"), "{err}");

        // scaled xy off by one ulp
        let mut bad = good.clone();
        let label = bad.children().unwrap()[0].as_xy().unwrap();
        bad.children_mut().unwrap()[0] = xyv(f32::from_bits(label[0].to_bits() + 1), label[1]);
        let err = msg(check_scaled(&src, &bad, &plan, chart, parent));
        assert!(err.contains("bad XY"), "{err}");

        // height changed
        let mut bad = good.clone();
        if let Node::Array { items, .. } = &mut bad.children_mut().unwrap()[2] {
            let [x, _, y] = items[0][2].as_xyz().unwrap();
            items[0][2] = xyzv(x, 78.0, y);
        }
        let err = msg(check_scaled(&src, &bad, &plan, chart, parent));
        assert!(err.contains("bad XYZ"), "{err}");

        // unscaled copy of a ruled array
        let mut bad = good.clone();
        bad.children_mut().unwrap()[1] = arr_xy(&[[1.0, 2.0], [3.0, 4.0]]);
        let err = msg(check_scaled(&src, &bad, &plan, chart, parent));
        assert!(err.contains("bad array point"), "{err}");

        // structure changed
        let mut bad = good;
        bad.children_mut().unwrap().pop();
        let err = msg(check_scaled(&src, &bad, &plan, chart, parent));
        assert!(err.contains("child count changed"), "{err}");
    }

    #[test]
    fn xy_rejects_non_finite_and_overflow() {
        let plan = Plan::proven();
        let sc = Scaler::new(&plan, plan.chart("eur_ind").unwrap());
        assert!(sc.xy([f32::NAN, 0.0]).is_err());
        assert!(sc.xy([f32::INFINITY, 0.0]).is_err());
        assert!(sc.xy([f32::MAX, 0.0]).is_err());
        assert!(sc.xy([0.0, 200.0]).is_ok());
    }

    #[test]
    fn float_audit_sorts_covered_preserved_and_unknown() {
        let root = Node::record(
            "root",
            0,
            vec![
                Node::record("mystery", 0, vec![f32v(1.5), xyv(-2.0, 3.0), arr_f32(&[4.0, -5.0])]),
                Node::record(
                    "trade_nodes",
                    0,
                    vec![Node::array(
                        "trade_nodes",
                        0,
                        vec![vec![s16("a"), xyv(1.0, 2.0)], vec![s16("b"), xyv(-3.0, 8.0)]],
                    )],
                ),
                Node::array(
                    "theatres",
                    0,
                    vec![vec![Node::record(
                        "theatre",
                        0,
                        vec![Node::record("wind_map", 0, vec![u32v(1), u32v(2), f32v(0.006)])],
                    )]],
                ),
            ],
        );
        let fields = enumerate_float_fields(&root);
        let (covered, preserved, unknown) = classify_float_fields(&fields);
        assert_eq!(covered.len(), 1);
        assert_eq!(covered[0]["path"], "/root/trade_nodes/trade_nodes[]");
        assert_eq!(covered[0]["index"], 1);
        assert_eq!(covered[0]["type"], "v12");
        assert_eq!(covered[0]["count"], 2);
        assert_eq!(covered[0]["min"], -3.0);
        assert_eq!(covered[0]["max"], 8.0);
        assert_eq!(covered[0]["action"], "xy");
        assert_eq!(preserved.len(), 1);
        assert_eq!(preserved[0]["path"], "/root/theatres[]/theatre/wind_map");
        assert!(preserved[0]["reason"].as_str().unwrap().contains("wind_map"));
        let unknown: Vec<_> = unknown
            .iter()
            .map(|u| (u["path"].as_str().unwrap(), u["index"].as_u64().unwrap(), u["type"].as_str().unwrap()))
            .collect();
        assert_eq!(
            unknown,
            vec![("/root/mystery", 0, "v10"), ("/root/mystery", 1, "v12"), ("/root/mystery", 2, "arr10")]
        );
    }

    #[test]
    fn bridges_scale_position_only_and_count_charts() {
        let plan = Plan::proven();
        let node = Node::record(
            "bridges",
            0,
            vec![arr_xyz(&[[0.0, 200.0, 1.25], [850.0, -40.0, -3.5]])],
        );
        let mut counts = BTreeMap::new();
        let out = scale_bridges(&plan, &node, &mut counts).unwrap();
        assert_eq!(counts["eur_ind"], 1);
        assert_eq!(counts["east_indies"], 1);
        let v = out.children().unwrap()[0].xyz_array().unwrap();
        assert_eq!(v[0][2], 1.25);
        assert_eq!(v[1][2], -3.5);
        let e = expect(&plan, plan.chart("eur_ind").unwrap(), [0.0, 200.0]);
        assert_eq!([v[0][0].to_bits(), v[0][1].to_bits()], [e[0].to_bits(), e[1].to_bits()]);
        // a point outside every chart/crop fails the whole section
        let lost = Node::record("bridges", 0, vec![arr_xyz(&[[-1000.0, 600.0, 0.0]])]);
        assert!(scale_bridges(&plan, &lost, &mut BTreeMap::new()).is_err());
    }

    fn mountains(points: &[[f32; 2]]) -> Node {
        Node::record(
            "mountain_data",
            0,
            vec![Node::record(
                "MOUNTAIN_OUTLINES",
                0,
                vec![
                    Node::array("OUTLINE_BLOCK", 0, vec![vec![arr_xy(points), u32v(7)]]),
                    Node::array("BOUNDS_BLOCK", 0, vec![vec![xyv(0.0, 150.0), xyv(100.0, 160.0)]]),
                ],
            )],
        )
    }

    #[test]
    fn mountain_outline_must_belong_to_one_chart() {
        let plan = Plan::proven();
        let mut scalers = BTreeMap::new();
        let mut pairs = Vec::new();
        let err = msg(scale_mountains(
            &plan,
            &mountains(&[[0.0, 200.0], [-500.0, 200.0]]),
            &mut scalers,
            &mut pairs,
        ));
        assert!(err.contains("mountain outline: points fall in charts"), "{err}");
        assert!(err.contains("america") && err.contains("eur_ind"), "{err}");
        assert!(pairs.is_empty());

        let src = mountains(&[[0.0, 200.0], [10.0, 210.0]]);
        let out = scale_mountains(&plan, &src, &mut scalers, &mut pairs).unwrap();
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.chart, "eur_ind");
        let chart = plan.chart("eur_ind").unwrap();
        check_scaled_items(
            "OUTLINE_BLOCK",
            &p.outline_src,
            &p.outline_dst,
            &plan,
            chart,
            "/root/mountain_data/MOUNTAIN_OUTLINES",
        )
        .unwrap();
        let c = check_scaled_items(
            "BOUNDS_BLOCK",
            &p.bounds_src,
            &p.bounds_dst,
            &plan,
            chart,
            "/root/mountain_data/MOUNTAIN_OUTLINES",
        )
        .unwrap();
        assert_eq!(c["scaled_xy_points"], 2);
        assert!(scalers.contains_key("eur_ind"));
        let block = out.path(&["MOUNTAIN_OUTLINES", "OUTLINE_BLOCK"]).unwrap();
        assert_eq!(block.items().unwrap()[0][1], u32v(7));
        assert_ne!(block.items().unwrap()[0][0], src.path(&["MOUNTAIN_OUTLINES", "OUTLINE_BLOCK"]).unwrap().items().unwrap()[0][0]);
    }

    #[test]
    fn trade_nodes_follow_their_position_chart() {
        let plan = Plan::proven();
        let node = Node::record(
            "trade_nodes",
            0,
            vec![
                Node::array(
                    "trade_nodes",
                    0,
                    vec![vec![s16("a"), xyv(0.0, 200.0)], vec![s16("b"), xyv(-500.0, 200.0)]],
                ),
                u32v(3),
            ],
        );
        let mut scalers = BTreeMap::new();
        let mut pairs = Vec::new();
        let out = scale_trade_nodes(&plan, &node, &mut scalers, &mut pairs).unwrap();
        assert_eq!(pairs.iter().map(|p| p.0.as_str()).collect::<Vec<_>>(), ["eur_ind", "america"]);
        let items = out.children().unwrap()[0].items().unwrap();
        let e = expect(&plan, plan.chart("america").unwrap(), [-500.0, 200.0]);
        assert_eq!(items[1][1].as_xy().unwrap().map(f32::to_bits), e.map(f32::to_bits));
        assert_eq!(items[1][0], s16("b"));
        assert_eq!(out.children().unwrap()[1], u32v(3));
        assert_eq!(scalers.len(), 2);
    }

    #[test]
    fn theatres_use_their_key_chart_and_unknown_keys_fail() {
        let plan = Plan::proven();
        let item = |key: &str| {
            vec![Node::record(
                "theatre",
                0,
                vec![s16(key), xyv(-180.0, 140.0), xyv(440.0, 520.0)],
            )]
        };
        let mut scalers = BTreeMap::new();
        let mut pairs = Vec::new();
        let ok = Node::array("theatres_and_region_keys", 0, vec![item("2")]);
        let out = scale_theatres(&plan, "theatres_and_region_keys", &ok, &mut scalers, &mut pairs).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].chart, "eur_ind");
        let rect = out.items().unwrap()[0][0].children().unwrap()[1].as_xy().unwrap();
        let e = expect(&plan, plan.chart("2").unwrap(), [-180.0, 140.0]);
        assert_eq!(rect.map(f32::to_bits), e.map(f32::to_bits));
        let chart = plan.chart("2").unwrap();
        let c = check_scaled_items("theatres_and_region_keys", &pairs[0].src, &pairs[0].dst, &plan, chart, "/root").unwrap();
        assert_eq!(c["scaled_xy_points"], 2);

        let missing = Node::array("theatres_and_region_keys", 0, vec![item("999")]);
        let err = msg(scale_theatres(&plan, "theatres_and_region_keys", &missing, &mut scalers, &mut pairs));
        assert!(err.contains("theatre 999"), "{err}");
    }
}
