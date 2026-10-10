//! Small shared helpers of the navigation migration: exact integer rounding, bounds/range accumulators with the POC's
//! `dump()` JSON shapes, an insertion-ordered counter and the crop-union containment test.

use anyhow::{Result, bail};
use serde_json::{Map, Value as Json, json};

use crate::esf::{Node, Value};

/// `-2147483647`: the dummy value of pathfinding corner vertices; never a legal position.
pub(super) const SENT_VERTEX: i32 = -2_147_483_647;
/// 2^20: the Q20 unit.
pub(super) const UNIT: i64 = 1 << 20;
pub(super) const I32_SPAN: i64 = 1 << 31;

/// Wrap an integer into the i32 range like the baseline engine's wrapping `sub`.
pub(super) fn wrap_i32(v: i64) -> i64 {
    v as i32 as i64
}

/// Floor division by a positive divisor (Python `//`).
pub(super) fn floor_div(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0);
    num.div_euclid(den)
}

/// Exact `round_half_even(num / den)` for `den > 0` (Python `round(Fraction)`).
pub(super) fn round_div_even(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0);
    let q = num.div_euclid(den);
    let r = num.rem_euclid(den);
    let twice = 2 * r;
    if twice < den {
        q
    } else if twice > den {
        q + 1
    } else if q.rem_euclid(2) == 0 {
        q
    } else {
        q + 1
    }
}

pub(super) fn sgn(v: i128) -> i32 {
    (v > 0) as i32 - (v < 0) as i32
}

/// Axis-aligned bounds accumulator (`Bounds` of the POC).
#[derive(Clone, Debug)]
pub(super) struct Bounds {
    pub lo: [f64; 2],
    pub hi: [f64; 2],
    pub n: usize,
}

impl Bounds {
    pub fn new() -> Self {
        Self {
            lo: [f64::INFINITY; 2],
            hi: [f64::NEG_INFINITY; 2],
            n: 0,
        }
    }

    pub fn add(&mut self, x: f64, y: f64) {
        self.n += 1;
        for (k, v) in [x, y].into_iter().enumerate() {
            if v < self.lo[k] {
                self.lo[k] = v;
            }
            if v > self.hi[k] {
                self.hi[k] = v;
            }
        }
    }

    pub fn dump(&self) -> Json {
        if self.n == 0 {
            return Json::Null;
        }
        json!({"count": self.n, "min": [self.lo[0], self.lo[1]], "max": [self.hi[0], self.hi[1]]})
    }
}

/// Scalar range accumulator (`Range` of the POC).
#[derive(Clone, Debug)]
pub(super) struct Range {
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
}

impl Range {
    pub fn new() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
            n: 0,
        }
    }

    pub fn add(&mut self, v: f64) {
        self.n += 1;
        self.lo = self.lo.min(v);
        self.hi = self.hi.max(v);
    }

    pub fn dump(&self) -> Json {
        if self.n == 0 {
            return Json::Null;
        }
        json!({"count": self.n, "min": self.lo, "max": self.hi})
    }
}

/// Counter whose keys keep the order of their first `add` (Python `Counter` as dumped by `dict(c)`).
#[derive(Clone, Debug, Default)]
pub(super) struct Counter {
    items: Vec<(String, i64)>,
}

impl Counter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, key: &str, by: i64) {
        match self.items.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v += by,
            None => self.items.push((key.to_string(), by)),
        }
    }

    pub fn get(&self, key: &str) -> i64 {
        self.items.iter().find(|(k, _)| k == key).map_or(0, |(_, v)| *v)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, i64)> {
        self.items.iter().map(|(k, v)| (k.as_str(), *v))
    }

    pub fn to_json(&self) -> Json {
        let mut m = Map::new();
        for (k, v) in &self.items {
            m.insert(k.clone(), json!(v));
        }
        Json::Object(m)
    }
}

/// `true` when the union of the closed rectangles `rects` covers the closed box `[lo, hi]` (shapely `covers`).
pub(super) fn rects_cover(rects: &[[[f64; 2]; 2]], lo: [f64; 2], hi: [f64; 2]) -> bool {
    let (lo, hi) = (
        [lo[0].min(hi[0]), lo[1].min(hi[1])],
        [lo[0].max(hi[0]), lo[1].max(hi[1])],
    );
    let breaks = |a: usize| -> Vec<f64> {
        let mut v = vec![lo[a], hi[a]];
        for r in rects {
            for e in [r[0][a], r[1][a]] {
                if e > lo[a] && e < hi[a] {
                    v.push(e);
                }
            }
        }
        v.sort_by(|p, q| p.total_cmp(q));
        v.dedup();
        v
    };
    let spans = |v: &[f64]| -> Vec<(f64, f64)> {
        if v.len() == 1 {
            vec![(v[0], v[0])]
        } else {
            v.windows(2).map(|w| (w[0], w[1])).collect()
        }
    };
    let (xs, ys) = (spans(&breaks(0)), spans(&breaks(1)));
    xs.iter().all(|&(a, b)| {
        ys.iter().all(|&(c, d)| {
            rects
                .iter()
                .any(|r| r[0][0] <= a && b <= r[1][0] && r[0][1] <= c && d <= r[1][1])
        })
    })
}

// ------------------------------------------------------------------------------------------ typed scalar access
/// Exactly an i32 scalar (Python `value(node, 4)`).
pub(super) fn node_i32(n: &Node) -> Result<i32> {
    match n {
        Node::Value(Value::I32(v)) => Ok(*v),
        other => bail!("expected i32 scalar, found {}", describe(other)),
    }
}

pub(super) fn node_u32(n: &Node) -> Result<u32> {
    match n {
        Node::Value(Value::U32(v)) => Ok(*v),
        other => bail!("expected u32 scalar, found {}", describe(other)),
    }
}

pub(super) fn node_u16(n: &Node) -> Result<u16> {
    match n {
        Node::Value(Value::U16(v)) => Ok(*v),
        other => bail!("expected u16 scalar, found {}", describe(other)),
    }
}

/// Short human description of a node for error messages.
pub(super) fn describe(n: &Node) -> String {
    match n {
        Node::Record { name, version, children } => format!("record {name} v{version} n={}", children.len()),
        Node::Array { name, version, items } => format!("array {name}[] v{version} n={}", items.len()),
        Node::Value(Value::Array { code, data }) => format!("packed array {code:#x} ({} bytes)", data.len()),
        Node::Value(v) => format!("{v:?}"),
    }
}

/// Packed array of exactly `code` (0x40 | value code) as raw bytes.
pub(super) fn packed(n: &Node, code: u8) -> Result<&[u8]> {
    match n {
        Node::Value(Value::Array { code: c, data }) if *c == code => Ok(data),
        other => bail!("expected packed array {code:#x}, found {}", describe(other)),
    }
}

pub(super) fn le_u32s(data: &[u8]) -> Result<Vec<u32>> {
    if data.len() % 4 != 0 {
        bail!("u32 array holds {} bytes", data.len());
    }
    Ok(data.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Index of the single child named `name` (record or record array).
pub(super) fn index_of(children: &[Node], parent: &str, name: &str) -> Result<usize> {
    let hits: Vec<usize> = children
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name() == Some(name))
        .map(|(i, _)| i)
        .collect();
    if hits.len() != 1 {
        bail!("{parent}: expected one {name}, found {}", hits.len());
    }
    Ok(hits[0])
}

/// `f32` rounding of an f64 result that must stay finite.
pub(super) fn to_f32(v: f64) -> Result<f32> {
    let out = v as f32;
    if !v.is_finite() || !out.is_finite() {
        bail!("non-finite float {v}");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_even_ties_and_signs() {
        assert_eq!(round_div_even(1, 2), 0);
        assert_eq!(round_div_even(3, 2), 2);
        assert_eq!(round_div_even(5, 2), 2);
        assert_eq!(round_div_even(-1, 2), 0);
        assert_eq!(round_div_even(-3, 2), -2);
        assert_eq!(round_div_even(-5, 2), -2);
        assert_eq!(round_div_even(7, 4), 2);
        assert_eq!(round_div_even(-7, 4), -2);
        assert_eq!(round_div_even(1, 1 << 20), 0);
    }

    #[test]
    fn rounding_matches_plan_fixed_value_for_stock_scale() {
        // The exact rational rule must equal the float rule of Plan::fixed_value for in-window values (ties included).
        let plan = crate::atlas::Plan::proven();
        let n = (plan.scale * (1u64 << 20) as f64) as i128;
        // N = 2 * 708679, so src = 2^18 * odd lands exactly on a half: 262144 * N / 2^20 = 354339.5 -> even 354340
        assert_eq!(round_div_even(262_144 * n, 1 << 20), 354_340);
        assert_eq!(round_div_even(786_432 * n, 1 << 20), 1_063_018);
        assert_eq!(round_div_even(-262_144 * n, 1 << 20), -354_340);
        for off in [0i128, -446_407_024, 472_040_960] {
            for src in [-157_287_008i128, -1, 0, 1, 2_097_152, 80_744_288, 524_287, 524_289, -524_287, -524_289, 262_144, 786_432, -262_144, -786_432] {
                let exact = round_div_even(src * n, 1 << 20) + off;
                let float = plan
                    .fixed_value(src as f64 / (1u64 << 20) as f64 * plan.scale + off as f64 / (1u64 << 20) as f64)
                    .unwrap();
                assert_eq!(exact, float as i128, "src {src} off {off}");
            }
        }
    }

    #[test]
    fn wrap_and_floor_division() {
        assert_eq!(wrap_i32((1 << 31) + 5), -(1 << 31) + 5);
        assert_eq!(wrap_i32(-5), -5);
        assert_eq!(floor_div(-1, 2), -1);
        assert_eq!(floor_div(5, 2), 2);
    }

    #[test]
    fn crop_union_cover() {
        let rects = [[[0.0, 0.0], [10.0, 10.0]], [[10.0, 0.0], [20.0, 10.0]]];
        assert!(rects_cover(&rects, [2.0, 2.0], [18.0, 8.0]));
        assert!(rects_cover(&rects, [0.0, 0.0], [20.0, 10.0]));
        assert!(!rects_cover(&rects, [2.0, 2.0], [18.0, 12.0]));
        let gap = [[[0.0, 0.0], [10.0, 10.0]], [[11.0, 0.0], [20.0, 10.0]]];
        assert!(!rects_cover(&gap, [2.0, 2.0], [18.0, 8.0]));
    }

    #[test]
    fn counter_keeps_first_insertion_order() {
        let mut c = Counter::new();
        c.add("b", 0);
        c.add("a", 2);
        c.add("b", 1);
        let keys: Vec<_> = c.iter().collect();
        assert_eq!(keys, vec![("b", 1), ("a", 2)]);
    }
}
