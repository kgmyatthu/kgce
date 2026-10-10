//! One coordinate axis of a chart's Q20 -> Q20 map: strictly increasing integers over every value that occurs.
//!
//! Start from the plain rule `round_half_even(src * N / 2^20) + offset` (exact integer arithmetic; `N = scale * 2^20`).
//! Every pin (a grid line `origin + k * cell` of some pathfinding area) is fixed to its exact image `src * N / 2^20 +
//! offset` (must be integral). Inside each interval between consecutive pins (and in the two open ends) values are
//! clamped to leave room for the values that follow and then pushed forward by the minimum needed to keep the order
//! strictly increasing. `repair_orientation` may afterwards move unpinned values by a few raw units (never past a
//! neighbour) to preserve polygon orientations.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, bail};
use serde_json::{Map, Value as Json, json};

use super::util::{Counter, UNIT, round_div_even};

pub(super) struct Axis {
    pub name: String,
    scale_n: i128,
    offset: i64,
    unit: i64,
    pub allv: Vec<i64>,
    pub init: Vec<i64>,
    pub t: Vec<i64>,
    pub pinned: Vec<bool>,
    base_t: Vec<i64>,
    n_values: usize,
    n_pins: usize,
    pins_not_in_geometry: usize,
}

fn run(
    name: &str,
    t: &mut [i64],
    init: &[i64],
    lo: usize,
    hi: usize,
    lower: Option<i64>,
    upper: Option<i64>,
) -> Result<()> {
    let mut prev = lower;
    let m = (hi - lo) as i64;
    for (j, i) in (lo..hi).enumerate() {
        let mut val = init[i];
        if let Some(u) = upper {
            val = val.min(u - (m - j as i64));
        }
        if let Some(p) = prev {
            val = val.max(p + 1);
        }
        t[i] = val;
        prev = Some(val);
    }
    if let (Some(u), Some(p)) = (upper, prev) {
        if m > 0 && p >= u {
            bail!("axis {name}: no room between pins {lower:?} and {upper:?} for {m} values");
        }
    }
    Ok(())
}

impl Axis {
    /// `expected(v)` is the plan's own float rule (`Plan::fixed_value(v / 2^20 * scale + offset)`); the exact rule must
    /// agree with it for every value.
    pub fn new(
        name: &str,
        values: &BTreeSet<i64>,
        pins: &BTreeSet<i64>,
        scale_n: i64,
        offset: i64,
        unit: i64,
        expected: &dyn Fn(i64) -> Result<i64>,
    ) -> Result<Self> {
        let scale_n = scale_n as i128;
        let allv: Vec<i64> = values.union(pins).copied().collect();
        let n = allv.len();
        let init: Vec<i64> = allv
            .iter()
            .map(|&v| (round_div_even(v as i128 * scale_n, UNIT as i128) + offset as i128) as i64)
            .collect();
        for (&v, &t) in allv.iter().zip(&init) {
            let want = expected(v)?;
            if t != want {
                bail!("axis {name}: plain rounding disagrees with the plan for {v}: exact {t}, plan {want}");
            }
        }
        let pinned: Vec<bool> = allv.iter().map(|v| pins.contains(v)).collect();
        let pin_idx: Vec<usize> = pinned.iter().enumerate().filter(|(_, p)| **p).map(|(i, _)| i).collect();
        if pin_idx.is_empty() {
            bail!("axis {name}: no grid-line pins");
        }
        let mut t = vec![0i64; n];
        for &i in &pin_idx {
            let num = allv[i] as i128 * scale_n;
            if num.rem_euclid(UNIT as i128) != 0 {
                bail!("axis {name}: grid line {} has no exact image", allv[i]);
            }
            t[i] = (num / UNIT as i128 + offset as i128) as i64;
        }
        let first = pin_idx[0];
        let last = *pin_idx.last().unwrap();
        let tf = t[first];
        run(name, &mut t, &init, 0, first, None, Some(tf))?;
        for w in pin_idx.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (ta, tb) = (t[a], t[b]);
            run(name, &mut t, &init, a + 1, b, Some(ta), Some(tb))?;
        }
        let tl = t[last];
        run(name, &mut t, &init, last + 1, n, Some(tl), None)?;
        let axis = Self {
            name: name.to_string(),
            scale_n,
            offset,
            unit,
            base_t: t.clone(),
            allv,
            init,
            t,
            pinned,
            n_values: values.len(),
            n_pins: pins.len(),
            pins_not_in_geometry: pins.difference(values).count(),
        };
        axis.check()?;
        Ok(axis)
    }

    /// Position of `v` in the sorted value list.
    pub fn index_of(&self, v: i64) -> Option<usize> {
        self.allv.binary_search(&v).ok()
    }

    /// `src * N + offset * 2^20` (the exact image times 2^20).
    fn exact_num(&self, v: i64) -> i128 {
        v as i128 * self.scale_n + self.offset as i128 * UNIT as i128
    }

    pub fn check(&self) -> Result<()> {
        let t = &self.t;
        if t.windows(2).any(|w| w[0] >= w[1]) {
            bail!("axis {}: quantization map is not strictly increasing", self.name);
        }
        for (i, &v) in self.allv.iter().enumerate() {
            if self.pinned[i] && t[i] as i128 * UNIT as i128 != self.exact_num(v) {
                bail!("axis {}: pinned grid line {v} moved", self.name);
            }
        }
        Ok(())
    }

    pub fn mapping(&self) -> HashMap<i64, i64> {
        self.allv.iter().copied().zip(self.t.iter().copied()).collect()
    }

    /// (report map, overrides): `overrides` holds the entries that differ from the plain rule.
    pub fn report(&self) -> (Map<String, Json>, BTreeMap<i64, i64>) {
        let dev: Vec<(i64, i64, i64)> = self
            .allv
            .iter()
            .zip(self.t.iter().zip(&self.init))
            .filter(|(_, (t, i))| t != i)
            .map(|(&v, (&t, &i))| (v, t, t - i))
            .collect();
        let mut hist: BTreeMap<i64, i64> = BTreeMap::new();
        for (_, _, d) in &dev {
            *hist.entry(d.abs()).or_default() += 1;
        }
        let order_only = self.base_t.iter().zip(&self.init).filter(|(a, b)| a != b).count();
        let errors: Vec<f64> = self
            .allv
            .iter()
            .zip(&self.t)
            .map(|(&v, &t)| (t as i128 * UNIT as i128 - self.exact_num(v)).abs() as f64 / UNIT as f64)
            .collect();
        let max_error = errors.iter().copied().fold(0.0f64, f64::max);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let max_nudge = hist.keys().next_back().copied().unwrap_or(0);
        let mut worst: Vec<&(i64, i64, i64)> = dev.iter().collect();
        worst.sort_by_key(|(_, _, d)| -d.abs());
        let worst: Vec<Json> = worst
            .into_iter()
            .take(5)
            .map(|&(v, t, d)| json!({"src_raw": v, "plain": t - d, "dst_raw": t, "nudge_raw_q18": d}))
            .collect();
        let mut histogram = Counter::new();
        for (k, c) in &hist {
            histogram.add(&k.to_string(), *c);
        }
        let mut m = Map::new();
        m.insert("distinct_values".into(), json!(self.allv.len()));
        m.insert("from_geometry".into(), json!(self.n_values));
        m.insert("grid_line_pins".into(), json!(self.n_pins));
        m.insert("pins_not_in_geometry".into(), json!(self.pins_not_in_geometry));
        m.insert("deviations_from_plain_rounding".into(), json!(dev.len()));
        m.insert("deviations_needed_for_strict_order_only".into(), json!(order_only));
        m.insert("max_nudge_raw_q18".into(), json!(max_nudge));
        m.insert("max_nudge_world_units".into(), json!(max_nudge as f64 / self.unit as f64));
        m.insert("nudge_histogram_raw_q18".into(), histogram.to_json());
        m.insert("max_error_vs_exact_raw_q18".into(), json!(max_error));
        m.insert("max_error_vs_exact_world_units".into(), json!(max_error / self.unit as f64));
        m.insert("mean_abs_error_vs_exact_raw_q18".into(), json!(mean_error));
        m.insert("worst".into(), Json::Array(worst));
        (m, dev.iter().map(|&(v, t, _)| (v, t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: i64 = 1_417_358;

    fn set(v: &[i64]) -> BTreeSet<i64> {
        v.iter().copied().collect()
    }

    fn plain(v: i64, off: i64) -> Result<i64> {
        Ok((round_div_even(v as i128 * N as i128, UNIT as i128) + off as i128) as i64)
    }

    #[test]
    fn strictly_monotonic_with_pins_and_pinned_values_exact() {
        // grid pins every 2^20 * 2 raw; many geometry values squeezed between them, some of them adjacent raw values
        let cell = 2 * UNIT;
        let pins: Vec<i64> = (0..6).map(|k| -3 * UNIT + k * cell).collect();
        let mut vals = BTreeSet::new();
        for p in &pins {
            for d in [1i64, 2, 3, 4, 5, 6, 7, 11, 12] {
                vals.insert(p + d);
            }
        }
        let off = 12_345;
        let axis = Axis::new("x", &vals, &set(&pins), N, off, UNIT, &|v| plain(v, off)).unwrap();
        assert!(axis.t.windows(2).all(|w| w[0] < w[1]));
        for p in &pins {
            let i = axis.index_of(*p).unwrap();
            assert!(axis.pinned[i]);
            assert_eq!(axis.t[i] as i128 * UNIT as i128, *p as i128 * N as i128 + off as i128 * UNIT as i128);
        }
        assert!(axis.index_of(pins[0] + 1).is_some());
        let (_, dev) = axis.report();
        assert!(dev.is_empty(), "scale >= 1 never needs a nudge without orientation repair");
    }

    #[test]
    fn run_pushes_forward_and_reports_missing_room() {
        // values that all round onto the same integer are separated by one raw unit each
        let init = [5i64, 5, 5, 5];
        let mut t = [0i64; 4];
        run("x", &mut t, &init, 0, 4, Some(4), None).unwrap();
        assert_eq!(t, [5, 6, 7, 8]);
        // an upper pin pulls the whole run down so that every value still fits strictly below it
        let mut t = [0i64; 3];
        run("x", &mut t, &[10, 10, 10], 0, 3, Some(0), Some(5)).unwrap();
        assert_eq!(t, [2, 3, 4]);
        // no room: three values strictly between 0 and 3
        let mut t = [0i64; 3];
        assert!(run("x", &mut t, &[1, 2, 3], 0, 3, Some(0), Some(3)).is_err());
    }

    #[test]
    fn missing_pins_and_float_rule_disagreement_are_errors() {
        let e = Axis::new("x", &set(&[1, 2]), &BTreeSet::new(), N, 0, UNIT, &|v| plain(v, 0));
        assert!(e.is_err());
        let e = Axis::new("x", &set(&[1, 2]), &set(&[0]), N, 0, UNIT, &|v| plain(v, 1));
        assert!(e.is_err());
    }

    #[test]
    fn pin_without_exact_image_is_rejected() {
        // N carries a single factor 2, so a pin that is not a multiple of 2^19 has no integral image
        let e = Axis::new("x", &set(&[]), &set(&[1]), N, 0, UNIT, &|v| plain(v, 0));
        assert!(e.is_err());
    }
}
