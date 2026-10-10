//! Output coordinate envelope of every position the navigation data carries, and the stock-engine window gates.

use anyhow::{Result, bail};
use serde_json::{Value as Json, json};

use super::util::{I32_SPAN, UNIT};

#[derive(Clone, Debug)]
struct Box2 {
    lo: [i64; 2],
    hi: [i64; 2],
    n: usize,
}

/// Raw fixed-point bounds per tag (insertion ordered, like the POC's dict, so ties name the first tag).
#[derive(Default)]
pub(super) struct Envelope {
    tags: Vec<(String, Box2)>,
}

impl Default for Box2 {
    fn default() -> Self {
        Self { lo: [i64::MAX; 2], hi: [i64::MIN; 2], n: 0 }
    }
}

/// `round_half_even(v * 2^20)` of an f32 (or f64) world coordinate.
pub(super) fn raw_of(v: f64) -> i64 {
    (v * UNIT as f64).round_ties_even() as i64
}

pub(super) const WINDOW_CHECKS: [&str; 3] = ["diff_window", "grid_index_window", "sum_window"];

impl Envelope {
    pub fn add(&mut self, tag: &str, x: i64, y: i64) {
        let at = match self.tags.iter().position(|(t, _)| t == tag) {
            Some(i) => i,
            None => {
                self.tags.push((tag.to_string(), Box2::default()));
                self.tags.len() - 1
            }
        };
        let b = &mut self.tags[at].1;
        b.n += 1;
        for (k, v) in [x, y].into_iter().enumerate() {
            b.lo[k] = b.lo[k].min(v);
            b.hi[k] = b.hi[k].max(v);
        }
    }

    pub fn add_f32(&mut self, tag: &str, p: [f32; 2]) {
        self.add(tag, raw_of(p[0] as f64), raw_of(p[1] as f64));
    }

    /// `(lo, hi, tag of lo, tag of hi)` per axis over all tags.
    fn total(&self) -> Result<([i64; 2], [i64; 2], [String; 2], [String; 2])> {
        let mut lo = [i64::MAX; 2];
        let mut hi = [i64::MIN; 2];
        let mut wlo = [String::new(), String::new()];
        let mut whi = [String::new(), String::new()];
        for (tag, b) in &self.tags {
            for k in 0..2 {
                if b.lo[k] < lo[k] {
                    lo[k] = b.lo[k];
                    wlo[k] = tag.clone();
                }
                if b.hi[k] > hi[k] {
                    hi[k] = b.hi[k];
                    whi[k] = tag.clone();
                }
            }
        }
        if self.tags.is_empty() {
            bail!("navigation envelope is empty: no position was recorded");
        }
        Ok((lo, hi, wlo, whi))
    }

    pub fn dump(&self) -> Json {
        let mut sorted: Vec<&(String, Box2)> = self.tags.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut m = serde_json::Map::new();
        for (tag, b) in sorted {
            m.insert(
                tag.clone(),
                json!({
                    "count": b.n,
                    "min_world": [b.lo[0] as f64 / UNIT as f64, b.lo[1] as f64 / UNIT as f64],
                    "max_world": [b.hi[0] as f64 / UNIT as f64, b.hi[1] as f64 / UNIT as f64],
                }),
            );
        }
        Json::Object(m)
    }
}

/// Intermediate-arithmetic checks for the stock Q20 executable on the FINAL positions of all navigation data:
///  * `diff_window`: every pair of positions differs by < 2^31 raw (2048 world units) per axis, so a wrapping i32
///    `sub pos_a, pos_b` (142 integer delta sites; lerps, headings, nearest-N sort) stays exact;
///  * `grid_index_window`: position - world_min lies in [0, 2^31) for every position, so the grid-index function
///    (wrapping sub, clamp at 0, divide by cell) never wraps/clamps and startX/startY are nominal;
///  * `sum_window`: |position| < 2^31/2 raw (1024 world units), so a + b of any two positions stays in i32.
///
/// Returns the report and the names of the failed checks.
pub(super) fn check_stock_window(env: &Envelope, wmin_raw: [i64; 2]) -> Result<(Json, Vec<String>)> {
    let (lo, hi, wlo, whi) = env.total()?;
    let mut res = serde_json::Map::new();
    let mut failed = Vec::new();
    for name in WINDOW_CHECKS {
        let mut per = Vec::new();
        let mut all_ok = true;
        for (k, ax) in ["x", "y"].into_iter().enumerate() {
            let (value, ok) = match name {
                "diff_window" => (json!(hi[k] - lo[k]), hi[k] - lo[k] < I32_SPAN),
                "grid_index_window" => (
                    json!([lo[k] - wmin_raw[k], hi[k] - wmin_raw[k]]),
                    lo[k] >= wmin_raw[k] && hi[k] - wmin_raw[k] < I32_SPAN,
                ),
                _ => (json!([2 * lo[k], 2 * hi[k]]), 2 * lo[k] >= -I32_SPAN && 2 * hi[k] < I32_SPAN),
            };
            all_ok &= ok;
            per.push(json!({
                "axis": ax, "ok": ok, "value_raw": value,
                "lowest_world": lo[k] as f64 / UNIT as f64, "lowest_tag": wlo[k],
                "highest_world": hi[k] as f64 / UNIT as f64, "highest_tag": whi[k],
            }));
        }
        if !all_ok {
            failed.push(name.to_string());
        }
        res.insert(name.to_string(), json!({"ok": all_ok, "axes": per}));
    }
    Ok((Json::Object(res), failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(v: f64) -> i64 {
        (v * UNIT as f64) as i64
    }

    #[test]
    fn stock_positions_pass_all_windows() {
        let mut env = Envelope::default();
        env.add("a", world(-669.0), world(-604.0));
        env.add("b", world(682.0), world(681.0));
        let (report, failed) = check_stock_window(&env, [world(-1280.0), world(-640.0)]).unwrap();
        assert!(failed.is_empty(), "{report}");
        assert_eq!(report["diff_window"]["ok"], json!(true));
    }

    #[test]
    fn diff_window_fails_at_2048_world_units() {
        let mut env = Envelope::default();
        env.add("lo", world(-1100.0), 0);
        env.add("hi", world(1000.0), 0);
        let (_, failed) = check_stock_window(&env, [world(-1280.0), world(-640.0)]).unwrap();
        assert!(failed.contains(&"diff_window".to_string()));
    }

    #[test]
    fn grid_index_window_fails_below_world_minimum_and_sum_window_beyond_1024() {
        let mut env = Envelope::default();
        env.add("below", world(-1281.0), world(-600.0));
        env.add("far", world(10.0), world(-600.0));
        let (_, failed) = check_stock_window(&env, [world(-1280.0), world(-640.0)]).unwrap();
        assert!(failed.contains(&"grid_index_window".to_string()));
        assert!(failed.contains(&"sum_window".to_string()));

        let mut env = Envelope::default();
        env.add("edge", world(1024.0), world(0.0));
        let (_, failed) = check_stock_window(&env, [world(-1280.0), world(-640.0)]).unwrap();
        assert!(failed.contains(&"sum_window".to_string()), "2 * 1024.0 world units = 2^31 raw is out of the window");
    }

    #[test]
    fn first_tag_wins_ties_and_empty_envelope_is_an_error() {
        let mut env = Envelope::default();
        env.add("first", 5, 5);
        env.add("second", 5, 5);
        let (report, _) = check_stock_window(&env, [0, 0]).unwrap();
        assert_eq!(report["diff_window"]["axes"][0]["lowest_tag"], json!("first"));
        assert!(check_stock_window(&Envelope::default(), [0, 0]).is_err());
    }
}
