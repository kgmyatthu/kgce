//! `scripting.lua` migration: the two camera calls follow the atlas (native port of `atlas_script`).

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use regex::Regex;
use serde_json::{Value as Json, json};

use crate::atlas::{Chart, Plan};

/// Lua calls with numeric arguments that are NOT world coordinates/distances and are copied verbatim.
const LUA_UNCHANGED_NUMERIC: [(&str, &str); 2] = [
    ("set_zoom_limit", "camera zoom limits (dimensionless factors)"),
    ("SetCameraZoom", "camera zoom factor"),
];

const OLD_COMMENT: &str = "-- camera bounds (xmin, ymax, xmax, ymin; same argument order as tut_napoleon): re-applied on every UI creation so saves made\n\t\t-- with the old Europe-sized bounds (+-640/+-320 stored in EPISODIC_RESTRICTIONS) are corrected on load";
const NEW_COMMENT: &str = "-- Stock-Q20 compact atlas camera bounds (xmin, ymax, xmax, ymin), identical to EPISODIC_RESTRICTIONS:\n\t\t-- output bbox of all theatre rects + the original margins. Requires a new campaign built from this startpos.";

/// Python-style `format(v, '.<prec>g')`.
pub fn fmt_g(v: f64, prec: usize) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if !v.is_finite() {
        return format!("{v}");
    }
    let sci = format!("{:.*e}", prec - 1, v);
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let strip = |s: &str| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s.to_string() }
    };
    if exp < -4 || exp >= prec as i32 {
        format!("{}e{}{:02}", strip(mantissa), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        strip(&format!("{:.*}", (prec as i32 - 1 - exp) as usize, v))
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '"' || c == '\''
}

/// True when `args` contains a numeric literal that is not part of an identifier or a quoted string.
fn has_number(args: &str) -> bool {
    let b: Vec<char> = args.chars().collect();
    for start in 0..b.len() {
        let j = start + usize::from(b[start] == '-');
        if j >= b.len() || !b[j].is_ascii_digit() || (start > 0 && is_word(b[start - 1])) {
            continue;
        }
        let mut k = j;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        let mut ends = Vec::new();
        if k + 1 < b.len() && b[k] == '.' && b[k + 1].is_ascii_digit() {
            let mut m = k + 1;
            while m < b.len() && b[m].is_ascii_digit() {
                m += 1;
            }
            ends.push(m);
        }
        ends.push(k);
        if ends.iter().any(|&e| e >= b.len() || !is_word(b[e])) {
            return true;
        }
    }
    false
}

fn parse_args(name: &str, args: &str) -> Result<Vec<f64>> {
    args.split(',').map(|v| v.trim().parse::<f64>().with_context(|| format!("{name}({args}): argument {:?} is not a number", v.trim()))).collect()
}

/// Rewrite the camera calls of `source`. `src_bounds` / `new_bounds` are `(xmin, ymin, xmax, ymax)`.
pub fn atlas_script(source: &str, plan: &Plan, src_bounds: [f32; 4], new_bounds: [f32; 4]) -> Result<(String, Json)> {
    let call = Regex::new(r"(set_map_bounds|SetCameraTargetInstant)\(([^()]*)\)").expect("static regex");
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut notes = serde_json::Map::new();
    let mut out = String::with_capacity(source.len());
    let mut last = 0;
    for m in call.captures_iter(source) {
        let whole = m.get(0).expect("match");
        let (name, args) = (&m[1], &m[2]);
        let parts = parse_args(name, args)?;
        *counts.entry(name.to_string()).or_default() += 1;
        out.push_str(&source[last..whole.start()]);
        last = whole.end();
        if name == "set_map_bounds" {
            ensure!(parts.len() == 4, "unexpected set_map_bounds call: {} arguments", parts.len());
            // Lua order (xmin, ymax, xmax, ymin) must equal the EPISODIC_RESTRICTIONS source rectangle; the generator
            // prints it with one decimal, so a difference below the rounding step is the same rectangle
            let want = [src_bounds[0], src_bounds[3], src_bounds[2], src_bounds[1]];
            ensure!(
                parts.iter().zip(want).all(|(p, w)| (p - f64::from(w)).abs() <= 0.05 + 1e-6),
                "script camera bounds {parts:?} differ from startpos EPISODIC_RESTRICTIONS {src_bounds:?}"
            );
            let n = new_bounds;
            notes.insert("set_map_bounds".into(), json!({"source": parts, "output": [n[0], n[3], n[2], n[1]]}));
            out.push_str(&format!("{name}({}, {}, {}, {})", fmt_g(f64::from(n[0]), 9), fmt_g(f64::from(n[3]), 9), fmt_g(f64::from(n[2]), 9), fmt_g(f64::from(n[1]), 9)));
        } else {
            ensure!(parts.len() == 2, "unexpected camera script call: {} arguments", parts.len());
            let pt = [parts[0], parts[1]];
            let owners: Vec<&Chart> = plan.charts.iter().filter(|c| c.theatre_rects.values().any(|r| Plan::contains(*r, pt))).collect();
            ensure!(owners.len() == 1, "camera target {pt:?} has {} owning theatre charts", owners.len());
            let moved = plan.xy(pt, owners[0]);
            notes.insert("SetCameraTargetInstant".into(), json!({"source": pt, "chart": owners[0].key, "output": moved}));
            let r3 = |v: f64| format!("{v:.3}").parse::<f64>().expect("fixed decimal");
            out.push_str(&format!("{name}({}, {})", fmt_g(r3(moved[0]), 10), fmt_g(r3(moved[1]), 10)));
        }
    }
    out.push_str(&source[last..]);
    // audit: no other call may carry numbers unless it is a documented non-spatial one
    let any_call = Regex::new(r"([A-Za-z_][\w.:]*)\(([^()]*)\)").expect("static regex");
    for m in any_call.captures_iter(&out) {
        let name = m[1].rsplit(':').next().unwrap_or("").rsplit('.').next().unwrap_or("");
        if has_number(&m[2]) && !matches!(name, "set_map_bounds" | "SetCameraTargetInstant") && !LUA_UNCHANGED_NUMERIC.iter().any(|(n, _)| *n == name) {
            bail!("script call {name}({}) carries numbers that are not classified as spatial/non-spatial: classify it before enabling the atlas", &m[2]);
        }
    }
    ensure!(
        counts.get("set_map_bounds") == Some(&1) && counts.get("SetCameraTargetInstant") == Some(&1) && counts.len() == 2,
        "unexpected camera call counts {counts:?}: expected exactly one set_map_bounds and one SetCameraTargetInstant"
    );
    let comment_replaced = out.contains(OLD_COMMENT);
    let text = out.replace(OLD_COMMENT, NEW_COMMENT);
    let report = json!({
        "calls": counts,
        "notes": notes,
        "camera_comment_replaced": comment_replaced,
        "unchanged_numeric": LUA_UNCHANGED_NUMERIC.iter().map(|(n, why)| (n.to_string(), why.to_string())).collect::<BTreeMap<_, _>>(),
    });
    Ok((text, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LUA: &str = "local x = 1\nlocal function OnUICreated(context)\n\t\tscripting.game_interface:set_map_bounds(-690.0, 380.0, 880.0, -150.0)\n\t\tscripting.game_interface:set_zoom_limit(1.00, 0.2)\n\t\tCampaignUI.SetCameraZoom(0.95)\n\t\tCampaignUI.SetCameraTargetInstant(100.5, 300.25)\n\t\tout.ting(\"Moving camera to initial position\")\nend\n";

    #[test]
    fn python_g_formatting_matches() {
        assert_eq!(fmt_g(f64::from(0.1f32), 9), "0.100000001");
        assert_eq!(fmt_g(-690.0, 9), "-690");
        assert_eq!(fmt_g(1234.5625, 9), "1234.5625");
        assert_eq!(fmt_g(1e-5, 9), "1e-05");
        assert_eq!(fmt_g(123456789012.0, 9), "1.23456789e+11");
        assert_eq!(fmt_g(0.0, 9), "0");
        assert_eq!(fmt_g(-239.125, 10), "-239.125");
    }

    #[test]
    fn number_audit_ignores_identifiers_and_strings_only() {
        assert!(has_number("1.5, 0.2"));
        assert!(has_number("x, -3"));
        assert!(!has_number("\"area 51\", name2"));
        assert!(!has_number("context.string"));
    }

    #[test]
    fn camera_calls_follow_the_atlas_and_other_numbers_fail_closed() {
        let plan = Plan::proven();
        let src = [-690.0f32, -150.0, 880.0, 380.0];
        let new = [-500.0f32, -100.0, 700.0, 300.0];
        let (text, report) = atlas_script(LUA, &plan, src, new).unwrap();
        assert!(text.contains("set_map_bounds(-500, 300, 700, -100)"), "{text}");
        let chart = plan.chart("2").unwrap();
        let moved = plan.xy([100.5, 300.25], chart);
        let r3 = |v: f64| format!("{v:.3}").parse::<f64>().unwrap();
        assert!(text.contains(&format!("SetCameraTargetInstant({}, {})", fmt_g(r3(moved[0]), 10), fmt_g(r3(moved[1]), 10))), "{text}");
        assert!(text.contains("set_zoom_limit(1.00, 0.2)") && text.contains("SetCameraZoom(0.95)"));
        assert_eq!(report["calls"]["set_map_bounds"], 1);
        // a script whose bounds do not equal the start position bounds is rejected
        assert!(atlas_script(LUA, &plan, [-640.0, -150.0, 880.0, 380.0], new).is_err());
        // a further spatial-looking call is not silently copied
        let extra = LUA.replace("end\n", "\t\tscripting.game_interface:teleport(10, 20)\nend\n");
        assert!(atlas_script(&extra, &plan, src, new).is_err());
        // an unowned camera target is rejected
        assert!(atlas_script(&LUA.replace("100.5, 300.25", "-1200, -600"), &plan, src, new).is_err());
    }
}
