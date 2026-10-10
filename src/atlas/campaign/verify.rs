//! Independent re-read of the converted start position against the source: structure, ownership and coverage
//! (native port of the accepted POC `Verifier`). It compares the source tree with the tree decoded from the
//! SERIALISED output, so a writer/parser asymmetry cannot hide a defect.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow};
use serde_json::{Value as Json, json};

use super::convert::{Conv, Counter, FIXED_PAIRS, Q, cached_point_offsets};
use super::nodes::*;
use super::visibility::{Tree, decode_window, tree_bbox};
use crate::atlas::{Chart, Plan};
use crate::esf::{Node, Value};

pub struct Verification {
    pub violations: Vec<(String, String)>,
    pub stats: Counter,
    pub changed_leaf_classes: BTreeMap<String, u64>,
}

impl Verification {
    pub fn to_json(&self) -> Json {
        json!({
            "violation_count": self.violations.len(),
            "violations": self.violations.iter().take(200).map(|(w, d)| json!({"at": w, "detail": d})).collect::<Vec<_>>(),
            "stats": self.stats,
            "changed_leaf_classes": self.changed_leaf_classes,
        })
    }
}

struct Verifier<'c, 'a> {
    c: &'c Conv<'a>,
    violations: Vec<(String, String)>,
    stats: Counter,
    changed_leaves: BTreeMap<String, u64>,
    item: usize,
}

fn stat(c: &mut Counter, key: &str, by: u64) {
    *c.entry(key.to_string()).or_default() += by;
}

fn u32s(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

pub fn verify(conv: &Conv<'_>, src: &Node, out: &Node) -> Result<Verification> {
    let mut v = Verifier { c: conv, violations: Vec::new(), stats: Counter::new(), changed_leaves: BTreeMap::new(), item: 0 };
    v.compare(src, out, &mut Vec::new());
    v.verify_hlci(src, out)?;
    v.verify_group_coverage(src, out)?;
    Ok(Verification { violations: v.violations, stats: v.stats, changed_leaf_classes: v.changed_leaves })
}

impl<'c, 'a> Verifier<'c, 'a> {
    fn violation(&mut self, at: impl Into<String>, detail: impl Into<String>) {
        self.violations.push((at.into(), detail.into()));
    }

    fn out_chart(&self, pt: [f64; 2]) -> Vec<&'a Chart> {
        let (plan, c): (&'a Plan, &'c Conv<'a>) = (self.c.plan, self.c);
        plan.charts.iter().filter(|ch| c.out_crops[&ch.key].iter().any(|r| Plan::contains(*r, pt))).collect()
    }

    fn check_point(&mut self, at: &str, src: [f64; 2], out: [f64; 2], tol: f64) {
        let found = self.out_chart(out);
        if found.len() != 1 {
            self.violation(at, format!("output point {out:?} lies in {} output crops", found.len()));
            return;
        }
        let chart = found[0];
        let back = self.c.plan.inverse_xy(out, chart);
        let slack = tol + 1.0 / (Q * self.c.plan.scale);
        if (back[0] - src[0]).abs() > slack || (back[1] - src[1]).abs() > slack {
            self.violation(at, format!("inverse map {back:?} != source {src:?}"));
            return;
        }
        if !chart.crop_rects.iter().any(|r| Plan::contains(*r, src)) {
            self.violation(at, format!("source point {src:?} not inside owning chart {} crops", chart.key));
            return;
        }
        let key = format!("chart.{}", chart.key);
        stat(&mut self.stats, "points_checked", 1);
        stat(&mut self.stats, &key, 1);
    }

    fn compare(&mut self, a: &Node, b: &Node, path: &mut Vec<String>) {
        match (a, b) {
            (Node::Record { name: na, version: va, children: ca }, Node::Record { name: nb, version: vb, children: cb }) => {
                if na != nb || va != vb {
                    self.violation(path.join("/"), format!("node mismatch {na} v{va} vs {nb} v{vb}"));
                    return;
                }
                path.push(na.clone());
                match na.as_str() {
                    "LINE_OF_SIGHT" => self.compare_los(ca, cb, path),
                    "QUAD_TREE_BIT_ARRAY" => {
                        if a != b {
                            self.violation(path.join("/"), "shroud tree changed");
                        }
                    }
                    _ => self.compare_items(ca, cb, path, na, false),
                }
                path.pop();
            }
            (Node::Array { name: na, version: va, items: ia }, Node::Array { name: nb, version: vb, items: ib }) => {
                if na != nb || va != vb {
                    self.violation(path.join("/"), format!("node mismatch {na}[] v{va} vs {nb}[] v{vb}"));
                    return;
                }
                let key = format!("{na}[]");
                path.push(key.clone());
                if na == "CAI_WORLD_REGION_HLCIS" {
                    // group deletions: checked id by id in verify_hlci
                } else if ia.len() != ib.len() {
                    self.violation(path.join("/"), "array length changed");
                } else if key == "TRADE_SEGMENTS[]" {
                    for (i, (x, y)) in ia.iter().zip(ib).enumerate() {
                        self.compare_segment(x, y, path, i);
                    }
                } else {
                    for (i, (x, y)) in ia.iter().zip(ib).enumerate() {
                        self.item = i;
                        self.compare_items(x, y, path, &key, true);
                    }
                }
                path.pop();
            }
            _ => self.violation(path.join("/"), "node kind changed"),
        }
    }

    fn compare_items(&mut self, x: &[Node], y: &[Node], path: &mut Vec<String>, name: &str, _in_array: bool) {
        if x.len() != y.len() {
            self.violation(path.join("/"), "child count changed");
            return;
        }
        for (at, (p, q)) in x.iter().zip(y).enumerate() {
            match (p, q) {
                (Node::Value(pv), Node::Value(qv)) => {
                    if pv != qv {
                        self.leaf_change(name, path, at, pv, qv, x, y);
                    }
                }
                (Node::Value(_), _) | (_, Node::Value(_)) => self.violation(format!("{}[{at}]", path.join("/")), "node kind changed"),
                _ => self.compare(p, q, path),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn leaf_change(&mut self, name: &str, path: &[String], at: usize, p: &Value, q: &Value, x: &[Node], y: &[Node]) {
        let at_s = format!("{}[{at}]", path.join("/"));
        *self.changed_leaves.entry(format!("{name}[{at}]")).or_default() += 1;
        if name == "CAI_REGION_HLCI" {
            return; // checked in verify_hlci
        }
        if let Some((_, starts)) = FIXED_PAIRS.iter().find(|(n, _)| *n == name) {
            if starts.iter().any(|&s| at == s || at == s + 1) {
                let s = if starts.contains(&at) { at } else { at - 1 };
                if at == s || x[s] == y[s] {
                    match (i32_at(x, s), i32_at(x, s + 1), i32_at(y, s), i32_at(y, s + 1)) {
                        (Ok(a), Ok(b), Ok(c), Ok(d)) => {
                            self.check_point(&at_s, [f64::from(a) / Q, f64::from(b) / Q], [f64::from(c) / Q, f64::from(d) / Q], 2e-6)
                        }
                        _ => self.violation(at_s, "fixed pair is not i32"),
                    }
                }
                return;
            }
        }
        match (name, at, p, q) {
            ("CAI_REGION", 1, Value::Array { code: ca, data: da }, Value::Array { code: cb, data: db }) => {
                let want: Vec<u32> = u32s(da).into_iter().filter(|i| !self.c.deleted_hlci_ids.contains(i)).collect();
                if u32s(db) != want || ca != cb {
                    self.violation(at_s, "CAI_REGION HLCI id list != source minus deleted groups");
                }
            }
            ("CAI_REGION", 4 | 5, _, _) => {
                if at == 4 || x[4] == y[4] {
                    match (f32_at(x, 4), f32_at(x, 5), f32_at(y, 4), f32_at(y, 5)) {
                        (Ok(a), Ok(b), Ok(c), Ok(d)) => self.check_point(&at_s, [f64::from(a), f64::from(b)], [f64::from(c), f64::from(d)], 1e-3),
                        _ => self.violation(at_s, "CAI_REGION centre is not f32"),
                    }
                }
            }
            ("CAI_REGION_BOUNDARY" | "TRADE_ROUTES[]", _, Value::F32(pv), Value::F32(qv)) => {
                if *qv != (f64::from(*pv) * self.c.plan.scale) as f32 {
                    self.violation(at_s, format!("distance {pv} -> {qv}"));
                }
            }
            ("EPISODIC_RESTRICTIONS", 9..=12, _, Value::F32(qv)) => {
                if *qv != self.c.new_bounds[at - 9] {
                    self.violation(at_s, "camera bound mismatch");
                }
            }
            ("CAMPAIGN_CAMERA", 2, Value::XYZ(pv), Value::XYZ(qv)) => {
                self.check_point(&at_s, [f64::from(pv[0]), f64::from(pv[2])], [f64::from(qv[0]), f64::from(qv[2])], 1e-3);
                if pv[1] != qv[1] {
                    self.violation(at_s, "camera height changed");
                }
            }
            ("PATHFINDING_GRID[]", 1, _, _) => self.compare_cached(x, y, &at_s),
            ("TRADE_NODES[]" | "PENDING_BATTLE", _, Value::XY(pv), Value::XY(qv)) => {
                self.check_point(&at_s, pv.map(f64::from), qv.map(f64::from), 1e-3)
            }
            _ => self.violation(at_s, format!("unexpected change of {name}[{at}]")),
        }
    }

    fn compare_cached(&mut self, x: &[Node], y: &[Node], at: &str) {
        let (Ok(ra), Ok(rb), Ok(count)) = (array_at(x, 1, 0x48), array_at(y, 1, 0x48), u32_at(x, 0)) else {
            self.violation(at, "cached polygon payload is not a u32 array");
            return;
        };
        if ra.len() != rb.len() {
            self.violation(at, "cached polygon byte length changed");
            return;
        }
        let (Ok(a), Ok(b)) = (cached_point_offsets(count, ra), cached_point_offsets(count, rb)) else {
            self.violation(at, "cached polygon payload cannot be parsed");
            return;
        };
        if a.iter().map(|k| k.0).ne(b.iter().map(|k| k.0)) {
            self.violation(at, "cached polygon layout changed");
            return;
        }
        let mut is_point = vec![false; ra.len()];
        for (o, _, _) in &a {
            is_point[*o..*o + 8].fill(true);
        }
        if ra.iter().zip(rb).enumerate().any(|(i, (u, v))| u != v && !is_point[i]) {
            self.violation(at, "non-point cached polygon byte changed");
            return;
        }
        let Some(chart_key) = self.c.grid_charts.get(self.item) else {
            self.violation(at, "grid has no owning chart");
            return;
        };
        let Ok(chart) = self.c.plan.chart(chart_key) else {
            self.violation(at, format!("unknown grid chart {chart_key}"));
            return;
        };
        let crops = &self.c.out_crops[&chart.key];
        let slack = 2e-6 + 1.0 / (Q * self.c.plan.scale);
        let mut checked = 0u64;
        for ((_, sx, sy), (_, tx, ty)) in a.iter().zip(&b) {
            let out = [f64::from(*tx) / Q, f64::from(*ty) / Q];
            if !crops.iter().any(|r| Plan::contains(*r, out)) {
                self.violation(at, format!("cached point ({tx},{ty}) outside grid chart {} output crops", chart.key));
                return;
            }
            let back = self.c.plan.inverse_xy(out, chart);
            if (back[0] - f64::from(*sx) / Q).abs() > slack || (back[1] - f64::from(*sy) / Q).abs() > slack {
                self.violation(at, format!("cached point ({sx},{sy})->({tx},{ty}) inverse mismatch"));
                return;
            }
            checked += 1;
        }
        stat(&mut self.stats, "cached_points_checked", checked);
    }

    fn compare_segment(&mut self, x: &[Node], y: &[Node], path: &[String], idx: usize) {
        let at = format!("{}[{idx}]", path.join("/"));
        let (Ok(count), Ok(count_out)) = (u32_at(x, 1), u32_at(y, 1)) else {
            self.violation(at, "segment shape changed");
            return;
        };
        let end = 2 + 4 * count as usize;
        if x.len() != y.len() || count != count_out || x.len() != end + 11 {
            self.violation(at, "segment shape changed");
            return;
        }
        let mut charts: BTreeSet<String> = BTreeSet::new();
        for i in 2..end {
            match (xy_at(x, i), xy_at(y, i)) {
                (Ok(a), Ok(b)) => {
                    self.check_point(&format!("{at}.{i}"), a.map(f64::from), b.map(f64::from), 1e-3);
                    charts.extend(self.out_chart(b.map(f64::from)).iter().map(|c| c.key.clone()));
                }
                _ => self.violation(&at, format!("segment point {i} is not xy")),
            }
        }
        if charts.len() != 1 {
            self.violation(&at, format!("segment output points span charts {charts:?}"));
        }
        for i in [end, end + 1] {
            match (array_at(x, i, 0x4A), array_at(y, i, 0x4A)) {
                (Ok(a), Ok(b)) => {
                    let scale = self.c.plan.scale;
                    let bad = a.chunks_exact(4).zip(b.chunks_exact(4)).any(|(u, v)| {
                        let (u, v) = (f32::from_le_bytes([u[0], u[1], u[2], u[3]]), f32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                        v != if u < 0.0 { u } else { (f64::from(u) * scale) as f32 }
                    });
                    if bad || a.len() != b.len() {
                        self.violation(&at, "distance array mismatch");
                    }
                }
                _ => self.violation(&at, "distance array missing"),
            }
        }
        for i in [end + 2, end + 7] {
            match (f32_at(x, i), f32_at(y, i)) {
                (Ok(a), Ok(b)) if b == (f64::from(a) * self.c.plan.scale) as f32 => {}
                _ => self.violation(&at, "segment length mismatch"),
            }
        }
        for i in [0, 1, end + 3, end + 4, end + 5, end + 6, end + 9, end + 10] {
            if x[i] != y[i] {
                self.violation(&at, format!("segment field {i} changed"));
            }
        }
        match (&x[end + 8], &y[end + 8]) {
            (Node::Record { children: a, .. }, Node::Record { children: b, .. }) => {
                let mut p = path.to_vec();
                p.push("LINE_OF_SIGHT".into());
                self.compare_los(a, b, &p);
            }
            _ => self.violation(&at, "segment LOS missing"),
        }
        stat(&mut self.stats, "segments_checked", 1);
    }

    fn compare_los(&mut self, a: &[Node], b: &[Node], path: &[String]) {
        let at = path.join("/");
        let (Ok(fa), Ok(fb)) = (bool_at(a, 0), bool_at(b, 0)) else {
            self.violation(at, "LOS flag is not bool");
            return;
        };
        if fa != fb || a.len() != b.len() {
            self.violation(at, "LOS flag/shape changed");
            return;
        }
        if a.len() == 1 {
            return;
        }
        let (Ok(ta), Ok(tb)) = (Tree::parse(&a[3]), Tree::parse(&b[3])) else {
            self.violation(at, "LOS tree cannot be parsed");
            return;
        };
        if (ta.w, ta.h, ta.span) != (tb.w, tb.h, tb.span) {
            self.violation(at, "LOS dimensions changed");
            return;
        }
        let (Ok(a1), Ok(a2), Ok(b1), Ok(b2)) = (xy_at(a, 1), xy_at(a, 2), xy_at(b, 1), xy_at(b, 2)) else {
            self.violation(at, "LOS bounds are not xy");
            return;
        };
        let vis = &self.c.vis;
        let (w, h, span) = (ta.w, ta.h, ta.span);
        let cs = vis.cell_size(w, h);
        let (ca, cb) = (vis.cell_range(a1, a2, w, h), vis.cell_range(b1, b2, w, h));
        let wa = [ca[0].max(0), ca[1].max(0), ca[2].min(w), ca[3].min(h)];
        let wb = [cb[0].max(0), cb[1].max(0), cb[2].min(w), cb[3].min(h)];
        let (Ok(sa), Ok(sb), Ok(bb)) = (decode_window(&ta.root, span, wa), decode_window(&tb.root, span, wb), tree_bbox(&tb.root, span)) else {
            self.violation(at, "LOS tree cannot be decoded");
            return;
        };
        if let Some(t) = bb {
            if t[0] < wb[0] || t[1] < wb[1] || t[2] >= wb[2] || t[3] >= wb[3] {
                self.violation(&at, "output bits outside output bbox");
            }
        }
        if sa.count() == 0 {
            if sb.count() > 0 {
                self.violation(at, "bits appeared in an empty LOS");
            }
            return;
        }
        let plan = self.c.plan;
        let mut ok = 0u64;
        for yy in 0..sb.h {
            for xx in 0..sb.w {
                if !sb.get(xx, yy) {
                    continue;
                }
                let pt = [vis.min[0] + ((xx as i64 + wb[0]) as f64 + 0.5) * cs[0], vis.min[1] + ((yy as i64 + wb[1]) as f64 + 0.5) * cs[1]];
                let found = self.out_chart(pt);
                if found.len() != 1 {
                    self.violation(at, format!("output LOS cell {pt:?} in {} output crops", found.len()));
                    return;
                }
                let chart = found[0];
                let sp = plan.inverse_xy(pt, chart);
                let sx = ((sp[0] - vis.min[0]) / cs[0]).floor() as i64 - wa[0];
                let sy = ((sp[1] - vis.min[1]) / cs[1]).floor() as i64 - wa[1];
                let set = sx >= 0 && sy >= 0 && (sx as usize) < sa.w && (sy as usize) < sa.h && sa.get(sx as usize, sy as usize);
                if !set || !chart.crop_rects.iter().any(|r| Plan::contains(*r, sp)) {
                    self.violation(at, format!("output LOS cell {pt:?} has no set source cell"));
                    return;
                }
                ok += 1;
            }
        }
        stat(&mut self.stats, "los_cells_checked", ok);
        stat(&mut self.stats, "los_checked", 1);
    }

    fn hlci_table(root: &Node) -> Result<Vec<(u32, Vec<Node>)>> {
        let world = find_named(root, "CAI_WORLD").ok_or_else(|| anyhow!("no CAI_WORLD"))?;
        world_items(world, "CAI_WORLD_REGION_HLCIS")?
            .iter()
            .map(|item| Ok((item_id(item)?, find_record(item, "CAI_REGION_HLCI")?.clone())))
            .collect()
    }

    fn verify_hlci(&mut self, src: &Node, out: &Node) -> Result<()> {
        let (s, o) = (Self::hlci_table(src)?, Self::hlci_table(out)?);
        let deleted = &self.c.deleted_hlci_ids;
        let want: Vec<u32> = s.iter().map(|k| k.0).filter(|i| !deleted.contains(i)).collect();
        if o.iter().map(|k| k.0).ne(want.iter().copied()) {
            self.violation("CAI_REGION_HLCI", "output HLCI ids/order != source minus deleted groups");
            return Ok(());
        }
        let by_id: BTreeMap<u32, &Vec<Node>> = s.iter().map(|(i, r)| (*i, r)).collect();
        for (hid, b) in &o {
            let a = by_id[hid];
            let at = format!("HLCI[{hid}]");
            if u32_at(a, 0)? != u32_at(b, 0)? || u32_at(a, 1)? != u32_at(b, 1)? {
                self.violation(at, "region/group id changed");
                continue;
            }
            let rid = u32_at(a, 0)?;
            let key = self.c.cai_region_key.get(&rid).ok_or_else(|| anyhow!("HLCI {hid}: unknown CAI region {rid}"))?.clone();
            let (old, new) = (u32_list(a, 2)?, u32_list(b, 2)?);
            if new != self.c.contract.new_areas(&key, &old)? {
                self.violation(&at, "area list != contract");
            }
            let p = [f64::from(i32_at(b, 3)?) / Q, f64::from(i32_at(b, 4)?) / Q];
            let ori = *self.c.out_regions.index.get(&key).ok_or_else(|| anyhow!("output regions lack {key}"))?;
            if !new.iter().any(|&n| self.c.out_regions.near(ori, n as usize, p, 0.05)) {
                let sri = *self.c.src_regions.index.get(&key).ok_or_else(|| anyhow!("source regions lack {key}"))?;
                let sp = [f64::from(i32_at(a, 3)?) / Q, f64::from(i32_at(a, 4)?) / Q];
                if old.iter().any(|&n| self.c.src_regions.near(sri, n as usize, sp, 1e-3)) {
                    self.violation(&at, format!("{key}: output point not inside its output areas {new:?}"));
                } else {
                    stat(&mut self.stats, "hlci_source_point_never_inside_group", 1);
                }
            }
            stat(&mut self.stats, "hlci_checked", 1);
        }
        stat(&mut self.stats, "hlci_deleted", self.c.deleted_hlci_ids.len() as u64);
        Ok(())
    }

    /// Every output area of every region must belong to an HLCI group (areas uncovered in the source stay exempt).
    fn verify_group_coverage(&mut self, src: &Node, out: &Node) -> Result<()> {
        let covers = |root: &Node| -> Result<BTreeMap<String, BTreeSet<u32>>> {
            let mut cov: BTreeMap<String, BTreeSet<u32>> = BTreeMap::new();
            for (_, rec) in Self::hlci_table(root)? {
                let rid = u32_at(&rec, 0)?;
                let key = self.c.cai_region_key.get(&rid).ok_or_else(|| anyhow!("unknown CAI region {rid}"))?;
                cov.entry(key.clone()).or_default().extend(u32_list(&rec, 2)?);
            }
            Ok(cov)
        };
        let (sc, oc) = (covers(src)?, covers(out)?);
        let empty = BTreeSet::new();
        for r in &self.c.contract.regions {
            let src_cov = sc.get(&r.key).unwrap_or(&empty);
            let out_cov = oc.get(&r.key).unwrap_or(&empty);
            let src_unc: BTreeSet<u32> = (0..r.source_areas as u32).filter(|i| !src_cov.contains(i)).collect();
            for j in 0..r.output_areas as u32 {
                if out_cov.contains(&j) {
                    continue;
                }
                let origin = r.area_source.get(j as usize).copied().flatten();
                if origin.is_some_and(|s| src_unc.contains(&s)) {
                    stat(&mut self.stats, "areas_uncovered_in_baseline_too", 1);
                    continue;
                }
                self.violations.push((format!("HLCI coverage {}", r.key), format!("output area {j} (source area {origin:?}) belongs to no HLCI group")));
            }
            stat(&mut self.stats, "areas_checked", u64::from(r.output_areas as u32));
        }
        Ok(())
    }
}
