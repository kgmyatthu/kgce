//! Vector display assets of the connected atlas: borders, roads, rivers, trade routes, coastline meshes, campaign trees,
//! bridge markers and (when present) `world.markers`. Port of `vector_scale.py`: horizontal positions move with the
//! chart affine, heights/orientation/sizes are preserved, everything is clipped exactly to the chart crops (pieces are
//! moved by the chart they fall into so nothing is stretched across relocated charts), decorations outside every chart
//! are dropped and recorded, unknown or unsupported input is refused.

mod coast;
mod geo;
mod markers;

use crate::{
    atlas::Plan,
    model::Progress,
    terrain::formats::{self as F, Spline},
};
use anyhow::{Context, Result, bail, ensure};
use geo::Geo;
use regex::Regex;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const SPLINE_UNIT: f64 = 39.37;
const MIN_POINTS: usize = 4;
const MAX_NUMBERED: usize = 14;

type Files = BTreeMap<String, Vec<u8>>;

#[derive(Default)]
struct SplineStats {
    source_files: usize,
    source_splines: usize,
    input_points: usize,
    output_points: usize,
    output_files: usize,
    dropped_short_pieces: usize,
    dropped_short_points: usize,
    fully_outside: Vec<String>,
    split: Vec<String>,
    per_chart: BTreeMap<String, usize>,
    renamed: Vec<Value>,
    dropped_files: Vec<String>,
    verified_points: usize,
    max_inverse_polyline_distance: f64,
}

impl SplineStats {
    fn report(&self) -> Value {
        json!({
            "source_files": self.source_files,
            "source_splines": self.source_splines,
            "input_points": self.input_points,
            "output_points": self.output_points,
            "output_files": self.output_files,
            "dropped_short_pieces": self.dropped_short_pieces,
            "dropped_short_points": self.dropped_short_points,
            "fully_outside_splines": self.fully_outside,
            "split_splines": self.split,
            "pieces_per_chart": self.per_chart,
            "renamed_files": self.renamed,
            "dropped_files": self.dropped_files,
            "verified_points": self.verified_points,
            "max_inverse_polyline_distance": self.max_inverse_polyline_distance,
        })
    }
}

struct Job<'a> {
    geo: Geo,
    disp: &'a Path,
    files: Files,
}

impl Job<'_> {
    fn put(&mut self, rel: String, data: Vec<u8>) -> Result<()> {
        ensure!(self.files.insert(rel.clone(), data).is_none(), "duplicate output {rel}");
        Ok(())
    }

    /// Strictly list a vector folder: every file must match `pattern`, directories only from `allow_dirs`.
    fn list_dir(&self, sub: &str, pattern: &str, allow_dirs: &[&str]) -> Result<Option<Vec<String>>> {
        let d = self.disp.join(sub);
        if !d.is_dir() {
            return Ok(None);
        }
        let re = Regex::new(&format!("^(?:{pattern})$"))?;
        let mut names = Vec::new();
        let mut entries: Vec<_> = fs::read_dir(&d)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.file_type()?.is_dir() {
                if allow_dirs.contains(&name.as_str()) {
                    continue;
                }
                bail!("{sub}/{name}: unexpected directory in vector folder");
            }
            ensure!(re.is_match(&name), "{sub}/{name}: unknown file in vector folder (unsupported, refusing to skip)");
            names.push(name);
        }
        Ok(Some(names))
    }

    /// One source spline -> pieces `(chart key, points)` in traversal order.
    fn transform_spline(&self, name: &str, pts: &[[f32; 3]], stats: &mut SplineStats) -> Vec<(String, Vec<[f32; 3]>)> {
        let geo = &self.geo;
        let xy: Vec<[f64; 2]> = pts.iter().map(|p| [p[0] as f64 * SPLINE_UNIT, p[2] as f64 * SPLINE_UNIT]).collect();
        let ex: Vec<f64> = pts.iter().map(|p| p[1] as f64).collect();
        stats.input_points += pts.len();
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in &xy {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let mut res: Vec<((usize, f64), usize, Vec<[f32; 3]>)> = Vec::new();
        for (ci, chart) in geo.charts.iter().enumerate() {
            if !chart.rects.iter().any(|r| hi[0] >= r[0][0] && hi[1] >= r[0][1] && lo[0] <= r[1][0] && lo[1] <= r[1][1]) {
                continue;
            }
            for piece in geo.clip_polyline(ci, &xy, &ex) {
                if piece.xy.len() < MIN_POINTS {
                    stats.dropped_short_pieces += 1;
                    stats.dropped_short_points += piece.xy.len();
                    continue;
                }
                let out: Vec<[f32; 3]> = piece
                    .xy
                    .iter()
                    .zip(&piece.ex)
                    .map(|(p, &h)| {
                        [
                            ((p[0] * geo.scale + chart.offset[0]) / SPLINE_UNIT) as f32,
                            h as f32,
                            ((p[1] * geo.scale + chart.offset[1]) / SPLINE_UNIT) as f32,
                        ]
                    })
                    .collect();
                *stats.per_chart.entry(chart.key.clone()).or_default() += 1;
                stats.output_points += piece.xy.len();
                res.push((piece.start, ci, out));
            }
        }
        res.sort_by(|a, b| a.0.0.cmp(&b.0.0).then(a.0.1.partial_cmp(&b.0.1).unwrap_or(std::cmp::Ordering::Equal)).then(a.1.cmp(&b.1)));
        if res.is_empty() {
            stats.fully_outside.push(name.to_string());
        } else if res.len() > 1 {
            stats.split.push(name.to_string());
        }
        res.into_iter().map(|(_, ci, p)| (geo.charts[ci].key.clone(), p)).collect()
    }

    fn read_spline_file(&self, sub: &str, fname: &str) -> Result<(u32, Vec<Spline>)> {
        let path = self.disp.join(sub).join(fname);
        let (ver, sp) = F::parse_spln(&fs::read(&path).with_context(|| format!("Read {}", path.display()))?)
            .with_context(|| format!("{sub}/{fname}"))?;
        ensure!(ver == 1, "{sub}/{fname}: unsupported SPLN version {ver}");
        Ok((ver, sp))
    }

    fn do_spline_series(&mut self, sub: &str) -> Result<Option<SplineStats>> {
        let Some(names) = self.list_dir(sub, r"[^/\\]+\.rigid_spline", &[])? else { return Ok(None) };
        let stems: Vec<String> = names.iter().map(|n| n.trim_end_matches(".rigid_spline").to_string()).collect();
        let stemset: BTreeSet<&str> = stems.iter().map(String::as_str).collect();
        let numbered = Regex::new(r"^(.+)_(\d+)$")?;
        // key -> (plain, members by number); BTreeMap keeps the source's sorted order
        let mut groups: BTreeMap<String, (bool, BTreeMap<u64, String>)> = BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for stem in &stems {
            let series = numbered
                .captures(stem)
                .filter(|c| stemset.contains(format!("{}_1", &c[1]).as_str()))
                .map(|c| (c[1].to_string(), c[2].parse::<u64>()));
            let (key, n, plain) = match series {
                Some((k, Ok(n))) => (k, n, false),
                _ => (stem.clone(), 0, true),
            };
            if !groups.contains_key(&key) {
                order.push(key.clone());
            }
            let g = groups.entry(key).or_insert((plain, BTreeMap::new()));
            g.1.insert(n, stem.clone());
        }
        for key in &order {
            let (plain, members) = &groups[key];
            if !plain {
                let ns: Vec<u64> = members.keys().copied().collect();
                ensure!(
                    ns == (1..=ns.len() as u64).collect::<Vec<_>>(),
                    "{sub}/{key}: source numbered files are not contiguous from 1: {ns:?}"
                );
            }
        }
        let mut stats = SplineStats::default();
        for key in &order {
            let (plain, members) = groups[key].clone();
            let mut pieces: Vec<(String, u32, Vec<[f32; 3]>)> = Vec::new();
            for stem in members.values() {
                let fname = format!("{stem}.rigid_spline");
                let (_, sp) = self.read_spline_file(sub, &fname)?;
                ensure!(sp.len() == 1, "{sub}/{fname}: {} splines in file (series files must hold exactly one)", sp.len());
                stats.source_files += 1;
                stats.source_splines += 1;
                let s = &sp[0];
                ensure!(s.points.len() >= 2, "{sub}/{fname}: spline with {} points", s.points.len());
                for (_chart, op) in self.transform_spline(&s.name, &s.points, &mut stats) {
                    pieces.push((s.name.clone(), s.flag, op));
                }
            }
            let src_names: Vec<String> = members.values().map(|s| format!("{s}.rigid_spline")).collect();
            if pieces.is_empty() {
                stats.dropped_files.extend(src_names);
                continue;
            }
            let out_names: Vec<String> = if pieces.len() == 1 && plain {
                vec![format!("{key}.rigid_spline")]
            } else {
                ensure!(pieces.len() <= MAX_NUMBERED, "{sub}/{key}: {} pieces exceed the loader's {MAX_NUMBERED}-file series", pieces.len());
                (1..=pieces.len()).map(|i| format!("{key}_{i}.rigid_spline")).collect()
            };
            if out_names != src_names {
                stats.renamed.push(json!({"source": src_names, "output": out_names}));
            }
            for (on, (name, flag, pts)) in out_names.into_iter().zip(pieces) {
                self.put(format!("{sub}/{on}"), F::write_spln(1, &[Spline { name, flag, points: pts }]))?;
                stats.output_files += 1;
            }
        }
        Ok(Some(stats))
    }

    fn do_rivers(&mut self) -> Result<Option<SplineStats>> {
        let Some(names) = self.list_dir("rivers", r"\d\d\.rigid_spline", &["textures"])? else { return Ok(None) };
        let nums: Vec<usize> = names.iter().map(|n| n[..2].parse().unwrap()).collect();
        ensure!(
            nums.iter().copied().eq(1..=nums.len()),
            "rivers: source numbering not contiguous from 01: {nums:?}"
        );
        let mut stats = SplineStats { source_files: names.len(), ..Default::default() };
        let mut pieces: Vec<(usize, Spline)> = Vec::new();
        for n in &nums {
            let fname = format!("{n:02}.rigid_spline");
            let (_, sp) = self.read_spline_file("rivers", &fname)?;
            ensure!(sp.len() == 1, "rivers/{fname}: {} splines in file", sp.len());
            stats.source_splines += 1;
            let s = &sp[0];
            for (_chart, op) in self.transform_spline(&s.name, &s.points, &mut stats) {
                pieces.push((*n, Spline { name: s.name.clone(), flag: s.flag, points: op }));
            }
        }
        ensure!(pieces.len() <= 99, "rivers: more than 99 output pieces (loader loops %02d)");
        if !pieces.iter().map(|p| p.0).eq(1..=pieces.len()) {
            stats.renamed.push(json!({
                "source_numbers": pieces.iter().map(|p| p.0).collect::<Vec<_>>(),
                "output_numbers": (1..=pieces.len()).collect::<Vec<_>>(),
            }));
        }
        for (i, (_, s)) in pieces.into_iter().enumerate() {
            self.put(format!("rivers/{:02}.rigid_spline", i + 1), F::write_spln(1, &[s]))?;
            stats.output_files += 1;
        }
        Ok(Some(stats))
    }

    fn do_traderoutes(&mut self) -> Result<Option<Value>> {
        let Some(names) = self.list_dir("traderoutes", r"[^/\\]+\.rigid_spline", &[])? else { return Ok(None) };
        let mut files = serde_json::Map::new();
        for fname in &names {
            let path = self.disp.join("traderoutes").join(fname);
            let (ver, sp) = F::parse_spln(&fs::read(&path)?).with_context(|| format!("traderoutes/{fname}"))?;
            let mut st = SplineStats { source_files: 1, source_splines: sp.len(), ..Default::default() };
            let mut out = Vec::new();
            let mut mapping = Vec::new();
            for (si, s) in sp.iter().enumerate() {
                ensure!(s.points.len() >= 2, "traderoutes/{fname}[{si}]: spline with {} points", s.points.len());
                let outp = self.transform_spline(&s.name, &s.points, &mut st);
                mapping.push(json!({"source_index": si, "name": s.name, "outputs": (out.len()..out.len() + outp.len()).collect::<Vec<_>>()}));
                for (_chart, op) in outp {
                    out.push(Spline { name: s.name.clone(), flag: s.flag, points: op });
                }
            }
            ensure!(!out.is_empty(), "traderoutes/{fname}: no spline survives the crop");
            st.output_files = 1;
            let mut rep = st.report();
            rep["output_splines"] = json!(out.len());
            rep["index_mapping"] = json!(mapping);
            self.put(format!("traderoutes/{fname}"), F::write_spln(ver, &out))?;
            files.insert(fname.clone(), rep);
        }
        Ok(Some(json!({"source_files": names.len(), "files": files})))
    }

    fn do_coast(&mut self) -> Result<Option<Value>> {
        let Some(names) = self.list_dir("coastline", r"coastline_group\d+\.rigid_mesh", &[])? else { return Ok(None) };
        let re = Regex::new(r"(\d+)")?;
        let mut nums: Vec<usize> = names.iter().map(|n| re.captures(n).unwrap()[1].parse().unwrap()).collect();
        nums.sort_unstable();
        ensure!(nums.iter().copied().eq(0..nums.len()), "coastline: source groups not contiguous from 0: {nums:?}");
        let (mut out_index, mut renumber) = (0usize, Vec::new());
        let (mut sv, mut st_, mut ov, mut ot) = (0usize, 0usize, 0usize, 0usize);
        let (mut src_area, mut clip_area) = (0.0f64, 0.0f64);
        let (mut clipped, mut dropped) = (0usize, 0usize);
        let mut per_group = serde_json::Map::new();
        let mut dropped_groups = Vec::new();
        for gi in nums {
            let fname = format!("coastline_group{gi}.rigid_mesh");
            let path = self.disp.join("coastline").join(&fname);
            let m = coast::parse(&fs::read(&path).with_context(|| format!("Read {}", path.display()))?).with_context(|| fname.clone())?;
            coast::check_layout(&m, &fname, true)?;
            let g = coast::clip_group(&self.geo, &m);
            sv += m.verts.len();
            st_ += m.idx.len() / 3;
            src_area += g.source_area;
            clip_area += g.clipped_area;
            clipped += g.clipped_triangles;
            dropped += g.dropped_triangles;
            per_group.insert(fname.clone(), g.stats.clone());
            if g.tris.is_empty() {
                dropped_groups.push(fname);
                continue;
            }
            let idx: Vec<u32> = g.tris.iter().flatten().map(|&i| i as u32).collect();
            let data = coast::build(&m.header, &g.verts, &idx);
            self.put(format!("coastline/coastline_group{out_index}.rigid_mesh"), data)?;
            renumber.push(json!([gi, out_index]));
            out_index += 1;
            ov += g.verts.len();
            ot += g.tris.len();
        }
        ensure!(out_index > 0, "coastline: no group survives the crop (group 0 is required)");
        Ok(Some(json!({
            "source_groups": renumber.len() + dropped_groups.len(),
            "output_groups": out_index,
            "source_vertices": sv,
            "source_triangles": st_,
            "output_vertices": ov,
            "output_triangles": ot,
            "clipped_triangles": clipped,
            "dropped_triangles": dropped,
            "source_area": src_area,
            "output_area_over_scale2": clip_area,
            "group_renumber": renumber,
            "dropped_groups": dropped_groups,
            "per_group": per_group,
        })))
    }

    fn do_trees(&mut self) -> Result<Option<Value>> {
        let Some(_) = self.list_dir("trees", r"campaign\.rigid_trees", &[])? else { return Ok(None) };
        let b = fs::read(self.disp.join("trees/campaign.rigid_trees"))?;
        let mut t = trees::parse(&b)?;
        ensure!(trees::write(&t) == b, "rigid_trees strict parse/write does not round-trip the source bytes");
        let geo = &self.geo;
        let ties0 = geo.ties.get();
        let (mut source, mut kept, mut dropped) = (0usize, 0usize, 0usize);
        let mut per_chart: BTreeMap<String, usize> = BTreeMap::new();
        let (mut placeholder, mut recomputed) = (0usize, 0usize);
        for m in &mut t.models {
            let mut new: Vec<[f32; 4]> = Vec::with_capacity(m.inst.len());
            for row in &m.inst {
                let (x, y) = (row[0] as f64, row[2] as f64);
                let Some(ci) = geo.chart_of_point(x, y) else {
                    dropped += 1;
                    continue;
                };
                let (nx, nz) = geo.xy(x, y, ci);
                new.push([nx as f32, row[1], nz as f32, row[3]]);
                *per_chart.entry(geo.charts[ci].key.clone()).or_default() += 1;
            }
            source += m.inst.len();
            kept += new.len();
            if m.bbox.iter().map(|&v| (v as f64 * 1e6).round() / 1e6).eq([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]) {
                placeholder += 1;
            } else {
                m.bbox = trees::recompute_bbox(&m.name, &m.bbox, &m.inst, &new)?;
                recomputed += 1;
            }
            m.inst = new;
        }
        let ties = geo.ties.get() - ties0;
        let data = trees::write(&t);
        trees::parse(&data)?;
        self.put("trees/campaign.rigid_trees".into(), data)?;
        Ok(Some(json!({
            "models": t.models.len(),
            "source_instances": source,
            "kept_instances": kept,
            "dropped_outside": dropped,
            "per_chart": per_chart,
            "boundary_ties": ties,
            "bbox": {"placeholder": placeholder, "recomputed": recomputed},
        })))
    }

    fn do_bridges(&mut self) -> Result<Option<Value>> {
        let Some(_) = self.list_dir("bridges", r"bridge\.markers", &[])? else { return Ok(None) };
        let b = fs::read(self.disp.join("bridges/bridge.markers"))?;
        let items = trees::parse_bridges(&b)?;
        ensure!(trees::write_bridges(&items) == b, "bridge.markers strict parse/write does not round-trip the source bytes");
        let geo = &self.geo;
        let (mut out, mut dropped) = (Vec::new(), Vec::new());
        let mut per_chart: BTreeMap<String, usize> = BTreeMap::new();
        for (name, m) in &items {
            ensure!(
                m[0][3] == 0.0 && m[1][3] == 0.0 && m[2][3] == 0.0 && m[3][3] == 1.0,
                "bridge {name}: last matrix column is not [0,0,0,1] (not a row-vector 4x4)"
            );
            let r = |i: usize| [m[i][0] as f64, m[i][1] as f64, m[i][2] as f64];
            let (r0, r1, r2) = (r(0), r(1), r(2));
            let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dot = r0[0] * r2[0] + r0[1] * r2[1] + r0[2] * r2[2];
            ensure!(
                (norm(r0) - 1.0).abs() < 1e-3
                    && (norm(r2) - 1.0).abs() < 1e-3
                    && dot.abs() < 1e-3
                    && (r1[1] - 1.0).abs() < 1e-6
                    && r1[0].abs() < 1e-6
                    && r1[2].abs() < 1e-6
                    && r0[1].abs() < 1e-6
                    && r2[1].abs() < 1e-6,
                "bridge {name}: matrix is not a pure yaw rotation (axis/convention assumption broken)"
            );
            let (x, y) = (m[3][0] as f64 * SPLINE_UNIT, m[3][2] as f64 * SPLINE_UNIT);
            let Some(ci) = geo.chart_of_point(x, y) else {
                dropped.push(name.clone());
                continue;
            };
            let (nx, ny) = geo.xy(x, y, ci);
            // Buildings keep their 1x footprint: only the translation moves (the accepted bridge span scale is 1.0).
            let mut nm = *m;
            nm[3][0] = (nx / SPLINE_UNIT) as f32;
            nm[3][2] = (ny / SPLINE_UNIT) as f32;
            out.push((name.clone(), nm));
            *per_chart.entry(geo.charts[ci].key.clone()).or_default() += 1;
        }
        let data = trees::write_bridges(&out);
        trees::parse_bridges(&data)?;
        let kept = out.len();
        self.put("bridges/bridge.markers".into(), data)?;
        Ok(Some(json!({"source": items.len(), "kept": kept, "dropped_outside": dropped, "span_scale": 1.0, "per_chart": per_chart})))
    }

    fn do_world_markers(&mut self) -> Result<Option<Value>> {
        let path = self.disp.join("world.markers");
        if !path.is_file() {
            return Ok(None);
        }
        let b = fs::read(&path)?;
        let w = markers::parse(&b)?;
        ensure!(markers::write(&w) == b, "world.markers strict parse/write does not round-trip the source bytes");
        let (out, rep) = markers::transform(&w, &self.geo)?;
        let data = markers::write(&out);
        markers::parse(&data)?;
        self.put("world.markers".into(), data)?;
        Ok(Some(rep))
    }

    /// Output points lie inside a transformed crop and (inverse-mapped) on a source polyline of the same name/flag,
    /// with heights inside the source's range.
    fn verify_splines(&self, sub: &str, stats: &mut SplineStats, source: &[Spline]) -> Result<()> {
        let mut by_name: BTreeMap<(&str, u32), Vec<&Spline>> = BTreeMap::new();
        for s in source {
            by_name.entry((s.name.as_str(), s.flag)).or_default().push(s);
        }
        let prefix = format!("{sub}/");
        let (mut worst, mut npts) = (0.0f64, 0usize);
        for (rel, data) in self.files.range(prefix.clone()..) {
            if !rel.starts_with(&prefix) {
                break;
            }
            if !rel.ends_with(".rigid_spline") {
                continue;
            }
            let (_, sp) = F::parse_spln(data)?;
            for s in sp {
                let cands = by_name.get(&(s.name.as_str(), s.flag)).with_context(|| format!("{rel}: no source spline named {}", s.name))?;
                let world: Vec<[f64; 2]> = s.points.iter().map(|p| [p[0] as f64 * SPLINE_UNIT, p[2] as f64 * SPLINE_UNIT]).collect();
                for (k, p) in world.iter().enumerate() {
                    ensure!(
                        !self.geo.inverse_chart_of_output(p[0], p[1], 1e-3).is_empty(),
                        "{rel}[{k}] outside every transformed crop"
                    );
                }
                let Some(&ci) = self.geo.inverse_chart_of_output(world[0][0], world[0][1], 1e-3).first() else { continue };
                let c = &self.geo.charts[ci];
                let inv: Vec<[f64; 2]> = world
                    .iter()
                    .map(|p| [(p[0] - c.offset[0]) / self.geo.scale, (p[1] - c.offset[1]) / self.geo.scale])
                    .collect();
                let mut best = f64::INFINITY;
                for cand in cands {
                    let poly: Vec<[f64; 2]> = cand.points.iter().map(|p| [p[0] as f64 * SPLINE_UNIT, p[2] as f64 * SPLINE_UNIT]).collect();
                    let d = inv.iter().map(|p| point_polyline_dist(*p, &poly)).fold(0.0f64, f64::max);
                    best = best.min(d);
                    let (hmin, hmax) = cand
                        .points
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1] as f64), a.1.max(p[1] as f64)));
                    if d <= 2e-3 {
                        ensure!(
                            s.points.iter().all(|p| (p[1] as f64) >= hmin - 1e-6 && (p[1] as f64) <= hmax + 1e-6),
                            "{rel}: heights leave the source range"
                        );
                    }
                }
                ensure!(best <= 2e-3, "{rel}: inverse-mapped points are {best} world units off every source polyline of {}", s.name);
                worst = worst.max(best);
                npts += s.points.len();
            }
        }
        stats.verified_points = npts;
        stats.max_inverse_polyline_distance = worst;
        Ok(())
    }

    fn source_splines(&self, sub: &str) -> Result<Vec<Spline>> {
        let mut all = Vec::new();
        let d = self.disp.join(sub);
        if !d.is_dir() {
            return Ok(all);
        }
        let mut names: Vec<_> = fs::read_dir(&d)?.collect::<std::io::Result<Vec<_>>>()?;
        names.sort_by_key(|e| e.file_name());
        for e in names {
            if e.file_name().to_string_lossy().ends_with(".rigid_spline") {
                all.extend(F::parse_spln(&fs::read(e.path())?)?.1);
            }
        }
        Ok(all)
    }
}

fn point_polyline_dist(p: [f64; 2], poly: &[[f64; 2]]) -> f64 {
    let mut best = f64::INFINITY;
    for w in poly.windows(2) {
        let (a, b) = (w[0], w[1]);
        let ab = [b[0] - a[0], b[1] - a[1]];
        let l2 = (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-18);
        let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0);
        let (dx, dy) = (a[0] + t * ab[0] - p[0], a[1] + t * ab[1] - p[1]);
        best = best.min((dx * dx + dy * dy).sqrt());
    }
    best
}

/// Transform every vector asset below `<map_dir>/display` in place (old series files of the handled folders are replaced).
pub fn transform(map_dir: &Path, plan: &Plan, progress: &Progress<'_>) -> Result<Value> {
    let disp = map_dir.join("display");
    ensure!(disp.is_dir(), "{} not found", disp.display());
    let mut job = Job { geo: Geo::new(plan)?, disp: &disp, files: Files::new() };
    progress("Atlas display: borders, roads, rivers and trade routes");
    let mut borders = job.do_spline_series("borders")?;
    let mut roads = job.do_spline_series("roads")?;
    let mut rivers = job.do_rivers()?;
    let traderoutes = job.do_traderoutes()?;
    progress("Atlas display: coastline, trees and bridges");
    let mut coast = job.do_coast()?.context("required vector asset coastline missing from source")?;
    let trees = job.do_trees()?;
    let bridges = job.do_bridges()?;
    let world_markers = job.do_world_markers()?;
    for (sub, stats) in [("borders", &mut borders), ("roads", &mut roads), ("rivers", &mut rivers)] {
        if let Some(stats) = stats {
            let src = job.source_splines(sub)?;
            job.verify_splines(sub, stats, &src)?;
        }
    }
    // clipped-area invariant of the coastline: output area / scale^2 equals the clipped source area
    let (mut out_area, mut degenerate_triangles) = (0.0f64, 0usize);
    for (rel, data) in &job.files {
        if rel.starts_with("coastline/") {
            let (area, degenerate) = coast::verify_group(&job.geo, data, rel)?;
            out_area += area;
            degenerate_triangles += degenerate;
        }
    }
    coast["degenerate_output_triangles"] = json!(degenerate_triangles);
    let expect = coast["output_area_over_scale2"].as_f64().unwrap_or(0.0);
    let ratio = out_area / (plan.scale * plan.scale) / expect;
    ensure!((ratio - 1.0).abs() < 1e-4, "coastline: output area / scale^2 differs from the clipped source area (ratio {ratio})");
    coast["output_area_over_scale2_reparsed"] = json!(out_area / (plan.scale * plan.scale));

    // Commit: replace the old series of every handled folder, then write the new files.
    for (sub, ext) in [("borders", ".rigid_spline"), ("roads", ".rigid_spline"), ("rivers", ".rigid_spline"), ("traderoutes", ".rigid_spline"), ("coastline", ".rigid_mesh")] {
        let d = disp.join(sub);
        if d.is_dir() {
            for e in fs::read_dir(&d)? {
                let e = e?;
                if e.file_type()?.is_file() && e.file_name().to_string_lossy().ends_with(ext) {
                    fs::remove_file(e.path())?;
                }
            }
        }
    }
    let count = job.files.len();
    let tie_points = job.geo.ties.get();
    for (rel, data) in job.files {
        let p = disp.join(&rel);
        fs::create_dir_all(p.parent().unwrap())?;
        fs::write(&p, data).with_context(|| format!("Write {}", p.display()))?;
    }
    Ok(json!({
        "files_written": count,
        "boundary_tie_points": tie_points,
        "borders": borders.as_ref().map(SplineStats::report),
        "roads": roads.as_ref().map(SplineStats::report),
        "rivers": rivers.as_ref().map(SplineStats::report),
        "traderoutes": traderoutes,
        "coastline": coast,
        "trees": trees,
        "bridges": bridges,
        "world.markers": world_markers,
        "world_markers_note": if world_markers.is_none() { "source has no active world.markers" } else { "transformed" },
    }))
}

/// `rigid_trees` and `bridge.markers` formats.
mod trees {
    use anyhow::{Context, Result, bail, ensure};

    pub struct Model {
        pub name: String,
        pub bbox: [f32; 6],
        pub inst: Vec<[f32; 4]>,
    }

    pub struct Trees {
        pub magic: u32,
        pub ver: u32,
        pub cell: f32,
        pub n6: u32,
        pub models: Vec<Model>,
    }

    fn string(b: &[u8], p: usize) -> Result<(String, usize)> {
        let n = u16::from_le_bytes(b.get(p..p + 2).context("truncated")?.try_into()?) as usize;
        let raw = b.get(p + 2..p + 2 + n * 2).context("truncated")?;
        let u: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok((String::from_utf16(&u)?, p + 2 + n * 2))
    }

    fn push_string(out: &mut Vec<u8>, s: &str) {
        let u: Vec<u16> = s.encode_utf16().collect();
        out.extend_from_slice(&(u.len() as u16).to_le_bytes());
        for c in u {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }

    fn f32s<const N: usize>(b: &[u8], p: usize) -> Result<[f32; N]> {
        let raw = b.get(p..p + 4 * N).context("truncated")?;
        Ok(std::array::from_fn(|i| f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap())))
    }

    pub fn parse(b: &[u8]) -> Result<Trees> {
        ensure!(b.len() >= 20, "rigid_trees truncated");
        let u = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        let (magic, ver, cell, n6, nmod) = (u(0), u(4), f32::from_le_bytes(b[8..12].try_into().unwrap()), u(12), u(16));
        ensure!(magic == 0x3D4D_4047 && ver == 1 && n6 == 6, "unsupported rigid_trees header");
        let mut p = 20;
        let mut models = Vec::new();
        for _ in 0..nmod {
            let (name, q) = string(b, p)?;
            let head = b.get(q..q + 8).context("truncated")?;
            let (u1, cnt) = (u32::from_le_bytes(head[..4].try_into()?), u32::from_le_bytes(head[4..].try_into()?) as usize);
            ensure!(u1 == 1, "rigid_trees model {name}: unexpected u32 {u1}");
            let bbox = f32s::<6>(b, q + 8)?;
            let q = q + 8 + 24;
            ensure!(q + cnt * 16 <= b.len(), "rigid_trees truncated");
            let inst = (0..cnt).map(|i| f32s::<4>(b, q + i * 16)).collect::<Result<Vec<_>>>()?;
            models.push(Model { name, bbox, inst });
            p = q + cnt * 16;
        }
        ensure!(p == b.len(), "rigid_trees trailing bytes {p} != {}", b.len());
        Ok(Trees { magic, ver, cell, n6, models })
    }

    pub fn write(t: &Trees) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&t.magic.to_le_bytes());
        out.extend_from_slice(&t.ver.to_le_bytes());
        out.extend_from_slice(&t.cell.to_le_bytes());
        out.extend_from_slice(&t.n6.to_le_bytes());
        out.extend_from_slice(&(t.models.len() as u32).to_le_bytes());
        for m in &t.models {
            push_string(&mut out, &m.name);
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&(m.inst.len() as u32).to_le_bytes());
            for v in m.bbox {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for i in &m.inst {
                for v in i {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// Non-placeholder bounding boxes must match `[min x, min c1, min worldY, max x, max c1, max worldY]` of the
    /// instances; anything else is refused. Moves with the instances.
    pub fn recompute_bbox(name: &str, bb: &[f32; 6], inst: &[[f32; 4]], new: &[[f32; 4]]) -> Result<[f32; 6]> {
        if inst.is_empty() {
            bail!("trees {name}: non-placeholder bbox on an empty model");
        }
        let ext = |rows: &[[f32; 4]]| {
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for r in rows {
                for k in 0..3 {
                    lo[k] = lo[k].min(r[k]);
                    hi[k] = hi[k].max(r[k]);
                }
            }
            [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
        };
        let layout = ext(inst);
        ensure!(
            layout.iter().zip(bb).all(|(a, b)| (*a as f64 - *b as f64).abs() <= 1e-3 + 1e-5 * (*b as f64).abs()),
            "trees {name}: bbox {bb:?} matches no known layout of instance extents {layout:?}"
        );
        Ok(if new.is_empty() { *bb } else { ext(new) })
    }

    pub fn parse_bridges(b: &[u8]) -> Result<Vec<(String, [[f32; 4]; 4])>> {
        let (mut p, mut out) = (0usize, Vec::new());
        while p < b.len() {
            let (name, q) = string(b, p)?;
            ensure!(q + 64 <= b.len(), "bridge.markers truncated");
            let flat = f32s::<16>(b, q)?;
            out.push((name, std::array::from_fn(|r| std::array::from_fn(|c| flat[r * 4 + c]))));
            p = q + 64;
        }
        ensure!(p == b.len(), "bridge.markers trailing bytes");
        Ok(out)
    }

    pub fn write_bridges(items: &[(String, [[f32; 4]; 4])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, m) in items {
            push_string(&mut out, name);
            for row in m {
                for v in row {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn trees_and_bridges_round_trip_and_reject_unknown_bbox_layouts() {
            let t = Trees {
                magic: 0x3D4D_4047,
                ver: 1,
                cell: 1024.0,
                n6: 6,
                models: vec![Model { name: "oak".into(), bbox: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], inst: vec![[1.0, 2.0, 3.0, 4.0]] }],
            };
            let b = write(&t);
            assert_eq!(write(&parse(&b).unwrap()), b);
            let inst = [[1.0, 5.0, 3.0, 1.0]];
            assert!(recompute_bbox("x", &[1.0, 5.0, 3.0, 1.0, 5.0, 3.0], &inst, &inst).is_ok());
            assert!(recompute_bbox("x", &[9.0, 5.0, 3.0, 1.0, 5.0, 3.0], &inst, &inst).is_err());
            let br = vec![("b".to_string(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [5.0, 6.0, 7.0, 1.0]])];
            assert_eq!(parse_bridges(&write_bridges(&br)).unwrap(), br);
        }
    }
}
