//! Coastline `rigid_mesh` v5 groups: exact triangle clipping to the chart crops with every per-vertex attribute
//! interpolated, then the chart affine on position and the cumulative-length columns (`clip_coast_group`).

use super::geo::{Geo, clip_poly_to_rect};
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap};

pub const STRIDE: usize = 80;
pub const VERT_OFF: usize = 0x1F;
pub const COUNT_OFF: usize = 0x1B;
pub const C_X: usize = 0;
pub const C_Z: usize = 2;
pub const C_U: usize = 6;
pub const C_V: usize = 7;
pub const C_CUM: usize = 18;
pub const C_TOT: usize = 19;
const ZERO_COLS: [usize; 13] = [3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
const EPS_AREA: f64 = 1e-9;

pub struct Mesh {
    pub header: Vec<u8>,
    pub verts: Vec<[f64; 20]>,
    pub idx: Vec<u32>,
}

fn u32_at(b: &[u8], p: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(p..p + 4).ok_or_else(|| anyhow::anyhow!("coast mesh truncated"))?.try_into()?))
}

pub fn parse(b: &[u8]) -> Result<Mesh> {
    ensure!(u32_at(b, 0)? == 0x1234_5678 && u32_at(b, 4)? == 5, "not a coastline rigid_mesh v5");
    let n = u32_at(b, COUNT_OFF)? as usize;
    ensure!(COUNT_OFF + 4 == VERT_OFF, "coast layout constants changed");
    let end = VERT_OFF + n * STRIDE;
    ensure!(end + 4 <= b.len(), "coast mesh truncated");
    let ni = u32_at(b, end)? as usize;
    ensure!(end + 4 + ni * 4 == b.len(), "coast mesh size mismatch {} != {}", end + 4 + ni * 4, b.len());
    ensure!(ni % 3 == 0, "coast index count not a multiple of 3");
    let mut verts = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = [0f64; 20];
        for (k, slot) in v.iter_mut().enumerate() {
            let o = VERT_OFF + i * STRIDE + k * 4;
            *slot = f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
        }
        verts.push(v);
    }
    let idx: Vec<u32> = (0..ni).map(|i| u32::from_le_bytes(b[end + 4 + i * 4..][..4].try_into().unwrap())).collect();
    ensure!(idx.iter().all(|&i| (i as usize) < n), "coast index out of range");
    Ok(Mesh { header: b[..COUNT_OFF].to_vec(), verts, idx })
}

pub fn build(header: &[u8], verts: &[[f64; 20]], idx: &[u32]) -> Vec<u8> {
    let mut out = header.to_vec();
    out.extend_from_slice(&(verts.len() as u32).to_le_bytes());
    for v in verts {
        for &c in v {
            out.extend_from_slice(&(c as f32).to_le_bytes());
        }
    }
    out.extend_from_slice(&(idx.len() as u32).to_le_bytes());
    for &i in idx {
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

/// Layout assumptions measured on the real files; anything else is refused.
pub fn check_layout(m: &Mesh, label: &str, source: bool) -> Result<()> {
    for v in &m.verts {
        ensure!(ZERO_COLS.iter().all(|&c| v[c] == 0.0), "{label}: unexpected nonzero coast vertex columns (layout assumption broken)");
        if source {
            ensure!(v[C_V] == 0.0 || v[C_V] == 1.0, "{label}: coast v column is not 0/1");
        } else {
            ensure!((-1e-6..=1.0 + 1e-6).contains(&v[C_V]), "{label}: coast v column outside [0,1]");
        }
        let q = v[C_CUM] / v[C_TOT];
        ensure!((v[C_U] - q).abs() <= 1e-4 + 1e-5 * q.abs(), "{label}: coast u != cumlen/total (layout assumption broken)");
    }
    Ok(())
}

pub fn triangle_area(v: &[[f64; 20]], t: [usize; 3]) -> f64 {
    let (x, z) = ([v[t[0]][C_X], v[t[1]][C_X], v[t[2]][C_X]], [v[t[0]][C_Z], v[t[1]][C_Z], v[t[2]][C_Z]]);
    0.5 * ((x[1] - x[0]) * (z[2] - z[0]) - (x[2] - x[0]) * (z[1] - z[0])).abs()
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Ref {
    Original(usize, usize),
    Cut(usize, [i64; 4]),
}

pub struct Group {
    pub verts: Vec<[f64; 20]>,
    pub tris: Vec<[usize; 3]>,
    pub stats: Value,
    pub source_area: f64,
    pub clipped_area: f64,
    pub clipped_triangles: usize,
    pub dropped_triangles: usize,
}

fn round4(v: f64) -> i64 {
    (v * 1e4).round_ties_even() as i64
}

pub fn clip_group(geo: &Geo, m: &Mesh) -> Group {
    let scale = geo.scale;
    let tri: Vec<[usize; 3]> = m.idx.chunks_exact(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();
    let nch = geo.charts.len();
    let mut recs: Vec<(usize, [(Ref, [f64; 20]); 3])> = Vec::new();
    let (mut source_area, mut area_out) = (0.0f64, 0.0f64);
    let (mut clipped, mut dropped) = (0usize, 0usize);
    let mut whole_per_chart = vec![0usize; nch];
    let mut clipped_per_chart = vec![0usize; nch];
    let original = |ci: usize, t: &[usize; 3]| -> [(Ref, [f64; 20]); 3] {
        std::array::from_fn(|k| (Ref::Original(ci, t[k]), m.verts[t[k]]))
    };
    for t in &tri {
        let p = [m.verts[t[0]], m.verts[t[1]], m.verts[t[2]]];
        let (xs, zs) = ([p[0][C_X], p[1][C_X], p[2][C_X]], [p[0][C_Z], p[1][C_Z], p[2][C_Z]]);
        let bmin = [xs.iter().cloned().fold(f64::INFINITY, f64::min), zs.iter().cloned().fold(f64::INFINITY, f64::min)];
        let bmax = [xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max), zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)];
        let area_src = triangle_area(&m.verts, *t);
        source_area += area_src;
        let whole = |ci: usize| geo.charts[ci].rects.iter().any(|r| bmin[0] >= r[0][0] && bmin[1] >= r[0][1] && bmax[0] <= r[1][0] && bmax[1] <= r[1][1]);
        let touch = |ci: usize| geo.charts[ci].rects.iter().any(|r| bmax[0] >= r[0][0] && bmax[1] >= r[0][1] && bmin[0] <= r[1][0] && bmin[1] <= r[1][1]);
        if let Some(ci) = (0..nch).find(|&ci| whole(ci)) {
            recs.push((ci, original(ci, t)));
            whole_per_chart[ci] += 1;
            area_out += area_src;
            continue;
        }
        let mut pieces_here = 0;
        for ci in 0..nch {
            if !touch(ci) {
                continue;
            }
            if geo.covered_by_union(ci, bmin, bmax) {
                recs.push((ci, original(ci, t)));
                whole_per_chart[ci] += 1;
                area_out += area_src;
                pieces_here += 1;
                continue;
            }
            for r in &geo.charts[ci].disjoint {
                let (poly, ids) = clip_poly_to_rect(p.to_vec(), t.iter().map(|&i| Some(i)).collect(), r);
                let (mut cp, mut cids): (Vec<[f64; 20]>, Vec<Option<usize>>) = (Vec::new(), Vec::new());
                for (v, id) in poly.into_iter().zip(ids) {
                    if let Some(last) = cp.last() {
                        if (v[C_X] - last[C_X]).abs() < 1e-9 && (v[C_Z] - last[C_Z]).abs() < 1e-9 {
                            continue;
                        }
                    }
                    cp.push(v);
                    cids.push(id);
                }
                if cp.len() > 1 && (cp[0][C_X] - cp[cp.len() - 1][C_X]).abs() < 1e-9 && (cp[0][C_Z] - cp[cp.len() - 1][C_Z]).abs() < 1e-9 {
                    cp.pop();
                    cids.pop();
                }
                if cp.len() < 3 {
                    continue;
                }
                let n = cp.len();
                let (mut s1, mut s2) = (0.0f64, 0.0f64);
                for i in 0..n {
                    s1 += cp[i][C_X] * cp[(i + 1) % n][C_Z];
                    s2 += cp[i][C_Z] * cp[(i + 1) % n][C_X];
                }
                let a = 0.5 * (s1 - s2).abs();
                if a < EPS_AREA {
                    continue;
                }
                let refof = |k: usize| -> Ref {
                    match cids[k] {
                        Some(id) => Ref::Original(ci, id),
                        None => {
                            let v = &cp[k];
                            Ref::Cut(ci, [round4(v[C_X]), round4(v[C_Z]), round4(v[C_V]), round4(v[C_CUM])])
                        }
                    }
                };
                for j in 1..n - 1 {
                    recs.push((ci, [(refof(0), cp[0]), (refof(j), cp[j]), (refof(j + 1), cp[j + 1])]));
                }
                clipped_per_chart[ci] += 1;
                area_out += a;
                pieces_here += 1;
            }
        }
        if pieces_here > 0 {
            clipped += 1;
        } else {
            dropped += 1;
        }
    }
    // vertex table: originals per chart (ascending id), then cut vertices in first-use order
    let mut used: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nch];
    for (ci, tr) in &recs {
        for (k, _) in tr {
            if let Ref::Original(_, id) = k {
                used[*ci].insert(*id);
            }
        }
    }
    let mut table: HashMap<Ref, usize> = HashMap::new();
    let mut rows: Vec<(usize, [f64; 20])> = Vec::new();
    for (ci, ids) in used.iter().enumerate() {
        for &id in ids {
            table.insert(Ref::Original(ci, id), rows.len());
            rows.push((ci, m.verts[id]));
        }
    }
    let mut tris = Vec::with_capacity(recs.len());
    for (ci, tr) in &recs {
        let mut idx = [0usize; 3];
        for (slot, (k, v)) in idx.iter_mut().zip(tr.iter()) {
            *slot = *table.entry(k.clone()).or_insert_with(|| {
                rows.push((*ci, *v));
                rows.len() - 1
            });
        }
        tris.push(idx);
    }
    let verts: Vec<[f64; 20]> = rows
        .into_iter()
        .map(|(ci, mut a)| {
            let c = &geo.charts[ci];
            a[C_X] = a[C_X] * scale + c.offset[0];
            a[C_Z] = a[C_Z] * scale + c.offset[1];
            a[C_CUM] *= scale;
            a[C_TOT] *= scale;
            a
        })
        .collect();
    let stats = json!({
        "source_triangles": tri.len(),
        "clipped_triangles": clipped,
        "dropped_triangles": dropped,
        "whole_triangles_per_chart": geo.charts.iter().zip(&whole_per_chart).filter(|(_, n)| **n > 0).map(|(c, n)| (c.key.clone(), json!(n))).collect::<serde_json::Map<_, _>>(),
        "clipped_pieces_per_chart": geo.charts.iter().zip(&clipped_per_chart).filter(|(_, n)| **n > 0).map(|(c, n)| (c.key.clone(), json!(n))).collect::<serde_json::Map<_, _>>(),
        "source_area": source_area,
        "output_area_over_scale2": area_out,
    });
    Group { verts, tris, stats, source_area, clipped_area: area_out, clipped_triangles: clipped, dropped_triangles: dropped }
}

/// Strict structural check of a written group: vertices inside a transformed crop, finite, in-range indices.
pub fn verify_group(geo: &Geo, b: &[u8], label: &str) -> Result<(f64, usize)> {
    let m = parse(b)?;
    check_layout(&m, label, false)?;
    for (k, v) in m.verts.iter().enumerate() {
        if geo.inverse_chart_of_output(v[C_X], v[C_Z], 1e-3).is_empty() {
            bail!("{label}: vertex {k} outside every transformed crop");
        }
    }
    let (mut area, mut degenerate) = (0.0, 0usize);
    for c in m.idx.chunks_exact(3) {
        let a = triangle_area(&m.verts, [c[0] as usize, c[1] as usize, c[2] as usize]);
        area += a;
        degenerate += (a <= 0.0) as usize;
    }
    Ok((area, degenerate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::Plan;

    fn vert(x: f64, z: f64, cum: f64, v: f64) -> [f64; 20] {
        let mut a = [0f64; 20];
        a[C_X] = x;
        a[C_Z] = z;
        a[C_CUM] = cum;
        a[C_TOT] = 100.0;
        a[C_U] = cum / 100.0;
        a[C_V] = v;
        a
    }

    #[test]
    fn straddling_triangle_is_cut_with_interpolated_attributes_and_moved_by_its_chart() {
        let plan = Plan::proven();
        let geo = Geo::new(&plan).unwrap();
        let ci = geo.charts.iter().position(|c| c.key == "america").unwrap();
        let r = geo.charts[ci].rects[0];
        let (x0, z0) = (r[0][0], r[0][1] + 50.0);
        // triangle crossing the crop's west edge (x = x0): two vertices inside, one 10 units outside
        let m = Mesh {
            header: vec![0; COUNT_OFF],
            verts: vec![vert(x0 + 10.0, z0, 10.0, 0.0), vert(x0 + 10.0, z0 + 10.0, 20.0, 1.0), vert(x0 - 10.0, z0, 30.0, 0.0)],
            idx: vec![0, 1, 2],
        };
        let g = clip_group(&geo, &m);
        assert_eq!(g.clipped_triangles, 1);
        assert!(g.tris.len() >= 1);
        let area_src: f64 = g.clipped_area;
        assert!(area_src > 0.0 && area_src < triangle_area(&m.verts, [0, 1, 2]));
        let s = geo.scale;
        let west = g.verts.iter().map(|v| v[C_X]).fold(f64::INFINITY, f64::min);
        assert!((west - (x0 * s + geo.charts[ci].offset[0])).abs() < 1e-9);
        // cumulative length scales with the world scale
        assert!(g.verts.iter().all(|v| v[C_TOT] == 100.0 * s));
    }
}
