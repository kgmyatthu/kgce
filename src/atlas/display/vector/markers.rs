//! `display/world.markers` (emitter markers, model markers, overlay groups). The stock NTW install ships none, but a
//! present file is strictly parsed, must round-trip byte for byte, and is transformed with the chart affine so no spatial
//! record is silently ignored (`parse_world_markers` / `transform_markers` of the accepted `vector_scale.py`).

use super::geo::Geo;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

const NAMES: [&str; 3] = ["CAMPAIGN_DISPLAY_EMITTER_MARKER", "CAMPAIGN_DISPLAY_MODEL_MARKER", "CAMPAIGN_DISPLAY_OVERLAY_MARKER"];

pub struct Group {
    pub kind: u32,
    pub name: String,
    /// `x, h, worldY, u, v, edge flag` per vertex.
    pub pts: Vec<[f32; 6]>,
    pub idx: Vec<u32>,
}

pub enum Section {
    /// Emitter (`hash, x, y, flags`) or model (`hash, x, y, angle, flags`) records as raw f32/u32 fields.
    Records { name: String, tag: [u8; 4], recs: Vec<Vec<f64>>, effs: Vec<(String, Vec<u32>)> },
    Overlay { name: String, tag: [u8; 4], groups: Vec<Group>, names: Vec<(String, Vec<u32>)> },
}

pub struct Markers {
    pub header: Vec<u8>,
    pub secs: Vec<Section>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.p..self.p + n).context("world.markers truncated")?;
        self.p += n;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        let raw = self.take(n * 2)?;
        let u: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok(String::from_utf16(&u)?)
    }
}

fn push_string(out: &mut Vec<u8>, s: &str) {
    let u: Vec<u16> = s.encode_utf16().collect();
    out.extend_from_slice(&(u.len() as u16).to_le_bytes());
    for c in u {
        out.extend_from_slice(&c.to_le_bytes());
    }
}

fn names_table(r: &mut Reader) -> Result<Vec<(String, Vec<u32>)>> {
    let mut out = Vec::new();
    for _ in 0..r.u16()? {
        ensure!(r.u16()? == 1, "world.markers: name record marker != 1");
        let nm = r.string()?;
        let k = r.u32()?;
        out.push((nm, (0..k).map(|_| r.u32()).collect::<Result<Vec<_>>>()?));
    }
    Ok(out)
}

pub fn parse(b: &[u8]) -> Result<Markers> {
    let mut r = Reader { b, p: 0 };
    let _repo = r.string()?;
    let header = b.get(..0x50).context("world.markers header truncated")?.to_vec();
    r.p = 0x50;
    let mut secs = Vec::new();
    for (si, expect) in NAMES.iter().enumerate() {
        let tag: [u8; 4] = r.take(4)?.try_into()?;
        ensure!(
            tag == [si as u8 + 1, 0xf2, 0x5d, 0x2a],
            "world.markers section {si}: bad tag {}",
            hex::encode(tag)
        );
        let name = r.string()?;
        ensure!(name == *expect, "world.markers section {si}: unexpected type {name}");
        ensure!(r.take(12)?.iter().all(|&v| v == 0), "world.markers: nonzero section pad");
        ensure!(r.take(4)? == tag, "world.markers: section tag mismatch");
        if si < 2 {
            let (ver, n) = (r.u32()?, r.u32()? as usize);
            ensure!(ver == 2, "world.markers section {si}: version {ver}");
            let mut recs = Vec::with_capacity(n);
            for _ in 0..n {
                let hash = r.u32()? as f64;
                let x = r.f32()? as f64;
                let y = r.f32()? as f64;
                if si == 0 {
                    recs.push(vec![hash, x, y, r.u32()? as f64]);
                } else {
                    let angle = r.f32()? as f64;
                    recs.push(vec![hash, x, y, angle, r.u32()? as f64]);
                }
            }
            let effs = if n > 0 { names_table(&mut r)? } else { Vec::new() };
            ensure!(r.take(12)?.iter().all(|&v| v == 0), "world.markers: nonzero section trailer");
            secs.push(Section::Records { name, tag, recs, effs });
        } else {
            let (ver, a) = (r.u32()?, r.u32()?);
            ensure!(ver == 5, "world.markers overlay version {ver}");
            let mut groups = Vec::new();
            for _ in 0..a {
                let kind = r.u32()?;
                let nm = r.string()?;
                let n = r.u32()? as usize;
                let mut pts = Vec::with_capacity(n);
                for _ in 0..n {
                    let mut p = [0f32; 6];
                    for v in &mut p {
                        *v = r.f32()?;
                    }
                    pts.push(p);
                }
                let m = r.u32()? as usize;
                let idx = (0..m).map(|_| r.u32()).collect::<Result<Vec<_>>>()?;
                groups.push(Group { kind, name: nm, pts, idx });
            }
            let names = names_table(&mut r)?;
            secs.push(Section::Overlay { name, tag, groups, names });
        }
    }
    ensure!(r.p == b.len(), "world.markers trailing bytes {} != {}", r.p, b.len());
    Ok(Markers { header, secs })
}

pub fn write(w: &Markers) -> Vec<u8> {
    let mut out = w.header.clone();
    let table = |out: &mut Vec<u8>, t: &[(String, Vec<u32>)]| {
        out.extend_from_slice(&(t.len() as u16).to_le_bytes());
        for (nm, idx) in t {
            out.extend_from_slice(&1u16.to_le_bytes());
            push_string(out, nm);
            out.extend_from_slice(&(idx.len() as u32).to_le_bytes());
            for i in idx {
                out.extend_from_slice(&i.to_le_bytes());
            }
        }
    };
    for (si, sec) in w.secs.iter().enumerate() {
        match sec {
            Section::Records { name, tag, recs, effs } => {
                out.extend_from_slice(tag);
                push_string(&mut out, name);
                out.extend_from_slice(&[0u8; 12]);
                out.extend_from_slice(tag);
                out.extend_from_slice(&2u32.to_le_bytes());
                out.extend_from_slice(&(recs.len() as u32).to_le_bytes());
                for r in recs {
                    out.extend_from_slice(&(r[0] as u32).to_le_bytes());
                    out.extend_from_slice(&(r[1] as f32).to_le_bytes());
                    out.extend_from_slice(&(r[2] as f32).to_le_bytes());
                    if si == 0 {
                        out.extend_from_slice(&(r[3] as u32).to_le_bytes());
                    } else {
                        out.extend_from_slice(&(r[3] as f32).to_le_bytes());
                        out.extend_from_slice(&(r[4] as u32).to_le_bytes());
                    }
                }
                if !recs.is_empty() {
                    table(&mut out, effs);
                }
                out.extend_from_slice(&[0u8; 12]);
            }
            Section::Overlay { name, tag, groups, names } => {
                out.extend_from_slice(tag);
                push_string(&mut out, name);
                out.extend_from_slice(&[0u8; 12]);
                out.extend_from_slice(tag);
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&(groups.len() as u32).to_le_bytes());
                for g in groups {
                    out.extend_from_slice(&g.kind.to_le_bytes());
                    push_string(&mut out, &g.name);
                    out.extend_from_slice(&(g.pts.len() as u32).to_le_bytes());
                    for p in &g.pts {
                        for v in p {
                            out.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    out.extend_from_slice(&(g.idx.len() as u32).to_le_bytes());
                    for i in &g.idx {
                        out.extend_from_slice(&i.to_le_bytes());
                    }
                }
                table(&mut out, names);
            }
        }
    }
    out
}

/// Emitter/model markers move with their chart (out-of-crop markers are dropped, effect indices remapped); overlay
/// groups translate rigidly by `T(c) - c` of their bbox centre so settlement/farm underlays keep their 1x footprint.
pub fn transform(w: &Markers, geo: &Geo) -> Result<(Markers, Value)> {
    let mut report = serde_json::Map::new();
    let mut secs = Vec::new();
    for sec in &w.secs {
        match sec {
            Section::Records { name, tag, recs, effs } => {
                let mut keep = std::collections::BTreeMap::new();
                let mut out = Vec::new();
                for (i, r) in recs.iter().enumerate() {
                    let Some(ci) = geo.chart_of_point(r[1], r[2]) else { continue };
                    let (nx, ny) = geo.xy(r[1], r[2], ci);
                    let mut r2 = r.clone();
                    r2[1] = nx as f32 as f64;
                    r2[2] = ny as f32 as f64;
                    keep.insert(i, out.len());
                    out.push(r2);
                }
                let mut effs2 = Vec::new();
                for (nm, idxs) in effs {
                    if let Some(&bad) = idxs.iter().find(|&&i| i as usize >= recs.len()) {
                        bail!("world.markers {name}: effect {nm} references marker {bad} out of range");
                    }
                    let ni: Vec<u32> = idxs.iter().filter_map(|&i| keep.get(&(i as usize)).map(|&k| k as u32)).collect();
                    if !ni.is_empty() {
                        effs2.push((nm.clone(), ni));
                    }
                }
                report.insert(name.clone(), json!({"source": recs.len(), "kept": out.len(), "dropped": recs.len() - out.len(), "effects_source": effs.len(), "effects_output": effs2.len()}));
                secs.push(Section::Records { name: name.clone(), tag: *tag, recs: out, effs: effs2 });
            }
            Section::Overlay { name, tag, groups, names } => {
                let mut gkeep = std::collections::BTreeMap::new();
                let mut out: Vec<Group> = Vec::new();
                for (gi, g) in groups.iter().enumerate() {
                    if g.pts.is_empty() {
                        gkeep.insert(gi, out.len());
                        out.push(Group { kind: g.kind, name: g.name.clone(), pts: Vec::new(), idx: g.idx.clone() });
                        continue;
                    }
                    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
                    for p in &g.pts {
                        for (k, c) in [0usize, 2].into_iter().enumerate() {
                            lo[k] = lo[k].min(p[c] as f64);
                            hi[k] = hi[k].max(p[c] as f64);
                        }
                    }
                    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
                    let Some(ci) = geo.chart_of_point(c[0], c[1]) else { continue };
                    let (nx, ny) = geo.xy(c[0], c[1], ci);
                    let (dx, dy) = (nx - c[0], ny - c[1]);
                    let pts = g
                        .pts
                        .iter()
                        .map(|p| {
                            let mut q = *p;
                            q[0] = (p[0] as f64 + dx) as f32;
                            q[2] = (p[2] as f64 + dy) as f32;
                            q
                        })
                        .collect();
                    gkeep.insert(gi, out.len());
                    out.push(Group { kind: g.kind, name: g.name.clone(), pts, idx: g.idx.clone() });
                }
                let mut names2 = Vec::new();
                for (nm, idxs) in names {
                    if let Some(&bad) = idxs.iter().find(|&&i| i as usize >= groups.len()) {
                        bail!("world.markers slot {nm}: group {bad} out of range");
                    }
                    let ni: Vec<u32> = idxs.iter().filter_map(|&i| gkeep.get(&(i as usize)).map(|&k| k as u32)).collect();
                    if !ni.is_empty() {
                        names2.push((nm.clone(), ni));
                    }
                }
                report.insert(name.clone(), json!({"source_groups": groups.len(), "kept_groups": out.len(), "dropped_groups": groups.len() - out.len(), "slots_source": names.len(), "slots_output": names2.len()}));
                secs.push(Section::Overlay { name: name.clone(), tag: *tag, groups: out, names: names2 });
            }
        }
    }
    Ok((Markers { header: w.header.clone(), secs }, Value::Object(report)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(s: &str) -> Vec<u8> {
        let mut o = Vec::new();
        push_string(&mut o, s);
        o
    }

    fn sample() -> Vec<u8> {
        let mut b = string("repo");
        b.resize(0x50, 0);
        for (si, name) in NAMES.iter().enumerate() {
            let tag = [si as u8 + 1, 0xf2, 0x5d, 0x2a];
            b.extend_from_slice(&tag);
            b.extend(string(name));
            b.extend_from_slice(&[0u8; 12]);
            b.extend_from_slice(&tag);
            if si < 2 {
                b.extend_from_slice(&2u32.to_le_bytes());
                b.extend_from_slice(&0u32.to_le_bytes());
                b.extend_from_slice(&[0u8; 12]);
            } else {
                b.extend_from_slice(&5u32.to_le_bytes());
                b.extend_from_slice(&0u32.to_le_bytes());
                b.extend_from_slice(&0u16.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn empty_marker_file_round_trips_byte_for_byte() {
        let b = sample();
        assert_eq!(write(&parse(&b).unwrap()), b);
    }

    #[test]
    fn unknown_trailing_data_is_refused() {
        let mut b = sample();
        b.push(0);
        assert!(parse(&b).is_err());
    }
}
