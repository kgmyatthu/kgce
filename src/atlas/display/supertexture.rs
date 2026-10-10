//! Virtual supertexture (`supertexture.stpi` / `.stpd`) of the connected atlas: every mip level is streamed tile by
//! tile through the chart mapping of that level (nearest sampling of the source tile pixels, blank tiles for
//! uncovered space, per-level blank/ocean fill block derived from the source). Port of `StBuilder` of the accepted
//! `raster_scale.py`; tiles are written level after level, row after row, with fresh sequential ids.

use super::{
    dxt,
    mapping::{Mapping, ceildiv, touch_grid},
    rasters::ModeCounter,
};
use crate::{atlas::Plan, model::Progress, terrain::formats as F};
use crate::terrain::art::stpx::encode_ntw_tile;
use anyhow::{Context, Result, bail, ensure};
use flate2::read::ZlibDecoder;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
};

pub struct Layer {
    pub tx: usize,
    pub ty: usize,
    /// `(offset, compressed size, uncompressed size, id, colour)`
    pub ents: Vec<[u32; 5]>,
}

pub struct StSource {
    pub levels: usize,
    pub w: usize,
    pub h: usize,
    pub t: usize,
    pub tb: usize,
    pub layers: Vec<Layer>,
    pub colors: Vec<u32>,
    pub tail: Vec<u8>,
    stpd: Vec<u8>,
}

impl StSource {
    pub fn parse(stpi: &[u8], stpd: Vec<u8>) -> Result<Self> {
        let p = F::parse_stpi(stpi)?;
        let (levels, w, h, t, tb) = (p.levels as usize, p.width as usize, p.height as usize, p.tile as usize, p.tile_bytes as usize);
        ensure!(
            tb == t * t && t % 4 == 0 && t > 0,
            "unsupported supertexture tile geometry T={t} TB={tb}"
        );
        let mut consumed = 24usize;
        let mut layers = Vec::new();
        let mut colors = Vec::new();
        for (lv, l) in p.layers.iter().enumerate() {
            let (tx, ty) = (l.tx as usize, l.ty as usize);
            ensure!(
                (w >> lv) == tx * t && (h >> lv) == ty * t,
                "stpi level {lv} dims {tx}x{ty} tiles do not match {w}x{h}>>{lv}"
            );
            consumed += 8 + 20 * l.entries.len();
            let mut colour = None;
            for e in &l.entries {
                ensure!(e[2] as usize == tb, "stpi usize != tile bytes");
                ensure!(e[0] as u64 + e[1] as u64 <= stpd.len() as u64, "stpi entry beyond stpd end");
                match colour {
                    None => colour = Some(e[4]),
                    Some(c) => ensure!(c == e[4], "stpi entry colour is not constant within level {lv}; cannot preserve it"),
                }
            }
            colors.push(colour.context("empty supertexture level")?);
            layers.push(Layer { tx, ty, ents: l.entries.clone() });
        }
        ensure!(p.layers.len() == levels, "stpi level count mismatch");
        let tail = stpi[consumed..].to_vec();
        ensure!(tail.len() == 0 || tail.len() == 4, "unexpected stpi tail");
        Ok(Self { levels, w, h, t, tb, layers, colors, tail, stpd })
    }

    fn decode_raw(&self, off: usize, cs: usize) -> Result<Vec<u8>> {
        let mv = &self.stpd[off..off + cs];
        let mut p = 0;
        let mut raw = Vec::with_capacity(self.tb);
        while p < cs {
            ensure!(p + 8 <= cs, "tile chunk header truncated");
            let c = u32::from_le_bytes(mv[p..p + 4].try_into().unwrap()) as usize;
            let u = u32::from_le_bytes(mv[p + 4..p + 8].try_into().unwrap()) as usize;
            ensure!(p + 8 + c <= cs, "tile chunk payload truncated");
            let before = raw.len();
            ZlibDecoder::new(&mv[p + 8..p + 8 + c]).read_to_end(&mut raw).context("inflate tile chunk")?;
            ensure!(raw.len() - before == u, "tile chunk usize mismatch");
            p += 8 + c;
        }
        ensure!(raw.len() == self.tb, "tile size mismatch");
        Ok(raw)
    }

    fn raw_tile(&self, level: usize, r: usize, c: usize) -> Result<Vec<u8>> {
        let l = &self.layers[level];
        let e = l.ents[r * l.tx + c];
        self.decode_raw(e[0] as usize, e[1] as usize)
    }
}

/// Per-worker LRU of decoded source tile pixels.
struct TileCache<'a> {
    st: &'a StSource,
    level: usize,
    cap: usize,
    tiles: HashMap<usize, Rc<Vec<u8>>>,
    order: VecDeque<usize>,
}

impl<'a> TileCache<'a> {
    fn new(st: &'a StSource, level: usize, cap: usize) -> Self {
        Self { st, level, cap, tiles: HashMap::new(), order: VecDeque::new() }
    }

    fn pixels(&mut self, r: usize, c: usize) -> Result<Rc<Vec<u8>>> {
        let l = &self.st.layers[self.level];
        let key = r * l.tx + c;
        if let Some(t) = self.tiles.get(&key) {
            let t = t.clone();
            self.order.retain(|&k| k != key);
            self.order.push_back(key);
            return Ok(t);
        }
        let raw = self.st.raw_tile(self.level, r, c)?;
        let b = self.st.t / 4;
        let px = Rc::new(dxt::blocks_to_pixels(&raw, b, b));
        self.tiles.insert(key, px.clone());
        self.order.push_back(key);
        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.tiles.remove(&old);
            }
        }
        Ok(px)
    }
}

struct Fill {
    block: [u8; 16],
    fraction: f64,
    distinct: usize,
    candidates: usize,
}

/// Source blank/ocean block of one level: mode block among blocks that lie fully outside every source crop.
fn fill_block(st: &StSource, map: &Mapping, level: usize) -> Result<Option<Fill>> {
    let l = &st.layers[level];
    let (t, b) = (st.t as i64, st.t / 4);
    let boxes = map.source_boxes();
    let touched = touch_grid(&boxes, 0, 0, l.ty, l.tx, t);
    let n = l.tx * l.ty;
    let cand: Vec<usize> = if n <= 256 {
        (0..n).collect()
    } else {
        let mut v: Vec<usize> = (0..n).filter(|&i| !touched[i]).collect();
        v.sort_by_key(|&i| l.ents[i][1]);
        v.truncate(24);
        v
    };
    let mut counter = ModeCounter::default();
    for k in cand {
        let (r, c) = (k / l.tx, k % l.tx);
        let raw = st.raw_tile(level, r, c)?;
        let touch = touch_grid(&boxes, r as i64 * t, c as i64 * t, b, b, 4);
        for (i, blk) in raw.chunks_exact(16).enumerate() {
            if !touch[i] {
                counter.add(blk);
            }
        }
    }
    Ok(counter.finish().map(|(block, fraction, distinct)| Fill { block, fraction, distinct, candidates: counter.total() }))
}


struct LevelCtx<'a> {
    st: &'a StSource,
    level: usize,
    map: Mapping,
    fill: Option<[u8; 16]>,
    fill_px: [[u8; 4]; 16],
    tx_out: usize,
    ty_out: usize,
    blank: Option<Vec<u8>>,
}

fn build_tile(ctx: &LevelCtx, cache: &mut TileCache, ty: usize, tx: usize) -> Result<Vec<u8>> {
    let t = ctx.st.t;
    let ti = t as i64;
    let nb = t / 4;
    let (x0, y0) = (tx as i64 * ti, ty as i64 * ti);
    let mut touched = vec![false; nb * nb];
    let mut any = false;
    for r in &ctx.map.rects {
        let (cx0, cx1) = (((r.ox0 - x0) >> 2).max(0), ceildiv(r.ox1 - x0, 4).min(nb as i64));
        let (cy0, cy1) = (((r.oy0 - y0) >> 2).max(0), ceildiv(r.oy1 - y0, 4).min(nb as i64));
        if cx0 < cx1 && cy0 < cy1 {
            any = true;
            for y in cy0..cy1 {
                for x in cx0..cx1 {
                    touched[y as usize * nb + x as usize] = true;
                }
            }
        }
    }
    if !any {
        return ctx.blank.clone().with_context(|| format!("level {}: untouched tile but no source fill block", ctx.level));
    }
    if ctx.fill.is_none() && !touched.iter().all(|&v| v) {
        bail!("level {}: region needs a fill block but none could be derived from the source", ctx.level);
    }
    let mut canvas = vec![0u8; t * t * 4];
    for y in 0..t {
        for x in 0..t {
            canvas[(y * t + x) * 4..][..4].copy_from_slice(&ctx.fill_px[(y & 3) * 4 + (x & 3)]);
        }
    }
    for r in &ctx.map.rects {
        let (ax, bx) = (r.ox0.max(x0), r.ox1.min(x0 + ti));
        let (ay, by) = (r.oy0.max(y0), r.oy1.min(y0 + ti));
        if ax >= bx || ay >= by {
            continue;
        }
        let cols: Vec<(usize, usize)> = (ax..bx)
            .map(|ox| {
                let sx = ctx.map.src_col(r, ox) as usize;
                (sx / t, sx % t)
            })
            .collect();
        for oy in ay..by {
            let sy = ctx.map.src_row(r, oy) as usize;
            let (tr, ly) = (sy / t, sy % t);
            let mut cur_tc = usize::MAX;
            let mut cur: Option<Rc<Vec<u8>>> = None;
            let drow = ((oy - y0) as usize * t + (ax - x0) as usize) * 4;
            for (i, &(tc, lx)) in cols.iter().enumerate() {
                if tc != cur_tc {
                    cur = Some(cache.pixels(tr, tc)?);
                    cur_tc = tc;
                }
                let src = cur.as_ref().unwrap();
                let s = (ly * t + lx) * 4;
                canvas[drow + i * 4..drow + i * 4 + 4].copy_from_slice(&src[s..s + 4]);
            }
        }
    }
    let mut raw = vec![0u8; nb * nb * 16];
    for by in 0..nb {
        for bx in 0..nb {
            let dst = &mut raw[(by * nb + bx) * 16..][..16];
            if touched[by * nb + bx] {
                dst.copy_from_slice(&dxt::encode_block(&dxt::gather_block(&canvas, t, bx, by)));
            } else {
                dst.copy_from_slice(&ctx.fill.expect("checked above"));
            }
        }
    }
    encode_ntw_tile(&raw)
}

/// Staged output files `(temporary, final)` of the supertexture pair plus the factual report.
pub struct Staged {
    pub files: Vec<(PathBuf, PathBuf)>,
    pub report: Value,
}

pub fn transform(map_dir: &Path, plan: &Plan, progress: &Progress<'_>) -> Result<Staged> {
    let dir = map_dir.join("display/supertexture");
    let (stpi_path, stpd_path) = (dir.join("supertexture.stpi"), dir.join("supertexture.stpd"));
    progress("Atlas display: reading source supertexture");
    let st = StSource::parse(
        &fs::read(&stpi_path).with_context(|| format!("Read {}", stpi_path.display()))?,
        fs::read(&stpd_path).with_context(|| format!("Read {}", stpd_path.display()))?,
    )?;
    ensure!(
        plan.supertexture_levels as usize == st.levels,
        "plan supertexture_levels {} != source {}",
        plan.supertexture_levels,
        st.levels
    );
    super::rasters::check_dims(plan, "supertexture", "display/supertexture/supertexture.stpi", st.w, st.h)?;
    let t = st.t;
    let mut maps = Vec::new();
    let mut fills = Vec::new();
    for lv in 0..st.levels {
        let l = &st.layers[lv];
        let m = Mapping::new(plan, l.tx * t, l.ty * t, &format!("supertexture level {lv}"))?;
        ensure!(
            m.out_w as usize % t == 0 && m.out_h as usize % t == 0,
            "level {lv}: output {}x{} is not a whole number of {t}px tiles",
            m.out_w,
            m.out_h
        );
        let fill = fill_block(&st, &m, lv)?;
        fills.push(fill);
        maps.push(m);
    }
    let out_w = maps[0].out_w as usize;
    let out_h = maps[0].out_h as usize;
    let (tmp_d, tmp_i) = (dir.join("supertexture.stpd.atlas-new"), dir.join("supertexture.stpi.atlas-new"));
    let mut writer = std::io::BufWriter::new(fs::File::create(&tmp_d).with_context(|| format!("Create {}", tmp_d.display()))?);
    let mut entries: Vec<(usize, usize, Vec<[u32; 5]>)> = Vec::new();
    let (mut pos, mut sid) = (0u64, 1u32);
    let mut level_reports = Vec::new();
    let mut blank_tiles_total = 0usize;
    let workers = super::threads();
    for lv in 0..st.levels {
        let map = maps[lv].clone();
        let (tx_out, ty_out) = (map.out_w as usize / t, map.out_h as usize / t);
        let fill = fills[lv].as_ref().map(|f| f.block);
        let nb = t / 4;
        let blank = match &fill {
            Some(b) => {
                let mut raw = Vec::with_capacity(nb * nb * 16);
                for _ in 0..nb * nb {
                    raw.extend_from_slice(b);
                }
                Some(encode_ntw_tile(&raw)?)
            }
            None => None,
        };
        let ctx = LevelCtx {
            st: &st,
            level: lv,
            map,
            fill,
            fill_px: fill.map(|b| dxt::decode_block(&b)).unwrap_or([[0u8; 4]; 16]),
            tx_out,
            ty_out,
            blank,
        };
        let touched_tiles = touch_grid(
            &ctx.map.rects.iter().map(|r| [r.ox0, r.oy0, r.ox1, r.oy1]).collect::<Vec<_>>(),
            0,
            0,
            ty_out,
            tx_out,
            t as i64,
        );
        if fill.is_none() && !touched_tiles.iter().all(|&v| v) {
            bail!("level {lv}: uncovered tiles but no source blank block found");
        }
        let colour = st.colors[lv];
        let mut level_entries: Vec<[u32; 5]> = Vec::with_capacity(tx_out * ty_out);
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let (send, recv) = mpsc::channel::<(usize, Result<Vec<Vec<u8>>>)>();
        let mut failure: Option<anyhow::Error> = None;
        std::thread::scope(|s| {
            for _ in 0..workers.min(ty_out.max(1)) {
                let send = send.clone();
                let (ctx, next, stop) = (&ctx, &next, &stop);
                s.spawn(move || {
                    let mut cache = TileCache::new(ctx.st, ctx.level, 24);
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let row = next.fetch_add(1, Ordering::Relaxed);
                        if row >= ctx.ty_out {
                            break;
                        }
                        let res = (0..ctx.tx_out).map(|tx| build_tile(ctx, &mut cache, row, tx)).collect::<Result<Vec<_>>>();
                        if send.send((row, res)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(send);
            let mut pending: BTreeMap<usize, Result<Vec<Vec<u8>>>> = BTreeMap::new();
            let mut want = 0usize;
            for (row, res) in recv {
                pending.insert(row, res);
                while let Some(res) = pending.remove(&want) {
                    want += 1;
                    if failure.is_some() {
                        continue;
                    }
                    match res {
                        Err(e) => {
                            failure = Some(e);
                            stop.store(true, Ordering::Relaxed);
                        }
                        Ok(tiles) => {
                            for c in tiles {
                                if pos + c.len() as u64 > u32::MAX as u64 {
                                    failure = Some(anyhow::anyhow!("supertexture.stpd exceeds the u32 offset range"));
                                    stop.store(true, Ordering::Relaxed);
                                    break;
                                }
                                if let Err(e) = writer.write_all(&c) {
                                    failure = Some(e.into());
                                    stop.store(true, Ordering::Relaxed);
                                    break;
                                }
                                level_entries.push([pos as u32, c.len() as u32, st.tb as u32, sid, colour]);
                                pos += c.len() as u64;
                                sid += 1;
                            }
                        }
                    }
                }
            }
        });
        if let Some(e) = failure {
            return Err(e.context(format!("Supertexture level {lv}")));
        }
        ensure!(level_entries.len() == tx_out * ty_out, "level {lv}: tile count mismatch");
        let content = touched_tiles.iter().filter(|&&v| v).count();
        blank_tiles_total += tx_out * ty_out - content;
        level_reports.push(json!({
            "level": lv,
            "tiles": [tx_out, ty_out],
            "tile_count": tx_out * ty_out,
            "content_tiles": content,
            "blank_tiles": tx_out * ty_out - content,
            "entry_colour": colour,
            "source_tiles": [st.layers[lv].tx, st.layers[lv].ty],
            "fill": fills[lv].as_ref().map(|f| json!({
                "fraction": f.fraction,
                "distinct": f.distinct,
                "candidate_blocks": f.candidates,
                "rgba": dxt::decode_block(&f.block)[0],
            })),
            "mapping": ctx.map.report(),
        }));
        progress(&format!("Atlas display: supertexture level {lv} done ({tx_out}x{ty_out} tiles, {} MiB)", pos >> 20));
        entries.push((tx_out, ty_out, level_entries));
    }
    writer.flush()?;
    drop(writer);
    let mut stpi = Vec::new();
    for v in [st.levels as u32, out_w as u32, out_h as u32, t as u32, st.tb as u32, st.levels as u32] {
        stpi.extend_from_slice(&v.to_le_bytes());
    }
    for (tx, ty, ents) in &entries {
        stpi.extend_from_slice(&(*tx as u32).to_le_bytes());
        stpi.extend_from_slice(&(*ty as u32).to_le_bytes());
        for e in ents {
            for v in e {
                stpi.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    stpi.extend_from_slice(&st.tail);
    fs::write(&tmp_i, &stpi).with_context(|| format!("Write {}", tmp_i.display()))?;
    let report = json!({
        "stpd_bytes": pos,
        "stpi_bytes": stpi.len(),
        "tile_ids": sid - 1,
        "blank_tiles": blank_tiles_total,
        "output_px": [out_w, out_h],
        "levels": level_reports,
    });
    Ok(Staged { files: vec![(tmp_d, stpd_path), (tmp_i, stpi_path)], report })
}

/// Re-read a written pair and check its structure (tile grid, entries contiguous, sequential ids, stpd size).
pub fn verify(stpi: &[u8], stpd_len: u64, expect_dims: (usize, usize)) -> Result<Value> {
    let p = F::parse_stpi(stpi)?;
    ensure!(
        (p.width as usize, p.height as usize) == expect_dims && p.levels > 0,
        "output stpi header mismatch"
    );
    let (mut pos, mut sid) = (0u64, 1u32);
    for (lv, l) in p.layers.iter().enumerate() {
        ensure!(
            p.tile > 0 && l.tx == (p.width >> lv) / p.tile && l.ty == (p.height >> lv) / p.tile,
            "level {lv}: tile grid mismatch"
        );
        for e in &l.entries {
            ensure!(e[0] as u64 == pos, "level {lv}: stpd not contiguous");
            ensure!(e[3] == sid, "level {lv}: ids not sequential");
            pos += e[1] as u64;
            sid += 1;
        }
    }
    ensure!(pos == stpd_len, "stpd size {stpd_len} != sum of entries {pos}");
    ensure!(sid > 1, "output stpi has no tiles");
    Ok(json!({"tile_ids": sid - 1, "stpd_bytes": pos}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_tiles_round_trip_through_the_source_decoder() {
        let raw: Vec<u8> = (0..512 * 512).map(|i| (i * 7 % 251) as u8).collect();
        let packed = encode_ntw_tile(&raw).unwrap();
        let mut stpd = packed.clone();
        stpd.extend_from_slice(&[0; 4]);
        let st = StSource {
            levels: 1,
            w: 512,
            h: 512,
            t: 512,
            tb: 512 * 512,
            layers: vec![Layer { tx: 1, ty: 1, ents: vec![[0, packed.len() as u32, 512 * 512, 1, 7]] }],
            colors: vec![7],
            tail: vec![],
            stpd,
        };
        assert_eq!(st.raw_tile(0, 0, 0).unwrap(), raw);
    }
}
