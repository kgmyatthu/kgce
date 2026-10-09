//! Strict parsers/writers for the ETW/NTW campaign display formats (every parser must consume its whole input).

use anyhow::{Context, Result, bail, ensure};
use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use std::io::{Read, Write};

pub fn u32_at(b: &[u8], p: usize) -> Result<u32> {
    let s = b
        .get(p..p + 4)
        .with_context(|| format!("Read past end of data at {p}"))?;
    Ok(u32::from_le_bytes(s.try_into()?))
}

pub fn u16_at(b: &[u8], p: usize) -> Result<u16> {
    let s = b
        .get(p..p + 2)
        .with_context(|| format!("Read past end of data at {p}"))?;
    Ok(u16::from_le_bytes(s.try_into()?))
}

fn utf16_at(b: &[u8], p: usize, units: usize) -> Result<String> {
    let s = b
        .get(p..p + units * 2)
        .with_context(|| format!("Read past end of data at {p}"))?;
    let u: Vec<u16> = s
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(String::from_utf16(&u)?)
}

fn push_utf16(out: &mut Vec<u8>, s: &str) {
    let u: Vec<u16> = s.encode_utf16().collect();
    out.extend_from_slice(&(u.len() as u16).to_le_bytes());
    for c in u {
        out.extend_from_slice(&c.to_le_bytes());
    }
}

// ------------------------------------------------------------------ TGA

pub struct TgaHeader {
    pub id_len: usize,
    pub image_type: u8,
    pub width: usize,
    pub height: usize,
    pub bpp: u8,
    pub desc: u8,
}

pub fn tga_header(b: &[u8]) -> Result<TgaHeader> {
    ensure!(b.len() >= 18, "TGA header truncated");
    Ok(TgaHeader {
        id_len: b[0] as usize,
        image_type: b[2],
        width: u16::from_le_bytes([b[12], b[13]]) as usize,
        height: u16::from_le_bytes([b[14], b[15]]) as usize,
        bpp: b[16],
        desc: b[17],
    })
}

/// 8-bit greyscale TGA -> pixels with rows TOP-first.
pub fn load_tga8(b: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    let h = tga_header(b)?;
    ensure!(
        h.image_type == 3 && h.bpp == 8,
        "Expected an 8-bit greyscale TGA"
    );
    let start = 18 + h.id_len;
    let data = b
        .get(start..start + h.width * h.height)
        .context("TGA pixel data truncated")?;
    if h.desc & 0x20 != 0 {
        return Ok((h.width, h.height, data.to_vec()));
    }
    let mut out = Vec::with_capacity(data.len());
    for y in (0..h.height).rev() {
        out.extend_from_slice(&data[y * h.width..(y + 1) * h.width]);
    }
    Ok((h.width, h.height, out))
}

// ------------------------------------------------------------------ DDS

/// 128-byte DDS header of an uncompressed 8-bit luminance surface (layout of the NTW fog / snow masks).
pub fn dds_header_l8(w: u32, h: u32) -> Vec<u8> {
    let mut hdr = vec![0u8; 128];
    hdr[0..4].copy_from_slice(b"DDS ");
    let put =
        |hdr: &mut Vec<u8>, off: usize, v: u32| hdr[off..off + 4].copy_from_slice(&v.to_le_bytes());
    put(&mut hdr, 4, 124);
    put(&mut hdr, 8, 0x81007); // CAPS|HEIGHT|WIDTH|PIXELFORMAT|LINEARSIZE
    put(&mut hdr, 12, h);
    put(&mut hdr, 16, w);
    put(&mut hdr, 20, w * h);
    put(&mut hdr, 76, 32);
    put(&mut hdr, 80, 0x20000); // DDPF_LUMINANCE
    put(&mut hdr, 88, 8);
    put(&mut hdr, 92, 0xFF);
    put(&mut hdr, 108, 0x1000);
    hdr
}

/// Give a DXT DDS the NTW style header (LINEARSIZE flag + size of the top level).
pub fn dds_with_linear_size(b: &[u8]) -> Result<Vec<u8>> {
    ensure!(b.len() >= 128 && &b[0..4] == b"DDS ", "Not a DDS file");
    let mut out = b.to_vec();
    let h = u32_at(b, 12)?;
    let w = u32_at(b, 16)?;
    let blk = if &b[84..88] == b"DXT1" { 8 } else { 16 };
    let size = w.div_ceil(4).max(1) * h.div_ceil(4).max(1) * blk;
    let flags = u32_at(b, 8)? | 0x80000;
    out[8..12].copy_from_slice(&flags.to_le_bytes());
    out[20..24].copy_from_slice(&size.to_le_bytes());
    Ok(out)
}

// ------------------------------------------------------------------ SPLN rigid_spline

pub struct Spline {
    pub name: String,
    pub flag: u32,
    pub points: Vec<[f32; 3]>,
}

pub fn parse_spln(b: &[u8]) -> Result<(u32, Vec<Spline>)> {
    ensure!(b.get(0..4) == Some(b"SPLN"), "Not an SPLN file");
    let version = u32_at(b, 4)?;
    let count = u32_at(b, 8)? as usize;
    let mut p = 12;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let nl = u16_at(b, p)? as usize;
        p += 2;
        let name = utf16_at(b, p, nl)?;
        p += nl * 2;
        let flag = u32_at(b, p)?;
        let n = u32_at(b, p + 4)? as usize;
        p += 8;
        let raw = b.get(p..p + n * 12).context("SPLN points truncated")?;
        let points = raw
            .chunks_exact(12)
            .map(|c| {
                [
                    f32::from_le_bytes(c[0..4].try_into().unwrap()),
                    f32::from_le_bytes(c[4..8].try_into().unwrap()),
                    f32::from_le_bytes(c[8..12].try_into().unwrap()),
                ]
            })
            .collect();
        p += n * 12;
        out.push(Spline { name, flag, points });
    }
    ensure!(p == b.len(), "SPLN has {} trailing bytes", b.len() - p);
    Ok((version, out))
}

pub fn write_spln(version: u32, splines: &[Spline]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"SPLN");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(splines.len() as u32).to_le_bytes());
    for s in splines {
        push_utf16(&mut out, &s.name);
        out.extend_from_slice(&s.flag.to_le_bytes());
        out.extend_from_slice(&(s.points.len() as u32).to_le_bytes());
        for p in &s.points {
            for v in p {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

// ------------------------------------------------------------------ coastline rigid_mesh (v5, 80 byte vertices)

const COAST_STRIDE: usize = 80;
const COAST_VERT_OFF: usize = 0x1F;

fn coast_vertex_count(b: &[u8]) -> Result<usize> {
    ensure!(u32_at(b, 0)? == 0x1234_5678, "Not a rigid_mesh");
    ensure!(u32_at(b, 4)? == 5, "Unsupported rigid_mesh version");
    Ok(u32_at(b, 0x1B)? as usize)
}

/// Strictly validate a coastline mesh; returns the vertex count.
pub fn parse_coast_mesh(b: &[u8]) -> Result<usize> {
    let n = coast_vertex_count(b)?;
    let p = COAST_VERT_OFF + n * COAST_STRIDE;
    let ni = u32_at(b, p)? as usize;
    ensure!(p + 4 + ni * 4 == b.len(), "Coastline mesh size mismatch");
    for i in 0..ni {
        ensure!(
            (u32_at(b, p + 4 + i * 4)? as usize) < n,
            "Coastline index out of range"
        );
    }
    Ok(n)
}

/// Copy of a coastline mesh with every vertex y (vertex offset +4) set to `y`.
pub fn set_coast_height(b: &[u8], y: f32) -> Result<Vec<u8>> {
    let n = coast_vertex_count(b)?;
    ensure!(
        b.len() >= COAST_VERT_OFF + n * COAST_STRIDE,
        "Coastline mesh truncated"
    );
    let mut out = b.to_vec();
    for i in 0..n {
        let o = COAST_VERT_OFF + i * COAST_STRIDE + 4;
        out[o..o + 4].copy_from_slice(&y.to_le_bytes());
    }
    Ok(out)
}

// ------------------------------------------------------------------ rigid_trees / bridge.markers

pub struct TreeModel {
    pub name: String,
    pub instances: usize,
}

pub fn parse_trees(b: &[u8]) -> Result<Vec<TreeModel>> {
    ensure!(
        u32_at(b, 0)? == 0x3D4D_4047 && u32_at(b, 4)? == 1,
        "Not a rigid_trees file"
    );
    let n_models = u32_at(b, 16)? as usize;
    let mut p = 20;
    let mut out = Vec::with_capacity(n_models);
    for _ in 0..n_models {
        let nl = u16_at(b, p)? as usize;
        let name = utf16_at(b, p + 2, nl)?;
        let q = p + 2 + nl * 2;
        ensure!(u32_at(b, q)? == 1, "Unexpected rigid_trees model marker");
        let count = u32_at(b, q + 4)? as usize;
        p = q + 8 + 24 + count * 16;
        ensure!(p <= b.len(), "rigid_trees instances truncated");
        out.push(TreeModel {
            name,
            instances: count,
        });
    }
    ensure!(p == b.len(), "rigid_trees has trailing bytes");
    Ok(out)
}

pub fn parse_bridges(b: &[u8]) -> Result<usize> {
    let mut p = 0;
    let mut n = 0;
    while p < b.len() {
        let nl = u16_at(b, p)? as usize;
        p += 2 + nl * 2 + 64;
        n += 1;
    }
    ensure!(p == b.len(), "bridge.markers truncated");
    Ok(n)
}

// ------------------------------------------------------------------ supertexture .stpi

pub struct StpiLayer {
    pub tx: u32,
    pub ty: u32,
    /// (offset, compressed size, uncompressed size, id, colour)
    pub entries: Vec<[u32; 5]>,
}

pub struct Stpi {
    pub levels: u32,
    pub width: u32,
    pub height: u32,
    pub tile: u32,
    pub tile_bytes: u32,
    pub layers: Vec<StpiLayer>,
}

pub fn parse_stpi(b: &[u8]) -> Result<Stpi> {
    let (levels, width, height, tile, tile_bytes) = (
        u32_at(b, 0)?,
        u32_at(b, 4)?,
        u32_at(b, 8)?,
        u32_at(b, 12)?,
        u32_at(b, 16)?,
    );
    ensure!(u32_at(b, 20)? == levels, "stpi level count mismatch");
    let mut p = 24;
    let mut layers = Vec::new();
    for _ in 0..levels {
        let (tx, ty) = (u32_at(b, p)?, u32_at(b, p + 4)?);
        p += 8;
        let count = tx as usize * ty as usize;
        let raw = b.get(p..p + count * 20).context("stpi entries truncated")?;
        let entries = raw
            .chunks_exact(20)
            .map(|c| {
                let mut e = [0u32; 5];
                for (i, v) in e.iter_mut().enumerate() {
                    *v = u32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap());
                }
                e
            })
            .collect();
        p += count * 20;
        layers.push(StpiLayer { tx, ty, entries });
    }
    ensure!(
        p == b.len() || p + 4 == b.len(),
        "stpi has unexpected trailing data"
    );
    Ok(Stpi {
        levels,
        width,
        height,
        tile,
        tile_bytes,
        layers,
    })
}

/// ETW tile (a single zlib stream of a DDS file: 128 byte header + 512x512 DXT5) -> NTW tile
/// (headerless DXT5 split into 32 KiB pieces, each stored as `[u32 csize][u32 usize][zlib]`).
pub fn etw_tile_to_ntw(z: &[u8]) -> Result<Vec<u8>> {
    let mut raw = Vec::with_capacity(262_144 + 128);
    ZlibDecoder::new(z)
        .read_to_end(&mut raw)
        .context("Inflate ETW supertexture tile")?;
    if raw.len() != 262_144 + 128 || &raw[0..4] != b"DDS " || &raw[84..88] != b"DXT5" {
        bail!(
            "ETW supertexture tile is not a 512x512 DXT5 DDS (inflated {} bytes)",
            raw.len()
        );
    }
    let mut out = Vec::new();
    for chunk in raw[128..].chunks(0x8000) {
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(9));
        enc.write_all(chunk)?;
        let c = enc.finish()?;
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(&c);
    }
    Ok(out)
}
