//! ESF (0xABCE) container used by Empire / Napoleon: Total War.
//!
//! Header: u32 magic 0xABCE, u32 0, u32 timestamp, u32 offset of the node-name table.
//! Name table: u16 count, then count * (u16 len + latin1 bytes).
//!
//! Values (`code`, payload):
//!   0x01 bool, 0x02 i8, 0x03 i16, 0x04 i32, 0x05 i64, 0x06 u8, 0x07 u16, 0x08 u32, 0x09 u64,
//!   0x0a f32, 0x0b f64, 0x0c (f32,f32), 0x0d (f32,f32,f32), 0x0e UTF-16 string (u16 char count),
//!   0x0f latin1 string (u16 byte count), 0x10 angle (u16), 0x40|code (0x41..=0x4f) packed array kept raw.
//! Records:
//!   0x80 name:u16 version:u8 end:u32, children...
//!   0x81 name:u16 version:u8 end:u32 count:u32, count * (end:u32, children...)
//!
//! The reader is strict: every offset is bounds checked, every record/array must end exactly at its
//! declared end, and anything the writer could not reproduce byte-for-byte (non 0/1 booleans, a non-zero
//! second header word, trailing bytes after the name table) is rejected instead of silently normalised.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;

pub const MAGIC: u32 = 0xABCE;
const MAX_DEPTH: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub struct Esf {
    pub timestamp: u32,
    pub names: Vec<String>,
    pub root: Node,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Record {
        name: String,
        version: u8,
        children: Vec<Node>,
    },
    Array {
        name: String,
        version: u8,
        items: Vec<Vec<Node>>,
    },
    Value(Value),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    XY([f32; 2]),
    XYZ([f32; 3]),
    String16(String),
    String8(String),
    Angle(u16),
    Array { code: u8, data: Vec<u8> },
}

impl Value {
    /// ESF type code of this value as stored in the file.
    pub fn code(&self) -> u8 {
        match self {
            Value::Bool(_) => 0x01,
            Value::I8(_) => 0x02,
            Value::I16(_) => 0x03,
            Value::I32(_) => 0x04,
            Value::I64(_) => 0x05,
            Value::U8(_) => 0x06,
            Value::U16(_) => 0x07,
            Value::U32(_) => 0x08,
            Value::U64(_) => 0x09,
            Value::F32(_) => 0x0A,
            Value::F64(_) => 0x0B,
            Value::XY(_) => 0x0C,
            Value::XYZ(_) => 0x0D,
            Value::String16(_) => 0x0E,
            Value::String8(_) => 0x0F,
            Value::Angle(_) => 0x10,
            Value::Array { code, .. } => *code,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Value::Bool(_) => "bool",
            Value::I8(_) => "i8",
            Value::I16(_) => "i16",
            Value::I32(_) => "i32",
            Value::I64(_) => "i64",
            Value::U8(_) => "u8",
            Value::U16(_) => "u16",
            Value::U32(_) => "u32",
            Value::U64(_) => "u64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::XY(_) => "xy",
            Value::XYZ(_) => "xyz",
            Value::String16(_) => "string16",
            Value::String8(_) => "string8",
            Value::Angle(_) => "angle",
            Value::Array { .. } => "packed array",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------------------------

fn take(d: &[u8], p: usize, n: usize) -> Result<&[u8]> {
    let end = p
        .checked_add(n)
        .ok_or_else(|| anyhow!("ESF offset overflow at {p:#x}"))?;
    d.get(p..end).ok_or_else(|| {
        anyhow!(
            "ESF truncated: need {n} bytes at {p:#x}, file has {:#x}",
            d.len()
        )
    })
}

fn rd_u16(d: &[u8], p: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(take(d, p, 2)?.try_into().unwrap()))
}

fn rd_u32(d: &[u8], p: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(take(d, p, 4)?.try_into().unwrap()))
}

fn scalar_size(code: u8) -> Option<usize> {
    Some(match code {
        0x01 | 0x02 | 0x06 => 1,
        0x03 | 0x07 | 0x10 => 2,
        0x04 | 0x08 | 0x0A => 4,
        0x05 | 0x09 | 0x0B | 0x0C => 8,
        0x0D => 12,
        _ => return None,
    })
}

struct Parser<'a> {
    d: &'a [u8],
    names: Vec<String>,
}

impl<'a> Parser<'a> {
    fn name(&self, id: u16) -> Result<String> {
        self.names.get(id as usize).cloned().ok_or_else(|| {
            anyhow!(
                "ESF node name index {id} outside the {} entry name table",
                self.names.len()
            )
        })
    }

    fn scalar(&self, code: u8, p: usize) -> Result<Value> {
        let b = take(self.d, p, scalar_size(code).unwrap())?;
        let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        Ok(match code {
            0x01 => match b[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                other => bail!("ESF bool at {p:#x} holds {other:#x}; only 0/1 round-trips"),
            },
            0x02 => Value::I8(b[0] as i8),
            0x03 => Value::I16(i16::from_le_bytes(b.try_into().unwrap())),
            0x04 => Value::I32(i32::from_le_bytes(b.try_into().unwrap())),
            0x05 => Value::I64(i64::from_le_bytes(b.try_into().unwrap())),
            0x06 => Value::U8(b[0]),
            0x07 => Value::U16(u16::from_le_bytes(b.try_into().unwrap())),
            0x08 => Value::U32(u32::from_le_bytes(b.try_into().unwrap())),
            0x09 => Value::U64(u64::from_le_bytes(b.try_into().unwrap())),
            0x0A => Value::F32(f32_at(0)),
            0x0B => Value::F64(f64::from_le_bytes(b.try_into().unwrap())),
            0x0C => Value::XY([f32_at(0), f32_at(4)]),
            0x0D => Value::XYZ([f32_at(0), f32_at(4), f32_at(8)]),
            0x10 => Value::Angle(u16::from_le_bytes(b.try_into().unwrap())),
            _ => unreachable!("scalar_size admitted code {code:#x}"),
        })
    }

    fn children(&self, mut p: usize, end: usize, depth: usize) -> Result<Vec<Node>> {
        if depth > MAX_DEPTH {
            bail!("ESF nesting deeper than {MAX_DEPTH} at {p:#x}");
        }
        let d = self.d;
        let mut out = Vec::new();
        while p < end {
            let code = d[p];
            let body = p + 1;
            match code {
                0x80 => {
                    if body + 7 > end {
                        bail!("ESF record header at {p:#x} crosses its parent end {end:#x}");
                    }
                    let name = self.name(rd_u16(d, body)?)?;
                    let version = d[body + 2];
                    let rend = rd_u32(d, body + 3)? as usize;
                    if rend < body + 7 || rend > end {
                        bail!(
                            "ESF record '{name}' at {p:#x} ends at {rend:#x}, outside {:#x}..{end:#x}",
                            body + 7
                        );
                    }
                    let children = self
                        .children(body + 7, rend, depth + 1)
                        .with_context(|| format!("in record '{name}' at {p:#x}"))?;
                    out.push(Node::Record {
                        name,
                        version,
                        children,
                    });
                    p = rend;
                }
                0x81 => {
                    if body + 11 > end {
                        bail!("ESF array header at {p:#x} crosses its parent end {end:#x}");
                    }
                    let name = self.name(rd_u16(d, body)?)?;
                    let version = d[body + 2];
                    let rend = rd_u32(d, body + 3)? as usize;
                    let count = rd_u32(d, body + 7)? as usize;
                    let mut q = body + 11;
                    if rend < q || rend > end {
                        bail!(
                            "ESF array '{name}' at {p:#x} ends at {rend:#x}, outside {q:#x}..{end:#x}"
                        );
                    }
                    if count > (rend - q) / 4 {
                        bail!(
                            "ESF array '{name}' at {p:#x} claims {count} items in {} bytes",
                            rend - q
                        );
                    }
                    let mut items = Vec::with_capacity(count);
                    for i in 0..count {
                        let iend = rd_u32(d, q)
                            .with_context(|| format!("item {i} of array '{name}'"))?
                            as usize;
                        if iend < q + 4 || iend > rend {
                            bail!(
                                "ESF array '{name}' item {i} at {q:#x} ends at {iend:#x}, outside {:#x}..{rend:#x}",
                                q + 4
                            );
                        }
                        items.push(
                            self.children(q + 4, iend, depth + 1).with_context(|| {
                                format!("in item {i} of array '{name}' at {p:#x}")
                            })?,
                        );
                        q = iend;
                    }
                    if q != rend {
                        bail!(
                            "ESF array '{name}' at {p:#x}: items stop at {q:#x} but the array ends at {rend:#x}"
                        );
                    }
                    out.push(Node::Array {
                        name,
                        version,
                        items,
                    });
                    p = rend;
                }
                0x0E => {
                    let n = rd_u16(d, body)? as usize;
                    let raw = take(d, body + 2, n * 2)?;
                    if body + 2 + n * 2 > end {
                        bail!("ESF string at {p:#x} crosses its parent end {end:#x}");
                    }
                    let units: Vec<u16> = raw
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let s = String::from_utf16(&units).map_err(|_| {
                        anyhow!("ESF UTF-16 string at {p:#x} holds an unpaired surrogate")
                    })?;
                    out.push(Node::Value(Value::String16(s)));
                    p = body + 2 + n * 2;
                }
                0x0F => {
                    let n = rd_u16(d, body)? as usize;
                    let raw = take(d, body + 2, n)?;
                    if body + 2 + n > end {
                        bail!("ESF string at {p:#x} crosses its parent end {end:#x}");
                    }
                    out.push(Node::Value(Value::String8(
                        raw.iter().map(|&c| c as char).collect(),
                    )));
                    p = body + 2 + n;
                }
                0x41..=0x4F => {
                    let aend = rd_u32(d, body)? as usize;
                    if aend < body + 4 || aend > end {
                        bail!(
                            "ESF packed array {code:#x} at {p:#x} ends at {aend:#x}, outside {:#x}..{end:#x}",
                            body + 4
                        );
                    }
                    out.push(Node::Value(Value::Array {
                        code,
                        data: d[body + 4..aend].to_vec(),
                    }));
                    p = aend;
                }
                _ if scalar_size(code).is_some() => {
                    let size = scalar_size(code).unwrap();
                    if body + size > end {
                        bail!("ESF value {code:#x} at {p:#x} crosses its parent end {end:#x}");
                    }
                    out.push(Node::Value(self.scalar(code, body)?));
                    p = body + size;
                }
                _ => bail!("unknown ESF type {code:#x} at {p:#x}"),
            }
        }
        if p != end {
            bail!(
                "ESF children overran their container: stopped at {p:#x}, container ends at {end:#x}"
            );
        }
        Ok(out)
    }
}

pub fn read(bytes: &[u8]) -> Result<Esf> {
    let d = bytes;
    let magic = rd_u32(d, 0).context("ESF header")?;
    if magic != MAGIC {
        bail!("unsupported ESF magic {magic:#x}");
    }
    let reserved = rd_u32(d, 4)?;
    if reserved != 0 {
        bail!("unsupported ESF header word {reserved:#x} (writer cannot reproduce it)");
    }
    let timestamp = rd_u32(d, 8)?;
    let names_off = rd_u32(d, 12)? as usize;
    if names_off < 16 {
        bail!("ESF name table offset {names_off:#x} lies inside the header");
    }
    let count = rd_u16(d, names_off).context("ESF name table")? as usize;
    let mut p = names_off + 2;
    let mut names = Vec::with_capacity(count);
    for i in 0..count {
        let n = rd_u16(d, p).with_context(|| format!("ESF name {i}"))? as usize;
        let raw = take(d, p + 2, n).with_context(|| format!("ESF name {i}"))?;
        names.push(raw.iter().map(|&c| c as char).collect::<String>());
        p += 2 + n;
    }
    if p != d.len() {
        bail!(
            "ESF has {} trailing bytes after the name table",
            d.len() - p
        );
    }
    let parser = Parser { d, names };
    let mut children = parser.children(16, names_off, 0)?;
    if children.len() != 1 || !matches!(children[0], Node::Record { .. }) {
        bail!(
            "ESF root must be a single record, found {} top-level nodes",
            children.len()
        );
    }
    Ok(Esf {
        timestamp,
        names: parser.names,
        root: children.remove(0),
    })
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

struct Writer {
    out: Vec<u8>,
    names: Vec<String>,
    index: HashMap<String, u16>,
}

fn latin1(s: &str, what: &str) -> Result<Vec<u8>> {
    s.chars()
        .map(|c| {
            u8::try_from(c as u32)
                .map_err(|_| anyhow!("{what} {s:?} contains {c:?}, which latin1 cannot store"))
        })
        .collect()
}

impl Writer {
    fn new(names: &[String]) -> Result<Self> {
        let mut w = Writer {
            out: Vec::new(),
            names: Vec::with_capacity(names.len()),
            index: HashMap::with_capacity(names.len()),
        };
        // The original table order is kept so that unchanged files round-trip exactly.
        for n in names {
            let id = u16::try_from(w.names.len())
                .map_err(|_| anyhow!("ESF name table exceeds 65535 entries"))?;
            w.names.push(n.clone());
            w.index.entry(n.clone()).or_insert(id);
        }
        Ok(w)
    }

    fn name_id(&mut self, n: &str) -> Result<u16> {
        if let Some(&id) = self.index.get(n) {
            return Ok(id);
        }
        let id = u16::try_from(self.names.len())
            .map_err(|_| anyhow!("ESF name table exceeds 65535 entries (adding '{n}')"))?;
        self.names.push(n.to_owned());
        self.index.insert(n.to_owned(), id);
        Ok(id)
    }

    fn pos32(&self) -> Result<u32> {
        u32::try_from(self.out.len()).map_err(|_| anyhow!("ESF larger than 4 GiB"))
    }

    fn patch(&mut self, at: usize, v: u32) {
        self.out[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn children(&mut self, nodes: &[Node]) -> Result<()> {
        for node in nodes {
            match node {
                Node::Record {
                    name,
                    version,
                    children,
                } => {
                    let id = self.name_id(name)?;
                    self.out.push(0x80);
                    self.out.extend_from_slice(&id.to_le_bytes());
                    self.out.push(*version);
                    let at = self.out.len();
                    self.out.extend_from_slice(&[0; 4]);
                    self.children(children)
                        .with_context(|| format!("in record '{name}'"))?;
                    let end = self.pos32()?;
                    self.patch(at, end);
                }
                Node::Array {
                    name,
                    version,
                    items,
                } => {
                    let id = self.name_id(name)?;
                    self.out.push(0x81);
                    self.out.extend_from_slice(&id.to_le_bytes());
                    self.out.push(*version);
                    let at = self.out.len();
                    self.out.extend_from_slice(&[0; 4]);
                    let count = u32::try_from(items.len())
                        .map_err(|_| anyhow!("array '{name}' has too many items"))?;
                    self.out.extend_from_slice(&count.to_le_bytes());
                    for (i, item) in items.iter().enumerate() {
                        let iat = self.out.len();
                        self.out.extend_from_slice(&[0; 4]);
                        self.children(item)
                            .with_context(|| format!("in item {i} of array '{name}'"))?;
                        let iend = self.pos32()?;
                        self.patch(iat, iend);
                    }
                    let end = self.pos32()?;
                    self.patch(at, end);
                }
                Node::Value(v) => self.value(v)?,
            }
        }
        Ok(())
    }

    fn value(&mut self, v: &Value) -> Result<()> {
        let out = &mut self.out;
        out.push(v.code());
        match v {
            Value::Bool(x) => out.push(*x as u8),
            Value::I8(x) => out.push(*x as u8),
            Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::U8(x) => out.push(*x),
            Value::U16(x) | Value::Angle(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::XY(x) => x
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
            Value::XYZ(x) => x
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
            Value::String16(s) => {
                let units: Vec<u16> = s.encode_utf16().collect();
                let n = u16::try_from(units.len())
                    .map_err(|_| anyhow!("UTF-16 ESF string longer than 65535 units"))?;
                out.extend_from_slice(&n.to_le_bytes());
                units
                    .iter()
                    .for_each(|u| out.extend_from_slice(&u.to_le_bytes()));
            }
            Value::String8(s) => {
                let raw = latin1(s, "ESF string")?;
                let n = u16::try_from(raw.len())
                    .map_err(|_| anyhow!("ESF string longer than 65535 bytes"))?;
                out.extend_from_slice(&n.to_le_bytes());
                out.extend_from_slice(&raw);
            }
            Value::Array { code, data } => {
                if !(0x41..=0x4F).contains(code) {
                    // value() already pushed the code; the error aborts the whole write so no cleanup is needed.
                    bail!("cannot write ESF packed array with type {code:#x}");
                }
                let end = out.len() + 4 + data.len();
                let end = u32::try_from(end).map_err(|_| anyhow!("ESF larger than 4 GiB"))?;
                out.extend_from_slice(&end.to_le_bytes());
                out.extend_from_slice(data);
            }
        }
        Ok(())
    }
}

pub fn write(file: &Esf) -> Result<Vec<u8>> {
    if !matches!(file.root, Node::Record { .. }) {
        bail!("ESF root must be a record");
    }
    let mut w = Writer::new(&file.names)?;
    w.out.extend_from_slice(&[0; 16]);
    w.children(std::slice::from_ref(&file.root))?;
    let names_off = w.pos32()?;
    let count = u16::try_from(w.names.len())
        .map_err(|_| anyhow!("ESF name table exceeds 65535 entries"))?;
    w.out.extend_from_slice(&count.to_le_bytes());
    for n in w.names.clone() {
        let raw = latin1(&n, "ESF node name")?;
        let len = u16::try_from(raw.len())
            .map_err(|_| anyhow!("ESF node name '{n}' longer than 65535 bytes"))?;
        w.out.extend_from_slice(&len.to_le_bytes());
        w.out.extend_from_slice(&raw);
    }
    w.out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    w.out[4..8].copy_from_slice(&0u32.to_le_bytes());
    w.out[8..12].copy_from_slice(&file.timestamp.to_le_bytes());
    w.out[12..16].copy_from_slice(&names_off.to_le_bytes());
    Ok(w.out)
}

/// Encoded size in bytes of one node (a node's size does not depend on the name table: ids are fixed width).
pub fn encoded_size(node: &Node) -> Result<usize> {
    let mut w = Writer::new(&[])?;
    w.children(std::slice::from_ref(node))?;
    Ok(w.out.len())
}

// ---------------------------------------------------------------------------------------------
// Node helpers
// ---------------------------------------------------------------------------------------------

impl Node {
    pub fn record(name: impl Into<String>, version: u8, children: Vec<Node>) -> Self {
        Node::Record {
            name: name.into(),
            version,
            children,
        }
    }

    pub fn array(name: impl Into<String>, version: u8, items: Vec<Vec<Node>>) -> Self {
        Node::Array {
            name: name.into(),
            version,
            items,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Node::Record { name, .. } | Node::Array { name, .. } => Some(name),
            Node::Value(_) => None,
        }
    }

    pub fn version(&self) -> Option<u8> {
        match self {
            Node::Record { version, .. } | Node::Array { version, .. } => Some(*version),
            Node::Value(_) => None,
        }
    }

    pub fn set_version(&mut self, v: u8) -> Result<()> {
        match self {
            Node::Record { version, .. } | Node::Array { version, .. } => {
                *version = v;
                Ok(())
            }
            Node::Value(value) => bail!("a {} value has no version", value.kind()),
        }
    }

    fn describe(&self) -> String {
        match self {
            Node::Record { name, .. } => format!("record '{name}'"),
            Node::Array { name, .. } => format!("record array '{name}'"),
            Node::Value(v) => format!("{} value", v.kind()),
        }
    }

    pub fn children(&self) -> Result<&Vec<Node>> {
        match self {
            Node::Record { children, .. } => Ok(children),
            other => bail!("{} has no direct children", other.describe()),
        }
    }

    pub fn children_mut(&mut self) -> Result<&mut Vec<Node>> {
        match self {
            Node::Record { children, .. } => Ok(children),
            other => bail!("{} has no direct children", other.describe()),
        }
    }

    pub fn items(&self) -> Result<&Vec<Vec<Node>>> {
        match self {
            Node::Array { items, .. } => Ok(items),
            other => bail!("{} has no items", other.describe()),
        }
    }

    pub fn items_mut(&mut self) -> Result<&mut Vec<Vec<Node>>> {
        match self {
            Node::Array { items, .. } => Ok(items),
            other => bail!("{} has no items", other.describe()),
        }
    }

    /// First record or record-array child named `name`.
    pub fn get(&self, name: &str) -> Result<&Node> {
        self.children()?
            .iter()
            .find(|c| c.name() == Some(name))
            .ok_or_else(|| anyhow!("{} has no child '{name}'", self.describe()))
    }

    pub fn get_mut(&mut self, name: &str) -> Result<&mut Node> {
        let what = self.describe();
        self.children_mut()?
            .iter_mut()
            .find(|c| c.name() == Some(name))
            .ok_or_else(|| anyhow!("{what} has no child '{name}'"))
    }

    pub fn path(&self, names: &[&str]) -> Result<&Node> {
        let mut cur = self;
        for n in names {
            cur = cur.get(n).with_context(|| format!("path {names:?}"))?;
        }
        Ok(cur)
    }

    pub fn path_mut(&mut self, names: &[&str]) -> Result<&mut Node> {
        let mut cur = self;
        for n in names {
            cur = cur.get_mut(n).with_context(|| format!("path {names:?}"))?;
        }
        Ok(cur)
    }

    /// All record/array children named `name`, in order.
    pub fn all(&self, name: &str) -> Result<Vec<&Node>> {
        Ok(self
            .children()?
            .iter()
            .filter(|c| c.name() == Some(name))
            .collect())
    }

    /// The scalar children of a record, in order (Python `vals`).
    pub fn values(&self) -> Result<Vec<&Value>> {
        Ok(self
            .children()?
            .iter()
            .filter_map(|c| {
                if let Node::Value(v) = c {
                    Some(v)
                } else {
                    None
                }
            })
            .collect())
    }

    /// The `index`-th scalar child of a record (Python `vals(node)[index]`).
    pub fn value_at(&self, index: usize) -> Result<&Value> {
        self.values()?.get(index).copied().ok_or_else(|| {
            anyhow!(
                "{} has fewer than {} scalar children",
                self.describe(),
                index + 1
            )
        })
    }

    pub fn value(&self) -> Result<&Value> {
        match self {
            Node::Value(v) => Ok(v),
            other => bail!("{} is not a scalar value", other.describe()),
        }
    }

    pub fn value_mut(&mut self) -> Result<&mut Value> {
        match self {
            Node::Value(v) => Ok(v),
            other => bail!("{} is not a scalar value", other.describe()),
        }
    }

    /// Visit every record and record array depth-first (pre-order); `path` holds the enclosing record names
    /// (array levels appear as `NAME[]`).
    pub fn walk<'a>(&'a self, f: &mut dyn FnMut(&'a Node, &[String])) {
        fn go<'a>(n: &'a Node, path: &mut Vec<String>, f: &mut dyn FnMut(&'a Node, &[String])) {
            match n {
                Node::Record { name, children, .. } => {
                    f(n, path);
                    path.push(name.clone());
                    children.iter().for_each(|c| go(c, path, f));
                    path.pop();
                }
                Node::Array { name, items, .. } => {
                    f(n, path);
                    path.push(format!("{name}[]"));
                    items.iter().flatten().for_each(|c| go(c, path, f));
                    path.pop();
                }
                Node::Value(_) => {}
            }
        }
        go(self, &mut Vec::new(), f);
    }

    pub fn as_u8(&self) -> Result<u8> {
        match self.value()? {
            Value::U8(v) => Ok(*v),
            other => bail!("expected u8, found {}", other.kind()),
        }
    }

    /// u8/u16/u32 widened losslessly.
    pub fn as_u32(&self) -> Result<u32> {
        match self.value()? {
            Value::U8(v) => Ok(*v as u32),
            Value::U16(v) => Ok(*v as u32),
            Value::U32(v) => Ok(*v),
            other => bail!("expected u32, found {}", other.kind()),
        }
    }

    pub fn as_u64(&self) -> Result<u64> {
        match self.value()? {
            Value::U8(v) => Ok(*v as u64),
            Value::U16(v) => Ok(*v as u64),
            Value::U32(v) => Ok(*v as u64),
            Value::U64(v) => Ok(*v),
            other => bail!("expected u64, found {}", other.kind()),
        }
    }

    /// i8/i16/i32 widened losslessly.
    pub fn as_i32(&self) -> Result<i32> {
        match self.value()? {
            Value::I8(v) => Ok(*v as i32),
            Value::I16(v) => Ok(*v as i32),
            Value::I32(v) => Ok(*v),
            other => bail!("expected i32, found {}", other.kind()),
        }
    }

    pub fn as_f32(&self) -> Result<f32> {
        match self.value()? {
            Value::F32(v) => Ok(*v),
            other => bail!("expected f32, found {}", other.kind()),
        }
    }

    pub fn as_str(&self) -> Result<&str> {
        match self.value()? {
            Value::String16(s) | Value::String8(s) => Ok(s),
            other => bail!("expected string, found {}", other.kind()),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self.value()? {
            Value::Bool(v) => Ok(*v),
            other => bail!("expected bool, found {}", other.kind()),
        }
    }

    pub fn as_xy(&self) -> Result<[f32; 2]> {
        match self.value()? {
            Value::XY(v) => Ok(*v),
            other => bail!("expected xy, found {}", other.kind()),
        }
    }

    pub fn as_xyz(&self) -> Result<[f32; 3]> {
        match self.value()? {
            Value::XYZ(v) => Ok(*v),
            other => bail!("expected xyz, found {}", other.kind()),
        }
    }

    /// ESF type code of a scalar/packed-array value; `None` for records and record arrays.
    pub fn code(&self) -> Option<u8> {
        match self {
            Node::Value(v) => Some(v.code()),
            _ => None,
        }
    }

    pub fn raw_array(&self) -> Result<&[u8]> {
        match self.value()? {
            Value::Array { data, .. } => Ok(data),
            other => bail!("expected packed array, found {}", other.kind()),
        }
    }

    fn typed_array(&self, code: u8, width: usize) -> Result<&[u8]> {
        match self.value()? {
            Value::Array { code: c, data } if *c == code => {
                if data.len() % width != 0 {
                    bail!(
                        "packed array {code:#x} holds {} bytes, not a multiple of {width}",
                        data.len()
                    );
                }
                Ok(data)
            }
            Value::Array { code: c, .. } => bail!("expected packed array {code:#x}, found {c:#x}"),
            other => bail!("expected packed array {code:#x}, found {}", other.kind()),
        }
    }

    pub fn u32_array(&self) -> Result<Vec<u32>> {
        Ok(self
            .typed_array(0x48, 4)?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn i32_array(&self) -> Result<Vec<i32>> {
        Ok(self
            .typed_array(0x44, 4)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn f32_array(&self) -> Result<Vec<f32>> {
        Ok(self
            .typed_array(0x4A, 4)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn xy_array(&self) -> Result<Vec<[f32; 2]>> {
        Ok(self
            .typed_array(0x4C, 8)?
            .chunks_exact(8)
            .map(|c| {
                [
                    f32::from_le_bytes(c[0..4].try_into().unwrap()),
                    f32::from_le_bytes(c[4..8].try_into().unwrap()),
                ]
            })
            .collect())
    }

    pub fn xyz_array(&self) -> Result<Vec<[f32; 3]>> {
        Ok(self
            .typed_array(0x4D, 12)?
            .chunks_exact(12)
            .map(|c| {
                [
                    f32::from_le_bytes(c[0..4].try_into().unwrap()),
                    f32::from_le_bytes(c[4..8].try_into().unwrap()),
                    f32::from_le_bytes(c[8..12].try_into().unwrap()),
                ]
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------------------------
// Constructors (mirror the Python `b/i32/u32/u8/f32/s16/xy/arr_*` helpers)
// ---------------------------------------------------------------------------------------------

pub fn b(v: bool) -> Node {
    Node::Value(Value::Bool(v))
}
pub fn u8v(v: u8) -> Node {
    Node::Value(Value::U8(v))
}
pub fn u16v(v: u16) -> Node {
    Node::Value(Value::U16(v))
}
pub fn i32v(v: i32) -> Node {
    Node::Value(Value::I32(v))
}
pub fn u32v(v: u32) -> Node {
    Node::Value(Value::U32(v))
}
pub fn i64v(v: i64) -> Node {
    Node::Value(Value::I64(v))
}
pub fn u64v(v: u64) -> Node {
    Node::Value(Value::U64(v))
}
pub fn f32v(v: f32) -> Node {
    Node::Value(Value::F32(v))
}
pub fn f64v(v: f64) -> Node {
    Node::Value(Value::F64(v))
}
pub fn xyv(x: f32, y: f32) -> Node {
    Node::Value(Value::XY([x, y]))
}
pub fn xyzv(x: f32, y: f32, z: f32) -> Node {
    Node::Value(Value::XYZ([x, y, z]))
}
pub fn angle(v: u16) -> Node {
    Node::Value(Value::Angle(v))
}
pub fn s16(v: impl Into<String>) -> Node {
    Node::Value(Value::String16(v.into()))
}
pub fn s8(v: impl Into<String>) -> Node {
    Node::Value(Value::String8(v.into()))
}

fn packed(code: u8, data: Vec<u8>) -> Node {
    Node::Value(Value::Array { code, data })
}

pub fn arr_u32(v: &[u32]) -> Node {
    packed(0x48, v.iter().flat_map(|x| x.to_le_bytes()).collect())
}
pub fn arr_i32(v: &[i32]) -> Node {
    packed(0x44, v.iter().flat_map(|x| x.to_le_bytes()).collect())
}
pub fn arr_f32(v: &[f32]) -> Node {
    packed(0x4A, v.iter().flat_map(|x| x.to_le_bytes()).collect())
}
pub fn arr_xy(v: &[[f32; 2]]) -> Node {
    packed(
        0x4C,
        v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect(),
    )
}
pub fn arr_xyz(v: &[[f32; 3]]) -> Node {
    packed(
        0x4D,
        v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Esf {
        let scalars = vec![
            b(true),
            b(false),
            Node::Value(Value::I8(-5)),
            Node::Value(Value::I16(-300)),
            i32v(-70000),
            i64v(-1 << 40),
            u8v(200),
            u16v(60000),
            u32v(0xDEAD_BEEF),
            u64v(1 << 50),
            f32v(1.5),
            f64v(-2.25),
            xyv(1.0, -2.0),
            xyzv(1.0, 2.0, 3.0),
            s16("Gr\u{fc}n \u{1F600}"),
            s8("caf\u{e9}"),
            angle(1234),
            arr_u32(&[1, 2, 3]),
            arr_xy(&[[0.5, 1.5]]),
            Node::Value(Value::Array {
                code: 0x46,
                data: vec![],
            }),
        ];
        let item = |n: u32| vec![u32v(n), Node::record("INNER", 3, vec![s16("x")])];
        let root = Node::record(
            "ROOT",
            7,
            vec![
                Node::record("SCALARS", 1, scalars),
                Node::array("ITEMS", 2, vec![item(1), vec![], item(3)]),
                Node::array("EMPTY", 0, vec![]),
            ],
        );
        // "UNUSED" is deliberately in the table but never referenced: it must survive a round trip.
        Esf {
            timestamp: 0x5E1F_0001,
            names: vec!["UNUSED".into(), "ROOT".into()],
            root,
        }
    }

    #[test]
    fn round_trips_every_value_type_byte_for_byte() {
        let original = sample();
        let bytes = write(&original).unwrap();
        let parsed = read(&bytes).unwrap();
        assert_eq!(parsed.timestamp, original.timestamp);
        assert_eq!(parsed.root, original.root);
        assert_eq!(&parsed.names[..2], &["UNUSED", "ROOT"]);
        assert_eq!(write(&parsed).unwrap(), bytes);
    }

    #[test]
    fn every_truncation_is_an_error_never_a_panic() {
        let bytes = write(&sample()).unwrap();
        for cut in 0..bytes.len() {
            assert!(
                read(&bytes[..cut]).is_err(),
                "prefix of {cut} bytes must be rejected"
            );
        }
    }

    #[test]
    fn every_single_byte_corruption_is_handled_without_panic() {
        let bytes = write(&sample()).unwrap();
        for i in 0..bytes.len() {
            for flip in [0x01u8, 0x80, 0xFF] {
                let mut bad = bytes.clone();
                bad[i] ^= flip;
                if let Ok(parsed) = read(&bad) {
                    // Whatever still parses must also be writable without panicking.
                    let _ = write(&parsed);
                }
            }
        }
    }

    #[test]
    fn rejects_bad_header_and_trailing_data() {
        let bytes = write(&sample()).unwrap();
        let mut bad_magic = bytes.clone();
        bad_magic[0] = 0xCF;
        assert!(read(&bad_magic).is_err());
        let mut reserved = bytes.clone();
        reserved[4] = 1;
        assert!(read(&reserved).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(read(&trailing).is_err());
        let mut inside_header = bytes.clone();
        inside_header[12..16].copy_from_slice(&8u32.to_le_bytes());
        assert!(read(&inside_header).is_err());
    }

    #[test]
    fn rejects_name_index_out_of_range_and_bad_container_ends() {
        let esf = Esf {
            timestamp: 1,
            names: vec![],
            root: Node::record("R", 0, vec![Node::array("A", 0, vec![vec![u32v(9)]])]),
        };
        let bytes = write(&esf).unwrap();
        // Record name id lives at offset 17 (16 header + 1 code byte).
        let mut bad_name = bytes.clone();
        bad_name[17..19].copy_from_slice(&99u16.to_le_bytes());
        assert!(read(&bad_name).is_err());
        // Root record end (offset 20..24) beyond the name table offset.
        let mut bad_end = bytes.clone();
        let off = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) + 1;
        bad_end[20..24].copy_from_slice(&off.to_le_bytes());
        assert!(read(&bad_end).is_err());
        // Root record end too small to hold its own header.
        let mut tiny_end = bytes.clone();
        tiny_end[20..24].copy_from_slice(&18u32.to_le_bytes());
        assert!(read(&tiny_end).is_err());
    }

    #[test]
    fn rejects_unknown_type_non_boolean_and_array_item_count_lies() {
        let esf = Esf {
            timestamp: 1,
            names: vec![],
            root: Node::record("R", 0, vec![b(true)]),
        };
        let bytes = write(&esf).unwrap();
        let value_at = 24; // code byte of the bool
        assert_eq!(bytes[value_at], 0x01);
        let mut unknown = bytes.clone();
        unknown[value_at] = 0x20;
        assert!(read(&unknown).is_err());
        let mut not_bool = bytes.clone();
        not_bool[value_at + 1] = 2;
        assert!(read(&not_bool).is_err());

        let esf = Esf {
            timestamp: 1,
            names: vec![],
            root: Node::record("R", 0, vec![Node::array("A", 0, vec![vec![]])]),
        };
        let bytes = write(&esf).unwrap();
        // Array count lives 11 bytes after the array code (code, name u16, version, end u32 -> count at +8).
        let count_at = 24 + 1 + 2 + 1 + 4;
        let mut lie = bytes.clone();
        lie[count_at..count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read(&lie).is_err());
    }

    #[test]
    fn writer_rejects_unwritable_data() {
        let mk = |v: Node| Esf {
            timestamp: 0,
            names: vec![],
            root: Node::record("R", 0, vec![v]),
        };
        assert!(
            write(&mk(Node::Value(Value::Array {
                code: 0x08,
                data: vec![]
            })))
            .is_err()
        );
        assert!(write(&mk(s8("snow\u{2603}"))).is_err());
        assert!(write(&mk(s8("x".repeat(70000)))).is_err());
        assert!(
            write(&Esf {
                timestamp: 0,
                names: vec![],
                root: u32v(1)
            })
            .is_err()
        );
        let deep_nodes = (0..10).fold(u32v(1), |inner, _| Node::record("N", 0, vec![inner]));
        assert!(
            write(&Esf {
                timestamp: 0,
                names: vec![],
                root: deep_nodes
            })
            .is_ok()
        );
    }

    #[test]
    fn rejects_pathological_nesting() {
        let nested =
            (0..MAX_DEPTH + 10).fold(u32v(1), |inner, _| Node::record("N", 0, vec![inner]));
        let bytes = write(&Esf {
            timestamp: 0,
            names: vec![],
            root: nested,
        })
        .unwrap();
        assert!(read(&bytes).is_err());
    }

    #[test]
    fn navigation_and_typed_accessors() {
        let mut esf = sample();
        let items = esf.root.path(&["ITEMS"]).unwrap().items().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0][0].as_u32().unwrap(), 1);
        assert!(esf.root.path(&["ITEMS", "INNER"]).is_err());
        let scalars = esf.root.get("SCALARS").unwrap();
        assert_eq!(scalars.value_at(0).unwrap(), &Value::Bool(true));
        assert_eq!(
            scalars.children().unwrap()[17].u32_array().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            scalars.children().unwrap()[18].xy_array().unwrap(),
            vec![[0.5, 1.5]]
        );
        assert!(scalars.children().unwrap()[17].i32_array().is_err());
        assert_eq!(
            scalars.children().unwrap()[14].as_str().unwrap(),
            "Gr\u{fc}n \u{1F600}"
        );
        assert!(scalars.children().unwrap()[14].as_u32().is_err());
        esf.root
            .path_mut(&["SCALARS"])
            .unwrap()
            .set_version(9)
            .unwrap();
        assert_eq!(esf.root.get("SCALARS").unwrap().version(), Some(9));
        assert!(u32v(1).set_version(1).is_err());
        let mut seen = 0;
        esf.root.walk(&mut |_, _| seen += 1);
        assert_eq!(seen, 1 + 1 + 1 + 2 + 1); // ROOT, SCALARS, ITEMS(+2 INNER records), EMPTY
    }

    #[test]
    fn encoded_size_matches_written_node() {
        let esf = sample();
        let node = esf.root.get("ITEMS").unwrap();
        let alone = Esf {
            timestamp: 0,
            names: vec![],
            root: Node::record("W", 0, vec![node.clone()]),
        };
        let whole = write(&alone).unwrap();
        let name_table: usize = 2 + alone_names(&whole);
        assert_eq!(
            whole.len(),
            16 + (1 + 2 + 1 + 4) + encoded_size(node).unwrap() + name_table
        );
    }

    fn alone_names(bytes: &[u8]) -> usize {
        let off = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        bytes.len() - off - 2
    }
}
