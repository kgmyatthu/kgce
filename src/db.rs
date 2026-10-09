//! Native Total War database codec: RPFM RON schema reader, Empire/Napoleon DB table codec,
//! `.loc` codec and typed access to the vanilla tables of an imported snapshot.
//!
//! Everything here is strict: truncated data, trailing bytes, invalid booleans, invalid UTF-8/UTF-16,
//! unsupported field types and rows that do not match the schema are errors, never silently repaired.

use crate::{
    model::Rules,
    snapshot::{Game, Snapshot},
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Number, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
    time::SystemTime,
};

pub const GUID_MARKER: [u8; 4] = [0xfd, 0xfe, 0xfc, 0xff];
pub const VERSION_MARKER: [u8; 4] = [0xfc, 0xfd, 0xfe, 0xff];
pub const LOC_MAGIC: [u8; 6] = [0xff, 0xfe, b'L', b'O', b'C', 0];

// ---------------------------------------------------------------------------------------------
// RON reader (the subset RPFM schemas use, but a real recursive-descent parser: structs, tuples,
// lists, maps, enum variants with payload, strings with escapes, numbers, comments)
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Ron {
    Ident(String),
    Int(i64),
    Float,
    Str(String),
    Seq(Vec<Ron>),
    Map(Vec<(Ron, Ron)>),
    Tuple(Vec<Ron>),
    Struct(Vec<(String, Ron)>),
    Variant(String, Box<Ron>),
}

struct RonParser<'a> {
    s: &'a [u8],
    p: usize,
}

impl<'a> RonParser<'a> {
    fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        let start = if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
            3
        } else {
            0
        };
        Self { s: bytes, p: start }
    }

    fn fail<T>(&self, msg: &str) -> Result<T> {
        let upto = self.p.min(self.s.len());
        let line = self.s[..upto].iter().filter(|&&c| c == b'\n').count() + 1;
        bail!("RON schema syntax error at line {line}: {msg}")
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.p).copied()
    }

    fn ws(&mut self) -> Result<()> {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => self.p += 1,
                Some(b'/') => match self.s.get(self.p + 1).copied() {
                    Some(b'/') => {
                        while let Some(c) = self.peek() {
                            if c == b'\n' {
                                break;
                            }
                            self.p += 1;
                        }
                    }
                    Some(b'*') => {
                        self.p += 2;
                        let mut depth = 1;
                        while depth > 0 {
                            let a = self.peek();
                            let b = self.s.get(self.p + 1).copied();
                            match (a, b) {
                                (Some(b'*'), Some(b'/')) => {
                                    depth -= 1;
                                    self.p += 2;
                                }
                                (Some(b'/'), Some(b'*')) => {
                                    depth += 1;
                                    self.p += 2;
                                }
                                (Some(_), _) => self.p += 1,
                                (None, _) => return self.fail("unterminated block comment"),
                            }
                        }
                    }
                    _ => return Ok(()),
                },
                _ => return Ok(()),
            }
        }
    }

    fn ident(&mut self) -> Result<String> {
        let start = self.p;
        match self.peek() {
            Some(c) if c == b'_' || c.is_ascii_alphabetic() => {}
            _ => return self.fail("expected identifier"),
        }
        while matches!(self.peek(), Some(c) if c == b'_' || c.is_ascii_alphanumeric()) {
            self.p += 1;
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.p]).into_owned())
    }

    fn looks_like_field(&self) -> bool {
        let s = self.s;
        let mut q = self.p;
        if !matches!(s.get(q), Some(c) if *c == b'_' || c.is_ascii_alphabetic()) {
            return false;
        }
        while matches!(s.get(q), Some(c) if *c == b'_' || c.is_ascii_alphanumeric()) {
            q += 1;
        }
        while matches!(s.get(q), Some(c) if c.is_ascii_whitespace()) {
            q += 1;
        }
        s.get(q) == Some(&b':')
    }

    fn string(&mut self) -> Result<String> {
        // positioned on the opening quote
        self.p += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => return self.fail("unterminated string"),
            };
            self.p += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = match self.peek() {
                        Some(e) => e,
                        None => return self.fail("unterminated escape"),
                    };
                    self.p += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'\'' => out.push(b'\''),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'0' => out.push(0),
                        b'u' => {
                            if self.peek() != Some(b'{') {
                                return self.fail("expected '{' after \\u");
                            }
                            self.p += 1;
                            let start = self.p;
                            while matches!(self.peek(), Some(h) if h.is_ascii_hexdigit()) {
                                self.p += 1;
                            }
                            let hex = String::from_utf8_lossy(&self.s[start..self.p]).into_owned();
                            if self.peek() != Some(b'}') {
                                return self.fail("expected '}' closing \\u escape");
                            }
                            self.p += 1;
                            let code = u32::from_str_radix(&hex, 16)
                                .map_err(|_| anyhow!("bad \\u escape"))?;
                            let ch = char::from_u32(code)
                                .ok_or_else(|| anyhow!("bad \\u escape value {code:x}"))?;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return self.fail("unknown string escape"),
                    }
                }
                other => out.push(other),
            }
        }
        String::from_utf8(out).map_err(|e| anyhow!("invalid UTF-8 in RON string: {e}"))
    }

    fn raw_string(&mut self) -> Result<String> {
        // positioned on 'r'
        self.p += 1;
        let mut hashes = 0usize;
        while self.peek() == Some(b'#') {
            hashes += 1;
            self.p += 1;
        }
        if self.peek() != Some(b'"') {
            return self.fail("malformed raw string");
        }
        self.p += 1;
        let start = self.p;
        loop {
            match self.peek() {
                None => return self.fail("unterminated raw string"),
                Some(b'"') => {
                    let end = self.p;
                    let mut q = self.p + 1;
                    let mut n = 0;
                    while n < hashes && self.s.get(q) == Some(&b'#') {
                        n += 1;
                        q += 1;
                    }
                    if n == hashes {
                        self.p = q;
                        return String::from_utf8(self.s[start..end].to_vec())
                            .map_err(|e| anyhow!("invalid UTF-8 in RON raw string: {e}"));
                    }
                    self.p += 1;
                }
                Some(_) => self.p += 1,
            }
        }
    }

    fn number(&mut self) -> Result<Ron> {
        let start = self.p;
        self.p += 1;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'+' || c == b'-')
        {
            self.p += 1;
        }
        let text: String = String::from_utf8_lossy(&self.s[start..self.p])
            .chars()
            .filter(|&c| c != '_')
            .collect();
        let body = text.trim_start_matches(['+', '-']);
        if body.starts_with("0x") || body.starts_with("0X") {
            let value = i64::from_str_radix(&body[2..], 16)
                .map_err(|_| anyhow!("bad hex number {text}"))?;
            return Ok(Ron::Int(if text.starts_with('-') { -value } else { value }));
        }
        if text.contains(['.', 'e', 'E']) {
            text.parse::<f64>()
                .map_err(|_| anyhow!("bad float {text}"))?;
            Ok(Ron::Float)
        } else {
            let v: i64 = text.parse().map_err(|_| anyhow!("bad integer {text}"))?;
            Ok(Ron::Int(v))
        }
    }

    fn value(&mut self) -> Result<Ron> {
        self.ws()?;
        match self.peek() {
            None => self.fail("unexpected end of input"),
            Some(b'"') => Ok(Ron::Str(self.string()?)),
            Some(b'[') => {
                self.p += 1;
                Ok(Ron::Seq(self.items(b']')?))
            }
            Some(b'{') => {
                self.p += 1;
                self.map()
            }
            Some(b'(') => {
                self.p += 1;
                self.paren()
            }
            Some(c) if c.is_ascii_digit() || c == b'-' || c == b'+' => self.number(),
            Some(c) if c == b'_' || c.is_ascii_alphabetic() => {
                if c == b'r' && matches!(self.s.get(self.p + 1), Some(b'"') | Some(b'#')) {
                    return Ok(Ron::Str(self.raw_string()?));
                }
                let name = self.ident()?;
                if self.peek() == Some(b'(') {
                    self.p += 1;
                    let inner = self.paren()?;
                    Ok(Ron::Variant(name, Box::new(inner)))
                } else {
                    Ok(Ron::Ident(name))
                }
            }
            Some(c) => self.fail(&format!("unexpected character {:?}", c as char)),
        }
    }

    fn items(&mut self, close: u8) -> Result<Vec<Ron>> {
        let mut out = Vec::new();
        loop {
            self.ws()?;
            if self.peek() == Some(close) {
                self.p += 1;
                return Ok(out);
            }
            out.push(self.value()?);
            self.ws()?;
            match self.peek() {
                Some(b',') => self.p += 1,
                Some(c) if c == close => {}
                _ => return self.fail("expected ',' or closing bracket"),
            }
        }
    }

    fn paren(&mut self) -> Result<Ron> {
        self.ws()?;
        if self.peek() == Some(b')') {
            self.p += 1;
            return Ok(Ron::Tuple(Vec::new()));
        }
        if self.looks_like_field() {
            let mut fields = Vec::new();
            loop {
                self.ws()?;
                if self.peek() == Some(b')') {
                    self.p += 1;
                    return Ok(Ron::Struct(fields));
                }
                let name = self.ident()?;
                self.ws()?;
                if self.peek() != Some(b':') {
                    return self.fail("expected ':' after field name");
                }
                self.p += 1;
                let v = self.value()?;
                fields.push((name, v));
                self.ws()?;
                match self.peek() {
                    Some(b',') => self.p += 1,
                    Some(b')') => {}
                    _ => return self.fail("expected ',' or ')' after struct field"),
                }
            }
        }
        Ok(Ron::Tuple(self.items(b')')?))
    }

    fn map(&mut self) -> Result<Ron> {
        let mut out = Vec::new();
        loop {
            self.ws()?;
            if self.peek() == Some(b'}') {
                self.p += 1;
                return Ok(Ron::Map(out));
            }
            let key = self.value()?;
            self.ws()?;
            if self.peek() != Some(b':') {
                return self.fail("expected ':' in map entry");
            }
            self.p += 1;
            let value = self.value()?;
            out.push((key, value));
            self.ws()?;
            match self.peek() {
                Some(b',') => self.p += 1,
                Some(b'}') => {}
                _ => return self.fail("expected ',' or '}' after map entry"),
            }
        }
    }

    fn document(&mut self) -> Result<Ron> {
        let v = self.value()?;
        self.ws()?;
        if self.p != self.s.len() {
            return self.fail("trailing data after schema");
        }
        Ok(v)
    }
}

fn ron_field<'a>(fields: &'a [(String, Ron)], name: &str) -> Result<&'a Ron> {
    fields
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v)
        .ok_or_else(|| anyhow!("schema struct has no field `{name}`"))
}

fn ron_struct(v: &Ron) -> Result<&[(String, Ron)]> {
    match v {
        Ron::Struct(f) => Ok(f),
        other => bail!("expected a RON struct, got {other:?}"),
    }
}

fn ron_str(v: &Ron) -> Result<&str> {
    match v {
        Ron::Str(s) => Ok(s),
        other => bail!("expected a RON string, got {other:?}"),
    }
}

fn ron_bool(v: &Ron) -> Result<bool> {
    match v {
        Ron::Ident(s) if s == "true" => Ok(true),
        Ron::Ident(s) if s == "false" => Ok(false),
        other => bail!("expected a RON bool, got {other:?}"),
    }
}

fn ron_reference(v: &Ron) -> Result<Option<(String, String)>> {
    match v {
        Ron::Ident(s) if s == "None" => Ok(None),
        Ron::Variant(name, payload) if name == "Some" => {
            if let Ron::Tuple(outer) = payload.as_ref() {
                if let [Ron::Tuple(pair)] = outer.as_slice() {
                    if let [Ron::Str(table), Ron::Str(field)] = pair.as_slice() {
                        return Ok(Some((table.clone(), field.clone())));
                    }
                }
            }
            bail!("unsupported is_reference payload {payload:?}")
        }
        other => bail!("unsupported is_reference value {other:?}"),
    }
}

/// Optional RPFM attributes (`is_bitwise`, `enum_values`, `is_part_of_colour`); absent attributes are neutral.
fn ron_column_meta(fs: &[(String, Ron)]) -> Result<ColumnMeta> {
    let mut meta = ColumnMeta::default();
    if let Some((_, v)) = fs.iter().find(|(n, _)| n == "is_bitwise") {
        meta.bitwise = match v {
            Ron::Int(n) if *n >= 0 => {
                u32::try_from(*n).map_err(|_| anyhow!("is_bitwise {n} too large"))?
            }
            other => bail!("unsupported is_bitwise value {other:?}"),
        };
    }
    if let Some((_, v)) = fs.iter().find(|(n, _)| n == "enum_values") {
        meta.has_enum = match v {
            Ron::Map(entries) => !entries.is_empty(),
            other => bail!("unsupported enum_values value {other:?}"),
        };
    }
    if let Some((_, v)) = fs.iter().find(|(n, _)| n == "is_part_of_colour") {
        meta.colour_group = match v {
            Ron::Ident(s) if s == "None" => None,
            Ron::Variant(name, payload) if name == "Some" => match payload.as_ref() {
                Ron::Tuple(items) => match items.as_slice() {
                    [Ron::Int(n)] => Some(
                        u8::try_from(*n)
                            .map_err(|_| anyhow!("is_part_of_colour {n} out of range"))?,
                    ),
                    _ => bail!("unsupported is_part_of_colour payload {payload:?}"),
                },
                other => bail!("unsupported is_part_of_colour payload {other:?}"),
            },
            other => bail!("unsupported is_part_of_colour value {other:?}"),
        };
    }
    Ok(meta)
}

// ---------------------------------------------------------------------------------------------
// Schema model
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum FieldType {
    Boolean,
    I16,
    I32,
    I64,
    F32,
    F64,
    ColourRgb,
    StringU8,
    StringU16,
    OptionalStringU8,
    OptionalStringU16,
    /// Types the codec cannot read or write (sequences, unknown blobs). Using such a table is an error.
    Unsupported(String),
}

impl FieldType {
    fn from_ron(v: &Ron) -> Self {
        match v {
            Ron::Ident(name) => match name.as_str() {
                "Boolean" => Self::Boolean,
                "I16" => Self::I16,
                "I32" => Self::I32,
                "I64" => Self::I64,
                "F32" => Self::F32,
                "F64" => Self::F64,
                "ColourRGB" => Self::ColourRgb,
                "StringU8" => Self::StringU8,
                "StringU16" => Self::StringU16,
                "OptionalStringU8" => Self::OptionalStringU8,
                "OptionalStringU16" => Self::OptionalStringU16,
                other => Self::Unsupported(other.to_owned()),
            },
            Ron::Variant(name, _) => Self::Unsupported(name.clone()),
            other => Self::Unsupported(format!("{other:?}")),
        }
    }

    pub fn is_optional(&self) -> bool {
        matches!(self, Self::OptionalStringU8 | Self::OptionalStringU16)
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub ty: FieldType,
    pub is_key: bool,
    /// `(table, field)` exactly as written in the schema (table name without `_tables`).
    pub reference: Option<(String, String)>,
}

/// Schema attributes of one field that RPFM's table processing (TSV layout) depends on but the binary codec ignores.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColumnMeta {
    /// `is_bitwise`: when above 1 RPFM expands the integer into that many boolean columns.
    pub bitwise: u32,
    /// `enum_values` is not empty: RPFM presents the integer as its enum name.
    pub has_enum: bool,
    /// `is_part_of_colour`: one channel of an RGB triplet that RPFM merges into a single hex column.
    pub colour_group: Option<u8>,
}

#[derive(Clone, Debug)]
pub struct Definition {
    pub version: i64,
    pub fields: Vec<Field>,
    /// One entry per field of `fields`, in the same order.
    pub columns: Vec<ColumnMeta>,
}

#[derive(Debug)]
pub struct Schema {
    pub version: i64,
    definitions: HashMap<String, Vec<Definition>>,
}

type SchemaStamp = (u64, Option<SystemTime>);
static SCHEMA_CACHE: LazyLock<Mutex<HashMap<PathBuf, (SchemaStamp, Arc<Schema>)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

impl Schema {
    pub fn parse(text: &str) -> Result<Self> {
        let root = RonParser::new(text).document()?;
        let top = ron_struct(&root).context("schema root")?;
        let version = match ron_field(top, "version")? {
            Ron::Int(v) => *v,
            other => bail!("schema version must be an integer, got {other:?}"),
        };
        let defs = match ron_field(top, "definitions")? {
            Ron::Map(entries) => entries,
            other => bail!("schema `definitions` must be a map, got {other:?}"),
        };
        let mut definitions = HashMap::with_capacity(defs.len());
        for (key, list) in defs {
            let table = ron_str(key)?.to_owned();
            let list = match list {
                Ron::Seq(items) => items,
                other => bail!("definitions of {table} must be a list, got {other:?}"),
            };
            let mut parsed = Vec::with_capacity(list.len());
            for def in list {
                let s = ron_struct(def).with_context(|| format!("definition of {table}"))?;
                let version = match ron_field(s, "version")? {
                    Ron::Int(v) => *v,
                    other => bail!("{table}: definition version must be an integer, got {other:?}"),
                };
                let fields_ron = match ron_field(s, "fields")? {
                    Ron::Seq(items) => items,
                    other => bail!("{table} v{version}: `fields` must be a list, got {other:?}"),
                };
                let mut fields = Vec::with_capacity(fields_ron.len());
                let mut columns = Vec::with_capacity(fields_ron.len());
                for f in fields_ron {
                    let fs = ron_struct(f).with_context(|| format!("{table} v{version} field"))?;
                    let name = ron_str(ron_field(fs, "name")?)?.to_owned();
                    let ty = FieldType::from_ron(ron_field(fs, "field_type")?);
                    let is_key = ron_bool(ron_field(fs, "is_key")?)
                        .with_context(|| format!("{table}.{name} is_key"))?;
                    let reference = ron_reference(ron_field(fs, "is_reference")?)
                        .with_context(|| format!("{table}.{name} is_reference"))?;
                    let meta = ron_column_meta(fs)
                        .with_context(|| format!("{table} v{version} field {name}"))?;
                    fields.push(Field {
                        name,
                        ty,
                        is_key,
                        reference,
                    });
                    columns.push(meta);
                }
                parsed.push(Definition {
                    version,
                    fields,
                    columns,
                });
            }
            if definitions.insert(table.clone(), parsed).is_some() {
                bail!("schema defines table {table} twice");
            }
        }
        Ok(Self {
            version,
            definitions,
        })
    }

    /// Parses (once per file stamp) and caches the schema at `path`.
    pub fn load(path: &Path) -> Result<Arc<Schema>> {
        let meta = fs::metadata(path).with_context(|| format!("Stat schema {}", path.display()))?;
        let stamp: SchemaStamp = (meta.len(), meta.modified().ok());
        let cache = &*SCHEMA_CACHE;
        if let Some((have, schema)) = cache
            .lock()
            .map_err(|_| anyhow!("schema cache poisoned"))?
            .get(path)
        {
            if *have == stamp {
                return Ok(schema.clone());
            }
        }
        let text =
            fs::read_to_string(path).with_context(|| format!("Read schema {}", path.display()))?;
        let schema = Arc::new(
            Schema::parse(&text).with_context(|| format!("Parse schema {}", path.display()))?,
        );
        cache
            .lock()
            .map_err(|_| anyhow!("schema cache poisoned"))?
            .insert(path.to_path_buf(), (stamp, schema.clone()));
        Ok(schema)
    }

    /// All definitions (every version) of `table` (name including `_tables`).
    pub fn definitions(&self, table: &str) -> &[Definition] {
        self.definitions
            .get(table)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn definition(&self, table: &str, version: i64) -> Option<&Definition> {
        self.definitions(table)
            .iter()
            .find(|d| d.version == version)
    }

    pub fn table_names(&self) -> impl Iterator<Item = &String> {
        self.definitions.keys()
    }
}

pub fn normalize_table_name(name: &str) -> String {
    if name.ends_with("_tables") {
        name.to_owned()
    } else {
        format!("{name}_tables")
    }
}

pub fn schema(rules: &Rules, game: Game) -> Result<Arc<Schema>> {
    let file = match game {
        Game::Ntw => "schema_nap.ron",
        Game::Etw => "schema_emp.ron",
    };
    Schema::load(&rules.root.join(file))
}

// ---------------------------------------------------------------------------------------------
// Table model and codec
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    Bool(bool),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Colour(u32),
    Str(String),
    Null,
}

impl Cell {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Cell::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Stable textual identity used for keys / reference checks (type-insensitive across integer widths).
    pub fn key(&self) -> String {
        match self {
            Cell::Bool(v) => v.to_string(),
            Cell::I16(v) => v.to_string(),
            Cell::I32(v) => v.to_string(),
            Cell::I64(v) => v.to_string(),
            Cell::Colour(v) => v.to_string(),
            Cell::F32(v) => format!("{:08x}", v.to_bits()),
            Cell::F64(v) => format!("{:016x}", v.to_bits()),
            Cell::Str(s) => s.clone(),
            Cell::Null => String::new(),
        }
    }

    pub fn to_json(&self) -> Result<Value> {
        Ok(match self {
            Cell::Bool(v) => Value::Bool(*v),
            Cell::I16(v) => Value::from(*v),
            Cell::I32(v) => Value::from(*v),
            Cell::I64(v) => Value::from(*v),
            Cell::Colour(v) => Value::from(*v),
            Cell::F32(v) => Value::Number(
                Number::from_f64(f64::from(*v))
                    .ok_or_else(|| anyhow!("non-finite F32 value cannot be represented"))?,
            ),
            Cell::F64(v) => Value::Number(
                Number::from_f64(*v)
                    .ok_or_else(|| anyhow!("non-finite F64 value cannot be represented"))?,
            ),
            Cell::Str(s) => Value::String(s.clone()),
            Cell::Null => Value::Null,
        })
    }

    pub fn from_json(ty: &FieldType, v: &Value) -> Result<Cell> {
        fn int(v: &Value, what: &str) -> Result<i64> {
            match v {
                Value::Number(n) => n
                    .as_i64()
                    .ok_or_else(|| anyhow!("expected {what}, got {n}")),
                other => bail!("expected {what}, got {other}"),
            }
        }
        Ok(match ty {
            FieldType::Boolean => match v {
                Value::Bool(b) => Cell::Bool(*b),
                other => bail!("expected boolean, got {other}"),
            },
            FieldType::I16 => Cell::I16(
                i16::try_from(int(v, "i16 integer")?)
                    .map_err(|_| anyhow!("{v} is out of range for I16"))?,
            ),
            FieldType::I32 => Cell::I32(
                i32::try_from(int(v, "i32 integer")?)
                    .map_err(|_| anyhow!("{v} is out of range for I32"))?,
            ),
            FieldType::I64 => Cell::I64(int(v, "i64 integer")?),
            FieldType::ColourRgb => match v {
                Value::Number(n) => Cell::Colour(
                    n.as_u64()
                        .and_then(|x| u32::try_from(x).ok())
                        .ok_or_else(|| anyhow!("{n} is out of range for ColourRGB"))?,
                ),
                other => bail!("expected colour integer, got {other}"),
            },
            FieldType::F32 => match v {
                Value::Number(n) => {
                    let x = n
                        .as_f64()
                        .ok_or_else(|| anyhow!("expected number, got {n}"))?;
                    let y = x as f32;
                    if !y.is_finite() {
                        bail!("{n} is out of range for F32");
                    }
                    Cell::F32(y)
                }
                other => bail!("expected number, got {other}"),
            },
            FieldType::F64 => match v {
                Value::Number(n) => Cell::F64(
                    n.as_f64()
                        .ok_or_else(|| anyhow!("expected number, got {n}"))?,
                ),
                other => bail!("expected number, got {other}"),
            },
            FieldType::StringU8 | FieldType::StringU16 => match v {
                Value::String(s) => Cell::Str(s.clone()),
                other => bail!("expected string, got {other}"),
            },
            FieldType::OptionalStringU8 | FieldType::OptionalStringU16 => match v {
                Value::Null => Cell::Null,
                Value::String(s) => Cell::Str(s.clone()),
                other => bail!("expected string or null, got {other}"),
            },
            FieldType::Unsupported(name) => bail!("unsupported field type {name}"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Table {
    /// Table name including `_tables`.
    pub name: String,
    pub guid: Option<String>,
    pub version: u32,
    /// Whether the binary carried an explicit version marker (kept for byte-exact roundtrips).
    pub version_marker: bool,
    /// The byte between header and row count (always 1 in shipped tables); preserved verbatim.
    pub flag: u8,
    pub fields: Vec<Field>,
    pub rows: Vec<Vec<Cell>>,
}

impl Table {
    pub fn column(&self, name: &str) -> Result<usize> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| anyhow!("table {} has no column `{name}`", self.name))
    }

    pub fn field_names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    /// Rows as JSON objects keyed by field name in schema order.
    pub fn json_rows(&self) -> Result<Vec<Value>> {
        let mut out = Vec::with_capacity(self.rows.len());
        for (i, row) in self.rows.iter().enumerate() {
            let mut obj = Map::with_capacity(self.fields.len());
            for (f, c) in self.fields.iter().zip(row) {
                obj.insert(
                    f.name.clone(),
                    c.to_json()
                        .with_context(|| format!("{} row {i} field {}", self.name, f.name))?,
                );
            }
            out.push(Value::Object(obj));
        }
        Ok(out)
    }

    /// Builds a table from JSON row objects; the key set of every row must equal the schema field set.
    pub fn from_json_rows(
        name: &str,
        version: u32,
        fields: &[Field],
        rows: &[Value],
    ) -> Result<Table> {
        let mut out = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            let obj = row
                .as_object()
                .with_context(|| format!("{name} row {i} is not an object"))?;
            let missing: Vec<&str> = fields
                .iter()
                .filter(|f| !obj.contains_key(&f.name))
                .map(|f| f.name.as_str())
                .collect();
            let extra: Vec<&str> = obj
                .keys()
                .filter(|k| !fields.iter().any(|f| &f.name == *k))
                .map(String::as_str)
                .collect();
            if !missing.is_empty() || !extra.is_empty() {
                bail!(
                    "{name} row {i}: fields do not match schema v{version} (missing {missing:?}, unknown {extra:?})"
                );
            }
            let mut cells = Vec::with_capacity(fields.len());
            for f in fields {
                let v = obj
                    .get(f.name.as_str())
                    .ok_or_else(|| anyhow!("{name} row {i}: missing {}", f.name))?;
                cells.push(
                    Cell::from_json(&f.ty, v)
                        .with_context(|| format!("{name} row {i} field {}", f.name))?,
                );
            }
            out.push(cells);
        }
        Ok(Table {
            name: name.to_owned(),
            guid: None,
            version,
            version_marker: version != 0,
            flag: 1,
            fields: fields.to_vec(),
            rows: out,
        })
    }
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let d = self.d;
        let end = self
            .p
            .checked_add(n)
            .filter(|&e| e <= d.len())
            .ok_or_else(|| {
                anyhow!(
                    "unexpected end of data at offset {} (need {n} more bytes, have {})",
                    self.p,
                    d.len() - self.p.min(d.len())
                )
            })?;
        let b = &d[self.p..end];
        self.p = end;
        Ok(b)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let b = self.take(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(b);
        Ok(a)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
}

fn decode_utf16(bytes: &[u8]) -> Result<String> {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    String::from_utf16(&units).map_err(|e| anyhow!("invalid UTF-16 text: {e}"))
}

fn encode_utf16_into(out: &mut Vec<u8>, s: &str) {
    for u in s.encode_utf16() {
        out.extend_from_slice(&u.to_le_bytes());
    }
}

fn read_string(r: &mut Reader, wide: bool) -> Result<String> {
    let n = r.u16()? as usize;
    if wide {
        decode_utf16(r.take(n * 2)?)
    } else {
        String::from_utf8(r.take(n)?.to_vec()).map_err(|e| anyhow!("invalid UTF-8 string: {e}"))
    }
}

fn read_cell(r: &mut Reader, ty: &FieldType) -> Result<Cell> {
    Ok(match ty {
        FieldType::Boolean => match r.u8()? {
            0 => Cell::Bool(false),
            1 => Cell::Bool(true),
            b => bail!("invalid boolean byte {b:#04x}"),
        },
        FieldType::I16 => Cell::I16(i16::from_le_bytes(r.array()?)),
        FieldType::I32 => Cell::I32(i32::from_le_bytes(r.array()?)),
        FieldType::I64 => Cell::I64(i64::from_le_bytes(r.array()?)),
        FieldType::F32 => Cell::F32(f32::from_le_bytes(r.array()?)),
        FieldType::F64 => Cell::F64(f64::from_le_bytes(r.array()?)),
        FieldType::ColourRgb => Cell::Colour(r.u32()?),
        FieldType::StringU8 => Cell::Str(read_string(r, false)?),
        FieldType::StringU16 => Cell::Str(read_string(r, true)?),
        FieldType::OptionalStringU8 | FieldType::OptionalStringU16 => match r.u8()? {
            0 => Cell::Null,
            1 => Cell::Str(read_string(r, matches!(ty, FieldType::OptionalStringU16))?),
            b => bail!("invalid optional-string flag {b:#04x}"),
        },
        FieldType::Unsupported(name) => bail!("unsupported field type {name}"),
    })
}

fn write_string(out: &mut Vec<u8>, s: &str, wide: bool) -> Result<()> {
    if wide {
        let n = s.encode_utf16().count();
        if n > 0xffff {
            bail!("string of {n} UTF-16 units does not fit a StringU16");
        }
        out.extend_from_slice(&(n as u16).to_le_bytes());
        encode_utf16_into(out, s);
    } else {
        if s.len() > 0xffff {
            bail!("string of {} bytes does not fit a StringU8", s.len());
        }
        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    Ok(())
}

fn write_cell(out: &mut Vec<u8>, ty: &FieldType, cell: &Cell) -> Result<()> {
    match (ty, cell) {
        (FieldType::Boolean, Cell::Bool(b)) => out.push(u8::from(*b)),
        (FieldType::I16, Cell::I16(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (FieldType::I32, Cell::I32(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (FieldType::I64, Cell::I64(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (FieldType::F32, Cell::F32(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (FieldType::F64, Cell::F64(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (FieldType::ColourRgb, Cell::Colour(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (FieldType::StringU8, Cell::Str(s)) => write_string(out, s, false)?,
        (FieldType::StringU16, Cell::Str(s)) => write_string(out, s, true)?,
        (FieldType::OptionalStringU8, Cell::Null) | (FieldType::OptionalStringU16, Cell::Null) => {
            out.push(0)
        }
        (FieldType::OptionalStringU8, Cell::Str(s)) => {
            out.push(1);
            write_string(out, s, false)?;
        }
        (FieldType::OptionalStringU16, Cell::Str(s)) => {
            out.push(1);
            write_string(out, s, true)?;
        }
        (FieldType::Unsupported(name), _) => bail!("unsupported field type {name}"),
        (ty, cell) => bail!("cell {cell:?} does not match field type {ty:?}"),
    }
    Ok(())
}

/// Decodes one DB table file. `name` includes `_tables`.
pub fn decode(name: &str, data: &[u8], schema: &Schema) -> Result<Table> {
    let mut r = Reader { d: data, p: 0 };
    let mut guid = None;
    let mut version = 0u32;
    let mut version_marker = false;
    if data.starts_with(&GUID_MARKER) {
        r.p = 4;
        let n = r.u16()? as usize;
        guid = Some(decode_utf16(r.take(n * 2)?).with_context(|| format!("{name}: GUID"))?);
    }
    if data[r.p..].starts_with(&VERSION_MARKER) {
        r.p += 4;
        version = r.u32()?;
        version_marker = true;
    }
    let flag = r.u8().with_context(|| format!("{name}: header"))?;
    let count = r.u32().with_context(|| format!("{name}: row count"))? as usize;
    let start = r.p;
    let all = schema.definitions(name);
    if all.is_empty() {
        bail!("no schema definition for {name}");
    }
    let candidates: Vec<&Definition> = all
        .iter()
        .filter(|d| d.version == i64::from(version))
        .collect();
    if candidates.is_empty() {
        let have: Vec<i64> = all.iter().map(|d| d.version).collect();
        bail!("no schema definition for {name} version {version} (schema has {have:?})");
    }
    let mut last = String::new();
    for def in candidates {
        if def.fields.is_empty() && count > 0 {
            last = "definition has no fields".into();
            continue;
        }
        r.p = start;
        let mut rows: Vec<Vec<Cell>> = Vec::with_capacity(count.min(data.len()));
        let mut failure = None;
        for i in 0..count {
            let mut row = Vec::with_capacity(def.fields.len());
            let mut bad = None;
            for f in &def.fields {
                match read_cell(&mut r, &f.ty) {
                    Ok(c) => row.push(c),
                    Err(e) => {
                        bad = Some(format!("row {i} field {}: {e:#}", f.name));
                        break;
                    }
                }
            }
            if let Some(b) = bad {
                failure = Some(b);
                break;
            }
            rows.push(row);
        }
        match failure {
            Some(f) => last = f,
            None if r.p == data.len() => {
                return Ok(Table {
                    name: name.to_owned(),
                    guid,
                    version,
                    version_marker,
                    flag,
                    fields: def.fields.clone(),
                    rows,
                });
            }
            None => last = format!("{} trailing bytes", data.len() - r.p),
        }
    }
    bail!("{name} v{version}: no schema definition fits the data ({last})")
}

pub fn encode(t: &Table) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    if let Some(g) = &t.guid {
        let units = g.encode_utf16().count();
        if units > 0xffff {
            bail!("{}: GUID too long", t.name);
        }
        out.extend_from_slice(&GUID_MARKER);
        out.extend_from_slice(&(units as u16).to_le_bytes());
        encode_utf16_into(&mut out, g);
    }
    if t.version_marker || t.version != 0 {
        out.extend_from_slice(&VERSION_MARKER);
        out.extend_from_slice(&t.version.to_le_bytes());
    }
    out.push(t.flag);
    let count = u32::try_from(t.rows.len()).map_err(|_| anyhow!("{}: too many rows", t.name))?;
    out.extend_from_slice(&count.to_le_bytes());
    for (i, row) in t.rows.iter().enumerate() {
        if row.len() != t.fields.len() {
            bail!(
                "{} row {i}: {} cells for {} fields",
                t.name,
                row.len(),
                t.fields.len()
            );
        }
        for (f, c) in t.fields.iter().zip(row) {
            write_cell(&mut out, &f.ty, c)
                .with_context(|| format!("{} row {i} field {}", t.name, f.name))?;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// LOC codec: FF FE 'L' 'O' 'C' 00, u32 version, u32 count, then per entry
// u16 n + UTF-16LE key, u16 n + UTF-16LE text, u8 tooltip flag.
// ---------------------------------------------------------------------------------------------

pub type LocEntry = (String, String, u8);

pub fn decode_loc(bytes: &[u8]) -> Result<(u32, Vec<LocEntry>)> {
    if !bytes.starts_with(&LOC_MAGIC) {
        bail!("not a LOC file (bad magic)");
    }
    let mut r = Reader {
        d: bytes,
        p: LOC_MAGIC.len(),
    };
    let version = r.u32()?;
    let count = r.u32()? as usize;
    let mut out = Vec::with_capacity(count.min(bytes.len() / 5));
    for i in 0..count {
        let entry = (|| -> Result<LocEntry> {
            let kn = r.u16()? as usize;
            let key = decode_utf16(r.take(kn * 2)?)?;
            let tn = r.u16()? as usize;
            let text = decode_utf16(r.take(tn * 2)?)?;
            let tip = r.u8()?;
            Ok((key, text, tip))
        })()
        .with_context(|| format!("LOC entry {i} of {count}"))?;
        out.push(entry);
    }
    if r.p != bytes.len() {
        bail!(
            "LOC has {} trailing bytes after {count} entries",
            bytes.len() - r.p
        );
    }
    Ok((version, out))
}

pub fn encode_loc(entries: &[LocEntry], version: u32) -> Result<Vec<u8>> {
    let count = u32::try_from(entries.len()).map_err(|_| anyhow!("too many LOC entries"))?;
    let mut out = Vec::with_capacity(14 + entries.len() * 64);
    out.extend_from_slice(&LOC_MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    for (key, text, tip) in entries {
        for (what, s) in [("key", key), ("text", text)] {
            let n = s.encode_utf16().count();
            if n > 0xffff {
                bail!("LOC {what} too long ({n} UTF-16 units): {key}");
            }
            out.extend_from_slice(&(n as u16).to_le_bytes());
            encode_utf16_into(&mut out, s);
        }
        out.push(*tip);
    }
    Ok(out)
}

/// key -> (text, tooltip flag). A repeated key keeps its last occurrence (the engine's dictionary semantics).
pub fn parse_loc(bytes: &[u8]) -> Result<BTreeMap<String, (String, u8)>> {
    let (_, entries) = decode_loc(bytes)?;
    Ok(entries.into_iter().map(|(k, t, f)| (k, (t, f))).collect())
}

// ---------------------------------------------------------------------------------------------
// Typed access to the vanilla tables inside an imported snapshot
// ---------------------------------------------------------------------------------------------

fn game_index(game: Game) -> u8 {
    match game {
        Game::Etw => 0,
        Game::Ntw => 1,
    }
}

/// Reads a snapshot member given a relative data path with either separator.
pub fn read_member(snapshot: &Snapshot, game: Game, rel: &str) -> Result<Vec<u8>> {
    let forward = rel.replace('\\', "/");
    match snapshot.read(game, &forward) {
        Ok(bytes) => Ok(bytes),
        Err(first) => {
            let back = rel.replace('/', "\\");
            snapshot.read(game, &back).map_err(|second| {
                anyhow!("cannot read {rel} from the source snapshot: {first:#}; {second:#}")
            })
        }
    }
}

/// Snapshot members of table `name` (`db/<name>_tables/*`), sorted by member name.
pub fn table_files(snapshot: &Snapshot, game: Game, name: &str) -> Vec<String> {
    let t = normalize_table_name(name);
    let mut names = snapshot.names(game, &format!("db/{t}/"));
    if names.is_empty() {
        names = snapshot.names(game, &format!("db\\{t}\\"));
    }
    names.retain(|n| !n.ends_with('/') && !n.ends_with('\\'));
    names.sort();
    names
}

fn fnv(mut h: u64, data: &[u8]) -> u64 {
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

type TableCacheKey = (PathBuf, u8, String);
static TABLE_CACHE: LazyLock<Mutex<HashMap<TableCacheKey, (u64, Arc<Table>)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Typed vanilla table (all member files concatenated in member-name order); `Ok(None)` when the snapshot
/// holds no file for the table. Files must agree on version and fields. Results are cached by content hash.
pub fn load_table_opt(
    snapshot: &Snapshot,
    game: Game,
    rules: &Rules,
    name: &str,
) -> Result<Option<Arc<Table>>> {
    let tname = normalize_table_name(name);
    let files = table_files(snapshot, game, &tname);
    if files.is_empty() {
        return Ok(None);
    }
    let mut blobs = Vec::with_capacity(files.len());
    let mut fingerprint = 0xcbf2_9ce4_8422_2325u64;
    for f in &files {
        if f.rsplit(['/', '\\']).next() == Some("nap_world") {
            bail!(
                "source snapshot contains generated pack member {f}; only untouched vanilla data may be used as source"
            );
        }
        let bytes = read_member(snapshot, game, f)?;
        fingerprint = fnv(fnv(fingerprint, f.as_bytes()), &bytes);
        blobs.push(bytes);
    }
    let key: TableCacheKey = (rules.root.clone(), game_index(game), tname.clone());
    let cache = &*TABLE_CACHE;
    if let Some((have, t)) = cache
        .lock()
        .map_err(|_| anyhow!("table cache poisoned"))?
        .get(&key)
    {
        if *have == fingerprint {
            return Ok(Some(t.clone()));
        }
    }
    let schema = schema(rules, game)?;
    let mut merged: Option<Table> = None;
    for (file, bytes) in files.iter().zip(&blobs) {
        let t = decode(&tname, bytes, &schema).with_context(|| format!("Decode {file}"))?;
        match merged.as_mut() {
            None => merged = Some(t),
            Some(m) => {
                if m.version != t.version || m.field_names() != t.field_names() {
                    bail!(
                        "{tname}: member {file} has version {} / different fields than the first member (version {})",
                        t.version,
                        m.version
                    );
                }
                m.rows.extend(t.rows);
            }
        }
    }
    let table = Arc::new(merged.ok_or_else(|| anyhow!("{tname}: no member decoded"))?);
    cache
        .lock()
        .map_err(|_| anyhow!("table cache poisoned"))?
        .insert(key, (fingerprint, table.clone()));
    Ok(Some(table))
}

pub fn load_table(
    snapshot: &Snapshot,
    game: Game,
    rules: &Rules,
    name: &str,
) -> Result<Arc<Table>> {
    load_table_opt(snapshot, game, rules, name)?.ok_or_else(|| {
        anyhow!(
            "table {} is not present in the source snapshot",
            normalize_table_name(name)
        )
    })
}

/// The tables of the target's NTW database that the conversion reads directly (as opposed to the tables it adds rows
/// to, and the tables those rows reference). Every campaign consumer goes through [`load_campaign_table`] with one of
/// these, so the list of direct reads, the reason shown to the user and the requirement check cannot drift apart.
/// Generic schema / foreign-key / generated-row infrastructure keeps using the name-based loaders.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CampaignTable {
    /// Faction keys and identity fields (key, name, flags, colours, settler rebellion faction).
    Factions,
    /// Region keys (key only).
    Regions,
    /// Unit keys (key only).
    Units,
    /// Land unit keys and `men` (strength).
    LandStats,
    /// Naval unit keys and crew counts (`num_marines`, `num_seamen`, `num_gunners`).
    NavalStats,
    /// Building level keys (`level_name`) and their `chain`.
    BuildingLevels,
    /// Building chain -> slot type buildability (`chain`, `slot`).
    BuildingSlots,
    /// Technology keys (key only).
    Technologies,
    /// Slot art templates (`culture`, `slot`, `template_model`, `template_model_art_quality_zero`).
    SlotArt,
    /// Exclusive unit / faction permissions the starting armies' availability is derived from.
    UnitPermissions,
}

impl CampaignTable {
    /// Every direct campaign input, once.
    pub const ALL: &'static [CampaignTable] = &[
        CampaignTable::Factions,
        CampaignTable::Regions,
        CampaignTable::Units,
        CampaignTable::LandStats,
        CampaignTable::NavalStats,
        CampaignTable::BuildingLevels,
        CampaignTable::BuildingSlots,
        CampaignTable::Technologies,
        CampaignTable::SlotArt,
        CampaignTable::UnitPermissions,
    ];

    /// Canonical table name (with `_tables`).
    pub fn name(self) -> &'static str {
        match self {
            CampaignTable::Factions => "factions_tables",
            CampaignTable::Regions => "regions_tables",
            CampaignTable::Units => "units_tables",
            CampaignTable::LandStats => "unit_stats_land_tables",
            CampaignTable::NavalStats => "unit_stats_naval_tables",
            CampaignTable::BuildingLevels => "building_levels_tables",
            CampaignTable::BuildingSlots => "building_chain_to_slots_tables",
            CampaignTable::Technologies => "technologies_tables",
            CampaignTable::SlotArt => "slots_art_tables",
            CampaignTable::UnitPermissions => "units_to_exclusive_faction_permissions_tables",
        }
    }

    /// Why the converter needs the table of the target, shown to the user.
    pub fn reason(self) -> &'static str {
        match self {
            CampaignTable::Factions => {
                "faction keys and identity: owner choices, faction mappings, start-position faction records"
            }
            CampaignTable::Regions => "region keys the generated start position refers to",
            CampaignTable::Units => {
                "unit keys: unit mappings, start armies, permission rows, start-position references"
            }
            CampaignTable::LandStats => {
                "land unit strength (men) written into the start-position armies"
            }
            CampaignTable::NavalStats => {
                "ship crew strength written into the start-position fleets"
            }
            CampaignTable::BuildingLevels => "building level keys of the building mapping",
            CampaignTable::BuildingSlots => {
                "building chain to slot validation of the building mapping"
            }
            CampaignTable::Technologies => {
                "technology keys checked in the generated start position"
            }
            CampaignTable::SlotArt => "slot art templates of the generated campaign map",
            CampaignTable::UnitPermissions => {
                "unit availability used to derive permissions for starting armies"
            }
        }
    }
}

/// A direct campaign input of the target (cached like [`load_table`]; same error when the target holds no file).
pub fn load_campaign_table(
    snapshot: &Snapshot,
    rules: &Rules,
    table: CampaignTable,
) -> Result<Arc<Table>> {
    load_table(snapshot, Game::Ntw, rules, table.name())
}

/// Vanilla table rows as JSON objects keyed by field name. `name` may omit `_tables`.
pub fn table(snapshot: &Snapshot, game: Game, rules: &Rules, name: &str) -> Result<Vec<Value>> {
    load_table(snapshot, game, rules, name)?.json_rows()
}

/// `(version, field names in schema order)` of a vanilla table.
pub fn table_meta(
    snapshot: &Snapshot,
    game: Game,
    rules: &Rules,
    name: &str,
) -> Result<(u32, Vec<String>)> {
    let t = load_table(snapshot, game, rules, name)?;
    Ok((t.version, t.field_names()))
}

/// Canonical `db/<table>_tables` directory names of the snapshot with their member files
/// (canonical `db/<table>_tables/<member>` paths, sorted).
pub fn table_directory(snapshot: &Snapshot, game: Game) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in snapshot.names(game, "db/") {
        let mut parts = path.splitn(3, '/');
        if let (Some("db"), Some(table), Some(member)) = (parts.next(), parts.next(), parts.next())
        {
            if !member.is_empty() && !member.contains('/') {
                out.entry(table.to_owned()).or_default().push(path.clone());
            }
        }
    }
    out
}

/// One DB member file decoded on its own.
#[derive(Clone, Debug)]
pub struct Member {
    /// Canonical `db/<table>_tables/<member>` path.
    pub path: String,
    pub table: Table,
    /// SHA-256 of the member bytes.
    pub sha256: String,
}

/// Decodes every member file of table `name` separately (no concatenation), in member-name order.
pub fn load_members(
    snapshot: &Snapshot,
    game: Game,
    rules: &Rules,
    name: &str,
) -> Result<Vec<Member>> {
    let tname = normalize_table_name(name);
    let schema = schema(rules, game)?;
    let mut out = Vec::new();
    for path in table_files(snapshot, game, &tname) {
        let bytes = read_member(snapshot, game, &path)?;
        let table = decode(&tname, &bytes, &schema).with_context(|| format!("Decode {path}"))?;
        let sha256 = {
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(&bytes))
        };
        out.push(Member {
            path,
            table,
            sha256,
        });
    }
    Ok(out)
}

/// Type-insensitive textual identity of a whole row (every cell).
pub fn row_signature(row: &[Cell]) -> String {
    row.iter().map(Cell::key).collect::<Vec<_>>().join("\u{1f}")
}

// ---------------------------------------------------------------------------------------------
// Tests (not run by the editor; `cargo test` exercises them)
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = r#"(
    version: 5,
    definitions: {
        "demo_tables": [
            (
                version: 2,
                fields: [
                    (name: "key", field_type: StringU8, is_key: true, default_value: None, is_filename: false, filename_relative_path: None, is_reference: None, lookup: None, description: "a \"quoted\" // not a comment", ca_order: -1, is_bitwise: 0, enum_values: {}, is_part_of_colour: None),
                    (name: "flag", field_type: Boolean, is_key: false, default_value: None, is_filename: false, filename_relative_path: None, is_reference: Some(("other", "key")), lookup: None, description: "", ca_order: -1, is_bitwise: 0, enum_values: {}, is_part_of_colour: None),
                    (name: "n", field_type: I32, is_key: false, default_value: None, is_filename: false, filename_relative_path: None, is_reference: None, lookup: None, description: "", ca_order: -1, is_bitwise: 0, enum_values: {}, is_part_of_colour: None),
                    (name: "f", field_type: F32, is_key: false, default_value: None, is_filename: false, filename_relative_path: None, is_reference: None, lookup: None, description: "", ca_order: -1, is_bitwise: 0, enum_values: {}, is_part_of_colour: None),
                    (name: "wide", field_type: StringU16, is_key: false, default_value: None, is_filename: false, filename_relative_path: None, is_reference: None, lookup: None, description: "", ca_order: -1, is_bitwise: 0, enum_values: {}, is_part_of_colour: None),
                    (name: "opt", field_type: OptionalStringU8, is_key: false, default_value: None, is_filename: false, filename_relative_path: None, is_reference: None, lookup: None, description: "", ca_order: -1, is_bitwise: 0, enum_values: {}, is_part_of_colour: None),
                    (name: "opt16", field_type: OptionalStringU16, is_key: false, default_value: None, is_filename: false, filename_relative_path: None, is_reference: None, lookup: None, description: "", ca_order: -1, is_bitwise: 0, enum_values: {}, is_part_of_colour: None),
                ],
                localised_fields: [],
                localised_key_order: [],
            ),
        ],
    },
    patches: {},
)"#;

    fn demo_rows() -> Vec<Value> {
        vec![
            serde_json::json!({"key": "alpha", "flag": true, "n": -7, "f": 0.5, "wide": "Gr\u{fc}\u{df}e \u{1f600}", "opt": null, "opt16": "x"}),
            serde_json::json!({"key": "", "flag": false, "n": 2147483647, "f": -1.25, "wide": "", "opt": "yes", "opt16": null}),
        ]
    }

    fn demo_table(schema: &Schema) -> Table {
        let def = schema.definition("demo_tables", 2).unwrap();
        Table::from_json_rows("demo_tables", 2, &def.fields, &demo_rows()).unwrap()
    }

    #[test]
    fn schema_parses_fields_references_and_escapes() {
        let schema = Schema::parse(SCHEMA).unwrap();
        assert_eq!(schema.version, 5);
        let def = schema.definition("demo_tables", 2).unwrap();
        assert_eq!(def.fields.len(), 7);
        assert!(def.fields[0].is_key);
        assert_eq!(
            def.fields[1].reference,
            Some(("other".to_owned(), "key".to_owned()))
        );
        assert_eq!(def.fields[5].ty, FieldType::OptionalStringU8);
        assert!(schema.definition("demo_tables", 3).is_none());
    }

    #[test]
    fn schema_keeps_rpfm_column_attributes_and_tolerates_their_absence() {
        let schema = Schema::parse(
            r#"(version: 5, definitions: { "c_tables": [ (version: 0, fields: [
                (name: "a", field_type: I32, is_key: false, is_reference: None),
                (name: "x_r", field_type: I32, is_key: false, is_reference: None, is_bitwise: 0, enum_values: {}, is_part_of_colour: Some(7)),
                (name: "flags", field_type: I32, is_key: false, is_reference: None, is_bitwise: 3, enum_values: {0: "a", 1: "b"}, is_part_of_colour: None),
            ], localised_fields: [], localised_key_order: []) ] }, patches: {})"#,
        )
        .unwrap();
        let def = schema.definition("c_tables", 0).unwrap();
        assert_eq!(def.columns.len(), def.fields.len());
        assert_eq!(def.columns[0], ColumnMeta::default());
        assert_eq!(def.columns[1].colour_group, Some(7));
        assert_eq!((def.columns[2].bitwise, def.columns[2].has_enum), (3, true));
        assert!(Schema::parse(
            r#"(version: 5, definitions: { "c_tables": [ (version: 0, fields: [
                (name: "a", field_type: I32, is_key: false, is_reference: None, is_part_of_colour: Some(300)),
            ]) ] }, patches: {})"#
        )
        .is_err());
    }

    #[test]
    fn schema_rejects_garbage() {
        assert!(Schema::parse("(version: 1, definitions: {").is_err());
        assert!(Schema::parse("(version: 1, definitions: {}) trailing").is_err());
        assert!(Schema::parse("(definitions: {})").is_err());
    }

    #[test]
    fn table_roundtrip_is_byte_exact_and_preserves_header() {
        let schema = Schema::parse(SCHEMA).unwrap();
        let mut t = demo_table(&schema);
        t.guid = Some("{1234-ABCD}".into());
        let bytes = encode(&t).unwrap();
        assert!(bytes.starts_with(&GUID_MARKER));
        let back = decode("demo_tables", &bytes, &schema).unwrap();
        assert_eq!(back.guid.as_deref(), Some("{1234-ABCD}"));
        assert_eq!(back.version, 2);
        assert_eq!(back.rows, t.rows);
        assert_eq!(encode(&back).unwrap(), bytes);
        let json = back.json_rows().unwrap();
        assert_eq!(json, demo_rows());
    }

    #[test]
    fn every_truncation_of_a_table_is_rejected() {
        let schema = Schema::parse(SCHEMA).unwrap();
        let bytes = encode(&demo_table(&schema)).unwrap();
        for cut in 0..bytes.len() {
            assert!(
                decode("demo_tables", &bytes[..cut], &schema).is_err(),
                "truncation at {cut} was accepted"
            );
        }
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(
            decode("demo_tables", &extended, &schema).is_err(),
            "trailing byte accepted"
        );
    }

    #[test]
    fn invalid_bool_and_unknown_version_are_rejected() {
        let schema = Schema::parse(SCHEMA).unwrap();
        let mut bytes = encode(&demo_table(&schema)).unwrap();
        // header: 4 marker + 4 version + 1 flag + 4 count, then row 0: key "alpha" (2 + 5) then the boolean
        let bool_at = 13 + 2 + 5;
        assert_eq!(bytes[bool_at], 1);
        bytes[bool_at] = 2;
        assert!(decode("demo_tables", &bytes, &schema).is_err());
        let mut wrong_version = encode(&demo_table(&schema)).unwrap();
        wrong_version[4] = 9;
        assert!(decode("demo_tables", &wrong_version, &schema).is_err());
    }

    #[test]
    fn json_rows_must_match_schema_exactly() {
        let schema = Schema::parse(SCHEMA).unwrap();
        let def = schema.definition("demo_tables", 2).unwrap();
        let mut rows = demo_rows();
        rows[0].as_object_mut().unwrap().remove("n");
        assert!(Table::from_json_rows("demo_tables", 2, &def.fields, &rows).is_err());
        let mut rows = demo_rows();
        rows[0]["n"] = serde_json::json!("seven");
        assert!(Table::from_json_rows("demo_tables", 2, &def.fields, &rows).is_err());
        let mut rows = demo_rows();
        rows[0]["flag"] = serde_json::json!(1);
        assert!(Table::from_json_rows("demo_tables", 2, &def.fields, &rows).is_err());
        let mut rows = demo_rows();
        rows[0]["n"] = serde_json::json!(4_294_967_296i64);
        assert!(Table::from_json_rows("demo_tables", 2, &def.fields, &rows).is_err());
        let mut rows = demo_rows();
        rows[0]["extra"] = serde_json::json!(1);
        assert!(Table::from_json_rows("demo_tables", 2, &def.fields, &rows).is_err());
    }

    #[test]
    fn loc_roundtrip_and_truncation() {
        let entries: Vec<LocEntry> = vec![
            ("a_key".into(), "Hello".into(), 0),
            (
                "b_key".into(),
                "Gr\u{fc}\u{df}e \u{1f600}\u{4e2d}".into(),
                1,
            ),
            ("empty".into(), String::new(), 0),
        ];
        let bytes = encode_loc(&entries, 1).unwrap();
        assert_eq!(&bytes[..6], &LOC_MAGIC);
        let (ver, back) = decode_loc(&bytes).unwrap();
        assert_eq!(ver, 1);
        assert_eq!(back, entries);
        assert_eq!(encode_loc(&back, ver).unwrap(), bytes);
        for cut in 0..bytes.len() {
            assert!(
                decode_loc(&bytes[..cut]).is_err(),
                "LOC truncation at {cut} was accepted"
            );
        }
        let mut extended = bytes.clone();
        extended.push(7);
        assert!(decode_loc(&extended).is_err());
        let mut bad_magic = bytes.clone();
        bad_magic[2] = b'X';
        assert!(decode_loc(&bad_magic).is_err());
        assert_eq!(parse_loc(&bytes).unwrap()["b_key"].1, 1);
    }

    #[test]
    fn loc_rejects_lone_surrogate_and_oversized_text() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&LOC_MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0xd800u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.push(0);
        assert!(decode_loc(&bytes).is_err());
        let long = "x".repeat(0x1_0000);
        assert!(encode_loc(&[("k".into(), long, 0)], 1).is_err());
    }
}
