//! Native reader/writer for the TSV files RPFM (Rusted PackFile Manager) exports for DB tables and `.loc`
//! files. This is not a generic CSV dialect; it mirrors RPFM's own implementation exactly, strictly.
//!
//! Upstream evidence (Frodo45127/rpfm, `master`):
//! * `rpfm_lib/src/files/mod.rs` `RFile::tsv_import_from_path` / `tsv_export_to_path`: tab delimiter, **no
//!   quoting at all** (`.quoting(false)` / `QuoteStyle::Never`, "otherwise Excel, Libreoffice ... break them"),
//!   line 1 = column names, line 2 = metadata `#<table>;<version>;<path in pack>` (a `Loc`/`Loc PackedFile`
//!   name marks a localisation file, anything else is a DB table), `\` in the path is turned into `/`.
//! * `rpfm_lib/src/files/table/local.rs` `TableInMemory::tsv_export`: header row, metadata row padded with
//!   `columns - 1` empty cells, then one line per row; booleans `true`/`false`; floats `{:.4}`; sequences base64.
//!   `tsv_import`: booleans via `utils::parse_str_as_bool` (`true`/`false`/`1`/`0`, case-insensitive), numbers via
//!   Rust's `str::parse`, colours via `u32::from_str_radix(.., 16)`; columns are matched **by name**.
//! * `rpfm_lib/src/files/table/mod.rs` `escape_special_chars` / `unescape_special_chars`: a newline or tab inside a
//!   string cell is written as the four/three characters `\\n` / `\\t` (two backslashes), reversed on import.
//! * `rpfm_lib/src/schema/mod.rs` `Definition::fields_processed` and `files/table/mod.rs` `decode_row_postprocess`:
//!   the TSV columns are the *processed* fields, not the binary ones: the `_r/_g/_b` parts of an
//!   `is_part_of_colour` group are removed and one `ColourRGB` column named `<prefix>_hex` (`unnamed colour
//!   group_<n>` when the part has no `_`) per group is appended at the end, value `RRGGBB` uppercase hex.
//!   `fields_processed_sorted(true)` (RPFM's default export) moves key columns to the front, stably.
//! * `rpfm_lib/src/binary/{reader,writer}.rs` optional strings: an empty string is stored as "absent" (flag 0),
//!   so RPFM cannot tell `None` from `Some("")`; this codec follows the same convention (`""` <-> `Cell::Null`).
//! * `rpfm_lib/src/files/loc/mod.rs`: loc TSV name `Loc`, columns `key`, `text`, `tooltip`, version 1 only.
//!
//! Deliberately stricter than upstream (which is `flexible` and silently defaults/ignores): the header must be
//! exactly the processed field set of one schema definition (any order), every row must have exactly one cell per
//! column, numbers/booleans must parse and fit, and anything RPFM would silently mangle (bare CR, blank lines, a
//! string already containing the literal `\\n`) is refused with `file: line N, column C (`name`)` diagnostics.

use crate::db::{Cell, Definition, Field, FieldType, LocEntry, Schema, Table};
use anyhow::{Result, anyhow, bail};
use std::collections::BTreeMap;

/// One decoded RPFM TSV file.
#[derive(Clone, Debug)]
pub enum Document {
    /// A DB table; `member` is the canonical `db/<table>/<member>` path from the TSV metadata row.
    Db { member: String, table: Table },
    /// A localisation file; `member` is the canonical `text/....loc` path from the TSV metadata row.
    Loc {
        member: String,
        version: u32,
        rows: Vec<LocEntry>,
    },
}

const LOC_NAME: &str = "Loc";
const LOC_NAME_OLD: &str = "Loc PackedFile";
const LOC_VERSION: u32 = 1;
const LOC_COLUMNS: [&str; 3] = ["key", "text", "tooltip"];
const MERGED_COLOUR_SUFFIX: &str = "_hex";
const MERGED_COLOUR_UNNAMED: &str = "unnamed colour group";
const MAX_STRING_UNITS: usize = 0xffff;

// ---------------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------------

fn check_component(component: &str) -> Result<()> {
    if component.is_empty() {
        bail!("empty path component (absolute path, `//` or trailing slash)");
    }
    if component == "." || component == ".." {
        bail!("`{component}` path component");
    }
    if let Some(c) = component
        .chars()
        .find(|c| !c.is_ascii() || c.is_ascii_control() || "<>:\"|?*;\\".contains(*c))
    {
        bail!("character {c:?} is not allowed in a data path");
    }
    if component.starts_with(' ') || component.ends_with(' ') || component.ends_with('.') {
        bail!("path component `{component}` starts/ends with a space or ends with a dot");
    }
    let stem = component
        .split('.')
        .next()
        .unwrap_or(component)
        .to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && stem.as_bytes()[3].is_ascii_digit()
            && stem.as_bytes()[3] != b'0');
    if reserved {
        bail!("`{component}` is a reserved device name");
    }
    Ok(())
}

/// Canonical data-relative member path: `\` -> `/`, ASCII-lowercase, no empty/`.`/`..` components, no
/// drive letters, control or Windows-reserved characters. Errors name the offending path.
pub fn canonical_member(path: &str) -> Result<String> {
    let normalized = path.replace('\\', "/");
    if normalized.is_empty() {
        bail!("empty member path");
    }
    let mut parts = Vec::new();
    for component in normalized.split('/') {
        check_component(component).map_err(|e| anyhow!("member path `{path}`: {e}"))?;
        parts.push(component.to_ascii_lowercase());
    }
    Ok(parts.join("/"))
}

fn db_member(table: &str, path: &str) -> Result<String> {
    let member = canonical_member(path)?;
    let parts: Vec<&str> = member.split('/').collect();
    match parts.as_slice() {
        ["db", folder, _] if *folder == table.to_ascii_lowercase() => Ok(member),
        _ => bail!(
            "member path `{path}` must be `db/{table}/<name>` (the metadata table is `{table}`)"
        ),
    }
}

fn loc_member(path: &str) -> Result<String> {
    let member = canonical_member(path)?;
    if !member.starts_with("text/") || !member.ends_with(".loc") {
        bail!("localisation member path `{path}` must be `text/....loc`");
    }
    Ok(member)
}

// ---------------------------------------------------------------------------------------------
// Text escapes (RPFM `escape_special_chars` / `unescape_special_chars`)
// ---------------------------------------------------------------------------------------------

fn unescape(s: &str) -> String {
    s.replace("\\\\t", "\t").replace("\\\\n", "\n")
}

/// Escapes a string cell for export; refuses text RPFM would corrupt on re-import.
fn escape(s: &str) -> Result<String, String> {
    if s.contains('\r') {
        return Err("contains a carriage return, which RPFM TSV cannot represent".into());
    }
    let escaped = s.replace('\n', "\\\\n").replace('\t', "\\\\t");
    if unescape(&escaped) != s {
        return Err(
            "already contains the literal text `\\\\n` or `\\\\t`, which RPFM would turn into a control character"
                .into(),
        );
    }
    Ok(escaped)
}

fn show(s: &str) -> String {
    const LIMIT: usize = 48;
    let mut out: String = s.chars().take(LIMIT).collect();
    if s.chars().count() > LIMIT {
        out.push('…');
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Layout: the processed (TSV) column list of a definition
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Source {
    /// Binary field index.
    Field(usize),
    /// Binary field indices of the red, green and blue channels.
    Colour([usize; 3]),
}

#[derive(Clone, Debug)]
struct Column {
    name: String,
    is_key: bool,
    source: Source,
}

fn layout(table: &str, def: &Definition) -> Result<Vec<Column>> {
    if def.columns.len() != def.fields.len() {
        bail!(
            "{table} v{}: schema column metadata is inconsistent",
            def.version
        );
    }
    let mut columns: Vec<Column> = Vec::new();
    // group -> (merged name, key flag of the first part, channel field indices)
    let mut groups: BTreeMap<u8, (String, bool, [Option<usize>; 3])> = BTreeMap::new();
    for (i, (field, meta)) in def.fields.iter().zip(&def.columns).enumerate() {
        if let FieldType::Unsupported(kind) = &field.ty {
            bail!("field `{}` has the unsupported type {kind}", field.name);
        }
        if meta.bitwise > 1 {
            bail!(
                "field `{}` is a bitwise field (RPFM expands it into boolean columns); not supported",
                field.name
            );
        }
        if meta.has_enum {
            bail!(
                "field `{}` has enum values (RPFM exports names instead of numbers); not supported",
                field.name
            );
        }
        let Some(group) = meta.colour_group else {
            columns.push(Column {
                name: field.name.clone(),
                is_key: field.is_key,
                source: Source::Field(i),
            });
            continue;
        };
        if !matches!(
            field.ty,
            FieldType::I16 | FieldType::I32 | FieldType::I64 | FieldType::F32 | FieldType::F64
        ) {
            bail!("colour channel `{}` is not a numeric field", field.name);
        }
        let mut split = field.name.rsplitn(2, '_');
        let channel = split.next().unwrap_or_default().to_lowercase();
        let merged = match split.next() {
            Some(prefix) => format!("{}{MERGED_COLOUR_SUFFIX}", prefix.to_lowercase()),
            None => format!("{MERGED_COLOUR_UNNAMED}_{group}"),
        };
        let slot = match channel.as_str() {
            "r" | "red" => 0,
            "g" | "green" => 1,
            "b" | "blue" => 2,
            _ => bail!(
                "colour channel `{}` is not named r/g/b (or red/green/blue)",
                field.name
            ),
        };
        let entry = groups
            .entry(group)
            .or_insert_with(|| (merged.clone(), field.is_key, [None; 3]));
        if entry.0 != merged {
            bail!(
                "colour group {group}: channel `{}` merges into `{merged}` but the group is `{}`",
                field.name,
                entry.0
            );
        }
        if entry.2[slot].replace(i).is_some() {
            bail!("colour group {group} has two {channel} channels");
        }
    }
    for (group, (name, is_key, parts)) in groups {
        let [Some(r), Some(g), Some(b)] = parts else {
            bail!("colour group {group} (`{name}`) does not have exactly one r, g and b channel");
        };
        columns.push(Column {
            name,
            is_key,
            source: Source::Colour([r, g, b]),
        });
    }
    let mut seen: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
    seen.sort_unstable();
    if let Some(w) = seen.windows(2).find(|w| w[0] == w[1]) {
        bail!("two TSV columns are both called `{}`", w[0]);
    }
    Ok(columns)
}

fn same_fields(a: &[Field], b: &[Field]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.name == y.name && x.ty == y.ty && x.is_key == y.is_key)
}

// ---------------------------------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------------------------------

/// Decodes the file text. UTF-8 (optionally with BOM, as Excel/LibreOffice write) only.
fn read_text<'a>(hint: &str, bytes: &'a [u8]) -> Result<&'a str> {
    if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[0xfe, 0xff]) {
        bail!("{hint}: the file is UTF-16; RPFM TSV files are UTF-8, re-save it as UTF-8");
    }
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes);
    std::str::from_utf8(bytes).map_err(|e| {
        let line = bytes[..e.valid_up_to()]
            .iter()
            .filter(|&&b| b == b'\n')
            .count()
            + 1;
        anyhow!("{hint}: line {line}: invalid UTF-8 text ({e})")
    })
}

/// `(1-based line number, line without terminator)`; `\n` and `\r\n` terminate, trailing blank lines are
/// dropped, a bare `\r` or a blank line inside the data is an error (RPFM's csv reader would split/skip them).
fn read_lines<'a>(hint: &str, text: &'a str) -> Result<Vec<(usize, &'a str)>> {
    let mut lines = Vec::new();
    for (i, raw) in text.split('\n').enumerate() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.contains('\r') {
            bail!("{hint}: line {}: bare carriage return inside a line", i + 1);
        }
        lines.push((i + 1, line));
    }
    while lines.last().is_some_and(|(_, l)| l.is_empty()) {
        lines.pop();
    }
    if let Some((n, _)) = lines.iter().find(|(_, l)| l.is_empty()) {
        bail!("{hint}: line {n}: blank line inside the table");
    }
    Ok(lines)
}

/// Table name (`<table>_tables`, lower-case) declared by the RPFM metadata line of a TSV file, if it has one.
/// Used where only the raw bytes are at hand (drift checks); decoding errors simply yield `None`.
pub fn declared_table(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)).ok()?;
    let line = text.split('\n').nth(1)?.trim_end_matches('\r');
    let spec = line.split('\t').next()?.strip_prefix('#')?;
    let name = spec.split(';').next()?;
    (!name.is_empty()).then(|| name.to_lowercase())
}

struct Meta {
    name: String,
    version: u32,
    path: String,
}

fn read_meta(hint: &str, line_no: usize, line: &str) -> Result<Meta> {
    let mut cells = line.split('\t');
    let first = cells.next().unwrap_or_default();
    if let Some(extra) = cells.position(|c| !c.is_empty()) {
        bail!(
            "{hint}: line {line_no}, column {}: the metadata row may only use its first cell",
            extra + 2
        );
    }
    let Some(spec) = first.strip_prefix('#') else {
        bail!(
            "{hint}: line {line_no}: the second line must be the RPFM metadata row `#<table>;<version>;<path>`, found `{}`",
            show(first)
        );
    };
    let parts: Vec<&str> = spec.split(';').collect();
    let [name, version, path] = parts.as_slice() else {
        bail!(
            "{hint}: line {line_no}: metadata `{}` must have exactly `<table>;<version>;<path>`",
            show(first)
        );
    };
    let version = version.parse::<u32>().map_err(|_| {
        anyhow!(
            "{hint}: line {line_no}: table version `{}` is not a non-negative integer",
            show(version)
        )
    })?;
    if name.is_empty() || path.is_empty() {
        bail!("{hint}: line {line_no}: metadata has an empty table name or path");
    }
    Ok(Meta {
        name: (*name).to_owned(),
        version,
        path: (*path).to_owned(),
    })
}

fn check_hint_folder(hint: &str, expected_table: Option<&str>) -> Result<()> {
    let normalized = hint.replace('\\', "/");
    let parts: Vec<&str> = normalized.split('/').collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let parent = parts[parts.len() - 2].to_ascii_lowercase();
    if parent.ends_with("_tables") && expected_table != Some(parent.as_str()) {
        match expected_table {
            Some(table) => bail!(
                "{hint}: the file sits in `{parent}` but its metadata says table `{table}`; refusing an ambiguous identity"
            ),
            None => bail!(
                "{hint}: the file sits in the table folder `{parent}` but its metadata says it is a localisation file"
            ),
        }
    }
    Ok(())
}

fn where_at(hint: &str, line: usize, column: usize, name: &str) -> String {
    format!("{hint}: line {line}, column {column} (`{name}`)")
}

// ---------------------------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------------------------

fn parse_bool(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(format!("`{}` is not a boolean (true/false/1/0)", show(s))),
    }
}

fn parse_string(s: &str, wide: bool) -> Result<String, String> {
    let text = unescape(s);
    let units = if wide {
        text.encode_utf16().count()
    } else {
        text.len()
    };
    if units > MAX_STRING_UNITS {
        return Err(format!(
            "text is {units} {} long, the limit is {MAX_STRING_UNITS}",
            if wide { "UTF-16 units" } else { "bytes" }
        ));
    }
    Ok(text)
}

fn parse_hex(s: &str, digits: std::ops::RangeInclusive<usize>) -> Result<u32, String> {
    if !digits.contains(&s.len()) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "`{}` is not a hexadecimal colour of {}..={} digits",
            show(s),
            digits.start(),
            digits.end()
        ));
    }
    u32::from_str_radix(s, 16).map_err(|e| format!("`{}`: {e}", show(s)))
}

fn parse_cell(ty: &FieldType, s: &str) -> Result<Cell, String> {
    fn number<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
        s.parse::<T>()
            .map_err(|_| format!("`{}` is not a valid {what}", show(s)))
    }
    Ok(match ty {
        FieldType::Boolean => Cell::Bool(parse_bool(s)?),
        FieldType::I16 => Cell::I16(number(s, "16-bit integer (-32768..=32767)")?),
        FieldType::I32 => Cell::I32(number(s, "32-bit integer")?),
        FieldType::I64 => Cell::I64(number(s, "64-bit integer")?),
        FieldType::F32 => {
            let v: f32 = number(s, "32-bit decimal number")?;
            if !v.is_finite() {
                return Err(format!("`{}` is not a finite 32-bit number", show(s)));
            }
            Cell::F32(v)
        }
        FieldType::F64 => {
            let v: f64 = number(s, "64-bit decimal number")?;
            if !v.is_finite() {
                return Err(format!("`{}` is not a finite 64-bit number", show(s)));
            }
            Cell::F64(v)
        }
        FieldType::ColourRgb => Cell::Colour(parse_hex(s, 1..=8)?),
        FieldType::StringU8 => Cell::Str(parse_string(s, false)?),
        FieldType::StringU16 => Cell::Str(parse_string(s, true)?),
        FieldType::OptionalStringU8 | FieldType::OptionalStringU16 if s.is_empty() => Cell::Null,
        FieldType::OptionalStringU8 => Cell::Str(parse_string(s, false)?),
        FieldType::OptionalStringU16 => Cell::Str(parse_string(s, true)?),
        FieldType::Unsupported(name) => return Err(format!("unsupported field type {name}")),
    })
}

/// Channel cell of the right numeric type from a byte.
fn channel_cell(ty: &FieldType, v: u8) -> Cell {
    match ty {
        FieldType::I16 => Cell::I16(i16::from(v)),
        FieldType::I32 => Cell::I32(i32::from(v)),
        FieldType::I64 => Cell::I64(i64::from(v)),
        FieldType::F32 => Cell::F32(f32::from(v)),
        _ => Cell::F64(f64::from(v)),
    }
}

// ---------------------------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------------------------

/// Decodes one RPFM TSV export. `path_hint` is any display path of the file (used for diagnostics and one
/// folder cross-check); identity (table, version, member) comes only from the metadata row.
pub fn decode(path_hint: &str, bytes: &[u8], schema: &Schema) -> Result<Document> {
    let text = read_text(path_hint, bytes)?;
    let lines = read_lines(path_hint, text)?;
    let [(_, header_line), (meta_no, meta_line), data @ ..] = lines.as_slice() else {
        bail!(
            "{path_hint}: expected a header line and the RPFM metadata line `#<table>;<version>;<path>`, found {} line(s)",
            lines.len()
        );
    };
    let meta = read_meta(path_hint, *meta_no, meta_line)?;
    let header: Vec<&str> = header_line.split('\t').collect();
    let mut sorted = header.clone();
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        bail!("{path_hint}: line 1: column `{}` appears twice", w[0]);
    }
    if meta.name == LOC_NAME || meta.name == LOC_NAME_OLD {
        check_hint_folder(path_hint, None)?;
        decode_loc(path_hint, &meta, *meta_no, &header, data)
    } else {
        decode_db(path_hint, schema, &meta, *meta_no, &header, data)
    }
}

fn missing_extra(expected: &[&str], header: &[&str]) -> String {
    let missing: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|c| !header.contains(c))
        .collect();
    let extra: Vec<&str> = header
        .iter()
        .copied()
        .filter(|c| !expected.contains(c))
        .collect();
    format!("missing columns {missing:?}, unexpected columns {extra:?}")
}

fn decode_db(
    hint: &str,
    schema: &Schema,
    meta: &Meta,
    meta_no: usize,
    header: &[&str],
    data: &[(usize, &str)],
) -> Result<Document> {
    let table = meta.name.as_str();
    let member =
        db_member(table, &meta.path).map_err(|e| anyhow!("{hint}: line {meta_no}: {e}"))?;
    let table_folder = table.to_ascii_lowercase();
    check_hint_folder(hint, Some(table_folder.as_str()))?;
    let all = schema.definitions(table);
    if all.is_empty() {
        bail!("{hint}: the schema has no table `{table}`");
    }
    let candidates: Vec<&Definition> = all
        .iter()
        .filter(|d| d.version == i64::from(meta.version))
        .collect();
    if candidates.is_empty() {
        let have: Vec<i64> = all.iter().map(|d| d.version).collect();
        bail!(
            "{hint}: {table} version {} is not in the schema (schema versions: {have:?}); re-export the table with the matching RPFM schema",
            meta.version
        );
    }
    let mut matches: Vec<(&Definition, Vec<Column>)> = Vec::new();
    let mut last_problem = String::new();
    for def in candidates {
        match layout(table, def) {
            Err(e) => last_problem = format!("{table} v{} cannot be used: {e}", def.version),
            Ok(columns) => {
                let names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
                if names.len() == header.len() && names.iter().all(|n| header.contains(n)) {
                    matches.push((def, columns));
                } else {
                    last_problem = format!(
                        "header does not match {table} v{}: {}{}",
                        def.version,
                        missing_extra(&names, header),
                        if def.columns.iter().any(|c| c.colour_group.is_some()) {
                            " (RPFM merges r/g/b colour parts into one `*_hex` column)"
                        } else {
                            ""
                        }
                    );
                }
            }
        }
    }
    let Some((def, columns)) = matches.first() else {
        bail!("{hint}: line 1: {last_problem}");
    };
    if matches[1..]
        .iter()
        .any(|(d, _)| !same_fields(&d.fields, &def.fields))
    {
        bail!(
            "{hint}: {table} version {} matches several different schema definitions; refusing an ambiguous layout",
            meta.version
        );
    }

    // header position -> column
    let order: Vec<usize> = header
        .iter()
        .map(|h| {
            columns
                .iter()
                .position(|c| c.name == *h)
                .unwrap_or_default()
        })
        .collect();
    let mut rows = Vec::with_capacity(data.len());
    for (line_no, line) in data {
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() != header.len() {
            bail!(
                "{hint}: line {line_no}: {} cells but the table has {} columns",
                cells.len(),
                header.len()
            );
        }
        let mut row = vec![Cell::Null; def.fields.len()];
        for (pos, cell) in cells.iter().enumerate() {
            let column = &columns[order[pos]];
            let fail =
                |msg: String| anyhow!("{}: {msg}", where_at(hint, *line_no, pos + 1, &column.name));
            match column.source {
                Source::Field(i) => row[i] = parse_cell(&def.fields[i].ty, cell).map_err(fail)?,
                Source::Colour(parts) => {
                    let rgb = parse_hex(cell, 6..=6).map_err(fail)?;
                    for (slot, field) in parts.into_iter().enumerate() {
                        let byte = ((rgb >> (16 - 8 * slot)) & 0xff) as u8;
                        row[field] = channel_cell(&def.fields[field].ty, byte);
                    }
                }
            }
        }
        rows.push(row);
    }
    Ok(Document::Db {
        member,
        table: Table {
            name: table.to_owned(),
            guid: None,
            version: meta.version,
            version_marker: meta.version != 0,
            flag: 1,
            fields: def.fields.clone(),
            rows,
        },
    })
}

fn decode_loc(
    hint: &str,
    meta: &Meta,
    meta_no: usize,
    header: &[&str],
    data: &[(usize, &str)],
) -> Result<Document> {
    let member = loc_member(&meta.path).map_err(|e| anyhow!("{hint}: line {meta_no}: {e}"))?;
    if meta.version != LOC_VERSION {
        bail!(
            "{hint}: line {meta_no}: localisation version {} is not supported (RPFM only writes version {LOC_VERSION})",
            meta.version
        );
    }
    if header.len() != LOC_COLUMNS.len() || !LOC_COLUMNS.iter().all(|c| header.contains(c)) {
        bail!(
            "{hint}: line 1: localisation header must be key/text/tooltip: {}",
            missing_extra(&LOC_COLUMNS, header)
        );
    }
    let pos_of = |name: &str| header.iter().position(|h| *h == name).unwrap_or_default();
    let (key_at, text_at, tip_at) = (pos_of("key"), pos_of("text"), pos_of("tooltip"));
    let mut rows = Vec::with_capacity(data.len());
    for (line_no, line) in data {
        let cells: Vec<&str> = line.split('\t').collect();
        if cells.len() != LOC_COLUMNS.len() {
            bail!(
                "{hint}: line {line_no}: {} cells but a localisation row has 3",
                cells.len()
            );
        }
        let text = |at: usize, name: &str| {
            parse_string(cells[at], true)
                .map_err(|m| anyhow!("{}: {m}", where_at(hint, *line_no, at + 1, name)))
        };
        let key = text(key_at, "key")?;
        let body = text(text_at, "text")?;
        let tip = parse_bool(cells[tip_at])
            .map_err(|m| anyhow!("{}: {m}", where_at(hint, *line_no, tip_at + 1, "tooltip")))?;
        rows.push((key, body, u8::from(tip)));
    }
    Ok(Document::Loc {
        member,
        version: meta.version,
        rows,
    })
}

// ---------------------------------------------------------------------------------------------
// Encode (RPFM-importable export)
// ---------------------------------------------------------------------------------------------

macro_rules! float_formatter {
    ($name:ident, $ty:ty) => {
        /// 4 decimals like RPFM (`{:.4}`) when that reads back to the exact same value, else the shortest exact form.
        fn $name(v: $ty) -> Result<String, String> {
            if !v.is_finite() {
                return Err("is not a finite number".into());
            }
            let fixed = format!("{v:.4}");
            match fixed.parse::<$ty>() {
                Ok(back) if back.to_bits() == v.to_bits() => Ok(fixed),
                _ => Ok(v.to_string()),
            }
        }
    };
}

float_formatter!(format_f32, f32);
float_formatter!(format_f64, f64);

fn format_cell(ty: &FieldType, cell: &Cell) -> Result<String, String> {
    Ok(match (ty, cell) {
        (FieldType::Boolean, Cell::Bool(v)) => v.to_string(),
        (FieldType::I16, Cell::I16(v)) => v.to_string(),
        (FieldType::I32, Cell::I32(v)) => v.to_string(),
        (FieldType::I64, Cell::I64(v)) => v.to_string(),
        (FieldType::F32, Cell::F32(v)) => format_f32(*v)?,
        (FieldType::F64, Cell::F64(v)) => format_f64(*v)?,
        (FieldType::ColourRgb, Cell::Colour(v)) => format!("{v:06X}"),
        (FieldType::StringU8 | FieldType::StringU16, Cell::Str(s)) => escape(s)?,
        (FieldType::OptionalStringU8 | FieldType::OptionalStringU16, Cell::Str(s)) => escape(s)?,
        (FieldType::OptionalStringU8 | FieldType::OptionalStringU16, Cell::Null) => String::new(),
        (FieldType::Unsupported(name), _) => return Err(format!("unsupported field type {name}")),
        (ty, cell) => return Err(format!("value {cell:?} does not match field type {ty:?}")),
    })
}

fn channel_byte(cell: &Cell) -> Result<u8, String> {
    let wide: Option<i64> = match cell {
        Cell::I16(v) => Some(i64::from(*v)),
        Cell::I32(v) => Some(i64::from(*v)),
        Cell::I64(v) => Some(*v),
        Cell::F32(v) if v.fract() == 0.0 => Some(*v as i64),
        Cell::F64(v) if v.fract() == 0.0 => Some(*v as i64),
        _ => None,
    };
    wide.and_then(|v| u8::try_from(v).ok())
        .ok_or_else(|| format!("colour channel value {cell:?} is not a whole number 0..=255"))
}

fn write_lines(lines: &[Vec<String>]) -> Vec<u8> {
    let mut out = String::new();
    for line in lines {
        out.push_str(&line.join("\t"));
        out.push('\n');
    }
    out.into_bytes()
}

fn metadata_row(name: &str, version: u32, member: &str, columns: usize) -> Vec<String> {
    let mut row = vec![format!("#{name};{version};{member}")];
    row.resize(columns, String::new());
    row
}

/// Exports `table` the way RPFM does (header, metadata row, key columns first, merged colour columns,
/// escaped strings, `\n` line ends, no BOM, no quoting). `member` is the table's `db/<table>/<name>` path.
pub fn encode_db(member: &str, table: &Table, schema: &Schema) -> Result<Vec<u8>> {
    let member = db_member(&table.name, member)?;
    let def = schema
        .definitions(&table.name)
        .iter()
        .find(|d| d.version == i64::from(table.version) && same_fields(&d.fields, &table.fields))
        .ok_or_else(|| {
            anyhow!(
                "{member}: the schema has no {} v{} definition with these fields",
                table.name,
                table.version
            )
        })?;
    let mut columns = layout(&table.name, def).map_err(|e| anyhow!("{member}: {e}"))?;
    if columns.is_empty() {
        bail!("{member}: the table has no columns");
    }
    columns.sort_by_key(|c| !c.is_key); // stable: keys first, RPFM `fields_processed_sorted(true)`
    let mut lines: Vec<Vec<String>> = Vec::with_capacity(table.rows.len() + 2);
    lines.push(columns.iter().map(|c| c.name.clone()).collect());
    lines.push(metadata_row(
        &table.name,
        table.version,
        &member,
        columns.len(),
    ));
    for (r, row) in table.rows.iter().enumerate() {
        if row.len() != def.fields.len() {
            bail!(
                "{member}: row {} has {} cells for {} fields",
                r + 1,
                row.len(),
                def.fields.len()
            );
        }
        let mut out = Vec::with_capacity(columns.len());
        for column in &columns {
            let text = match column.source {
                Source::Field(i) => format_cell(&def.fields[i].ty, &row[i]),
                Source::Colour(parts) => parts
                    .iter()
                    .map(|&p| channel_byte(&row[p]))
                    .collect::<Result<Vec<u8>, String>>()
                    .map(|c| format!("{:02X}{:02X}{:02X}", c[0], c[1], c[2])),
            }
            .map_err(|m| anyhow!("{member}: row {} column `{}`: {m}", r + 1, column.name))?;
            out.push(text);
        }
        if columns.len() == 1 && out[0].is_empty() {
            bail!(
                "{member}: row {} is an empty single cell, which RPFM TSV cannot represent (it is a blank line)",
                r + 1
            );
        }
        lines.push(out);
    }
    Ok(write_lines(&lines))
}

/// Exports localisation rows as RPFM `Loc` TSV (`key`, `text`, `tooltip`); `version` must be 1.
pub fn encode_loc(member: &str, version: u32, rows: &[LocEntry]) -> Result<Vec<u8>> {
    let member = loc_member(member)?;
    if version != LOC_VERSION {
        bail!(
            "{member}: localisation version {version} cannot be written (RPFM only knows version {LOC_VERSION})"
        );
    }
    let mut lines: Vec<Vec<String>> = Vec::with_capacity(rows.len() + 2);
    lines.push(LOC_COLUMNS.iter().map(|c| (*c).to_owned()).collect());
    lines.push(metadata_row(LOC_NAME, version, &member, LOC_COLUMNS.len()));
    for (r, (key, text, tip)) in rows.iter().enumerate() {
        let cell = |column: &str, s: &str| {
            if s.encode_utf16().count() > MAX_STRING_UNITS {
                return Err(anyhow!(
                    "{member}: row {} column `{column}`: text is too long",
                    r + 1
                ));
            }
            escape(s).map_err(|m| anyhow!("{member}: row {} column `{column}`: {m}", r + 1))
        };
        let tooltip = match tip {
            0 => "false",
            1 => "true",
            other => bail!("{member}: row {} tooltip flag {other} is not 0 or 1", r + 1),
        };
        lines.push(vec![
            cell("key", key)?,
            cell("text", text)?,
            tooltip.to_owned(),
        ]);
    }
    Ok(write_lines(&lines))
}

#[cfg(test)]
mod tests;
