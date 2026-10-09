//! Database rows, localisation and lookup map for a region that was carved out of another one (vanilla NTW:
//! `eur_samogitia` out of `eur_lithuania`).
//!
//! A land region is described outside the campaign map and the start position by exactly these data:
//!  * `regions_tables` — key, continent, lookup colour (`r`,`g`,`b`), battle name; the colour must be unique and is the
//!    palette colour of the region in the map's lookup TGA (`campaign_map_playable_areas.overlay_file`);
//!  * `campaign_map_settlements_tables` — one row per settlement (`slot_count` 1, `slot_type` settlement). The three
//!    settlement slots themselves (`settlement_road`, `settlement_1_slot`, `settlement_fortification`) exist only in the
//!    start position and `regions.esf`, never in a DB table; `campaign_map_slots_tables` and
//!    `campaign_map_towns_and_ports_tables` hold resource / town / port slots only, which the new region does not have;
//!  * `text\localisation.loc` — region and settlement names.
//!
//! [`scan_references`] finds every table row of the vanilla data that mentions the source region, [`generate`] builds
//! the new rows by cloning the source rows, and [`build_pack_members`] packs them as additive `db/<table>/zz_samogitia`
//! members plus the merged `text\localisation.loc` (the pack carries no loose files).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::db::{self, Cell, LocEntry, Schema, Table};
use crate::pack::{self, Pack};

pub const SRC_REGION: &str = "eur_lithuania";
pub const NEW_REGION: &str = "eur_samogitia";
pub const SRC_SETTLEMENT: &str = "settlement:eur_lithuania:vilnius";
pub const NEW_SETTLEMENT: &str = "settlement:eur_samogitia:siauliai";
pub const NEW_REGION_NAME: &str = "Samogitia";
pub const NEW_SETTLEMENT_NAME: &str = "Siauliai";
/// Lookup colour of the new region: unique among `regions_tables` and the lookup TGA palette.
pub const NEW_COLOUR: [u8; 3] = [142, 180, 106];
/// Base name of the added table members (`db\<table>\zz_samogitia`).
pub const MEMBER: &str = "zz_samogitia";
/// Strings the start position must store so that the loc keys below resolve. The loc key is the engine prefix plus the
/// string; Vilnius uses `settlement:eur_lithuania:vilnius-1220936178` (eur_napoleon) / `...vilnius1169683589`
/// (mp_eur_napoleon) and `eur_lithuaniaeur_napoleon` / `eur_lithuaniamp_eur_napoleon` for the rebel faction name.
pub const STARTPOS_SETTLEMENT_NAME_EUR_NAPOLEON: &str = "settlement:eur_samogitia:siauliai-1000001";
pub const STARTPOS_SETTLEMENT_NAME_MP_EUR_NAPOLEON: &str = "settlement:eur_samogitia:siauliai1000002";
pub const STARTPOS_REBEL_NAME_EUR_NAPOLEON: &str = "eur_samogitiaeur_napoleon";
pub const STARTPOS_REBEL_NAME_MP_EUR_NAPOLEON: &str = "eur_samogitiamp_eur_napoleon";
const REBEL_TEXT: &str = "Lithuanian Rebels";

pub const LOC_MEMBER: &str = "text\\localisation.loc";

/// Strings that identify the source region in a table cell (lower case): the region key, which is also embedded in
/// every slot / settlement / town key (`settlement:eur_lithuania:vilnius`, `pasture:eur_lithuania:brest-litovsk`, ...),
/// and the settlement name.
const NEEDLES: [&str; 2] = ["eur_lithuania", "vilnius"];

/// Orders packs the way the engine loads them (the snapshot reader uses the same rule): boot, release/movie, patch, mod.
fn pack_rank(kind: u32) -> u32 {
    match kind {
        pack::TYPE_BOOT => 0,
        pack::TYPE_PATCH => 2,
        pack::TYPE_MOD => 3,
        _ => 1,
    }
}

/// The vanilla packs of a game `data` folder in load order: every boot / release / patch pack (movie and mod packs, such
/// as converter output or earlier tests, are not vanilla and are ignored).
pub fn open_vanilla_packs(data: &Path) -> Result<Vec<Pack>> {
    let mut packs = Vec::new();
    for e in fs::read_dir(data).with_context(|| format!("list {}", data.display()))? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_lowercase();
        if !e.file_type()?.is_file() || !name.ends_with(".pack") {
            continue;
        }
        let p = Pack::open(&e.path())?;
        if p.type_ <= pack::TYPE_PATCH {
            packs.push((name, p));
        }
    }
    packs.sort_by(|a, b| (pack_rank(a.1.type_), &a.0).cmp(&(pack_rank(b.1.type_), &b.0)));
    Ok(packs.into_iter().map(|(_, p)| p).collect())
}

fn db_table_of(member: &str) -> Option<&str> {
    let mut parts = member.split('\\');
    if !parts.next()?.eq_ignore_ascii_case("db") {
        return None;
    }
    let table = parts.next()?;
    parts.next()?;
    Some(table)
}

fn utf16(s: &str) -> Vec<u8> {
    s.encode_utf16().flat_map(u16::to_le_bytes).collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

/// One DB table member with rows that mention the source region.
#[derive(Clone, Debug)]
pub struct TableScan {
    pub pack: String,
    pub member: String,
    pub table: String,
    pub rows: usize,
    /// `(row index, [(column, cell text)])` of every matching row.
    pub hits: Vec<(usize, Vec<(String, String)>)>,
}

#[derive(Clone, Debug, Default)]
pub struct Scan {
    pub tables: Vec<TableScan>,
    /// Members the schema cannot decode whose raw bytes mention a needle (must be empty for a trustworthy scan).
    pub undecodable_hits: Vec<String>,
    pub db_members: usize,
    pub undecodable: usize,
}

/// Every DB row of the vanilla packs whose text cells contain `eur_lithuania` or `vilnius` (case-insensitive).
pub fn scan_references(packs: &[Pack], schema: &Schema) -> Result<Scan> {
    let mut scan = Scan::default();
    for p in packs {
        let pack_name = p
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for e in &p.entries {
            let Some(table) = db_table_of(&e.name) else {
                continue;
            };
            scan.db_members += 1;
            let bytes = p.read_entry(e)?;
            let t = match db::decode(&table.to_lowercase(), &bytes, schema) {
                Ok(t) => t,
                Err(_) => {
                    scan.undecodable += 1;
                    let lower = bytes.to_ascii_lowercase();
                    if NEEDLES
                        .iter()
                        .any(|n| contains(&lower, n.as_bytes()) || contains(&lower_utf16(&bytes), &utf16(n)))
                    {
                        scan.undecodable_hits.push(format!("{pack_name}:{}", e.name));
                    }
                    continue;
                }
            };
            let mut hits = Vec::new();
            for (i, row) in t.rows.iter().enumerate() {
                let cols: Vec<(String, String)> = row
                    .iter()
                    .zip(&t.fields)
                    .filter_map(|(c, f)| match c {
                        Cell::Str(s) => {
                            let l = s.to_lowercase();
                            NEEDLES
                                .iter()
                                .any(|n| l.contains(n))
                                .then(|| (f.name.clone(), s.clone()))
                        }
                        _ => None,
                    })
                    .collect();
                if !cols.is_empty() {
                    hits.push((i, cols));
                }
            }
            if !hits.is_empty() {
                scan.tables.push(TableScan {
                    pack: pack_name.clone(),
                    member: e.name.clone(),
                    table: t.name.clone(),
                    rows: t.rows.len(),
                    hits,
                });
            }
        }
    }
    Ok(scan)
}

/// ASCII-lower-cased copy of UTF-16LE text bytes (only the low byte of each unit is touched).
fn lower_utf16(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .enumerate()
        .map(|(i, b)| if i % 2 == 0 { b.to_ascii_lowercase() } else { *b })
        .collect()
}

/// All members of `table` (`<name>_tables`) of the packs, rows concatenated in load order. The header of the first member
/// is kept; members must agree on version and fields.
pub fn merged_table(packs: &[Pack], table: &str, schema: &Schema) -> Result<Table> {
    let mut merged: Option<Table> = None;
    for p in packs {
        for e in &p.entries {
            if db_table_of(&e.name).is_none_or(|t| !t.eq_ignore_ascii_case(table)) {
                continue;
            }
            let t = db::decode(table, &p.read_entry(e)?, schema).with_context(|| format!("decode {}", e.name))?;
            match merged.as_mut() {
                None => merged = Some(t),
                Some(m) => {
                    ensure!(
                        m.version == t.version && m.field_names() == t.field_names(),
                        "{}: members disagree on version / fields",
                        e.name
                    );
                    m.rows.extend(t.rows);
                }
            }
        }
    }
    merged.ok_or_else(|| anyhow!("no vanilla member for {table}"))
}

fn set(t: &Table, row: &mut [Cell], column: &str, value: Cell) -> Result<()> {
    row[t.column(column)?] = value;
    Ok(())
}

fn find_row(t: &Table, column: &str, key: &str) -> Result<Option<usize>> {
    let c = t.column(column)?;
    Ok(t.rows.iter().position(|r| r[c].as_str() == Some(key)))
}

fn string(row: &[Cell], t: &Table, column: &str) -> Result<String> {
    row[t.column(column)?]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("{}.{column} is not a string", t.name))
}

/// The additive rows of one table.
#[derive(Clone, Debug)]
pub struct NewRows {
    /// Table with the vanilla header and only the new rows.
    pub table: Table,
    /// Vanilla row count of the merged table the rows are added to.
    pub vanilla_rows: usize,
    /// The vanilla row each new row was cloned from.
    pub source_row: Vec<Cell>,
}

#[derive(Clone, Debug)]
pub struct Generated {
    pub regions: NewRows,
    pub settlements: NewRows,
    /// New `text\localisation.loc` entries `(key, text, tooltip flag)`.
    pub loc: Vec<LocEntry>,
}

/// Builds the new rows from the vanilla source rows. Fails when the source rows are missing, the new keys already exist,
/// or [`NEW_COLOUR`] is used by a vanilla region.
pub fn generate(packs: &[Pack], schema: &Schema) -> Result<Generated> {
    // ---- regions_tables
    let regions = merged_table(packs, "regions_tables", schema)?;
    let src = find_row(&regions, "key", SRC_REGION)?
        .ok_or_else(|| anyhow!("regions_tables has no row {SRC_REGION}"))?;
    ensure!(
        find_row(&regions, "key", NEW_REGION)?.is_none(),
        "regions_tables already has {NEW_REGION}"
    );
    let [r, g, b] = NEW_COLOUR.map(i32::from);
    let (cr, cg, cb) = (regions.column("r")?, regions.column("g")?, regions.column("b")?);
    for row in &regions.rows {
        if row[cr] == Cell::I32(r) && row[cg] == Cell::I32(g) && row[cb] == Cell::I32(b) {
            bail!("lookup colour {NEW_COLOUR:?} is already used by region {}", string(row, &regions, "key")?);
        }
    }
    let source_row = regions.rows[src].clone();
    let mut row = source_row.clone();
    set(&regions, &mut row, "key", Cell::Str(NEW_REGION.into()))?;
    set(&regions, &mut row, "r", Cell::I32(r))?;
    set(&regions, &mut row, "g", Cell::I32(g))?;
    set(&regions, &mut row, "b", Cell::I32(b))?;
    set(&regions, &mut row, "battle_name", Cell::Str(NEW_REGION_NAME.into()))?;
    let regions_new = NewRows {
        vanilla_rows: regions.rows.len(),
        table: Table {
            rows: vec![row],
            ..regions
        },
        source_row,
    };

    // ---- campaign_map_settlements_tables
    let settlements = merged_table(packs, "campaign_map_settlements_tables", schema)?;
    let src = find_row(&settlements, "settlement_id", SRC_SETTLEMENT)?
        .ok_or_else(|| anyhow!("campaign_map_settlements_tables has no row {SRC_SETTLEMENT}"))?;
    ensure!(
        find_row(&settlements, "settlement_id", NEW_SETTLEMENT)?.is_none(),
        "campaign_map_settlements_tables already has {NEW_SETTLEMENT}"
    );
    let source_row = settlements.rows[src].clone();
    let mut row = source_row.clone();
    set(&settlements, &mut row, "settlement_id", Cell::Str(NEW_SETTLEMENT.into()))?;
    set(&settlements, &mut row, "region", Cell::Str(NEW_REGION.into()))?;
    set(&settlements, &mut row, "default_onscreen_name", Cell::Str(NEW_SETTLEMENT_NAME.into()))?;
    let settlements_new = NewRows {
        vanilla_rows: settlements.rows.len(),
        table: Table {
            rows: vec![row],
            ..settlements
        },
        source_row,
    };

    // ---- localisation (key patterns of the vanilla eur_lithuania / vilnius strings)
    let loc: Vec<LocEntry> = [
        (format!("regions_onscreen_{NEW_REGION}"), NEW_REGION_NAME),
        (format!("regions_battle_name_{NEW_REGION}"), NEW_REGION_NAME),
        (
            format!("start_pos_regions_rebel_faction_name_{STARTPOS_REBEL_NAME_EUR_NAPOLEON}"),
            REBEL_TEXT,
        ),
        (
            format!("start_pos_regions_rebel_faction_name_{STARTPOS_REBEL_NAME_MP_EUR_NAPOLEON}"),
            REBEL_TEXT,
        ),
        (
            format!("start_pos_settlements_onscreen_name_{STARTPOS_SETTLEMENT_NAME_EUR_NAPOLEON}"),
            NEW_SETTLEMENT_NAME,
        ),
        (
            format!("start_pos_settlements_onscreen_name_{STARTPOS_SETTLEMENT_NAME_MP_EUR_NAPOLEON}"),
            NEW_SETTLEMENT_NAME,
        ),
    ]
    .into_iter()
    .map(|(k, t)| (k, t.to_owned(), 0u8))
    .collect();

    Ok(Generated {
        regions: regions_new,
        settlements: settlements_new,
        loc,
    })
}

/// The effective vanilla `text\localisation.loc`: the member of the last pack (in load order) that has one — the patch
/// pack's complete file replaces the base pack's.
pub fn effective_loc(packs: &[Pack]) -> Result<(u32, Vec<LocEntry>, String)> {
    let p = packs
        .iter()
        .rev()
        .find(|p| p.find(LOC_MEMBER).is_some())
        .ok_or_else(|| anyhow!("no pack contains {LOC_MEMBER}"))?;
    let (version, entries) = db::decode_loc(&p.read(LOC_MEMBER)?)?;
    let name = p.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok((version, entries, name))
}

/// `db\<table>\zz_samogitia` members plus the merged full `text\localisation.loc` (vanilla entries followed by the new
/// ones; the vanilla file is replaced as a whole because the engine loads one `text\localisation.loc` per path).
pub fn build_pack_members(packs: &[Pack], schema: &Schema) -> Result<(Generated, BTreeMap<String, Vec<u8>>)> {
    let generated = generate(packs, schema)?;
    let mut files = BTreeMap::new();
    for rows in [&generated.regions, &generated.settlements] {
        let name = format!("db\\{}\\{MEMBER}", rows.table.name);
        files.insert(name, db::encode(&rows.table)?);
    }
    let (version, mut entries, _) = effective_loc(packs)?;
    let have: std::collections::HashSet<&str> = entries.iter().map(|e| e.0.as_str()).collect();
    for (k, ..) in &generated.loc {
        ensure!(!have.contains(k.as_str()), "vanilla localisation already has {k}");
    }
    entries.extend(generated.loc.iter().cloned());
    files.insert(LOC_MEMBER.to_owned(), db::encode_loc(&entries, version)?);
    Ok((generated, files))
}
