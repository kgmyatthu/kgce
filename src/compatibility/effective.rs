//! Effective target database: every DB member of the TSV dataset is decoded separately. Conflicting keys the
//! conversion reads or writes require resolution; an explicit member choice removes only the losing rows. All other
//! rows, including conflicting ones, stay untouched. Original member paths and row order are preserved.

use crate::{
    content::{self, PortTable},
    db::{self, Cell, Table},
    model::Rules,
    snapshot::{Game, Snapshot},
};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One decoded DB member of the effective target with the rows removed from it so far.
#[derive(Clone, Debug)]
pub struct MemberRows {
    /// Canonical `db/<table>/<member>` path of the TSV export.
    pub path: String,
    pub origin: String,
    pub sha256: String,
    pub table: Table,
    pub removed: BTreeSet<usize>,
}

impl MemberRows {
    pub fn live_rows(&self) -> impl Iterator<Item = (usize, &Vec<Cell>)> {
        self.table
            .rows
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.removed.contains(i))
    }
}

#[derive(Clone, Debug)]
pub struct SchemaProblem {
    pub table: String,
    pub detail: String,
    /// Only conflicting rows between members cannot be detected; the table itself is readable. Matters only to
    /// tables whose row values the conversion reads or whose rows it must not collide with.
    pub identity_only: bool,
}

#[derive(Debug, Default)]
pub struct Scan {
    /// `<table>_tables` -> its members (member-name order).
    pub tables: BTreeMap<String, Vec<MemberRows>>,
    pub schema: Vec<SchemaProblem>,
}

/// Whether the decoded rows of all members, concatenated, equal the original game's table (same version, columns and
/// rows in any order). Compares decoded content, never file bytes: a TSV round trip rewrites the binary header.
fn equals_original(base: &Snapshot, rules: &Rules, dir: &str, members: &[db::Member]) -> bool {
    let Ok(Some(original)) = db::load_table_opt(base, Game::Ntw, rules, dir) else {
        return false;
    };
    let Some(first) = members.first() else {
        return false;
    };
    if original.version != first.table.version
        || original.field_names() != first.table.field_names()
    {
        return false;
    }
    let mut want: Vec<String> = original.rows.iter().map(|r| db::row_signature(r)).collect();
    let mut have: Vec<String> = members
        .iter()
        .flat_map(|m| m.table.rows.iter().map(|r| db::row_signature(r)))
        .collect();
    want.sort_unstable();
    have.sort_unstable();
    want == have
}

/// Decodes every DB table of the effective target. The TSV dataset is authoritative: there is no vanilla member to
/// fall back to, so a table is either decoded from the dataset or absent. `base` is only used to recognise tables
/// whose content equals the original game's.
pub fn scan(effective: &Snapshot, base: &Snapshot, rules: &Rules) -> Result<Scan> {
    scan_tables(effective, base, rules, None)
}

/// [`scan`] restricted to the named tables (build time only needs the tables the conversion adds rows to).
pub fn scan_tables(
    effective: &Snapshot,
    base: &Snapshot,
    rules: &Rules,
    only: Option<&[String]>,
) -> Result<Scan> {
    let mut out = Scan::default();
    for (dir, _members) in db::table_directory(effective, Game::Ntw) {
        if only.is_some_and(|names| !names.contains(&dir)) {
            continue;
        }
        let decoded = match db::load_members(effective, Game::Ntw, rules, &dir) {
            Ok(decoded) => decoded,
            Err(error) => {
                out.schema.push(SchemaProblem {
                    table: dir,
                    detail: format!("{error:#}"),
                    identity_only: false,
                });
                continue;
            }
        };
        if let Some(first) = decoded.first() {
            if decoded.len() > 1
                && !content::has_row_identity(&dir, &first.table.fields)
                && !equals_original(base, rules, &dir, &decoded)
            {
                out.schema.push(SchemaProblem {
                    table: dir.clone(),
                    identity_only: true,
                    detail: format!(
                        "{dir} is split over {} TSV files and differs from the original game's table, but it has no supported row-key definition in the schema; conflicting rows between the files cannot be detected. Export the table as one TSV file.",
                        decoded.len()
                    ),
                });
                continue;
            }
            if let Some(other) = decoded.iter().find(|m| {
                m.table.version != first.table.version
                    || m.table.field_names() != first.table.field_names()
            }) {
                out.schema.push(SchemaProblem {
                    table: dir.clone(),
                    identity_only: false,
                    detail: format!(
                        "members {} (version {}) and {} (version {}) of {dir} have different schema versions or fields; the engine cannot merge them",
                        first.path, first.table.version, other.path, other.table.version
                    ),
                });
                continue;
            }
        }
        let rows = decoded
            .into_iter()
            .map(|m| MemberRows {
                origin: effective.member_origin(Game::Ntw, &m.path),
                path: m.path,
                sha256: m.sha256,
                table: m.table,
                removed: BTreeSet::new(),
            })
            .collect::<Vec<_>>();
        out.tables.insert(dir, rows);
    }
    Ok(out)
}

/// A key held with different content by several rows.
#[derive(Clone, Debug)]
pub struct Conflict {
    pub id: String,
    /// `(member index, row index)`
    pub entries: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Default)]
pub struct Duplicates {
    /// Conflicts between members, grouped by the (sorted) set of members that hold the key.
    pub cross: BTreeMap<Vec<usize>, Vec<Conflict>>,
    /// Conflicts where one member itself holds a key twice with different content.
    pub within: Vec<Conflict>,
}

pub fn identity_names(table: &str, members: &[MemberRows]) -> Vec<String> {
    members
        .first()
        .map(|m| content::identity_columns(table, &m.table.fields))
        .unwrap_or_default()
}

/// Finds rows sharing an identity with different content. Rows equal in every cell are harmless duplicates.
pub fn duplicates(table: &str, members: &[MemberRows]) -> Result<Duplicates> {
    let names = identity_names(table, members);
    let mut by_id: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for (mi, m) in members.iter().enumerate() {
        for (ri, row) in m.live_rows() {
            let id = content::identity_of_row(&m.table.fields, &names, row)
                .ok_or_else(|| anyhow!("{table}: identity column missing in {}", m.path))?;
            by_id.entry(id).or_default().push((mi, ri));
        }
    }
    let mut out = Duplicates::default();
    for (id, entries) in by_id {
        if entries.len() < 2 {
            continue;
        }
        // Rows equal in every cell are harmless; any other pair sharing a key is a decision of the dataset author.
        let sigs: Vec<String> = entries
            .iter()
            .map(|(m, r)| db::row_signature(&members[*m].table.rows[*r]))
            .collect();
        if sigs.iter().all(|s| *s == sigs[0]) {
            continue;
        }
        let mut within = false;
        for (i, (m, _)) in entries.iter().enumerate() {
            if entries
                .iter()
                .enumerate()
                .any(|(j, (m2, _))| i != j && m2 == m && sigs[i] != sigs[j])
            {
                within = true;
            }
        }
        let conflict = Conflict {
            id,
            entries: entries.clone(),
        };
        if within {
            out.within.push(conflict);
        } else {
            let mut set: Vec<usize> = entries.iter().map(|(m, _)| *m).collect();
            set.sort_unstable();
            set.dedup();
            out.cross.entry(set).or_default().push(conflict);
        }
    }
    Ok(out)
}

/// The members of `group` other than `winner` lose their rows of every conflict.
pub fn keep_member(members: &mut [MemberRows], conflicts: &[Conflict], winner: usize) {
    for c in conflicts {
        for (m, r) in &c.entries {
            if *m != winner {
                members[*m].removed.insert(*r);
            }
        }
    }
}

/// A port row whose identity already exists in the target.
#[derive(Clone, Debug)]
pub struct PortCollision {
    pub id: String,
    pub identical: bool,
    pub port_row: Vec<Cell>,
    /// Matching live target rows `(member index, row index)`.
    pub target: Vec<(usize, usize)>,
}

/// Whether the port table and the target members describe the same table layout.
pub fn same_layout(pt: &PortTable, members: &[MemberRows]) -> bool {
    members.iter().all(|m| {
        m.table.version == pt.table.version && m.table.field_names() == pt.table.field_names()
    })
}

pub fn port_collisions(pt: &PortTable, members: &[MemberRows]) -> Vec<PortCollision> {
    let mut target: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (mi, m) in members.iter().enumerate() {
        for (ri, row) in m.live_rows() {
            if let Some(id) = content::identity_of_row(&m.table.fields, &pt.identity, row) {
                target.entry(id).or_default().push((mi, ri));
            }
        }
    }
    let mut out = Vec::new();
    for row in &pt.table.rows {
        let id = pt.identity_of(row);
        let Some(entries) = target.get(&id) else {
            continue;
        };
        let identical = entries
            .iter()
            .all(|(m, r)| members[*m].table.rows[*r] == *row);
        out.push(PortCollision {
            id,
            identical,
            port_row: row.clone(),
            target: entries.clone(),
        });
    }
    out
}

/// Every member that lost rows, re-encoded: `(canonical path, bytes, rows removed)`.
pub fn rewrite(members: &[MemberRows]) -> Result<Vec<(String, Vec<u8>, usize)>> {
    let mut out = Vec::new();
    for m in members {
        if m.removed.is_empty() {
            continue;
        }
        let mut table = m.table.clone();
        let mut index = 0usize;
        table.rows.retain(|_| {
            index += 1;
            !m.removed.contains(&(index - 1))
        });
        out.push((m.path.clone(), db::encode(&table)?, m.removed.len()));
    }
    Ok(out)
}
