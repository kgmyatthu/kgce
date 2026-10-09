//! FACTION record assembly (NTW v18 schema) for the edited world. Port of `startpos/factions_b.py` plus the
//! ownership-edit policy:
//!
//! * faction alive + its NTW record is alive: the NTW record is used unchanged (own ids);
//! * faction alive + its NTW record is dead (or absent): a live NTW donor record is cloned with fresh ids and carries the
//!   faction's own identity (names, flag/colours, name allocators), ministers/governors come from the donor;
//! * faction landless (with a live NTW record, or without a record): the dead NTW donor record (`hungary`) with
//!   the faction's identity; such a faction is inactive (no governorship, no forces);
//! * faction landless and its NTW record is dead: the NTW record unchanged.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, anyhow, bail};

use crate::db::{Cell, Table};
use crate::esf::{Node, Value};
use crate::model::Rules;

use super::context::{BuildContext, FactionInfo, FactionMeta, GovPost, MetaChar};
use super::helpers::*;

pub const ID_LO: u32 = 500_000_000;
pub const ID_HI: u32 = 1_000_000_000;
const PROTECTED: [&str; 3] = [
    "NAME_ALLOCATION_DETAILS",
    "LAND_UNIT_NAME_ALLOCATOR",
    "NAVAL_UNIT_NAME_ALLOCATOR",
];

pub const DONOR_ISLAMIC: &str = "ottomans";
pub const DONOR_WEST: &str = "hessen";
/// NTW faction that is dead in eur_napoleon (shape of a non-playable dead record)
pub const DEAD_DONOR: &str = "hungary";
const DEPLOYED_TYPES: [&str; 2] = ["gentleman", "rake"];

pub fn theatre_post(theatre: &str) -> Result<&'static str> {
    Ok(match theatre {
        "2" => "governor_europe",
        "1" => "governor_america",
        "3" => "governor_india",
        other => bail!("unknown theatre key {other}"),
    })
}

/// How the FACTION record of a faction is obtained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Structure {
    /// NTW record of the faction itself, unchanged
    Own,
    /// live NTW donor (named faction of the target start position) cloned with fresh ids
    Live(String),
    /// dead NTW donor cloned with fresh ids
    Dead(String),
}

fn own_is_alive(own: &Node) -> Result<bool> {
    Ok(!items(get(own, &["CHARACTER_ARRAY"])?)?.is_empty())
}

/// Faction records of the (target) `eur_napoleon` start position: which factions have a record and whether it is live
/// (has ministers / governors) or dead, plus the unit keys of its armies (template instances).
#[derive(Clone, Debug, Default)]
pub struct StartposStructure {
    pub order: Vec<String>,
    pub alive: BTreeMap<String, bool>,
    pub units: BTreeSet<String>,
}

impl StartposStructure {
    pub fn from_templates(tmpl: &Templates) -> Result<Self> {
        let mut out = StartposStructure::default();
        for k in &tmpl.fac_order {
            let alive = own_is_alive(&tmpl.fac_by_key[k])
                .with_context(|| format!("faction record {k} of the target start position"))?;
            out.order.push(k.clone());
            out.alive.insert(k.clone(), alive);
        }
        out.units = tmpl.unit_order.iter().cloned().collect();
        Ok(out)
    }

    pub fn is_live(&self, key: &str) -> bool {
        self.alive.get(key) == Some(&true)
    }

    pub fn live_factions(&self) -> Vec<&str> {
        self.order
            .iter()
            .filter(|k| self.is_live(k))
            .map(String::as_str)
            .collect()
    }

    /// Dead record used as shape template of landless factions: `DEAD_DONOR` when it is dead, else the first dead record.
    pub fn dead_donor(&self) -> Option<&str> {
        if self.alive.get(DEAD_DONOR) == Some(&false) {
            return Some(DEAD_DONOR);
        }
        self.order
            .iter()
            .find(|k| self.alive.get(*k) == Some(&false))
            .map(String::as_str)
    }
}

fn live_list(sp: &StartposStructure) -> String {
    let live = sp.live_factions();
    let shown: Vec<&str> = live.iter().copied().take(12).collect();
    format!(
        "{}{}",
        shown.join(", "),
        if live.len() > shown.len() {
            ", ..."
        } else {
            ""
        }
    )
}

fn live_donor_checked(
    key: &str,
    donor: &str,
    sp: &StartposStructure,
    what: &str,
) -> Result<String> {
    if sp.is_live(donor) {
        return Ok(donor.to_string());
    }
    let why = match sp.alive.get(donor) {
        None => "has no record in the target start position",
        Some(_) => "is a dead record in the target start position",
    };
    bail!(
        "faction {key}: {what} {donor} {why}; choose a live faction with mappings.faction_templates[\"{key}\"] (live: {})",
        live_list(sp)
    )
}

fn dead_donor_checked(key: &str, sp: &StartposStructure) -> Result<String> {
    sp.dead_donor().map(str::to_string).ok_or_else(|| {
        anyhow!(
            "faction {key} is landless but the target start position has no dead faction record to use as its template"
        )
    })
}

/// Structure plan of one faction against the faction records of the target start position. Used identically by the
/// generator and by `campaign_profile` validation, so every problem is reported before conversion starts.
pub fn structure_for(
    info: &FactionInfo,
    key: &str,
    sp: &StartposStructure,
    rules: &Rules,
) -> Result<Structure> {
    if let Some(d) = &info.donor {
        live_donor_checked(key, d, sp, "the chosen structural donor")?;
    }
    // the faction is a faction of the target database; its record counts only if the start position holds it
    match sp.alive.get(key).copied() {
        Some(own_alive) => Ok(match (info.alive, own_alive) {
            (true, true) | (false, false) => Structure::Own,
            (true, false) => Structure::Live(match &info.donor {
                Some(d) => d.clone(),
                None => live_donor_checked(
                    key,
                    live_donor_for(info, rules),
                    sp,
                    "the default structural donor",
                )?,
            }),
            (false, true) => Structure::Dead(dead_donor_checked(key, sp)?),
        }),
        None if info.alive => match &info.donor {
            Some(d) => Ok(Structure::Live(d.clone())),
            None => bail!(
                "faction {key} exists in the target database but has no record in the target start position \
                 (campaigns/eur_napoleon/startpos.esf) and no structural donor; set mappings.faction_templates[\"{key}\"] \
                 to a live faction (live: {})",
                live_list(sp)
            ),
        },
        None => Ok(Structure::Dead(dead_donor_checked(key, sp)?)),
    }
}

/// donor for a previously dead NTW faction that now holds territory: Islamic / Indian capitals use the Ottoman kit,
/// everything else the minor western kit
fn live_donor_for(info: &FactionInfo, rules: &Rules) -> &'static str {
    if let Some(cap) = &info.capital {
        if let Some(r) = rules.regions.get(cap) {
            let islamic = r.get("religion").and_then(|x| x.as_str()) == Some("rel_islamic");
            let india = r.get("theatre_name").and_then(|x| x.as_str()) == Some("india");
            if islamic || india {
                return DONOR_ISLAMIC;
            }
        }
    }
    DONOR_WEST
}

/// NTW faction whose GOVERNMENT type the faction's settlements use. `has_own`: the target start position holds a record
/// of the faction itself.
pub fn government_donor(st: &Structure, key: &str, has_own: bool) -> String {
    match st {
        Structure::Own => key.to_string(),
        Structure::Live(d) => d.clone(),
        Structure::Dead(d) => {
            if has_own {
                key.to_string()
            } else {
                d.clone()
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ re-id

fn scan_ids(chs: &[Node], found: &mut BTreeSet<u32>) {
    for c in chs {
        match c {
            Node::Value(Value::I32(x)) => {
                let v = *x as u32;
                if (ID_LO..ID_HI).contains(&v) {
                    found.insert(v);
                }
            }
            Node::Value(Value::U32(v)) => {
                if (ID_LO..ID_HI).contains(v) {
                    found.insert(*v);
                }
            }
            Node::Value(Value::Array { code, data }) if matches!(*code & 0x3f, 4 | 8) => {
                for ch in data.chunks_exact(4) {
                    let v = u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                    if (ID_LO..ID_HI).contains(&v) {
                        found.insert(v);
                    }
                }
            }
            Node::Record { name, children, .. } => {
                if !PROTECTED.contains(&name.as_str()) {
                    scan_ids(children, found);
                }
            }
            Node::Array { name, items, .. } => {
                if !PROTECTED.contains(&name.as_str()) {
                    for it in items {
                        scan_ids(it, found);
                    }
                }
            }
            _ => {}
        }
    }
}

fn apply_ids(chs: &mut [Node], map: &BTreeMap<u32, u32>) {
    for c in chs.iter_mut() {
        match c {
            Node::Record { name, children, .. } => {
                if !PROTECTED.contains(&name.as_str()) {
                    apply_ids(children, map);
                }
            }
            Node::Array { name, items, .. } => {
                if !PROTECTED.contains(&name.as_str()) {
                    for it in items.iter_mut() {
                        apply_ids(it, map);
                    }
                }
            }
            _ => {}
        }
    }
    remap_scalars(chs, map);
}

/// Give every object id found in the subtree a fresh id (name allocators excluded). Returns old -> new.
pub fn reid_subtree(children: &mut [Node], ids: &mut IdAlloc) -> BTreeMap<u32, u32> {
    let mut found = BTreeSet::new();
    scan_ids(children, &mut found);
    let map: BTreeMap<u32, u32> = found.into_iter().map(|v| (v, ids.world())).collect();
    apply_ids(children, &map);
    map
}

// ------------------------------------------------------------------------------------------------ character helpers

pub struct CharInfo {
    pub wid: u32,
    pub kind: String,
    pub army: u32,
    pub unit: u32,
}

pub fn char_info(ch: &Node) -> Result<CharInfo> {
    let k = kids(ch)?;
    Ok(CharInfo {
        wid: nth_id(k, 0)?,
        kind: nth_text(k, 1)?.to_string(),
        army: nth_id(k, 2)?,
        unit: nth_id(k, 3)?,
    })
}

/// id of the government post a minister holds (7th scalar of CHARACTER; 0 = unassigned)
pub fn char_post_id(ch: &Node) -> Result<u32> {
    nth_id(kids(ch)?, 6)
}

/// link a minister to its post: post-id scalar + clear the 'unassigned' bool (scalar 9)
pub fn assign_post(ch: &mut Node, post_id: u32) -> Result<()> {
    let k = kids_mut(ch)?;
    let t = scalar_idx(k);
    if t.len() < 10 || ncode(&k[t[6]]) != Some(8) || ncode(&k[t[9]]) != Some(1) {
        bail!("character does not have the expected post-id / unassigned fields");
    }
    k[t[6]] = u32v(post_id);
    k[t[9]] = b(false);
    Ok(())
}

pub fn char_pos(ch: &Node) -> Result<(i32, i32)> {
    let loc = kids(get(ch, &["LOCOMOTABLE"])?)?;
    let x = as_number(nth_val(loc, 0)?).ok_or_else(|| anyhow!("LOCOMOTABLE x"))? as i32;
    let y = as_number(nth_val(loc, 1)?).ok_or_else(|| anyhow!("LOCOMOTABLE y"))? as i32;
    Ok((x, y))
}

pub fn set_char_pos(ch: &mut Node, x: i32, y: i32) -> Result<()> {
    let loc = kids_mut(get_mut(ch, &["LOCOMOTABLE"])?)?;
    let idx = scalar_idx(loc);
    if idx.len() < 4 {
        bail!("LOCOMOTABLE has fewer than four scalars");
    }
    loc[idx[0]] = i32v(x);
    loc[idx[1]] = i32v(y);
    loc[idx[2]] = i32v(x);
    loc[idx[3]] = i32v(y);
    Ok(())
}

/// drop cached path / LOS polygon of a freshly (re)placed character
pub fn clear_char_motion(ch: &mut Node) -> Result<()> {
    {
        let loc = kids_mut(get_mut(ch, &["LOCOMOTABLE"])?)?;
        for c in loc.iter_mut() {
            if is_u32_arr(c) {
                *c = arr_u32(&[]);
            }
        }
    }
    for c in kids_mut(ch)?.iter_mut() {
        if is_u32_arr(c) {
            *c = arr_u32(&[]);
        } else if is_record(c, "LINE_OF_SIGHT") {
            *c = Node::record("LINE_OF_SIGHT", 1, vec![b(false)]);
        }
    }
    Ok(())
}

fn mode_pos(positions: &[(i32, i32)]) -> (i32, i32) {
    let mut counts: Vec<((i32, i32), usize)> = Vec::new();
    for p in positions {
        if *p == (0, 0) {
            continue;
        }
        match counts.iter_mut().find(|(q, _)| q == p) {
            Some(e) => e.1 += 1,
            None => counts.push((*p, 1)),
        }
    }
    let mut best: Option<((i32, i32), usize)> = None;
    for (p, n) in counts {
        if best.map_or(true, |(_, bn)| n > bn) {
            best = Some((p, n));
        }
    }
    best.map(|b| b.0).unwrap_or((0, 0))
}

/// neutral relationship record (clone of a NTW neutral one) towards `other_id`
pub fn empty_diplomacy_relation(tmpl: &Templates, other_id: u32) -> Result<Node> {
    for k in &tmpl.fac_order {
        let rec = &tmpl.fac_by_key[k];
        for it in items(get(
            rec,
            &["DIPLOMACY_MANAGER", "DIPLOMACY_RELATIONSHIPS_ARRAY"],
        )?)? {
            let r = &it[0];
            let st: Vec<&str> = kids(r)?
                .iter()
                .filter(|c| ncode(c) == Some(14))
                .filter_map(as_text)
                .collect();
            if st == ["neutral", "neutral"] {
                let mut n = r.clone();
                kids_mut(&mut n)?[0] = i32_from_id(other_id);
                return Ok(n);
            }
        }
    }
    bail!("no neutral relationship template")
}

// ------------------------------------------------------------------------------------------------ make_faction

fn top_utf16(rec: &Node) -> Result<Vec<(usize, String)>> {
    Ok(kids(rec)?
        .iter()
        .enumerate()
        .filter(|(_, c)| ncode(c) == Some(14))
        .filter_map(|(i, c)| as_text(c).map(|s| (i, s.to_string())))
        .collect())
}

fn top_index(rec: &Node, name: &str) -> Result<usize> {
    idx_of(kids(rec)?, name)
}

fn first_scalar_index(rec: &Node) -> Result<usize> {
    scalar_idx(kids(rec)?)
        .first()
        .copied()
        .ok_or_else(|| anyhow!("faction record has no scalar"))
}

/// replace the same-named records of `rec` by the ones of `src` (pairwise in order)
fn copy_named(rec: &mut Node, src: &Node, name: &str) -> Result<()> {
    let src_l: Vec<Node> = kids(src)?
        .iter()
        .filter(|c| is_record(c, name))
        .cloned()
        .collect();
    let dst_i: Vec<usize> = kids(rec)?
        .iter()
        .enumerate()
        .filter(|(_, c)| is_record(c, name))
        .map(|(i, _)| i)
        .collect();
    let ch = kids_mut(rec)?;
    for (i, c) in dst_i.into_iter().zip(src_l) {
        ch[i] = c;
    }
    Ok(())
}

/// reid_subtree gave the donor's own faction id a fresh value that is wrong for faction K: every nested reference to the
/// donor's id must carry K's own id instead
fn own_id_fix(fid: u32, rec: &mut Node, meta: &mut FactionMeta, donor_rec: &Node) -> Result<()> {
    let did = nth_id(kids(donor_rec)?, 0)?;
    if let Some(&new) = meta.idmap.get(&did) {
        if new != fid {
            let m = BTreeMap::from([(new, fid)]);
            remap_ids(kids_mut(rec)?, &m);
        }
    }
    meta.idmap.insert(did, fid);
    Ok(())
}

// ------------------------------------------------------------------------------------------------ target identity

/// What the start position stores of a faction's identity, read from its row of the target `factions` table: the English
/// screen name, both flag paths and the primary / secondary / uniform colours (8-bit).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DbIdentity {
    pub screen_name: String,
    pub flag: String,
    pub republic_flag: String,
    pub republic_flag_defaulted: bool,
    /// primary r,g,b, secondary r,g,b, uniform r,g,b
    pub colours: [u8; 9],
}

const COLOUR_COLUMNS: [&str; 9] = [
    "primary_colour_r",
    "primary_colour_g",
    "primary_colour_b",
    "secondary_colour_r",
    "secondary_colour_g",
    "secondary_colour_b",
    "uniform_colour_r",
    "uniform_colour_g",
    "uniform_colour_b",
];

fn cell_f64(c: &Cell) -> Option<f64> {
    match c {
        Cell::I16(v) => Some(f64::from(*v)),
        Cell::I32(v) => Some(f64::from(*v)),
        Cell::I64(v) => Some(*v as f64),
        Cell::F32(v) => Some(f64::from(*v)),
        Cell::F64(v) => Some(*v),
        _ => None,
    }
}

/// Identity of faction `key` as the start position can store it. Errors (with the reason) when the row is missing or a
/// field cannot be represented: the converter then blocks instead of letting a donor's identity stand in.
pub fn identity_from_table(t: &Table, key: &str) -> Result<DbIdentity> {
    let kc = t.column("key")?;
    let row = t
        .rows
        .iter()
        .find(|r| r[kc].as_str() == Some(key))
        .ok_or_else(|| anyhow!("faction {key} is not in the target factions table"))?;
    identity_from_cells(key, &|col: &str| Ok(&row[t.column(col)?]))
}

/// [`identity_from_table`] over a column accessor of the faction's row.
pub fn identity_from_cells<'a>(
    key: &str,
    cell: &dyn Fn(&str) -> Result<&'a Cell>,
) -> Result<DbIdentity> {
    let text = |col: &str| -> Result<Option<String>> {
        Ok(cell(col)?
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_owned))
    };
    let screen_name = text("screen_name")?.ok_or_else(|| {
        anyhow!("faction {key}: the target database row has no screen_name, so the start position cannot carry its name")
    })?;
    let flag = text("flags_path")?.ok_or_else(|| {
        anyhow!("faction {key}: the target database row has no flags_path, so the start position cannot carry its flag")
    })?;
    let (republic_flag, republic_flag_defaulted) = match text("republican_flag_path")? {
        Some(p) => (p, false),
        None => (flag.clone(), true),
    };
    let mut colours = [0u8; 9];
    for (slot, col) in colours.iter_mut().zip(COLOUR_COLUMNS) {
        let v = cell_f64(cell(col)?).ok_or_else(|| {
            anyhow!("faction {key}: target database column {col} is not a number")
        })?;
        if v.fract() != 0.0 || !(0.0..=255.0).contains(&v) {
            bail!(
                "faction {key}: target database column {col} = {v} cannot be stored in the start position (needs an integer 0..255)"
            );
        }
        *slot = v as u8;
    }
    Ok(DbIdentity {
        screen_name,
        flag,
        republic_flag,
        republic_flag_defaulted,
        colours,
    })
}

/// Writes `identity` into a donor FACTION record: second identity string (name) and the two FACTION_FLAG_AND_COLOURS
/// records (flag path + nine colour bytes). Fails when the donor's layout differs, instead of leaving donor values.
pub fn apply_db_identity(rec: &mut Node, identity: &DbIdentity) -> Result<()> {
    let strings = top_utf16(rec)?;
    if strings.len() < 3 {
        bail!("faction record lacks its three identity strings");
    }
    let name_at = strings[1].0;
    kids_mut(rec)?[name_at] = s16(identity.screen_name.clone());
    let mut seen = 0;
    for c in kids_mut(rec)?.iter_mut() {
        if !is_record(c, "FACTION_FLAG_AND_COLOURS") {
            continue;
        }
        let k = kids_mut(c)?;
        if k.len() != 10 || ncode(&k[0]) != Some(14) || k[1..].iter().any(|n| ncode(n) != Some(6)) {
            bail!(
                "FACTION_FLAG_AND_COLOURS of the donor does not have the layout flag path + 9 colour bytes"
            );
        }
        k[0] = s16(if seen == 0 {
            identity.flag.clone()
        } else {
            identity.republic_flag.clone()
        });
        for (i, v) in identity.colours.iter().enumerate() {
            k[1 + i] = u8v(*v);
        }
        seen += 1;
    }
    if seen != 2 {
        bail!(
            "the donor record has {seen} FACTION_FLAG_AND_COLOURS records, expected 2 (flag and republican flag)"
        );
    }
    Ok(())
}

/// FACTION_INFOS entry (campaign selection data) of a target database faction the target start position has no entry
/// for: the structural donor's entry carries only the layout. Key, flag path and the leader
/// portrait / description keys are the faction's own - the donor's are blanked, never shown as the faction's.
/// Fails when the donor entry does not have the expected string layout.
pub fn donor_faction_info(donor: &[Node], key: &str, identity: &DbIdentity) -> Result<Vec<Node>> {
    // [0] key, [1] leader portrait, [5] description loc key, [6] flag path
    if donor.len() < 8 || [0, 1, 5, 6].iter().any(|&i| ncode(&donor[i]) != Some(14)) {
        bail!(
            "faction {key}: the FACTION_INFOS entry of the structural donor does not have the expected layout (key, portrait, description key and flag path as strings)"
        );
    }
    let mut it = donor.to_vec();
    it[0] = s16(key);
    it[1] = s16("");
    it[5] = s16("");
    it[6] = s16(identity.flag.clone());
    Ok(it)
}

/// Returns (FACTION record, meta) for faction `k`.
pub fn make_faction(
    ctx: &mut BuildContext,
    tmpl: &Templates,
    k: &str,
    st: &Structure,
) -> Result<(Node, FactionMeta)> {
    let info = ctx
        .factions
        .get(k)
        .ok_or_else(|| anyhow!("unknown faction {k}"))?
        .clone();
    let fid = *ctx
        .fid
        .get(k)
        .ok_or_else(|| anyhow!("no id for faction {k}"))?;
    let mut meta = FactionMeta {
        key: k.to_string(),
        donor: k.to_string(),
        ..Default::default()
    };
    // the target start position holds a record of this very faction
    let has_own = tmpl.fac_by_key.contains_key(k);
    let mut rec: Node;
    match st {
        Structure::Own => {
            rec = tmpl
                .fac_by_key
                .get(k)
                .ok_or_else(|| anyhow!("faction {k} has no record in the target start position"))?
                .clone();
            meta.own_structure = true;
        }
        Structure::Live(dn) | Structure::Dead(dn) => {
            let dn: &str = dn.as_str();
            let donor_rec = tmpl.fac_by_key.get(dn).ok_or_else(|| {
                anyhow!("{k}: structural donor {dn} has no record in the target start position")
            })?;
            rec = donor_rec.clone();
            meta.idmap = reid_subtree(kids_mut(&mut rec)?, &mut ctx.ids);
            meta.donor = if has_own && matches!(st, Structure::Dead(_)) {
                k.to_string()
            } else {
                dn.to_string()
            };
            let id_i = first_scalar_index(&rec)?;
            kids_mut(&mut rec)?[id_i] = i32_from_id(fid);
            own_id_fix(fid, &mut rec, &mut meta, donor_rec)?;
            if has_own {
                // identity / look of the real faction carried over from its own NTW record
                let own = &tmpl.fac_by_key[k];
                copy_named(&mut rec, own, "FACTION_FLAG_AND_COLOURS")?;
                for name in PROTECTED {
                    copy_named(&mut rec, own, name)?;
                }
                let u_own = top_utf16(own)?;
                let u_dst = top_utf16(&rec)?;
                if u_own.len() < 3 || u_dst.len() < 3 {
                    bail!("{k}: faction record lacks its three identity strings");
                }
                let ch = kids_mut(&mut rec)?;
                for n in 0..3 {
                    ch[u_dst[n].0] = s16(u_own[n].1.clone());
                }
            } else {
                // A faction of the target database without a record in the target start position: the donor supplies the
                // record structure only. Name, flags and colours are the target database row's; the donor's identity is
                // never presented as the faction's own.
                let u_d = top_utf16(&rec)?;
                if u_d.len() < 3 {
                    bail!("{k}: faction record lacks its three identity strings");
                }
                kids_mut(&mut rec)?[u_d[0].0] = s16(k);
                // state religion: the ETW counterpart's when there is one, else the donor's
                if let Some(etw) = ctx.etw_factions.get(&info.etw_source).cloned() {
                    let u_e: Vec<String> = kids(&etw)?
                        .iter()
                        .filter(|c| ncode(c) == Some(14))
                        .filter_map(|c| as_text(c).map(str::to_string))
                        .collect();
                    if u_e.len() >= 3 {
                        kids_mut(&mut rec)?[u_d[2].0] = s16(u_e[2].clone());
                    }
                }
                let table = crate::db::load_campaign_table(
                    ctx.snapshot,
                    ctx.rules,
                    crate::db::CampaignTable::Factions,
                )?;
                let identity = identity_from_table(&table, k)?;
                apply_db_identity(&mut rec, &identity).with_context(|| {
                    format!("{k}: structural donor {dn} cannot carry the identity of the target database row")
                })?;
                ctx.notes.push(format!(
                    "Faction {k} has no record in the target start position: record structure from {dn}, name / flags / colours from the target database{}",
                    if identity.republic_flag_defaulted {
                        " (no republican flag defined: the main flag is used for both)"
                    } else {
                        ""
                    }
                ));
            }
        }
    }
    kids_mut(get_mut(&mut rec, &["CAMPAIGN_PLAYER_SETUP"])?)?[2] = s16(k);
    Ok((rec, meta))
}

/// First governor post (POSTS_ARRAY item) of a live record: the western kit first, then the start position's order.
fn governor_post_donor(tmpl: &Templates) -> Result<Option<Vec<Node>>> {
    let first = std::iter::once(DONOR_WEST.to_string());
    for key in first.chain(tmpl.fac_order.iter().cloned()) {
        let Some(rec) = tmpl.fac_by_key.get(&key) else {
            continue;
        };
        if !own_is_alive(rec)? {
            continue;
        }
        for it in items(get(rec, &["GOVERNMENT", "POSTS_ARRAY"])?)? {
            let is_governor = kids(&it[0])
                .ok()
                .and_then(|c| text_at(c, 1).ok())
                .is_some_and(|s| s.starts_with("governor"));
            if is_governor {
                return Ok(Some(it.clone()));
            }
        }
    }
    Ok(None)
}

// ------------------------------------------------------------------------------------------------ finish_faction

/// capital relocation, character clean-up, governorships, capital region ids
pub fn finish_faction(
    ctx: &mut BuildContext,
    tmpl: &Templates,
    k: &str,
    rec: &mut Node,
    meta: &mut FactionMeta,
) -> Result<()> {
    let info = ctx
        .factions
        .get(k)
        .ok_or_else(|| anyhow!("unknown faction {k}"))?
        .clone();
    let fid = ctx.fid[k];
    let cap_key = info.capital.clone();
    let new_cap = match &cap_key {
        Some(c) => {
            ctx.region_info
                .get(c)
                .ok_or_else(|| anyhow!("{k}: capital {c} is not a region"))?
                .pos
        }
        None => (0, 0),
    };
    let inv: BTreeMap<u32, u32> = meta.idmap.iter().map(|(a, b)| (*b, *a)).collect();

    let old_items = std::mem::take(items_mut(get_mut(rec, &["CHARACTER_ARRAY"])?)?);
    let mut stationary = Vec::new();
    for it in &old_items {
        let ci = char_info(&it[0])?;
        if ci.army == 0 && !DEPLOYED_TYPES.contains(&ci.kind.as_str()) {
            stationary.push(char_pos(&it[0])?);
        }
    }
    let old_cap = mode_pos(&stationary);
    let mut kept: Vec<Vec<Node>> = Vec::new();
    for mut it in old_items {
        let ci = char_info(&it[0])?;
        if ci.army != 0 || DEPLOYED_TYPES.contains(&ci.kind.as_str()) {
            continue;
        }
        if char_pos(&it[0])? == old_cap && old_cap != (0, 0) {
            if new_cap == (0, 0) {
                set_char_pos(&mut it[0], 0, 0)?;
            } else {
                set_char_pos(&mut it[0], new_cap.0, new_cap.1)?;
            }
        }
        clear_char_motion(&mut it[0])?;
        meta.chars.push(MetaChar {
            wid: ci.wid,
            twid: inv.get(&ci.wid).copied().unwrap_or(ci.wid),
            kind: ci.kind,
            unit: ci.unit,
        });
        kept.push(it);
    }

    // Posts held by a removed (deployed) character (e.g. faction_leader = Napoleon General): the engine dereferences the
    // post's character at post-load and crashes on NULL. Hand such posts to a spare minister.
    let kept_ids: BTreeSet<u32> = kept
        .iter()
        .map(|it| char_info(&it[0]).map(|c| c.wid))
        .collect::<Result<_>>()?;
    let mut spare: Vec<usize> = Vec::new();
    for (i, it) in kept.iter().enumerate() {
        if char_info(&it[0])?.kind == "minister" && char_post_id(&it[0])? == 0 {
            spare.push(i);
        }
    }
    {
        let posts = items_mut(get_mut(rec, &["GOVERNMENT", "POSTS_ARRAY"])?)?;
        for it in posts.iter_mut() {
            let pk = kids_mut(&mut it[0])?;
            let cw = id_at(pk, 2)?;
            if cw != 0 && !kept_ids.contains(&cw) {
                if spare.is_empty() {
                    bail!("{k}: no spare minister for post {}", text_at(pk, 1)?);
                }
                let ni = spare.remove(0);
                let nwid = char_info(&kept[ni][0])?.wid;
                pk[2] = u32v(nwid);
                let post_id = id_at(pk, 0)?;
                assign_post(&mut kept[ni][0], post_id)?;
            }
        }
    }
    items_mut(get_mut(rec, &["ARMY_ARRAY"])?)?.clear();

    // capital region id (the two i32 after the FORT_UPGRADE_MANAGER records)
    if info.alive {
        if let Some(c) = &cap_key {
            let cid = ctx.region_info[c].world_id;
            let ch = kids_mut(rec)?;
            let fum: Vec<usize> = ch
                .iter()
                .enumerate()
                .filter(|(_, c)| is_record(c, "FORT_UPGRADE_MANAGER"))
                .map(|(i, _)| i)
                .collect();
            let j = *fum
                .last()
                .ok_or_else(|| anyhow!("{k}: no FORT_UPGRADE_MANAGER"))?
                + 1;
            if ncode(&ch[j]) != Some(4) || ncode(&ch[j + 1]) != Some(4) {
                bail!("{k}: capital region id fields not found after FORT_UPGRADE_MANAGER");
            }
            ch[j] = i32_from_id(cid);
            ch[j + 1] = i32_from_id(cid);
        }
    }

    // governorships
    let owned: Vec<String> = ctx
        .region_info
        .iter()
        .filter(|(_, v)| v.owner == k)
        .map(|(key, _)| key.clone())
        .collect();
    meta.owned = owned.clone();
    meta.gov_posts.clear();
    if info.alive {
        let cap = cap_key
            .clone()
            .ok_or_else(|| anyhow!("{k} is alive but has no capital"))?;
        let mut th_count: BTreeMap<String, usize> = BTreeMap::new();
        for r in &owned {
            *th_count
                .entry(ctx.region_info[r].theatre.clone())
                .or_default() += 1;
        }
        let main_th = ctx.region_info[&cap].theatre.clone();
        let mut theatres: Vec<String> = th_count.keys().cloned().collect();
        theatres.sort_by(|a, b| ((*a != main_th), a).cmp(&((*b != main_th), b)));

        let posts = items_mut(get_mut(rec, &["GOVERNMENT", "POSTS_ARRAY"])?)?;
        let mut existing: Vec<usize> = Vec::new();
        for (i, it) in posts.iter().enumerate() {
            if text_at(kids(&it[0])?, 1)?.starts_with("governor") {
                existing.push(i);
            }
        }
        // a structure without governor posts (never the case for the shipped NTW records) borrows one from a live record
        // of the target start position (the western kit when it has governor posts)
        let (base_post, base_is_fresh): (Vec<Node>, bool) = match existing.first() {
            Some(&i) => (posts[i].clone(), false),
            None => {
                let found = governor_post_donor(tmpl)?.ok_or_else(|| {
                    anyhow!("{k}: neither the faction nor any live faction of the target start position has a governor post")
                })?;
                (found, true)
            }
        };
        let remaining: Vec<Vec<Node>> = std::mem::take(posts)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !existing.contains(i))
            .map(|(_, it)| it)
            .collect();
        *posts = remaining;

        for (n, th) in theatres.iter().enumerate() {
            let fresh = n > 0 || base_is_fresh;
            let mut post_item: Vec<Node> = if fresh {
                vec![base_post[0].clone()]
            } else {
                base_post.clone()
            };
            if fresh {
                // fresh post, governorship and governor (cloned minister)
                let mut cands: Vec<usize> = Vec::new();
                for (i, it) in kept.iter().enumerate() {
                    if char_info(&it[0])?.kind == "minister" && char_post_id(&it[0])? == 0 {
                        cands.push(i);
                    }
                }
                let src_i = match cands.last() {
                    Some(&i) => i,
                    None => {
                        let mut last = None;
                        for (i, it) in kept.iter().enumerate() {
                            if char_info(&it[0])?.kind == "minister" {
                                last = Some(i);
                            }
                        }
                        last.ok_or_else(|| {
                            anyhow!("{k}: no minister available to become governor")
                        })?
                    }
                };
                let mut nc = kept[src_i][0].clone();
                let old_id = char_info(&nc)?.wid;
                let new_id = ctx.ids.world();
                remap_ids(kids_mut(&mut nc)?, &BTreeMap::from([(old_id, new_id)]));
                if new_cap != (0, 0) {
                    set_char_pos(&mut nc, new_cap.0, new_cap.1)?;
                }
                let src_meta = meta
                    .chars
                    .iter()
                    .find(|m| m.wid == old_id)
                    .cloned()
                    .ok_or_else(|| {
                        anyhow!("{k}: governor template character {old_id} is not tracked")
                    })?;
                meta.chars.push(MetaChar {
                    wid: new_id,
                    ..src_meta
                });
                let post_id = ctx.ids.world();
                {
                    let pk = kids_mut(&mut post_item[0])?;
                    pk[0] = i32_from_id(post_id);
                    pk[2] = u32v(new_id);
                }
                assign_post(&mut nc, post_id)?;
                kept.push(vec![nc]);
            }
            let p = &mut post_item[0];
            let gkeys: Vec<String> = owned
                .iter()
                .filter(|r| ctx.region_info[*r].theatre == *th)
                .cloned()
                .collect();
            let world_ids: Vec<u32> = gkeys.iter().map(|r| ctx.region_info[r].world_id).collect();
            let post_key = theatre_post(th)?;
            let pk = kids_mut(p)?;
            pk[1] = s16(post_key);
            let gs_i = pk
                .iter()
                .position(|c| is_record(c, "GOVERNORSHIP"))
                .ok_or_else(|| anyhow!("{k}: post without GOVERNORSHIP"))?;
            let new_gov_id = if fresh { Some(ctx.ids.world()) } else { None };
            let char_id = id_at(pk, 2)?;
            let gsk = kids_mut(&mut pk[gs_i])?;
            let ai = gsk
                .iter()
                .position(is_u32_arr)
                .ok_or_else(|| anyhow!("{k}: GOVERNORSHIP without region list"))?;
            gsk[ai] = arr_u32(&world_ids);
            let fi = gsk
                .iter()
                .position(|c| ncode(c) == Some(8))
                .ok_or_else(|| anyhow!("{k}: GOVERNORSHIP without faction id"))?;
            gsk[fi] = u32v(fid);
            let gi = gsk
                .iter()
                .position(|c| ncode(c) == Some(4))
                .ok_or_else(|| anyhow!("{k}: GOVERNORSHIP without id"))?;
            if let Some(g) = new_gov_id {
                gsk[gi] = i32_from_id(g);
            }
            let gov_id = id_at(gsk, gi)?;
            meta.gov_posts.push(GovPost {
                char_id,
                gov_id,
                theatre: th.clone(),
                post: post_key.to_string(),
                regions: gkeys,
                cai: 0,
            });
            posts.push(post_item);
        }
        // the list of governorship ids kept at faction level (array right after MORGUE)
        let mi = top_index(rec, "MORGUE")?;
        let ch = kids_mut(rec)?;
        let gov_ids: Vec<u32> = meta.gov_posts.iter().map(|g| g.gov_id).collect();
        let j = (mi + 1..ch.len())
            .find(|&j| is_u32_arr(&ch[j]))
            .ok_or_else(|| anyhow!("{k}: governorship id list not found after MORGUE"))?;
        ch[j] = arr_u32(&gov_ids);
    }
    *items_mut(get_mut(rec, &["CHARACTER_ARRAY"])?)? = kept;
    Ok(())
}

// ------------------------------------------------------------------------------------------------ diplomacy / shapes

pub fn set_player_flags(cps: &mut Node, playable: bool) -> Result<()> {
    let ch = kids_mut(cps)?;
    let idx: Vec<usize> = ch
        .iter()
        .enumerate()
        .filter(|(_, c)| ncode(c) == Some(1))
        .map(|(i, _)| i)
        .collect();
    if idx.len() != 3 {
        bail!(
            "CAMPAIGN_PLAYER_SETUP has {} bool flags, expected 3",
            idx.len()
        );
    }
    ch[idx[0]] = b(false);
    ch[idx[1]] = b(playable);
    ch[idx[2]] = b(false);
    Ok(())
}

/// Diplomacy: NTW eur states between factions that both use their own live NTW record are kept (wars/alliances),
/// everything else is neutral.
pub fn build_diplomacy(ctx: &mut BuildContext, tmpl: &Templates) -> Result<()> {
    let fkeys = ctx.fkeys.clone();
    let mut war: BTreeSet<(String, String)> = BTreeSet::new();
    for k in &fkeys {
        let keep_k = ctx.fmeta[k].own_structure && ctx.factions[k].alive;
        let rec = ctx
            .frec
            .get_mut(k)
            .ok_or_else(|| anyhow!("no record for {k}"))?;
        let arr = get_mut(rec, &["DIPLOMACY_MANAGER", "DIPLOMACY_RELATIONSHIPS_ARRAY"])?;
        let old: BTreeMap<u32, Node> = items(arr)?
            .iter()
            .map(|it| Ok((id_at(kids(&it[0])?, 0)?, it[0].clone())))
            .collect::<Result<_>>()?;
        let mut new_items: Vec<Vec<Node>> = Vec::new();
        for o in &fkeys {
            if o == k {
                continue;
            }
            let oid = ctx.fid[o];
            let keep = keep_k
                && ctx.fmeta[o].own_structure
                && ctx.factions[o].alive
                && old.contains_key(&oid);
            let r = if keep {
                old[&oid].clone()
            } else {
                empty_diplomacy_relation(tmpl, oid)?
            };
            let st = kids(&r)?
                .iter()
                .find(|c| ncode(c) == Some(14))
                .and_then(as_text);
            if st == Some("war") {
                war.insert((k.clone(), o.clone()));
            }
            new_items.push(vec![r]);
        }
        *items_mut(arr)? = new_items;
    }
    ctx.war = war;
    Ok(())
}

/// playable factions carry MISSION_MANAGER / SHROUD / VICTORY_CONDITIONS, others not
pub fn fix_shapes(ctx: &mut BuildContext, tmpl: &Templates) -> Result<()> {
    // shape template of a playable faction: france when the target start position has it (vanilla), else the first record
    // that carries the playable-only records
    let has_playable_shape = |rec: &Node| -> bool {
        kids(rec).is_ok_and(|c| {
            c.iter().any(|n| is_record(n, "CAMPAIGN_MISSION_MANAGER"))
                && c.iter()
                    .any(|n| is_record(n, "CAMPAIGN_VICTORY_CONDITIONS"))
        })
    };
    let fr = std::iter::once("france".to_string())
        .chain(tmpl.fac_order.iter().cloned())
        .filter_map(|k| tmpl.fac_by_key.get(&k))
        .find(|rec| has_playable_shape(*rec))
        .ok_or_else(|| {
            anyhow!("no faction record of the target start position carries a mission manager and victory conditions to use as playable template")
        })?;
    let mm_t = get(fr, &["CAMPAIGN_MISSION_MANAGER"])?.clone();
    let vc_t = get(fr, &["CAMPAIGN_VICTORY_CONDITIONS"])?.clone();
    let etw_britain = ctx
        .etw_factions
        .get("britain")
        .ok_or_else(|| anyhow!("ETW britain record missing"))?;
    let empty_tree = kids(get(etw_britain, &["CAMPAIGN_SHROUD"])?)?
        .get(2)
        .ok_or_else(|| anyhow!("ETW shroud has no trees"))?
        .clone();
    let sh_t = Node::record(
        "CAMPAIGN_SHROUD",
        1,
        vec![empty_tree.clone(), empty_tree.clone(), empty_tree, b(true)],
    );
    let fkeys = ctx.fkeys.clone();
    for k in &fkeys {
        let want = ctx.factions[k].playable;
        let rec = ctx
            .frec
            .get_mut(k)
            .ok_or_else(|| anyhow!("no record for {k}"))?;
        set_player_flags(get_mut(rec, &["CAMPAIGN_PLAYER_SETUP"])?, want)?;
        let ch = kids_mut(rec)?;
        let has = ch.iter().any(|c| is_record(c, "CAMPAIGN_SHROUD"));
        if has {
            let i = idx_of(ch, "CAMPAIGN_SHROUD")?;
            ch[i] = sh_t.clone();
        }
        if want && !has {
            let fi = idx_of(ch, "FAMILY")?;
            ch.insert(fi + 1, mm_t.clone());
            let mi = idx_of(ch, "MORGUE")?;
            ch.insert(mi + 3, sh_t.clone());
            let ui = idx_of(ch, "FORT_UPGRADE_MANAGER")?;
            ch.insert(ui, vc_t.clone());
        } else if has && !want {
            for nm in [
                "CAMPAIGN_MISSION_MANAGER",
                "CAMPAIGN_SHROUD",
                "CAMPAIGN_VICTORY_CONDITIONS",
            ] {
                let i = idx_of(ch, nm).with_context(|| format!("{k}: removing {nm}"))?;
                ch.remove(i);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn rules() -> Rules {
        Rules {
            root: PathBuf::new(),
            regions: json!({
                "delhi": {"religion": "rel_islamic", "theatre_name": "india"},
                "paris": {"religion": "rel_catholic", "theatre_name": "europe"}
            }),
            factions: json!({}),
            units: json!({}),
            buildings: json!({}),
            buildings_by_slot_type: json!({}),
            slot_types: json!({}),
            armies: json!({}),
        }
    }

    fn records(records: &[(&str, bool)]) -> StartposStructure {
        StartposStructure {
            order: records.iter().map(|(k, _)| k.to_string()).collect(),
            alive: records.iter().map(|(k, a)| (k.to_string(), *a)).collect(),
            units: BTreeSet::new(),
        }
    }

    fn vanilla_like() -> StartposStructure {
        records(&[
            ("france", true),
            ("belgium", false),
            ("hungary", false),
            ("ottomans", true),
            ("hessen", true),
        ])
    }

    fn info(alive: bool, capital: Option<&str>) -> FactionInfo {
        FactionInfo {
            alive,
            capital: capital.map(str::to_owned),
            playable: false,
            regions: usize::from(alive),
            etw_source: String::new(),
            donor: None,
        }
    }

    fn plan(key: &str, info: &FactionInfo, sp: &StartposStructure) -> Result<Structure> {
        let mut info = info.clone();
        if info.etw_source.is_empty() {
            info.etw_source = key.to_owned();
        }
        structure_for(&info, key, sp, &rules())
    }

    #[test]
    fn own_records_are_used_unchanged_and_revived_or_buried_through_donors() {
        let sp = vanilla_like();
        assert_eq!(
            plan("france", &info(true, Some("paris")), &sp).unwrap(),
            Structure::Own
        );
        assert_eq!(
            plan("belgium", &info(false, None), &sp).unwrap(),
            Structure::Own
        );
        // landless faction with a live record is buried into the dead template
        assert_eq!(
            plan("france", &info(false, None), &sp).unwrap(),
            Structure::Dead("hungary".into())
        );
        // revived faction: capital decides the default kit
        assert_eq!(
            plan("belgium", &info(true, Some("paris")), &sp).unwrap(),
            Structure::Live(DONOR_WEST.into())
        );
        assert_eq!(
            plan("belgium", &info(true, Some("delhi")), &sp).unwrap(),
            Structure::Live(DONOR_ISLAMIC.into())
        );
    }

    #[test]
    fn explicit_donor_wins_and_must_be_live() {
        let sp = vanilla_like();
        let mut revived = info(true, Some("paris"));
        revived.donor = Some("france".into());
        assert_eq!(
            plan("belgium", &revived, &sp).unwrap(),
            Structure::Live("france".into())
        );
        revived.donor = Some("hungary".into());
        let err = format!("{:#}", plan("belgium", &revived, &sp).unwrap_err());
        assert!(
            err.contains("dead record") && err.contains("faction_templates"),
            "{err}"
        );
        revived.donor = Some("atlantis".into());
        let err = format!("{:#}", plan("belgium", &revived, &sp).unwrap_err());
        assert!(err.contains("no record"), "{err}");
    }

    #[test]
    fn target_factions_without_a_start_position_record_need_a_donor() {
        let sp = vanilla_like();
        // exists in the target database, no record in the start position: the capital's kit never stands in
        let alive = info(true, Some("paris"));
        let err = format!("{:#}", plan("mod_france", &alive, &sp).unwrap_err());
        assert!(
            err.contains("faction_templates") && err.contains("live: france, ottomans, hessen"),
            "{err}"
        );
        let mut islamic = info(true, Some("delhi"));
        assert!(plan("mod_mughal", &islamic, &sp).is_err());
        islamic.donor = Some("ottomans".into());
        assert_eq!(
            plan("mod_mughal", &islamic, &sp).unwrap(),
            Structure::Live("ottomans".into())
        );
        let mut with_donor = alive.clone();
        with_donor.donor = Some("hessen".into());
        assert_eq!(
            plan("mod_france", &with_donor, &sp).unwrap(),
            Structure::Live("hessen".into())
        );
        // landless: the dead template
        assert_eq!(
            plan("mod_france", &info(false, None), &sp).unwrap(),
            Structure::Dead("hungary".into())
        );
    }

    #[test]
    fn missing_default_donors_and_dead_templates_block_with_the_fix() {
        // the target start position has no ottomans: a revived faction with an Islamic capital needs an explicit donor
        let sp = records(&[
            ("belgium", false),
            ("france", true),
            ("hungary", false),
            ("hessen", true),
        ]);
        let err = format!(
            "{:#}",
            plan("belgium", &info(true, Some("delhi")), &sp).unwrap_err()
        );
        assert!(
            err.contains("ottomans") && err.contains("faction_templates"),
            "{err}"
        );
        // no dead record at all: landless factions cannot be built
        let all_live = records(&[("france", true), ("hessen", true)]);
        let err = format!(
            "{:#}",
            plan("x", &info(false, None), &all_live).unwrap_err()
        );
        assert!(err.contains("no dead faction record"), "{err}");
        // another dead record stands in when hungary is absent
        let other_dead = records(&[("france", true), ("belgium", false)]);
        assert_eq!(other_dead.dead_donor(), Some("belgium"));
    }

    #[test]
    fn government_follows_the_structure_that_provides_the_record() {
        assert_eq!(government_donor(&Structure::Own, "france", true), "france");
        assert_eq!(
            government_donor(&Structure::Live("hessen".into()), "x", false),
            "hessen"
        );
        // buried faction with its own live record keeps its government type; without a record the dead template's
        assert_eq!(
            government_donor(&Structure::Dead("hungary".into()), "france", true),
            "france"
        );
        assert_eq!(
            government_donor(&Structure::Dead("hungary".into()), "x", false),
            "hungary"
        );
    }

    fn row(overrides: &[(&str, Cell)]) -> BTreeMap<String, Cell> {
        let mut m = BTreeMap::new();
        m.insert("screen_name".to_string(), Cell::Str("Mod Empire".into()));
        m.insert(
            "flags_path".to_string(),
            Cell::Str("data\\ui\\flags\\mod".into()),
        );
        m.insert(
            "republican_flag_path".to_string(),
            Cell::Str("data\\ui\\flags\\mod_republic".into()),
        );
        for (i, c) in COLOUR_COLUMNS.iter().enumerate() {
            m.insert(c.to_string(), Cell::F32(i as f32 * 10.0));
        }
        for (k, v) in overrides {
            m.insert(k.to_string(), v.clone());
        }
        m
    }

    fn identity(row: &BTreeMap<String, Cell>) -> Result<DbIdentity> {
        identity_from_cells("mod_empire", &|c: &str| {
            row.get(c).ok_or_else(|| anyhow!("no column {c}"))
        })
    }

    #[test]
    fn database_identity_is_read_from_the_target_row() {
        let id = identity(&row(&[])).unwrap();
        assert_eq!(id.screen_name, "Mod Empire");
        assert_eq!(id.flag, "data\\ui\\flags\\mod");
        assert_eq!(id.republic_flag, "data\\ui\\flags\\mod_republic");
        assert!(!id.republic_flag_defaulted);
        assert_eq!(id.colours, [0, 10, 20, 30, 40, 50, 60, 70, 80]);
        let id = identity(&row(&[("republican_flag_path", Cell::Null)])).unwrap();
        assert_eq!(id.republic_flag, id.flag);
        assert!(id.republic_flag_defaulted);
    }

    #[test]
    fn identity_the_record_cannot_hold_is_refused_with_the_reason() {
        for (overrides, needle) in [
            (vec![("screen_name", Cell::Null)], "screen_name"),
            (vec![("flags_path", Cell::Str(String::new()))], "flags_path"),
            (
                vec![("secondary_colour_g", Cell::F32(12.5))],
                "secondary_colour_g",
            ),
            (
                vec![("uniform_colour_b", Cell::F32(300.0))],
                "uniform_colour_b",
            ),
            (
                vec![("primary_colour_r", Cell::Str("red".into()))],
                "primary_colour_r",
            ),
        ] {
            let err = format!("{:#}", identity(&row(&overrides)).unwrap_err());
            assert!(err.contains(needle) && err.contains("mod_empire"), "{err}");
        }
    }

    fn flag(path: &str) -> Node {
        let mut c = vec![s16(path)];
        c.extend((0..9).map(|_| u8v(1)));
        Node::record("FACTION_FLAG_AND_COLOURS", 1, c)
    }

    fn donor_record(flags: usize) -> Node {
        let mut c = vec![s16("hessen"), s16("Hessen"), s16("rel_protestant")];
        for _ in 0..flags {
            c.push(flag("data\\ui\\flags\\hessen"));
        }
        Node::record("FACTION", 1, c)
    }

    #[test]
    fn donor_record_carries_the_database_name_flags_and_colours_not_the_donors() {
        let mut rec = donor_record(2);
        apply_db_identity(&mut rec, &identity(&row(&[])).unwrap()).unwrap();
        let ch = kids(&rec).unwrap();
        assert_eq!(as_text(&ch[0]), Some("hessen")); // the key is set by the caller
        assert_eq!(as_text(&ch[1]), Some("Mod Empire"));
        assert_eq!(as_text(&ch[2]), Some("rel_protestant"));
        let flags: Vec<&Node> = ch
            .iter()
            .filter(|n| is_record(n, "FACTION_FLAG_AND_COLOURS"))
            .collect();
        assert_eq!(
            as_text(&kids(flags[0]).unwrap()[0]),
            Some("data\\ui\\flags\\mod")
        );
        assert_eq!(
            as_text(&kids(flags[1]).unwrap()[0]),
            Some("data\\ui\\flags\\mod_republic")
        );
        let colours: Vec<f64> = kids(flags[1])
            .unwrap()
            .iter()
            .filter_map(as_number)
            .collect();
        assert_eq!(
            colours,
            vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        );
    }

    #[test]
    fn donor_with_an_unexpected_flag_layout_is_refused_instead_of_keeping_its_look() {
        let mut one_flag = donor_record(1);
        let err = format!(
            "{:#}",
            apply_db_identity(&mut one_flag, &identity(&row(&[])).unwrap()).unwrap_err()
        );
        assert!(err.contains("expected 2"), "{err}");
    }

    fn info_entry(portrait: Node) -> Vec<Node> {
        vec![
            s16("hessen"),
            portrait,
            b(true),
            b(true),
            i32v(1),
            s16("start_pos_factions_description_hessen"),
            s16("data\\ui\\flags\\hessen"),
            i32v(5),
        ]
    }

    #[test]
    fn donor_faction_info_carries_the_faction_key_and_flag_but_not_the_donors_text_keys() {
        let entry = donor_faction_info(
            &info_entry(s16("data\\ui\\portraits\\hessen")),
            "mod_empire",
            &identity(&row(&[])).unwrap(),
        )
        .unwrap();
        assert_eq!(entry.len(), 8);
        assert_eq!(as_text(&entry[0]), Some("mod_empire"));
        assert_eq!(as_text(&entry[1]), Some(""));
        assert_eq!(as_text(&entry[5]), Some(""));
        assert_eq!(as_text(&entry[6]), Some("data\\ui\\flags\\mod"));
        // numeric/bool fields stay the donor's until the caller sets playable flag, order and region count
        assert_eq!(as_number(&entry[7]), Some(5.0));
        // a donor entry whose portrait is not a string is refused instead of being guessed
        let err = format!(
            "{:#}",
            donor_faction_info(
                &info_entry(i32v(1)),
                "mod_empire",
                &identity(&row(&[])).unwrap()
            )
            .unwrap_err()
        );
        assert!(
            err.contains("expected layout") && err.contains("mod_empire"),
            "{err}"
        );
    }
}
