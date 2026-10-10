//! `campaigns/<campaign>/startpos.esf`: add a region cloned from an existing one (the "split" of `eur_lithuania` into
//! `eur_lithuania` + `eur_samogitia`).
//!
//! Everything that makes the region exist in a vanilla NTW `eur_napoleon` startpos was traced on the vanilla file
//! (`work/research/ssplit/*.py`); [`add_region`] writes all of it, [`check`] re-verifies the result:
//!
//! * `WORLD/REGION_MANAGER/REGIONS_ARRAY`: a REGION clone of the donor (child 0 key, 4 region id, 20 owner, 22 governorship,
//!   25 rebel name, 27 recruitment manager id, 41 name key) with the donor's settlement (SIEGEABLE_GARRISON_RESIDENCE ids,
//!   fixed 12.20 positions, name key), the requested settlement slots (fresh slot ids, keys, positions; every
//!   non-settlement slot of the donor is dropped) and zeroed garrison pointers.
//! * owner faction `GOVERNMENT/POSTS_ARRAY/.../GOVERNORSHIP` region id list, `CAMPAIGN_PREOPEN_MAP_INFO/REGION_OWNERSHIPS_BY_THEATRE`,
//!   `CAMPAIGN_TRADE_MANAGER/SETTLEMENT_INDICES`.
//! * trade node numbering: node indices are `ports, settlements, trade nodes` (the generated world startpos leaves a
//!   gap between the last settlement and the first trade node); the new settlement takes index `ports + settlements`
//!   and every node reference at or above it (`TRADE_ROUTES`, `TRADE_NODES`, `DOMESTIC_TRADE_ROUTE` child 6) moves up
//!   by one, exactly as `trade_routes::add_settlement` renumbers when it appends a settlement node. One direct (empty
//!   spline list) `TRADE_ROUTES` entry per land neighbour is appended at the end so no existing route index changes.
//! * owner: the donor's owner by default; `NewRegion::owner` hands the residences / buildings to another faction that
//!   already governs the donor's theatre (REGION owner and governorship link, owner CAI faction and governorship lists).
//! * `CAI_INTERFACE/CAI_WORLD`: CAI_REGION, CAI_SETTLEMENT, CAI_BUILDING_SLOT x3, CAI_REGION_HLCI, CAI_REGION_BOUNDARY (one
//!   per neighbour), the theatre region list (index = regions.esf index), the owner CAI_FACTION owned-region list, the CAI
//!   governorship region list and the neighbours' boundary lists; root pool analysers #11 / #20 receive one per-region
//!   belief each (as `startpos::cai` does for every region); `CAI_CENTRAL_BDI_POOL` child 0 (the next CAI id) is advanced.
//!
//! Activation of a wilderness region ([`NewRegion::existing_cai`], [`new_region_for_activation`]): the generated world
//! startpos already holds a `CAI_WORLD_REGIONS` item (with its beliefs, usually its boundaries, HLCIs and theatre entry)
//! for every wilderness polygon at its `regions.esf` index, but no REGION in `REGIONS_ARRAY`. The same steps run; the
//! CAI item is updated in place (settlement id, world region id, governorship, settlement position, empty slot list),
//! a settlement HLCI of the donor's kind is moved to the settlement (or added), boundaries the item lacks are added
//! (existing ones keep their ETW lengths), and the theatre list gets the region only when it is missing.
//!
//! Not written (see [`Report::skipped`]): the per-faction manager pools and the root pool's other per-region analyser
//! beliefs - the engine creates those lazily, the converter ships every region without them.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::esf::{self, Node, Value};
use crate::startpos::cai::nodes as nd;
use crate::regionmap::geometry::P;
use crate::regionmap::regions::{RegionData, settlement_center};
use crate::startpos::helpers::{IdAlloc, collect_used};

const WORLD_ID_START: u32 = 1_400_000_000;
const FIXED: f64 = 1_048_576.0; // 12.20 fixed point
/// CAI_SITUATED kind / CAI_REGION_HLCI kind of a settlement
const KIND_SETTLEMENT: u32 = 4;
const ROLE_ROAD: &str = "settlement_road";
const ROLE_FORT: &str = "settlement_fortification";

#[derive(Clone, Debug, Default)]
pub struct NewRegion {
    pub key: String,
    /// donor region the new region is cloned from
    pub like: String,
    /// `settlement:<region>:<name>`
    pub settlement_key: String,
    /// the new settlement's slot keys (`settlement:<region>:<name>:<role>:<n>`), roles `settlement_road`,
    /// `settlement_<k>_slot`, `settlement_fortification`
    pub slot_keys: Vec<String>,
    /// settlement centre in world units
    pub position: [f32; 2],
    /// same order as `slot_keys`
    pub slot_positions: Vec<[f32; 2]>,
    pub neighbours: Vec<String>,
    /// the complete string of the settlement name localisation (`start_pos_settlements_onscreen_name_<settlement
    /// key>-<n>`), which is also the key of its loc row
    pub settlement_loc: String,
    /// REGION child 25 (a plain literal in the vanilla startpos)
    pub rebel_name: String,
    /// shared border length (world units) with each of `neighbours`; missing entries are estimated
    pub border_lengths: Vec<f32>,
    /// new length of the donor's own boundary with the given region; `<= 0` removes that boundary
    pub donor_borders: Vec<(String, f32)>,
    /// faction key (as in the generated startpos' FACTION records) that owns the new region; `None` = the donor's
    /// owner. A different owner must already hold a governorship of the donor's theatre.
    pub owner: Option<String>,
    /// activation of a wilderness region: the generated start position already holds the `CAI_WORLD_REGIONS` item at the
    /// `regions.esf` index (settlement id, slot list, owner and governorship ids empty), usually with its boundaries,
    /// HLCIs and theatre entry. That item is updated in place instead of a new CAI region being appended; missing
    /// boundaries / HLCIs / theatre entries are added. `like` is the playable donor region.
    pub existing_cai: bool,
    /// Index of the region in `regions.esf`. When set, [`add_region`] checks that the CAI region it adds or updates sits
    /// at exactly this index of `CAI_WORLD_REGIONS` (the engine addresses CAI regions by their `regions.esf` index).
    pub map_index: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub region_id: u32,
    pub settlement_id: u32,
    pub recruitment_id: u32,
    /// `(role, old id, new id)` of every cloned slot
    pub slots: Vec<(String, u32, u32)>,
    pub cai_region: u32,
    pub cai_settlement: u32,
    /// first / last CAI id allocated
    pub cai_range: (u32, u32),
    pub settlement_index: u32,
    pub owner: String,
    pub owner_id: u32,
    pub added: Vec<String>,
    pub skipped: Vec<String>,
    pub notes: Vec<String>,
}

// ------------------------------------------------------------------------------------------------ small helpers

fn fixed(v: f32) -> i32 {
    (v as f64 * FIXED).round() as i32
}

fn unfixed(v: i32) -> f32 {
    (v as f64 / FIXED) as f32
}

fn text(n: &Node) -> Option<&str> {
    nd::string_of(n)
}

/// replace a string value keeping its encoding (UTF-16 or latin1)
fn set_text(n: &mut Node, s: &str) -> Result<()> {
    match n {
        Node::Value(Value::String16(x)) | Node::Value(Value::String8(x)) => {
            *x = s.to_string();
            Ok(())
        }
        other => bail!("expected a string, found {other:?}"),
    }
}

/// replace an object id keeping its integer type (i32 / u32)
fn set_id(n: &mut Node, id: u32) -> Result<()> {
    match n {
        Node::Value(Value::U32(x)) => *x = id,
        Node::Value(Value::I32(x)) => *x = id as i32,
        other => bail!("expected an object id, found {other:?}"),
    }
    Ok(())
}

fn set_fixed(n: &mut Node, v: i32) -> Result<()> {
    match n {
        Node::Value(Value::I32(x)) => {
            *x = v;
            Ok(())
        }
        other => bail!("expected a fixed point i32, found {other:?}"),
    }
}

fn id_of(n: &Node) -> Result<u32> {
    match n {
        Node::Value(Value::U32(x)) => Ok(*x),
        Node::Value(Value::I32(x)) => Ok(*x as u32),
        other => bail!("expected an object id, found {other:?}"),
    }
}

fn fixed_of(n: &Node) -> Result<i32> {
    match n {
        Node::Value(Value::I32(x)) => Ok(*x),
        other => bail!("expected a fixed point i32, found {other:?}"),
    }
}

fn arr_push(n: &mut Node, v: u32) -> Result<()> {
    match n {
        Node::Value(Value::Array { code: 0x48, data }) => {
            data.extend_from_slice(&v.to_le_bytes());
            Ok(())
        }
        other => bail!("expected a u32 array, found {other:?}"),
    }
}

fn arr_remove(n: &mut Node, v: u32) -> Result<bool> {
    let vals = nd::arr_vals(n);
    if !nd::is_u32_arr(n) {
        bail!("expected a u32 array, found {n:?}");
    }
    let kept: Vec<u32> = vals.iter().copied().filter(|x| *x != v).collect();
    let changed = kept.len() != vals.len();
    *n = nd::arrn(&kept);
    Ok(changed)
}

/// (world id, key) of a FACTION record: first i32 scalar and first string scalar (same rule as `pipeline::inspect_owners`)
fn faction_id_key(f: &Node) -> Option<(u32, String)> {
    let ch = f.children().ok()?;
    let id = ch.iter().find_map(|n| match n {
        Node::Value(Value::I32(x)) => Some(*x as u32),
        _ => None,
    })?;
    let key = ch.iter().find_map(text)?;
    Some((id, key.to_string()))
}

/// A FACTION of the startpos: world id, key and the government key its buildings carry.
#[derive(Clone, Debug)]
struct OwnerRec {
    id: u32,
    key: String,
    government: String,
}

fn faction_records(world: &Node) -> Result<Vec<&Node>> {
    Ok(world
        .get("FACTION_ARRAY")?
        .items()?
        .iter()
        .filter_map(|it| it.iter().find(|n| nd::is_record_named(n, "FACTION")))
        .collect())
}

fn owner_rec(f: &Node) -> Result<OwnerRec> {
    let (id, key) = faction_id_key(f).ok_or_else(|| anyhow!("FACTION without id / key"))?;
    let government = f
        .get("GOVERNMENT")?
        .children()?
        .get(1)
        .and_then(text)
        .ok_or_else(|| anyhow!("faction {key} has no government key"))?
        .to_string();
    Ok(OwnerRec { id, key, government })
}

/// id of the GOVERNORSHIP of faction `faction_id` that lists regions of `theatre` (REGION child 23)
fn governorship_in_theatre(world: &Node, faction_id: u32, theatre: &str) -> Result<Option<u32>> {
    let mut theatre_of: HashMap<u32, &str> = HashMap::new();
    for it in world.path(&["REGION_MANAGER", "REGIONS_ARRAY"])?.items()? {
        let ch = it[0].children()?;
        if let Some(t) = ch.get(23).and_then(text) {
            theatre_of.insert(id_of(&ch[4])?, t);
        }
    }
    let f = faction_records(world)?
        .into_iter()
        .find(|f| faction_id_key(f).map(|x| x.0) == Some(faction_id))
        .ok_or_else(|| anyhow!("faction {faction_id} not found"))?;
    for post in f.path(&["GOVERNMENT", "POSTS_ARRAY"])?.items()? {
        for rec in post.iter().filter(|n| nd::is_record_named(n, "CHARACTER_POST")) {
            for g in rec.children()?.iter().filter(|n| nd::is_record_named(n, "GOVERNORSHIP")) {
                let gc = g.children()?;
                if nd::arr_vals(&gc[2]).iter().any(|r| theatre_of.get(r) == Some(&theatre)) {
                    return Ok(Some(id_of(&gc[1])?));
                }
            }
        }
    }
    Ok(None)
}

/// hand the cloned region's garrison residences and buildings from `from` to `to`; `count` = (residences, buildings)
fn retarget_owner(n: &mut Node, from: &OwnerRec, to: &OwnerRec, count: &mut (usize, usize)) -> Result<()> {
    match n {
        Node::Record { name, children, .. } => {
            if name == "GARRISON_RESIDENCE" && !children.is_empty() && id_of(&children[0]).ok() == Some(from.id) {
                set_id(&mut children[0], to.id)?;
                count.0 += 1;
            } else if name == "BUILDING" && children.len() > 3 && text(&children[2]) == Some(from.key.as_str()) {
                set_text(&mut children[2], &to.key)?;
                if text(&children[3]) == Some(from.government.as_str()) {
                    set_text(&mut children[3], &to.government)?;
                }
                count.1 += 1;
            }
            for c in children {
                retarget_owner(c, from, to, count)?;
            }
        }
        Node::Array { items, .. } => {
            for it in items {
                for c in it {
                    retarget_owner(c, from, to, count)?;
                }
            }
        }
        Node::Value(_) => {}
    }
    Ok(())
}

fn slot_role(key: &str) -> Result<&str> {
    key.split(':')
        .nth(3)
        .ok_or_else(|| anyhow!("slot key {key} has no role segment"))
}

fn region_key_of(item: &[Node]) -> Option<&str> {
    item.first()?.children().ok()?.first().and_then(text)
}

struct CaiIds {
    next: u32,
    first: u32,
}

impl CaiIds {
    fn take(&mut self) -> u32 {
        let v = self.next;
        self.next += 1;
        v
    }
}

#[derive(Clone, Debug)]
struct Slot {
    role: String,
    key: String,
    pos: [f32; 2],
}

struct WorldOut {
    region_id: u32,
    settlement_id: u32,
    owner_key: String,
    owner_id: u32,
    gov_id: u32,
    /// (role, old world slot id, new world slot id)
    slots: Vec<(String, u32, u32)>,
    old_settlement_id: u32,
    settlement_fixed: (i32, i32),
    /// the donor settlement's fixed point position (the donor's settlement HLCI carries it)
    old_settlement_fixed: (i32, i32),
    /// the owner differs from the donor's: CAI faction / governorship references follow `owner_id` / `gov_id`
    owner_changed: bool,
}

// ------------------------------------------------------------------------------------------------ entry point

pub fn add_region(root: &mut Node, spec: &NewRegion) -> Result<Report> {
    ensure!(
        !spec.slot_keys.is_empty() && spec.slot_keys.len() == spec.slot_positions.len(),
        "slot_keys / slot_positions mismatch"
    );
    ensure!(
        spec.existing_cai || !spec.neighbours.is_empty(),
        "a region needs neighbours"
    );
    ensure!(
        spec.border_lengths.is_empty() || spec.border_lengths.len() == spec.neighbours.len(),
        "border_lengths must be empty or parallel to neighbours"
    );
    ensure!(
        !spec.settlement_loc.is_empty() && !spec.rebel_name.is_empty(),
        "settlement_loc and rebel_name are required"
    );
    if let Some(index) = spec.map_index {
        let at = cai_index_for(root, spec)?;
        ensure!(
            at == index,
            "region {}: regions.esf index {index} but its CAI region would be at index {at}; the CAI region list and regions.esf must stay in the same order",
            spec.key
        );
    }
    let slots: Vec<Slot> = spec
        .slot_keys
        .iter()
        .zip(&spec.slot_positions)
        .map(|(k, p)| {
            Ok(Slot {
                role: slot_role(k)?.to_string(),
                key: k.clone(),
                pos: *p,
            })
        })
        .collect::<Result<_>>()?;
    let keys: BTreeSet<&str> = slots.iter().map(|s| s.key.as_str()).collect();
    ensure!(keys.len() == slots.len(), "two requested slots share a key");

    let used = collect_used(&[&*root]);
    let mut ids = IdAlloc::new(used, WORLD_ID_START, 0);
    let mut rep = Report::default();

    let w = add_to_world(root, spec, &slots, &mut ids, &mut rep).context("WORLD")?;
    rep.region_id = w.region_id;
    rep.settlement_id = w.settlement_id;
    rep.owner = w.owner_key.clone();
    rep.owner_id = w.owner_id;
    rep.slots = w.slots.clone();
    add_to_cai(root, spec, &w, &mut rep).context("CAI_INTERFACE")?;
    add_to_trade(root, spec, &w, &mut rep).context("trade manager")?;
    Ok(rep)
}

// ------------------------------------------------------------------------------------------------ WORLD

/// clear the garrison pointers of every SIEGEABLE_GARRISON_RESIDENCE below `n` (as `startpos::core` does)
fn clear_garrisons(n: &mut Node) -> Result<()> {
    match n {
        Node::Record { name, children, .. } => {
            if name == "SIEGEABLE_GARRISON_RESIDENCE" {
                ensure!(
                    children.len() > 14
                        && matches!(children[12], Node::Value(Value::U32(_)))
                        && nd::is_u32_arr(&children[13])
                        && nd::is_u32_arr(&children[14]),
                    "unexpected SIEGEABLE_GARRISON_RESIDENCE layout"
                );
                children[12] = esf::u32v(0);
                children[13] = esf::arr_u32(&[]);
                children[14] = esf::arr_u32(&[]);
            }
            for c in children {
                clear_garrisons(c)?;
            }
        }
        Node::Array { items, .. } => {
            for it in items {
                for c in it {
                    clear_garrisons(c)?;
                }
            }
        }
        Node::Value(_) => {}
    }
    Ok(())
}

/// every integer >= 1_000_000 below `n` (world ids and similar)
fn big_ints(n: &Node, out: &mut BTreeSet<u32>) {
    match n {
        Node::Record { name, .. } if name == "LINE_OF_SIGHT" => {}
        Node::Record { children, .. } => children.iter().for_each(|c| big_ints(c, out)),
        Node::Array { items, .. } => items.iter().flatten().for_each(|c| big_ints(c, out)),
        Node::Value(Value::U32(v)) if *v >= 1_000_000 && *v != u32::MAX => {
            out.insert(*v);
        }
        Node::Value(Value::I32(v)) if (*v as u32) >= 1_000_000 && *v != -1 => {
            out.insert(*v as u32);
        }
        _ => {}
    }
}

/// retarget one REGION_SLOT record
fn retarget_slot(slot: &mut Node, new: &Slot, new_id: u32) -> Result<u32> {
    let ch = slot.children_mut()?;
    ensure!(ch.len() > 7, "REGION_SLOT too short");
    let old_id = id_of(&ch[2])?;
    {
        let sgr = ch[0].children_mut()?;
        ensure!(id_of(&sgr[1])? == old_id, "slot residence id differs from the slot id");
        set_id(&mut sgr[1], new_id)?;
    }
    set_id(&mut ch[2], new_id)?;
    set_text(&mut ch[3], &new.key)?;
    set_fixed(&mut ch[6], fixed(new.pos[0]))?;
    set_fixed(&mut ch[7], fixed(new.pos[1]))?;
    Ok(old_id)
}

fn add_to_world(
    root: &mut Node,
    spec: &NewRegion,
    slots: &[Slot],
    ids: &mut IdAlloc,
    rep: &mut Report,
) -> Result<WorldOut> {
    let model = root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])?;
    let world = model.get_mut("WORLD")?;

    // ---- donor
    let like = {
        let items = world.path(&["REGION_MANAGER", "REGIONS_ARRAY"])?.items()?;
        ensure!(
            !items.iter().any(|it| region_key_of(it) == Some(&spec.key)),
            "region {} already exists",
            spec.key
        );
        items
            .iter()
            .find(|it| region_key_of(it) == Some(&spec.like))
            .ok_or_else(|| anyhow!("donor region {} not found", spec.like))?[0]
            .clone()
    };
    let mut region = like.clone();
    let (donor_owner_id, donor_gov_id, theatre_name) = {
        let ch = region.children()?;
        ensure!(ch.len() >= 42, "REGION has {} children, expected 42+", ch.len());
        let theatre = text(&ch[23]).ok_or_else(|| anyhow!("REGION child 23 is not the theatre name"))?;
        (id_of(&ch[20])?, id_of(&ch[22])?, theatre.to_string())
    };
    let donor_owner = {
        let f = faction_records(world)?
            .into_iter()
            .find(|f| faction_id_key(f).map(|x| x.0) == Some(donor_owner_id))
            .ok_or_else(|| anyhow!("owner faction {donor_owner_id} of {} not found", spec.like))?;
        owner_rec(f)?
    };
    // a requested owner other than the donor's takes over residences and buildings and brings its own governorship
    let new_owner: Option<(OwnerRec, u32)> = match spec.owner.as_deref() {
        Some(k) if k != donor_owner.key => {
            let f = faction_records(world)?
                .into_iter()
                .find(|f| faction_id_key(f).map(|x| x.1.as_str() == k).unwrap_or(false))
                .ok_or_else(|| anyhow!("owner faction {k} is not in the startpos"))?;
            let o = owner_rec(f)?;
            let g = governorship_in_theatre(world, o.id, &theatre_name)?.ok_or_else(|| {
                anyhow!("owner {k} holds no governorship of the {theatre_name} theatre (it needs a region there)")
            })?;
            Some((o, g))
        }
        _ => None,
    };
    let (owner_id, gov_id) = match &new_owner {
        Some((o, g)) => (o.id, *g),
        None => (donor_owner_id, donor_gov_id),
    };
    let old_settlement_key = {
        let st = region.get("SETTLEMENT")?.children()?;
        text(&st[3]).ok_or_else(|| anyhow!("SETTLEMENT child 3 is not a string"))?.to_string()
    };

    // ---- fresh ids
    let region_id = ids.world();
    let settlement_id = ids.world();
    let recruitment_id = ids.world();
    rep.recruitment_id = recruitment_id;
    let mut slot_out: Vec<(String, u32, u32)> = Vec::new();

    {
        let ch = region.children_mut()?;
        ensure!(text(&ch[0]) == Some(spec.like.as_str()), "REGION child 0 is not the key");
        set_text(&mut ch[0], &spec.key)?;
        ensure!(matches!(ch[4], Node::Value(Value::I32(_))), "REGION child 4 is not the region id");
        set_id(&mut ch[4], region_id)?;
        ensure!(text(&ch[25]).is_some(), "REGION child 25 is not the rebel name");
        set_text(&mut ch[25], &spec.rebel_name)?;
    }
    if let Some((o, g)) = &new_owner {
        let ch = region.children_mut()?;
        set_id(&mut ch[20], o.id)?;
        set_id(&mut ch[22], *g)?;
        let mut count = (0, 0);
        retarget_owner(&mut region, &donor_owner, o, &mut count)?;
        ensure!(count.0 > 0, "cloned region holds no garrison residence of {}", donor_owner.key);
        rep.added.push(format!(
            "owner {} -> {}: {} garrison residences and {} buildings handed over, governorship {g}",
            donor_owner.key, o.key, count.0, count.1
        ));
    }
    {
        let ch = region.children_mut()?;
        ensure!(
            nd::is_record_named(&ch[27], "REGION_RECRUITMENT_MANAGER"),
            "REGION child 27 is {:?}, not REGION_RECRUITMENT_MANAGER",
            nd::name_of(&ch[27])
        );
        let rm = ch[27].children_mut()?;
        set_id(&mut rm[2], recruitment_id)?;
        ensure!(
            nd::is_record_named(&ch[41], "CAMPAIGN_LOCALISATION"),
            "REGION child 41 is not the region name localisation"
        );
        let loc = ch[41].children_mut()?;
        set_text(&mut loc[0], &format!("regions_onscreen_{}", spec.key))?;
    }

    // ---- slots
    // Slots of one role (a capital settlement has several `settlement_<k>_slot`) pair up by their trailing number:
    // the i-th donor slot of a role becomes the i-th requested slot of that role.
    let ordinal = |key: &str| key.rsplit(':').next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(u32::MAX);
    let mut by_role: HashMap<&str, Vec<&Slot>> = HashMap::new();
    for s in slots {
        by_role.entry(s.role.as_str()).or_default().push(s);
    }
    for v in by_role.values_mut() {
        v.sort_by_key(|s| ordinal(&s.key));
    }
    {
        let sm = region.get_mut("REGION_SLOT_MANAGER")?;
        // array slots: keep only the requested settlement slots
        {
            let arr = sm.get_mut("REGION_SLOT_ARRAY")?.items_mut()?;
            let mut donors: HashMap<String, Vec<(u32, usize)>> = HashMap::new();
            for (i, it) in arr.iter().enumerate() {
                let key = text(&it[0].children()?[3]).ok_or_else(|| anyhow!("REGION_SLOT without a key"))?;
                if key.starts_with("settlement:") {
                    donors.entry(slot_role(key)?.to_string()).or_default().push((ordinal(key), i));
                }
            }
            for v in donors.values_mut() {
                v.sort_unstable();
            }
            let mut kept = Vec::new();
            for (i, mut it) in std::mem::take(arr).into_iter().enumerate() {
                let key = text(&it[0].children()?[3])
                    .ok_or_else(|| anyhow!("REGION_SLOT without a key"))?
                    .to_string();
                if !key.starts_with("settlement:") {
                    continue; // town / pasture / mine ... slots are dropped
                }
                let role = slot_role(&key)?;
                let rank = donors[role].iter().position(|d| d.1 == i).expect("ranked donor slot");
                let Some(new) = by_role.get(role).and_then(|v| v.get(rank)).copied() else {
                    continue;
                };
                let new_id = ids.world();
                let old = retarget_slot(&mut it[0], new, new_id)?;
                slot_out.push((new.role.clone(), old, new_id));
                kept.push(it);
            }
            *arr = kept;
        }
        for (container, role) in [("ROAD_SLOT", ROLE_ROAD), ("FORTIFICATION_SLOT", ROLE_FORT)] {
            let c = sm.get_mut(container)?.children_mut()?;
            let slot = c
                .iter_mut()
                .find(|n| nd::is_record_named(n, "REGION_SLOT"))
                .ok_or_else(|| anyhow!("{container} has no REGION_SLOT"))?;
            match by_role.get(role).and_then(|v| v.first()).copied() {
                Some(new) => {
                    let new_id = ids.world();
                    let old = retarget_slot(slot, new, new_id)?;
                    slot_out.push((role.to_string(), old, new_id));
                }
                None => bail!("requested slots lack the {role} slot the settlement needs"),
            }
        }
    }
    ensure!(
        slot_out.len() == slots.len(),
        "the donor lacks a slot for one of the requested roles ({} of {} cloned)",
        slot_out.len(),
        slots.len()
    );

    // ---- settlement
    let old_settlement_id;
    let old_settlement_fixed;
    let settlement_fixed = (fixed(spec.position[0]), fixed(spec.position[1]));
    {
        let st = region.get_mut("SETTLEMENT")?.children_mut()?;
        ensure!(st.len() > 5, "SETTLEMENT too short");
        {
            let sgr = st[0].children_mut()?;
            ensure!(sgr.len() > 14, "settlement residence too short");
            old_settlement_id = id_of(&sgr[1])?;
            old_settlement_fixed = (fixed_of(&sgr[10])?, fixed_of(&sgr[11])?);
            set_id(&mut sgr[1], settlement_id)?;
            set_fixed(&mut sgr[10], settlement_fixed.0)?;
            set_fixed(&mut sgr[11], settlement_fixed.1)?;
        }
        ensure!(
            nd::is_record_named(&st[1], "CAMPAIGN_LOCALISATION"),
            "SETTLEMENT child 1 is not the name localisation"
        );
        set_text(&mut st[1].children_mut()?[0], &spec.settlement_loc)?;
        ensure!(id_of(&st[4])? == old_settlement_id, "SETTLEMENT child 4 is not the residence id");
        set_id(&mut st[4], settlement_id)?;
        ensure!(
            text(&st[3]) == Some(old_settlement_key.as_str()) && text(&st[5]) == Some(old_settlement_key.as_str()),
            "SETTLEMENT key children differ"
        );
        set_text(&mut st[3], &spec.settlement_key)?;
        set_text(&mut st[5], &spec.settlement_key)?;
    }
    clear_garrisons(&mut region)?;

    // ---- every world id the clone still shares with the donor must be a shared one (owner, governorship, constant)
    {
        let (mut new_ids, mut donor_ids) = (BTreeSet::new(), BTreeSet::new());
        big_ints(&region, &mut new_ids);
        big_ints(&like, &mut donor_ids);
        let left: Vec<u32> = new_ids
            .intersection(&donor_ids)
            .copied()
            .filter(|v| (100_000_000..4_000_000_000).contains(v))
            .filter(|v| *v != owner_id && *v != gov_id && *v != 2_036_301_432)
            .collect();
        if !left.is_empty() {
            rep.notes.push(format!(
                "REGION clone still holds integers of the donor that are neither owner, governorship nor the shared constant: {left:?}"
            ));
        }
    }

    // ---- owner faction and its governorship
    let owner_key = {
        let fa = world.get("FACTION_ARRAY")?.items()?;
        let f = fa
            .iter()
            .filter_map(|it| it.iter().find(|n| nd::is_record_named(n, "FACTION")))
            .find(|f| faction_id_key(f).map(|x| x.0) == Some(owner_id))
            .ok_or_else(|| anyhow!("owner faction {owner_id} not found"))?;
        faction_id_key(f).ok_or_else(|| anyhow!("FACTION without id / key"))?.1
    };
    {
        let fa = world.get_mut("FACTION_ARRAY")?.items_mut()?;
        let faction = fa
            .iter_mut()
            .filter_map(|it| it.iter_mut().find(|n| nd::is_record_named(n, "FACTION")))
            .find(|f| faction_id_key(f).map(|x| x.0) == Some(owner_id))
            .ok_or_else(|| anyhow!("owner faction {owner_id} not found"))?;
        let posts = faction.path_mut(&["GOVERNMENT", "POSTS_ARRAY"])?.items_mut()?;
        let mut done = 0;
        for post in posts.iter_mut() {
            for rec in post.iter_mut() {
                if !nd::is_record_named(rec, "CHARACTER_POST") {
                    continue;
                }
                for g in rec.children_mut()?.iter_mut() {
                    if nd::is_record_named(g, "GOVERNORSHIP") {
                        let gc = g.children_mut()?;
                        if id_of(&gc[1])? == gov_id {
                            arr_push(&mut gc[2], region_id)?;
                            done += 1;
                        }
                    }
                }
            }
        }
        ensure!(done == 1, "governorship {gov_id} found {done} times in {owner_key}");
    }
    rep.added.push(format!(
        "owner {owner_key}: GOVERNORSHIP {gov_id} region list += region id {region_id}"
    ));

    world
        .path_mut(&["REGION_MANAGER", "REGIONS_ARRAY"])?
        .items_mut()?
        .push(vec![region]);
    rep.added.push(format!(
        "WORLD/REGION_MANAGER/REGIONS_ARRAY: REGION {} (region id {region_id}, settlement {settlement_id}, recruitment manager {recruitment_id}, slots {:?})",
        spec.key,
        slot_out.iter().map(|s| (s.0.as_str(), s.2)).collect::<Vec<_>>()
    ));

    // ---- REGION_OWNERSHIPS_BY_THEATRE
    {
        let by_theatre = root
            .path_mut(&["CAMPAIGN_PREOPEN_MAP_INFO", "REGION_OWNERSHIPS_BY_THEATRE"])?
            .items_mut()?;
        let mut done = 0;
        for th in by_theatre.iter_mut() {
            let own = th
                .iter_mut()
                .find(|n| nd::is_array_named(n, "REGION_OWNERSHIPS"))
                .ok_or_else(|| anyhow!("theatre without REGION_OWNERSHIPS"))?
                .items_mut()?;
            if let Some(d) = own.iter().find(|o| text(&o[0]) == Some(spec.like.as_str())).cloned() {
                ensure!(
                    text(&d[1]) == Some(donor_owner.key.as_str()),
                    "REGION_OWNERSHIPS says {:?} owns {}, the REGION says {}",
                    d.get(1).and_then(text),
                    spec.like,
                    donor_owner.key
                );
                let mut n = d;
                set_text(&mut n[0], &spec.key)?;
                set_text(&mut n[1], &owner_key)?;
                own.push(n);
                done += 1;
            }
        }
        ensure!(done == 1, "{} is listed in {done} theatres of REGION_OWNERSHIPS_BY_THEATRE", spec.like);
    }
    rep.added.push(format!(
        "CAMPAIGN_PREOPEN_MAP_INFO/REGION_OWNERSHIPS_BY_THEATRE: ({}, {owner_key})",
        spec.key
    ));

    Ok(WorldOut {
        region_id,
        settlement_id,
        owner_key,
        owner_id,
        gov_id,
        slots: slot_out,
        old_settlement_id,
        settlement_fixed,
        old_settlement_fixed,
        owner_changed: new_owner.is_some(),
    })
}

// ------------------------------------------------------------------------------------------------ trade manager

/// (neighbour settlement position as f32 world units) from the neighbour's REGION record
fn settlement_pos(region: &Node) -> Result<[f32; 2]> {
    let sgr = region.path(&["SETTLEMENT", "SIEGEABLE_GARRISON_RESIDENCE"])?.children()?;
    Ok([unfixed(fixed_of(&sgr[10])?), unfixed(fixed_of(&sgr[11])?)])
}

fn add_to_trade(root: &mut Node, spec: &NewRegion, w: &WorldOut, rep: &mut Report) -> Result<()> {
    let model = root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])?;
    // trade neighbours: the neighbours that are regions of the startpos (sea regions have no settlement node)
    let mut npos: Vec<[f32; 2]> = Vec::new();
    let mut trade_nb: Vec<&String> = Vec::new();
    {
        let items = model.path(&["WORLD", "REGION_MANAGER", "REGIONS_ARRAY"])?.items()?;
        for nb in &spec.neighbours {
            if let Some(r) = items.iter().find(|it| region_key_of(it) == Some(nb.as_str())) {
                npos.push(settlement_pos(&r[0])?);
                trade_nb.push(nb);
            } else {
                rep.notes.push(format!("{nb} is not a region with a settlement: no trade route to it"));
            }
        }
    }
    let tm = model.get_mut("CAMPAIGN_TRADE_MANAGER")?;

    // ---- SETTLEMENT_INDICES: node indices are ports, then settlements, then (after a possible gap) the trade nodes
    let (index, nb_index, npos) = {
        let n_ports = tm.get("PORT_INDICES")?.items()?.len() as u32;
        let si = tm.get_mut("SETTLEMENT_INDICES")?.items_mut()?;
        let mut map: HashMap<String, u32> = HashMap::new();
        for it in si.iter() {
            map.insert(
                text(&it[0]).ok_or_else(|| anyhow!("SETTLEMENT_INDICES key"))?.to_string(),
                id_of(&it[1])?,
            );
        }
        ensure!(!map.contains_key(&spec.key), "{} already has a settlement index", spec.key);
        let index = n_ports + si.len() as u32;
        ensure!(
            map.values().all(|v| (n_ports..index).contains(v)) && map.len() == si.len(),
            "SETTLEMENT_INDICES are not the contiguous range {n_ports}..{index}"
        );
        // a land region without a settlement (no entry in SETTLEMENT_INDICES) has no trade node to route to
        let mut nb_index: Vec<u32> = Vec::new();
        let mut kept_pos: Vec<[f32; 2]> = Vec::new();
        for (n, p) in trade_nb.iter().zip(&npos) {
            match map.get(*n) {
                Some(i) => {
                    nb_index.push(*i);
                    kept_pos.push(*p);
                }
                None => rep.notes.push(format!("{n} has no settlement index: no trade route to it")),
            }
        }
        si.push(vec![esf::s16(&spec.key), esf::u32v(index)]);
        (index, nb_index, kept_pos)
    };
    rep.settlement_index = index;
    rep.added.push(format!(
        "CAMPAIGN_TRADE_MANAGER/SETTLEMENT_INDICES: ({}, {index})",
        spec.key
    ));

    // ---- trade node renumbering: node indices are ports, settlements, then the trade nodes (which need not follow the
    // settlements directly: the generated world startpos leaves a gap). Every reference in `index..=last node` moves up one.
    let mut node_idx: Vec<u32> = tm
        .get("TRADE_NODES")?
        .items()?
        .iter()
        .map(|it| id_of(it.get(1).ok_or_else(|| anyhow!("TRADE_NODES item without its node index"))?))
        .collect::<Result<_>>()?;
    node_idx.sort_unstable();
    ensure!(!node_idx.is_empty(), "no trade nodes");
    ensure!(
        node_idx[0] >= index && node_idx.windows(2).all(|p| p[1] == p[0] + 1),
        "trade node indices {node_idx:?} are not one run at or above the new settlement index {index}"
    );
    let (lo, hi) = (index, node_idx[node_idx.len() - 1] + 1);
    let bump = |n: &mut Node| -> Result<bool> {
        let v = id_of(n)?;
        if v >= lo && v < hi {
            set_id(n, v + 1)?;
            Ok(true)
        } else {
            Ok(false)
        }
    };
    let mut shifted = (0usize, 0usize, 0usize);
    for it in tm.get_mut("TRADE_ROUTES")?.items_mut()? {
        ensure!(it.len() == 4, "TRADE_ROUTES item with {} children", it.len());
        for i in 0..2 {
            if bump(&mut it[i])? {
                shifted.0 += 1;
            }
        }
    }
    for it in tm.get_mut("TRADE_NODES")?.items_mut()? {
        ensure!(it.len() == 2, "TRADE_NODES item with {} children", it.len());
        ensure!(bump(&mut it[1])?, "trade node index outside {lo}..{hi}");
        shifted.1 += 1;
    }
    let mut foreign = 0usize;
    for faction in tm.get_mut("DOMESTIC_TRADE_ROUTES")?.items_mut()? {
        let routes = faction
            .iter_mut()
            .find(|n| nd::is_array_named(n, "FACTION_DOMESTIC_TRADE_ROUTES_ARRAY"))
            .ok_or_else(|| anyhow!("DOMESTIC_TRADE_ROUTES item without its route array"))?
            .items_mut()?;
        for r in routes {
            let rec = r
                .iter_mut()
                .find(|n| nd::is_record_named(n, "DOMESTIC_TRADE_ROUTE"))
                .ok_or_else(|| anyhow!("DOMESTIC_TRADE_ROUTE missing"))?
                .children_mut()?;
            for (i, c) in rec.iter_mut().enumerate() {
                if let Node::Value(Value::U32(v)) = c {
                    if *v >= lo && *v < hi {
                        if i == 6 {
                            *v += 1;
                            shifted.2 += 1;
                        } else {
                            foreign += 1;
                        }
                    }
                }
            }
        }
    }
    if foreign > 0 {
        rep.notes.push(format!(
            "{foreign} DOMESTIC_TRADE_ROUTE integers other than child 6 fall into the trade node range {lo}..{hi} and were left alone"
        ));
    }
    {
        // international routes: node-index children are 3, 4, 8, 9, 13, 14 in vanilla and none reaches the trade nodes
        let mut hits = 0usize;
        for faction in tm.get("INTERNATIONAL_TRADE_ROUTES")?.items()? {
            for arr in faction.iter().filter(|n| nd::is_array_named(n, "FACTION_INTERNATIONAL_TRADE_ROUTES_ARRAY")) {
                for r in arr.items()? {
                    for rec in r.iter().filter(|n| nd::is_record_named(n, "INTERNATIONAL_TRADE_ROUTE")) {
                        for i in [3usize, 4, 8, 9, 13, 14] {
                            if let Some(Node::Value(Value::U32(v))) = rec.children()?.get(i) {
                                if *v >= lo && *v < hi {
                                    hits += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        if hits > 0 {
            rep.notes.push(format!(
                "{hits} INTERNATIONAL_TRADE_ROUTE node children fall into the trade node range {lo}..{hi} and were left alone"
            ));
        }
    }
    rep.added.push(format!(
        "trade node indices {lo}..{} -> {}..{}: {} TRADE_ROUTES endpoints, {} TRADE_NODES, {} DOMESTIC_TRADE_ROUTE nodes",
        hi - 1,
        lo + 1,
        hi,
        shifted.0,
        shifted.1,
        shifted.2
    ));

    // ---- direct routes neighbour -> new settlement (appended: no existing route index changes)
    let me = [unfixed(w.settlement_fixed.0), unfixed(w.settlement_fixed.1)];
    let routes = tm.get_mut("TRADE_ROUTES")?.items_mut()?;
    let mut order: Vec<usize> = (0..nb_index.len()).collect();
    order.sort_by_key(|&i| nb_index[i]);
    for i in order {
        let length = (((me[0] - npos[i][0]) as f64).powi(2) + ((me[1] - npos[i][1]) as f64).powi(2)).sqrt() as f32;
        ensure!(nb_index[i] < index, "route endpoints must satisfy from < to");
        routes.push(vec![
            esf::u32v(nb_index[i]),
            esf::u32v(index),
            esf::arr_u32(&[]),
            esf::f32v(length),
        ]);
    }
    rep.added.push(format!(
        "CAMPAIGN_TRADE_MANAGER/TRADE_ROUTES: {} direct routes ending at node {index} (appended)",
        nb_index.len()
    ));
    Ok(())
}

// ------------------------------------------------------------------------------------------------ CAI

fn spec_str(item: &[Node], i: usize) -> Option<&str> {
    nd::spec_kids(item).ok()?.get(i).and_then(text)
}

fn spec_id(item: &[Node], i: usize) -> Result<u32> {
    nd::sc_at(nd::spec_kids(item)?, i)
}

fn world_items<'a>(world: &'a Node, name: &str) -> Result<&'a Vec<Vec<Node>>> {
    nd::get_items(world, &[name])
}

fn world_items_mut<'a>(world: &'a mut Node, name: &str) -> Result<&'a mut Vec<Vec<Node>>> {
    nd::arr_items_mut(nd::get_mut(world, &[name])?)
}

fn add_to_cai(root: &mut Node, spec: &NewRegion, w: &WorldOut, rep: &mut Report) -> Result<()> {
    let ci = root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"])?;
    let next = nd::sc_at(nd::kids(nd::get(ci, &["CAI_CENTRAL_BDI_POOL"])?)?, 0)?;
    let mut cid = CaiIds { next, first: next };

    // ---- read phase -----------------------------------------------------------------------------------------------
    let world = nd::get(ci, &["CAI_WORLD"])?;
    let regions = world_items(world, "CAI_WORLD_REGIONS")?;
    let region_cai: HashMap<String, u32> = regions
        .iter()
        .map(|it| Ok((spec_str(it, 10).ok_or_else(|| anyhow!("CAI_REGION without key"))?.to_string(), nd::item_id(it)?)))
        .collect::<Result<_>>()?;
    // activation: the CAI region of the key is already there and is updated in place
    let existing: Option<(usize, Vec<Node>)> = if spec.existing_cai {
        let (index, item) = regions
            .iter()
            .enumerate()
            .find(|(_, it)| spec_str(it, 10) == Some(spec.key.as_str()))
            .ok_or_else(|| anyhow!("CAI has no region {} to activate", spec.key))?;
        let s = nd::spec_kids(item)?;
        ensure!(
            nd::sc_at(s, 2)? == 0 && nd::arr_at(s, 3)?.is_empty() && nd::sc_at(s, 11)? == 0 && nd::sc_at(s, 12)? == 0,
            "CAI region {} is not an empty wilderness entry (settlement {}, {} slots, world id {}, governorship {})",
            spec.key,
            nd::sc_at(s, 2)?,
            nd::arr_at(s, 3)?.len(),
            nd::sc_at(s, 11)?,
            nd::sc_at(s, 12)?
        );
        Some((index, item.clone()))
    } else {
        ensure!(!region_cai.contains_key(&spec.key), "CAI already has region {}", spec.key);
        None
    };
    let new_index = existing.as_ref().map_or(regions.len(), |e| e.0);
    let like_item = regions
        .iter()
        .find(|it| spec_str(it, 10) == Some(spec.like.as_str()))
        .ok_or_else(|| anyhow!("CAI has no region {}", spec.like))?
        .clone();
    let like_cai = nd::item_id(&like_item)?;
    let like_spec = nd::spec_kids(&like_item)?.clone();
    let like_settle_cai = nd::sc_at(&like_spec, 2)?;
    let donor_gov_cai = nd::sc_at(&like_spec, 12)?;
    ensure!(
        nd::sc_at(&like_spec, 11)? != 0,
        "donor CAI region without a world region id"
    );
    let _theatre_list = nd::arr_at(&like_spec, 0)?;
    let like_hlcis = nd::arr_at(&like_spec, 1)?;
    let nb_cai: Vec<u32> = spec
        .neighbours
        .iter()
        .map(|n| region_cai.get(n).copied().ok_or_else(|| anyhow!("CAI has no region {n}")))
        .collect::<Result<_>>()?;
    let like_settle = world_items(world, "CAI_WORLD_SETTLEMENTS")?
        .iter()
        .find(|it| nd::item_id(it).ok() == Some(like_settle_cai))
        .ok_or_else(|| anyhow!("donor CAI settlement {like_settle_cai} missing"))?
        .clone();
    let donor_owner_cai = nd::sc_at(nd::kids(nd::rec_by_name(&like_settle, "OWNED_DIRECT")?)?, 0)?;
    // a different owner brings its own CAI faction and the CAI governorship of its world governorship
    let (owner_cai, gov_cai) = if w.owner_changed {
        let faction = world_items(world, "CAI_WORLD_FACTIONS")?
            .iter()
            .find(|it| nd::spec_kids(it).ok().and_then(|s| nd::sc_at(s, 6).ok()) == Some(w.owner_id))
            .ok_or_else(|| anyhow!("no CAI faction for world faction {}", w.owner_id))?;
        let governorship = world_items(world, "CAI_WORLD_GOVERNORSHIPS")?
            .iter()
            .find(|it| nd::spec_kids(it).ok().and_then(|s| nd::sc_at(s, 0).ok()) == Some(w.gov_id))
            .ok_or_else(|| anyhow!("no CAI governorship for world governorship {}", w.gov_id))?;
        (nd::item_id(faction)?, nd::item_id(governorship)?)
    } else {
        (donor_owner_cai, donor_gov_cai)
    };
    // building slots of the donor settlement keyed by their world slot id
    let like_slot_ids: Vec<u32> = w.slots.iter().map(|s| s.1).collect();
    let mut slot_templates: Vec<(u32, Vec<Node>)> = Vec::new();
    for it in world_items(world, "CAI_WORLD_BUILDING_SLOTS")? {
        if spec_id(it, 2)? == like_settle_cai {
            slot_templates.push((spec_id(it, 0)?, it.clone()));
        }
    }
    {
        let have: BTreeSet<u32> = slot_templates.iter().map(|s| s.0).collect();
        for id in &like_slot_ids {
            ensure!(have.contains(id), "no CAI_BUILDING_SLOT for the donor slot {id}");
        }
    }
    // The donor's settlement HLCI: the one at the donor settlement's position, else one of the vanilla settlement
    // kind (4), else the donor's only HLCI. The generated world startpos keeps the ETW HLCIs, whose kind numbers are
    // not 4, so the clone keeps the template's kind.
    let hlci_tmpl = {
        let cands: Vec<&Vec<Node>> = world_items(world, "CAI_WORLD_REGION_HLCIS")?
            .iter()
            .filter(|it| {
                let s = nd::spec_kids(it).ok();
                s.and_then(|s| nd::sc_at(s, 0).ok()) == Some(like_cai)
                    && like_hlcis.contains(&nd::item_id(it).unwrap_or(0))
            })
            .collect();
        let field = |it: &Vec<Node>, i: usize| nd::spec_kids(it).ok().and_then(|s| nd::sc_at(s, i).ok());
        let at_settlement = |it: &&Vec<Node>| {
            field(it, 3) == Some(w.old_settlement_fixed.0 as u32) && field(it, 4) == Some(w.old_settlement_fixed.1 as u32)
        };
        cands
            .iter()
            .find(|it| at_settlement(it))
            .or_else(|| cands.iter().find(|it| field(it, 1) == Some(KIND_SETTLEMENT)))
            .or_else(|| if cands.len() == 1 { cands.first() } else { None })
            .ok_or_else(|| {
                anyhow!(
                    "donor region {} has no identifiable settlement HLCI (CAI region {like_cai} lists {like_hlcis:?}; candidates (id, kind): {:?})",
                    spec.like,
                    cands.iter().map(|it| (nd::item_id(it).unwrap_or(0), field(it, 1))).collect::<Vec<_>>()
                )
            })?
            .to_vec()
    };
    let bnd_tmpl = world_items(world, "CAI_WORLD_REGION_BOUNDARIES")?
        .first()
        .ok_or_else(|| anyhow!("no boundary to clone"))?
        .clone();
    let existing_boundaries: Vec<(u32, u32, u32, f32)> = world_items(world, "CAI_WORLD_REGION_BOUNDARIES")?
        .iter()
        .map(|it| {
            let s = nd::spec_kids(it)?;
            let len = match s.get(2) {
                Some(Node::Value(Value::F32(l))) => *l,
                other => bail!("boundary length is {other:?}"),
            };
            Ok((nd::item_id(it)?, nd::sc_at(s, 0)?, nd::sc_at(s, 1)?, len))
        })
        .collect::<Result<_>>()?;

    // ---- allocate ---------------------------------------------------------------------------------------------------
    let region_id = match &existing {
        Some((_, item)) => nd::item_id(item)?,
        None => cid.take(),
    };
    // an existing settlement HLCI of the wilderness region (same kind as the donor's) moves to the new settlement
    let existing_hlci: Option<u32> = match &existing {
        Some((_, item)) => {
            let listed = nd::arr_at(nd::spec_kids(item)?, 1)?;
            let kind = nd::sc_at(nd::spec_kids(&hlci_tmpl)?, 1)?;
            let mut found = None;
            for it in world_items(world, "CAI_WORLD_REGION_HLCIS")? {
                let s = nd::spec_kids(it)?;
                if listed.contains(&nd::item_id(it)?) && nd::sc_at(s, 0)? == region_id && nd::sc_at(s, 1)? == kind {
                    ensure!(found.is_none(), "CAI region {} has two HLCIs of kind {kind}", spec.key);
                    found = Some(nd::item_id(it)?);
                }
            }
            found
        }
        None => None,
    };
    let hlci_id = match existing_hlci {
        Some(id) => id,
        None => cid.take(),
    };
    let slot_cai: Vec<(u32, u32)> = like_slot_ids.iter().map(|old| (*old, cid.take())).collect();
    let settle_id = cid.take();
    rep.cai_region = region_id;
    rep.cai_settlement = settle_id;

    // ---- boundaries: lengths ------------------------------------------------------------------------------------------
    let donor_len = |other: u32| -> Option<f32> {
        existing_boundaries
            .iter()
            .find(|b| (b.1 == like_cai && b.2 == other) || (b.1 == other && b.2 == like_cai))
            .map(|b| b.3)
    };
    // a boundary the activated region already has with a neighbour keeps its ETW length
    let boundary_between = |a: u32, b: u32| -> Option<u32> {
        existing_boundaries
            .iter()
            .find(|e| (e.1 == a && e.2 == b) || (e.1 == b && e.2 == a))
            .map(|e| e.0)
    };
    let mut bnd_ids: Vec<u32> = Vec::new(); // every boundary of the region, in neighbour order (new and kept)
    let mut new_bnd_ids: Vec<u32> = Vec::new();
    let mut new_boundaries: Vec<Vec<Node>> = Vec::new();
    for (i, nb) in nb_cai.iter().enumerate() {
        if existing.is_some() {
            if let Some(id) = boundary_between(*nb, region_id) {
                bnd_ids.push(id);
                continue;
            }
        }
        let id = cid.take();
        bnd_ids.push(id);
        new_bnd_ids.push(id);
        let len = match spec.border_lengths.get(i) {
            Some(l) => *l,
            None => {
                let l = donor_len(*nb).map(|l| l / 2.0).unwrap_or(25.0);
                rep.notes.push(format!(
                    "border {}<->{}: no length given, estimated {l:.2}",
                    spec.key, spec.neighbours[i]
                ));
                l
            }
        };
        let mut it = bnd_tmpl.clone();
        nd::pristine_header(&mut it, Some(id))?;
        let s = nd::spec_kids_mut(&mut it)?;
        // (a, b) = (neighbour, new) while the new region has the highest index; an activated region sits in the middle
        let (a, b) = if *nb < region_id { (*nb, region_id) } else { (region_id, *nb) };
        nd::put(s, 0, nd::u32n(a))?;
        nd::put(s, 1, nd::u32n(b))?;
        nd::put(s, 2, nd::f32n(len))?;
        new_boundaries.push(it);
    }
    if existing.is_some() {
        rep.notes.push(format!(
            "{} of {} neighbour boundaries of {} already existed in the start position (ETW lengths kept)",
            bnd_ids.len() - new_bnd_ids.len(),
            bnd_ids.len(),
            spec.key
        ));
    }

    // ---- items --------------------------------------------------------------------------------------------------------
    let fx = unfixed(w.settlement_fixed.0);
    let fy = unfixed(w.settlement_fixed.1);
    let mut region_item = match &existing {
        Some((_, item)) => item.clone(),
        None => like_item.clone(),
    };
    if existing.is_none() {
        nd::pristine_header(&mut region_item, Some(region_id))?;
    }
    {
        let r = nd::spec_kids_mut(&mut region_item)?;
        if existing.is_some() {
            // the wilderness entry keeps its HLCIs and boundaries; the theatre list is the donor's
            nd::put(r, 0, like_spec[0].clone())?;
            let mut hlcis = nd::arr_at(r, 1)?;
            if existing_hlci.is_none() {
                hlcis.push(hlci_id);
            }
            nd::put(r, 1, nd::arrn(&hlcis))?;
            let mut bnds = nd::arr_at(r, 7)?;
            bnds.extend(&new_bnd_ids);
            nd::put(r, 7, nd::arrn(&bnds))?;
        } else {
            nd::put(r, 1, nd::arrn(&[hlci_id]))?;
            nd::put(r, 7, nd::arrn(&bnd_ids))?;
        }
        nd::put(r, 2, nd::u32n(settle_id))?;
        nd::put(r, 3, nd::empty_arr())?; // no town / resource slots
        nd::put(r, 4, nd::f32n(fx))?;
        nd::put(r, 5, nd::f32n(fy))?;
        nd::put(r, 6, nd::empty_arr())?;
        nd::put(r, 9, nd::empty_arr())?; // no mobiles
        let ch = r;
        ensure!(text(&ch[10]).is_some(), "CAI_REGION child 10 is not the key");
        set_text(&mut ch[10], &spec.key)?;
        nd::put(ch, 11, nd::u32n(w.region_id))?;
        nd::put(ch, 12, nd::u32n(gov_cai))?;
        nd::put(ch, 13, nd::empty_arr())?; // no factions present
    }
    nd::set_owner(&mut region_item, "OWNED_INDIRECT", settle_id)?;

    let mut settle_item = like_settle.clone();
    nd::pristine_header(&mut settle_item, Some(settle_id))?;
    if w.owner_changed {
        nd::set_owner(&mut settle_item, "OWNED_DIRECT", owner_cai)?;
    }
    {
        let (x, y) = w.settlement_fixed;
        let (theatre, kind) = {
            let sit = nd::kids(nd::rec_by_name(&settle_item, "CAI_SITUATED")?)?;
            (nd::arr_at(sit, 3)?, nd::sc_at(sit, 4)?)
        };
        ensure!(theatre.len() == 1, "settlement theatre list {theatre:?}");
        nd::set_situated(&mut settle_item, x, y, region_id, theatre[0], kind)?;
        let g = nd::kids_mut(nd::rec_by_name_mut(&mut settle_item, "CAI_GARRISONABLE")?)?;
        nd::put(g, 0, nd::u32n(0))?; // no garrison mobile
        nd::put(g, 2, nd::empty_arr())?;
        let cs = nd::kids_mut(nd::rec_by_name_mut(&mut settle_item, "CAI_SETTLEMENT")?)?;
        let old_slots = nd::arr_at(cs, 0)?;
        let new_slots: Vec<u32> = old_slots
            .iter()
            .filter_map(|s| {
                // the donor settlement's slot list maps through the world slot id of that CAI slot
                slot_templates
                    .iter()
                    .find(|t| nd::item_id(&t.1).ok() == Some(*s))
                    .and_then(|t| slot_cai.iter().find(|c| c.0 == t.0))
                    .map(|c| c.1)
            })
            .collect();
        ensure!(new_slots.len() == slot_cai.len(), "CAI_SETTLEMENT slot list does not cover the cloned slots");
        nd::put(cs, 0, nd::arrn(&new_slots))?;
        ensure!(nd::sc_at(cs, 2)? == w.old_settlement_id, "CAI_SETTLEMENT world id is not the donor settlement");
        nd::put(cs, 2, nd::u32n(w.settlement_id))?;
        if w.owner_changed {
            nd::put(cs, 1, nd::u32n(owner_cai))?; // CAI_SETTLEMENT owner faction
        }
    }

    let mut slot_items: Vec<Vec<Node>> = Vec::new();
    for (old_world, new_cai) in &slot_cai {
        let t = slot_templates.iter().find(|t| t.0 == *old_world).unwrap();
        let new_world = w.slots.iter().find(|s| s.1 == *old_world).unwrap().2;
        let mut it = t.1.clone();
        nd::pristine_header(&mut it, Some(*new_cai))?;
        let s = nd::spec_kids_mut(&mut it)?;
        nd::put(s, 0, nd::u32n(new_world))?;
        nd::put(s, 2, nd::u32n(settle_id))?;
        slot_items.push(it);
    }

    // a new HLCI is cloned from the donor's; a reused one only moves to the new settlement (below)
    let mut hlci_item = hlci_tmpl.clone();
    nd::pristine_header(&mut hlci_item, Some(hlci_id))?;
    {
        let h = nd::spec_kids_mut(&mut hlci_item)?;
        nd::put(h, 0, nd::u32n(region_id))?;
        nd::put(h, 2, nd::arrn(&[0]))?;
        nd::put(h, 3, nd::i32n(w.settlement_fixed.0))?;
        nd::put(h, 4, nd::i32n(w.settlement_fixed.1))?;
    }

    // ---- write phase ----------------------------------------------------------------------------------------------------
    let world = nd::get_mut(ci, &["CAI_WORLD"])?;
    match &existing {
        Some((index, _)) => {
            let slot = world_items_mut(world, "CAI_WORLD_REGIONS")?
                .get_mut(*index)
                .ok_or_else(|| anyhow!("CAI region index {index} vanished"))?;
            ensure!(spec_str(slot, 10) == Some(spec.key.as_str()), "CAI region {index} is not {}", spec.key);
            *slot = region_item;
        }
        None => world_items_mut(world, "CAI_WORLD_REGIONS")?.push(region_item),
    }
    world_items_mut(world, "CAI_WORLD_SETTLEMENTS")?.push(settle_item);
    world_items_mut(world, "CAI_WORLD_BUILDING_SLOTS")?.extend(slot_items);
    if existing_hlci.is_some() {
        let it = world_items_mut(world, "CAI_WORLD_REGION_HLCIS")?
            .iter_mut()
            .find(|it| nd::item_id(it).ok() == Some(hlci_id))
            .ok_or_else(|| anyhow!("HLCI {hlci_id} vanished"))?;
        let h = nd::spec_kids_mut(it)?;
        nd::put(h, 3, nd::i32n(w.settlement_fixed.0))?;
        nd::put(h, 4, nd::i32n(w.settlement_fixed.1))?;
    } else {
        world_items_mut(world, "CAI_WORLD_REGION_HLCIS")?.push(hlci_item);
    }
    world_items_mut(world, "CAI_WORLD_REGION_BOUNDARIES")?.extend(new_boundaries);
    rep.added.push(format!(
        "CAI_WORLD: CAI_REGION {region_id} (index {new_index}), CAI_SETTLEMENT {settle_id}, {} CAI_BUILDING_SLOT {:?}, CAI_REGION_HLCI {hlci_id}, {} CAI_REGION_BOUNDARY {:?}",
        slot_cai.len(),
        slot_cai.iter().map(|s| s.1).collect::<Vec<_>>(),
        bnd_ids.len(),
        bnd_ids
    ));

    // neighbours' boundary lists
    {
        let regs = world_items_mut(world, "CAI_WORLD_REGIONS")?;
        for (nb, bid) in nb_cai.iter().zip(&bnd_ids) {
            if !new_bnd_ids.contains(bid) {
                continue;
            }
            let it = regs
                .iter_mut()
                .find(|it| nd::item_id(it).ok() == Some(*nb))
                .ok_or_else(|| anyhow!("neighbour CAI region {nb} vanished"))?;
            let s = nd::spec_kids_mut(it)?;
            arr_push(&mut s[7], *bid)?;
        }
    }
    rep.added.push("CAI_WORLD_REGIONS: boundary lists of the neighbours += new boundary".into());

    // donor boundary changes
    for (key, len) in &spec.donor_borders {
        let other = *region_cai
            .get(key)
            .ok_or_else(|| anyhow!("donor_borders: CAI has no region {key}"))?;
        let (bid, _, _, _) = *existing_boundaries
            .iter()
            .find(|b| (b.1 == like_cai && b.2 == other) || (b.1 == other && b.2 == like_cai))
            .ok_or_else(|| anyhow!("donor_borders: {} has no boundary with {key}", spec.like))?;
        if *len > 0.0 {
            let b = world_items_mut(world, "CAI_WORLD_REGION_BOUNDARIES")?
                .iter_mut()
                .find(|it| nd::item_id(it).ok() == Some(bid))
                .unwrap();
            nd::put(nd::spec_kids_mut(b)?, 2, nd::f32n(*len))?;
            rep.added.push(format!("CAI_REGION_BOUNDARY {bid} ({} <-> {key}) length -> {len:.2}", spec.like));
        } else {
            world_items_mut(world, "CAI_WORLD_REGION_BOUNDARIES")?.retain(|it| nd::item_id(it).ok() != Some(bid));
            for it in world_items_mut(world, "CAI_WORLD_REGIONS")?.iter_mut() {
                let id = nd::item_id(it)?;
                if id == like_cai || id == other {
                    arr_remove(&mut nd::spec_kids_mut(it)?[7], bid)?;
                }
            }
            rep.added.push(format!("CAI_REGION_BOUNDARY {bid} ({} <-> {key}) removed", spec.like));
        }
    }

    // theatre region list: the CAI regions of the donor's theatre in CAI region order (one list per theatre; the
    // vanilla Europe startpos has one list holding every region). A new region has the highest CAI id; an activated
    // wilderness region is normally listed already and is inserted in order when it is not.
    {
        let th = world_items_mut(world, "CAI_WORLD_THEATRES")?;
        let mut done = 0;
        for it in th.iter_mut() {
            let s = nd::spec_kids_mut(it)?;
            if existing.is_some() && nd::arr_vals(&s[3]).contains(&region_id) {
                ensure!(
                    nd::arr_vals(&s[3]).contains(&like_cai),
                    "wilderness region {} is listed by a different CAI theatre than its donor {}",
                    spec.key,
                    spec.like
                );
                done += 1;
                rep.notes.push(format!("CAI theatre list already holds {} (CAI id {region_id})", spec.key));
            } else if nd::is_u32_arr(&s[3]) && nd::arr_vals(&s[3]).contains(&like_cai) {
                let mut vals = nd::arr_vals(&s[3]);
                if existing.is_some() {
                    ensure!(
                        vals.windows(2).all(|p| p[0] <= p[1]),
                        "the donor theatre's region list is not in CAI id order, cannot insert {}",
                        spec.key
                    );
                    let at = vals.partition_point(|v| *v < region_id);
                    vals.insert(at, region_id);
                    nd::put(s, 3, nd::arrn(&vals))?;
                } else {
                    if vals.windows(2).any(|p| p[0] > p[1]) {
                        rep.notes.push("the donor theatre's region list is not in CAI id order; the new region was appended".into());
                    }
                    arr_push(&mut s[3], region_id)?;
                }
                done += 1;
            }
        }
        ensure!(done == 1, "donor region is in {done} CAI theatres");
    }
    rep.added.push(format!("CAI_WORLD_THEATRES: region list of the donor's theatre lists {region_id} (CAI region index {new_index})"));

    // owner CAI faction owned-region list
    {
        let fac = world_items_mut(world, "CAI_WORLD_FACTIONS")?;
        let mut done = 0;
        for it in fac.iter_mut() {
            if nd::item_id(it).ok() == Some(owner_cai) {
                let s = nd::spec_kids_mut(it)?;
                ensure!(
                    nd::arr_vals(&s[0]).contains(&like_cai) != w.owner_changed,
                    "owner CAI faction {} the donor region",
                    if w.owner_changed { "already lists" } else { "does not list" }
                );
                ensure!(
                    nd::sc_at(s, 6)? == w.owner_id,
                    "CAI faction {owner_cai} is not world faction {}",
                    w.owner_id
                );
                arr_push(&mut s[0], region_id)?;
                done += 1;
            }
        }
        ensure!(done == 1, "owner CAI faction {owner_cai} found {done} times");
    }
    rep.added.push(format!("CAI_WORLD_FACTIONS: CAI faction {owner_cai} owned regions += {region_id}"));

    // CAI governorship
    {
        let gv = world_items_mut(world, "CAI_WORLD_GOVERNORSHIPS")?;
        let mut done = 0;
        for it in gv.iter_mut() {
            let s = nd::spec_kids_mut(it)?;
            if nd::sc_at(s, 0)? == w.gov_id {
                ensure!(
                    nd::arr_vals(&s[3]).contains(&like_cai) != w.owner_changed,
                    "CAI governorship {} the donor region",
                    if w.owner_changed { "already lists" } else { "does not list" }
                );
                arr_push(&mut s[3], region_id)?;
                done += 1;
            }
        }
        ensure!(done == 1, "CAI governorship of world governorship {} found {done} times", w.gov_id);
    }
    rep.added.push(format!("CAI_WORLD_GOVERNORSHIPS: governorship {} region list += {region_id}", w.gov_id));

    // ---- root pool analysers #11 (border patrol) / #20 (region occupancy). The generated start position holds these
    // beliefs for every CAI region, so an activated wilderness region has them already.
    if existing.is_some() {
        let have = region_belief_count(ci, region_id)?;
        ensure!(
            have == REGION_ROOT.len(),
            "CAI region {} has {have} of the {} root region beliefs",
            spec.key,
            REGION_ROOT.len()
        );
        rep.notes.push(format!("{}: border patrol / occupancy beliefs already present", spec.key));
    } else {
        add_region_beliefs(ci, region_id, &mut cid, rep)?;
    }

    nd::put(
        nd::kids_mut(nd::get_mut(ci, &["CAI_CENTRAL_BDI_POOL"])?)?,
        0,
        nd::u32n(cid.next),
    )?;
    rep.cai_range = (cid.first, cid.next - 1);
    rep.added.push(format!(
        "CAI_CENTRAL_BDI_POOL next CAI id {next} -> {} ({} ids allocated)",
        cid.next,
        cid.next - cid.first
    ));

    rep.skipped.extend([
        "CAI_WORLD_REGION_SLOTS / CAI_WORLD_RESOURCE_MOBILES: the clone has no town, resource or port slot".to_string(),
        "CAI_WORLD_TRADE_ROUTE_SEGMENTS: segments carry no region list that needs the new region".to_string(),
        format!(
            "CAI_INTERFACE_MANAGERS (42 faction pools) and the root pool's other per-region beliefs (region / settlement base value, unit availability, border / group analysis ...): vanilla holds hundreds of beliefs for the donor ({like_cai}); the converter ships every region without them and the engine rebuilds them, so none is cloned for {region_id}"
        ),
        "CAI_WORLD [32] belief registry: only lists beliefs of the unit-availability/recruitment families that are not created here".to_string(),
        "CAI_HISTORY events (CAI_HISTORY_EVENT_INVASION_REQUESTED region key lists): historical scripts, left as vanilla".to_string(),
    ]);
    Ok(())
}

const REGION_ROOT: [(u32, &str); 2] = [(11, "CAI_BORDER_PATROL_ANALYSIS"), (20, "CAI_REGION_OCCUPANCY_ANALYSIS")];

/// How many of the root region analysers (#11 / #20) already hold a belief of `region_cai`.
fn region_belief_count(ci: &Node, region_cai: u32) -> Result<usize> {
    let rp = nd::get(ci, &["CAI_BDI_POOL"])?;
    let mut n = 0;
    for (d, _) in REGION_ROOT {
        let desire = nd::get_items(rp, &["CAI_BDI_POOL_DESIRES"])?
            .iter()
            .find(|it| nd::sc_at(it, 2).ok() == Some(d))
            .ok_or_else(|| anyhow!("root desire {d} missing"))?;
        let pairs = nd::arr_at(nd::kids(nd::rec_by_name(desire, "CAI_ANALYSER")?)?, 1)?;
        if pairs.chunks_exact(2).any(|p| p[0] == region_cai) {
            n += 1;
        }
    }
    Ok(n)
}

/// One belief per root desire #11 / #20, built exactly like `startpos::cai::CaiBuilder::add_region_beliefs`.
fn add_region_beliefs(ci: &mut Node, region_cai: u32, cid: &mut CaiIds, rep: &mut Report) -> Result<()> {
    let rp = nd::get_mut(ci, &["CAI_BDI_POOL"])?;
    for (d, expect) in REGION_ROOT {
        // template: the first belief of the analyser that owns no other belief
        let (tmpl, owns_tmpl) = {
            let desires = nd::get_items(rp, &["CAI_BDI_POOL_DESIRES"])?;
            let desire = desires
                .iter()
                .find(|it| nd::sc_at(it, 2).ok() == Some(d))
                .ok_or_else(|| anyhow!("root desire {d} missing"))?;
            let ids = nd::arr_at(nd::kids(nd::rec_by_name(desire, "CAI_ANALYSER")?)?, 0)?;
            let beliefs = nd::get_items(rp, &["CAI_BDI_POOL_BELIEFS"])?;
            let by_id: HashMap<u32, &Vec<Node>> = beliefs
                .iter()
                .filter_map(|it| nd::sc_at(it, 2).ok().map(|id| (id, it)))
                .collect();
            let mut cand = None;
            for id in ids {
                let b = by_id
                    .get(&id)
                    .ok_or_else(|| anyhow!("analyser {d}: belief {id} is not in the root pool"))?;
                if nd::item_array(b, 15)?.is_empty() {
                    cand = Some((*b).clone());
                    break;
                }
            }
            let owns = nd::item_array(desire, 15)?
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("root desire {d} owns nothing"))?;
            (cand.ok_or_else(|| anyhow!("analyser {d} has no belief without BLOCK_OWNS"))?, owns)
        };
        ensure!(
            nd::spec_name(&tmpl)? == expect,
            "root desire {d}: template is {}, expected {expect}",
            nd::spec_name(&tmpl)?
        );
        let position = {
            let desires = nd::get_items(rp, &["CAI_BDI_POOL_DESIRES"])?;
            let desire = desires.iter().find(|it| nd::sc_at(it, 2).ok() == Some(d)).unwrap();
            let n = nd::item_array(desire, 15)?.len();
            ensure!(
                nd::sc_at(desire, 17)? as usize == n,
                "root desire {d}: BLOCK_OWNS count child {} != {n}",
                nd::sc_at(desire, 17)?
            );
            n as u32
        };
        let bid = cid.take();
        let mut b = tmpl;
        nd::put(&mut b, 2, nd::u32n(bid))?;
        nd::put(nd::spec_kids_mut(&mut b)?, 0, nd::u32n(region_cai))?;
        nd::put(&mut b, 10, nd::arrn(&[d, position]))?;
        nd::put(&mut b, 14, nd::arrn(&[d]))?;
        nd::put(&mut b, 19, nd::empty_arr())?;
        nd::put(&mut b, 20, nd::arrn(&if d == 20 { vec![region_cai] } else { Vec::new() }))?;
        nd::put(&mut b, 21, nd::empty_arr())?;
        nd::item_array_mut(&mut b, 15)?.clear();
        nd::put(&mut b, 17, nd::u32n(0))?;
        if d == 20 {
            let sc = nd::spec_kids_mut(&mut b)?;
            ensure!(sc.get(1).map(nd::is_u32_arr).unwrap_or(false), "occupancy belief child 1 is not a u32 list");
            nd::put(sc, 1, nd::empty_arr())?;
        } else {
            // a fresh region has no specific patrol areas
            let sc = nd::spec_kids_mut(&mut b)?;
            ensure!(sc.len() > 1, "border patrol belief too short");
            match &mut sc[1] {
                Node::Array { items, .. } => items.clear(),
                other => bail!("border patrol belief child 1 is {other:?}"),
            }
        }
        nd::arr_items_mut(nd::get_mut(rp, &["CAI_BDI_POOL_BELIEFS"])?)?.push(b);

        let desires = nd::arr_items_mut(nd::get_mut(rp, &["CAI_BDI_POOL_DESIRES"])?)?;
        let it = desires
            .iter_mut()
            .find(|it| nd::sc_at(it, 2).ok() == Some(d))
            .unwrap();
        let mut o = owns_tmpl;
        nd::put(&mut o, 0, nd::u32n(bid))?;
        nd::put(&mut o, 1, nd::u32n(d))?;
        nd::item_array_mut(it, 15)?.push(o);
        nd::put(it, 17, nd::u32n(position + 1))?;
        ensure!(nd::is_u32_arr(&it[19]), "desire {d} child 19 is not a u32 list");
        arr_push(&mut it[19], bid)?;
        let ca = nd::kids_mut(nd::rec_by_name_mut(it, "CAI_ANALYSER")?)?;
        arr_push(&mut ca[0], bid)?;
        arr_push(&mut ca[1], region_cai)?;
        arr_push(&mut ca[1], bid)?;
        rep.added.push(format!(
            "CAI_BDI_POOL: {expect} belief {bid} for the region, root desire {d} BLOCK_OWNS[{position}] / analyser lists"
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ verification

/// Structural cross-check of an edited startpos against its vanilla original. Returns the list of problems (empty =
/// consistent). Mirrors `startpos::cai::check` (reference fields are learned from the vanilla CAI graph and enforced on
/// the edited one) and adds the WORLD id cross references the split relies on.
pub fn check(vanilla: &Node, edited: &Node, spec: &NewRegion) -> Result<Vec<String>> {
    let mut problems = Vec::new();
    let vm = vanilla.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])?;
    let em = edited.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])?;
    cai_check(vm.get("CAI_INTERFACE")?, em.get("CAI_INTERFACE")?, &mut problems)?;
    world_check(edited, spec, &mut problems)?;
    Ok(problems)
}

type Key = (u32, u32, u8);

#[derive(Default)]
struct Paths {
    kids: Vec<HashMap<String, u32>>,
    parent: Vec<(u32, String)>,
}

impl Paths {
    fn new() -> Self {
        Self {
            kids: vec![HashMap::new()],
            parent: vec![(0, "/CAI_INTERFACE".to_string())],
        }
    }
    fn child(&mut self, pid: u32, name: &str) -> u32 {
        if let Some(id) = self.kids[pid as usize].get(name) {
            return *id;
        }
        let id = self.parent.len() as u32;
        self.parent.push((pid, name.to_string()));
        self.kids.push(HashMap::new());
        self.kids[pid as usize].insert(name.to_string(), id);
        id
    }
    fn render(&self, mut pid: u32) -> String {
        let mut parts = Vec::new();
        loop {
            let (p, n) = &self.parent[pid as usize];
            parts.push(n.clone());
            if pid == 0 {
                break;
            }
            pid = *p;
        }
        parts.reverse();
        parts.join("/")
    }
}

fn walk_fields(ch: &[Node], pid: u32, paths: &mut Paths, out: &mut HashMap<Key, Vec<u32>>) {
    const QUAD: [&str; 4] = ["bottom_left", "top_left", "bottom_right", "top_right"];
    for (i, c) in ch.iter().enumerate() {
        match c {
            Node::Record { name, children, .. } => {
                let nm = if QUAD.contains(&name.as_str()) { "Q" } else { name.as_str() };
                let cp = paths.child(pid, nm);
                walk_fields(children, cp, paths, out);
            }
            Node::Array { name, items, .. } => {
                let cp = paths.child(pid, &format!("{name}[]"));
                for it in items {
                    walk_fields(it, cp, paths, out);
                }
            }
            Node::Value(Value::U32(v)) => out.entry((pid, i as u32, 0)).or_default().push(*v),
            Node::Value(Value::Array { code: 0x48, data }) if !data.is_empty() => {
                let e = out.entry((pid, i as u32, 1)).or_default();
                for b in data.chunks_exact(4) {
                    e.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
            }
            _ => {}
        }
    }
}

fn item_hdr(it: &[Node]) -> Option<u32> {
    let i = it.iter().position(|c| nd::is_record_named(c, nd::PROPSET))?;
    it[i + 1..].iter().find(|c| nd::is_value(c)).and_then(nd::scalar)
}

fn pool_arrays(pool: &Node) -> Result<Vec<(&str, &Vec<Vec<Node>>)>> {
    Ok(nd::kids(pool)?
        .iter()
        .filter_map(|a| match a {
            Node::Array { name, items, .. } if name.starts_with("CAI_BDI_POOL_") => Some((name.as_str(), items)),
            _ => None,
        })
        .collect())
}

/// every CAI object id
fn universe(ci: &Node) -> Result<HashSet<u32>> {
    let mut u = HashSet::new();
    let world = nd::get(ci, &["CAI_WORLD"])?;
    for a in nd::kids(world)? {
        if let Node::Array { items, .. } = a {
            for it in items {
                if let Some(h) = item_hdr(it) {
                    u.insert(h);
                }
            }
        }
    }
    for it in nd::get_items(world, &["CAI_WORLD_FACTIONS"])? {
        let r_ = nd::spec_kids(it)?;
        let pi = nd::find_idx(r_, nd::PROPSET).ok_or_else(|| anyhow!("CAI_FACTION without PROPSET"))?;
        u.insert(nd::sc_at(r_, pi + 1)?);
    }
    let flat: Vec<&Node> = nd::kids(ci)?.iter().filter(|c| !matches!(c, Node::Array { .. })).collect();
    if let Some(i) = flat.iter().position(|c| nd::is_record_named(c, nd::PROPSET)) {
        if let Some(h) = flat[i + 1..].iter().find(|c| nd::is_value(c)).and_then(|c| nd::scalar(c)) {
            u.insert(h);
        }
    }
    let wh = nd::kids(world)?
        .iter()
        .find(|c| nd::is_u32(c))
        .and_then(nd::scalar)
        .ok_or_else(|| anyhow!("CAI_WORLD without header id"))?;
    u.insert(wh);
    let mut add_pool = |p: &Node| -> Result<()> {
        for (_, items) in pool_arrays(p)? {
            for it in items {
                if let Some(h) = item_hdr(it) {
                    u.insert(h);
                }
            }
        }
        Ok(())
    };
    add_pool(nd::get(ci, &["CAI_INTERFACE_DIRECTOR_BDI_POOL", "CAI_BDI_POOL"])?)?;
    add_pool(nd::get(ci, &["CAI_BDI_POOL"])?)?;
    for m in nd::get_items(ci, &["CAI_INTERFACE_MANAGERS"])? {
        add_pool(m.first().ok_or_else(|| anyhow!("empty CAI manager item"))?)?;
    }
    Ok(u)
}

fn is_ref(v: u32, floor: u32) -> bool {
    v >= floor && v != 0xFFFF_FFFF
}

fn cai_check(vci: &Node, eci: &Node, problems: &mut Vec<String>) -> Result<()> {
    const LEARN_RATE: f64 = 0.98;
    const HDR_LISTS: [usize; 8] = [9, 10, 13, 14, 18, 19, 20, 21];
    let mut paths = Paths::new();
    let vu = universe(vci)?;
    let mut vf = HashMap::new();
    walk_fields(nd::kids(vci)?, 0, &mut paths, &mut vf);
    let mut refs: HashSet<Key> = HashSet::new();
    for (key, vals) in &vf {
        let nv: Vec<u32> = vals.iter().copied().filter(|v| is_ref(*v, 100)).collect();
        if nv.len() < 3 {
            continue;
        }
        let distinct: HashSet<u32> = nv.iter().copied().collect();
        if distinct.len() < 2 {
            continue;
        }
        let hit = nv.iter().filter(|v| vu.contains(v)).count();
        if hit as f64 / nv.len() as f64 >= LEARN_RATE {
            refs.insert(*key);
        }
    }
    let eu = universe(eci)?;
    let mut ef = HashMap::new();
    walk_fields(nd::kids(eci)?, 0, &mut paths, &mut ef);
    // dangling ids in learned reference fields (vanilla's own dangling values are tolerated)
    let mut vanilla_missing: HashSet<(Key, u32)> = HashSet::new();
    for key in &refs {
        if let Some(vals) = vf.get(key) {
            for v in vals {
                if is_ref(*v, 100) && !vu.contains(v) {
                    vanilla_missing.insert((*key, *v));
                }
            }
        }
    }
    let mut dangling: Vec<(Key, Vec<u32>)> = Vec::new();
    for key in &refs {
        if let Some(vals) = ef.get(key) {
            let miss: Vec<u32> = vals
                .iter()
                .copied()
                .filter(|v| is_ref(*v, 100) && !eu.contains(v) && !vanilla_missing.contains(&(*key, *v)))
                .collect();
            if !miss.is_empty() {
                dangling.push((*key, miss));
            }
        }
    }
    dangling.sort_by_key(|d| d.0);
    for (key, miss) in dangling {
        problems.push(format!(
            "DANGLING CAI id in {}#{}: {} values, e.g. {:?}",
            paths.render(key.0),
            key.1,
            miss.len(),
            &miss[..miss.len().min(3)]
        ));
    }

    // duplicate ids, header lists, header #10 pairs
    let mut pools: Vec<(String, &Node)> = vec![
        ("root".into(), nd::get(eci, &["CAI_BDI_POOL"])?),
        ("director".into(), nd::get(eci, &["CAI_INTERFACE_DIRECTOR_BDI_POOL", "CAI_BDI_POOL"])?),
    ];
    for (k, m) in nd::get_items(eci, &["CAI_INTERFACE_MANAGERS"])?.iter().enumerate() {
        pools.push((format!("manager {k}"), m.first().ok_or_else(|| anyhow!("empty manager"))?));
    }
    let mut ids_seen: HashMap<u32, usize> = HashMap::new();
    let mut allit: HashMap<u32, &Vec<Node>> = HashMap::new();
    let mut unresolved: BTreeMap<(String, usize), (usize, u32, u32)> = BTreeMap::new();
    for (tag, p) in &pools {
        let class = if tag == "root" || tag == "director" { tag.clone() } else { "manager".to_string() };
        for (name, items) in pool_arrays(p)? {
            if name == "CAI_BDI_POOL_FAILED" {
                continue;
            }
            for it in items {
                let id = nd::sc_at(it, 2)?;
                *ids_seen.entry(id).or_insert(0) += 1;
                allit.insert(id, it);
                for i in HDR_LISTS {
                    if let Some(c) = it.get(i) {
                        if nd::is_u32_arr(c) {
                            for v in nd::arr_vals(c) {
                                if v >= 100 && !eu.contains(&v) {
                                    unresolved.entry((class.clone(), i)).or_insert((0, v, id)).0 += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (k, (n, v, id)) in unresolved {
        problems.push(format!("UNRESOLVED pool header list #{} of {} pools: {n} values, e.g. {v} in item {id}", k.1, k.0));
    }
    for a in nd::kids(nd::get(eci, &["CAI_WORLD"])?)? {
        if let Node::Array { items, .. } = a {
            for it in items {
                if let Some(h) = item_hdr(it) {
                    *ids_seen.entry(h).or_insert(0) += 1;
                    allit.entry(h).or_insert(it);
                }
            }
        }
    }
    let mut dups: Vec<u32> = ids_seen.iter().filter(|(_, n)| **n > 1).map(|(k, _)| *k).collect();
    dups.sort_unstable();
    if !dups.is_empty() {
        problems.push(format!("DUPLICATE CAI ids: {} e.g. {:?}", dups.len(), &dups[..dups.len().min(6)]));
    }
    let mut xids: Vec<u32> = allit.keys().copied().collect();
    xids.sort_unstable();
    for xid in xids {
        let it = allit[&xid];
        if it.len() <= 10 || !nd::is_u32_arr(&it[10]) || nd::is_record_named(&it[0], nd::PROPSET) {
            continue;
        }
        let l = nd::arr_vals(&it[10]);
        for pair in l.chunks_exact(2) {
            let (t_, ix) = (pair[0], pair[1] as usize);
            let Some(target) = allit.get(&t_) else {
                problems.push(format!("PAIR item {xid} -> {t_}: target is not a CAI item"));
                continue;
            };
            let Some(ow) = target.iter().find(|x| nd::is_array_named(x, nd::BLOCK_OWNS)) else {
                problems.push(format!("PAIR item {xid} -> {t_}: target has no BLOCK_OWNS"));
                continue;
            };
            let owned = nd::arr_items(ow)?;
            if ix >= owned.len() {
                problems.push(format!("PAIR item {xid} -> {t_}: position {ix} of {}", owned.len()));
            } else if owned[ix].first().and_then(nd::scalar) != Some(xid) {
                problems.push(format!("PAIR item {xid} -> {t_}: BLOCK_OWNS[{ix}] is another item"));
            }
        }
    }

    // next-id counter above every id
    let counter = nd::sc_at(nd::kids(nd::get(eci, &["CAI_CENTRAL_BDI_POOL"])?)?, 0)?;
    let max = eu.iter().max().copied().unwrap_or(0);
    if counter <= max {
        problems.push(format!("CAI_CENTRAL_BDI_POOL next id {counter} <= highest id {max}"));
    }
    Ok(())
}

fn world_check(edited: &Node, spec: &NewRegion, problems: &mut Vec<String>) -> Result<()> {
    let model = edited.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL"])?;
    let world = model.get("WORLD")?;
    let regions = world.path(&["REGION_MANAGER", "REGIONS_ARRAY"])?.items()?;
    let mut region_ids: HashMap<String, u32> = HashMap::new();
    let mut owners: HashMap<String, u32> = HashMap::new();
    let mut seen_ids: HashMap<u32, String> = HashMap::new();
    let mut dup = |id: u32, what: String, problems: &mut Vec<String>| {
        if let Some(prev) = seen_ids.insert(id, what.clone()) {
            problems.push(format!("world id {id} used by both {prev} and {what}"));
        }
    };
    let mut slot_ids: HashSet<u32> = HashSet::new();
    let mut settlement_ids: HashMap<String, u32> = HashMap::new();
    for it in regions {
        let r = &it[0];
        let ch = r.children()?;
        let key = text(&ch[0]).ok_or_else(|| anyhow!("REGION key"))?.to_string();
        let id = id_of(&ch[4])?;
        dup(id, format!("region {key}"), problems);
        region_ids.insert(key.clone(), id);
        owners.insert(key.clone(), id_of(&ch[20])?);
        let sgr = r.path(&["SETTLEMENT", "SIEGEABLE_GARRISON_RESIDENCE"])?.children()?;
        let sid = id_of(&sgr[1])?;
        dup(sid, format!("settlement of {key}"), problems);
        settlement_ids.insert(key.clone(), sid);
        let rm = id_of(&r.get("REGION_RECRUITMENT_MANAGER")?.children()?[2])?;
        dup(rm, format!("recruitment manager of {key}"), problems);
        let sm = r.get("REGION_SLOT_MANAGER")?;
        let mut slots: Vec<&Node> = sm.get("REGION_SLOT_ARRAY")?.items()?.iter().map(|i| &i[0]).collect();
        for c in ["ROAD_SLOT", "FORTIFICATION_SLOT"] {
            if let Some(s) = sm.get(c)?.children()?.iter().find(|n| nd::is_record_named(n, "REGION_SLOT")) {
                slots.push(s);
            }
        }
        for s in slots {
            let sc = s.children()?;
            let sidv = id_of(&sc[2])?;
            ensure_eq(&mut *problems, id_of(&sc[0].children()?[1])?, sidv, || format!("slot residence id of {key}"));
            dup(sidv, format!("slot {}", text(&sc[3]).unwrap_or("?")), problems);
            slot_ids.insert(sidv);
        }
    }
    if region_ids.len() != regions.len() {
        problems.push(format!("REGIONS_ARRAY has duplicate keys ({} items, {} keys)", regions.len(), region_ids.len()));
    }
    if !region_ids.contains_key(&spec.key) {
        problems.push(format!("{} is not in REGIONS_ARRAY", spec.key));
    }

    // governorship lists: every region exactly once, REGION child 22 resolves
    let mut gov_regions: HashMap<u32, Vec<u32>> = HashMap::new();
    for it in world.get("FACTION_ARRAY")?.items()? {
        for f in it.iter().filter(|n| nd::is_record_named(n, "FACTION")) {
            for post in f.path(&["GOVERNMENT", "POSTS_ARRAY"])?.items()? {
                for rec in post.iter().filter(|n| nd::is_record_named(n, "CHARACTER_POST")) {
                    for g in rec.children()?.iter().filter(|n| nd::is_record_named(n, "GOVERNORSHIP")) {
                        let gc = g.children()?;
                        gov_regions.insert(id_of(&gc[1])?, nd::arr_vals(&gc[2]));
                    }
                }
            }
        }
    }
    let mut listed: HashMap<u32, usize> = HashMap::new();
    for v in gov_regions.values() {
        for r in v {
            *listed.entry(*r).or_insert(0) += 1;
        }
    }
    for it in regions {
        let ch = it[0].children()?;
        let key = text(&ch[0]).unwrap_or("?");
        let rid = id_of(&ch[4])?;
        let gid = id_of(&ch[22])?;
        match gov_regions.get(&gid) {
            None => problems.push(format!("{key}: REGION child 22 {gid} is not a governorship")),
            Some(l) if !l.contains(&rid) => problems.push(format!("{key}: governorship {gid} does not list the region")),
            _ => {}
        }
        if listed.get(&rid).copied().unwrap_or(0) != 1 {
            problems.push(format!("{key}: listed by {} governorships", listed.get(&rid).copied().unwrap_or(0)));
        }
    }
    for (g, l) in &gov_regions {
        for r in l {
            if !region_ids.values().any(|v| v == r) {
                problems.push(format!("governorship {g} lists unknown region id {r}"));
            }
        }
    }

    // REGION_OWNERSHIPS_BY_THEATRE and SETTLEMENT_INDICES agree with the regions
    let mut factions: HashMap<u32, String> = HashMap::new();
    for it in world.get("FACTION_ARRAY")?.items()? {
        for f in it.iter().filter(|n| nd::is_record_named(n, "FACTION")) {
            let (id, key) = faction_id_key(f).ok_or_else(|| anyhow!("FACTION without id / key"))?;
            factions.insert(id, key);
        }
    }
    let mut owned_listed: HashMap<String, String> = HashMap::new();
    for th in edited
        .path(&["CAMPAIGN_PREOPEN_MAP_INFO", "REGION_OWNERSHIPS_BY_THEATRE"])?
        .items()?
    {
        for own in th.iter().filter(|n| nd::is_array_named(n, "REGION_OWNERSHIPS")) {
            for o in own.items()? {
                owned_listed.insert(text(&o[0]).unwrap_or("?").to_string(), text(&o[1]).unwrap_or("?").to_string());
            }
        }
    }
    for (key, owner) in &owners {
        let name = factions.get(owner).cloned().unwrap_or_default();
        if owned_listed.get(key) != Some(&name) {
            problems.push(format!("{key}: REGION owner {name:?} but REGION_OWNERSHIPS says {:?}", owned_listed.get(key)));
        }
    }
    if owned_listed.len() != owners.len() {
        problems.push(format!("REGION_OWNERSHIPS lists {} regions, REGIONS_ARRAY {}", owned_listed.len(), owners.len()));
    }
    let tm = model.get("CAMPAIGN_TRADE_MANAGER")?;
    let si = tm.get("SETTLEMENT_INDICES")?.items()?;
    let keys: BTreeSet<&str> = si.iter().filter_map(|it| text(&it[0])).collect();
    // every settlement index belongs to a region and the new region has one (a few regions have none: islands)
    if keys.len() != si.len() || !keys.iter().all(|k| region_ids.contains_key(*k)) || !keys.contains(spec.key.as_str()) {
        problems.push("SETTLEMENT_INDICES holds duplicates / unknown regions or lacks the new region".to_string());
    }
    let vals: BTreeSet<u32> = si.iter().filter_map(|it| id_of(&it[1]).ok()).collect();
    let port_vals: BTreeSet<u32> = tm
        .get("PORT_INDICES")?
        .items()?
        .iter()
        .filter_map(|it| id_of(&it[1]).ok())
        .collect();
    let n_ports = tm.get("PORT_INDICES")?.items()?.len() as u32;
    if port_vals != (0..n_ports).collect::<BTreeSet<u32>>() {
        problems.push(format!("PORT_INDICES values are not the contiguous range 0..{n_ports}"));
    }
    let settle_end = n_ports + si.len() as u32;
    let expect: BTreeSet<u32> = (n_ports..settle_end).collect();
    if vals != expect {
        problems.push(format!("SETTLEMENT_INDICES values are not the contiguous range {n_ports}..{settle_end}"));
    }
    // trade nodes: one contiguous run above the settlements (a gap between the two is allowed)
    let mut node_idx: Vec<u32> = tm.get("TRADE_NODES")?.items()?.iter().map(|it| id_of(&it[1]).unwrap_or(0)).collect();
    node_idx.sort_unstable();
    if node_idx.first().is_none_or(|f| *f < settle_end) || node_idx.windows(2).any(|p| p[1] != p[0] + 1) {
        problems.push(format!("TRADE_NODES indices {node_idx:?} are not one contiguous run at or above {settle_end}"));
    }
    let nodes: BTreeSet<u32> = node_idx.iter().copied().collect();
    let known = |v: u32| v < settle_end || nodes.contains(&v);
    for it in tm.get("TRADE_ROUTES")?.items()? {
        let (a, b) = (id_of(&it[0])?, id_of(&it[1])?);
        if !known(a) || !known(b) || a >= b {
            problems.push(format!("TRADE_ROUTES entry {a} -> {b} references an unknown node / is unordered"));
            break;
        }
    }

    // CAI cross references into the world
    let cw = model.path(&["CAI_INTERFACE", "CAI_WORLD"])?;
    let mut cai_region_of: HashMap<u32, String> = HashMap::new();
    for it in nd::get_items(cw, &["CAI_WORLD_REGIONS"])? {
        let s = nd::spec_kids(it)?;
        let key = text(&s[10]).unwrap_or("?").to_string();
        let wid = nd::sc_at(s, 11)?;
        if let Some(rid) = region_ids.get(&key) {
            if *rid != wid {
                problems.push(format!("CAI_REGION {key} points at world id {wid}, the REGION has {rid}"));
            }
        } else if wid != 0 && !key.starts_with("eur_map") && region_ids.values().any(|v| *v == wid) {
            problems.push(format!("CAI_REGION {key} points at the world id of another region"));
        }
        cai_region_of.insert(nd::item_id(it)?, key);
    }
    // the new region is listed by exactly one theatre (a few vanilla regions, e.g. kalahari, are listed by none)
    let mut in_theatres: HashMap<u32, usize> = HashMap::new();
    for it in nd::get_items(cw, &["CAI_WORLD_THEATRES"])? {
        for r in nd::arr_vals(&nd::spec_kids(it)?[3]) {
            *in_theatres.entry(r).or_insert(0) += 1;
        }
    }
    for (id, key) in &cai_region_of {
        let n = in_theatres.get(id).copied().unwrap_or(0);
        if key == &spec.key && n != 1 {
            problems.push(format!("CAI region {key} is listed by {n} theatres"));
        }
    }
    // the new / activated CAI region: its settlement, HLCIs and boundaries hang together
    {
        let item = nd::get_items(cw, &["CAI_WORLD_REGIONS"])?
            .iter()
            .find(|it| spec_str(it, 10) == Some(spec.key.as_str()))
            .ok_or_else(|| anyhow!("{} is not a CAI region", spec.key))?;
        let rid = nd::item_id(item)?;
        let s = nd::spec_kids(item)?;
        let settle = nd::sc_at(s, 2)?;
        match nd::get_items(cw, &["CAI_WORLD_SETTLEMENTS"])?.iter().find(|it| nd::item_id(it).ok() == Some(settle)) {
            None => problems.push(format!("CAI region {} has no CAI settlement (id {settle})", spec.key)),
            Some(st) => {
                if settlement_ids.get(&spec.key) != Some(&spec_id(st, 2)?) {
                    problems.push(format!("CAI settlement of {} is not the settlement of its REGION", spec.key));
                }
                let sit = nd::kids(nd::rec_by_name(st, "CAI_SITUATED")?)?;
                if nd::sc_at(sit, 2)? != rid {
                    problems.push(format!("CAI settlement of {} sits in another CAI region", spec.key));
                }
                let owned = nd::arr_at(nd::kids(nd::rec_by_name(st, "CAI_SETTLEMENT")?)?, 0)?;
                let slots = nd::arr_at(s, 3)?;
                if !slots.is_empty() {
                    problems.push(format!("CAI region {} lists {} town / resource slots", spec.key, slots.len()));
                }
                for slot in owned {
                    let ok = nd::get_items(cw, &["CAI_WORLD_BUILDING_SLOTS"])?
                        .iter()
                        .any(|b| nd::item_id(b).ok() == Some(slot) && spec_id(b, 2).ok() == Some(settle));
                    if !ok {
                        problems.push(format!("CAI settlement of {} lists slot {slot} which is not its building slot", spec.key));
                    }
                }
            }
        }
        let hlcis = nd::arr_at(s, 1)?;
        if hlcis.is_empty() {
            problems.push(format!("CAI region {} has no HLCI", spec.key));
        }
        for h in &hlcis {
            let ok = nd::get_items(cw, &["CAI_WORLD_REGION_HLCIS"])?
                .iter()
                .any(|it| nd::item_id(it).ok() == Some(*h) && spec_id(it, 0).ok() == Some(rid));
            if !ok {
                problems.push(format!("CAI region {} lists HLCI {h} that does not belong to it", spec.key));
            }
        }
        let listed = nd::arr_at(s, 7)?;
        let mut by_boundary: HashMap<u32, (u32, u32)> = HashMap::new();
        for b in nd::get_items(cw, &["CAI_WORLD_REGION_BOUNDARIES"])? {
            let (a, c) = (spec_id(b, 0)?, spec_id(b, 1)?);
            by_boundary.insert(nd::item_id(b)?, (a, c));
            if (a == rid || c == rid) && !listed.contains(&nd::item_id(b)?) {
                problems.push(format!("CAI boundary {} of {} is not in its boundary list", nd::item_id(b)?, spec.key));
            }
        }
        for b in &listed {
            match by_boundary.get(b) {
                Some((x, y)) if *x == rid || *y == rid => {}
                _ => problems.push(format!("CAI region {} lists boundary {b} that does not touch it", spec.key)),
            }
        }
        for nb in &spec.neighbours {
            let Some((nid, _)) = cai_region_of.iter().find(|(_, k)| *k == nb) else { continue };
            if !by_boundary.values().any(|(x, y)| (*x == rid && y == nid) || (*y == rid && x == nid)) {
                problems.push(format!("CAI region {} has no boundary with its neighbour {nb}", spec.key));
            }
        }
    }
    for it in nd::get_items(cw, &["CAI_WORLD_SETTLEMENTS"])? {
        let s = nd::spec_kids(it)?;
        let wid = nd::sc_at(s, 2)?;
        if !settlement_ids.values().any(|v| *v == wid) {
            problems.push(format!("CAI_SETTLEMENT {} points at unknown world settlement {wid}", nd::item_id(it)?));
        }
    }
    let mut slot_users: HashMap<u32, usize> = HashMap::new();
    for it in nd::get_items(cw, &["CAI_WORLD_BUILDING_SLOTS"])? {
        let wid = spec_id(it, 0)?;
        if !slot_ids.contains(&wid) {
            problems.push(format!("CAI_BUILDING_SLOT {} points at unknown world slot {wid}", nd::item_id(it)?));
        }
        *slot_users.entry(wid).or_insert(0) += 1;
    }
    for (id, n) in slot_users {
        if n > 1 {
            problems.push(format!("world slot {id} has {n} CAI_BUILDING_SLOTs"));
        }
    }
    // owned-region list of every CAI faction equals the regions its world faction owns
    let cai_region_by_world: HashMap<u32, u32> = nd::get_items(cw, &["CAI_WORLD_REGIONS"])?
        .iter()
        .filter_map(|it| Some((nd::sc_at(nd::spec_kids(it).ok()?, 11).ok()?, nd::item_id(it).ok()?)))
        .collect();
    for it in nd::get_items(cw, &["CAI_WORLD_FACTIONS"])? {
        let s = nd::spec_kids(it)?;
        let wf = nd::sc_at(s, 6)?;
        let Some(name) = factions.get(&wf) else { continue };
        let mut want: Vec<u32> = owners
            .iter()
            .filter(|(_, o)| **o == wf)
            .filter_map(|(k, _)| cai_region_by_world.get(&region_ids[k]).copied())
            .collect();
        want.sort_unstable();
        let mut have = nd::arr_vals(&s[0]);
        have.sort_unstable();
        if want != have {
            problems.push(format!("CAI faction of {name}: owned regions {have:?} but the world says {want:?}"));
        }
    }
    Ok(())
}

fn ensure_eq(problems: &mut Vec<String>, a: u32, b: u32, what: impl FnOnce() -> String) {
    if a != b {
        problems.push(format!("{}: {a} != {b}", what()));
    }
}

/// CAI boundary lengths (world units as stored in `CAI_WORLD_REGION_BOUNDARIES`) of region `key` with each neighbour. The
/// stored lengths are not the polygon lengths of `regions.esf` (they run 1x-2x larger), so a split should scale them by the
/// polygon-length ratio instead of writing polygon lengths.
pub fn cai_boundary_lengths(root: &Node, key: &str) -> Result<BTreeMap<String, f32>> {
    let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"])?;
    let mut name_of: HashMap<u32, String> = HashMap::new();
    for it in nd::get_items(world, &["CAI_WORLD_REGIONS"])? {
        name_of.insert(nd::item_id(it)?, spec_str(it, 10).unwrap_or("?").to_string());
    }
    let mut out = BTreeMap::new();
    for it in nd::get_items(world, &["CAI_WORLD_REGION_BOUNDARIES"])? {
        let s = nd::spec_kids(it)?;
        let (a, b) = (&name_of[&nd::sc_at(s, 0)?], &name_of[&nd::sc_at(s, 1)?]);
        let len = match s.get(2) {
            Some(Node::Value(Value::F32(l))) => *l,
            other => bail!("boundary length is {other:?}"),
        };
        if a == key {
            *out.entry(b.clone()).or_insert(0.0) += len;
        } else if b == key {
            *out.entry(a.clone()).or_insert(0.0) += len;
        }
    }
    Ok(out)
}

/// Number of CAI regions of the start position: the `regions.esf` index the next added region must have.
pub fn cai_region_count(root: &Node) -> Result<usize> {
    let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"])?;
    Ok(nd::get_items(world, &["CAI_WORLD_REGIONS"])?.len())
}

/// Describe an applied map split for the start position of `root` (the generated startpos with every earlier split
/// already added). CAI boundary lengths are not the polygon lengths of `regions.esf` (see [`cai_boundary_lengths`]):
/// each border keeps the source's CAI length scaled by the share of the polygon border it gets, and the new region's
/// border with the source uses the mean CAI/polygon ratio of the source's borders.
pub fn new_region_for(
    root: &Node,
    split: &crate::model::RegionSplit,
    applied: &crate::regionmap::apply::AppliedSplit,
    rules: &crate::model::Rules,
) -> Result<NewRegion> {
    ensure!(
        applied.key == split.key && applied.source == split.source,
        "applied split {} does not belong to split {}",
        applied.key,
        split.key
    );
    let rule = rules
        .regions
        .get(&split.key)
        .ok_or_else(|| anyhow!("the region rules have no entry for {}", split.key))?;
    let rebel_name = rule
        .get("rebel_name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("the region rules of {} have no rebel_name", split.key))?
        .to_string();
    let owner = rule
        .get("owner")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("the region rules of {} have no owner", split.key))?
        .to_string();
    let cai = cai_boundary_lengths(root, &split.source)?;
    let before = &applied.source_border_lengths_before;
    let after = &applied.source_border_lengths_after;
    let polygon = &applied.border_lengths;
    // mean CAI length / polygon length over the source's borders (1.0 when it has none to learn from)
    let ratios: Vec<f64> = before
        .iter()
        .filter(|(_, l)| **l > 0.0)
        .filter_map(|(k, l)| cai.get(k).map(|c| *c as f64 / l))
        .collect();
    let ratio = if ratios.is_empty() { 1.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
    let mut neighbours = Vec::new();
    let mut border_lengths = Vec::new();
    let mut donor_borders = Vec::new();
    for nb in &applied.neighbours {
        let poly = polygon.get(nb).copied().unwrap_or(0.0);
        if *nb == split.source {
            neighbours.push(nb.clone());
            border_lengths.push((poly * ratio) as f32);
            continue;
        }
        // a neighbour the source has no CAI boundary with (e.g. a sea region) gets none
        let Some(cl) = cai.get(nb).copied() else { continue };
        let b = before.get(nb).copied().unwrap_or(0.0);
        neighbours.push(nb.clone());
        if b > 0.0 {
            border_lengths.push((cl as f64 * poly / b) as f32);
            donor_borders.push((nb.clone(), (cl as f64 * after.get(nb).copied().unwrap_or(0.0) / b) as f32));
        } else {
            border_lengths.push((poly * ratio) as f32);
        }
    }
    ensure!(!neighbours.is_empty(), "split {} has no neighbours with a CAI boundary", split.key);
    Ok(NewRegion {
        key: split.key.clone(),
        like: split.source.clone(),
        settlement_key: applied.settlement_key.clone(),
        slot_keys: applied.slot_keys.clone(),
        position: applied.center,
        slot_positions: applied.slot_positions.clone(),
        neighbours,
        settlement_loc: format!("start_pos_settlements_onscreen_name_{}", applied.settlement_key),
        rebel_name,
        border_lengths,
        donor_borders,
        owner: Some(owner),
        existing_cai: false,
        map_index: Some(applied.new_index),
    })
}

/// Describe an applied activation for the start position of `root` (the generated startpos with every earlier
/// activation already added). The wilderness region keeps its place in the start position's CAI region list
/// (`existing_cai`), which also usually holds its boundaries with the neighbours (their ETW lengths stay); a boundary
/// the start position lacks gets the polygon border length scaled by the mean CAI/polygon ratio of the donor's
/// borders. The donor's own polygon is unchanged, so no donor border shrinks.
pub fn new_region_for_activation(
    root: &Node,
    activation: &crate::model::RegionActivation,
    applied: &crate::regionmap::apply::AppliedActivation,
    rules: &crate::model::Rules,
) -> Result<NewRegion> {
    ensure!(
        applied.key == activation.key && applied.donor == activation.donor,
        "applied activation {} does not belong to activation {}",
        applied.key,
        activation.key
    );
    let rule = rules
        .regions
        .get(&activation.key)
        .ok_or_else(|| anyhow!("the region rules have no entry for {}", activation.key))?;
    let rebel_name = rule
        .get("rebel_name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("the region rules of {} have no rebel_name", activation.key))?
        .to_string();
    let owner = rule
        .get("owner")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("the region rules of {} have no owner", activation.key))?
        .to_string();
    let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"])?;
    let cai_keys: Vec<String> = nd::get_items(world, &["CAI_WORLD_REGIONS"])?
        .iter()
        .map(|it| spec_str(it, 10).unwrap_or("?").to_string())
        .collect();
    let index = cai_keys
        .iter()
        .position(|k| *k == activation.key)
        .ok_or_else(|| anyhow!("the start position has no CAI region {}", activation.key))?;
    ensure!(
        index == applied.index,
        "Activation {}: the map region index is {} but the start position holds the CAI region at index {index}",
        activation.key,
        applied.index
    );
    let cai = cai_boundary_lengths(root, &activation.donor)?;
    let ratios: Vec<f64> = applied
        .donor_border_lengths
        .iter()
        .filter(|(_, l)| **l > 0.0)
        .filter_map(|(k, l)| cai.get(k).map(|c| *c as f64 / l))
        .collect();
    let ratio = if ratios.is_empty() { 1.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
    let mut neighbours = Vec::new();
    let mut border_lengths = Vec::new();
    for nb in &applied.neighbours {
        if !cai_keys.contains(nb) {
            continue;
        }
        neighbours.push(nb.clone());
        border_lengths.push((applied.border_lengths.get(nb).copied().unwrap_or(0.0) * ratio) as f32);
    }
    Ok(NewRegion {
        key: activation.key.clone(),
        like: activation.donor.clone(),
        settlement_key: applied.settlement_key.clone(),
        slot_keys: applied.slot_keys.clone(),
        position: applied.center,
        slot_positions: applied.slot_positions.clone(),
        neighbours,
        settlement_loc: format!("start_pos_settlements_onscreen_name_{}", applied.settlement_key),
        rebel_name,
        border_lengths,
        donor_borders: Vec::new(),
        owner: Some(owner),
        existing_cai: true,
        map_index: Some(applied.index),
    })
}

/// Index the CAI region of `spec` has (activation) or gets (new region) in `CAI_WORLD_REGIONS`.
fn cai_index_for(root: &Node, spec: &NewRegion) -> Result<usize> {
    let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"])?;
    let items = nd::get_items(world, &["CAI_WORLD_REGIONS"])?;
    if spec.existing_cai {
        items
            .iter()
            .position(|it| spec_str(it, 10) == Some(spec.key.as_str()))
            .ok_or_else(|| anyhow!("CAI has no region {} to activate", spec.key))
    } else {
        Ok(items.len())
    }
}

// ------------------------------------------------------------------------------------------------ carved regions

/// Describe an applied activation of a region that has no CAI entry in the start position yet (a region carved out of
/// the `unexplorable` backdrop, [`crate::model::RegionActivation::carve_from`]): like [`new_region_for_activation`], but
/// `existing_cai` is false and the CAI region is appended, so the region must be the last one of `regions.esf`
/// (`applied.index` == CAI region count, asserted by [`add_region`] as well).
///
/// Boundaries: with a playable land neighbour the region gets a CAI boundary with each playable land neighbour that has
/// a CAI entry; an island without one (e.g. Sumatra) gets one with every adjacent sea region that has a CAI entry,
/// like its donor Ceylon (boundary with `indian_ocean`). Lengths are the polygon lengths of `rd` (the map after the
/// activation) scaled by the donor's mean CAI/polygon ratio.
pub fn new_region_for_carve(
    root: &Node,
    rd: &RegionData,
    activation: &crate::model::RegionActivation,
    applied: &crate::regionmap::apply::AppliedActivation,
    rules: &crate::model::Rules,
) -> Result<NewRegion> {
    ensure!(
        applied.key == activation.key && applied.donor == activation.donor,
        "applied activation {} does not belong to activation {}",
        applied.key,
        activation.key
    );
    let rule = rules
        .regions
        .get(&activation.key)
        .ok_or_else(|| anyhow!("the region rules have no entry for {}", activation.key))?;
    let rebel_name = rule
        .get("rebel_name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("the region rules of {} have no rebel_name", activation.key))?
        .to_string();
    let owner = rule
        .get("owner")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("the region rules of {} have no owner", activation.key))?
        .to_string();
    let count = cai_region_count(root)?;
    ensure!(
        count == applied.index,
        "{}: the map region index is {} but the next CAI region index is {count}; the region must be the last in regions.esf",
        activation.key,
        applied.index
    );
    let cai = cai_boundary_lengths(root, &activation.donor)?;
    let ratios: Vec<f64> = applied
        .donor_border_lengths
        .iter()
        .filter(|(_, l)| **l > 0.0)
        .filter_map(|(k, l)| cai.get(k).map(|c| *c as f64 / l))
        .collect();
    let ratio = if ratios.is_empty() { 1.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
    let has_cai = |ri: usize| ri < count;
    let pick = |want: &dyn Fn(usize, &str) -> bool| -> Result<(Vec<String>, Vec<f32>)> {
        let mut neighbours = Vec::new();
        let mut lengths = Vec::new();
        for (nb, len) in &applied.border_lengths {
            let ri = rd.index(nb)?;
            if *len <= 0.0 || !has_cai(ri) || !want(ri, nb) {
                continue;
            }
            neighbours.push(nb.clone());
            lengths.push((*len * ratio) as f32);
        }
        Ok((neighbours, lengths))
    };
    let mut chosen = pick(&|ri, nb| rd.regions[ri].kind == "land" && applied.neighbours.iter().any(|n| n == nb))?;
    if chosen.0.is_empty() {
        chosen = pick(&|ri, _| rd.regions[ri].kind == "sea")?;
    }
    let (neighbours, border_lengths) = chosen;
    ensure!(!neighbours.is_empty(), "{}: no neighbour with a CAI entry", activation.key);
    Ok(NewRegion {
        key: activation.key.clone(),
        like: activation.donor.clone(),
        settlement_key: applied.settlement_key.clone(),
        slot_keys: applied.slot_keys.clone(),
        position: applied.center,
        slot_positions: applied.slot_positions.clone(),
        neighbours,
        settlement_loc: format!("start_pos_settlements_onscreen_name_{}", applied.settlement_key),
        rebel_name,
        border_lengths,
        donor_borders: Vec::new(),
        owner: Some(owner),
        existing_cai: false,
        map_index: Some(applied.index),
    })
}

// ------------------------------------------------------------------------------------------------ boundaries

/// Add the `CAI_REGION_BOUNDARY` items a region gained by activation / merge: one per land region it borders in `rd`
/// (shared polygon border > 0) that has a CAI region and no boundary with `key` yet (e.g. Burma | Bengal). A boundary that
/// exists but is missing from either region's boundary list (child 7) is listed again. New lengths are the polygon
/// border length scaled by the mean CAI/polygon ratio of the region's existing boundaries (1 without any). New ids come
/// from `CAI_CENTRAL_BDI_POOL`. Returns one note per change.
pub fn ensure_boundaries(root: &mut Node, rd: &RegionData, key: &str) -> Result<Vec<String>> {
    let ri = rd.index(key)?;
    let lengths = rd.border_lengths(ri);
    let ci = root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"])?;
    let next = nd::sc_at(nd::kids(nd::get(ci, &["CAI_CENTRAL_BDI_POOL"])?)?, 0)?;
    let mut cid = CaiIds { next, first: next };
    let world = nd::get_mut(ci, &["CAI_WORLD"])?;

    let mut cai_of: HashMap<String, u32> = HashMap::new();
    for it in world_items(world, "CAI_WORLD_REGIONS")? {
        cai_of.insert(spec_str(it, 10).ok_or_else(|| anyhow!("CAI_REGION without key"))?.to_string(), nd::item_id(it)?);
    }
    let me = *cai_of.get(key).ok_or_else(|| anyhow!("CAI has no region {key}"))?;
    let name_of: HashMap<u32, &str> = cai_of.iter().map(|(k, v)| (*v, k.as_str())).collect();
    // (id, a, b) of the boundaries
    let mut bnds: Vec<(u32, u32, u32)> = Vec::new();
    let mut ratios: Vec<f64> = Vec::new();
    for it in world_items(world, "CAI_WORLD_REGION_BOUNDARIES")? {
        let s = nd::spec_kids(it)?;
        let (a, b) = (nd::sc_at(s, 0)?, nd::sc_at(s, 1)?);
        bnds.push((nd::item_id(it)?, a, b));
        let other = if a == me {
            b
        } else if b == me {
            a
        } else {
            continue;
        };
        if let (Some(Node::Value(Value::F32(l))), Some(nb)) = (s.get(2), name_of.get(&other)) {
            if let Some(poly) = rd.index(nb).ok().and_then(|n| lengths.get(&n)).filter(|p| **p > 0.0) {
                ratios.push(*l as f64 / poly);
            }
        }
    }
    let ratio = if ratios.is_empty() { 1.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
    let tmpl = world_items(world, "CAI_WORLD_REGION_BOUNDARIES")?
        .first()
        .ok_or_else(|| anyhow!("no boundary to clone"))?
        .clone();
    let listed = |regions: &Vec<Vec<Node>>, region: u32, id: u32| -> Result<bool> {
        let it = regions
            .iter()
            .find(|it| nd::item_id(it).ok() == Some(region))
            .ok_or_else(|| anyhow!("CAI region {region} vanished"))?;
        Ok(nd::arr_at(nd::spec_kids(it)?, 7)?.contains(&id))
    };

    let mut notes = Vec::new();
    let mut new_items: Vec<Vec<Node>> = Vec::new();
    let mut to_list: Vec<(u32, u32)> = Vec::new(); // (region cai id, boundary id)
    for (nbi, poly) in &lengths {
        let nb = &rd.regions[*nbi];
        if *poly <= 0.0 || nb.kind != "land" {
            continue;
        }
        let Some(&other) = cai_of.get(&nb.key) else { continue };
        if other == me {
            continue;
        }
        let regions = world_items(world, "CAI_WORLD_REGIONS")?;
        match bnds.iter().find(|b| (b.1 == me && b.2 == other) || (b.1 == other && b.2 == me)) {
            Some((id, _, _)) => {
                for r in [me, other] {
                    if !listed(regions, r, *id)? {
                        to_list.push((r, *id));
                        notes.push(format!("CAI_REGION_BOUNDARY {id} ({key} | {}) listed again by CAI region {r}", nb.key));
                    }
                }
            }
            None => {
                let id = cid.take();
                let len = (*poly * ratio) as f32;
                let mut it = tmpl.clone();
                nd::pristine_header(&mut it, Some(id))?;
                let s = nd::spec_kids_mut(&mut it)?;
                let (a, b) = if other < me { (other, me) } else { (me, other) };
                nd::put(s, 0, nd::u32n(a))?;
                nd::put(s, 1, nd::u32n(b))?;
                nd::put(s, 2, nd::f32n(len))?;
                new_items.push(it);
                bnds.push((id, a, b));
                to_list.push((me, id));
                to_list.push((other, id));
                notes.push(format!("CAI_REGION_BOUNDARY {id} added: {key} | {} length {len:.2}", nb.key));
            }
        }
    }
    if notes.is_empty() {
        return Ok(notes);
    }
    world_items_mut(world, "CAI_WORLD_REGION_BOUNDARIES")?.extend(new_items);
    for it in world_items_mut(world, "CAI_WORLD_REGIONS")?.iter_mut() {
        let id = nd::item_id(it)?;
        for (_, b) in to_list.iter().filter(|(r, _)| *r == id) {
            arr_push(&mut nd::spec_kids_mut(it)?[7], *b)?;
        }
    }
    nd::put(nd::kids_mut(nd::get_mut(ci, &["CAI_CENTRAL_BDI_POOL"])?)?, 0, nd::u32n(cid.next))?;
    Ok(notes)
}

// ------------------------------------------------------------------------------------------------ HLCI rebuild

/// One region whose HLCIs were rebuilt.
#[derive(Clone, Debug, Default)]
pub struct HlciRegionReport {
    pub key: String,
    pub cai_id: u32,
    /// The HLCI ids the CAI region listed before / lists now (one per component).
    pub old: Vec<u32>,
    pub new: Vec<u32>,
    /// `(area index, component id)` per area
    pub areas: Vec<(usize, u32)>,
}

#[derive(Clone, Debug, Default)]
pub struct HlciReport {
    pub regions: Vec<HlciRegionReport>,
    /// HLCI ids dropped from the region lists and from `CAI_WORLD_REGION_HLCIS`
    pub removed: Vec<u32>,
    /// HLCI ids cloned with fresh CAI ids
    pub created: Vec<u32>,
    /// `CAI_WORLD_REGION_HLCIS` size afterwards
    pub total: usize,
}


struct ExHlci {
    id: u32,
    comp: u32,
    pt: P,
    area: Option<usize>,
    /// index into `CAI_WORLD_REGION_HLCIS`
    at: usize,
}

fn fixed64(v: f64) -> i32 {
    (v * FIXED).round() as i32
}

fn hlci_layout_matches(
    cai_id: u32,
    areas: &[super::regions::Area],
    listed: &[u32],
    by_id: &HashMap<u32, usize>,
    hlcis: &[Vec<Node>],
) -> Result<bool> {
    let mut covered = vec![false; areas.len()];
    let mut components = BTreeSet::new();
    for id in listed {
        let Some(&at) = by_id.get(id) else { return Ok(false) };
        let s = nd::spec_kids(&hlcis[at])?;
        let component = nd::sc_at(s, 1)?;
        if nd::sc_at(s, 0)? != cai_id || !components.insert(component) {
            return Ok(false);
        }
        let indices = nd::arr_at(s, 2)?;
        if indices.is_empty() {
            return Ok(false);
        }
        for ai in indices {
            let Some(area) = areas.get(ai as usize) else { return Ok(false) };
            if area.adjoining_id as u32 != component || covered[ai as usize] {
                return Ok(false);
            }
            covered[ai as usize] = true;
        }
    }
    Ok(covered.into_iter().all(|v| v))
}

/// Rebuild region HLCIs in the stock shape: `[region CAI id, component id, [area indices], x, y]`,
/// with one record per distinct component and every area covered exactly once.
///
/// Explicit `keys` force reconstruction after geometry edits. Empty `keys` repairs only invalid groups;
/// valid multi-area groups keep their original records and representatives.
/// Settlement representatives and existing component ids are preferred, then an existing point in the group.
/// Missing groups receive fresh CAI ids; surplus records are removed.
pub fn rebuild_region_hlcis(root: &mut Node, rd: &RegionData, keys: &[String]) -> Result<HlciReport> {
    let ci = root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"])?;
    let next = nd::sc_at(nd::kids(nd::get(ci, &["CAI_CENTRAL_BDI_POOL"])?)?, 0)?;
    let mut cid = CaiIds { next, first: next };
    let world = nd::get_mut(ci, &["CAI_WORLD"])?;

    // (item index, key, CAI id, listed HLCI ids) of every CAI region
    let mut regions: Vec<(usize, String, u32, Vec<u32>)> = Vec::new();
    for (i, it) in world_items(world, "CAI_WORLD_REGIONS")?.iter().enumerate() {
        let key = spec_str(it, 10).ok_or_else(|| anyhow!("CAI_REGION without key"))?.to_string();
        regions.push((i, key, nd::item_id(it)?, nd::arr_at(nd::spec_kids(it)?, 1)?));
    }
    for k in keys {
        ensure!(regions.iter().any(|r| &r.1 == k), "CAI has no region {k}");
        ensure!(rd.index(k).is_ok(), "regions.esf has no region {k}");
    }

    let mut report = HlciReport::default();
    let mut removed: BTreeSet<u32> = BTreeSet::new();
    let mut updates: Vec<(usize, Vec<u32>)> = Vec::new();
    {
        let hl = world_items_mut(world, "CAI_WORLD_REGION_HLCIS")?;
        ensure!(!hl.is_empty(), "CAI_WORLD_REGION_HLCIS is empty");
        let mut by_id: HashMap<u32, usize> = HashMap::new();
        for (i, it) in hl.iter().enumerate() {
            by_id.insert(nd::item_id(it)?, i);
        }
        for (item_index, key, cai_id, listed) in &regions {
            if !keys.is_empty() && !keys.contains(key) {
                continue;
            }
            let Ok(ri) = rd.index(key) else { continue };
            let areas = &rd.regions[ri].areas;
            if keys.is_empty() && hlci_layout_matches(*cai_id, areas, listed, &by_id, hl)? {
                continue;
            }
            let sett = settlement_center(&rd.regions[ri]).ok().map(|p| [p[0] as f64, p[1] as f64]);
            let settlement_area = sett.and_then(|p| rd.area_at(ri, p));
            let mut groups: Vec<(u32, Vec<u32>)> = Vec::new();
            let mut group_index = HashMap::new();
            for (ai, area) in areas.iter().enumerate() {
                let comp = area.adjoining_id as u32;
                let gi = *group_index.entry(comp).or_insert_with(|| {
                    groups.push((comp, Vec::new()));
                    groups.len() - 1
                });
                groups[gi].1.push(ai as u32);
            }
            let mut ex: Vec<ExHlci> = Vec::new();
            for id in listed {
                let Some(&at) = by_id.get(id) else { continue };
                let s = nd::spec_kids(&hl[at])?;
                let pt = [fixed_of(&s[3])? as f64 / FIXED, fixed_of(&s[4])? as f64 / FIXED];
                ex.push(ExHlci {
                    id: *id,
                    comp: nd::sc_at(s, 1)?,
                    pt,
                    area: rd.area_at(ri, pt),
                    at,
                });
            }
            let mut used: HashSet<u32> = HashSet::new();
            let mut new_ids: Vec<u32> = Vec::new();
            for (comp, indices) in &groups {
                let in_group = |ai: Option<usize>| ai.is_some_and(|ai| indices.contains(&(ai as u32)));
                let sett_here = sett.filter(|_| in_group(settlement_area));
                let mut pick: Option<usize> = None;
                if let Some(s) = sett_here {
                    pick = ex
                        .iter()
                        .position(|e| !used.contains(&e.id) && (e.pt[0] - s[0]).abs() < 1e-3 && (e.pt[1] - s[1]).abs() < 1e-3);
                }
                if pick.is_none() {
                    pick = ex.iter().position(|e| !used.contains(&e.id) && e.comp == *comp);
                }
                if pick.is_none() {
                    pick = ex.iter().position(|e| !used.contains(&e.id) && in_group(e.area));
                }
                let point = match (sett_here, pick) {
                    (Some(s), _) => s,
                    (None, Some(k)) if in_group(ex[k].area) => ex[k].pt,
                    _ => match indices.iter().copied().find(|&ai| areas[ai as usize].outlines.iter().any(|o| o.outer)) {
                        Some(ai) => rd.pole_of_inaccessibility(ri, ai as usize),
                        None => match pick {
                            Some(k) => ex[k].pt,
                            None => bail!("{key}: component {comp} has no outer ring"),
                        },
                    },
                };
                let (id, at) = match pick {
                    Some(k) => (ex[k].id, ex[k].at),
                    None => match ex.iter().find(|e| !used.contains(&e.id)) {
                        Some(e) => (e.id, e.at),
                        None => {
                            let mut clone = ex.first().map_or_else(|| hl[0].clone(), |e| hl[e.at].clone());
                            let hdr = nd::hdr_index(&clone)?;
                            let id = cid.take();
                            set_id(&mut clone[hdr], id)?;
                            hl.push(clone);
                            by_id.insert(id, hl.len() - 1);
                            report.created.push(id);
                            (id, hl.len() - 1)
                        }
                    },
                };
                used.insert(id);
                let h = nd::spec_kids_mut(&mut hl[at])?;
                set_id(&mut h[0], *cai_id)?;
                set_id(&mut h[1], *comp)?;
                nd::put(h, 2, nd::arrn(indices))?;
                nd::put(h, 3, nd::i32n(fixed64(point[0])))?;
                nd::put(h, 4, nd::i32n(fixed64(point[1])))?;
                new_ids.push(id);
            }
            for e in &ex {
                if !used.contains(&e.id) {
                    removed.insert(e.id);
                }
            }
            report.regions.push(HlciRegionReport {
                key: key.clone(),
                cai_id: *cai_id,
                old: listed.clone(),
                new: new_ids.clone(),
                areas: areas.iter().enumerate().map(|(i, a)| (i, a.adjoining_id as u32)).collect(),
            });
            updates.push((*item_index, new_ids));
        }
        if !removed.is_empty() {
            hl.retain(|it| nd::item_id(it).is_ok_and(|id| !removed.contains(&id)));
        }
        report.total = hl.len();
    }
    if !removed.is_empty() {
        for (index, item) in world_items(world, "CAI_WORLD_TRADE_ROUTE_SEGMENTS")?.iter().enumerate() {
            let reference = nd::sc_at(nd::spec_kids(item)?, 1)?;
            ensure!(
                !removed.contains(&reference) || reference as usize == index,
                "CAI trade-route segment {index} still references removed HLCI {reference}"
            );
        }
    }
    let items = world_items_mut(world, "CAI_WORLD_REGIONS")?;
    for (index, ids) in updates {
        nd::put(nd::spec_kids_mut(&mut items[index])?, 1, nd::arrn(&ids))?;
    }
    if cid.next != cid.first {
        nd::put(nd::kids_mut(nd::get_mut(ci, &["CAI_CENTRAL_BDI_POOL"])?)?, 0, nd::u32n(cid.next))?;
    }
    report.removed = removed.into_iter().collect();
    Ok(report)
}

// ------------------------------------------------------------------------------------------------ map consistency

/// What the start position is checked against besides itself ([`check_world`]).
pub struct MapChecks<'a> {
    pub regions: &'a RegionData,
    /// regions.esf region key -> theatre index (position in `theatres[]`, which is the pathfinding area index)
    pub theatre_of: BTreeMap<String, usize>,
    /// per pathfinding area: the region list (`regions.esf` indices, `Grid::region_ids`)
    pub pathfinding_regions: &'a [Vec<i16>],
    /// number of cells of the sea grid; `None` skips the check
    pub sea_grid_cells: Option<usize>,
}

impl<'a> MapChecks<'a> {
    /// `regions_root`: the `regions.esf` tree `regions` was parsed from (its `theatres` / `theatres_and_region_keys`
    /// say which theatre lists a region); `pathfinding_regions[i]` = region list of pathfinding area `i`.
    pub fn new(
        regions: &'a RegionData,
        regions_root: &Node,
        pathfinding_regions: &'a [Vec<i16>],
        sea_grid_cells: Option<usize>,
    ) -> Result<Self> {
        let mut theatre_keys: Vec<String> = Vec::new();
        for item in regions_root.get("theatres")?.items()? {
            let ch: &[Node] = if let [Node::Record { children, .. }] = item.as_slice() { children } else { item };
            let key = ch
                .iter()
                .find_map(text)
                .ok_or_else(|| anyhow!("theatre without a key"))?
                .to_string();
            theatre_keys.push(key);
        }
        let mut theatre_of = BTreeMap::new();
        for item in regions_root.get("theatres_and_region_keys")?.items()? {
            let entry = &item[0];
            let key = entry
                .children()?
                .iter()
                .find_map(text)
                .ok_or_else(|| anyhow!("theatre entry without a key"))?;
            let idx = theatre_keys
                .iter()
                .position(|k| k == key)
                .ok_or_else(|| anyhow!("theatres_and_region_keys lists theatre {key}, theatres[] does not"))?;
            for it in entry.get("region_keys")?.items()? {
                theatre_of.insert(it[0].as_str()?.to_string(), idx);
            }
        }
        Ok(Self { regions, theatre_of, pathfinding_regions, sea_grid_cells })
    }
}

/// Number of `name` nodes (records, or items of record arrays) below `n`.
fn count_named(n: &Node, name: &str) -> usize {
    match n {
        Node::Record { name: nm, children, .. } => {
            if nm == name {
                1
            } else {
                children.iter().map(|c| count_named(c, name)).sum()
            }
        }
        Node::Array { name: nm, items, .. } => {
            if nm == name {
                items.len()
            } else {
                items.iter().flatten().map(|c| count_named(c, name)).sum()
            }
        }
        _ => 0,
    }
}

/// Consistency of the start position `root` with the final map: what the engine rejects or crashes on and `check`
/// (which compares against the vanilla start position) cannot see.
///
/// * `CAI_WORLD_REGIONS[i]` is the region `regions.esf` has at index `i` (and both lists have the same length);
/// * every component of every region has one HLCI containing all its area indices; each area is covered exactly
///   once, each HLCI belongs to its region, and no HLCI is listed by two regions or by none;
/// * `CAI_SEA_GRID_CELL` count == `sea_grid_cells`;
/// * every settled region is in the pathfinding region list of the theatre that lists it.
pub fn check_world(root: &Node, map: &MapChecks<'_>) -> Result<Vec<String>> {
    let mut problems = Vec::new();
    let rd = map.regions;
    let ci = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"])?;
    let world = nd::get(ci, &["CAI_WORLD"])?;
    let cai_regions = nd::get_items(world, &["CAI_WORLD_REGIONS"])?;
    if cai_regions.len() != rd.regions.len() {
        problems.push(format!(
            "CAI_WORLD_REGIONS has {} regions, regions.esf has {}",
            cai_regions.len(),
            rd.regions.len()
        ));
    }
    let mut hlci_at: HashMap<u32, &Vec<Node>> = HashMap::new();
    for it in nd::get_items(world, &["CAI_WORLD_REGION_HLCIS"])? {
        let id = nd::item_id(it)?;
        if hlci_at.insert(id, it).is_some() {
            problems.push(format!("CAI_REGION_HLCI id {id} exists twice"));
        }
    }
    let mut listed_by: HashMap<u32, u32> = HashMap::new();
    for (i, it) in cai_regions.iter().enumerate() {
        let key = spec_str(it, 10).unwrap_or("?");
        let cai_id = nd::item_id(it)?;
        match rd.regions.get(i) {
            Some(r) if r.key == key => {}
            Some(r) => problems.push(format!("CAI region {i} is {key}, regions.esf region {i} is {}", r.key)),
            None => problems.push(format!("CAI region {i} ({key}) has no regions.esf region")),
        }
        // by key, so one misplaced region does not hide the HLCI findings of the others
        let Ok(ri) = rd.index(key) else {
            problems.push(format!("CAI region {key} is not in regions.esf"));
            continue;
        };
        let areas = &rd.regions[ri].areas;
        let listed = nd::arr_at(nd::spec_kids(it)?, 1)?;
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut components = BTreeSet::new();
        for id in &listed {
            if let Some(other) = listed_by.insert(*id, cai_id) {
                problems.push(format!("{key}: HLCI {id} is also listed by CAI region {other}"));
            }
            let Some(h) = hlci_at.get(id) else {
                problems.push(format!("{key}: listed HLCI {id} does not exist"));
                continue;
            };
            let s = nd::spec_kids(h)?;
            if nd::sc_at(s, 0)? != cai_id {
                problems.push(format!("{key}: HLCI {id} belongs to CAI region {}", nd::sc_at(s, 0)?));
            }
            let comp = nd::sc_at(s, 1)?;
            if !components.insert(comp) {
                problems.push(format!("{key}: multiple HLCIs for component {comp}"));
            }
            let indices = nd::arr_at(s, 2)?;
            if indices.is_empty() {
                problems.push(format!("{key}: HLCI {id} covers no areas"));
            }
            for ai in indices {
                let Some(area) = areas.get(ai as usize) else {
                    problems.push(format!("{key}: HLCI {id} names area {ai}, the region has {}", areas.len()));
                    continue;
                };
                if !seen.insert(ai) {
                    problems.push(format!("{key}: area {ai} is covered more than once"));
                }
                if comp != area.adjoining_id as u32 {
                    problems.push(format!(
                        "{key}: HLCI {id} component {comp} but area {ai} has component {}",
                        area.adjoining_id
                    ));
                }
            }
        }
        for ai in 0..areas.len() {
            if !seen.contains(&(ai as u32)) {
                problems.push(format!("{key}: area {ai} has no HLCI"));
            }
        }
    }
    for (id, h) in &hlci_at {
        if !listed_by.contains_key(id) {
            problems.push(format!("HLCI {id} (region {}) is listed by no CAI region", nd::sc_at(nd::spec_kids(h)?, 0)?));
        }
    }
    if let Some(want) = map.sea_grid_cells {
        let have = count_named(ci, "CAI_SEA_GRID_CELL");
        if have != want {
            problems.push(format!("CAI_SEA_GRID_CELL count {have}, the sea grid has {want} cells"));
        }
    }
    for (ri, r) in rd.regions.iter().enumerate() {
        if r.settlement.is_none() {
            continue;
        }
        let Some(theatre) = map.theatre_of.get(&r.key) else {
            problems.push(format!("settled region {} is listed by no theatre", r.key));
            continue;
        };
        match map.pathfinding_regions.get(*theatre) {
            Some(list) if list.contains(&(ri as i16)) => {}
            Some(_) => problems.push(format!(
                "settled region {} (index {ri}) is missing from the pathfinding region list of theatre {theatre}",
                r.key
            )),
            None => problems.push(format!("settled region {}: pathfinding.esf has no area {theatre}", r.key)),
        }
    }
    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXP5: &str = "F:/NTW2/work/exp5";

    fn load(path: &str) -> Option<esf::Esf> {
        let bytes = std::fs::read(path).ok()?;
        Some(esf::read(&bytes).unwrap_or_else(|e| panic!("{path}: {e:#}")))
    }

    fn cai_regions(root: &Node) -> Vec<(String, Vec<u32>)> {
        let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"]).unwrap();
        nd::get_items(world, &["CAI_WORLD_REGIONS"])
            .unwrap()
            .iter()
            .map(|it| (spec_str(it, 10).unwrap().to_string(), nd::arr_at(nd::spec_kids(it).unwrap(), 1).unwrap()))
            .collect()
    }


    fn hlci_fields(root: &Node, id: u32) -> (u32, Vec<u32>, i32, i32) {
        let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"]).unwrap();
        let it = nd::get_items(world, &["CAI_WORLD_REGION_HLCIS"])
            .unwrap()
            .iter()
            .find(|it| nd::item_id(it).unwrap() == id)
            .unwrap();
        let s = nd::spec_kids(it).unwrap();
        (nd::sc_at(s, 1).unwrap(), nd::arr_at(s, 2).unwrap(), fixed_of(&s[3]).unwrap(), fixed_of(&s[4]).unwrap())
    }

    fn hlci_fixture(components: &[u16], groups: &[(u32, &[u32])]) -> (esf::Esf, RegionData) {
        use super::super::regions::{Area, Outline, Region};
        let mut rd = RegionData { vertices: Vec::new(), regions: Vec::new(), theatres: Vec::new() };
        let mut areas = Vec::new();
        for (ai, &component) in components.iter().enumerate() {
            let base = rd.vertices.len() as u32;
            let x = ai as f32 * 20.0;
            rd.vertices.extend([[x, 0.0], [x + 10.0, 0.0], [x + 10.0, 10.0], [x, 10.0]]);
            areas.push(Area {
                land_bridge: false, channel_coast: false, passable: true, adjoining_id: component,
                faces: vec![[base, base + 1, base + 2], [base, base + 2, base + 3]],
                outlines: vec![Outline { outer: true, ring: (base..base + 4).collect(), runs: Vec::new() }],
                impassable_id: u16::MAX, island_id: 0,
            });
        }
        rd.regions.push(Region { key: "sea".into(), kind: "sea".into(), areas, tail: -1, settlement: None });
        let mut region = vec![nd::u32n(0); 11];
        let ids: Vec<u32> = (100..100 + groups.len() as u32).collect();
        region[1] = nd::arrn(&ids);
        region[10] = Node::Value(Value::String16("sea".into()));
        let hlcis = groups.iter().enumerate().map(|(i, &(component, indices))| {
            let x = indices.first().copied().unwrap_or(0) as f64 * 20.0 + 5.0;
            vec![nd::u32n(ids[i]), Node::record("CAI_REGION_HLCI", 0, vec![
                nd::u32n(50), nd::u32n(component), nd::arrn(indices),
                nd::i32n(fixed64(x)), nd::i32n(fixed64(5.0)),
            ])]
        }).collect();
        let world = Node::record("CAI_WORLD", 0, vec![
            Node::array("CAI_WORLD_REGIONS", 0, vec![vec![nd::u32n(50), Node::record("CAI_REGION", 0, region)]]),
            Node::array("CAI_WORLD_REGION_HLCIS", 0, hlcis),
            Node::array("CAI_WORLD_TRADE_ROUTE_SEGMENTS", 0, Vec::new()),
        ]);
        let ci = Node::record("CAI_INTERFACE", 0, vec![
            world, Node::record("CAI_CENTRAL_BDI_POOL", 0, vec![nd::u32n(1000)]),
        ]);
        let root = Node::record("ROOT", 0, vec![Node::record("CAMPAIGN_ENV", 0, vec![
            Node::record("CAMPAIGN_MODEL", 0, vec![ci]),
        ])]);
        (esf::Esf { timestamp: 1, names: Vec::new(), root }, rd)
    }

    fn hlci_shape(root: &Node) -> Vec<(u32, Vec<u32>)> {
        cai_regions(root)[0].1.iter().map(|&id| {
            let (component, indices, _, _) = hlci_fields(root, id);
            (component, indices)
        }).collect()
    }

    fn hlci_problems(root: &Node, rd: &RegionData) -> Vec<String> {
        check_world(root, &MapChecks {
            regions: rd, theatre_of: BTreeMap::new(), pathfinding_regions: &[], sea_grid_cells: None,
        }).unwrap()
    }

    #[test]
    fn valid_multi_area_hlcis_are_preserved_byte_for_byte() {
        let (mut file, rd) = hlci_fixture(&[31, 31, 110], &[(31, &[0, 1]), (110, &[2])]);
        let before = esf::write(&file).unwrap();
        rebuild_region_hlcis(&mut file.root, &rd, &[]).unwrap();
        assert_eq!(esf::write(&file).unwrap(), before);
        assert_eq!(hlci_problems(&file.root, &rd), Vec::<String>::new());
    }

    #[test]
    fn newly_passable_sea_areas_share_one_existing_component_hlci() {
        let (mut file, rd) = hlci_fixture(
            &[31, 31, 31, 31, 31], &[(31, &[0, 1]), (110, &[2]), (111, &[3]), (112, &[4])],
        );
        let report = rebuild_region_hlcis(&mut file.root, &rd, &[]).unwrap();
        assert_eq!(hlci_shape(&file.root), vec![(31, vec![0, 1, 2, 3, 4])]);
        assert_eq!(hlci_fields(&file.root, 100), (31, vec![0, 1, 2, 3, 4], fixed64(5.0), fixed64(5.0)));
        assert_eq!(report.removed, vec![101, 102, 103]);
        assert_eq!(report.created, Vec::<u32>::new());
        let bytes = esf::write(&file).unwrap();
        let back = esf::read(&bytes).unwrap();
        assert_eq!(hlci_shape(&back.root), vec![(31, vec![0, 1, 2, 3, 4])]);
        assert_eq!(hlci_problems(&back.root, &rd), Vec::<String>::new());
        rebuild_region_hlcis(&mut file.root, &rd, &[]).unwrap();
        assert_eq!(esf::write(&file).unwrap(), bytes);
    }

    #[test]
    fn separated_components_allocate_distinct_hlcis_and_advance_the_pool() {
        let (mut file, rd) = hlci_fixture(&[31, 110, 111], &[(31, &[0, 1, 2])]);
        let report = rebuild_region_hlcis(&mut file.root, &rd, &[]).unwrap();
        assert_eq!(hlci_shape(&file.root), vec![(31, vec![0]), (110, vec![1]), (111, vec![2])]);
        assert_eq!(report.created, vec![1000, 1001]);
        let pool = file.root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_CENTRAL_BDI_POOL"]).unwrap();
        assert_eq!(nd::sc_at(nd::kids(pool).unwrap(), 0).unwrap(), 1002);
        for id in cai_regions(&file.root)[0].1.iter().copied() {
            let (_, indices, x, y) = hlci_fields(&file.root, id);
            assert_eq!(rd.area_at(0, [x as f64 / FIXED, y as f64 / FIXED]), Some(indices[0] as usize));
        }
        assert_eq!(hlci_problems(&file.root, &rd), Vec::<String>::new());
    }

    #[test]
    fn equal_hlci_and_area_counts_do_not_hide_stale_indices() {
        let (mut file, rd) = hlci_fixture(&[31, 110, 111], &[(31, &[0]), (110, &[1]), (111, &[99])]);
        assert_eq!(hlci_problems(&file.root, &rd).len(), 2);
        rebuild_region_hlcis(&mut file.root, &rd, &[]).unwrap();
        assert_eq!(hlci_shape(&file.root), vec![(31, vec![0]), (110, vec![1]), (111, vec![2])]);
        assert_eq!(hlci_problems(&file.root, &rd), Vec::<String>::new());
    }

    #[test]
    fn hlci_validation_rejects_duplicate_components_and_invalid_area_coverage() {
        let cases: &[(&[(u32, &[u32])], usize)] = &[
            (&[(31, &[0]), (31, &[1])], 1),
            (&[(31, &[0, 0])], 2),
            (&[(31, &[0, 2])], 2),
            (&[(31, &[0])], 1),
            (&[(110, &[0, 1])], 2),
        ];
        for &(groups, violations) in cases {
            let (file, rd) = hlci_fixture(&[31, 31], groups);
            assert_eq!(hlci_problems(&file.root, &rd).len(), violations, "{groups:?}");
        }
    }

    #[test]
    fn removing_an_hlci_with_a_live_trade_segment_reference_is_rejected() {
        let (mut file, rd) = hlci_fixture(&[31, 31], &[(31, &[0]), (110, &[1])]);
        let world = file.root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"]).unwrap();
        world_items_mut(world, "CAI_WORLD_TRADE_ROUTE_SEGMENTS").unwrap().push(vec![
            nd::u32n(200), Node::record("CAI_TRADE_ROUTE_SEGMENT", 0, vec![nd::u32n(0), nd::u32n(101)]),
        ]);
        assert!(rebuild_region_hlcis(&mut file.root, &rd, &[]).is_err());
    }

    #[test]
    fn hlci_rebuild_reproduces_the_validated_burma_startpos() {
        let dir = format!("{EXP5}/variantB/out/data");
        let (Some(pre), Some(golden), Some(regions)) = (
            load(&format!("{dir}/campaigns/wld_napoleon/startpos.pre_hlci.esf")),
            load(&format!("{dir}/campaigns/wld_napoleon/startpos.esf")),
            load(&format!("{dir}/campaign_maps/nap_world/regions.esf")),
        ) else {
            return;
        };
        let rd = RegionData::parse(&regions.root).unwrap();
        let mut sp = pre;
        let report = rebuild_region_hlcis(&mut sp.root, &rd, &["indochina".to_string()]).unwrap();
        assert_eq!(report.regions.len(), 1);
        let r = &report.regions[0];
        assert_eq!(r.key, "indochina");
        assert_eq!(r.new, [40220, 312, 313, 314]);
        assert_eq!(r.areas, [(0, 857), (1, 290), (2, 291), (3, 292)]);
        assert_eq!(report.removed, [309, 310, 311, 315, 316]);
        assert!(report.created.is_empty());
        // Rangoon point on the settlement area, areas in order
        let rangoon = rd.regions[rd.index("indochina").unwrap()].settlement.as_ref().unwrap().children().unwrap()[0].as_xy().unwrap();
        let (comp, areas, x, y) = hlci_fields(&sp.root, 40220);
        assert_eq!((comp, areas), (857, vec![0]));
        assert_eq!((x, y), (fixed(rangoon[0]), fixed(rangoon[1])));
        for (id, area) in [(312, 1), (313, 2), (314, 3)] {
            assert_eq!(hlci_fields(&sp.root, id).1, vec![area]);
        }
        let bytes = esf::write(&sp).unwrap();
        assert!(bytes == std::fs::read(format!("{dir}/campaigns/wld_napoleon/startpos.esf")).unwrap(), "rebuilt startpos differs from the validated one");
        assert_eq!(esf::write(&golden).unwrap(), bytes);
    }


    #[test]
    fn ensure_boundaries_adds_a_missing_land_border_once() {
        let dir = format!("{EXP5}/variantB/out/data");
        let (Some(mut sp), Some(regions)) = (
            load(&format!("{dir}/campaigns/wld_napoleon/startpos.esf")),
            load(&format!("{dir}/campaign_maps/nap_world/regions.esf")),
        ) else {
            return;
        };
        let rd = RegionData::parse(&regions.root).unwrap();
        // the validated start position has the Burma | Bengal boundary: nothing to do
        let ids = |root: &Node| -> (u32, u32, Vec<u32>) {
            let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"]).unwrap();
            let find = |k: &str| {
                nd::get_items(world, &["CAI_WORLD_REGIONS"]).unwrap().iter().find(|it| spec_str(it, 10) == Some(k)).unwrap().clone()
            };
            let (a, b) = (find("indochina"), find("bengal"));
            let (ai, bi) = (nd::item_id(&a).unwrap(), nd::item_id(&b).unwrap());
            let shared: Vec<u32> = nd::get_items(world, &["CAI_WORLD_REGION_BOUNDARIES"])
                .unwrap()
                .iter()
                .filter(|it| {
                    let s = nd::spec_kids(it).unwrap();
                    let (x, y) = (nd::sc_at(s, 0).unwrap(), nd::sc_at(s, 1).unwrap());
                    (x == ai && y == bi) || (x == bi && y == ai)
                })
                .map(|it| nd::item_id(it).unwrap())
                .collect();
            (ai, bi, shared)
        };
        let (burma, bengal, shared) = ids(&sp.root);
        assert_eq!(shared.len(), 1, "the validated start position holds Burma | Bengal");
        // other land borders the validated start position lacks may be added here; they do not matter below
        ensure_boundaries(&mut sp.root, &rd, "indochina").unwrap();
        let pool_next = |root: &Node| {
            let ci = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"]).unwrap();
            nd::sc_at(nd::kids(nd::get(ci, &["CAI_CENTRAL_BDI_POOL"]).unwrap()).unwrap(), 0).unwrap()
        };
        let boundary_count = |root: &Node| {
            let world = root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"]).unwrap();
            nd::get_items(world, &["CAI_WORLD_REGION_BOUNDARIES"]).unwrap().len()
        };
        // expected length: polygon border length x the mean CAI/polygon ratio of Burma's other boundaries
        let polygon = rd.border_lengths(rd.index("indochina").unwrap());
        let bengal_poly = polygon[&rd.index("bengal").unwrap()];
        let ratios: Vec<f64> = cai_boundary_lengths(&sp.root, "indochina")
            .unwrap()
            .iter()
            .filter(|(k, _)| k.as_str() != "bengal")
            .filter_map(|(k, l)| polygon.get(&rd.index(k).unwrap()).filter(|p| **p > 0.0).map(|p| *l as f64 / p))
            .collect();
        let ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let want_len = (bengal_poly * ratio) as f32;
        // drop the boundary everywhere, then ensure restores exactly that one
        let drop_id = shared[0];
        {
            let world = nd::get_mut(sp.root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"]).unwrap(), &["CAI_WORLD"]).unwrap();
            world_items_mut(world, "CAI_WORLD_REGION_BOUNDARIES").unwrap().retain(|it| nd::item_id(it).unwrap() != drop_id);
            for it in world_items_mut(world, "CAI_WORLD_REGIONS").unwrap().iter_mut() {
                let id = nd::item_id(it).unwrap();
                if id == burma || id == bengal {
                    arr_remove(&mut nd::spec_kids_mut(it).unwrap()[7], drop_id).unwrap();
                }
            }
        }
        let (pool_before, count_before) = (pool_next(&sp.root), boundary_count(&sp.root));
        let notes = ensure_boundaries(&mut sp.root, &rd, "indochina").unwrap();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("bengal"), "{notes:?}");
        // the new id is the pool's next id; the pool advanced past it
        let (_, _, now) = ids(&sp.root);
        assert_eq!(now, [pool_before]);
        assert_eq!(pool_next(&sp.root), pool_before + 1);
        assert_eq!(boundary_count(&sp.root), count_before + 1);
        let back = esf::read(&esf::write(&sp).unwrap()).unwrap();
        let world = back.root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"]).unwrap();
        // the record: ascending CAI ids, the expected length
        let rec = nd::get_items(world, &["CAI_WORLD_REGION_BOUNDARIES"]).unwrap().iter().find(|it| nd::item_id(it).unwrap() == pool_before).unwrap();
        let s = nd::spec_kids(rec).unwrap();
        assert_eq!((nd::sc_at(s, 0).unwrap(), nd::sc_at(s, 1).unwrap()), (burma.min(bengal), burma.max(bengal)));
        let Some(Node::Value(Value::F32(len))) = s.get(2) else { panic!("boundary length is {:?}", s.get(2)) };
        assert!((len - want_len).abs() <= want_len * 1e-4, "length {len}, expected {want_len}");
        // both regions list it exactly once
        for k in ["indochina", "bengal"] {
            let it = nd::get_items(world, &["CAI_WORLD_REGIONS"]).unwrap().iter().find(|it| spec_str(it, 10) == Some(k)).unwrap();
            let list = nd::arr_at(nd::spec_kids(it).unwrap(), 7).unwrap();
            assert_eq!(list.iter().filter(|b| **b == pool_before).count(), 1, "{k} lists the new boundary once");
        }
        // idempotent: a second call changes nothing
        assert!(ensure_boundaries(&mut sp.root, &rd, "indochina").unwrap().is_empty());
        assert_eq!((pool_next(&sp.root), boundary_count(&sp.root)), (pool_before + 1, count_before + 1));
        // a boundary that exists but is missing from Bengal's list is listed again, without a new record
        {
            let world = nd::get_mut(sp.root.path_mut(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE"]).unwrap(), &["CAI_WORLD"]).unwrap();
            for it in world_items_mut(world, "CAI_WORLD_REGIONS").unwrap().iter_mut() {
                if nd::item_id(it).unwrap() == bengal {
                    arr_remove(&mut nd::spec_kids_mut(it).unwrap()[7], pool_before).unwrap();
                }
            }
        }
        let relisted = ensure_boundaries(&mut sp.root, &rd, "indochina").unwrap();
        assert_eq!(relisted.len(), 1, "{relisted:?}");
        assert_eq!((pool_next(&sp.root), boundary_count(&sp.root)), (pool_before + 1, count_before + 1));
        assert!(ensure_boundaries(&mut sp.root, &rd, "indochina").unwrap().is_empty());
    }

    #[test]
    fn add_region_checks_the_regions_esf_index() {
        let Some(mut sp) = load(&format!("{EXP5}/variantB/out/data/campaigns/wld_napoleon/startpos.esf")) else { return };
        let at = cai_region_count(&sp.root).unwrap();
        let spec = NewRegion {
            key: "nowhere".into(),
            like: "bengal".into(),
            settlement_key: "settlement:nowhere:x".into(),
            slot_keys: vec!["settlement:nowhere:x:settlement_road:1".into()],
            slot_positions: vec![[0.0, 0.0]],
            neighbours: vec!["bengal".into()],
            settlement_loc: "loc".into(),
            rebel_name: "Rebels".into(),
            map_index: Some(at + 1),
            ..NewRegion::default()
        };
        let err = add_region(&mut sp.root, &spec).unwrap_err().to_string();
        assert!(err.contains("regions.esf index"), "{err}");
    }
}
