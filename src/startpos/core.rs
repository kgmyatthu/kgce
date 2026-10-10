//! Build orchestration of `campaigns/wld_napoleon/startpos.esf` (port of `startpos/build.py` and `regions_b.py`) and the
//! consistency / reference checks (`check_refs.py`) that run before anything is written.
//!
//! Skeleton = NTW `eur_napoleon` startpos; regions / factions / forces / trade / pathfinder come from the ETW `main` startpos
//! converted with the curated rules (post `Rules::apply_project`, so user ownership edits are already in `rules.regions` /
//! `rules.factions`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};

use crate::db::CampaignTable;
use crate::esf::{self, Node};
use crate::map::MapArtifacts;
use crate::model::{Progress, Rules};
use crate::snapshot::{Game, Snapshot};

use super::cai;
use super::context::{BuildContext, FactionInfo, RegionInfo};
use super::factions::{self, StartposStructure, Structure};
use super::forces::ForceBuilder;
use super::helpers::*;

pub const CAMPAIGN_KEY: &str = "wld_napoleon";
pub const MAP_NAME: &str = "nap_world";
const CAMERA_MARGIN: f64 = 60.0;
const ETW_STARTPOS: &str = "campaigns/main/startpos.esf";
const NTW_STARTPOS: &str = "campaigns/eur_napoleon/startpos.esf";

pub struct Generated {
    pub startpos: Vec<u8>,
    pub lua: String,
    /// relocations and other facts worth reporting
    pub notes: Vec<String>,
}

// ------------------------------------------------------------------------------------------------ regions

#[derive(Default, Debug)]
struct SlotStats {
    buildings: usize,
    dropped: usize,
}

fn slot_keys_for(map: &MapArtifacts, region: &str) -> Result<HashMap<String, Option<String>>> {
    let regs = map.slot_mapping.get("regions").unwrap_or(&map.slot_mapping);
    let r = regs
        .get(region)
        .ok_or_else(|| anyhow!("slot mapping has no region {region}"))?;
    let slots = r
        .get("slots")
        .and_then(|s| s.as_array())
        .ok_or_else(|| anyhow!("slot mapping of {region} has no slots"))?;
    let mut out = HashMap::new();
    for s in slots {
        let key = if s.get("dropped").and_then(Json::as_bool).unwrap_or(false) {
            None
        } else {
            Some(jstr(s, "key")?.to_string())
        };
        out.insert(jstr(s, "etw_key")?.to_string(), key);
    }
    Ok(out)
}

struct SlotRules<'a> {
    slot_types: &'a Json,
    buildings: &'a Json,
    by_slot: &'a Json,
    keys: HashMap<String, Option<String>>,
    owner_key: &'a str,
    owner_id: u32,
    owner_gov: &'a str,
}

fn nonempty(v: Option<&Json>) -> Option<String> {
    v.and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// convert one REGION_SLOT in place; dropped slots record their id
fn convert_slot(
    sl: &mut Node,
    is_fort: bool,
    rules: &SlotRules,
    stats: &mut SlotStats,
    dropped: &mut Vec<u32>,
) -> Result<()> {
    let (id, name, name_idx) = {
        let ch = kids(sl)?;
        let idx = scalar_idx(ch);
        if idx.len() < 2 {
            bail!("region slot has fewer than two scalars");
        }
        (id_at(ch, idx[0])?, text_at(ch, idx[1])?.to_string(), idx[1])
    };
    let Some(key) = rules
        .keys
        .get(&name)
        .ok_or_else(|| anyhow!("slot {name} is not in the slot mapping"))?
    else {
        dropped.push(id);
        return Ok(());
    };
    kids_mut(sl)?[name_idx] = s16(key.clone());
    kids_mut(get_mut(
        sl,
        &["SIEGEABLE_GARRISON_RESIDENCE", "GARRISON_RESIDENCE"],
    )?)?[0] = u32v(rules.owner_id);
    let bm = kids_mut(get_mut(sl, &["BUILDING_MANAGER"])?)?;
    if is_fort {
        *bm = vec![b(false), b(false)];
        return Ok(());
    }
    let st_name = rules.slot_types.get(&name).and_then(|v| v.as_str());
    for bi in 0..bm.len() {
        if !is_record(&bm[bi], "BUILDING") {
            continue;
        }
        let old_key = text_at(kids(&bm[bi])?, 1)?.to_string();
        let nk = nonempty(
            rules
                .by_slot
                .get(&old_key)
                .and_then(|m| st_name.and_then(|s| m.get(s))),
        )
        .or_else(|| nonempty(rules.buildings.get(&old_key)));
        match nk {
            None => {
                stats.dropped += 1;
                *bm = vec![b(false), b(false)];
                break;
            }
            Some(nk) => {
                let bv = kids_mut(&mut bm[bi])?;
                bv[1] = s16(nk);
                bv[2] = s16(rules.owner_key);
                bv[3] = s16(rules.owner_gov);
                stats.buildings += 1;
            }
        }
    }
    Ok(())
}

fn convert_region(
    r_etw: &Node,
    rules: &Rules,
    map: &MapArtifacts,
    owner_key: &str,
    owner_id: u32,
    owner_gov: &str,
    rinfo: &Json,
) -> Result<(Node, SlotStats, Vec<u32>)> {
    let mut r = r_etw.clone();
    match &mut r {
        Node::Record { version, .. } => *version = 5,
        _ => bail!("REGION is not a record"),
    }
    let region_key = text_at(kids(&r)?, 0)?.to_string();
    {
        let ch = kids_mut(&mut r)?;
        expect_code(ch, 18, 1, &region_key)?;
        ch.insert(18, i32v(0)); // v4 -> v5: one extra i32 before the bool
        expect_code(ch, 20, 8, &region_key)?;
        ch[20] = u32v(owner_id);
        expect_code(ch, 23, 14, &region_key)?;
        expect_code(ch, 24, 14, &region_key)?;
        ch[23] = s16(jstr(rinfo, "theatre_name")?);
        ch[24] = s16(jstr(rinfo, "rebel_faction")?);
        ch[25] = s16(jstr(rinfo, "rebel_name")?);
        ch[26] = s16(jstr(rinfo, "sc_culture")?);
    }
    let res: Vec<Vec<Node>> = rinfo
        .get("unit_resources")
        .and_then(|u| u.as_array())
        .ok_or_else(|| anyhow!("{region_key}: no unit_resources"))?
        .iter()
        .map(|k| {
            k.as_str()
                .map(|s| vec![s16(s)])
                .ok_or_else(|| anyhow!("{region_key}: unit resource is not a string"))
        })
        .collect::<Result<_>>()?;
    *items_mut(get_mut(&mut r, &["RESOURCES_ARRAY"])?)? = res;
    items_mut(get_mut(&mut r, &["FORT_ARRAY"])?)?.clear(); // forts do not exist in NTW
    {
        let rb = items_mut(get_mut(
            &mut r,
            &["POPULATION", "REGION_FACTORS", "RELIGION_BREAKDOWN"],
        )?)?;
        let nb = rinfo.get("religion_breakdown");
        let mut out = Vec::new();
        for it in rb.iter() {
            let k = as_text(&it[0])
                .ok_or_else(|| anyhow!("{region_key}: religion key"))?
                .to_string();
            let old = as_number(&it[1]).unwrap_or(0.0);
            let v = nb
                .and_then(|m| m.get(&k))
                .and_then(|x| x.as_f64())
                .unwrap_or(old);
            out.push(vec![s16(k), f32v(v as f32)]);
        }
        *rb = out;
    }

    // slots
    let slot_rules = SlotRules {
        slot_types: &rules.slot_types,
        buildings: &rules.buildings,
        by_slot: &rules.buildings_by_slot_type,
        keys: slot_keys_for(map, &region_key)?,
        owner_key,
        owner_id,
        owner_gov,
    };
    let mut stats = SlotStats::default();
    let mut dropped: Vec<u32> = Vec::new();
    {
        let arr = items_mut(get_mut(
            &mut r,
            &["REGION_SLOT_MANAGER", "REGION_SLOT_ARRAY"],
        )?)?;
        for it in arr.iter_mut() {
            convert_slot(&mut it[0], false, &slot_rules, &mut stats, &mut dropped)
                .with_context(|| format!("region {region_key}"))?;
        }
    }
    convert_slot(
        get_mut(&mut r, &["REGION_SLOT_MANAGER", "ROAD_SLOT", "REGION_SLOT"])?,
        false,
        &slot_rules,
        &mut stats,
        &mut dropped,
    )
    .with_context(|| format!("region {region_key} road slot"))?;
    convert_slot(
        get_mut(
            &mut r,
            &["REGION_SLOT_MANAGER", "FORTIFICATION_SLOT", "REGION_SLOT"],
        )?,
        true,
        &slot_rules,
        &mut stats,
        &mut dropped,
    )
    .with_context(|| format!("region {region_key} fortification slot"))?;
    // settlement garrison residence owner
    kids_mut(get_mut(
        &mut r,
        &["SETTLEMENT", "SIEGEABLE_GARRISON_RESIDENCE"],
    )?)?
    .first_mut()
    .map(|g| kids_mut(g).map(|k| k[0] = u32v(owner_id)))
    .ok_or_else(|| anyhow!("{region_key}: settlement garrison residence is empty"))??;
    // drop the removed slots from the slot array
    {
        let arr = items_mut(get_mut(
            &mut r,
            &["REGION_SLOT_MANAGER", "REGION_SLOT_ARRAY"],
        )?)?;
        let mut kept = Vec::new();
        for it in std::mem::take(arr) {
            let first = scalar_idx(kids(&it[0])?)
                .first()
                .copied()
                .ok_or_else(|| anyhow!("slot without id"))?;
            if !dropped.contains(&id_at(kids(&it[0])?, first)?) {
                kept.push(it);
            }
        }
        *arr = kept;
    }
    Ok((r, stats, dropped))
}

fn build_regions(ctx: &mut BuildContext, gov: &BTreeMap<String, String>) -> Result<()> {
    let etw_model = ctx.etw_model.clone();
    let rules = ctx.rules;
    let map = ctx.map;
    let mut total = SlotStats::default();
    for it in items(get(
        &etw_model,
        &["WORLD", "REGION_MANAGER", "REGIONS_ARRAY"],
    )?)? {
        let r0 = &it[0];
        let key = text_at(kids(r0)?, 0)?.to_string();
        let info = rules
            .regions
            .get(&key)
            .ok_or_else(|| anyhow!("region {key} of the ETW world is not in the region rules"))?;
        let owner = jstr(info, "owner")?.to_string();
        let owner_id = *ctx
            .fid
            .get(&owner)
            .ok_or_else(|| anyhow!("region {key}: owner {owner} is not a faction"))?;
        let owner_gov = gov
            .get(&owner)
            .ok_or_else(|| anyhow!("no government for {owner}"))?;
        let (mut r, st, dropped) =
            convert_region(r0, rules, map, &owner, owner_id, owner_gov, info)?;
        ctx.dropped_slots.extend(dropped);
        total.buildings += st.buildings;
        total.dropped += st.dropped;
        // clear stale garrison pointers (army ids from the 1700 setup) in every garrison residence
        let mut problem: Option<String> = None;
        walk_mut(kids_mut(&mut r)?, &mut |n| {
            if is_record(n, "SIEGEABLE_GARRISON_RESIDENCE") {
                if let Node::Record { children: ch, .. } = n {
                    if ch.len() > 14
                        && ncode(&ch[12]) == Some(8)
                        && ncode(&ch[13]) == Some(0x48)
                        && ncode(&ch[14]) == Some(0x48)
                    {
                        ch[12] = u32v(0);
                        ch[13] = arr_u32(&[]);
                        ch[14] = arr_u32(&[]);
                    } else {
                        problem = Some(format!(
                            "{key}: unexpected SIEGEABLE_GARRISON_RESIDENCE layout"
                        ));
                    }
                }
            }
        });
        if let Some(p) = problem {
            bail!(p);
        }
        let sg = kids(get(&r, &["SETTLEMENT", "SIEGEABLE_GARRISON_RESIDENCE"])?)?;
        let pos = (
            as_number(sg.get(10).ok_or_else(|| anyhow!("{key}: settlement x"))?)
                .ok_or_else(|| anyhow!("{key}: settlement x"))? as i32,
            as_number(sg.get(11).ok_or_else(|| anyhow!("{key}: settlement y"))?)
                .ok_or_else(|| anyhow!("{key}: settlement y"))? as i32,
        );
        let sg_id = id_at(sg, 1)?;
        let world_id = nth_id(kids(&r)?, 1)?;
        let index = ctx.regions.len();
        ctx.region_info.insert(
            key.clone(),
            RegionInfo {
                key: key.clone(),
                index,
                owner,
                theatre: jstr(info, "theatre")?.to_string(),
                theatre_name: jstr(info, "theatre_name")?.to_string(),
                pos,
                world_id,
                sg_id,
                garrison: None,
            },
        );
        ctx.region_order.push(key);
        ctx.regions.push(r);
    }
    // split regions are not part of the ETW world, and activated wilderness regions have no REGION record in it:
    // both enter the start position through `regionmap::startpos_split`
    let want = rules
        .regions
        .as_object()
        .map(|o| o.keys().filter(|k| !ctx.added_keys.contains(*k)).count())
        .unwrap_or(0);
    if want != ctx.regions.len() {
        bail!(
            "region rules define {want} regions but the ETW world has {}",
            ctx.regions.len()
        );
    }
    ctx.notes.push(format!(
        "regions converted: {} ({} buildings, {} without NTW equivalent)",
        ctx.regions.len(),
        total.buildings,
        total.dropped
    ));
    Ok(())
}

// ------------------------------------------------------------------------------------------------ trade / pathfinder

fn trade_manager(ctx: &BuildContext) -> Result<Node> {
    let mut tm = get(&ctx.etw_model, &["CAMPAIGN_TRADE_MANAGER"])?.clone();
    for nm in ["DOMESTIC_TRADE_ROUTES", "INTERNATIONAL_TRADE_ROUTES"] {
        let arr = get_mut(&mut tm, &[nm])?;
        let (name, version) = {
            let first = items(arr)?
                .first()
                .and_then(|it| it.get(1))
                .ok_or_else(|| anyhow!("{nm} has no template item"))?;
            match first {
                Node::Array { name, version, .. } => (name.clone(), *version),
                _ => bail!("{nm}: item[1] is not a record array"),
            }
        };
        *items_mut(arr)? = ctx
            .fkeys
            .iter()
            .map(|k| vec![s16(k.clone()), Node::array(name.clone(), version, vec![])])
            .collect();
    }
    items_mut(get_mut(&mut tm, &["THEATRES"])?)?.clear();
    // segments reference (by world handle) the per-faction trade routes that were just emptied: clear both route-handle
    // lists or the loader dereferences NULL
    for it in items_mut(get_mut(&mut tm, &["TRADE_SEGMENTS"])?)?.iter_mut() {
        let idx: Vec<usize> = it
            .iter()
            .enumerate()
            .filter(|(_, c)| is_u32_arr(c))
            .map(|(i, _)| i)
            .collect();
        if idx.len() != 2 {
            bail!(
                "trade segment has {} route-handle lists, expected 2",
                idx.len()
            );
        }
        for i in idx {
            it[i] = arr_u32(&[]);
        }
    }
    Ok(tm)
}

fn pathfinder(ctx: &BuildContext) -> Result<Node> {
    let mut pf = get(&ctx.etw_model, &["CAMPAIGN_PATHFINDER"])?.clone();
    for grid in items_mut(get_mut(&mut pf, &["PATHFINDING_GRID"])?)?.iter_mut() {
        for c in grid.iter_mut() {
            if is_record(c, "OBSTACLE_LISTS") {
                for x in kids_mut(c)?.iter_mut() {
                    if is_array(x, "CHARACTER_OBSTACLE") || is_array(x, "FORT_OBSTACLE") {
                        items_mut(x)?.clear();
                    } else if is_u32_arr(x) {
                        *x = arr_u32(&[]);
                    }
                }
            }
        }
    }
    Ok(pf)
}

// ------------------------------------------------------------------------------------------------ header / setup

fn players(ctx: &BuildContext, template: &[Node]) -> Result<Vec<Vec<Node>>> {
    let mut out = Vec::new();
    for k in &ctx.fkeys {
        let mut it = template.to_vec();
        kids_mut(&mut it[0])?[2] = s16(k.clone());
        factions::set_player_flags(&mut it[0], ctx.factions[k].playable)?;
        out.push(it);
    }
    Ok(out)
}

fn header_and_setup(
    ctx: &BuildContext,
    root: &mut Node,
    etw_root: &Node,
    gm: &GlobalMap,
) -> Result<[f64; 4]> {
    // thumbnails of the ETW theatres
    let etw_maps = items(get(etw_root, &["SAVE_GAME_HEADER", "MAPS"])?)?.clone();
    *items_mut(get_mut(root, &["SAVE_GAME_HEADER", "MAPS"])?)? = etw_maps;

    let ps_t: Vec<Node> = items(get(
        root,
        &[
            "CAMPAIGN_PREOPEN_MAP_INFO",
            "CAMPAIGN_PLAYERS_SETUP",
            "PLAYERS_ARRAY",
        ],
    )?)?
    .first()
    .ok_or_else(|| anyhow!("PREOPEN players array is empty"))?
    .clone();
    let new_players = players(ctx, &ps_t)?;
    {
        let pre = get_mut(root, &["CAMPAIGN_PREOPEN_MAP_INFO"])?;
        *items_mut(get_mut(pre, &["CAMPAIGN_PLAYERS_SETUP", "PLAYERS_ARRAY"])?)? =
            new_players.clone();
        {
            let ch = kids_mut(pre)?;
            let sc = scalar_idx(ch);
            if sc.len() < 2 {
                bail!("PREOPEN_MAP_INFO has fewer than two scalars");
            }
            ch[sc[0]] = s16(CAMPAIGN_KEY);
            ch[sc[1]] = s16(MAP_NAME);
        }
        // victory options: no region lists; only playable factions
        {
            let vo = items_mut(get_mut(pre, &["VICTORY_CONDITION_OPTIONS"])?)?;
            for it in vo.iter_mut() {
                if let Some(Node::Array { items: blocks, .. }) = it.get_mut(1) {
                    for blk in blocks.iter_mut() {
                        for vc in blk.iter_mut() {
                            if let Node::Record { children, .. } = vc {
                                for c in children.iter_mut() {
                                    if is_array(c, "REGION_KEYS") {
                                        items_mut(c)?.clear();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let template = vo
                .first()
                .ok_or_else(|| anyhow!("VICTORY_CONDITION_OPTIONS is empty"))?
                .clone();
            let mut new_vo: Vec<Vec<Node>> = Vec::new();
            for it in std::mem::take(vo) {
                let k = as_text(&it[0]).unwrap_or("");
                if ctx.factions.get(k).map_or(false, |f| f.playable) {
                    new_vo.push(it);
                }
            }
            for k in &ctx.fkeys {
                if ctx.factions[k].playable
                    && !new_vo.iter().any(|it| as_text(&it[0]) == Some(k.as_str()))
                {
                    let mut it = template.clone();
                    it[0] = s16(k.clone());
                    new_vo.push(it);
                }
            }
            *vo = new_vo;
        }
        // faction infos (playable flags, order, region counts)
        let fi = items_mut(get_mut(pre, &["FACTION_INFOS"])?)?;
        let ntw_info: HashMap<String, Vec<Node>> = fi
            .iter()
            .filter_map(|it| as_text(&it[0]).map(|k| (k.to_string(), it.clone())))
            .collect();
        let mut out = Vec::new();
        let mut order = 0;
        for k in &ctx.fkeys {
            let mut it = match ntw_info.get(k) {
                Some(i) => i.clone(),
                None => {
                    // a faction of the target database the target start position has no entry for: the structural
                    // donor's entry carries only the layout; key, flag, portrait and description are the faction's own
                    let donor = &ctx.fmeta[k].donor;
                    let Some(entry) = (donor != k).then(|| ntw_info.get(donor)).flatten() else {
                        bail!(
                            "faction {k} has no FACTION_INFOS entry in the target start position and no structural donor entry to build one from"
                        );
                    };
                    let table = crate::db::load_campaign_table(
                        ctx.snapshot,
                        ctx.rules,
                        CampaignTable::Factions,
                    )?;
                    let identity = factions::identity_from_table(&table, k)
                        .with_context(|| format!("FACTION_INFOS entry of {k}"))?;
                    factions::donor_faction_info(entry, k, &identity)?
                }
            };
            if it.len() < 8 {
                bail!("FACTION_INFOS entry of {k} is too short");
            }
            let playable = ctx.factions[k].playable;
            it[2] = b(playable);
            it[3] = b(playable);
            if playable {
                order += 1;
            }
            it[4] = i32v(if playable { order } else { 0 });
            it[7] = i32v(ctx.fmeta[k].owned.len() as i32);
            out.push(it);
        }
        *fi = out;
        // ownership by theatre
        let ro = items_mut(get_mut(pre, &["REGION_OWNERSHIPS_BY_THEATRE"])?)?;
        let mut items_out = Vec::new();
        for tn in ["america", "europe", "india"] {
            let rows: Vec<Vec<Node>> = ctx
                .region_order
                .iter()
                .filter(|k| ctx.region_info[*k].theatre_name == tn)
                .map(|k| vec![s16(k.clone()), s16(ctx.region_info[k].owner.clone())])
                .collect();
            items_out.push(vec![s16(tn), Node::array("REGION_OWNERSHIPS", 0, rows)]);
        }
        *ro = items_out;
    }
    // environment
    let env = get_mut(root, &["CAMPAIGN_ENV"])?;
    {
        let cs = get_mut(env, &["CAMPAIGN_SETUP"])?;
        kids_mut(cs)?[0] = s16(CAMPAIGN_KEY);
        *items_mut(get_mut(cs, &["CAMPAIGN_PLAYERS_SETUP", "PLAYERS_ARRAY"])?)? = new_players;
    }
    kids_mut(get_mut(
        env,
        &["CAMPAIGN_CAMERA_MANAGER", "CAMPAIGN_CAMERA"],
    )?)?[2] = xyz([-54.0, 0.0, 293.0]);
    // camera target clamp: EPISODIC_RESTRICTIONS children 9..12 = (x_min, y_min, x_max, y_max)
    let bounds = gm.camera_bounds(CAMERA_MARGIN);
    let er = kids_mut(get_mut(env, &["CAMPAIGN_MODEL", "EPISODIC_RESTRICTIONS"])?)?;
    for (i, v) in (9..13).zip(bounds) {
        expect_code(er, i, 10, "EPISODIC_RESTRICTIONS")?;
        er[i] = f32v(v as f32);
    }
    Ok(bounds)
}

// ------------------------------------------------------------------------------------------------ scripting.lua

const LUA: &str = "-- scripting.lua for the 'Grand Campaign' (wld_napoleon): minimal NTW campaign script.
-- Based on eur_napoleon/scripting.lua with all Europe-specific scripted missions/advice removed
-- (they referenced eur_* region keys that do not exist on nap_world).
local scripting = require \"EpisodicScripting\"
local new_game = false

scripting.SetCampaign(\"wld_napoleon\")

local function OnNewCampaignStarted(context)
\tnew_game = true
\tscripting.game_interface:show_shroud(false)
end

local function OnUICreated(context)
\tif context.string == \"Campaign UI\" then
\t\t-- camera bounds (xmin, ymax, xmax, ymin; same argument order as tut_napoleon): re-applied on every UI creation so saves made
\t\t-- with the old Europe-sized bounds (+-640/+-320 stored in EPISODIC_RESTRICTIONS) are corrected on load
\t\tscripting.game_interface:set_map_bounds(%(bx0), %(by1), %(bx1), %(by0))
\t\tif new_game then
\t\t\tout.ting(\"Moving camera to initial position\")
\t\t\tscripting.game_interface:set_zoom_limit(1.00, 0.2)
\t\t\tCampaignUI.SetCameraZoom(0.95)
\t\t\tCampaignUI.SetCameraTargetInstant(%(cx), %(cy))
\t\t\tnew_game = false
\t\tend
\tend
end

scripting.AddEventCallBack(\"NewCampaignStarted\", OnNewCampaignStarted)
scripting.AddEventCallBack(\"UICreated\", OnUICreated)
";

fn write_lua(ctx: &BuildContext, bounds: [f64; 4]) -> Result<String> {
    let france = ctx
        .region_info
        .get("france")
        .ok_or_else(|| anyhow!("region france missing"))?;
    let (x, y) = (unfixed(france.pos.0), unfixed(france.pos.1));
    Ok(LUA
        .replace("%(bx0)", &format!("{:.1}", bounds[0]))
        .replace("%(by0)", &format!("{:.1}", bounds[1]))
        .replace("%(bx1)", &format!("{:.1}", bounds[2]))
        .replace("%(by1)", &format!("{:.1}", bounds[3]))
        .replace("%(cx)", &format!("{x:.3}"))
        .replace("%(cy)", &format!("{y:.3}")))
}

// ------------------------------------------------------------------------------------------------ consistency checks

/// Ownership coherence of the generated world (before the CAI is built on top of it).
fn validate_world(ctx: &BuildContext) -> Result<()> {
    let mut seen_chars: HashSet<u32> = HashSet::new();
    for (key, ri) in &ctx.region_info {
        let owner_id = ctx.fid[&ri.owner];
        let rec = &ctx.regions[ri.index];
        let ch = kids(rec)?;
        if id_at(ch, 20)? != owner_id {
            bail!("region {key}: owner id does not match {}", ri.owner);
        }
        let mut bad: Option<String> = None;
        walk(ch, &mut |n| {
            if let Node::Record { name, children, .. } = n {
                if name == "BUILDING" {
                    if children.get(2).and_then(as_text) != Some(ri.owner.as_str()) {
                        bad = Some(format!("region {key}: building not owned by {}", ri.owner));
                    }
                } else if name == "SIEGEABLE_GARRISON_RESIDENCE"
                    && children.first().and_then(|g| match g {
                        Node::Record { children: gc, .. } => gc.first().and_then(as_id),
                        _ => None,
                    }) != Some(owner_id)
                {
                    bad = Some(format!(
                        "region {key}: garrison residence not owned by {}",
                        ri.owner
                    ));
                }
            }
        });
        if let Some(m) = bad {
            bail!(m);
        }
    }
    for k in &ctx.fkeys {
        let info = &ctx.factions[k];
        let meta = &ctx.fmeta[k];
        let rec = &ctx.frec[k];
        let owned: Vec<&String> = ctx
            .region_info
            .iter()
            .filter(|(_, r)| r.owner == *k)
            .map(|(key, _)| key)
            .collect();
        if owned.len() != meta.owned.len() || owned.iter().zip(&meta.owned).any(|(a, b)| *a != b) {
            bail!("faction {k}: owned region list is stale");
        }
        if info.alive != !owned.is_empty() {
            bail!(
                "faction {k}: alive flag disagrees with its {} owned regions",
                owned.len()
            );
        }
        let chars = items(get(rec, &["CHARACTER_ARRAY"])?)?;
        if info.alive {
            if chars.is_empty() {
                bail!("faction {k} is alive but has no ministers or governors");
            }
            let mut covered: BTreeSet<&String> = BTreeSet::new();
            for g in &meta.gov_posts {
                covered.extend(g.regions.iter());
            }
            if covered.len() != owned.len() || owned.iter().any(|r| !covered.contains(*r)) {
                bail!("faction {k}: governorships do not cover exactly its owned regions");
            }
            let cap = info
                .capital
                .as_ref()
                .ok_or_else(|| anyhow!("faction {k} has no capital"))?;
            if ctx.region_info[cap].owner != *k {
                bail!(
                    "faction {k}: capital {cap} is owned by {}",
                    ctx.region_info[cap].owner
                );
            }
        } else {
            if !chars.is_empty() || !meta.gov_posts.is_empty() {
                bail!("faction {k} is landless but still carries characters / governorships");
            }
            if ctx.forces.iter().any(|f| f.faction == *k) {
                bail!("faction {k} is landless but has starting forces");
            }
        }
        for it in chars {
            let c = &it[0];
            let ci = factions::char_info(c)?;
            if !seen_chars.insert(ci.wid) {
                bail!("character id {} is used twice", ci.wid);
            }
            let det = kids(get(c, &["CHARACTER_DETAILS"])?)?;
            if id_at(det, 10)? != ci.wid {
                bail!(
                    "faction {k}: CHARACTER_DETAILS[10] of character {} does not point back at it",
                    ci.wid
                );
            }
        }
        if !items(get(rec, &["ARMY_ARRAY"])?)?.is_empty() && !info.alive {
            bail!("faction {k}: landless faction has armies");
        }
    }
    for f in &ctx.forces {
        let ri = &ctx.region_info[&f.region];
        if ri.owner != f.faction {
            bail!(
                "force of {} stands in {} owned by {}",
                f.faction,
                f.region,
                ri.owner
            );
        }
    }
    Ok(())
}

/// After the CAI: every region's governorship link must name a governorship of its owner.
fn validate_regions_after_cai(ctx: &BuildContext) -> Result<()> {
    for (key, ri) in &ctx.region_info {
        let gid = id_at(kids(&ctx.regions[ri.index])?, 22)?;
        let ok = ctx.fmeta[&ri.owner]
            .gov_posts
            .iter()
            .any(|g| g.gov_id == gid && g.theatre == ri.theatre);
        if !ok {
            bail!(
                "region {key}: governorship link {gid} is not a governorship of {} for theatre {}",
                ri.owner,
                ri.theatre
            );
        }
    }
    Ok(())
}

#[derive(Default)]
struct Refs {
    units: BTreeSet<String>,
    buildings: BTreeSet<String>,
    factions: BTreeSet<String>,
    regions: BTreeSet<String>,
    techs: BTreeSet<String>,
}

fn scan_refs(children: &[Node], in_array: &str, refs: &mut Refs) {
    for c in children {
        match c {
            Node::Record {
                name, children: ch, ..
            } => {
                match name.as_str() {
                    "UNIT_RECORD_KEY" | "LAND_RECORD_KEY" | "NAVAL_RECORD_KEY" => {
                        if let Some(s) = ch.first().and_then(as_text) {
                            refs.units.insert(s.to_string());
                        }
                    }
                    "BUILDING" => {
                        if let Some(s) = ch.get(1).and_then(as_text) {
                            refs.buildings.insert(s.to_string());
                        }
                        if let Some(s) = ch.get(2).and_then(as_text) {
                            refs.factions.insert(s.to_string());
                        }
                    }
                    "REGION" if in_array == "REGIONS_ARRAY" => {
                        if let Some(s) = ch.first().and_then(as_text) {
                            refs.regions.insert(s.to_string());
                        }
                        if let Some(s) = ch
                            .iter()
                            .filter(|x| ncode(x) == Some(14))
                            .filter_map(as_text)
                            .nth(2)
                        {
                            refs.factions.insert(s.to_string());
                        }
                    }
                    "CAMPAIGN_PLAYER_SETUP" => {
                        if let Some(s) = ch.get(2).and_then(as_text) {
                            refs.factions.insert(s.to_string());
                        }
                    }
                    "FACTION_TECHNOLOGY_MANAGER" => {
                        for ta in ch {
                            if let Node::Array { name, items, .. } = ta {
                                if name == "techs" {
                                    for it in items {
                                        if let Some(s) = it.first().and_then(as_text) {
                                            refs.techs.insert(s.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
                scan_refs(ch, "", refs);
            }
            Node::Array { name, items, .. } => {
                if name == "REGION_OWNERSHIPS" {
                    for it in items {
                        if let Some(s) = it.first().and_then(as_text) {
                            refs.regions.insert(s.to_string());
                        }
                        if let Some(s) = it.get(1).and_then(as_text) {
                            refs.factions.insert(s.to_string());
                        }
                    }
                }
                for it in items {
                    scan_refs(it, name, refs);
                }
            }
            Node::Value(_) => {}
        }
    }
}

fn db_keys(
    snapshot: &Snapshot,
    rules: &Rules,
    map: &MapArtifacts,
    table: CampaignTable,
    col: &str,
) -> Result<HashSet<String>> {
    let mut out: HashSet<String> = HashSet::new();
    for row in crate::db::load_campaign_table(snapshot, rules, table)?.json_rows()? {
        if let Some(s) = row.get(col).and_then(|v| v.as_str()) {
            out.insert(s.to_string());
        }
    }
    // Faction identities come exclusively from the imported database, never generated assets.
    if matches!(table, CampaignTable::Factions) {
        return Ok(out);
    }
    // curated rows of the conversion (assets/rules/db_rows) and rows generated by the map step
    let mut add_rows = |v: &Json| {
        if let Some(rows) = v.get("rows").and_then(|r| r.as_array()) {
            for r in rows {
                if let Some(s) = r.get(col).and_then(|x| x.as_str()) {
                    out.insert(s.to_string());
                }
            }
        }
    };
    let dir = rules.root.join("db_rows");
    let full = table.name();
    let stem = full.strip_suffix("_tables").unwrap_or(full);
    let (a, b2) = (format!("__{stem}.json"), format!("__{full}.json"));
    if dir.is_dir() {
        let mut names: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for n in names {
            if n.ends_with(&a) || n.ends_with(&b2) {
                let bytes = std::fs::read(dir.join(&n)).with_context(|| format!("Read {n}"))?;
                add_rows(
                    &serde_json::from_slice::<Json>(&bytes)
                        .with_context(|| format!("Parse {n}"))?,
                );
            }
        }
    }
    for (name, v) in &map.db_rows {
        if name == full || name == stem {
            add_rows(v);
        }
    }
    Ok(out)
}

/// Every faction key must exist in the imported target; other references may also use generated campaign rows.
fn check_refs(snapshot: &Snapshot, rules: &Rules, map: &MapArtifacts, root: &Node, faction_keys: HashSet<String>) -> Result<()> {
    let mut refs = Refs::default();
    if let Node::Record { children, .. } = root {
        scan_refs(children, "", &mut refs);
    }
    let known = [
        (
            "factions",
            &refs.factions,
            faction_keys,
        ),
        (
            "regions",
            &refs.regions,
            db_keys(snapshot, rules, map, CampaignTable::Regions, "key")?,
        ),
        (
            "units",
            &refs.units,
            db_keys(snapshot, rules, map, CampaignTable::Units, "key")?,
        ),
        (
            "building levels",
            &refs.buildings,
            db_keys(
                snapshot,
                rules,
                map,
                CampaignTable::BuildingLevels,
                "level_name",
            )?,
        ),
        (
            "technologies",
            &refs.techs,
            db_keys(snapshot, rules, map, CampaignTable::Technologies, "key")?,
        ),
    ];
    let mut errors = Vec::new();
    for (what, set, db) in &known {
        let missing: Vec<&String> = set
            .iter()
            .filter(|k| !k.is_empty() && !db.contains(*k))
            .collect();
        if !missing.is_empty() {
            errors.push(format!(
                "{what}: {} unresolved ({})",
                missing.len(),
                missing
                    .iter()
                    .take(20)
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    if !errors.is_empty() {
        bail!(
            "startpos references keys that do not exist in the NTW databases: {}",
            errors.join("; ")
        );
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ main entry

/// Faction records of the effective NTW `eur_napoleon` start position of `snapshot` (the original game's, or the
/// structural template supplied by a dataset adapter). Parsed once per distinct file content.
pub fn startpos_structure(snapshot: &Snapshot) -> Result<Arc<StartposStructure>> {
    static CACHE: LazyLock<Mutex<HashMap<[u8; 32], Arc<StartposStructure>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let bytes = snapshot
        .read(Game::Ntw, NTW_STARTPOS)
        .with_context(|| format!("Read {NTW_STARTPOS} of the target"))?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    if let Some(hit) = CACHE
        .lock()
        .map_err(|_| anyhow!("start position cache poisoned"))?
        .get(&digest)
    {
        return Ok(hit.clone());
    }
    let parsed = esf::read(&bytes).with_context(|| format!("Parse {NTW_STARTPOS}"))?;
    let model = model_of(&parsed.root).with_context(|| {
        format!("{NTW_STARTPOS} is not a campaign start position (no CAMPAIGN_ENV/CAMPAIGN_MODEL)")
    })?;
    let tmpl = Templates::new(model).with_context(|| {
        format!("{NTW_STARTPOS} does not have the structure the converter can use as template")
    })?;
    let structure = Arc::new(StartposStructure::from_templates(&tmpl)?);
    CACHE
        .lock()
        .map_err(|_| anyhow!("start position cache poisoned"))?
        .insert(digest, structure.clone());
    Ok(structure)
}

/// Per-faction rule data of resolved rules (see `campaign_profile::prepare_rules` for the fields).
pub fn faction_infos(rules: &Rules) -> Result<BTreeMap<String, FactionInfo>> {
    let obj = rules
        .factions
        .as_object()
        .ok_or_else(|| anyhow!("faction rules must be an object"))?;
    let mut out = BTreeMap::new();
    for (k, v) in obj {
        out.insert(
            k.clone(),
            FactionInfo {
                alive: v
                    .get("alive")
                    .and_then(|x| x.as_bool())
                    .ok_or_else(|| anyhow!("faction {k}: alive"))?,
                capital: v
                    .get("capital")
                    .and_then(|x| x.as_str())
                    .map(str::to_string),
                playable: v.get("playable").and_then(|x| x.as_bool()).unwrap_or(false),
                regions: v.get("regions").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                etw_source: v
                    .get("etw_source")
                    .and_then(|x| x.as_str())
                    .unwrap_or(k)
                    .to_string(),
                donor: v.get("donor").and_then(|x| x.as_str()).map(str::to_string),
            },
        );
    }
    Ok(out)
}

/// `edited_regions`: the generated `campaign_maps/nap_world/regions.esf` when it holds edited theatre rectangles (theatre
/// extensions); the camera bounds (EPISODIC_RESTRICTIONS 9..12, `scripting.lua`) then follow those rectangles instead
/// of the ETW originals. `sea_grids`: the generated `campaign_maps/nap_world/sea_grids.esf`, whose cell layout decides
/// the CAI sea grid cell of every fleet.
pub fn generate(
    snapshot: &Snapshot,
    rules: &Rules,
    map: &MapArtifacts,
    added_keys: &BTreeSet<String>,
    edited_regions: Option<&Node>,
    sea_grids: &Node,
    progress: &Progress<'_>,
) -> Result<Generated> {
    let faction_keys = db_keys(snapshot, rules, map, CampaignTable::Factions, "key")?;
    for key in rules.factions.as_object().context("Faction rules must be an object")?.keys() {
        if !faction_keys.contains(key) {
            bail!("Faction {key} is absent from the imported target database; creating factions is not supported");
        }
    }
    progress("Startpos: reading the ETW and NTW start positions");
    let etw = read_esf(snapshot, Game::Etw, ETW_STARTPOS)?;
    let ntw = read_esf(snapshot, Game::Ntw, NTW_STARTPOS)?;
    let etw_model = Arc::new(model_of(&etw.root)?.clone());
    let ntw_model = Arc::new(model_of(&ntw.root)?.clone());
    let tmpl = Templates::new(&ntw_model)?;
    let used = collect_used(&[&etw.root, &ntw.root]);
    let mut ids = IdAlloc::new(used, 1_400_000_000, 20_000);
    let unit_db = UnitDb::load(snapshot, rules)?;
    let mut gm = GlobalMap::load(snapshot)?;
    if let Some(root) = edited_regions {
        gm.use_theatres_of(root).context("theatre rectangles of the generated regions.esf")?;
    }

    // ---- plan
    let finfo = faction_infos(rules)?;
    let mut etw_fkey = BTreeMap::new();
    let mut etw_factions: BTreeMap<String, Arc<Node>> = BTreeMap::new();
    for it in items(get(&etw_model, &["WORLD", "FACTION_ARRAY"])?)? {
        let k = faction_key(&it[0])?;
        etw_fkey.insert(nth_id(kids(&it[0])?, 0)?, k.clone());
        etw_factions.insert(k, Arc::new(it[0].clone()));
    }
    let mut ntw_fkey = BTreeMap::new();
    for (k, rec) in &tmpl.fac_by_key {
        ntw_fkey.insert(nth_id(kids(rec)?, 0)?, k.clone());
    }
    let mut fkeys: Vec<String> = tmpl
        .fac_order
        .iter()
        .filter(|k| finfo.contains_key(*k))
        .cloned()
        .collect();
    let mut rest: Vec<String> = finfo
        .keys()
        .filter(|k| !tmpl.fac_order.contains(k))
        .cloned()
        .collect();
    rest.sort();
    fkeys.extend(rest);
    let mut fid: BTreeMap<String, u32> = BTreeMap::new();
    let mut structures: BTreeMap<String, Structure> = BTreeMap::new();
    let mut gov: BTreeMap<String, String> = BTreeMap::new();
    let sp = StartposStructure::from_templates(&tmpl)?;
    // which faction each ETW start position faction stands for now ("" = none): ETW trade/CAI objects follow it
    let mut etw_to_key: BTreeMap<String, String> = etw_factions
        .keys()
        .map(|k| (k.clone(), String::new()))
        .collect();
    for (k, info) in &finfo {
        etw_to_key.insert(info.etw_source.clone(), k.clone());
    }
    for k in &fkeys {
        let info = &finfo[k];
        // the target start position holds a record of this very faction
        let has_own = tmpl.fac_by_key.contains_key(k);
        if has_own {
            let rec = &tmpl.fac_by_key[k];
            fid.insert(k.clone(), nth_id(kids(rec)?, 0)?);
        } else {
            fid.insert(k.clone(), ids.world());
        }
        let st = factions::structure_for(info, k, &sp, rules)
            .with_context(|| format!("Structure of faction {k}"))?;
        let donor = factions::government_donor(&st, k, has_own);
        let drec = tmpl
            .fac_by_key
            .get(&donor)
            .ok_or_else(|| anyhow!("donor {donor} missing"))?;
        gov.insert(
            k.clone(),
            text_at(kids(get(drec, &["GOVERNMENT"])?)?, 1)?.to_string(),
        );
        structures.insert(k.clone(), st);
    }
    let rebel_id = {
        let reb = get(&ntw_model, &["WORLD", "REBEL_FACTION"])?;
        nth_id(
            kids(
                kids(reb)?
                    .first()
                    .ok_or_else(|| anyhow!("REBEL_FACTION is empty"))?,
            )?,
            0,
        )?
    };

    let mut ctx = BuildContext {
        snapshot,
        rules,
        map,
        sea_grids,
        added_keys: added_keys.clone(),
        etw_model,
        ntw_model,
        etw_fkey,
        ntw_fkey,
        etw_factions,
        etw_to_key,
        fkeys: fkeys.clone(),
        fid,
        rebel_id,
        tech_id: BTreeMap::new(),
        factions: finfo,
        fmeta: BTreeMap::new(),
        frec: BTreeMap::new(),
        war: BTreeSet::new(),
        regions: Vec::new(),
        region_order: Vec::new(),
        region_info: BTreeMap::new(),
        dropped_slots: BTreeSet::new(),
        forces: Vec::new(),
        ids,
        notes: Vec::new(),
    };

    // ---- regions
    progress("Startpos: converting 137 regions to the NTW schema");
    build_regions(&mut ctx, &gov)?;

    // ---- factions
    progress("Startpos: assembling factions, ministers and governorships");
    for k in &fkeys {
        let (mut rec, mut meta) = factions::make_faction(&mut ctx, &tmpl, k, &structures[k])?;
        factions::finish_faction(&mut ctx, &tmpl, k, &mut rec, &mut meta)?;
        ctx.frec.insert(k.clone(), rec);
        ctx.fmeta.insert(k.clone(), meta);
    }
    for k in &fkeys {
        let tm = kids(get(&ctx.frec[k], &["FACTION_TECHNOLOGY_MANAGER"])?)?;
        let id = tm
            .iter()
            .find(|c| ncode(c) == Some(4))
            .and_then(as_id)
            .ok_or_else(|| anyhow!("{k}: technology tree id missing"))?;
        ctx.tech_id.insert(k.clone(), id);
    }

    // ---- forces
    progress("Startpos: deploying starting armies and fleets");
    ForceBuilder::new(&tmpl, &unit_db, &gm).run(&mut ctx)?;
    for n in &ctx.notes {
        progress(&format!("Startpos: {n}"));
    }

    factions::build_diplomacy(&mut ctx, &tmpl)?;
    factions::fix_shapes(&mut ctx, &tmpl)?;
    validate_world(&ctx).context("Generated world is not coherent")?;

    // ---- CAI
    progress("Startpos: building the campaign AI world");
    let cai_out = cai::build(&mut ctx).context("Campaign AI build")?;
    validate_regions_after_cai(&ctx).context("Governorship links")?;

    // ---- assemble
    progress("Startpos: assembling the start position");
    let mut root = ntw.root.clone();
    {
        let mut fa: Vec<Vec<Node>> = Vec::new();
        for k in &ctx.fkeys {
            fa.push(vec![
                ctx.frec
                    .remove(k)
                    .ok_or_else(|| anyhow!("no record for {k}"))?,
            ]);
        }
        let regs: Vec<Vec<Node>> = std::mem::take(&mut ctx.regions)
            .into_iter()
            .map(|r| vec![r])
            .collect();
        let model = model_of_mut(&mut root)?;
        {
            let world = get_mut(model, &["WORLD"])?;
            *items_mut(get_mut(world, &["FACTION_ARRAY"])?)? = fa;
            *items_mut(get_mut(world, &["REGION_MANAGER", "REGIONS_ARRAY"])?)? = regs;
            let sp = items_mut(get_mut(world, &["SPYING_ARRAY"])?)?;
            let t = sp
                .first()
                .ok_or_else(|| anyhow!("SPYING_ARRAY is empty"))?
                .clone();
            *sp = ctx
                .fkeys
                .iter()
                .map(|k| {
                    let mut it = t.clone();
                    it[0] = s16(k.clone());
                    it
                })
                .collect();
            let rf = get(world, &["REBEL_FACTION", "FACTION"])?;
            if !items(get(rf, &["CHARACTER_ARRAY"])?)?.is_empty()
                || !items(get(rf, &["ARMY_ARRAY"])?)?.is_empty()
            {
                bail!("NTW rebel faction unexpectedly has characters or armies");
            }
        }
        // CAI
        let ci = kids_mut(model)?
            .iter()
            .position(|c| is_record(c, "CAI_INTERFACE"))
            .ok_or_else(|| anyhow!("CAI_INTERFACE missing"))?;
        kids_mut(model)?[ci] = cai_out.interface;
        // CAMPAIGN_DIRECTOR/CDIR_INTERFACE child 0 = CAI id of the director pool's CDIR_WORLD belief: follow the director-pool re-id
        for it in items_mut(get_mut(model, &["CAMPAIGN_DIRECTOR"])?)?.iter_mut() {
            let cd = it
                .iter_mut()
                .find(|c| is_record(c, "CDIR_INTERFACE"))
                .ok_or_else(|| anyhow!("CDIR_INTERFACE missing"))?;
            let ch = kids_mut(cd)?;
            let old = id_at(ch, 0)?;
            let new = cai_out.director_map.get(&old).ok_or_else(|| {
                anyhow!(
                    "CDIR_INTERFACE refers to CAI id {old} that the director pool does not contain"
                )
            })?;
            ch[0] = u32v(*new);
        }
        // trade manager + pathfinder from the ETW world
        let tm = trade_manager(&ctx)?;
        let pf = pathfinder(&ctx)?;
        let mk = kids_mut(model)?;
        let i = idx_of(mk, "CAMPAIGN_TRADE_MANAGER")?;
        mk[i] = tm;
        let i = idx_of(mk, "CAMPAIGN_PATHFINDER")?;
        mk[i] = pf;
        // map data
        let md = kids_mut(get_mut(model, &["CAMPAIGN_MAP_DATA"])?)?;
        md[0] = s16(format!("campaign_maps/{MAP_NAME}"));
        md[1] = s16(format!("campaign_maps/{MAP_NAME}/display"));
        md[2] = s16(MAP_NAME);
        md[3] = b(false);
    }
    let bounds = header_and_setup(&ctx, &mut root, &etw.root, &gm)?;
    let lua = write_lua(&ctx, bounds)?;

    progress("Startpos: checking database references");
    check_refs(snapshot, rules, map, &root, faction_keys)?;

    let file = esf::Esf {
        timestamp: ntw.timestamp,
        names: ntw.names.clone(),
        root,
    };
    let startpos = esf::write(&file).context("Write startpos.esf")?;
    Ok(Generated {
        startpos,
        lua,
        notes: ctx.notes,
    })
}
