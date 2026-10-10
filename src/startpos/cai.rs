//! CAI_INTERFACE reconstruction (world model objects from the ETW map + pristine NTW AI pools).
//!
//! Native port of `etw_map_port/startpos/cai_b.py` (accepted path: attitude extension OFF). The CAI_WORLD is rebuilt from the
//! ETW map objects (regions, boundaries, settlements, slots, HLCIs, trade segments, sea grid) plus the generated factions,
//! tech trees, governorships, characters, mobiles and units of the `BuildContext`; the per-faction BDI manager pools are
//! clones of the NTW pool of a faction without territory (greece) / the NTW rebel manager with re-allocated ids, and the root
//! pool receives the per-faction root beliefs and the per-region beliefs of root analysers #11 / #20.
//!
//! Every ownership dependent list (faction owned regions / theatres / hostile capitals / governorships / mobiles / characters,
//! region governors / present factions / mobiles, settlement owners and garrisons, sea grid cell mobiles, REGION child 22
//! governorship links) is derived from the (already edited) `BuildContext`, so edited ownership propagates everywhere.

mod check;
pub(crate) mod nodes;

use std::collections::{BTreeMap, BTreeSet, HashSet};

use anyhow::{Context, Result, anyhow, bail};

use crate::esf::{Node, Value};

use super::context::{BuildContext, ForceKind};
use nodes::*;

pub use check::{CheckReport, fleet_cell_problems};

/// Result of the CAI build.
pub struct CaiOutput {
    /// complete CAI_INTERFACE record to splice into the NTW CAMPAIGN_MODEL
    pub interface: Node,
    /// director pool re-id (NTW id -> new id); CAMPAIGN_DIRECTOR/CDIR_INTERFACE child 0 must follow it
    pub director_map: BTreeMap<u32, u32>,
}

/// Switches of the accepted conversion (all OFF by default, exactly as the accepted build).
#[derive(Clone, Copy, Debug, Default)]
pub struct CaiOptions {
    /// Python `CAI_ATTITUDES=1`: per-subject attitude beliefs/goals for all other factions (speculative, off)
    pub attitudes: bool,
}

const TRANSITION_AREAS: bool = false;
const REBEL: &str = "__rebel__";
const BASE_MANAGER_INDEX: usize = 5; // NTW eur_napoleon manager of a faction without territory (greece)
const CAI_START: u32 = 20_000;
/// per-faction ROOT beliefs referenced by a faction manager pool (analysers #9 diplomatic, #21 active recruitment, #23 owned regions)
const ROOT_TYPES: [&str; 3] = [
    "CAI_ACTIVE_RECRUITMENT_ANALYSIS",
    "CAI_OWNED_REGIONS_ANALYSIS",
    "CAI_DIPLOMATIC_ANALYSIS",
];
const ATT_B: &str = "CAI_FACTION_TO_FACTION_ATTITUDE_ANALYSIS";
const ATT_G: &str = "CAI_BDI_GOAL_MANAGE_FACTION_TO_FACTION_ATTITUDE";
const ATT_M: &str = "CAI_FACTION_BDI_POOL_ATTITUDE_MANAGER";
/// root desire id -> root belief type created once per CAI_WORLD region
const REGION_ROOT: [(u32, &str); 2] = [
    (11, "CAI_BORDER_PATROL_ANALYSIS"),
    (20, "CAI_REGION_OCCUPANCY_ANALYSIS"),
];

/// CAI_WORLD arrays rebuilt from scratch by `build_world`
const WORLD_ARRAYS: [&str; 18] = [
    "CAI_WORLD_THEATRES",
    "CAI_WORLD_REGIONS",
    "CAI_WORLD_REGION_BOUNDARIES",
    "CAI_WORLD_BUILDING_SLOTS",
    "CAI_WORLD_REGION_SLOTS",
    "CAI_WORLD_TRADING_POSTS",
    "CAI_WORLD_SETTLEMENTS",
    "CAI_WORLD_FACTIONS",
    "CAI_WORLD_RESOURCE_MOBILES",
    "CAI_WORLD_UNITS",
    "CAI_WORLD_CHARACTERS",
    "CAI_WORLD_FORTS",
    "CAI_WORLD_TRADE_ROUTE_SEGMENTS",
    "CAI_WORLD_REGION_HLCIS",
    "CAI_WORLD_TRANSITION_AREAS",
    "CAI_WORLD_TECHNOLOGY_TREES",
    "CAI_WORLD_GOVERNORSHIPS",
    "CAI_WORLD_SEA_GRID_CELLS",
];

/// Build the CAI_INTERFACE with the accepted defaults.
pub fn build(ctx: &mut BuildContext) -> Result<CaiOutput> {
    build_with(ctx, CaiOptions::default())
}

pub fn build_with(ctx: &mut BuildContext, options: CaiOptions) -> Result<CaiOutput> {
    let etw = ctx.etw_model.clone();
    let ntw = ctx.ntw_model.clone();
    let mut cb = CaiBuilder::new(&etw, &ntw, ctx.sea_grids, ctx, options)?;
    cb.allocate(ctx)?;
    let linked = cb.link_regions(ctx)?;
    cb.build_entity_items(ctx)?;
    let interface = cb.build_interface(ctx)?;
    let report = check::verify(ctx, &ntw, &interface).context("CAI consistency check")?;
    ctx.notes.push(format!(
        "CAI: {} factions, {} regions, {} mobiles, {} units, {} characters, {} governorships, {} root beliefs ({} region beliefs), {} region governorship links; {}",
        cb.fkeys.len(),
        cb.region_cai.len(),
        ctx.forces.len(),
        ctx.forces.iter().map(|f| f.units.len()).sum::<usize>(),
        cb.char_cai.len(),
        ctx.fmeta.values().map(|m| m.gov_posts.len()).sum::<usize>(),
        cb.new_root_beliefs_len,
        cb.region_beliefs,
        linked,
        report.summary(),
    ));
    if report.bad() > 0 {
        bail!(
            "generated CAI graph is inconsistent:\n{}",
            report.problems().join("\n")
        );
    }
    let sea = crate::regionmap::sea_grids::SeaGrids::parse(ctx.sea_grids).context("Parse the generated sea_grids.esf")?;
    let fleet_problems = fleet_cell_problems(&interface, &sea)?;
    if !fleet_problems.is_empty() {
        bail!("generated CAI sea grid cells disagree with sea_grids.esf:\n{}", fleet_problems.join("\n"));
    }
    Ok(CaiOutput {
        interface,
        director_map: cb.dir_map,
    })
}

struct SeaGrid {
    key: String,
    p0: [f64; 2],
    p1: [f64; 2],
    size: f64,
    cols: usize,
    rows: usize,
    offset: usize,
}

/// (theatre key, local cell index, global cell index)
type Cell = (String, usize, usize);

struct CaiBuilder<'m> {
    options: CaiOptions,
    cw_e: &'m Node,
    cw_n: &'m Node,
    ci_n: &'m Node,

    next_cai: u32,
    reserved: HashSet<u32>,

    region_cai: BTreeMap<String, u32>,
    inv_region: BTreeMap<u32, String>,
    region_theatres: BTreeMap<u32, Vec<u32>>,
    theatre_cai: BTreeMap<String, u32>,
    cai_theatre_key: BTreeMap<u32, String>,
    seagrid: Vec<SeaGrid>,

    /// factions in order incl. the rebel faction (last)
    fkeys: Vec<String>,
    faction_cai: BTreeMap<String, u32>,
    tech_cai: BTreeMap<String, u32>,
    char_cai: BTreeMap<u32, u32>,
    mobile_cai: BTreeMap<u32, u32>,
    mobile_items: Vec<Vec<Node>>,
    unit_items: Vec<Vec<Node>>,
    char_items: Vec<Vec<Node>>,
    faction_mobiles: BTreeMap<String, Vec<u32>>,
    faction_chars: BTreeMap<String, Vec<u32>>,
    region_mobiles: BTreeMap<u32, Vec<u32>>,
    theatre_mobiles: BTreeMap<u32, Vec<u32>>,
    cell_mobiles: BTreeMap<(String, usize), Vec<u32>>,
    /// region cai id -> CAI ids of the factions with a force in it (order of first appearance)
    region_present: BTreeMap<u32, Vec<u32>>,
    force_cell: Vec<Option<Cell>>,

    // NTW templates (Python `Templates`)
    t_char: BTreeMap<u32, &'m Vec<Node>>,
    t_unit: BTreeMap<u32, &'m Vec<Node>>,
    t_mobile: BTreeMap<u32, &'m Vec<Node>>,
    ntw_fcai: BTreeMap<String, u32>,

    // NTW root pool beliefs (id -> item / record type name)
    ntw_root_b: BTreeMap<u32, &'m Vec<Node>>,
    ntw_root_type: BTreeMap<u32, String>,
    new_root_beliefs: Vec<Vec<Node>>,
    new_root_beliefs_len: usize,
    region_beliefs: usize,
    analyser_entries: BTreeMap<String, Vec<(u32, u32)>>,
    region_tmpl: BTreeMap<u32, Vec<Node>>,
    owns_tmpl: Option<Vec<Node>>,
    dir_map: BTreeMap<u32, u32>,
}

fn sorted(v: Option<&Vec<u32>>) -> Vec<u32> {
    let mut v = v.cloned().unwrap_or_default();
    v.sort_unstable();
    v
}

impl<'m> CaiBuilder<'m> {
    fn new(
        etw: &'m Node,
        ntw: &'m Node,
        sea_root: &Node,
        ctx: &BuildContext,
        options: CaiOptions,
    ) -> Result<Self> {
        let cw_e = get(etw, &["CAI_INTERFACE", "CAI_WORLD"]).context("ETW CAI_WORLD")?;
        let cw_n = get(ntw, &["CAI_INTERFACE", "CAI_WORLD"]).context("NTW CAI_WORLD")?;
        let ci_n = get(ntw, &["CAI_INTERFACE"]).context("NTW CAI_INTERFACE")?;

        let mut region_cai = BTreeMap::new();
        let mut inv_region = BTreeMap::new();
        let mut region_theatres = BTreeMap::new();
        let mut reserved: HashSet<u32> = HashSet::new();
        for it in get_items(cw_e, &["CAI_WORLD_REGIONS"])? {
            let spec = spec_kids(it)?;
            let key = spec
                .iter()
                .find(|c| is_string16(c))
                .and_then(string_of)
                .ok_or_else(|| anyhow!("ETW CAI region without key"))?
                .to_string();
            let id = item_id(it)?;
            region_cai.insert(key.clone(), id);
            inv_region.insert(id, key);
            region_theatres.insert(
                id,
                arr_vals(
                    spec.first()
                        .ok_or_else(|| anyhow!("ETW CAI region without theatre list"))?,
                ),
            );
        }
        let mut theatre_cai = BTreeMap::new();
        for it in get_items(cw_e, &["CAI_WORLD_THEATRES"])? {
            let spec = spec_kids(it)?;
            let key = string_of(
                spec.first()
                    .ok_or_else(|| anyhow!("empty ETW CAI theatre"))?,
            )
            .ok_or_else(|| anyhow!("ETW CAI theatre key is not a string"))?;
            theatre_cai.insert(key.to_string(), item_id(it)?);
        }
        let cai_theatre_key: BTreeMap<u32, String> =
            theatre_cai.iter().map(|(k, v)| (*v, k.clone())).collect();
        // ETW CAI world ids that survive into the output must never be handed out again
        if let Node::Record { children, .. } = cw_e {
            for c in children {
                if let Node::Array { items, .. } = c {
                    for it in items {
                        if let Ok(id) = item_id(it) {
                            reserved.insert(id);
                        }
                    }
                }
            }
        }
        for it in get_items(cw_e, &["CAI_WORLD_FACTIONS"])? {
            let spec = spec_kids(it)?;
            if let Some(pi) = find_idx(spec, PROPSET) {
                if let Some(h) = spec.get(pi + 1).and_then(scalar) {
                    reserved.insert(h);
                }
            }
        }
        let seagrid = parse_seagrid(sea_root, &theatre_cai)?;
        let cells = get_items(cw_e, &["CAI_WORLD_SEA_GRID_CELLS"])?.len();
        let grid_cells: usize = seagrid.iter().map(|g| g.cols * g.rows).sum();
        if cells != grid_cells {
            bail!("the generated sea_grids.esf has {grid_cells} cells but the CAI world holds {cells} sea grid cells");
        }

        // NTW templates
        let mut t_char = BTreeMap::new();
        for it in get_items(cw_n, &["CAI_WORLD_CHARACTERS"])? {
            t_char.insert(sc_at(spec_kids(it)?, 3)?, it);
        }
        let mut t_unit = BTreeMap::new();
        for it in get_items(cw_n, &["CAI_WORLD_UNITS"])? {
            t_unit.insert(sc_at(spec_kids(it)?, 1)?, it);
        }
        let mut t_mobile = BTreeMap::new();
        for it in get_items(cw_n, &["CAI_WORLD_RESOURCE_MOBILES"])? {
            t_mobile.insert(sc_at(spec_kids(it)?, 10)?, it);
        }
        // NTW cai faction ids by key
        let mut ntw_fcai = BTreeMap::new();
        for it in get_items(cw_n, &["CAI_WORLD_FACTIONS"])? {
            let wid = sc_at(spec_kids(it)?, 6)?;
            let key = ctx
                .ntw_fkey
                .get(&wid)
                .cloned()
                .unwrap_or_else(|| REBEL.to_string());
            ntw_fcai.insert(key, item_id(it)?);
        }
        // NTW root pool beliefs
        let rp_n = get(ci_n, &["CAI_BDI_POOL"])?;
        let mut ntw_root_b = BTreeMap::new();
        let mut ntw_root_type = BTreeMap::new();
        for it in get_items(rp_n, &["CAI_BDI_POOL_BELIEFS"])? {
            let id = sc_at(it, 2)?;
            ntw_root_type.insert(id, spec_name(it)?.to_string());
            ntw_root_b.insert(id, it);
        }

        let mut fkeys: Vec<String> = ctx.fkeys.clone();
        fkeys.push(REBEL.to_string());
        Ok(Self {
            options,
            cw_e,
            cw_n,
            ci_n,
            next_cai: CAI_START,
            reserved,
            region_cai,
            inv_region,
            region_theatres,
            theatre_cai,
            cai_theatre_key,
            seagrid,
            fkeys,
            faction_cai: BTreeMap::new(),
            tech_cai: BTreeMap::new(),
            char_cai: BTreeMap::new(),
            mobile_cai: BTreeMap::new(),
            mobile_items: Vec::new(),
            unit_items: Vec::new(),
            char_items: Vec::new(),
            faction_mobiles: BTreeMap::new(),
            faction_chars: BTreeMap::new(),
            region_mobiles: BTreeMap::new(),
            theatre_mobiles: BTreeMap::new(),
            cell_mobiles: BTreeMap::new(),
            region_present: BTreeMap::new(),
            force_cell: Vec::new(),
            t_char,
            t_unit,
            t_mobile,
            ntw_fcai,
            ntw_root_b,
            ntw_root_type,
            new_root_beliefs: Vec::new(),
            new_root_beliefs_len: 0,
            region_beliefs: 0,
            analyser_entries: BTreeMap::new(),
            region_tmpl: BTreeMap::new(),
            owns_tmpl: None,
            dir_map: BTreeMap::new(),
        })
    }

    /// next free CAI id (small ints; ETW CAI ids that are kept are skipped)
    fn cai(&mut self) -> u32 {
        while self.reserved.contains(&self.next_cai) {
            self.next_cai += 1;
        }
        let v = self.next_cai;
        self.next_cai += 1;
        v
    }

    /// (theatre key, local index, global index) of the sea grid cell containing (x, y)
    fn sea_cell(&self, x: f64, y: f64) -> Option<Cell> {
        for g in &self.seagrid {
            let (x0, y0, x1, y1) = (g.p0[0], g.p0[1], g.p1[0], g.p1[1]);
            if x0 <= x && x < x1 && y0 <= y && y < y1 {
                let col = (((x - x0) / g.size) as usize).min(g.cols - 1);
                let row = (((y - y0) / g.size) as usize).min(g.rows - 1);
                let loc = row * g.cols + col;
                return Some((g.key.clone(), loc, g.offset + loc));
            }
        }
        None
    }

    fn theatre_of(&self, x: f64, y: f64, region_cai: u32) -> Result<u32> {
        let cands = self
            .region_theatres
            .get(&region_cai)
            .cloned()
            .unwrap_or_default();
        for tid in &cands {
            let k = self
                .cai_theatre_key
                .get(tid)
                .ok_or_else(|| anyhow!("CAI region theatre {tid} is not a CAI theatre"))?;
            for g in &self.seagrid {
                if &g.key == k && g.p0[0] <= x && x < g.p1[0] && g.p0[1] <= y && y < g.p1[1] {
                    return Ok(*tid);
                }
            }
        }
        match cands.first() {
            Some(t) => Ok(*t),
            None => self
                .theatre_cai
                .get("2")
                .copied()
                .ok_or_else(|| anyhow!("no europe theatre in the ETW CAI world")),
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    fn allocate(&mut self, ctx: &mut BuildContext) -> Result<()> {
        let fkeys = self.fkeys.clone();
        for k in &fkeys {
            let base = self.cai();
            self.cai();
            let tech = self.cai();
            self.faction_cai.insert(k.clone(), base);
            self.tech_cai.insert(k.clone(), tech);
        }
        // characters that are not on the map
        for k in &ctx.fkeys {
            let meta = ctx
                .fmeta
                .get(k)
                .ok_or_else(|| anyhow!("no faction meta for {k}"))?;
            for ch in &meta.chars {
                let id = self.cai();
                self.char_cai.insert(ch.wid, id);
            }
        }
        // governorships
        for k in &ctx.fkeys {
            let meta = ctx
                .fmeta
                .get_mut(k)
                .ok_or_else(|| anyhow!("no faction meta for {k}"))?;
            for gp in &mut meta.gov_posts {
                gp.cai = self.cai();
            }
        }
        // forces
        self.force_cell = Vec::with_capacity(ctx.forces.len());
        for fi in 0..ctx.forces.len() {
            let cai_char = self.cai();
            let cai_mobile = self.cai();
            let n_units = ctx.forces[fi].units.len();
            if n_units == 0 || ctx.forces[fi].unit_ids.len() != n_units {
                bail!(
                    "force {fi} of {}: {} unit keys but {} unit ids",
                    ctx.forces[fi].faction,
                    n_units,
                    ctx.forces[fi].unit_ids.len()
                );
            }
            let cai_units: Vec<u32> = (0..n_units).map(|_| self.cai()).collect();
            let f = &mut ctx.forces[fi];
            f.cai_char = cai_char;
            f.cai_mobile = cai_mobile;
            f.cai_units = cai_units;
            self.char_cai.insert(f.wid_char, cai_char);
            self.mobile_cai.insert(f.wid_army, cai_mobile);
            let x = f.x as f64 / 1048576.0;
            let y = f.y as f64 / 1048576.0;
            let rc = *self.region_cai.get(&f.region_at).ok_or_else(|| {
                anyhow!(
                    "force of {} stands in {} which has no CAI region",
                    f.faction,
                    f.region_at
                )
            })?;
            f.region_cai = rc;
            let theatre = self.theatre_of(x, y, rc)?;
            f.theatre_cai = theatre;
            let cell = if f.kind == ForceKind::Navy {
                self.sea_cell(x, y)
            } else {
                None
            };
            f.cell = cell.as_ref().map(|c| (c.1 as i32, c.2 as i32));
            self.faction_mobiles
                .entry(f.faction.clone())
                .or_default()
                .push(cai_mobile);
            self.faction_chars
                .entry(f.faction.clone())
                .or_default()
                .push(cai_char);
            self.region_mobiles.entry(rc).or_default().push(cai_mobile);
            self.theatre_mobiles
                .entry(theatre)
                .or_default()
                .push(cai_mobile);
            let fc = *self
                .faction_cai
                .get(&f.faction)
                .ok_or_else(|| anyhow!("force of unknown faction {}", f.faction))?;
            let present = self.region_present.entry(rc).or_default();
            if !present.contains(&fc) {
                present.push(fc);
            }
            if let Some(c) = &cell {
                self.cell_mobiles
                    .entry((c.0.clone(), c.1))
                    .or_default()
                    .push(cai_mobile);
            }
            self.force_cell.push(cell);
        }
        for k in &ctx.fkeys {
            for ch in &ctx.fmeta[k].chars {
                let id = self.char_cai[&ch.wid];
                self.faction_chars.entry(k.clone()).or_default().push(id);
            }
        }
        Ok(())
    }

    fn region_gov_cai(&self, ctx: &BuildContext, key: &str) -> Result<u32> {
        let Some(ri) = ctx.region_info.get(key) else {
            return Ok(0);
        };
        let meta = ctx
            .fmeta
            .get(&ri.owner)
            .ok_or_else(|| anyhow!("region {key}: owner {} has no faction meta", ri.owner))?;
        for gp in &meta.gov_posts {
            if gp.theatre == ri.theatre {
                return Ok(gp.cai);
            }
        }
        Ok(0)
    }

    /// REGION (world) child 22 = id of the GOVERNORSHIP (world object) that covers the region: the owner's governor post of the
    /// region's theatre. Returns the number of regions linked.
    fn link_regions(&self, ctx: &mut BuildContext) -> Result<usize> {
        let mut links: Vec<(usize, u32)> = Vec::with_capacity(ctx.region_info.len());
        for (key, ri) in &ctx.region_info {
            let meta = ctx
                .fmeta
                .get(&ri.owner)
                .ok_or_else(|| anyhow!("region {key}: owner {} has no faction meta", ri.owner))?;
            let mut gid = 0;
            for gp in &meta.gov_posts {
                if gp.theatre == ri.theatre {
                    gid = gp.gov_id;
                }
            }
            if gid == 0 {
                bail!(
                    "region {key}: owner {} has no governorship for theatre {}",
                    ri.owner,
                    ri.theatre
                );
            }
            links.push((ri.index, gid));
        }
        let n = links.len();
        for (index, gid) in links {
            let rec = ctx
                .regions
                .get_mut(index)
                .ok_or_else(|| anyhow!("region index {index} out of range"))?;
            let ch = kids_mut(rec)?;
            if !ch.get(22).map(is_u32).unwrap_or(false) {
                bail!(
                    "REGION child 22 is not a u32 governorship link: {:?}",
                    ch.get(22)
                );
            }
            put(ch, 22, u32n(gid))?;
        }
        Ok(n)
    }

    // ---------------------------------------------------------------------------------------------------------------
    fn build_entity_items(&mut self, ctx: &BuildContext) -> Result<()> {
        let mut chars: Vec<Vec<Node>> = Vec::new();
        let mut mobiles: Vec<Vec<Node>> = Vec::new();
        let mut units: Vec<Vec<Node>> = Vec::new();
        let mut settlement_of_region: BTreeMap<u32, u32> = BTreeMap::new();
        for sit_item in get_items(self.cw_e, &["CAI_WORLD_SETTLEMENTS"])? {
            let sit = kids(rec_by_name(sit_item, "CAI_SITUATED")?)?;
            settlement_of_region.insert(sc_at(sit, 2)?, item_id(sit_item)?);
        }
        // non-deployed characters
        let mut gov_of_char: BTreeMap<u32, u32> = BTreeMap::new();
        for k in &ctx.fkeys {
            for gp in &ctx.fmeta[k].gov_posts {
                gov_of_char.insert(gp.char_id, gp.cai);
            }
        }
        let mut leader_of_char: BTreeMap<u32, u32> = BTreeMap::new();
        for k in &ctx.fkeys {
            let frec = ctx
                .frec
                .get(k)
                .ok_or_else(|| anyhow!("no faction record for {k}"))?;
            for p in get_items(frec, &["GOVERNMENT", "POSTS_ARRAY"])? {
                let post = kids(
                    p.first()
                        .ok_or_else(|| anyhow!("empty POSTS_ARRAY item of {k}"))?,
                )?;
                if post.get(1).and_then(string_of) == Some("faction_leader") {
                    let c = sc_at(post, 2)?;
                    if c != 0 {
                        leader_of_char.insert(c, self.faction_cai[k]);
                    }
                }
            }
        }
        for k in &ctx.fkeys {
            let fc = self.faction_cai[k];
            for ch in &ctx.fmeta[k].chars {
                let tmpl = self.t_char.get(&ch.twid).ok_or_else(|| {
                    anyhow!(
                        "faction {k}: character template {} has no NTW CAI_CHARACTER",
                        ch.twid
                    )
                })?;
                let mut it: Vec<Node> = (*tmpl).clone();
                pristine_header(&mut it, Some(self.char_cai[&ch.wid]))?;
                set_owner(&mut it, "OWNED_INDIRECT", fc)?;
                let cc = spec_kids_mut(&mut it)?;
                put(cc, 3, u32n(ch.wid))?;
                put(cc, 0, u32n(0))?;
                put(cc, 1, u32n(0))?;
                put(cc, 2, u32n(0))?;
                // [4] = CAI governorship held by the character, [5] = owning faction if it is the faction leader
                put(cc, 4, u32n(gov_of_char.get(&ch.wid).copied().unwrap_or(0)))?;
                put(
                    cc,
                    5,
                    u32n(leader_of_char.get(&ch.wid).copied().unwrap_or(0)),
                )?;
                chars.push(it);
            }
        }
        let cells_items = get_items(self.cw_e, &["CAI_WORLD_SEA_GRID_CELLS"])?;
        for (fi, f) in ctx.forces.iter().enumerate() {
            let fc = *self
                .faction_cai
                .get(&f.faction)
                .ok_or_else(|| anyhow!("force of unknown faction {}", f.faction))?;
            // character
            let tch = self.t_char.get(&f.t_char).ok_or_else(|| {
                anyhow!(
                    "force of {}: NTW CAI_CHARACTER template {} missing",
                    f.faction,
                    f.t_char
                )
            })?;
            let mut it: Vec<Node> = (*tch).clone();
            pristine_header(&mut it, Some(f.cai_char))?;
            set_owner(&mut it, "OWNED_INDIRECT", fc)?;
            {
                let cc = spec_kids_mut(&mut it)?;
                put(cc, 0, u32n(f.cai_mobile))?;
                put(cc, 4, u32n(0))?;
                put(cc, 5, u32n(0))?;
                put(cc, 1, u32n(f.cai_units[0]))?;
                put(cc, 2, u32n(f.cai_mobile))?;
                put(cc, 3, u32n(f.wid_char))?;
            }
            chars.push(it);
            // mobile
            let tm = self.t_mobile.get(&f.t_army).ok_or_else(|| {
                anyhow!(
                    "force of {}: NTW CAI mobile template {} missing",
                    f.faction,
                    f.t_army
                )
            })?;
            let mut it: Vec<Node> = (*tm).clone();
            pristine_header(&mut it, Some(f.cai_mobile))?;
            set_owner(&mut it, "OWNED_DIRECT", fc)?;
            let kind = if f.kind == ForceKind::Navy { 6 } else { 4 };
            set_situated(&mut it, f.x, f.y, f.region_cai, f.theatre_cai, kind)?;
            {
                let settlement = if f.garrison {
                    *settlement_of_region.get(&f.region_cai).ok_or_else(|| {
                        anyhow!(
                            "garrison force of {} in {} has no CAI settlement",
                            f.faction,
                            f.region_at
                        )
                    })?
                } else {
                    0
                };
                let m = spec_kids_mut(&mut it)?;
                put(m, 0, u32n(f.cai_char))?;
                // [1] = CAI settlement the force garrisons (NTW: every mobile standing in a settlement; field forces / navies 0)
                put(m, 1, u32n(settlement))?;
                put(m, 4, arrn(&[f.cai_char]))?;
                put(m, 5, arrn(&f.cai_units))?;
                put(m, 10, u32n(f.wid_army))?;
                if f.kind == ForceKind::Navy {
                    let gi = match &self.force_cell[fi] {
                        Some(c) => {
                            let cell_item = cells_items
                                .get(c.2)
                                .ok_or_else(|| anyhow!("sea grid cell {} out of range", c.2))?;
                            vec![item_id(cell_item)?]
                        }
                        None => Vec::new(),
                    };
                    put(m, 15, arrn(&gi))?;
                }
            }
            mobiles.push(it);
            // units
            let tu = self.t_unit.get(&f.t_unit).ok_or_else(|| {
                anyhow!(
                    "force of {}: NTW CAI_UNIT template {} missing",
                    f.faction,
                    f.t_unit
                )
            })?;
            for (i, (uid, cid)) in f.unit_ids.iter().zip(f.cai_units.iter()).enumerate() {
                let mut it: Vec<Node> = (*tu).clone();
                pristine_header(&mut it, Some(*cid))?;
                let uc = spec_kids_mut(&mut it)?;
                put(uc, 0, u32n(if i == 0 { f.cai_char } else { 0 }))?;
                put(uc, 1, u32n(*uid))?;
                put(uc, 2, u32n(f.cai_mobile))?;
                units.push(it);
            }
        }
        self.char_items = chars;
        self.mobile_items = mobiles;
        self.unit_items = units;
        Ok(())
    }

    // ---------------------------------------------------------------------------------------------------------------
    fn build_world(&mut self, ctx: &BuildContext) -> Result<Node> {
        let cw_e = self.cw_e;
        let cw_n = self.cw_n;
        let e_arr = |name: &str| -> Result<&'m Vec<Vec<Node>>> { get_items(cw_e, &[name]) };

        // skeleton (headers, CAI_TIME, CAI_HISTORY ...) of the NTW world; rebuilt arrays start empty
        let mut cw = match cw_n {
            Node::Record {
                name,
                version,
                children,
            } => Node::Record {
                name: name.clone(),
                version: *version,
                children: children
                    .iter()
                    .map(|c| match c {
                        Node::Array { name, version, .. }
                            if WORLD_ARRAYS.contains(&name.as_str()) =>
                        {
                            empty_array_like(name, *version)
                        }
                        other => other.clone(),
                    })
                    .collect(),
            },
            _ => bail!("NTW CAI_WORLD is not a record"),
        };
        let mut out: BTreeMap<&'static str, Vec<Vec<Node>>> = BTreeMap::new();

        // ---- theatres
        let mut th_items = Vec::new();
        for it in e_arr("CAI_WORLD_THEATRES")? {
            let mut it = it.clone();
            let tid = item_id(&it)?;
            pristine_header(&mut it, Some(tid))?;
            let rec = spec_rec_mut(&mut it)?;
            if let Node::Record { version, .. } = &mut *rec {
                *version = 3;
            }
            let t = kids_mut(rec)?;
            put(t, 1, arrn(&sorted(self.theatre_mobiles.get(&tid))))?;
            if !TRANSITION_AREAS {
                put(t, 5, empty_arr())?;
            }
            t.push(u32n(2));
            th_items.push(it);
        }
        out.insert("CAI_WORLD_THEATRES", th_items);

        // ---- regions
        let mut reg_items = Vec::new();
        for it in e_arr("CAI_WORLD_REGIONS")? {
            let mut it = it.clone();
            let rid = item_id(&it)?;
            pristine_header(&mut it, Some(rid))?;
            let key = self
                .inv_region
                .get(&rid)
                .ok_or_else(|| anyhow!("CAI region {rid} has no key"))?
                .clone();
            let gov = self.region_gov_cai(ctx, &key)?;
            let mobs = sorted(self.region_mobiles.get(&rid));
            let present = self.region_present.get(&rid).cloned().unwrap_or_default();
            let r = spec_kids_mut(&mut it)?;
            put(r, 9, arrn(&mobs))?;
            put(r, 6, empty_arr())?; // ETW: fort slots; NTW: always empty
            put(r, 12, u32n(gov))?;
            put(r, 13, arrn(&present))?;
            reg_items.push(it);
        }
        out.insert("CAI_WORLD_REGIONS", reg_items);

        // ---- region boundaries / building slots / region slots / trading posts
        let pristine_all = |name: &str| -> Result<Vec<Vec<Node>>> {
            let mut v = Vec::new();
            for it in e_arr(name)? {
                let mut it = it.clone();
                let id = item_id(&it)?;
                pristine_header(&mut it, Some(id))?;
                v.push(it);
            }
            Ok(v)
        };
        out.insert(
            "CAI_WORLD_REGION_BOUNDARIES",
            pristine_all("CAI_WORLD_REGION_BOUNDARIES")?,
        );
        let mut dropped_bs: BTreeSet<u32> = BTreeSet::new();
        let mut bs_items = Vec::new();
        for it in e_arr("CAI_WORLD_BUILDING_SLOTS")? {
            let mut it = it.clone();
            if ctx.dropped_slots.contains(&sc_at(spec_kids(&it)?, 0)?) {
                dropped_bs.insert(item_id(&it)?);
                continue;
            }
            let id = item_id(&it)?;
            pristine_header(&mut it, Some(id))?;
            bs_items.push(it);
        }
        out.insert("CAI_WORLD_BUILDING_SLOTS", bs_items);
        out.insert(
            "CAI_WORLD_REGION_SLOTS",
            pristine_all("CAI_WORLD_REGION_SLOTS")?,
        );
        let mut tps = pristine_all("CAI_WORLD_TRADING_POSTS")?;
        for it in &mut tps {
            put(spec_kids_mut(it)?, 3, u32n(0))?; // occupying resource mobile (ETW 1700 ship ids) -> none
        }
        out.insert("CAI_WORLD_TRADING_POSTS", tps);

        // ---- settlements
        let mut st_items = Vec::new();
        for it in e_arr("CAI_WORLD_SETTLEMENTS")? {
            let mut it = it.clone();
            let id = item_id(&it)?;
            pristine_header(&mut it, Some(id))?;
            let region = sc_at(kids(rec_by_name(&it, "CAI_SITUATED")?)?, 2)?;
            let key = self
                .inv_region
                .get(&region)
                .ok_or_else(|| anyhow!("settlement {id}: CAI region {region} has no key"))?;
            let ri = ctx
                .region_info
                .get(key)
                .ok_or_else(|| anyhow!("settlement of region {key} has no region info"))?;
            let oc = *self
                .faction_cai
                .get(&ri.owner)
                .ok_or_else(|| anyhow!("region {key}: unknown owner {}", ri.owner))?;
            put(
                kids_mut(rec_by_name_mut(&mut it, "OWNED_DIRECT")?)?,
                0,
                u32n(oc),
            )?;
            let gm = match ri.garrison {
                Some(army) => self.mobile_cai.get(&army).copied().unwrap_or(0),
                None => 0,
            };
            {
                let g = kids_mut(rec_by_name_mut(&mut it, "CAI_GARRISONABLE")?)?;
                put(g, 0, u32n(gm))?;
                put(g, 2, empty_arr())?; // ETW 1700 garrison mobiles (NTW: empty)
            }
            {
                let cs = kids_mut(rec_by_name_mut(&mut it, "CAI_SETTLEMENT")?)?;
                let slots: Vec<u32> = arr_at(cs, 0)?
                    .into_iter()
                    .filter(|x| !dropped_bs.contains(x))
                    .collect();
                put(cs, 0, arrn(&slots))?;
                put(cs, 1, u32n(oc))?;
            }
            st_items.push(it);
        }
        out.insert("CAI_WORLD_SETTLEMENTS", st_items);

        // ---- factions / mobiles / units / characters / forts
        out.insert("CAI_WORLD_FACTIONS", self.faction_items(ctx)?);
        out.insert(
            "CAI_WORLD_RESOURCE_MOBILES",
            std::mem::take(&mut self.mobile_items),
        );
        out.insert("CAI_WORLD_UNITS", std::mem::take(&mut self.unit_items));
        out.insert("CAI_WORLD_CHARACTERS", std::mem::take(&mut self.char_items));
        out.insert("CAI_WORLD_FORTS", Vec::new());

        // ---- trade route segments: ETW faction / faction-hdr2 / technology tree / fort ids follow the re-allocated objects
        let mut etw_map: BTreeMap<u32, u32> = BTreeMap::new();
        let mut e_fac: BTreeMap<u32, String> = BTreeMap::new();
        for it in e_arr("CAI_WORLD_FACTIONS")? {
            let r_ = spec_kids(it)?;
            let pi =
                find_idx(r_, PROPSET).ok_or_else(|| anyhow!("ETW CAI_FACTION without PROPSET"))?;
            let etw_key = ctx
                .etw_fkey
                .get(&sc_at(r_, 6)?)
                .cloned()
                .unwrap_or_else(|| REBEL.to_string()); // not in FACTION_ARRAY = REBEL_FACTION
            // the ETW faction now stands for the faction the rules / mappings made of it ("" = no faction of the world)
            let key = ctx.etw_to_key.get(&etw_key).cloned().unwrap_or(etw_key);
            let id = item_id(it)?;
            etw_map.insert(id, self.faction_cai.get(&key).copied().unwrap_or(0)); // ETW-only factions (thirteen_colonies, pirates ...) -> 0
            etw_map.insert(
                sc_at(r_, pi + 1)?,
                self.faction_cai.get(&key).map(|v| v + 1).unwrap_or(0),
            );
            e_fac.insert(id, key);
        }
        for it in e_arr("CAI_WORLD_TECHNOLOGY_TREES")? {
            let owner = sc_at(kids(rec_by_name(it, "OWNED_DIRECT")?)?, 0)?;
            let key = e_fac.get(&owner).ok_or_else(|| {
                anyhow!("ETW technology tree owned by unknown CAI faction {owner}")
            })?;
            let v = if key != REBEL {
                self.tech_cai.get(key).copied().unwrap_or(0)
            } else {
                0
            };
            etw_map.insert(item_id(it)?, v);
        }
        for it in e_arr("CAI_WORLD_FORTS")? {
            etw_map.insert(item_id(it)?, 0); // forts do not exist in NTW
        }
        let mut segs = pristine_all("CAI_WORLD_TRADE_ROUTE_SEGMENTS")?;
        for it in &mut segs {
            let sr = spec_kids_mut(it)?;
            let v = sc_at(sr, 1)?;
            if let Some(m) = etw_map.get(&v) {
                put(sr, 1, u32n(*m))?;
            } else if dropped_bs.contains(&v) {
                put(sr, 1, u32n(0))?;
            }
        }
        out.insert("CAI_WORLD_TRADE_ROUTE_SEGMENTS", segs);
        out.insert(
            "CAI_WORLD_REGION_HLCIS",
            pristine_all("CAI_WORLD_REGION_HLCIS")?,
        );
        out.insert("CAI_WORLD_TRANSITION_AREAS", Vec::new());

        // ---- tech trees
        let tmpl_tt = get_items(cw_n, &["CAI_WORLD_TECHNOLOGY_TREES"])?
            .first()
            .ok_or_else(|| anyhow!("NTW has no technology tree template"))?;
        let mut tt = Vec::new();
        for k in &ctx.fkeys {
            let mut it = tmpl_tt.clone();
            pristine_header(&mut it, Some(self.tech_cai[k]))?;
            set_owner(&mut it, "OWNED_DIRECT", self.faction_cai[k])?;
            let tech = *ctx
                .tech_id
                .get(k)
                .ok_or_else(|| anyhow!("no technology tree id for {k}"))?;
            put(spec_kids_mut(&mut it)?, 0, u32n(tech))?;
            tt.push(it);
        }
        out.insert("CAI_WORLD_TECHNOLOGY_TREES", tt);

        // ---- governorships
        let tmpl_gv = get_items(cw_n, &["CAI_WORLD_GOVERNORSHIPS"])?
            .first()
            .ok_or_else(|| anyhow!("NTW has no governorship template"))?;
        let mut gv = Vec::new();
        for k in &ctx.fkeys {
            for gp in &ctx.fmeta[k].gov_posts {
                let mut it = tmpl_gv.clone();
                pristine_header(&mut it, Some(gp.cai))?;
                set_owner(&mut it, "OWNED_DIRECT", self.faction_cai[k])?;
                let theatre = *self.theatre_cai.get(&gp.theatre).ok_or_else(|| {
                    anyhow!("governorship of {k}: unknown theatre {}", gp.theatre)
                })?;
                let governor = *self.char_cai.get(&gp.char_id).ok_or_else(|| {
                    anyhow!(
                        "governorship of {k}/{}: governor {} has no CAI character",
                        gp.theatre,
                        gp.char_id
                    )
                })?;
                let regions = gp
                    .regions
                    .iter()
                    .map(|r| {
                        self.region_cai.get(r).copied().ok_or_else(|| {
                            anyhow!("governorship of {k}: region {r} has no CAI region")
                        })
                    })
                    .collect::<Result<Vec<u32>>>()?;
                let s = spec_kids_mut(&mut it)?;
                put(s, 0, u32n(gp.gov_id))?;
                put(s, 1, u32n(theatre))?;
                put(s, 2, u32n(governor))?;
                put(s, 3, arrn(&regions))?;
                gv.push(it);
            }
        }
        out.insert("CAI_WORLD_GOVERNORSHIPS", gv);

        // ---- sea grid cells
        let mut cells = Vec::new();
        for (gi, it) in e_arr("CAI_WORLD_SEA_GRID_CELLS")?.iter().enumerate() {
            let mut it = it.clone();
            let id = item_id(&it)?;
            pristine_header(&mut it, Some(id))?;
            let g = self
                .seagrid
                .iter()
                .find(|g| g.offset <= gi && gi < g.offset + g.cols * g.rows)
                .ok_or_else(|| anyhow!("sea grid cell {gi} belongs to no grid"))?;
            let loc = gi - g.offset;
            let ms = sorted(self.cell_mobiles.get(&(g.key.clone(), loc)));
            let s = spec_kids_mut(&mut it)?;
            let mob = s
                .iter_mut()
                .find(|c| matches!(c, Node::Array { .. }))
                .ok_or_else(|| anyhow!("sea grid cell without mobile array"))?;
            *arr_items_mut(mob)? = ms.iter().map(|m| vec![u32n(*m)]).collect();
            cells.push(it);
        }
        out.insert("CAI_WORLD_SEA_GRID_CELLS", cells);

        // ---- install the rebuilt arrays
        {
            let ch = kids_mut(&mut cw)?;
            for (name, items) in out {
                let idx =
                    find_idx(ch, name).ok_or_else(|| anyhow!("NTW CAI_WORLD has no {name}"))?;
                *arr_items_mut(&mut ch[idx])? = items;
            }
            // world header: belief id registry is empty with empty pools
            for c in ch.iter_mut() {
                if matches!(&*c, Node::Value(Value::Array { code: 0x48, data }) if data.len() > 1000)
                {
                    *c = empty_arr();
                }
            }
            // history: no events
            let h = ch
                .iter_mut()
                .find(|c| is_record_named(c, "CAI_HISTORY"))
                .ok_or_else(|| anyhow!("NTW CAI_WORLD has no CAI_HISTORY"))?;
            for c in kids_mut(h)?.iter_mut() {
                if let Node::Array { name, items, .. } = c {
                    if name == "CAI_HISTORY_EVENTS" {
                        items.clear();
                    }
                }
            }
        }
        Ok(cw)
    }

    // ---------------------------------------------------------------------------------------------------------------
    fn faction_items(&self, ctx: &BuildContext) -> Result<Vec<Vec<Node>>> {
        let nf = get_items(self.cw_n, &["CAI_WORLD_FACTIONS"])?;
        fn find_faction(nf: &Vec<Vec<Node>>, fc: u32) -> Result<&Vec<Node>> {
            for it in nf {
                if sc_at(kids(rec_by_name(it, "OWNED_DIRECT")?)?, 0)? == fc {
                    return Ok(it);
                }
            }
            bail!("NTW CAI_WORLD_FACTIONS has no faction {fc}")
        }
        let find = |fc: u32| find_faction(nf, fc);
        // template of an ordinary faction: france when the target start position has it (vanilla), else any other faction
        let tmpl_key = if self.ntw_fcai.contains_key("france") {
            "france"
        } else {
            self.ntw_fcai
                .keys()
                .find(|k| k.as_str() != REBEL)
                .map(String::as_str)
                .ok_or_else(|| {
                    anyhow!("target start position has no ordinary CAI faction to use as template")
                })?
        };
        let tmpl = find(
            *self
                .ntw_fcai
                .get(tmpl_key)
                .ok_or_else(|| anyhow!("NTW has no {tmpl_key} CAI faction"))?,
        )?;
        let reb = find(
            *self
                .ntw_fcai
                .get(REBEL)
                .ok_or_else(|| anyhow!("NTW has no rebel CAI faction"))?,
        )?;
        let mut items = Vec::with_capacity(self.fkeys.len());
        for k in &self.fkeys {
            let fc = self.faction_cai[k];
            let rebel = k == REBEL;
            let mut it = (if rebel { reb } else { tmpl }).clone();
            pristine_header(&mut it, Some(fc))?;
            put(
                kids_mut(rec_by_name_mut(&mut it, "OWNED_DIRECT")?)?,
                0,
                u32n(fc),
            )?;

            let (owned, thr, mobs, chrs, cap, gov_cais) =
                if rebel {
                    (
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        0u32,
                        Vec::new(),
                    )
                } else {
                    let meta = ctx
                        .fmeta
                        .get(k)
                        .ok_or_else(|| anyhow!("no faction meta for {k}"))?;
                    let info = ctx
                        .factions
                        .get(k)
                        .ok_or_else(|| anyhow!("no faction info for {k}"))?;
                    let mut owned = Vec::with_capacity(meta.owned.len());
                    let mut thr: BTreeSet<u32> = BTreeSet::new();
                    for r in &meta.owned {
                        owned.push(
                            *self
                                .region_cai
                                .get(r)
                                .ok_or_else(|| anyhow!("{k} owns {r} which has no CAI region"))?,
                        );
                        let ri = ctx
                            .region_info
                            .get(r)
                            .ok_or_else(|| anyhow!("{k} owns unknown region {r}"))?;
                        thr.insert(*self.theatre_cai.get(&ri.theatre).ok_or_else(|| {
                            anyhow!("region {r}: unknown theatre {}", ri.theatre)
                        })?);
                    }
                    owned.sort_unstable();
                    let cap = match (&info.capital, info.alive) {
                        (Some(c), true) => *self
                            .region_cai
                            .get(c)
                            .ok_or_else(|| anyhow!("capital {c} of {k} has no CAI region"))?,
                        _ => 0,
                    };
                    (
                        owned,
                        thr.into_iter().collect::<Vec<u32>>(),
                        sorted(self.faction_mobiles.get(k)),
                        sorted(self.faction_chars.get(k)),
                        cap,
                        meta.gov_posts.iter().map(|g| g.cai).collect::<Vec<u32>>(),
                    )
                };
            let hostile = if !rebel && cap != 0 {
                let mut h = vec![cap];
                for o in &ctx.fkeys {
                    if o != k && ctx.at_war(k, o) {
                        let oi = &ctx.factions[o];
                        if oi.alive {
                            if let Some(c) = &oi.capital {
                                h.push(*self.region_cai.get(c).ok_or_else(|| {
                                    anyhow!("capital {c} of {o} has no CAI region")
                                })?);
                            }
                        }
                    }
                }
                h
            } else {
                Vec::new()
            };
            let mut prim = 0;
            if !rebel {
                for f in &ctx.forces {
                    if &f.faction == k && f.kind == ForceKind::Army && f.role == "General" {
                        prim = f.cai_char; // NTW: CAI_WORLD_CHARACTERS id of the primary general
                        break;
                    }
                }
            }

            let ch = spec_kids_mut(&mut it)?;
            let pi = ch
                .iter()
                .position(|c| is_record_named(c, PROPSET))
                .ok_or_else(|| anyhow!("CAI_FACTION without PROPSET"))?;
            put(ch, 0, arrn(&owned))?;
            put(ch, 1, arrn(&thr))?;
            let marker: &[u32] = if owned.is_empty() { &[] } else { &[4] };
            put(ch, 2, arrn(marker))?;
            put(ch, 3, arrn(&mobs))?;
            put(ch, 4, arrn(&chrs))?;
            put(ch, 5, u32n(cap))?;
            put(
                ch,
                6,
                u32n(if rebel {
                    ctx.rebel_id
                } else {
                    *ctx.fid
                        .get(k)
                        .ok_or_else(|| anyhow!("no world id for faction {k}"))?
                }),
            )?;
            put(ch, 7, empty_arr())?;
            put(ch, 8, empty_arr())?;
            put(ch, 9, u32n(if rebel { 0 } else { self.tech_cai[k] }))?; // NTW rebels have no technology tree
            put(ch, 10, arrn(&gov_cais))?;
            // embedded header: scalars between the PROPSET record (pi) and CAI_BDI_NEW_TURN
            let ni = ch
                .iter()
                .position(|c| is_record_named(c, "CAI_BDI_NEW_TURN"))
                .ok_or_else(|| anyhow!("CAI_FACTION without CAI_BDI_NEW_TURN"))?;
            if ni <= pi {
                bail!("CAI_FACTION: CAI_BDI_NEW_TURN before PROPSET");
            }
            let mut hdr_item: Vec<Node> = Vec::with_capacity(ni - pi);
            hdr_item.push(ch[pi].clone());
            hdr_item.extend(ch[pi + 1..ni].iter().cloned());
            pristine_header(&mut hdr_item, Some(fc + 1))?;
            let _ = ch.splice(pi + 1..ni, hdr_item.into_iter().skip(1));
            let ni = ch
                .iter()
                .position(|c| is_record_named(c, "CAI_BDI_NEW_TURN"))
                .ok_or_else(|| anyhow!("CAI_FACTION lost CAI_BDI_NEW_TURN"))?;
            put(ch, ni + 1, u32n(cap))?;
            put(ch, ni + 2, arrn(&hostile))?;
            put(ch, ni + 3, u32n(prim))?;
            items.push(it);
        }
        Ok(items)
    }

    // ---------------------------------------------------------------------------------------------------------------
    /// `{root belief type: NTW root belief id}` referenced by an NTW faction pool; requires exactly one of each root type
    fn base_triple(&self, pool: &Node) -> Result<BTreeMap<String, u32>> {
        let mut found: BTreeMap<String, BTreeSet<u32>> = BTreeMap::new();
        fn visit(ch: &[Node], this: &CaiBuilder, found: &mut BTreeMap<String, BTreeSet<u32>>) {
            for c in ch {
                match c {
                    Node::Record { children, .. } => visit(children, this, found),
                    Node::Array { items, .. } => {
                        for it in items {
                            visit(it, this, found);
                        }
                    }
                    Node::Value(Value::U32(v)) => note(*v, this, found),
                    Node::Value(Value::Array { code: 0x48, data }) => {
                        for b in data.chunks_exact(4) {
                            note(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), this, found);
                        }
                    }
                    _ => {}
                }
            }
        }
        fn note(v: u32, this: &CaiBuilder, found: &mut BTreeMap<String, BTreeSet<u32>>) {
            if let Some(t) = this.ntw_root_type.get(&v) {
                found.entry(t.clone()).or_default().insert(v);
            }
        }
        visit(kids(pool)?, self, &mut found);
        let ok = found.len() == ROOT_TYPES.len()
            && ROOT_TYPES
                .iter()
                .all(|t| found.get(*t).map(|s| s.len() == 1).unwrap_or(false));
        if !ok {
            bail!(
                "faction pool root belief references are not exactly one of each root type: {:?}",
                found
            );
        }
        Ok(found
            .into_iter()
            .map(|(t, v)| (t, *v.iter().next().unwrap()))
            .collect())
    }

    /// CAI_DIPLOMATIC_ANALYSIS of faction `k` must hold one FACTIONINFO per OTHER faction (the AI planning round looks the target
    /// faction record up unconditionally), the subject list #20 = [k, others...] and the ally / war / relation lists.
    fn extend_diplomatic(&self, ctx: &BuildContext, k: &str, rb: &mut Vec<Node>) -> Result<()> {
        let others: Vec<u32> = self
            .fkeys
            .iter()
            .filter(|o| o.as_str() != k)
            .map(|o| self.faction_cai[o])
            .collect();
        {
            let spec = spec_kids_mut(rb)?;
            let fi_i = spec
                .iter()
                .position(|c| matches!(c, Node::Array { .. }))
                .ok_or_else(|| anyhow!("diplomatic analysis without FACTIONINFO array"))?;
            let old = arr_items(&spec[fi_i])?.clone();
            if old.len() != 41 {
                bail!(
                    "NTW diplomatic analysis holds {} FACTIONINFO entries, expected 41",
                    old.len()
                );
            }
            let mut items = Vec::with_capacity(others.len());
            for (j, o) in others.iter().enumerate() {
                let mut it = old[j % old.len()].clone();
                put(&mut it, 0, u32n(*o))?;
                items.push(it);
            }
            *arr_items_mut(&mut spec[fi_i])? = items;
        }
        let mut subjects = vec![self.faction_cai[k]];
        subjects.extend(others.iter().copied());
        put(rb, 20, arrn(&subjects))?;
        if k != REBEL {
            let fid_key: BTreeMap<u32, &String> =
                ctx.fkeys.iter().map(|o| (ctx.fid[o], o)).collect();
            let frec = ctx
                .frec
                .get(k)
                .ok_or_else(|| anyhow!("no faction record for {k}"))?;
            let mut state: BTreeMap<&str, String> = BTreeMap::new();
            for r_ in get_items(
                frec,
                &["DIPLOMACY_MANAGER", "DIPLOMACY_RELATIONSHIPS_ARRAY"],
            )? {
                let rc = kids(
                    r_.first()
                        .ok_or_else(|| anyhow!("empty diplomacy relationship of {k}"))?,
                )?;
                if let Some(o) = fid_key.get(&sc_at(rc, 0)?) {
                    if let Some(s) = rc.iter().find(|c| is_string16(c)).and_then(string_of) {
                        state.insert(o.as_str(), s.to_string());
                    }
                }
            }
            let st = |o: &str| state.get(o).map(String::as_str);
            let allies: Vec<u32> = self
                .fkeys
                .iter()
                .filter(|o| st(o) == Some("allied"))
                .map(|o| self.faction_cai[o])
                .collect();
            let war: Vec<u32> = self
                .fkeys
                .iter()
                .filter(|o| o.as_str() != k && (st(o) == Some("war") || o.as_str() == REBEL))
                .map(|o| self.faction_cai[o])
                .collect();
            let union: Vec<u32> = allies
                .iter()
                .chain(war.iter())
                .copied()
                .collect::<BTreeSet<u32>>()
                .into_iter()
                .collect();
            let spec = spec_kids_mut(rb)?;
            for (idx, lst) in [(3usize, &allies), (4, &war), (5, &allies), (6, &union)] {
                if !spec.get(idx).map(is_u32_arr).unwrap_or(false) {
                    bail!("diplomatic analysis child {idx} is not a u32 list");
                }
                put(spec, idx, arrn(lst))?;
            }
        }
        Ok(())
    }

    /// One attitude belief + management goal (+ trade status entry, analyser / manager lists, BLOCK_OWNS) per subject faction.
    /// Speculative and OFF in the accepted conversion (`CaiOptions::attitudes`).
    fn extend_attitudes(&mut self, k: &str, pool: &mut Node) -> Result<()> {
        if !self.options.attitudes || k == REBEL {
            return Ok(());
        }
        let pool_ch = kids_mut(pool)?;
        let bel_i = find_idx(pool_ch, "CAI_BDI_POOL_BELIEFS")
            .ok_or_else(|| anyhow!("pool without beliefs"))?;
        let des_i = find_idx(pool_ch, "CAI_BDI_POOL_DESIRES")
            .ok_or_else(|| anyhow!("pool without desires"))?;
        let mut bel = std::mem::take(arr_items_mut(&mut pool_ch[bel_i])?);
        let mut des = std::mem::take(arr_items_mut(&mut pool_ch[des_i])?);
        let result = self.extend_attitudes_inner(k, &mut bel, &mut des);
        *arr_items_mut(&mut pool_ch[bel_i])? = bel;
        *arr_items_mut(&mut pool_ch[des_i])? = des;
        result
    }

    fn extend_attitudes_inner(
        &mut self,
        k: &str,
        bel: &mut Vec<Vec<Node>>,
        des: &mut Vec<Vec<Node>>,
    ) -> Result<()> {
        let subjects: Vec<u32> = self
            .fkeys
            .iter()
            .filter(|o| o.as_str() != k && o.as_str() != REBEL)
            .map(|o| self.faction_cai[o])
            .collect();
        let mut bs: Vec<usize> = (0..bel.len())
            .filter(|i| spec_name(&bel[*i]).map(|n| n == ATT_B).unwrap_or(false))
            .collect();
        let mut gs: Vec<usize> = (0..des.len())
            .filter(|i| spec_name(&des[*i]).map(|n| n == ATT_G).unwrap_or(false))
            .collect();
        let ms: Vec<usize> = (0..des.len())
            .filter(|i| spec_name(&des[*i]).map(|n| n == ATT_M).unwrap_or(false))
            .collect();
        if bs.is_empty() || gs.is_empty() || ms.len() != 1 {
            bail!(
                "attitude pool of {k}: {} beliefs, {} goals, {} managers",
                bs.len(),
                gs.len(),
                ms.len()
            );
        }
        let m_i = ms[0];
        let bids: BTreeSet<u32> = bs
            .iter()
            .map(|i| sc_at(&bel[*i], 2))
            .collect::<Result<_>>()?;
        let a_cands: Vec<usize> = (0..des.len())
            .filter(|i| {
                spec_name(&des[*i])
                    .map(|n| n == "CAI_ANALYSER")
                    .unwrap_or(false)
                    && rec_by_name(&des[*i], "CAI_ANALYSER")
                        .and_then(kids)
                        .map(|c| {
                            c.first()
                                .map(|n| arr_vals(n).into_iter().collect::<BTreeSet<u32>>() == bids)
                                .unwrap_or(false)
                        })
                        .unwrap_or(false)
            })
            .collect();
        if a_cands.len() != 1 {
            bail!("attitude pool of {k}: {} attitude analysers", a_cands.len());
        }
        let a_i = a_cands[0];
        let ta: Vec<usize> = (0..bel.len())
            .filter(|i| {
                spec_name(&bel[*i])
                    .map(|n| n == "CAI_FACTION_TRADE_ANALYSIS")
                    .unwrap_or(false)
            })
            .collect();
        if ta.len() > 1 {
            bail!("attitude pool of {k}: several trade analyses");
        }
        let mut subj_of: BTreeMap<u32, u32> = BTreeMap::new();
        for i in &bs {
            subj_of.insert(sc_at(&bel[*i], 2)?, sc_at(spec_kids(&bel[*i])?, 0)?);
        }
        let present: BTreeSet<u32> = subj_of.values().copied().collect();
        let subj_set: BTreeSet<u32> = subjects.iter().copied().collect();
        if !present.is_subset(&subj_set) {
            bail!("attitude pool of {k}: beliefs for non subject factions");
        }
        let mut g_by_b: BTreeMap<u32, usize> = BTreeMap::new();
        for i in &gs {
            g_by_b.insert(sc_at(spec_kids(&des[*i])?, 2)?, *i);
        }
        if g_by_b.keys().copied().collect::<BTreeSet<u32>>() != bids {
            bail!("attitude pool of {k}: goals and beliefs do not pair up");
        }
        let t_b = bs[0];
        let t_b_id = sc_at(&bel[t_b], 2)?;
        let t_g = g_by_b[&t_b_id];
        let t_g_id = sc_at(&des[t_g], 2)?;
        let t_subject = subj_of[&t_b_id];
        let (tb_item, tg_item) = (bel[t_b].clone(), des[t_g].clone());
        for s in subjects.iter().filter(|x| !present.contains(x)) {
            let (nb, ng) = (self.cai(), self.cai());
            let mp: BTreeMap<u32, u32> = [(t_b_id, nb), (t_g_id, ng), (t_subject, *s)]
                .into_iter()
                .collect();
            let mut b_it = tb_item.clone();
            remap_children(&mut b_it, &mp);
            bel.push(b_it);
            bs.push(bel.len() - 1);
            let mut g_it = tg_item.clone();
            remap_children(&mut g_it, &mp);
            des.push(g_it);
            gs.push(des.len() - 1);
            subj_of.insert(nb, *s);
            g_by_b.insert(nb, des.len() - 1);
        }
        let b_order: Vec<u32> = bs
            .iter()
            .map(|i| sc_at(&bel[*i], 2))
            .collect::<Result<_>>()?;
        let owns_t = item_array(&des[a_i], 15)?
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("attitude analyser has no BLOCK_OWNS template"))?;
        let owns = |item: u32, owner: u32| -> Result<Vec<Node>> {
            let mut o = owns_t.clone();
            put(&mut o, 0, u32n(item))?;
            put(&mut o, 1, u32n(owner))?;
            Ok(o)
        };
        let g_order: Vec<usize> = b_order.iter().map(|b| g_by_b[b]).collect();
        let a_id = sc_at(&des[a_i], 2)?;
        {
            let a = &mut des[a_i];
            *item_array_mut(a, 15)? = b_order
                .iter()
                .map(|b| owns(*b, a_id))
                .collect::<Result<_>>()?;
            put(a, 17, u32n(b_order.len() as u32))?;
            let pairs: Vec<u32> = b_order.iter().flat_map(|b| [*b, subj_of[b]]).collect();
            put(a, 19, arrn(&pairs))?;
            let ca = kids_mut(rec_by_name_mut(a, "CAI_ANALYSER")?)?;
            put(ca, 0, arrn(&b_order))?;
            let sp: Vec<u32> = b_order.iter().flat_map(|b| [subj_of[b], *b]).collect();
            put(ca, 1, arrn(&sp))?;
        }
        let g_ids: Vec<u32> = g_order
            .iter()
            .map(|i| sc_at(&des[*i], 2))
            .collect::<Result<_>>()?;
        let m_id = sc_at(&des[m_i], 2)?;
        {
            let m = &mut des[m_i];
            *item_array_mut(m, 15)? = g_ids
                .iter()
                .map(|g| owns(*g, m_id))
                .collect::<Result<_>>()?;
            put(m, 17, u32n(g_ids.len() as u32))?;
            let t: Vec<u32> = g_ids
                .iter()
                .zip(b_order.iter())
                .flat_map(|(g, b)| [*g, *b, subj_of[b]])
                .collect();
            put(m, 19, arrn(&t))?;
            put(spec_kids_mut(m)?, 0, arrn(&g_ids))?;
        }
        for (i, (b, g)) in b_order.iter().zip(g_ids.iter()).enumerate() {
            let bi = *bs
                .iter()
                .find(|x| sc_at(&bel[**x], 2).map(|v| v == *b).unwrap_or(false))
                .ok_or_else(|| anyhow!("belief {b} vanished"))?;
            put(&mut bel[bi], 10, arrn(&[a_id, i as u32, *g, 0]))?;
            put(&mut des[g_by_b[b]], 10, arrn(&[m_id, i as u32]))?;
        }
        // trade analysis: one CAI_FACTION_TRADE_STATUS per subject
        if let Some(t) = ta.first() {
            let spec = spec_kids_mut(&mut bel[*t])?;
            let si = spec
                .iter()
                .position(|c| matches!(c, Node::Array { .. }))
                .ok_or_else(|| anyhow!("trade analysis without status array"))?;
            let st = arr_items_mut(&mut spec[si])?;
            let mut have: BTreeMap<u32, Vec<Node>> = BTreeMap::new();
            for x in st.iter() {
                have.insert(sc_at(kids(&x[0])?, 1)?, x.clone());
            }
            for s in &subjects {
                if !have.contains_key(s) {
                    let mut x = st
                        .first()
                        .ok_or_else(|| anyhow!("empty trade status array"))?
                        .clone();
                    put(kids_mut(&mut x[0])?, 1, u32n(*s))?;
                    have.insert(*s, x);
                }
            }
            *st = subjects.iter().map(|s| have[s].clone()).collect();
        }
        Ok(())
    }

    fn faction_manager(
        &mut self,
        ctx: &BuildContext,
        k: &str,
        base_mgr: &Vec<Node>,
        base_fcai: u32,
        other_map: &BTreeMap<u32, u32>,
    ) -> Result<Vec<Node>> {
        let mut item: Vec<Node> = base_mgr.clone();
        if item.len() != 3 {
            bail!(
                "faction manager item has {} parts, expected pool / pool-facade / manager",
                item.len()
            );
        }
        let base_trip = self
            .base_triple(&item[0])
            .with_context(|| format!("faction manager of {k}"))?;
        let mut mapping: BTreeMap<u32, u32> = BTreeMap::new();
        for arrn_ in [
            "CAI_BDI_POOL_BELIEFS",
            "CAI_BDI_POOL_DESIRES",
            "CAI_BDI_POOL_INTENTIONS",
        ] {
            for it in get_items(&item[0], &[arrn_])? {
                let id = sc_at(it, 2)?;
                let new = self.cai();
                mapping.insert(id, new);
            }
        }
        let mut trip: BTreeMap<&str, u32> = BTreeMap::new();
        for t in ROOT_TYPES {
            let new = self.cai();
            trip.insert(t, new);
        }
        for t in ROOT_TYPES {
            mapping.insert(base_trip[t], trip[t]);
        }
        mapping.insert(base_fcai, self.faction_cai[k]);
        for (a, b) in other_map {
            mapping.insert(*a, *b);
        }
        for part in item.iter_mut() {
            remap_children(kids_mut(part)?, &mapping);
        }
        // the faction's three root beliefs: clones of the base faction's, same id mapping; references to any other NTW root
        // belief (e.g. the relation analysis of the base faction) are dropped (those beliefs do not exist in our root pool)
        let keep: BTreeSet<u32> = base_trip.values().copied().collect();
        let drop: BTreeSet<u32> = self
            .ntw_root_b
            .keys()
            .copied()
            .filter(|x| !keep.contains(x))
            .collect();
        for t in ROOT_TYPES {
            let mut rb: Vec<Node> = self.ntw_root_b[&base_trip[t]].clone();
            scrub_arrays(&mut rb, &drop);
            remap_children(&mut rb, &mapping);
            if sc_at(&rb, 2)? != trip[t] {
                bail!(
                    "root belief {t}: id {} was not remapped to {}",
                    sc_at(&rb, 2)?,
                    trip[t]
                );
            }
            // header #10 = [(provider id, position)] pairs: "I am entry <position> of <provider>'s belief list"; the provider is the root
            // analyser whose list we fill in the same order
            let pr = arr_at(&rb, 10)?;
            if pr.len() != 2 {
                bail!(
                    "root belief {t}: header #10 holds {} values, expected one (provider, position) pair",
                    pr.len()
                );
            }
            let position = self.analyser_entries.get(t).map(Vec::len).unwrap_or(0) as u32;
            put(&mut rb, 10, arrn(&[pr[0], position]))?;
            if t == "CAI_DIPLOMATIC_ANALYSIS" {
                self.extend_diplomatic(ctx, k, &mut rb)?;
            }
            self.new_root_beliefs.push(rb);
            self.analyser_entries
                .entry(t.to_string())
                .or_default()
                .push((self.faction_cai[k], trip[t]));
        }
        // theatre references
        let eur = *self
            .theatre_cai
            .get("2")
            .ok_or_else(|| anyhow!("no europe theatre"))?;
        retarget_theatres(&mut item[0], eur)?;
        self.extend_attitudes(k, &mut item[0])?;
        // faction manager: own faction id
        put(kids_mut(&mut item[2])?, 0, u32n(self.faction_cai[k]))?;
        Ok(item)
    }

    /// One root belief per CAI_WORLD region under root analysers #11 (CAI_BORDER_PATROL_ANALYSIS) and #20
    /// (CAI_REGION_OCCUPANCY_ANALYSIS), exactly as NTW. The analyser keeps a hash table region cai id -> belief that the first AI round
    /// looks up unconditionally.
    fn add_region_beliefs(&mut self, rp: &mut Node) -> Result<()> {
        let regions: Vec<u32> = get_items(self.cw_e, &["CAI_WORLD_REGIONS"])?
            .iter()
            .map(|it| item_id(it))
            .collect::<Result<_>>()?;
        let desires = arr_items_mut(get_mut(rp, &["CAI_BDI_POOL_DESIRES"])?)?;
        for it in desires.iter_mut() {
            let d = sc_at(it, 2)?;
            let Some(tmpl) = self.region_tmpl.get(&d).cloned() else {
                continue;
            };
            let expect = REGION_ROOT
                .iter()
                .find(|(id, _)| *id == d)
                .map(|(_, n)| *n)
                .ok_or_else(|| anyhow!("unexpected region root desire {d}"))?;
            if spec_name(&tmpl)? != expect {
                bail!(
                    "region root desire {d}: template is {}, expected {expect}",
                    spec_name(&tmpl)?
                );
            }
            let mut ents: Vec<(u32, u32)> = Vec::with_capacity(regions.len());
            for (i, r) in regions.iter().enumerate() {
                let mut b_ = tmpl.clone();
                let bid = self.cai();
                put(&mut b_, 2, u32n(bid))?;
                put(spec_kids_mut(&mut b_)?, 0, u32n(*r))?;
                put(&mut b_, 10, arrn(&[d, i as u32]))?;
                put(&mut b_, 14, arrn(&[d]))?;
                put(&mut b_, 19, empty_arr())?;
                let occupancy: Vec<u32> = if d == 20 { vec![*r] } else { Vec::new() };
                put(&mut b_, 20, arrn(&occupancy))?;
                put(&mut b_, 21, empty_arr())?;
                item_array_mut(&mut b_, 15)?.clear();
                put(&mut b_, 17, u32n(0))?;
                if d == 20 {
                    let sc = spec_kids_mut(&mut b_)?;
                    if !sc.get(1).map(is_u32_arr).unwrap_or(false) {
                        bail!("region occupancy belief child 1 is not a u32 list");
                    }
                    put(sc, 1, empty_arr())?;
                }
                self.new_root_beliefs.push(b_);
                self.region_beliefs += 1;
                ents.push((*r, bid));
            }
            let owns_tmpl = self
                .owns_tmpl
                .clone()
                .ok_or_else(|| anyhow!("no BLOCK_OWNS template found in the NTW root pool"))?;
            *item_array_mut(it, 15)? = ents
                .iter()
                .map(|(_, bid)| {
                    let mut o = owns_tmpl.clone();
                    put(&mut o, 0, u32n(*bid))?;
                    put(&mut o, 1, u32n(d))?;
                    Ok(o)
                })
                .collect::<Result<_>>()?;
            put(it, 17, u32n(ents.len() as u32))?;
            let bids: Vec<u32> = ents.iter().map(|(_, b)| *b).collect();
            put(it, 19, if d == 11 { arrn(&bids) } else { empty_arr() })?;
            let ca = kids_mut(rec_by_name_mut(it, "CAI_ANALYSER")?)?;
            put(ca, 0, arrn(&bids))?;
            let pairs: Vec<u32> = ents.iter().flat_map(|(r, b)| [*r, *b]).collect();
            put(ca, 1, arrn(&pairs))?;
        }
        Ok(())
    }

    fn build_interface(&mut self, ctx: &BuildContext) -> Result<Node> {
        let ci_n = self.ci_n;
        let world = self.build_world(ctx)?;

        // CAI_INTERFACE: everything the NTW baseline carries is kept, except the pieces rebuilt below (the world, the root pool
        // beliefs and the manager list are replaced, so they are not cloned in the first place)
        let (ci_name, ci_version, ci_children) = match ci_n {
            Node::Record {
                name,
                version,
                children,
            } => (name.clone(), *version, children),
            _ => bail!("NTW CAI_INTERFACE is not a record"),
        };
        let mut ch: Vec<Node> = Vec::with_capacity(ci_children.len());
        for c in ci_children {
            match c {
                Node::Record { name, .. } if name == "CAI_WORLD" => ch.push(world_placeholder()),
                Node::Record {
                    name,
                    version,
                    children,
                } if name == "CAI_BDI_POOL" => ch.push(Node::Record {
                    name: name.clone(),
                    version: *version,
                    children: children
                        .iter()
                        .map(|x| match x {
                            Node::Array { name, version, .. } if name == "CAI_BDI_POOL_BELIEFS" => {
                                empty_array_like(name, *version)
                            }
                            other => other.clone(),
                        })
                        .collect(),
                }),
                Node::Array { name, version, .. } if name == "CAI_INTERFACE_MANAGERS" => {
                    ch.push(empty_array_like(name, *version))
                }
                other => ch.push(other.clone()),
            }
        }
        let wi = ch
            .iter()
            .position(|c| is_record_named(c, "CAI_WORLD"))
            .ok_or_else(|| anyhow!("NTW CAI_INTERFACE has no CAI_WORLD"))?;
        ch[wi] = world;
        let rp_i = ch
            .iter()
            .position(|c| is_record_named(c, "CAI_BDI_POOL"))
            .ok_or_else(|| anyhow!("NTW CAI_INTERFACE has no root pool"))?;

        // root pool: beliefs = the per-faction ones created by the faction managers below; analysers reset (their belief lists
        // are rebuilt after the managers); header list #20 (= [3], constant provider) of every root desire is kept
        let mut analyser_of_type: BTreeMap<&str, u32> = BTreeMap::new();
        {
            let rp = &mut ch[rp_i];
            // per-region root belief templates (taken from the still unreset NTW analysers)
            for it in get_items(rp, &["CAI_BDI_POOL_DESIRES"])? {
                let d = sc_at(it, 2)?;
                if d == 11 || d == 20 {
                    let ids = arr_at(kids(rec_by_name(it, "CAI_ANALYSER")?)?, 0)?;
                    let mut cand = None;
                    for i_ in ids {
                        let b_ = self.ntw_root_b.get(&i_).ok_or_else(|| {
                            anyhow!("analyser {d}: belief {i_} not in the NTW root pool")
                        })?;
                        if item_array(b_, 15)?.is_empty() {
                            cand = Some((*b_).clone());
                            break;
                        }
                    }
                    self.region_tmpl.insert(
                        d,
                        cand.ok_or_else(|| {
                            anyhow!("analyser {d} has no belief without BLOCK_OWNS")
                        })?,
                    );
                }
            }
            if self.region_tmpl.keys().copied().collect::<Vec<u32>>() != vec![11, 20] {
                bail!("NTW root pool region analysers #11 / #20 not found");
            }
            let owns_tmpl_src: BTreeSet<u32> = {
                let mut s = BTreeSet::new();
                for it in get_items(rp, &["CAI_BDI_POOL_DESIRES"])? {
                    if !item_array(it, 15)?.is_empty() {
                        s.insert(sc_at(it, 2)?);
                    }
                }
                s
            };
            for it in arr_items_mut(get_mut(rp, &["CAI_BDI_POOL_DESIRES"])?)?.iter_mut() {
                let did = sc_at(it, 2)?;
                let hdr19 = it.get(19).map(arr_vals).unwrap_or_default();
                for t in ROOT_TYPES {
                    if hdr19
                        .iter()
                        .any(|v| self.ntw_root_type.get(v).map(String::as_str) == Some(t))
                    {
                        analyser_of_type.insert(t, did);
                    }
                }
                if owns_tmpl_src.contains(&did) && self.owns_tmpl.is_none() {
                    // [belief id, desire id, f32, f32, u32]
                    self.owns_tmpl = item_array(it, 15)?.first().cloned();
                }
                pristine_hdr_keep_id(it)?;
                put(it, 17, u32n(0))?; // header u32 #17 = number of BLOCK_OWNS items
                for s in it.iter_mut() {
                    if let Node::Record { name, children, .. } = s {
                        if name.as_str() == PROPSET {
                            continue;
                        }
                        for c in children.iter_mut() {
                            if is_u32_arr(c) {
                                // only CAI_ANALYSER carries lists (belief ids, pairs)
                                *c = empty_arr();
                            }
                        }
                    }
                }
            }
        }
        if analyser_of_type.len() != ROOT_TYPES.len() {
            bail!(
                "root analysers of {:?} not found: {:?}",
                ROOT_TYPES,
                analyser_of_type
            );
        }
        // managers
        let mg_n = get_items(ci_n, &["CAI_INTERFACE_MANAGERS"])?;
        let base = mg_n
            .get(BASE_MANAGER_INDEX)
            .ok_or_else(|| anyhow!("NTW has no base faction manager {BASE_MANAGER_INDEX}"))?;
        let pairs: Vec<u32> = ci_children
            .iter()
            .find(
                |c| matches!(c, Node::Value(Value::Array { code: 0x48, data }) if data.len() > 100),
            )
            .map(arr_vals)
            .ok_or_else(|| anyhow!("NTW CAI_INTERFACE has no manager pair list"))?;
        let ntw_order: Vec<u32> = pairs.iter().step_by(2).copied().collect();
        let base_fcai = *ntw_order
            .get(BASE_MANAGER_INDEX)
            .ok_or_else(|| anyhow!("manager pair list too short"))?;
        let ntw_others: Vec<u32> = ntw_order
            .iter()
            .copied()
            .filter(|x| *x != base_fcai)
            .collect();
        let mut mgrs: Vec<Vec<Node>> = Vec::with_capacity(self.fkeys.len());
        let mut new_pairs: Vec<u32> = Vec::new();
        let fkeys = self.fkeys.clone();
        for k in &fkeys {
            let others: Vec<u32> = fkeys
                .iter()
                .filter(|o| *o != k)
                .map(|o| self.faction_cai[o])
                .collect();
            if k == REBEL {
                // rebels use NTW's own rebel manager (type 2, no AI flag): the generic type-11 manager fails its post-load
                let r_base = mg_n.last().ok_or_else(|| anyhow!("NTW has no managers"))?;
                let r_fcai = *ntw_order
                    .last()
                    .ok_or_else(|| anyhow!("manager pair list empty"))?;
                let r_others: Vec<u32> =
                    ntw_order.iter().copied().filter(|x| *x != r_fcai).collect();
                let other_map: BTreeMap<u32, u32> = r_others
                    .iter()
                    .enumerate()
                    .map(|(i, old)| (*old, others[i % others.len()]))
                    .collect();
                let m = self.faction_manager(ctx, k, r_base, r_fcai, &other_map)?;
                mgrs.push(m);
                new_pairs.push(self.faction_cai[k]);
                new_pairs.push(
                    *pairs
                        .last()
                        .ok_or_else(|| anyhow!("manager pair list empty"))?,
                );
                continue;
            }
            // NTW has (n_ntw-1) other factions in the baseline; map them onto ours (extra ours are not attitudes-tracked)
            let mut other_map: BTreeMap<u32, u32> = BTreeMap::new();
            for (i, old) in ntw_others.iter().enumerate() {
                other_map.insert(
                    *old,
                    if others.is_empty() {
                        *old
                    } else {
                        others[i % others.len()]
                    },
                );
            }
            let m = self.faction_manager(ctx, k, base, base_fcai, &other_map)?;
            mgrs.push(m);
            new_pairs.push(self.faction_cai[k]);
            new_pairs.push(11);
        }
        {
            let arr_i = ch
                .iter()
                .position(|c| matches!(c, Node::Value(Value::Array { code: 0x48, data }) if data.len() > 100))
                .ok_or_else(|| anyhow!("CAI_INTERFACE has no manager pair list"))?;
            ch[arr_i] = arrn(&new_pairs);
            let mi = find_idx(&ch, "CAI_INTERFACE_MANAGERS")
                .ok_or_else(|| anyhow!("CAI_INTERFACE has no managers array"))?;
            *arr_items_mut(&mut ch[mi])? = mgrs;
        }
        // root pool: add the per-region and per-faction root beliefs and register them in their analysers
        {
            let mut rp = std::mem::replace(&mut ch[rp_i], world_placeholder());
            self.add_region_beliefs(&mut rp)?;
            self.new_root_beliefs_len = self.new_root_beliefs.len();
            *arr_items_mut(get_mut(&mut rp, &["CAI_BDI_POOL_BELIEFS"])?)? =
                std::mem::take(&mut self.new_root_beliefs);
            let owns_tmpl = self
                .owns_tmpl
                .clone()
                .ok_or_else(|| anyhow!("no BLOCK_OWNS template found in the NTW root pool"))?;
            for it in arr_items_mut(get_mut(&mut rp, &["CAI_BDI_POOL_DESIRES"])?)?.iter_mut() {
                let did = sc_at(it, 2)?;
                for t in ROOT_TYPES {
                    if analyser_of_type[t] != did {
                        continue;
                    }
                    let ents = self.analyser_entries.get(t).cloned().unwrap_or_default();
                    let bids: Vec<u32> = ents.iter().map(|(_, b)| *b).collect();
                    put(it, 19, arrn(&bids))?;
                    // BLOCK_OWNS: one [belief id, this desire id, 0.0, 0.0, 1] per belief, in list order; the belief's header pair
                    // (desire id, position) indexes it
                    let mut owns: Vec<Vec<Node>> = Vec::with_capacity(ents.len());
                    for b_ in &bids {
                        let mut o = owns_tmpl.clone();
                        put(&mut o, 0, u32n(*b_))?;
                        put(&mut o, 1, u32n(did))?;
                        owns.push(o);
                    }
                    *item_array_mut(it, 15)? = owns;
                    put(it, 17, u32n(ents.len() as u32))?;
                    // base record of every analyser: [belief ids], [(world object id, belief id)]
                    let s = kids_mut(rec_by_name_mut(it, "CAI_ANALYSER")?)?;
                    let arrs: Vec<usize> = s
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| is_u32_arr(c))
                        .map(|(i, _)| i)
                        .collect();
                    if arrs.len() != 2 {
                        bail!(
                            "CAI_ANALYSER of root desire {did} has {} u32 lists, expected 2",
                            arrs.len()
                        );
                    }
                    put(s, arrs[0], arrn(&bids))?;
                    let pairs: Vec<u32> = ents.iter().flat_map(|(f_, b_)| [*f_, *b_]).collect();
                    put(s, arrs[1], arrn(&pairs))?;
                }
            }
            ch[rp_i] = rp;
        }
        // invasion tracking records reference NTW factions -> none at start
        {
            let inv = ch
                .iter_mut()
                .find(|c| is_record_named(c, "CAI_INVASION_TRACKING_SYSTEM_INFORMATION"))
                .ok_or_else(|| anyhow!("CAI_INTERFACE has no invasion tracking record"))?;
            for c in kids_mut(inv)?.iter_mut() {
                if let Node::Array { items, .. } = c {
                    items.clear();
                }
            }
        }
        // director pool (NTW ids 3998..4013, referenced only inside the pool): they collide with ETW's CAI_WORLD_TRADE_ROUTE_SEGMENTS
        // ids (NTW has no duplicate CAI ids), so give them fresh ids
        {
            let di = ch
                .iter()
                .position(|c| is_record_named(c, "CAI_INTERFACE_DIRECTOR_BDI_POOL"))
                .ok_or_else(|| anyhow!("CAI_INTERFACE has no director pool"))?;
            let mut dmap: BTreeMap<u32, u32> = BTreeMap::new();
            for arrn_ in [
                "CAI_BDI_POOL_BELIEFS",
                "CAI_BDI_POOL_DESIRES",
                "CAI_BDI_POOL_INTENTIONS",
            ] {
                for it in get_items(&ch[di], &["CAI_BDI_POOL", arrn_])? {
                    let id = sc_at(it, 2)?;
                    let new = self.cai();
                    dmap.insert(id, new);
                }
            }
            for part in kids_mut(&mut ch[di])?.iter_mut() {
                match part {
                    Node::Record { children, .. } => remap_children(children, &dmap),
                    Node::Array { items, .. } => {
                        for it in items.iter_mut() {
                            remap_children(it, &dmap);
                        }
                    }
                    scalar => remap_children(std::slice::from_mut(scalar), &dmap),
                }
            }
            self.dir_map = dmap;
        }
        Ok(Node::Record {
            name: ci_name,
            version: ci_version,
            children: ch,
        })
    }
}

fn world_placeholder() -> Node {
    Node::Record {
        name: "CAI_WORLD".to_string(),
        version: 0,
        children: Vec::new(),
    }
}

/// `CAI_MILITARY_RANK_THEATRE_ANALYSIS` / `CAI_BDI_GOAL_NAVY_DISTRIBUTION` carry a hard-wired NTW theatre id 28 -> our europe theatre
fn retarget_theatres(pool: &mut Node, eur: u32) -> Result<()> {
    for arrn_ in ["CAI_BDI_POOL_BELIEFS", "CAI_BDI_POOL_DESIRES"] {
        for it in arr_items_mut(get_mut(pool, &[arrn_])?)?.iter_mut() {
            let rec = spec_rec_mut(it)?;
            let is_target = matches!(
                name_of(rec),
                Some("CAI_MILITARY_RANK_THEATRE_ANALYSIS") | Some("CAI_BDI_GOAL_NAVY_DISTRIBUTION")
            );
            if is_target {
                for c in kids_mut(rec)?.iter_mut() {
                    if matches!(c, Node::Value(Value::U32(28))) {
                        *c = u32n(eur);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Sea grid parameters per theatre (cols, rows, size, p0, p1) from the generated `sea_grids.esf`.
fn parse_seagrid(root: &Node, theatre_cai: &BTreeMap<String, u32>) -> Result<Vec<SeaGrid>> {
    let ch = kids(root)?;
    let mut out: Vec<SeaGrid> = Vec::new();
    for p in 0..ch.len() {
        let Node::Value(Value::String16(key)) = &ch[p] else {
            continue;
        };
        if !theatre_cai.contains_key(key) || p + 5 >= ch.len() {
            continue;
        }
        let (Node::Value(Value::XY(p0)), Node::Value(Value::XY(p1)), Node::Value(Value::F32(size))) =
            (&ch[p + 1], &ch[p + 2], &ch[p + 3])
        else {
            continue;
        };
        let cols = scalar(&ch[p + 4])
            .ok_or_else(|| anyhow!("sea grid {key}: columns are not an integer"))?
            as usize;
        let rows = scalar(&ch[p + 5])
            .ok_or_else(|| anyhow!("sea grid {key}: rows are not an integer"))?
            as usize;
        if cols == 0 || rows == 0 {
            bail!("sea grid {key} has {cols}x{rows} cells");
        }
        out.push(SeaGrid {
            key: key.clone(),
            p0: [p0[0] as f64, p0[1] as f64],
            p1: [p1[0] as f64, p1[1] as f64],
            size: *size as f64,
            cols,
            rows,
            offset: 0,
        });
    }
    let mut off = 0;
    for g in &mut out {
        g.offset = off;
        off += g.cols * g.rows;
    }
    if out.is_empty() {
        bail!("The generated sea_grids.esf holds no sea grid for the CAI theatres");
    }
    Ok(out)
}
