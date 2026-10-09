//! Deployed armies / navies (ARMY/NAVY records + leader CHARACTER) from `rules.armies`. Port of `forces_b.py` and
//! `entities.make_unit`, plus the displacement policy for ownership edits: a starting force whose region is no longer owned by
//! its faction is rehomed deterministically to an owned region (same theatre preferred, then nearest settlement); a fleet
//! needs an owned region with a port, otherwise the build fails with an explicit error.

use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow, bail};
use regex::Regex;

use crate::esf::Node;

use super::context::{BuildContext, Force, ForceKind};
use super::factions::{char_info, clear_char_motion, set_char_pos};
use super::helpers::*;

static GUNS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"_(\d{2,3})(?:_|$)").expect("static regex"));

/// land-force offsets from the settlement point (world units)
static OFFSETS: LazyLock<Vec<(f64, f64)>> = LazyLock::new(|| {
    let mut v = Vec::new();
    for r in [3.0f64, 5.0, 7.0, 9.0, 12.0] {
        for a in (0..360).step_by(45) {
            let rad = (a as f64).to_radians();
            v.push((r * rad.cos(), r * rad.sin()));
        }
    }
    v
});

// ------------------------------------------------------------------------------------------------ units

/// Current and maximum strength of a land unit instance.
fn set_land_men(u: &mut Node, key: &str, men: u32) -> Result<()> {
    let un = kids_mut(get_mut(u, &["UNIT"])?)?;
    let idx = scalar_idx(un);
    if idx.len() < 3 {
        bail!("{key}: UNIT record has fewer than 3 scalars");
    }
    un[idx[1]] = u32v(men);
    un[idx[2]] = u32v(men);
    Ok(())
}

/// Strength and crew composition (marines, seamen, gunners) of a ship instance.
fn set_ship_crew(u: &mut Node, key: &str, (mar, sea, gun): (u32, u32, u32)) -> Result<()> {
    set_land_men(u, key, mar + sea + gun)?;
    let sv = kids_mut(get_mut(u, &["SHIP_DAMAGE_INFO"])?)?;
    let ii = scalar_idx(sv);
    if ii.len() < 11 {
        bail!("{key}: SHIP_DAMAGE_INFO has fewer than 11 scalars");
    }
    for (slot, v) in [(5, mar), (6, sea), (7, gun), (8, mar), (9, sea), (10, gun)] {
        sv[ii[slot]] = i32v(v as i32);
    }
    Ok(())
}

/// Clone a unit instance (LAND_UNIT / NAVAL_UNIT) for `key` using a template instance.
pub fn make_unit(
    tmpl: &Templates,
    db: &UnitDb,
    key: &str,
    new_id: u32,
    leader_id: u32,
    fkey: &str,
) -> Result<Node> {
    let is_naval_key = !db.land_men.contains_key(key);
    let mut u: Node;
    if let Some(lst) = tmpl.unit_inst.get(key) {
        u = lst[0].clone();
        // the template carries the original unit's counts: follow the target's tables where it changed them
        if is_naval_key {
            if let Some(&crew) = db.naval_changed.get(key) {
                set_ship_crew(&mut u, key, crew)?;
            }
        } else if let Some(&men) = db.land_changed.get(key) {
            set_land_men(&mut u, key, men)?;
        }
    } else {
        // analogous template (same prefix and kind), then set key + counts
        let pr = unit_prefix(key);
        let same_kind = |k: &&String| db.land_men.contains_key(k.as_str()) != is_naval_key;
        let cand = tmpl
            .unit_order
            .iter()
            .filter(same_kind)
            .find(|k| unit_prefix(k) == pr)
            .or_else(|| tmpl.unit_order.iter().find(same_kind))
            .ok_or_else(|| anyhow!("no template unit instance for {key}"))?;
        u = tmpl.unit_inst[cand][0].clone();
        {
            let first = kids_mut(&mut u)?
                .first_mut()
                .ok_or_else(|| anyhow!("unit without record key"))?;
            kids_mut(first)?[0] = s16(key);
        }
        let un = kids_mut(get_mut(&mut u, &["UNIT"])?)?;
        kids_mut(&mut un[0])?[0] = s16(key);
        let idx = scalar_idx(un);
        if idx.len() < 11 {
            bail!("{key}: UNIT record has fewer than 11 scalars");
        }
        let men = if !is_naval_key {
            db.land_men[key]
        } else {
            let c = db.naval_crew.get(key).ok_or_else(|| {
                anyhow!("unit {key} is neither in unit_stats_land nor unit_stats_naval")
            })?;
            c.0 + c.1 + c.2
        };
        un[idx[1]] = u32v(men);
        un[idx[2]] = u32v(men);
        if let Some((a, bb)) = tmpl.pair_for_prefix(&pr) {
            un[idx[3]] = i32v(a);
            un[idx[10]] = i32v(bb);
        }
        if is_naval_key {
            let (mar, sea, gun) = db.naval_crew[key];
            let guns: Option<u32> = GUNS_RE.captures(key).and_then(|c| c[1].parse().ok());
            let sv = kids_mut(get_mut(&mut u, &["SHIP_DAMAGE_INFO"])?)?;
            let ii = scalar_idx(sv);
            if ii.len() < 14 {
                bail!("{key}: SHIP_DAMAGE_INFO has fewer than 14 scalars");
            }
            sv[ii[5]] = i32v(mar as i32);
            sv[ii[6]] = i32v(sea as i32);
            sv[ii[7]] = i32v(gun as i32);
            sv[ii[8]] = i32v(mar as i32);
            sv[ii[9]] = i32v(sea as i32);
            sv[ii[10]] = i32v(gun as i32);
            if let Some(g) = guns {
                sv[ii[11]] = u32v(g);
                sv[ii[13]] = u32v(g);
            }
        }
    }
    let un = kids_mut(get_mut(&mut u, &["UNIT"])?)?;
    let idx = scalar_idx(un);
    if idx.len() < 7 {
        bail!("{key}: UNIT record has fewer than 7 scalars");
    }
    un[idx[0]] = i32_from_id(new_id);
    un[idx[6]] = u32v(leader_id);
    kids_mut(child_mut(un, "COMMANDER_DETAILS")?)?[2] = s16(fkey);
    Ok(u)
}

// ------------------------------------------------------------------------------------------------ ports / placement

/// port slot positions (name, x, y) of a converted region record
pub fn port_positions(rec: &Node) -> Result<Vec<(String, i32, i32)>> {
    let mut out = Vec::new();
    for it in items(get(rec, &["REGION_SLOT_MANAGER", "REGION_SLOT_ARRAY"])?)? {
        let sv = kids(&it[0])?;
        let name = nth_text(sv, 1)?;
        if name.starts_with("port:") {
            let x = as_number(nth_val(sv, 4)?).ok_or_else(|| anyhow!("port slot x"))? as i32;
            let y = as_number(nth_val(sv, 5)?).ok_or_else(|| anyhow!("port slot y"))? as i32;
            out.push((name.to_string(), x, y));
        }
    }
    Ok(out)
}

pub struct ForceBuilder<'a> {
    tmpl: &'a Templates,
    db: &'a UnitDb,
    map: &'a GlobalMap,
    cursor: HashMap<(String, String), usize>,
    taken: HashMap<String, Vec<(i32, i32)>>,
}

impl<'a> ForceBuilder<'a> {
    pub fn new(tmpl: &'a Templates, db: &'a UnitDb, map: &'a GlobalMap) -> Self {
        Self {
            tmpl,
            db,
            map,
            cursor: HashMap::new(),
            taken: HashMap::new(),
        }
    }

    fn land_position(&mut self, ctx: &BuildContext, region: &str, k: &str) -> (i32, i32, bool) {
        let ri = &ctx.region_info[region];
        let (px, py) = ri.pos;
        let used = self.taken.entry(region.to_string()).or_default();
        if ri.garrison.is_none() && ri.owner == k {
            used.push((px, py));
            return (px, py, true);
        }
        for (dx, dy) in OFFSETS.iter() {
            let x = px + fixed(*dx);
            let y = py + fixed(*dy);
            match self.map.locate(unfixed(x), unfixed(y)) {
                Some((key, "land")) if key == region => {}
                _ => continue,
            }
            if used
                .iter()
                .all(|(ux, uy)| (unfixed(x - ux)).hypot(unfixed(y - uy)) >= 2.5)
            {
                used.push((x, y));
                return (x, y, false);
            }
        }
        used.push((px, py));
        (px, py, false)
    }

    fn sea_position(&mut self, ctx: &BuildContext, region: &str) -> Result<(i32, i32, String)> {
        let ri = &ctx.region_info[region];
        let ports = port_positions(&ctx.regions[ri.index])?;
        let (px, py) = ri.pos;
        let used = self.taken.entry(format!("sea:{region}")).or_default();
        if ports.is_empty() {
            bail!("no port in {region}");
        }
        let (_, x0, y0) = &ports[used.len() % ports.len()];
        let (dx, dy) = (unfixed(x0 - px), unfixed(y0 - py));
        let n = dx.hypot(dy);
        let n = if n == 0.0 { 1.0 } else { n };
        let (ux, uy) = (dx / n, dy / n);
        for dist in [0.0f64, 4.0, 8.0, 12.0, 16.0, 20.0, 26.0, 32.0] {
            for side in [0.0f64, 4.0, -4.0, 8.0, -8.0] {
                let x = unfixed(*x0) + ux * dist - uy * side;
                let y = unfixed(*y0) + uy * dist + ux * side;
                let key = match self.map.locate(x, y) {
                    Some((key, "sea")) => key,
                    _ => continue,
                };
                let (fx, fy) = (fixed(x), fixed(y));
                if used
                    .iter()
                    .all(|(a, c)| unfixed(fx - a).hypot(unfixed(fy - c)) >= 4.0)
                {
                    used.push((fx, fy));
                    return Ok((fx, fy, key.to_string()));
                }
            }
        }
        bail!("no sea position near port of {region}")
    }

    /// Region a starting force is deployed to: its configured region while its faction still owns it, otherwise the
    /// nearest owned region (same theatre first; fleets only consider regions with a port).
    fn resolve_region(
        &self,
        ctx: &BuildContext,
        k: &str,
        region: &str,
        kind: ForceKind,
    ) -> Result<String> {
        let ri = ctx
            .region_info
            .get(region)
            .ok_or_else(|| anyhow!("{k}: force region {region} is not a region"))?;
        let navy = kind.is_navy();
        let has_port = |r: &str| -> Result<bool> {
            Ok(!port_positions(&ctx.regions[ctx.region_info[r].index])?.is_empty())
        };
        if ri.owner == k && (!navy || has_port(region)?) {
            return Ok(region.to_string());
        }
        let (ox, oy) = (unfixed(ri.pos.0), unfixed(ri.pos.1));
        let mut best: Option<((bool, u64), String)> = None;
        for cand in &ctx.fmeta[k].owned {
            let ci = &ctx.region_info[cand];
            if navy && !has_port(cand)? {
                continue;
            }
            let d = (unfixed(ci.pos.0) - ox).hypot(unfixed(ci.pos.1) - oy);
            let score = (ci.theatre != ri.theatre, (d * 1000.0).round() as u64);
            if best
                .as_ref()
                .map_or(true, |(s, key)| (score, cand) < (*s, key))
            {
                best = Some((score, cand.clone()));
            }
        }
        match best {
            Some((_, r)) => Ok(r),
            None if navy => bail!(
                "{k}: its starting fleet was based at {region}, which {k} no longer owns, and {k} owns no region with a port; \
                 give {k} a coastal region or reassign {region}"
            ),
            None => bail!("{k}: starting army at {region} cannot be relocated, {k} owns no region"),
        }
    }

    fn make_force(
        &mut self,
        ctx: &mut BuildContext,
        k: &str,
        entry: &serde_json::Value,
    ) -> Result<()> {
        let kind = match jstr(entry, "kind")? {
            "army" => ForceKind::Army,
            "navy" => ForceKind::Navy,
            other => bail!("{k}: unknown force kind {other}"),
        };
        let units: Vec<String> = entry
            .get("units")
            .and_then(|u| u.as_array())
            .ok_or_else(|| anyhow!("{k}: force without units"))?
            .iter()
            .map(|u| {
                u.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("{k}: unit key is not a string"))
            })
            .collect::<Result<_>>()?;
        if units.is_empty() {
            bail!("{k}: force without units");
        }
        // every unit must be of the force's kind in the effective (target) unit tables; a key that exists nowhere is an
        // unresolved reference, never silently replaced by a vanilla unit
        for key in &units {
            let land = self.db.land_men.contains_key(key);
            let naval = self.db.naval_crew.contains_key(key);
            let templated = self.tmpl.unit_inst.contains_key(key);
            let (ok, want) = match kind {
                ForceKind::Army => (land || (!naval && templated), "land"),
                ForceKind::Navy => (naval || (!land && templated), "naval"),
            };
            if !ok {
                let found = match (land, naval) {
                    (true, _) => "a land unit",
                    (_, true) => "a naval unit",
                    _ => "not defined in unit_stats_land / unit_stats_naval of the target database",
                };
                bail!(
                    "{k}: the starting {} uses unit {key}, which is {found}; map it to a {want} unit of the target with mappings.units[\"{key}\"]",
                    if kind.is_navy() { "fleet" } else { "army" }
                );
            }
        }
        let configured = jstr(entry, "region")?.to_string();
        let region = self.resolve_region(ctx, k, &configured, kind)?;
        if region != configured {
            let msg = format!(
                "Relocated {} of {k} from {configured} to {region}: {configured} is no longer owned by {k}",
                if kind.is_navy() { "fleet" } else { "army" }
            );
            ctx.notes.push(msg);
        }
        let general = entry
            .get("general")
            .and_then(|g| g.as_bool())
            .unwrap_or(false);
        let role = if kind == ForceKind::Army {
            if general { "General" } else { "colonel" }
        } else if units.len() >= 4 {
            "admiral"
        } else {
            "captain"
        };
        let donor = ctx.fmeta[k].donor.clone();
        let (tmpl, unit_db): (&'a Templates, &'a UnitDb) = (self.tmpl, self.db);
        let pool = tmpl.pick_pool(&donor, role)?;
        let cur = self
            .cursor
            .entry((k.to_string(), role.to_string()))
            .or_insert(0);
        let (tch, tarmy) = &pool[*cur % pool.len()];
        *cur += 1;
        let tinfo = char_info(tch)?;
        let t_army_id = nth_id(kids(get(tarmy, &["MILITARY_FORCE"])?)?, 0)?;
        let wid_char = ctx.ids.world();
        let wid_army = ctx.ids.world();
        let unit_ids: Vec<u32> = units.iter().map(|_| ctx.ids.world()).collect();
        let mapping = BTreeMap::from([
            (tinfo.wid, wid_char),
            (t_army_id, wid_army),
            (tinfo.unit, unit_ids[0]),
        ]);
        let mut ch = tch.clone();
        remap_ids(kids_mut(&mut ch)?, &mapping); // nested: CHARACTER_DETAILS holds the character id too
        // CHARACTER child[8] = id of the CHARACTER_POST held, child[9] = recruitment slot handle: a deployed leader holds neither
        {
            let c = kids_mut(&mut ch)?;
            for ix in [8usize, 9] {
                expect_code(c, ix, 8, "force leader")?;
                c[ix] = u32v(0);
            }
        }
        {
            let det = kids_mut(get_mut(&mut ch, &["CHARACTER_DETAILS"])?)?;
            for c in det.iter_mut() {
                if ncode(c) == Some(14) {
                    if let Some(s) = as_text(c) {
                        if ctx.factions.contains_key(s) {
                            *c = s16(k);
                        }
                    }
                }
            }
        }
        let (x, y, garrison, reg_at, sea_key) = if kind == ForceKind::Army {
            let (x, y, g) = self.land_position(ctx, &region, k);
            (x, y, g, region.clone(), None)
        } else {
            let (x, y, sk) = self.sea_position(ctx, &region)?;
            (x, y, false, sk.clone(), Some(sk))
        };
        set_char_pos(&mut ch, x, y)?;
        clear_char_motion(&mut ch)?;
        let mut army = tarmy.clone();
        {
            let a = kids_mut(&mut army)?;
            a[0] = Node::record(
                "MILITARY_FORCE",
                1,
                vec![u32v(wid_army), u32v(wid_char), arr_u32(&[])],
            );
            for c in a.iter_mut() {
                if ncode(c) == Some(4) {
                    *c = i32_from_id(wid_army);
                }
            }
        }
        let lnames: Vec<Node> = kids(get(&ch, &["CHARACTER_DETAILS"])?)?
            .iter()
            .filter(|c| is_record(c, "CAMPAIGN_LOCALISATION"))
            .take(2)
            .cloned()
            .collect();
        if lnames.len() < 2 {
            bail!("{k}: leader template lacks two CAMPAIGN_LOCALISATION records");
        }
        let mut new_items: Vec<Vec<Node>> = Vec::new();
        for (i, key) in units.iter().enumerate() {
            let mut u = make_unit(
                tmpl,
                unit_db,
                key,
                unit_ids[i],
                if i == 0 { wid_char } else { 0 },
                k,
            )
            .with_context(|| format!("{k}: unit {key}"))?;
            if i == 0 {
                let cd = kids_mut(get_mut(&mut u, &["UNIT", "COMMANDER_DETAILS"])?)?;
                cd[0] = lnames[0].clone();
                cd[1] = lnames[1].clone();
            }
            new_items.push(vec![u]);
        }
        *items_mut(get_mut(&mut army, &["UNITS_ARRAY"])?)? = new_items;
        // ARMY child[5] = id of the SIEGEABLE_GARRISON_RESIDENCE the army sits in (garrison force) / 0
        if kind == ForceKind::Army {
            kids_mut(&mut army)?[5] = u32v(0);
        }
        if garrison {
            let idx = ctx.region_info[&region].index;
            let sg = kids_mut(get_mut(
                &mut ctx.regions[idx],
                &["SETTLEMENT", "SIEGEABLE_GARRISON_RESIDENCE"],
            )?)?;
            sg[12] = u32v(wid_army);
            let sg_id = id_at(sg, 1)?;
            kids_mut(&mut army)?[5] = u32v(sg_id);
            ctx.region_info
                .get_mut(&region)
                .ok_or_else(|| anyhow!("region {region}"))?
                .garrison = Some(wid_army);
        }
        let frec = ctx
            .frec
            .get_mut(k)
            .ok_or_else(|| anyhow!("no faction record for {k}"))?;
        items_mut(get_mut(frec, &["CHARACTER_ARRAY"])?)?.push(vec![ch]);
        items_mut(get_mut(frec, &["ARMY_ARRAY"])?)?.push(vec![army]);
        ctx.forces.push(Force {
            faction: k.to_string(),
            kind,
            region,
            region_at: reg_at,
            sea_key,
            role: role.to_string(),
            units,
            x,
            y,
            garrison,
            wid_char,
            wid_army,
            unit_ids,
            t_char: tinfo.wid,
            t_army: t_army_id,
            t_unit: tinfo.unit,
            cai_char: 0,
            cai_mobile: 0,
            cai_units: Vec::new(),
            region_cai: 0,
            theatre_cai: 0,
            cell: None,
        });
        Ok(())
    }

    /// All starting forces of every alive faction, in faction order. Landless factions get none.
    pub fn run(&mut self, ctx: &mut BuildContext) -> Result<()> {
        let armies = ctx.rules.armies.clone();
        for k in ctx.fkeys.clone() {
            if !ctx.factions[&k].alive {
                continue;
            }
            let Some(list) = armies.get(&k).and_then(|v| v.as_array()) else {
                continue;
            };
            for entry in list {
                self.make_force(ctx, &k, entry)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf::{i32v, u32v};

    fn unit(men: u32) -> Node {
        Node::record(
            "NAVAL_UNIT",
            1,
            vec![
                Node::record("UNIT", 1, vec![i32v(7), u32v(men), u32v(men)]),
                Node::record("SHIP_DAMAGE_INFO", 1, (0..12).map(i32v).collect()),
            ],
        )
    }

    fn scalars(u: &Node, record: &str) -> Vec<f64> {
        kids(get(u, &[record]).unwrap())
            .unwrap()
            .iter()
            .filter_map(as_number)
            .collect()
    }

    #[test]
    fn changed_land_strength_replaces_current_and_maximum_men_only() {
        let mut u = unit(120);
        set_land_men(&mut u, "Inf_Line", 211).unwrap();
        assert_eq!(scalars(&u, "UNIT"), vec![7.0, 211.0, 211.0]);
    }

    #[test]
    fn changed_ship_crew_rewrites_strength_and_all_six_crew_counters() {
        let mut u = unit(500);
        set_ship_crew(&mut u, "Frigate", (10, 200, 30)).unwrap();
        assert_eq!(scalars(&u, "UNIT"), vec![7.0, 240.0, 240.0]);
        let sv = scalars(&u, "SHIP_DAMAGE_INFO");
        assert_eq!(&sv[5..11], &[10.0, 200.0, 30.0, 10.0, 200.0, 30.0]);
        // untouched counters keep the template's values
        assert_eq!(&sv[..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sv[11], 11.0);
    }

    #[test]
    fn records_without_the_expected_scalars_are_refused() {
        let mut short = Node::record(
            "LAND_UNIT",
            1,
            vec![Node::record("UNIT", 1, vec![i32v(7), u32v(5)])],
        );
        assert!(set_land_men(&mut short, "x", 9).is_err());
    }
}
