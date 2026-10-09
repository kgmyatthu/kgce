//! Experiment 4 (throwaway): activate the backdrop polygon `indochina` ("Burma") of the enlarged India theatre as a
//! playable region and stage every changed campaign file under F:/NTW2/work/exp4/out/data (nothing is written into the
//! game folder).  Re-runnable: the regions.esf that the pathfinding step left in the staging map dir is snapshotted to
//! F:/NTW2/work/exp4/burma_in on the first run and restored before every later run.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use kgc_port_editor::db::{self, Cell, Schema, Table};
use kgc_port_editor::esf;
use kgc_port_editor::map::MapArtifacts;
use kgc_port_editor::model::{Project, RegionActivation, Rules};
use kgc_port_editor::pack::{self, Pack};
use kgc_port_editor::regionmap::apply;
use kgc_port_editor::regionmap::regions::RegionData;
use kgc_port_editor::regionmap::startpos_split;

const GAME_DATA: &str = "F:/NTW2/Napoleon Total War/data";
const DEFAULT_ROOT: &str = "F:/NTW2/work/exp4";
const EXP3_MAP: &str = "F:/NTW2/work/exp3/out/data/campaign_maps/nap_world";
const ASSETS: &str = "F:/NTW2/kgc_port_editor/assets";

const KEY: &str = "indochina";
const DONOR: &str = "bengal";
const THEATRE_LOOKUP: &str = "india_lookup.tga";
const LOC_MEMBER: &str = "text\\localisation.loc";

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn copy(from: &Path, to: &Path) -> Result<()> {
    if let Some(dir) = to.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::copy(from, to).with_context(|| format!("copy {} -> {}", from.display(), to.display()))?;
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<usize> {
    if to.exists() {
        fs::remove_dir_all(to)?;
    }
    fs::create_dir_all(to)?;
    let mut n = 0;
    for e in fs::read_dir(from)? {
        let e = e?;
        if e.file_type()?.is_file() {
            copy(&e.path(), &to.join(e.file_name()))?;
            n += 1;
        }
    }
    Ok(n)
}

/// Palette (BGR -> RGB) of an 8-bit colour-mapped TGA.
fn tga_palette(d: &[u8]) -> Result<Vec<[u8; 3]>> {
    let at = 18 + d[0] as usize;
    let first = u16::from_le_bytes([d[3], d[4]]) as usize;
    let count = u16::from_le_bytes([d[5], d[6]]) as usize;
    ensure!(d[1] == 1 && d[7] == 24 && first == 0, "not a 24-bit colour-mapped TGA");
    Ok((0..count).map(|i| [d[at + 3 * i + 2], d[at + 3 * i + 1], d[at + 3 * i]]).collect())
}

/// The colour with the largest distance to every colour in `used` (grid search), channels kept off the extremes.
fn pick_colour(used: &[[u8; 3]]) -> [u8; 3] {
    let dist = |a: [u8; 3], b: [u8; 3]| -> i64 {
        (0..3).map(|k| (a[k] as i64 - b[k] as i64).pow(2)).sum()
    };
    let mut best = ([0u8; 3], -1i64);
    for r in (45..=210u16).step_by(5) {
        for g in (45..=210u16).step_by(5) {
            for b in (45..=210u16).step_by(5) {
                let c = [r as u8, g as u8, b as u8];
                let m = used.iter().map(|&u| dist(c, u)).min().unwrap_or(i64::MAX);
                if m > best.1 {
                    best = (c, m);
                }
            }
        }
    }
    best.0
}

fn member(pack: &Pack, name: &str) -> Result<Vec<u8>> {
    pack.read(name)
}

/// Decode a DB member and check that it re-encodes to the same bytes (so a rewrite only changes what we touch).
fn decode_checked(name: &str, bytes: &[u8], schema: &Schema) -> Result<Table> {
    let t = db::decode(name, bytes, schema)?;
    ensure!(db::encode(&t)? == bytes, "{name} does not round-trip byte-exact");
    Ok(t)
}

fn row_json(t: &Table, i: usize) -> Result<Value> {
    Ok(t.json_rows()?.swap_remove(i))
}

fn cell_row(t: &Table, row: &Value) -> Result<Vec<Cell>> {
    t.fields
        .iter()
        .map(|f| Cell::from_json(&f.ty, row.get(&f.name).with_context(|| format!("missing {}", f.name))?))
        .collect()
}

fn main() -> Result<()> {
    let game_map = PathBuf::from(GAME_DATA).join("campaign_maps/nap_world");
    let root = std::env::var("EXP4_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_string());
    let out_data = format!("{root}/out/data");
    let map_dir = PathBuf::from(&out_data).join("campaign_maps/nap_world");
    let exp3 = PathBuf::from(EXP3_MAP);
    let snap = PathBuf::from(format!("{root}/burma_in"));
    fs::create_dir_all(&map_dir)?;
    fs::create_dir_all(&snap)?;

    // ---------------------------------------------------------------- 1. stage the map files
    // regions.esf / pathfinding.esf come from the pathfinding step (exp4 staging dir); until they exist, exp3's.
    let regions_path = map_dir.join("regions.esf");
    let pf_path = map_dir.join("pathfinding.esf");
    let snap_regions = snap.join("regions.esf");
    let snap_sha = snap.join("regions.out.sha");
    let mut notes: Vec<String> = Vec::new();
    let regions_src = if regions_path.exists() {
        let cur = fs::read(&regions_path)?;
        let previous_out = fs::read_to_string(&snap_sha).ok();
        if previous_out.as_deref() == Some(sha(&cur).as_str()) && snap_regions.exists() {
            notes.push("regions.esf: restored the pre-Burma regions.esf of the previous run".into());
            snap_regions.clone()
        } else {
            fs::write(&snap_regions, &cur)?;
            notes.push("regions.esf: using the regions.esf found in the exp4 staging dir".into());
            snap_regions.clone()
        }
    } else {
        notes.push("regions.esf: NOT in the exp4 staging dir yet, developing against exp3's file".into());
        exp3.join("regions.esf")
    };
    fs::write(&regions_path, fs::read(&regions_src)?)?;
    if !pf_path.exists() {
        notes.push("pathfinding.esf: NOT in the exp4 staging dir yet, using exp3's file".into());
        copy(&exp3.join("pathfinding.esf"), &pf_path)?;
    }
    // everything else: exp4 if present (kept), else exp3, else the installed copy
    for name in ["sea_grids.esf", "stratradar_india.tga", "india_map.tga"] {
        let dst = map_dir.join(name);
        if !dst.exists() {
            let src = if exp3.join(name).exists() { exp3.join(name) } else { game_map.join(name) };
            copy(&src, &dst)?;
        }
    }
    // files the activation modifies: always start from the installed (== exp3) originals
    for name in ["poi.esf", "trade_routes.esf"] {
        copy(&game_map.join(name), &map_dir.join(name))?;
    }
    let lookup_src = if exp3.join(THEATRE_LOOKUP).exists() { exp3.join(THEATRE_LOOKUP) } else { game_map.join(THEATRE_LOOKUP) };
    copy(&lookup_src, &map_dir.join(THEATRE_LOOKUP))?;
    let nb = copy_dir(&game_map.join("display/borders"), &map_dir.join("display/borders"))?;
    println!("staged map files in {} ({nb} border files copied)", map_dir.display());

    // ---------------------------------------------------------------- 2. rules, project, map artifacts
    let assets = kgc_port_editor::local_assets::ensure(
        &kgc_port_editor::pipeline::default_workspace(Path::new(ASSETS)),
        Path::new(ASSETS),
        &|message| println!("{message}"),
    )?;
    let mut rules = Rules::load(&assets.join("rules"))?;
    let schema = Schema::load(&assets.join("rules/schema_nap.ron"))?;
    let pack_in = Pack::open(&Path::new(GAME_DATA).join("nap_world.pack"))?;

    // lookup colour: the one farthest from every colour of the theatre's palette and of regions_tables
    let regions_member = "db\\regions_tables\\nap_world";
    let settle_member = "db\\campaign_map_settlements_tables\\nap_world";
    let areas_member = "db\\campaign_map_playable_areas_tables\\nap_world";
    let regions_old = decode_checked("regions_tables", &member(&pack_in, regions_member)?, &schema)?;
    let settle_old = decode_checked("campaign_map_settlements_tables", &member(&pack_in, settle_member)?, &schema)?;
    let areas_old = decode_checked("campaign_map_playable_areas_tables", &member(&pack_in, areas_member)?, &schema)?;
    let mut used = tga_palette(&fs::read(map_dir.join(THEATRE_LOOKUP))?)?;
    for row in regions_old.json_rows()? {
        used.push([row["r"].as_u64().unwrap_or(0) as u8, row["g"].as_u64().unwrap_or(0) as u8, row["b"].as_u64().unwrap_or(0) as u8]);
    }
    for (_, r) in rules.regions.as_object().context("regions rules")? {
        if let Some(c) = r.get("lookup_rgb").and_then(Value::as_array) {
            if c.len() == 3 {
                used.push([c[0].as_u64().unwrap_or(0) as u8, c[1].as_u64().unwrap_or(0) as u8, c[2].as_u64().unwrap_or(0) as u8]);
            }
        }
    }
    let colour = pick_colour(&used);
    println!("lookup colour {colour:?} (checked against {} used colours)", used.len());

    let activation = RegionActivation {
        key: KEY.into(),
        donor: DONOR.into(),
        name: "Burma".into(),
        settlement: "rangoon".into(),
        settlement_name: "Rangoon".into(),
        center: [683.9, 119.8],
        colour,
        ..Default::default()
    };
    let mut project = Project::default();
    project.activations.push(activation.clone());
    rules.apply_project(&project)?;
    let rule = rules.regions.get(KEY).context("rules have no indochina after apply_project")?.clone();
    println!("rules.regions[{KEY}]: {rule}");

    let mut map = MapArtifacts {
        slot_mapping: json!({ "regions": {} }),
        trade_nodes: Value::Null,
        db_rows: BTreeMap::from([
            (
                "campaign_map_playable_areas_tables".to_string(),
                json!({ "table": "campaign_map_playable_areas_tables", "version": areas_old.version, "rows": areas_old.json_rows()? }),
            ),
            (
                "campaign_map_settlements_tables".to_string(),
                json!({ "table": "campaign_map_settlements_tables", "version": settle_old.version, "rows": settle_old.json_rows()? }),
            ),
        ]),
        loc_rows: Vec::new(),
    };

    // ---------------------------------------------------------------- 3. map: apply the activation
    let before = RegionData::parse(&esf::read(&fs::read(&regions_path)?)?.root)?;
    let ri = before.index(KEY)?;
    let (applied, stats) = apply::apply_activations_with_stats(&map_dir, &project.activations, &rules, &mut map)?;
    ensure!(applied.len() == 1, "expected one applied activation");
    let ap = &applied[0];
    let st = &stats[0];
    let after = RegionData::parse(&esf::read(&fs::read(&regions_path)?)?.root)?;
    // regions.esf may differ from the input only in the new region's settlement / tail
    ensure!(before.regions.len() == after.regions.len() && before.vertices == after.vertices, "regions.esf geometry changed");
    for (i, (a, b)) in before.regions.iter().zip(&after.regions).enumerate() {
        ensure!(a.key == b.key && a.kind == b.kind && a.areas.len() == b.areas.len(), "region {i} changed shape");
        for (x, y) in a.areas.iter().zip(&b.areas) {
            ensure!(
                x.passable == y.passable
                    && x.adjoining_id == y.adjoining_id
                    && x.impassable_id == y.impassable_id
                    && x.island_id == y.island_id
                    && x.faces == y.faces,
                "region {} area flags/faces changed",
                a.key
            );
        }
        if i != ri {
            ensure!(a.tail == b.tail && a.settlement.is_some() == b.settlement.is_some(), "region {} changed", a.key);
        }
    }
    ensure!(after.regions[ri].settlement.is_some(), "indochina has no settlement after apply");
    let passable_areas: Vec<usize> =
        after.regions[ri].areas.iter().enumerate().filter(|(_, a)| a.passable).map(|(i, _)| i).collect();
    println!(
        "regions.esf: index {ri} tail {} -> {}, passable areas {:?}, applied: theatre {}, neighbours {:?}",
        before.regions[ri].tail, after.regions[ri].tail, passable_areas, ap.theatre, ap.neighbours
    );
    if passable_areas != [2] {
        notes.push(format!(
            "WARNING: indochina passable areas are {passable_areas:?} in the regions.esf that was used (expected [2]); the pathfinding step's flipped regions.esf is not staged yet"
        ));
    }

    // ---------------------------------------------------------------- 4. start position
    let startpos_in = PathBuf::from(GAME_DATA).join("campaigns/wld_napoleon/startpos.esf");
    let startpos_out = PathBuf::from(&out_data).join("campaigns/wld_napoleon/startpos.esf");
    let mut sp = esf::read(&fs::read(&startpos_in)?)?;
    let spec = startpos_split::new_region_for_activation(&sp.root, &activation, ap, &rules)?;
    let sp_before = sp.root.clone();
    let report = startpos_split::add_region(&mut sp.root, &spec)?;
    let sp_bytes = esf::write(&sp)?;
    let sp_back = esf::read(&sp_bytes)?;
    let problems = startpos_split::check(&sp_before, &sp_back.root, &spec)?;
    if !problems.is_empty() {
        bail!("start position check found {} problems:\n  {}", problems.len(), problems.join("\n  "));
    }
    if let Some(dir) = startpos_out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(&startpos_out, &sp_bytes)?;
    println!(
        "startpos: check() empty; region id {} settlement id {} owner {} settlement index {} CAI ids {:?} neighbours {:?}",
        report.region_id, report.settlement_id, report.owner, report.settlement_index, report.cai_range, spec.neighbours
    );
    for n in &report.notes {
        println!("  note: {n}");
    }

    // ---------------------------------------------------------------- 5. pack
    let mut members: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for e in &pack_in.entries {
        members.insert(e.name.clone(), pack_in.read_entry(e)?);
    }
    // sanity: an unmodified rewrite reproduces the installed pack byte for byte
    let probe = snap.join("probe.pack");
    pack::write_pack(&probe, &members, pack_in.type_)?;
    ensure!(
        fs::read(&probe)? == fs::read(&pack_in.path)?,
        "write_pack of the unmodified members differs from the installed pack"
    );
    fs::remove_file(&probe)?;

    // regions_tables: the pack already has a (cont_impassable, black) row for the backdrop key; replace it in place
    let mut regions_new = regions_old.clone();
    let key_col = regions_new.column("key")?;
    let at = regions_new
        .rows
        .iter()
        .position(|r| r[key_col].as_str() == Some(KEY))
        .with_context(|| format!("regions_tables has no row {KEY}"))?;
    let mut row = row_json(&regions_old, at)?;
    let old_row = row.clone();
    row["continent"] = json!(rule["continent"].as_str().context("rule continent")?);
    row["r"] = json!(colour[0]);
    row["g"] = json!(colour[1]);
    row["b"] = json!(colour[2]);
    row["battle_name"] = json!(activation.settlement_name);
    regions_new.rows[at] = cell_row(&regions_old, &row)?;
    println!("regions_tables: {old_row} -> {row}");
    members.insert(regions_member.to_string(), db::encode(&regions_new)?);

    // campaign_map_settlements_tables: rows from map.db_rows
    let rows = map.db_rows["campaign_map_settlements_tables"]["rows"].as_array().context("settlement rows")?.clone();
    let mut settle_new = settle_old.clone();
    settle_new.rows = rows.iter().map(|r| cell_row(&settle_old, r)).collect::<Result<_>>()?;
    let id_col = settle_old.column("settlement_id")?;
    let old_ids: Vec<&str> = settle_old.rows.iter().filter_map(|r| r[id_col].as_str()).collect();
    let new_ids: Vec<&str> = settle_new.rows.iter().filter_map(|r| r[id_col].as_str()).collect();
    ensure!(new_ids.len() == old_ids.len() + 1, "expected exactly one new settlement row");
    let added: Vec<&&str> = new_ids.iter().filter(|k| !old_ids.contains(k)).collect();
    ensure!(added.len() == 1, "settlement ids added: {added:?}");
    // every old row unchanged and in order
    let mut it = settle_new.rows.iter();
    for r in &settle_old.rows {
        let mut n = it.next().context("settlement rows shorter")?;
        if n[id_col].as_str() == Some(*added[0]) {
            n = it.next().context("settlement rows shorter")?;
        }
        ensure!(n == r, "an existing settlement row changed");
    }
    println!("campaign_map_settlements_tables: + {added:?}");
    members.insert(settle_member.to_string(), db::encode(&settle_new)?);

    // localisation.loc: existing keys are replaced in place, new ones appended
    let loc_bytes = member(&pack_in, LOC_MEMBER)?;
    let (loc_version, mut entries) = db::decode_loc(&loc_bytes)?;
    ensure!(db::encode_loc(&entries, loc_version)? == loc_bytes, "localisation.loc does not round-trip byte-exact");
    let mut wanted: Vec<(String, String, u8)> = map
        .loc_rows
        .iter()
        .map(|(k, t, tip, target)| {
            assert_eq!(target, "localisation");
            (k.clone(), t.clone(), *tip)
        })
        .collect();
    wanted.push((
        format!("start_pos_regions_rebel_faction_name_{KEY}wld_napoleon"),
        rule["rebel_name"].as_str().context("rebel_name")?.to_string(),
        0,
    ));
    let (mut replaced, mut appended) = (Vec::new(), Vec::new());
    for (key, text, tip) in wanted {
        match entries.iter_mut().find(|e| e.0 == key) {
            Some(e) => {
                replaced.push(format!("{key}: {:?} -> {text:?}", e.1));
                e.1 = text;
                e.2 = tip;
            }
            None => {
                appended.push(format!("{key} = {text:?}"));
                entries.push((key, text, tip));
            }
        }
    }
    println!("localisation.loc replaced: {replaced:#?}\nlocalisation.loc appended: {appended:#?}");
    members.insert(LOC_MEMBER.to_string(), db::encode_loc(&entries, loc_version)?);

    let pack_out = PathBuf::from(&out_data).join("nap_world.pack");
    pack::write_pack(&pack_out, &members, pack_in.type_)?;

    // ---------------------------------------------------------------- 6. re-read the outputs
    let p2 = Pack::open(&pack_out)?;
    ensure!(p2.entries.len() == pack_in.entries.len(), "member count changed");
    let mut changed_members = Vec::new();
    for e in &pack_in.entries {
        if p2.read(&e.name)? != pack_in.read_entry(e)? {
            changed_members.push(e.name.clone());
        }
    }
    println!("pack members changed: {changed_members:?}");
    let rt = decode_checked("regions_tables", &p2.read(regions_member)?, &schema)?;
    ensure!(rt.rows.len() == regions_old.rows.len(), "regions_tables row count");
    println!("  regions_tables[{KEY}] = {}", row_json(&rt, at)?);
    let st2 = decode_checked("campaign_map_settlements_tables", &p2.read(settle_member)?, &schema)?;
    ensure!(st2.rows.len() == settle_old.rows.len() + 1, "settlement row count");
    let loc2 = db::parse_loc(&p2.read(LOC_MEMBER)?)?;
    for (k, v) in [
        ("regions_onscreen_indochina", "Burma"),
        ("regions_battle_name_indochina", "Rangoon"),
        ("start_pos_settlements_onscreen_name_settlement:indochina:rangoon", "Rangoon"),
        ("start_pos_regions_rebel_faction_name_indochinawld_napoleon", rule["rebel_name"].as_str().unwrap()),
    ] {
        let got = loc2.get(k).map(|t| t.0.as_str());
        ensure!(got == Some(v), "loc {k} = {got:?}, expected {v:?}");
    }
    // the start position strings that must resolve in the loc
    let sp_check = esf::read(&fs::read(&startpos_out)?)?;
    ensure!(esf::write(&sp_check)? == sp_bytes, "startpos does not round-trip");
    let rd = RegionData::parse(&esf::read(&fs::read(&regions_path)?)?.root)?;
    let r = &rd.regions[rd.index(KEY)?];
    ensure!(r.settlement.is_some(), "re-read regions.esf: no settlement");
    println!("re-read OK: regions.esf ({} regions), startpos.esf, nap_world.pack ({} members)", rd.regions.len(), p2.entries.len());

    fs::write(&snap_sha, sha(&fs::read(&regions_path)?))?;

    // ---------------------------------------------------------------- 7. summary
    println!("\n=== SUMMARY ===");
    for n in &notes {
        println!("{n}");
    }
    println!(
        "pathfinding area {} | poi: {:?} | trade node {} ({} routes) | lookup {} painted {} px (palette index {}) | sea cells listing {}",
        st.pathfinding_area, st.poi, st.trade_node, st.trade_routes, st.lookup_file, st.lookup.painted, st.lookup.palette_index, st.sea_cells_listing
    );
    println!("borders written {:?} removed {:?}", st.border_files_written, st.border_files_removed);
    let rel = |p: &Path| p.strip_prefix(&out_data).unwrap().to_string_lossy().replace('\\', "/");
    println!("\nINSTALL (relative to the game's data folder; staged under {out_data}):");
    let mut install = vec![
        "campaign_maps/nap_world/regions.esf".to_string(),
        "campaign_maps/nap_world/pathfinding.esf".to_string(),
        "campaign_maps/nap_world/poi.esf".to_string(),
        "campaign_maps/nap_world/trade_routes.esf".to_string(),
        format!("campaign_maps/nap_world/{}", st.lookup_file),
        rel(&startpos_out),
        rel(&pack_out),
    ];
    install.extend(st.border_files_written.iter().map(|f| format!("campaign_maps/nap_world/display/borders/{f}")));
    for f in &install {
        println!("  {f}");
    }
    if !st.border_files_removed.is_empty() {
        println!("DELETE from the game folder: {:?}", st.border_files_removed);
    }
    Ok(())
}
