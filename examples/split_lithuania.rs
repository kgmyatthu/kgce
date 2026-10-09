//! Vanilla NTW experiment: carve `eur_samogitia` out of the northern tongue of `eur_lithuania` and stage every
//! changed campaign file under OUT (nothing is written into the game folder).
//! usage: split_lithuania NTW_DATA_DIR OUT_DIR
use std::fmt::Write as _;
use std::path::Path;

use kgc_port_editor::esf;
use kgc_port_editor::terrain::formats::{Spline, write_spln};
use kgc_port_editor::regionmap::pathfinding::Grid;
use kgc_port_editor::regionmap::poi::{self, Poi, SETTLEMENT};
use kgc_port_editor::regionmap::apply;
use kgc_port_editor::regionmap::campaign_rows as cr;
use kgc_port_editor::regionmap::sea_grids;
use kgc_port_editor::regionmap::startpos_split::{self, NewRegion};
use kgc_port_editor::{db, pack};
use kgc_port_editor::regionmap::trade_routes;
use kgc_port_editor::regionmap::regions::{
    RegionData, SettlementSpec, Split, add_theatre_region_key, settlement_center,
};

/// World rectangle (x0, y0, x1, y1) that the vanilla NTW `europe_lookup.tga` covers.
const NTW_EUROPE_RECT: [f64; 4] = [-410.0, -190.0, 340.0, 195.0];
const SRC: &str = "eur_lithuania";
const NEW: &str = "eur_samogitia";
const TOWN: &str = "siauliai";
/// ring positions in area 1 of eur_lithuania: 203 = (90.18, 81.35) on the Belarus border, 238 = (78.33, 82.69) on the
/// Courland border; forward arc 203..=238 is the northern tongue.
const A: usize = 203;
const B: usize = 238;
/// cut points ordered from `a` (east, Belarus border) towards `b` (west, Courland border)
const INTERIOR: [[f32; 2]; 4] = [[88.5, 81.8], [86.0, 82.2], [83.0, 82.6], [80.0, 82.8]];
const CENTER: [f32; 2] = [83.3, 87.3];

fn svg(rd: &RegionData, keys: &[&str], path: &Path) {
    let mut s = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"700\" height=\"1000\" style=\"background:#fff\">",
    );
    let p = |v: [f64; 2]| ((v[0] - 55.0) * 12.0, (100.0 - v[1]) * 12.0);
    let colours = ["#fb7", "#7fb", "#bbf"];
    for (ci, key) in keys.iter().enumerate() {
        let ri = rd.index(key).unwrap();
        for a in &rd.regions[ri].areas {
            for t in &a.faces {
                let pts: Vec<String> = t
                    .iter()
                    .map(|&i| {
                        let (x, y) = p(rd.v(i));
                        format!("{x:.1},{y:.1}")
                    })
                    .collect();
                let _ = write!(
                    s,
                    "<polygon points=\"{}\" fill=\"{}\" fill-opacity=\"0.5\" stroke=\"#999\" stroke-width=\"0.3\"/>",
                    pts.join(" "),
                    colours[ci % colours.len()]
                );
            }
            for o in &a.outlines {
                let n = o.ring.len();
                for run in &o.runs {
                    let pts: Vec<String> = (run.from..=run.to + 1)
                        .map(|k| {
                            let (x, y) = p(rd.v(o.ring[k as usize % n]));
                            format!("{x:.1},{y:.1}")
                        })
                        .collect();
                    let nb = rd.regions.get((run.code >> 16) as usize).map(|r| r.key.as_str()).unwrap_or("none");
                    let colour = if nb == "all" { "#06c" } else if keys.contains(&nb) { "#c00" } else { "#090" };
                    let _ = write!(
                        s,
                        "<polyline points=\"{}\" fill=\"none\" stroke=\"{colour}\" stroke-width=\"2\"/>",
                        pts.join(" ")
                    );
                    let (x, y) = p(rd.v(o.ring[((run.from + run.to) / 2) as usize % n]));
                    let _ = write!(s, "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"8\" fill=\"{colour}\">{}[{}-{}]</text>", nb.trim_start_matches("eur_"), run.from, run.to);
                }
            }
        }
        if let Some(sett) = &rd.regions[ri].settlement {
            for slot in sett.children().unwrap()[2].items().unwrap() {
                let xy = slot[3].as_xy().unwrap();
                let (x, y) = p([xy[0] as f64, xy[1] as f64]);
                let _ = write!(s, "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"3\" fill=\"#f0f\"/><text x=\"{:.1}\" y=\"{y:.1}\" font-size=\"8\">{}</text>", x + 4.0, slot[0].as_str().unwrap().split(':').nth(2).unwrap_or(""));
            }
        }
    }
    s.push_str("</svg>");
    std::fs::write(path, s).unwrap();
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let map_out = out.join("campaign_maps/nap_europe");
    std::fs::create_dir_all(&map_out)?;

    // ---- regions.esf ---------------------------------------------------------------------------------------
    let bytes = std::fs::read(data.join("campaign_maps/nap_europe/regions.esf"))?;
    let mut file = esf::read(&bytes)?;
    let mut rd = RegionData::parse(&file.root)?;
    let src_center = settlement_center(&rd.regions[rd.index(SRC)?])?;
    let new_index = rd.split(&Split {
        key: SRC.into(),
        area: 1,
        a: A,
        b: B,
        interior: INTERIOR.to_vec(),
        new_key: NEW.into(),
        settlement: SettlementSpec {
            name: TOWN.into(),
            center: CENTER,
        },
    })?;
    rd.store(&mut file.root)?;
    add_theatre_region_key(&mut file.root, SRC, NEW, CENTER)?;
    let cells = rd.rebuild_query_index(&mut file.root)?;
    let regions_bytes = esf::write(&file)?;
    std::fs::write(map_out.join("regions.esf"), &regions_bytes)?;
    svg(&rd, &[SRC, NEW], &out.join("split.svg"));
    let new = &rd.regions[new_index];
    println!(
        "regions.esf: {NEW} is region #{new_index} with {} ring vertices, {} faces, runs {:?}; {SRC} now {} ring vertices, {} faces; {cells} query cells rebuilt; settlement moved from {:?} to {:?}; {} bytes",
        new.areas[0].outlines[0].ring.len(),
        new.areas[0].faces.len(),
        new.areas[0].outlines[0].runs.iter().map(|r| (rd.regions[(r.code >> 16) as usize].key.clone(), r.code & 0xffff, r.from, r.to)).collect::<Vec<_>>(),
        rd.regions[rd.index(SRC)?].areas[1].outlines[0].ring.len(),
        rd.regions[rd.index(SRC)?].areas[1].faces.len(),
        src_center,
        CENTER,
        regions_bytes.len()
    );

    // ---- display/borders (cosmetic border lines, world / 40, resampled every 0.3 units) ------------------------
    let borders = map_out.join("display/borders");
    std::fs::create_dir_all(&borders)?;
    for (key, area) in [(SRC, 1usize), (NEW, 0usize)] {
        let ri = rd.index(key)?;
        let ring = &rd.regions[ri].areas[area].outlines[0].ring;
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for k in 0..ring.len() {
            let a = rd.v(ring[k]);
            let b = rd.v(ring[(k + 1) % ring.len()]);
            let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let steps = (len / 0.3).ceil().max(1.0) as usize;
            for s in 0..steps {
                let t = s as f64 / steps as f64;
                pts.push([((a[0] + (b[0] - a[0]) * t) / 40.0) as f32, 0.0, ((a[1] + (b[1] - a[1]) * t) / 40.0) as f32]);
            }
        }
        pts.push(pts[0]);
        let spline = Spline { name: format!("border:{key}:1"), flag: 1, points: pts };
        std::fs::write(borders.join(format!("{key}_1.rigid_spline")), write_spln(1, &[spline]))?;
    }
    println!("display/borders: {SRC}_1 and {NEW}_1 rigid_splines written");

    // ---- pathfinding.esf -----------------------------------------------------------------------------------
    let bytes = std::fs::read(data.join("campaign_maps/nap_europe/pathfinding.esf"))?;
    let mut file = esf::read(&bytes)?;
    let mut grid = Grid::parse(&file.root)?;
    let src_index = rd.index(SRC)? as i16;
    let in_new = |p: [f64; 2]| rd.locate(p).is_some_and(|c| (c >> 16) as usize == new_index);
    let (pieces, empties) = grid.relabel(src_index, new_index as i16, &in_new)?;
    grid.store(&mut file.root)?;
    let pf_bytes = esf::write(&file)?;
    std::fs::write(map_out.join("pathfinding.esf"), &pf_bytes)?;
    println!(
        "pathfinding.esf: {pieces} pieces and {empties} empty cells relabelled to {NEW}; {} regions, {} paths; {} bytes",
        grid.region_ids.len(),
        grid.paths.len(),
        pf_bytes.len()
    );

    // ---- poi.esf -------------------------------------------------------------------------------------------
    let bytes = std::fs::read(data.join("campaign_maps/nap_europe/poi.esf"))?;
    let mut file = esf::read(&bytes)?;
    let mut pois = poi::parse(&file.root)?;
    let src_index = src_index as usize;
    let new_adjacent = rd.adjacent(new_index);
    let key_of = |i: usize| rd.regions[i].key.clone();
    let dist = |i: usize, p: &Poi| rd.distance_to_region(i, [p.x, p.y]) as f32;
    let centre = [CENTER[0] as f64, CENTER[1] as f64];
    let settlements: std::collections::HashMap<String, [f64; 2]> = pois
        .iter()
        .filter(|p| p.kind == SETTLEMENT)
        .map(|p| (p.region.clone(), [p.x, p.y]))
        .chain(std::iter::once((NEW.to_string(), centre)))
        .collect();
    let (mut moved, mut touched) = (0usize, 0usize);
    let mut template: Option<Poi> = None;
    let old: Vec<Poi> = pois.clone();
    for p in pois.iter_mut() {
        let at = rd.locate([p.x, p.y]).map(|c| (c >> 16) as usize);
        if p.region == SRC && at == Some(new_index) {
            p.region = NEW.into();
            p.settlement_distance = ((p.x - centre[0]).powi(2) + (p.y - centre[1]).powi(2)).sqrt() as f32;
            if !p.neighbours.is_empty() {
                let n: Vec<(String, f32)> = new_adjacent.iter().map(|&i| (key_of(i), dist(i, p))).collect();
                p.neighbours.clear();
                for (k, d) in n {
                    p.set_neighbour(&k, d);
                }
            }
            moved += 1;
        } else if p.region == SRC {
            if p.kind == SETTLEMENT {
                template = Some(p.clone());
            }
            if !p.neighbours.is_empty() {
                let d = dist(new_index, p);
                p.set_neighbour(NEW, d);
            }
            touched += 1;
        } else if rd.index(&p.region).ok().is_some_and(|ri| new_adjacent.contains(&ri)) {
            if p.neighbours.iter().any(|(n, _)| n == SRC) {
                let (ds, dn) = (dist(src_index, p), dist(new_index, p));
                p.set_neighbour(SRC, ds);
                p.set_neighbour(NEW, dn);
                touched += 1;
            }
            let closer_to_new = dist(new_index, p) < dist(src_index, p);
            for entry in p.border.iter_mut() {
                if entry.0 == SRC && closer_to_new {
                    entry.0 = NEW.into();
                    entry.1 = ((p.x - centre[0]).powi(2) + (p.y - centre[1]).powi(2)).sqrt() as f32;
                    touched += 1;
                }
            }
        }
    }
    // border points of the moved region that name the old region's settlement
    for p in pois.iter_mut().filter(|p| p.region == NEW) {
        for entry in p.border.iter_mut() {
            if let Some(s) = settlements.get(&entry.0) {
                entry.1 = ((p.x - s[0]).powi(2) + (p.y - s[1]).powi(2)).sqrt() as f32;
            }
        }
    }
    let mut sett = template.ok_or_else(|| anyhow::anyhow!("{SRC} has no settlement POI"))?;
    sett.index = pois.len() as u32;
    sett.region = NEW.into();
    sett.x = centre[0];
    sett.y = centre[1];
    sett.settlement_distance = 0.0;
    sett.neighbours.clear();
    for &i in &new_adjacent {
        let d = rd.distance_to_region(i, centre) as f32;
        sett.set_neighbour(&key_of(i), d);
    }
    pois.push(sett);
    poi::store(&mut file.root, &pois)?;
    let poi_bytes = esf::write(&file)?;
    std::fs::write(map_out.join("poi.esf"), &poi_bytes)?;
    println!(
        "poi.esf: {moved} points moved to {NEW}, {touched} neighbour lists updated, settlement point #{} added ({} -> {} points); {} bytes",
        pois.len() - 1,
        old.len(),
        pois.len(),
        poi_bytes.len()
    );

    // ---- trade_routes.esf ----------------------------------------------------------------------------------
    let bytes = std::fs::read(data.join("campaign_maps/nap_europe/trade_routes.esf"))?;
    let mut file = esf::read(&bytes)?;
    let neighbour_settlements: Vec<(String, [f32; 2])> = new_adjacent
        .iter()
        .map(|&i| {
            let s = settlements[&rd.regions[i].key];
            (rd.regions[i].key.clone(), [s[0] as f32, s[1] as f32])
        })
        .collect();
    let refs: Vec<(&str, [f32; 2])> = neighbour_settlements.iter().map(|(k, p)| (k.as_str(), *p)).collect();
    let report = trade_routes::add_settlement(&mut file.root, NEW, CENTER, &refs)?;
    let tr_bytes = esf::write(&file)?;
    std::fs::write(map_out.join("trade_routes.esf"), &tr_bytes)?;
    println!("trade_routes.esf: node {} added with {} routes; {} bytes", report.node, report.routes_added, tr_bytes.len());

    // ---- sea_grids.esf -------------------------------------------------------------------------------------
    let bytes = std::fs::read(data.join("campaign_maps/nap_europe/sea_grids.esf"))?;
    let mut file = esf::read(&bytes)?;
    let changed = sea_grids::add_region(&mut file.root, &rd, new_index)?;
    let sg_bytes = esf::write(&file)?;
    std::fs::write(map_out.join("sea_grids.esf"), &sg_bytes)?;
    println!("sea_grids.esf: {changed} cell land list(s) changed; {} bytes", sg_bytes.len());

    // ---- DB rows + localisation pack, lookup TGA ------------------------------------------------------------
    let packs = cr::open_vanilla_packs(data)?;
    let schema = db::Schema::load(Path::new("assets/rules/schema_nap.ron"))?;
    let (_generated, files) = cr::build_pack_members(&packs, &schema)?;
    pack::write_pack(&out.join("zz_samogitia_test.pack"), &files, pack::TYPE_MOD)?;
    let painted = apply::paint_lookup_tga(
        &data.join("campaign_maps/nap_europe/europe_lookup.tga"),
        &map_out.join("europe_lookup.tga"),
        &rd,
        new_index,
        NTW_EUROPE_RECT,
        cr::NEW_COLOUR,
        None,
    )?
    .painted;
    println!("zz_samogitia_test.pack: {} members; europe_lookup.tga: {painted} pixels painted", files.len());

    // ---- startpos.esf --------------------------------------------------------------------------------------
    let vanilla_rd = RegionData::parse(&esf::read(&std::fs::read(data.join("campaign_maps/nap_europe/regions.esf"))?)?.root)?;
    let sp_bytes = std::fs::read(data.join("campaigns/eur_napoleon/startpos.esf"))?;
    let vanilla_sp = esf::read(&sp_bytes)?;
    let cai = startpos_split::cai_boundary_lengths(&vanilla_sp.root, SRC)?;
    let vb = vanilla_rd.border_lengths(vanilla_rd.index(SRC)?);
    let nb_len = rd.border_lengths(new_index);
    let sb = rd.border_lengths(src_index);
    // CAI boundary lengths scale with the polygon lengths: each donor border is divided in the polygon ratio
    let mut border_lengths: Vec<f32> = Vec::new();
    let mut donor_borders: Vec<(String, f32)> = Vec::new();
    let mut neighbours: Vec<String> = Vec::new();
    for &i in &new_adjacent {
        let key = rd.regions[i].key.clone();
        if key == SRC {
            continue;
        }
        let before = vb.get(&vanilla_rd.index(&key)?).copied().unwrap_or(0.0);
        let cl = cai.get(&key).copied().unwrap_or(0.0) as f64;
        border_lengths.push((cl * nb_len.get(&i).copied().unwrap_or(0.0) / before) as f32);
        donor_borders.push((key.clone(), (cl * sb.get(&i).copied().unwrap_or(0.0) / before) as f32));
        neighbours.push(key);
    }
    let ratios: Vec<f64> = vb
        .iter()
        .filter_map(|(i, l)| cai.get(&vanilla_rd.regions[*i].key).map(|c| *c as f64 / l))
        .collect();
    let ratio: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
    border_lengths.push((nb_len[&src_index] * ratio) as f32);
    neighbours.push(SRC.into());
    let new_sett = rd.regions[new_index].settlement.as_ref().unwrap().children()?[2].items()?.clone();
    let spec = NewRegion {
        key: NEW.into(),
        like: SRC.into(),
        settlement_key: format!("settlement:{NEW}:{TOWN}"),
        slot_keys: new_sett.iter().map(|s| s[0].as_str().map(String::from)).collect::<Result<_, _>>()?,
        slot_positions: new_sett.iter().map(|s| s[3].as_xy()).collect::<Result<_, _>>()?,
        position: CENTER,
        neighbours,
        settlement_loc: format!("start_pos_settlements_onscreen_name_{}", cr::STARTPOS_SETTLEMENT_NAME_EUR_NAPOLEON),
        rebel_name: "Samogitian Rebels".into(),
        border_lengths,
        donor_borders,
        owner: None,
        existing_cai: false,
        map_index: Some(new_index),
    };
    let mut edited = vanilla_sp.clone();
    let report = startpos_split::add_region(&mut edited.root, &spec)?;
    let out_sp = esf::write(&edited)?;
    let problems = startpos_split::check(&vanilla_sp.root, &esf::read(&out_sp)?.root, &spec)?;
    std::fs::create_dir_all(out.join("campaigns/eur_napoleon"))?;
    std::fs::write(out.join("campaigns/eur_napoleon/startpos.esf"), &out_sp)?;
    println!(
        "startpos.esf: region id {} owner {} settlement index {} CAI ids {:?}; {} structures added, {} consistency problems; {} bytes",
        report.region_id, report.owner, report.settlement_index, report.cai_range, report.added.len(), problems.len(), out_sp.len()
    );
    for p in &problems {
        println!("  ! {p}");
    }
    Ok(())
}
