//! Structural comparison of a generated campaign map against the validated golden files (acceptance 2 of the
//! theatre-extension integration).
//!
//! ```text
//! cargo run --release --example compare_golden -- --generated <data dir | NTW ZIP> --golden <data dir>
//!        [--theatre 3] [--settlement indochina --settlement sumatra] [--allow-region unexplorable]
//! ```
//!
//! `--generated` is the `data` folder of a build, or `GrandCampaign-NTW.zip` (members `data/...`). `--golden` is a
//! `data` folder of the validated experiment (variant B = Burma union only: `F:/NTW2/work/exp5/variantB/out/data`; the
//! full exp5 set is NOT game-validated and is only a structural reference). Compared, region by region (matched by
//! key): area count, passable flag, bounding box, component id, impassable / island ids and the rings (vertex counts,
//! coordinates within 1e-3 after vertex matching); pathfinding of the theatre: region list, path table, number of landing
//! crossings (+-2), settlement nodes; the sea grid lists and connections; and the generated start position (HLCI
//! counts, region list, sea cells) and trade routes. Component ids are reported separately: the port deliberately gives
//! an activated area the component of the connected passable land it borders (Bengal's 13), the old golden used a fresh
//! id (857).
//!
//! Exit status 1 when a hard deviation (anything but component ids) is found.

use anyhow::{Context, Result, bail};
use kgc_port_editor::{
    esf,
    regionmap::{
        pathfinding::Grid,
        pathsynth,
        regions::{RegionData, settlement_center},
        sea_grids::SeaGrids,
        startpos_split::{MapChecks, check_world},
        trade_routes::routeless_nodes,
    },
};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

const MAP: &str = "campaign_maps/nap_world";

enum Source {
    Dir(PathBuf),
    Zip(zip::ZipArchive<fs::File>),
}

impl Source {
    fn open(path: &Path) -> Result<Self> {
        if path.is_dir() {
            Ok(Self::Dir(path.into()))
        } else {
            Ok(Self::Zip(zip::ZipArchive::new(fs::File::open(path).with_context(|| path.display().to_string())?)?))
        }
    }

    fn read(&mut self, relative: &str) -> Result<Vec<u8>> {
        match self {
            Self::Dir(root) => fs::read(root.join(relative)).with_context(|| format!("{}/{relative}", root.display())),
            Self::Zip(zip) => {
                let mut member = zip
                    .by_name(&format!("data/{relative}"))
                    .with_context(|| format!("zip member data/{relative}"))?;
                let mut bytes = Vec::new();
                member.read_to_end(&mut bytes)?;
                Ok(bytes)
            }
        }
    }

    fn esf(&mut self, relative: &str) -> Result<esf::Esf> {
        esf::read(&self.read(relative)?).with_context(|| relative.to_string())
    }
}

struct Maps {
    esf: esf::Esf,
    rd: RegionData,
    pf: esf::Esf,
    sea: esf::Esf,
    trade: esf::Esf,
}

fn load(source: &mut Source) -> Result<Maps> {
    let regions = source.esf(&format!("{MAP}/regions.esf"))?;
    Ok(Maps {
        rd: RegionData::parse(&regions.root)?,
        esf: regions,
        pf: source.esf(&format!("{MAP}/pathfinding.esf"))?,
        sea: source.esf(&format!("{MAP}/sea_grids.esf"))?,
        trade: source.esf(&format!("{MAP}/trade_routes.esf"))?,
    })
}

#[derive(Default)]
struct Report {
    hard: Vec<String>,
    components: Vec<String>,
    info: Vec<String>,
}

impl Report {
    fn hard(&mut self, text: String) {
        self.hard.push(text);
    }
}

/// `[min, max]` of the ring vertices of an area (exact f32 min/max).
fn bbox(rd: &RegionData, ri: usize, ai: usize) -> [[f32; 2]; 2] {
    let mut bb = [[f32::MAX; 2], [f32::MIN; 2]];
    for o in &rd.regions[ri].areas[ai].outlines {
        for &v in &o.ring {
            let p = rd.vertices[v as usize];
            for k in 0..2 {
                bb[0][k] = bb[0][k].min(p[k]);
                bb[1][k] = bb[1][k].max(p[k]);
            }
        }
    }
    bb
}

fn ring(rd: &RegionData, ids: &[u32]) -> Vec<[f32; 2]> {
    ids.iter().map(|&v| rd.vertices[v as usize]).collect()
}

/// Every vertex of `a` has a vertex of `b` within `eps` and the counts agree (rings are matched as vertex sets: the
/// start vertex and the run layout may legitimately differ).
fn same_vertices(a: &[[f32; 2]], b: &[[f32; 2]], eps: f32) -> bool {
    let near = |p: &[f32; 2], list: &[[f32; 2]]| list.iter().any(|q| (p[0] - q[0]).abs() <= eps && (p[1] - q[1]).abs() <= eps);
    a.len() == b.len() && a.iter().all(|p| near(p, b)) && b.iter().all(|p| near(p, a))
}

fn compare_regions(gen_: &Maps, gold: &Maps, allow: &BTreeSet<String>, report: &mut Report) {
    report.info.push(format!(
        "regions: generated {}, golden {}",
        gen_.rd.regions.len(),
        gold.rd.regions.len()
    ));
    let mut matched = 0usize;
    let mut areas = 0usize;
    for (gi, region) in gen_.rd.regions.iter().enumerate() {
        let Ok(oi) = gold.rd.index(&region.key) else {
            report.info.push(format!("region {} exists only in the generated map", region.key));
            continue;
        };
        let other = &gold.rd.regions[oi];
        if allow.contains(&region.key) {
            report.info.push(format!("region {} skipped (--allow-region)", region.key));
            continue;
        }
        matched += 1;
        if region.areas.len() != other.areas.len() || region.kind != other.kind {
            report.hard(format!(
                "region {}: {} areas ({}) against golden {} ({})",
                region.key,
                region.areas.len(),
                region.kind,
                other.areas.len(),
                other.kind
            ));
            continue;
        }
        for (ai, (a, b)) in region.areas.iter().zip(&other.areas).enumerate() {
            areas += 1;
            let at = format!("region {} area {ai}", region.key);
            if a.passable != b.passable {
                report.hard(format!("{at}: passable {} against golden {}", a.passable, b.passable));
            }
            if a.adjoining_id != b.adjoining_id {
                report.components.push(format!("{at}: component {} against golden {}", a.adjoining_id, b.adjoining_id));
            }
            if a.impassable_id != b.impassable_id && (a.impassable_id == 65535 || b.impassable_id == 65535 || a.passable) {
                report.hard(format!("{at}: impassable id {} against golden {}", a.impassable_id, b.impassable_id));
            }
            if a.island_id != b.island_id {
                report.hard(format!("{at}: island id {} against golden {}", a.island_id, b.island_id));
            }
            if bbox(&gen_.rd, gi, ai) != bbox(&gold.rd, oi, ai) {
                report.hard(format!("{at}: bounding box {:?} against golden {:?}", bbox(&gen_.rd, gi, ai), bbox(&gold.rd, oi, ai)));
            }
            if a.outlines.len() != b.outlines.len() {
                report.hard(format!("{at}: {} outlines against golden {}", a.outlines.len(), b.outlines.len()));
                continue;
            }
            for (oi2, (x, y)) in a.outlines.iter().zip(&b.outlines).enumerate() {
                if x.outer != y.outer || !same_vertices(&ring(&gen_.rd, &x.ring), &ring(&gold.rd, &y.ring), 1e-3) {
                    report.hard(format!("{at} outline {oi2}: ring differs ({} against {} vertices)", x.ring.len(), y.ring.len()));
                }
            }
        }
    }
    for region in &gold.rd.regions {
        if gen_.rd.index(&region.key).is_err() {
            report.hard(format!("golden region {} is missing in the generated map", region.key));
        }
    }
    report.info.push(format!("regions compared: {matched} ({areas} areas)"));
}

fn landing_halves(grid: &Grid) -> usize {
    let Some(sea) = grid.paths.iter().position(|p| p.is_empty()) else { return 0 };
    grid.cells
        .iter()
        .flat_map(|c| &c.pieces)
        .filter(|p| p.a & 0xF == 7 && usize::from(p.path_id()) == sea)
        .count()
}

use kgc_port_editor::regionmap::stages::settlement_node_at as node_at;

fn compare_pathfinding(gen_: &Maps, gold: &Maps, theatre: &str, settlements: &[String], report: &mut Report) -> Result<()> {
    let area = |m: &Maps| m.rd.theatre_rects().iter().position(|(k, _)| k == theatre);
    let (Some(ga), Some(oa)) = (area(gen_), area(gold)) else { bail!("theatre {theatre} is missing") };
    let grid = Grid::parse_area(&gen_.pf.root, ga)?;
    let golden = Grid::parse_area(&gold.pf.root, oa)?;
    report.info.push(format!(
        "pathfinding theatre {theatre}: generated {}x{} cells, golden {}x{}",
        grid.cols, grid.rows, golden.cols, golden.rows
    ));
    if (grid.cols, grid.rows) != (golden.cols, golden.rows) {
        report.hard(format!("pathfinding grid size {}x{} against golden {}x{}", grid.cols, grid.rows, golden.cols, golden.rows));
    }
    let problems = pathsynth::validate(&grid, &gen_.rd);
    report.info.push(format!("pathfinding rule checker: {} problem(s)", problems.len()));
    for p in problems.iter().take(20) {
        report.hard(format!("pathfinding validator: {p}"));
    }
    // region lists and path tables are compared by region KEY: indices differ when a region was carved
    let keys = |g: &Grid, rd: &RegionData| -> Vec<String> { g.region_ids.iter().map(|&i| rd.regions[i as usize].key.clone()).collect() };
    let (gk, ok) = (keys(&grid, &gen_.rd), keys(&golden, &gold.rd));
    if gk != ok {
        let (a, b): (BTreeSet<_>, BTreeSet<_>) = (gk.iter().collect(), ok.iter().collect());
        report.hard(format!(
            "pathfinding region list differs: only generated {:?}, only golden {:?}",
            a.difference(&b).collect::<Vec<_>>(),
            b.difference(&a).collect::<Vec<_>>()
        ));
    }
    let table = |g: &Grid, rd: &RegionData| -> Vec<Vec<String>> {
        g.paths.iter().map(|p| p.iter().map(|&i| rd.regions[i as usize].key.clone()).collect()).collect()
    };
    if table(&grid, &gen_.rd) != table(&golden, &gold.rd) {
        report.hard(format!(
            "pathfinding path table differs ({} paths against golden {})",
            grid.paths.len(),
            golden.paths.len()
        ));
    }
    let (n, m) = (landing_halves(&grid), landing_halves(&golden));
    report.info.push(format!("landing crossings (sea halves): generated {n}, golden {m}"));
    if n.abs_diff(m) > 2 {
        report.hard(format!("landing crossings {n} against golden {m} (more than 2 apart)"));
    }
    for key in settlements {
        let Ok(ri) = gen_.rd.index(key) else {
            report.info.push(format!("settlement {key}: not in the generated map"));
            continue;
        };
        let centre = settlement_center(&gen_.rd.regions[ri])?;
        let point = [f64::from(centre[0]), f64::from(centre[1])];
        if !node_at(&grid, ri as i16, point) {
            report.hard(format!("settlement node of {key} is missing at {centre:?}"));
        } else {
            let groups: Vec<Vec<String>> = grid
                .paths
                .iter()
                .filter(|p| p.len() > 1 && p.contains(&(ri as i16)))
                .map(|p| p.iter().map(|&i| gen_.rd.regions[i as usize].key.clone()).collect())
                .collect();
            report.info.push(format!("settlement {key}: type-7 node present; group paths {groups:?}"));
        }
        if let Ok(oi) = gold.rd.index(key) {
            let oc = settlement_center(&gold.rd.regions[oi])?;
            if oc != centre {
                report.info.push(format!("settlement {key}: centre {centre:?}, golden {oc:?}"));
            }
            if !node_at(&golden, oi as i16, [f64::from(oc[0]), f64::from(oc[1])]) {
                report.info.push(format!("settlement {key}: the golden grid has no node at its own centre?"));
            }
        }
    }
    Ok(())
}

fn compare_sea(gen_: &Maps, gold: &Maps, theatre: &str, report: &mut Report) -> Result<()> {
    let (a, b) = (SeaGrids::parse(&gen_.sea.root)?, SeaGrids::parse(&gold.sea.root)?);
    let (Some(x), Some(y)) = (a.grid(theatre), b.grid(theatre)) else { bail!("sea grid of theatre {theatre} is missing") };
    if x.cells != y.cells || x.min != y.min || x.max != y.max || (x.xsize, x.ysize) != (y.xsize, y.ysize) {
        report.hard(format!("sea grid boxes differ: {:?}..{:?} {}x{} against golden {:?}..{:?} {}x{}", x.min, x.max, x.xsize, x.ysize, y.min, y.max, y.xsize, y.ysize));
    }
    let differing: Vec<usize> = (0..x.areas.len().min(y.areas.len()))
        .filter(|&i| {
            let (p, q) = (&x.areas[i], &y.areas[i]);
            (&p.lands, &p.seas, &p.ports, &p.numbers) != (&q.lands, &q.seas, &q.ports, &q.numbers)
        })
        .collect();
    report.info.push(format!("sea grid: {} cells, {} with different lists", x.areas.len(), differing.len()));
    for i in differing.iter().take(10) {
        report.hard(format!("sea grid cell {i}: lands/seas/ports/nums differ ({:?} against {:?})", x.areas[*i], y.areas[*i]));
    }
    if x.connections != y.connections {
        report.hard(format!("sea grid connections differ ({} against golden {})", x.connections.len(), y.connections.len()));
    }
    Ok(())
}

fn check_startpos(source: &mut Source, gen_: &Maps, report: &mut Report) -> Result<()> {
    let start = source.esf("campaigns/wld_napoleon/startpos.esf")?;
    let mut lists = Vec::new();
    for area in 0..Grid::area_count(&gen_.pf.root)? {
        lists.push(Grid::parse_area(&gen_.pf.root, area)?.region_ids);
    }
    let sea = SeaGrids::parse(&gen_.sea.root)?;
    let cells: usize = sea.grids.iter().map(|g| g.cells.len()).sum();
    let checks = MapChecks::new(&gen_.rd, &gen_.esf.root, &lists, Some(cells))?;
    let problems = check_world(&start.root, &checks)?;
    report.info.push(format!("start position: {} problem(s) (HLCI counts, region list, sea cells)", problems.len()));
    for p in problems.iter().take(20) {
        report.hard(format!("startpos: {p}"));
    }
    let routeless = routeless_nodes(&gen_.trade.root)?;
    report.info.push(format!("trade nodes without a route: {}", routeless.len()));
    for node in routeless {
        report.hard(format!("trade_routes.esf: node {node} has no route"));
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut generated, mut golden, mut theatre) = (None, None, "3".to_string());
    let (mut settlements, mut allow) = (Vec::new(), BTreeSet::new());
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--generated" => generated = Some(PathBuf::from(value()?)),
            "--golden" => golden = Some(PathBuf::from(value()?)),
            "--theatre" => theatre = value()?,
            "--settlement" => settlements.push(value()?),
            "--allow-region" => {
                allow.insert(value()?);
            }
            other => bail!("unknown argument {other}"),
        }
    }
    let (generated, golden) = (
        generated.context("--generated is required")?,
        golden.context("--golden is required")?,
    );
    let mut gen_source = Source::open(&generated)?;
    let gen_ = load(&mut gen_source)?;
    let gold = load(&mut Source::open(&golden)?)?;
    if settlements.is_empty() {
        settlements = vec!["indochina".into(), "sumatra".into()];
    }
    let mut report = Report::default();
    compare_regions(&gen_, &gold, &allow, &mut report);
    compare_pathfinding(&gen_, &gold, &theatre, &settlements, &mut report)?;
    compare_sea(&gen_, &gold, &theatre, &mut report)?;
    check_startpos(&mut gen_source, &gen_, &mut report)?;

    println!("== info");
    for line in &report.info {
        println!("  {line}");
    }
    println!("== component id differences ({}; deliberate: connected passable land component, e.g. 13 instead of 857)", report.components.len());
    for line in report.components.iter().take(30) {
        println!("  {line}");
    }
    println!("== hard deviations ({})", report.hard.len());
    for line in &report.hard {
        println!("  {line}");
    }
    if !report.hard.is_empty() {
        std::process::exit(1);
    }
    println!("OK: no hard deviation");
    Ok(())
}
