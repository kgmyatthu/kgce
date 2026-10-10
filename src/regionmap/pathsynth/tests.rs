//! Tests against the validated experiment outputs under F:/NTW2/work (skipped when a file is missing).

use std::path::{Path, PathBuf};

use super::model::Model;
use super::validate;
use crate::esf::{self, Esf};
use crate::regionmap::pathfinding::Grid;
use crate::regionmap::regions::RegionData;

pub const WORK: &str = "F:/NTW2/work";
pub const AREA: usize = 3;

pub fn vanilla_pf() -> PathBuf {
    Path::new(WORK).join("exp1/backup/campaign_maps/nap_world/pathfinding.esf")
}
pub fn vanilla_regions() -> PathBuf {
    Path::new(WORK).join("exp1/backup/campaign_maps/nap_world/regions.esf")
}
pub fn exp3_pf() -> PathBuf {
    Path::new(WORK).join("exp3/out/data/campaign_maps/nap_world/pathfinding.esf")
}
pub fn exp3_regions() -> PathBuf {
    Path::new(WORK).join("exp3/out/data/campaign_maps/nap_world/regions.esf")
}
pub fn golden_pf() -> PathBuf {
    Path::new(WORK).join("exp5/variantB/out/data/campaign_maps/nap_world/pathfinding.esf")
}
pub fn golden_regions() -> PathBuf {
    Path::new(WORK).join("exp5/variantB/out/data/campaign_maps/nap_world/regions.esf")
}
pub fn exp5_regions() -> PathBuf {
    Path::new(WORK).join("exp5/out/data/campaign_maps/nap_world/regions.esf")
}
pub fn exp5_pf() -> PathBuf {
    Path::new(WORK).join("exp5/out/data/campaign_maps/nap_world/pathfinding.esf")
}

pub fn load(p: &Path) -> Option<(Esf, Vec<u8>)> {
    let bytes = std::fs::read(p).ok()?;
    Some((esf::read(&bytes).expect("ESF parse"), bytes))
}

fn round_trip(path: PathBuf) {
    let Some((mut file, bytes)) = load(&path) else {
        eprintln!("skip: {} missing", path.display());
        return;
    };
    let n = Grid::area_count(&file.root).unwrap();
    for area in 0..n {
        let grid = Grid::parse_area(&file.root, area).unwrap();
        grid.store(&mut file.root).unwrap();
    }
    assert_eq!(esf::write(&file).unwrap(), bytes, "{} does not round trip byte-exact", path.display());
}

#[test]
fn vanilla_round_trip_is_byte_exact() {
    round_trip(vanilla_pf());
}

#[test]
fn exp3_round_trip_is_byte_exact() {
    round_trip(exp3_pf());
}

#[test]
fn model_tables_rebuild_identically() {
    for path in [vanilla_pf(), exp3_pf(), golden_pf()] {
        let Some((mut file, bytes)) = load(&path) else { continue };
        let mut grid = Grid::parse_area(&file.root, AREA).unwrap();
        let model = Model::from_grid(&grid).unwrap();
        model.write_into(&mut grid).unwrap();
        grid.store(&mut file.root).unwrap();
        assert_eq!(esf::write(&file).unwrap(), bytes, "{} tables do not rebuild identically", path.display());
    }
}

fn validated(pf: PathBuf, regions: PathBuf) {
    let (Some((pff, _)), Some((rf, _))) = (load(&pf), load(&regions)) else {
        eprintln!("skip: {} / {} missing", pf.display(), regions.display());
        return;
    };
    let rd = RegionData::parse(&rf.root).unwrap();
    let grid = Grid::parse_area(&pff.root, AREA).unwrap();
    let problems = validate(&grid, &rd);
    assert!(problems.is_empty(), "{} problems in {}:\n{}", problems.len(), pf.display(), problems.join("\n"));
}

#[test]
fn validator_accepts_vanilla() {
    validated(vanilla_pf(), vanilla_regions());
}

#[test]
fn validator_accepts_exp3() {
    validated(exp3_pf(), exp3_regions());
}

#[test]
fn validator_accepts_golden_variant_b() {
    validated(golden_pf(), golden_regions());
}

// ---------------------------------------------------------------------------------------------- synthesis
use super::{Options, SlotKind, Target, TargetOptions, synthesize};
use super::model::{OUTSIDE, MCell};

fn target(rd: &RegionData, key: &str, kind: SlotKind) -> Target {
    Target { region: rd.index(key).unwrap(), settlement_slot_kind: kind, options: TargetOptions::default() }
}

/// Per cell the sorted set of (piece type, path id) (run cells: the run's type and path id).
fn cell_signature(c: &MCell) -> Vec<(u8, u16)> {
    let mut v: Vec<(u8, u16)> = if c.pieces.is_empty() {
        c.run.map(|(a, pid)| vec![((a & 15) as u8, pid)]).unwrap_or_default()
    } else {
        c.pieces.iter().map(|p| (p.ptype(), p.pid)).collect()
    };
    v.sort();
    v.dedup();
    v
}

#[test]
fn burma_synthesis_matches_golden_variant_b() {
    let (Some((pff, _)), Some((rf, _)), Some((gf, _))) = (load(&exp3_pf()), load(&golden_regions()), load(&golden_pf())) else {
        eprintln!("skip: experiment files missing");
        return;
    };
    let rd = RegionData::parse(&rf.root).unwrap();
    let mut grid = Grid::parse_area(&pff.root, AREA).unwrap();
    let t0 = std::time::Instant::now();
    let report = synthesize(&mut grid, &rd, &[target(&rd, "indochina", SlotKind::FourSlot)], &Options::default()).unwrap();
    eprintln!("burma synthesis: {:?}, report {report:#?}", t0.elapsed());
    let problems = validate(&grid, &rd);
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
    let burma = rd.index("indochina").unwrap() as i16;
    let bengal = rd.index("bengal").unwrap() as i16;
    assert_eq!(*grid.region_ids.last().unwrap(), burma, "region list must end with the new region");
    assert_eq!(report.new_groups.len(), 1);
    let mut want = vec![bengal, burma];
    want.sort_by_key(|r| grid.region_ids.iter().position(|x| x == r));
    assert_eq!(report.new_groups[0].0, want);
    let sea = grid.paths.iter().position(|p| p.is_empty()).unwrap();
    assert!(grid.paths[..sea].iter().all(|p| p.len() == 1) && grid.paths[sea + 1..].iter().all(|p| p.len() >= 2));
    let landings = report.per_region[0].landings;
    assert!((70..=76).contains(&landings), "{landings} landing crossings, expected 73 +-3");
    // octagon node around Rangoon
    let model = Model::from_grid(&grid).unwrap();
    let path = grid.single_path(burma).unwrap();
    let rangoon = [683.9f64, 119.8];
    let mut node = None;
    for (idx, cell) in model.cells.iter().enumerate() {
        for p in &cell.pieces {
            if p.ptype() == 7 && p.pid == path {
                let poly: Vec<[f64; 2]> = p.pts.iter().map(|q| [(q[0] as i64 + model.origin[0] as i64) as f64 / 1048576.0, (q[1] as i64 + model.origin[1] as i64) as f64 / 1048576.0]).collect();
                if crate::regionmap::geometry::point_in_ring(rangoon, &poly) {
                    node = Some((idx, poly.len()));
                }
            }
        }
    }
    assert!(node.is_some(), "no type-7 node piece around Rangoon");
    // band cells carry the group path
    let gpid = report.new_groups[0].1;
    let count_band = |m: &Model, pid: u16| m.cells.iter().filter(|c| c.pieces.iter().any(|p| p.ptype() == 0 && p.pid == pid) || c.run.is_some_and(|(a, p)| c.pieces.is_empty() && a & 15 == 0 && p == pid)).count();
    let band_cells = count_band(&model, gpid);
    let golden = Grid::parse_area(&gf.root, AREA).unwrap();
    let gpos = golden.paths.iter().position(|p| p.len() == 2 && p.contains(&burma) && p.contains(&bengal)).expect("golden has the Burma|Bengal group");
    let golden_band = count_band(&Model::from_grid(&golden).unwrap(), gpos as u16);
    eprintln!("band cells: {band_cells} (golden {golden_band})");
    assert!(band_cells > 0 && (band_cells as f64 - golden_band as f64).abs() <= 0.2 * golden_band as f64 + 2.0, "{band_cells} band cells, golden {golden_band}");
    let gm = Model::from_grid(&golden).unwrap();
    assert_eq!((model.cols, model.rows), (gm.cols, gm.rows));
    let same = model.cells.iter().zip(&gm.cells).filter(|(a, b)| cell_signature(a) == cell_signature(b)).count();
    let frac = same as f64 / model.cells.len() as f64;
    eprintln!("cells with the golden type/pid set: {same}/{} = {frac:.5}", model.cells.len());
    assert!(frac >= 0.99, "only {:.4} of the cells agree with the golden file", frac);
    eprintln!("path table equals the golden one: {}", grid.paths == golden.paths);
    let _ = OUTSIDE;
}

#[test]
fn burma_and_sumatra_synthesis_is_valid() {
    let (Some((pff, _)), Some((rf, _))) = (load(&exp3_pf()), load(&exp5_regions())) else {
        eprintln!("skip: experiment files missing");
        return;
    };
    let rd = RegionData::parse(&rf.root).unwrap();
    let mut grid = Grid::parse_area(&pff.root, AREA).unwrap();
    let t0 = std::time::Instant::now();
    let targets = [target(&rd, "indochina", SlotKind::FourSlot), target(&rd, "sumatra", SlotKind::OneSlot)];
    let report = synthesize(&mut grid, &rd, &targets, &Options::default()).unwrap();
    eprintln!("burma+sumatra synthesis: {:?}, report {report:#?}", t0.elapsed());
    let problems = validate(&grid, &rd);
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
    let unexplorable = rd.index("unexplorable").unwrap() as i16;
    assert!(grid.paths.iter().all(|p| !(p.len() > 1 && p.contains(&unexplorable))), "a group path contains unexplorable");
    assert_eq!(report.per_region.len(), 2);
}

#[test]
fn extend_reproduces_exp3_grid() {
    use crate::regionmap::pathfinding::TransitionArea;
    let (Some((pff, _)), Some((rf, _)), Some((ef, _))) = (load(&vanilla_pf()), load(&vanilla_regions()), load(&exp3_pf())) else {
        eprintln!("skip: experiment files missing");
        return;
    };
    let rd = RegionData::parse(&rf.root).unwrap();
    let mut grid = Grid::parse_area(&pff.root, AREA).unwrap();
    let east = TransitionArea { id: 11, old: [[660.63, 17.73], [678.66, 118.5]], new: [[770.63, 17.73], [788.66, 118.5]] };
    let west = TransitionArea { id: 10, old: [[436.42, 17.49], [454.61, 105.5]], new: [[436.42, 17.49], [454.61, 105.5]] };
    let report = grid.extend([[440.0, -80.0], [790.0, 300.0]], &rd, &[west, east]).unwrap();
    assert_eq!((report.frame_cells_to_sea, report.frame_pieces_to_sea), (0, 0), "India has no impassable frame at its moved edges");
    eprintln!("extend report: {report:?}");
    let expect = Grid::parse_area(&ef.root, AREA).unwrap();
    assert_eq!((grid.cols, grid.rows), (expect.cols, expect.rows));
    assert_eq!((grid.x0, grid.y0, grid.origin_index), (expect.x0, expect.y0, expect.origin_index));
    let (m, e) = (Model::from_grid(&grid).unwrap(), Model::from_grid(&expect).unwrap());
    let same_sig = m.cells.iter().zip(&e.cells).filter(|(a, b)| cell_signature(a) == cell_signature(b)).count();
    let same_hdr = m.cells.iter().zip(&e.cells).filter(|(a, b)| a.hdr == b.hdr).count();
    let same_word = m.cells.iter().zip(&e.cells).filter(|(a, b)| a.run == b.run && a.pieces.iter().map(|p| p.a).eq(b.pieces.iter().map(|p| p.a))).count();
    let n = m.cells.len();
    eprintln!("cells equal to exp3: type/pid {same_sig}/{n}, header {same_hdr}/{n}, words {same_word}/{n}");
    assert!(same_sig as f64 / n as f64 >= 0.999);
    assert!(same_hdr as f64 / n as f64 >= 0.999);
    assert!(same_word as f64 / n as f64 >= 0.999);
}

fn passable_sea(c: &crate::regionmap::pathfinding::Cell) -> bool {
    match c.empty {
        Some((a, _)) if c.pieces.is_empty() => a & 15 == 1 && a >> 24 != 0,
        _ => c.pieces.iter().any(|p| p.a & 15 == 1 && p.a >> 24 != 0),
    }
}

fn has_outside(c: &crate::regionmap::pathfinding::Cell) -> bool {
    match c.empty {
        Some((a, _)) if c.pieces.is_empty() => a & 15 == 2,
        _ => c.pieces.iter().any(|p| p.a & 15 == 2),
    }
}

/// Cell kinds and paths of a cell, independent of the a-words that the neighbours influence.
fn cell_kinds(c: &crate::regionmap::pathfinding::Cell) -> (Vec<(u32, u16)>, Option<(u32, u16)>, [u8; 8]) {
    (c.pieces.iter().map(|p| (p.a & 15, p.path_id())).collect(), c.empty.map(|(a, id)| (a & 15, id)), c.header)
}

/// Cells whose centre lies in `rect`.
fn cells_centred_in(grid: &Grid, rect: crate::regionmap::geometry::Rect) -> Vec<usize> {
    (0..grid.cells.len())
        .filter(|&i| {
            let o = grid.cell_origin(i);
            let (cx, cy) = (o[0] + grid.size / 2.0, o[1] + grid.size / 2.0);
            f64::from(rect[0][0]) <= cx && cx <= f64::from(rect[1][0]) && f64::from(rect[0][1]) <= cy && cy <= f64::from(rect[1][1])
        })
        .collect()
}

/// Passable-sea cells (4-neighbourhood) reachable from `from`.
fn sea_component(grid: &Grid, from: usize) -> std::collections::BTreeSet<usize> {
    let mut seen = std::collections::BTreeSet::new();
    if !passable_sea(&grid.cells[from]) {
        return seen;
    }
    let mut queue = std::collections::VecDeque::from([from]);
    seen.insert(from);
    while let Some(i) = queue.pop_front() {
        let (c, r) = ((i % grid.cols) as i64, (i / grid.cols) as i64);
        for (dc, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nc, nr) = (c + dc, r + dr);
            if nc < 0 || nr < 0 || nc >= grid.cols as i64 || nr >= grid.rows as i64 {
                continue;
            }
            let n = nr as usize * grid.cols + nc as usize;
            if passable_sea(&grid.cells[n]) && seen.insert(n) {
                queue.push_back(n);
            }
        }
    }
    seen
}

/// America's enlarged west edge: the exit slides to the new edge, but the stock west edge was wrapped in an impassable
/// frame whose outside cells stayed behind as a wall across the sea, sealing the moved exit off from every ship.
#[test]
fn america_west_extension_clears_the_old_frame_and_keeps_the_moved_exit_reachable() {
    use crate::regionmap::pathfinding::TransitionArea;
    use crate::regionmap::stages::{new_problems, problem_identities};
    let (Some((mut pff, _)), Some((mut rf, _))) = (load(&vanilla_pf()), load(&vanilla_regions())) else {
        eprintln!("skip: vanilla files missing");
        return;
    };
    let mut rd = RegionData::parse(&rf.root).unwrap();
    let area = rd.theatre_rects().iter().position(|(key, _)| key == "1").unwrap();
    let stock = Grid::parse_area(&pff.root, area).unwrap();
    let preserved = rd.transition_areas("1").unwrap();
    let baseline = problem_identities(&stock, &rd);

    let change = rd.extend_theatre("1", [[-888.0, 0.0], [-340.0, 460.0]]).unwrap();
    assert_eq!(change.moved.len(), 1, "only the exit on the moved west edge slides");
    let exit = change.moved[0];
    assert_eq!((exit.id, exit.old, exit.new), (1, [[-760.0, 80.0], [-740.0, 120.0]], [[-888.0, 80.0], [-868.0, 120.0]]));
    let moves: Vec<TransitionArea> = change.moved.iter().map(|m| TransitionArea { id: m.id, old: m.old, new: m.new }).collect();
    let mut grid = stock.clone();
    let report = grid.extend(change.new_rect, &rd, &moves).unwrap();
    assert_eq!(report.offset, [64, 0]);
    let old_to_new = |i: usize| (i / stock.cols + report.offset[1]) * grid.cols + i % stock.cols + report.offset[0];

    // the moved exit lies on traversable sea ...
    let zone = cells_centred_in(&grid, exit.new);
    assert!(!zone.is_empty() && zone.iter().all(|&i| passable_sea(&grid.cells[i])), "the moved exit is not open sea");
    // ... that is connected to the sea the stock exit used and to the preserved south exit
    let reach = sea_component(&grid, zone[0]);
    let stock_exit_sea: Vec<usize> = cells_centred_in(&stock, exit.old).into_iter().filter(|&i| passable_sea(&stock.cells[i])).collect();
    assert!(!stock_exit_sea.is_empty());
    assert!(stock_exit_sea.iter().all(|&i| reach.contains(&old_to_new(i))), "the moved exit is walled off from the stock exit sea");
    for ta in preserved.iter().filter(|t| t.id == 2) {
        let south: Vec<usize> = cells_centred_in(&stock, ta.rect).into_iter().filter(|&i| passable_sea(&stock.cells[i])).collect();
        assert!(!south.is_empty() && south.iter().all(|&i| reach.contains(&old_to_new(i))), "the moved exit cannot reach exit {}", ta.id);
    }

    // no stale mask: the old west frame (outside cells x -760..-750) is open sea along the exit and its neighbourhood
    for i in 0..stock.cells.len() {
        let o = stock.cell_origin(i);
        let (cx, cy) = (o[0] + 1.0, o[1] + 1.0);
        if -760.0 < cx && cx < -750.0 && (10.0..=120.0).contains(&cy) {
            let cell = &grid.cells[old_to_new(i)];
            assert!(!has_outside(cell) && passable_sea(cell), "stale outside frame at ({cx}, {cy})");
        }
    }

    // preserved exits keep their exact cells
    for ta in preserved.iter().filter(|t| t.id != exit.id) {
        for i in cells_centred_in(&stock, ta.rect) {
            assert_eq!(cell_kinds(&grid.cells[old_to_new(i)]), cell_kinds(&stock.cells[i]), "preserved exit {} changed", ta.id);
        }
    }
    let new_problems = new_problems(&baseline, &grid, &rd);
    assert!(new_problems.is_empty(), "{} new rule problems:\n{}", new_problems.len(), new_problems.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));

    // both files survive serialization with the moved exit, the preserved exits and the cleared frame intact
    grid.store(&mut pff.root).unwrap();
    let back = Grid::parse_area(&esf::read(&esf::write(&pff).unwrap()).unwrap().root, area).unwrap();
    assert_eq!((back.cols, back.rows, back.x0, back.y0, back.origin_index), (grid.cols, grid.rows, grid.x0, grid.y0, grid.origin_index));
    assert!(
        back.cells.iter().zip(&grid.cells).all(|(a, b)| {
            a.header == b.header && a.empty == b.empty && a.pieces.iter().map(|p| (p.a, p.b)).eq(b.pieces.iter().map(|p| (p.a, p.b)))
        }),
        "the extended grid changed through store / write / read / parse"
    );
    rd.store(&mut rf.root).unwrap();
    let back = RegionData::parse(&esf::read(&esf::write(&rf).unwrap()).unwrap().root).unwrap();
    assert_eq!(back.theatre_rect("1").unwrap(), change.new_rect);
    let exits = back.transition_areas("1").unwrap();
    assert_eq!(exits.iter().find(|t| t.id == exit.id).unwrap().rect, exit.new, "the moved exit was not stored");
    for t in preserved.iter().filter(|t| t.id != exit.id) {
        assert_eq!(exits.iter().find(|u| u.id == t.id).unwrap().rect, t.rect, "preserved exit {} changed on store", t.id);
    }
}

/// A two-region world and grid with no external files: a passable ocean, an L-shaped impassable frame (a west band
/// 16 units wide, wider than any fitted reach, plus a 4-unit south band that stays an edge frame), an isolated
/// impassable square in the interior, and a land backdrop block on the west edge.
fn synthetic_frame_world() -> (Grid, RegionData) {
    use crate::regionmap::pathfinding::{Cell, TRANSITION};
    use crate::regionmap::regions::{Area, Outline, Region};
    let mut vertices: Vec<[f32; 2]> = Vec::new();
    let mut region = |key: &str, kind: &str, areas: Vec<(bool, Vec<[f32; 2]>)>| -> Region {
        let areas = areas
            .into_iter()
            .map(|(passable, ring)| {
                let first = vertices.len() as u32;
                vertices.extend_from_slice(&ring);
                Area {
                    land_bridge: false,
                    channel_coast: false,
                    passable,
                    adjoining_id: 0,
                    faces: vec![],
                    outlines: vec![Outline { outer: true, ring: (first..first + ring.len() as u32).collect(), runs: vec![] }],
                    impassable_id: if passable { 65535 } else { 286 },
                    island_id: 0,
                }
            })
            .collect();
        Region { key: key.into(), kind: kind.into(), areas, tail: 0, settlement: None }
    };
    let regions = vec![
        region("ocean", "sea", vec![(true, vec![[-30.0, -10.0], [50.0, -10.0], [50.0, 40.0], [-30.0, 40.0]])]),
        region(
            "impassable",
            "sea",
            vec![
                (false, vec![[0.0, 0.0], [40.0, 0.0], [40.0, 4.0], [16.0, 4.0], [16.0, 20.0], [0.0, 20.0]]),
                (false, vec![[24.0, 10.0], [30.0, 10.0], [30.0, 16.0], [24.0, 16.0]]),
            ],
        ),
        region("backdrop", "land", vec![(false, vec![[0.0, 20.0], [16.0, 20.0], [16.0, 28.0], [0.0, 28.0]])]),
    ];
    let rd = RegionData { vertices, regions, theatres: vec![] };
    let (cols, rows) = (20usize, 14usize);
    let mut cells = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let (x, y) = (2.0 * c as f64 + 1.0, 2.0 * r as f64 + 1.0);
            let outside = (x < 16.0 && y < 20.0) || y < 4.0 || (x < 16.0 && y >= 20.0) || (24.0 < x && x < 30.0 && 10.0 < y && y < 16.0);
            cells.push(if outside {
                Cell { header: [0x1f, 0xff, 0x1f, 0xff, 0xff, 0x1f, 0xff, 0x1f], pieces: vec![], empty: Some((2, TRANSITION)) }
            } else {
                Cell { header: [0x14; 8], pieces: vec![], empty: Some((0xff11_0111, 0)) }
            });
        }
    }
    let grid = Grid {
        area: 0, x0: 0.0, y0: 0.0, size: 2.0, cols, rows,
        origin_index: [640, 320], region_ids: vec![], paths: vec![vec![]], cells,
        vertices: vec![[0, 0]; 4], polylines: [(0, vec![0, 2, 3, 1])].into(),
    };
    (grid, rd)
}

/// A moved west exit must reach the old sea whatever the width of the impassable frame that hugged the old edge;
/// only that frame is released (down to its diagonal split against the unmoved south band): land backdrop on the old
/// edge, the unmoved edge's band and an isolated interior impassable area keep their outside cells.
#[test]
fn frame_wider_than_any_fitted_reach_is_released_without_touching_other_outside_cells() {
    use crate::regionmap::pathfinding::TransitionArea;
    let (stock, rd) = synthetic_frame_world();
    let mut grid = stock.clone();
    let exit = TransitionArea { id: 1, old: [[0.0, 6.0], [4.0, 14.0]], new: [[-24.0, 6.0], [-20.0, 14.0]] };
    let report = grid.extend([[-24.0, 0.0], [40.0, 28.0]], &rd, &[exit]).unwrap();
    assert_eq!(report.offset, [12, 0]);
    let new_index = |c: usize, r: usize| r * grid.cols + c + report.offset[0];
    let in_frame = |c: usize, r: usize| {
        let (x, y) = (2.0 * c as f64 + 1.0, 2.0 * r as f64 + 1.0);
        (x < 16.0 && y < 20.0) || y < 4.0
    };
    let mut released = 0;
    for r in 0..stock.rows {
        for c in 0..stock.cols {
            let cell = &grid.cells[new_index(c, r)];
            if in_frame(c, r) && c < r {
                // strictly nearer to the moved west edge than to the unmoved south edge
                assert!(passable_sea(cell) && !has_outside(cell), "frame cell ({c}, {r}) kept its outside wall");
                released += 1;
            } else if in_frame(c, r) {
                assert!(has_outside(cell), "unmoved south band cell ({c}, {r}) was released");
            }
        }
    }
    assert_eq!(report.frame_cells_to_sea, released);
    assert!(grid.cells[new_index(7, 9)].empty.is_some_and(|(a, _)| a & 15 == 1), "the 14-unit-deep part of the frame must be released");
    for r in 10..14 {
        for c in 0..8 {
            assert!(has_outside(&grid.cells[new_index(c, r)]), "land backdrop cell ({c}, {r}) was released");
        }
    }
    for r in 5..8 {
        for c in 12..15 {
            assert!(has_outside(&grid.cells[new_index(c, r)]), "isolated impassable area cell ({c}, {r}) was released");
        }
    }
    // the moved exit lies on new sea that is connected to the old sea through the released frame
    let zone = cells_centred_in(&grid, exit.new);
    assert!(!zone.is_empty() && zone.iter().all(|&i| passable_sea(&grid.cells[i])));
    let reach = sea_component(&grid, zone[0]);
    assert!(zone.iter().all(|i| reach.contains(i)));
    assert!(reach.contains(&new_index(10, 5)), "the moved exit is walled off from the old sea");
}

/// India's enlarged east edge has no frame (the validated experiment output must not change): the exit slides to the new
/// edge, every cell its partially covered window touches is either open sea or genuine outside (type 2, `TRANSITION`)
/// navigation geometry, and the old window keeps no outside piece behind.
#[test]
fn india_extension_moves_the_east_exit_onto_coherent_navigation_geometry() {
    use crate::regionmap::pathfinding::{TRANSITION, TransitionArea};
    let (Some((pff, _)), Some((rf, _))) = (load(&vanilla_pf()), load(&vanilla_regions())) else {
        eprintln!("skip: vanilla files missing");
        return;
    };
    let mut rd = RegionData::parse(&rf.root).unwrap();
    let stock = Grid::parse_area(&pff.root, AREA).unwrap();
    let mut grid = stock.clone();
    let change = rd.extend_theatre("3", [[440.0, -80.0], [790.0, 300.0]]).unwrap();
    let moves: Vec<TransitionArea> = change.moved.iter().map(|m| TransitionArea { id: m.id, old: m.old, new: m.new }).collect();
    assert_eq!(change.moved.iter().map(|m| m.id).collect::<Vec<_>>(), vec![11]);
    let report = grid.extend(change.new_rect, &rd, &moves).unwrap();
    assert_eq!((report.frame_cells_to_sea, report.frame_pieces_to_sea), (0, 0), "India has no impassable frame at its moved edges");
    let exit = change.moved[0];
    let old_to_new = |i: usize| (i / stock.cols + report.offset[1]) * grid.cols + i % stock.cols + report.offset[0];

    // the 770.63..788.66 window is not cell aligned: every cell that overlaps it at all is considered
    let touched: Vec<usize> = (0..grid.cells.len())
        .filter(|&i| {
            let o = grid.cell_origin(i);
            let s = grid.size;
            f64::from(exit.new[0][0]) < o[0] + s && o[0] < f64::from(exit.new[1][0]) && f64::from(exit.new[0][1]) < o[1] + s && o[1] < f64::from(exit.new[1][1])
        })
        .collect();
    assert!(touched.len() > cells_centred_in(&grid, exit.new).len(), "the window must cut cells in two");
    let (mut sea, mut outside) = (0, 0);
    for &i in &touched {
        let cell = &grid.cells[i];
        if passable_sea(cell) {
            sea += 1;
        } else {
            // new outside cells are whole type-2 runs of the TRANSITION path with the outside header
            assert_eq!(cell.empty.map(|(a, id)| (a, id)), Some((0x0000_0002, TRANSITION)), "cell {i} of the moved exit is neither sea nor outside geometry: {cell:?}");
            assert!(cell.pieces.is_empty() && has_outside(cell));
            outside += 1;
        }
    }
    assert!(sea > outside, "{sea} sea cells against {outside} outside cells in the moved exit window");
    let reach = sea_component(&grid, *touched.iter().find(|&&i| passable_sea(&grid.cells[i])).unwrap());
    assert!(touched.iter().filter(|&&i| passable_sea(&grid.cells[i])).all(|i| reach.contains(i)), "part of the moved east exit is cut off");

    // the exit's old window keeps no outside piece (its sea is plain sea again)
    for i in cells_centred_in(&stock, exit.old) {
        assert!(grid.cells[old_to_new(i)].pieces.iter().all(|p| p.a & 15 != 2), "stale outside piece in the old east exit window, cell {i}");
    }
    // the unmoved west exit keeps its exact cells
    for i in cells_centred_in(&stock, [[436.42, 17.49], [454.61, 105.5]]) {
        assert_eq!(cell_kinds(&grid.cells[old_to_new(i)]), cell_kinds(&stock.cells[i]), "the preserved west exit changed at cell {i}");
    }
}

/// The connected atlas accepts India's enlarged rectangle but America's original one: extending for an atlas project
/// moves only India's east exit, and the atlas's uniform chart transform keeps the exit's inset from the theatre edge.
#[test]
fn atlas_project_moves_only_the_india_exit_and_the_chart_transform_keeps_its_inset() {
    use crate::atlas::Plan;
    use crate::model::Project;
    use crate::regionmap::chain;
    let Some((rf, _)) = load(&vanilla_regions()) else {
        eprintln!("skip: vanilla regions missing");
        return;
    };
    let mut rd = RegionData::parse(&rf.root).unwrap();
    let america = (rd.theatre_rect("1").unwrap(), rd.transition_areas("1").unwrap());
    let mut project = Project::default();
    project.atlas = Some(Plan::proven());
    let changes = chain::extend_for_project(&mut rd, &project).unwrap();
    assert_eq!(changes.iter().map(|c| c.theatre.as_str()).collect::<Vec<_>>(), vec!["3"], "only India differs from the source rectangles");
    assert_eq!((rd.theatre_rect("1").unwrap(), rd.transition_areas("1").unwrap()), america, "America keeps its rectangle and exits");
    let change = &changes[0];
    assert_eq!(change.moved.iter().map(|m| m.id).collect::<Vec<_>>(), vec![11]);
    let exit = change.moved[0];
    let plan = project.atlas.as_ref().unwrap();
    let chart = plan.chart("3").unwrap();
    let map = |p: [f32; 2]| plan.xy(p.map(f64::from), chart);
    let (lo, hi) = (map(change.new_rect[0]), map(change.new_rect[1]));
    let (elo, ehi) = (map(exit.new[0]), map(exit.new[1]));
    assert!(lo[0] <= elo[0] && ehi[0] <= hi[0] && lo[1] <= elo[1] && ehi[1] <= hi[1], "the mapped exit leaves the mapped theatre");
    let source_inset = f64::from(change.new_rect[1][0] - exit.new[1][0]) / f64::from(change.new_rect[1][0] - change.new_rect[0][0]);
    let atlas_inset = (hi[0] - ehi[0]) / (hi[0] - lo[0]);
    assert!((source_inset - atlas_inset).abs() < 1e-9, "the chart transform changed the exit's inset: {source_inset} vs {atlas_inset}");
}

/// Mixed corner-relative and absolute vertices expose half-cell shifts that whole-cell runs hide.
fn mixed_vertex_grid() -> Grid {
    use crate::regionmap::pathfinding::{Cell, Piece, fixed_of};
    Grid {
        area: 0, x0: 440.0, y0: 0.0, size: 2.0, cols: 2, rows: 1,
        origin_index: [860, 320], region_ids: vec![], paths: vec![vec![]],
        cells: vec![
            Cell {
                header: [0; 8],
                pieces: vec![Piece { a: 1, b: 0 }, Piece { a: 1, b: 5 }],
                empty: None,
            },
            Cell { header: [0; 8], pieces: vec![], empty: Some((1, 0)) },
        ],
        vertices: vec![[0, 0], [0, 0], [0, 0], [0, 0], [fixed_of(442.0), fixed_of(1.0)], [fixed_of(440.0), fixed_of(1.0)]],
        polylines: [(0, vec![0, 2, 4, 5]), (5, vec![5, 4, 3, 1])].into(),
    }
}

#[test]
fn off_grid_extension_is_rejected_without_mutating_polygons() {
    let rd = RegionData { vertices: vec![], regions: vec![], theatres: vec![] };
    for rect in [
        [[440.0, -83.0], [801.0, 280.0]],
        [[440.0, -84.0], [801.0, 280.0]],
        [[439.5, -84.0], [802.0, 280.0]],
        [[440.0, -84.0], [802.0, 280.5]],
    ] {
        let mut grid = mixed_vertex_grid();
        let before: Vec<_> = grid.cells[0].pieces.iter().map(|p| grid.piece_polygon(0, p).unwrap()).collect();
        let error = grid.extend(rect, &rd, &[]).expect_err("fractional cell bounds must not be rounded");
        assert!(error.to_string().contains("not aligned"));
        assert_eq!((grid.x0, grid.y0, grid.cols, grid.rows), (440.0, 0.0, 2, 1));
        let after: Vec<_> = grid.cells[0].pieces.iter().map(|p| grid.piece_polygon(0, p).unwrap()).collect();
        assert_eq!(after, before);
    }
}

#[test]
fn odd_theatre_edges_preserve_original_world_polygons() {
    use crate::regionmap::regions::Theatre;
    let mut rd = RegionData {
        vertices: vec![], regions: vec![],
        theatres: vec![Theatre {
            key: "3".into(), rect: [[440.0, 0.0], [444.0, 2.0]],
            transitions: vec![], extended: false,
        }],
    };
    let change = rd.extend_theatre("3", [[440.0, -83.0], [801.0, 280.0]]).unwrap();
    assert_eq!(change.new_rect, [[440.0, -84.0], [802.0, 280.0]]);
    let mut grid = mixed_vertex_grid();
    let before: Vec<_> = grid.cells[0].pieces.iter().map(|p| grid.piece_polygon(0, p).unwrap()).collect();
    let report = grid.extend(change.new_rect, &rd, &[]).unwrap();
    let index = report.offset[1] * grid.cols + report.offset[0];
    assert_eq!(grid.cell_origin(index), [440.0, 0.0]);
    let after: Vec<_> = grid.cells[index].pieces.iter().map(|p| grid.piece_polygon(index, p).unwrap()).collect();
    assert_eq!(after, before, "extension must not distort or translate the old polygons");
    assert_eq!([grid.x0 + grid.cols as f64 * grid.size, grid.y0 + grid.rows as f64 * grid.size], [802.0, 280.0]);
}

fn centroid_of_ring(pts: &[[f64; 2]]) -> [f64; 2] {
    crate::regionmap::pathfinding::centroid(pts)
}

#[test]
fn split_gets_node_band_and_group() {
    use crate::regionmap::geometry::{point_segment_distance, polygon_area};
    use crate::regionmap::regions::{Split, SettlementSpec, settlement_center};
    let (Some((pff, _)), Some((rf, _))) = (load(&exp3_pf()), load(&exp3_regions())) else {
        eprintln!("skip: experiment files missing");
        return;
    };
    let rd0 = RegionData::parse(&rf.root).unwrap();
    let grid0 = Grid::parse_area(&pff.root, AREA).unwrap();
    // the biggest single-outline passable area of a settled region that the grid lists
    let mut best: Option<(usize, usize, f64)> = None;
    for &r in &grid0.region_ids {
        let r = r as usize;
        let reg = &rd0.regions[r];
        if reg.kind != "land" || reg.settlement.is_none() {
            continue;
        }
        for (ai, a) in reg.areas.iter().enumerate() {
            if !a.passable || a.outlines.len() != 1 || a.outlines[0].ring.len() < 40 {
                continue;
            }
            let pts: Vec<[f64; 2]> = a.outlines[0].ring.iter().map(|&v| rd0.v(v)).collect();
            let ar = polygon_area(&pts);
            if best.is_none_or(|b| ar > b.2) {
                best = Some((r, ai, ar));
            }
        }
    }
    let (src, ai, _) = best.expect("a splittable region");
    let n = rd0.regions[src].areas[ai].outlines[0].ring.len();
    let old_settlement = settlement_center(&rd0.regions[src]).unwrap();
    let mut done = None;
    'search: for a in (0..n).step_by(3) {
        for b in (0..n).step_by(5) {
            if a == b {
                continue;
            }
            let mut rd = rd0.clone();
            let probe = Split { key: rd0.regions[src].key.clone(), area: ai, a, b, interior: vec![], new_key: "splitnew".into(), settlement: SettlementSpec { name: "x".into(), center: [0.0, 0.0] } };
            let Ok(new) = rd.split(&probe) else { continue };
            // centre: centroid of the largest triangle of the new area
            let na = &rd.regions[new].areas[0];
            let tri = na.faces.iter().max_by(|x, y| {
                let ar = |t: &[u32; 3]| polygon_area(&[rd.v(t[0]), rd.v(t[1]), rd.v(t[2])]);
                ar(x).partial_cmp(&ar(y)).unwrap()
            });
            let Some(t) = tri else { continue };
            let c = centroid_of_ring(&[rd.v(t[0]), rd.v(t[1]), rd.v(t[2])]);
            let ring: Vec<[f64; 2]> = na.outlines[0].ring.iter().map(|&v| rd.v(v)).collect();
            let dmin = (0..ring.len()).map(|i| point_segment_distance(c, ring[i], ring[(i + 1) % ring.len()])).fold(f64::MAX, f64::min);
            if dmin < 4.0 || polygon_area(&ring) < 800.0 || rd.in_region(new, [old_settlement[0] as f64, old_settlement[1] as f64]) {
                continue;
            }
            let center = [c[0] as f32, c[1] as f32];
            let mut rd = rd0.clone();
            let new = rd.split(&Split { settlement: SettlementSpec { name: "x".into(), center }, ..probe }).unwrap();
            done = Some((rd, new, center, a, b));
            break 'search;
        }
    }
    let (rd, new, center, a, b) = done.expect("a usable cut");
    eprintln!("split {} area {ai} ring {n} cut {a}->{b}, new region {new} centre {center:?}", rd.regions[src].key);
    let mut grid = grid0.clone();
    let before = grid.paths.len();
    let in_new = |p: [f64; 2]| rd.in_region(new, p);
    grid.relabel(src as i16, new as i16, &in_new).unwrap();
    let t0 = std::time::Instant::now();
    let report = super::synthesize_split(&mut grid, &rd, src, new, SlotKind::FourSlot, &Options::default()).unwrap();
    eprintln!("split synthesis {:?}: {report:#?}", t0.elapsed());
    let problems = validate(&grid, &rd);
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
    // the node holds the settlement
    let model = Model::from_grid(&grid).unwrap();
    let sp = grid.single_path(new as i16).unwrap();
    let p = [((center[0] as f64 - grid.x0) * 1048576.0).round(), ((center[1] as f64 - grid.y0) * 1048576.0).round()];
    let hit = model.cells.iter().flat_map(|c| c.pieces.iter()).filter(|q| q.ptype() == 7 && q.pid == sp).any(|q| {
        let poly: Vec<[f64; 2]> = q.pts.iter().map(|v| [v[0] as f64, v[1] as f64]).collect();
        crate::regionmap::geometry::point_in_ring(p, &poly)
    });
    assert!(hit, "no type-7 node of the new region around its settlement");
    // a (source,new) group exists, every pid is in the table, group members ascend
    let pos = |r: i16| grid.region_ids.iter().position(|&x| x == r).unwrap();
    let gid = grid.paths.iter().position(|g| g.len() == 2 && g.contains(&(src as i16)) && g.contains(&(new as i16)));
    assert!(gid.is_some(), "no ({src},{new}) group path; paths {:?}", &grid.paths[grid.paths.iter().position(|p| p.is_empty()).unwrap()..]);
    for g in grid.paths.iter().filter(|g| g.len() > 1) {
        assert!(g.windows(2).all(|w| pos(w[0]) < pos(w[1])), "group {g:?} not ascending");
    }
    let used: std::collections::BTreeSet<u16> = model.cells.iter().flat_map(|c| if c.pieces.is_empty() { c.run.map(|r| r.1).into_iter().collect::<Vec<_>>() } else { c.pieces.iter().map(|p| p.pid).collect() }).collect();
    assert!(used.iter().all(|&p| p == 1023 || (p as usize) < grid.paths.len()), "stale pid");
    assert!(used.contains(&(gid.unwrap() as u16)), "the group path is not used");
    eprintln!("paths before {before}, after {}", grid.paths.len());
    // every land piece of the new region lies on its side
    let sea = grid.paths.iter().position(|p| p.is_empty()).unwrap();
    let mut wrong = 0;
    for (idx, cell) in model.cells.iter().enumerate() {
        let (c, r) = (idx % model.cols, idx / model.cols);
        for q in &cell.pieces {
            if q.ptype() != 0 || (q.pid as usize) > sea {
                continue;
            }
            let poly: Vec<[f64; 2]> = q.pts.iter().map(|v| [(v[0] as i64 + model.origin[0] as i64) as f64 / 1048576.0, (v[1] as i64 + model.origin[1] as i64) as f64 / 1048576.0]).collect();
            let cen = centroid_of_ring(&poly);
            let _ = (c, r);
            let in_n = rd.in_region(new, cen);
            if (q.pid == sp) != in_n && polygon_area(&poly) > 0.2 && (grid.paths[q.pid as usize][0] == new as i16 || grid.paths[q.pid as usize][0] == src as i16) {
                wrong += 1;
            }
        }
    }
    assert!(wrong <= 3, "{wrong} single-path land pieces lie on the wrong side of the cut");
}

#[test]
fn split_of_a_fresh_region_synthesises_both_halves() {
    use crate::regionmap::geometry::{point_segment_distance, polygon_area};
    use crate::regionmap::regions::{Split, SettlementSpec, settlement_center};
    let (Some((pff, _)), Some((rf, _))) = (load(&exp3_pf()), load(&golden_regions())) else {
        eprintln!("skip: experiment files missing");
        return;
    };
    let rd0 = RegionData::parse(&rf.root).unwrap();
    let grid0 = Grid::parse_area(&pff.root, AREA).unwrap();
    let src = rd0.index("indochina").unwrap();
    let ai = rd0.regions[src].areas.iter().position(|a| a.passable && a.outlines.len() == 1).expect("the merged Burma area");
    let n = rd0.regions[src].areas[ai].outlines[0].ring.len();
    let rangoon = settlement_center(&rd0.regions[src]).unwrap();
    let mut tried = 0;
    let mut ok = None;
    'search: for a in (0..n).step_by(2) {
        for b in (0..n).step_by(3) {
            if a == b {
                continue;
            }
            let mut rd = rd0.clone();
            let probe = Split { key: "indochina".into(), area: ai, a, b, interior: vec![], new_key: "arakan".into(), settlement: SettlementSpec { name: "x".into(), center: [0.0, 0.0] } };
            let Ok(new) = rd.split(&probe) else { continue };
            let na = &rd.regions[new].areas[0];
            let ring: Vec<[f64; 2]> = na.outlines[0].ring.iter().map(|&v| rd.v(v)).collect();
            if polygon_area(&ring) < 300.0 || rd.in_region(new, [rangoon[0] as f64, rangoon[1] as f64]) {
                continue;
            }
            let Some(t) = na.faces.iter().max_by(|x, y| {
                let ar = |t: &[u32; 3]| polygon_area(&[rd.v(t[0]), rd.v(t[1]), rd.v(t[2])]);
                ar(x).partial_cmp(&ar(y)).unwrap()
            }) else { continue };
            let c = centroid_of_ring(&[rd.v(t[0]), rd.v(t[1]), rd.v(t[2])]);
            let dmin = (0..ring.len()).map(|i| point_segment_distance(c, ring[i], ring[(i + 1) % ring.len()])).fold(f64::MAX, f64::min);
            let sring: Vec<[f64; 2]> = rd.regions[src].areas[ai].outlines[0].ring.iter().map(|&v| rd.v(v)).collect();
            let dsrc = (0..sring.len()).map(|i| point_segment_distance([rangoon[0] as f64, rangoon[1] as f64], sring[i], sring[(i + 1) % sring.len()])).fold(f64::MAX, f64::min);
            if dmin < 4.0 || dsrc < 4.0 {
                continue;
            }
            let center = [c[0] as f32, c[1] as f32];
            let mut rd = rd0.clone();
            let new = rd.split(&Split { settlement: SettlementSpec { name: "x".into(), center }, ..probe }).unwrap();
            tried += 1;
            let mut grid = grid0.clone();
            let targets = [target(&rd, "indochina", SlotKind::FourSlot), Target { region: new, settlement_slot_kind: SlotKind::OneSlot, options: TargetOptions::default() }];
            match synthesize(&mut grid, &rd, &targets, &Options::default()) {
                Ok(report) => {
                    let problems = validate(&grid, &rd);
                    eprintln!("cut {a}->{b}: {} problems, landings {:?}, groups {:?}, warnings {:?}", problems.len(), report.per_region.iter().map(|r| r.landings).collect::<Vec<_>>(), report.new_groups, report.warnings);
                    if problems.is_empty() {
                        ok = Some((rd, new, grid, a, b));
                        break 'search;
                    }
                }
                Err(e) => eprintln!("cut {a}->{b}: {e}"),
            }
            if tried >= 12 {
                break 'search;
            }
        }
    }
    let (rd, new, grid, a, b) = ok.expect("one cut of the Burma region synthesises cleanly");
    let src_id = src as i16;
    assert!(grid.paths.iter().any(|p| p.len() == 2 && p.contains(&src_id) && p.contains(&(new as i16))), "no group path between the halves");
    eprintln!("cut {a}->{b} of Burma synthesised: {} regions listed", grid.region_ids.len());
    let _ = rd;
}
