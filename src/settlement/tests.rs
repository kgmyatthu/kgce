//! Behavioural tests of the settlement move engine on small synthetic maps: placement rules, the all-or-nothing replay,
//! exact agreement between the requested and the stored position, the pathfinding node relocation and the consistent
//! update of the start position and the trade network. Everything is deterministic and needs no game files.

use std::collections::BTreeMap;

use super::{Kind, Move, nav, plan, preview, resolve, startpos, targets, trade, validate, validate_structure};
use crate::{
    esf::{self, Node},
    regionmap::{
        pathfinding::{Cell, Grid},
        pathsynth::{geometry as g, land::Frame, model::Model},
        regions::{Area, Outline, Region, RegionData},
    },
};

// ---------------------------------------------------------------------------------------------------- region fixtures

fn ring(rd: &mut RegionData, points: &[[f32; 2]]) -> Vec<u32> {
    let base = rd.vertices.len() as u32;
    rd.vertices.extend_from_slice(points);
    (0..points.len() as u32).map(|i| base + i).collect()
}

/// One area: the rectangle `[x0, y0, x1, y1]` with an optional rectangular hole.
fn area(rd: &mut RegionData, rect: [f32; 4], hole: Option<[f32; 4]>, passable: bool) -> Area {
    let [x0, y0, x1, y1] = rect;
    let outer = ring(rd, &[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    let mut outlines = vec![Outline { outer: true, ring: outer, runs: Vec::new() }];
    if let Some([hx0, hy0, hx1, hy1]) = hole {
        let inner = ring(rd, &[[hx0, hy0], [hx0, hy1], [hx1, hy1], [hx1, hy0]]);
        outlines.push(Outline { outer: false, ring: inner, runs: Vec::new() });
    }
    Area {
        land_bridge: false,
        channel_coast: false,
        passable,
        adjoining_id: 0,
        faces: Vec::new(),
        outlines,
        impassable_id: if passable { 65535 } else { 1 },
        island_id: 0,
    }
}

fn slot(key: &str, kind: &str, at: [f32; 2], height: f32) -> Vec<Node> {
    let foot = [[at[0] - 0.3, at[1] - 0.3], [at[0] + 0.3, at[1] - 0.3], [at[0], at[1] + 0.3]];
    vec![
        esf::s16(key),
        esf::s16(kind),
        esf::xyzv(at[0], height, at[1]),
        esf::xyv(at[0], at[1]),
        esf::u16v(57344),
        esf::b(true),
        esf::s16(""),
        esf::arr_xy(&foot),
        esf::arr_xy(&[]),
        esf::arr_xy(&foot),
    ]
}

fn links(name: &str) -> Node {
    Node::record(name, 0, vec![Node::array("links", 0, Vec::new())])
}

/// A converted-format `settlement_and_slots` record: a four-slot capital with road and fortification slots plus the
/// given `(kind, name, position)` town / port slots.
fn settlement(region: &str, name: &str, center: [f32; 2], others: &[(&str, &str, [f32; 2])]) -> Node {
    let mut slots = Vec::new();
    for (i, d) in [[-0.8f32, 0.0], [0.8, 0.0], [0.0, 0.8], [0.0, -0.8]].iter().enumerate() {
        slots.push(slot(
            &format!("settlement:{region}:{name}:settlement_4_slot:{i}"),
            "settlement_4_slot",
            [center[0] + d[0], center[1] + d[1]],
            0.27,
        ));
    }
    slots.push(slot(
        &format!("settlement:{region}:{name}:settlement_road:4"),
        "settlement_road",
        [center[0] - 1.0, center[1] - 1.0],
        0.09,
    ));
    slots.push(slot(
        &format!("settlement:{region}:{name}:settlement_fortification:5"),
        "settlement_fortification",
        [center[0] + 1.0, center[1] + 1.0],
        0.09,
    ));
    for (prefix, town, at) in others {
        let kind = if *prefix == "port" { "port" } else { "town-commercial" };
        slots.push(slot(&format!("{prefix}:{region}:{town}"), kind, *at, 0.1));
    }
    let footprint: Vec<[f32; 2]> = (0..8)
        .map(|k| {
            let a = k as f32 * std::f32::consts::FRAC_PI_4;
            [center[0] + 1.5 * a.cos(), center[1] + 1.5 * a.sin()]
        })
        .collect();
    Node::record(
        "settlement_and_slots",
        0,
        vec![
            esf::xyv(center[0], center[1]),
            esf::arr_xy(&footprint),
            Node::array("slot_descriptions", 1, slots),
            links("roads"),
            links("railways"),
            links("canals"),
        ],
    )
}

fn land(key: &str, areas: Vec<Area>, settlement: Node) -> Region {
    Region { key: key.into(), kind: "land".into(), areas, tail: 2, settlement: Some(settlement) }
}

/// alpha (40 x 30, a hole at x 28..34, y 10..20) borders beta (40 x 30); delta has three areas, the last impassable.
fn world() -> RegionData {
    let mut rd = RegionData { vertices: Vec::new(), regions: Vec::new(), theatres: Vec::new() };
    let alpha = area(&mut rd, [0.0, 0.0, 40.0, 30.0], Some([28.0, 10.0, 34.0, 20.0]), true);
    let node = settlement("alpha", "ajaccio", [20.0, 15.0], &[("town", "bastia", [8.0, 15.0]), ("port", "harbour", [38.5, 5.0])]);
    rd.regions.push(land("alpha", vec![alpha], node));
    let beta = area(&mut rd, [40.0, 0.0, 80.0, 30.0], None, true);
    rd.regions.push(land("beta", vec![beta], settlement("beta", "calvi", [60.0, 15.0], &[])));
    let areas = vec![
        area(&mut rd, [100.0, 0.0, 120.0, 30.0], None, true),
        area(&mut rd, [130.0, 0.0, 150.0, 30.0], None, true),
        area(&mut rd, [160.0, 0.0, 180.0, 30.0], None, false),
    ];
    rd.regions.push(land("delta", areas, settlement("delta", "sartene", [110.0, 15.0], &[])));
    rd
}

fn mv(region: &str, slot: &str, position: [f32; 2]) -> Move {
    Move { region: region.into(), slot: slot.into(), position }
}

fn failure(result: anyhow::Result<()>) -> String {
    format!("{:#}", result.expect_err("the move must be refused"))
}

/// (key, ground position, height, duplicated xy) of every slot description of region `ri`.
fn slot_table(rd: &RegionData, ri: usize) -> Vec<(String, [f32; 2], f32, [f32; 2])> {
    let settlement = rd.regions[ri].settlement.as_ref().unwrap();
    settlement.children().unwrap()[2]
        .items()
        .unwrap()
        .iter()
        .map(|slot| {
            let xyz = slot[2].as_xyz().unwrap();
            (slot[0].as_str().unwrap().to_string(), [xyz[0], xyz[2]], xyz[1], slot[3].as_xy().unwrap())
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------------- resolution

#[test]
fn targets_have_stable_ids_and_resolve_by_id_slot_key_or_name() {
    let rd = world();
    let list = targets(&rd, 0).unwrap();
    let ids: Vec<(&str, Kind)> = list.iter().map(|target| (target.slot.as_str(), target.kind)).collect();
    assert_eq!(
        ids,
        vec![
            ("settlement:alpha:ajaccio", Kind::Capital),
            ("town:alpha:bastia", Kind::Town),
            ("port:alpha:harbour", Kind::Port)
        ]
    );
    assert_eq!((list[0].general_slots, list[0].members.len(), list[0].position), (4, 6, [20.0, 15.0]));
    for spelling in ["settlement:alpha:ajaccio", "settlement:alpha:ajaccio:settlement_4_slot:2", "ajaccio"] {
        assert_eq!(resolve(&rd, "alpha", spelling).unwrap().slot, "settlement:alpha:ajaccio", "{spelling}");
    }
    assert_eq!(resolve(&rd, "alpha", "bastia").unwrap().kind, Kind::Town);
    assert!(resolve(&rd, "alpha", "nowhere").is_err());
    assert!(resolve(&rd, "beta", "bastia").is_err(), "a town of another region is not addressable");
    assert!(resolve(&rd, "nowhere", "ajaccio").is_err());
}

#[test]
fn a_bare_name_shared_by_capital_and_town_must_be_spelled_out() {
    let mut rd = RegionData { vertices: Vec::new(), regions: Vec::new(), theatres: Vec::new() };
    let a = area(&mut rd, [0.0, 0.0, 40.0, 30.0], None, true);
    rd.regions.push(land("eps", vec![a], settlement("eps", "x", [20.0, 15.0], &[("town", "x", [8.0, 15.0])])));
    assert!(format!("{:#}", resolve(&rd, "eps", "x").unwrap_err()).contains("ambiguous"));
    assert_eq!(resolve(&rd, "eps", "town:eps:x").unwrap().kind, Kind::Town);
    assert_eq!(resolve(&rd, "eps", "settlement:eps:x").unwrap().kind, Kind::Capital);
}

#[test]
fn structure_check_rejects_non_finite_and_repeated_targets() {
    assert!(validate_structure(&[mv("alpha", "ajaccio", [1.0, 2.0])]).is_ok());
    assert!(validate_structure(&[mv("alpha", "ajaccio", [f32::NAN, 2.0])]).is_err());
    assert!(validate_structure(&[mv("alpha", "ajaccio", [f32::INFINITY, 2.0])]).is_err());
    assert!(validate_structure(&[mv("alpha", "ajaccio", [1.0, 2.0]), mv("alpha", "ajaccio", [3.0, 2.0])]).is_err());
    assert!(validate_structure(&[mv("", "ajaccio", [1.0, 2.0])]).is_err());
}

// --------------------------------------------------------------------------------------------------------- placement

#[test]
fn placement_stays_inside_the_own_passable_area_with_node_clearance_and_spacing() {
    let rd = world();
    let capital = resolve(&rd, "alpha", "ajaccio").unwrap();
    assert!(super::check(&rd, &capital, [20.0, 22.0]).is_ok());
    assert!(super::check(&rd, &capital, [13.0, 15.0]).is_ok());
    let refused = |at: [f32; 2]| failure(super::check(&rd, &capital, at));
    assert!(refused([45.0, 15.0]).contains("leave its region"), "inside the neighbour");
    assert!(refused([31.0, 15.0]).contains("leave its region"), "inside the hole");
    assert!(refused([-5.0, 15.0]).contains("leave its region"), "outside everything");
    assert!(refused([1.5, 15.0]).contains("border or coast"), "too close to the outer ring");
    assert!(refused([26.5, 15.0]).contains("border or coast"), "too close to the hole ring");
    assert!(refused([10.5, 15.0]).contains("apart"), "closer than 3.2 to the town");
    assert!(refused([f32::NAN, 15.0]).contains("finite"));

    let town = resolve(&rd, "alpha", "bastia").unwrap();
    assert!(super::check(&rd, &town, [2.0, 15.0]).is_ok(), "a town node is smaller than a capital's");
    assert!(super::check(&rd, &town, [0.8, 15.0]).is_err());

    let port = resolve(&rd, "alpha", "port:alpha:harbour").unwrap();
    assert!(failure(super::check(&rd, &port, [30.0, 25.0])).contains("port"));
}

#[test]
fn a_settlement_never_changes_landmass_or_enters_impassable_ground() {
    let rd = world();
    let capital = resolve(&rd, "delta", "sartene").unwrap();
    assert!(failure(super::check(&rd, &capital, [140.0, 15.0])).contains("another landmass"));
    assert!(failure(super::check(&rd, &capital, [170.0, 15.0])).contains("another landmass"));
    assert!(super::check(&rd, &capital, [112.0, 20.0]).is_ok());
}

#[test]
fn validate_uses_the_slot_as_it_stands_now() {
    let mut rd = world();
    assert!(validate(&rd, &mv("alpha", "ajaccio", [20.0, 22.0])).is_ok());
    preview(&mut rd, &[mv("alpha", "ajaccio", [20.0, 22.0])]).unwrap();
    // the capital left (20, 15): the town may now stand there, but not next to the capital's new place
    assert!(validate(&rd, &mv("alpha", "bastia", [20.0, 15.0])).is_ok());
    assert!(validate(&rd, &mv("alpha", "bastia", [20.0, 24.0])).is_err());
}

// ------------------------------------------------------------------------------------------------------------ replay

#[test]
fn preview_moves_the_whole_capital_cluster_by_one_offset_and_keeps_heights() {
    let before = world();
    let mut rd = before.clone();
    preview(&mut rd, &[mv("alpha", "settlement:alpha:ajaccio", [20.0, 22.0])]).unwrap();

    let capital = resolve(&rd, "alpha", "ajaccio").unwrap();
    assert_eq!(capital.position, [20.0, 22.0], "the stored centre is the requested position, bit for bit");
    let (old, new) = (slot_table(&before, 0), slot_table(&rd, 0));
    assert_eq!(old.len(), new.len());
    for ((old_key, old_at, old_height, _), (new_key, new_at, new_height, new_xy)) in old.iter().zip(&new) {
        assert_eq!(old_key, new_key);
        assert_eq!(old_height, new_height, "slot heights are per slot type, not per place");
        assert_eq!(new_xy, new_at, "the duplicated map position follows the ground position");
        if old_key.starts_with("settlement:") {
            assert!((new_at[0] - old_at[0]).abs() < 1e-5 && (new_at[1] - old_at[1] - 7.0).abs() < 1e-5, "{old_key}: {new_at:?}");
        } else {
            assert_eq!(old_at, new_at, "{old_key} is not part of the capital");
        }
    }
    // the footprint polygon of the settlement moved with it
    let footprint = |rd: &RegionData| rd.regions[0].settlement.as_ref().unwrap().children().unwrap()[1].xy_array().unwrap();
    for (a, b) in footprint(&before).iter().zip(footprint(&rd)) {
        assert!((b[0] - a[0]).abs() < 1e-5 && (b[1] - a[1] - 7.0).abs() < 1e-5);
    }
    // other regions are untouched, and the replay is deterministic
    assert_eq!(before.regions[1].settlement, rd.regions[1].settlement);
    let mut again = before.clone();
    preview(&mut again, &[mv("alpha", "ajaccio", [20.0, 22.0])]).unwrap();
    assert_eq!(again.regions[0].settlement, rd.regions[0].settlement);
}

#[test]
fn a_town_move_changes_only_its_slot() {
    let before = world();
    let mut rd = before.clone();
    preview(&mut rd, &[mv("alpha", "town:alpha:bastia", [10.25, 20.5])]).unwrap();
    let (old, new) = (slot_table(&before, 0), slot_table(&rd, 0));
    let changed: Vec<&str> = old
        .iter()
        .zip(&new)
        .filter(|(a, b)| a.1 != b.1 || a.2 != b.2 || a.3 != b.3)
        .map(|(a, _)| a.0.as_str())
        .collect();
    assert_eq!(changed, vec!["town:alpha:bastia"]);
    assert_eq!(resolve(&rd, "alpha", "bastia").unwrap().position, [10.25, 20.5]);
    assert_eq!(resolve(&rd, "alpha", "ajaccio").unwrap().position, [20.0, 15.0]);
}

#[test]
fn a_refused_move_leaves_the_data_exactly_as_it_was() {
    let before = world();
    let mut rd = before.clone();
    let moves = [mv("alpha", "ajaccio", [20.0, 22.0]), mv("beta", "calvi", [-50.0, 15.0])];
    assert!(preview(&mut rd, &moves).is_err());
    assert_eq!(rd.regions[0].settlement, before.regions[0].settlement, "the first move must have been rolled back");
    assert_eq!(rd.regions[1].settlement, before.regions[1].settlement);
}

#[test]
fn two_spellings_of_one_target_are_one_target() {
    let mut rd = world();
    let moves = [mv("alpha", "town:alpha:bastia", [10.0, 20.0]), mv("alpha", "bastia", [12.0, 20.0])];
    assert!(failure(preview(&mut rd, &moves)).contains("more than once"));
}

#[test]
fn later_moves_see_the_earlier_ones_and_a_move_onto_the_current_place_is_skipped() {
    let mut rd = world();
    // against the original capital place (20, 15) the town spot (16, 15) is 4 u away, against the moved one 2 u
    let moves = [mv("alpha", "ajaccio", [14.0, 15.0]), mv("alpha", "bastia", [16.0, 15.0])];
    assert!(failure(preview(&mut rd, &moves)).contains("apart"));

    let mut rd = world();
    let before = rd.clone();
    let plans = plan(&mut rd, &[mv("alpha", "bastia", [8.0, 15.0])]).unwrap();
    assert!(plans.is_empty());
    assert_eq!(rd.regions[0].settlement, before.regions[0].settlement);
}

// ---------------------------------------------------------------------------------------------- pathfinding node

fn land_grid(cols: usize, rows: usize) -> Grid {
    Grid {
        area: 0,
        x0: 0.0,
        y0: 0.0,
        size: 2.0,
        cols,
        rows,
        origin_index: [0, 0],
        region_ids: vec![0],
        paths: vec![vec![0], Vec::new()],
        cells: (0..cols * rows).map(|_| Cell { header: [0; 8], pieces: Vec::new(), empty: Some((0, 0)) }).collect(),
        vertices: vec![[-2147483647, -2147483647]; 4],
        polylines: BTreeMap::from([(0usize, vec![0u32, 2, 3, 1])]),
    }
}

/// A grid of one all-land region with a town node carved at `at`.
fn grid_with_town_node(at: [f32; 2]) -> Grid {
    let mut grid = land_grid(12, 12);
    let mut model = Model::from_grid(&grid).unwrap();
    let frame = Frame { origin: [0.0, 0.0], cs: model.cs };
    let mut touched = BTreeMap::new();
    nav::carve(&mut model, &mut touched, 0, nav::NodeShape::Town, at, &frame).unwrap();
    nav::normalize_runs(&mut model, &mut touched);
    model.write_into(&mut grid).unwrap();
    grid
}

/// (union area of the type-7 pieces, summed area of every piece and run cell, node contains point) of a grid.
fn node_report(grid: &Grid, probe: [f32; 2]) -> (f64, f64, bool) {
    let model = Model::from_grid(grid).unwrap();
    let mut node: Vec<g::Shapes> = Vec::new();
    let mut total = 0.0;
    for (index, cell) in model.cells.iter().enumerate() {
        let (c, r) = (index % model.cols, index / model.cols);
        if cell.pieces.is_empty() {
            total += g::area(&vec![g::rect_shape([c as i32 * model.cs, r as i32 * model.cs], [(c as i32 + 1) * model.cs, (r as i32 + 1) * model.cs])]);
        }
        for piece in &cell.pieces {
            let shapes: g::Shapes = vec![vec![piece.pts.clone()]];
            total += g::area(&shapes);
            if piece.ptype() == 7 {
                node.push(shapes);
            }
        }
    }
    let refs: Vec<&g::Shapes> = node.iter().collect();
    let union = g::union_many(&refs);
    let at = [(f64::from(probe[0]) * g::FIX).round(), (f64::from(probe[1]) * g::FIX).round()];
    (g::area(&union), total, g::contains_point(&union, at))
}

#[test]
fn a_node_moves_without_leaving_a_gap_or_an_overlap() {
    let town = [11.0f32, 11.0];
    let target = [17.3f32, 8.1];
    let mut grid = grid_with_town_node(town);
    let (area_before, total_before, at_old) = node_report(&grid, town);
    assert!(at_old && (area_before - 3.1100).abs() < 0.01, "the town node is an octagon of 3.11 u²: {area_before}");
    assert!((total_before - 12.0 * 12.0 * 4.0).abs() < 1e-6);

    let outcome = nav::relocate(&mut grid, 0, nav::NodeShape::Town, town, target).unwrap();
    assert!(matches!(outcome, nav::Outcome::Moved { old_pieces, .. } if old_pieces >= 1));
    let (area_old_place, total, _) = node_report(&grid, town);
    let (_, _, at_new) = node_report(&grid, target);
    let (_, _, still_at_old) = node_report(&grid, town);
    assert!(at_new, "the node now holds the new point");
    assert!(!still_at_old, "the old place is land again");
    assert!((area_old_place - 3.1100).abs() < 0.01, "one node, same size: {area_old_place}");
    assert!((total - total_before).abs() < 1e-6, "pieces tile the grid exactly: {total}");
}

#[test]
fn a_node_that_does_not_fit_is_refused_and_the_grid_stays() {
    let town = [11.0f32, 11.0];
    let mut grid = grid_with_town_node(town);
    let before = Model::from_grid(&grid).unwrap();
    let error = format!("{:#}", nav::relocate(&mut grid, 0, nav::NodeShape::Town, town, [0.5, 11.0]).unwrap_err());
    assert!(error.contains("leaves the pathfinding grid") || error.contains("does not fit"), "{error}");
    let after = Model::from_grid(&grid).unwrap();
    let pieces = |m: &Model| m.cells.iter().map(|cell| cell.pieces.iter().map(|p| (p.a, p.pts.clone())).collect::<Vec<_>>()).collect::<Vec<_>>();
    assert_eq!(pieces(&before), pieces(&after));
}

#[test]
fn a_settlement_without_a_stock_node_gets_none() {
    let mut grid = land_grid(12, 12);
    let outcome = nav::relocate(&mut grid, 0, nav::NodeShape::City, [11.0, 11.0], [15.0, 15.0]).unwrap();
    assert_eq!(outcome, nav::Outcome::Absent);
    let (area, _, _) = node_report(&grid, [15.0, 15.0]);
    assert_eq!(area, 0.0);
}

// ------------------------------------------------------------------------------- start position and trade network

fn fixed(v: f32) -> i32 {
    (f64::from(v) * 1_048_576.0).round() as i32
}

fn los() -> Node {
    Node::record(
        "LINE_OF_SIGHT",
        0,
        vec![esf::b(true), esf::xyv(0.0, 0.0), esf::xyv(8.0, 8.0), Node::record("QUAD_TREE_BIT_ARRAY", 0, vec![esf::arr_u32(&[1, 2, 3])])],
    )
}

fn residence(id: u32, at: (i32, i32)) -> Node {
    Node::record(
        "SIEGEABLE_GARRISON_RESIDENCE",
        0,
        vec![
            Node::record("GARRISON_RESIDENCE", 0, Vec::new()),
            esf::u32v(id),
            esf::u32v(0),
            esf::u32v(0),
            esf::u32v(0),
            esf::u32v(0),
            esf::i32v(0),
            esf::b(false),
            esf::b(false),
            esf::b(false),
            esf::i32v(at.0),
            esf::i32v(at.1),
            esf::u32v(0),
            esf::arr_u32(&[]),
            esf::arr_u32(&[]),
            los(),
            esf::b(false),
        ],
    )
}

fn region_slot(key: &str, id: u32, at: (i32, i32)) -> Node {
    Node::record(
        "REGION_SLOT",
        0,
        vec![
            residence(id, (0, 0)),
            Node::record("BUILDING_MANAGER", 0, Vec::new()),
            esf::u32v(id),
            esf::s16(key),
            esf::i32v(0),
            esf::i32v(0),
            esf::i32v(at.0),
            esf::i32v(at.1),
            esf::i32v(0),
            esf::i32v(1),
        ],
    )
}

fn region_record(rd: &RegionData, index: usize, world_id: u32, settlement_id: u32) -> Node {
    let region = &rd.regions[index];
    let settlement = region.settlement.as_ref().unwrap();
    let center = settlement.children().unwrap()[0].as_xy().unwrap();
    let (mut array, mut road, mut fortification) = (Vec::new(), None, None);
    for (n, description) in settlement.children().unwrap()[2].items().unwrap().iter().enumerate() {
        let key = description[0].as_str().unwrap();
        let at = description[3].as_xy().unwrap();
        let record = region_slot(key, world_id * 10 + n as u32, (fixed(at[0]), fixed(at[1])));
        if key.contains("settlement_road") {
            road = Some(record);
        } else if key.contains("settlement_fortification") {
            fortification = Some(record);
        } else {
            array.push(vec![record]);
        }
    }
    let manager = Node::record(
        "REGION_SLOT_MANAGER",
        0,
        vec![
            Node::array("REGION_SLOT_ARRAY", 0, array),
            Node::record("ROAD_SLOT", 0, vec![esf::b(true), road.unwrap()]),
            Node::record("FORTIFICATION_SLOT", 0, vec![esf::b(true), fortification.unwrap()]),
        ],
    );
    Node::record(
        "REGION",
        0,
        vec![
            esf::s16(region.key.clone()),
            Node::record("POPULATION", 0, Vec::new()),
            Node::record("TRAITS", 0, Vec::new()),
            manager,
            esf::i32v(world_id as i32),
            Node::record(
                "SETTLEMENT",
                0,
                vec![
                    residence(settlement_id, (fixed(center[0]), fixed(center[1]))),
                    Node::record("CAMPAIGN_LOCALISATION", 0, Vec::new()),
                    esf::i32v(1),
                    esf::s16(format!("settlement:{}", region.key)),
                    esf::i32v(settlement_id as i32),
                    esf::s16(format!("settlement:{}", region.key)),
                ],
            ),
        ],
    )
}

fn situated(at: (i32, i32), region_cai: u32) -> Node {
    Node::record("CAI_SITUATED", 0, vec![esf::i32v(at.0), esf::i32v(at.1), esf::u32v(region_cai), esf::arr_u32(&[1]), esf::u32v(13)])
}

fn cai_region(key: &str, id: u32, world_id: u32, at: [f32; 2]) -> Vec<Node> {
    vec![
        esf::u32v(id),
        Node::record(
            "CAI_REGION",
            0,
            vec![
                esf::arr_u32(&[]),
                esf::arr_u32(&[]),
                esf::u32v(0),
                esf::arr_u32(&[]),
                esf::f32v(at[0]),
                esf::f32v(at[1]),
                esf::arr_u32(&[]),
                esf::arr_u32(&[]),
                esf::b(false),
                esf::arr_u32(&[]),
                esf::s16(key),
                esf::u32v(world_id),
                esf::u32v(0),
                esf::arr_u32(&[]),
            ],
        ),
    ]
}

fn character(at: (i32, i32)) -> Vec<Node> {
    vec![Node::record(
        "CHARACTER",
        0,
        vec![
            Node::record("LOCOMOTABLE", 0, vec![esf::i32v(at.0), esf::i32v(at.1), esf::i32v(at.0), esf::i32v(at.1), esf::f32v(0.5)]),
            Node::record("CHARACTER_DETAILS", 0, Vec::new()),
            los(),
        ],
    )]
}

fn expand(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    (0..points.len() - 1).step_by(3).flat_map(|s| points[s..s + 4].to_vec()).collect()
}

const SPLINE: [[f32; 2]; 4] = [[38.0, 5.0], [30.0, 8.0], [24.0, 12.0], [20.4, 15.3]];

fn segment(points: &[[f32; 2]; 4]) -> Vec<Node> {
    let length = (trade::bezier_len(points) * 1.001) as f32;
    let mut item = vec![Node::array("COMMERCE_RAIDS", 0, Vec::new()), esf::u32v(1)];
    item.extend(points.iter().map(|p| esf::xyv(p[0], p[1])));
    item.extend([
        esf::arr_f32(&[length]),
        esf::arr_f32(&[0.0]),
        esf::f32v(length),
        esf::u32v(1000),
        esf::arr_u32(&[]),
        esf::arr_u32(&[]),
        esf::u32v(3),
        esf::f32v(length),
        los(),
        esf::i32v(0),
        esf::b(false),
    ]);
    item
}

fn trade_file() -> Node {
    Node::record(
        "TRADE_ROUTES_FILE",
        0,
        vec![
            Node::array("SPLINES", 0, vec![vec![esf::arr_xy(&SPLINE), esf::b(true)]]),
            Node::array("PORTS", 0, vec![vec![esf::s16("port:alpha:harbour")]]),
            Node::array("SETTLEMENTS", 0, vec![vec![esf::s16("alpha")], vec![esf::s16("beta")]]),
            Node::array(
                "ROUTES",
                0,
                vec![
                    vec![esf::u32v(0), esf::u32v(1), esf::arr_u32(&[0]), esf::f32v(25.0)],
                    vec![esf::u32v(1), esf::u32v(2), esf::arr_u32(&[]), esf::f32v(40.5)],
                ],
            ),
        ],
    )
}

/// A minimal start position of `rd`'s alpha and beta: regions with residences and slots, the CAI records that mirror
/// them, characters (one garrisoned in alpha's capital, one beside it, one far away) and the trade manager.
fn start_position(rd: &RegionData) -> Node {
    let alpha = rd.regions[0].settlement.as_ref().unwrap().children().unwrap()[0].as_xy().unwrap();
    let beta = rd.regions[1].settlement.as_ref().unwrap().children().unwrap()[0].as_xy().unwrap();
    let (alpha_at, beta_at) = ((fixed(alpha[0]), fixed(alpha[1])), (fixed(beta[0]), fixed(beta[1])));
    let bastia = (fixed(8.0), fixed(15.0));
    let harbour = (fixed(38.5), fixed(5.0));
    let world = Node::record(
        "WORLD",
        0,
        vec![Node::record(
            "REGION_MANAGER",
            0,
            vec![Node::array("REGIONS_ARRAY", 0, vec![vec![region_record(rd, 0, 1000, 1001)], vec![region_record(rd, 1, 2000, 2001)]])],
        )],
    );
    let cai = Node::record(
        "CAI_INTERFACE",
        0,
        vec![Node::record(
            "CAI_WORLD",
            0,
            vec![
                Node::array("CAI_WORLD_REGIONS", 0, vec![cai_region("alpha", 300, 1000, alpha), cai_region("beta", 301, 2000, beta)]),
                Node::array(
                    "CAI_WORLD_SETTLEMENTS",
                    0,
                    vec![
                        vec![situated(alpha_at, 300), esf::u32v(310), Node::record("CAI_SETTLEMENT", 0, vec![esf::arr_u32(&[]), esf::u32v(0), esf::u32v(1001)])],
                        vec![situated(beta_at, 301), esf::u32v(311), Node::record("CAI_SETTLEMENT", 0, vec![esf::arr_u32(&[]), esf::u32v(0), esf::u32v(2001)])],
                    ],
                ),
                Node::array(
                    "CAI_WORLD_REGION_HLCIS",
                    0,
                    vec![
                        vec![esf::u32v(320), Node::record("CAI_REGION_HLCI", 0, vec![esf::u32v(300), esf::u32v(5), esf::arr_u32(&[0]), esf::i32v(alpha_at.0), esf::i32v(alpha_at.1)])],
                        vec![esf::u32v(321), Node::record("CAI_REGION_HLCI", 0, vec![esf::u32v(300), esf::u32v(5), esf::arr_u32(&[1]), esf::i32v(fixed(3.0)), esf::i32v(fixed(3.0))])],
                        vec![esf::u32v(322), Node::record("CAI_REGION_HLCI", 0, vec![esf::u32v(301), esf::u32v(5), esf::arr_u32(&[0]), esf::i32v(beta_at.0), esf::i32v(beta_at.1)])],
                    ],
                ),
                Node::array(
                    "CAI_WORLD_REGION_SLOTS",
                    0,
                    vec![
                        vec![situated(bastia, 300), esf::u32v(330), Node::record("CAI_REGION_SLOT", 0, vec![esf::i32v(1), esf::u32v(1442)])],
                        vec![situated(harbour, 300), esf::u32v(331), Node::record("CAI_REGION_SLOT", 0, vec![esf::i32v(1), esf::u32v(1443)])],
                    ],
                ),
                Node::array(
                    "CAI_WORLD_RESOURCE_MOBILES",
                    0,
                    vec![
                        vec![situated(alpha_at, 300), esf::u32v(340), Node::record("CAI_MOBILE", 0, Vec::new())],
                        vec![situated((fixed(1.0), fixed(1.0)), 300), esf::u32v(341), Node::record("CAI_MOBILE", 0, Vec::new())],
                    ],
                ),
            ],
        )],
    );
    let characters = Node::array(
        "CHARACTER_ARRAY",
        0,
        vec![character(alpha_at), character((alpha_at.0 + fixed(0.5), alpha_at.1)), character((fixed(1.0), fixed(1.0)))],
    );
    let segments = Node::array("TRADE_SEGMENTS", 0, vec![segment(&SPLINE)]);
    let routes = Node::array(
        "TRADE_ROUTES",
        0,
        vec![
            vec![esf::u32v(0), esf::u32v(1), esf::arr_u32(&[0]), esf::f32v(26.0)],
            vec![esf::u32v(1), esf::u32v(2), esf::arr_u32(&[]), esf::f32v(41.0)],
            vec![esf::u32v(0), esf::u32v(2), esf::arr_u32(&[]), esf::f32v(50.0)],
        ],
    );
    let manager = Node::record(
        "CAMPAIGN_TRADE_MANAGER",
        0,
        vec![
            Node::array("PORT_INDICES", 0, vec![vec![esf::s16("port:alpha:harbour"), esf::u32v(0)]]),
            Node::array("SETTLEMENT_INDICES", 0, vec![vec![esf::s16("alpha"), esf::u32v(1)], vec![esf::s16("beta"), esf::u32v(2)]]),
            segments,
            routes,
        ],
    );
    Node::record(
        "ROOT",
        0,
        vec![Node::record("CAMPAIGN_ENV", 0, vec![Node::record("CAMPAIGN_MODEL", 0, vec![world, cai, characters, manager])])],
    )
}

fn cai_items<'a>(root: &'a Node, name: &str) -> &'a Vec<Vec<Node>> {
    root.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAI_INTERFACE", "CAI_WORLD"]).unwrap().get(name).unwrap().items().unwrap()
}

fn position_of(item: &[Node]) -> (i32, i32) {
    let kids = item.iter().find(|n| n.name() == Some("CAI_SITUATED")).unwrap().children().unwrap();
    (kids[0].as_i32().unwrap(), kids[1].as_i32().unwrap())
}

fn locomotable(item: &[Node]) -> Vec<i32> {
    let character = item[0].children().unwrap();
    character[0].children().unwrap()[..4].iter().map(|n| n.as_i32().unwrap()).collect()
}

fn line_of_sight_active(item: &[Node]) -> bool {
    let character = item[0].children().unwrap();
    character[2].children().unwrap().len() > 1
}

#[test]
fn moving_a_capital_updates_trade_roads_and_the_start_position_consistently() {
    let before = world();
    let mut start = start_position(&before);
    let mut roads = trade_file();
    let mut rd = before.clone();
    let mut positions = trade::Positions::of(&rd).unwrap();
    let plans = plan(&mut rd, &[mv("alpha", "ajaccio", [20.0, 22.0])]).unwrap();
    positions.moved(&plans);

    // ---- trade_routes.esf: the road end and its handle carry the offset, lengths follow
    let change = trade::apply(&mut roads, &plans, &positions).unwrap();
    let shifted = |v: f32| (f64::from(v) + 7.0) as f32;
    let new_spline = [SPLINE[0], SPLINE[1], [SPLINE[2][0], shifted(SPLINE[2][1])], [SPLINE[3][0], shifted(SPLINE[3][1])]];
    let stored = roads.get("SPLINES").unwrap().items().unwrap()[0][0].xy_array().unwrap();
    assert_eq!(stored, new_spline.to_vec());
    assert_eq!(change.splines.len(), 1);
    let grown = trade::chain_len(&new_spline) - trade::chain_len(&SPLINE);
    let lengths: Vec<f32> = roads.get("ROUTES").unwrap().items().unwrap().iter().map(|r| r[3].as_f32().unwrap()).collect();
    assert!((f64::from(lengths[0]) - (25.0 + grown)).abs() < 1e-4, "{lengths:?}");
    let direct = ((20.0f64 - 60.0).powi(2) + (22.0f64 - 15.0).powi(2)).sqrt() - 40.0;
    assert!((f64::from(lengths[1]) - (40.5 + direct)).abs() < 1e-4, "{lengths:?}");

    // ---- startpos.esf
    let notes = startpos::apply(&mut start, &plans, &change, &positions).unwrap();
    assert!(notes.iter().any(|note| note.starts_with("startpos.esf")));
    startpos::verify(&start, &plans).unwrap();
    let new_at = (fixed(20.0), fixed(22.0));
    let old_at = (fixed(20.0), fixed(15.0));

    let regions = cai_items(&start, "CAI_WORLD_REGIONS");
    let kids = regions[0][1].children().unwrap();
    assert_eq!((kids[4].as_f32().unwrap(), kids[5].as_f32().unwrap()), (20.0, 22.0));
    let beta = regions[1][1].children().unwrap();
    assert_eq!((beta[4].as_f32().unwrap(), beta[5].as_f32().unwrap()), (60.0, 15.0), "beta did not move");
    let settlements = cai_items(&start, "CAI_WORLD_SETTLEMENTS");
    assert_eq!(position_of(&settlements[0]), new_at);
    assert_eq!(position_of(&settlements[1]), (fixed(60.0), fixed(15.0)));
    let hlci: Vec<(i32, i32)> = cai_items(&start, "CAI_WORLD_REGION_HLCIS")
        .iter()
        .map(|item| {
            let kids = item[1].children().unwrap();
            (kids[3].as_i32().unwrap(), kids[4].as_i32().unwrap())
        })
        .collect();
    assert_eq!(hlci[0], new_at, "the settlement HLCI moved");
    assert_eq!(hlci[1], (fixed(3.0), fixed(3.0)), "another HLCI of the region stays");
    let slots = cai_items(&start, "CAI_WORLD_REGION_SLOTS");
    assert_eq!(position_of(&slots[0]), (fixed(8.0), fixed(15.0)), "the capital move leaves the town alone");
    let mobiles = cai_items(&start, "CAI_WORLD_RESOURCE_MOBILES");
    assert_eq!(position_of(&mobiles[0]), new_at);
    assert_eq!(position_of(&mobiles[1]), (fixed(1.0), fixed(1.0)));

    // the garrison goes with the city, a neighbour and a distant army stay, and the moved ones lose their cached sight
    let characters = start.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL"]).unwrap().get("CHARACTER_ARRAY").unwrap().items().unwrap();
    assert_eq!(locomotable(&characters[0]), vec![new_at.0, new_at.1, new_at.0, new_at.1]);
    assert!(!line_of_sight_active(&characters[0]));
    assert_eq!(locomotable(&characters[1]), vec![old_at.0 + fixed(0.5), old_at.1, old_at.0 + fixed(0.5), old_at.1]);
    assert!(line_of_sight_active(&characters[1]));
    assert_eq!(locomotable(&characters[2])[..2], [fixed(1.0), fixed(1.0)]);
    assert!(line_of_sight_active(&characters[2]));

    // trade segment mirrors the spline, lengths are consistent, routes follow
    let manager = start.path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "CAMPAIGN_TRADE_MANAGER"]).unwrap();
    let segment = &manager.get("TRADE_SEGMENTS").unwrap().items().unwrap()[0];
    let points: Vec<[f32; 2]> = segment[2..6].iter().map(|n| n.as_xy().unwrap()).collect();
    assert_eq!(points, expand(&new_spline));
    let (a1, total) = (segment[6].f32_array().unwrap(), segment[8].as_f32().unwrap());
    assert!((a1.iter().sum::<f32>() - total).abs() < 1e-4 && segment[13].as_f32().unwrap() == total);
    let old_total = (trade::bezier_len(&SPLINE) * 1.001) as f32;
    let expected = old_total * (trade::bezier_len(&new_spline) / trade::bezier_len(&SPLINE)) as f32;
    assert!((total - expected).abs() < 1e-3, "{total} vs {expected}");
    assert_eq!(segment[14].children().unwrap().len(), 1, "the segment's cached sight was dropped");
    let routes: Vec<f32> = manager.get("TRADE_ROUTES").unwrap().items().unwrap().iter().map(|r| r[3].as_f32().unwrap()).collect();
    assert!((f64::from(routes[0]) - (26.0 + f64::from(total - old_total))).abs() < 1e-3, "{routes:?}");
    assert!((f64::from(routes[1]) - (41.0 + direct)).abs() < 1e-3, "{routes:?}");
    assert_eq!(routes[2], 50.0, "a route between other nodes is untouched");
}

#[test]
fn a_start_position_that_disagrees_with_the_map_is_refused() {
    let mut other = world();
    preview(&mut other, &[mv("alpha", "ajaccio", [19.0, 15.0])]).unwrap();
    let mut start = start_position(&other);
    let mut rd = world();
    let mut positions = trade::Positions::of(&rd).unwrap();
    let plans = plan(&mut rd, &[mv("alpha", "ajaccio", [20.0, 22.0])]).unwrap();
    positions.moved(&plans);
    let mut roads = trade_file();
    let change = trade::apply(&mut roads, &plans, &positions).unwrap();
    let error = format!("{:#}", startpos::apply(&mut start, &plans, &change, &positions).unwrap_err());
    assert!(error.contains("start position keeps"), "{error}");
}

#[test]
fn a_road_end_far_from_its_settlement_is_not_carried_away() {
    let mut rd = world();
    let mut positions = trade::Positions::of(&rd).unwrap();
    let plans = plan(&mut rd, &[mv("alpha", "ajaccio", [20.0, 22.0])]).unwrap();
    positions.moved(&plans);
    let mut roads = trade_file();
    // the road of alpha ends 10 u away from its centre: the node cannot be attributed to the settlement
    roads.get_mut("SPLINES").unwrap().items_mut().unwrap()[0][0] =
        esf::arr_xy(&[[38.0, 5.0], [30.0, 8.0], [24.0, 12.0], [30.0, 15.0]]);
    let error = format!("{:#}", trade::apply(&mut roads, &plans, &positions).unwrap_err());
    assert!(error.contains("expected a trade node near the settlement"), "{error}");
}
