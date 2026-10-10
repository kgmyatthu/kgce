//! `trade_routes.esf`: root children `SPLINES[]` (items `[xy[] control points, bool land]`), `PORTS[]`,
//! `SETTLEMENTS[]`, `ROUTES[]` (items `[u32 from, u32 to, u32[] spline ids, f32 length]`) and `TRADE_NODE_ROUTES[]`
//! (item 0 = `[xy[] sea trade-node positions]`, item i > 0 = `[s16 node name, ROUTES[] of (u32 port, u32[] spline ids,
//! f32 length)]`).
//!
//! Every control point is mapped through its owning chart. A spline must lie wholly inside the crops of ONE chart; one
//! that spans charts or leaves every crop is never mapped and stops the run with an inventory. Every route length is
//! `old * scale` (cross-chart hops are scaled, not recomputed from atlas geometry).

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use serde_json::{Map, Value as Json, json};

use super::ctx::Ctx;
use super::diff::Touched;
use super::util::{Bounds, Counter, Range, le_u32s, packed};
use crate::esf::{self, Node, Value};

fn ids_of(n: &Node) -> Result<Vec<u32>> {
    le_u32s(packed(n, 0x48)?)
}

fn layout_is(item: &[Node], want: &[u8]) -> bool {
    // codes: 0x08 u32, 0x0A f32, 0x01 bool, 0x48 u32 array, 0x4C xy array
    item.len() == want.len()
        && item.iter().zip(want).all(|(n, &w)| match n {
            Node::Value(Value::U32(_)) => w == 0x08,
            Node::Value(Value::F32(_)) => w == 0x0A,
            Node::Value(Value::Bool(_)) => w == 0x01,
            Node::Value(Value::Array { code, .. }) => *code == w,
            _ => false,
        })
}

fn layout_desc(item: &[Node]) -> Vec<String> {
    item.iter().map(super::util::describe).collect()
}

/// Maps an xy f32 array through the owning charts of its points.
fn scale_xy_array(
    ctx: &mut Ctx,
    node: &Node,
    bounds_old: &mut Bounds,
    bounds_new: &mut Bounds,
    label: &str,
    spline: Option<usize>,
    tag: &str,
) -> Result<(Node, usize, BTreeSet<usize>)> {
    let data = match node {
        Node::Value(Value::Array { code: 0x4C, data }) if data.len() % 8 == 0 => data,
        other => bail!("expected xy f32 array, found {}", super::util::describe(other)),
    };
    let points: Vec<[f32; 2]> = data
        .chunks_exact(8)
        .map(|c| {
            [
                f32::from_le_bytes(c[0..4].try_into().unwrap()),
                f32::from_le_bytes(c[4..8].try_into().unwrap()),
            ]
        })
        .collect();
    let world: Vec<[f64; 2]> = points.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let (charts, present) = ctx.point_charts(&world, label, spline)?;
    let mut out = Vec::with_capacity(points.len());
    for (p, &chart) in points.iter().zip(&charts) {
        let q = ctx.xy_f32(*p, chart)?;
        bounds_old.add(p[0] as f64, p[1] as f64);
        bounds_new.add(q[0] as f64, q[1] as f64);
        ctx.note_f(tag, q);
        out.push(q);
    }
    Ok((esf::arr_xy(&out), out.len(), present))
}

pub(super) fn do_trade_routes(ctx: &mut Ctx, root: &mut Node, touched: &mut Touched) -> Result<Json> {
    let Node::Record { children, .. } = root else {
        bail!("trade_routes.esf root is not a record");
    };
    let names: Vec<Option<&str>> = children.iter().map(Node::name).collect();
    if names != [Some("SPLINES"), Some("PORTS"), Some("SETTLEMENTS"), Some("ROUTES"), Some("TRADE_NODE_ROUTES")] {
        bail!("unexpected trade_routes children {names:?}");
    }
    let [spl, ports, setts, routes, tnr] = children.as_mut_slice() else {
        bail!("unexpected trade_routes children");
    };
    let (spl, routes, tnr) = (spl.items_mut()?, routes.items_mut()?, tnr.items_mut()?);
    let (ports, setts) = (ports.items()?, setts.items()?);
    for (list, set) in [(ports, &mut ctx.refs.trade_ports), (setts, &mut ctx.refs.trade_settlements)] {
        for item in list {
            match item.first() {
                Some(n) => {
                    set.insert(n.as_str()?.to_string());
                }
                None => bail!("empty PORTS/SETTLEMENTS item"),
            }
        }
    }
    // ---- SPLINES[]
    let (mut sb_old, mut sb_new) = (Bounds::new(), Bounds::new());
    let mut land = [0i64; 2];
    let mut control_points = 0usize;
    for (i, item) in spl.iter_mut().enumerate() {
        let is_land = match item.as_slice() {
            [_, Node::Value(Value::Bool(b))] => *b,
            _ => bail!("SPLINES[{i}] layout {:?}", layout_desc(item)),
        };
        land[is_land as usize] += 1;
        let (node, n, present) =
            scale_xy_array(ctx, &item[0], &mut sb_old, &mut sb_new, &format!("SPLINES[{i}]"), Some(i), "trade.spline")?;
        item[0] = node;
        ctx.spline_charts.insert(i, present);
        if n < 4 || (n - 1) % 3 != 0 {
            bail!("SPLINES[{i}] has {n} control points");
        }
        control_points += n;
        touched.add(&[0, i, 0]);
    }
    // route layouts are validated before anything is raised so the inventory can count users
    let mut route_ids: Vec<Vec<u32>> = Vec::with_capacity(routes.len());
    for (i, item) in routes.iter().enumerate() {
        if !layout_is(item, &[0x08, 0x08, 0x48, 0x0A]) {
            bail!("ROUTES[{i}] layout {:?}", layout_desc(item));
        }
        route_ids.push(ids_of(&item[2])?);
    }
    let first_ok = tnr.first().is_some_and(|f| layout_is(f, &[0x4C]));
    if !first_ok {
        bail!("TRADE_NODE_ROUTES[0] layout {:?}", tnr.first().map(|f| layout_desc(f)));
    }
    let mut node_route_ids: Vec<Vec<u32>> = Vec::new();
    for (i, item) in tnr.iter().enumerate().skip(1) {
        let ok = matches!(item.as_slice(), [Node::Value(Value::String16(_) | Value::String8(_)), Node::Array { name, version: 0, .. }] if name == "ROUTES");
        if !ok {
            bail!("TRADE_NODE_ROUTES[{i}] layout {:?}", layout_desc(item));
        }
        for (j, r) in item[1].items()?.iter().enumerate() {
            if !layout_is(r, &[0x08, 0x48, 0x0A]) {
                bail!("TRADE_NODE_ROUTES[{i}].ROUTES[{j}] layout {:?}", layout_desc(r));
            }
            node_route_ids.push(ids_of(&r[1])?);
        }
    }
    ctx.raise_if_blocked(&route_ids, &node_route_ids)?;
    // ---- ROUTES[]
    let (mut rl_old, mut rl_new, mut ratio) = (Range::new(), Range::new(), Range::new());
    let mut direct = Counter::new();
    for (i, item) in routes.iter_mut().enumerate() {
        let length = item[3].as_f32()?;
        let out = ctx.distance(length, None)?;
        item[3] = esf::f32v(out);
        touched.add(&[3, i, 3]);
        direct.add(if route_ids[i].is_empty() { "direct_empty_spline_list" } else { "spline_route" }, 1);
        rl_old.add(length as f64);
        rl_new.add(out as f64);
        if length > 0.0 {
            ratio.add(out as f64 / length as f64);
        }
    }
    // ---- TRADE_NODE_ROUTES[]
    let (mut nb_old, mut nb_new) = (Bounds::new(), Bounds::new());
    let (node, n_nodes, _) = scale_xy_array(ctx, &tnr[0][0], &mut nb_old, &mut nb_new, "TRADE_NODE_ROUTES[0]", None, "trade.node")?;
    tnr[0][0] = node;
    touched.add(&[4, 0, 0]);
    let (mut tr_old, mut tr_new) = (Range::new(), Range::new());
    let mut tn_routes = 0usize;
    let mut node_names = Vec::new();
    let n_splines = spl.len();
    for i in 1..tnr.len() {
        node_names.push(tnr[i][0].as_str()?.to_string());
        let list = tnr[i][1].items_mut()?;
        for (j, r) in list.iter_mut().enumerate() {
            let ids = ids_of(&r[1])?;
            if ids.iter().max().is_some_and(|m| *m as usize / 2 >= n_splines) {
                bail!("TRADE_NODE_ROUTES[{i}].ROUTES[{j}] references missing spline");
            }
            let port = r[0].as_u32()? as usize;
            if port >= ports.len() {
                bail!("TRADE_NODE_ROUTES[{i}].ROUTES[{j}] destination {port} is not a port");
            }
            let length = r[2].as_f32()?;
            let out = ctx.distance(length, None)?;
            r[2] = esf::f32v(out);
            touched.add(&[4, i, 1, j, 2]);
            tr_old.add(length as f64);
            tr_new.add(out as f64);
            tn_routes += 1;
        }
    }
    let mut detail = Map::new();
    detail.insert(
        "splines".into(),
        json!({
            "count": spl.len(), "control_points": control_points, "land": land[1], "sea": land[0],
            "bounds_old": sb_old.dump(), "bounds_new": sb_new.dump(),
        }),
    );
    detail.insert("ports".into(), json!(ports.len()));
    detail.insert("settlements".into(), json!(setts.len()));
    detail.insert(
        "routes".into(),
        json!({
            "count": routes.len(), "kinds": direct.to_json(), "length_old": rl_old.dump(), "length_new": rl_new.dump(),
            "new_over_old": ratio.dump(),
        }),
    );
    detail.insert(
        "trade_node_routes".into(),
        json!({
            "trade_nodes": node_names.len(), "node_names": node_names, "node_positions": n_nodes,
            "node_position_bounds_old": nb_old.dump(), "node_position_bounds_new": nb_new.dump(),
            "routes": tn_routes, "length_old": tr_old.dump(), "length_new": tr_new.dump(),
            "note": "item 0 = xy array of the sea trade-node positions; item i>0 = [node name, ROUTES[] of [destination port index, spline-id u32[], length f32]]",
        }),
    );
    // ---- atlas bookkeeping
    let chain_charts = |ids: &[u32]| -> Result<BTreeSet<usize>> {
        let mut ks = BTreeSet::new();
        for j in ids {
            match ctx.spline_charts.get(&(*j as usize / 2)) {
                Some(s) => ks.extend(s.iter().copied()),
                None => bail!("route references missing spline {}", *j as usize / 2),
            }
        }
        Ok(ks)
    };
    let mut per_chart: BTreeMap<String, i64> = BTreeMap::new();
    for charts in ctx.spline_charts.values() {
        for c in charts {
            *per_chart.entry(ctx.chart_key(*c).to_string()).or_default() += 1;
        }
    }
    let mut route_kinds: BTreeMap<usize, i64> = BTreeMap::new();
    let mut node_route_kinds: BTreeMap<usize, i64> = BTreeMap::new();
    for ids in &route_ids {
        *route_kinds.entry(chain_charts(ids)?.len()).or_default() += 1;
    }
    for ids in &node_route_ids {
        *node_route_kinds.entry(chain_charts(ids)?.len()).or_default() += 1;
    }
    let keyed = |m: &BTreeMap<usize, i64>| -> Json { Json::Object(m.iter().map(|(k, v)| (k.to_string(), json!(v))).collect()) };
    detail.insert(
        "atlas".into(),
        json!({
            "splines_per_chart": per_chart,
            "routes_by_charts_visited": keyed(&route_kinds),
            "node_routes_by_charts_visited": keyed(&node_route_kinds),
            "length_policy": "every ROUTES[]/TRADE_NODE_ROUTES[] length = old * scale; the stored length is the control polygon length of the chained splines plus the straight gaps between consecutive spline ends (cross-chart hops), so hops are scaled, not recomputed from atlas geometry",
        }),
    );
    Ok(Json::Object(detail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::Plan;

    fn spline(points: &[[f32; 2]]) -> Vec<Node> {
        vec![esf::arr_xy(points), esf::b(true)]
    }

    fn tree(splines: Vec<Vec<Node>>, route_ids: &[u32], length: f32, node: [f32; 2]) -> Node {
        Node::record(
            "ROOT",
            0,
            vec![
                Node::array("SPLINES", 0, splines),
                Node::array("PORTS", 0, vec![vec![esf::s16("port:france:brest")]]),
                Node::array("SETTLEMENTS", 0, vec![vec![esf::s16("france")]]),
                Node::array("ROUTES", 0, vec![vec![esf::u32v(0), esf::u32v(1), esf::arr_u32(route_ids), esf::f32v(length)]]),
                Node::array(
                    "TRADE_NODE_ROUTES",
                    0,
                    vec![
                        vec![esf::arr_xy(&[node])],
                        vec![
                            esf::s16("trade_node_0"),
                            Node::array("ROUTES", 0, vec![vec![esf::u32v(0), esf::arr_u32(route_ids), esf::f32v(length / 2.0)]]),
                        ],
                    ],
                ),
            ],
        )
    }

    fn europe_spline() -> Vec<Node> {
        spline(&[[0.0, 300.0], [1.0, 301.0], [2.0, 302.0], [3.0, 303.0]])
    }

    #[test]
    fn single_chart_spline_and_all_lengths_scale() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let mut root = tree(vec![europe_spline()], &[0], 100.0, [5.0, 310.0]);
        let mut touched = Touched::default();
        let detail = do_trade_routes(&mut ctx, &mut root, &mut touched).unwrap();
        let europe = plan.chart("2").unwrap();
        let p = plan.xy([1.0, 301.0], europe);
        let Node::Record { children, .. } = &root else { panic!() };
        let pts = children[0].items().unwrap()[0][0].xy_array().unwrap();
        assert_eq!(pts[1], [p[0] as f32, p[1] as f32]);
        let routes = children[3].items().unwrap();
        assert_eq!(routes[0][3], esf::f32v((100.0f64 * plan.scale) as f32));
        // spline points, route length, node position, node-route length
        assert_eq!(touched.paths.len(), 4);
        assert_eq!(detail["atlas"]["splines_per_chart"]["eur_ind"], json!(1));
        assert_eq!(detail["atlas"]["routes_by_charts_visited"]["1"], json!(1));
        assert!(ctx.refs.trade_ports.contains("port:france:brest") && ctx.refs.trade_settlements.contains("france"));
    }

    #[test]
    fn spline_spanning_two_charts_stops_the_run_with_an_inventory() {
        let plan = Plan::proven();
        let mut ctx = Ctx::new(&plan).unwrap();
        let cross = spline(&[[0.0, 300.0], [-500.0, 200.0], [-500.0, 210.0], [-500.0, 220.0]]);
        let mut root = tree(vec![europe_spline(), cross], &[2], 100.0, [5.0, 310.0]);
        let err = do_trade_routes(&mut ctx, &mut root, &mut Touched::default()).unwrap_err().to_string();
        assert!(err.contains("SPLINES[1]") && err.contains("cross_chart") && err.contains("\"routes_using\":1"), "{err}");
    }

    #[test]
    fn bad_layouts_and_references_are_errors() {
        let plan = Plan::proven();
        // 5 control points: (n - 1) % 3 != 0
        let mut ctx = Ctx::new(&plan).unwrap();
        let five = spline(&[[0.0, 300.0], [1.0, 301.0], [2.0, 302.0], [3.0, 303.0], [4.0, 304.0]]);
        assert!(do_trade_routes(&mut ctx, &mut tree(vec![five], &[0], 1.0, [5.0, 310.0]), &mut Touched::default()).is_err());
        // route referencing a missing spline
        let mut ctx = Ctx::new(&plan).unwrap();
        assert!(do_trade_routes(&mut ctx, &mut tree(vec![europe_spline()], &[4], 1.0, [5.0, 310.0]), &mut Touched::default()).is_err());
        // negative route length
        let mut ctx = Ctx::new(&plan).unwrap();
        assert!(do_trade_routes(&mut ctx, &mut tree(vec![europe_spline()], &[0], -1.0, [5.0, 310.0]), &mut Touched::default()).is_err());
        // trade node outside every chart
        let mut ctx = Ctx::new(&plan).unwrap();
        assert!(do_trade_routes(&mut ctx, &mut tree(vec![europe_spline()], &[0], 1.0, [1000.0, 600.0]), &mut Touched::default()).is_err());
    }
}
