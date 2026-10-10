//! Read access to the `pathfinding_areas` of `pathfinding.esf`: layout validation, grid header, piece polygons.
//!
//! Area item = `[vertices[] (i32 x, i32 y), u32[] polygon index stream, grid_data]`; `grid_data` children:
//! `[i32 x0, i32 y0, u16 startX, u16 startY, i32 cell, u32 cols, u32 rows, u32, u16, u16, i16[], u16[], grid_cells[]]`;
//! a cell item is `[u8[8] header, boundaries[] (u32 a, u32 b), ...]`, run descriptors carry 5 children with a 12-byte
//! line list at index 4; a piece's polygon is `b & 0x1FFFFF` into the index stream (`count, ids...`; ids 0..=3 are the
//! cell corners BL, TL, BR, TR, larger ids index the vertex table).

use anyhow::{Result, anyhow, bail};

use super::util::{index_of, le_u32s, node_i32, node_u16, node_u32, packed};
use crate::esf::{Node, Value};

pub(super) struct GridHead {
    pub ox: i32,
    pub oy: i32,
    pub sx: u16,
    pub sy: u16,
    pub cs: i32,
    pub cols: u32,
    pub rows: u32,
}

/// Root child index and items of the single `pathfinding_areas` array.
pub(super) fn areas(root: &Node) -> Result<(usize, &Vec<Vec<Node>>)> {
    let at = index_of(root.children()?, "pathfinding.esf root", "pathfinding_areas")?;
    Ok((at, root.children()?[at].items()?))
}

fn value_is(n: &Node, want: u8) -> bool {
    matches!(
        (n, want),
        (Node::Value(Value::I32(_)), 0x04)
            | (Node::Value(Value::U16(_)), 0x07)
            | (Node::Value(Value::U32(_)), 0x08)
    ) || matches!(n, Node::Value(Value::Array { code, .. }) if *code == want)
}

/// The POC's `GRID_CODES` layout check of one area item plus the header of its grid.
pub(super) fn checked_head(k: usize, item: &[Node]) -> Result<GridHead> {
    let vertices_ok = matches!(item.first(), Some(Node::Array { name, .. }) if name == "vertices");
    let stream_ok = matches!(item.get(1), Some(Node::Value(Value::Array { code: 0x48, .. })));
    let grid_ok = matches!(item.get(2), Some(Node::Record { name, .. }) if name == "grid_data");
    if item.len() != 3 || !vertices_ok || !stream_ok || !grid_ok {
        bail!(
            "area {k}: unexpected layout {:?}",
            item.iter().map(|n| n.name().map(str::to_string)).collect::<Vec<_>>()
        );
    }
    let g = item[2].children()?;
    const CODES: [u8; 12] = [0x04, 0x04, 0x07, 0x07, 0x04, 0x08, 0x08, 0x08, 0x07, 0x07, 0x43, 0x47];
    let cells_ok = matches!(g.get(12), Some(Node::Array { name, version: 0, .. }) if name == "grid_cells");
    if g.len() != 13 || !cells_ok || !g.iter().zip(CODES).all(|(n, c)| value_is(n, c)) {
        bail!("area {k}: unexpected grid_data layout");
    }
    Ok(GridHead {
        ox: node_i32(&g[0])?,
        oy: node_i32(&g[1])?,
        sx: node_u16(&g[2])?,
        sy: node_u16(&g[3])?,
        cs: node_i32(&g[4])?,
        cols: node_u32(&g[5])?,
        rows: node_u32(&g[6])?,
    })
}

/// Vertex table of an area as `(x, y)` pairs (dummies included).
pub(super) fn vertex_pairs(item: &[Node]) -> Result<Vec<[i32; 2]>> {
    item[0]
        .items()?
        .iter()
        .enumerate()
        .map(|(vi, v)| {
            if v.len() != 2 {
                bail!("vertex {vi}: {} entries", v.len());
            }
            Ok([node_i32(&v[0])?, node_i32(&v[1])?])
        })
        .collect()
}

pub(super) fn cells(item: &[Node]) -> Result<&Vec<Vec<Node>>> {
    item[2].children()?[12].items()
}

/// Calls `f(col, row, ids)` for every boundary piece of every cell in file order and returns the number of cells
/// walked (run descriptors stand for several).
pub(super) fn for_each_piece(
    k: usize,
    item: &[Node],
    cols: i64,
    mut f: impl FnMut(i64, i64, &[u32]) -> Result<()>,
) -> Result<i64> {
    if cols <= 0 {
        bail!("area {k}: grid has {cols} columns");
    }
    let stream = le_u32s(packed(&item[1], 0x48)?)?;
    let mut idx = 0i64;
    for cell in cells(item)? {
        let (col, row) = (idx % cols, idx / cols);
        let extra = if cell.len() == 5 { cell[4].raw_array()?.len() / 12 } else { 0 };
        idx += 1 + extra as i64;
        let Some(Node::Array { name, items, .. }) = cell.get(1) else {
            bail!("area {k}: unexpected cell layout");
        };
        if name != "boundaries" {
            bail!("area {k}: unexpected cell layout");
        }
        for b in items {
            let off = (node_u32(b.get(1).ok_or_else(|| anyhow!("area {k}: piece without second word"))?)? & 0x1F_FFFF) as usize;
            let n = *stream.get(off).ok_or_else(|| anyhow!("area {k}: polygon offset {off} outside the index stream"))? as usize;
            let ids = stream
                .get(off + 1..off + 1 + n)
                .ok_or_else(|| anyhow!("area {k}: polygon at {off} runs past the index stream"))?;
            f(col, row, ids)?;
        }
    }
    Ok(idx)
}

/// Cell corner `id` (0..=3: BL, TL, BR, TR) of cell `(col, row)`.
pub(super) fn corner(h: &GridHead, col: i64, row: i64, cell: i64, id: u32) -> [i64; 2] {
    let (cx, cy) = (h.ox as i64 + col * cell, h.oy as i64 + row * cell);
    match id {
        0 => [cx, cy],
        1 => [cx, cy + cell],
        2 => [cx + cell, cy],
        _ => [cx + cell, cy + cell],
    }
}

/// Every pathfinding piece polygon of every area (corner ids resolved against their cell, vertex ids against the
/// vertex table) in Q20 source coordinates, tagged with its area index.
pub(super) fn path_polygons(root: &Node) -> Result<Vec<(usize, Vec<[i64; 2]>)>> {
    let mut out = Vec::new();
    for (k, item) in areas(root)?.1.iter().enumerate() {
        let head = checked_head(k, item)?;
        let pairs = vertex_pairs(item)?;
        let cs = head.cs as i64;
        for_each_piece(k, item, head.cols as i64, |col, row, ids| {
            let mut pts = Vec::with_capacity(ids.len());
            for &i in ids {
                pts.push(if i < 4 {
                    corner(&head, col, row, cs, i)
                } else {
                    let p = pairs.get(i as usize).ok_or_else(|| anyhow!("area {k}: vertex id {i} outside the vertex table"))?;
                    [p[0] as i64, p[1] as i64]
                });
            }
            out.push((k, pts));
            Ok(())
        })?;
    }
    Ok(out)
}

#[cfg(test)]
pub(super) mod test_support {
    use super::*;
    use crate::esf;

    /// One area: a 3x1 grid with origin (0, 0) and cell 2^20; cell 0 holds one triangle piece using the corners and
    /// vertex 4, cell 1 is a run descriptor covering itself plus one following cell (no pieces).
    pub fn small_root() -> Node {
        let cell = 1 << 20;
        let vertices = Node::array(
            "vertices",
            0,
            vec![
                vec![esf::i32v(-2147483647), esf::i32v(-2147483647)],
                vec![esf::i32v(-2147483647), esf::i32v(-2147483647)],
                vec![esf::i32v(-2147483647), esf::i32v(-2147483647)],
                vec![esf::i32v(-2147483647), esf::i32v(-2147483647)],
                vec![esf::i32v(cell / 2), esf::i32v(cell / 4)],
            ],
        );
        let stream = esf::arr_u32(&[3, 0, 4, 2]);
        let piece = vec![esf::u32v(0), esf::u32v(0)];
        let header = Node::Value(Value::Array { code: 0x46, data: vec![0; 8] });
        let cell0 = vec![header.clone(), Node::array("boundaries", 0, vec![piece])];
        let cell1 = vec![
            header,
            Node::array("boundaries", 0, vec![]),
            esf::u32v(0),
            esf::u16v(0),
            Node::Value(Value::Array { code: 0x46, data: vec![0; 12] }),
        ];
        let grid = Node::record(
            "grid_data",
            0,
            vec![
                esf::i32v(0),
                esf::i32v(0),
                esf::u16v(7),
                esf::u16v(9),
                esf::i32v(cell),
                esf::u32v(3),
                esf::u32v(1),
                esf::u32v(0),
                esf::u16v(0),
                esf::u16v(0),
                Node::Value(Value::Array { code: 0x43, data: vec![] }),
                Node::Value(Value::Array { code: 0x47, data: vec![] }),
                Node::array("grid_cells", 0, vec![cell0, cell1]),
            ],
        );
        Node::record("ROOT", 0, vec![Node::array("pathfinding_areas", 0, vec![vec![vertices, stream, grid]])])
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::small_root;
    use super::*;

    #[test]
    fn layout_head_and_polygons_of_a_small_area() {
        let root = small_root();
        let (at, items) = areas(&root).unwrap();
        assert_eq!(at, 0);
        let head = checked_head(0, &items[0]).unwrap();
        assert_eq!((head.ox, head.oy, head.cs, head.cols, head.rows, head.sx, head.sy), (0, 0, 1 << 20, 3, 1, 7, 9));
        let polys = path_polygons(&root).unwrap();
        assert_eq!(polys.len(), 1);
        // ids 0 (BL corner), 4 (vertex), 2 (BR corner)
        assert_eq!(polys[0].1, vec![[0, 0], [1 << 19, 1 << 18], [1 << 20, 0]]);
    }

    #[test]
    fn cells_walked_counts_run_descriptors() {
        let root = small_root();
        let items = areas(&root).unwrap().1;
        let walked = for_each_piece(0, &items[0], 3, |_, _, _| Ok(())).unwrap();
        assert_eq!(walked, 3);
    }

    #[test]
    fn unexpected_layouts_are_rejected() {
        let mut root = small_root();
        if let Node::Record { children, .. } = &mut root {
            if let Node::Array { items, .. } = &mut children[0] {
                items[0].pop();
            }
        }
        let (_, items) = areas(&root).unwrap();
        assert!(checked_head(0, &items[0]).is_err());
    }
}
