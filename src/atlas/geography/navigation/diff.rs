//! Parallel tree diff: proves the transform changed ONLY the values it deliberately touched and nothing else
//! (shape, names, versions, strings, packed-array sizes and every untouched value are identical).

use std::collections::HashSet;

use anyhow::{Result, bail};

use crate::esf::{Node, Value};

/// Tree path: record child index, or array item index followed by the element index inside the item.
pub(super) type Path = Vec<usize>;

/// Paths the transform deliberately modified (Python `Touched`).
#[derive(Default)]
pub(super) struct Touched {
    pub paths: HashSet<Path>,
}

impl Touched {
    pub fn add(&mut self, path: &[usize]) {
        self.paths.insert(path.to_vec());
    }
}

fn value_changed(path: &[usize], a: &Value, b: &Value) -> Result<bool> {
    use Value::*;
    Ok(match (a, b) {
        (String16(x), String16(y)) | (String8(x), String8(y)) => {
            if x != y {
                bail!("{path:?}: string changed");
            }
            false
        }
        (Array { code: c1, data: d1 }, Array { code: c2, data: d2 }) => {
            if c1 != c2 || d1.len() != d2.len() {
                bail!("{path:?}: array code/length changed");
            }
            d1 != d2
        }
        (F32(x), F32(y)) => x.to_bits() != y.to_bits(),
        (F64(x), F64(y)) => x.to_bits() != y.to_bits(),
        (XY(x), XY(y)) => x.iter().zip(y).any(|(p, q)| p.to_bits() != q.to_bits()),
        (XYZ(x), XYZ(y)) => x.iter().zip(y).any(|(p, q)| p.to_bits() != q.to_bits()),
        (x, y) if std::mem::discriminant(x) == std::mem::discriminant(y) => x != y,
        (x, y) => bail!("{path:?}: scalar code {:#x} -> {:#x}", x.code(), y.code()),
    })
}

/// Parallel walk; errors on ANY shape difference, records the paths of changed scalar/array leaves.
pub(super) fn tree_diff(a: &Node, b: &Node, path: &mut Path, changed: &mut Vec<Path>) -> Result<()> {
    match (a, b) {
        (Node::Value(x), Node::Value(y)) => {
            if value_changed(path, x, y)? {
                changed.push(path.clone());
            }
        }
        (
            Node::Record { name: na, version: va, children: ca },
            Node::Record { name: nb, version: vb, children: cb },
        ) => {
            if na != nb || va != vb || ca.len() != cb.len() {
                bail!("{path:?}: record {na} v{va} n={} -> {nb} v{vb} n={}", ca.len(), cb.len());
            }
            for (i, (ea, eb)) in ca.iter().zip(cb).enumerate() {
                path.push(i);
                tree_diff(ea, eb, path, changed)?;
                path.pop();
            }
        }
        (
            Node::Array { name: na, version: va, items: ia },
            Node::Array { name: nb, version: vb, items: ib },
        ) => {
            if na != nb || va != vb || ia.len() != ib.len() {
                bail!("{path:?}: record array {na} v{va} n={} -> {nb} v{vb} n={}", ia.len(), ib.len());
            }
            for (i, (xa, xb)) in ia.iter().zip(ib).enumerate() {
                if xa.len() != xb.len() {
                    bail!("{path:?}+{i}: item width {} -> {}", xa.len(), xb.len());
                }
                for (j, (ea, eb)) in xa.iter().zip(xb).enumerate() {
                    path.push(i);
                    path.push(j);
                    tree_diff(ea, eb, path, changed)?;
                    path.pop();
                    path.pop();
                }
            }
        }
        _ => bail!("{path:?}: node kind changed"),
    }
    Ok(())
}

/// Diff of two whole trees.
pub(super) fn diff(old: &Node, new: &Node) -> Result<Vec<Path>> {
    let mut changed = Vec::new();
    tree_diff(old, new, &mut Vec::new(), &mut changed)?;
    Ok(changed)
}

/// Fail when a changed leaf was not declared in `touched` (a stray modification).
pub(super) fn ensure_only_touched(file: &str, changed: &[Path], touched: &Touched) -> Result<()> {
    let stray: Vec<&Path> = changed.iter().filter(|p| !touched.paths.contains(*p)).collect();
    if let Some(first) = stray.first() {
        bail!("{file}: {} values changed outside the transform, first {first:?}", stray.len());
    }
    Ok(())
}

pub(super) fn count_nodes(node: &Node) -> usize {
    match node {
        Node::Value(_) => 1,
        Node::Record { children, .. } => 1 + children.iter().map(count_nodes).sum::<usize>(),
        Node::Array { items, .. } => 1 + items.iter().flatten().map(count_nodes).sum::<usize>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf;

    fn sample() -> Node {
        Node::record(
            "ROOT",
            0,
            vec![
                esf::u32v(2),
                esf::s16("name"),
                Node::array("ITEMS", 0, vec![vec![esf::i32v(1), esf::f32v(1.5)], vec![esf::i32v(2), esf::f32v(2.5)]]),
                esf::arr_u32(&[1, 2, 3]),
            ],
        )
    }

    #[test]
    fn identical_trees_have_no_changes() {
        assert!(diff(&sample(), &sample()).unwrap().is_empty());
        assert_eq!(count_nodes(&sample()), 1 + 1 + 1 + 1 + 4 + 1);
    }

    #[test]
    fn declared_change_is_accepted_and_stray_change_rejected() {
        let old = sample();
        let mut new = sample();
        if let Node::Record { children, .. } = &mut new {
            if let Node::Array { items, .. } = &mut children[2] {
                items[1][1] = esf::f32v(9.0);
            }
        }
        let changed = diff(&old, &new).unwrap();
        assert_eq!(changed, vec![vec![2, 1, 1]]);
        let mut touched = Touched::default();
        touched.add(&[2, 1, 1]);
        ensure_only_touched("x", &changed, &touched).unwrap();
        // a second, undeclared change in the packed array must be reported as stray
        if let Node::Record { children, .. } = &mut new {
            children[3] = esf::arr_u32(&[1, 2, 4]);
        }
        let changed = diff(&old, &new).unwrap();
        assert_eq!(changed.len(), 2);
        let err = ensure_only_touched("x", &changed, &touched).unwrap_err().to_string();
        assert!(err.contains("outside the transform"), "{err}");
    }

    #[test]
    fn shape_string_and_width_changes_are_errors() {
        let old = sample();
        let mut renamed = sample();
        if let Node::Record { children, .. } = &mut renamed {
            children[1] = esf::s16("other");
        }
        assert!(diff(&old, &renamed).is_err());
        let mut longer = sample();
        if let Node::Record { children, .. } = &mut longer {
            children[3] = esf::arr_u32(&[1, 2, 3, 4]);
        }
        assert!(diff(&old, &longer).is_err());
        let mut wider = sample();
        if let Node::Record { children, .. } = &mut wider {
            if let Node::Array { items, .. } = &mut children[2] {
                items[0].push(esf::b(true));
            }
        }
        assert!(diff(&old, &wider).is_err());
        let mut retyped = sample();
        if let Node::Record { children, .. } = &mut retyped {
            children[0] = esf::i32v(2);
        }
        assert!(diff(&old, &retyped).is_err());
    }

    #[test]
    fn float_comparison_is_bitwise() {
        let mut a = sample();
        let mut b = sample();
        if let (Node::Record { children: ca, .. }, Node::Record { children: cb, .. }) = (&mut a, &mut b) {
            if let (Node::Array { items: ia, .. }, Node::Array { items: ib, .. }) = (&mut ca[2], &mut cb[2]) {
                ia[0][1] = esf::f32v(0.0);
                ib[0][1] = esf::f32v(-0.0);
            }
        }
        assert_eq!(diff(&a, &b).unwrap().len(), 1);
    }
}
