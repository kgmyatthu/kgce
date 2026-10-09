//! Typed ESF tree helpers used by the CAI builder (mirror of the Python `common` / `entities` helpers the CAI port needs).
//!
//! Python's "scalar tuple" children map to `Node::Value`; a "CAI world item" is one entry of a `Node::Array` (`Vec<Node>`).

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, bail};

use crate::esf::{Node, Value};

pub const PROPSET: &str = "CAI_BDI_COMPONENT_PROPERTY_SET";
pub const BLOCK_OWNS: &str = "CAI_BDI_COMPONENT_BLOCK_OWNS";
/// Number of scalars/arrays of the shared BDI component header following the PROPSET record of every CAI item.
pub const HEADER_LEN: usize = 23;

// ---- constructors -------------------------------------------------------------------------------------------------
pub fn u32n(v: u32) -> Node {
    Node::Value(Value::U32(v))
}
pub fn i32n(v: i32) -> Node {
    Node::Value(Value::I32(v))
}
pub fn f32n(v: f32) -> Node {
    Node::Value(Value::F32(v))
}
pub fn booln(v: bool) -> Node {
    Node::Value(Value::Bool(v))
}

pub fn arrn(v: &[u32]) -> Node {
    let mut data = Vec::with_capacity(v.len() * 4);
    for x in v {
        data.extend_from_slice(&x.to_le_bytes());
    }
    Node::Value(Value::Array { code: 0x48, data })
}

pub fn empty_arr() -> Node {
    arrn(&[])
}

pub fn empty_array_like(name: &str, version: u8) -> Node {
    Node::Array {
        name: name.to_string(),
        version,
        items: Vec::new(),
    }
}

// ---- scalar access ------------------------------------------------------------------------------------------------
/// integer scalar as u32 (Python `c[1]` of a u32 / i32 / small-int tuple)
pub fn scalar(n: &Node) -> Option<u32> {
    match n {
        Node::Value(v) => match v {
            Value::U32(x) => Some(*x),
            Value::I32(x) => Some(*x as u32),
            Value::U16(x) => Some(*x as u32),
            Value::I16(x) => Some(*x as u32),
            Value::U8(x) => Some(*x as u32),
            Value::I8(x) => Some(*x as u32),
            _ => None,
        },
        _ => None,
    }
}

pub fn is_value(n: &Node) -> bool {
    matches!(n, Node::Value(_))
}
pub fn is_u32(n: &Node) -> bool {
    matches!(n, Node::Value(Value::U32(_)))
}
pub fn is_u32_arr(n: &Node) -> bool {
    matches!(n, Node::Value(Value::Array { code: 0x48, .. }))
}
pub fn is_string16(n: &Node) -> bool {
    matches!(n, Node::Value(Value::String16(_)))
}

pub fn string_of(n: &Node) -> Option<&str> {
    match n {
        Node::Value(Value::String16(s)) | Node::Value(Value::String8(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// elements of a packed array value (any element-size-4 array); empty for non-arrays
pub fn arr_vals(n: &Node) -> Vec<u32> {
    match n {
        Node::Value(Value::Array { data, .. }) => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => Vec::new(),
    }
}

pub fn sc_at(ch: &[Node], i: usize) -> Result<u32> {
    let n = ch
        .get(i)
        .ok_or_else(|| anyhow!("child index {i} out of range ({} children)", ch.len()))?;
    scalar(n).ok_or_else(|| anyhow!("child {i} is not an integer scalar: {:?}", n))
}

pub fn arr_at(ch: &[Node], i: usize) -> Result<Vec<u32>> {
    let n = ch
        .get(i)
        .ok_or_else(|| anyhow!("child index {i} out of range ({} children)", ch.len()))?;
    if !is_u32_arr(n) {
        bail!("child {i} is not a u32 array: {:?}", n);
    }
    Ok(arr_vals(n))
}

pub fn put(ch: &mut [Node], i: usize, n: Node) -> Result<()> {
    let len = ch.len();
    match ch.get_mut(i) {
        Some(slot) => {
            *slot = n;
            Ok(())
        }
        None => bail!("child index {i} out of range ({len} children)"),
    }
}

// ---- structure access ---------------------------------------------------------------------------------------------
pub fn name_of(n: &Node) -> Option<&str> {
    match n {
        Node::Record { name, .. } | Node::Array { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

pub fn is_record(n: &Node) -> bool {
    matches!(n, Node::Record { .. })
}
pub fn is_record_named(n: &Node, name: &str) -> bool {
    matches!(n, Node::Record { name: nm, .. } if nm == name)
}
pub fn is_array_named(n: &Node, name: &str) -> bool {
    matches!(n, Node::Array { name: nm, .. } if nm == name)
}

pub fn kids(n: &Node) -> Result<&Vec<Node>> {
    match n {
        Node::Record { children, .. } => Ok(children),
        _ => bail!("expected a record, found {:?}", name_of(n)),
    }
}

pub fn kids_mut(n: &mut Node) -> Result<&mut Vec<Node>> {
    match n {
        Node::Record { children, .. } => Ok(children),
        other => bail!("expected a record, found {:?}", name_of(other)),
    }
}

pub fn arr_items(n: &Node) -> Result<&Vec<Vec<Node>>> {
    match n {
        Node::Array { items, .. } => Ok(items),
        _ => bail!("expected a record array, found {:?}", name_of(n)),
    }
}

pub fn arr_items_mut(n: &mut Node) -> Result<&mut Vec<Vec<Node>>> {
    match n {
        Node::Array { items, .. } => Ok(items),
        other => bail!("expected a record array, found {:?}", name_of(other)),
    }
}

pub fn find_idx(ch: &[Node], name: &str) -> Option<usize> {
    ch.iter().position(|c| name_of(c) == Some(name))
}

/// Python `get(node, *path)`: navigate record children by Record / RecordArray name
pub fn get<'a>(node: &'a Node, path: &[&str]) -> Result<&'a Node> {
    let mut cur = node;
    for p in path {
        let ch = kids(cur).map_err(|e| anyhow!("{e} while looking for {p}"))?;
        cur = ch
            .iter()
            .find(|c| name_of(c) == Some(*p))
            .ok_or_else(|| anyhow!("missing node {p} in {:?}", name_of(cur)))?;
    }
    Ok(cur)
}

pub fn get_mut<'a>(node: &'a mut Node, path: &[&str]) -> Result<&'a mut Node> {
    let mut cur = node;
    for p in path {
        let owner = name_of(cur).map(str::to_owned);
        let ch = kids_mut(cur)?;
        let idx = ch
            .iter()
            .position(|c| name_of(c) == Some(*p))
            .ok_or_else(|| anyhow!("missing node {p} in {:?}", owner))?;
        cur = &mut ch[idx];
    }
    Ok(cur)
}

pub fn get_items<'a>(node: &'a Node, path: &[&str]) -> Result<&'a Vec<Vec<Node>>> {
    arr_items(get(node, path)?)
}

// ---- item (array element) helpers ---------------------------------------------------------------------------------
/// index of the first scalar of a CAI world item = BDI header id
pub fn hdr_index(item: &[Node]) -> Result<usize> {
    item.iter()
        .position(is_value)
        .ok_or_else(|| anyhow!("CAI item without header scalar"))
}

pub fn item_id(item: &[Node]) -> Result<u32> {
    let i = hdr_index(item)?;
    scalar(&item[i]).ok_or_else(|| anyhow!("CAI item header is not an id: {:?}", item[i]))
}

/// last Record of an item (CAI specialisation record)
pub fn spec_rec(item: &[Node]) -> Result<&Node> {
    item.iter()
        .rev()
        .find(|c| is_record(c))
        .ok_or_else(|| anyhow!("CAI item without specialisation record"))
}

pub fn spec_rec_mut(item: &mut [Node]) -> Result<&mut Node> {
    item.iter_mut()
        .rev()
        .find(|c| is_record(c))
        .ok_or_else(|| anyhow!("CAI item without specialisation record"))
}

pub fn spec_kids(item: &[Node]) -> Result<&Vec<Node>> {
    kids(spec_rec(item)?)
}
pub fn spec_kids_mut(item: &mut [Node]) -> Result<&mut Vec<Node>> {
    kids_mut(spec_rec_mut(item)?)
}
pub fn spec_name(item: &[Node]) -> Result<&str> {
    name_of(spec_rec(item)?).ok_or_else(|| anyhow!("unnamed specialisation record"))
}

pub fn rec_by_name<'a>(item: &'a [Node], name: &str) -> Result<&'a Node> {
    item.iter()
        .find(|c| is_record_named(c, name))
        .ok_or_else(|| anyhow!("CAI item has no record {name}"))
}

pub fn rec_by_name_mut<'a>(item: &'a mut [Node], name: &str) -> Result<&'a mut Node> {
    item.iter_mut()
        .find(|c| is_record_named(c, name))
        .ok_or_else(|| anyhow!("CAI item has no record {name}"))
}

pub fn set_owner(item: &mut [Node], which: &str, owner_cai: u32) -> Result<()> {
    let r = kids_mut(rec_by_name_mut(item, which)?)?;
    put(r, 0, u32n(owner_cai))
}

pub fn set_situated(
    item: &mut [Node],
    x: i32,
    y: i32,
    region_cai: u32,
    theatre_cai: u32,
    kind: u32,
) -> Result<()> {
    let r = kids_mut(rec_by_name_mut(item, "CAI_SITUATED")?)?;
    put(r, 0, i32n(x))?;
    put(r, 1, i32n(y))?;
    put(r, 2, u32n(region_cai))?;
    put(r, 3, arrn(&[theatre_cai]))?;
    put(r, 4, u32n(kind))
}

/// The RecordArray-valued item element at `idx` (e.g. BLOCK_OWNS at 15 of a pool item)
pub fn item_array_mut(item: &mut [Node], idx: usize) -> Result<&mut Vec<Vec<Node>>> {
    let len = item.len();
    match item.get_mut(idx) {
        Some(Node::Array { items, .. }) => Ok(items),
        _ => bail!("item element {idx} is not a record array ({len} elements)"),
    }
}

pub fn item_array(item: &[Node], idx: usize) -> Result<&Vec<Vec<Node>>> {
    match item.get(idx) {
        Some(Node::Array { items, .. }) => Ok(items),
        _ => bail!(
            "item element {idx} is not a record array ({} elements)",
            item.len()
        ),
    }
}

/// Reset the shared BDI component header (23 scalars + BLOCK_OWNS array) of a CAI world item (Python `pristine_header`).
pub fn pristine_header(item: &mut Vec<Node>, new_id: Option<u32>) -> Result<()> {
    let i0 = hdr_index(item)?;
    let hid = match new_id {
        Some(v) => v,
        None => scalar(&item[i0]).ok_or_else(|| anyhow!("CAI item header is not an id"))?,
    };
    if i0 + HEADER_LEN > item.len() {
        bail!(
            "CAI item too short for a BDI header ({} elements, header at {i0})",
            item.len()
        );
    }
    if !is_array_named(&item[i0 + 13], BLOCK_OWNS) {
        bail!(
            "CAI item header element 13 is not BLOCK_OWNS: {:?}",
            name_of(&item[i0 + 13])
        );
    }
    let hdr = vec![
        u32n(hid),
        f32n(100.0),
        f32n(100.0),
        f32n(0.0),
        f32n(100.0),
        u32n(1),
        booln(false),
        empty_arr(),
        empty_arr(),
        u32n(0),
        u32n(0),
        empty_arr(),
        empty_arr(),
        empty_array_like(BLOCK_OWNS, 0),
        u32n(0),
        u32n(0),
        empty_arr(),
        empty_arr(),
        empty_arr(),
        empty_arr(),
        booln(false),
        u32n(0),
        booln(false),
    ];
    debug_assert_eq!(hdr.len(), HEADER_LEN);
    item.splice(i0..i0 + HEADER_LEN, hdr);
    Ok(())
}

/// Pool item `[u32 type, PROPSET, u32 id, f32...]`: keep the id, flags and the constant provider list #20, empty every other list.
pub fn pristine_hdr_keep_id(it: &mut [Node]) -> Result<()> {
    let i0 = 2;
    if i0 + HEADER_LEN > it.len() {
        bail!("pool item too short ({} elements)", it.len());
    }
    for n in 0..HEADER_LEN {
        if n == 18 {
            continue;
        }
        let idx = i0 + n;
        let replacement = match &it[idx] {
            Node::Value(Value::Array { code: 0x48, .. }) => Some(empty_arr()),
            Node::Array { name, version, .. } => Some(empty_array_like(name, *version)),
            _ => None,
        };
        if let Some(r) = replacement {
            it[idx] = r;
        }
    }
    Ok(())
}

// ---- id remapping --------------------------------------------------------------------------------------------------
/// Replace scalar u32/i32 values and packed u32/i32 array elements equal to mapping keys, recursively through records and
/// record-array items (Python `remap_scalars` + `remap_scalars_nested`).
pub fn remap_children(ch: &mut [Node], map: &BTreeMap<u32, u32>) {
    for c in ch.iter_mut() {
        match c {
            Node::Value(Value::U32(x)) => {
                if let Some(n) = map.get(x) {
                    *x = *n;
                }
            }
            Node::Value(Value::I32(x)) => {
                if let Some(n) = map.get(&(*x as u32)) {
                    *x = *n as i32;
                }
            }
            Node::Value(Value::Array { code, data }) if *code == 0x48 || *code == 0x44 => {
                for chunk in data.chunks_exact_mut(4) {
                    let v = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if let Some(n) = map.get(&v) {
                        chunk.copy_from_slice(&n.to_le_bytes());
                    }
                }
            }
            Node::Value(_) => {}
            Node::Record { children, .. } => remap_children(children, map),
            Node::Array { items, .. } => {
                for it in items.iter_mut() {
                    remap_children(it, map);
                }
            }
        }
    }
}

/// Remove the values in `drop` from every packed u32 array below `ch`.
pub fn scrub_arrays(ch: &mut [Node], drop: &BTreeSet<u32>) {
    for c in ch.iter_mut() {
        match c {
            Node::Value(Value::Array { code: 0x48, data }) if !data.is_empty() => {
                if data
                    .chunks_exact(4)
                    .any(|b| drop.contains(&u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                {
                    let kept: Vec<u8> = data
                        .chunks_exact(4)
                        .filter(|b| !drop.contains(&u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                        .flatten()
                        .copied()
                        .collect();
                    *data = kept;
                }
            }
            Node::Value(_) => {}
            Node::Record { children, .. } => scrub_arrays(children, drop),
            Node::Array { items, .. } => {
                for it in items.iter_mut() {
                    scrub_arrays(it, drop);
                }
            }
        }
    }
}
