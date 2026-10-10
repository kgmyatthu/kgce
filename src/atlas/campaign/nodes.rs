//! Typed access to the flat child lists of start-position records (the POC addressed them by index with `value(..)` /
//! `set_value(..)`; every accessor here fails when the child has another ESF type, so a layout change cannot be
//! converted silently).

use anyhow::{Result, anyhow, bail};

use crate::esf::{Node, Value};

/// `i32` sentinel (`-2147483647`) the engine stores for "no position".
pub const SENT: i32 = -2_147_483_647;

fn describe(n: Option<&Node>) -> String {
    match n {
        None => "no child".to_string(),
        Some(Node::Record { name, .. }) => format!("record {name}"),
        Some(Node::Array { name, .. }) => format!("record array {name}"),
        Some(Node::Value(v)) => format!("value code {:#04x}", v.code()),
    }
}

fn value_at<'a>(ch: &'a [Node], at: usize, want: &str) -> Result<&'a Value> {
    match ch.get(at) {
        Some(Node::Value(v)) => Ok(v),
        other => bail!("child {at}: expected {want}, found {}", describe(other)),
    }
}

pub fn i32_at(ch: &[Node], at: usize) -> Result<i32> {
    match value_at(ch, at, "i32")? {
        Value::I32(v) => Ok(*v),
        v => bail!("child {at}: expected i32, found value code {:#04x}", v.code()),
    }
}

pub fn set_i32(ch: &mut [Node], at: usize, new: i32) -> Result<()> {
    i32_at(ch, at)?;
    ch[at] = Node::Value(Value::I32(new));
    Ok(())
}

pub fn u32_at(ch: &[Node], at: usize) -> Result<u32> {
    match value_at(ch, at, "u32")? {
        Value::U32(v) => Ok(*v),
        v => bail!("child {at}: expected u32, found value code {:#04x}", v.code()),
    }
}

pub fn f32_at(ch: &[Node], at: usize) -> Result<f32> {
    match value_at(ch, at, "f32")? {
        Value::F32(v) => Ok(*v),
        v => bail!("child {at}: expected f32, found value code {:#04x}", v.code()),
    }
}

pub fn set_f32(ch: &mut [Node], at: usize, new: f32) -> Result<()> {
    f32_at(ch, at)?;
    ch[at] = Node::Value(Value::F32(new));
    Ok(())
}

pub fn bool_at(ch: &[Node], at: usize) -> Result<bool> {
    match value_at(ch, at, "bool")? {
        Value::Bool(v) => Ok(*v),
        v => bail!("child {at}: expected bool, found value code {:#04x}", v.code()),
    }
}

pub fn xy_at(ch: &[Node], at: usize) -> Result<[f32; 2]> {
    match value_at(ch, at, "xy")? {
        Value::XY(v) => Ok(*v),
        v => bail!("child {at}: expected xy, found value code {:#04x}", v.code()),
    }
}

pub fn set_xy(ch: &mut [Node], at: usize, new: [f32; 2]) -> Result<()> {
    xy_at(ch, at)?;
    ch[at] = Node::Value(Value::XY(new));
    Ok(())
}

pub fn xyz_at(ch: &[Node], at: usize) -> Result<[f32; 3]> {
    match value_at(ch, at, "xyz")? {
        Value::XYZ(v) => Ok(*v),
        v => bail!("child {at}: expected xyz, found value code {:#04x}", v.code()),
    }
}

pub fn set_xyz(ch: &mut [Node], at: usize, new: [f32; 3]) -> Result<()> {
    xyz_at(ch, at)?;
    ch[at] = Node::Value(Value::XYZ(new));
    Ok(())
}

pub fn str_at(ch: &[Node], at: usize) -> Result<&str> {
    match value_at(ch, at, "string")? {
        Value::String16(s) | Value::String8(s) => Ok(s),
        v => bail!("child {at}: expected string, found value code {:#04x}", v.code()),
    }
}

/// Raw bytes of the packed array of ESF type `code` at `at`.
pub fn array_at(ch: &[Node], at: usize, code: u8) -> Result<&[u8]> {
    match value_at(ch, at, "packed array")? {
        Value::Array { code: c, data } if *c == code => Ok(data),
        v => bail!("child {at}: expected packed array {code:#04x}, found value code {:#04x}", v.code()),
    }
}

pub fn set_array(ch: &mut [Node], at: usize, code: u8, data: Vec<u8>) -> Result<()> {
    array_at(ch, at, code)?;
    ch[at] = Node::Value(Value::Array { code, data });
    Ok(())
}

/// Byte length of the packed array of any type at `at`.
pub fn array_len(ch: &[Node], at: usize) -> Result<usize> {
    match value_at(ch, at, "packed array")? {
        Value::Array { data, .. } => Ok(data.len()),
        v => bail!("child {at}: expected packed array, found value code {:#04x}", v.code()),
    }
}

/// Elements of a packed `u32` array (`0x48`).
pub fn u32_list(ch: &[Node], at: usize) -> Result<Vec<u32>> {
    let data = array_at(ch, at, 0x48)?;
    if data.len() % 4 != 0 {
        bail!("child {at}: u32 array of {} bytes", data.len());
    }
    Ok(data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

pub fn u32_list_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// True when child `at` is an empty packed `u32` array.
pub fn is_empty_u32_array(ch: &[Node], at: usize) -> bool {
    matches!(ch.get(at), Some(Node::Value(Value::Array { code: 0x48, data })) if data.is_empty())
}

/// Children of a record child.
pub fn record_children<'a>(n: &'a Node, name: &str) -> Result<&'a Vec<Node>> {
    match n {
        Node::Record { name: nm, children, .. } if nm == name => Ok(children),
        other => Err(anyhow!("expected record {name}, found {}", describe(Some(other)))),
    }
}

pub fn record_children_mut<'a>(n: &'a mut Node, name: &str) -> Result<&'a mut Vec<Node>> {
    match n {
        Node::Record { name: actual, children, .. } => {
            if actual != name {
                bail!("expected record {name}, found record {actual}");
            }
            Ok(children)
        }
        other => Err(anyhow!("expected record {name}, found {}", describe(Some(&*other)))),
    }
}

/// First record child (of an array item) called `name`.
pub fn find_record<'a>(item: &'a [Node], name: &str) -> Result<&'a Vec<Node>> {
    item.iter()
        .find_map(|c| match c {
            Node::Record { name: n, children, .. } if n == name => Some(children),
            _ => None,
        })
        .ok_or_else(|| anyhow!("item has no record {name}"))
}

pub fn find_record_mut<'a>(item: &'a mut [Node], name: &str) -> Result<&'a mut Vec<Node>> {
    item.iter_mut()
        .find_map(|c| match c {
            Node::Record { name: n, children, .. } if n == name => Some(children),
            _ => None,
        })
        .ok_or_else(|| anyhow!("item has no record {name}"))
}

/// Id scalar (first scalar child) of a CAI world item.
pub fn item_id(item: &[Node]) -> Result<u32> {
    match item.iter().find_map(|c| if let Node::Value(v) = c { Some(v) } else { None }) {
        Some(Value::U32(v)) => Ok(*v),
        other => Err(anyhow!("CAI world item has no u32 id (found {other:?})")),
    }
}

/// Depth-first search for the first record called `name`.
pub fn find_named_mut<'a>(node: &'a mut Node, name: &str) -> Option<&'a mut Node> {
    if matches!(node, Node::Record { name: n, .. } if n == name) {
        return Some(node);
    }
    match node {
        Node::Record { children, .. } => children.iter_mut().find_map(|c| find_named_mut(c, name)),
        Node::Array { items, .. } => items.iter_mut().flatten().find_map(|c| find_named_mut(c, name)),
        Node::Value(_) => None,
    }
}

pub fn find_named<'a>(node: &'a Node, name: &str) -> Option<&'a Node> {
    match node {
        Node::Record { name: n, .. } if n == name => Some(node),
        Node::Record { children, .. } => children.iter().find_map(|c| find_named(c, name)),
        Node::Array { items, .. } => items.iter().flatten().find_map(|c| find_named(c, name)),
        Node::Value(_) => None,
    }
}

/// Visit every record called `name` (depth-first).
pub fn for_each_record<'a>(node: &'a Node, name: &str, f: &mut dyn FnMut(&'a Node)) {
    match node {
        Node::Record { name: n, children, .. } => {
            if n == name {
                f(node);
            }
            children.iter().for_each(|c| for_each_record(c, name, f));
        }
        Node::Array { items, .. } => items.iter().flatten().for_each(|c| for_each_record(c, name, f)),
        Node::Value(_) => {}
    }
}

/// Children of the record array `name` below `world` (a record).
pub fn world_items<'a>(world: &'a Node, name: &str) -> Result<&'a Vec<Vec<Node>>> {
    match world {
        Node::Record { children, .. } => children
            .iter()
            .find_map(|c| match c {
                Node::Array { name: n, items, .. } if n == name => Some(items),
                _ => None,
            })
            .ok_or_else(|| anyhow!("no record array {name}")),
        _ => Err(anyhow!("expected a record while looking for {name}")),
    }
}

pub fn world_items_mut<'a>(world: &'a mut Node, name: &str) -> Result<&'a mut Vec<Vec<Node>>> {
    match world {
        Node::Record { children, .. } => children
            .iter_mut()
            .find_map(|c| match c {
                Node::Array { name: n, items, .. } if n == name => Some(items),
                _ => None,
            })
            .ok_or_else(|| anyhow!("no record array {name}")),
        _ => Err(anyhow!("expected a record while looking for {name}")),
    }
}
