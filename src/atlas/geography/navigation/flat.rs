//! Cursor over the flat scalar stream of `poi.esf` / `sea_grids.esf` that can replace the value it just consumed.

use anyhow::{Result, bail};

use super::diff::Touched;
use crate::esf::{Node, Value};

pub(super) struct Flat<'a> {
    vals: &'a mut Vec<Node>,
    pos: usize,
    touched: &'a mut Touched,
}

impl<'a> Flat<'a> {
    pub fn new(vals: &'a mut Vec<Node>, touched: &'a mut Touched) -> Self {
        Self { vals, pos: 0, touched }
    }

    fn next(&mut self, want: &str) -> Result<&Value> {
        let at = self.pos;
        if at >= self.vals.len() {
            bail!("stream exhausted at {at}, expected {want}");
        }
        self.pos += 1;
        match &self.vals[at] {
            Node::Value(v) => Ok(v),
            other => bail!("expected {want} at {at}, found {}", super::util::describe(other)),
        }
    }

    pub fn u32(&mut self) -> Result<u32> {
        let at = self.pos;
        match self.next("u32")? {
            Value::U32(v) => Ok(*v),
            v => bail!("expected scalar u32 at {at}, found {v:?}"),
        }
    }

    pub fn i32(&mut self) -> Result<i32> {
        let at = self.pos;
        match self.next("i32")? {
            Value::I32(v) => Ok(*v),
            v => bail!("expected scalar i32 at {at}, found {v:?}"),
        }
    }

    pub fn bool(&mut self) -> Result<bool> {
        let at = self.pos;
        match self.next("bool")? {
            Value::Bool(v) => Ok(*v),
            v => bail!("expected scalar bool at {at}, found {v:?}"),
        }
    }

    pub fn f32(&mut self) -> Result<f32> {
        let at = self.pos;
        match self.next("f32")? {
            Value::F32(v) => Ok(*v),
            v => bail!("expected scalar f32 at {at}, found {v:?}"),
        }
    }

    pub fn xy(&mut self) -> Result<[f32; 2]> {
        let at = self.pos;
        match self.next("xy")? {
            Value::XY(v) => Ok(*v),
            v => bail!("expected scalar xy at {at}, found {v:?}"),
        }
    }

    pub fn string(&mut self) -> Result<String> {
        let at = self.pos;
        match self.next("string")? {
            Value::String16(s) | Value::String8(s) => Ok(s.clone()),
            v => bail!("expected string at {at}, found {v:?}"),
        }
    }

    fn slot(&mut self, back: usize) -> Result<&mut Value> {
        if back == 0 || back > self.pos {
            bail!("replace {back} back from {}", self.pos);
        }
        let idx = self.pos - back;
        self.touched.add(&[idx]);
        match &mut self.vals[idx] {
            Node::Value(v) => Ok(v),
            other => bail!("replace type mismatch at {idx}: {}", super::util::describe(other)),
        }
    }

    /// Replace the i32 consumed `back` positions ago (1 = last).
    pub fn replace_i32(&mut self, back: usize, new: i32) -> Result<()> {
        match self.slot(back)? {
            Value::I32(v) => *v = new,
            other => bail!("replace type mismatch: {other:?} is not i32"),
        }
        Ok(())
    }

    pub fn replace_f32(&mut self, back: usize, new: f32) -> Result<()> {
        match self.slot(back)? {
            Value::F32(v) => *v = new,
            other => bail!("replace type mismatch: {other:?} is not f32"),
        }
        Ok(())
    }

    pub fn replace_xy(&mut self, back: usize, new: [f32; 2]) -> Result<()> {
        match self.slot(back)? {
            Value::XY(v) => *v = new,
            other => bail!("replace type mismatch: {other:?} is not xy"),
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<()> {
        if self.pos != self.vals.len() {
            bail!("unconsumed stream: {} values", self.vals.len() - self.pos);
        }
        Ok(())
    }
}

/// Read-only variant of [`Flat`].
pub(super) struct Reader<'a> {
    vals: &'a [Node],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(vals: &'a [Node]) -> Self {
        Self { vals, pos: 0 }
    }

    fn next(&mut self, want: &str) -> Result<&'a Value> {
        let at = self.pos;
        match self.vals.get(at) {
            None => bail!("stream exhausted at {at}, expected {want}"),
            Some(Node::Value(v)) => {
                self.pos += 1;
                Ok(v)
            }
            Some(other) => bail!("expected {want} at {at}, found {}", super::util::describe(other)),
        }
    }

    pub fn u32(&mut self) -> Result<u32> {
        let at = self.pos;
        match self.next("u32")? {
            Value::U32(v) => Ok(*v),
            v => bail!("expected scalar u32 at {at}, found {v:?}"),
        }
    }

    pub fn i32(&mut self) -> Result<i32> {
        let at = self.pos;
        match self.next("i32")? {
            Value::I32(v) => Ok(*v),
            v => bail!("expected scalar i32 at {at}, found {v:?}"),
        }
    }

    pub fn bool(&mut self) -> Result<bool> {
        let at = self.pos;
        match self.next("bool")? {
            Value::Bool(v) => Ok(*v),
            v => bail!("expected scalar bool at {at}, found {v:?}"),
        }
    }

    pub fn f32(&mut self) -> Result<f32> {
        let at = self.pos;
        match self.next("f32")? {
            Value::F32(v) => Ok(*v),
            v => bail!("expected scalar f32 at {at}, found {v:?}"),
        }
    }

    pub fn string(&mut self) -> Result<&'a str> {
        let at = self.pos;
        match self.next("string")? {
            Value::String16(s) | Value::String8(s) => Ok(s.as_str()),
            v => bail!("expected string at {at}, found {v:?}"),
        }
    }

    pub fn finish(&self) -> Result<()> {
        if self.pos != self.vals.len() {
            bail!("unconsumed stream: {} values", self.vals.len() - self.pos);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esf;

    #[test]
    fn take_checks_types_and_replace_marks_touched() {
        let mut vals = vec![esf::u32v(1), esf::i32v(5), esf::f32v(2.0)];
        let mut touched = Touched::default();
        let mut f = Flat::new(&mut vals, &mut touched);
        assert_eq!(f.u32().unwrap(), 1);
        assert_eq!(f.i32().unwrap(), 5);
        f.replace_i32(1, 7).unwrap();
        assert!(f.u32().is_err());
        assert_eq!(touched.paths.len(), 1);
        assert!(touched.paths.contains(&vec![1usize]));
        assert_eq!(vals[1], esf::i32v(7));
    }

    #[test]
    fn replace_type_mismatch_and_unconsumed_stream_are_errors() {
        let mut vals = vec![esf::u32v(1), esf::f32v(2.0)];
        let mut touched = Touched::default();
        let mut f = Flat::new(&mut vals, &mut touched);
        f.u32().unwrap();
        assert!(f.replace_f32(1, 1.0).is_err());
        assert!(f.finish().is_err());
        f.f32().unwrap();
        f.finish().unwrap();
    }
}
