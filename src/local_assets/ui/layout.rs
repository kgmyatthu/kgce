//! Reader, editor and writer of the binary UI layout files (`Version039`) of Napoleon: Total War.
//!
//! The format follows the open etwng layout converter. Every field of the supported version is kept, so an
//! unedited layout writes back to the identical bytes; the generator verifies that for each source before it
//! edits anything. Other layout versions are rejected.

use anyhow::{Context, Result, bail};
use std::collections::{BTreeSet, HashMap};

pub(super) const SUPPORTED_VERSION: &str = "039";

#[derive(Clone, Debug)]
pub(super) struct Tga {
    pub id: i32,
    pub path: String,
    pub width: i32,
    pub height: i32,
    pub int1: i32,
}

#[derive(Clone, Debug)]
pub(super) struct TgaUse {
    pub id: i32,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub flags: [u8; 7],
    pub position: i32,
    pub flags45: [u8; 2],
    pub ints: [i32; 3],
}

#[derive(Clone, Debug)]
pub(super) struct Transition {
    pub kind: i32,
    pub id: i32,
    pub str1: String,
    pub int1: i32,
    pub int2: i32,
}

#[derive(Clone, Debug)]
pub(super) struct State {
    pub id: i32,
    pub title: String,
    pub width: i32,
    pub height: i32,
    pub state_text: Vec<u16>,
    pub tooltip: Vec<u16>,
    pub ints7_11: [i32; 5],
    pub flag7: u8,
    pub localisation_id: Vec<u16>,
    pub tooltip_id: Vec<u16>,
    pub font: String,
    pub int12: i32,
    pub int13: i32,
    pub int14: i32,
    pub ints15_17: [i32; 3],
    pub flags8_10: [u8; 3],
    pub normalt0: String,
    pub ints18_21: [i32; 4],
    pub state_description: String,
    pub event_text: String,
    pub tga_uses: Vec<TgaUse>,
    pub int23: i32,
    pub int24: i32,
    pub transitions: Vec<Transition>,
}

#[derive(Clone, Debug)]
pub(super) struct Effect {
    pub name: String,
    pub flag: i16,
    /// Two float bit patterns followed by ten integers per phase.
    pub phases: Vec<[u32; 12]>,
}

#[derive(Clone, Debug)]
pub(super) struct Entry {
    pub id: i32,
    pub title: String,
    pub x: i32,
    pub y: i32,
    pub flags123: [u8; 3],
    pub flags11_14: [u8; 4],
    pub parent_name: String,
    pub int1: i32,
    pub tooltip: Vec<u16>,
    pub tooltip_text: Vec<u16>,
    pub int3: i32,
    pub flag4: u8,
    /// Id of the current/default state (0 or one of the entry's own state ids).
    pub int4: i32,
    pub script: String,
    pub tgas: Vec<Tga>,
    pub int5: i32,
    pub int6: i32,
    pub states: Vec<State>,
    pub int26: i32,
    pub events: Vec<String>,
    pub int27: i32,
    pub effects: Vec<Effect>,
    pub children: Vec<Entry>,
    pub template: String,
}

/// The file holds one root entry (title of the layout) below the version header.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub root: Entry,
}

pub(super) fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

// ------------------------------------------------------------------------------------------------ reading

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.p + n > self.b.len() {
            bail!("layout truncated at byte {}", self.p);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn i32(&mut self) -> Result<i32> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(self.i32()? as u32)
    }
    fn i16(&mut self) -> Result<i16> {
        let s = self.take(2)?;
        Ok(i16::from_le_bytes([s[0], s[1]]))
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(self.i16()? as u16)
    }
    fn count(&mut self) -> Result<usize> {
        let n = self.i32()?;
        if n < 0 || n as usize > self.b.len() {
            bail!("implausible element count {n} at byte {}", self.p);
        }
        Ok(n as usize)
    }
    fn ascii(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        let s = self.take(n)?;
        if !s.is_ascii() {
            bail!("non-ASCII text in layout string at byte {}", self.p);
        }
        Ok(String::from_utf8_lossy(s).into_owned())
    }
    fn utf16(&mut self) -> Result<Vec<u16>> {
        let n = self.u16()? as usize;
        let s = self.take(n * 2)?;
        Ok(s.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
    fn ints<const N: usize>(&mut self) -> Result<[i32; N]> {
        let mut a = [0; N];
        for v in &mut a {
            *v = self.i32()?;
        }
        Ok(a)
    }
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }
}

fn read_entry(r: &mut Rd) -> Result<Entry> {
    let id = r.i32()?;
    let title = r.ascii()?;
    let x = r.i32()?;
    let y = r.i32()?;
    let flags123 = r.bytes()?;
    let flags11_14 = r.bytes()?;
    let parent_name = r.ascii()?;
    let int1 = r.i32()?;
    let tooltip = r.utf16()?;
    let tooltip_text = r.utf16()?;
    let int3 = r.i32()?;
    let flag4 = r.u8()?;
    let int4 = r.i32()?;
    let script = r.ascii()?;
    let n = r.count()?;
    let mut tgas = Vec::with_capacity(n);
    for _ in 0..n {
        tgas.push(Tga {
            id: r.i32()?,
            path: r.ascii()?,
            width: r.i32()?,
            height: r.i32()?,
            int1: r.i32()?,
        });
    }
    let int5 = r.i32()?;
    let int6 = r.i32()?;
    let n = r.count()?;
    let mut states = Vec::with_capacity(n);
    for _ in 0..n {
        states.push(read_state(r)?);
    }
    let int26 = r.i32()?;
    let mut events = Vec::new();
    loop {
        let e = r.ascii()?;
        if e == "events_end" {
            break;
        }
        events.push(e);
    }
    let int27 = r.i32()?;
    let n = r.count()?;
    let mut effects = Vec::with_capacity(n);
    for _ in 0..n {
        let name = r.ascii()?;
        let flag = r.i16()?;
        let pn = r.count()?;
        let mut phases = Vec::with_capacity(pn);
        for _ in 0..pn {
            let mut ph = [0u32; 12];
            for w in &mut ph {
                *w = r.u32()?;
            }
            phases.push(ph);
        }
        effects.push(Effect { name, flag, phases });
    }
    let n = r.count()?;
    let mut children = Vec::with_capacity(n);
    for _ in 0..n {
        children.push(read_entry(r)?);
    }
    let template = r.ascii()?;
    Ok(Entry {
        id,
        title,
        x,
        y,
        flags123,
        flags11_14,
        parent_name,
        int1,
        tooltip,
        tooltip_text,
        int3,
        flag4,
        int4,
        script,
        tgas,
        int5,
        int6,
        states,
        int26,
        events,
        int27,
        effects,
        children,
        template,
    })
}

fn read_state(r: &mut Rd) -> Result<State> {
    let id = r.i32()?;
    let title = r.ascii()?;
    let width = r.i32()?;
    let height = r.i32()?;
    let state_text = r.utf16()?;
    let tooltip = r.utf16()?;
    let ints7_11 = r.ints()?;
    let flag7 = r.u8()?;
    let localisation_id = r.utf16()?;
    let tooltip_id = r.utf16()?;
    let font = r.ascii()?;
    let int12 = r.i32()?;
    let int13 = r.i32()?;
    let int14 = r.i32()?;
    let ints15_17 = r.ints()?;
    let flags8_10 = r.bytes()?;
    let normalt0 = r.ascii()?;
    let ints18_21 = r.ints()?;
    let state_description = r.ascii()?;
    let event_text = r.ascii()?;
    let n = r.count()?;
    let mut tga_uses = Vec::with_capacity(n);
    for _ in 0..n {
        tga_uses.push(TgaUse {
            id: r.i32()?,
            x: r.i32()?,
            y: r.i32()?,
            width: r.i32()?,
            height: r.i32()?,
            flags: r.bytes()?,
            position: r.i32()?,
            flags45: r.bytes()?,
            ints: r.ints()?,
        });
    }
    let int23 = r.i32()?;
    let int24 = r.i32()?;
    let n = r.count()?;
    let mut transitions = Vec::with_capacity(n);
    for _ in 0..n {
        transitions.push(Transition {
            kind: r.i32()?,
            id: r.i32()?,
            str1: r.ascii()?,
            int1: r.i32()?,
            int2: r.i32()?,
        });
    }
    Ok(State {
        id,
        title,
        width,
        height,
        state_text,
        tooltip,
        ints7_11,
        flag7,
        localisation_id,
        tooltip_id,
        font,
        int12,
        int13,
        int14,
        ints15_17,
        flags8_10,
        normalt0,
        ints18_21,
        state_description,
        event_text,
        tga_uses,
        int23,
        int24,
        transitions,
    })
}

impl Layout {
    pub(super) fn parse(bytes: &[u8]) -> Result<Layout> {
        if bytes.len() < 10 || &bytes[..7] != b"Version" {
            bail!("not a UI layout file");
        }
        let v = std::str::from_utf8(&bytes[7..10]).unwrap_or("?");
        if v != SUPPORTED_VERSION {
            bail!("UI layout version {v} is not supported (only Version{SUPPORTED_VERSION})");
        }
        let mut r = Rd { b: bytes, p: 10 };
        let root = read_entry(&mut r)?;
        if r.p != bytes.len() {
            bail!("{} trailing bytes after the layout tree", bytes.len() - r.p);
        }
        Ok(Layout { root })
    }

    pub(super) fn write(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(b"Version");
        o.extend_from_slice(SUPPORTED_VERSION.as_bytes());
        write_entry(&mut o, &self.root);
        o
    }

    /// Structural checks the engine relies on: unique entry ids, `int4` is 0 or an own state id, texture
    /// uses refer to textures of the entry or an ancestor, transitions target own states.
    pub(super) fn validate(&self, name: &str) -> Result<()> {
        let mut seen = BTreeSet::new();
        fn ids(e: &Entry, seen: &mut BTreeSet<i32>, name: &str) -> Result<()> {
            if !seen.insert(e.id) {
                bail!("{name}: duplicate uiEntry id {}", e.id);
            }
            for c in &e.children {
                ids(c, seen, name)?;
            }
            Ok(())
        }
        ids(&self.root, &mut seen, name)?;
        fn walk(e: &Entry, anc: &BTreeSet<i32>, name: &str) -> Result<()> {
            let own: BTreeSet<i32> = e.tgas.iter().map(|t| t.id).collect();
            let states: BTreeSet<i32> = e.states.iter().map(|s| s.id).collect();
            if e.int4 != 0 && !states.contains(&e.int4) {
                bail!("{name}: {} int4={} is not an own state id", e.title, e.int4);
            }
            for s in &e.states {
                for u in &s.tga_uses {
                    if !own.contains(&u.id) && !anc.contains(&u.id) {
                        bail!("{name}: {} texture use {} is dangling", e.title, u.id);
                    }
                }
                for t in &s.transitions {
                    if !states.contains(&t.id) {
                        bail!("{name}: {} transition target {} is dangling", e.title, t.id);
                    }
                }
            }
            let next: BTreeSet<i32> = anc.union(&own).copied().collect();
            for c in &e.children {
                walk(c, &next, name)?;
            }
            Ok(())
        }
        walk(&self.root, &BTreeSet::new(), name)
    }
}

// ------------------------------------------------------------------------------------------------ writing

fn w_i32(o: &mut Vec<u8>, v: i32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn w_ascii(o: &mut Vec<u8>, s: &str) {
    o.extend_from_slice(&(s.len() as u16).to_le_bytes());
    o.extend_from_slice(s.as_bytes());
}
fn w_utf16(o: &mut Vec<u8>, s: &[u16]) {
    o.extend_from_slice(&(s.len() as u16).to_le_bytes());
    for u in s {
        o.extend_from_slice(&u.to_le_bytes());
    }
}

fn write_entry(o: &mut Vec<u8>, e: &Entry) {
    w_i32(o, e.id);
    w_ascii(o, &e.title);
    w_i32(o, e.x);
    w_i32(o, e.y);
    o.extend_from_slice(&e.flags123);
    o.extend_from_slice(&e.flags11_14);
    w_ascii(o, &e.parent_name);
    w_i32(o, e.int1);
    w_utf16(o, &e.tooltip);
    w_utf16(o, &e.tooltip_text);
    w_i32(o, e.int3);
    o.push(e.flag4);
    w_i32(o, e.int4);
    w_ascii(o, &e.script);
    w_i32(o, e.tgas.len() as i32);
    for t in &e.tgas {
        w_i32(o, t.id);
        w_ascii(o, &t.path);
        w_i32(o, t.width);
        w_i32(o, t.height);
        w_i32(o, t.int1);
    }
    w_i32(o, e.int5);
    w_i32(o, e.int6);
    w_i32(o, e.states.len() as i32);
    for s in &e.states {
        write_state(o, s);
    }
    w_i32(o, e.int26);
    for ev in &e.events {
        w_ascii(o, ev);
    }
    w_ascii(o, "events_end");
    w_i32(o, e.int27);
    w_i32(o, e.effects.len() as i32);
    for f in &e.effects {
        w_ascii(o, &f.name);
        o.extend_from_slice(&f.flag.to_le_bytes());
        w_i32(o, f.phases.len() as i32);
        for ph in &f.phases {
            for w in ph {
                o.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    w_i32(o, e.children.len() as i32);
    for c in &e.children {
        write_entry(o, c);
    }
    w_ascii(o, &e.template);
}

fn write_state(o: &mut Vec<u8>, s: &State) {
    w_i32(o, s.id);
    w_ascii(o, &s.title);
    w_i32(o, s.width);
    w_i32(o, s.height);
    w_utf16(o, &s.state_text);
    w_utf16(o, &s.tooltip);
    for v in s.ints7_11 {
        w_i32(o, v);
    }
    o.push(s.flag7);
    w_utf16(o, &s.localisation_id);
    w_utf16(o, &s.tooltip_id);
    w_ascii(o, &s.font);
    w_i32(o, s.int12);
    w_i32(o, s.int13);
    w_i32(o, s.int14);
    for v in s.ints15_17 {
        w_i32(o, v);
    }
    o.extend_from_slice(&s.flags8_10);
    w_ascii(o, &s.normalt0);
    for v in s.ints18_21 {
        w_i32(o, v);
    }
    w_ascii(o, &s.state_description);
    w_ascii(o, &s.event_text);
    w_i32(o, s.tga_uses.len() as i32);
    for u in &s.tga_uses {
        w_i32(o, u.id);
        w_i32(o, u.x);
        w_i32(o, u.y);
        w_i32(o, u.width);
        w_i32(o, u.height);
        o.extend_from_slice(&u.flags);
        w_i32(o, u.position);
        o.extend_from_slice(&u.flags45);
        for v in u.ints {
            w_i32(o, v);
        }
    }
    w_i32(o, s.int23);
    w_i32(o, s.int24);
    w_i32(o, s.transitions.len() as i32);
    for t in &s.transitions {
        w_i32(o, t.kind);
        w_i32(o, t.id);
        w_ascii(o, &t.str1);
        w_i32(o, t.int1);
        w_i32(o, t.int2);
    }
}

// ------------------------------------------------------------------------------------------------ editing

impl Entry {
    pub(super) fn child(&self, title: &str) -> Result<&Entry> {
        self.children
            .iter()
            .find(|c| c.title == title)
            .with_context(|| format!("layout entry {title} not found below {}", self.title))
    }

    pub(super) fn child_mut(&mut self, title: &str) -> Result<&mut Entry> {
        let parent = self.title.clone();
        self.children
            .iter_mut()
            .find(|c| c.title == title)
            .with_context(|| format!("layout entry {title} not found below {parent}"))
    }

    /// First entry titled `title` in document order (this entry included).
    pub(super) fn find_mut(&mut self, title: &str) -> Result<&mut Entry> {
        fn go<'a>(e: &'a mut Entry, title: &str) -> Option<&'a mut Entry> {
            if e.title == title {
                return Some(e);
            }
            e.children.iter_mut().find_map(|c| go(c, title))
        }
        let from = self.title.clone();
        go(self, title).with_context(|| format!("layout entry {title} not found below {from}"))
    }

    /// Resizes an entry that has exactly one state with one texture use (state size and every use).
    pub(super) fn set_size(
        &mut self,
        w: i32,
        h: i32,
        tga_w: Option<i32>,
        tga_h: Option<i32>,
    ) -> Result<()> {
        let Some(st) = self.states.first_mut() else {
            bail!("layout entry {} has no state", self.title)
        };
        st.width = w;
        st.height = h;
        for u in &mut st.tga_uses {
            u.width = tga_w.unwrap_or(w);
            u.height = tga_h.unwrap_or(h);
        }
        Ok(())
    }

    /// Scales the entry's own states and texture uses (not child positions) by `s`.
    pub(super) fn scale(&mut self, s: f64) {
        for st in &mut self.states {
            st.width = round_even(st.width as f64 * s);
            st.height = round_even(st.height as f64 * s);
            for u in &mut st.tga_uses {
                u.x = round_even(u.x as f64 * s);
                u.y = round_even(u.y as f64 * s);
                u.width = round_even(u.width as f64 * s);
                u.height = round_even(u.height as f64 * s);
            }
        }
    }

    fn visit_ids(&self, f: &mut dyn FnMut(i32)) {
        f(self.id);
        for s in &self.states {
            f(s.id);
            for u in &s.tga_uses {
                f(u.id);
            }
        }
        for t in &self.tgas {
            f(t.id);
        }
        for c in &self.children {
            c.visit_ids(f);
        }
    }
}

/// Round half to even (Python `round`).
fn round_even(v: f64) -> i32 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        (r - v.signum()) as i32
    } else {
        r as i32
    }
}

/// Allocates layout ids that collide with no id present in the layout.
pub(super) struct IdAlloc {
    used: BTreeSet<i32>,
    next: i32,
}

impl IdAlloc {
    pub(super) fn new(layout: &Layout, start: i32) -> Self {
        let mut used = BTreeSet::new();
        layout.root.visit_ids(&mut |i| {
            used.insert(i);
        });
        Self { used, next: start }
    }

    fn fresh(&mut self) -> i32 {
        while self.used.contains(&self.next) {
            self.next += 1;
        }
        let v = self.next;
        self.used.insert(v);
        self.next += 1;
        v
    }

    /// Deep copy with fresh ids for every entry, state and texture defined inside, and every reference to them
    /// (texture uses, transition targets, the entry's current-state id) rewritten.
    pub(super) fn clone_entry(&mut self, e: &Entry) -> Entry {
        // Allocation order: every entry (pre-order), then every state, then every texture.
        let (mut entries, mut states, mut tgas) = (Vec::new(), Vec::new(), Vec::new());
        fn walk(e: &Entry, entries: &mut Vec<i32>, states: &mut Vec<i32>, tgas: &mut Vec<i32>) {
            entries.push(e.id);
            for s in &e.states {
                states.push(s.id);
            }
            for t in &e.tgas {
                tgas.push(t.id);
            }
            for c in &e.children {
                walk(c, entries, states, tgas);
            }
        }
        walk(e, &mut entries, &mut states, &mut tgas);
        let mut map: HashMap<i32, i32> = HashMap::new();
        for old in entries.into_iter().chain(states).chain(tgas) {
            if !map.contains_key(&old) {
                let n = self.fresh();
                map.insert(old, n);
            }
        }
        let mut c = e.clone();
        remap(&mut c, &map);
        c
    }
}

fn remap(e: &mut Entry, map: &HashMap<i32, i32>) {
    let m = |v: &mut i32| {
        if let Some(n) = map.get(v) {
            *v = *n;
        }
    };
    m(&mut e.id);
    m(&mut e.int4);
    for t in &mut e.tgas {
        m(&mut t.id);
    }
    for s in &mut e.states {
        m(&mut s.id);
        for u in &mut s.tga_uses {
            m(&mut u.id);
        }
        for t in &mut s.transitions {
            m(&mut t.id);
        }
    }
    for c in &mut e.children {
        remap(c, map);
    }
}
