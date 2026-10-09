//! Lua 5.1 bytecode reader and structured decompiler.
//!
//! The Napoleon front-end and campaign HUD scripts ship as compiled Lua 5.1 chunks. The local UI rebuild needs
//! their source text so that the small authored changes can be applied as readable text patches. The decompiler
//! covers the code shapes produced by the stock `luac` for those scripts (locals from the debug information,
//! if/elseif/else, while, numeric and generic `for`, `and`/`or` values, comparison values, table constructors,
//! closures, method calls). Anything else fails with the function path and instruction index instead of
//! producing guessed source.

use anyhow::{Result, bail};

const OP_MOVE: u32 = 0;
const OP_LOADK: u32 = 1;
const OP_LOADBOOL: u32 = 2;
const OP_LOADNIL: u32 = 3;
const OP_GETUPVAL: u32 = 4;
const OP_GETGLOBAL: u32 = 5;
const OP_GETTABLE: u32 = 6;
const OP_SETGLOBAL: u32 = 7;
const OP_SETUPVAL: u32 = 8;
const OP_SETTABLE: u32 = 9;
const OP_NEWTABLE: u32 = 10;
const OP_SELF: u32 = 11;
const OP_ADD: u32 = 12;
const OP_POW: u32 = 17;
const OP_UNM: u32 = 18;
const OP_NOT: u32 = 19;
const OP_LEN: u32 = 20;
const OP_CONCAT: u32 = 21;
const OP_JMP: u32 = 22;
const OP_EQ: u32 = 23;
const OP_LT: u32 = 24;
const OP_LE: u32 = 25;
const OP_TEST: u32 = 26;
const OP_TESTSET: u32 = 27;
const OP_CALL: u32 = 28;
const OP_TAILCALL: u32 = 29;
const OP_RETURN: u32 = 30;
const OP_FORLOOP: u32 = 31;
const OP_FORPREP: u32 = 32;
const OP_TFORLOOP: u32 = 33;
const OP_SETLIST: u32 = 34;
const OP_CLOSE: u32 = 35;
const OP_CLOSURE: u32 = 36;
const OP_VARARG: u32 = 37;

#[derive(Clone, Debug)]
pub(super) enum Const {
    Nil,
    Bool(bool),
    Num(f64),
    Str(Vec<u8>),
}

#[derive(Clone, Debug)]
pub(super) struct LocVar {
    name: String,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
pub(super) struct Proto {
    num_params: usize,
    is_vararg: u8,
    code: Vec<u32>,
    consts: Vec<Const>,
    protos: Vec<Proto>,
    locvars: Vec<LocVar>,
    upvalues: Vec<String>,
    nups: usize,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
    size_t: usize,
    num: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.p + n > self.b.len() {
            bail!("truncated Lua chunk at byte {}", self.p);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn count(&mut self) -> Result<usize> {
        let n = self.u32()? as usize;
        if n > self.b.len() {
            bail!("implausible element count {n} in Lua chunk");
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<Option<Vec<u8>>> {
        let n = if self.size_t == 4 {
            self.u32()? as usize
        } else {
            let s = self.take(8)?;
            u64::from_le_bytes(s.try_into().unwrap()) as usize
        };
        if n == 0 {
            return Ok(None);
        }
        let s = self.take(n)?;
        Ok(Some(s[..n - 1].to_vec()))
    }
    fn number(&mut self) -> Result<f64> {
        if self.num == 4 {
            let s = self.take(4)?;
            Ok(f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64)
        } else {
            let s = self.take(8)?;
            Ok(f64::from_le_bytes(s.try_into().unwrap()))
        }
    }
}

fn read_proto(r: &mut Rd) -> Result<Proto> {
    r.string()?; // source name
    r.u32()?; // line defined
    r.u32()?; // last line defined
    let nups = r.u8()? as usize;
    let num_params = r.u8()? as usize;
    let is_vararg = r.u8()?;
    r.u8()?; // max stack
    let n = r.count()?;
    let mut code = Vec::with_capacity(n);
    for _ in 0..n {
        code.push(r.u32()?);
    }
    let n = r.count()?;
    let mut consts = Vec::with_capacity(n);
    for _ in 0..n {
        consts.push(match r.u8()? {
            0 => Const::Nil,
            1 => Const::Bool(r.u8()? != 0),
            3 => Const::Num(r.number()?),
            4 => Const::Str(r.string()?.unwrap_or_default()),
            t => bail!("unknown constant type {t}"),
        });
    }
    let n = r.count()?;
    let mut protos = Vec::with_capacity(n);
    for _ in 0..n {
        protos.push(read_proto(r)?);
    }
    let n = r.count()?;
    r.take(n * 4)?; // line info
    let n = r.count()?;
    let mut locvars = Vec::with_capacity(n);
    for _ in 0..n {
        let name = String::from_utf8_lossy(&r.string()?.unwrap_or_default()).into_owned();
        let start = r.u32()? as usize;
        let end = r.u32()? as usize;
        locvars.push(LocVar { name, start, end });
    }
    let n = r.count()?;
    let mut upvalues = Vec::with_capacity(n);
    for _ in 0..n {
        upvalues.push(String::from_utf8_lossy(&r.string()?.unwrap_or_default()).into_owned());
    }
    Ok(Proto {
        num_params,
        is_vararg,
        code,
        consts,
        protos,
        locvars,
        upvalues,
        nups,
    })
}

/// Parses a compiled Lua 5.1 chunk. Needs the debug information (local and upvalue names).
pub(super) fn parse(bytes: &[u8]) -> Result<Proto> {
    if bytes.len() < 12 || &bytes[..4] != b"\x1bLua" {
        bail!("not a compiled Lua chunk");
    }
    if bytes[4] != 0x51
        || bytes[5] != 0
        || bytes[6] != 1
        || bytes[7] != 4
        || bytes[9] != 4
        || bytes[11] != 0
    {
        bail!("unsupported Lua bytecode header {:02x?}", &bytes[4..12]);
    }
    let (size_t, num) = (bytes[8] as usize, bytes[10] as usize);
    if !matches!(size_t, 4 | 8) || !matches!(num, 4 | 8) {
        bail!("unsupported Lua bytecode header {:02x?}", &bytes[4..12]);
    }
    let mut r = Rd {
        b: bytes,
        p: 12,
        size_t,
        num,
    };
    let p = read_proto(&mut r)?;
    if r.p != bytes.len() {
        bail!("{} trailing bytes after the Lua chunk", bytes.len() - r.p);
    }
    Ok(p)
}

// ------------------------------------------------------------------------------------------------ AST

#[derive(Clone, PartialEq, Debug)]
enum Expr {
    Nil,
    True,
    False,
    Num(f64),
    Str(Vec<u8>),
    Vararg,
    Name(String),
    Index(Box<Expr>, Box<Expr>),
    Call {
        f: Box<Expr>,
        args: Vec<Expr>,
        multi: bool,
    },
    Method {
        obj: Box<Expr>,
        name: String,
        args: Vec<Expr>,
        multi: bool,
    },
    Bin(&'static str, Box<Expr>, Box<Expr>),
    Un(&'static str, Box<Expr>),
    Table(Vec<(Option<Expr>, Expr)>),
    Func(Box<FuncAst>),
    /// Pending `SELF` result (method reference); only valid until the following call consumes it.
    SelfRef(Box<Expr>, String),
    /// Extra result slot of a multi-value producer (or of `SELF`).
    Cont,
}

#[derive(Clone, PartialEq, Debug)]
struct FuncAst {
    params: Vec<String>,
    vararg: bool,
    body: Vec<Stmt>,
}

#[derive(Clone, PartialEq, Debug)]
enum Stmt {
    Local(Vec<String>, Vec<Expr>),
    LocalFunc(String, FuncAst),
    Assign(Expr, Expr),
    Call(Expr),
    If(Vec<(Expr, Vec<Stmt>)>, Option<Vec<Stmt>>),
    While(Expr, Vec<Stmt>),
    NumFor(String, Expr, Expr, Option<Expr>, Vec<Stmt>),
    GenFor(Vec<String>, Vec<Expr>, Vec<Stmt>),
    Return(Vec<Expr>),
    Break,
}

// ------------------------------------------------------------------------------------------------ rendering

const KEYWORDS: [&str; 22] = [
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if", "in", "local",
    "nil", "not", "or", "repeat", "return", "then", "true", "until", "while", "goto",
];

fn is_ident(s: &[u8]) -> bool {
    let Some(&f) = s.first() else { return false };
    if !(f.is_ascii_alphabetic() || f == b'_')
        || !s.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
    {
        return false;
    }
    !KEYWORDS.contains(&std::str::from_utf8(s).unwrap_or(""))
}

fn quote(s: &[u8]) -> String {
    let mut o = String::from("\"");
    for &c in s {
        match c {
            b'\\' => o.push_str("\\\\"),
            b'"' => o.push_str("\\\""),
            b'\n' => o.push_str("\\n"),
            b'\r' => o.push_str("\\r"),
            b'\t' => o.push_str("\\t"),
            0x20..=0x7e => o.push(c as char),
            _ => o.push_str(&format!("\\{c:03}")),
        }
    }
    o.push('"');
    o
}

fn number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    if (v as f32) as f64 == v {
        format!("{}", v as f32)
    } else {
        format!("{v}")
    }
}

fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Bin(op, ..) => match *op {
            "or" => 1,
            "and" => 2,
            "<" | ">" | "<=" | ">=" | "~=" | "==" => 3,
            ".." => 4,
            "+" | "-" => 5,
            "*" | "/" | "%" => 6,
            _ => 8,
        },
        Expr::Un(..) => 7,
        Expr::Func(_) => 0,
        _ => 10,
    }
}

fn pad(ind: usize) -> String {
    "  ".repeat(ind)
}

fn sub(e: &Expr, min: u8, ind: usize) -> String {
    let s = expr(e, ind);
    if prec(e) < min { format!("({s})") } else { s }
}

fn prefix(e: &Expr, ind: usize) -> String {
    match e {
        Expr::Name(_) | Expr::Index(..) | Expr::Call { .. } | Expr::Method { .. } => expr(e, ind),
        _ => format!("({})", expr(e, ind)),
    }
}

fn explist(l: &[Expr], ind: usize) -> String {
    let mut v: Vec<String> = l.iter().map(|e| expr(e, ind)).collect();
    // A single-value call in tail position keeps its truncation explicit.
    if let Some(last) = l.last() {
        let trunc = matches!(
            last,
            Expr::Call { multi: false, .. } | Expr::Method { multi: false, .. }
        );
        if trunc && l.len() > 1 {
            let n = v.len();
            v[n - 1] = format!("({})", v[n - 1]);
        }
    }
    v.join(", ")
}

fn args_text(args: &[Expr], ind: usize) -> String {
    let mut v: Vec<String> = args.iter().map(|e| expr(e, ind)).collect();
    if let Some(Expr::Call { multi: false, .. } | Expr::Method { multi: false, .. }) = args.last() {
        let n = v.len();
        v[n - 1] = format!("({})", v[n - 1]);
    }
    v.join(", ")
}

fn func_head(f: &FuncAst) -> String {
    let mut p = f.params.clone();
    if f.vararg {
        p.push("...".into());
    }
    p.join(", ")
}

fn func_text(f: &FuncAst, head: &str, ind: usize) -> String {
    let mut lines = vec![format!("{head}({})", func_head(f))];
    stmts(&mut lines, &f.body, ind + 1);
    lines.push(format!("{}end", pad(ind)));
    lines.join("\n")
}

fn expr(e: &Expr, ind: usize) -> String {
    match e {
        Expr::Nil => "nil".into(),
        Expr::True => "true".into(),
        Expr::False => "false".into(),
        Expr::Num(v) => number(*v),
        Expr::Str(s) => quote(s),
        Expr::Vararg => "...".into(),
        Expr::Name(n) => n.clone(),
        Expr::Index(o, k) => match &**k {
            Expr::Str(s) if is_ident(s) => {
                format!("{}.{}", prefix(o, ind), String::from_utf8_lossy(s))
            }
            _ => format!("{}[{}]", prefix(o, ind), expr(k, ind)),
        },
        Expr::Call { f, args, .. } => format!("{}({})", prefix(f, ind), args_text(args, ind)),
        Expr::Method {
            obj, name, args, ..
        } => format!("{}:{}({})", prefix(obj, ind), name, args_text(args, ind)),
        Expr::Bin(op, l, r) => {
            let p = prec(e);
            let right_assoc = matches!(*op, ".." | "^");
            let assoc_same = matches!(*op, "and" | "or");
            let ls = sub(l, if right_assoc { p + 1 } else { p }, ind);
            let rs = sub(r, if right_assoc || assoc_same { p } else { p + 1 }, ind);
            format!("{ls} {op} {rs}")
        }
        Expr::Un(op, x) => {
            let s = sub(x, 7, ind);
            if *op == "not" {
                format!("not {s}")
            } else if s.starts_with('-') {
                format!("{op} {s}")
            } else {
                format!("{op}{s}")
            }
        }
        Expr::Table(fields) => {
            if fields.is_empty() {
                return "{}".into();
            }
            let item = |(k, v): &(Option<Expr>, Expr), ind: usize| match k {
                None => expr(v, ind),
                Some(Expr::Str(s)) if is_ident(s) => {
                    format!("{} = {}", String::from_utf8_lossy(s), expr(v, ind))
                }
                Some(k) => format!("[{}] = {}", expr(k, ind), expr(v, ind)),
            };
            if fields.len() == 1 && !matches!(fields[0].1, Expr::Func(_) | Expr::Table(_)) {
                return format!("{{{}}}", item(&fields[0], ind));
            }
            let mut s = String::from("{\n");
            let n = fields.len();
            for (i, f) in fields.iter().enumerate() {
                s.push_str(&pad(ind + 1));
                s.push_str(&item(f, ind + 1));
                s.push_str(if i + 1 < n { ",\n" } else { "\n" });
            }
            s.push_str(&pad(ind));
            s.push('}');
            s
        }
        Expr::Func(f) => func_text(f, "function", ind),
        Expr::SelfRef(..) | Expr::Cont => "<internal>".into(),
    }
}

fn stmts(out: &mut Vec<String>, list: &[Stmt], ind: usize) {
    let p = pad(ind);
    for s in list {
        match s {
            Stmt::Local(names, vals) => {
                let mut l = format!("{p}local {}", names.join(", "));
                if !vals.is_empty() {
                    l.push_str(" = ");
                    l.push_str(&explist(vals, ind));
                }
                out.push(l);
            }
            Stmt::LocalFunc(n, f) => out.push(format!(
                "{p}{}",
                func_text(f, &format!("local function {n}"), ind)
            )),
            Stmt::Assign(t, v) => {
                if let Expr::Func(f) = v {
                    let name = match t {
                        Expr::Name(n) => Some(n.clone()),
                        Expr::Index(o, k) => match (&**o, &**k) {
                            (Expr::Name(_) | Expr::Index(..), Expr::Str(s)) if is_ident(s) => {
                                Some(format!("{}.{}", expr(o, ind), String::from_utf8_lossy(s)))
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(n) = name {
                        out.push(format!(
                            "{p}{}",
                            func_text(f, &format!("function {n}"), ind)
                        ));
                        continue;
                    }
                }
                out.push(format!("{p}{} = {}", expr(t, ind), expr(v, ind)));
            }
            Stmt::Call(c) => out.push(format!("{p}{}", expr(c, ind))),
            Stmt::If(arms, els) => {
                for (i, (c, b)) in arms.iter().enumerate() {
                    out.push(format!(
                        "{p}{} {} then",
                        if i == 0 { "if" } else { "elseif" },
                        expr(c, ind)
                    ));
                    stmts(out, b, ind + 1);
                }
                if let Some(e) = els {
                    out.push(format!("{p}else"));
                    stmts(out, e, ind + 1);
                }
                out.push(format!("{p}end"));
            }
            Stmt::While(c, b) => {
                out.push(format!("{p}while {} do", expr(c, ind)));
                stmts(out, b, ind + 1);
                out.push(format!("{p}end"));
            }
            Stmt::NumFor(v, a, b, st, body) => {
                let step = st
                    .as_ref()
                    .map(|s| format!(", {}", expr(s, ind)))
                    .unwrap_or_default();
                out.push(format!(
                    "{p}for {v} = {}, {}{step} do",
                    expr(a, ind),
                    expr(b, ind)
                ));
                stmts(out, body, ind + 1);
                out.push(format!("{p}end"));
            }
            Stmt::GenFor(vars, ex, body) => {
                out.push(format!(
                    "{p}for {} in {} do",
                    vars.join(", "),
                    explist(ex, ind)
                ));
                stmts(out, body, ind + 1);
                out.push(format!("{p}end"));
            }
            Stmt::Return(v) => {
                if v.is_empty() {
                    out.push(format!("{p}return"));
                } else {
                    out.push(format!("{p}return {}", explist(v, ind)));
                }
            }
            Stmt::Break => out.push(format!("{p}break")),
        }
    }
}

// ------------------------------------------------------------------------------------------------ decompiler

fn neg(e: Expr) -> Expr {
    match e {
        Expr::Un("not", x) => *x,
        Expr::Bin("==", a, b) => Expr::Bin("~=", a, b),
        Expr::Bin("~=", a, b) => Expr::Bin("==", a, b),
        o => Expr::Un("not", Box::new(o)),
    }
}

fn and(l: Expr, r: Expr) -> Expr {
    Expr::Bin("and", Box::new(l), Box::new(r))
}

fn or(l: Expr, r: Expr) -> Expr {
    Expr::Bin("or", Box::new(l), Box::new(r))
}

#[derive(Clone)]
enum V {
    T,
    F,
    E(Expr),
}

struct Item {
    taken: Expr,
    target: usize,
    start: usize,
    pc: usize,
}

struct Chain {
    cond: Expr,
    /// First instruction after the last jump.
    b: usize,
    /// Target of the last jump.
    f: usize,
}

struct Dec<'a> {
    p: &'a Proto,
    path: String,
    regs: Vec<Option<Expr>>,
    /// head pc -> pc of the backward jump of a `while` loop
    while_heads: std::collections::HashMap<usize, usize>,
}

fn op_of(i: u32) -> u32 {
    i & 0x3f
}
fn a_of(i: u32) -> usize {
    ((i >> 6) & 0xff) as usize
}
fn b_of(i: u32) -> usize {
    ((i >> 23) & 0x1ff) as usize
}
fn c_of(i: u32) -> usize {
    ((i >> 14) & 0x1ff) as usize
}
fn bx_of(i: u32) -> usize {
    (i >> 14) as usize
}
fn sbx_of(i: u32) -> i64 {
    (i >> 14) as i64 - 131071
}

fn is_test(op: u32) -> bool {
    matches!(op, OP_EQ | OP_LT | OP_LE | OP_TEST | OP_TESTSET)
}

impl<'a> Dec<'a> {
    fn new(p: &'a Proto, path: String) -> Result<Self> {
        let mut while_heads = std::collections::HashMap::new();
        let mut test_backs = Vec::new();
        for (pc, &i) in p.code.iter().enumerate() {
            if op_of(i) == OP_JMP && sbx_of(i) < 0 {
                let head = (pc as i64 + 1 + sbx_of(i)) as usize;
                if pc > 0 && op_of(p.code[pc - 1]) == OP_TFORLOOP {
                    continue;
                }
                if pc > 0 && is_test(op_of(p.code[pc - 1])) {
                    test_backs.push((pc, head));
                    continue;
                }
                while_heads.insert(head, pc);
            }
        }
        // A test jumping backwards is either an `if` at the end of a while body (its jump was threaded to the
        // loop head) or a repeat-until loop, which is not supported.
        for (pc, head) in test_backs {
            if !while_heads.contains_key(&head) {
                bail!("{path}: repeat-until loop at instruction {pc} is not supported");
            }
        }
        Ok(Self {
            p,
            path,
            regs: vec![None; 256],
            while_heads,
        })
    }

    fn err<T>(&self, pc: usize, msg: &str) -> Result<T> {
        bail!("{}: instruction {pc}: {msg}", self.path)
    }

    fn active(&self, pc: usize) -> impl Iterator<Item = &LocVar> {
        self.p
            .locvars
            .iter()
            .filter(move |l| l.start <= pc && pc < l.end)
    }

    fn nactive(&self, pc: usize) -> usize {
        self.active(pc).count()
    }

    fn local_name(&self, reg: usize, pc: usize) -> Option<&str> {
        self.active(pc).nth(reg).map(|l| l.name.as_str())
    }

    fn konst(&self, k: usize, pc: usize) -> Result<Expr> {
        Ok(match self.p.consts.get(k) {
            Some(Const::Nil) => Expr::Nil,
            Some(Const::Bool(true)) => Expr::True,
            Some(Const::Bool(false)) => Expr::False,
            Some(Const::Num(n)) => Expr::Num(*n),
            Some(Const::Str(s)) => Expr::Str(s.clone()),
            None => return self.err(pc, "constant index out of range"),
        })
    }

    fn read(&mut self, r: usize, pc: usize) -> Result<Expr> {
        if let Some(n) = self.local_name(r, pc) {
            return Ok(Expr::Name(n.to_owned()));
        }
        match self.regs[r].take() {
            Some(Expr::Cont) | None => {
                self.err(pc, &format!("register {r} read before it was written"))
            }
            Some(e) => Ok(e),
        }
    }

    fn rk(&mut self, x: usize, pc: usize) -> Result<Expr> {
        if x >= 256 {
            self.konst(x - 256, pc)
        } else {
            self.read(x, pc)
        }
    }

    fn upname(&self, i: usize, pc: usize) -> Result<String> {
        match self.p.upvalues.get(i) {
            Some(n) if !n.is_empty() => Ok(n.clone()),
            _ => self.err(pc, "upvalue name missing from the debug information"),
        }
    }

    fn write(&mut self, a: usize, pc: usize, e: Expr, out: &mut Vec<Stmt>) {
        if let Some(n) = self.local_name(a, pc) {
            out.push(Stmt::Assign(Expr::Name(n.to_owned()), e));
        } else {
            self.regs[a] = Some(e);
        }
    }

    /// Emits `local ...` for variables whose scope starts right after instruction `pc`.
    fn decl(&mut self, pc: usize, out: &mut Vec<Stmt>) -> Result<()> {
        let new: Vec<&LocVar> = self
            .p
            .locvars
            .iter()
            .filter(|l| l.start == pc + 1 && !l.name.starts_with('('))
            .collect();
        if new.is_empty() {
            return Ok(());
        }
        let base = self.nactive(pc);
        let names: Vec<String> = new.iter().map(|l| l.name.clone()).collect();
        let mut vals = Vec::new();
        for k in 0..names.len() {
            match self.regs[base + k].take() {
                Some(Expr::Cont) => {}
                Some(e) => vals.push(e),
                None => return self.err(pc, &format!("local {} has no pending value", names[k])),
            }
        }
        while matches!(vals.last(), Some(Expr::Nil)) {
            vals.pop();
        }
        out.push(Stmt::Local(names, vals));
        Ok(())
    }

    // ---- purity scanning (no state changes)

    fn decl_at(&self, pc: usize) -> bool {
        self.p
            .locvars
            .iter()
            .any(|l| l.start == pc && !l.name.starts_with('('))
    }

    /// Skips instructions that only compute pending values. Returns the first other instruction.
    fn scan_pure(&self, from: usize, limit: usize, decl_ok: Option<usize>) -> usize {
        let code = &self.p.code;
        let mut open: Vec<usize> = Vec::new();
        let mut pc = from;
        while pc < limit {
            let i = code[pc];
            let op = op_of(i);
            let a = a_of(i);
            let mut len = 1;
            let pure = match op {
                OP_MOVE | OP_LOADK | OP_GETUPVAL | OP_GETGLOBAL | OP_GETTABLE | OP_SELF
                | OP_UNM | OP_NOT | OP_LEN | OP_CONCAT | OP_VARARG => a >= self.nactive(pc),
                OP_ADD..=OP_POW => a >= self.nactive(pc),
                OP_LOADNIL => a >= self.nactive(pc),
                OP_LOADBOOL => c_of(i) == 0 && a >= self.nactive(pc),
                OP_NEWTABLE => {
                    open.push(a);
                    a >= self.nactive(pc)
                }
                OP_CALL => c_of(i) != 1 && a >= self.nactive(pc),
                OP_SETTABLE | OP_SETLIST => open.contains(&a),
                OP_CLOSURE => {
                    len = 1 + self.p.protos.get(bx_of(i)).map_or(0, |c| c.nups);
                    a >= self.nactive(pc)
                }
                _ => false,
            };
            if !pure {
                return pc;
            }
            let last = pc + len - 1;
            if self.decl_at(last + 1) && decl_ok != Some(last + 1) {
                return pc;
            }
            pc += len;
        }
        pc
    }

    /// Runs the straight-line instructions in `[from, to)`; none may produce a statement.
    fn run(&mut self, from: usize, to: usize) -> Result<()> {
        let mut pc = from;
        let mut scratch = Vec::new();
        while pc < to {
            pc = self.step(pc, &mut scratch)?;
        }
        if !scratch.is_empty() {
            return self.err(from, "statement inside an expression");
        }
        Ok(())
    }

    // ---- conditions

    /// Jump instruction after a test; returns its forward target.
    fn test_target(&self, p: usize) -> Option<usize> {
        let j = *self.p.code.get(p + 1)?;
        if op_of(j) != OP_JMP {
            return None;
        }
        Some((p as i64 + 2 + sbx_of(j)) as usize)
    }

    fn item_expr(&mut self, p: usize) -> Result<Expr> {
        let i = self.p.code[p];
        let (a, b, c) = (a_of(i), b_of(i), c_of(i));
        Ok(match op_of(i) {
            OP_EQ | OP_LT | OP_LE => {
                let l = self.rk(b, p)?;
                let r = self.rk(c, p)?;
                let op = match op_of(i) {
                    OP_EQ => {
                        return Ok(Expr::Bin(
                            if a == 1 { "==" } else { "~=" },
                            Box::new(l),
                            Box::new(r),
                        ));
                    }
                    OP_LT => "<",
                    _ => "<=",
                };
                let e = Expr::Bin(op, Box::new(l), Box::new(r));
                if a == 1 {
                    e
                } else {
                    Expr::Un("not", Box::new(e))
                }
            }
            OP_TEST => {
                let v = self.read(a, p)?;
                if c == 1 {
                    v
                } else {
                    Expr::Un("not", Box::new(v))
                }
            }
            OP_TESTSET => {
                let v = self.read(b, p)?;
                if c == 1 {
                    v
                } else {
                    Expr::Un("not", Box::new(v))
                }
            }
            _ => return self.err(p, "not a test instruction"),
        })
    }

    /// `TEST r 1; JMP end` that closes the first operand of `cond and A or B` (value passes to `end`).
    fn is_value_tail(&self, stop: usize) -> bool {
        let i = self.p.code[stop];
        if op_of(i) != OP_TEST || c_of(i) != 1 {
            return false;
        }
        let Some(t) = self.test_target(stop) else {
            return false;
        };
        t > stop + 2
            && self.scan_pure(stop + 2, t, Some(t)) == t
            && a_of(self.p.code[t - 1]) == a_of(i)
            && Self::writes_reg(op_of(self.p.code[t - 1]))
    }

    /// Greedy chain of consecutive tests starting at `p` (operand computations in between must be pure).
    fn chain_positions(&self, p: usize) -> Vec<(usize, usize)> {
        let mut v = vec![(p, self.test_target(p).unwrap_or(0))];
        let mut cur = p + 2;
        loop {
            let stop = self.scan_pure(cur, self.p.code.len(), None);
            if stop + 1 < self.p.code.len()
                && is_test(op_of(self.p.code[stop]))
                && !self.is_value_tail(stop)
            {
                if let Some(t) = self.test_target(stop) {
                    v.push((stop, t));
                    cur = stop + 2;
                    continue;
                }
            }
            break;
        }
        v
    }

    fn valid_chain(items: &[(usize, usize)], start0: usize, labels: &[usize]) -> bool {
        let starts: Vec<usize> = std::iter::once(start0)
            .chain(items.iter().map(|(pc, _)| pc + 2))
            .collect();
        items
            .iter()
            .enumerate()
            .all(|(i, &(_, t))| labels.contains(&t) || starts.iter().skip(i + 2).any(|&s| s == t))
    }

    /// Evaluates the chain's operands and folds its jumps into one condition.
    /// `labels`: (pc, outcome) for jump targets, `fall`: outcome of falling out of the last jump.
    fn eval_chain(
        &mut self,
        cur: usize,
        pos: &[(usize, usize)],
        labels: &[(usize, bool)],
        fall: bool,
    ) -> Result<Expr> {
        let mut items = Vec::new();
        let mut at = cur;
        for &(pc, target) in pos {
            self.run(at, pc)?;
            let start = at;
            let taken = self.item_expr(pc)?;
            items.push(Item {
                taken,
                target,
                start,
                pc,
            });
            at = pc + 2;
        }
        match res(&items, 0, labels, fall)? {
            V::E(e) => Ok(e),
            V::T => Ok(Expr::True),
            V::F => Ok(Expr::False),
        }
    }

    /// Statement-context chain: falling out of the last jump means "true", `f` means "false".
    fn cond_chain(&mut self, cur: usize, p: usize) -> Result<Chain> {
        let mut pos = self.chain_positions(p);
        let (b, f);
        loop {
            let last = *pos.last().unwrap();
            let bb = last.0 + 2;
            let ff = last.1;
            if Self::valid_chain(&pos, cur, &[bb, ff]) || pos.len() == 1 {
                b = bb;
                f = ff;
                break;
            }
            pos.pop();
        }
        let labels = [(f, false), (b, true)];
        let cond = self.eval_chain(cur, &pos, &labels, true)?;
        Ok(Chain { cond, b, f })
    }

    /// `x = a < b` style comparison values: [tests] LOADBOOL r 0 1 / LOADBOOL r 1 0.
    fn bool_value(&mut self, p: usize) -> Result<Option<(usize, Expr, usize)>> {
        let full = self.chain_positions(p);
        let mut pos = full.clone();
        loop {
            let last = *pos.last().unwrap();
            let b = last.0 + 2;
            let c = &self.p.code;
            let shape = b + 1 < c.len()
                && op_of(c[b]) == OP_LOADBOOL
                && b_of(c[b]) == 0
                && c_of(c[b]) == 1
                && op_of(c[b + 1]) == OP_LOADBOOL
                && b_of(c[b + 1]) == 1
                && c_of(c[b + 1]) == 0
                && a_of(c[b]) == a_of(c[b + 1]);
            if shape && Self::valid_chain(&pos, p, &[b, b + 1]) && !self.decl_at(b) {
                let r = a_of(c[b]);
                let labels = [(b, false), (b + 1, true)];
                let e = self.eval_chain(p, &pos, &labels, false)?;
                return Ok(Some((r, e, b + 2)));
            }
            if pos.len() == 1 {
                return Ok(None);
            }
            pos.pop();
        }
    }

    /// `x or y` / `x and y` values left in a register by TEST/TESTSET jumping to the end of the expression.
    fn logic_value(&mut self, p: usize) -> Result<Option<(usize, Expr, usize)>> {
        let code = &self.p.code;
        let first = code[p];
        if !matches!(op_of(first), OP_TEST | OP_TESTSET) {
            return Ok(None);
        }
        let Some(end) = self.test_target(p) else {
            return Ok(None);
        };
        let r = a_of(first);
        let mut pos = vec![p];
        let mut cur = p + 2;
        loop {
            if cur >= end {
                return Ok(None);
            }
            let stop = self.scan_pure(cur, end, Some(end));
            if stop == end {
                // final segment must be non-empty and end by writing r
                if cur >= end {
                    return Ok(None);
                }
                let last = self.last_instr_before(end);
                if !matches!(last, Some(l) if a_of(self.p.code[l]) == r && Self::writes_reg(op_of(self.p.code[l])))
                {
                    return Ok(None);
                }
                break;
            }
            let i = self.p.code[stop];
            if matches!(op_of(i), OP_TEST | OP_TESTSET)
                && a_of(i) == r
                && self.test_target(stop) == Some(end)
            {
                pos.push(stop);
                cur = stop + 2;
            } else {
                return Ok(None);
            }
        }
        // evaluate
        let mut parts: Vec<(Expr, &'static str)> = Vec::new();
        let mut at = p;
        for &pc in &pos {
            self.run(at, pc)?;
            let i = self.p.code[pc];
            let v = if op_of(i) == OP_TEST {
                self.read(a_of(i), pc)?
            } else {
                self.read(b_of(i), pc)?
            };
            parts.push((v, if c_of(i) == 1 { "or" } else { "and" }));
            at = pc + 2;
        }
        self.run(at, end)?;
        let mut acc = match self.regs[r].take() {
            Some(e) => e,
            None => return self.err(p, "logical value has no result"),
        };
        for (v, op) in parts.into_iter().rev() {
            acc = Expr::Bin(op, Box::new(v), Box::new(acc));
        }
        Ok(Some((r, acc, end)))
    }

    fn last_instr_before(&self, end: usize) -> Option<usize> {
        // the instruction that ends right before `end`; CLOSURE pseudo-instructions are not written by any of our shapes
        end.checked_sub(1)
    }

    fn writes_reg(op: u32) -> bool {
        matches!(
            op,
            OP_MOVE
                | OP_LOADK
                | OP_LOADBOOL
                | OP_LOADNIL
                | OP_GETUPVAL
                | OP_GETGLOBAL
                | OP_GETTABLE
                | OP_NEWTABLE
                | OP_ADD..=OP_CONCAT | OP_CALL | OP_VARARG
        )
    }

    /// `cond and A or B` with the final TEST/JMP after A.
    fn ternary_value(
        &mut self,
        p: usize,
        ch: Chain,
    ) -> Result<Result<(usize, Expr, usize), Chain>> {
        let (b, f) = (ch.b, ch.f);
        let c = &self.p.code;
        if f < b + 3 || f + 1 > c.len() {
            return Ok(Err(ch));
        }
        let (t, j) = (c[f - 2], c[f - 1]);
        if op_of(t) != OP_TEST || c_of(t) != 1 || op_of(j) != OP_JMP || sbx_of(j) < 0 {
            return Ok(Err(ch));
        }
        let end = (f as i64 + sbx_of(j)) as usize;
        let r = a_of(t);
        if end <= f
            || self.scan_pure(b, f - 2, None) != f - 2
            || self.scan_pure(f, end, Some(end)) != end
        {
            return Ok(Err(ch));
        }
        let l1 = f - 3;
        let l2 = end - 1;
        if a_of(self.p.code[l1]) != r || a_of(self.p.code[l2]) != r {
            return Ok(Err(ch));
        }
        let _ = p;
        self.run(b, f - 2)?;
        let a_val = self.read(r, f - 2)?;
        self.run(f, end)?;
        let b_val = self.read(r, end)?;
        Ok(Ok((r, or(and(ch.cond, a_val), b_val), end)))
    }

    // ---- structures

    fn block(&mut self, mut pc: usize, end: usize, exit: Option<usize>) -> Result<Vec<Stmt>> {
        let mut out = Vec::new();
        while pc < end {
            if let Some(&j) = self.while_heads.get(&pc) {
                if j < end {
                    pc = self.while_loop(pc, j, &mut out)?;
                    continue;
                }
            }
            let i = self.p.code[pc];
            match op_of(i) {
                OP_EQ | OP_LT | OP_LE | OP_TEST | OP_TESTSET => {
                    pc = self.conditional(pc, end, exit, &mut out)?
                }
                OP_JMP => {
                    let target = (pc as i64 + 1 + sbx_of(i)) as usize;
                    if target > pc
                        && target < self.p.code.len()
                        && op_of(self.p.code[target]) == OP_TFORLOOP
                    {
                        pc = self.gen_for(pc, target, &mut out)?;
                    } else if exit == Some(target) {
                        out.push(Stmt::Break);
                        pc += 1;
                    } else {
                        return self.err(pc, "unstructured jump");
                    }
                }
                OP_FORPREP => pc = self.num_for(pc, &mut out)?,
                OP_RETURN | OP_TAILCALL => pc = self.ret(pc, &mut out)?,
                OP_FORLOOP | OP_TFORLOOP => return self.err(pc, "loop end outside its loop"),
                _ => pc = self.step(pc, &mut out)?,
            }
        }
        if pc != end {
            return self.err(pc, "statement runs past the end of its block");
        }
        Ok(out)
    }

    /// Jump targets of the last statement of a block are threaded by the compiler to the target of the jump that
    /// follows the block (loop head, else-skip, loop exit). Such a target means "the end of this block".
    fn clamp(&self, t: usize, end: usize) -> usize {
        if t != end && end < self.p.code.len() {
            let j = self.p.code[end];
            if op_of(j) == OP_JMP && (end as i64 + 1 + sbx_of(j)) as usize == t {
                return end;
            }
        }
        t
    }

    fn conditional(
        &mut self,
        p: usize,
        end: usize,
        exit: Option<usize>,
        out: &mut Vec<Stmt>,
    ) -> Result<usize> {
        if self.test_target(p).is_none() {
            return self.err(p, "test not followed by a jump");
        }
        // value shapes first
        let snapshot = self.regs.clone();
        if let Some((r, e, next)) = self.logic_value(p)? {
            self.regs[r] = Some(e);
            self.decl(next - 1, out)?;
            return Ok(next);
        }
        self.regs = snapshot.clone();
        if let Some((r, e, next)) = self.bool_value(p)? {
            self.regs[r] = Some(e);
            self.decl(next - 1, out)?;
            return Ok(next);
        }
        self.regs = snapshot;
        let ch = self.cond_chain(p, p)?;
        let ch = match self.ternary_value(p, ch)? {
            Ok((r, e, next)) => {
                self.regs[r] = Some(e);
                self.decl(next - 1, out)?;
                return Ok(next);
            }
            Err(ch) => ch,
        };
        let (b, f) = (ch.b, self.clamp(ch.f, end));
        let code = &self.p.code;
        let mut else_end = None;
        if f > b {
            let j = code[f - 1];
            if op_of(j) == OP_JMP {
                let t = (f as i64 + sbx_of(j)) as usize;
                let e = self.clamp(t, end);
                if exit != Some(t) && e >= f && e <= end {
                    else_end = Some(e);
                }
            }
        }
        let (then, els, next) = match else_end {
            Some(e) => {
                let then = self.block(b, f - 1, exit)?;
                let els = if e > f {
                    Some(self.block(f, e, exit)?)
                } else {
                    None
                };
                (then, els, e)
            }
            None => (self.block(b, f, exit)?, None, f),
        };
        let mut arms = vec![(ch.cond, then)];
        let mut tail = els;
        // `else if` without further statements is an `elseif`
        while let Some(t) = tail.take() {
            if t.len() == 1 {
                if let Stmt::If(a2, e2) = &t[0] {
                    arms.extend(a2.iter().cloned());
                    tail = e2.clone();
                    continue;
                }
            }
            tail = Some(t);
            break;
        }
        out.push(Stmt::If(arms, tail));
        Ok(next)
    }

    fn while_loop(&mut self, head: usize, j: usize, out: &mut Vec<Stmt>) -> Result<usize> {
        let stop = self.scan_pure(head, j, None);
        let cond;
        let body_start;
        if stop < j && is_test(op_of(self.p.code[stop])) {
            let ch = self.cond_chain(head, stop)?;
            if ch.f != j + 1 {
                return self.err(head, "while condition does not exit at the loop end");
            }
            cond = ch.cond;
            body_start = ch.b;
        } else {
            cond = Expr::True;
            body_start = head;
        }
        let body = self.block(body_start, j, Some(j + 1))?;
        out.push(Stmt::While(cond, body));
        Ok(j + 1)
    }

    fn num_for(&mut self, pc: usize, out: &mut Vec<Stmt>) -> Result<usize> {
        let i = self.p.code[pc];
        let a = a_of(i);
        let target = (pc as i64 + 1 + sbx_of(i)) as usize;
        if target >= self.p.code.len() || op_of(self.p.code[target]) != OP_FORLOOP {
            return self.err(pc, "numeric for without matching loop instruction");
        }
        let init = self.take_pending(a, pc)?;
        let limit = self.take_pending(a + 1, pc)?;
        let step = self.take_pending(a + 2, pc)?;
        let var = match self.local_name(a + 3, pc + 1) {
            Some(n) => n.to_owned(),
            None => return self.err(pc, "numeric for variable name missing"),
        };
        let body = self.block(pc + 1, target, Some(target + 1))?;
        let step = if step == Expr::Num(1.0) {
            None
        } else {
            Some(step)
        };
        out.push(Stmt::NumFor(var, init, limit, step, body));
        Ok(target + 1)
    }

    fn gen_for(&mut self, pc: usize, t: usize, out: &mut Vec<Stmt>) -> Result<usize> {
        let tf = self.p.code[t];
        let (a, n) = (a_of(tf), c_of(tf));
        let mut ex = Vec::new();
        for r in a..a + 3 {
            match self.regs[r].take() {
                Some(Expr::Cont) | None => {}
                Some(e) => ex.push(e),
            }
        }
        if ex.is_empty() {
            return self.err(pc, "generic for without expression list");
        }
        let mut vars = Vec::new();
        for k in 0..n {
            match self.local_name(a + 3 + k, pc + 1) {
                Some(v) => vars.push(v.to_owned()),
                None => return self.err(pc, "generic for variable name missing"),
            }
        }
        if t + 1 >= self.p.code.len() || op_of(self.p.code[t + 1]) != OP_JMP {
            return self.err(t, "generic for without back jump");
        }
        let body = self.block(pc + 1, t, Some(t + 2))?;
        out.push(Stmt::GenFor(vars, ex, body));
        Ok(t + 2)
    }

    fn take_pending(&mut self, r: usize, pc: usize) -> Result<Expr> {
        match self.regs[r].take() {
            Some(Expr::Cont) | None => self.err(pc, &format!("register {r} has no pending value")),
            Some(e) => Ok(e),
        }
    }

    /// Collects call arguments `regs[from..from+n]`, or every contiguous pending register for `n == None`.
    fn args(&mut self, from: usize, n: Option<usize>, pc: usize) -> Result<Vec<Expr>> {
        let mut v = Vec::new();
        match n {
            Some(n) => {
                for r in from..from + n {
                    v.push(self.read(r, pc)?);
                }
            }
            None => {
                let mut r = from;
                while r < self.regs.len()
                    && self.regs[r].is_some()
                    && self.local_name(r, pc).is_none()
                {
                    v.push(self.regs[r].take().unwrap());
                    r += 1;
                }
            }
        }
        Ok(v)
    }

    fn call_expr(&mut self, pc: usize, multi: bool) -> Result<Expr> {
        let i = self.p.code[pc];
        let (a, b) = (a_of(i), b_of(i));
        let f = self.take_pending(a, pc)?;
        match f {
            Expr::SelfRef(obj, name) => {
                self.regs[a + 1] = None; // Cont placeholder
                let args = if b == 0 {
                    self.args(a + 2, None, pc)?
                } else {
                    self.args(a + 2, Some(b - 2), pc)?
                };
                Ok(Expr::Method {
                    obj,
                    name,
                    args,
                    multi,
                })
            }
            f => {
                let args = if b == 0 {
                    self.args(a + 1, None, pc)?
                } else {
                    self.args(a + 1, Some(b - 1), pc)?
                };
                Ok(Expr::Call {
                    f: Box::new(f),
                    args,
                    multi,
                })
            }
        }
    }

    fn ret(&mut self, pc: usize, out: &mut Vec<Stmt>) -> Result<usize> {
        let i = self.p.code[pc];
        let (a, b) = (a_of(i), b_of(i));
        if op_of(i) == OP_TAILCALL {
            let call = self.call_expr(pc, true)?;
            out.push(Stmt::Return(vec![call]));
            if pc + 1 < self.p.code.len() && op_of(self.p.code[pc + 1]) == OP_RETURN {
                return Ok(pc + 2);
            }
            return Ok(pc + 1);
        }
        let vals = match b {
            1 => Vec::new(),
            0 => self.args(a, None, pc)?,
            n => self.args(a, Some(n - 1), pc)?,
        };
        out.push(Stmt::Return(vals));
        Ok(pc + 1)
    }

    /// Processes one non-structural instruction.
    fn step(&mut self, pc: usize, out: &mut Vec<Stmt>) -> Result<usize> {
        let i = self.p.code[pc];
        let (op, a, b, c, bx) = (op_of(i), a_of(i), b_of(i), c_of(i), bx_of(i));
        let mut next = pc + 1;
        match op {
            OP_MOVE => {
                let v = self.read(b, pc)?;
                self.write(a, pc, v, out);
            }
            OP_LOADK => {
                let k = self.konst(bx, pc)?;
                self.write(a, pc, k, out);
            }
            OP_LOADBOOL => {
                if c != 0 {
                    return self.err(pc, "unsupported conditional LOADBOOL");
                }
                self.write(a, pc, if b != 0 { Expr::True } else { Expr::False }, out);
            }
            OP_LOADNIL => {
                for r in a..=b {
                    self.write(r, pc, Expr::Nil, out);
                }
            }
            OP_GETUPVAL => {
                let n = self.upname(b, pc)?;
                self.write(a, pc, Expr::Name(n), out);
            }
            OP_GETGLOBAL => match self.konst(bx, pc)? {
                Expr::Str(s) => {
                    let n = String::from_utf8_lossy(&s).into_owned();
                    self.write(a, pc, Expr::Name(n), out);
                }
                _ => return self.err(pc, "global name is not a string"),
            },
            OP_GETTABLE => {
                let o = self.read(b, pc)?;
                let k = self.rk(c, pc)?;
                self.write(a, pc, Expr::Index(Box::new(o), Box::new(k)), out);
            }
            OP_SETGLOBAL => {
                let v = self.read(a, pc)?;
                match self.konst(bx, pc)? {
                    Expr::Str(s) => out.push(Stmt::Assign(
                        Expr::Name(String::from_utf8_lossy(&s).into_owned()),
                        v,
                    )),
                    _ => return self.err(pc, "global name is not a string"),
                }
            }
            OP_SETUPVAL => {
                let v = self.read(a, pc)?;
                let n = self.upname(b, pc)?;
                out.push(Stmt::Assign(Expr::Name(n), v));
            }
            OP_SETTABLE => {
                let building =
                    a >= self.nactive(pc) && matches!(self.regs[a], Some(Expr::Table(_)));
                let k = self.rk(b, pc)?;
                let v = self.rk(c, pc)?;
                if building {
                    if let Some(Expr::Table(f)) = &mut self.regs[a] {
                        f.push((Some(k), v));
                    }
                } else {
                    let o = self.read(a, pc)?;
                    out.push(Stmt::Assign(Expr::Index(Box::new(o), Box::new(k)), v));
                }
            }
            OP_NEWTABLE => self.write(a, pc, Expr::Table(Vec::new()), out),
            OP_SELF => {
                let o = self.read(b, pc)?;
                let name = match self.rk(c, pc)? {
                    Expr::Str(s) if is_ident(&s) => String::from_utf8_lossy(&s).into_owned(),
                    _ => return self.err(pc, "method name is not an identifier"),
                };
                self.regs[a] = Some(Expr::SelfRef(Box::new(o), name));
                self.regs[a + 1] = Some(Expr::Cont);
            }
            OP_ADD..=OP_POW => {
                let l = self.rk(b, pc)?;
                let r = self.rk(c, pc)?;
                let name = ["+", "-", "*", "/", "%", "^"][(op - OP_ADD) as usize];
                self.write(a, pc, Expr::Bin(name, Box::new(l), Box::new(r)), out);
            }
            OP_UNM | OP_NOT | OP_LEN => {
                let v = self.read(b, pc)?;
                let name = match op {
                    OP_UNM => "-",
                    OP_NOT => "not",
                    _ => "#",
                };
                self.write(a, pc, Expr::Un(name, Box::new(v)), out);
            }
            OP_CONCAT => {
                let mut parts = Vec::new();
                for r in b..=c {
                    parts.push(self.read(r, pc)?);
                }
                let mut acc = parts.pop().unwrap();
                while let Some(p) = parts.pop() {
                    acc = Expr::Bin("..", Box::new(p), Box::new(acc));
                }
                self.write(a, pc, acc, out);
            }
            OP_CALL => {
                let call = self.call_expr(pc, c == 0)?;
                match c {
                    1 => out.push(Stmt::Call(call)),
                    0 => self.regs[a] = Some(call),
                    n => {
                        self.write(a, pc, call, out);
                        for k in 1..n - 1 {
                            self.regs[a + k] = Some(Expr::Cont);
                        }
                    }
                }
            }
            OP_CLOSE => {}
            OP_CLOSURE => {
                let Some(child) = self.p.protos.get(bx) else {
                    return self.err(pc, "closure index out of range");
                };
                let f = decompile_func(child, format!("{}.{}", self.path, bx))?;
                next = pc + 1 + child.nups;
                let declared = self.local_name(a, pc).filter(|_| {
                    self.p
                        .locvars
                        .iter()
                        .any(|l| l.start == pc && !l.name.starts_with('('))
                });
                if let Some(n) = declared {
                    out.push(Stmt::LocalFunc(n.to_owned(), f));
                } else {
                    self.write(a, pc, Expr::Func(Box::new(f)), out);
                }
            }
            OP_VARARG => {
                if b == 0 {
                    self.regs[a] = Some(Expr::Vararg);
                } else {
                    self.write(a, pc, Expr::Vararg, out);
                    for k in 1..b - 1 {
                        self.regs[a + k] = Some(Expr::Cont);
                    }
                }
            }
            OP_SETLIST => {
                if c == 0 {
                    return self.err(pc, "SETLIST with extended batch");
                }
                let vals = if b == 0 {
                    self.args(a + 1, None, pc)?
                } else {
                    self.args(a + 1, Some(b), pc)?
                };
                match &mut self.regs[a] {
                    Some(Expr::Table(f)) => f.extend(vals.into_iter().map(|v| (None, v))),
                    _ => return self.err(pc, "SETLIST on a value that is not a pending table"),
                }
            }
            _ => return self.err(pc, &format!("unsupported opcode {op}")),
        }
        self.decl(next - 1, out)?;
        Ok(next)
    }
}

fn res(items: &[Item], i: usize, labels: &[(usize, bool)], fall: bool) -> Result<V> {
    let nt = if i + 1 == items.len() {
        if fall { V::T } else { V::F }
    } else {
        res(items, i + 1, labels, fall)?
    };
    let t = items[i].target;
    let tk = if let Some(&(_, v)) = labels.iter().find(|(pc, _)| *pc == t) {
        if v { V::T } else { V::F }
    } else {
        match (i + 1..items.len()).find(|&j| items[j].start == t) {
            Some(j) => res(items, j, labels, fall)?,
            None => bail!(
                "jump at instruction {} leaves the condition at {t}",
                items[i].pc
            ),
        }
    };
    let v = items[i].taken.clone();
    Ok(match (tk, nt) {
        (V::T, V::T) | (V::F, V::F) => bail!("constant condition at instruction {}", items[i].pc),
        (V::T, V::F) => V::E(v),
        (V::F, V::T) => V::E(neg(v)),
        (V::T, V::E(x)) => V::E(or(v, x)),
        (V::F, V::E(x)) => V::E(and(neg(v), x)),
        (V::E(y), V::F) => V::E(and(v, y)),
        (V::E(y), V::T) => V::E(or(neg(v), y)),
        (V::E(y), V::E(x)) => {
            if let Expr::Bin("and", l, r) = &x {
                if **r == y {
                    return Ok(V::E(and(or(v, (**l).clone()), y)));
                }
            }
            bail!("unsupported condition shape at instruction {}", items[i].pc)
        }
    })
}

fn decompile_func(p: &Proto, path: String) -> Result<FuncAst> {
    let n = p.code.len();
    if n == 0 || op_of(p.code[n - 1]) != OP_RETURN {
        bail!("{path}: function does not end with RETURN");
    }
    let params: Vec<String> = p
        .locvars
        .iter()
        .take(p.num_params)
        .map(|l| l.name.clone())
        .collect();
    if params.len() != p.num_params {
        bail!("{path}: parameter names missing from the debug information");
    }
    let mut d = Dec::new(p, path)?;
    let body = d.block(0, n - 1, None)?;
    Ok(FuncAst {
        params,
        vararg: p.is_vararg & 2 != 0,
        body,
    })
}

/// Source text of a compiled chunk, one entry per line, 2-space indentation, no trailing newline.
pub(super) fn decompile(p: &Proto, name: &str) -> Result<Vec<String>> {
    let f = decompile_func(p, name.to_owned())?;
    let mut lines = Vec::new();
    stmts(&mut lines, &f.body, 0);
    Ok(lines
        .into_iter()
        .flat_map(|l| l.split('\n').map(str::to_owned).collect::<Vec<_>>())
        .collect())
}
