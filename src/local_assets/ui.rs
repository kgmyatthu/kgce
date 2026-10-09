//! Local rebuild of the world campaign front end and campaign HUD repairs (`ui/**`).
//!
//! Inputs: the user's Napoleon: Total War UI files from the verified snapshot plus the authored recipe in
//! `<assets>/policy/ui` (`ui.json` and `scripts/*.patch`). Nothing from the game is stored in the program or in
//! the recipe:
//!
//! * layouts (`main`, `spain_main`, `grand_campaign`) are parsed, edited structurally (new menu button,
//!   three theatre previews, 18-cell faction grid) and written back in the same binary format;
//! * Lua scripts are decompiled from the shipped Lua 5.1 bytecode and the authored text patches are applied.
//!   They are written as plain Lua source under the original `.luac` names, which the game's loader accepts.

mod layout;
mod lua;

use crate::model::Progress;
use crate::snapshot::{Game, Snapshot};
use anyhow::{Context, Result, bail};
use layout::{IdAlloc, Layout, utf16};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct Policy {
    version: u32,
    world_layout: String,
    source_world_layout: String,
    button: Button,
    main_menu: MainMenu,
    world: World,
    scripts: Vec<ScriptSpec>,
}

#[derive(Deserialize)]
struct Button {
    title: String,
    text: String,
    loc_key: String,
    heading_loc_key: String,
}

#[derive(Deserialize)]
struct MainMenu {
    layouts: Vec<String>,
    id_start: i32,
    pitch: i32,
    allowed_stock_pitch: Vec<i32>,
}

#[derive(Deserialize)]
struct World {
    id_start: i32,
    templates: Templates,
    maps: Maps,
    factions: Factions,
}

#[derive(Deserialize)]
struct Templates {
    root: String,
    main_panel: String,
    faction_selection: String,
    advanced_options: String,
}

#[derive(Deserialize)]
struct Maps {
    container_y: i32,
    prototype: String,
    previews: Vec<Preview>,
}

#[derive(Deserialize)]
struct Preview {
    component: String,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

#[derive(Deserialize)]
struct Factions {
    area_x: i32,
    area_y: i32,
    area_w: i32,
    area_h: i32,
    columns: i32,
    rows: i32,
    pitch_x: i32,
    pitch_y: i32,
    scale: f64,
}

#[derive(Deserialize)]
struct ScriptSpec {
    source: String,
    output: String,
    patch: String,
}

const FRONTEND: &str = "ui/frontend ui/";

/// Writes `ui/**` below `output_assets`.
pub(super) fn generate(
    snapshot: &Snapshot,
    shipped_policy: &Path,
    output_assets: &Path,
    progress: &Progress,
) -> Result<()> {
    let policy_dir = shipped_policy.join("policy").join("ui");
    let text = fs::read_to_string(policy_dir.join("ui.json"))
        .with_context(|| format!("read {}", policy_dir.join("ui.json").display()))?;
    let policy: Policy = serde_json::from_str(&text).context("parse policy/ui/ui.json")?;
    if policy.version != 1 {
        bail!(
            "policy/ui/ui.json version {} is not supported",
            policy.version
        );
    }
    let snap = snapshot.original();
    let read = |p: &str| -> Result<Vec<u8>> {
        snap.read(Game::Ntw, p)
            .with_context(|| format!("Napoleon original {p}"))
    };

    let out_ui = output_assets.join("ui");
    if out_ui.exists() {
        fs::remove_dir_all(&out_ui).with_context(|| format!("clear {}", out_ui.display()))?;
    }
    let mut written: Vec<(String, Vec<u8>)> = Vec::new();

    for name in &policy.main_menu.layouts {
        progress(&format!("UI: main menu layout {name}"));
        let path = format!("{FRONTEND}{name}");
        let src = read(&path)?;
        let mut layout = parse_checked(&src, &path)?;
        add_menu_button(&mut layout, &policy, name)?;
        layout.validate(name)?;
        written.push((path, layout.write()));
    }

    progress("UI: world campaign selection layout");
    let src_path = format!("{FRONTEND}{}", policy.source_world_layout);
    let src = read(&src_path)?;
    let mut layout = parse_checked(&src, &src_path)?;
    build_world_layout(&mut layout, &policy)?;
    layout.validate(&policy.world_layout)?;
    written.push((format!("{FRONTEND}{}", policy.world_layout), layout.write()));

    for spec in &policy.scripts {
        progress(&format!("UI: script {}", spec.output));
        let bytes = read(&spec.source)?;
        crate::snapshot::validate_rel(&spec.patch)
            .context("UI patch must be relative to policy/ui/scripts")?;
        let patch_path = policy_dir.join("scripts").join(&spec.patch);
        let patch = fs::read_to_string(&patch_path)
            .with_context(|| format!("read {}", patch_path.display()))?;
        let text = build_script(&bytes, &patch, &spec.source, &spec.patch)?;
        written.push((spec.output.clone(), text.into_bytes()));
    }

    for (rel, bytes) in written {
        crate::snapshot::validate_rel(&rel)
            .context("UI output must stay inside the local asset tree")?;
        if !rel.starts_with("ui/") {
            bail!("UI recipe output must be below ui/: {rel}");
        }
        let dest = output_assets.join(&rel);
        if let Some(dir) = dest.parent() {
            fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        fs::write(&dest, bytes).with_context(|| format!("write {}", dest.display()))?;
    }
    Ok(())
}

/// Parses a layout and proves the writer reproduces the original bytes before anything is edited.
fn parse_checked(src: &[u8], path: &str) -> Result<Layout> {
    let layout = Layout::parse(src).with_context(|| format!("layout {path}"))?;
    if layout.write() != src {
        bail!(
            "layout {path} does not round-trip through the layout writer; this game version is not supported"
        );
    }
    Ok(layout)
}

// ------------------------------------------------------------------------------------------------ layouts

/// Adds the world campaign button as the last entry of the single-player list of a main menu layout.
fn add_menu_button(layout: &mut Layout, p: &Policy, name: &str) -> Result<()> {
    let mut ids = IdAlloc::new(layout, p.main_menu.id_start);
    let pitch = p.main_menu.pitch;
    let top = layout
        .root
        .children
        .first_mut()
        .with_context(|| format!("layout {name} has no menu entry"))?;
    let group = top.child_mut("single_player_expanded")?;
    let sp = group.child_mut("single_player")?;
    let stock = sp.child("sp_battle")?.y - sp.child("grand_campaign")?.y;
    if !p.main_menu.allowed_stock_pitch.contains(&stock) {
        bail!("layout {name}: unexpected single-player button pitch {stock}");
    }
    let last = sp.child("sp_battle")?;
    let last_y = last.y;
    let mut button = ids.clone_entry(last);
    button.title = p.button.title.clone();
    button.parent_name = "copy of sp_battle".to_owned();
    button.y = last_y + pitch;
    let text = button.child_mut("button_txt")?;
    let st = text
        .states
        .first_mut()
        .with_context(|| format!("layout {name}: button text has no state"))?;
    st.state_text = utf16(&p.button.text);
    st.localisation_id = utf16(&p.button.loc_key);
    let idx = sp
        .children
        .iter()
        .position(|c| c.title == "sp_battle")
        .unwrap();
    sp.children.insert(idx + 1, button);
    // everything below the list moves down by one row
    sp.child_mut("flourishes")?.y += pitch;
    for t in ["multiplayer", "options"] {
        group.child_mut(t)?.y += pitch;
    }
    Ok(())
}

fn build_world_layout(layout: &mut Layout, p: &Policy) -> Result<()> {
    let mut ids = IdAlloc::new(layout, p.world.id_start);
    let t = &p.world.templates;
    let gc = layout.root.child_mut(&p.source_world_layout)?;
    gc.title = p.world_layout.clone();
    gc.template = t.root.clone();

    let heading = gc.find_mut("heading_txt")?;
    let hs = heading
        .states
        .first_mut()
        .context("heading_txt has no state")?;
    hs.state_text = utf16(&p.button.text);
    hs.localisation_id = utf16(&p.button.heading_loc_key);

    // the forward button looks its owner up by name
    let fwd = gc.find_mut("button_forward")?;
    let from = format!("Parent(\"{}\")", p.source_world_layout);
    if !fwd.script.contains(&from) {
        bail!(
            "button_forward script does not refer to the layout by name; this game version is not supported"
        );
    }
    fwd.script = fwd
        .script
        .replace(&from, &format!("Parent(\"{}\")", p.world_layout));

    let main_panel = gc.child_mut("main_panel")?;
    main_panel.template = t.main_panel.clone();
    main_panel.child_mut("advanced_options")?.template = t.advanced_options.clone();

    // theatre previews replace the single europe map
    let maps = main_panel.child_mut("maps")?;
    let proto = maps.child(&p.world.maps.prototype)?.clone();
    let idx = maps
        .children
        .iter()
        .position(|c| c.title == p.world.maps.prototype)
        .unwrap();
    maps.children.remove(idx);
    for (n, m) in p.world.maps.previews.iter().enumerate() {
        let mut c = ids.clone_entry(&proto);
        c.title = m.component.clone();
        c.parent_name = m.component.clone();
        c.x = m.x;
        c.y = m.y;
        c.set_size(m.w, m.h, None, None)?;
        c.child_mut("map_overlay")?.set_size(m.w, m.h, None, None)?;
        let edges = c.child_mut("scroll_edges")?;
        edges.set_size(10, m.h + 10, Some(10), Some(m.h + 9))?;
        let right = edges.child_mut("copy of scroll_edge_L")?;
        right.x = m.w + 5;
        right.set_size(10, m.h + 9, None, None)?;
        maps.children.insert(idx + n, c);
    }
    maps.y = p.world.maps.container_y;

    // faction grid
    let f = &p.world.factions;
    let fs = main_panel.child_mut("faction_selection")?;
    fs.template = t.faction_selection.clone();
    fs.x = f.area_x;
    fs.y = f.area_y;
    fs.set_size(f.area_w, f.area_h, None, None)?;
    if fs.children.len() < 2
        || fs.children[0].title != "placeholder_faction_1"
        || fs.children[1].title != "placeholder_faction_2"
    {
        bail!("faction_selection placeholders not found; this game version is not supported");
    }
    let empty_proto = ids.clone_entry(&fs.children[1]);
    fs.children.truncate(1);
    {
        let ph1 = &mut fs.children[0];
        ph1.child_mut("faction_button")?.scale(f.scale);
        ph1.scale(f.scale);
    }
    for i in 1..=(f.columns * f.rows) {
        let (row, col) = ((i - 1) / f.columns, (i - 1) % f.columns);
        if i == 1 {
            let ph = &mut fs.children[0];
            ph.x = col * f.pitch_x;
            ph.y = row * f.pitch_y;
        } else {
            let mut ph = ids.clone_entry(&empty_proto);
            ph.title = format!("placeholder_faction_{i}");
            ph.parent_name = ph.title.clone();
            ph.scale(f.scale);
            ph.x = col * f.pitch_x;
            ph.y = row * f.pitch_y;
            fs.children.push(ph);
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ scripts

fn build_script(bytecode: &[u8], patch: &str, source: &str, patch_name: &str) -> Result<String> {
    let proto = lua::parse(bytecode).with_context(|| format!("compiled script {source}"))?;
    let lines = lua::decompile(&proto, source).with_context(|| format!("decompile {source}"))?;
    let ops = parse_patch(patch, patch_name)?;
    let lines =
        apply_patch(lines, &ops).with_context(|| format!("patch {patch_name} on {source}"))?;
    check_blocks(&lines).with_context(|| format!("patched {source}"))?;
    let mut out = lines.join("\n");
    out.push('\n');
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    After,
    Before,
    Replace,
    Through,
    Function,
}

struct Op {
    kind: Kind,
    scope: Option<String>,
    function: Option<String>,
    anchors: Vec<String>,
    end_anchors: Vec<String>,
    body: Option<Vec<String>>,
    header_line: usize,
}

fn parse_patch(text: &str, name: &str) -> Result<Vec<Op>> {
    let mut ops: Vec<Op> = Vec::new();
    let mut in_body = false;
    let mut in_end = false;
    for (n, raw) in text.lines().enumerate() {
        let n = n + 1;
        let line = raw.trim_end_matches('\r');
        if let Some(h) = line.strip_prefix("@@") {
            let h = h.trim();
            if h == "body" {
                let Some(op) = ops.last_mut() else {
                    bail!("{name}:{n}: body before any operation")
                };
                if op.body.is_some() {
                    bail!(
                        "{name}:{n}: operation at line {} has two bodies",
                        op.header_line
                    );
                }
                op.body = Some(Vec::new());
                in_body = true;
                continue;
            }
            let mut words = h.split_whitespace();
            let kind = match words.next() {
                Some("after") => Kind::After,
                Some("before") => Kind::Before,
                Some("replace") => Kind::Replace,
                Some("through") => Kind::Through,
                Some("function") => Kind::Function,
                other => bail!("{name}:{n}: unknown operation {other:?}"),
            };
            let (mut scope, mut function) = (None, None);
            let rest: Vec<&str> = words.collect();
            match (kind, rest.as_slice()) {
                (Kind::Function, [f]) => function = Some((*f).to_owned()),
                (Kind::Function, _) => {
                    bail!("{name}:{n}: `function` needs exactly one function name")
                }
                (_, []) => {}
                (_, ["in", f]) => scope = Some((*f).to_owned()),
                _ => bail!("{name}:{n}: malformed operation header `{h}`"),
            }
            ops.push(Op {
                kind,
                scope,
                function,
                anchors: Vec::new(),
                end_anchors: Vec::new(),
                body: None,
                header_line: n,
            });
            in_body = false;
            in_end = false;
            continue;
        }
        if in_body {
            ops.last_mut()
                .unwrap()
                .body
                .as_mut()
                .unwrap()
                .push(line.to_owned());
            continue;
        }
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(op) = ops.last_mut() else {
            bail!("{name}:{n}: text before the first operation")
        };
        let Some(a) = line.strip_prefix('=') else {
            bail!("{name}:{n}: expected an anchor line starting with `=`")
        };
        let a = a.trim().to_owned();
        if a == ".." {
            if op.kind != Kind::Through || in_end || op.anchors.is_empty() {
                bail!(
                    "{name}:{n}: `..` is only valid inside a `through` operation after its first anchor"
                );
            }
            in_end = true;
        } else if in_end {
            op.end_anchors.push(a);
        } else {
            op.anchors.push(a);
        }
    }
    for op in &mut ops {
        if let Some(b) = &mut op.body {
            while b.last().is_some_and(|l| l.trim().is_empty()) {
                b.pop();
            }
        }
        let ok = match op.kind {
            Kind::Function => op.function.is_some(),
            Kind::Through => !op.anchors.is_empty() && !op.end_anchors.is_empty(),
            _ => !op.anchors.is_empty(),
        };
        if !ok {
            bail!("{name}:{}: operation lacks its anchors", op.header_line);
        }
        if matches!(op.kind, Kind::After | Kind::Before) && op.body.is_none() {
            bail!("{name}:{}: insert operation without a body", op.header_line);
        }
    }
    Ok(ops)
}

fn indent_of(l: &str) -> usize {
    l.len() - l.trim_start().len()
}

/// Line range `[start, end]` of the top-level function `name` (its `function` line to its closing `end`).
fn function_range(lines: &[String], name: &str) -> Result<(usize, usize)> {
    let head = format!("function {name}(");
    let mut found = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with(&head));
    let Some((s, _)) = found.next() else {
        bail!("function {name} not found")
    };
    if found.next().is_some() {
        bail!("function {name} is defined more than once");
    }
    let e = lines[s + 1..]
        .iter()
        .position(|l| l == "end")
        .with_context(|| format!("function {name} has no end"))?;
    Ok((s, s + 1 + e))
}

fn match_at(lines: &[String], i: usize, anchors: &[String]) -> bool {
    i + anchors.len() <= lines.len()
        && anchors
            .iter()
            .enumerate()
            .all(|(k, a)| lines[i + k].trim() == a)
}

fn apply_patch(mut lines: Vec<String>, ops: &[Op]) -> Result<Vec<String>> {
    for op in ops {
        let at = format!("operation at line {}", op.header_line);
        let body = |ind: usize| -> Vec<String> {
            op.body
                .as_ref()
                .map(|b| {
                    b.iter()
                        .map(|l| {
                            if l.is_empty() {
                                String::new()
                            } else {
                                format!("{}{l}", " ".repeat(ind))
                            }
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        if op.kind == Kind::Function {
            let (s, e) = function_range(&lines, op.function.as_deref().unwrap())
                .with_context(|| at.clone())?;
            lines.splice(s..=e, body(0));
            continue;
        }
        let (lo, hi) = match &op.scope {
            Some(f) => function_range(&lines, f).with_context(|| at.clone())?,
            None => (0, lines.len().saturating_sub(1)),
        };
        let hits: Vec<usize> = (lo..=hi)
            .filter(|&i| match_at(&lines, i, &op.anchors))
            .collect();
        if hits.len() != 1 {
            bail!(
                "{at}: anchor `{}` matches {} times (exactly one required)",
                op.anchors[0],
                hits.len()
            );
        }
        let s = hits[0];
        let ind = indent_of(&lines[s]);
        let n = op.anchors.len();
        match op.kind {
            Kind::After => {
                let b = body(ind);
                lines.splice(s + n..s + n, b);
            }
            Kind::Before => {
                let b = body(ind);
                lines.splice(s..s, b);
            }
            Kind::Replace => {
                let b = body(ind);
                lines.splice(s..s + n, b);
            }
            Kind::Through => {
                let m = op.end_anchors.len();
                let end = (s + n..=hi).find(|&j| match_at(&lines, j, &op.end_anchors));
                let Some(j) = end else {
                    bail!(
                        "{at}: end anchor `{}` not found after the start anchor",
                        op.end_anchors[0]
                    )
                };
                let b = body(ind);
                lines.splice(s..j + m, b);
            }
            Kind::Function => unreachable!(),
        }
    }
    Ok(lines)
}

/// Cheap syntax sanity check of the patched text: `function`/`if`/`do`/`repeat` blocks balance with `end`/`until`
/// and the text never closes a block that is not open. Strings and comments are skipped.
fn check_blocks(lines: &[String]) -> Result<()> {
    let mut depth: i64 = 0;
    for (n, l) in lines.iter().enumerate() {
        let b = l.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c == b'-' && b.get(i + 1) == Some(&b'-') {
                break;
            }
            if c == b'"' || c == b'\'' {
                i += 1;
                while i < b.len() && b[i] != c {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
                continue;
            }
            if c.is_ascii_alphabetic() || c == b'_' {
                let s = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                match &l[s..i] {
                    "function" | "if" | "do" | "repeat" => depth += 1,
                    "end" | "until" => depth -= 1,
                    _ => {}
                }
                if depth < 0 {
                    bail!("line {}: block closed that was never opened", n + 1);
                }
                continue;
            }
            i += 1;
        }
    }
    if depth != 0 {
        bail!("{depth} unclosed block(s)");
    }
    Ok(())
}
