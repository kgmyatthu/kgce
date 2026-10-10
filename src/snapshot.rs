//! Immutable, manifest-verified snapshot of the original ETW and NTW data the converter needs.
//!
//! `import` copies the relevant originals byte-for-byte out of both installed games (loose files and
//! pack members, with the same precedence as the game: loose files beat packs, patch packs beat release
//! packs, later pack names win; NTW only reads its vanilla packs, never mod packs such as nap_world.pack)
//! into `<destination>/etw/…` and `<destination>/ntw/…`, hashing the source stream and the written copy
//! and refusing any mismatch. Every file is recorded in `manifest.json` (SHA-256, size, origin game /
//! path / pack / member). Afterwards the tool runs from the snapshot alone: `Snapshot` never touches the
//! installations, never writes, and refuses files whose size or hash no longer matches the manifest.
//!
//! Paths use `/` separators and are looked up case-insensitively (`\` is accepted); `names()` returns the
//! canonical lower-case `/` form.

use crate::model::Progress;
use crate::pack::{self, Pack};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::ops::Bound::{Included, Unbounded};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use walkdir::WalkDir;

const FORMAT: &str = "kgc-snapshot";
const VERSION: u32 = 1;
const MANIFEST: &str = "manifest.json";
const INDEX_DIR: &str = "source_index";
pub(crate) const BUF: usize = 1 << 20;

/// NTW packs that make up the vanilla game. Anything else in the folder (nap_world.pack, test packs…) is a mod.
pub(crate) const NTW_VANILLA_PACKS: [&str; 11] = [
    "battleterrain.pack",
    "boot.pack",
    "buildings.pack",
    "data.pack",
    "local_en.pack",
    "local_en_patch.pack",
    "media.pack",
    "rigidmodels.pack",
    "sound.pack",
    "variantmodels.pack",
    "variantmodels2.pack",
];
/// Files that earlier editors captured into snapshots (the campaign-building shaders now ship with the
/// editor, see `building_scale`). New imports no longer select them; a snapshot that still holds them stays
/// valid and re-importing it is not refused for lacking them in the selection.
const RETIRED_NTW_SOURCES: [&str; 2] = [
    "fx/campaignbuilding.fx",
    "fx/campaignbuilding_vertex.fx_fragment",
];
const NTW_REF_MAPS: [&str; 5] = [
    "nap_europe",
    "nap_egypt",
    "nap_italy",
    "nap_spain",
    "nap_tut",
];
/// Our own output / backups living inside the NTW folder: never part of the source index either.
const NTW_MOD_DIRS: [&str; 2] = ["campaign_maps/nap_world/", "campaigns/wld_napoleon/"];

const ETW_REQUIRED: [&str; 8] = [
    "campaigns/main/startpos.esf",
    "campaign_maps/global_map/regions.esf",
    "text/localisation.loc",
    "world_.stpi",
    "world_.stpd",
    "heightmaps/default.tga",
    "battleterrain/tiles/metadata.dat",
    "campaign/terrain/campaign_detail.dds",
];
const NTW_REQUIRED: [&str; 4] = [
    "campaigns/eur_napoleon/startpos.esf",
    "campaign_maps/nap_europe/regions.esf",
    "text/localisation.loc",
    "text/ui.loc",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Game {
    Etw,
    Ntw,
}

impl Game {
    pub const ALL: [Game; 2] = [Game::Etw, Game::Ntw];

    /// Folder name inside the snapshot.
    pub fn dir(self) -> &'static str {
        match self {
            Game::Etw => "etw",
            Game::Ntw => "ntw",
        }
    }

    fn slot(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Origin {
    /// A loose file below the game's data folder (`file` is the relative path as stored on disk).
    Loose { file: String },
    /// A member of `file` (pack file name); `member` is the name as stored in the pack.
    Pack { file: String, member: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileEntry {
    pub game: Game,
    /// Canonical lower-case `/` path, also the location below `<root>/<game>/`.
    pub path: String,
    /// The name with its original capitalisation.
    pub stored: String,
    pub size: u64,
    pub sha256: String,
    pub origin: Origin,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackInfo {
    pub file: String,
    pub kind: u32,
    pub size: u64,
    pub members: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Where the data was read from (provenance only; nothing reads it again).
    pub data_dir: String,
    pub packs: Vec<PackInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexInfo {
    pub game: Game,
    pub size: u64,
    pub sha256: String,
    pub entries: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    /// Language of the NTW localisation packs this snapshot holds (snapshots made before language
    /// support only ever held English).
    #[serde(default = "english")]
    pub language: String,
    pub sources: BTreeMap<String, SourceInfo>,
    pub indexes: Vec<IndexInfo>,
    pub files: Vec<FileEntry>,
}

#[derive(Debug)]
struct Inner {
    manifest: Manifest,
    by_key: HashMap<(Game, String), usize>,
    names: [Vec<String>; 2],
    verified: Mutex<HashSet<usize>>,
    source: [Vec<String>; 2],
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub root: PathBuf,
    inner: Arc<Inner>,
    overlay: Option<Arc<Overlay>>,
}

/// Shared immutable payload of an in-memory overlay member.
pub type Bytes = Arc<Vec<u8>>;

/// One in-memory member handed to `derive_ntw`.
pub(crate) struct OvInput {
    pub entry: FileEntry,
    pub bytes: Bytes,
    pub priority: usize,
    pub origin: String,
}

#[derive(Debug)]
struct OvFile {
    entry: FileEntry,
    bytes: Bytes,
    priority: usize,
    origin: String,
}

#[derive(Clone, Debug, Default)]
struct Overlay {
    /// Canonical NTW name -> winning member.
    files: BTreeMap<String, Arc<OvFile>>,
    /// Base (or earlier overlay) names hidden from every view.
    removed: BTreeSet<String>,
    /// Names that exist in the source but are not readable members.
    src_added: BTreeSet<String>,
    /// Highest precedence handed out so far.
    top: usize,
}

enum Loc<'a> {
    Base(usize, &'a FileEntry),
    Ov(&'a OvFile),
}

impl OvFile {
    fn read_all(&self) -> Vec<u8> {
        self.bytes.as_ref().clone()
    }
}

// ---------------------------------------------------------------------------------------------
// Names and paths
// ---------------------------------------------------------------------------------------------

/// Canonical snapshot name: `/` separators, lower case.
pub(crate) fn canon(name: &str) -> String {
    name.replace('\\', "/").to_lowercase()
}

/// Rejects anything that could leave the snapshot folder or is not a plain relative file name on Windows.
pub(crate) fn validate_rel(path: &str) -> Result<()> {
    if path.is_empty() {
        bail!("empty path");
    }
    for part in path.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            bail!("unsafe path {path:?}: empty, '.' or '..' component");
        }
        if part.ends_with('.') || part.ends_with(' ') {
            bail!("unsafe path {path:?}: component {part:?} ends with a dot or space");
        }
        if let Some(c) = part
            .chars()
            .find(|&c| (c as u32) < 0x20 || c == '\u{7f}' || "<>:\"|?*\\".contains(c))
        {
            bail!("unsafe path {path:?}: component {part:?} contains {c:?}");
        }
        let stem = part.split('.').next().unwrap_or("").to_lowercase();
        let reserved = matches!(stem.as_str(), "con" | "prn" | "aux" | "nul")
            || ((stem.starts_with("com") || stem.starts_with("lpt"))
                && stem.len() == 4
                && stem.as_bytes()[3].is_ascii_digit()
                && stem.as_bytes()[3] != b'0');
        if reserved {
            bail!("unsafe path {path:?}: {part:?} is a reserved device name");
        }
    }
    Ok(())
}

pub(crate) fn validate_sha(s: &str) -> Result<()> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        bail!("{s:?} is not a lower-case SHA-256");
    }
    Ok(())
}

fn english() -> String {
    "en".to_owned()
}

/// `local_fr.pack` / `local_fr_patch.pack` (lower-case file name) -> `fr`. Other `local_*` packs are mods.
pub(crate) fn base_language_pack(lower: &str) -> Option<String> {
    let rest = lower.strip_prefix("local_")?.strip_suffix(".pack")?;
    let code = rest.strip_suffix("_patch").unwrap_or(rest);
    (matches!(code.len(), 2 | 3) && code.bytes().all(|c| c.is_ascii_lowercase()))
        .then(|| code.to_owned())
}

/// Language recorded by the snapshot in `root` without opening its files; `None` if there is no manifest.
pub fn language_of(root: &Path) -> Result<Option<String>> {
    let mpath = root.join(MANIFEST);
    if !mpath.is_file() {
        return Ok(None);
    }
    let raw = fs::read(&mpath).with_context(|| format!("read {}", mpath.display()))?;
    let manifest: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("snapshot manifest {} is not valid", mpath.display()))?;
    Ok(Some(manifest.language))
}

fn disk_path(root: &Path, game: Game, path: &str) -> PathBuf {
    let mut p = root.join(game.dir());
    p.extend(path.split('/'));
    p
}

fn index_path(root: &Path, game: Game) -> PathBuf {
    root.join(INDEX_DIR).join(format!("{}.tsv", game.dir()))
}

/// Reads the list of every original file of one game (extracted or not), checked against the manifest.
fn load_source_index(root: &Path, manifest: &Manifest, game: Game) -> Result<Vec<String>> {
    let info = manifest
        .indexes
        .iter()
        .find(|i| i.game == game)
        .ok_or_else(|| anyhow!("manifest has no source index for {}", game.dir()))?;
    let p = index_path(root, game);
    let bytes = fs::read(&p).with_context(|| format!("read source index {}", p.display()))?;
    if bytes.len() as u64 != info.size || hex::encode(Sha256::digest(&bytes)) != info.sha256 {
        bail!("source index {} is corrupt", p.display());
    }
    let text = String::from_utf8(bytes)
        .with_context(|| format!("source index {} is not UTF-8", p.display()))?;
    let names: Vec<String> = text
        .lines()
        .map(|l| l.split('\t').next().unwrap_or("").to_owned())
        .collect();
    if names.len() as u64 != info.entries || !names.windows(2).all(|w| w[0] < w[1]) {
        bail!("source index {} is malformed", p.display());
    }
    Ok(names)
}

pub(crate) fn pump(
    r: &mut dyn Read,
    mut w: Option<&mut dyn Write>,
    h: &mut Sha256,
    buf: &mut [u8],
) -> Result<u64> {
    let mut total = 0u64;
    loop {
        let n = match r.read(buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        };
        h.update(&buf[..n]);
        if let Some(w) = w.as_mut() {
            w.write_all(&buf[..n])?;
        }
        total += n as u64;
    }
    Ok(total)
}

/// Lock that survives a panicked holder: the guarded sets are caches, so a poisoned state is still valid.
fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn hash_file(path: &Path, buf: &mut [u8]) -> Result<(u64, String)> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let n = pump(&mut f, None, &mut h, buf).with_context(|| format!("read {}", path.display()))?;
    Ok((n, hex::encode(h.finalize())))
}

// ---------------------------------------------------------------------------------------------
// What is relevant
// ---------------------------------------------------------------------------------------------

fn etw_wanted(k: &str) -> bool {
    const EXACT: [&str; 5] = [
        "campaigns/main/startpos.esf",
        "heightmaps/default.tga",
        "world_.stpi",
        "world_.stpd",
        "battleterrain/tiles/metadata.dat",
    ];
    const PREFIX: [&str; 9] = [
        "campaign_maps/global_map/",
        "db/",
        "campaign/coastline/",
        "campaign/terrain/",
        "rigidmodels/campaignborders/",
        "rigidmodels/campaignroads/",
        "rigidmodels/campaignrivers/",
        "rigidmodels/campaigntrees/",
        "rigidmodels/campaignbridges/",
    ];
    if EXACT.contains(&k)
        || PREFIX.iter().any(|p| k.starts_with(p))
        || k.starts_with("rigidmodels/campaignbuildings/campaignprops/")
    {
        return true;
    }
    k.strip_prefix("text/")
        .is_some_and(|rest| !rest.contains('/') && rest.ends_with(".loc"))
}

fn is_image(k: &str) -> bool {
    k.ends_with(".dds") || k.ends_with(".tga")
}

fn ntw_wanted(k: &str) -> bool {
    const EXACT: [&str; 5] = [
        "text/ui.loc",
        "text/localisation.loc",
        "campaigns/eur_napoleon/startpos.esf",
        "campaigns/eur_napoleon/scripting.lua",
        "battleterrain/tiles/metadata.dat",
    ];
    if EXACT.contains(&k) || k.starts_with("db/") || k.starts_with("ui/frontend ui/") {
        return true;
    }
    if let Some(rest) = k.strip_prefix("ui/campaign ui/") {
        return !is_image(rest);
    }
    // Reference maps: their logic files, and their display data except the multi-hundred-MB supertexture
    // payload; bulky images are kept from nap_europe only (header/layout references).
    if let Some(rest) = k.strip_prefix("campaign_maps/") {
        let Some((map, tail)) = rest.split_once('/') else {
            return false;
        };
        if !NTW_REF_MAPS.contains(&map) {
            return false;
        }
        return match tail.strip_prefix("display/") {
            Some(display) => {
                !display.ends_with(".stpd") && (!is_image(display) || map == "nap_europe")
            }
            None => {
                !tail.contains('/')
                    && (tail.ends_with(".esf") || tail.ends_with(".tga") || tail == "metadata.dat")
            }
        };
    }
    false
}

fn wanted(game: Game, k: &str) -> bool {
    match game {
        Game::Etw => etw_wanted(k),
        Game::Ntw => ntw_wanted(k),
    }
}

// ---------------------------------------------------------------------------------------------
// Reading the installations
// ---------------------------------------------------------------------------------------------

enum Src {
    Loose(PathBuf, String),
    Member(usize, usize),
}

struct Eff {
    stored: String,
    size: u64,
    src: Src,
}

struct Scan {
    data_dir: PathBuf,
    packs: Vec<Pack>,
    index: BTreeMap<String, Eff>,
    /// NTW only: the base language whose localisation packs were read.
    language: Option<String>,
}

fn pack_order(kind: u32) -> u32 {
    match kind {
        pack::TYPE_BOOT => 0,
        pack::TYPE_PATCH => 2,
        pack::TYPE_MOD => 3,
        _ => 1, // release and movie packs
    }
}

impl Scan {
    fn new(
        game: Game,
        data_dir: &Path,
        requested_language: Option<&str>,
        progress: &Progress,
    ) -> Result<Scan> {
        let label = game.dir().to_uppercase();
        let mut listed: Vec<(String, PathBuf)> = Vec::new();
        for e in fs::read_dir(data_dir).with_context(|| format!("list {}", data_dir.display()))? {
            let e = e?;
            let name = e.file_name().to_string_lossy().into_owned();
            let lower = name.to_lowercase();
            if !e.file_type()?.is_file() || !lower.ends_with(".pack") {
                continue;
            }
            listed.push((lower, e.path()));
        }
        // NTW localisation lives in local_<xx>.pack / local_<xx>_patch.pack; exactly one language is used.
        let mut chosen: Option<String> = None;
        if game == Game::Ntw {
            let detected: BTreeSet<String> = listed
                .iter()
                .filter_map(|(n, _)| base_language_pack(n))
                .collect();
            let list = detected.iter().cloned().collect::<Vec<_>>().join(", ");
            chosen = match requested_language {
                Some(l) => {
                    let l = l.trim().to_lowercase();
                    if !detected.contains(&l) {
                        bail!(
                            "{label} data folder {} has no local_{l}.pack language pack (found: {})",
                            data_dir.display(),
                            if list.is_empty() {
                                "none".to_owned()
                            } else {
                                list
                            }
                        );
                    }
                    Some(l)
                }
                None => match detected.len() {
                    0 => bail!(
                        "{label} data folder {} has no local_<language>.pack (for example local_en.pack); is this the Napoleon: Total War data folder?",
                        data_dir.display()
                    ),
                    1 => detected.into_iter().next(),
                    _ => bail!(
                        "{label} data folder {} holds language packs for several languages ({list}); choose which language the converter should use instead of guessing",
                        data_dir.display()
                    ),
                },
            };
        }
        let mut files: Vec<(String, PathBuf)> = Vec::new();
        for (lower, path) in listed {
            if game == Game::Ntw {
                let vanilla = NTW_VANILLA_PACKS.contains(&lower.as_str())
                    && base_language_pack(&lower).is_none();
                if !vanilla && base_language_pack(&lower) != chosen {
                    continue;
                }
            }
            files.push((lower, path));
        }
        let mut packs = Vec::with_capacity(files.len());
        for (lower, path) in files {
            progress(&format!("{label}: reading pack index {lower}"));
            packs.push((lower, Pack::open(&path)?));
        }
        packs.sort_by(|a, b| (pack_order(a.1.type_), &a.0).cmp(&(pack_order(b.1.type_), &b.0)));
        if packs.is_empty() {
            bail!(
                "{label} data folder {} contains no vanilla .pack files",
                data_dir.display()
            );
        }
        if game == Game::Ntw && !packs.iter().any(|(n, _)| n == "data.pack") {
            bail!(
                "{label} data folder {} has no data.pack; is this the Napoleon: Total War data folder?",
                data_dir.display()
            );
        }
        let packs: Vec<Pack> = packs.into_iter().map(|(_, p)| p).collect();

        let mut index: BTreeMap<String, Eff> = BTreeMap::new();
        for (pi, p) in packs.iter().enumerate() {
            for (ei, e) in p.entries.iter().enumerate() {
                index.insert(
                    canon(&e.name),
                    Eff {
                        stored: e.name.replace('\\', "/"),
                        size: e.size as u64,
                        src: Src::Member(pi, ei),
                    },
                );
            }
        }
        // Loose files beat packs.
        progress(&format!("{label}: scanning loose files"));
        let walker = WalkDir::new(data_dir)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                !(e.file_type().is_dir()
                    && e.file_name()
                        .to_string_lossy()
                        .eq_ignore_ascii_case("backup"))
            });
        for entry in walker {
            let entry = entry.with_context(|| format!("walk {}", data_dir.display()))?;
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = entry.path().strip_prefix(data_dir)?;
            let mut stored = String::new();
            for c in rel.components() {
                let Component::Normal(part) = c else {
                    bail!("unexpected path component in {}", entry.path().display())
                };
                if !stored.is_empty() {
                    stored.push('/');
                }
                stored.push_str(
                    part.to_str()
                        .ok_or_else(|| anyhow!("{} is not valid UTF-8", entry.path().display()))?,
                );
            }
            let key = stored.to_lowercase();
            if !key.contains('/') && key.ends_with(".pack") {
                continue;
            }
            if game == Game::Ntw && NTW_MOD_DIRS.iter().any(|d| key.starts_with(d)) {
                continue;
            }
            let size = entry.metadata()?.len();
            index.insert(
                key,
                Eff {
                    stored: stored.clone(),
                    size,
                    src: Src::Loose(entry.path().to_owned(), stored),
                },
            );
        }
        Ok(Scan {
            data_dir: data_dir.to_owned(),
            packs,
            index,
            language: chosen,
        })
    }

    fn open(&self, e: &Eff) -> Result<Box<dyn Read>> {
        let reader: Box<dyn Read> = match &e.src {
            Src::Loose(path, _) => {
                Box::new(File::open(path).with_context(|| format!("open {}", path.display()))?)
            }
            Src::Member(pi, ei) => {
                let p = &self.packs[*pi];
                Box::new(p.reader(&p.entries[*ei])?)
            }
        };
        Ok(reader)
    }

    fn origin(&self, e: &Eff) -> Origin {
        match &e.src {
            Src::Loose(_, rel) => Origin::Loose { file: rel.clone() },
            Src::Member(pi, ei) => {
                let p = &self.packs[*pi];
                Origin::Pack {
                    file: p
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    member: p.entries[*ei].name.clone(),
                }
            }
        }
    }

    fn source_info(&self) -> SourceInfo {
        SourceInfo {
            data_dir: self.data_dir.display().to_string(),
            packs: self
                .packs
                .iter()
                .map(|p| PackInfo {
                    file: p
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    kind: p.type_,
                    size: fs::metadata(&p.path).map(|m| m.len()).unwrap_or(0),
                    members: p.entries.len() as u64,
                })
                .collect(),
        }
    }

    /// `name\tsize\tloose|pack\tfile` for every effective vanilla file (also the ones that are not extracted).
    fn index_tsv(&self) -> Result<Vec<u8>> {
        let mut out = String::new();
        for (key, e) in &self.index {
            if key.chars().any(|c| c.is_control()) {
                bail!("file name {key:?} contains control characters");
            }
            let (kind, file) = match self.origin(e) {
                Origin::Loose { file } => ("loose", file),
                Origin::Pack { file, .. } => ("pack", file),
            };
            out.push_str(&format!("{key}\t{}\t{kind}\t{file}\n", e.size));
        }
        Ok(out.into_bytes())
    }
}

pub(crate) fn absolute(path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}

/// Canonical form of a path that may not exist yet: canonicalise the deepest existing ancestor.
pub(crate) fn canonical_lenient(path: &Path) -> Result<PathBuf> {
    let path = absolute(path)?;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = path.as_path();
    loop {
        if cur.exists() {
            let mut base = cur.canonicalize()?;
            base.extend(tail.iter().rev());
            return Ok(base);
        }
        match (cur.file_name(), cur.parent()) {
            (Some(n), Some(parent)) => {
                tail.push(n.to_owned());
                cur = parent;
            }
            _ => bail!("{} has no existing ancestor", path.display()),
        }
    }
}

fn lower_components(p: &Path) -> Vec<String> {
    p.components()
        .map(|c| c.as_os_str().to_string_lossy().to_lowercase())
        .collect()
}

pub(crate) fn is_inside(child: &Path, parent: &Path) -> bool {
    let (c, p) = (lower_components(child), lower_components(parent));
    c.len() >= p.len() && c[..p.len()] == p[..]
}

// ---------------------------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------------------------

type Plan<'a> = Vec<(Game, String, &'a Eff)>;

fn make_plan<'a>(scans: &'a [Scan; 2]) -> Result<Plan<'a>> {
    let mut plan = Vec::new();
    for game in Game::ALL {
        let scan = &scans[game.slot()];
        let before = plan.len();
        for (key, eff) in &scan.index {
            if wanted(game, key) {
                validate_rel(key).with_context(|| {
                    format!("{} source file {:?}", game.dir().to_uppercase(), eff.stored)
                })?;
                plan.push((game, key.clone(), eff));
            }
        }
        let have: HashSet<&str> = plan[before..].iter().map(|(_, k, _)| k.as_str()).collect();
        let required: &[&str] = if game == Game::Etw {
            ETW_REQUIRED.as_slice()
        } else {
            NTW_REQUIRED.as_slice()
        };
        for r in required {
            if !have.contains(r) {
                bail!(
                    "{} data folder {} lacks required file {r}",
                    game.dir().to_uppercase(),
                    scan.data_dir.display()
                );
            }
        }
        if !have.iter().any(|k| k.starts_with("db/")) {
            bail!(
                "{} data folder {} contains no db tables",
                game.dir().to_uppercase(),
                scan.data_dir.display()
            );
        }
    }
    Ok(plan)
}

/// Imports both installations. The NTW language pack is auto-detected; a data folder that holds several
/// languages is refused until the caller chooses one with `import_with_language`.
pub fn import(
    etw_data: &Path,
    ntw_data: &Path,
    destination: &Path,
    progress: &Progress,
) -> Result<Snapshot> {
    import_with_language(etw_data, ntw_data, destination, None, progress)
}

/// Like `import`, with an explicit NTW base language (`local_<language>.pack`, `local_<language>_patch.pack`).
/// The chosen language is recorded in the snapshot manifest; ETW is only ever read for its geography.
pub fn import_with_language(
    etw_data: &Path,
    ntw_data: &Path,
    destination: &Path,
    language: Option<&str>,
    progress: &Progress,
) -> Result<Snapshot> {
    // Accept the installation root as well as its data directory. Both remain read-only.
    let etw_root = etw_data.join("data");
    let ntw_root = ntw_data.join("data");
    let etw_data = if etw_root.is_dir() { etw_root.as_path() } else { etw_data };
    let ntw_data = if ntw_root.is_dir() { ntw_root.as_path() } else { ntw_data };
    crate::safety::ensure_writable_path(destination)?;
    for (label, dir) in [("ETW", etw_data), ("NTW", ntw_data)] {
        if !dir.is_dir() {
            bail!("{label} data folder {} is not a directory", dir.display());
        }
    }
    if destination
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        bail!(
            "snapshot destination {} must not contain '..'",
            destination.display()
        );
    }
    let dest = absolute(destination)?;
    let dest_real = canonical_lenient(destination)?;
    for (label, dir) in [("ETW", etw_data), ("NTW", ntw_data)] {
        let dir = dir.canonicalize()?;
        if is_inside(&dest_real, &dir) || is_inside(&dir, &dest_real) {
            bail!(
                "snapshot destination {} overlaps the {label} installation {}; the installations are never written",
                dest.display(),
                dir.display()
            );
        }
    }
    let existing = dest.join(MANIFEST).is_file();
    // An existing snapshot keeps the language it was imported with unless the caller asks for another.
    let language = match language.map(|l| l.trim().to_lowercase()) {
        Some(l) => Some(l),
        None if existing => language_of(&dest)?,
        None => None,
    };
    if !existing && dest.exists() {
        let empty = dest.is_dir() && fs::read_dir(&dest)?.next().is_none();
        if !empty {
            bail!(
                "{} exists and is not an empty folder or a snapshot; refusing to write into it",
                dest.display()
            );
        }
    }

    let scans = [
        Scan::new(Game::Etw, etw_data, None, progress)?,
        Scan::new(Game::Ntw, ntw_data, language.as_deref(), progress)?,
    ];
    let plan = make_plan(&scans)?;
    progress(&format!(
        "Source import: {} relevant files selected",
        plan.len()
    ));

    if existing {
        progress("Existing snapshot found: verifying it against the installations");
        let snap = Snapshot::open(&dest)?;
        snap.verify(progress).context("existing snapshot is damaged; refusing to overwrite it (delete the folder to import again)")?;
        if Some(snap.language()) != scans[Game::Ntw.slot()].language.as_deref() {
            bail!(
                "the existing snapshot holds the '{}' language packs but '{}' was requested; refusing to overwrite (delete the snapshot folder to import again)",
                snap.language(),
                scans[Game::Ntw.slot()].language.as_deref().unwrap_or("?")
            );
        }
        check_same_sources(&snap, &plan, &scans, progress)?;
        progress("Snapshot is up to date");
        return Ok(snap);
    }

    let parent = dest
        .parent()
        .ok_or_else(|| anyhow!("destination {} has no parent folder", dest.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let staging = parent.join(format!(".{name}.importing-{}-{stamp}", std::process::id()));
    fs::create_dir(&staging).with_context(|| format!("create {}", staging.display()))?;
    let built = build_snapshot(&staging, &plan, &scans, progress);
    match built {
        Ok(()) => {
            if dest.exists() {
                fs::remove_dir(&dest)
                    .with_context(|| format!("replace empty folder {}", dest.display()))?;
            }
            fs::rename(&staging, &dest)
                .with_context(|| format!("move {} to {}", staging.display(), dest.display()))?;
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(e);
        }
    }
    progress("Snapshot complete");
    Snapshot::open(&dest)
}

/// Streams one source file into `dest` (which must not exist), hashing the source stream and the written
/// copy and refusing any size or hash mismatch.
fn copy_member(
    scan: &Scan,
    eff: &Eff,
    game: Game,
    path: &str,
    dest: &Path,
    buf: &mut [u8],
) -> Result<FileEntry> {
    fs::create_dir_all(dest.parent().unwrap())
        .with_context(|| format!("create folder for {}", dest.display()))?;
    let mut src = scan.open(eff)?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dest)
        .with_context(|| format!("create {}", dest.display()))?;
    let mut h = Sha256::new();
    let copied = pump(&mut src, Some(&mut out as &mut dyn Write), &mut h, buf)
        .with_context(|| format!("copy {}", eff.stored))?;
    if copied != eff.size {
        bail!(
            "{} {}: read {copied} bytes but the source declares {}; the installation changed or is damaged",
            game.dir().to_uppercase(),
            eff.stored,
            eff.size
        );
    }
    out.flush()?;
    drop(out);
    let sha = hex::encode(h.finalize());
    let (copy_len, copy_sha) = hash_file(dest, buf)?;
    if copy_len != copied || copy_sha != sha {
        bail!(
            "{} {}: copy hash {copy_sha} ({copy_len} bytes) differs from source hash {sha} ({copied} bytes)",
            game.dir().to_uppercase(),
            eff.stored
        );
    }
    Ok(FileEntry {
        game,
        path: path.to_owned(),
        stored: eff.stored.clone(),
        size: copied,
        sha256: sha,
        origin: scan.origin(eff),
    })
}

fn build_snapshot(
    staging: &Path,
    plan: &Plan,
    scans: &[Scan; 2],
    progress: &Progress,
) -> Result<()> {
    let mut buf = vec![0u8; BUF];
    let mut files = Vec::with_capacity(plan.len());
    for (n, (game, path, eff)) in plan.iter().enumerate() {
        if n % 50 == 0 || eff.size > (4 << 20) {
            progress(&format!(
                "Importing {}/{}: {} {}",
                n + 1,
                plan.len(),
                game.dir().to_uppercase(),
                path
            ));
        }
        let dest = disk_path(staging, *game, path);
        files.push(copy_member(
            &scans[game.slot()],
            eff,
            *game,
            path,
            &dest,
            &mut buf,
        )?);
    }

    let mut indexes = Vec::new();
    fs::create_dir_all(staging.join(INDEX_DIR))?;
    for game in Game::ALL {
        let tsv = scans[game.slot()].index_tsv()?;
        let path = index_path(staging, game);
        fs::write(&path, &tsv).with_context(|| format!("write {}", path.display()))?;
        indexes.push(IndexInfo {
            game,
            size: tsv.len() as u64,
            sha256: hex::encode(Sha256::digest(&tsv)),
            entries: tsv.iter().filter(|&&c| c == b'\n').count() as u64,
        });
    }
    let mut sources = BTreeMap::new();
    for game in Game::ALL {
        sources.insert(game.dir().to_owned(), scans[game.slot()].source_info());
    }
    let manifest = Manifest {
        format: FORMAT.into(),
        version: VERSION,
        language: scans[Game::Ntw.slot()]
            .language
            .clone()
            .ok_or_else(|| anyhow!("the NTW language was not determined"))?,
        sources,
        indexes,
        files,
    };
    let manifest_path = staging.join(MANIFEST);
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;

    // Immutable from here on: every file becomes read-only.
    for e in &manifest.files {
        set_readonly(&disk_path(staging, e.game, &e.path))?;
    }
    for game in Game::ALL {
        set_readonly(&index_path(staging, game))?;
    }
    set_readonly(&manifest_path)?;
    Ok(())
}

pub(crate) fn set_readonly(path: &Path) -> Result<()> {
    let mut perm = fs::metadata(path)?.permissions();
    perm.set_readonly(true);
    fs::set_permissions(path, perm).with_context(|| format!("protect {}", path.display()))
}

/// An existing snapshot is accepted only if every selected source file hashes to what it recorded.
/// Returns an error for any selected file the snapshot lacks.
fn check_same_sources(
    snap: &Snapshot,
    plan: &Plan,
    scans: &[Scan; 2],
    progress: &Progress,
) -> Result<()> {
    let mut want: HashSet<(Game, &str)> = HashSet::new();
    for (game, path, _) in plan.iter() {
        want.insert((*game, path.as_str()));
        if !snap.inner.by_key.contains_key(&(*game, path.clone())) {
            bail!(
                "snapshot lacks {} {path}, which the installation provides; refusing to overwrite (delete the snapshot folder to import again)",
                game.dir()
            );
        }
    }
    for f in &snap.inner.manifest.files {
        let retired = f.game == Game::Ntw && RETIRED_NTW_SOURCES.contains(&f.path.as_str());
        if !retired && !want.contains(&(f.game, f.path.as_str())) {
            bail!(
                "snapshot holds {} {}, which the installation no longer provides; refusing to overwrite (delete the snapshot folder to import again)",
                f.game.dir(),
                f.path
            );
        }
    }
    let mut buf = vec![0u8; BUF];
    for (n, (game, path, eff)) in plan.iter().enumerate() {
        if n % 50 == 0 || eff.size > (4 << 20) {
            progress(&format!(
                "Comparing source {}/{}: {} {}",
                n + 1,
                plan.len(),
                game.dir().to_uppercase(),
                path
            ));
        }
        let entry = &snap.inner.manifest.files[snap.inner.by_key[&(*game, path.clone())]];
        let mut src = scans[game.slot()].open(eff)?;
        let mut h = Sha256::new();
        let len = pump(&mut src, None, &mut h, &mut buf)
            .with_context(|| format!("read {}", eff.stored))?;
        let sha = hex::encode(h.finalize());
        if len != entry.size || sha != entry.sha256 {
            bail!(
                "{} {}: the installation now differs from the snapshot (hash {sha} vs recorded {}); refusing to overwrite (delete the snapshot folder to import again)",
                game.dir().to_uppercase(),
                eff.stored,
                entry.sha256
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Reading a snapshot
// ---------------------------------------------------------------------------------------------

impl Snapshot {
    pub fn open(root: &Path) -> Result<Self> {
        let mpath = root.join(MANIFEST);
        let raw = fs::read(&mpath)
            .with_context(|| format!("read snapshot manifest {}", mpath.display()))?;
        let manifest: Manifest = serde_json::from_slice(&raw)
            .with_context(|| format!("snapshot manifest {} is not valid", mpath.display()))?;
        if manifest.format != FORMAT || manifest.version != VERSION {
            bail!(
                "{} is not a {FORMAT} v{VERSION} manifest (found {} v{})",
                mpath.display(),
                manifest.format,
                manifest.version
            );
        }
        let mut by_key = HashMap::with_capacity(manifest.files.len());
        let mut names: [Vec<String>; 2] = [Vec::new(), Vec::new()];
        for (i, e) in manifest.files.iter().enumerate() {
            validate_rel(&e.path).with_context(|| format!("manifest entry {i}"))?;
            validate_rel(&e.stored).with_context(|| format!("manifest entry {i}"))?;
            if e.path != canon(&e.path) || canon(&e.stored) != e.path {
                bail!(
                    "manifest entry {i}: path {:?} is not the canonical form of {:?}",
                    e.path,
                    e.stored
                );
            }
            validate_sha(&e.sha256).with_context(|| format!("manifest entry {}", e.path))?;
            if by_key.insert((e.game, e.path.clone()), i).is_some() {
                bail!("manifest lists {} {} twice", e.game.dir(), e.path);
            }
            names[e.game.slot()].push(e.path.clone());
        }
        for game in Game::ALL {
            let idx: Vec<&IndexInfo> = manifest.indexes.iter().filter(|i| i.game == game).collect();
            if idx.len() != 1 {
                bail!(
                    "manifest must hold exactly one source index for {}",
                    game.dir()
                );
            }
            validate_sha(&idx[0].sha256)?;
            names[game.slot()].sort();
        }
        for e in &manifest.files {
            let p = disk_path(root, e.game, &e.path);
            let meta = fs::symlink_metadata(&p)
                .with_context(|| format!("snapshot file {} {} is missing", e.game.dir(), e.path))?;
            if !meta.is_file() {
                bail!(
                    "snapshot entry {} {} is not a regular file",
                    e.game.dir(),
                    e.path
                );
            }
            if meta.len() != e.size {
                bail!(
                    "snapshot file {} {} is corrupt: {} bytes on disk, manifest records {}",
                    e.game.dir(),
                    e.path,
                    meta.len(),
                    e.size
                );
            }
        }
        let source = [
            load_source_index(root, &manifest, Game::Etw)?,
            load_source_index(root, &manifest, Game::Ntw)?,
        ];
        Ok(Snapshot {
            root: root.to_owned(),
            inner: Arc::new(Inner {
                manifest,
                by_key,
                names,
                verified: Mutex::new(HashSet::new()),
                source,
            }),
            overlay: None,
        })
    }

    /// Language of the NTW localisation this snapshot holds (`en` for snapshots that predate the field).
    pub fn language(&self) -> &str {
        &self.inner.manifest.language
    }

    pub fn manifest(&self) -> &Manifest {
        &self.inner.manifest
    }

    pub fn entry(&self, game: Game, path: &str) -> Result<&FileEntry> {
        Ok(match self.locate(game, path)? {
            Loc::Base(_, e) => e,
            Loc::Ov(f) => &f.entry,
        })
    }

    /// Every visible file of `game` (overlays honoured), in no particular order.
    pub fn entries(&self, game: Game) -> impl Iterator<Item = &FileEntry> {
        let ov = if game == Game::Ntw {
            self.overlay.as_deref()
        } else {
            None
        };
        let base = self.inner.manifest.files.iter().filter(move |e| {
            e.game == game
                && ov.is_none_or(|o| !o.removed.contains(&e.path) && !o.files.contains_key(&e.path))
        });
        base.chain(
            ov.into_iter()
                .flat_map(|o| o.files.values().map(|f| &f.entry)),
        )
    }

    pub fn exists(&self, game: Game, path: &str) -> bool {
        self.locate(game, path).is_ok()
    }

    fn locate(&self, game: Game, path: &str) -> Result<Loc<'_>> {
        let key = canon(path);
        validate_rel(&key).with_context(|| format!("snapshot path {path:?}"))?;
        if game == Game::Ntw {
            if let Some(ov) = &self.overlay {
                if let Some(f) = ov.files.get(&key) {
                    return Ok(Loc::Ov(f));
                }
                if ov.removed.contains(&key) {
                    bail!("snapshot has no NTW file {path}");
                }
            }
        }
        let i =
            *self.inner.by_key.get(&(game, key)).ok_or_else(|| {
                anyhow!("snapshot has no {} file {path}", game.dir().to_uppercase())
            })?;
        Ok(Loc::Base(i, &self.inner.manifest.files[i]))
    }

    fn check_on_disk(&self, e: &FileEntry) -> Result<PathBuf> {
        let p = disk_path(&self.root, e.game, &e.path);
        let meta = fs::symlink_metadata(&p)
            .with_context(|| format!("snapshot file {} {} is missing", e.game.dir(), e.path))?;
        if !meta.is_file() || meta.len() != e.size {
            bail!(
                "snapshot file {} {} is corrupt: not a regular file of {} bytes",
                e.game.dir(),
                e.path,
                e.size
            );
        }
        Ok(p)
    }

    /// Whole file, checked against the manifest (size and SHA-256) on every read.
    pub fn read(&self, game: Game, path: &str) -> Result<Vec<u8>> {
        let e = match self.locate(game, path)? {
            Loc::Ov(f) => return Ok(f.read_all()),
            Loc::Base(_, e) => e,
        };
        let p = self.check_on_disk(e)?;
        let bytes = fs::read(&p).with_context(|| format!("read {}", p.display()))?;
        let sha = hex::encode(Sha256::digest(&bytes));
        if bytes.len() as u64 != e.size || sha != e.sha256 {
            bail!(
                "snapshot file {} {} is corrupt: hash {sha} differs from manifest {}",
                e.game.dir(),
                e.path,
                e.sha256
            );
        }
        Ok(bytes)
    }

    /// Real file for seek-style readers. The file is hash-verified the first time it is requested through
    /// this snapshot instance (and every time through `read`/`verify`). Overlay members that live in memory
    /// or inside a pack have no standalone file and report an error.
    pub fn path(&self, game: Game, path: &str) -> Result<PathBuf> {
        let (i, e) = match self.locate(game, path)? {
            Loc::Ov(_) => {
                bail!("{game:?} {path} is an in-memory overlay member and has no standalone file")
            }
            Loc::Base(i, e) => (i, e),
        };
        let p = self.check_on_disk(e)?;
        if locked(&self.inner.verified).contains(&i) {
            return Ok(p);
        }
        let (len, sha) = hash_file(&p, &mut vec![0u8; BUF])?;
        if len != e.size || sha != e.sha256 {
            bail!(
                "snapshot file {} {} is corrupt: hash {sha} differs from manifest {}",
                e.game.dir(),
                e.path,
                e.sha256
            );
        }
        locked(&self.inner.verified).insert(i);
        Ok(p)
    }

    /// Sorted canonical (lower-case, `/`) names of the visible files starting with `prefix`.
    pub fn names(&self, game: Game, prefix: &str) -> Vec<String> {
        let prefix = canon(prefix);
        let base = self.inner.names[game.slot()]
            .iter()
            .filter(|n| n.starts_with(&prefix));
        match (game, &self.overlay) {
            (Game::Ntw, Some(ov)) => {
                let mut v: Vec<String> = base
                    .filter(|n| !ov.removed.contains(*n) && !ov.files.contains_key(*n))
                    .cloned()
                    .collect();
                v.extend(
                    ov.files
                        .range::<str, _>((Included(prefix.as_str()), Unbounded))
                        .take_while(|(k, _)| k.starts_with(&prefix))
                        .map(|(k, _)| k.clone()),
                );
                v.sort();
                v
            }
            _ => base.cloned().collect(),
        }
    }

    /// Whether the original installation (or an overlay) held this file, including the many files that
    /// were not extracted.
    pub fn has_in_source(&self, game: Game, path: &str) -> bool {
        let key = canon(path);
        if game == Game::Ntw {
            if let Some(ov) = &self.overlay {
                if ov.files.contains_key(&key) {
                    return true;
                }
                if ov.removed.contains(&key) {
                    return false;
                }
                if ov.src_added.contains(&key) {
                    return true;
                }
            }
        }
        self.inner.source[game.slot()].binary_search(&key).is_ok()
    }

    /// Canonical names of every original file starting with `prefix` (extracted or not).
    pub fn names_in_source(&self, game: Game, prefix: &str) -> Vec<String> {
        let prefix = canon(prefix);
        let base = self.inner.source[game.slot()]
            .iter()
            .filter(|n| n.starts_with(&prefix));
        match (game, &self.overlay) {
            (Game::Ntw, Some(ov)) => {
                let mut v: Vec<String> = base
                    .filter(|n| !ov.removed.contains(*n) && !ov.files.contains_key(*n))
                    .cloned()
                    .collect();
                v.extend(
                    ov.files
                        .range::<str, _>((Included(prefix.as_str()), Unbounded))
                        .take_while(|(k, _)| k.starts_with(&prefix))
                        .map(|(k, _)| k.clone()),
                );
                v.extend(
                    ov.src_added
                        .range::<str, _>((Included(prefix.as_str()), Unbounded))
                        .take_while(|k| k.starts_with(&prefix))
                        .cloned(),
                );
                v.sort();
                v.dedup();
                v
            }
            _ => base.cloned().collect(),
        }
    }

    /// Precedence of the member that currently wins: 0 for the original snapshot (and for everything of
    /// ETW), 1 for target dataset members, one above the previous overlay for converter-resolved members.
    pub fn member_priority(&self, game: Game, path: &str) -> usize {
        match self.locate(game, path) {
            Ok(Loc::Ov(f)) => f.priority,
            _ => 0,
        }
    }

    /// Human-readable origin of the winning member (never contains absolute paths).
    pub fn member_origin(&self, game: Game, path: &str) -> String {
        match self.locate(game, path) {
            Ok(Loc::Ov(f)) => f.origin.clone(),
            Ok(Loc::Base(_, e)) => match &e.origin {
                Origin::Loose { file } => format!("original snapshot: loose {file}"),
                Origin::Pack { file, member } => format!("original snapshot: {file} {member}"),
            },
            Err(_) => "not present".to_owned(),
        }
    }

    /// Cheap derived snapshot in which NTW members are replaced/added by `files` (canonicalised names ->
    /// bytes), `removed` NTW names disappear everywhere (members and source names) and `extra_names` are
    /// known to exist in the source (`has_in_source`/`names_in_source`) without being readable. The
    /// original snapshot and ETW are untouched; the payloads are shared, never copied again.
    pub fn with_ntw_overlay(
        &self,
        files: BTreeMap<String, Vec<u8>>,
        removed: Vec<String>,
        extra_names: Vec<String>,
    ) -> Result<Self> {
        let members = files
            .into_iter()
            .map(|(name, bytes)| {
                let origin = format!("resolved by converter: {}", name.replace('\\', "/"));
                (name, (bytes, origin))
            })
            .collect();
        self.derive_ntw(members, removed, extra_names)
    }

    /// Like `with_ntw_overlay`, with a caller-supplied human-readable origin per member (the target dataset
    /// names the TSV file each member came from). Origins must not contain absolute host paths.
    pub fn with_ntw_members(
        &self,
        members: BTreeMap<String, (Vec<u8>, String)>,
        removed: Vec<String>,
    ) -> Result<Self> {
        self.derive_ntw(members, removed, Vec::new())
    }

    fn derive_ntw(
        &self,
        members: BTreeMap<String, (Vec<u8>, String)>,
        removed: Vec<String>,
        extra: Vec<String>,
    ) -> Result<Self> {
        let mut ov = self.overlay.as_deref().cloned().unwrap_or_default();
        let priority = ov.top + 1;
        let mut files: Vec<OvInput> = Vec::with_capacity(members.len());
        for (name, (bytes, origin)) in members {
            let key = canon(&name);
            validate_rel(&key).with_context(|| format!("overlay member {name:?}"))?;
            if files.iter().any(|f| f.entry.path == key) {
                bail!("overlay member {key} is provided twice");
            }
            let stored = name.replace('\\', "/");
            files.push(OvInput {
                entry: FileEntry {
                    game: Game::Ntw,
                    path: key,
                    stored: stored.clone(),
                    size: bytes.len() as u64,
                    sha256: hex::encode(Sha256::digest(&bytes)),
                    origin: Origin::Loose {
                        file: format!("({origin})/{stored}"),
                    },
                },
                bytes: Arc::new(bytes),
                priority,
                origin,
            });
        }
        let mut removed_keys = BTreeSet::new();
        for name in removed {
            let key = canon(&name);
            validate_rel(&key).with_context(|| format!("removed member {name:?}"))?;
            removed_keys.insert(key);
        }
        let mut extra_keys = BTreeSet::new();
        for name in extra {
            let key = canon(&name);
            validate_rel(&key).with_context(|| format!("extra source name {name:?}"))?;
            extra_keys.insert(key);
        }
        for key in &removed_keys {
            if files.iter().any(|f| &f.entry.path == key) || extra_keys.contains(key) {
                bail!("overlay both provides and removes {key}");
            }
        }
        for key in &removed_keys {
            ov.files.remove(key);
            ov.src_added.remove(key);
            ov.removed.insert(key.clone());
        }
        for key in extra_keys {
            ov.removed.remove(&key);
            if !ov.files.contains_key(&key) {
                ov.src_added.insert(key);
            }
        }
        for f in files {
            let key = f.entry.path.clone();
            ov.removed.remove(&key);
            ov.src_added.remove(&key);
            ov.top = ov.top.max(f.priority);
            ov.files.insert(
                key,
                Arc::new(OvFile {
                    entry: f.entry,
                    bytes: f.bytes,
                    priority: f.priority,
                    origin: f.origin,
                }),
            );
        }
        Ok(Snapshot {
            root: self.root.clone(),
            inner: self.inner.clone(),
            overlay: Some(Arc::new(ov)),
        })
    }

    /// The immutable original snapshot behind this one: same files and root, no overlay. Cheap (shares
    /// the manifest); use it to compare modded members with their vanilla baseline.
    pub fn original(&self) -> Snapshot {
        Snapshot {
            root: self.root.clone(),
            inner: self.inner.clone(),
            overlay: None,
        }
    }

    /// Full integrity check: every file and index hashes to its manifest record and nothing else exists
    /// inside the snapshot folder.
    pub fn verify(&self, progress: &Progress) -> Result<()> {
        let files = &self.inner.manifest.files;
        let mut buf = vec![0u8; BUF];
        for (n, e) in files.iter().enumerate() {
            if n % 100 == 0 || e.size > (4 << 20) {
                progress(&format!(
                    "Verifying snapshot {}/{}: {} {}",
                    n + 1,
                    files.len(),
                    e.game.dir().to_uppercase(),
                    e.path
                ));
            }
            let p = self.check_on_disk(e)?;
            let (len, sha) = hash_file(&p, &mut buf)?;
            if len != e.size || sha != e.sha256 {
                bail!(
                    "snapshot file {} {} is corrupt: hash {sha} differs from manifest {}",
                    e.game.dir(),
                    e.path,
                    e.sha256
                );
            }
            locked(&self.inner.verified).insert(n);
        }
        for info in &self.inner.manifest.indexes {
            let p = index_path(&self.root, info.game);
            let (len, sha) = hash_file(&p, &mut buf)?;
            if len != info.size || sha != info.sha256 {
                bail!("snapshot source index {} is corrupt", p.display());
            }
        }
        let mut allowed: HashSet<String> = files
            .iter()
            .map(|e| format!("{}/{}", e.game.dir(), e.path))
            .collect();
        allowed.insert(MANIFEST.to_owned());
        allowed.extend(
            Game::ALL
                .iter()
                .map(|g| format!("{INDEX_DIR}/{}.tsv", g.dir())),
        );
        for entry in WalkDir::new(&self.root).follow_links(false) {
            let entry = entry?;
            let ty = entry.file_type();
            if ty.is_symlink() {
                bail!("snapshot contains a link at {}", entry.path().display());
            }
            if !ty.is_file() {
                continue;
            }
            let rel = entry
                .path()
                .strip_prefix(&self.root)?
                .to_string_lossy()
                .replace('\\', "/")
                .to_lowercase();
            if !allowed.contains(&rel) {
                bail!(
                    "snapshot contains an unlisted file {}",
                    entry.path().display()
                );
            }
        }
        progress("Snapshot verified");
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::pack::{TYPE_MOVIE, TYPE_PATCH, TYPE_RELEASE, test_util::TempDir, write_pack};

    pub(crate) fn quiet() -> impl Fn(&str) + Send + Sync {
        |_: &str| {}
    }

    fn put(root: &Path, rel: &str, data: &[u8]) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, data).unwrap();
    }

    fn pack_of(path: &Path, kind: u32, members: &[(&str, Vec<u8>)]) {
        let map: BTreeMap<String, Vec<u8>> = members
            .iter()
            .map(|(n, d)| (n.to_string(), d.clone()))
            .collect();
        write_pack(path, &map, kind).unwrap();
    }

    /// A miniature ETW + NTW installation with every file the importer requires, plus decoys.
    pub(crate) struct Installs {
        pub(crate) dir: TempDir,
        pub(crate) etw: PathBuf,
        pub(crate) ntw: PathBuf,
        pub(crate) dest: PathBuf,
    }

    pub(crate) fn installs() -> Installs {
        let dir = TempDir::new("snap");
        let etw = dir.path().join("etw_data");
        let ntw = dir.path().join("ntw_data");
        let dest = dir.path().join("project").join("snapshot");
        fs::create_dir_all(&etw).unwrap();
        fs::create_dir_all(&ntw).unwrap();
        let big: Vec<u8> = (0..3_000_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();

        pack_of(
            &etw.join("main.pack"),
            TYPE_RELEASE,
            &[
                (
                    "db\\factions_tables\\factions",
                    b"etw-factions-release".to_vec(),
                ),
                ("db\\regions_tables\\regions", b"etw-regions".to_vec()),
                ("text\\localisation.loc", b"etw-loc".to_vec()),
                ("text\\sub\\nested.loc", b"not wanted".to_vec()),
                (
                    "campaigns\\main\\startpos.esf",
                    b"pack-startpos-overridden-by-loose".to_vec(),
                ),
                ("heightmaps\\default.tga", b"hm".to_vec()),
                ("battleterrain\\tiles\\metadata.dat", b"MTGI".to_vec()),
                ("campaign\\terrain\\Campaign_Detail.dds", b"dds".to_vec()),
                ("campaign\\terrain\\rocks1_diffuse.dds", b"rocks".to_vec()),
                ("sounds\\bang.wav", b"decoy".to_vec()),
            ],
        );
        pack_of(
            &etw.join("patch.pack"),
            TYPE_PATCH,
            &[(
                "db\\factions_tables\\factions",
                b"etw-factions-patched".to_vec(),
            )],
        );
        pack_of(
            &etw.join("supertexture.pack"),
            TYPE_RELEASE,
            &[
                ("world_.stpi", b"stpi".to_vec()),
                ("world_.stpd", big.clone()),
            ],
        );
        put(&etw, "campaigns/main/startpos.esf", b"loose-startpos");
        put(&etw, "campaign_maps/global_map/regions.esf", b"regions");
        put(&etw, "campaign_maps/global_map/europe_map.tga", b"tga");
        put(&etw, "campaign_maps/natives_map/regions.esf", b"decoy");
        put(&etw, "backup/campaigns/main/startpos.esf", b"decoy backup");

        pack_of(
            &ntw.join("data.pack"),
            TYPE_RELEASE,
            &[
                ("db\\factions_tables\\factions", b"ntw-factions".to_vec()),
                ("ui\\frontend ui\\main", b"layout".to_vec()),
                ("ui\\campaign ui\\travel_destinations.luac", b"lua".to_vec()),
                ("ui\\campaign ui\\pips\\icon.tga", b"image".to_vec()),
                ("models\\tree_a.rigid_model", b"model".to_vec()),
            ],
        );
        pack_of(
            &ntw.join("local_en.pack"),
            TYPE_RELEASE,
            &[
                ("text\\localisation.loc", b"ntw-loc".to_vec()),
                ("text\\ui.loc", b"ntw-ui-loc".to_vec()),
            ],
        );
        pack_of(
            &ntw.join("nap_world.pack"),
            TYPE_MOVIE,
            &[
                ("db\\factions_tables\\nap_world", b"MOD".to_vec()),
                ("text\\localisation.loc", b"MOD-LOC".to_vec()),
            ],
        );
        put(&ntw, "campaigns/eur_napoleon/startpos.esf", b"ntw-startpos");
        put(
            &ntw,
            "campaigns/wld_napoleon/startpos.esf",
            b"our own output",
        );
        put(&ntw, "campaign_maps/nap_europe/regions.esf", b"nap-regions");
        put(
            &ntw,
            "campaign_maps/nap_europe/display/environment/spring.lighting",
            b"light",
        );
        put(
            &ntw,
            "campaign_maps/nap_europe/display/fogmask/fogmask.dds",
            b"fog",
        );
        put(
            &ntw,
            "campaign_maps/nap_europe/display/supertexture/supertexture.stpd",
            &big,
        );
        put(
            &ntw,
            "campaign_maps/nap_egypt/display/fogmask/fogmask.dds",
            b"not wanted: images only from europe",
        );
        put(
            &ntw,
            "campaign_maps/nap_egypt/display/arrows/arrows.rigid_model",
            b"arrows",
        );
        put(
            &ntw,
            "campaign_maps/nap_world/regions.esf",
            b"our own output",
        );
        put(
            &ntw,
            "campaign_maps/nap_europe/backup/regions.esf",
            b"decoy backup",
        );
        put(&ntw, "campaign_maps/my_spain/regions.esf", b"unrelated map");
        Installs {
            dir,
            etw,
            ntw,
            dest,
        }
    }

    fn sha(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn unlock(path: &Path) {
        let mut perm = fs::metadata(path).unwrap().permissions();
        perm.set_readonly(false);
        fs::set_permissions(path, perm).unwrap();
    }

    #[test]
    fn import_copies_exactly_the_relevant_originals_with_game_precedence() {
        let i = installs();
        let snap = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();

        // Loose beats pack, patch beats release, mod pack ignored, case-insensitive and either separator.
        assert_eq!(
            snap.read(Game::Etw, "campaigns\\main\\startpos.esf")
                .unwrap(),
            b"loose-startpos"
        );
        assert_eq!(
            snap.read(Game::Etw, "DB/factions_tables/factions").unwrap(),
            b"etw-factions-patched"
        );
        assert_eq!(
            snap.read(Game::Ntw, "text/localisation.loc").unwrap(),
            b"ntw-loc"
        );
        assert_eq!(
            snap.read(Game::Ntw, "db/factions_tables/factions").unwrap(),
            b"ntw-factions"
        );
        assert_eq!(
            snap.read(Game::Etw, "campaign/terrain/campaign_detail.dds")
                .unwrap(),
            b"dds"
        );
        assert_eq!(
            snap.read(Game::Etw, "world_.stpd").unwrap().len(),
            3_000_000
        );

        // Names mirror the originals, lower-cased, and exclude everything not relevant.
        assert_eq!(
            snap.names(Game::Ntw, "db\\"),
            ["db/factions_tables/factions"]
        );
        assert_eq!(
            snap.names(Game::Ntw, "campaign_maps/"),
            [
                "campaign_maps/nap_egypt/display/arrows/arrows.rigid_model",
                "campaign_maps/nap_europe/display/environment/spring.lighting",
                "campaign_maps/nap_europe/display/fogmask/fogmask.dds",
                "campaign_maps/nap_europe/regions.esf",
            ]
        );
        assert_eq!(
            snap.names(Game::Ntw, "ui/"),
            [
                "ui/campaign ui/travel_destinations.luac",
                "ui/frontend ui/main"
            ]
        );
        assert_eq!(snap.names(Game::Etw, "text/"), ["text/localisation.loc"]);
        assert!(!snap.exists(Game::Etw, "sounds/bang.wav"));
        assert!(!snap.exists(Game::Etw, "campaign_maps/natives_map/regions.esf"));
        assert!(!snap.exists(Game::Etw, "backup/campaigns/main/startpos.esf"));
        assert!(!snap.exists(Game::Ntw, "campaigns/wld_napoleon/startpos.esf"));
        assert!(
            !snap
                .names(Game::Ntw, "")
                .iter()
                .any(|n| n.contains("nap_world") || n.contains("my_spain"))
        );

        // Provenance is recorded and real files are byte-identical to what was copied.
        let e = snap
            .entry(Game::Etw, "db/factions_tables/factions")
            .unwrap();
        assert_eq!(e.sha256, sha(b"etw-factions-patched"));
        assert!(
            matches!(&e.origin, Origin::Pack { file, member } if file == "patch.pack" && member == "db\\factions_tables\\factions")
        );
        assert!(
            matches!(&snap.entry(Game::Etw, "campaigns/main/startpos.esf").unwrap().origin, Origin::Loose { file } if file == "campaigns/main/startpos.esf")
        );
        assert_eq!(
            snap.entry(Game::Etw, "campaign/terrain/campaign_detail.dds")
                .unwrap()
                .stored,
            "campaign/terrain/Campaign_Detail.dds"
        );
        assert_eq!(
            fs::read(snap.path(Game::Etw, "world_.stpi").unwrap()).unwrap(),
            b"stpi"
        );

        // Files that were not extracted are still known to exist in the originals.
        assert!(snap.has_in_source(Game::Ntw, "models\\tree_A.rigid_model"));
        assert!(snap.has_in_source(Game::Etw, "sounds/bang.wav"));
        assert!(!snap.has_in_source(Game::Ntw, "db/factions_tables/nap_world"));
        assert_eq!(
            snap.names_in_source(Game::Ntw, "models/"),
            ["models/tree_a.rigid_model"]
        );
        snap.verify(&quiet()).unwrap();
    }

    #[test]
    fn snapshot_is_standalone_and_read_only_after_import() {
        let i = installs();
        import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        fs::remove_dir_all(&i.etw).unwrap();
        fs::remove_dir_all(&i.ntw).unwrap();
        let snap = Snapshot::open(&i.dest).unwrap();
        snap.verify(&quiet()).unwrap();
        assert_eq!(
            snap.read(Game::Ntw, "campaigns/eur_napoleon/startpos.esf")
                .unwrap(),
            b"ntw-startpos"
        );
        let p = snap
            .path(Game::Ntw, "campaigns/eur_napoleon/startpos.esf")
            .unwrap();
        assert!(p.starts_with(&i.dest));
        assert!(
            fs::write(&p, b"overwrite").is_err(),
            "snapshot files must be read-only"
        );
    }

    #[test]
    fn import_is_idempotent_and_refuses_changed_sources_or_damaged_snapshots() {
        let i = installs();
        let first = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        let again = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        assert_eq!(first.manifest().files.len(), again.manifest().files.len());

        // A changed original is refused, never silently merged or overwritten.
        let loose = i.etw.join("campaigns/main/startpos.esf");
        fs::write(&loose, b"edited-by-user").unwrap();
        let err = format!(
            "{:#}",
            import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap_err()
        );
        assert!(err.contains("differs from the snapshot"), "{err}");
        assert_eq!(
            first
                .read(Game::Etw, "campaigns/main/startpos.esf")
                .unwrap(),
            b"loose-startpos"
        );
        fs::write(&loose, b"loose-startpos").unwrap();

        // A new relevant original that the snapshot lacks is refused too.
        put(&i.etw, "db/new_tables/new", b"new table");
        assert!(import(&i.etw, &i.ntw, &i.dest, &quiet()).is_err());
        fs::remove_dir_all(i.etw.join("db")).unwrap();
        import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        fs::remove_file(i.etw.join("campaign_maps/global_map/europe_map.tga")).unwrap();
        assert!(
            import(&i.etw, &i.ntw, &i.dest, &quiet()).is_err(),
            "removing a relevant source file must be refused as well"
        );
    }

    #[test]
    fn campaign_building_shaders_are_not_imported_and_old_snapshots_holding_them_stay_valid() {
        let i = installs();
        let top = b"top shader".to_vec();
        put(&i.ntw, "fx/campaignbuilding.fx", &top);
        put(&i.ntw, "fx/campaignbuilding_vertex.fx_fragment", b"vertex fragment");
        let fresh = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        assert!(!fresh.exists(Game::Ntw, "fx/campaignbuilding.fx"));
        assert!(!fresh.exists(Game::Ntw, "fx/campaignbuilding_vertex.fx_fragment"));

        // Rewrite the snapshot the way an earlier editor captured the same two installed files.
        let mut manifest = fresh.manifest().clone();
        let origin = manifest.files.iter().find(|f| f.game == Game::Ntw).unwrap().origin.clone();
        for (path, data) in [
            ("fx/campaignbuilding.fx", top.as_slice()),
            ("fx/campaignbuilding_vertex.fx_fragment", b"vertex fragment".as_slice()),
        ] {
            let dest = disk_path(&i.dest, Game::Ntw, path);
            put(&i.dest, &format!("ntw/{path}"), data);
            set_readonly(&dest).unwrap();
            manifest.files.push(FileEntry {
                game: Game::Ntw,
                path: path.to_owned(),
                stored: path.to_owned(),
                size: data.len() as u64,
                sha256: hex::encode(Sha256::digest(data)),
                origin: origin.clone(),
            });
        }
        manifest.files.sort_by(|a, b| (a.game, &a.path).cmp(&(b.game, &b.path)));
        let manifest_path = i.dest.join(MANIFEST);
        unlock(&manifest_path);
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        set_readonly(&manifest_path).unwrap();

        let old = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        assert_eq!(old.read(Game::Ntw, "fx/campaignbuilding.fx").unwrap(), top);
        assert_eq!(old.manifest().files.len(), fresh.manifest().files.len() + 2);
        old.verify(&quiet()).unwrap();

        // The installed shader is not selected, so its removal or change never matters to the import.
        fs::remove_file(i.ntw.join("fx/campaignbuilding.fx")).unwrap();
        import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap().verify(&quiet()).unwrap();
    }

    #[test]
    fn tampering_is_detected_by_read_path_verify_and_open() {
        let i = installs();
        let snap = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        let victim = disk_path(&i.dest, Game::Etw, "heightmaps/default.tga");
        unlock(&victim);
        fs::write(&victim, b"HM").unwrap(); // same size, different content
        assert!(snap.read(Game::Etw, "heightmaps/default.tga").is_err());
        assert!(snap.path(Game::Etw, "heightmaps/default.tga").is_err());
        assert!(snap.verify(&quiet()).is_err());
        assert!(
            import(&i.etw, &i.ntw, &i.dest, &quiet()).is_err(),
            "a damaged snapshot must not be accepted as up to date"
        );
        // The untouched files still read fine.
        assert_eq!(snap.read(Game::Etw, "world_.stpi").unwrap(), b"stpi");

        fs::write(&victim, b"truncated?").unwrap(); // size change
        assert!(Snapshot::open(&i.dest).is_err());
        fs::remove_file(&victim).unwrap();
        assert!(Snapshot::open(&i.dest).is_err());
    }

    #[test]
    fn unlisted_files_and_bad_manifests_are_rejected() {
        let i = installs();
        let snap = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        put(&i.dest, "etw/db/sneaky_tables/extra", b"x");
        let err = format!("{:#}", snap.verify(&quiet()).unwrap_err());
        assert!(err.contains("unlisted"), "{err}");
        fs::remove_file(i.dest.join("etw/db/sneaky_tables/extra")).unwrap();
        snap.verify(&quiet()).unwrap();

        let manifest_path = i.dest.join(MANIFEST);
        unlock(&manifest_path);
        let original = fs::read_to_string(&manifest_path).unwrap();
        let rewrite = |from: &str, to: &str| {
            fs::write(&manifest_path, original.replacen(from, to, 1)).unwrap()
        };
        rewrite(
            "\"path\": \"world_.stpi\"",
            "\"path\": \"../manifest.json\"",
        );
        assert!(
            Snapshot::open(&i.dest).is_err(),
            "path traversal in the manifest"
        );
        rewrite(
            "\"path\": \"world_.stpi\"",
            "\"path\": \"C:/windows/win.ini\"",
        );
        assert!(
            Snapshot::open(&i.dest).is_err(),
            "drive path in the manifest"
        );
        rewrite(&sha(b"stpi"), &"0".repeat(64));
        let tampered = Snapshot::open(&i.dest).unwrap();
        assert!(
            tampered.read(Game::Etw, "world_.stpi").is_err(),
            "recorded hash no longer matches the file"
        );
        rewrite("\"version\": 1", "\"version\": 2");
        assert!(Snapshot::open(&i.dest).is_err());
        fs::write(&manifest_path, b"{ not json").unwrap();
        assert!(Snapshot::open(&i.dest).is_err());
    }

    #[test]
    fn lookups_reject_traversal_and_unknown_files() {
        let i = installs();
        let snap = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        for bad in [
            "../manifest.json",
            "..\\manifest.json",
            "db/../../manifest.json",
            "/etc/passwd",
            "C:/x",
            "",
            "db//x",
            "./world_.stpi",
        ] {
            assert!(snap.read(Game::Etw, bad).is_err(), "{bad:?}");
            assert!(snap.path(Game::Etw, bad).is_err(), "{bad:?}");
            assert!(!snap.exists(Game::Etw, bad), "{bad:?}");
        }
        assert!(snap.read(Game::Etw, "world_.nope").is_err());
        assert!(
            snap.read(Game::Ntw, "world_.stpi").is_err(),
            "files belong to one game only"
        );
    }

    #[test]
    fn hostile_pack_member_names_abort_the_import_without_leaving_output() {
        let i = installs();
        // write_pack itself refuses '..', so craft the hostile pack by patching the index bytes.
        let evil = i.etw.join("evil.pack");
        pack_of(&evil, TYPE_PATCH, &[("db\\xx\\evil_file", b"pwn".to_vec())]);
        let mut bytes = fs::read(&evil).unwrap();
        let at = bytes
            .windows(14)
            .position(|w| w == b"db\\xx\\evil_fil")
            .unwrap();
        bytes[at..at + 6].copy_from_slice(b"db\\..\\");
        fs::write(&evil, &bytes).unwrap();
        assert_eq!(
            Pack::open(&evil).unwrap().entries[0].name,
            "db\\..\\evil_file"
        );
        let err = format!(
            "{:#}",
            import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap_err()
        );
        assert!(err.contains("unsafe path"), "{err}");
        assert!(!i.dest.exists());
        let leftovers: Vec<_> = fs::read_dir(i.dest.parent().unwrap())
            .map(|r| r.flatten().map(|e| e.path()).collect::<Vec<_>>())
            .unwrap_or_default();
        assert!(
            leftovers.is_empty(),
            "staging folders must be cleaned up: {leftovers:?}"
        );
    }

    #[test]
    fn destination_must_not_overlap_installs_or_clobber_other_folders() {
        let i = installs();
        assert!(import(&i.etw, &i.ntw, &i.etw.join("snapshot"), &quiet()).is_err());
        assert!(import(&i.etw, &i.ntw, &i.ntw, &quiet()).is_err());
        assert!(import(&i.etw, &i.ntw, i.etw.parent().unwrap(), &quiet()).is_err());
        fs::create_dir_all(&i.dest).unwrap();
        put(&i.dest, "precious.txt", b"keep");
        assert!(import(&i.etw, &i.ntw, &i.dest, &quiet()).is_err());
        assert_eq!(fs::read(i.dest.join("precious.txt")).unwrap(), b"keep");
        fs::remove_file(i.dest.join("precious.txt")).unwrap();
        // An empty existing folder is fine.
        import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
    }

    #[test]
    fn wrong_or_incomplete_installations_fail_clearly() {
        let i = installs();
        fs::remove_file(i.ntw.join("data.pack")).unwrap();
        assert!(
            format!(
                "{:#}",
                import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap_err()
            )
            .contains("data.pack")
        );
        let j = installs();
        fs::remove_file(j.etw.join("supertexture.pack")).unwrap();
        assert!(
            format!(
                "{:#}",
                import(&j.etw, &j.ntw, &j.dest, &quiet()).unwrap_err()
            )
            .contains("world_.stpi")
        );
        assert!(!j.dest.exists());
        let missing = j.etw.join("nope");
        assert!(import(&missing, &j.ntw, &j.dest, &quiet()).is_err());
    }

    #[test]
    fn path_rules() {
        for ok in [
            "db/a_tables/a",
            "ui/frontend ui/main.luac",
            "world_.stpd",
            "campaign_maps/nap_europe/display/x.rigid_spline",
        ] {
            validate_rel(ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
        }
        for bad in [
            "",
            "/a",
            "a/",
            "a//b",
            "a/./b",
            "a/../b",
            "..",
            "a\\b",
            "c:/a",
            "a/b:stream",
            "a/con",
            "a/CON.txt",
            "a/lpt1",
            "a/x.",
            "a/x ",
            "a/q?",
            "a/\u{1}",
            "a/com3.dat",
        ] {
            assert!(validate_rel(bad).is_err(), "{bad:?} must be rejected");
        }
        assert!(validate_rel("a/com0").is_ok() && validate_rel("a/comx").is_ok());
        assert!(
            etw_wanted("db/anything_tables/x")
                && etw_wanted("text/ui.loc")
                && !etw_wanted("text/sub/ui.loc")
                && !etw_wanted("text/readme.txt")
        );
        assert!(
            ntw_wanted("campaign_maps/nap_tut/display/world.markers")
                && !ntw_wanted("campaign_maps/nap_world/display/world.markers")
        );
        assert!(
            ntw_wanted("campaign_maps/nap_europe/display/fogmask/fogmask.dds")
                && !ntw_wanted("campaign_maps/nap_italy/display/fogmask/fogmask.dds")
        );
        assert!(
            !ntw_wanted("campaign_maps/nap_europe/display/supertexture/supertexture.stpd")
                && !ntw_wanted("campaign_maps/nap_europe/backup/regions.esf")
        );
        assert!(
            is_inside(
                Path::new("C:/Games/NTW/data/snap"),
                Path::new("c:/games/ntw/DATA")
            ) && !is_inside(Path::new("C:/Games/NTW2"), Path::new("C:/Games/NTW"))
        );
    }

    #[test]
    fn overlay_replaces_hides_and_adds_without_touching_the_original_snapshot() {
        let i = installs();
        let snap = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        let top = snap
            .with_ntw_overlay(
                BTreeMap::from([
                    (
                        "DB\\factions_tables\\factions".to_owned(),
                        b"resolved".to_vec(),
                    ),
                    ("db/units_tables/mod_units".to_owned(), b"units".to_vec()),
                ]),
                vec!["text/ui.loc".into()],
                vec!["models/mod_only.rigid_model".into()],
            )
            .unwrap();

        assert_eq!(
            top.read(Game::Ntw, "db/factions_tables/factions").unwrap(),
            b"resolved"
        );
        assert_eq!(
            top.read(Game::Ntw, "db/units_tables/mod_units").unwrap(),
            b"units"
        );
        assert_eq!(
            top.entry(Game::Ntw, "db/units_tables/mod_units")
                .unwrap()
                .sha256,
            sha(b"units")
        );
        assert!(
            top.names(Game::Ntw, "db/")
                .contains(&"db/units_tables/mod_units".to_owned())
        );
        assert!(!top.exists(Game::Ntw, "text/ui.loc"));
        assert!(top.read(Game::Ntw, "text/ui.loc").is_err());
        assert!(
            !top.names(Game::Ntw, "text/")
                .contains(&"text/ui.loc".to_owned())
        );
        assert!(!top.has_in_source(Game::Ntw, "text/ui.loc"));
        assert!(top.has_in_source(Game::Ntw, "models/mod_only.rigid_model"));
        assert!(!top.exists(Game::Ntw, "models/mod_only.rigid_model"));
        assert!(top.has_in_source(Game::Ntw, "models/tree_a.rigid_model"));
        assert_eq!(
            top.member_priority(Game::Ntw, "db/units_tables/mod_units"),
            1
        );
        assert_eq!(top.member_priority(Game::Ntw, "text/localisation.loc"), 0);
        assert_eq!(
            top.entries(Game::Ntw)
                .filter(|e| e.path == "text/ui.loc")
                .count(),
            0
        );
        // Stacked overlays rank higher and may bring a hidden name back.
        let again = top
            .with_ntw_overlay(
                BTreeMap::from([("text/ui.loc".to_owned(), b"back".to_vec())]),
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
        assert_eq!(again.read(Game::Ntw, "text/ui.loc").unwrap(), b"back");
        assert_eq!(again.member_priority(Game::Ntw, "text/ui.loc"), 2);
        assert!(
            top.with_ntw_overlay(
                BTreeMap::from([("x/y".to_owned(), vec![1])]),
                vec!["x/y".into()],
                Vec::new()
            )
            .is_err()
        );
        assert!(
            top.with_ntw_overlay(
                BTreeMap::from([("../escape".to_owned(), vec![1])]),
                Vec::new(),
                Vec::new()
            )
            .is_err()
        );

        // The original snapshot and ETW are unchanged.
        assert_eq!(
            snap.read(Game::Ntw, "db/factions_tables/factions").unwrap(),
            b"ntw-factions"
        );
        assert_eq!(snap.read(Game::Ntw, "text/ui.loc").unwrap(), b"ntw-ui-loc");
        assert_eq!(
            snap.member_priority(Game::Ntw, "db/factions_tables/factions"),
            0
        );
        assert_eq!(
            top.read(Game::Etw, "db/factions_tables/factions").unwrap(),
            b"etw-factions-patched"
        );
        snap.verify(&quiet()).unwrap();
    }

    #[test]
    fn target_members_carry_their_origin_and_converter_overlays_stack_above_them() {
        let i = installs();
        let snap = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        let target = snap
            .with_ntw_members(
                BTreeMap::from([(
                    "db/x_tables/y".to_owned(),
                    (b"from-tsv".to_vec(), "TSV x/y.tsv".to_owned()),
                )]),
                vec!["text/ui.loc".into()],
            )
            .unwrap();
        assert_eq!(
            target.read(Game::Ntw, "db/x_tables/y").unwrap(),
            b"from-tsv"
        );
        assert_eq!(
            target.member_origin(Game::Ntw, "db/x_tables/y"),
            "TSV x/y.tsv"
        );
        assert_eq!(target.member_priority(Game::Ntw, "db/x_tables/y"), 1);
        // Overlay members live in memory: no standalone file for seek-style readers.
        assert!(target.path(Game::Ntw, "db/x_tables/y").is_err());
        assert!(!target.exists(Game::Ntw, "text/ui.loc"));
        // The same member twice (differing only by case/separators) is refused, not last-wins.
        assert!(
            snap.with_ntw_members(
                BTreeMap::from([
                    ("db/a/b".to_owned(), (vec![1], "o".to_owned())),
                    ("DB\\A\\B".to_owned(), (vec![2], "o".to_owned())),
                ]),
                Vec::new(),
            )
            .is_err()
        );
        let more = target
            .with_ntw_overlay(
                BTreeMap::from([("db/z".to_owned(), vec![0])]),
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
        assert_eq!(more.member_priority(Game::Ntw, "db/z"), 2);
        assert_eq!(
            more.original().member_priority(Game::Ntw, "db/x_tables/y"),
            0
        );
        assert!(!more.original().exists(Game::Ntw, "db/x_tables/y"));
    }

    #[test]
    fn base_language_is_detected_recorded_and_never_guessed() {
        let i = installs();
        // English only: unchanged behaviour, recorded as such.
        let en = import(&i.etw, &i.ntw, &i.dest, &quiet()).unwrap();
        assert_eq!(en.language(), "en");
        assert_eq!(language_of(&i.dest).unwrap().as_deref(), Some("en"));
        assert_eq!(language_of(&i.dest.join("nowhere")).unwrap(), None);

        // A second language pack makes the folder ambiguous: refused until chosen explicitly.
        pack_of(
            &i.ntw.join("local_fr.pack"),
            TYPE_RELEASE,
            &[
                ("text\\localisation.loc", b"ntw-loc-fr".to_vec()),
                ("text\\ui.loc", b"ntw-ui-loc-fr".to_vec()),
            ],
        );
        pack_of(
            &i.ntw.join("local_fr_patch.pack"),
            TYPE_PATCH,
            &[("text\\ui.loc", b"ntw-ui-loc-fr-patched".to_vec())],
        );
        let other = i.dir.path().join("other_snapshot");
        let err = format!(
            "{:#}",
            import(&i.etw, &i.ntw, &other, &quiet()).unwrap_err()
        );
        assert!(
            err.contains("several languages") && err.contains("en, fr"),
            "{err}"
        );
        assert!(!other.exists());
        let err = format!(
            "{:#}",
            import_with_language(&i.etw, &i.ntw, &other, Some("de"), &quiet()).unwrap_err()
        );
        assert!(err.contains("local_de.pack"), "{err}");

        let fr = import_with_language(&i.etw, &i.ntw, &other, Some("FR"), &quiet()).unwrap();
        assert_eq!(fr.language(), "fr");
        assert_eq!(
            fr.read(Game::Ntw, "text/localisation.loc").unwrap(),
            b"ntw-loc-fr"
        );
        assert_eq!(
            fr.read(Game::Ntw, "text/ui.loc").unwrap(),
            b"ntw-ui-loc-fr-patched"
        );
        fr.verify(&quiet()).unwrap();
        // Re-importing without a choice keeps the recorded language; asking for another one is refused.
        assert_eq!(
            import(&i.etw, &i.ntw, &other, &quiet()).unwrap().language(),
            "fr"
        );
        assert!(import_with_language(&i.etw, &i.ntw, &other, Some("en"), &quiet()).is_err());
        // Mod-looking local_* packs are never base language packs.
        assert_eq!(
            base_language_pack("local_fr_patch.pack").as_deref(),
            Some("fr")
        );
        assert_eq!(base_language_pack("local_fr_mymod.pack"), None);
        assert_eq!(base_language_pack("local_french.pack"), None);

        // A manifest written before language support is English.
        let mut raw: serde_json::Value =
            serde_json::from_slice(&fs::read(i.dest.join(MANIFEST)).unwrap()).unwrap();
        raw.as_object_mut().unwrap().remove("language");
        let old: Manifest = serde_json::from_value(raw).unwrap();
        assert_eq!(old.language, "en");
    }
}
