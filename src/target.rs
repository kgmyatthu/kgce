//! Immutable TSV target datasets: the modded Napoleon: Total War database a new campaign is ported into.
//!
//! A mod author exports its database tables and localisation with RPFM as TSV files. `import` reads one
//! folder of such files, decodes every file against the NTW schema (`crate::tsv`) and stores the result
//! under `<workspace>/targets/<id>/`:
//!
//! * `target.json`      the manifest (name, language, per-file hashes and member identities); `id` and
//!   `fingerprint` derive from its content, so any different input is a NEW dataset;
//! * `tsv/NNNNN.tsv`    byte-exact, read-only copies of the imported TSV files;
//! * `members/NNNNN.bin` the game-format member each TSV decodes to (`db/<table>/<member>`,
//!   `text/<name>.loc`), frozen at import time.
//!
//! The dataset is authoritative and complete: its tables are NOT an additive overlay. `effective_snapshot`
//! hides every vanilla NTW database member and localisation file, so a table the TSVs do not provide is
//! missing, never silently taken from the original game. Several TSV files may provide members of the same
//! table (distinct member names are kept apart so conflicts stay visible); two files naming the same member
//! are refused. The import folder is only ever read; later edits to it never change an imported dataset.
//!
//! TSV files carry no campaign, UI or script data. `ImportSpec::resources_reviewed` records that the user
//! knows resources that were not supplied use the original conversion templates; such resources can be
//! supplied separately as an adapter (`import_adapter`), a folder of already merged files bound to the exact
//! dataset fingerprint.
//!
//! Import asks for only the existing-faction campaign inputs the conversion needs (`compatibility::required_tables`) plus
//! the two full localisation files. These are conversion INPUTS (analysis catalog, unit stats for force generation, ...),
//! not an export manifest: the export contains only the campaign additions/overrides, explicit conflict-resolution
//! replacements and the merged localisation, and the installed target mod supplies every unchanged table (land/naval
//! unit stats, existing faction identities, ...). Optional tables a folder also supplies are imported and retained for
//! analysis only when asked; they are not required, are not forced into the export, and the references of the rows a
//! dataset provides never become further import requirements.
//!
//! Nothing here writes outside `<workspace>/targets` and `<workspace>/adapters`.

mod adapter;
mod scan;

pub use adapter::{AdapterFile, AdapterManifest, adapter_files, adapter_manifest, import_adapter};

use crate::compatibility;
use crate::db;
use crate::model::{Progress, Rules};
use crate::snapshot::{
    Game, Snapshot, canon, canonical_lenient, hash_file, is_inside, set_readonly, validate_rel,
    validate_sha,
};
use crate::tsv::{self, Document};
use anyhow::{Context, Result, anyhow, bail};
use scan::{Limits, RawFile};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

const FORMAT: &str = "kgc-target";
const VERSION: u32 = 1;
const TARGETS: &str = "targets";
const TARGET_JSON: &str = "target.json";

/// Localisation members every modded target must provide in full: the converter never substitutes base text.
pub const REQUIRED_LOC: [&str; 2] = ["text/localisation.loc", "text/ui.loc"];

const LIMITS: Limits = Limits {
    files: 50_000,
    file_bytes: 256 << 20,
    total_bytes: 1 << 30,
};

// ---------------------------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------------------------

/// The dataset a project is bound to. Ids start with `t`; pack-profile ids of older versions never match.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetBinding {
    pub id: String,
    pub fingerprint: String,
}

/// Merged resources bound to exactly one dataset fingerprint.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterBinding {
    pub id: String,
    pub fingerprint: String,
    pub target_fingerprint: String,
}

/// The user's selection of a TSV folder.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportSpec {
    pub name: String,
    /// Two/three letter language code (`en`, `fr`...) of the text in the localisation TSVs.
    pub language: String,
    pub folder: PathBuf,
    /// The user confirmed that resources the TSVs cannot carry (campaign, UI, scripts) and that are not
    /// supplied as an adapter use the original conversion templates. Export is blocked while false.
    #[serde(default)]
    pub resources_reviewed: bool,
    /// Also import the optional tables the folder provides, retained as analysis inputs only (never required, not
    /// exported unless the build mutates them). Off: only the required tables and the localisation files enter the
    /// dataset, the rest of the folder is ignored.
    #[serde(default)]
    pub extras: bool,
}

/// A table (database directory such as `units_tables`, or a localisation member such as
/// `text/ui.loc`) the conversion needs from the target, with the reason.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Requirement {
    pub table: String,
    pub reason: String,
}

/// What `import` would do, without writing anything.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Preview {
    /// TSV files that were read.
    pub files: usize,
    /// Database directories and localisation members the folder provides, sorted.
    pub tables: Vec<String>,
    /// Everything the conversion needs: database tables plus the two full localisation files.
    pub required: Vec<Requirement>,
    /// Required names the folder does not provide.
    pub missing: Vec<String>,
    /// Errors that make `import` refuse the folder (unreadable files, field errors, duplicates...).
    pub problems: Vec<String>,
    pub warnings: Vec<String>,
    /// Optional tables the folder provides that `import` leaves out (required-only import).
    #[serde(default)]
    pub ignored: Vec<String>,
}

impl Preview {
    /// Provided tables (lower-case names) that are not required: optional, retained for analysis only.
    pub fn extras(&self) -> Vec<&str> {
        let required: BTreeSet<String> = self
            .required
            .iter()
            .map(|r| r.table.to_lowercase())
            .collect();
        self.tables
            .iter()
            .filter(|t| !required.contains(t.as_str()))
            .map(String::as_str)
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Summary {
    pub binding: TargetBinding,
    pub name: String,
    pub language: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Listing {
    pub datasets: Vec<Summary>,
    /// One message per dataset folder that could not be read or failed its integrity check.
    pub problems: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MemberKind {
    Db,
    Loc,
}

/// One imported TSV file and the member it decodes to.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFile {
    /// Path inside the import folder, `/` separators, stored capitalisation.
    pub file: String,
    pub size: u64,
    pub sha256: String,
    pub kind: MemberKind,
    /// Canonical lower-case data-relative member path.
    pub member: String,
    pub rows: usize,
    pub member_size: u64,
    pub member_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub id: String,
    pub name: String,
    pub language: String,
    pub fingerprint: String,
    pub resources_reviewed: bool,
    /// Only required tables (and localisation) were imported; folder files of other tables are not drift.
    #[serde(default)]
    pub required_only: bool,
    pub created: u64,
    /// Sorted by lower-case `file`.
    pub files: Vec<DataFile>,
    pub warnings: Vec<String>,
    /// Absolute folder the files were imported from (local only: used by `Dataset::changes`, never exported).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Small helpers shared with the adapter module
// ---------------------------------------------------------------------------------------------

fn sha_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn targets_dir(workspace: &Path) -> PathBuf {
    workspace.join(TARGETS)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `prefix` + lower-case alphanumerics only: ids become folder names.
fn check_id(id: &str, prefix: char, what: &str) -> Result<()> {
    if id.len() < 2
        || id.len() > 64
        || !id.starts_with(prefix)
        || !id
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
    {
        bail!("{id:?} is not a valid {what} id (ids of this editor start with '{prefix}')");
    }
    Ok(())
}

/// Reads a regular file only (links and folders are refused).
fn secure_read(path: &Path) -> Result<Vec<u8>> {
    let meta =
        fs::symlink_metadata(path).with_context(|| format!("{} is missing", path.display()))?;
    if !meta.is_file() {
        bail!(
            "{} is not a regular file (links and folders are refused)",
            path.display()
        );
    }
    fs::read(path).with_context(|| format!("read {}", path.display()))
}

fn is_real_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|m| m.is_dir())
}

fn unlock_tree(dir: &Path) {
    for e in WalkDir::new(dir).follow_links(false).into_iter().flatten() {
        if e.file_type().is_file() {
            if let Ok(meta) = fs::metadata(e.path()) {
                let mut perm = meta.permissions();
                perm.set_readonly(false);
                let _ = fs::set_permissions(e.path(), perm);
            }
        }
    }
}

fn discard_tree(dir: &Path) {
    unlock_tree(dir);
    let _ = fs::remove_dir_all(dir);
}

/// The destination must neither contain nor sit inside the selected source.
fn ensure_no_overlap(dest: &Path, source: &Path, what: &str) -> Result<()> {
    let d = canonical_lenient(dest)?;
    let s = canonical_lenient(source)?;
    if is_inside(&d, &s) || is_inside(&s, &d) {
        bail!(
            "{what} destination {} overlaps the selected folder {}; inputs are read-only and the editor workspace must live outside them",
            dest.display(),
            source.display()
        );
    }
    Ok(())
}

fn staging_dir(parent: &Path) -> Result<PathBuf> {
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let staging = parent.join(format!(".importing-{}-{stamp}", std::process::id()));
    fs::create_dir(&staging).with_context(|| format!("create {}", staging.display()))?;
    Ok(staging)
}

/// Writes a new read-only file (never overwrites) and returns nothing; parents are created.
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(
        path.parent()
            .ok_or_else(|| anyhow!("no parent for {}", path.display()))?,
    )?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    out.write_all(bytes)?;
    out.flush()?;
    drop(out);
    let (len, sha) = hash_file(path, &mut vec![0u8; 1 << 16])?;
    if len != bytes.len() as u64 || sha != sha_hex(bytes) {
        bail!(
            "{}: the written copy differs from its source",
            path.display()
        );
    }
    set_readonly(path)
}

/// Every file below `root` must be listed in `allowed` (lower-case `/` relative paths); no links.
fn check_tree(root: &Path, allowed: &HashSet<String>, what: &str) -> Result<()> {
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry?;
        let ty = entry.file_type();
        if ty.is_symlink() {
            bail!("{what} contains a link at {}", entry.path().display());
        }
        if !ty.is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/")
            .to_lowercase();
        if !allowed.contains(&rel) {
            bail!(
                "{what} contains an unlisted file {}",
                entry.path().display()
            );
        }
    }
    Ok(())
}

fn valid_language(code: &str) -> bool {
    matches!(code.len(), 2 | 3) && code.bytes().all(|c| c.is_ascii_lowercase())
}

fn fingerprint_of(m: &Manifest) -> Result<String> {
    let mut c = m.clone();
    c.id.clear();
    c.fingerprint.clear();
    c.created = 0;
    c.source = None;
    Ok(sha_hex(&serde_json::to_vec(&c)?))
}

fn id_from(fingerprint: &str) -> String {
    format!("t{}", &fingerprint[..24])
}

fn tsv_name(index: usize) -> String {
    format!("tsv/{index:05}.tsv")
}

fn member_name(index: usize) -> String {
    format!("members/{index:05}.bin")
}

/// The only member shapes a dataset holds: `db/<table>_tables/<name>` and `text/<name>.loc`.
fn check_member(kind: MemberKind, member: &str) -> Result<()> {
    let key = canon(member);
    validate_rel(&key)?;
    let parts: Vec<&str> = key.split('/').collect();
    let ok = match (kind, parts.as_slice()) {
        (MemberKind::Db, ["db", dir, name]) => {
            dir.len() > "_tables".len() && dir.ends_with("_tables") && !name.is_empty()
        }
        (MemberKind::Loc, ["text", name]) => name.len() > ".loc".len() && name.ends_with(".loc"),
        _ => false,
    };
    if !ok {
        bail!(
            "{member:?} is not a database member db/<table>_tables/<name> or a localisation member text/<name>.loc"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Reading and analysing the folder
// ---------------------------------------------------------------------------------------------

struct Entry {
    file: String,
    tsv: Vec<u8>,
    kind: MemberKind,
    member: String,
    rows: usize,
    encoded: Vec<u8>,
}

/// Decodes every TSV; one problem per file that cannot be used, one per member provided twice.
fn interpret(files: Vec<RawFile>, schema: &db::Schema, problems: &mut Vec<String>) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::with_capacity(files.len());
    let mut owner: HashMap<String, String> = HashMap::new();
    for f in files {
        let decoded = tsv::decode(&f.rel, &f.bytes, schema).and_then(|doc| match doc {
            Document::Db { member, table } => {
                check_member(MemberKind::Db, &member).with_context(|| f.rel.clone())?;
                let key = canon(&member);
                let dir = key.split('/').nth(1).unwrap_or_default();
                if !table.name.eq_ignore_ascii_case(dir) {
                    bail!(
                        "{}: table {} does not match its member {member}",
                        f.rel,
                        table.name
                    );
                }
                let encoded = db::encode(&table).with_context(|| f.rel.clone())?;
                Ok((MemberKind::Db, key, table.rows.len(), encoded))
            }
            Document::Loc {
                member,
                version,
                rows,
            } => {
                check_member(MemberKind::Loc, &member).with_context(|| f.rel.clone())?;
                let encoded = db::encode_loc(&rows, version).with_context(|| f.rel.clone())?;
                Ok((MemberKind::Loc, canon(&member), rows.len(), encoded))
            }
        });
        let (kind, member, rows, encoded) = match decoded {
            Ok(v) => v,
            Err(e) => {
                problems.push(format!("{e:#}"));
                continue;
            }
        };
        if let Some(first) = owner.insert(member.clone(), f.rel.clone()) {
            problems.push(format!(
                "{first} and {} both provide {member}; export each table member only once",
                f.rel
            ));
            continue;
        }
        out.push(Entry {
            file: f.rel,
            tsv: f.bytes,
            kind,
            member,
            rows,
            encoded,
        });
    }
    out
}

struct Analysis {
    entries: Vec<Entry>,
    preview: Preview,
    /// Import-time observations that are part of the stored dataset (analysis-dependent warnings are not).
    notes: Vec<String>,
    /// Absolute input folder when it could be opened.
    source: Option<PathBuf>,
}

fn list_some(names: &[String]) -> String {
    let more = names.len().saturating_sub(5);
    let head = names.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    if more > 0 {
        format!("{head} and {more} more")
    } else {
        head
    }
}

/// The existing-faction campaign inputs plus the two full localisation members, sorted by name. Nothing is
/// derived from the rows a dataset provides.
pub fn required_inputs(base: &Snapshot, rules: &Rules) -> Result<Vec<Requirement>> {
    let mut required = compatibility::required_tables(base, rules)?;
    add_required_loc(&mut required);
    Ok(required)
}

fn add_required_loc(required: &mut Vec<Requirement>) {
    for loc in REQUIRED_LOC {
        required.push(Requirement {
            table: loc.to_owned(),
            reason: "full target localisation: base game text is never used as a stand-in for the target's text".into(),
        });
    }
    required.sort_by(|a, b| a.table.cmp(&b.table));
}

/// Required names no provided table matches. `provided` holds lower-case directory and member names.
fn missing_tables(required: &[Requirement], provided: &BTreeSet<String>) -> Vec<String> {
    required
        .iter()
        .filter(|r| !provided.contains(&r.table.to_lowercase()))
        .map(|r| r.table.clone())
        .collect()
}

fn analyze(
    workspace: &Path,
    assets: &Path,
    spec: &ImportSpec,
    progress: &Progress,
) -> Result<Analysis> {
    let sources = workspace.join("sources");
    let base = Snapshot::open(&sources).with_context(|| {
        format!(
            "the original game snapshot {} is needed to know which tables the conversion requires; import the original games first",
            sources.display()
        )
    })?;
    let rules = Rules::load(&assets.join("rules"))?;
    let schema = db::schema(&rules, Game::Ntw)?;
    let mut preview = Preview::default();
    preview.required = required_inputs(&base, &rules)?;

    let name = spec.name.trim();
    if name.is_empty() || name.chars().count() > 100 || name.chars().any(|c| c.is_control()) {
        preview
            .problems
            .push("the dataset needs a name of 1-100 printable characters".into());
    }
    if !valid_language(spec.language.trim().to_lowercase().as_str()) {
        preview.problems.push(format!(
            "language {:?} is not a language code such as en, fr or de",
            spec.language
        ));
    }

    let mut entries = Vec::new();
    let mut source = None;
    let mut notes: Vec<String> = Vec::new();
    match scan::open_folder(&spec.folder, "TSV") {
        Err(e) => preview.problems.push(format!("{e:#}")),
        Ok(folder) => {
            if let Err(e) = ensure_no_overlap(&targets_dir(workspace), &folder, "dataset") {
                preview.problems.push(format!("{e:#}"));
            }
            progress(&format!("Reading {}", folder.display()));
            let tree = scan::read_tree(
                &folder,
                &|key: &str| key.ends_with(".tsv"),
                &LIMITS,
                progress,
            );
            preview.problems.extend(tree.problems);
            if !tree.ignored.is_empty() {
                let note = format!(
                    "{} file(s) that are not .tsv were ignored: {}",
                    tree.ignored.len(),
                    list_some(&tree.ignored)
                );
                notes.push(note.clone());
                preview.warnings.push(note);
            }
            if tree.files.is_empty() && preview.problems.is_empty() {
                preview
                    .problems
                    .push("the folder contains no .tsv files".into());
            }
            preview.files = tree.files.len();
            progress(&format!("Decoding {} TSV files", tree.files.len()));
            entries = interpret(tree.files, &schema, &mut preview.problems);
            if !spec.extras {
                let ignored = keep_required(&mut entries, &preview.required);
                if !ignored.is_empty() {
                    let note = format!(
                        "{} optional table(s) in the folder are not imported (required tables only): {}",
                        ignored.len(),
                        list_some(&ignored)
                    );
                    notes.push(note.clone());
                    preview.warnings.push(note);
                }
                preview.ignored = ignored;
            }
            source = Some(folder);
        }
    }

    let mut tables: BTreeSet<String> = BTreeSet::new();
    for e in &entries {
        tables.insert(match e.kind {
            MemberKind::Db => e.member.split('/').nth(1).unwrap_or_default().to_owned(),
            MemberKind::Loc => e.member.clone(),
        });
    }
    preview.missing = missing_tables(&preview.required, &tables);
    preview.tables = tables.into_iter().collect();
    if !preview.missing.is_empty() && !entries.is_empty() {
        preview.warnings.push(format!(
            "{} required table(s) are not provided: {}; the conversion is blocked until a dataset provides them, vanilla tables are never used instead",
            preview.missing.len(),
            list_some(&preview.missing)
        ));
    }
    if !spec.resources_reviewed {
        preview.warnings.push(
            "resources are not confirmed: TSV files carry no campaign, UI or script data; confirm that resources you do not supply as an adapter use the original conversion templates, otherwise export stays blocked".into(),
        );
    }
    preview.problems.sort();
    preview.problems.dedup();
    Ok(Analysis {
        entries,
        preview,
        notes,
        source,
    })
}

/// Drops the database entries of tables that are not required (localisation entries always stay) and returns
/// the ignored table names, sorted.
fn keep_required(entries: &mut Vec<Entry>, required: &[Requirement]) -> Vec<String> {
    let required: BTreeSet<String> = required.iter().map(|r| r.table.to_lowercase()).collect();
    let mut ignored: BTreeSet<String> = BTreeSet::new();
    entries.retain(|e| match e.kind {
        MemberKind::Loc => true,
        MemberKind::Db => {
            let table = e.member.split('/').nth(1).unwrap_or_default().to_lowercase();
            let keep = required.contains(&table);
            if !keep {
                ignored.insert(table);
            }
            keep
        }
    });
    ignored.into_iter().collect()
}

/// Reads and decodes the selected folder (nothing is written) and reports the required and missing
/// tables, field errors and warnings. Errors only when the workspace itself cannot answer (no source
/// snapshot, unreadable rules).
pub fn preview(
    workspace: &Path,
    assets: &Path,
    spec: &ImportSpec,
    progress: &Progress,
) -> Result<Preview> {
    Ok(analyze(workspace, assets, spec, progress)?.preview)
}

// ---------------------------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------------------------

/// Creates a NEW immutable dataset from the folder (never overwrites: identical content yields the
/// existing dataset, anything different gets a new id). Refuses malformed or incomplete datasets:
/// every required DB table and both full localisation members must be supplied.
pub fn import(
    workspace: &Path,
    assets: &Path,
    spec: &ImportSpec,
    progress: &Progress,
) -> Result<TargetBinding> {
    let an = analyze(workspace, assets, spec, progress)?;
    let mut blockers = an.preview.problems.clone();
    blockers.extend(an.preview.missing.iter().map(|m| {
        format!("{m} is required: the dataset never falls back to original database or localisation rows")
    }));
    if !blockers.is_empty() {
        bail!(
            "The TSV folder cannot be imported:\n - {}",
            blockers.join("\n - ")
        );
    }
    publish(workspace, spec, &an, progress)
}

/// Stores an analysed folder as a new dataset (or returns the identical existing one).
fn publish(
    workspace: &Path,
    spec: &ImportSpec,
    an: &Analysis,
    progress: &Progress,
) -> Result<TargetBinding> {
    let source = an
        .source
        .as_ref()
        .ok_or_else(|| anyhow!("no input folder"))?;
    let dir = targets_dir(workspace);
    crate::safety::ensure_writable_path(&dir)?;
    ensure_no_overlap(&dir, source, "dataset")?;

    let staging = staging_dir(&dir)?;
    let manifest = match build(&staging, spec, an, source, progress) {
        Ok(m) => m,
        Err(e) => {
            discard_tree(&staging);
            return Err(e);
        }
    };
    let binding = TargetBinding {
        id: manifest.id.clone(),
        fingerprint: manifest.fingerprint.clone(),
    };
    let final_dir = dir.join(&binding.id);
    if final_dir.exists() {
        discard_tree(&staging);
        load(workspace, &binding)
            .and_then(|d| d.verify(progress))
            .with_context(|| {
                format!(
                    "dataset {} already exists but is damaged; it is never overwritten",
                    binding.id
                )
            })?;
        progress("An identical dataset already exists");
        return Ok(binding);
    }
    if let Err(e) = fs::rename(&staging, &final_dir) {
        discard_tree(&staging);
        return Err(anyhow!(e).context(format!("publish dataset {}", binding.id)));
    }
    progress(&format!("Dataset {} imported", binding.id));
    Ok(binding)
}

fn build(
    staging: &Path,
    spec: &ImportSpec,
    an: &Analysis,
    source: &Path,
    progress: &Progress,
) -> Result<Manifest> {
    let mut order: Vec<&Entry> = an.entries.iter().collect();
    order.sort_by_key(|e| e.file.to_lowercase());
    let mut files = Vec::with_capacity(order.len());
    for (i, e) in order.iter().enumerate() {
        if i % 200 == 0 {
            progress(&format!("Storing {}/{}: {}", i + 1, order.len(), e.file));
        }
        write_new(&staging.join(tsv_name(i)), &e.tsv)?;
        write_new(&staging.join(member_name(i)), &e.encoded)?;
        files.push(DataFile {
            file: e.file.clone(),
            size: e.tsv.len() as u64,
            sha256: sha_hex(&e.tsv),
            kind: e.kind,
            member: e.member.clone(),
            rows: e.rows,
            member_size: e.encoded.len() as u64,
            member_sha256: sha_hex(&e.encoded),
        });
    }
    let mut manifest = Manifest {
        format: FORMAT.to_owned(),
        version: VERSION,
        id: String::new(),
        name: spec.name.trim().to_owned(),
        language: spec.language.trim().to_lowercase(),
        fingerprint: String::new(),
        resources_reviewed: spec.resources_reviewed,
        required_only: !spec.extras,
        created: now_secs(),
        files,
        warnings: an.notes.clone(),
        source: Some(source.to_string_lossy().into_owned()),
    };
    manifest.fingerprint = fingerprint_of(&manifest)?;
    manifest.id = id_from(&manifest.fingerprint);
    write_new(
        &staging.join(TARGET_JSON),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

// ---------------------------------------------------------------------------------------------
// Listing and loading
// ---------------------------------------------------------------------------------------------

fn read_manifest(root: &Path, id: &str) -> Result<Manifest> {
    check_id(id, 't', "dataset")?;
    if !is_real_dir(root) {
        bail!("dataset {id} does not exist (or is a link)");
    }
    let raw = secure_read(&root.join(TARGET_JSON))?;
    let m: Manifest = serde_json::from_slice(&raw)
        .with_context(|| format!("dataset {id}: {TARGET_JSON} is not valid"))?;
    if m.format != FORMAT || m.version != VERSION {
        bail!(
            "dataset {id}: unsupported format {} v{}",
            m.format,
            m.version
        );
    }
    validate_sha(&m.fingerprint)?;
    if fingerprint_of(&m)? != m.fingerprint {
        bail!(
            "dataset {id} was modified after import: its content no longer matches its fingerprint"
        );
    }
    if m.id != id || id_from(&m.fingerprint) != id {
        bail!("dataset folder {id} holds a different dataset ({})", m.id);
    }
    let mut files_seen = BTreeSet::new();
    let mut members_seen = BTreeSet::new();
    let mut last: Option<String> = None;
    for (i, f) in m.files.iter().enumerate() {
        let key = f.file.to_lowercase();
        validate_rel(&key).with_context(|| format!("dataset {id}: file {i}"))?;
        validate_sha(&f.sha256)?;
        validate_sha(&f.member_sha256)?;
        check_member(f.kind, &f.member)
            .with_context(|| format!("dataset {id}: file {}", f.file))?;
        if f.member != canon(&f.member) {
            bail!("dataset {id}: member {:?} is not canonical", f.member);
        }
        if last.as_ref().is_some_and(|l| *l >= key) || !files_seen.insert(key.clone()) {
            bail!("dataset {id}: file list is not strictly sorted and unique");
        }
        if !members_seen.insert(f.member.clone()) {
            bail!("dataset {id}: member {} is provided twice", f.member);
        }
        last = Some(key);
    }
    Ok(m)
}

/// Every dataset in the workspace; damaged folders are reported in `problems` instead of failing.
pub fn list(workspace: &Path) -> Result<Listing> {
    let dir = targets_dir(workspace);
    let mut out = Listing::default();
    if !dir.exists() {
        return Ok(out);
    }
    let mut found: Vec<(String, u64, Summary)> = Vec::new();
    for e in fs::read_dir(&dir).with_context(|| format!("list {}", dir.display()))? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue; // interrupted import staging
        }
        match read_manifest(&e.path(), &name) {
            Ok(m) => found.push((
                m.name.to_lowercase(),
                m.created,
                Summary {
                    binding: TargetBinding {
                        id: m.id,
                        fingerprint: m.fingerprint,
                    },
                    name: m.name,
                    language: m.language,
                },
            )),
            Err(err) => out.problems.push(format!("{name}: {err:#}")),
        }
    }
    found.sort_by(|a, b| (&a.0, a.1, &a.2.binding.id).cmp(&(&b.0, b.1, &b.2.binding.id)));
    out.datasets = found.into_iter().map(|f| f.2).collect();
    out.problems.sort();
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub manifest: Manifest,
    root: PathBuf,
}

/// Loads a dataset (manifest and file presence; contents are hashed by `verify` and on use). The binding
/// must match the stored fingerprint exactly: a replaced dataset is reported as stale, never accepted.
pub fn load(workspace: &Path, binding: &TargetBinding) -> Result<Dataset> {
    check_id(&binding.id, 't', "dataset")?;
    let root = targets_dir(workspace).join(&binding.id);
    let manifest = read_manifest(&root, &binding.id)?;
    if manifest.fingerprint != binding.fingerprint {
        bail!(
            "dataset {} is stale: the project is bound to fingerprint {} but the stored dataset is {}; select the dataset again",
            binding.id,
            binding.fingerprint,
            manifest.fingerprint
        );
    }
    for (i, f) in manifest.files.iter().enumerate() {
        for (rel, size) in [(tsv_name(i), f.size), (member_name(i), f.member_size)] {
            let p = root.join(&rel);
            let meta = fs::symlink_metadata(&p)
                .with_context(|| format!("dataset {}: {rel} is missing", binding.id))?;
            if !meta.is_file() || meta.len() != size {
                bail!("dataset {}: {rel} is corrupt (size)", binding.id);
            }
        }
    }
    Ok(Dataset { manifest, root })
}

impl Dataset {
    pub fn binding(&self) -> TargetBinding {
        TargetBinding {
            id: self.manifest.id.clone(),
            fingerprint: self.manifest.fingerprint.clone(),
        }
    }

    /// Portable dependency lock for export: no absolute host paths; names, hashes and member identities of
    /// every imported TSV file (the imported inputs, not the files shipped in the pack).
    pub fn export_manifest(&self) -> Value {
        let files: Vec<Value> = self
            .manifest
            .files
            .iter()
            .map(|f| {
                json!({
                    "file": f.file,
                    "size": f.size,
                    "sha256": f.sha256,
                    "kind": f.kind,
                    "member": f.member,
                    "rows": f.rows,
                    "member_sha256": f.member_sha256,
                })
            })
            .collect();
        json!({
            "format": "kgc-target-lock",
            "version": VERSION,
            "target_id": self.manifest.id,
            "fingerprint": self.manifest.fingerprint,
            "name": self.manifest.name,
            "language": self.manifest.language,
            "resources_reviewed": self.manifest.resources_reviewed,
            "warnings": self.manifest.warnings,
            "files": files,
        })
    }

    fn read_checked(&self, rel: &str, size: u64, sha: &str) -> Result<Vec<u8>> {
        let bytes = secure_read(&self.root.join(rel))?;
        if bytes.len() as u64 != size || sha_hex(&bytes) != sha {
            bail!(
                "dataset {}: {rel} no longer matches the hash recorded at import; the dataset is corrupt",
                self.manifest.id
            );
        }
        Ok(bytes)
    }

    /// `base` as the target sees the game: every vanilla NTW database member and localisation file is hidden
    /// and the dataset's members are added under their own canonical paths (priority 1, origin
    /// `TSV <file>`). A table the dataset does not provide does not exist: there is no vanilla fallback.
    /// Everything else (campaign, UI, models...) stays the original game's. Bytes are verified against
    /// the import hashes.
    pub fn effective_snapshot(&self, base: &Snapshot) -> Result<Snapshot> {
        let mut members: BTreeMap<String, (Vec<u8>, String)> = BTreeMap::new();
        for (i, f) in self.manifest.files.iter().enumerate() {
            let bytes = self.read_checked(&member_name(i), f.member_size, &f.member_sha256)?;
            members.insert(f.member.clone(), (bytes, format!("TSV {}", f.file)));
        }
        for loc in REQUIRED_LOC {
            if !members.contains_key(loc) {
                bail!(
                    "dataset {} does not provide {loc}; the target's full text is required and base game text is never used instead",
                    self.manifest.id
                );
            }
        }
        let mut hidden: BTreeSet<String> = BTreeSet::new();
        for prefix in ["db/", "text/"] {
            for name in base
                .names_in_source(Game::Ntw, prefix)
                .into_iter()
                .chain(base.names(Game::Ntw, prefix))
            {
                if (prefix == "db/" || name.ends_with(".loc")) && !members.contains_key(&name) {
                    hidden.insert(name);
                }
            }
        }
        base.with_ntw_members(members, hidden.into_iter().collect())
    }

    /// Integrity of everything stored inside the dataset: manifest, every TSV copy and member; no
    /// unlisted file or link. The import folder is not read here (see `changes`).
    pub fn verify(&self, progress: &Progress) -> Result<()> {
        let fresh = read_manifest(&self.root, &self.manifest.id)?;
        if serde_json::to_value(&fresh)? != serde_json::to_value(&self.manifest)? {
            bail!(
                "dataset {} changed on disk since it was loaded",
                self.manifest.id
            );
        }
        let mut allowed: HashSet<String> = HashSet::from([TARGET_JSON.to_owned()]);
        let total = self.manifest.files.len();
        for (i, f) in self.manifest.files.iter().enumerate() {
            if i % 200 == 0 {
                progress(&format!("Verifying dataset {}/{total}: {}", i + 1, f.file));
            }
            self.read_checked(&tsv_name(i), f.size, &f.sha256)?;
            self.read_checked(&member_name(i), f.member_size, &f.member_sha256)?;
            allowed.insert(tsv_name(i));
            allowed.insert(member_name(i));
        }
        check_tree(&self.root, &allowed, "dataset")?;
        progress("Dataset verified");
        Ok(())
    }

    /// Compares the folder the dataset was imported from with the imported copies. Nothing is written and
    /// the dataset stays authoritative either way: an empty list means the folder is unchanged; edits,
    /// additions and removals are described one by one.
    pub fn changes(&self) -> Result<Vec<String>> {
        let Some(src) = &self.manifest.source else {
            return Ok(vec![
                "the folder this dataset was imported from was not recorded; it cannot be compared"
                    .into(),
            ]);
        };
        let root = match scan::open_folder(Path::new(src), "TSV") {
            Ok(r) => r,
            Err(e) => {
                return Ok(vec![format!(
                    "the import folder is missing or unreadable ({e:#}); the imported copy is unaffected"
                )]);
            }
        };
        let tree = scan::read_tree(
            &root,
            &|k: &str| k.ends_with(".tsv"),
            &LIMITS,
            &|_: &str| {},
        );
        let now: BTreeMap<String, &RawFile> = tree
            .files
            .iter()
            .map(|f| (f.rel.to_lowercase(), f))
            .collect();
        let mut out = Vec::new();
        let mut added = Vec::new();
        let mut changed = Vec::new();
        let mut removed = Vec::new();
        let mut known = HashSet::new();
        for f in &self.manifest.files {
            let key = f.file.to_lowercase();
            known.insert(key.clone());
            match now.get(&key) {
                None => removed.push(f.file.clone()),
                Some(n) if n.bytes.len() as u64 != f.size || sha_hex(&n.bytes) != f.sha256 => {
                    changed.push(f.file.clone())
                }
                Some(_) => {}
            }
        }
        let tables: HashSet<String> = self
            .manifest
            .files
            .iter()
            .filter_map(|f| f.member.split('/').nth(1).map(str::to_lowercase))
            .collect();
        for (key, f) in &now {
            if !known.contains(key) {
                // required-only datasets ignore the folder's optional tables by design
                if self.manifest.required_only
                    && crate::tsv::declared_table(&f.bytes).is_some_and(|t| !tables.contains(&t))
                {
                    continue;
                }
                added.push(f.rel.clone());
            }
        }
        for (what, list) in [("changed", changed), ("added", added), ("removed", removed)] {
            if !list.is_empty() {
                out.push(format!(
                    "{} TSV file(s) {what} in the import folder since import: {}",
                    list.len(),
                    list_some(&list)
                ));
            }
        }
        out.extend(
            tree.problems
                .into_iter()
                .map(|p| format!("import folder problem: {p}")),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
