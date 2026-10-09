//! Adapters: already merged, loose resources that TSV files cannot carry, bound to one exact dataset.
//!
//! An adapter is a folder of data-relative files that may only be custom UI (`ui/`), campaign scripts
//! (`script/`, the generated world script `campaigns/wld_napoleon/scripting.lua`) and an optional start
//! position template (`campaigns/eur_napoleon/startpos.esf`, a donor that is read and never shipped over the
//! original campaign). Database tables and localisation belong in the TSV dataset and packs are never
//! imported. The files are copied byte-exact into `<workspace>/adapters/<id>/`, hashed, made read-only and
//! bound to the dataset fingerprint they were merged for.

use super::scan::{self, Limits};
use super::{
    AdapterBinding, TargetBinding, check_id, check_tree, discard_tree, ensure_no_overlap,
    is_real_dir, load, secure_read, sha_hex, staging_dir, write_new,
};
use crate::model::Progress;
use crate::snapshot::{canon, validate_rel, validate_sha};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const ADAPTERS: &str = "adapters";
const ADAPTER_FORMAT: &str = "kgc-adapter";
const ADAPTER_VERSION: u32 = 2;
const ADAPTER_JSON: &str = "adapter.json";
const LIMITS: Limits = Limits {
    files: 20_000,
    file_bytes: 128 << 20,
    total_bytes: 512 << 20,
};
const PREFIXES: [&str; 2] = ["ui/", "script/"];
const FILES: [&str; 2] = [
    "campaigns/wld_napoleon/scripting.lua",
    "campaigns/eur_napoleon/startpos.esf",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterManifest {
    pub format: String,
    pub version: u32,
    pub id: String,
    pub fingerprint: String,
    pub target_id: String,
    pub target_fingerprint: String,
    pub files: Vec<AdapterFile>,
}

fn adapters_dir(workspace: &Path) -> PathBuf {
    workspace.join(ADAPTERS)
}

/// Whether a canonical path may be part of an adapter.
fn allowed(path: &str) -> bool {
    !path.ends_with(".pack")
        && (FILES.contains(&path) || PREFIXES.iter().any(|p| path.starts_with(p)))
}

fn fingerprint_of(m: &AdapterManifest) -> Result<String> {
    let mut c = m.clone();
    c.id.clear();
    c.fingerprint.clear();
    Ok(sha_hex(&serde_json::to_vec(&c)?))
}

fn stored(root: &Path, path: &str) -> PathBuf {
    let mut p = root.join("files");
    p.extend(path.split('/'));
    p
}

/// Copies a folder of merged resources into an immutable adapter bound to the exact fingerprint of
/// `binding`. Files outside the allowed paths, packs, links and unsafe names refuse the whole import.
pub fn import_adapter(
    workspace: &Path,
    binding: &TargetBinding,
    folder: &Path,
    progress: &Progress,
) -> Result<AdapterBinding> {
    let target = load(workspace, binding)?;
    let folder = scan::open_folder(folder, "adapter")?;
    let dir = adapters_dir(workspace);
    crate::safety::ensure_writable_path(&dir)?;
    ensure_no_overlap(&dir, &folder, "adapter")?;
    let tree = scan::read_tree(&folder, &|_: &str| true, &LIMITS, progress);
    let mut problems = tree.problems;
    let mut files = BTreeMap::new();
    for f in tree.files {
        let key = canon(&f.rel);
        if !allowed(&key) {
            problems.push(format!(
                "{} is not an adapter file; adapters hold only ui/, script/, {} and {} (database tables and text belong in the TSV dataset, packs are never imported)",
                f.rel, FILES[0], FILES[1]
            ));
            continue;
        }
        files.insert(key, f.bytes);
    }
    if problems.is_empty() && files.is_empty() {
        problems.push(format!(
            "the adapter folder {} contains no files",
            folder.display()
        ));
    }
    if !problems.is_empty() {
        problems.sort();
        bail!(
            "The adapter folder cannot be imported:\n - {}",
            problems.join("\n - ")
        );
    }

    let staging = staging_dir(&dir)?;
    let built = (|| -> Result<AdapterManifest> {
        let mut list = Vec::with_capacity(files.len());
        for (n, (path, bytes)) in files.iter().enumerate() {
            if n % 200 == 0 {
                progress(&format!(
                    "Importing adapter {}/{}: {path}",
                    n + 1,
                    files.len()
                ));
            }
            write_new(&stored(&staging, path), bytes)?;
            list.push(AdapterFile {
                path: path.clone(),
                size: bytes.len() as u64,
                sha256: sha_hex(bytes),
            });
        }
        let mut manifest = AdapterManifest {
            format: ADAPTER_FORMAT.to_owned(),
            version: ADAPTER_VERSION,
            id: String::new(),
            fingerprint: String::new(),
            target_id: target.manifest.id.clone(),
            target_fingerprint: target.manifest.fingerprint.clone(),
            files: list,
        };
        manifest.fingerprint = fingerprint_of(&manifest)?;
        manifest.id = format!("a{}", &manifest.fingerprint[..24]);
        write_new(
            &staging.join(ADAPTER_JSON),
            &serde_json::to_vec_pretty(&manifest)?,
        )?;
        Ok(manifest)
    })();
    let manifest = match built {
        Ok(m) => m,
        Err(e) => {
            discard_tree(&staging);
            return Err(e);
        }
    };
    let ab = AdapterBinding {
        id: manifest.id.clone(),
        fingerprint: manifest.fingerprint.clone(),
        target_fingerprint: manifest.target_fingerprint.clone(),
    };
    let final_dir = dir.join(&ab.id);
    if final_dir.exists() {
        discard_tree(&staging);
        adapter_files(workspace, &ab, binding).with_context(|| {
            format!(
                "adapter {} already exists but is damaged; it is never overwritten",
                ab.id
            )
        })?;
        return Ok(ab);
    }
    if let Err(e) = fs::rename(&staging, &final_dir) {
        discard_tree(&staging);
        return Err(anyhow!(e).context(format!("publish adapter {}", ab.id)));
    }
    progress(&format!("Adapter {} imported", ab.id));
    Ok(ab)
}

fn load_adapter(
    workspace: &Path,
    adapter: &AdapterBinding,
    binding: &TargetBinding,
) -> Result<(AdapterManifest, PathBuf)> {
    check_id(&adapter.id, 'a', "adapter")?;
    if adapter.target_fingerprint != binding.fingerprint {
        bail!(
            "adapter {} was imported for dataset fingerprint {} but the project uses {}; the adapter is stale, import it again for the current dataset",
            adapter.id,
            adapter.target_fingerprint,
            binding.fingerprint
        );
    }
    load(workspace, binding)?;
    let root = adapters_dir(workspace).join(&adapter.id);
    if !is_real_dir(&root) {
        bail!("adapter {} does not exist (or is a link)", adapter.id);
    }
    let raw = secure_read(&root.join(ADAPTER_JSON))?;
    let m: AdapterManifest = serde_json::from_slice(&raw)
        .with_context(|| format!("adapter {}: {ADAPTER_JSON} is not valid", adapter.id))?;
    if m.format != ADAPTER_FORMAT || m.version != ADAPTER_VERSION {
        bail!(
            "adapter {}: unsupported format {} v{}",
            adapter.id,
            m.format,
            m.version
        );
    }
    if fingerprint_of(&m)? != m.fingerprint
        || m.fingerprint != adapter.fingerprint
        || m.id != adapter.id
        || m.target_fingerprint != adapter.target_fingerprint
        || m.target_fingerprint != binding.fingerprint
        || m.target_id != binding.id
    {
        bail!(
            "adapter {} does not match its binding: it was modified or belongs to another dataset",
            adapter.id
        );
    }
    let mut listed: HashSet<String> = HashSet::from([ADAPTER_JSON.to_owned()]);
    let mut last: Option<&str> = None;
    for f in &m.files {
        validate_rel(&f.path)?;
        validate_sha(&f.sha256)?;
        if f.path != canon(&f.path)
            || !allowed(&f.path)
            || last.is_some_and(|l| l >= f.path.as_str())
        {
            bail!(
                "adapter {}: file list is not canonical, sorted and limited to adapter paths",
                adapter.id
            );
        }
        last = Some(&f.path);
        listed.insert(format!("files/{}", f.path));
    }
    check_tree(&root, &listed, "adapter")?;
    Ok((m, root))
}

/// The recorded file list (canonical lower-case `/` data-relative paths with size and SHA-256) of an adapter
/// bound to exactly this dataset. Contents are verified by `adapter_files`.
pub fn adapter_manifest(
    workspace: &Path,
    adapter: &AdapterBinding,
    binding: &TargetBinding,
) -> Result<AdapterManifest> {
    Ok(load_adapter(workspace, adapter, binding)?.0)
}

/// All adapter files keyed by canonical lower-case `/` data-relative path, hash-verified, after checking
/// that the adapter is bound to exactly this dataset fingerprint.
pub fn adapter_files(
    workspace: &Path,
    adapter: &AdapterBinding,
    binding: &TargetBinding,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let (m, root) = load_adapter(workspace, adapter, binding)?;
    let mut out = BTreeMap::new();
    for f in &m.files {
        let bytes = secure_read(&stored(&root, &f.path))?;
        if bytes.len() as u64 != f.size || sha_hex(&bytes) != f.sha256 {
            bail!(
                "adapter {} file {} is corrupt: it no longer matches its recorded hash",
                adapter.id,
                f.path
            );
        }
        out.insert(f.path.clone(), bytes);
    }
    Ok(out)
}
