//! Rebuild conversion resources from the user's verified originals, never from shipped replacement assets.
mod rules;
mod ui;

use crate::{
    model::Progress,
    snapshot::{self, Snapshot},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use walkdir::WalkDir;

// Bump whenever the generators' semantics change. Terrain has its own source-bound cache.
const RECIPE: u32 = 1;
const MANIFEST: &str = "local-assets.json";

#[derive(Serialize, Deserialize, PartialEq)]
struct Artifact {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    recipe: u32,
    fingerprint: String,
    files: Vec<Artifact>,
}

fn files(root: &Path, prefixes: &[&str], buffer: &mut [u8]) -> Result<Vec<Artifact>> {
    let mut paths = Vec::new();
    for prefix in prefixes {
        let path = root.join(prefix);
        for entry in WalkDir::new(&path).follow_links(false) {
            let entry =
                entry.with_context(|| format!("Read recipe resources {}", path.display()))?;
            if entry.file_type().is_symlink() {
                bail!(
                    "Local asset resources must not be symbolic links: {}",
                    entry.path().display()
                );
            }
            if entry.file_type().is_file() {
                paths.push(entry.into_path());
            }
        }
    }
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let (size, sha256) = snapshot::hash_file(&path, buffer)?;
            Ok(Artifact {
                path: path
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
                size,
                sha256,
            })
        })
        .collect()
}

pub fn ensure(workspace: &Path, shipped_policy: &Path, progress: &Progress<'_>) -> Result<PathBuf> {
    let snapshot = Snapshot::open(&workspace.join("sources")).context(
        "Import your Empire and Napoleon installations before rebuilding local resources",
    )?;
    ensure_from_snapshot(workspace, shipped_policy, &snapshot, progress)
}

pub(crate) fn ensure_from_snapshot(
    workspace: &Path,
    shipped_policy: &Path,
    snapshot: &Snapshot,
    progress: &Progress<'_>,
) -> Result<PathBuf> {
    let parent = workspace.join("derived");
    crate::safety::ensure_writable_path(&parent)?;
    snapshot.verify(progress)?;
    let mut buffer = vec![0; 1024 * 1024];
    let policy = files(
        shipped_policy,
        &[
            "rules/schema_emp.ron",
            "rules/schema_nap.ron",
            "policy/rules",
            "policy/ui",
        ],
        &mut buffer,
    )?;
    let mut hash = Sha256::new();
    hash.update(RECIPE.to_le_bytes());
    hash.update(snapshot.language().as_bytes());
    for file in &snapshot.manifest().files {
        hash.update(file.game.dir().as_bytes());
        hash.update([0]);
        hash.update(file.path.as_bytes());
        hash.update([0]);
        hash.update(file.sha256.as_bytes());
        hash.update(file.size.to_le_bytes());
    }
    for file in &policy {
        hash.update(file.path.as_bytes());
        hash.update([0]);
        hash.update(file.sha256.as_bytes());
    }
    let fingerprint = format!("{:x}", hash.finalize());
    let destination = parent.join(&fingerprint);
    let assets = destination.join("assets");
    if destination.exists() {
        let manifest: Manifest = serde_json::from_slice(&fs::read(destination.join(MANIFEST))?)
            .context("Read locally rebuilt asset manifest")?;
        if manifest.recipe != RECIPE
            || manifest.fingerprint != fingerprint
            || files(&assets, &["rules", "ui"], &mut buffer)? != manifest.files
        {
            bail!(
                "Locally rebuilt resources are damaged: {}. Move this derived cache aside and rebuild; the original snapshots and project must be kept.",
                destination.display()
            );
        }
        progress("Verified locally rebuilt rules and interface resources");
        return Ok(assets);
    }
    fs::create_dir_all(&parent)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let staging = parent.join(format!(".rebuilding-{}-{stamp}", std::process::id()));
    fs::create_dir(&staging)?;
    let staged_assets = staging.join("assets");
    let result = (|| -> Result<()> {
        fs::create_dir(&staged_assets)?;
        progress("Rebuilding conversion rules from your original game data");
        rules::generate(snapshot, shipped_policy, &staged_assets, progress)?;
        progress("Rebuilding the campaign interface from your original Napoleon files");
        ui::generate(snapshot, shipped_policy, &staged_assets, progress)?;
        let manifest = Manifest {
            recipe: RECIPE,
            fingerprint,
            files: files(&staged_assets, &["rules", "ui"], &mut buffer)?,
        };
        fs::write(
            staging.join(MANIFEST),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        fs::rename(&staging, &destination).context("Publish locally rebuilt resources")?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&staging);
        return Err(
            error.context("Local resource rebuild failed; no replacement assets were published")
        );
    }
    progress("Local conversion resources ready; terrain will be rebuilt during export");
    Ok(assets)
}
