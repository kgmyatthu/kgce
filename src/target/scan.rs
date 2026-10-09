//! Reads a user folder of loose files into memory without ever following links, leaving the folder, or
//! accepting a name that cannot become a safe path. The folder is only read, never written.

use crate::model::Progress;
use crate::snapshot::{absolute, validate_rel};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use walkdir::WalkDir;

/// A regular file below the folder; `rel` keeps the stored capitalisation and uses `/`.
pub(super) struct RawFile {
    pub rel: String,
    pub bytes: Vec<u8>,
}

#[derive(Default)]
pub(super) struct Tree {
    /// Sorted by lower-case relative path.
    pub files: Vec<RawFile>,
    /// Relative paths of regular files the caller's filter does not take.
    pub ignored: Vec<String>,
    /// Links, special files, unsafe or colliding names, files over a limit... every one blocks an import.
    pub problems: Vec<String>,
}

pub(super) struct Limits {
    pub files: usize,
    pub file_bytes: u64,
    pub total_bytes: u64,
}

/// Validates a user-chosen input folder and returns its absolute path.
pub(super) fn open_folder(path: &Path, what: &str) -> Result<PathBuf> {
    if path.as_os_str().is_empty() {
        bail!("the {what} folder is empty");
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        bail!("the {what} folder {} must not contain '..'", path.display());
    }
    let abs = absolute(path)?;
    let meta = fs::symlink_metadata(&abs).with_context(|| {
        format!(
            "the {what} folder {} does not exist or cannot be read",
            abs.display()
        )
    })?;
    if meta.file_type().is_symlink() {
        bail!(
            "the {what} folder {} is a link; links are never followed",
            abs.display()
        );
    }
    if !meta.is_dir() {
        bail!("{} is not a folder", abs.display());
    }
    Ok(abs)
}

/// Reads every regular file for which `keep(lower-case relative path)` is true; other regular files are
/// listed in `ignored`.
pub(super) fn read_tree(
    root: &Path,
    keep: &dyn Fn(&str) -> bool,
    limits: &Limits,
    progress: &Progress,
) -> Tree {
    let mut tree = Tree::default();
    let mut found: BTreeMap<String, RawFile> = BTreeMap::new();
    let mut total = 0u64;
    for entry in WalkDir::new(root).follow_links(false).sort_by_file_name() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tree.problems.push(format!("cannot read the folder: {e}"));
                continue;
            }
        };
        if entry.depth() == 0 {
            continue;
        }
        let ty = entry.file_type();
        if ty.is_symlink() {
            tree.problems.push(format!(
                "{} is a link; links are never followed or imported",
                entry.path().display()
            ));
            continue;
        }
        if ty.is_dir() {
            continue;
        }
        if !ty.is_file() {
            tree.problems
                .push(format!("{} is not a regular file", entry.path().display()));
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(root) else {
            tree.problems
                .push(format!("{} is outside the folder", entry.path().display()));
            continue;
        };
        let mut stored = String::new();
        let mut bad = false;
        for c in rel.components() {
            match c {
                Component::Normal(part) => match part.to_str() {
                    Some(s) => {
                        if !stored.is_empty() {
                            stored.push('/');
                        }
                        stored.push_str(s);
                    }
                    None => bad = true,
                },
                _ => bad = true,
            }
        }
        if bad {
            tree.problems.push(format!(
                "{} is not a plain UTF-8 relative path",
                entry.path().display()
            ));
            continue;
        }
        let key = stored.to_lowercase();
        if !keep(&key) {
            tree.ignored.push(stored);
            continue;
        }
        if let Err(e) = validate_rel(&key) {
            tree.problems.push(format!("{stored}: {e:#}"));
            continue;
        }
        if found.contains_key(&key) {
            tree.problems.push(format!(
                "{stored} exists twice, differing only by capitalisation"
            ));
            continue;
        }
        if found.len() >= limits.files {
            tree.problems.push(format!(
                "the folder holds more than {} importable files",
                limits.files
            ));
            break;
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                tree.problems.push(format!("{stored}: {e}"));
                continue;
            }
        };
        if meta.len() > limits.file_bytes {
            tree.problems.push(format!(
                "{stored} is {} bytes; the limit for one file is {}",
                meta.len(),
                limits.file_bytes
            ));
            continue;
        }
        total += meta.len();
        if total > limits.total_bytes {
            tree.problems.push(format!(
                "the folder holds more than {} bytes of importable files",
                limits.total_bytes
            ));
            break;
        }
        match fs::read(entry.path()) {
            Ok(bytes) if bytes.len() as u64 == meta.len() => {
                found.insert(key, RawFile { rel: stored, bytes });
            }
            Ok(_) => tree
                .problems
                .push(format!("{stored} changed while it was being read")),
            Err(e) => tree.problems.push(format!("{stored}: {e}")),
        }
        if found.len() % 500 == 0 && !found.is_empty() {
            progress(&format!(
                "Reading {}: {} files",
                root.display(),
                found.len()
            ));
        }
    }
    tree.files = found.into_values().collect();
    tree.problems.sort();
    tree
}
