//! UI / campaign-script / template policy. The generated campaign ships its own UI scripts and campaign script.
//! TSV data carries no campaign, UI or script information, so nothing about the target's own resources can be
//! inferred. The policy is explicit instead: the user confirms (`resources_reviewed`) that unprovided resources use
//! the original conversion templates, and may supply already merged resources as a dataset-bound adapter. Adapter
//! files are never silently ignored: each one is validated against the allowed paths and then consumed (emitted
//! into the pack, or used as the structural donor of the template snapshot).

use super::Draft;
use crate::{esf, model::Progress};
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use walkdir::WalkDir;

/// Structural donor of the template snapshot (never shipped).
pub const DONOR_PATH: &str = "campaigns/eur_napoleon/startpos.esf";
/// Campaign script of the generated campaign (written by the start-position builder; an adapter may supply a
/// merged one).
pub const SCRIPT_PATH: &str = "campaigns/wld_napoleon/scripting.lua";

pub struct Input<'a> {
    pub assets: &'a Path,
    /// A TSV dataset is the target (resources come from the conversion templates unless an adapter supplies them).
    pub tsv: bool,
    /// Verified adapter files (canonical lower-case `/` keys), when an adapter is bound.
    pub adapter: Option<&'a BTreeMap<String, Vec<u8>>>,
}

#[derive(Default)]
pub struct Outcome {
    /// Pack member name (backslashes) -> bytes: the generated campaign's UI resources with adapter files applied.
    pub members: BTreeMap<String, Vec<u8>>,
    /// Adapter-provided structural donor for `campaigns/eur_napoleon/startpos.esf`.
    pub donor: Option<Vec<u8>>,
    pub issues: Vec<Draft>,
    pub changes: Vec<(String, String, String)>,
    pub warnings: Vec<String>,
}

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn canon(name: &str) -> String {
    name.replace('\\', "/").to_lowercase()
}

/// The generated campaign's own UI resources below `assets/ui`, keyed by pack member name.
pub fn bundled(assets: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut out = BTreeMap::new();
    for entry in WalkDir::new(assets.join("ui")) {
        let entry = entry?;
        if entry.file_type().is_file() {
            let key = entry
                .path()
                .strip_prefix(assets)?
                .to_string_lossy()
                .replace('/', "\\");
            if out.insert(key.clone(), fs::read(entry.path())?).is_some() {
                bail!("Duplicate output pack member {key}");
            }
        }
    }
    Ok(out)
}

/// Structural check of a start-position donor: the records the converter reads must exist.
pub fn check_startpos(bytes: &[u8]) -> Result<()> {
    let file = esf::read(bytes).context("not a readable ESF file")?;
    let world = file
        .root
        .path(&["CAMPAIGN_ENV", "CAMPAIGN_MODEL", "WORLD"])
        .context("missing CAMPAIGN_ENV/CAMPAIGN_MODEL/WORLD")?;
    world
        .get("FACTION_ARRAY")
        .and_then(|n| n.items())
        .context("missing WORLD/FACTION_ARRAY")?;
    world
        .path(&["REGION_MANAGER", "REGIONS_ARRAY"])
        .and_then(|n| n.items())
        .context("missing WORLD/REGION_MANAGER/REGIONS_ARRAY")?;
    Ok(())
}

pub(super) fn adapter_path_allowed(path: &str) -> bool {
    path.starts_with("ui/")
        || path.starts_with("script/")
        || path == SCRIPT_PATH
        || path == DONOR_PATH
}

pub fn plan(input: &Input<'_>, progress: &Progress<'_>) -> Result<Outcome> {
    let mut out = Outcome::default();
    let ours = bundled(input.assets)?;
    let empty = BTreeMap::new();
    let adapter = input.adapter.unwrap_or(&empty);
    let mut consumed: BTreeSet<String> = BTreeSet::new();

    // ---- adapter paths must be resource paths this tool knows how to consume
    for path in adapter.keys() {
        if !adapter_path_allowed(path) {
            let mut h = super::Hasher::new("adapter-path");
            h.part(path);
            out.issues.push(Draft::blocker(
                format!("adapter:path:{path}"),
                h.finish(),
                "adapter",
                format!("Adapter file {path} is not allowed"),
                format!(
                    "Adapters may only supply UI resources (ui/…), scripts (script/…), the merged campaign script ({SCRIPT_PATH}) and the structural donor ({DONOR_PATH}). {path} would override database, text, map, start-position or other game data and cannot be used; remove it from the adapter and import the adapter again."
                ),
            ));
        }
    }

    // ---- the generated campaign's own UI resources
    progress("Checking UI and script resources against the target");
    for (member, bytes) in &ours {
        let path = canon(member);
        if let Some(merged) = adapter.get(&path) {
            consumed.insert(path.clone());
            out.members.insert(member.clone(), merged.clone());
            out.changes.push((
                "adapter_resource".into(),
                path,
                "merged adapter file replaces the bundled resource".into(),
            ));
            continue;
        }
        out.members.insert(member.clone(), bytes.clone());
    }

    if input.tsv {
        out.warnings.push(if adapter.is_empty() {
            "The TSV data contains no UI, script or start-position information. You confirmed that the resources come from the original conversion templates; the mod's own UI/script changes are not part of this export and no mod resource compatibility is claimed.".to_owned()
        } else {
            format!(
                "{} resource file(s) come from the supplied adapter; everything else uses the original conversion templates. The TSV data itself says nothing about the mod's UI/scripts, and the adapter files are not checked against the mod or played.",
                adapter.len()
            )
        });
    }

    // ---- donor and extra adapter files
    if let Some(bytes) = adapter.get(DONOR_PATH) {
        consumed.insert(DONOR_PATH.to_owned());
        match check_startpos(bytes) {
            Ok(()) => {
                out.donor = Some(bytes.clone());
                out.changes.push((
                    "adapter_donor".into(),
                    DONOR_PATH.into(),
                    "adapter start position used as structural template only (not shipped)".into(),
                ));
                out.warnings.push(
                    "The adapter start-position template is checked structurally only, never run in the game."
                        .into(),
                );
            }
            Err(error) => {
                let mut h = super::Hasher::new("donor-structure");
                h.part(&sha(bytes));
                out.issues.push(Draft::blocker(
                    "template:adapter".into(),
                    h.finish(),
                    "template",
                    "Adapter start-position template is not usable".into(),
                    format!("{DONOR_PATH} from the adapter failed the structural check: {error:#}"),
                ));
            }
        }
    }
    for (path, bytes) in adapter {
        if !adapter_path_allowed(path) || path == DONOR_PATH {
            continue;
        }
        let member = path.replace('/', "\\");
        if out.members.contains_key(&member) || ours.keys().any(|m| canon(m) == *path) {
            continue; // already consumed as replacement of a bundled resource
        }
        consumed.insert(path.clone());
        out.members.insert(member, bytes.clone());
        out.changes.push((
            "adapter_resource".into(),
            path.clone(),
            "adapter file shipped as additional pack member".into(),
        ));
    }

    // ---- every adapter file must have been consumed or rejected
    for path in adapter.keys() {
        if adapter_path_allowed(path) && !consumed.contains(path) {
            bail!("adapter file {path} was not consumed by the resource plan");
        }
    }
    Ok(out)
}
