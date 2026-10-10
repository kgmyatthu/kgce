//! Campaign-building visual scale: a separate, auto-loaded movie pack affecting
//! geometry rendered by the campaign-building shader without changing model assets,
//! database, start position, characters or the executable.
//!
//! The pack carries exactly two members, derived from the two original NTW shader sources that ship inside
//! the editor (`assets/shaders/`, embedded byte-for-byte with `include_bytes!`, CRLF line endings included).
//! They are verbatim, unmodified game files: Creative Assembly's work, bundled only as the input of this
//! derivation and never written to disk by themselves. Building export therefore depends on nothing
//! imported; no source snapshot, target dataset or installed (possibly already scaled) shader is read.
//!
//! * `fx\campaignbuilding.fx` gets `#define CAMPAIGN_BUILDING_MODEL_SCALE <scale>f` prepended, and
//! * `fx\campaignbuilding_vertex.fx_fragment` multiplies the model-space vertex position by that constant in
//!   each of its three vertex transforms (visible, shadow and depth).
//!
//! Nothing else is overridden, so unrelated shaders (battle units, terrain, flags, …) keep loading from the
//! game's own packs. The build needs no shader compiler: the game compiles the shader when it starts. The
//! accepted proof-of-concept scale is 0.5; its payloads are checked against the hashes of the
//! compile-verified proof-of-concept pack.

use crate::pack::{self, TYPE_MOVIE};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// Stable movie-pack filename: installing another scale replaces this file rather
/// than stacking overrides. Other mods overriding these shaders may still conflict.
pub const PACK_NAME: &str = "zz_campaign_buildings_scale.pack";

/// Smallest and largest saved scale. 1.0 means original size and produces no pack.
pub const MIN_SCALE: f32 = 0.1;
pub const MAX_SCALE: f32 = 1.0;
/// The only scale with a visually verified proof of concept.
pub const PROVEN_SCALE: f32 = 0.5;

/// Pack member names of the two shaders.
const TOP: &str = "fx\\campaignbuilding.fx";
const VERTEX: &str = "fx\\campaignbuilding_vertex.fx_fragment";

/// The unmodified NTW shader sources, embedded verbatim (never re-encoded; `assets/` is exempt from
/// line-ending conversion in `.gitattributes`).
static TOP_SOURCE: &[u8] = include_bytes!("../assets/shaders/campaignbuilding.fx");
static VERTEX_SOURCE: &[u8] = include_bytes!("../assets/shaders/campaignbuilding_vertex.fx_fragment");
/// Provenance recorded in the export metadata.
const SOURCE_PROVENANCE: &str = "bundled with the editor: unmodified original Napoleon: Total War game shader (assets/shaders), not read from any installation or import";

const TRANSFORM: &[u8] = b"mul(float4(v.position,1), instance_world)";
const TRANSFORM_SCALED: &[u8] =
    b"mul(float4(v.position * CAMPAIGN_BUILDING_MODEL_SCALE,1), instance_world)";

/// SHA-256 of the unmodified sources of the accepted proof of concept and of its half-scale payloads.
const STOCK_TOP_SHA: &str = "5906870eaa0d99a6228659792a4d8cf22bb2ae9f3e79f7e90eaff363c5638b8d";
const STOCK_VERTEX_SHA: &str = "07a5132feb6ff292ce1b5387236b6b04598b3e25a4abb2dc2a32055801d6c580";
const PROVEN_TOP_SHA: &str = "3e3988d84e5d7370acd48287a3d7bf4a9206836f0994d0c08e50f4828d2c52e8";
const PROVEN_VERTEX_SHA: &str = "595b961371e25fde9730bfe9374a697db213898c3e765b8fff146291d2544056";

fn sha(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// Scales the editor accepts: finite and within `MIN_SCALE..=MAX_SCALE`.
pub fn validate_scale(scale: f32) -> Result<()> {
    ensure!(
        scale.is_finite() && (MIN_SCALE..=MAX_SCALE).contains(&scale),
        "The campaign building scale {scale} must be a number from {MIN_SCALE} to {MAX_SCALE} (1 keeps the original size)"
    );
    Ok(())
}

/// Whether a valid scale changes anything (1.0 keeps the original size and emits no pack).
pub fn is_active(scale: f32) -> bool {
    scale != MAX_SCALE
}

/// Checks that the bundled sources are intact and the scale is usable, without writing anything, so callers
/// can fail before spending time on the rest of an export.
pub fn preflight(scale: f32) -> Result<()> {
    validate_scale(scale)?;
    if !is_active(scale) {
        return Ok(());
    }
    build(scale).map(|_| ())
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn replace_all(haystack: &[u8], needle: &[u8], with: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(haystack.len() + 64);
    let mut at = 0;
    while let Some(i) = find(&haystack[at..], needle) {
        out.extend_from_slice(&haystack[at..at + i]);
        out.extend_from_slice(with);
        at += i + needle.len();
    }
    out.extend_from_slice(&haystack[at..]);
    out
}

/// HLSL literal of the scale: the shortest decimal that parses back to the same `f32`, so 0.5 gives `0.5f`.
fn literal(scale: f32) -> String {
    format!("{scale}f")
}

fn describe(scale: f32) -> String {
    if scale == PROVEN_SCALE {
        "half-size".to_owned()
    } else {
        format!("{scale}x-size")
    }
}

fn derive(scale: f32) -> Result<BTreeMap<String, Vec<u8>>> {
    validate_scale(scale)?;
    ensure!(
        sha(TOP_SOURCE) == STOCK_TOP_SHA && sha(VERTEX_SOURCE) == STOCK_VERTEX_SHA,
        "The campaign building shader sources bundled with this editor are not the verified original files (damaged build or altered line endings); refusing to derive a shader pack"
    );
    let mut scaled_top = format!(
        "// Same campaign models and positions; {} model-space geometry.\r\n#define CAMPAIGN_BUILDING_MODEL_SCALE {}\r\n",
        describe(scale),
        literal(scale)
    )
    .into_bytes();
    scaled_top.extend_from_slice(TOP_SOURCE);
    let scaled_vertex = replace_all(VERTEX_SOURCE, TRANSFORM, TRANSFORM_SCALED);

    if scale == PROVEN_SCALE {
        ensure!(
            sha(&scaled_top) == PROVEN_TOP_SHA && sha(&scaled_vertex) == PROVEN_VERTEX_SHA,
            "The half-size building shaders differ from the accepted compile-verified payloads"
        );
    }
    Ok(BTreeMap::from([(TOP.to_owned(), scaled_top), (VERTEX.to_owned(), scaled_vertex)]))
}

/// Pack members (`fx\…` names) that scale campaign buildings by `scale`; empty for 1.0.
pub fn build(scale: f32) -> Result<BTreeMap<String, Vec<u8>>> {
    validate_scale(scale)?;
    if !is_active(scale) {
        return Ok(BTreeMap::new());
    }
    derive(scale)
}

/// Writes `<data>/zz_campaign_buildings_scale.pack` and returns the export metadata, or `None` (nothing
/// written) for scale 1.0. The pack is re-read and every member compared before returning.
pub fn export(data: &Path, scale: f32) -> Result<Option<Value>> {
    validate_scale(scale)?;
    if !is_active(scale) {
        return Ok(None);
    }
    let members = derive(scale)?;
    let path = data.join(PACK_NAME);
    pack::write_pack(&path, &members, TYPE_MOVIE)?;
    let reread = pack::Pack::open(&path)?;
    ensure!(
        reread.type_ == TYPE_MOVIE && reread.entries.len() == members.len(),
        "The generated building scale pack failed its type or entry check"
    );
    for (name, payload) in &members {
        ensure!(
            reread.read(name)? == *payload,
            "The generated building scale pack member {name} failed its read-back check"
        );
    }
    let bytes = std::fs::read(&path)?;
    let sources: [(&str, &[u8]); 2] = [(TOP, TOP_SOURCE), (VERTEX, VERTEX_SOURCE)];
    let mut described = serde_json::Map::new();
    for (name, source) in sources {
        described.insert(
            name.to_owned(),
            json!({
                "source": SOURCE_PROVENANCE,
                "source_sha256": sha(source),
                "override_sha256": sha(&members[name]),
            }),
        );
    }
    Ok(Some(json!({
        "enabled": true,
        "scale": scale.to_string().parse::<f64>()?,
        "pack_file": PACK_NAME,
        "pack_type": "movie (4), auto-loaded",
        "pack_sha256": sha(&bytes),
        "pack_bytes": bytes.len(),
        "members": described,
        "compiler": "none needed: the game compiles the shaders at startup",
        "scale_status": if scale == PROVEN_SCALE {
            "proven: payloads identical to the accepted compile-verified half-size pack"
        } else {
            "custom: same transform as the accepted half-size pack, not individually verified in the game"
        },
        "scope": "Geometry using the campaign-building shader in all campaigns and saves on this installation; separately rendered settlement pieces, ground underlays and CPU selection bounds are not rescaled.",
        "unchanged": [
            "model assets", "settlements", "world positions", "database", "startpos",
            "characters (generals, armies, fleets)", "battle unit shaders", "executable",
        ],
        "restart_required": true,
        "rollback": "Remove the pack file from the game's data folder and restart the game. No model, database or save data was changed, so saves work with or without it.",
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::test_util::TempDir;

    /// Every occurrence of `needle`, so the transform count is asserted without a production-only helper.
    fn occurrences(haystack: &[u8], needle: &[u8]) -> usize {
        let (mut n, mut at) = (0, 0);
        while let Some(i) = find(&haystack[at..], needle) {
            n += 1;
            at += i + needle.len();
        }
        n
    }

    #[test]
    fn bundled_sources_are_the_verified_originals_with_the_expected_structure() {
        assert_eq!(sha(TOP_SOURCE), STOCK_TOP_SHA);
        assert_eq!(sha(VERTEX_SOURCE), STOCK_VERTEX_SHA);
        assert!(find(TOP_SOURCE, b"\r\n").is_some(), "line endings must stay CRLF");
        assert_eq!(occurrences(VERTEX_SOURCE, TRANSFORM), 3);
        assert!(find(TOP_SOURCE, b"CAMPAIGN_BUILDING_MODEL_SCALE").is_none());
        assert!(find(VERTEX_SOURCE, b"CAMPAIGN_BUILDING_MODEL_SCALE").is_none());
        assert!(find(&TOP_SOURCE.to_ascii_lowercase(), b"campaignbuilding_vertex.fx_fragment").is_some());
    }

    #[test]
    fn invalid_scales_are_refused_and_full_size_emits_nothing() {
        for bad in [f32::NAN, f32::INFINITY, 0.0, 0.05, 1.01, -0.5] {
            assert!(validate_scale(bad).is_err(), "{bad}");
            assert!(preflight(bad).is_err(), "{bad}");
        }
        let out = TempDir::new("building_scale_full");
        assert!(export(out.path(), f32::NAN).is_err());
        assert!(export(out.path(), 0.0).is_err());
        assert!(build(1.0).unwrap().is_empty());
        assert!(export(out.path(), 1.0).unwrap().is_none());
        assert!(!out.path().join(PACK_NAME).exists());
    }

    #[test]
    fn scales_exactly_the_three_transforms_into_a_movie_pack_without_any_import() {
        preflight(0.25).unwrap();
        let out = TempDir::new("building_scale_pack");
        let meta = export(out.path(), 0.25).unwrap().unwrap();
        assert_eq!(meta["restart_required"], true);
        assert_eq!(meta["members"][TOP]["source_sha256"], STOCK_TOP_SHA);
        assert_eq!(meta["members"][VERTEX]["source_sha256"], STOCK_VERTEX_SHA);
        for name in [TOP, VERTEX] {
            let source = meta["members"][name]["source"].as_str().unwrap();
            assert!(source.starts_with("bundled with the editor"), "{source}");
        }
        let pack = pack::Pack::open(&out.path().join(PACK_NAME)).unwrap();
        assert_eq!(pack.type_, TYPE_MOVIE);
        assert_eq!(pack.entries.len(), 2);
        let top = pack.read(TOP).unwrap();
        assert!(top.starts_with(
            b"// Same campaign models and positions; 0.25x-size model-space geometry.\r\n#define CAMPAIGN_BUILDING_MODEL_SCALE 0.25f\r\n"
        ));
        assert!(top.ends_with(TOP_SOURCE));
        let vertex = pack.read(VERTEX).unwrap();
        assert_eq!(occurrences(&vertex, b"v.position * CAMPAIGN_BUILDING_MODEL_SCALE"), 3);
        assert_eq!(occurrences(&vertex, b"v.position,1"), 0);
        assert_eq!(build(0.25).unwrap().len(), 2);
    }

    #[test]
    fn proven_scale_reproduces_the_accepted_pack_byte_for_byte() {
        let out = TempDir::new("building_scale_proven");
        let meta = export(out.path(), PROVEN_SCALE).unwrap().unwrap();
        let bytes = std::fs::read(out.path().join(PACK_NAME)).unwrap();
        assert_eq!(
            sha(&bytes),
            "6642854f8e9688e7d3a2aa6086d2731ddca9e379bf866dd5271d7c03a42d79e9"
        );
        assert_eq!(bytes.len(), 8470);
        assert_eq!(meta["pack_sha256"], sha(&bytes));
        assert_eq!(meta["members"][TOP]["override_sha256"], PROVEN_TOP_SHA);
        assert_eq!(meta["members"][VERTEX]["override_sha256"], PROVEN_VERTEX_SHA);
        assert!(meta["scale_status"].as_str().unwrap().starts_with("proven"));
    }
}
