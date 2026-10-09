//! Pathfinding synthesis for newly passable land: land zone, coast bands, landing crossings, settlement nodes, region
//! border bands with group paths, sea next to new coasts, a-words / headers, vertex tables and run-length encoding
//! (ports of work/exp4/pathsynth.py, crossings.py, work/exp5/pathsynth5.py and the pf_decode.py rule checker).
//!
//! Usage: `synthesize(&mut grid, &rd, &targets, &Options::default())` after the regions of the targets are passable
//! (activation / merge / carve / clipping done), then `grid.store(&mut root)`; `validate(&grid, &rd)` returns the list
//! of rule violations (empty = valid).

pub mod crossings;
pub mod finish;
pub mod geometry;
pub mod hygiene;
pub mod land;
pub mod model;
pub mod nodes;
pub mod rules;
mod split;
mod synth;
pub mod validate;

#[cfg(test)]
mod tests;

pub use nodes::SlotKind;
pub use split::synthesize_split;
pub use synth::synthesize;
pub use validate::{validate, validate_all};

/// Extra per-target switches (none yet; kept so callers can pass `Default::default()`).
#[derive(Clone, Debug, Default)]
pub struct TargetOptions;

/// One region to give pathfinding: its passable areas become land, its settlement gets a node of the shape of its
/// slot type.
#[derive(Clone, Debug)]
pub struct Target {
    /// regions.esf region index
    pub region: usize,
    pub settlement_slot_kind: SlotKind,
    pub options: TargetOptions,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// wanted arc distance between landing crossings along a coast (map units); the default 7.5 is the value of the
    /// experiment scripts and reproduces the golden Burma landing count (73; 7.0 gives 78)
    pub crossing_spacing: f64,
    /// at least one landing per this much coast (map units)
    pub min_per: f64,
    /// regions (keys) that never form group paths
    pub no_group_regions: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            crossing_spacing: 7.5,
            min_per: 15.0,
            no_group_regions: vec!["unexplorable".into()],
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegionReport {
    pub region: usize,
    pub path_id: u16,
    pub landings: usize,
    pub node_clipped: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub warnings: Vec<String>,
    pub per_region: Vec<RegionReport>,
    /// new group paths: (member regions.esf indices in region-list order, final path id)
    pub new_groups: Vec<(Vec<i16>, u16)>,
    pub touched_cells: usize,
    pub cells_to_runs: usize,
    pub vertices_merged: usize,
    pub degenerate_dropped: usize,
    pub tjunctions: usize,
    pub edge_vertices_synced: usize,
}
