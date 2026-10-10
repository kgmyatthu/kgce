//! `BuildContext`: the Rust mirror of the Python `Build` object (`bd`) that the CAI builder consumes.
//!
//! Ownership model: core (this module's owner) fills everything up to and including the diplomacy / shapes stage; the CAI
//! owner (`cai::build`) is called afterwards, reads the data below, may mutate `regions` (WORLD REGION child[22] =
//! governorship link) and the CAI-assigned fields (`Force::cai_*`, `GovPost::cai`), and returns `CaiOutput`.
//! Core then splices `CaiOutput::interface` into the CAMPAIGN_MODEL and remaps CDIR_INTERFACE child 0 through
//! `CaiOutput::director_map`.
//!
//! All world object ids are `u32` (the ESF stores them as u32/i32 with the same bit pattern for the values used here).
//! World coordinates in `Force::x/y` are the fixed-point 12.20 integers stored in LOCOMOTABLE (`fixed()` of world units).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::esf::Node;
use crate::model::Rules;
use crate::snapshot::Snapshot;

use super::helpers::IdAlloc;

/// Per-faction rule data after `Rules::apply_project` (mirror of Python `bd.F[key]`).
#[derive(Clone, Debug)]
pub struct FactionInfo {
    /// owns at least one region (post-edit)
    pub alive: bool,
    /// capital region key (post-edit), `None` for landless factions
    pub capital: Option<String>,
    pub playable: bool,
    /// number of owned regions (post-edit)
    pub regions: usize,
    /// key of the faction in the ETW world rules/start position this faction was derived from (differs from the faction
    /// key when the user mapped it onto a target-mod faction). Every faction is a faction of the target database; its
    /// FACTION record structure depends on whether the target start position holds a record for it (see
    /// `factions::structure_for`)
    pub etw_source: String,
    /// explicit structural donor (a live faction record of the target start position)
    pub donor: Option<String>,
}

/// A non-deployed character kept in a faction record (ministers, reserve generals, governors).
#[derive(Clone, Debug)]
pub struct MetaChar {
    /// CHARACTER id in OUR world (CHARACTER child[2] / CHARACTER_DETAILS[10])
    pub wid: u32,
    /// id of the NTW template character it was cloned from (key of the NTW CAI_WORLD_CHARACTERS template); equals `wid` for
    /// characters that are unmodified NTW characters of the faction's own NTW record
    pub twid: u32,
    pub kind: String,
    pub unit: u32,
}

/// A governorship (post + GOVERNORSHIP record) of a faction for one theatre.
#[derive(Clone, Debug)]
pub struct GovPost {
    /// world id of the governor CHARACTER
    pub char_id: u32,
    /// world id of the GOVERNORSHIP record
    pub gov_id: u32,
    /// theatre key ("1" america, "2" europe, "3" india)
    pub theatre: String,
    /// post key ("governor_europe", ...)
    pub post: String,
    /// owned region keys of that theatre (sorted)
    pub regions: Vec<String>,
    /// CAI governorship id; assigned by the CAI owner (0 until then)
    pub cai: u32,
}

#[derive(Clone, Debug, Default)]
pub struct FactionMeta {
    pub key: String,
    /// non-deployed characters of the faction
    pub chars: Vec<MetaChar>,
    /// owned region keys (sorted, post-edit)
    pub owned: Vec<String>,
    pub gov_posts: Vec<GovPost>,
    /// NTW faction key that provided the record structure (own key when the NTW record is used unmodified)
    pub donor: String,
    /// donor-id -> fresh-id mapping applied when the record was cloned from a donor
    pub idmap: BTreeMap<u32, u32>,
    /// true when the NTW eur_napoleon FACTION record of this very faction (live structure, own ids) is used unchanged
    pub own_structure: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceKind {
    Army,
    Navy,
}

impl ForceKind {
    pub fn is_navy(self) -> bool {
        self == ForceKind::Navy
    }
}

/// A deployed army / navy (ARMY or NAVY record + leader CHARACTER, both stored in `frec[faction]`).
#[derive(Clone, Debug)]
pub struct Force {
    pub faction: String,
    pub kind: ForceKind,
    /// region the army stands in / region whose port the navy sails from (post-edit: always owned by `faction`)
    pub region: String,
    /// region key the force stands in; for navies the sea region key from the regions.esf locator
    pub region_at: String,
    /// sea region key for navies
    pub sea_key: Option<String>,
    /// "General" | "colonel" | "admiral" | "captain"
    pub role: String,
    pub units: Vec<String>,
    /// fixed 12.20 coordinates as stored in LOCOMOTABLE
    pub x: i32,
    pub y: i32,
    /// garrison force of its region's settlement (SIEGEABLE_GARRISON_RESIDENCE[12] == wid_army)
    pub garrison: bool,
    pub wid_char: u32,
    pub wid_army: u32,
    pub unit_ids: Vec<u32>,
    /// NTW template ids (character / army / first unit) the instance was cloned from
    pub t_char: u32,
    pub t_army: u32,
    pub t_unit: u32,
    // ---- assigned by the CAI owner
    pub cai_char: u32,
    pub cai_mobile: u32,
    pub cai_units: Vec<u32>,
    pub region_cai: u32,
    pub theatre_cai: u32,
    pub cell: Option<(i32, i32)>,
}

/// Per region (mirror of Python `bd.region_info[key]`).
#[derive(Clone, Debug)]
pub struct RegionInfo {
    pub key: String,
    /// index into `BuildContext::regions` (the REGION record, converted to NTW v5)
    pub index: usize,
    /// owning faction key (post-edit)
    pub owner: String,
    /// theatre key ("1" america, "2" europe, "3" india)
    pub theatre: String,
    pub theatre_name: String,
    /// settlement position (fixed 12.20): SIEGEABLE_GARRISON_RESIDENCE children 10, 11
    pub pos: (i32, i32),
    /// REGION child 4 (i32): world id of the region
    pub world_id: u32,
    /// world id of the SIEGEABLE_GARRISON_RESIDENCE of the settlement
    pub sg_id: u32,
    /// world id of the garrison army (set by forces)
    pub garrison: Option<u32>,
}

pub struct BuildContext<'a> {
    pub snapshot: &'a Snapshot,
    pub rules: &'a Rules,
    pub map: &'a crate::map::MapArtifacts,
    /// root of the generated `campaign_maps/nap_world/sea_grids.esf`: the theatre rectangles and cell sizes of the
    /// finished map. The CAI sea grid cell of every fleet is derived from it (never from the ETW original, whose
    /// theatre grids the build moves)
    pub sea_grids: &'a Node,
    /// keys of the activated wilderness regions and of the user's region splits: present in `rules.regions`, without a
    /// REGION record in the ETW start position
    pub added_keys: BTreeSet<String>,

    /// CAMPAIGN_MODEL record of the pristine ETW `main` startpos
    pub etw_model: Arc<Node>,
    /// CAMPAIGN_MODEL record of the pristine NTW `eur_napoleon` startpos
    pub ntw_model: Arc<Node>,
    /// ETW FACTION world id -> key
    pub etw_fkey: BTreeMap<u32, String>,
    /// NTW FACTION world id -> key
    pub ntw_fkey: BTreeMap<u32, String>,
    /// ETW FACTION records by key
    pub etw_factions: BTreeMap<String, Arc<Node>>,
    /// ETW start position faction key -> key of the faction that now stands for it (identity unless the user mapped it)
    pub etw_to_key: BTreeMap<String, String>,

    /// final faction order (NTW order first, then the remaining target factions sorted); rebel faction NOT included
    pub fkeys: Vec<String>,
    /// world faction ids by key
    pub fid: BTreeMap<String, u32>,
    /// NTW REBEL_FACTION world id
    pub rebel_id: u32,
    /// technology tree id (FACTION_TECHNOLOGY_MANAGER child with code 4) by faction key
    pub tech_id: BTreeMap<String, u32>,
    pub factions: BTreeMap<String, FactionInfo>,
    pub fmeta: BTreeMap<String, FactionMeta>,
    /// finished FACTION records by key
    pub frec: BTreeMap<String, Node>,
    /// (K, O) pairs at war according to the generated diplomacy
    pub war: BTreeSet<(String, String)>,

    /// REGION records in output order (`region_order[i]` is the key of `regions[i]`)
    pub regions: Vec<Node>,
    pub region_order: Vec<String>,
    pub region_info: BTreeMap<String, RegionInfo>,
    /// slot ids (first scalar of REGION_SLOT) of ETW slots that do not exist on the new map
    pub dropped_slots: BTreeSet<u32>,

    pub forces: Vec<Force>,
    pub ids: IdAlloc,
    /// human-readable notes (relocations, conversions); core forwards them to the progress callback / report
    pub notes: Vec<String>,
}

impl<'a> BuildContext<'a> {
    pub fn at_war(&self, a: &str, b: &str) -> bool {
        self.war.contains(&(a.to_string(), b.to_string()))
            || self.war.contains(&(b.to_string(), a.to_string()))
    }

    pub fn region_rec(&self, key: &str) -> Option<&Node> {
        self.region_info.get(key).map(|r| &self.regions[r.index])
    }

    pub fn region_rec_mut(&mut self, key: &str) -> Option<&mut Node> {
        let idx = self.region_info.get(key)?.index;
        self.regions.get_mut(idx)
    }

    pub fn theatre_of(&self, region_key: &str) -> Option<&str> {
        self.region_info.get(region_key).map(|r| r.theatre.as_str())
    }
}
