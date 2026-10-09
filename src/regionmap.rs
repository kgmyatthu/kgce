//! Campaign-map region editing: typed access to the geometry files of an ETW/NTW campaign map and the operations
//! that change region borders (split a region along a user-drawn polyline, rebuild derived indexes).

pub mod geometry;
pub mod regions;
pub mod clip;
pub mod merge;

pub mod pathfinding;
pub mod pathsynth;
pub mod apply;
pub mod poi;
pub mod trade_routes;
pub mod sea_grids;
pub mod chain;
pub mod stages;
pub mod theatres;
pub mod campaign_rows;
pub mod startpos_split;
