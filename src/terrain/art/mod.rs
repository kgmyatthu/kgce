//! Native generation of the fixed America / India theatre artwork from the user's own ETW supertexture and
//! heightmap plus a public NOAA elevation grid (port of the former offline Python painters).

pub mod america;
pub mod dem;
pub mod grid;
pub mod india;
pub mod painter;
pub mod prims;
pub mod resample;
pub mod stpx;

use std::collections::BTreeMap;

/// Output of one theatre generator.
pub struct TheatreResult {
    /// RAW 262144-byte DXT5 per changed tile (all seven mip levels), keyed by global tile index.
    pub tiles: BTreeMap<usize, Vec<u8>>,
    /// Complete new raw heightmap (8192 x 4096, bottom-first, TGA header stripped).
    pub heights: Vec<u8>,
    /// Numbers the generator measured about itself (changed tiles per level, void/heal px, coast agreement ...).
    pub report: serde_json::Value,
}
