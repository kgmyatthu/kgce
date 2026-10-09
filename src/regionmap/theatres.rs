//! Fixed campaign theatre bounds, shared by preview and export rather than stored as project settings.

use super::geometry::Rect;

/// Europe is unchanged; India matches the game-validated rectangle, leaving a 10-unit gap to East Indies.
/// America's west edge is the nearest outward 2-unit grid line containing all of `western_plain`:
/// its 331 ring vertices span x [-886.510193, -736.021240], y [162.857834, 341.924622].
pub const FIXED_BOUNDS: &[(&str, Rect)] = &[
    ("1", [[-888.0, 0.0], [-340.0, 460.0]]), // America — west edge covers western_plain
    ("2", [[-180.0, 140.0], [440.0, 520.0]]), // Europe — vanilla, unchanged
    ("3", [[440.0, -80.0], [790.0, 300.0]]), // India — validated; 10-unit gap to EastIndies at x 800
];
