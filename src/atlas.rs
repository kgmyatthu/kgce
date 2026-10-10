//! Reproducible stock-Q20 connected atlas. Project edits use source coordinates;
//! this final export pass moves geography, navigation, display and campaign together.

pub mod campaign;
pub mod display;
pub mod geography;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type Point = [f64; 2];
pub type Rect = [Point; 2];
pub const SCALE: f64 = 1.3516979217529297;
const PRESET: &str = include_str!("../assets/policy/connected_atlas.json");

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chart {
    pub key: String,
    pub theatres: Vec<String>,
    pub theatre_rects: BTreeMap<String, Rect>,
    pub crop_rects: Vec<Rect>,
    pub offset: Point,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub format: u32,
    pub scale: f64,
    pub fixed_bits: u32,
    pub source_fixed_bits: u32,
    pub supertexture_levels: u32,
    pub world: Rect,
    pub source_world: Rect,
    pub charts: Vec<Chart>,
    pub background_region: serde_json::Value,
    pub interior_point_window: Rect,
}

impl Plan {
    pub fn proven() -> Self {
        serde_json::from_str(PRESET).expect("bundled connected atlas policy")
    }

    /// The accepted arrangement is deliberately indivisible: arbitrary offsets or
    /// scale increases invalidate the native Q20/navigation/camera proof.
    pub fn validate(&self) -> Result<()> {
        ensure!(*self == Self::proven(), "Unsupported atlas settings: choose the proven connected atlas preset; arbitrary scaling or theatre placement is not stock-engine verified");
        Ok(())
    }

    pub fn chart(&self, key: &str) -> Result<&Chart> {
        self.charts.iter().find(|c| c.key == key || c.theatres.iter().any(|t| t == key))
            .with_context(|| format!("Atlas has no chart/theatre {key}"))
    }

    pub fn contains(rect: Rect, point: Point) -> bool {
        (0..2).all(|i| rect[0][i] <= point[i] && point[i] <= rect[1][i])
    }

    pub fn chart_at(&self, point: Point, allow_crop: bool) -> Result<&Chart> {
        let mut found = self.charts.iter().filter(|c| c.theatre_rects.values().any(|r| Self::contains(*r, point)));
        if let Some(chart) = found.next() {
            ensure!(found.next().is_none(), "Ambiguous atlas point {point:?}");
            return Ok(chart);
        }
        if allow_crop {
            let mut found = self.charts.iter().filter(|c| c.crop_rects.iter().any(|r| Self::contains(*r, point)));
            if let Some(chart) = found.next() {
                ensure!(found.next().is_none(), "Ambiguous atlas crop point {point:?}");
                return Ok(chart);
            }
        }
        bail!("Unmapped atlas point {point:?}")
    }

    pub fn xy(&self, point: Point, chart: &Chart) -> Point {
        std::array::from_fn(|i| point[i] * self.scale + chart.offset[i])
    }

    pub fn inverse_xy(&self, point: Point, chart: &Chart) -> Point {
        std::array::from_fn(|i| (point[i] - chart.offset[i]) / self.scale)
    }

    pub fn fixed_value(&self, value: f64) -> Result<i32> {
        let raw = (value * (1u64 << self.fixed_bits) as f64).round_ties_even();
        ensure!(raw.is_finite() && raw >= i32::MIN as f64 && raw <= i32::MAX as f64,
            "Q{} coordinate outside i32: {value}", self.fixed_bits);
        Ok(raw as i32)
    }

    pub fn fixed_xy(&self, point: [i32; 2], chart: &Chart) -> Result<[i32; 2]> {
        let moved = self.xy(point.map(|v| v as f64 / (1u64 << self.source_fixed_bits) as f64), chart);
        Ok([self.fixed_value(moved[0])?, self.fixed_value(moved[1])?])
    }

    pub fn fixed_distance(&self, raw: i32) -> Result<i32> {
        self.fixed_value(raw as f64 / (1u64 << self.source_fixed_bits) as f64 * self.scale)
    }

    /// Snap only stored camera theatre edges, never arbitrary geometry points.
    pub fn camera_rect(&self, chart: &Chart, theatre: &str) -> Result<[[f32; 2]; 2]> {
        let source = chart.theatre_rects.get(theatre).with_context(|| format!("Chart {} has no theatre {theatre}", chart.key))?;
        Ok(source.map(|p| self.xy(p, chart).map(|v| ((v * 4096.0).round_ties_even() / 4096.0) as f32)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connected_seam_is_identical_in_camera_and_affine_spaces() {
        let plan = Plan::proven();
        let europe = plan.chart("2").unwrap();
        let india = plan.chart("3").unwrap();
        assert!(std::ptr::eq(europe, india));
        assert_eq!(plan.xy([440.0, 200.0], europe), plan.xy([440.0, 200.0], india));
        assert_eq!(plan.camera_rect(europe, "2").unwrap()[1][0], plan.camera_rect(india, "3").unwrap()[0][0]);
        let point = [-175.75, 519.25];
        let back = plan.inverse_xy(plan.xy(point, europe), europe);
        for axis in 0..2 { assert!((back[axis] - point[axis]).abs() < 1e-10); }
    }

    #[test]
    fn unsupported_layout_and_fixedpoint_overflow_are_rejected() {
        let mut plan = Plan::proven();
        plan.scale = 2.0;
        assert!(plan.validate().is_err());
        assert!(plan.fixed_value(2048.0).is_err());
        assert!(plan.fixed_value(f64::NAN).is_err());
        assert_eq!(plan.fixed_value(-2048.0).unwrap(), i32::MIN);
    }
}
