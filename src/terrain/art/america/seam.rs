//! Join generated land to sound original terrain, not to the old dark parchment rim.
//! Work on the heightmap lattice; colour correction is sampled continuously at texture resolution.

use crate::terrain::art::{
    grid::{Arr2, Rgbf},
    painter::smoothstep,
    prims,
};
use anyhow::{Result, ensure};

pub(super) const RIM_UNITS: f32 = 8.0;
pub(super) const ORIGINAL_UNITS: f32 = 20.0;
pub(super) const GENERATED_UNITS: f32 = 40.0;
const FILTER_UNITS: f64 = 3.0;

pub(super) struct Transition {
    pub original: Arr2<f32>,
    pub correction: Arr2<f32>,
    near: Arr2<u32>,
    sigma: f64,
}

impl Transition {
    /// Ownership controls the feather; reference eligibility only controls its sampled correction.
    /// Dark rock underlays remain original land, but cannot stain generated land via colour matching.
    pub fn new(
        generated: &Arr2<bool>,
        stock: &Arr2<bool>,
        references: &Arr2<bool>,
        cells_per_unit: f32,
    ) -> Result<Self> {
        ensure!(
            generated.h == stock.h
                && generated.w == stock.w
                && references.h == stock.h
                && references.w == stock.w,
            "America transition grid sizes differ"
        );
        ensure!(
            generated.d.iter().any(|&v| v) && stock.d.iter().any(|&v| v),
            "America transition needs generated and original land"
        );
        let to_generated = prims::edt(&generated.map(|v| !v));
        let to_stock = prims::edt(&stock.map(|v| !v));
        let anchors = Arr2::from_vec(
            stock.h,
            stock.w,
            (0..stock.d.len())
                .map(|k| {
                    stock.d[k] && references.d[k] && to_generated.d[k] > RIM_UNITS * cells_per_unit
                })
                .collect(),
        );
        ensure!(
            anchors.d.iter().any(|&v| v),
            "America has no original land beyond its decorative rim"
        );
        let (_, near) = prims::edt_with_indices(&anchors.map(|v| !v));
        let original = Arr2::from_vec(
            stock.h,
            stock.w,
            (0..stock.d.len())
                .map(|k| {
                    if generated.d[k] {
                        0.0
                    } else if stock.d[k] {
                        smoothstep(
                            (to_generated.d[k] / cells_per_unit - RIM_UNITS)
                                / (ORIGINAL_UNITS - RIM_UNITS),
                        )
                    } else {
                        1.0
                    }
                })
                .collect(),
        );
        let correction = to_stock.map(|d| 1.0 - smoothstep(d / (GENERATED_UNITS * cells_per_unit)));
        Ok(Self {
            original,
            correction,
            near,
            sigma: FILTER_UNITS * cells_per_unit as f64,
        })
    }

    /// Extend the original-minus-generated residual, then smooth the field rather than copying nearest heights.
    /// This retains DEM slopes/detail and removes the constant-height shelf made by nearest-value tapering.
    pub fn residual(&self, sample: impl Fn(usize) -> f32) -> Arr2<f32> {
        let field = self.near.map(|i| sample(i as usize));
        prims::gaussian_filter(&field, self.sigma)
    }

    pub fn colour_residual(&self, sample: impl Fn(usize) -> Rgbf) -> Arr2<Rgbf> {
        let field = self.near.map(|i| sample(i as usize));
        prims::gaussian_filter_rgb(&field, self.sigma)
    }

    pub fn mix(&self, k: usize, generated: f32, original: f32, residual: f32) -> f32 {
        let w = self.original.d[k];
        let corrected = generated + self.correction.d[k] * residual;
        corrected * (1.0 - w) + original * w
    }
}

/// One interpolation stencil for weights and RGB; no per-pixel allocations.
pub(super) struct Sample {
    indices: [usize; 4],
    weights: [f32; 4],
}

impl Sample {
    pub fn texture(r: usize, c: usize, h: usize, w: usize) -> Self {
        let y = ((r as f32 + 0.5) / 8.0 - 0.5).clamp(0.0, (h - 1) as f32);
        let x = ((c as f32 + 0.5) / 8.0 - 0.5).clamp(0.0, (w - 1) as f32);
        let (r0, c0) = (y.floor() as usize, x.floor() as usize);
        let (r1, c1) = ((r0 + 1).min(h - 1), (c0 + 1).min(w - 1));
        let (fy, fx) = (y - r0 as f32, x - c0 as f32);
        Self {
            indices: [r0 * w + c0, r0 * w + c1, r1 * w + c0, r1 * w + c1],
            weights: [
                (1.0 - fy) * (1.0 - fx),
                (1.0 - fy) * fx,
                fy * (1.0 - fx),
                fy * fx,
            ],
        }
    }

    pub fn scalar(&self, values: &[f32]) -> f32 {
        self.indices
            .iter()
            .zip(&self.weights)
            .map(|(&i, &w)| values[i] * w)
            .sum()
    }

    pub fn rgb(&self, values: &[Rgbf]) -> Rgbf {
        let mut out = [0.0; 3];
        for (&i, &w) in self.indices.iter().zip(&self.weights) {
            for ch in 0..3 {
                out[ch] += values[i][ch] * w;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_offset_slopes_without_a_rim_or_boundary_step() {
        let (h, w, edge) = (12, 192, 96);
        let generated = Arr2::from_vec(h, w, (0..h * w).map(|k| k % w < edge).collect());
        let stock = generated.map(|v| !v);
        let t = Transition::new(&generated, &stock, &stock, 1.0).unwrap();
        let dem = |k: usize| 30.0 + (k % w) as f32 * 0.25;
        let original = |k: usize| dem(k) + if k % w < edge + 8 { -20.0 } else { 30.0 };
        let residual = t.residual(|k| original(k) - dem(k));
        let row: Vec<f32> = (0..w)
            .map(|k| t.mix(k, dem(k), original(k), residual.d[k]))
            .collect();
        assert!(
            (row[edge] - row[edge - 1]).abs() < 0.5,
            "old edge must not be a cliff"
        );
        for c in edge..edge + 8 {
            assert!(
                (row[c] - (dem(c) + 30.0)).abs() < 0.01,
                "decorative rim leaked into join"
            );
        }
        for c in 0..edge - 40 {
            assert_eq!(row[c], dem(c));
        }
        for c in edge + 20..w {
            assert_eq!(row[c], original(c));
        }
        assert!(row.windows(2).all(|p| (p[1] - p[0]).abs() < 1.5));
    }

    #[test]
    fn water_and_dark_underlays_cannot_supply_anchors_or_preserve_the_old_rim() {
        let (h, w) = (40, 160);
        let generated =
            Arr2::from_vec(h, w, (0..h * w).map(|k| k / w < 20 && k % w < 80).collect());
        let stock = Arr2::from_vec(
            h,
            w,
            (0..h * w).map(|k| k / w < 20 && k % w >= 80).collect(),
        );
        let references = Arr2::from_vec(h, w, (0..h * w).map(|k| k % w >= 110).collect());
        let t = Transition::new(&generated, &stock, &references, 1.0).unwrap();
        let residual = t.residual(|k| {
            if stock.d[k] && references.d[k] {
                12.0
            } else {
                -200.0
            }
        });
        assert_eq!(
            t.original.d[82], 0.0,
            "excluding a dark reference must not preserve the dark rim"
        );
        assert!(residual.d.iter().all(|v| (*v - 12.0).abs() < 0.001));
        assert!(t.original.d[20 * w..].iter().all(|&v| v == 1.0));
    }
}
