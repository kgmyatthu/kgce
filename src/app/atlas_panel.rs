//! Saved atlas/building settings and source/display coordinate boundary.
use super::{Editor, Polygon, polygon_with_triangles};
use crate::atlas::{Chart, Plan};
use eframe::egui::{self, Pos2, Rect};

impl Editor {
    pub(super) fn atlas_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Campaign scale & layout").default_open(true).show(ui, |ui| {
            ui.label(if self.project.atlas.is_some() { "Map: 1.3516979217529297× · connected atlas" } else { "Map: original scale · fixed source theatres" });
            ui.small("Europe–India stays connected. Stock executable; no patching.");
            if ui.add_enabled(!self.busy, egui::Button::new("Apply proven setup")).clicked() {
                let mut next = self.project.clone();
                next.atlas = Some(Plan::proven());
                next.building_scale = 0.5;
                self.set_spatial_project(next);
            }
            if self.project.atlas.is_some() && ui.add_enabled(!self.busy, egui::Button::new("Use original map layout")).clicked() {
                let mut next = self.project.clone();
                next.atlas = None;
                self.set_spatial_project(next);
            }
            let mut factor = self.project.building_scale;
            let response = ui.add_enabled(!self.busy, egui::Slider::new(&mut factor, 0.1..=1.0).text("Building size").fixed_decimals(2));
            if response.changed() {
                self.project.building_scale = factor;
                self.dirty = true;
                self.ui.analysis_wanted = self.source_ready;
            }
            ui.small("0.50 = proven half size; 1.00 = no override. Separate pack affects campaign buildings, not characters.");
            if let Some(plan) = &self.project.atlas {
                ui.collapsing("Saved theatre arrangement", |ui| {
                    for chart in &plan.charts {
                        for theatre in &chart.theatres {
                            if let Ok(rect) = plan.camera_rect(chart, theatre) {
                                ui.small(format!("{} ({theatre}): ({:.3}, {:.3})–({:.3}, {:.3})", chart.key, rect[0][0], rect[0][1], rect[1][0], rect[1][1]));
                            }
                        }
                    }
                });
            }
            ui.small("Save project to reproduce. Export a new campaign; existing saves are not rescaled.");
        });
    }

    fn set_spatial_project(&mut self, project: crate::model::Project) {
        match self.install_project(project) {
            Ok(()) => {
                self.leave_split_tool();
                self.leave_activate_tool();
                self.leave_settlement_tool();
                self.dirty = true;
                self.ui.analysis_wanted = self.source_ready;
                self.fit_map();
                self.push_message("Campaign layout and building settings changed. Save the project to reproduce this export.".into());
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    pub(super) fn display_point(&self, source: [f32; 2]) -> [f32; 2] {
        let Some(plan) = &self.project.atlas else { return source };
        let point = source.map(f64::from);
        plan.chart_at(point, true).map(|chart| plan.xy(point, chart).map(|v| v as f32)).unwrap_or([f32::NAN; 2])
    }

    pub(super) fn display_to_source(&self, display: [f32; 2]) -> [f32; 2] {
        let Some(plan) = &self.project.atlas else { return display };
        for chart in &plan.charts {
            let point = plan.inverse_xy(display.map(f64::from), chart);
            if chart.crop_rects.iter().any(|rect| Plan::contains(*rect, point)) {
                return point.map(|v| v as f32);
            }
        }
        [f32::NAN; 2]
    }

    pub(super) fn to_display_world(&self, point: Pos2, rect: Rect) -> [f32; 2] {
        let scale = self.scale(rect);
        [(point.x - rect.center().x) / scale + self.center.x,
         self.center.y - (point.y - rect.center().y) / scale]
    }

    pub(super) fn screen_display(&self, point: [f32; 2], rect: Rect) -> Pos2 {
        let scale = self.scale(rect);
        Pos2::new(rect.center().x + (point[0] - self.center.x) * scale,
                  rect.center().y - (point[1] - self.center.y) * scale)
    }

    pub(super) fn rendered_polygons(&self) -> &[Polygon] {
        if self.project.atlas.is_some() { &self.atlas_polygons } else { &self.polygons }
    }

    pub(super) fn draw_atlas_frames(&self, painter: &egui::Painter, rect: Rect) {
        let Some(plan) = &self.project.atlas else { return };
        let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(110, 165, 195));
        for chart in &plan.charts {
            for theatre in &chart.theatres {
                let Ok([lo, hi]) = plan.camera_rect(chart, theatre) else { continue };
                let corners = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
                    .map(|p| self.screen_display(p, rect));
                for i in 0..4 { painter.line_segment([corners[i], corners[(i + 1) % 4]], stroke); }
                let name = match theatre.as_str() {
                    "1" => "America", "2" => "Europe", "3" => "India",
                    "2113354257" => "West Africa", "836795134" => "Madagascar",
                    "1197997136" => "Brazil", "-1133129049" => "East Indies", _ => theatre,
                };
                painter.text(corners[3] + egui::vec2(4.0, 4.0), egui::Align2::LEFT_TOP, name,
                    egui::FontId::proportional(12.0), stroke.color);
            }
        }
    }

    pub(super) fn rebuild_atlas_preview(&mut self) {
        self.atlas_polygons.clear();
        let Some(plan) = &self.project.atlas else { return };
        let crops: Vec<_> = plan.charts.iter().map(|chart| (chart, disjoint_crops(chart))).collect();
        for source in &self.polygons {
            let mut triangles = Vec::new();
            for (chart, rects) in &crops {
                for rect in rects {
                    if f64::from(source.bounds[2]) < rect[0][0] || f64::from(source.bounds[0]) > rect[1][0]
                        || f64::from(source.bounds[3]) < rect[0][1] || f64::from(source.bounds[1]) > rect[1][1] { continue; }
                    for triangle in &source.triangles {
                        clip_triangle(*triangle, *rect, plan, chart, &mut triangles);
                    }
                }
            }
            if !triangles.is_empty() {
                self.atlas_polygons.push(polygon_with_triangles(source.key.clone(), triangles, source.wild));
            }
        }
    }
}

// Partition rectangle unions once, so overlapping Europe/India crops do not
// double-paint geometry. Rectangle boundaries preserve holes in input triangles.
fn disjoint_crops(chart: &Chart) -> Vec<crate::atlas::Rect> {
    let mut xs: Vec<_> = chart.crop_rects.iter().flat_map(|r| [r[0][0], r[1][0]]).collect();
    let mut ys: Vec<_> = chart.crop_rects.iter().flat_map(|r| [r[0][1], r[1][1]]).collect();
    xs.sort_by(f64::total_cmp); xs.dedup();
    ys.sort_by(f64::total_cmp); ys.dedup();
    let mut result = Vec::new();
    for x in xs.windows(2) { for y in ys.windows(2) {
        if chart.crop_rects.iter().any(|r| Plan::contains(*r, [(x[0] + x[1]) * 0.5, (y[0] + y[1]) * 0.5])) {
            result.push([[x[0], y[0]], [x[1], y[1]]]);
        }
    }}
    result
}

fn clip_triangle(triangle: [[f32; 2]; 3], rect: crate::atlas::Rect, plan: &Plan, chart: &Chart, out: &mut Vec<[[f32; 2]; 3]>) {
    let mut polygon = [[0.0; 2]; 8];
    for (slot, point) in polygon.iter_mut().zip(triangle) { *slot = point.map(f64::from); }
    let mut count = 3;
    for (axis, bound, lower) in [(0, rect[0][0], true), (0, rect[1][0], false), (1, rect[0][1], true), (1, rect[1][1], false)] {
        if count < 3 { return; }
        let mut clipped = [[0.0; 2]; 8];
        let mut next = 0;
        let inside = |p: [f64; 2]| if lower { p[axis] >= bound } else { p[axis] <= bound };
        let mut previous = polygon[count - 1];
        for &current in &polygon[..count] {
            if inside(current) != inside(previous) {
                let t = (bound - previous[axis]) / (current[axis] - previous[axis]);
                clipped[next] = std::array::from_fn(|i| previous[i] + t * (current[i] - previous[i]));
                next += 1;
            }
            if inside(current) { clipped[next] = current; next += 1; }
            previous = current;
        }
        polygon = clipped; count = next;
    }
    for i in 1..count.saturating_sub(1) {
        out.push([polygon[0], polygon[i], polygon[i + 1]].map(|p| plan.xy(p, chart).map(|v| v as f32)));
    }
}
