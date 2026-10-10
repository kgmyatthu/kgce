//! Region borders and settlement markers drawn over the Ownership map (visual only; selection hit-tests the filled
//! polygons). Built from the same `RegionData` the polygons come from, so split regions get theirs automatically.
//!
//! Every capital, town and port marker carries the stable identity the settlement engine addresses it by
//! ([`PlaceId`]: region key and slot id), so the settlement tool hit-tests and moves exactly the marker that was drawn.
//! Drawing goes through the display transform ([`Editor::to_screen`] semantics): with the connected atlas on, a point
//! belongs to one chart, a border never connects points of two charts, and culling is done in screen space.

use super::Editor;
use crate::{model::Rules, regionmap::regions::RegionData, settlement};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

/// A closed outline ring in world coordinates.
struct Strip {
    land: bool,
    points: Vec<[f32; 2]>,
    /// `[min x, min y, max x, max y]`
    bounds: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PlaceKind {
    City,
    Town,
    Port,
    Resource,
}

/// Stable identity of a settlement marker: the region and the slot id the settlement engine resolves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PlaceId {
    pub(super) region: String,
    pub(super) slot: String,
    /// ports are shown but cannot be moved
    pub(super) movable: bool,
}

pub(super) struct Place {
    pub(super) kind: PlaceKind,
    pub(super) at: [f32; 2],
    pub(super) label: String,
    /// `None` for resource slots
    pub(super) id: Option<PlaceId>,
}

#[derive(Default)]
pub(super) struct Overlay {
    strips: Vec<Strip>,
    pub(super) places: Vec<Place>,
}

/// How a settlement marker is decorated while the settlement tool is active.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MarkerState {
    Normal,
    /// the project moves it away from its original position
    Moved,
    /// being dragged: the marker stays dim at its current position, the candidate is drawn by the tool
    Dragged,
    /// shown but not movable (ports)
    Locked,
}

impl Overlay {
    pub(super) fn build(data: &RegionData, rules: &Rules) -> Self {
        let mut overlay = Self::default();
        for (region_index, region) in data.regions.iter().enumerate() {
            let land = region.kind == "land";
            for area in &region.areas {
                for outline in &area.outlines {
                    let points: Vec<[f32; 2]> = outline
                        .ring
                        .iter()
                        .map(|&id| data.vertices[id as usize])
                        .collect();
                    let mut bounds = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
                    for point in &points {
                        bounds[0] = bounds[0].min(point[0]);
                        bounds[1] = bounds[1].min(point[1]);
                        bounds[2] = bounds[2].max(point[0]);
                        bounds[3] = bounds[3].max(point[1]);
                    }
                    if points.len() > 1 {
                        overlay.strips.push(Strip { land, points, bounds });
                    }
                }
            }
            let Some(settlement) = &region.settlement else {
                continue;
            };
            let Ok(children) = settlement.children() else {
                continue;
            };
            let targets = settlement::targets(data, region_index).unwrap_or_default();
            if let Some(at) = children.first().and_then(|node| node.as_xy().ok()) {
                let label = rules.regions[&region.key]["capital_settlement_name"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| region.key.clone());
                let id = targets
                    .iter()
                    .find(|target| target.kind == settlement::Kind::Capital)
                    .map(|target| PlaceId { region: region.key.clone(), slot: target.slot.clone(), movable: true });
                overlay.places.push(Place {
                    kind: PlaceKind::City,
                    at,
                    label,
                    id,
                });
            }
            let Some(slots) = children.get(2).and_then(|node| node.items().ok()) else {
                continue;
            };
            for slot in slots {
                let (Some(key), Some(position)) = (
                    slot.first().and_then(|node| node.as_str().ok()),
                    slot.get(2),
                ) else {
                    continue;
                };
                let Some(at) = position
                    .as_xy()
                    .ok()
                    .or_else(|| position.as_xyz().ok().map(|v| [v[0], v[2]]))
                else {
                    continue;
                };
                let mut parts = key.split(':');
                let kind = match parts.next() {
                    Some("settlement") => continue,
                    Some("town") => PlaceKind::Town,
                    Some("port") => PlaceKind::Port,
                    _ => PlaceKind::Resource,
                };
                let label = parts.nth(1).unwrap_or(key).replace('_', " ");
                let id = match kind {
                    PlaceKind::Town | PlaceKind::Port => Some(PlaceId {
                        region: region.key.clone(),
                        slot: key.to_string(),
                        movable: kind == PlaceKind::Town,
                    }),
                    _ => None,
                };
                overlay.places.push(Place { kind, at, label, id });
            }
        }
        overlay
    }
}

impl Editor {
    /// Screen position of a source-space point and the atlas chart it belongs to (always chart 0 without an atlas);
    /// `None` where the atlas has no chart (gaps between charts).
    fn overlay_point(&self, point: [f32; 2], rect: Rect) -> Option<(usize, Pos2)> {
        let position = match &self.project.atlas {
            None => return Some((0, self.screen_display(point, rect))),
            Some(plan) => {
                let source = point.map(f64::from);
                let chart = plan.chart_at(source, true).ok()?;
                let index = plan.charts.iter().position(|candidate| candidate.key == chart.key)?;
                (index, plan.xy(source, chart).map(|value| value as f32))
            }
        };
        let at = self.screen_display(position.1, rect);
        (at.x.is_finite() && at.y.is_finite()).then_some((position.0, at))
    }

    /// Screen position of the marker at `point` (`None` when it has no place on the map).
    pub(super) fn marker_screen(&self, point: [f32; 2], rect: Rect) -> Option<Pos2> {
        self.overlay_point(point, rect).map(|(_, at)| at)
    }

    pub(super) fn draw_overlay(&self, painter: &egui::Painter, rect: Rect) {
        let view = rect.expand(64.0);
        // a strip is skipped when its bounding box, mapped corner by corner into one chart, lies off screen
        let visible = |bounds: [f32; 4]| {
            let corners = [[bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[0], bounds[3]], [bounds[2], bounds[3]]];
            let mapped: Vec<(usize, Pos2)> = corners.iter().filter_map(|corner| self.overlay_point(*corner, rect)).collect();
            if mapped.len() < 4 || mapped.iter().any(|(chart, _)| *chart != mapped[0].0) {
                return true;
            }
            let (mut low, mut high) = (mapped[0].1, mapped[0].1);
            for (_, at) in &mapped {
                low = low.min(*at);
                high = high.max(*at);
            }
            Rect::from_min_max(low, high).intersects(view)
        };
        let land = Stroke::new(1.0, Color32::from_rgba_unmultiplied(34, 34, 34, 153));
        let water = Stroke::new(1.0, Color32::from_rgba_unmultiplied(150, 190, 215, 40));
        for strip in self.overlay.strips.iter().filter(|strip| visible(strip.bounds)) {
            let stroke = if strip.land { land } else { water };
            // one polyline per run of points in the same chart; skip vertices closer than a pixel to the previous one
            let mut runs: Vec<(usize, Vec<Pos2>)> = Vec::new();
            let mut broken = false;
            for point in &strip.points {
                match self.overlay_point(*point, rect) {
                    Some((chart, at)) => match runs.last_mut() {
                        Some((last_chart, points)) if *last_chart == chart => {
                            if points.last().is_none_or(|last| last.distance_sq(at) >= 0.64) {
                                points.push(at);
                            }
                        }
                        _ => {
                            broken |= !runs.is_empty();
                            runs.push((chart, vec![at]));
                        }
                    },
                    None => broken |= !runs.is_empty(),
                }
            }
            if runs.len() == 1 && !broken && runs[0].1.len() > 1 {
                painter.add(egui::Shape::closed_line(runs.remove(0).1, stroke));
            } else {
                for (_, points) in runs.into_iter().filter(|(_, points)| points.len() > 1) {
                    painter.add(egui::Shape::line(points, stroke));
                }
            }
        }
        let font = FontId::proportional(11.0);
        let outline = Stroke::new(1.0, Color32::BLACK);
        let editing = self.settlement_tool_active();
        for place in &self.overlay.places {
            let Some((_, at)) = self.overlay_point(place.at, rect) else {
                continue;
            };
            if !view.contains(at) {
                continue;
            }
            let label_zoom = match place.kind {
                PlaceKind::City => 1.4,
                _ => 2.5,
            };
            let state = match (&place.id, editing) {
                (Some(id), true) => self.marker_state(id),
                _ => MarkerState::Normal,
            };
            if editing && place.kind != PlaceKind::Resource {
                match state {
                    MarkerState::Moved => {
                        painter.circle_stroke(at, 9.5, Stroke::new(2.0, Color32::from_rgb(110, 225, 130)));
                    }
                    MarkerState::Dragged => {
                        painter.circle_stroke(at, 9.5, Stroke::new(1.5, Color32::from_white_alpha(90)));
                    }
                    MarkerState::Normal if place.id.is_some() => {
                        painter.circle_stroke(at, 8.0, Stroke::new(1.0, Color32::from_white_alpha(70)));
                    }
                    _ => {}
                }
            }
            match place.kind {
                PlaceKind::City => {
                    painter.circle(at, 4.0, Color32::from_rgb(255, 240, 200), outline);
                }
                PlaceKind::Town => {
                    let square = Rect::from_center_size(at, Vec2::splat(6.0));
                    painter.rect(square, 0.0, Color32::from_rgb(225, 225, 225), outline, egui::StrokeKind::Middle);
                }
                PlaceKind::Port => {
                    painter.add(egui::Shape::convex_polygon(
                        vec![
                            at + Vec2::new(0.0, -4.5),
                            at + Vec2::new(4.5, 3.5),
                            at + Vec2::new(-4.5, 3.5),
                        ],
                        if state == MarkerState::Locked { Color32::from_rgb(70, 100, 160) } else { Color32::from_rgb(70, 140, 255) },
                        outline,
                    ));
                }
                PlaceKind::Resource => {
                    painter.add(egui::Shape::convex_polygon(
                        vec![
                            at + Vec2::new(0.0, -4.0),
                            at + Vec2::new(4.0, 0.0),
                            at + Vec2::new(0.0, 4.0),
                            at + Vec2::new(-4.0, 0.0),
                        ],
                        Color32::from_rgb(235, 170, 60),
                        outline,
                    ));
                }
            }
            if self.zoom >= label_zoom {
                let text_at = at + Vec2::new(7.0, 0.0);
                painter.text(
                    text_at + Vec2::new(1.0, 1.0),
                    egui::Align2::LEFT_CENTER,
                    &place.label,
                    font.clone(),
                    Color32::BLACK,
                );
                painter.text(
                    text_at,
                    egui::Align2::LEFT_CENTER,
                    &place.label,
                    font.clone(),
                    if place.kind == PlaceKind::City {
                        Color32::WHITE
                    } else {
                        Color32::from_rgb(210, 220, 230)
                    },
                );
            }
        }
    }
}
