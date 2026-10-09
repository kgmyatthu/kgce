//! Region borders and settlement markers drawn over the Ownership map (visual only; selection hit-tests the filled
//! polygons). Built from the same `RegionData` the polygons come from, so split regions get theirs automatically.

use super::Editor;
use crate::{model::Rules, regionmap::regions::RegionData};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

/// A closed outline ring in world coordinates.
struct Strip {
    land: bool,
    points: Vec<[f32; 2]>,
    /// `[min x, min y, max x, max y]`
    bounds: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaceKind {
    City,
    Town,
    Port,
    Resource,
}

struct Place {
    kind: PlaceKind,
    at: [f32; 2],
    label: String,
}

#[derive(Default)]
pub(super) struct Overlay {
    strips: Vec<Strip>,
    places: Vec<Place>,
}

impl Overlay {
    pub(super) fn build(data: &RegionData, rules: &Rules) -> Self {
        let mut overlay = Self::default();
        for region in &data.regions {
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
            if let Some(at) = children.first().and_then(|node| node.as_xy().ok()) {
                let label = rules.regions[&region.key]["capital_settlement_name"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| region.key.clone());
                overlay.places.push(Place {
                    kind: PlaceKind::City,
                    at,
                    label,
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
                overlay.places.push(Place { kind, at, label });
            }
        }
        overlay
    }
}

impl Editor {
    pub(super) fn draw_overlay(&self, painter: &egui::Painter, rect: Rect) {
        let scale = self.scale(rect);
        let to = |point: [f32; 2]| {
            Pos2::new(
                rect.center().x + (point[0] - self.center.x) * scale,
                rect.center().y - (point[1] - self.center.y) * scale,
            )
        };
        let (low, high) = (
            self.to_world(rect.left_bottom(), rect),
            self.to_world(rect.right_top(), rect),
        );
        let visible = |bounds: [f32; 4]| {
            bounds[2] >= low[0] && bounds[0] <= high[0] && bounds[3] >= low[1] && bounds[1] <= high[1]
        };
        let land = Stroke::new(1.0, Color32::from_rgba_unmultiplied(34, 34, 34, 153));
        let water = Stroke::new(1.0, Color32::from_rgba_unmultiplied(150, 190, 215, 40));
        for strip in self.overlay.strips.iter().filter(|strip| visible(strip.bounds)) {
            // skip vertices closer than a pixel to the previous one: at small zoom most of a ring is sub-pixel
            let mut points: Vec<Pos2> = Vec::with_capacity(strip.points.len().min(4096));
            for point in &strip.points {
                let at = to(*point);
                if points.last().is_none_or(|last| last.distance_sq(at) >= 0.64) {
                    points.push(at);
                }
            }
            if points.len() > 1 {
                painter.add(egui::Shape::closed_line(
                    points,
                    if strip.land { land } else { water },
                ));
            }
        }
        let font = FontId::proportional(11.0);
        let outline = Stroke::new(1.0, Color32::BLACK);
        for place in &self.overlay.places {
            if !visible([place.at[0], place.at[1], place.at[0], place.at[1]]) {
                continue;
            }
            let at = to(place.at);
            let label_zoom = match place.kind {
                PlaceKind::City => 1.4,
                _ => 2.5,
            };
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
                        Color32::from_rgb(70, 140, 255),
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
