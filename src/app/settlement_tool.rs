//! Settlement tool of the Ownership map: drag a capital (circle) or town (square) marker to a new place inside its own
//! region. The project stores [`Move`]s in source coordinates (`project.settlement_moves`); the map geometry replays
//! them (`settlement::preview`), so the markers, the next move's validation and the export always agree.
//!
//! * press and drag a marker with the primary button: the candidate follows the pointer and is validated every frame
//!   (`settlement::validate`: inside the region's own passable area, outside holes, node clearance, spacing); green is
//!   valid, red shows why not
//! * release over a valid place: one move is committed through [`Editor::install_project`] (all or nothing) and one undo
//!   step is recorded; release over an invalid place, outside the map or `Esc` while dragging: nothing changes
//! * right and middle drags still pan, the wheel still zooms; this tool only reacts to the primary button
//! * `Ctrl+Z` / the Undo button take back the last committed move; every move can also be reset from the list

use super::{
    Editor,
    overlay::{MarkerState, PlaceId, PlaceKind},
    split_tool::MapTool,
};
use crate::{
    map::splits,
    settlement::{self, Move},
};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

/// A marker is grabbed when the press is within this many points of it.
const HIT_RADIUS: f32 = 10.0;
const VALID: Color32 = Color32::from_rgb(110, 225, 130);
const INVALID: Color32 = Color32::from_rgb(255, 120, 120);

/// State of the settlement tool.
#[derive(Default)]
pub(super) struct SettlementDraft {
    drag: Option<Drag>,
    /// `project.settlement_moves` before each committed move, newest last
    history: Vec<Vec<Move>>,
    /// last committed or cancelled action (green)
    note: Option<String>,
    /// why the last release changed nothing (red)
    error: Option<String>,
}

struct Drag {
    id: PlaceId,
    kind: PlaceKind,
    label: String,
    /// where the marker stands now (and keeps standing unless the release commits)
    origin: [f32; 2],
    /// pointer in source coordinates; `None` over a gap of the atlas or off the map
    candidate: Option<[f32; 2]>,
    verdict: Result<(), String>,
}

impl Editor {
    pub(super) fn settlement_tool_active(&self) -> bool {
        self.tool == MapTool::Settlement
    }

    /// Decoration of the marker `id` while the tool is active.
    pub(super) fn marker_state(&self, id: &PlaceId) -> MarkerState {
        if let Some(drag) = &self.settlement_draft.drag {
            if drag.id == *id {
                return MarkerState::Dragged;
            }
        }
        if !id.movable {
            return MarkerState::Locked;
        }
        if self
            .project
            .settlement_moves
            .iter()
            .any(|movement| movement.region == id.region && movement.slot == id.slot)
        {
            return MarkerState::Moved;
        }
        MarkerState::Normal
    }

    /// "Move settlements" toggle above the region list.
    pub(super) fn settlement_toggle(&mut self, ui: &mut egui::Ui) {
        let active = self.tool == MapTool::Settlement;
        let enabled = active || (self.geometry.is_some() && !self.busy);
        let response = ui
            .add_enabled(
                enabled,
                egui::Button::new("Move settlements")
                    .selected(active)
                    .min_size(Vec2::new(ui.available_width(), 26.0)),
            )
            .on_hover_text("Drag a capital or town marker to a new place inside its own region")
            .on_disabled_hover_text("Import the original game files first");
        if response.clicked() {
            if active {
                self.leave_settlement_tool();
            } else {
                self.leave_split_tool();
                self.leave_activate_tool();
                self.tool = MapTool::Settlement;
                self.settlement_draft = SettlementDraft::default();
                self.show_overlay = true;
                self.canonical_moves();
                self.mesh_dirty = true;
                self.push_message("Settlement tool: drag a capital or town marker inside its region.".into());
            }
        }
    }

    pub(super) fn leave_settlement_tool(&mut self) {
        if self.tool == MapTool::Settlement {
            self.tool = MapTool::Select;
        }
        self.settlement_draft = SettlementDraft::default();
        self.mesh_dirty = true;
    }

    /// Escape cancels a drag in progress, else leaves the tool; Ctrl+Z undoes the last committed move.
    pub(super) fn settlement_keys(&mut self, ctx: &egui::Context) {
        if self.tool != MapTool::Settlement {
            return;
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            if self.settlement_draft.drag.take().is_some() {
                self.settlement_draft.error = None;
                self.settlement_draft.note = Some("Move cancelled; nothing changed.".into());
            } else {
                self.leave_settlement_tool();
            }
        } else if !ctx.wants_keyboard_input()
            && self.settlement_draft.drag.is_none()
            && ctx.input(|input| input.modifiers.command && input.key_pressed(egui::Key::Z))
        {
            self.undo_settlement_move();
        }
    }

    /// Rewrites the slot of every stored move to its canonical id (a move set with a bare name from the command line
    /// matches its marker by id).
    fn canonical_moves(&mut self) {
        let Some(geometry) = &self.geometry else { return };
        let mut moves = self.project.settlement_moves.clone();
        let mut changed = false;
        for movement in &mut moves {
            if let Ok(target) = settlement::resolve(&geometry.data, &movement.region, &movement.slot) {
                if target.slot != movement.slot {
                    movement.slot = target.slot;
                    changed = true;
                }
            }
        }
        if changed {
            let mut next = self.project.clone();
            next.settlement_moves = moves;
            if self.install_project(next).is_ok() {
                self.dirty = true;
            }
        }
    }

    /// The marker (with a stable id) nearest to `point` within the grab radius.
    fn marker_at(&self, point: Pos2, rect: Rect) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (index, place) in self.overlay.places.iter().enumerate() {
            if place.id.is_none() {
                continue;
            }
            let Some(at) = self.marker_screen(place.at, rect) else { continue };
            let distance = at.distance(point);
            if distance <= HIT_RADIUS && best.is_none_or(|(_, nearest)| distance < nearest) {
                best = Some((index, distance));
            }
        }
        best.map(|(index, _)| index)
    }

    /// A plain primary click: report the marker under the pointer and select its region.
    pub(super) fn settlement_click(&mut self, point: Pos2, rect: Rect) {
        let Some(index) = self.marker_at(point, rect) else { return };
        let place = &self.overlay.places[index];
        let Some(id) = place.id.clone() else { return };
        let at = place.at;
        let label = place.label.clone();
        self.settlement_draft.error = None;
        self.settlement_draft.note = Some(format!(
            "{label} ({}) stands at ({:.2}, {:.2}){}",
            id.slot,
            at[0],
            at[1],
            if id.movable { "" } else { "; ports cannot be moved" }
        ));
        self.select(id.region, false);
    }

    /// Pointer handling of the map panel while the tool is active: begin, follow and finish a marker drag.
    pub(super) fn settlement_input(&mut self, ui: &egui::Ui, response: &egui::Response, rect: Rect) {
        if self.tool != MapTool::Settlement {
            return;
        }
        let primary = egui::PointerButton::Primary;
        if self.settlement_draft.drag.is_none() {
            if let Some(hover) = response.hover_pos() {
                if self.marker_at(hover, rect).is_some() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            }
        }
        if response.drag_started_by(primary) && !self.busy {
            if let Some(press) = ui.input(|input| input.pointer.press_origin()) {
                self.begin_drag(press, rect);
            }
        }
        if self.settlement_draft.drag.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            if response.dragged_by(primary) || response.drag_stopped_by(primary) {
                if let Some(pointer) = ui.input(|input| input.pointer.latest_pos()).or_else(|| response.interact_pointer_pos()) {
                    self.follow_drag(pointer, rect);
                }
            }
        }
        if response.drag_stopped_by(primary) {
            self.finish_drag();
        }
    }

    fn begin_drag(&mut self, press: Pos2, rect: Rect) {
        self.settlement_draft.error = None;
        self.settlement_draft.note = None;
        let Some(index) = self.marker_at(press, rect) else { return };
        let place = &self.overlay.places[index];
        let Some(id) = place.id.clone() else { return };
        if !id.movable {
            self.settlement_draft.error = Some(format!(
                "{} is a port: its landing crossings, dock and trade-route node follow the coastline, so ports stay where they are.",
                place.label
            ));
            return;
        }
        self.settlement_draft.drag = Some(Drag {
            id,
            kind: place.kind,
            label: place.label.clone(),
            origin: place.at,
            candidate: Some(place.at),
            verdict: Ok(()),
        });
    }

    fn follow_drag(&mut self, pointer: Pos2, rect: Rect) {
        let candidate = rect.contains(pointer)
            .then(|| self.to_world(pointer, rect))
            .filter(|point| point.iter().all(|value| value.is_finite()));
        let Some(drag) = &self.settlement_draft.drag else { return };
        if drag.candidate == candidate && candidate.is_some() {
            return;
        }
        let verdict = match (candidate, &self.geometry) {
            (Some(position), Some(geometry)) => settlement::validate(
                &geometry.data,
                &Move { region: drag.id.region.clone(), slot: drag.id.slot.clone(), position },
            )
            .map_err(|error| format!("{error:#}")),
            (None, _) => Err("The pointer is outside the visible map or between theatre charts.".to_string()),
            (_, None) => Err("The map geometry is not loaded.".to_string()),
        };
        if let Some(drag) = &mut self.settlement_draft.drag {
            drag.candidate = candidate;
            drag.verdict = verdict;
        }
    }

    /// The button went up: commit a valid candidate once, anything else leaves the project untouched.
    fn finish_drag(&mut self) {
        let Some(drag) = self.settlement_draft.drag.take() else { return };
        let Some(position) = drag.candidate else {
            self.settlement_draft.error = Some("Released outside the map: nothing moved.".into());
            return;
        };
        if let Err(reason) = drag.verdict {
            self.settlement_draft.error = Some(format!("Not moved: {reason}"));
            return;
        }
        if position == drag.origin {
            return;
        }
        self.commit_move(&drag.id, &drag.label, position);
    }

    /// Replaces the move of `id` (or drops it when `position` is the original place) and installs the new project; one
    /// undo step on success, the project untouched on failure.
    fn commit_move(&mut self, id: &PlaceId, label: &str, position: [f32; 2]) {
        let Some(base) = self.base_regions.clone() else { return };
        let previous = self.project.settlement_moves.clone();
        let mut next = self.project.clone();
        next.settlement_moves.clear();
        let data = match splits::replay(&base, &next) {
            Ok((data, _)) => data,
            Err(error) => {
                self.settlement_draft.error = Some(format!("{error:#}"));
                return;
            }
        };
        let original = match settlement::resolve(&data, &id.region, &id.slot) {
            Ok(target) => target.position,
            Err(error) => {
                self.settlement_draft.error = Some(format!("{error:#}"));
                return;
            }
        };
        // every stored move except the one of this target (whatever its spelling of the slot)
        next.settlement_moves = previous
            .iter()
            .filter(|movement| {
                movement.region != id.region
                    || settlement::resolve(&data, &movement.region, &movement.slot)
                        .map_or(true, |target| target.slot != id.slot)
            })
            .cloned()
            .collect();
        let back_home = (position[0] - original[0]).abs() < 1e-3 && (position[1] - original[1]).abs() < 1e-3;
        if !back_home {
            next.settlement_moves.push(Move { region: id.region.clone(), slot: id.slot.clone(), position });
        }
        match self.install_project(next) {
            Ok(()) => {
                self.dirty = true;
                self.settlement_draft.history.push(previous);
                self.settlement_draft.error = None;
                let text = if back_home {
                    format!("{label} is back at its original place.")
                } else {
                    format!("Moved {label} to ({:.2}, {:.2}).", position[0], position[1])
                };
                self.settlement_draft.note = Some(text.clone());
                self.push_message(format!("{text} Save or export to persist."));
            }
            Err(error) => self.settlement_draft.error = Some(format!("{error:#}")),
        }
    }

    /// Takes back the last committed move.
    pub(super) fn undo_settlement_move(&mut self) {
        let Some(previous) = self.settlement_draft.history.pop() else {
            self.settlement_draft.note = Some("Nothing to undo.".into());
            return;
        };
        if !self.install_moves(previous.clone(), "Undid the last settlement move.", false) {
            self.settlement_draft.history.push(previous);
        }
    }

    /// Resets one move of the list to the original place.
    fn reset_move(&mut self, index: usize) {
        let mut moves = self.project.settlement_moves.clone();
        if index >= moves.len() {
            return;
        }
        moves.remove(index);
        self.install_moves(moves, "Settlement returned to its original place.", true);
    }

    /// Installs `moves` as the project's move list (all or nothing); `record` stores the current list as an undo step.
    /// Returns whether it was installed.
    fn install_moves(&mut self, moves: Vec<Move>, note: &str, record: bool) -> bool {
        let before = self.project.settlement_moves.clone();
        let mut next = self.project.clone();
        next.settlement_moves = moves;
        match self.install_project(next) {
            Ok(()) => {
                self.dirty = true;
                if record {
                    self.settlement_draft.history.push(before);
                }
                self.settlement_draft.error = None;
                self.settlement_draft.note = Some(note.to_string());
                self.push_message(format!("{note} Save or export to persist."));
                true
            }
            Err(error) => {
                self.settlement_draft.error = Some(format!("{error:#}"));
                false
            }
        }
    }

    /// Candidate, origin line and verdict over the map.
    pub(super) fn draw_settlement_overlay(&self, painter: &egui::Painter, rect: Rect, ui: &egui::Ui) {
        let draft = &self.settlement_draft;
        let text: Option<(String, bool)> = if let Some(drag) = &draft.drag {
            let colour = if drag.verdict.is_ok() { VALID } else { INVALID };
            if let (Some(from), Some(candidate)) = (self.marker_screen(drag.origin, rect), drag.candidate) {
                if let Some(to) = self.marker_screen(candidate, rect) {
                    painter.line_segment([from, to], Stroke::new(1.5, colour.gamma_multiply(0.8)));
                    let outline = Stroke::new(1.5, colour);
                    match drag.kind {
                        PlaceKind::Town => {
                            painter.rect(Rect::from_center_size(to, Vec2::splat(10.0)), 0.0, colour.gamma_multiply(0.35), outline, egui::StrokeKind::Middle);
                        }
                        _ => {
                            painter.circle(to, 7.0, colour.gamma_multiply(0.35), outline);
                        }
                    }
                }
            }
            Some(match &drag.verdict {
                Ok(()) => (format!("{}: release to move · navigation clearance checked on export · Esc cancels", drag.label), false),
                Err(reason) => (format!("{}: {reason}", drag.label), true),
            })
        } else if let Some(error) = &draft.error {
            Some((error.clone(), true))
        } else if let Some(note) = &draft.note {
            Some((note.clone(), false))
        } else {
            Some(("Drag a capital (circle) or town (square) inside its region · Esc leaves · Right-drag pans".to_string(), false))
        };
        if let Some((text, bad)) = text {
            let galley = painter.layout(
                text,
                egui::TextStyle::Body.resolve(ui.style()),
                if bad { Color32::from_rgb(255, 150, 150) } else { Color32::WHITE },
                (rect.width() - 40.0).max(120.0),
            );
            let at = rect.left_top() + Vec2::new(12.0, 12.0);
            painter.rect_filled(Rect::from_min_size(at, galley.size()).expand(5.0), 4.0, Color32::from_black_alpha(190));
            painter.galley(at, galley, Color32::WHITE);
        }
    }

    /// Sidebar controls while the tool is active.
    pub(super) fn settlement_panel(&mut self, ui: &mut egui::Ui) {
        ui.strong("Move settlements");
        ui.small("Press a capital (circle) or town (square), drag inside its region and release. Green passes region/spacing checks; red names a problem and cancels the move. Export additionally checks navigation-node clearance against roads, coasts and borders. Esc cancels; right or middle drag pans.");
        ui.small("Ports are fixed. The export moves the settlement slots, pathfinding node, points of interest, trade roads and start position of the moved settlement together.");
        if self.project.atlas.is_some() {
            ui.small("Positions are stored in the original coordinates; the atlas scales them when exporting.");
        }
        let mut undo = false;
        let mut done = false;
        ui.horizontal(|ui| {
            undo = ui
                .add_enabled(!self.settlement_draft.history.is_empty() && !self.busy, egui::Button::new("Undo move"))
                .on_hover_text("Ctrl+Z")
                .clicked();
            done = ui.button("Done").clicked();
        });
        if let Some(error) = &self.settlement_draft.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        } else if let Some(note) = &self.settlement_draft.note {
            ui.label(note);
        }
        if !self.project.settlement_moves.is_empty() {
            ui.separator();
            ui.strong(format!("Moved settlements ({})", self.project.settlement_moves.len()));
        }
        let mut reset = None;
        for (index, movement) in self.project.settlement_moves.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "{} ({}) → {:.2}, {:.2}",
                    movement.slot.rsplit(':').next().unwrap_or(&movement.slot).replace('_', " "),
                    movement.region,
                    movement.position[0],
                    movement.position[1]
                ));
                if ui.add_enabled(!self.busy, egui::Button::new("Reset")).clicked() {
                    reset = Some(index);
                }
            });
        }
        if undo {
            self.undo_settlement_move();
        } else if let Some(index) = reset {
            self.reset_move(index);
        }
        if done {
            self.leave_settlement_tool();
        }
    }
}
