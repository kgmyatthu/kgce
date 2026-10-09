//! Region split tool of the Ownership map: pick a region, draw a cut between two of its border vertices, place the
//! new settlement and add the split to the project. The geometry work lives in `map::splits`.

use super::{Editor, contains, label};
use crate::{
    map::splits::{self, Snap, Trial},
    model::{RegionSplit, Rules},
    regionmap::regions::RegionData,
};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum MapTool {
    #[default]
    Select,
    Split,
    Activate,
}

/// The cut being drawn and the fields of the new region.
#[derive(Default)]
pub(super) struct SplitDraft {
    source: Option<String>,
    /// first border vertex of the cut
    start: Option<Snap>,
    /// cut points between the end points, in click order (from `start` towards `end`)
    clicks: Vec<[f32; 2]>,
    end: Option<Snap>,
    /// both halves of the finished cut
    trial: Option<Trial>,
    placing: bool,
    center: Option<[f32; 2]>,
    name: String,
    key: String,
    key_edited: bool,
    settlement_name: String,
    settlement: String,
    settlement_edited: bool,
    colour: [u8; 3],
    error: Option<String>,
}

const NEW_HALF: Color32 = Color32::from_rgba_premultiplied(35, 105, 128, 120);
const KEPT_HALF: Color32 = Color32::from_rgba_premultiplied(128, 70, 30, 60);

impl SplitDraft {
    fn cut(&self) -> Option<RegionSplit> {
        Some(splits::build_split(
            self.source.as_deref()?,
            self.start?,
            self.end?,
            &self.clicks,
        ))
    }

    /// Recomputes both halves of the finished cut; `trial` stays empty (and `error` is set) when the cut is invalid.
    fn refresh(&mut self, data: &RegionData) {
        self.trial = None;
        let Some(cut) = self.cut() else { return };
        let result = data.index(&cut.source).and_then(|ri| {
            splits::check_cut(
                &splits::ring_points(data, ri, cut.area),
                cut.a,
                cut.b,
                &splits::clicks_of(&cut),
            )?;
            splits::trial_split(data, &cut)
        });
        match result {
            Ok(trial) => {
                self.error = None;
                self.trial = Some(trial);
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    /// Exchanges the halves: the cut runs the other way, so the other side becomes the new region.
    fn swap(&mut self, data: &RegionData) {
        std::mem::swap(&mut self.start, &mut self.end);
        self.clicks.reverse();
        self.refresh(data);
        if self.trial.is_none() {
            std::mem::swap(&mut self.start, &mut self.end);
            self.clicks.reverse();
            self.refresh(data);
            return;
        }
        if let (Some(center), Some(trial)) = (self.center, &self.trial) {
            if !trial.new_half_contains(center) {
                self.center = None;
            }
        }
    }

    /// Takes back the most recent step.
    fn undo(&mut self) {
        self.error = None;
        if self.placing {
            self.placing = false;
        } else if self.center.is_some() {
            self.center = None;
        } else if self.end.is_some() {
            self.end = None;
            self.trial = None;
        } else if self.clicks.pop().is_some() {
        } else if self.start.is_some() {
            self.start = None;
        } else {
            self.source = None;
        }
    }

    fn hint(&self) -> &'static str {
        if self.source.is_none() {
            "1. Click the region to split."
        } else if self.start.is_none() {
            "2. Click a vertex of the region border to start the cut."
        } else if self.end.is_none() {
            "3. Click inside the region to bend the cut; click a border vertex to end it."
        } else if self.trial.as_ref().is_some_and(Trial::source_settlement_in_new_half) {
            "The source region's own settlement lies in the cyan half: press Swap halves so it stays with the source."
        } else if self.center.is_none() {
            "4. Check the halves (cyan = new region), then Place settlement and click inside the cyan half."
        } else {
            "5. Name the region and settlement, then Apply split."
        }
    }

    /// The project entry, or why the drawing cannot become one.
    fn finish(&self, rules: &Rules) -> Result<RegionSplit, String> {
        let mut split = self.cut().ok_or("Finish the cut on the region border first.")?;
        let trial = self.trial.as_ref().ok_or("The cut is not valid.")?;
        let center = self.center.ok_or("Place the settlement inside the new half.")?;
        if trial.source_settlement_in_new_half() {
            return Err("The source region's own settlement lies in the new half; use Swap halves.".into());
        }
        if !trial.new_half_contains(center) {
            return Err("The settlement lies outside the new half; place it again.".into());
        }
        split.key = self.key.trim().into();
        split.name = self.name.trim().into();
        split.settlement = self.settlement.trim().into();
        split.settlement_name = self.settlement_name.trim().into();
        split.center = center;
        split.colour = self.colour;
        splits::validate_fields(&split, rules).map_err(|error| format!("{error:#}"))?;
        Ok(split)
    }
}

fn default_key(source: &str, name: &str) -> String {
    let slug = splits::slug(name);
    if slug.is_empty() {
        String::new()
    } else {
        format!("{source}_{slug}")
    }
}

fn fill(painter: &egui::Painter, triangles: &[[[f32; 2]; 3]], to: &dyn Fn([f32; 2]) -> Pos2, color: Color32) {
    let mut mesh = egui::Mesh::default();
    mesh.vertices.reserve(triangles.len() * 3);
    for triangle in triangles {
        let base = mesh.vertices.len() as u32;
        for point in triangle {
            mesh.vertices.push(egui::epaint::Vertex {
                pos: to(*point),
                uv: egui::epaint::WHITE_UV,
                color,
            });
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    painter.add(egui::Shape::mesh(mesh));
}

impl Editor {
    /// A region clicked in the sidebar list: it becomes the region to split while the tool waits for one.
    pub(super) fn list_click(&mut self, key: String, multiple: bool) {
        if self.tool == MapTool::Split && self.draft.source.is_none() {
            self.draft.error = None;
            self.pick_source(key);
        } else {
            self.select(key, multiple);
        }
    }

    /// Key of the editable (campaign) region whose polygon contains `world`.
    pub(super) fn region_at(&self, world: [f32; 2]) -> Option<String> {
        self.polygon_at(world, false)
    }

    /// Key of the wilderness polygon (no campaign data yet, can be activated) that contains `world`.
    pub(super) fn wild_at(&self, world: [f32; 2]) -> Option<String> {
        self.polygon_at(world, true)
    }

    fn polygon_at(&self, world: [f32; 2], wild: bool) -> Option<String> {
        self.polygons
            .iter()
            .find(|polygon| {
                polygon.wild == wild
                    && world[0] >= polygon.bounds[0]
                    && world[0] <= polygon.bounds[2]
                    && world[1] >= polygon.bounds[1]
                    && world[1] <= polygon.bounds[3]
                    && polygon
                        .triangles
                        .iter()
                        .any(|triangle| contains(*triangle, world))
            })
            .map(|polygon| polygon.key.clone())
    }

    pub(super) fn to_screen(&self, world: [f32; 2], rect: Rect) -> Pos2 {
        let scale = self.scale(rect);
        Pos2::new(
            rect.center().x + (world[0] - self.center.x) * scale,
            rect.center().y - (world[1] - self.center.y) * scale,
        )
    }

    /// Snap distance in world units: 2% of the visible map width.
    fn snap_radius(&self, rect: Rect) -> f64 {
        0.02 * f64::from(rect.width()) / f64::from(self.scale(rect))
    }

    pub(super) fn leave_split_tool(&mut self) {
        self.tool = MapTool::Select;
        self.draft = SplitDraft::default();
    }

    /// "Split region" toggle above the region list.
    pub(super) fn split_toggle(&mut self, ui: &mut egui::Ui) {
        let active = self.tool == MapTool::Split;
        let enabled = active || (self.geometry.is_some() && !self.busy);
        let response = ui
            .add_enabled(
                enabled,
                egui::Button::new("Split region")
                    .selected(active)
                    .min_size(Vec2::new(ui.available_width(), 26.0)),
            )
            .on_hover_text("Cut a region in two along a line you draw on the map")
            .on_disabled_hover_text("Import the original game files first");
        if response.clicked() {
            if active {
                self.leave_split_tool();
            } else {
                self.leave_activate_tool();
                self.tool = MapTool::Split;
                self.draft = SplitDraft::default();
                self.push_message("Split tool: click the region to split.".into());
            }
        }
    }

    /// Escape leaves the tool, Backspace takes back the last step.
    pub(super) fn split_keys(&mut self, ctx: &egui::Context) {
        if self.tool != MapTool::Split {
            return;
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.leave_split_tool();
        } else if !ctx.wants_keyboard_input()
            && ctx.input(|input| input.key_pressed(egui::Key::Backspace))
        {
            self.draft.undo();
        }
    }

    fn pick_source(&mut self, key: String) {
        let Some(rules) = &self.rules else { return };
        let Some(geometry) = &self.geometry else {
            return;
        };
        let Ok(index) = geometry.data.index(&key) else {
            self.draft.error = Some(format!("{key} is not in the region geometry."));
            return;
        };
        if splits::cuttable_areas(&geometry.data, index).is_empty() {
            self.draft.error = Some(format!(
                "{key} cannot be split yet: every one of its areas has inner holes."
            ));
            return;
        }
        self.draft = SplitDraft {
            colour: splits::pick_colour(rules),
            source: Some(key.clone()),
            ..SplitDraft::default()
        };
        self.select(key, false);
    }

    /// A primary click on the map while the tool is active.
    pub(super) fn split_click(&mut self, world: [f32; 2], rect: Rect) {
        self.draft.error = None;
        let radius = self.snap_radius(rect);
        if self.draft.placing {
            let inside = self
                .draft
                .trial
                .as_ref()
                .is_some_and(|trial| trial.new_half_contains(world));
            if inside {
                self.draft.center = Some(world);
                self.draft.placing = false;
            } else {
                self.draft.error = Some("The settlement must lie inside the new (cyan) half.".into());
            }
            return;
        }
        let Some(source) = self.draft.source.clone() else {
            match self.region_at(world) {
                Some(key) => self.pick_source(key),
                None => match self.wild_at(world) {
                    Some(key) => {
                        self.draft.error = Some(format!(
                            "{} is wilderness without campaign data and cannot be split. Activate it first (Activate wilderness region).",
                            splits::default_name(&key)
                        ))
                    }
                    None => self.draft.error = Some("Click a region.".into()),
                },
            }
            return;
        };
        if self.draft.end.is_some() {
            self.draft.error =
                Some("The cut is finished: Place settlement, Swap halves or Undo point.".into());
            return;
        }
        let Some(geometry) = &self.geometry else {
            return;
        };
        let data = &geometry.data;
        let Ok(index) = data.index(&source) else {
            self.draft.error = Some(format!("{source} is not in the region geometry."));
            return;
        };
        let draft = &mut self.draft;
        let near = splits::nearest_vertex(data, index, draft.start.map(|start| start.area), world);
        let Some(start) = draft.start else {
            match near {
                Some((snap, distance)) if distance <= radius => draft.start = Some(snap),
                _ => {
                    draft.error = Some(
                        "Start the cut on the region border: click closer to one of its vertices (zoom in for precision)."
                            .into(),
                    )
                }
            }
            return;
        };
        match near {
            Some((snap, distance)) if distance <= radius && snap.pos != start.pos => {
                draft.end = Some(snap);
                draft.refresh(data);
                if draft.trial.is_none() {
                    draft.end = None;
                }
            }
            _ => {
                let ring = splits::ring_points(data, index, start.area);
                if splits::inside_ring(&ring, world) {
                    draft.clicks.push(world);
                } else {
                    draft.error = Some("Cut points must lie inside the region; end the cut on a border vertex.".into());
                }
            }
        }
    }

    /// Preview of the cut over the map.
    pub(super) fn draw_split_overlay(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        hover: Option<Pos2>,
        ui: &egui::Ui,
    ) {
        let Some(geometry) = &self.geometry else {
            return;
        };
        let draft = &self.draft;
        let to = |point: [f32; 2]| self.to_screen(point, rect);
        if let Some(trial) = &draft.trial {
            fill(painter, &trial.kept_triangles, &to, KEPT_HALF);
            fill(painter, &trial.new_triangles, &to, NEW_HALF);
            for (ring, color) in [
                (&trial.kept_ring, Color32::from_rgb(255, 160, 80)),
                (&trial.new_ring, Color32::from_rgb(90, 220, 255)),
            ] {
                painter.add(egui::Shape::closed_line(
                    ring.iter().map(|point| to(*point)).collect(),
                    Stroke::new(2.0, color),
                ));
            }
        }
        let mut line: Vec<Pos2> = Vec::new();
        if let Some(start) = draft.start {
            line.push(to(start.point));
        }
        line.extend(draft.clicks.iter().map(|point| to(*point)));
        if let Some(end) = draft.end {
            line.push(to(end.point));
        }
        if line.len() >= 2 {
            painter.add(egui::Shape::line(line.clone(), Stroke::new(2.5, Color32::WHITE)));
        }
        let world_hover = hover.map(|point| self.to_world(point, rect));
        if let (Some(from), Some(point), true) = (line.last(), hover, draft.end.is_none() && draft.start.is_some()) {
            painter.line_segment([*from, point], Stroke::new(1.0, Color32::from_white_alpha(120)));
        }
        for point in &draft.clicks {
            painter.circle(to(*point), 4.0, Color32::WHITE, Stroke::new(1.0, Color32::BLACK));
        }
        if let Some(start) = draft.start {
            painter.circle(to(start.point), 6.5, Color32::from_rgb(90, 220, 90), Stroke::new(1.5, Color32::BLACK));
        }
        if let Some(end) = draft.end {
            painter.circle(to(end.point), 6.5, Color32::from_rgb(255, 150, 40), Stroke::new(1.5, Color32::BLACK));
        }
        if let Some(center) = draft.center {
            let at = to(center);
            painter.circle(at, 8.0, Color32::from_rgb(250, 213, 112), Stroke::new(2.0, Color32::BLACK));
            painter.line_segment([at - Vec2::new(5.0, 0.0), at + Vec2::new(5.0, 0.0)], Stroke::new(1.5, Color32::BLACK));
            painter.line_segment([at - Vec2::new(0.0, 5.0), at + Vec2::new(0.0, 5.0)], Stroke::new(1.5, Color32::BLACK));
        }
        // the border vertex a click would snap to
        if let (Some(source), Some(world), true) = (&draft.source, world_hover, draft.end.is_none() && !draft.placing) {
            if let Ok(index) = geometry.data.index(source) {
                let near = splits::nearest_vertex(&geometry.data, index, draft.start.map(|start| start.area), world);
                if let Some((snap, distance)) = near {
                    if distance <= self.snap_radius(rect) {
                        painter.circle_stroke(to(snap.point), 9.0, Stroke::new(2.0, Color32::from_rgb(90, 220, 255)));
                    }
                }
            }
        }
        let text = draft.error.as_deref().unwrap_or_else(|| {
            if draft.placing {
                "Click inside the cyan half to place the settlement."
            } else {
                draft.hint()
            }
        });
        let galley = painter.layout_no_wrap(
            text.to_owned(),
            egui::TextStyle::Body.resolve(ui.style()),
            if draft.error.is_some() { Color32::from_rgb(255, 150, 150) } else { Color32::WHITE },
        );
        let at = rect.left_top() + Vec2::new(12.0, 12.0);
        painter.rect_filled(
            Rect::from_min_size(at, galley.size()).expand(5.0),
            4.0,
            Color32::from_black_alpha(190),
        );
        painter.galley(at, galley, Color32::WHITE);
    }

    /// Sidebar controls while the tool is active.
    pub(super) fn split_panel(&mut self, ui: &mut egui::Ui) {
        ui.strong("Cut a new region");
        let busy = self.busy;
        let data = self.geometry.as_ref().map(|geometry| &geometry.data);
        let draft = &mut self.draft;
        match &draft.source {
            Some(source) => {
                ui.horizontal(|ui| {
                    ui.label("Splitting");
                    ui.monospace(source);
                });
            }
            None => {
                ui.label("Click the region to split on the map.");
            }
        }
        ui.small(draft.hint());
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(draft.source.is_some(), egui::Button::new("Undo point"))
                .on_hover_text("Backspace")
                .clicked()
            {
                draft.undo();
            }
            if ui
                .add_enabled(draft.trial.is_some(), egui::Button::new("Swap halves"))
                .on_hover_text("Exchange which side of the cut becomes the new region")
                .clicked()
            {
                if let Some(data) = data {
                    draft.swap(data);
                }
            }
            if ui
                .add_enabled(draft.source.is_some(), egui::Button::new("Restart cut"))
                .clicked()
            {
                let colour = draft.colour;
                *draft = SplitDraft {
                    colour,
                    ..SplitDraft::default()
                };
            }
        });
        if draft.end.is_some() {
            ui.small("Cyan half = new region, orange half keeps the source region.");
        }
        ui.separator();
        ui.label("New region name");
        if ui.text_edit_singleline(&mut draft.name).changed() && !draft.key_edited {
            draft.key = default_key(draft.source.as_deref().unwrap_or(""), &draft.name);
        }
        ui.label("Region key");
        if ui.text_edit_singleline(&mut draft.key).changed() {
            draft.key_edited = !draft.key.is_empty();
        }
        ui.label("Settlement name");
        if ui.text_edit_singleline(&mut draft.settlement_name).changed() && !draft.settlement_edited {
            draft.settlement = splits::slug(&draft.settlement_name);
        }
        ui.label("Settlement key");
        if ui.text_edit_singleline(&mut draft.settlement).changed() {
            draft.settlement_edited = !draft.settlement.is_empty();
        }
        ui.horizontal(|ui| {
            ui.label("Map colour");
            ui.color_edit_button_srgb(&mut draft.colour);
            ui.monospace(format!(
                "#{:02x}{:02x}{:02x}",
                draft.colour[0], draft.colour[1], draft.colour[2]
            ));
        });
        ui.small("Unused by every other region; it identifies the region on the generated map.");
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(draft.trial.is_some(), egui::Button::new("Place settlement").selected(draft.placing))
                .clicked()
            {
                draft.placing = !draft.placing;
            }
            match draft.center {
                Some(center) => ui.monospace(format!("{:.2}, {:.2}", center[0], center[1])),
                None => ui.label("not placed"),
            };
        });
        let mut apply = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            apply = ui
                .add_enabled(
                    !busy
                        && draft.center.is_some()
                        && draft.trial.as_ref().is_some_and(|trial| !trial.source_settlement_in_new_half()),
                    egui::Button::new("Apply split").fill(Color32::from_rgb(61, 92, 73)),
                )
                .clicked();
            cancel = ui.button("Cancel").clicked();
        });
        if let Some(error) = &draft.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        }
        if cancel {
            self.leave_split_tool();
        } else if apply {
            self.apply_split();
        }
    }

    fn apply_split(&mut self) {
        let Some(rules) = &self.rules else { return };
        let split = match self.draft.finish(rules) {
            Ok(split) => split,
            Err(error) => {
                self.draft.error = Some(error);
                return;
            }
        };
        let mut next = self.project.clone();
        // the new region starts under the same owner its source region has now
        if let Some(owner) = self.project.owners.get(&split.source).cloned() {
            next.owners.insert(split.key.clone(), owner);
        }
        let (key, name, source) = (split.key.clone(), split.name.clone(), split.source.clone());
        next.splits.push(split);
        match self.install_project(next) {
            Ok(()) => {
                self.dirty = true;
                self.ui.analysis_wanted = self.source_ready;
                self.leave_split_tool();
                self.select(key, false);
                self.refresh_factions();
                self.push_message(format!(
                    "Split {name} off {}. Save or export to persist.",
                    label(&source)
                ));
            }
            Err(error) => self.draft.error = Some(format!("{error:#}")),
        }
    }

    /// The splits of the project with their remove buttons.
    pub(super) fn splits_list(&mut self, ui: &mut egui::Ui) {
        if self.project.splits.is_empty() {
            return;
        }
        ui.separator();
        ui.strong(format!("Region splits ({})", self.project.splits.len()));
        let mut remove = None;
        for (index, split) in self.project.splits.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("{} (from {})", split.name, label(&split.source)));
                if ui
                    .add_enabled(!self.busy, egui::Button::new("Remove split"))
                    .on_hover_text(format!("Remove {} and every split cut from it", split.key))
                    .clicked()
                {
                    remove = Some(index);
                }
            });
        }
        if remove.is_some() {
            self.confirm_remove_split = remove;
        }
    }

    /// Confirmation of a split removal (it also drops the later splits that cut the removed region).
    pub(super) fn remove_split_window(&mut self, ctx: &egui::Context) {
        let Some(index) = self.confirm_remove_split else {
            return;
        };
        if index >= self.project.splits.len() {
            self.confirm_remove_split = None;
            return;
        }
        let removed = splits::dependent_splits(&self.project.splits, index);
        let mut decision = None;
        egui::Window::new("Remove region split?")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("These regions are merged back into their source and their owner entries are dropped:");
                for &position in &removed {
                    let split = &self.project.splits[position];
                    ui.monospace(format!("{} ({})", split.key, split.name));
                }
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        decision = Some(false);
                    }
                    if ui.button("Remove split and dependents").clicked() {
                        decision = Some(true);
                    }
                });
            });
        match decision {
            Some(true) => {
                self.confirm_remove_split = None;
                self.remove_splits(&removed);
            }
            Some(false) => self.confirm_remove_split = None,
            None => {}
        }
    }

    fn remove_splits(&mut self, removed: &[usize]) {
        let mut next = self.project.clone();
        let keys: Vec<String> = removed
            .iter()
            .map(|&position| self.project.splits[position].key.clone())
            .collect();
        next.splits = self
            .project
            .splits
            .iter()
            .enumerate()
            .filter(|(position, _)| !removed.contains(position))
            .map(|(_, split)| split.clone())
            .collect();
        for key in &keys {
            next.owners.remove(key);
        }
        match self.install_project(next) {
            Ok(()) => {
                self.dirty = true;
                self.ui.analysis_wanted = self.source_ready;
                self.refresh_factions();
                self.push_message(format!("Removed split region(s): {}.", keys.join(", ")));
            }
            Err(error) => self.error = Some(format!("Remove split: {error:#}")),
        }
    }
}
