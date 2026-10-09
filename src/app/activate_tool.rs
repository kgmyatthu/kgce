//! Wilderness activation tool of the Ownership map: pick a hatched wilderness polygon (or an area of a backdrop
//! region to carve a new region from), choose which of its areas become passable and whether they merge, choose the
//! playable region that serves as template, name it, place its settlement and add the activation to the project.
//! The geometry work lives in `map::splits` and `regionmap::chain`.

use super::{Editor, label, split_tool::MapTool};
use crate::{
    map::splits,
    model::{AreaSelection, RegionActivation, Rules},
    regionmap::{chain, regions::RegionData},
    tsv_ui::workflow,
};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

/// Which areas of the picked region become passable.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum AreaChoice {
    /// only the area holding the settlement (it must be passable already)
    #[default]
    Centre,
    All,
    /// the areas ticked in the list / clicked on the map
    Picked,
}

/// The wilderness region being activated and the fields of the new campaign region.
#[derive(Default)]
pub(super) struct ActivationDraft {
    /// regions.esf key of the wilderness region (with `carve_from`: the key of the region to create)
    pub(super) key: Option<String>,
    /// acceptable donors: the first `adjacent` are neighbours of the region
    donors: Vec<String>,
    adjacent: usize,
    donor: String,
    name: String,
    settlement_name: String,
    settlement: String,
    settlement_edited: bool,
    colour: [u8; 3],
    placing: bool,
    center: Option<[f32; 2]>,
    error: Option<String>,
    /// the user is choosing a backdrop area to carve a new region from
    carve_mode: bool,
    /// `(source region key, area index)` of the carved area
    carve_from: Option<(String, usize)>,
    choice: AreaChoice,
    picked: Vec<usize>,
    merge: bool,
    /// map clicks toggle areas of the region
    picking_areas: bool,
    /// the geometry with the area options applied (`None` while they change nothing), and why it failed
    preview: Option<RegionData>,
    preview_error: Option<String>,
}

impl ActivationDraft {
    fn hint(&self) -> &'static str {
        if self.key.is_none() && self.carve_mode {
            "1. Click an area of a backdrop region (outlined) to carve a new region from it."
        } else if self.key.is_none() {
            "1. Click a hatched wilderness polygon."
        } else if self.center.is_none() && !self.placing {
            "2. Choose areas and template region, name region and settlement, then Place settlement."
        } else if self.center.is_none() {
            "3. Click inside the region's passable ground (outlined green) to place the settlement."
        } else {
            "4. Check the entries, then Apply activation."
        }
    }

    /// Takes back the most recent step.
    fn undo(&mut self) {
        self.error = None;
        if self.placing {
            self.placing = false;
        } else if self.center.is_some() {
            self.center = None;
        } else {
            let carve_mode = self.carve_mode && self.carve_from.is_none();
            *self = Self::default();
            self.carve_mode = carve_mode;
        }
    }

    /// The area options as a project entry (only the fields the geometry needs are meaningful).
    fn options(&self) -> Option<RegionActivation> {
        let key = self.key.clone()?;
        let areas = match (&self.carve_from, self.choice) {
            (Some((_, area)), _) => AreaSelection::Indices(vec![*area]),
            (None, AreaChoice::Centre) => AreaSelection::Centre,
            (None, AreaChoice::All) => AreaSelection::All,
            (None, AreaChoice::Picked) => AreaSelection::Indices(self.picked.clone()),
        };
        Some(RegionActivation {
            key,
            areas,
            merge: self.merge && self.carve_from.is_none(),
            carve_from: self.carve_from.as_ref().map(|(source, _)| source.clone()),
            ..RegionActivation::default()
        })
    }

    /// Recomputes the geometry preview for the current area options on `data` (the project's geometry).
    fn refresh(&mut self, data: &RegionData) {
        self.preview = None;
        self.preview_error = None;
        let Some(options) = self.options() else { return };
        if let Err(error) = options.validate_options() {
            self.preview_error = Some(format!("{error:#}"));
            return;
        }
        if !chain::changes_geometry(&options) {
            return;
        }
        let mut copy = data.clone();
        match chain::prepare_activation(&mut copy, &options) {
            Ok(_) => self.preview = Some(copy),
            Err(error) => self.preview_error = Some(format!("{error:#}")),
        }
    }

    /// The project entry, or why the form cannot become one.
    fn finish(
        &self,
        rules: &Rules,
        data: &RegionData,
        theatres: &[crate::map::TheatreRect],
        project_splits: &[crate::model::RegionSplit],
    ) -> Result<RegionActivation, String> {
        self.key.clone().ok_or("Click a wilderness polygon first.")?;
        let center = self.center.ok_or("Place the settlement inside the region.")?;
        if let Some(error) = &self.preview_error {
            return Err(error.clone());
        }
        let mut activation = self.options().ok_or("Click a wilderness polygon first.")?;
        activation.donor = self.donor.clone();
        activation.name = self.name.trim().into();
        activation.settlement = self.settlement.trim().into();
        activation.settlement_name = self.settlement_name.trim().into();
        activation.center = center;
        activation.colour = self.colour;
        splits::validate_activation(&activation, rules, data, theatres, project_splits)
            .map_err(|error| format!("{error:#}"))?;
        Ok(activation)
    }
}

impl Editor {
    /// "Activate wilderness region" toggle above the region list.
    pub(super) fn activate_toggle(&mut self, ui: &mut egui::Ui) {
        let active = self.tool == MapTool::Activate;
        let enabled = active || (self.geometry.is_some() && !self.busy);
        let response = ui
            .add_enabled(
                enabled,
                egui::Button::new("Activate wilderness region")
                    .selected(active)
                    .min_size(Vec2::new(ui.available_width(), 26.0)),
            )
            .on_hover_text("Turn a hatched wilderness polygon of a playable theatre into a campaign region")
            .on_disabled_hover_text("Import the original game files first");
        if response.clicked() {
            if active {
                self.leave_activate_tool();
            } else {
                self.leave_split_tool();
                self.tool = MapTool::Activate;
                self.activation = ActivationDraft::default();
                self.mesh_dirty = true;
                self.push_message("Activate tool: click a hatched wilderness polygon.".into());
            }
        }
    }

    pub(super) fn leave_activate_tool(&mut self) {
        if self.tool == MapTool::Activate {
            self.tool = MapTool::Select;
        }
        self.activation = ActivationDraft::default();
        self.mesh_dirty = true;
    }

    /// Escape leaves the tool, Backspace takes back the last step.
    pub(super) fn activate_keys(&mut self, ctx: &egui::Context) {
        if self.tool != MapTool::Activate {
            return;
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.leave_activate_tool();
        } else if !ctx.wants_keyboard_input() && ctx.input(|input| input.key_pressed(egui::Key::Backspace)) {
            self.activation.undo();
            self.refresh_preview();
            self.mesh_dirty = true;
        }
    }

    /// Recomputes the draft's geometry preview from the project's current geometry.
    fn refresh_preview(&mut self) {
        if let Some(geometry) = &self.geometry {
            self.activation.refresh(&geometry.data);
        }
    }

    /// The geometry the draft works on: the preview of its area options, else the project's.
    fn draft_data(&self) -> Option<&RegionData> {
        self.activation.preview.as_ref().or(self.geometry.as_ref().map(|g| &g.data))
    }

    /// Donors and defaults for the region `key` (which must exist in `data`).
    fn start_draft(&mut self, key: String, data: &RegionData, carve_from: Option<(String, usize)>) {
        let Some(rules) = &self.rules else { return };
        let found = splits::activatable(data, &self.theatres, rules, &key);
        let found = match found {
            Ok(found) => found,
            Err(error) => {
                self.activation.error = Some(format!("{error:#}"));
                return;
            }
        };
        let donors = splits::donor_choices(data, rules, &self.project.splits, &key, &found.theatre);
        let adjacent = splits::donor_candidates(data, rules, &self.project.splits, &key, &found.theatre).len();
        let Some(donor) = donors.first().cloned() else {
            self.activation.error = Some(format!("{} has no playable region of its theatre to copy.", label(&key)));
            return;
        };
        let name = splits::default_name(&key);
        let carve_mode = self.activation.carve_mode;
        self.activation = ActivationDraft {
            donors,
            adjacent,
            donor,
            settlement: splits::slug(&name),
            settlement_name: name.clone(),
            name,
            colour: splits::pick_colour(rules),
            key: Some(key),
            carve_mode,
            carve_from,
            // a region that has no passable area yet needs its areas chosen: start with all of them
            choice: if found.passable_area.is_none() { AreaChoice::All } else { AreaChoice::Centre },
            ..ActivationDraft::default()
        };
        self.mesh_dirty = true;
    }

    fn pick_wild(&mut self, key: String) {
        let Some(geometry) = &self.geometry else { return };
        let data = geometry.data.clone();
        self.start_draft(key, &data, None);
        self.refresh_preview();
    }

    /// Carve mode: the backdrop area under `world` becomes a new region.
    fn pick_carve_source(&mut self, world: [f32; 2]) {
        let Some(rules) = &self.rules else { return };
        let Some(geometry) = &self.geometry else { return };
        let data = &geometry.data;
        let Some(code) = data.locate([f64::from(world[0]), f64::from(world[1])]) else {
            self.activation.error = Some("Click an area of a backdrop region.".into());
            return;
        };
        let (ri, area) = ((code >> 16) as usize, (code & 0xFFFF) as usize);
        let region = &data.regions[ri];
        if region.kind != "land" || region.settlement.is_some() || rules.regions.get(&region.key).is_some() {
            self.activation.error = Some(format!("{} is not a backdrop region.", label(&region.key)));
            return;
        }
        if region.areas[area].passable {
            self.activation.error = Some("That area is passable land already; activate its region instead.".into());
            return;
        }
        let source = region.key.clone();
        let key = format!("{}_{area}", splits::slug(&source));
        let mut copy = data.clone();
        let options = RegionActivation {
            key: key.clone(),
            areas: AreaSelection::Indices(vec![area]),
            carve_from: Some(source.clone()),
            ..RegionActivation::default()
        };
        if let Err(error) = chain::prepare_activation(&mut copy, &options) {
            self.activation.error = Some(format!("{error:#}"));
            return;
        }
        self.start_draft(key, &copy, Some((source, area)));
        self.refresh_preview();
    }

    /// A primary click on the map while the tool is active.
    pub(super) fn activate_click(&mut self, world: [f32; 2]) {
        self.activation.error = None;
        if self.activation.placing {
            let found = self.draft_data().zip(self.activation.key.as_deref()).map(|(data, key)| {
                let ri = data.index(key).ok()?;
                splits::passable_area_at(data, ri, world)?;
                Some(splits::check_settlement_clearance(data, ri, world))
            });
            match found.flatten() {
                Some(Ok(())) => {
                    self.activation.center = Some(world);
                    self.activation.placing = false;
                }
                Some(Err(error)) => self.activation.error = Some(format!("{error:#}")),
                None => {
                    self.activation.error = Some("The settlement must lie inside a passable area of the region.".into());
                }
            }
            return;
        }
        if self.activation.picking_areas {
            // areas are chosen on the project's geometry (indices before any merge), never on the merged preview
            let hit = self.geometry.as_ref().map(|g| &g.data).zip(self.activation.key.as_deref()).and_then(|(data, key)| {
                let ri = data.index(key).ok()?;
                data.area_at(ri, [f64::from(world[0]), f64::from(world[1])])
            });
            match hit {
                Some(area) => self.toggle_area(area),
                None => self.activation.error = Some("Click inside one of the region's areas.".into()),
            }
            return;
        }
        if self.activation.carve_mode && self.activation.key.is_none() {
            self.pick_carve_source(world);
            return;
        }
        match self.wild_at(world) {
            Some(key) if self.activation.key.as_deref() != Some(key.as_str()) => self.pick_wild(key),
            Some(_) => {}
            None => self.activation.error = Some("Click a hatched wilderness polygon.".into()),
        }
    }

    fn toggle_area(&mut self, area: usize) {
        let draft = &mut self.activation;
        draft.choice = AreaChoice::Picked;
        match draft.picked.iter().position(|&a| a == area) {
            Some(at) => {
                draft.picked.remove(at);
            }
            None => draft.picked.push(area),
        }
        draft.picked.sort_unstable();
        draft.center = None;
        self.refresh_preview();
        self.mesh_dirty = true;
    }

    /// Labels of the wilderness polygons large enough to hold one.
    pub(super) fn draw_wild_labels(&self, painter: &egui::Painter, rect: Rect) {
        let scale = self.scale(rect);
        let font = FontId::proportional(11.0);
        let to = |point: [f32; 2]| {
            Pos2::new(
                rect.center().x + (point[0] - self.center.x) * scale,
                rect.center().y - (point[1] - self.center.y) * scale,
            )
        };
        for polygon in self.polygons.iter().filter(|polygon| polygon.wild) {
            if (polygon.bounds[2] - polygon.bounds[0]) * scale < 36.0 {
                continue;
            }
            let at = to(polygon.label_at);
            if !rect.contains(at) {
                continue;
            }
            let text = format!("{}\n(wilderness)", splits::default_name(&polygon.key));
            painter.text(
                at + Vec2::new(1.0, 1.0),
                egui::Align2::CENTER_CENTER,
                &text,
                font.clone(),
                Color32::BLACK,
            );
            painter.text(
                at,
                egui::Align2::CENTER_CENTER,
                &text,
                font.clone(),
                Color32::from_rgb(215, 222, 215),
            );
        }
    }

    /// Settlement marker, placement guide and hint over the map.
    pub(super) fn draw_activate_overlay(&self, painter: &egui::Painter, rect: Rect, ui: &egui::Ui) {
        let Some(rules) = &self.rules else { return };
        let draft = &self.activation;
        let scale = self.scale(rect);
        let to = |point: [f32; 2]| {
            Pos2::new(
                rect.center().x + (point[0] - self.center.x) * scale,
                rect.center().y - (point[1] - self.center.y) * scale,
            )
        };
        let outline = |data: &RegionData, ri: usize, area: usize, stroke: Stroke| {
            for outline in &data.regions[ri].areas[area].outlines {
                painter.add(egui::Shape::closed_line(
                    outline.ring.iter().map(|&id| to(data.vertices[id as usize])).collect(),
                    stroke,
                ));
            }
        };
        let faint = Stroke::new(1.2, Color32::from_rgba_unmultiplied(230, 230, 230, 110));
        if let (Some(geometry), None, true) = (&self.geometry, &draft.key, draft.carve_mode) {
            // candidates of a carve: every impassable area of the backdrop regions
            for (ri, region) in geometry.data.regions.iter().enumerate() {
                if region.kind != "land" || region.settlement.is_some() || rules.regions.get(&region.key).is_some() {
                    continue;
                }
                for (ai, area) in region.areas.iter().enumerate() {
                    if !area.passable {
                        outline(&geometry.data, ri, ai, faint);
                    }
                }
            }
        }
        if let (Some(data), Some(key)) = (self.draft_data(), &draft.key) {
            let green = Stroke::new(2.0, Color32::from_rgb(90, 230, 120));
            let result = data.index(key).ok();
            if draft.placing || draft.carve_from.is_some() {
                // the geometry the settlement goes into: passable areas of the region as it will be built
                if let Some(ri) = result {
                    for (ai, area) in data.regions[ri].areas.iter().enumerate() {
                        if area.passable {
                            outline(data, ri, ai, green);
                        }
                    }
                }
            } else {
                // area selection works on the region as the project has it (indices before any merge)
                let source = self.geometry.as_ref().map(|g| &g.data);
                if let Some((src, ri)) = source.and_then(|d| Some((d, d.index(key).ok()?))) {
                    let font = FontId::proportional(12.0);
                    for (ai, area) in src.regions[ri].areas.iter().enumerate() {
                        let chosen = match draft.choice {
                            AreaChoice::Centre => false,
                            AreaChoice::All => true,
                            AreaChoice::Picked => draft.picked.contains(&ai),
                        };
                        let stroke = if chosen {
                            Stroke::new(3.0, Color32::from_rgb(250, 213, 112))
                        } else if area.passable {
                            green
                        } else {
                            faint
                        };
                        outline(src, ri, ai, stroke);
                        if draft.picking_areas || draft.choice == AreaChoice::Picked {
                            if let Some(first) = area.outlines.iter().find(|o| o.outer).and_then(|o| o.ring.first()) {
                                painter.text(
                                    to(src.vertices[*first as usize]) + Vec2::new(4.0, -4.0),
                                    egui::Align2::LEFT_BOTTOM,
                                    ai.to_string(),
                                    font.clone(),
                                    Color32::WHITE,
                                );
                            }
                        }
                    }
                }
                // the merged / newly passable result, separate from the selection
                if let (Some(_), Some(ri)) = (&draft.preview, result) {
                    for (ai, area) in data.regions[ri].areas.iter().enumerate() {
                        if area.passable {
                            outline(data, ri, ai, Stroke::new(1.5, Color32::from_rgb(90, 200, 250)));
                        }
                    }
                }
            }
        }
        if let Some(center) = draft.center {
            let at = to(center);
            painter.circle(at, 8.0, Color32::from_rgb(250, 213, 112), Stroke::new(2.0, Color32::BLACK));
            painter.line_segment([at - Vec2::new(5.0, 0.0), at + Vec2::new(5.0, 0.0)], Stroke::new(1.5, Color32::BLACK));
            painter.line_segment([at - Vec2::new(0.0, 5.0), at + Vec2::new(0.0, 5.0)], Stroke::new(1.5, Color32::BLACK));
        }
        let text = draft
            .error
            .as_deref()
            .or(draft.preview_error.as_deref())
            .unwrap_or_else(|| draft.hint());
        let failed = draft.error.is_some() || draft.preview_error.is_some();
        let galley = painter.layout_no_wrap(
            text.to_owned(),
            egui::TextStyle::Body.resolve(ui.style()),
            if failed { Color32::from_rgb(255, 150, 150) } else { Color32::WHITE },
        );
        let at = rect.left_top() + Vec2::new(12.0, 12.0);
        painter.rect_filled(Rect::from_min_size(at, galley.size()).expand(5.0), 4.0, Color32::from_black_alpha(190));
        painter.galley(at, galley, Color32::WHITE);
    }

    /// Sidebar controls while the tool is active.
    pub(super) fn activate_panel(&mut self, ui: &mut egui::Ui) {
        let Some(rules) = &self.rules else { return };
        ui.strong("Activate a wilderness region");
        let busy = self.busy;
        if self.activation.key.is_none() {
            ui.label(if self.activation.carve_mode {
                "Click an outlined area of a backdrop region on the map."
            } else {
                "Click a hatched wilderness polygon on the map."
            });
            ui.small(self.activation.hint());
            let mut carve = self.activation.carve_mode;
            if ui
                .checkbox(&mut carve, "Carve into new region")
                .on_hover_text("Make a NEW region out of one area of a backdrop region (e.g. an island of `unexplorable`)")
                .changed()
            {
                self.activation.carve_mode = carve;
                self.mesh_dirty = true;
            }
            if let Some(error) = &self.activation.error {
                ui.colored_label(Color32::LIGHT_RED, error);
            }
            if ui.button("Cancel").clicked() {
                self.leave_activate_tool();
            }
            return;
        }
        let mut refresh = false;
        let data_for_areas: Option<Vec<(usize, bool, usize)>> = self.geometry.as_ref().map(|g| &g.data).zip(self.activation.key.as_deref()).and_then(
            |(data, key)| {
                let ri = data.index(key).ok()?;
                Some(
                    data.regions[ri]
                        .areas
                        .iter()
                        .enumerate()
                        .map(|(i, a)| (i, a.passable, a.outlines.iter().map(|o| o.ring.len()).sum()))
                        .collect(),
                )
            },
        );
        let draft = &mut self.activation;
        let key = draft.key.clone().unwrap_or_default();
        match &draft.carve_from {
            Some((source, area)) => {
                ui.horizontal(|ui| {
                    ui.label("Carving");
                    ui.monospace(format!("{source}:{area}"));
                });
                ui.label("Key of the new region");
                if ui.text_edit_singleline(draft.key.get_or_insert_with(String::new)).changed() {
                    refresh = true;
                }
            }
            None => {
                ui.horizontal(|ui| {
                    ui.label("Activating");
                    ui.monospace(&key);
                });
            }
        }
        ui.small(draft.hint());
        ui.horizontal_wrapped(|ui| {
            if ui.button("Undo step").on_hover_text("Backspace").clicked() {
                draft.undo();
                refresh = true;
            }
        });
        if draft.key.is_none() {
            self.mesh_dirty = true;
            return;
        }
        if draft.carve_from.is_none() {
            ui.separator();
            ui.label("Areas that become passable");
            let before = (draft.choice, draft.picked.clone(), draft.merge);
            ui.radio_value(&mut draft.choice, AreaChoice::Centre, "Only the area with the settlement");
            ui.radio_value(&mut draft.choice, AreaChoice::All, "All areas");
            ui.horizontal(|ui| {
                ui.radio_value(&mut draft.choice, AreaChoice::Picked, "Chosen areas");
                if ui
                    .add(egui::Button::new("Click areas on the map").selected(draft.picking_areas))
                    .clicked()
                {
                    draft.picking_areas = !draft.picking_areas;
                    draft.choice = AreaChoice::Picked;
                }
            });
            if draft.choice == AreaChoice::Picked {
                if let Some(areas) = &data_for_areas {
                    egui::ScrollArea::vertical().id_salt("activation-areas").max_height(150.0).show(ui, |ui| {
                        for &(index, passable, vertices) in areas {
                            let mut on = draft.picked.contains(&index);
                            let text = format!(
                                "Area {index} · {vertices} vertices{}",
                                if passable { " · passable" } else { "" }
                            );
                            if ui.checkbox(&mut on, text).changed() {
                                match (on, draft.picked.iter().position(|&a| a == index)) {
                                    (true, None) => draft.picked.push(index),
                                    (false, Some(at)) => {
                                        draft.picked.remove(at);
                                    }
                                    _ => {}
                                }
                                draft.picked.sort_unstable();
                            }
                        }
                    });
                }
            }
            ui.add_enabled(
                draft.choice != AreaChoice::Centre,
                egui::Checkbox::new(&mut draft.merge, "Merge the selected areas into one"),
            )
            .on_hover_text("Areas that touch are united into a single polygon; areas that touch nothing stay as they are");
            if draft.choice == AreaChoice::Centre {
                draft.merge = false;
            }
            if before != (draft.choice, draft.picked.clone(), draft.merge) {
                draft.center = None;
                refresh = true;
            }
            if let Some(error) = &draft.preview_error {
                ui.colored_label(Color32::LIGHT_RED, error);
            }
        }
        ui.separator();
        ui.label("Template region (owner, culture, religion, settlement slots)");
        let donor_text = |key: &str, adjacent: bool| {
            format!(
                "{} · {}{}",
                label(key),
                label(workflow::effective_owner(&self.project, rules, key)),
                if adjacent { " (next to it)" } else { "" }
            )
        };
        egui::ComboBox::from_id_salt("activation-donor")
            .width(ui.available_width() - 8.0)
            .selected_text(donor_text(&draft.donor, false))
            .show_ui(ui, |ui| {
                for (index, donor) in draft.donors.iter().enumerate() {
                    ui.selectable_value(&mut draft.donor, donor.clone(), donor_text(donor, index < draft.adjacent));
                }
            });
        ui.label("Region name");
        ui.text_edit_singleline(&mut draft.name);
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
            if ui.add(egui::Button::new("Place settlement").selected(draft.placing)).clicked() {
                draft.placing = !draft.placing;
                draft.picking_areas = false;
            }
            match draft.center {
                Some(center) => ui.monospace(format!("{:.2}, {:.2}", center[0], center[1])),
                None => ui.label("not placed"),
            };
        });
        ui.small("The settlement keeps at least 2.1 units from the outline of its area.");
        let (mut apply, mut cancel) = (false, false);
        ui.horizontal(|ui| {
            apply = ui
                .add_enabled(
                    !busy && draft.center.is_some() && draft.preview_error.is_none(),
                    egui::Button::new("Apply activation").fill(Color32::from_rgb(61, 92, 73)),
                )
                .clicked();
            cancel = ui.button("Cancel").clicked();
        });
        if let Some(error) = &draft.error {
            ui.colored_label(Color32::LIGHT_RED, error);
        }
        if refresh {
            self.mesh_dirty = true;
        }
        if cancel {
            self.leave_activate_tool();
        } else if apply {
            self.apply_activation();
        } else if refresh {
            self.refresh_preview();
        }
    }

    fn apply_activation(&mut self) {
        let Some(rules) = &self.rules else { return };
        let Some(geometry) = &self.geometry else {
            return;
        };
        let activation = match self.activation.finish(rules, &geometry.data, &self.theatres, &self.project.splits) {
            Ok(activation) => activation,
            Err(error) => {
                self.activation.error = Some(error);
                return;
            }
        };
        let mut next = self.project.clone();
        let (key, name, donor) = (activation.key.clone(), activation.name.clone(), activation.donor.clone());
        // the region starts under the owner its template region has now
        if let Some(owner) = self.project.owners.get(&donor).cloned() {
            next.owners.insert(key.clone(), owner);
        }
        next.activations.push(activation);
        match next.validate_structure().and_then(|()| self.install_project(next)) {
            Ok(()) => {
                self.dirty = true;
                self.ui.analysis_wanted = self.source_ready;
                self.leave_activate_tool();
                self.select(key, false);
                self.refresh_factions();
                self.push_message(format!(
                    "Activated {name} with {} as template. Save or export to persist.",
                    label(&donor)
                ));
            }
            Err(error) => self.activation.error = Some(format!("{error:#}")),
        }
    }

    /// The activations of the project with their remove buttons.
    pub(super) fn activations_list(&mut self, ui: &mut egui::Ui) {
        if self.project.activations.is_empty() {
            return;
        }
        ui.separator();
        ui.strong(format!("Activated wilderness ({})", self.project.activations.len()));
        let mut remove = None;
        for (index, activation) in self.project.activations.iter().enumerate() {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("{} (template {})", activation.name, label(&activation.donor)));
                if ui
                    .add_enabled(!self.busy, egui::Button::new("Remove activation"))
                    .on_hover_text(format!("Return {} to wilderness", activation.key))
                    .clicked()
                {
                    remove = Some(index);
                }
            });
        }
        if remove.is_some() {
            self.confirm_remove_activation = remove;
        }
    }

    /// Confirmation of an activation removal (it also drops the splits cut from the region and its owner entry).
    pub(super) fn remove_activation_window(&mut self, ctx: &egui::Context) {
        let Some(index) = self.confirm_remove_activation else {
            return;
        };
        let Some(removal) = self
            .project
            .activations
            .get(index)
            .and_then(|activation| {
                splits::removal_of_activation(&self.project.activations, &self.project.splits, &activation.key)
            })
        else {
            self.confirm_remove_activation = None;
            return;
        };
        let mut decision = None;
        egui::Window::new("Remove wilderness activation?")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("These regions return to wilderness (or disappear) and their owner entries are dropped:");
                for key in &removal.keys {
                    ui.monospace(key);
                }
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        decision = Some(false);
                    }
                    if ui.button("Remove activation and dependents").clicked() {
                        decision = Some(true);
                    }
                });
            });
        match decision {
            Some(true) => {
                self.confirm_remove_activation = None;
                self.remove_activation(&removal);
            }
            Some(false) => self.confirm_remove_activation = None,
            None => {}
        }
    }

    fn remove_activation(&mut self, removal: &splits::ActivationRemoval) {
        let mut next = self.project.clone();
        removal.remove_from(&mut next);
        match self.install_project(next) {
            Ok(()) => {
                self.dirty = true;
                self.ui.analysis_wanted = self.source_ready;
                self.refresh_factions();
                self.push_message(format!("Removed activation of {}.", removal.keys.join(", ")));
            }
            Err(error) => self.error = Some(format!("Remove activation: {error:#}")),
        }
    }
}
