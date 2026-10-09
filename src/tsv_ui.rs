//! Native panels for the TSV workflow: import RPFM TSV tables -> mappings -> compatibility review ->
//! ownership map -> export review.
//!
//! The panels never touch the filesystem or run workers themselves: they edit the [`Project`] in memory
//! and return [`Request`]s which `app.rs` executes on background threads. Results are fed back through
//! the `set_*` methods.

pub mod workflow;

use crate::{
    compatibility::{Analysis, Issue},
    model::{Project, TargetMode},
    target::{
        AdapterBinding, Dataset, ImportSpec, Listing, Preview, Requirement, Summary, TargetBinding,
    },
};
use eframe::egui::{self, Color32, RichText};
use std::{collections::BTreeMap, path::PathBuf};
use workflow::{MAPPING_FIELDS, mapping_title};

/// Shared desktop styling, including standalone panel verification windows.
pub fn configure_visuals(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.all_styles_mut(|style| {
        for (text_style, size) in [
            (egui::TextStyle::Heading, 22.0),
            (egui::TextStyle::Body, 16.0),
            (egui::TextStyle::Button, 16.0),
            (egui::TextStyle::Small, 13.0),
        ] {
            style
                .text_styles
                .insert(text_style, egui::FontId::proportional(size));
        }
        style
            .text_styles
            .insert(egui::TextStyle::Monospace, egui::FontId::monospace(15.0));
    });
}

const AMBER: Color32 = Color32::from_rgb(250, 213, 112);
const RED: Color32 = Color32::from_rgb(240, 120, 120);
const GREEN: Color32 = Color32::from_rgb(130, 210, 150);
const CATEGORIES: [(&str, &str); 14] = [
    ("target", "TSV dataset"),
    ("resources", "Resource review"),
    ("language", "Language"),
    ("loc", "Localisation"),
    ("schema", "Database schema"),
    ("reference", "References"),
    ("db_key", "Database key collisions"),
    ("mapping", "Faction / unit / building mapping"),
    ("ui", "UI resources"),
    ("script", "Campaign scripts"),
    ("template", "Templates"),
    ("adapter", "Resource adapter"),
    ("port", "Port"),
    ("ownership", "Ownership"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Tables,
    Mappings,
    Review,
    Map,
    Export,
}

impl Tab {
    pub fn title(self) -> &'static str {
        match self {
            Tab::Tables => "1 · Import TSV",
            Tab::Mappings => "2 · Mappings",
            Tab::Review => "3 · Compatibility",
            Tab::Map => "4 · Ownership map",
            Tab::Export => "5 · Export review",
        }
    }
}

/// Selection the ownership map is asked to make when another panel sends the user there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OwnerFocus {
    /// One region whose owner override is invalid.
    Region(String),
    /// Every region still awaiting an owner.
    Pending,
}

/// Work the panels ask the application to run on a background thread.
#[derive(Debug)]
pub enum Request {
    LoadRequirements,
    /// Re-list datasets and (re)load details of the project's pinned dataset.
    RefreshDatasets(Option<TargetBinding>),
    Preview(ImportSpec),
    Import(ImportSpec),
    Check(TargetBinding),
    ImportAdapter(TargetBinding, PathBuf),
    Analyze,
    Export,
}

/// Manifest facts about one stored dataset that the panels display.
#[derive(Clone, Debug)]
pub struct DatasetInfo {
    pub binding: TargetBinding,
    pub name: String,
    pub language: String,
    pub fingerprint: String,
    pub resources_reviewed: bool,
}

impl DatasetInfo {
    pub fn of(dataset: &Dataset) -> Self {
        Self {
            binding: dataset.binding(),
            name: dataset.manifest.name.clone(),
            language: dataset.manifest.language.clone(),
            fingerprint: dataset.manifest.fingerprint.clone(),
            resources_reviewed: dataset.manifest.resources_reviewed,
        }
    }
}

/// Shared mutable state handed to a panel for one frame.
pub struct PanelCtx<'a> {
    pub project: &'a mut Project,
    pub dirty: &'a mut bool,
    /// An exclusive worker (import/export/dataset/adapter) is running.
    pub busy: bool,
    pub source_ready: bool,
}

#[derive(Default)]
pub struct TsvUi {
    pub tab: Option<Tab>,
    name: String,
    language: String,
    folder_input: String,
    resources_reviewed: bool,
    import_extras: bool,
    adapter_folder: String,
    notice: Option<String>,
    /// Last rejected mapping or resolution edit.
    edit_error: Option<String>,
    requirements: Vec<Requirement>,
    requirements_loaded: bool,
    datasets: Vec<Summary>,
    dataset_problems: Vec<String>,
    datasets_loaded: bool,
    dataset_filter: String,
    info: Option<Result<DatasetInfo, String>>,
    info_requested: Option<TargetBinding>,
    checked: Option<(String, Vec<String>)>,
    preview: Option<(ImportSpec, Preview)>,
    pub analysis: Option<Analysis>,
    pub analyzed_signature: Option<String>,
    pub analysis_running: bool,
    /// Set when a project edit invalidated the analysis; `app.rs` restarts it once idle.
    pub analysis_wanted: bool,
    /// What the review/export panels ask the ownership map to select (consumed by `app.rs` once the map is shown).
    pub focus_owner: Option<OwnerFocus>,
    mapping_field: usize,
    mapping_filter: String,
    new_from: String,
    new_to: String,
    issue_choice: BTreeMap<String, String>,
    change_filter: String,
    show_info: bool,
}

impl TsvUi {
    pub fn tab(&self) -> Tab {
        self.tab.unwrap_or(Tab::Map)
    }

    /// The next background read the panels need, one at a time (each one holds the busy flag).
    pub fn pending(
        &mut self,
        project: &Project,
        busy: bool,
        source_ready: bool,
    ) -> Option<Request> {
        if busy {
            return None;
        }
        if source_ready && !self.requirements_loaded {
            self.requirements_loaded = true;
            return Some(Request::LoadRequirements);
        }
        if !self.datasets_loaded || self.info_requested != project.target {
            self.datasets_loaded = true;
            self.info_requested = project.target.clone();
            return Some(Request::RefreshDatasets(project.target.clone()));
        }
        None
    }

    /// The original snapshot changed, so the derived table requirements must be recomputed.
    pub fn sources_changed(&mut self) {
        self.requirements_loaded = false;
    }

    pub fn set_requirements(&mut self, requirements: Vec<Requirement>) {
        self.requirements = requirements;
    }

    pub fn set_datasets(&mut self, listing: Listing, info: Option<Result<DatasetInfo, String>>) {
        self.datasets = listing.datasets;
        self.dataset_problems = listing.problems;
        self.info = info;
    }

    pub fn set_preview(&mut self, spec: ImportSpec, preview: Preview) {
        self.notice = None;
        self.preview = Some((spec, preview));
    }

    pub fn set_checked(&mut self, info: DatasetInfo, changes: Vec<String>) {
        self.checked = Some((info.binding.id.clone(), changes));
        if self.info_requested.as_ref() == Some(&info.binding) {
            self.info = Some(Ok(info));
        }
    }

    /// A new immutable dataset was stored: refresh the list.
    pub fn dataset_imported(&mut self) {
        self.notice = None;
        self.datasets_loaded = false;
    }

    pub fn adapter_imported(&mut self, ctx: &mut PanelCtx<'_>, adapter: AdapterBinding) {
        ctx.project.adapter = Some(adapter);
        self.project_changed(ctx);
    }

    pub fn set_analysis(&mut self, analysis: Analysis, signature: String) {
        self.analysis = Some(analysis);
        self.analyzed_signature = Some(signature);
    }

    pub fn clear_analysis(&mut self) {
        self.analysis = None;
        self.analyzed_signature = None;
    }

    pub fn dropped_folder(&mut self, path: PathBuf) {
        if path.is_dir() {
            self.folder_input = path.display().to_string();
            self.notice = None;
        } else {
            self.notice = Some(format!(
                "{} is not a folder; drop the folder that contains the RPFM TSV exports.",
                path.display()
            ));
        }
    }

    /// Project edits that change analysis inputs call this to schedule a fresh analysis.
    fn project_changed(&mut self, ctx: &mut PanelCtx<'_>) {
        *ctx.dirty = true;
        if ctx.source_ready {
            self.analysis_wanted = true;
        }
    }

    fn mode_changed(&mut self, ctx: &mut PanelCtx<'_>) {
        self.clear_analysis();
        self.project_changed(ctx);
    }

    pub fn target_line(&self, project: &Project) -> String {
        match (project.target_mode, &project.target) {
            (TargetMode::Vanilla, _) => "Vanilla Napoleon".into(),
            (TargetMode::Tsv, None) => "TSV mode · no dataset selected (export blocked)".into(),
            (TargetMode::Tsv, Some(binding)) => {
                if let Some(Ok(info)) = &self.info {
                    if info.binding == *binding {
                        return format!("TSV dataset {} · language {}", info.name, info.language);
                    }
                }
                match self
                    .datasets
                    .iter()
                    .find(|summary| summary.binding == *binding)
                {
                    Some(summary) => format!(
                        "TSV dataset {} · language {}",
                        summary.name, summary.language
                    ),
                    None => format!("TSV dataset {}", binding.id),
                }
            }
        }
    }

    pub fn status_line(&self, project: &Project) -> (String, Color32) {
        if let Some(reason) = workflow::dataset_blocker(project) {
            return (reason.into(), RED);
        }
        match &self.analysis {
            None if self.analysis_running => ("Analyzing…".into(), AMBER),
            None => ("Not analyzed".into(), AMBER),
            Some(analysis) => {
                let stale = self.analyzed_signature.as_deref()
                    != Some(workflow::signature(project).as_str());
                if stale {
                    return ("Analysis out of date".into(), AMBER);
                }
                match workflow::standing(project, analysis) {
                    workflow::Standing::Blocked(open) => {
                        let waiting = workflow::pending_count(project, analysis);
                        if waiting > 0 {
                            (
                                format!("{open} blocking issue(s) · {waiting} region(s) awaiting an owner"),
                                RED,
                            )
                        } else {
                            (format!("{open} blocking issue(s)"), RED)
                        }
                    }
                    workflow::Standing::PendingOwnership(waiting) => (
                        format!("{waiting} region(s) awaiting an owner · remaining checks pending"),
                        AMBER,
                    ),
                    workflow::Standing::NotReady => ("Analysis not ready".into(), RED),
                    workflow::Standing::Ready => ("Ready · gameplay not verified".into(), AMBER),
                }
            }
        }
    }

    /// One action that takes the user from any panel to the ownership map with every region still awaiting an owner
    /// selected, ready for "Assign faction to selected".
    fn pending_jump(&mut self, ui: &mut egui::Ui, project: &Project, analysis: &Analysis) {
        let waiting = workflow::pending_count(project, analysis);
        if waiting > 0
            && ui
                .button(format!("Assign ownership ({waiting} region(s))"))
                .clicked()
        {
            self.focus_owner = Some(OwnerFocus::Pending);
            self.tab = Some(Tab::Map);
        }
    }

    fn spec(&self) -> ImportSpec {
        let folder = PathBuf::from(self.folder_input.trim());
        let name = match self.name.trim() {
            "" => folder
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            name => name.to_string(),
        };
        ImportSpec {
            name,
            language: self.language.trim().into(),
            folder,
            resources_reviewed: self.resources_reviewed,
            extras: self.import_extras,
        }
    }

    // ------------------------------------------------------------------ import tab

    pub fn tables_panel(&mut self, ui: &mut egui::Ui, ctx: &mut PanelCtx<'_>) -> Vec<Request> {
        let mut requests = Vec::new();
        egui::ScrollArea::vertical()
            .id_salt("tables-scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.heading("Target database: RPFM TSV tables");
                ui.label(
                    "Export the mod's database and localisation tables from RPFM (Export TSV) into one folder. The editor only reads that folder and stores an immutable, hash-verified copy. TSV tables are authoritative conversion inputs: a table that is required but not exported blocks the import. Imported tables are not an export list: the export contains only the campaign additions/overrides and explicit conflict resolutions (plus merged localisation); the installed target mod must supply unchanged tables such as land/naval unit stats and existing factions. Extra tables you also export are retained for analysis only when you opt in.",
                );
                self.mode_selector(ui, ctx);
                ui.separator();
                self.dataset_list(ui, ctx, &mut requests);
                ui.separator();
                self.import_form(ui, ctx, &mut requests);
                ui.separator();
                self.adapter_section(ui, ctx, &mut requests);
                ui.separator();
                ui.colored_label(
                    AMBER,
                    "No exported campaign is gameplay-verified. Compatibility checks are structural; test in Napoleon before relying on a TSV-based export.",
                );
            });
        requests
    }

    fn mode_selector(&mut self, ui: &mut egui::Ui, ctx: &mut PanelCtx<'_>) {
        ui.horizontal(|ui| {
            let tsv = ctx.project.target_mode == TargetMode::Tsv;
            if ui
                .add_enabled(
                    !ctx.busy,
                    egui::SelectableLabel::new(!tsv, "Vanilla Napoleon"),
                )
                .clicked()
                && tsv
            {
                workflow::use_vanilla(ctx.project);
                self.mode_changed(ctx);
            }
            if ui
                .add_enabled(
                    !ctx.busy,
                    egui::SelectableLabel::new(tsv, "Modded: TSV dataset"),
                )
                .clicked()
                && !tsv
            {
                workflow::select_tsv_mode(ctx.project);
                self.mode_changed(ctx);
            }
        });
        match (ctx.project.target_mode, &ctx.project.target) {
            (TargetMode::Vanilla, _) => {
                ui.label("Vanilla: the shipped Napoleon files in the verified source snapshot are the target. Only the editor's own content is merged.");
            }
            (TargetMode::Tsv, None) => {
                ui.colored_label(
                    RED,
                    "TSV mode has no dataset yet. Import one below or pin a stored dataset; export is blocked until then (it never falls back to vanilla).",
                );
            }
            (TargetMode::Tsv, Some(_)) => {}
        }
    }

    fn dataset_list(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &mut PanelCtx<'_>,
        requests: &mut Vec<Request>,
    ) {
        ui.horizontal(|ui| {
            ui.strong(format!("Stored datasets ({})", self.datasets.len()));
            ui.add(
                egui::TextEdit::singleline(&mut self.dataset_filter)
                    .hint_text("Filter datasets")
                    .desired_width(200.0),
            );
            if ui
                .add_enabled(!ctx.busy, egui::Button::new("Refresh"))
                .clicked()
            {
                requests.push(Request::RefreshDatasets(ctx.project.target.clone()));
            }
        });
        for problem in &self.dataset_problems {
            ui.colored_label(RED, format!("Dataset problem: {problem}"));
        }
        if self.datasets.is_empty() {
            ui.label("No datasets stored in this workspace yet. Import one below.");
        }
        let filter = self.dataset_filter.to_lowercase();
        let mut pin = None;
        for summary in &self.datasets {
            if !filter.is_empty()
                && !summary.name.to_lowercase().contains(&filter)
                && !summary.binding.id.to_lowercase().contains(&filter)
            {
                continue;
            }
            let pinned = ctx.project.target.as_ref() == Some(&summary.binding);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(&summary.name);
                    ui.label(format!("language: {}", summary.language));
                    if pinned {
                        ui.colored_label(GREEN, "Pinned to this project");
                    }
                });
                ui.monospace(format!(
                    "id {} · fingerprint {}",
                    summary.binding.id,
                    short(&summary.binding.fingerprint)
                ));
                if pinned {
                    match &self.info {
                        Some(Ok(info)) if info.binding == summary.binding => {
                            if info.resources_reviewed {
                                ui.colored_label(GREEN, "Resource limits reviewed at import.");
                            } else {
                                ui.colored_label(RED, "Resource limits were NOT confirmed at import: export is blocked. Import the folder again with the confirmation ticked.");
                            }
                        }
                        Some(Err(error)) => {
                            ui.colored_label(RED, format!("Cannot load this dataset: {error}"));
                        }
                        _ => {}
                    }
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!ctx.busy && !pinned, egui::Button::new("Use for this project (pin)"))
                        .on_hover_text("Pins this exact dataset version (by fingerprint) and switches the project to TSV mode.")
                        .clicked()
                    {
                        pin = Some(summary.binding.clone());
                    }
                    if ui
                        .add_enabled(!ctx.busy, egui::Button::new("Check dataset"))
                        .on_hover_text("Re-verify the stored copy and compare the original TSV folder. Nothing is written.")
                        .clicked()
                    {
                        requests.push(Request::Check(summary.binding.clone()));
                    }
                });
                if let Some((id, changes)) = &self.checked {
                    if *id == summary.binding.id {
                        if changes.is_empty() {
                            ui.colored_label(GREEN, "Stored copy verified; the original folder is unchanged.");
                        } else {
                            ui.colored_label(
                                AMBER,
                                format!(
                                    "{} difference(s) between the stored dataset and the TSV folder. Import the folder again to adopt them; saved resolutions stay valid when their issue inputs are unchanged.",
                                    changes.len()
                                ),
                            );
                            egui::ScrollArea::vertical()
                                .id_salt(("dataset-changes", &summary.binding.id))
                                .max_height(140.0)
                                .show(ui, |ui| {
                                    for change in changes {
                                        ui.monospace(change);
                                    }
                                });
                        }
                    }
                }
            });
        }
        if let Some(binding) = pin {
            workflow::pin_target(ctx.project, binding);
            self.mode_changed(ctx);
        }
        if let Some(target) = &ctx.project.target {
            if self.datasets_loaded
                && !self.datasets.is_empty()
                && !self
                    .datasets
                    .iter()
                    .any(|summary| summary.binding == *target)
            {
                ui.colored_label(
                    RED,
                    format!(
                        "Pinned dataset {} is missing from this workspace's dataset list.",
                        target.id
                    ),
                );
            }
        }
    }

    fn import_form(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &mut PanelCtx<'_>,
        requests: &mut Vec<Request>,
    ) {
        ui.strong("Import a TSV folder");
        ui.label("Drop the folder onto this window or type its path. Only RPFM-exported .tsv files are read; everything else is ignored and the folder is never modified.");
        ui.horizontal(|ui| {
            ui.label("Folder");
            ui.add(
                egui::TextEdit::singleline(&mut self.folder_input)
                    .hint_text("Folder containing the RPFM TSV exports")
                    .desired_width(520.0),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.add(
                egui::TextEdit::singleline(&mut self.name)
                    .hint_text("defaults to the folder name")
                    .desired_width(260.0),
            );
            ui.label("Localisation language");
            ui.add(
                egui::TextEdit::singleline(&mut self.language)
                    .hint_text("e.g. en, fr")
                    .desired_width(90.0),
            );
        });
        ui.label("The language names the target text: full text/localisation.loc and text/ui.loc TSVs must be present; English or base text is never substituted for missing target text.");
        ui.colored_label(
            AMBER,
            format!(
                "{}: {}",
                workflow::OWNERSHIP_LABEL,
                workflow::OWNERSHIP_HELP
            ),
        );
        ui.small("Only the tables listed as required for the existing-faction campaign are imported; every other table in the folder is ignored, and the references of your rows never add requirements.");
        ui.checkbox(
            &mut self.import_extras,
            "Also import the optional tables found in the folder (retained for analysis only; never required, not forced into the export)",
        );
        ui.checkbox(
            &mut self.resources_reviewed,
            "I have reviewed what TSV cannot provide: these tables carry no campaign, UI or script resources. Anything not provided separately uses the original Napoleon conversion templates; already-merged resources can be imported below.",
        );
        if !self.resources_reviewed {
            ui.colored_label(
                AMBER,
                "Without this confirmation the dataset can be imported but exporting a ZIP stays blocked.",
            );
        }
        if let Some(notice) = &self.notice {
            ui.colored_label(RED, notice);
        }
        let spec = self.spec();
        let fresh = self
            .preview
            .as_ref()
            .is_some_and(|(scanned, _)| same_source(scanned, &spec));
        let basic = !spec.name.is_empty() && !spec.language.is_empty() && spec.folder.is_dir();
        let clean = fresh
            && self.preview.as_ref().is_some_and(|(_, preview)| {
                preview.problems.is_empty() && preview.missing.is_empty()
            });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    basic && !ctx.busy && ctx.source_ready,
                    egui::Button::new("Scan folder"),
                )
                .on_hover_text("Parses the TSVs (read only) and checks them against the required tables before importing.")
                .on_disabled_hover_text("Needs an existing folder, a name, a language and a verified source snapshot.")
                .clicked()
            {
                requests.push(Request::Preview(spec.clone()));
            }
            if ui
                .add_enabled(
                    basic && clean && !ctx.busy && ctx.source_ready,
                    egui::Button::new("Import dataset").fill(Color32::from_rgb(61, 92, 73)),
                )
                .on_disabled_hover_text("Scan the current folder first; import is blocked while tables are missing or the scan reports problems.")
                .clicked()
            {
                requests.push(Request::Import(spec.clone()));
            }
            if ui.button("Clear form").clicked() {
                self.folder_input.clear();
                self.name.clear();
                self.resources_reviewed = false;
                self.notice = None;
                self.preview = None;
            }
        });
        self.requirement_checklist(ui, fresh);
    }

    fn requirement_checklist(&self, ui: &mut egui::Ui, fresh: bool) {
        let scanned = self.preview.as_ref().map(|(_, preview)| preview);
        let (required, missing): (&[Requirement], &[String]) = match scanned {
            Some(preview) => (preview.required.as_slice(), preview.missing.as_slice()),
            None => (self.requirements.as_slice(), &[][..]),
        };
        if let Some(preview) = scanned {
            if !fresh {
                ui.colored_label(
                    AMBER,
                    "Folder or settings changed since the last scan; scan again before importing.",
                );
            }
            let extras = preview.extras().len();
            ui.label(format!(
                "Scanned {} TSV file(s); {} table(s) will be imported ({extras} optional), {} optional table(s) ignored.",
                preview.files,
                preview.tables.len(),
                preview.ignored.len()
            ));
            if extras > 0 {
                ui.small("Optional extras are not required and add no requirements; they are retained for analysis only and are not forced into the export unless the build changes them.");
            }
            if !preview.problems.is_empty() {
                ui.colored_label(
                    RED,
                    format!("{} parse/validation problem(s):", preview.problems.len()),
                );
                egui::ScrollArea::vertical()
                    .id_salt("preview-problems")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for problem in &preview.problems {
                            ui.colored_label(RED, problem);
                        }
                    });
            }
            for warning in &preview.warnings {
                ui.colored_label(AMBER, warning);
            }
        }
        let header = match scanned {
            Some(_) if missing.is_empty() => {
                RichText::new(format!("Required tables ({}): all present", required.len()))
                    .color(GREEN)
            }
            Some(_) => RichText::new(format!(
                "Required tables ({}): {} missing",
                required.len(),
                missing.len()
            ))
            .color(RED),
            None => RichText::new(format!(
                "Required tables ({}) — scan a folder to check them",
                required.len()
            )),
        };
        egui::CollapsingHeader::new(header)
            .id_salt("required-tables")
            .default_open(!missing.is_empty() || scanned.is_none())
            .show(ui, |ui| {
                if required.is_empty() {
                    ui.label("The required table list is derived from the verified originals; it appears once they are loaded.");
                }
                egui::ScrollArea::vertical()
                    .id_salt("required-tables-rows")
                    .max_height(260.0)
                    .show(ui, |ui| {
                        for requirement in required {
                            let (mark, color) = match scanned {
                                Some(_) if missing.contains(&requirement.table) => ("✗", RED),
                                Some(_) => ("✓", GREEN),
                                None => ("•", Color32::LIGHT_GRAY),
                            };
                            ui.horizontal_wrapped(|ui| {
                                ui.colored_label(color, mark);
                                ui.monospace(&requirement.table);
                                ui.small("[Required]");
                                ui.label(format!("— {}", requirement.reason));
                            });
                        }
                        for table in missing {
                            if !required.iter().any(|requirement| requirement.table == *table) {
                                ui.horizontal_wrapped(|ui| {
                                    ui.colored_label(RED, "✗");
                                    ui.monospace(table);
                                });
                            }
                        }
                    });
            });
    }

    fn adapter_section(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &mut PanelCtx<'_>,
        requests: &mut Vec<Request>,
    ) {
        ui.strong("Resource adapter (optional)");
        ui.label("TSV tables cannot carry custom UI, campaign scripts or start positions. If review reports such a gap, import a folder of already-merged resources (ui/, script/, campaigns/wld_napoleon/scripting.lua, or a start position donor at campaigns/eur_napoleon/startpos.esf) prepared for exactly this dataset. DB and localisation files are not accepted here.");
        let Some(target) = ctx.project.target.clone() else {
            ui.label("Adapters bind to a pinned TSV dataset.");
            return;
        };
        match &ctx.project.adapter {
            Some(adapter) => {
                let current = adapter.target_fingerprint == target.fingerprint;
                ui.monospace(format!(
                    "adapter {} · {} · bound to dataset {}",
                    adapter.id,
                    short(&adapter.fingerprint),
                    short(&adapter.target_fingerprint)
                ));
                if !current {
                    ui.colored_label(RED, "This adapter was prepared for a different dataset fingerprint and will block export.");
                }
                if ui
                    .add_enabled(!ctx.busy, egui::Button::new("Remove adapter"))
                    .clicked()
                {
                    ctx.project.adapter = None;
                    self.project_changed(ctx);
                }
            }
            None => {
                ui.label("No adapter imported.");
            }
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.adapter_folder)
                    .hint_text("Folder of merged resources (data-relative paths)")
                    .desired_width(520.0),
            );
            if ui
                .add_enabled(
                    !ctx.busy && ctx.source_ready && !self.adapter_folder.trim().is_empty(),
                    egui::Button::new("Import adapter folder"),
                )
                .clicked()
            {
                requests.push(Request::ImportAdapter(
                    target.clone(),
                    PathBuf::from(self.adapter_folder.trim()),
                ));
            }
        });
    }

    // -------------------------------------------------------------- mappings tab

    pub fn mapping_panel(&mut self, ui: &mut egui::Ui, ctx: &mut PanelCtx<'_>) -> Vec<Request> {
        let mut requests = Vec::new();
        ui.heading("Faction, unit and building mapping");
        ui.label("Map port (ETW) identities onto the imported TSV content. Targets come from the analyzed catalog of the selected dataset; nothing is silently substituted. Units map only to NTW stand-in keys. Faction donors name an existing target faction whose campaign structure seeds a new identity.");
        ui.colored_label(
            AMBER,
            "Faction mappings stay optional. Fix regions without an existing owner by assigning an existing target faction in the Ownership map; no mapping is needed for every absent faction.",
        );
        let Some(analysis) = self.analysis.clone() else {
            ui.colored_label(
                AMBER,
                "Run analysis first so the effective target catalog is known.",
            );
            if ui
                .add_enabled(
                    !ctx.busy && !self.analysis_running && ctx.source_ready,
                    egui::Button::new("Analyze"),
                )
                .clicked()
            {
                requests.push(Request::Analyze);
            }
            return requests;
        };
        let stale =
            self.analyzed_signature.as_deref() != Some(workflow::signature(ctx.project).as_str());
        if stale {
            ui.colored_label(
                AMBER,
                "Catalog is from an earlier analysis of different project inputs.",
            );
        }
        let waiting = workflow::pending_count(ctx.project, &analysis);
        if waiting > 0 {
            ui.colored_label(AMBER, workflow::pending_message(waiting));
            self.pending_jump(ui, ctx.project, &analysis);
        }
        ui.horizontal(|ui| {
            for (index, field) in MAPPING_FIELDS.iter().enumerate() {
                let count =
                    workflow::mapping_ref(&ctx.project.mappings, field).map_or(0, BTreeMap::len);
                let open = analysis
                    .issues
                    .iter()
                    .filter(|issue| {
                        !issue.resolved
                            && issue
                                .mapping
                                .as_ref()
                                .is_some_and(|hint| hint.field == *field)
                    })
                    .count();
                let mut text = format!("{} ({count})", mapping_title(field));
                if open > 0 {
                    text.push_str(&format!(" · {open} needed"));
                }
                if ui
                    .selectable_label(self.mapping_field == index, text)
                    .clicked()
                {
                    self.mapping_field = index;
                    self.new_from.clear();
                    self.new_to.clear();
                }
            }
        });
        let field = MAPPING_FIELDS[self.mapping_field.min(MAPPING_FIELDS.len() - 1)];
        let targets = workflow::target_options(&analysis, field);
        let sources = workflow::source_options(&analysis, field);
        ui.separator();

        let needed: Vec<&Issue> = analysis
            .issues
            .iter()
            .filter(|issue| {
                !issue.resolved
                    && issue
                        .mapping
                        .as_ref()
                        .is_some_and(|hint| hint.field == field)
            })
            .collect();
        if !needed.is_empty() {
            ui.colored_label(
                RED,
                format!("Mappings required by analysis ({}):", needed.len()),
            );
            egui::ScrollArea::vertical()
                .id_salt("needed-mappings")
                .max_height(120.0)
                .show(ui, |ui| {
                    for issue in needed {
                        let key = issue
                            .mapping
                            .as_ref()
                            .map(|hint| hint.key.clone())
                            .unwrap_or_default();
                        ui.horizontal(|ui| {
                            if ui.small_button("Map…").clicked() {
                                self.new_from = key.clone();
                                self.new_to.clear();
                            }
                            ui.monospace(&key);
                            ui.label(&issue.title);
                        });
                    }
                });
            ui.separator();
        }

        ui.horizontal(|ui| {
            let (from_label, to_label) = if field == "faction_templates" {
                ("Target faction", "Donor faction")
            } else {
                ("Port key", "Target key")
            };
            ui.label(from_label);
            search_select(
                ui,
                ("map-from", field),
                &mut self.new_from,
                &sources,
                "search…",
            );
            ui.label("→");
            ui.label(to_label);
            search_select(ui, ("map-to", field), &mut self.new_to, &targets, "search…");
            if ui
                .add_enabled(
                    !ctx.busy && !self.new_from.is_empty() && !self.new_to.is_empty(),
                    egui::Button::new("Set mapping"),
                )
                .clicked()
            {
                match workflow::set_mapping(
                    ctx.project,
                    &analysis,
                    field,
                    &self.new_from,
                    &self.new_to,
                ) {
                    Ok(()) => {
                        self.new_from.clear();
                        self.new_to.clear();
                        self.edit_error = None;
                        self.project_changed(ctx);
                    }
                    Err(error) => self.edit_error = Some(format!("{error:#}")),
                }
            }
        });
        if let Some(error) = &self.edit_error {
            ui.colored_label(RED, error);
        }
        ui.add(
            egui::TextEdit::singleline(&mut self.mapping_filter)
                .hint_text("Filter current mappings")
                .desired_width(260.0),
        );
        let filter = self.mapping_filter.to_lowercase();
        let mut remove = None;
        let mut change = None;
        if let Some(map) = workflow::mapping_ref(&ctx.project.mappings, field) {
            let rows: Vec<(&String, &String)> = map
                .iter()
                .filter(|(from, to)| {
                    filter.is_empty()
                        || from.to_lowercase().contains(&filter)
                        || to.to_lowercase().contains(&filter)
                })
                .collect();
            if rows.is_empty() {
                ui.label("No mappings in this category.");
            }
            egui::ScrollArea::vertical()
                .id_salt(("mapping-rows", field))
                .show_rows(ui, 26.0, rows.len(), |ui, range| {
                    for (from, to) in &rows[range] {
                        ui.horizontal(|ui| {
                            ui.monospace(from.as_str());
                            ui.label("→");
                            let valid = targets.iter().any(|target| target == *to);
                            let mut value = (*to).clone();
                            if search_select(
                                ui,
                                ("row", field, from.as_str()),
                                &mut value,
                                &targets,
                                "",
                            ) {
                                change = Some(((*from).clone(), value));
                            }
                            if !valid {
                                ui.colored_label(RED, "not in effective target catalog");
                            }
                            if ui
                                .add_enabled(!ctx.busy, egui::Button::new("Remove"))
                                .clicked()
                            {
                                remove = Some((*from).clone());
                            }
                        });
                    }
                });
        }
        let mut edited = false;
        if let Some((from, to)) = change {
            match workflow::set_mapping(ctx.project, &analysis, field, &from, &to) {
                Ok(()) => edited = true,
                Err(error) => self.edit_error = Some(format!("{error:#}")),
            }
        }
        if let Some(from) = remove {
            if workflow::remove_mapping(ctx.project, field, &from).unwrap_or(false) {
                edited = true;
            }
        }
        if edited {
            self.project_changed(ctx);
        }
        requests
    }

    // -------------------------------------------------------- compatibility tab

    pub fn review_panel(&mut self, ui: &mut egui::Ui, ctx: &mut PanelCtx<'_>) -> Vec<Request> {
        let mut requests = Vec::new();
        ui.heading("Compatibility review");
        ui.colored_label(
            AMBER,
            "GAMEPLAY NOT VERIFIED — checks are structural only. Never claim a campaign works until it has been played in Napoleon.",
        );
        ui.label("Duplicate conflicts in untouched tables outside campaign inputs are warnings; campaign inputs and tables receiving converted rows still require resolution.");
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !ctx.busy && !self.analysis_running && ctx.source_ready,
                    egui::Button::new("Analyze"),
                )
                .on_hover_text("Reads the verified sources and the selected TSV dataset; no game files are touched.")
                .clicked()
            {
                requests.push(Request::Analyze);
            }
            if self.analysis_running {
                ui.spinner();
            }
        });
        if let Some(reason) = workflow::dataset_blocker(ctx.project) {
            ui.colored_label(RED, reason);
            if ui.button("Open TSV import").clicked() {
                self.tab = Some(Tab::Tables);
            }
        }
        let Some(analysis) = self.analysis.clone() else {
            ui.label("No analysis yet.");
            return requests;
        };
        let stale = workflow::stale_resolutions(ctx.project, &analysis);
        if !stale.is_empty() {
            ui.colored_label(
                AMBER,
                format!(
                    "{} saved resolution(s) no longer match the current issues and are ignored:",
                    stale.len()
                ),
            );
            for (id, reason) in &stale {
                ui.monospace(format!("{id}: {reason}"));
            }
            if ui
                .add_enabled(!ctx.busy, egui::Button::new("Discard stale resolutions"))
                .clicked()
            {
                for (id, _) in &stale {
                    ctx.project.resolutions.remove(id);
                }
                self.project_changed(ctx);
            }
        }
        let waiting = workflow::pending_count(ctx.project, &analysis);
        if waiting > 0 {
            ui.colored_label(AMBER, workflow::pending_message(waiting));
            self.pending_jump(ui, ctx.project, &analysis);
        }
        ui.checkbox(&mut self.show_info, "Show informational notes");
        egui::ScrollArea::vertical()
            .id_salt("review-scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.issues(ui, ctx, &analysis);
                if !analysis.warnings.is_empty() {
                    egui::CollapsingHeader::new(format!("Warnings ({})", analysis.warnings.len()))
                        .default_open(true)
                        .show(ui, |ui| {
                            for warning in &analysis.warnings {
                                ui.colored_label(AMBER, warning);
                            }
                        });
                }
            });
        requests
    }

    fn issues(&mut self, ui: &mut egui::Ui, ctx: &mut PanelCtx<'_>, analysis: &Analysis) {
        let mut categories: Vec<(&str, &str)> = CATEGORIES.to_vec();
        for issue in &analysis.issues {
            if !categories.iter().any(|(key, _)| *key == issue.category) {
                categories.push((issue.category.as_str(), issue.category.as_str()));
            }
        }
        let mut apply: Option<(String, String)> = None;
        let mut clear: Option<String> = None;
        let mut goto_mapping: Option<String> = None;
        let mut goto_tables = false;
        let mut goto_owner: Option<OwnerFocus> = None;
        let mut shown_any = false;
        for (key, title) in categories {
            let issues: Vec<&Issue> = analysis
                .issues
                .iter()
                .filter(|issue| {
                    issue.category == key && (self.show_info || issue.severity != "info")
                })
                .collect();
            if issues.is_empty() {
                continue;
            }
            shown_any = true;
            let open = issues
                .iter()
                .filter(|i| i.severity == "blocker" && !i.resolved)
                .count();
            let header = if open > 0 {
                RichText::new(format!("{title} — {open} blocking of {}", issues.len())).color(RED)
            } else {
                RichText::new(format!("{title} ({})", issues.len())).color(GREEN)
            };
            egui::CollapsingHeader::new(header)
                .id_salt(("issues", key))
                .default_open(open > 0)
                .show(ui, |ui| {
                    for issue in issues {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let (text, color) = match (issue.severity.as_str(), issue.resolved) {
                                    ("blocker", false) => ("BLOCKER", RED),
                                    ("blocker", true) => ("resolved", GREEN),
                                    ("warning", _) => ("warning", AMBER),
                                    _ => ("note", Color32::LIGHT_GRAY),
                                };
                                ui.colored_label(color, text);
                                ui.strong(&issue.title);
                            });
                            ui.label(&issue.detail);
                            ui.small(format!("id {} · input {}", issue.id, short(&issue.input_hash)));
                            let saved = ctx.project.resolutions.get(&issue.id);
                            if !issue.choices.is_empty() {
                                let current = self
                                    .issue_choice
                                    .get(&issue.id)
                                    .cloned()
                                    .or_else(|| {
                                        saved
                                            .filter(|r| r.input_hash == issue.input_hash)
                                            .map(|r| r.choice.clone())
                                    })
                                    .unwrap_or_default();
                                ui.horizontal_wrapped(|ui| {
                                    for choice in &issue.choices {
                                        if ui
                                            .selectable_label(current == *choice, choice_label(choice))
                                            .clicked()
                                        {
                                            self.issue_choice.insert(issue.id.clone(), choice.clone());
                                        }
                                    }
                                });
                                ui.horizontal(|ui| {
                                    let already = saved.is_some_and(|r| {
                                        r.input_hash == issue.input_hash && r.choice == current
                                    });
                                    if ui
                                        .add_enabled(
                                            !ctx.busy && !current.is_empty() && !already,
                                            egui::Button::new("Apply choice"),
                                        )
                                        .clicked()
                                    {
                                        apply = Some((issue.id.clone(), current.clone()));
                                    }
                                    if saved.is_some()
                                        && ui.add_enabled(!ctx.busy, egui::Button::new("Clear choice")).clicked()
                                    {
                                        clear = Some(issue.id.clone());
                                    }
                                });
                            } else if issue.severity == "blocker" && !issue.resolved {
                                ui.horizontal(|ui| match (&issue.mapping, issue.category.as_str()) {
                                    _ if workflow::owner_blocker_region(issue).is_some() => {
                                        ui.label("This region's owner override is not an existing target faction: assign one in the Ownership map.");
                                        if ui.button("Open Ownership map").clicked() {
                                            goto_owner = workflow::owner_blocker_region(issue)
                                                .map(|region| OwnerFocus::Region(region.to_owned()));
                                        }
                                    }
                                    (Some(hint), _) => {
                                        if ui.button(format!("Open {} mapping", mapping_title(&hint.field))).clicked() {
                                            goto_mapping = Some(hint.field.clone());
                                            self.new_from = hint.key.clone();
                                        }
                                    }
                                    (None, "ui" | "script" | "template" | "adapter") => {
                                        ui.label("Needs a dataset-bound merged adapter.");
                                        if ui.button("Open adapter import").clicked() {
                                            goto_tables = true;
                                        }
                                    }
                                    _ => {
                                        ui.label("No in-app resolution: fix the TSV export (or confirm the resource review), import it again and re-run analysis.");
                                        if ui.button("Open TSV import").clicked() {
                                            goto_tables = true;
                                        }
                                    }
                                });
                            }
                        });
                    }
                });
        }
        if !shown_any {
            match workflow::standing(ctx.project, analysis) {
                workflow::Standing::Ready => {
                    ui.colored_label(GREEN, "No issues reported. Gameplay is still unverified.");
                }
                // The pending-ownership banner above already says why nothing else is reported.
                workflow::Standing::PendingOwnership(_) | workflow::Standing::Blocked(_) => {}
                workflow::Standing::NotReady => {
                    ui.colored_label(
                        AMBER,
                        "No issues listed, but the analysis is not ready; re-run analysis.",
                    );
                }
            }
        }
        let mut edited = false;
        if let Some((id, choice)) = apply {
            match workflow::apply_resolution(ctx.project, analysis, &id, &choice, None) {
                Ok(()) => {
                    self.issue_choice.remove(&id);
                    edited = true;
                }
                Err(error) => self.edit_error = Some(format!("{error:#}")),
            }
        }
        if let Some(id) = clear {
            ctx.project.resolutions.remove(&id);
            self.issue_choice.remove(&id);
            edited = true;
        }
        if edited {
            self.project_changed(ctx);
        }
        if let Some(error) = &self.edit_error {
            ui.colored_label(RED, error);
        }
        if let Some(field) = goto_mapping {
            if let Some(index) = MAPPING_FIELDS.iter().position(|f| *f == field) {
                self.mapping_field = index;
            }
            self.tab = Some(Tab::Mappings);
        }
        if goto_tables {
            self.tab = Some(Tab::Tables);
        }
        if let Some(focus) = goto_owner {
            self.focus_owner = Some(focus);
            self.tab = Some(Tab::Map);
        }
    }

    // -------------------------------------------------------------- export tab

    pub fn export_panel(&mut self, ui: &mut egui::Ui, ctx: &mut PanelCtx<'_>) -> Vec<Request> {
        let mut requests = Vec::new();
        ui.heading("Export review");
        ui.colored_label(
            AMBER,
            "GAMEPLAY NOT VERIFIED — the ZIP contains data/... paths for manual extraction; the app never writes to a game folder. It carries only campaign additions/overrides, explicit conflict-resolution DB replacements and merged localisation; unchanged imported tables (e.g. land/naval unit stats, existing factions) must come from the installed target mod.",
        );
        ui.add_space(6.0);
        ui.collapsing("Terrain relief of the generated map", |ui| {
            ui.label(
                "The ETW world squeezes continents into far fewer map units than Napoleon's own maps and paints \
                 uplands about three times higher, which Napoleon renders as noisy, bulging terrain. The curve keeps \
                 lowlands, compresses uplands and smooths land only (coastlines and sea are never changed).",
            );
            ui.small("Fixed theatre frames and matching terrain are included automatically, before the relief curve.");
            let t = &mut ctx.project.terrain;
            let before = t.clone();
            ui.horizontal(|ui| {
                ui.label("Lowland kept up to height");
                ui.add(egui::Slider::new(&mut t.knee, 0..=255));
            });
            ui.horizontal(|ui| {
                ui.label("Upland scale");
                ui.add(egui::Slider::new(&mut t.upland_scale, 0.05..=1.0).fixed_decimals(2));
            });
            ui.horizontal(|ui| {
                ui.label("Smoothing (pixels, 3.2 px per map unit)");
                ui.add(egui::Slider::new(&mut t.smooth_sigma, 0.0..=8.0).fixed_decimals(1));
            });
            ui.horizontal(|ui| {
                if ui.button("Napoleon-like (default)").clicked() {
                    *t = crate::model::TerrainRelief::default();
                }
                if ui.button("Original ETW relief").clicked() {
                    *t = crate::model::TerrainRelief::original();
                }
                ui.label(if t.is_original() {
                    "original relief curve; fixed theatre terrain still included"
                } else {
                    "applied to display/heightmap/heightmap.tga and the fog mask at build"
                });
            });
            if *t != before {
                *ctx.dirty = true;
            }
        });
        let blocker = workflow::export_blocker(
            ctx.project,
            self.analysis.as_ref(),
            self.analyzed_signature.as_deref(),
        );
        let current =
            self.analyzed_signature.as_deref() == Some(workflow::signature(ctx.project).as_str());
        let open = self
            .analysis
            .as_ref()
            .map_or(0, workflow::blocker_count);
        let gate = |ui: &mut egui::Ui, ok: bool, text: String| {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(if ok { GREEN } else { RED }, if ok { "✓" } else { "✗" });
                ui.label(text);
            });
        };
        gate(
            ui,
            ctx.source_ready,
            if ctx.source_ready {
                "Original snapshot verified".into()
            } else {
                "Original snapshot required: import the original game files first".into()
            },
        );
        match (ctx.project.target_mode, &ctx.project.target) {
            (TargetMode::Vanilla, _) => gate(ui, true, "Target: vanilla Napoleon".into()),
            (TargetMode::Tsv, None) => gate(ui, false, "TSV mode: no dataset selected".into()),
            (TargetMode::Tsv, Some(binding)) => {
                gate(
                    ui,
                    true,
                    format!(
                        "TSV dataset {} ({})",
                        binding.id,
                        short(&binding.fingerprint)
                    ),
                );
                match &self.info {
                    Some(Ok(info)) if info.binding == *binding => gate(
                        ui,
                        info.resources_reviewed,
                        if info.resources_reviewed {
                            "Resource limits reviewed (original templates apply to anything not supplied)".into()
                        } else {
                            "Resource review not confirmed: import the folder again with the confirmation".into()
                        },
                    ),
                    Some(Err(error)) => {
                        gate(ui, false, format!("Dataset cannot be loaded: {error}"))
                    }
                    _ => gate(ui, false, "Dataset details are still loading".into()),
                }
            }
        }
        gate(
            ui,
            self.analysis.is_some() && current,
            if self.analysis.is_none() {
                "Analysis has not been run".into()
            } else if current {
                "Analysis is current".into()
            } else {
                "Analysis is out of date".into()
            },
        );
        gate(
            ui,
            self.analysis.is_some() && current && open == 0,
            if self.analysis.is_some() && current {
                format!("{open} unresolved blocking issue(s)")
            } else {
                "Blocking issues are unknown until the analysis is current".into()
            },
        );
        match &self.analysis {
            Some(analysis) if current => match workflow::pending_count(ctx.project, analysis) {
                0 => gate(ui, true, "No region awaits a target owner".into()),
                waiting => {
                    ui.horizontal_wrapped(|ui| {
                        ui.colored_label(AMBER, "!");
                        ui.label(format!(
                            "{waiting} region(s) await a target owner in the Ownership map (not a missing import or mapping)"
                        ));
                    });
                }
            },
            _ => gate(
                ui,
                false,
                "Regions awaiting a target owner are unknown until the analysis is current".into(),
            ),
        }
        gate(
            ui,
            true,
            format!(
                "{}: no new faction keys; the starting forces of removed factions are not transferred to the owners of their territory.",
                workflow::OWNERSHIP_LABEL
            ),
        );
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !ctx.busy && !self.analysis_running && ctx.source_ready,
                    egui::Button::new("Analyze"),
                )
                .clicked()
            {
                requests.push(Request::Analyze);
            }
            if self.analysis_running {
                ui.spinner();
            }
            let export = ui.add_enabled(
                blocker.is_none() && !ctx.busy && !self.analysis_running && ctx.source_ready,
                egui::Button::new("Save & Export ZIP").fill(Color32::from_rgb(61, 92, 73)),
            );
            let export = match &blocker {
                Some(reason) => export.on_disabled_hover_text(reason),
                None => export,
            };
            if export.clicked() {
                requests.push(Request::Export);
            }
        });
        if let Some(reason) = &blocker {
            let waiting_only = workflow::only_pending_ownership(
                ctx.project,
                self.analysis.as_ref(),
                self.analyzed_signature.as_deref(),
            );
            ui.colored_label(
                if waiting_only { AMBER } else { RED },
                format!("Export unavailable: {reason}"),
            );
        }
        let Some(analysis) = self.analysis.clone() else {
            return requests;
        };
        self.pending_jump(ui, ctx.project, &analysis);
        egui::Grid::new("analysis-summary")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Target");
                ui.strong(&analysis.target_name);
                ui.end_row();
                ui.label("Language");
                ui.label(&analysis.language);
                ui.end_row();
                ui.label("Analysis fingerprint");
                ui.monospace(short(&analysis.fingerprint));
                ui.end_row();
                ui.label("Mappings");
                ui.label(workflow::mapping_count(ctx.project).to_string());
                ui.end_row();
                ui.label("Ownership overrides");
                ui.label(ctx.project.owners.len().to_string());
                ui.end_row();
                ui.label("Ownership");
                ui.label(workflow::OWNERSHIP_LABEL);
                ui.end_row();
                ui.label("Gameplay tested");
                ui.colored_label(
                    if analysis.gameplay_tested {
                        GREEN
                    } else {
                        AMBER
                    },
                    if analysis.gameplay_tested {
                        "yes"
                    } else {
                        "no"
                    },
                );
                ui.end_row();
            });
        egui::ScrollArea::vertical()
            .id_salt("export-scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if !analysis.warnings.is_empty() {
                    egui::CollapsingHeader::new(format!("Warnings ({})", analysis.warnings.len()))
                        .default_open(true)
                        .show(ui, |ui| {
                            for warning in &analysis.warnings {
                                ui.colored_label(AMBER, warning);
                            }
                        });
                }
                egui::CollapsingHeader::new(format!(
                    "Planned changes ({})",
                    analysis.changes.len()
                ))
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.change_filter)
                            .hint_text("Filter by operation, table, key, or source")
                            .desired_width(320.0),
                    );
                    let filter = self.change_filter.to_lowercase();
                    let rows: Vec<_> = analysis
                        .changes
                        .iter()
                        .filter(|change| {
                            filter.is_empty()
                                || [
                                    &change.operation,
                                    &change.table,
                                    &change.key,
                                    &change.source,
                                    &change.detail,
                                ]
                                .iter()
                                .any(|field| field.to_lowercase().contains(&filter))
                        })
                        .collect();
                    ui.label(format!("{} shown", rows.len()));
                    egui::ScrollArea::vertical()
                        .id_salt("changes-scroll")
                        .max_height(320.0)
                        .show_rows(ui, 18.0, rows.len(), |ui, range| {
                            for change in &rows[range] {
                                ui.monospace(format!(
                                    "{} {} {} ← {} {}",
                                    change.operation,
                                    change.table,
                                    change.key,
                                    change.source,
                                    change.detail
                                ));
                            }
                        });
                });
            });
        requests
    }
}

/// The scan only depends on what is read, not on the resource-review confirmation.
fn same_source(a: &ImportSpec, b: &ImportSpec) -> bool {
    a.name == b.name && a.language == b.language && a.folder == b.folder
}

fn choice_label(choice: &str) -> String {
    match choice {
        "keep_target" => "Keep target mod's version".into(),
        "use_port" => "Use ported version".into(),
        other => match other.strip_prefix("member:") {
            Some(path) => format!("Use {path}"),
            None => other.into(),
        },
    }
}

pub fn short(hash: &str) -> &str {
    &hash[..hash.len().min(12)]
}

/// Combo box with a type-to-filter list; returns true when a different option was chosen.
pub fn search_select(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    selected: &mut String,
    options: &[String],
    placeholder: &str,
) -> bool {
    let mut changed = false;
    let text = if selected.is_empty() {
        placeholder.to_string()
    } else {
        selected.clone()
    };
    egui::ComboBox::from_id_salt(id)
        .width(240.0)
        .selected_text(text)
        .show_ui(ui, |ui| {
            let filter_id = ui.make_persistent_id("search-filter");
            let mut filter = ui
                .data_mut(|data| data.get_temp::<String>(filter_id))
                .unwrap_or_default();
            ui.add(egui::TextEdit::singleline(&mut filter).hint_text("Search…"));
            ui.data_mut(|data| data.insert_temp(filter_id, filter.clone()));
            let query = filter.to_lowercase();
            let mut shown = 0usize;
            egui::ScrollArea::vertical()
                .max_height(260.0)
                .show(ui, |ui| {
                    for option in options {
                        if !query.is_empty() && !option.to_lowercase().contains(&query) {
                            continue;
                        }
                        if ui.selectable_label(*option == *selected, option).clicked()
                            && *option != *selected
                        {
                            *selected = option.clone();
                            changed = true;
                        }
                        shown += 1;
                        if shown >= 300 {
                            ui.weak("More matches; refine the search.");
                            break;
                        }
                    }
                    if shown == 0 {
                        ui.weak("No matches.");
                    }
                });
        });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> ImportSpec {
        ImportSpec {
            name: name.into(),
            language: "en".into(),
            folder: "tsv".into(),
            resources_reviewed: false,
            extras: false,
        }
    }

    #[test]
    fn requirements_load_once_per_original_snapshot() {
        let project = Project::default();
        let mut ui = TsvUi::default();
        assert!(matches!(
            ui.pending(&project, false, true),
            Some(Request::LoadRequirements)
        ));
        assert!(matches!(
            ui.pending(&project, false, true),
            Some(Request::RefreshDatasets(_))
        ));
        assert!(ui.pending(&project, false, true).is_none());
        ui.sources_changed();
        assert!(matches!(
            ui.pending(&project, false, true),
            Some(Request::LoadRequirements)
        ));
    }

    #[test]
    fn a_scan_stays_fresh_across_the_resource_review_but_not_a_changed_folder() {
        let mut reviewed = spec("mod");
        reviewed.resources_reviewed = true;
        assert!(same_source(&reviewed, &spec("mod")));
        assert!(!same_source(&spec("other"), &spec("mod")));
    }
}
