mod activate_tool;
mod overlay;
mod split_tool;

use crate::{
    compatibility::{self, Analysis},
    local_assets,
    map::{self, RegionMap, RegionPolygon, TheatreRect, splits},
    model::{BuildReport, Progress, Project, Rules, TargetMode},
    pipeline,
    regionmap::regions::RegionData,
    snapshot::{self, Snapshot},
    target::{self, AdapterBinding, ImportSpec, Listing, Preview, Requirement, TargetBinding},
    tsv_ui::{self, DatasetInfo, OwnerFocus, PanelCtx, Request, Tab, TsvUi, search_select, workflow},
};
use activate_tool::ActivationDraft;
use overlay::Overlay;
use split_tool::{MapTool, SplitDraft};
use anyhow::{Context, Result};
use eframe::egui::{self, Color32, Pos2, Rect, Vec2};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread,
};

struct RegionRow {
    key: String,
    name: String,
    theatre: String,
    /// created by a split of this project
    split: bool,
    /// a wilderness region activated by this project
    activated: bool,
}
struct Polygon {
    key: String,
    triangles: Vec<[[f32; 2]; 3]>,
    bounds: [f32; 4],
    /// ETW land polygon without campaign rules that can be activated (drawn hatched, not selectable)
    wild: bool,
    /// a point inside the polygon where its label goes
    label_at: [f32; 2],
}
/// The project's region geometry: `RegionData` with every split applied and its triangulation (all regions).
struct Geometry {
    data: RegionData,
    raw: Vec<RegionPolygon>,
}

fn rows_of(rules: &Rules, project: &Project) -> Result<Vec<RegionRow>> {
    Ok(rules
        .regions
        .as_object()
        .context("Invalid region rules")?
        .iter()
        .map(|(key, value)| RegionRow {
            key: key.clone(),
            name: value["name"].as_str().unwrap_or(key).into(),
            theatre: value["theatre_name"].as_str().unwrap_or("").into(),
            split: project.splits.iter().any(|split| split.key == *key),
            activated: project.activations.iter().any(|activation| activation.key == *key),
        })
        .collect())
}

fn polygon_of(polygon: &RegionPolygon, wild: bool) -> Polygon {
    let mut bounds = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    // the centroid of the largest triangle lies inside the polygon
    let (mut largest, mut label_at) = (0.0f32, [0.0f32; 2]);
    for triangle in &polygon.triangles {
        for point in triangle {
            bounds[0] = bounds[0].min(point[0]);
            bounds[1] = bounds[1].min(point[1]);
            bounds[2] = bounds[2].max(point[0]);
            bounds[3] = bounds[3].max(point[1]);
        }
        let [a, b, c] = *triangle;
        let size = ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
        if size > largest {
            largest = size;
            label_at = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
        }
    }
    Polygon {
        key: polygon.key.clone(),
        triangles: polygon.triangles.clone(),
        bounds,
        wild,
        label_at,
    }
}
enum Event {
    Progress(String),
    /// Original geometry with the splits (the project's at the time) applied.
    MapReady(Box<RegionMap>, Box<Project>, PathBuf, Box<Rules>),
    Built(BuildReport),
    Verified,
    Error(String),
    Requirements(Vec<Requirement>),
    Datasets(Listing, Option<Result<DatasetInfo, String>>),
    Preview(ImportSpec, Preview),
    TargetImported(TargetBinding),
    TargetChecked(DatasetInfo, Vec<String>),
    AdapterImported(AdapterBinding),
    Analyzed(Box<Analysis>, String),
    AnalysisFailed(String),
}

pub struct Editor {
    assets: PathBuf,
    derived_assets: Option<PathBuf>,
    workspace: PathBuf,
    project: Project,
    rules: Option<Rules>,
    rows: Vec<RegionRow>,
    factions: Vec<String>,
    filtered: Vec<usize>,
    search: String,
    theatre: String,
    selected: BTreeSet<String>,
    assign_to: String,
    etw_data: String,
    ntw_data: String,
    base_language: String,
    source_ready: bool,
    busy: bool,
    dirty: bool,
    status: String,
    error: Option<String>,
    output: Option<PathBuf>,
    messages: VecDeque<String>,
    tx: Sender<Event>,
    rx: Receiver<Event>,
    polygons: Vec<Polygon>,
    bounds: [f32; 4],
    center: Vec2,
    zoom: f32,
    mesh: Option<Arc<egui::Mesh>>,
    mesh_dirty: bool,
    last_rect: Rect,
    confirm_reload: bool,
    confirm_reset: bool,
    ui: TsvUi,
    shown_tab: Tab,
    /// `workflow::catalog_key` of the analysis that is running now.
    running_key: Option<String>,
    /// `workflow::catalog_key` the stored analysis was made for; its catalog only feeds owner choices while it matches.
    analysis_key: Option<String>,
    /// `analysis_key` matches the current target settings; refreshed once per frame so per-region drawing never
    /// rehashes the catalog key.
    pending_valid: bool,
    /// regions.esf as shipped; every split change replays the project's splits on a copy of it.
    base_regions: Option<RegionData>,
    /// `base_regions` with the project's splits applied, and its triangulation (all regions, sea included).
    geometry: Option<Geometry>,
    tool: MapTool,
    draft: SplitDraft,
    /// region borders and settlement markers of `geometry`
    overlay: Overlay,
    show_overlay: bool,
    /// index into `project.splits` whose removal awaits confirmation
    confirm_remove_split: Option<usize>,
    /// theatre rectangles of regions.esf
    theatres: Vec<TheatreRect>,
    /// land regions without campaign rules that can be activated
    wild: BTreeMap<String, splits::Activatable>,
    activation: ActivationDraft,
    /// index into `project.activations` whose removal awaits confirmation
    confirm_remove_activation: Option<usize>,
    /// diagonal stripes tiled over the wilderness polygons
    hatch: Option<egui::TextureHandle>,
    wild_mesh: Option<Arc<egui::Mesh>>,
}

pub fn run(assets: PathBuf, workspace: PathBuf) -> Result<()> {
    let app = Editor::new(assets, workspace)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 900.0])
            .with_min_inner_size([960.0, 680.0])
            .with_active(false),
        ..Default::default()
    };
    eframe::run_native(
        "KGC Port Editor",
        options,
        Box::new(move |cc| {
            tsv_ui::configure_visuals(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| anyhow::anyhow!("Native window: {error}"))
}

impl Editor {
    fn new(assets: PathBuf, workspace: PathBuf) -> Result<Self> {
        let project_path = workspace.join("project.json");
        let project = if project_path.exists() {
            Project::load(&project_path)?
        } else {
            Project::default()
        };
        let factions = workflow::owner_choices(None);
        let games_root = assets
            .parent()
            .and_then(Path::parent)
            .unwrap_or(Path::new("."));
        let etw_data = games_root
            .join("Empire Total War/data")
            .display()
            .to_string();
        let ntw_data = games_root
            .join("Napoleon Total War/data")
            .display()
            .to_string();
        let (tx, rx) = mpsc::channel();
        let mut editor = Self {
            assets,
            derived_assets: None,
            workspace,
            project,
            rules: None,
            filtered: Vec::new(),
            rows: Vec::new(),
            factions,
            search: String::new(),
            theatre: String::new(),
            selected: BTreeSet::new(),
            assign_to: "france".into(),
            etw_data,
            ntw_data,
            base_language: String::new(),
            source_ready: false,
            busy: false,
            dirty: false,
            status: "Import original game files to begin.".into(),
            error: None,
            output: None,
            messages: VecDeque::new(),
            tx,
            rx,
            polygons: Vec::new(),
            bounds: [-820.0, -290.0, 980.0, 580.0],
            center: Vec2::new(80.0, 145.0),
            zoom: 1.0,
            mesh: None,
            mesh_dirty: true,
            last_rect: Rect::NOTHING,
            confirm_reload: false,
            confirm_reset: false,
            ui: TsvUi::default(),
            shown_tab: Tab::Map,
            running_key: None,
            analysis_key: None,
            pending_valid: false,
            base_regions: None,
            geometry: None,
            tool: MapTool::Select,
            draft: SplitDraft::default(),
            overlay: Overlay::default(),
            show_overlay: true,
            confirm_remove_split: None,
            theatres: Vec::new(),
            wild: BTreeMap::new(),
            activation: ActivationDraft::default(),
            confirm_remove_activation: None,
            hatch: None,
            wild_mesh: None,
        };
        if editor.project.target_mode == TargetMode::Tsv && editor.project.target.is_none() {
            editor.ui.tab = Some(Tab::Tables);
        }
        if editor.workspace.join("sources").exists() {
            editor.load_snapshot();
        }
        Ok(editor)
    }

    fn push_message(&mut self, message: String) {
        self.status = message.clone();
        self.messages.push_back(message);
        if self.messages.len() > 250 {
            self.messages.pop_front();
        }
    }

    fn load_snapshot(&mut self) {
        self.busy = true;
        self.error = None;
        let root = self.workspace.join("sources");
        let workspace = self.workspace.clone();
        let assets = self.assets.clone();
        let project = self.project.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let progress = |message: &str| {
                let _ = tx.send(Event::Progress(message.into()));
            };
            let result = (|| -> Result<_> {
                let snapshot = Snapshot::open(&root)?;
                let assets = local_assets::ensure_from_snapshot(&workspace, &assets, &snapshot, &progress)?;
                let rules = splits::project_rules(&assets.join("rules"), &project)?;
                workflow::validate_owners(&project, &rules)?;
                let map = map::region_polygons(&snapshot, &project)?;
                Ok((map, project, assets, rules))
            })();
            let _ = tx.send(match result {
                Ok((map, project, assets, rules)) => Event::MapReady(Box::new(map), Box::new(project), assets, Box::new(rules)),
                Err(error) => Event::Error(format!("{error:#}")),
            });
        });
    }

    fn import_sources(&mut self) {
        self.busy = true;
        self.error = None;
        let etw = PathBuf::from(&self.etw_data);
        let ntw = PathBuf::from(&self.ntw_data);
        let language = self.base_language.clone();
        let root = self.workspace.join("sources");
        let workspace = self.workspace.clone();
        let assets = self.assets.clone();
        let project = self.project.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let progress = |message: &str| {
                let _ = tx.send(Event::Progress(message.into()));
            };
            let result = (|| -> Result<_> {
                let language = (!language.trim().is_empty()).then(|| language.trim().to_string());
                let snapshot = snapshot::import_with_language(
                    &etw,
                    &ntw,
                    &root,
                    language.as_deref(),
                    &progress,
                )?;
                let assets = local_assets::ensure_from_snapshot(&workspace, &assets, &snapshot, &progress)?;
                let rules = splits::project_rules(&assets.join("rules"), &project)?;
                workflow::validate_owners(&project, &rules)?;
                let map = map::region_polygons(&snapshot, &project)?;
                Ok((map, project, assets, rules))
            })();
            let _ = tx.send(match result {
                Ok((map, project, assets, rules)) => Event::MapReady(Box::new(map), Box::new(project), assets, Box::new(rules)),
                Err(error) => Event::Error(format!("{error:#}")),
            });
        });
    }

    fn verify_sources(&mut self) {
        self.busy = true;
        self.error = None;
        let root = self.workspace.join("sources");
        let tx = self.tx.clone();
        thread::spawn(move || {
            let progress = |message: &str| {
                let _ = tx.send(Event::Progress(message.into()));
            };
            let result = Snapshot::open(&root).and_then(|snapshot| snapshot.verify(&progress));
            let _ = tx.send(match result {
                Ok(()) => Event::Verified,
                Err(error) => Event::Error(format!("{error:#}")),
            });
        });
    }

    fn save(&mut self) -> bool {
        match self.project.save(&self.workspace.join("project.json")) {
            Ok(()) => {
                self.dirty = false;
                self.push_message("Project saved. Original game data is unchanged.".into());
                true
            }
            Err(error) => {
                self.error = Some(format!("Save project: {error:#}"));
                false
            }
        }
    }

    fn reload(&mut self) {
        let path = self.workspace.join("project.json");
        let result = if path.exists() {
            Project::load(&path)
        } else {
            Ok(Project::default())
        };
        match result {
            Ok(project) => match self.install_project(project) {
                Ok(()) => {
                    self.leave_split_tool();
                    self.leave_activate_tool();
                    self.confirm_remove_split = None;
                    self.confirm_remove_activation = None;
                    self.dirty = false;
                    self.ui.analysis_wanted = self.source_ready;
                    self.refresh_factions();
                    self.push_message("Reloaded the saved project (ownership, wilderness activations and region splits).".into());
                }
                Err(error) => self.error = Some(format!("Reload project: {error:#}")),
            },
            Err(error) => self.error = Some(format!("Reload project: {error:#}")),
        }
    }

    fn generate(&mut self) {
        let Some(assets) = self.derived_assets.clone() else {
            self.error = Some("Import original games before exporting.".into());
            return;
        };
        if let Some(reason) = self.export_blocker() {
            self.error = Some(format!("Export unavailable: {reason}"));
            return;
        }
        if !self.save() {
            return;
        }
        self.busy = true;
        self.error = None;
        let workspace = self.workspace.clone();
        let project = self.project.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let progress = |message: &str| {
                let _ = tx.send(Event::Progress(message.into()));
            };
            let result = pipeline::generate(&workspace, &assets, &project, &progress);
            let _ = tx.send(match result {
                Ok(report) => Event::Built(report),
                Err(error) => Event::Error(format!("{error:#}")),
            });
        });
    }

    fn receive(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                Event::Progress(message) => self.push_message(message),
                Event::Error(error) => {
                    self.busy = false;
                    self.status =
                        "Operation failed; previous generated output was preserved.".into();
                    self.error = Some(error);
                }
                Event::Requirements(requirements) => {
                    self.busy = false;
                    self.ui.set_requirements(requirements);
                }
                Event::Datasets(listing, info) => {
                    self.busy = false;
                    self.ui.set_datasets(listing, info);
                }
                Event::Preview(spec, preview) => {
                    self.busy = false;
                    let open = preview.problems.len() + preview.missing.len();
                    self.ui.set_preview(spec, preview);
                    self.push_message(format!(
                        "TSV scan complete: {open} problem(s) or missing table(s) to resolve before import."
                    ));
                }
                Event::TargetImported(binding) => {
                    self.busy = false;
                    let id = binding.id.clone();
                    workflow::pin_target(&mut self.project, binding);
                    self.dirty = true;
                    self.ui.clear_analysis();
                    self.ui.dataset_imported();
                    self.ui.analysis_wanted = true;
                    self.push_message(format!(
                        "Imported immutable TSV dataset {id} and pinned it to this project. Save to keep the pin."
                    ));
                }
                Event::TargetChecked(info, changes) => {
                    self.busy = false;
                    let id = info.binding.id.clone();
                    self.push_message(if changes.is_empty() {
                        format!("Dataset {id}: stored copy verified, the TSV folder is unchanged.")
                    } else {
                        format!(
                            "Dataset {id}: {} difference(s) from the TSV folder found.",
                            changes.len()
                        )
                    });
                    self.ui.set_checked(info, changes);
                }
                Event::AdapterImported(adapter) => {
                    self.busy = false;
                    let id = adapter.id.clone();
                    let mut panel = PanelCtx {
                        project: &mut self.project,
                        dirty: &mut self.dirty,
                        busy: false,
                        source_ready: self.source_ready,
                    };
                    self.ui.adapter_imported(&mut panel, adapter);
                    self.push_message(format!(
                        "Imported adapter {id}; re-run analysis to check it."
                    ));
                }
                Event::Analyzed(analysis, signature) => {
                    self.ui.analysis_running = false;
                    let key = workflow::catalog_key(&self.project);
                    if self.running_key.take().as_deref() != Some(key.as_str()) {
                        // Started before the target or adapter changed: its catalog and
                        // verdicts describe a different target, so it must neither show nor offer owners.
                        self.ui.analysis_wanted = self.source_ready;
                        self.push_message(
                            "Discarded an analysis of the previous target settings; analyzing again."
                                .into(),
                        );
                        continue;
                    }
                    if signature != workflow::signature(&self.project) {
                        self.ui.analysis_wanted = self.source_ready;
                    }
                    let blockers = workflow::blocker_count(&analysis);
                    let waiting = analysis.unassigned_regions.len();
                    let message = format!(
                        "Analysis finished for {}: {blockers} blocking issue(s), {waiting} region(s) awaiting an owner, {} warning(s). Gameplay not verified.",
                        analysis.target_name,
                        analysis.warnings.len()
                    );
                    self.ui.set_analysis(*analysis, signature);
                    self.analysis_key = Some(key);
                    self.refresh_factions();
                    self.mesh_dirty = true;
                    self.push_message(message);
                }
                Event::AnalysisFailed(error) => {
                    self.ui.analysis_running = false;
                    self.ui.analysis_wanted = false;
                    self.status = "Analysis failed.".into();
                    self.error = Some(error);
                }
                Event::Verified => {
                    self.busy = false;
                    self.source_ready = true;
                    self.ui.analysis_wanted = true;
                    self.push_message("Original snapshot passed every manifest hash check.".into());
                }
                Event::Built(report) => {
                    self.busy = false;
                    self.output = Some(report.output.clone());
                    self.push_message(format!(
                        "Generated {} regions and {} factions; {} ownership overrides. {}",
                        report.region_count,
                        report.faction_count,
                        report.ownership_changes,
                        report.output.display()
                    ));
                    for warning in report.warnings {
                        self.messages.push_back(warning);
                    }
                }
                Event::MapReady(map, loaded, assets, rules) => {
                    let rows = match rows_of(&rules, &loaded) {
                        Ok(rows) => rows,
                        Err(error) => {
                            self.busy = false;
                            self.error = Some(format!("{error:#}"));
                            continue;
                        }
                    };
                    self.derived_assets = Some(assets);
                    self.rules = Some(*rules);
                    self.rows = rows;
                    self.busy = false;
                    self.source_ready = true;
                    let RegionMap {
                        base,
                        theatres,
                        data,
                        polygons,
                    } = *map;
                    self.base_regions = Some(base);
                    self.theatres = theatres;
                    self.geometry = Some(Geometry {
                        data,
                        raw: polygons,
                    });
                    self.rebuild_polygons();
                    if loaded.activations != self.project.activations || loaded.splits != self.project.splits {
                        // The project's activations or splits changed while the map was loading.
                        if let Err(error) = self.install_project(self.project.clone()) {
                            self.error = Some(format!("{error:#}"));
                        }
                    }
                    self.fit_map();
                    self.ui.sources_changed();
                    self.ui.analysis_wanted = true;
                    self.refresh_factions();
                    self.refresh_filter();
                    self.push_message(format!(
                        "Verified originals loaded. {} editable region polygons.",
                        self.polygons.len()
                    ));
                }
            }
        }
    }

    fn owner<'a>(&'a self, key: &'a str) -> &'a str {
        self.rules.as_ref().map_or("", |rules| workflow::effective_owner(&self.project, rules, key))
    }

    /// The stored analysis was made for the current target and adapter, so its catalog is valid.
    fn catalog_current(&self) -> bool {
        self.ui.analysis.is_some()
            && self.analysis_key.as_deref() == Some(workflow::catalog_key(&self.project).as_str())
    }

    /// Whether the region still awaits a target owner according to the stored analysis (only trusted while
    /// `pending_valid`, i.e. that analysis belongs to the current target settings). No allocation.
    fn is_pending(&self, key: &str) -> bool {
        self.pending_valid
            && self
                .ui
                .analysis
                .as_ref()
                .is_some_and(|analysis| workflow::is_pending(&self.project, analysis, key))
    }

    fn pending_total(&self) -> usize {
        match self.ui.analysis.as_ref() {
            Some(analysis) if self.pending_valid => {
                workflow::pending_count(&self.project, analysis)
            }
            _ => 0,
        }
    }

    /// Selects every pending region, ready for one bulk assignment. Choices are never guessed: the faction is
    /// left for the user to pick.
    fn select_pending(&mut self) {
        self.selected = self
            .rows
            .iter()
            .filter(|row| self.is_pending(&row.key))
            .map(|row| row.key.clone())
            .collect();
        self.assign_to.clear();
        self.mesh_dirty = true;
    }

    /// A target catalog is loaded and the region's owner is not one of its factions.
    fn needs_owner(&self, key: &str) -> bool {
        let owner = self.owner(key);
        !self.factions.is_empty()
            && self
                .factions
                .binary_search_by(|faction| faction.as_str().cmp(owner))
                .is_err()
    }

    fn refresh_factions(&mut self) {
        let analysis = if self.catalog_current() {
            self.ui.analysis.as_ref()
        } else {
            None
        };
        self.factions = workflow::owner_choices(analysis);
        if !self.factions.contains(&self.assign_to) {
            // Never preselect a guess: the user picks an existing target faction.
            self.assign_to.clear();
        }
    }

    fn export_blocker(&self) -> Option<String> {
        workflow::export_blocker(
            &self.project,
            self.ui.analysis.as_ref(),
            self.ui.analyzed_signature.as_deref(),
        )
    }

    fn pump_analysis(&mut self) {
        if self.ui.analysis_wanted && self.source_ready && !self.busy && !self.ui.analysis_running {
            self.start_analysis();
        }
    }

    fn start_analysis(&mut self) {
        let Some(assets) = self.derived_assets.clone() else {
            self.error = Some("Import original games before analysis.".into());
            return;
        };
        self.ui.analysis_wanted = false;
        self.ui.analysis_running = true;
        self.error = None;
        let workspace = self.workspace.clone();
        let project = self.project.clone();
        let signature = workflow::signature(&project);
        self.running_key = Some(workflow::catalog_key(&project));
        let tx = self.tx.clone();
        thread::spawn(move || {
            let progress = |message: &str| {
                let _ = tx.send(Event::Progress(message.into()));
            };
            let result = compatibility::analyze(&workspace, &assets, &project, &progress);
            let _ = tx.send(match result {
                Ok(analysis) => Event::Analyzed(Box::new(analysis), signature),
                Err(error) => Event::AnalysisFailed(format!("{error:#}")),
            });
        });
    }

    /// Runs one exclusive background job; its event releases `busy`.
    fn spawn_job<F>(&mut self, job: F)
    where
        F: FnOnce(&Path, &Progress<'_>) -> Result<Event> + Send + 'static,
    {
        self.busy = true;
        self.error = None;
        let workspace = self.workspace.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let progress = |message: &str| {
                let _ = tx.send(Event::Progress(message.into()));
            };
            let event = job(&workspace, &progress)
                .unwrap_or_else(|error| Event::Error(format!("{error:#}")));
            let _ = tx.send(event);
        });
    }

    fn run_requests(&mut self, requests: Vec<Request>) {
        for request in requests {
            match request {
                Request::LoadRequirements => {
                    let Some(assets) = self.derived_assets.clone() else { continue };
                    self.spawn_job(move |workspace, _| {
                        let base = Snapshot::open(&workspace.join("sources"))?;
                        let rules = Rules::load(&assets.join("rules"))?;
                        Ok(Event::Requirements(target::required_inputs(&base, &rules)?))
                    })
                }
                Request::RefreshDatasets(pinned) => self.spawn_job(move |workspace, _| {
                    let listing = target::list(workspace)?;
                    let info = pinned.map(|binding| {
                        target::load(workspace, &binding)
                            .map(|dataset| DatasetInfo::of(&dataset))
                            .map_err(|error| format!("{error:#}"))
                    });
                    Ok(Event::Datasets(listing, info))
                }),
                Request::Preview(spec) => {
                    let Some(assets) = self.derived_assets.clone() else { continue };
                    self.spawn_job(move |workspace, progress| {
                        let preview = target::preview(workspace, &assets, &spec, progress)?;
                        Ok(Event::Preview(spec, preview))
                    })
                }
                Request::Import(spec) => {
                    let Some(assets) = self.derived_assets.clone() else { continue };
                    self.spawn_job(move |workspace, progress| {
                        Ok(Event::TargetImported(target::import(
                            workspace, &assets, &spec, progress,
                        )?))
                    })
                }
                Request::Check(binding) => self.spawn_job(move |workspace, progress| {
                    let dataset = target::load(workspace, &binding)?;
                    dataset.verify(progress)?;
                    let changes = dataset.changes()?;
                    Ok(Event::TargetChecked(DatasetInfo::of(&dataset), changes))
                }),
                Request::ImportAdapter(binding, folder) => {
                    self.spawn_job(move |workspace, progress| {
                        Ok(Event::AdapterImported(target::import_adapter(
                            workspace, &binding, &folder, progress,
                        )?))
                    })
                }
                Request::Analyze => {
                    if !self.busy && !self.ui.analysis_running {
                        self.start_analysis();
                    }
                }
                Request::Export => self.generate(),
            }
        }
    }

    fn refresh_filter(&mut self) {
        let query = self.search.to_lowercase();
        self.filtered = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                (self.theatre.is_empty() || row.theatre == self.theatre)
                    && (query.is_empty()
                        || row.key.contains(&query)
                        || row.name.to_lowercase().contains(&query)
                        || self.owner(&row.key).contains(&query))
            })
            .map(|(index, _)| index)
            .collect();
    }

    fn select(&mut self, key: String, multiple: bool) {
        if multiple {
            if !self.selected.remove(&key) {
                self.selected.insert(key);
            }
        } else {
            let owner = self.owner(&key);
            self.assign_to = if self
                .factions
                .binary_search_by(|faction| faction.as_str().cmp(owner))
                .is_err()
            {
                String::new()
            } else {
                owner.to_string()
            };
            self.selected.clear();
            self.selected.insert(key);
        }
        self.mesh_dirty = true;
    }

    /// Editable polygons: every triangulated region the rules know plus the wilderness land polygons that can be
    /// activated (sea regions and other rule-less land have none).
    fn rebuild_polygons(&mut self) {
        let Some(rules) = &self.rules else { return };
        self.wild = self
            .geometry
            .iter()
            .flat_map(|geometry| {
                geometry
                    .data
                    .regions
                    .iter()
                    .filter(|region| region.kind == "land" && rules.regions.get(&region.key).is_none())
                    .filter_map(|region| {
                        splits::activatable(&geometry.data, &self.theatres, rules, &region.key)
                            .ok()
                            .map(|found| (region.key.clone(), found))
                    })
            })
            .collect();
        self.polygons = self
            .geometry
            .iter()
            .flat_map(|geometry| &geometry.raw)
            .filter_map(|polygon| {
                let wild = self.wild.contains_key(&polygon.key);
                (wild || rules.regions.get(&polygon.key).is_some()).then(|| polygon_of(polygon, wild))
            })
            .collect();
        self.overlay = self
            .geometry
            .as_ref()
            .map(|geometry| Overlay::build(&geometry.data, rules))
            .unwrap_or_default();
        self.mesh_dirty = true;
    }

    /// Makes `project` current: derives the rules of its activated and split regions, replays its activations and
    /// splits on the original geometry and rebuilds the region list and polygons. Everything fallible runs first, so an
    /// error leaves the editor exactly as it was.
    fn install_project(&mut self, project: Project) -> Result<()> {
        let Some(assets) = &self.derived_assets else {
            self.project = project;
            return Ok(());
        };
        let rules = splits::project_rules(&assets.join("rules"), &project)?;
        workflow::validate_owners(&project, &rules)?;
        let rows = rows_of(&rules, &project)?;
        let (geometry, theatres) = match &self.base_regions {
            Some(base) => {
                let (data, theatres) = splits::replay(base, &project)?;
                let previous = self
                    .geometry
                    .as_ref()
                    .map(|geometry| (&geometry.data, geometry.raw.as_slice()));
                let raw = map::polygons_of(&data, previous)?;
                (Some(Geometry { data, raw }), Some(theatres))
            }
            None => (None, None),
        };
        self.selected.retain(|key| rules.regions.get(key).is_some());
        self.rules = Some(rules);
        self.rows = rows;
        self.project = project;
        self.geometry = geometry;
        if let Some(theatres) = theatres {
            self.theatres = theatres;
        }
        self.rebuild_polygons();
        self.refresh_filter();
        Ok(())
    }

    fn apply_assignment(&mut self) {
        let Some(rules) = &self.rules else { return };
        let analysis = if self.catalog_current() {
            self.ui.analysis.as_ref()
        } else {
            None
        };
        // All-or-nothing: a rejected region leaves the project exactly as it was.
        let mut next = self.project.clone();
        for key in &self.selected {
            if let Err(error) =
                workflow::assign_owner(&mut next, rules, analysis, key, &self.assign_to)
            {
                self.error = Some(format!("{error:#}"));
                return;
            }
        }
        self.project = next;
        self.dirty = true;
        self.ui.analysis_wanted = self.source_ready;
        self.mesh_dirty = true;
        self.refresh_filter();
        self.push_message(format!(
            "Assigned {} selected region(s) to {}. Save or export to persist.",
            self.selected.len(),
            label(&self.assign_to)
        ));
    }

    fn fit_map(&mut self) {
        if !self.polygons.is_empty() {
            self.bounds = [
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ];
            for polygon in &self.polygons {
                if !polygon.triangles.is_empty() {
                    for i in 0..2 {
                        self.bounds[i] = self.bounds[i].min(polygon.bounds[i]);
                        self.bounds[i + 2] = self.bounds[i + 2].max(polygon.bounds[i + 2]);
                    }
                }
            }
        }
        for theatre in &self.theatres {
            if self.bounds[0].is_finite() {
                self.bounds[0] = self.bounds[0].min(theatre.lo[0]);
                self.bounds[1] = self.bounds[1].min(theatre.lo[1]);
                self.bounds[2] = self.bounds[2].max(theatre.hi[0]);
                self.bounds[3] = self.bounds[3].max(theatre.hi[1]);
            }
        }
        self.center = Vec2::new(
            (self.bounds[0] + self.bounds[2]) * 0.5,
            (self.bounds[1] + self.bounds[3]) * 0.5,
        );
        self.zoom = 1.0;
        self.mesh_dirty = true;
    }

    fn scale(&self, rect: Rect) -> f32 {
        ((rect.width() - 24.0) / (self.bounds[2] - self.bounds[0]).max(1.0))
            .min((rect.height() - 24.0) / (self.bounds[3] - self.bounds[1]).max(1.0))
            * self.zoom
    }

    fn to_world(&self, point: Pos2, rect: Rect) -> [f32; 2] {
        let scale = self.scale(rect);
        [
            (point.x - rect.center().x) / scale + self.center.x,
            self.center.y - (point.y - rect.center().y) / scale,
        ]
    }

    /// Pixels (points) per tile of the wilderness hatch texture.
    const HATCH_TILE: f32 = 12.0;

    fn hatch_texture(ctx: &egui::Context) -> egui::TextureHandle {
        let size = Self::HATCH_TILE as usize;
        let pixels = (0..size * size)
            .map(|i| {
                let (x, y) = (i % size, i / size);
                if (x + y) % size < 2 {
                    Color32::WHITE
                } else {
                    Color32::TRANSPARENT
                }
            })
            .collect();
        ctx.load_texture(
            "wilderness-hatch",
            egui::ColorImage {
                size: [size, size],
                pixels,
            },
            egui::TextureOptions {
                wrap_mode: egui::TextureWrapMode::Repeat,
                ..egui::TextureOptions::NEAREST
            },
        )
    }

    fn redraw_mesh(&mut self, rect: Rect) {
        let count = self.polygons.iter().map(|p| p.triangles.len() * 3).sum();
        let mut mesh = egui::Mesh::default();
        mesh.vertices.reserve(count);
        mesh.indices.reserve(count);
        let mut hatch = self.hatch.as_ref().map(|texture| egui::Mesh::with_texture(texture.id()));
        let scale = self.scale(rect);
        // Pending state is read from the stored analysis per polygon; nothing is cloned for the render pass.
        for polygon in &self.polygons {
            let picked = self.activation.key.as_deref() == Some(polygon.key.as_str());
            let color = if polygon.wild {
                if picked {
                    Color32::from_rgb(250, 213, 112)
                } else {
                    Color32::from_rgb(66, 72, 68)
                }
            } else if self.selected.contains(&polygon.key) {
                Color32::from_rgb(250, 213, 112)
            } else if self.is_pending(&polygon.key) {
                Color32::from_rgb(235, 150, 40)
            } else if self.needs_owner(&polygon.key) {
                Color32::from_rgb(220, 40, 40)
            } else {
                faction_color(self.owner(&polygon.key))
            };
            let lines = if picked {
                Color32::from_rgba_unmultiplied(70, 50, 10, 160)
            } else {
                Color32::from_rgba_unmultiplied(165, 175, 165, 120)
            };
            for triangle in &polygon.triangles {
                let base = mesh.vertices.len() as u32;
                let points = triangle.map(|point| {
                    Pos2::new(
                        rect.center().x + (point[0] - self.center.x) * scale,
                        rect.center().y - (point[1] - self.center.y) * scale,
                    )
                });
                for pos in points {
                    mesh.vertices.push(egui::epaint::Vertex {
                        pos,
                        uv: egui::epaint::WHITE_UV,
                        color,
                    });
                }
                mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
                if let (true, Some(hatch)) = (polygon.wild, hatch.as_mut()) {
                    let base = hatch.vertices.len() as u32;
                    for pos in points {
                        hatch.vertices.push(egui::epaint::Vertex {
                            pos,
                            uv: Pos2::new(
                                (pos.x - rect.min.x) / Self::HATCH_TILE,
                                (pos.y - rect.min.y) / Self::HATCH_TILE,
                            ),
                            color: lines,
                        });
                    }
                    hatch.indices.extend_from_slice(&[base, base + 1, base + 2]);
                }
            }
        }
        self.mesh = Some(Arc::new(mesh));
        self.wild_mesh = hatch.map(Arc::new);
        self.mesh_dirty = false;
        self.last_rect = rect;
    }

    fn map_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("World ownership");
            ui.separator();
            ui.label(match self.tool {
                MapTool::Split => "Split tool · Click a region, then its border to start a cut · Esc leaves · Scroll to zoom · Right-drag to pan",
                MapTool::Activate => "Activate tool · Click a hatched wilderness polygon · Esc leaves · Scroll to zoom · Right-drag to pan",
                MapTool::Select => "Click to select · Ctrl-click multi-select · Scroll to zoom · Right-drag to pan",
            });
            if ui.button("Fit map").clicked() {
                self.fit_map();
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.strong(workflow::OWNERSHIP_LABEL);
            ui.small(workflow::OWNERSHIP_HELP);
        });
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 5.0, Color32::from_rgb(18, 34, 46));
        if self.polygons.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Import original ETW files to display the region map",
                egui::TextStyle::Heading.resolve(ui.style()),
                Color32::LIGHT_GRAY,
            );
            return;
        }
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = ui.input(|i| i.pointer.delta());
            let scale = self.scale(rect);
            self.center.x -= delta.x / scale;
            self.center.y += delta.y / scale;
            self.mesh_dirty = true;
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.01 {
                let point = response.hover_pos().unwrap_or(rect.center());
                let before = self.to_world(point, rect);
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.6, 16.0);
                let after = self.to_world(point, rect);
                self.center.x += before[0] - after[0];
                self.center.y += before[1] - after[1];
                self.mesh_dirty = true;
            }
        }
        if response.clicked_by(egui::PointerButton::Primary) {
            if let Some(point) = response.interact_pointer_pos() {
                let world = self.to_world(point, rect);
                match self.tool {
                    MapTool::Split => self.split_click(world, rect),
                    MapTool::Activate => self.activate_click(world),
                    MapTool::Select => {
                        if let Some(key) = self.region_at(world) {
                            self.select(key, ui.input(|i| i.modifiers.ctrl));
                        } else if let Some(key) = self.wild_at(world) {
                            self.push_message(format!(
                                "{} is wilderness without campaign data; use 'Activate wilderness region' to make it playable.",
                                splits::default_name(&key)
                            ));
                        }
                    }
                }
            }
        }
        if self.hatch.is_none() {
            self.hatch = Some(Self::hatch_texture(ui.ctx()));
            self.mesh_dirty = true;
        }
        if self.mesh_dirty || self.last_rect != rect {
            self.redraw_mesh(rect);
        }
        if let Some(mesh) = &self.mesh {
            painter.add(egui::Shape::Mesh(mesh.clone()));
        }
        if let Some(mesh) = &self.wild_mesh {
            painter.add(egui::Shape::Mesh(mesh.clone()));
        }
        if self.show_overlay {
            self.draw_overlay(&painter, rect);
        }
        if self.tool == MapTool::Split {
            self.draw_split_overlay(&painter, rect, response.hover_pos(), ui);
        }
        self.draw_wild_labels(&painter, rect);
        if self.tool == MapTool::Activate {
            self.draw_activate_overlay(&painter, rect, ui);
        }
        painter.text(
            rect.left_bottom() + Vec2::new(12.0, -12.0),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "{} regions · {} ownership overrides · {:.1}×",
                self.rows.len(),
                self.project.owners.len(),
                self.zoom
            ),
            egui::TextStyle::Body.resolve(ui.style()),
            Color32::LIGHT_GRAY,
        );
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        ui.heading("Regions");
        let mut changed = ui
            .add(egui::TextEdit::singleline(&mut self.search).hint_text("Search region or owner"))
            .changed();
        egui::ComboBox::from_id_salt("theatre-filter")
            .selected_text(if self.theatre.is_empty() {
                "All theatres".into()
            } else {
                label(&self.theatre)
            })
            .show_ui(ui, |ui| {
                for (key, title) in [
                    ("", "All theatres"),
                    ("america", "America"),
                    ("europe", "Europe"),
                    ("india", "India"),
                ] {
                    changed |= ui
                        .selectable_value(&mut self.theatre, key.into(), title)
                        .changed();
                }
            });
        if changed {
            self.refresh_filter();
        }
        ui.label(format!(
            "{} shown / {} total",
            self.filtered.len(),
            self.rows.len()
        ));
        ui.checkbox(&mut self.show_overlay, "Show borders & towns");
        self.split_toggle(ui);
        self.activate_toggle(ui);
        ui.label("Theatre bounds are fixed").on_hover_ui(|ui| {
            for (key, rect) in crate::regionmap::theatres::FIXED_BOUNDS {
                let name = match *key {
                    "1" => "America — Western Plains",
                    "2" => "Europe — original",
                    "3" => "India — Thailand and Indonesia",
                    _ => key,
                };
                ui.label(format!("{name}: {rect:?}"));
            }
        });
        let list_height = if self.tool != MapTool::Select {
            (ui.available_height() * 0.3).max(120.0)
        } else {
            (ui.available_height() - 360.0).max(140.0)
        };
        // Pending state is read from the stored analysis per region; nothing is cloned for the render pass.
        egui::ScrollArea::vertical()
            .id_salt("region-list")
            .max_height(list_height)
            .show_rows(ui, 42.0, self.filtered.len(), |ui, range| {
                for index in range {
                    let row = &self.rows[self.filtered[index]];
                    let key = row.key.clone();
                    let owner = self.owner(&key).to_string();
                    let selected = self.selected.contains(&key);
                    let missing = self.needs_owner(&key);
                    let awaiting = self.is_pending(&key);
                    let text = format!(
                        "{}{}{}\n{}",
                        if self.project.owners.contains_key(&key) {
                            "* "
                        } else {
                            ""
                        },
                        row.name,
                        if row.split {
                            " (split)"
                        } else if row.activated {
                            " (activated)"
                        } else {
                            ""
                        },
                        if awaiting {
                            format!(
                                "{} · original owner unavailable in target — awaiting an existing faction",
                                label(&owner)
                            )
                        } else if missing {
                            format!(
                                "{} · not in target database — assign an existing faction",
                                label(&owner)
                            )
                        } else {
                            label(&owner)
                        }
                    );
                    let response = ui.add_sized(
                        [ui.available_width(), 42.0],
                        egui::SelectableLabel::new(
                            selected,
                            egui::RichText::new(text).color(if selected {
                                Color32::from_rgb(250, 213, 112)
                            } else if awaiting {
                                Color32::from_rgb(235, 150, 40)
                            } else if missing {
                                Color32::LIGHT_RED
                            } else {
                                Color32::WHITE
                            }),
                        ),
                    );
                    if response.clicked() {
                        self.list_click(key, ui.input(|i| i.modifiers.ctrl));
                    }
                }
            });
        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("sidebar-lower")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                match self.tool {
                    MapTool::Split => self.split_panel(ui),
                    MapTool::Activate => self.activate_panel(ui),
                    MapTool::Select => self.assign_panel(ui),
                }
                self.activations_list(ui);
                self.splits_list(ui);
            });
    }

    fn assign_panel(&mut self, ui: &mut egui::Ui) {
        ui.strong(format!("{} selected", self.selected.len()));
        if self.selected.len() == 1 {
            if let Some(key) = self.selected.first() {
                ui.monospace(key);
                ui.label(format!("Owner: {}", label(self.owner(key))));
            }
        }
        if self.factions.is_empty() {
            ui.colored_label(
                Color32::from_rgb(250, 213, 112),
                "Target faction catalog not loaded yet: owner choices appear once analysis has read the target database (Export review → Analyze). Nothing is guessed.",
            );
        } else {
            let waiting = self.pending_total();
            let invalid: Vec<String> = self
                .rows
                .iter()
                .filter(|row| self.needs_owner(&row.key) && !self.is_pending(&row.key))
                .map(|row| row.key.clone())
                .collect();
            if waiting > 0 {
                ui.colored_label(
                    Color32::from_rgb(250, 213, 112),
                    format!(
                        "{} region(s) await a target owner: their original owner is unavailable in the imported target. Pick an existing target faction below; remaining compatibility checks and export follow after the next analysis.",
                        waiting
                    ),
                );
                if ui.button("Select regions awaiting an owner").clicked() {
                    self.select_pending();
                }
            }
            if !invalid.is_empty() {
                ui.colored_label(
                    Color32::LIGHT_RED,
                    format!(
                        "{} region(s) have an owner that is not in the target database; export stays blocked until each is assigned an existing faction.",
                        invalid.len()
                    ),
                );
                if ui.button("Select regions with an invalid owner").clicked() {
                    self.selected = invalid.into_iter().collect();
                    self.mesh_dirty = true;
                }
            }
        }
        ui.label("New owner (existing factions of the target database)");
        search_select(
            ui,
            "assign-faction",
            &mut self.assign_to,
            &self.factions,
            "choose faction…",
        );
        if ui
            .add_enabled(
                !self.selected.is_empty() && !self.busy && self.factions.contains(&self.assign_to),
                egui::Button::new("Assign faction to selected")
                    .min_size(Vec2::new(ui.available_width(), 30.0)),
            )
            .clicked()
        {
            self.apply_assignment();
        }
        if ui
            .add_enabled(
                !self.selected.is_empty() && !self.busy,
                egui::Button::new("Restore selected to 1805 owners"),
            )
            .clicked()
        {
            for key in &self.selected {
                self.project.owners.remove(key);
            }
            self.dirty = true;
            self.ui.analysis_wanted = self.source_ready;
            self.mesh_dirty = true;
            self.refresh_filter();
        }
        ui.small("Ownership changes are used when generating a new campaign. Original ETW files and existing saves are never edited.");
        ui.small(workflow::OWNERSHIP_HELP);
    }
}

impl eframe::App for Editor {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive();
        self.pending_valid = self.catalog_current();
        if self.source_ready {
            self.split_keys(ctx);
            self.activate_keys(ctx);
        }
        self.pump_analysis();
        if let Some(request) = self.ui.pending(&self.project, self.busy, self.source_ready) {
            self.run_requests(vec![request]);
        }
        if self.busy || self.ui.analysis_running {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if self.ui.tab() == Tab::Tables {
            for file in ctx.input(|input| input.raw.dropped_files.clone()) {
                if let Some(path) = file.path {
                    self.ui.dropped_folder(path);
                }
            }
        }
        let export_blocker = self.export_blocker();
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("KGC Port Editor");
                ui.separator();
                ui.label("ETW originals → Napoleon world campaign");
                if self.dirty {
                    ui.colored_label(Color32::from_rgb(250, 213, 112), "Unsaved changes");
                }
                if self.busy || self.ui.analysis_running {
                    ui.spinner();
                }
            });
            ui.horizontal(|ui| {
                ui.strong(format!("Target: {}", self.ui.target_line(&self.project)));
                let (status, color) = self.ui.status_line(&self.project);
                ui.colored_label(color, status);
                ui.colored_label(
                    Color32::from_rgb(250, 213, 112),
                    "Gameplay not verified in-game",
                );
            });
            ui.horizontal(|ui| {
                let open = self
                    .ui
                    .analysis
                    .as_ref()
                    .map_or(0, workflow::blocker_count);
                let waiting = self.pending_total();
                let mut tab = self.ui.tab();
                for candidate in [Tab::Tables, Tab::Mappings, Tab::Review, Tab::Map, Tab::Export] {
                    let text = if candidate == Tab::Review && open > 0 {
                        format!("{} ({open})", candidate.title())
                    } else if candidate == Tab::Map && waiting > 0 {
                        format!("{} ({waiting} awaiting owner)", candidate.title())
                    } else {
                        candidate.title().to_string()
                    };
                    ui.selectable_value(&mut tab, candidate, text);
                }
                self.ui.tab = Some(tab);
            });
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.busy, egui::Button::new("Save project"))
                    .clicked()
                {
                    self.save();
                }
                if ui
                    .add_enabled(!self.busy, egui::Button::new("Reload project"))
                    .clicked()
                {
                    if self.dirty {
                        self.confirm_reload = true;
                    } else {
                        self.reload();
                    }
                }
                if ui
                    .add_enabled(
                        self.source_ready && !self.busy && !self.project.owners.is_empty(),
                        egui::Button::new("Reset all ownership"),
                    )
                    .clicked()
                {
                    self.confirm_reset = true;
                }
                ui.separator();
                let export = ui.add_enabled(
                    self.source_ready
                        && !self.busy
                        && !self.ui.analysis_running
                        && export_blocker.is_none(),
                    egui::Button::new("Save & Export ZIP").fill(Color32::from_rgb(61, 92, 73)),
                );
                let export = match &export_blocker {
                    Some(reason) => export.on_disabled_hover_text(reason),
                    None => export,
                };
                if export.clicked() {
                    self.generate();
                }
                if ui
                    .add_enabled(
                        self.source_ready && !self.busy,
                        egui::Button::new("Verify originals"),
                    )
                    .clicked()
                {
                    self.verify_sources();
                }
                ui.label(if self.source_ready {
                    "Original snapshot ready"
                } else {
                    "Original snapshot required"
                });
                ui.label("ZIP only; no game installation writes.");
            });
            if let Some(reason) = &export_blocker {
                ui.colored_label(
                    if workflow::only_pending_ownership(
                        &self.project,
                        self.ui.analysis.as_ref(),
                        self.ui.analyzed_signature.as_deref(),
                    ) {
                        Color32::from_rgb(250, 213, 112)
                    } else {
                        Color32::LIGHT_RED
                    },
                    format!("Export unavailable: {reason}"),
                );
            }
            egui::CollapsingHeader::new("Original game sources / workspace")
                .default_open(!self.source_ready)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Empire installation");
                        ui.add_enabled(
                            !self.busy,
                            egui::TextEdit::singleline(&mut self.etw_data).desired_width(650.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Napoleon installation");
                        ui.add_enabled(
                            !self.busy,
                            egui::TextEdit::singleline(&mut self.ntw_data).desired_width(650.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("NTW base language (blank = auto-detect)");
                        ui.add_enabled(
                            !self.busy,
                            egui::TextEdit::singleline(&mut self.base_language)
                                .hint_text("auto (e.g. en, fr; needed only if several local_<xx>.pack exist)")
                                .desired_width(420.0),
                        );
                    });
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(!self.busy, egui::Button::new("Import originals & rebuild resources"))
                            .clicked()
                        {
                            self.import_sources();
                        }
                        ui.label("Select each game folder or its data folder. Originals are read-only; resources are rebuilt in your workspace.");
                    });
                    ui.monospace(format!("Workspace: {}", self.workspace.display()));
                });
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.label(&self.status);
            if let Some(output) = &self.output {
                ui.monospace(format!("Export ZIP: {}", output.display()));
            }
            if let Some(error) = &self.error {
                ui.colored_label(Color32::LIGHT_RED, error);
            }
            egui::CollapsingHeader::new("Conversion log").show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(130.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for message in &self.messages {
                            ui.monospace(message);
                        }
                    });
            });
        });
        if !self.source_ready {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Rebuild from your own installed games");
                ui.label("Empire supplies the world, geography and campaign data. Napoleon supplies its campaign and interface templates.");
                ui.label("Enter both installation paths above, then import. No replacement game assets are distributed with the editor.");
            });
            return;
        }
        let tab = self.ui.tab();
        if tab != self.shown_tab {
            self.shown_tab = tab;
            if tab == Tab::Map {
                // Mapping and dataset edits change resolved owners and selectable factions.
                self.refresh_factions();
                self.mesh_dirty = true;
                self.refresh_filter();
            }
        }
        if let Some(focus) = self.ui.focus_owner.take() {
            match focus {
                OwnerFocus::Region(key) => {
                    if self.rows.iter().any(|row| row.key == key) {
                        self.select(key, false);
                    }
                }
                OwnerFocus::Pending => self.select_pending(),
            }
        }
        if tab == Tab::Map {
            egui::SidePanel::left("regions")
                .default_width(300.0)
                .min_width(265.0)
                .resizable(true)
                .show(ctx, |ui| self.sidebar(ui));
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut panel = PanelCtx {
                project: &mut self.project,
                dirty: &mut self.dirty,
                busy: self.busy,
                source_ready: self.source_ready,
            };
            let requests = match tab {
                Tab::Map => {
                    self.map_panel(ui);
                    Vec::new()
                }
                Tab::Tables => self.ui.tables_panel(ui, &mut panel),
                Tab::Mappings => self.ui.mapping_panel(ui, &mut panel),
                Tab::Review => self.ui.review_panel(ui, &mut panel),
                Tab::Export => self.ui.export_panel(ui, &mut panel),
            };
            self.run_requests(requests);
        });
        if self.confirm_reload {
            egui::Window::new("Discard unsaved ownership edits?")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Reloading replaces the current edits with your saved project.");
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.confirm_reload = false;
                        }
                        if ui.button("Discard and reload").clicked() {
                            self.confirm_reload = false;
                            self.reload();
                        }
                    });
                });
        }
        self.remove_split_window(ctx);
        self.remove_activation_window(ctx);
        if self.confirm_reset {
            egui::Window::new("Restore all 1805 ownership?")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("This clears the project's ownership overrides. Originals remain unchanged.");
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.confirm_reset = false;
                        }
                        if ui.button("Restore default owners").clicked() {
                            self.confirm_reset = false;
                            self.project.owners.clear();
                            self.dirty = true;
                            self.ui.analysis_wanted = self.source_ready;
                            self.mesh_dirty = true;
                            self.refresh_filter();
                        }
                    });
                });
        }
    }
}

fn label(key: &str) -> String {
    key.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
fn faction_color(key: &str) -> Color32 {
    let hash = key.bytes().fold(2166136261u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(16777619)
    });
    Color32::from_rgb(
        65 + (hash & 127) as u8,
        65 + ((hash >> 8) & 127) as u8,
        65 + ((hash >> 16) & 127) as u8,
    )
}
fn contains(triangle: [[f32; 2]; 3], point: [f32; 2]) -> bool {
    let cross = |a: [f32; 2], b: [f32; 2]| {
        (point[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (point[1] - b[1])
    };
    let d = [
        cross(triangle[0], triangle[1]),
        cross(triangle[1], triangle[2]),
        cross(triangle[2], triangle[0]),
    ];
    !(d.iter().any(|x| *x < 0.0) && d.iter().any(|x| *x > 0.0))
}
