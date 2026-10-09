use anyhow::{Context, Result, bail};
use kgc_port_editor::{
    app,
    local_assets,
    compatibility::{self, Analysis},
    map::{self, splits},
    model::{self, Mappings, Project, RegionActivation, RegionSplit, Rules},
    pipeline,
    snapshot::{self, Snapshot},
    target,
    tsv_ui::workflow,
};
use serde_json::{Value, json};
use std::{
    cell::OnceCell,
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const HELP: &str = "KGC Port Editor - native ETW -> NTW campaign conversion

USAGE
  KgcPortEditor [gui] [--workspace DIR] [--assets DIR]
  KgcPortEditor import --etw INSTALL_PATH --ntw INSTALL_PATH [--language CODE] [--workspace DIR]
    Each path accepts an installed game folder or its data folder. Rules and UI are rebuilt locally from verified originals.
    --language CODE (e.g. en, fr, de) selects the NTW base language; omitted = auto-detect from local_<xx>.pack, which errors if several languages are installed.
  KgcPortEditor verify-sources [--workspace DIR]
  KgcPortEditor assign --region KEY --faction KEY [--project FILE]
    --faction must be an existing faction of the target database (checked against its real catalog; the project need not be export-ready).
  KgcPortEditor build [--project FILE] [--workspace DIR] [--assets DIR]
  KgcPortEditor inspect --startpos FILE

TSV TARGET DATA (RPFM-exported TSV tables; the folder is only read and stored as an immutable, hash-verified dataset)
  KgcPortEditor tables requirements [--project FILE] [--workspace DIR] [--assets DIR]
  KgcPortEditor tables preview --folder DIR --language CODE [--name NAME] [--resources-reviewed] [--with-extras] [--project FILE] [--workspace DIR] [--assets DIR]
  KgcPortEditor tables import --folder DIR --language CODE [--name NAME] [--resources-reviewed] [--with-extras] [--pin] [--project FILE] [--workspace DIR] [--assets DIR]
  KgcPortEditor tables list [--project FILE]
  KgcPortEditor tables use (--target ID | --vanilla) [--project FILE]
  KgcPortEditor tables check [--target ID] [--project FILE]
    `requirements` lists the campaign inputs the existing-faction conversion needs from the target, with reasons;
    `preview` and `import` use the same list. Only these tables are imported: other tables in the folder are ignored
    unless --with-extras is given, in which case they are retained as analysis inputs without becoming required (they are
    NOT forced into the export), and the references of the rows you supply are never demanded as further tables. Imported
    tables are conversion inputs, not an export list: `build` exports only the campaign tables it adds or overrides, explicit
    conflict-resolution DB replacements and the merged localisation; the installed target mod must supply every unchanged table
    (land/naval unit stats, existing faction identities, ...). `preview` parses the folder and reports parse
    errors and missing required tables (exit 3 when it would refuse). --language CODE names the language of the
    target text; full text/localisation.loc and text/ui.loc TSVs are required.
    --resources-reviewed confirms you know TSVs carry no campaign/UI/script resources (unprovided ones use the
    original conversion templates; merged resources can be added with `adapter import`). Without it `build` stays
    blocked. `use --target` selects TSV mode with that exact dataset; a project in TSV mode without a dataset
    cannot build. `check` exits 4 when the TSV folder differs from the stored dataset.

ANALYSIS AND REVIEW (never writes to a game folder)
  KgcPortEditor analyze [--project FILE]
    Prints the full analysis JSON (issues, planned changes, effective target catalog).
    Exit 0 = ready, 3 = not export-ready: blocking issues remain and/or regions still need an owner (`unassigned_regions` in the JSON,
    listed on stderr). Pending owners are NOT compatibility issues; owner-dependent checks resume once every region has an owner.
    Gameplay is never verified (gameplay_tested=false).
  KgcPortEditor resolve list [--project FILE]
  KgcPortEditor resolve set --issue ID --choice CHOICE [--input-hash HASH] [--project FILE]
  KgcPortEditor resolve clear --issue ID [--project FILE]
  KgcPortEditor resolve apply --spec FILE.json [--project FILE]
  KgcPortEditor resolve discard-stale [--project FILE]
    CHOICE is one of the issue's listed choices (keep_target | use_port | member:<path>). A resolution
    is bound to the issue's input hash; changed inputs make it stale. apply spec JSON:
      {\"db_key:...\": \"use_port\", \"other\": {\"choice\": \"keep_target\", \"input_hash\": \"<hash>\"}}

MAPPINGS (validated against the effective target catalog of the selected TSV dataset)
  KgcPortEditor mapping list [--project FILE]
  KgcPortEditor mapping search --kind KIND [--side source|target] [--query TEXT] [--limit N] [--project FILE]
  KgcPortEditor mapping set --kind KIND --from KEY --to KEY [--project FILE]
  KgcPortEditor mapping remove --kind KIND --from KEY [--project FILE]
  KgcPortEditor mapping import --spec FILE.json [--replace] [--project FILE]
    KIND: factions | units | buildings | faction_templates (faction -> donor faction).
    spec JSON: {\"factions\": {\"france\": \"mod_france\"}, \"units\": {}, \"buildings\": {}, \"faction_templates\": {}}

RESOURCE ADAPTERS (merged custom UI/script/template resources bound to the exact TSV dataset)
  KgcPortEditor adapter import --folder DIR [--project FILE]
  KgcPortEditor adapter clear [--project FILE]

REGION SPLITS (cut a region in two; `build` applies every split to the generated world map; the GUI draws them on the Ownership map)
  KgcPortEditor split list [--project FILE]
  KgcPortEditor split add --source KEY --area N --a I --b J --interior x,y;x,y --name NAME --settlement-name NAME --center x,y [--key KEY] [--settlement SLUG] [--colour RRGGBB] [--project FILE] [--workspace DIR]
  KgcPortEditor split remove --key KEY [--project FILE]
    --source is a region of regions.esf (or the key of an earlier split); --area is the index of its area; --a and --b are positions in
    that area's outer ring where the cut starts and ends (the forward arc a..=b plus the cut becomes the new region); --interior lists the
    cut points from a towards b, in ETW world units, exactly as stored in the project (an empty value means a straight cut). --center is the
    new settlement and must lie inside the new region. --key defaults to <source>_<name>, --settlement to the settlement name, --colour to an
    unused map colour. `add` replays the existing splits on the original geometry first and refuses an invalid cut without touching the
    project. `remove` also removes every later split cut from the removed region.

TERRAIN RELIEF (height curve + land-only smoothing of the generated world heightmap; default matches Napoleon's own maps)
  KgcPortEditor terrain show [--project FILE]
  KgcPortEditor terrain set [--knee 0-255] [--upland-scale 0.05-1] [--smooth 0-8] [--painted DIR|none] [--project FILE]
  KgcPortEditor terrain original [--project FILE]      (original relief curve; fixed theatre terrain remains)
  KgcPortEditor terrain default [--project FILE]

WILDERNESS ACTIVATION (turn a wilderness_* land polygon of a playable theatre (wilderness_canada, wilderness_khiva, ...) into a campaign region; `build` applies every activation before the splits)
  KgcPortEditor activate list [--project FILE]
  KgcPortEditor activate add --key KEY --donor REGION --name NAME --settlement-name NAME --center x,y [--settlement SLUG] [--colour RRGGBB] [--areas all|centre|I,J,K] [--merge] [--carve-from SOURCE] [--project FILE] [--workspace DIR]
  KgcPortEditor activate remove --key KEY [--project FILE]
    --key is a land region of regions.esf that has no settlement and no campaign rules and lies inside one theatre rectangle (e.g.
    wilderness_great_plains); --donor is a playable region of the same theatre whose owner, culture, religion, rebels, resources and
    settlement slots the new region takes over. --center is the new settlement and must lie inside a passable area of the region.
    --settlement defaults to the settlement name, --colour to an unused map colour. `remove` also removes every split cut from the region
    and its owner override.
    --areas chooses which polygons (areas) of the region become passable: `centre` (default) = only the one holding --center (it must be
    passable already), `all`, or a comma list of area indices as numbered in the map after theatre extension. --merge unites the selected
    areas that touch each other into ONE area (offshore islets that touch nothing stay as they are). --carve-from SOURCE makes --key a NEW
    region built from the single area given by --areas (one index) of the backdrop region SOURCE (e.g. an island of `unexplorable`); --key
    must then be a new key, --name/--settlement-name/--center/--donor describe the new region (the donor supplies the rules and settlement
    slots). The settlement point must keep at least 2.1 u from the outline of its area.

FIXED THEATRE BOUNDS
  KgcPortEditor theatre list [--workspace DIR]
    Read-only: Europe stays original; India covers Thailand and Indonesia; America includes Western Plains.
    Bounds are built in and cannot be set or reset by a project. Preview and export use the same rectangles.
    Fixed terrain is generated natively from your Empire originals and public elevation data on first export, then cached locally.
    terrain set --painted DIR is an optional advanced override (--painted none clears it).

PROJECT
  KgcPortEditor project show [--project FILE]
    Ownership has one behaviour, 'Existing factions only'. It creates no new faction keys: no output faction may be absent
    from the target database, and YOU choose each region's owner with `assign` among the target's existing factions (list
    them with `mapping search --kind factions --side target`). Regions whose original ETW owner is not an existing target
    faction and that have no assignment yet are PENDING ownership (`unassigned_regions` of `analyze`, `resolve list`): they are
    not mapping or compatibility issues, `build` refuses until each has an owner, and no recipient is ever picked
    automatically. While any is pending, the owner-dependent compatibility checks have not run; they resume after the
    recipients are selected, so further blockers can then appear. An explicit owner you saved that is not an existing target
    faction is a real error: issue `ownership:<region>:invalid_owner`, repaired with `assign`. The starting forces of removed
    factions are NOT transferred to the new owners of their territory, and target-only owners get no invented army. Projects
    saved by older versions keep their owners, mappings, resolutions and bindings; no owner is reassigned automatically.

Original files are copied and hash-verified. Export: workspace/generated/ntw/GrandCampaign-NTW.zip. The ZIP contains data/... paths for manual extraction into the Napoleon game folder; the app never installs or overwrites game files. It holds only campaign additions/overrides, explicit conflict-resolution DB replacements and the merged localisation: imported tables are conversion inputs and are not re-exported when unchanged, so the installed target mod must supply them. `build` refuses to export while any blocking compatibility issue is unresolved or any region still needs an owner. Exported campaigns are NOT gameplay-verified. Project ownership affects new campaigns, not existing saves.";

const OPTIONS: &[&str] = &[
    "--workspace",
    "--assets",
    "--etw",
    "--ntw",
    "--project",
    "--region",
    "--faction",
    "--startpos",
    "--spec",
    "--target",
    "--folder",
    "--name",
    "--kind",
    "--side",
    "--from",
    "--to",
    "--issue",
    "--choice",
];
const VALUE_OPTIONS: &[&str] = &[
    "--input-hash",
    "--query",
    "--limit",
    "--language",
    "--source",
    "--area",
    "--knee",
    "--upland-scale",
    "--smooth",
    "--a",
    "--b",
    "--interior",
    "--key",
    "--settlement",
    "--settlement-name",
    "--center",
    "--colour",
    "--donor",
    "--areas",
    "--carve-from",
    "--painted",
];
const FLAGS: &[&str] = &["--vanilla", "--pin", "--replace", "--resources-reviewed", "--merge"];
const SUBCOMMANDS: &[&str] = &["tables", "mapping", "resolve", "adapter", "project", "split", "terrain", "activate", "theatre"];

fn main() {
    if let Err(error) = run() {
        eprintln!("KGC Port Editor: {error:#}");
        std::process::exit(1);
    }
}

struct Args {
    command: String,
    sub: Option<String>,
    options: BTreeMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        Self::parse_from(std::env::args().skip(1))
    }

    fn parse_from(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut args = args.into_iter().peekable();
        let command = args.next().unwrap_or_else(|| "gui".into());
        let mut sub = None;
        if SUBCOMMANDS.contains(&command.as_str()) {
            match args.peek() {
                Some(next) if !next.starts_with("--") => sub = args.next(),
                _ => bail!("{command} requires a subcommand; use --help"),
            }
        }
        let mut options = BTreeMap::new();
        let mut flags = Vec::new();
        while let Some(key) = args.next() {
            if FLAGS.contains(&key.as_str()) {
                if flags.contains(&key) {
                    bail!("Duplicate option {key}");
                }
                flags.push(key);
                continue;
            }
            if !OPTIONS.contains(&key.as_str()) && !VALUE_OPTIONS.contains(&key.as_str()) {
                bail!("Unknown option {key}; use --help");
            }
            let value = args
                .next()
                .with_context(|| format!("Missing value for {key}"))?;
            if value.starts_with("--") {
                bail!("Missing value for {key}");
            }
            if options.insert(key.clone(), value).is_some() {
                bail!("Duplicate option {key}");
            }
        }
        Ok(Self {
            command,
            sub,
            options,
            flags,
        })
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|flag| flag == name)
    }

    fn required(&self, key: &str) -> Result<&str> {
        self.options.get(key).map(String::as_str).with_context(|| {
            format!(
                "{} requires {key}",
                match &self.sub {
                    Some(sub) => format!("{} {sub}", self.command),
                    None => self.command.clone(),
                }
            )
        })
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    if matches!(args.command.as_str(), "--help" | "help" | "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let assets = match args.options.get("--assets") {
        Some(path) => PathBuf::from(path),
        None => pipeline::assets_dir()?,
    };
    let workspace = args
        .options
        .get("--workspace")
        .map(PathBuf::from)
        .unwrap_or_else(|| pipeline::default_workspace(&assets));
    let project_path = args
        .options
        .get("--project")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("project.json"));
    let progress = |message: &str| println!("{message}");
    let quiet = |message: &str| eprintln!("{message}");
    let env = Env {
        assets: &assets,
        workspace: &workspace,
        project_path: &project_path,
        derived_assets: OnceCell::new(),
    };
    match args.command.as_str() {
        "gui" => app::run(assets, workspace),
        "import" => {
            snapshot::import_with_language(
                &PathBuf::from(args.required("--etw")?),
                &PathBuf::from(args.required("--ntw")?),
                &workspace.join("sources"),
                args.options.get("--language").map(String::as_str),
                &progress,
            )?;
            let derived = env.runtime_assets()?;
            println!("Original source snapshot and locally rebuilt resources ready: {}", derived.display());
            Ok(())
        }
        "verify-sources" => {
            Snapshot::open(&workspace.join("sources"))?.verify(&progress)?;
            println!("Source manifest verified.");
            Ok(())
        }
        "assign" => assign(
            &env,
            args.required("--region")?,
            args.required("--faction")?,
            &quiet,
        ),
        "build" => {
            let project = env.load_project()?;
            if let Some(reason) = workflow::dataset_blocker(&project) {
                bail!(
                    "Export refused: {reason} (`tables import --pin` or `tables use --target ID`)."
                );
            }
            let analysis = env.analyze(&project, &quiet)?;
            let blocking = workflow::blockers(&analysis).len();
            let pending = analysis.unassigned_regions.len();
            if !analysis.ready || blocking > 0 || pending > 0 {
                print_blockers(&analysis);
                if pending > 0 && blocking == 0 {
                    bail!(
                        "Export refused: {pending} region(s) still need an owner. Choose an existing target faction for each with `assign --region KEY --faction EXISTING_KEY`; the remaining compatibility checks run after every region has an owner. Run `analyze` for the list."
                    );
                }
                if pending > 0 {
                    bail!(
                        "Export refused: {blocking} blocking compatibility issue(s) and {pending} region(s) still needing an owner (`assign --region KEY --faction EXISTING_KEY`; checks that depend on the owners run after assignment). Run `analyze` for the full report."
                    );
                }
                bail!(
                    "Export refused: {} blocking compatibility issue(s). Run `analyze` for the full report.",
                    blocking.max(1)
                );
            }
            let report = pipeline::generate(&workspace, env.runtime_assets()?, &project, &progress)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            eprintln!(
                "Gameplay not verified: test the generated campaign in Napoleon before relying on it."
            );
            Ok(())
        }
        "inspect" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&pipeline::inspect_owners(&PathBuf::from(
                    args.required("--startpos")?
                ))?)?
            );
            Ok(())
        }
        "analyze" => {
            let project = env.load_project()?;
            let analysis = env.analyze(&project, &quiet)?;
            println!("{}", serde_json::to_string_pretty(&analysis)?);
            if !analysis.ready
                || !workflow::blockers(&analysis).is_empty()
                || !analysis.unassigned_regions.is_empty()
            {
                print_blockers(&analysis);
                std::process::exit(3);
            }
            Ok(())
        }
        "tables" => tables(&env, &args, &quiet),
        "mapping" => mapping(&env, &args, &quiet),
        "resolve" => resolve(&env, &args, &quiet),
        "adapter" => adapter(&env, &args, &quiet),
        "split" => split(&env, &args),
        "terrain" => terrain(&env, &args),
        "activate" => activate(&env, &args),
        "theatre" => theatre(&env, &args),
        "project" => match args.sub.as_deref() {
            Some("show") => {
                println!("{}", serde_json::to_string_pretty(&env.load_project()?)?);
                Ok(())
            }
            other => bail!("Unknown project subcommand {other:?}; use --help"),
        },
        _ => bail!("Unknown command {}; use --help", args.command),
    }
}

struct Env<'a> {
    assets: &'a Path,
    workspace: &'a Path,
    project_path: &'a Path,
    derived_assets: OnceCell<PathBuf>,
}

impl Env<'_> {
    fn runtime_assets(&self) -> Result<&Path> {
        if self.derived_assets.get().is_none() {
            let assets = local_assets::ensure(self.workspace, self.assets, &|message| eprintln!("{message}"))?;
            let _ = self.derived_assets.set(assets);
        }
        Ok(self.derived_assets.get().expect("local assets initialized").as_path())
    }

    fn load_project(&self) -> Result<Project> {
        if self.project_path.exists() {
            Project::load(self.project_path)
        } else {
            Ok(Project::default())
        }
    }

    fn analyze(
        &self,
        project: &Project,
        progress: &(dyn Fn(&str) + Send + Sync),
    ) -> Result<Analysis> {
        compatibility::analyze(self.workspace, self.runtime_assets()?, project, progress)
    }
}

fn print_blockers(analysis: &Analysis) {
    if !analysis.unassigned_regions.is_empty() {
        // Pending owner selection is a task, not a compatibility issue: the original ETW owner has no existing
        // target faction and the user has not picked a recipient yet. No recipient is ever chosen automatically.
        eprintln!(
            "OWNERSHIP PENDING: {} region(s) need an existing target faction as owner: {}",
            analysis.unassigned_regions.len(),
            analysis.unassigned_regions.join(", ")
        );
        eprintln!(
            "  Assign each with `assign --region KEY --faction EXISTING_KEY` (`mapping search --kind factions --side target` lists the existing factions; no faction is created and no forces are transferred)."
        );
        eprintln!(
            "  Compatibility checks that depend on the owners were not run yet; they resume once every region has an owner, so the issues below may be incomplete."
        );
    }
    for issue in workflow::blockers(analysis) {
        eprintln!(
            "BLOCKER [{}] {} ({}) choices: {}",
            issue.category,
            issue.title,
            issue.id,
            if let Some(region) = workflow::owner_blocker_region(issue) {
                format!(
                    "none; assign an existing target faction: `assign --region {region} --faction KEY` (`mapping search --kind factions --side target`)"
                )
            } else if issue.choices.is_empty() {
                "none (mapping or adapter required)".to_string()
            } else {
                issue.choices.join(", ")
            }
        );
    }
}

fn read_json(path: &str) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("Read spec {path}"))?;
    serde_json::from_slice(&bytes).with_context(|| format!("Parse JSON spec {path}"))
}

fn assign(
    env: &Env<'_>,
    region: &str,
    faction: &str,
    quiet: &(dyn Fn(&str) + Send + Sync),
) -> Result<()> {
    let mut project = env.load_project()?;
    // Split regions are owner targets too: derive their rule entries before validating the region key.
    let rules = kgc_port_editor::map::splits::project_rules(&env.runtime_assets()?.join("rules"), &project)?;
    if let Some(reason) = workflow::dataset_blocker(&project) {
        bail!("Cannot read the target faction catalog: {reason}");
    }
    // Only the target's faction list is needed, and it does not depend on the owners: analysing without them
    // keeps a project whose current owners are missing or invalid repairable, and blocking issues are fine.
    let mut probe = project.clone();
    probe.owners.clear();
    let analysis = env.analyze(&probe, quiet)?;
    workflow::assign_owner(&mut project, &rules, Some(&analysis), region, faction)?;
    project.save(env.project_path)?;
    println!(
        "Assigned {region} → {faction}; saved {}",
        env.project_path.display()
    );
    Ok(())
}

fn parse_xy(text: &str) -> Result<[f32; 2]> {
    let (x, y) = text
        .trim()
        .split_once(',')
        .with_context(|| format!("'{text}' is not x,y"))?;
    Ok([
        x.trim().parse().with_context(|| format!("'{x}' is not a number"))?,
        y.trim().parse().with_context(|| format!("'{y}' is not a number"))?,
    ])
}

fn parse_colour(text: &str) -> Result<[u8; 3]> {
    let hex = text.trim().trim_start_matches('#');
    if hex.len() != 6 || !hex.is_ascii() {
        bail!("--colour must be six hex digits (rrggbb)");
    }
    let channel = |at: usize| u8::from_str_radix(&hex[at..at + 2], 16).context("--colour must be hex");
    Ok([channel(0)?, channel(2)?, channel(4)?])
}

fn terrain(env: &Env<'_>, args: &Args) -> Result<()> {
    let mut project = env.load_project()?;
    match args.sub.as_deref() {
        Some("show") => {
            println!("{}", serde_json::to_string_pretty(&project.terrain)?);
            println!("painted terrain: {}", project.painted_terrain.as_ref().map_or("none".into(), |p| p.display().to_string()));
            return Ok(());
        }
        Some("original") => project.terrain = kgc_port_editor::model::TerrainRelief::original(),
        Some("default") => project.terrain = kgc_port_editor::model::TerrainRelief::default(),
        Some("set") => {
            if let Some(v) = args.options.get("--knee") {
                project.terrain.knee = v.parse().context("--knee must be 0-255")?;
            }
            if let Some(v) = args.options.get("--upland-scale") {
                project.terrain.upland_scale = v.parse().context("--upland-scale must be a number")?;
            }
            if let Some(v) = args.options.get("--smooth") {
                project.terrain.smooth_sigma = v.parse().context("--smooth must be a number")?;
            }
            if let Some(v) = args.options.get("--painted") {
                project.painted_terrain = if v.eq_ignore_ascii_case("none") { None } else { Some(PathBuf::from(v)) };
            }
            project.terrain.validate()?;
        }
        other => bail!("Unknown terrain subcommand {other:?}; use --help"),
    }
    project.save(env.project_path)?;
    println!(
        "{}; saved {}",
        serde_json::to_string(&project.terrain)?,
        env.project_path.display()
    );
    Ok(())
}

fn activate(env: &Env<'_>, args: &Args) -> Result<()> {
    let rules_root = env.runtime_assets()?.join("rules");
    let mut project = env.load_project()?;
    match args.sub.as_deref() {
        Some("list") => {
            println!("{}", serde_json::to_string_pretty(&project.activations)?);
            Ok(())
        }
        Some("add") => {
            let rules = splits::project_rules(&rules_root, &project)?;
            let settlement_name = args.required("--settlement-name")?.trim().to_string();
            let activation = RegionActivation {
                key: args.required("--key")?.to_string(),
                donor: args.required("--donor")?.to_string(),
                name: args.required("--name")?.trim().to_string(),
                settlement: match args.options.get("--settlement") {
                    Some(slug) => slug.clone(),
                    None => splits::slug(&settlement_name),
                },
                colour: match args.options.get("--colour") {
                    Some(colour) => parse_colour(colour)?,
                    None => splits::pick_colour(&rules),
                },
                center: parse_xy(args.required("--center")?)?,
                settlement_name,
                areas: match args.options.get("--areas") {
                    Some(text) => model::AreaSelection::parse(text)?,
                    None => model::AreaSelection::default(),
                },
                merge: args.flag("--merge"),
                carve_from: args.options.get("--carve-from").cloned(),
            };
            activation.validate_options()?;
            let snapshot = Snapshot::open(&env.workspace.join("sources"))?;
            let (base, _) = map::region_source(&snapshot)?;
            let (data, rects_now) = splits::replay(&base, &project)?;
            splits::validate_activation(&activation, &rules, &data, &rects_now, &project.splits)?;
            // the whole project must replay on the original geometry with the new activation in place
            let key = activation.key.clone();
            if let Some(owner) = project.owners.get(&activation.donor).cloned() {
                project.owners.insert(key.clone(), owner);
            }
            project.activations.push(activation);
            project.validate_structure()?;
            splits::replay(&base, &project)?;
            splits::project_rules(&rules_root, &project)?;
            project.save(env.project_path)?;
            println!("Activated {key}; saved {}", env.project_path.display());
            Ok(())
        }
        Some("remove") => {
            let key = args.required("--key")?;
            let removal = splits::removal_of_activation(&project.activations, &project.splits, key)
                .with_context(|| format!("No activation with key {key}"))?;
            removal.remove_from(&mut project);
            splits::project_rules(&rules_root, &project)?;
            project.save(env.project_path)?;
            println!(
                "Removed activation of {}; saved {}",
                removal.keys.join(", "),
                env.project_path.display()
            );
            Ok(())
        }
        other => bail!("Unknown activate subcommand {other:?}; use --help"),
    }
}


fn theatre(env: &Env<'_>, args: &Args) -> Result<()> {
    anyhow::ensure!(args.sub.as_deref() == Some("list"), "Theatre bounds are fixed; only 'theatre list' is available");
    let snapshot = Snapshot::open(&env.workspace.join("sources"))?;
    let (mut data, _) = map::region_source(&snapshot)?;
    kgc_port_editor::regionmap::chain::extend(&mut data)?;
    for (key, [lo, hi]) in data.theatre_rects() {
        println!("theatre {key}: ({}, {})..({}, {}) [fixed]", lo[0], lo[1], hi[0], hi[1]);
    }
    Ok(())
}

fn split(env: &Env<'_>, args: &Args) -> Result<()> {
    let rules_root = env.runtime_assets()?.join("rules");
    let mut project = env.load_project()?;
    match args.sub.as_deref() {
        Some("list") => {
            println!("{}", serde_json::to_string_pretty(&project.splits)?);
            Ok(())
        }
        Some("add") => {
            let rules = splits::project_rules(&rules_root, &project)?;
            let name = args.required("--name")?.trim().to_string();
            let settlement_name = args.required("--settlement-name")?.trim().to_string();
            let source = args.required("--source")?.to_string();
            let number = |option: &str| -> Result<usize> {
                args.required(option)?
                    .parse()
                    .with_context(|| format!("{option} must be a non-negative integer"))
            };
            let interior = args
                .required("--interior")?
                .split(';')
                .filter(|point| !point.trim().is_empty())
                .map(parse_xy)
                .collect::<Result<Vec<_>>>()?;
            let split = RegionSplit {
                key: match args.options.get("--key") {
                    Some(key) => key.clone(),
                    None => format!("{source}_{}", splits::slug(&name)),
                },
                settlement: match args.options.get("--settlement") {
                    Some(slug) => slug.clone(),
                    None => splits::slug(&settlement_name),
                },
                colour: match args.options.get("--colour") {
                    Some(colour) => parse_colour(colour)?,
                    None => splits::pick_colour(&rules),
                },
                area: number("--area")?,
                a: number("--a")?,
                b: number("--b")?,
                center: parse_xy(args.required("--center")?)?,
                source,
                interior,
                name,
                settlement_name,
            };
            splits::validate_fields(&split, &rules)?;
            // replay the project's splits on the original geometry, then try the new cut
            let snapshot = Snapshot::open(&env.workspace.join("sources"))?;
            let (base, _) = map::region_source(&snapshot)?;
            let (mut data, _) = splits::replay(&base, &project)?;
            let source_index = data.index(&split.source)?;
            if !splits::cuttable_areas(&data, source_index).contains(&split.area) {
                bail!(
                    "Area {} of {} cannot be split: it does not exist or has inner holes (cuttable areas: {:?})",
                    split.area,
                    split.source,
                    splits::cuttable_areas(&data, source_index)
                );
            }
            splits::check_cut(
                &splits::ring_points(&data, source_index, split.area),
                split.a,
                split.b,
                &splits::clicks_of(&split),
            )?;
            let trial = splits::trial_split(&data, &split)?;
            if !trial.new_half_contains(split.center) {
                bail!(
                    "--center {:?} lies outside the new region; the forward arc a..=b plus the cut is the new region",
                    split.center
                );
            }
            if trial.source_settlement_in_new_half() {
                bail!("The source region's own settlement would end up in the new region; swap --a and --b (and reverse --interior)");
            }
            if let Some(owner) = project.owners.get(&split.source).cloned() {
                project.owners.insert(split.key.clone(), owner);
            }
            let key = split.key.clone();
            project.splits.push(split);
            splits::project_rules(&rules_root, &project)?;
            project.save(env.project_path)?;
            println!("Added split {key}; saved {}", env.project_path.display());
            Ok(())
        }
        Some("remove") => {
            let key = args.required("--key")?;
            let index = project
                .splits
                .iter()
                .position(|split| split.key == key)
                .with_context(|| format!("No split with key {key}"))?;
            let removed = splits::dependent_splits(&project.splits, index);
            let keys: Vec<String> = removed.iter().map(|&at| project.splits[at].key.clone()).collect();
            let mut position = 0;
            project.splits.retain(|_| {
                position += 1;
                !removed.contains(&(position - 1))
            });
            for key in &keys {
                project.owners.remove(key);
            }
            splits::project_rules(&rules_root, &project)?;
            project.save(env.project_path)?;
            println!("Removed split(s) {}; saved {}", keys.join(", "), env.project_path.display());
            Ok(())
        }
        other => bail!("Unknown split subcommand {other:?}; use --help"),
    }
}

fn dataset_json(summary: &target::Summary, pinned: bool) -> Value {
    json!({
        "id": summary.binding.id,
        "fingerprint": summary.binding.fingerprint,
        "name": summary.name,
        "language": summary.language,
        "pinned": pinned,
    })
}

fn find_dataset(env: &Env<'_>, id: &str) -> Result<target::TargetBinding> {
    target::list(env.workspace)?
        .datasets
        .into_iter()
        .find(|summary| summary.binding.id == id)
        .map(|summary| summary.binding)
        .with_context(|| format!("No TSV dataset {id}; run `tables list`"))
}

fn import_spec(args: &Args) -> Result<target::ImportSpec> {
    let folder = PathBuf::from(args.required("--folder")?);
    let name = match args.options.get("--name") {
        Some(name) => name.clone(),
        None => folder
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| !name.is_empty())
            .context("--name is required when the folder path has no folder name")?,
    };
    Ok(target::ImportSpec {
        name,
        language: args.required("--language")?.to_string(),
        folder,
        resources_reviewed: args.flag("--resources-reviewed"),
        extras: args.flag("--with-extras"),
    })
}

fn tables(env: &Env<'_>, args: &Args, quiet: &(dyn Fn(&str) + Send + Sync)) -> Result<()> {
    match args.sub.as_deref() {
        Some("requirements") => {
            let base = Snapshot::open(&env.workspace.join("sources"))?;
            let rules = Rules::load(&env.runtime_assets()?.join("rules"))?;
            let required = target::required_inputs(&base, &rules)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({ "required": required }))?
            );
            Ok(())
        }
        Some("preview") => {
            let spec = import_spec(args)?;
            let preview = target::preview(env.workspace, env.runtime_assets()?, &spec, quiet)?;
            println!("{}", serde_json::to_string_pretty(&preview)?);
            if !preview.problems.is_empty() || !preview.missing.is_empty() {
                eprintln!(
                    "Import would be refused: {} problem(s), {} missing required table(s).",
                    preview.problems.len(),
                    preview.missing.len()
                );
                std::process::exit(3);
            }
            Ok(())
        }
        Some("import") => {
            let spec = import_spec(args)?;
            let binding = target::import(env.workspace, env.runtime_assets()?, &spec, quiet)?;
            if args.flag("--pin") {
                let mut project = env.load_project()?;
                workflow::pin_target(&mut project, binding.clone());
                project.save(env.project_path)?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "id": binding.id,
                    "fingerprint": binding.fingerprint,
                    "resources_reviewed": spec.resources_reviewed,
                    "pinned": args.flag("--pin"),
                }))?
            );
            if !spec.resources_reviewed {
                eprintln!(
                    "Resources not confirmed: `build` stays blocked until the folder is imported with --resources-reviewed."
                );
            }
            Ok(())
        }
        Some("list") => {
            let project = env.load_project()?;
            let listing = target::list(env.workspace)?;
            let datasets: Vec<Value> = listing
                .datasets
                .iter()
                .map(|summary| {
                    dataset_json(summary, project.target.as_ref() == Some(&summary.binding))
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "datasets": datasets,
                    "problems": listing.problems,
                    "target_mode": project.target_mode,
                    "pinned": project.target.as_ref().map(|b| b.id.clone()),
                }))?
            );
            Ok(())
        }
        Some("check") => {
            let binding = match args.options.get("--target") {
                Some(id) => find_dataset(env, id)?,
                None => env
                    .load_project()?
                    .target
                    .context("No --target given and the project has no selected dataset")?,
            };
            let dataset = target::load(env.workspace, &binding)?;
            dataset.verify(quiet)?;
            let changes = dataset.changes()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "target": binding.id,
                    "snapshot_verified": true,
                    "resources_reviewed": dataset.manifest.resources_reviewed,
                    "external_changes": changes,
                }))?
            );
            if !changes.is_empty() {
                eprintln!(
                    "{} difference(s) from the TSV folder; import it again to adopt them.",
                    changes.len()
                );
                std::process::exit(4);
            }
            Ok(())
        }
        Some("use") => {
            let mut project = env.load_project()?;
            match (args.flag("--vanilla"), args.options.get("--target")) {
                (true, None) => workflow::use_vanilla(&mut project),
                (false, Some(id)) => workflow::pin_target(&mut project, find_dataset(env, id)?),
                _ => bail!("tables use requires exactly one of --target ID or --vanilla"),
            }
            project.save(env.project_path)?;
            println!(
                "Project target: {}; saved {}",
                project
                    .target
                    .as_ref()
                    .map_or("vanilla Napoleon".to_string(), |b| format!(
                        "TSV dataset {}",
                        b.id
                    )),
                env.project_path.display()
            );
            Ok(())
        }
        other => bail!("Unknown tables subcommand {other:?}; use --help"),
    }
}

fn mapping(env: &Env<'_>, args: &Args, quiet: &(dyn Fn(&str) + Send + Sync)) -> Result<()> {
    match args.sub.as_deref() {
        Some("list") => {
            println!(
                "{}",
                serde_json::to_string_pretty(&env.load_project()?.mappings)?
            );
            Ok(())
        }
        Some("search") => {
            let project = env.load_project()?;
            let analysis = env.analyze(&project, quiet)?;
            let kind = args.required("--kind")?;
            let options = match args.options.get("--side").map(String::as_str) {
                None | Some("target") => workflow::target_options(&analysis, kind),
                Some("source") => workflow::source_options(&analysis, kind),
                Some(other) => bail!("--side must be source or target, not {other}"),
            };
            if !workflow::MAPPING_FIELDS.contains(&kind) {
                bail!(
                    "Unknown mapping kind {kind}; use one of {}",
                    workflow::MAPPING_FIELDS.join(", ")
                );
            }
            let query = args
                .options
                .get("--query")
                .map(|q| q.to_lowercase())
                .unwrap_or_default();
            let limit: usize = match args.options.get("--limit") {
                Some(value) => value.parse().context("--limit must be a number")?,
                None => 100,
            };
            let matches: Vec<&String> = options
                .iter()
                .filter(|key| query.is_empty() || key.to_lowercase().contains(&query))
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "kind": kind,
                    "total_matches": matches.len(),
                    "matches": matches.iter().take(limit).collect::<Vec<_>>(),
                }))?
            );
            Ok(())
        }
        Some("set") => {
            let mut project = env.load_project()?;
            let analysis = env.analyze(&project, quiet)?;
            let (kind, from, to) = (
                args.required("--kind")?,
                args.required("--from")?,
                args.required("--to")?,
            );
            workflow::set_mapping(&mut project, &analysis, kind, from, to)?;
            project.save(env.project_path)?;
            println!(
                "Mapped {kind} {from} → {to}; saved {}",
                env.project_path.display()
            );
            Ok(())
        }
        Some("remove") => {
            let mut project = env.load_project()?;
            let (kind, from) = (args.required("--kind")?, args.required("--from")?);
            if !workflow::remove_mapping(&mut project, kind, from)? {
                bail!("No {kind} mapping for {from}");
            }
            project.save(env.project_path)?;
            println!("Removed {kind} mapping for {from}");
            Ok(())
        }
        Some("import") => {
            let spec: Mappings = serde_json::from_value(read_json(args.required("--spec")?)?)
                .context("Invalid mapping spec; expected {factions,units,buildings,faction_templates} string maps")?;
            let mut project = env.load_project()?;
            let analysis = env.analyze(&project, quiet)?;
            if args.flag("--replace") {
                project.mappings = Mappings::default();
            }
            let mut count = 0;
            for field in workflow::MAPPING_FIELDS {
                if let Some(entries) = workflow::mapping_ref(&spec, field) {
                    for (from, to) in entries {
                        workflow::set_mapping(&mut project, &analysis, field, from, to)
                            .with_context(|| format!("{field} {from} → {to}"))?;
                        count += 1;
                    }
                }
            }
            project.save(env.project_path)?;
            println!(
                "Imported {count} mapping(s); saved {}",
                env.project_path.display()
            );
            Ok(())
        }
        other => bail!("Unknown mapping subcommand {other:?}; use --help"),
    }
}

fn resolve(env: &Env<'_>, args: &Args, quiet: &(dyn Fn(&str) + Send + Sync)) -> Result<()> {
    let mut project = env.load_project()?;
    let analysis = env.analyze(&project, quiet)?;
    match args.sub.as_deref() {
        Some("list") => {
            let issues: Vec<Value> = analysis
                .issues
                .iter()
                .map(|issue| {
                    json!({
                        "id": issue.id,
                        "severity": issue.severity,
                        "category": issue.category,
                        "title": issue.title,
                        "detail": issue.detail,
                        "input_hash": issue.input_hash,
                        "choices": issue.choices,
                        "mapping": issue.mapping.as_ref().map(|hint| json!({"field": hint.field, "key": hint.key})),
                        "resolved": issue.resolved,
                        "saved_choice": project.resolutions.get(&issue.id).map(|r| r.choice.clone()),
                    })
                })
                .collect();
            let stale: Vec<Value> = workflow::stale_resolutions(&project, &analysis)
                .into_iter()
                .map(|(id, reason)| json!({"id": id, "reason": reason}))
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ready": analysis.ready,
                    "unassigned_regions": analysis.unassigned_regions,
                    "gameplay_tested": analysis.gameplay_tested,
                    "issues": issues,
                    "stale_resolutions": stale,
                }))?
            );
            return Ok(());
        }
        Some("set") => {
            workflow::apply_resolution(
                &mut project,
                &analysis,
                args.required("--issue")?,
                args.required("--choice")?,
                args.options.get("--input-hash").map(String::as_str),
            )?;
        }
        Some("clear") => {
            let id = args.required("--issue")?;
            if project.resolutions.remove(id).is_none() {
                bail!("No saved resolution for {id}");
            }
        }
        Some("apply") => {
            let spec = read_json(args.required("--spec")?)?;
            let entries = spec
                .as_object()
                .context("Resolution spec must be a JSON object keyed by issue id")?;
            for (id, value) in entries {
                let (choice, hash) = match value {
                    Value::String(choice) => (choice.as_str(), None),
                    Value::Object(object) => (
                        object
                            .get("choice")
                            .and_then(Value::as_str)
                            .with_context(|| format!("{id}: missing \"choice\""))?,
                        object.get("input_hash").and_then(Value::as_str),
                    ),
                    _ => bail!("{id}: expected a choice string or {{choice, input_hash}}"),
                };
                workflow::apply_resolution(&mut project, &analysis, id, choice, hash)
                    .with_context(|| format!("Resolution for {id}"))?;
            }
        }
        Some("discard-stale") => {
            let stale = workflow::stale_resolutions(&project, &analysis);
            for (id, _) in &stale {
                project.resolutions.remove(id);
            }
            println!("Discarded {} stale resolution(s).", stale.len());
        }
        other => bail!("Unknown resolve subcommand {other:?}; use --help"),
    }
    project.save(env.project_path)?;
    println!(
        "Saved {}; re-run `analyze` to confirm the issue is resolved.",
        env.project_path.display()
    );
    Ok(())
}

fn adapter(env: &Env<'_>, args: &Args, quiet: &(dyn Fn(&str) + Send + Sync)) -> Result<()> {
    let mut project = env.load_project()?;
    match args.sub.as_deref() {
        Some("import") => {
            let dataset = project.target.clone().context(
                "Adapters bind to a selected TSV dataset; run `tables use --target ID` first",
            )?;
            let binding = target::import_adapter(
                env.workspace,
                &dataset,
                &PathBuf::from(args.required("--folder")?),
                quiet,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "id": binding.id,
                    "fingerprint": binding.fingerprint,
                    "target_fingerprint": binding.target_fingerprint,
                }))?
            );
            project.adapter = Some(binding);
        }
        Some("clear") => project.adapter = None,
        other => bail!("Unknown adapter subcommand {other:?}; use --help"),
    }
    project.save(env.project_path)?;
    println!("Saved {}; re-run `analyze`.", env.project_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kgc_port_editor::model::TargetMode;

    fn args(list: &[&str]) -> Result<Args> {
        Args::parse_from(list.iter().map(|item| item.to_string()))
    }

    /// Assets with minimal rules plus a workspace, removed again on drop.
    struct Scratch {
        root: PathBuf,
        assets: PathBuf,
        workspace: PathBuf,
        project: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("kgc_cli_{name}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            let mapping = root.join("assets/rules/mapping");
            std::fs::create_dir_all(&mapping).unwrap();
            for (file, json) in [
                (
                    "regions",
                    r#"{"acadia": {"owner": "france"}, "silesia": {"owner": "prussia"}}"#,
                ),
                ("factions", r#"{"france": {}, "prussia": {}}"#),
                ("units", "{}"),
                ("buildings", "{}"),
                ("buildings_by_slot_type", "{}"),
                ("slot_types", "{}"),
                ("armies", "{}"),
            ] {
                std::fs::write(mapping.join(format!("{file}.json")), json).unwrap();
            }
            Self {
                assets: root.join("assets"),
                workspace: root.join("workspace"),
                project: root.join("workspace/project.json"),
                root,
            }
        }

        fn env(&self) -> Env<'_> {
            Env {
                assets: &self.assets,
                workspace: &self.workspace,
                project_path: &self.project,
                derived_assets: OnceCell::from(self.assets.clone()),
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn the_recycling_toggle_is_gone_from_the_command_line() {
        assert!(args(&["project", "recycling", "--enabled", "true"]).is_err());
        assert!(args(&["project", "show", "--enabled", "true"]).is_err());
        assert_eq!(
            args(&["project", "show", "--project", "p.json"])
                .unwrap()
                .sub
                .as_deref(),
            Some("show")
        );
    }

    #[test]
    fn assign_needs_a_readable_target_catalog_before_analysis_and_leaves_the_project_untouched() {
        let scratch = Scratch::new("assign");
        let env = scratch.env();
        let quiet = |_: &str| {};
        // A TSV-mode project without a dataset has no catalog: refused before any analysis.
        let mut project = Project::default();
        project.owners.insert("silesia".into(), "france".into());
        project.target_mode = TargetMode::Tsv;
        project.save(&scratch.project).unwrap();
        let error = assign(&env, "acadia", "target_only_faction", &quiet).unwrap_err();
        assert!(format!("{error:#}").contains("Cannot read the target faction catalog"));
        let after = Project::load(&scratch.project).unwrap();
        assert!(!after.owners.contains_key("acadia"));
        assert_eq!(after.owners["silesia"], "france");
    }
}
