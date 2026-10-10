//! Dataset and adapter store behaviour. TSV decoding itself is covered by the codec; these tests feed already
//! decoded entries so that nothing depends on a schema file.

use super::*;
use crate::pack::test_util::TempDir;
use crate::snapshot::{self, tests as snap_tests};

fn quiet() -> impl Fn(&str) + Send + Sync {
    |_: &str| {}
}

fn db_entry(file: &str, member: &str, rows: usize) -> Entry {
    let dir = member.split('/').nth(1).unwrap().to_owned();
    let table = db::Table {
        name: dir,
        guid: None,
        version: 0,
        version_marker: false,
        flag: 1,
        fields: Vec::new(),
        rows: vec![Vec::new(); rows],
    };
    Entry {
        file: file.to_owned(),
        tsv: format!("key\n#{dir};0;{member}\n{file} {rows}\n", dir = member.split('/').nth(1).unwrap()).into_bytes(),
        kind: MemberKind::Db,
        member: member.to_owned(),
        rows,
        encoded: db::encode(&table).unwrap(),
    }
}

fn loc_entry(file: &str, member: &str, text: &str) -> Entry {
    Entry {
        file: file.to_owned(),
        tsv: format!("#loc {file} {text}\n").into_bytes(),
        kind: MemberKind::Loc,
        member: member.to_owned(),
        rows: 1,
        encoded: db::encode_loc(&[("key".to_owned(), text.to_owned(), 0)], 1).unwrap(),
    }
}

fn entries() -> Vec<Entry> {
    vec![
        db_entry("units_tables_a.tsv", "db/units_tables/a_units", 2),
        db_entry("db/units_tables/b.tsv", "db/units_tables/b_units", 3),
        db_entry("factions.tsv", "db/factions_tables/mod_factions", 1),
        loc_entry("loc/localisation.tsv", "text/localisation.loc", "texte"),
        loc_entry("loc/ui.tsv", "text/ui.loc", "interface"),
    ]
}

fn write_folder(root: &Path, list: &[Entry]) {
    for e in list {
        let p = root.join(&e.file);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, &e.tsv).unwrap();
    }
}

fn spec(name: &str, folder: &Path) -> ImportSpec {
    ImportSpec {
        name: name.to_owned(),
        language: "fr".to_owned(),
        folder: folder.to_owned(),
        resources_reviewed: true,
        extras: true,
    }
}

fn analysis(list: Vec<Entry>, folder: &Path) -> Analysis {
    Analysis {
        entries: list,
        preview: Preview::default(),
        notes: Vec::new(),
        source: Some(folder.to_owned()),
    }
}

struct Env {
    dir: TempDir,
    ws: PathBuf,
    input: PathBuf,
}

fn env() -> Env {
    let dir = TempDir::new("target");
    let ws = dir.path().join("ws");
    let input = dir.path().join("tsv");
    fs::create_dir_all(&ws).unwrap();
    write_folder(&input, &entries());
    Env { dir, ws, input }
}

fn publish_default(e: &Env) -> TargetBinding {
    publish(
        &e.ws,
        &spec("Mod", &e.input),
        &analysis(entries(), &e.input),
        &quiet(),
    )
    .unwrap()
}

fn unlock(path: &Path) {
    let mut perm = fs::metadata(path).unwrap().permissions();
    perm.set_readonly(false);
    fs::set_permissions(path, perm).unwrap();
}

#[test]
fn datasets_are_pinned_idempotent_and_independent_of_later_folder_edits() {
    let e = env();
    let binding = publish_default(&e);
    assert!(binding.id.starts_with('t'));
    let data = load(&e.ws, &binding).unwrap();
    data.verify(&quiet()).unwrap();
    assert_eq!(data.manifest.language, "fr");
    assert!(data.manifest.resources_reviewed);
    assert!(data.changes().unwrap().is_empty());

    // The same content imported again, or from a copy of the folder elsewhere, is the same dataset.
    assert_eq!(publish_default(&e), binding);
    let copy = e.dir.path().join("copy");
    write_folder(&copy, &entries());
    let again = publish(
        &e.ws,
        &spec("Mod", &copy),
        &analysis(entries(), &copy),
        &quiet(),
    )
    .unwrap();
    assert_eq!(again, binding);
    // A different name or different data is a new dataset next to it.
    let renamed = publish(
        &e.ws,
        &spec("Other", &e.input),
        &analysis(entries(), &e.input),
        &quiet(),
    )
    .unwrap();
    assert_ne!(renamed.id, binding.id);
    let mut other = entries();
    other[0] = db_entry("units_tables_a.tsv", "db/units_tables/a_units", 5);
    let changed = publish(
        &e.ws,
        &spec("Mod", &e.input),
        &analysis(other, &e.input),
        &quiet(),
    )
    .unwrap();
    assert_ne!(changed.fingerprint, binding.fingerprint);
    let listing = list(&e.ws).unwrap();
    assert!(listing.problems.is_empty());
    assert_eq!(listing.datasets.len(), 3);

    // Edits of the external folder are reported but never change the stored dataset.
    fs::write(e.input.join("units_tables_a.tsv"), b"edited").unwrap();
    fs::remove_file(e.input.join("factions.tsv")).unwrap();
    fs::write(e.input.join("new.tsv"), b"new").unwrap();
    let report = data.changes().unwrap().join("\n");
    assert!(
        report.contains("changed") && report.contains("units_tables_a.tsv"),
        "{report}"
    );
    assert!(
        report.contains("removed") && report.contains("factions.tsv"),
        "{report}"
    );
    assert!(
        report.contains("added") && report.contains("new.tsv"),
        "{report}"
    );
    data.verify(&quiet()).unwrap();
    let reloaded = load(&e.ws, &binding).unwrap();
    assert_eq!(reloaded.manifest.files.len(), 5);
    fs::remove_dir_all(&e.input).unwrap();
    assert!(data.changes().unwrap()[0].contains("missing"));
    data.verify(&quiet()).unwrap();

    // Stale bindings and identifiers of pack profiles are never accepted.
    let stale = TargetBinding {
        id: binding.id.clone(),
        fingerprint: "0".repeat(64),
    };
    assert!(format!("{:#}", load(&e.ws, &stale).unwrap_err()).contains("stale"));
    let old = TargetBinding {
        id: format!("p{}", &binding.id[1..]),
        fingerprint: binding.fingerprint.clone(),
    };
    assert!(load(&e.ws, &old).is_err());
}

#[test]
fn export_manifest_is_portable_and_lists_member_identities() {
    let e = env();
    let binding = publish_default(&e);
    let manifest = load(&e.ws, &binding).unwrap().export_manifest();
    let text = manifest.to_string();
    assert!(!text.contains(&e.dir.path().to_string_lossy().replace('\\', "\\\\")));
    assert_eq!(manifest["fingerprint"], binding.fingerprint);
    let members: Vec<&str> = manifest["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["member"].as_str().unwrap())
        .collect();
    assert!(members.contains(&"db/units_tables/b_units") && members.contains(&"text/ui.loc"));
}

#[test]
fn tampering_with_stored_files_is_detected() {
    let e = env();
    let binding = publish_default(&e);
    let data = load(&e.ws, &binding).unwrap();
    let root = e.ws.join("targets").join(&binding.id);

    // Files are read-only; an edit that gets past that is caught by hashes, on use and by verify.
    let member = root.join(member_name(0));
    assert!(fs::write(&member, b"x").is_err());
    unlock(&member);
    let original = fs::read(&member).unwrap();
    let mut forged = original.clone();
    *forged.last_mut().unwrap() ^= 1;
    fs::write(&member, &forged).unwrap();
    assert!(data.verify(&quiet()).is_err());
    let i = snap_tests::installs();
    let base = snapshot::import(&i.etw, &i.ntw, &i.dest, &snap_tests::quiet()).unwrap();
    assert!(data.effective_snapshot(&base).is_err());
    fs::write(&member, &original).unwrap();
    data.verify(&quiet()).unwrap();

    // An unlisted file is refused; so is an edited manifest.
    fs::write(root.join("tsv/extra.tsv"), b"x").unwrap();
    assert!(format!("{:#}", data.verify(&quiet()).unwrap_err()).contains("unlisted"));
    fs::remove_file(root.join("tsv/extra.tsv")).unwrap();
    let json = root.join(TARGET_JSON);
    unlock(&json);
    let text = fs::read_to_string(&json)
        .unwrap()
        .replace("\"fr\"", "\"en\"");
    fs::write(&json, text).unwrap();
    assert!(load(&e.ws, &binding).is_err());
    assert!(!list(&e.ws).unwrap().problems.is_empty());
}

#[test]
fn effective_snapshot_is_the_dataset_alone_for_tables_and_text() {
    let i = snap_tests::installs();
    let base = snapshot::import(&i.etw, &i.ntw, &i.dest, &snap_tests::quiet()).unwrap();
    let e = env();
    let mut list_ = entries();
    list_.push(loc_entry("loc/extra.tsv", "text/extra.loc", "supplement"));
    let binding = publish(
        &e.ws,
        &spec("Mod", &e.input),
        &analysis(list_, &e.input),
        &quiet(),
    )
    .unwrap();
    let data = load(&e.ws, &binding).unwrap();
    let eff = data.effective_snapshot(&base).unwrap();

    // The vanilla factions table is hidden everywhere; only the dataset's members exist.
    assert!(base.exists(Game::Ntw, "db/factions_tables/factions"));
    assert!(!eff.exists(Game::Ntw, "db/factions_tables/factions"));
    assert!(!eff.has_in_source(Game::Ntw, "db/factions_tables/factions"));
    assert_eq!(
        eff.names(Game::Ntw, "db/"),
        vec![
            "db/factions_tables/mod_factions",
            "db/units_tables/a_units",
            "db/units_tables/b_units"
        ]
    );
    for name in eff.names(Game::Ntw, "db/") {
        assert_eq!(eff.member_priority(Game::Ntw, &name), 1);
    }
    assert_eq!(
        eff.member_origin(Game::Ntw, "db/units_tables/b_units"),
        "TSV db/units_tables/b.tsv"
    );
    let stored = fs::read(e.ws.join("targets").join(&binding.id).join(member_name(0))).unwrap();
    assert_eq!(
        eff.read(Game::Ntw, "db/units_tables/b_units").unwrap(),
        stored
    );

    // Full target text replaces the base text; additional text members are kept.
    let (_, rows) = db::decode_loc(&eff.read(Game::Ntw, "text/ui.loc").unwrap()).unwrap();
    assert_eq!(rows[0].1, "interface");
    assert!(eff.exists(Game::Ntw, "text/extra.loc"));

    // Everything that is not a table or text, and the other game, still comes from the original snapshot.
    assert_eq!(
        eff.read(Game::Ntw, "ui/frontend ui/main").unwrap(),
        b"layout"
    );
    assert_eq!(
        eff.read(Game::Ntw, "campaigns/eur_napoleon/startpos.esf")
            .unwrap(),
        b"ntw-startpos"
    );
    assert_eq!(
        eff.read(Game::Etw, "db/factions_tables/factions").unwrap(),
        b"etw-factions-patched"
    );
    assert_eq!(
        eff.original()
            .read(Game::Ntw, "db/factions_tables/factions")
            .unwrap(),
        b"ntw-factions"
    );
    assert_eq!(
        eff.original().read(Game::Ntw, "text/ui.loc").unwrap(),
        b"ntw-ui-loc"
    );

    // Without the full localisation there is no effective target: base text is never a stand-in.
    let mut partial = entries();
    partial.retain(|x| x.member != "text/ui.loc");
    let incomplete = publish(
        &e.ws,
        &spec("Partial", &e.input),
        &analysis(partial, &e.input),
        &quiet(),
    )
    .unwrap();
    let err = load(&e.ws, &incomplete)
        .unwrap()
        .effective_snapshot(&base)
        .unwrap_err();
    assert!(format!("{err:#}").contains("text/ui.loc"));
}

#[test]
fn the_import_folder_cannot_contain_the_workspace() {
    let dir = TempDir::new("target_overlap");
    let input = dir.path().join("tsv");
    let ws = input.join("workspace");
    write_folder(&input, &entries());
    let err = publish(
        &ws,
        &spec("Mod", &input),
        &analysis(entries(), &input),
        &quiet(),
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("overlaps"));
    assert!(!ws.join("targets").exists() || fs::read_dir(ws.join("targets")).unwrap().count() == 0);
}

#[test]
fn only_database_and_localisation_member_shapes_are_accepted() {
    for ok in ["db/units_tables/main", "DB\\Units_Tables\\Main"] {
        check_member(MemberKind::Db, ok).unwrap();
    }
    for bad in [
        "db/units_tables",
        "db/units/main",
        "db/_tables/main",
        "db/units_tables/a/b",
        "text/ui.loc",
        "db/../units_tables/x",
        "campaigns/eur_napoleon/startpos.esf",
    ] {
        assert!(check_member(MemberKind::Db, bad).is_err(), "{bad}");
    }
    check_member(MemberKind::Loc, "text/ui.loc").unwrap();
    for bad in [
        "text/.loc",
        "text/sub/ui.loc",
        "db/units_tables/ui.loc",
        "text/ui.txt",
        "../text/ui.loc",
    ] {
        assert!(check_member(MemberKind::Loc, bad).is_err(), "{bad}");
    }
}

#[test]
fn folder_reading_ignores_other_files_and_enforces_limits() {
    let dir = TempDir::new("target_scan");
    let root = dir.path().join("in");
    for (rel, data) in [
        ("a.tsv", "1"),
        ("sub/B.TSV", "22"),
        ("readme.md", "ignored"),
        ("sub/mod.pack", "ignored"),
    ] {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, data).unwrap();
    }
    let keep = |k: &str| k.ends_with(".tsv");
    let big = Limits {
        files: 10,
        file_bytes: 100,
        total_bytes: 1000,
    };
    let tree = scan::read_tree(&root, &keep, &big, &quiet());
    assert!(tree.problems.is_empty(), "{:?}", tree.problems);
    let names: Vec<&str> = tree.files.iter().map(|f| f.rel.as_str()).collect();
    assert_eq!(names, ["a.tsv", "sub/B.TSV"]);
    assert_eq!(tree.ignored.len(), 2);

    let few = Limits { files: 1, ..big };
    assert!(
        !scan::read_tree(&root, &keep, &few, &quiet())
            .problems
            .is_empty()
    );
    let small = Limits {
        file_bytes: 1,
        ..big
    };
    let tree = scan::read_tree(&root, &keep, &small, &quiet());
    assert_eq!(tree.files.len(), 1);
    assert!(tree.problems[0].contains("sub/B.TSV"));
    let total = Limits {
        total_bytes: 2,
        ..big
    };
    assert!(
        !scan::read_tree(&root, &keep, &total, &quiet())
            .problems
            .is_empty()
    );

    assert!(scan::open_folder(&root, "TSV").is_ok());
    assert!(scan::open_folder(&root.join("a.tsv"), "TSV").is_err());
    assert!(scan::open_folder(&root.join("missing"), "TSV").is_err());
    assert!(scan::open_folder(Path::new(""), "TSV").is_err());
    assert!(scan::open_folder(&root.join("..").join("in"), "TSV").is_err());
}

fn adapter_folder(root: &Path) {
    for (rel, data) in [
        ("ui/custom/panel.txt", "ui"),
        ("script/Campaign.lua", "lua"),
        ("campaigns/wld_napoleon/scripting.lua", "world script"),
        ("campaigns/eur_napoleon/startpos.esf", "donor"),
    ] {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, data).unwrap();
    }
}

#[test]
fn adapters_are_bound_to_the_exact_dataset_and_verified() {
    let e = env();
    let binding = publish_default(&e);
    let folder = e.dir.path().join("adapter");
    adapter_folder(&folder);
    let adapter = import_adapter(&e.ws, &binding, &folder, &quiet()).unwrap();
    assert!(adapter.id.starts_with('a'));
    assert_eq!(adapter.target_fingerprint, binding.fingerprint);
    let files = adapter_files(&e.ws, &adapter, &binding).unwrap();
    assert_eq!(
        files.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "campaigns/eur_napoleon/startpos.esf",
            "campaigns/wld_napoleon/scripting.lua",
            "script/campaign.lua",
            "ui/custom/panel.txt"
        ]
    );
    assert_eq!(files["script/campaign.lua"], b"lua");
    assert_eq!(
        adapter_manifest(&e.ws, &adapter, &binding)
            .unwrap()
            .files
            .len(),
        4
    );
    assert_eq!(
        import_adapter(&e.ws, &binding, &folder, &quiet()).unwrap(),
        adapter
    );

    // Another dataset cannot use it, whatever binding it presents.
    let other = publish(
        &e.ws,
        &spec("Other", &e.input),
        &analysis(entries(), &e.input),
        &quiet(),
    )
    .unwrap();
    assert!(format!("{:#}", adapter_files(&e.ws, &adapter, &other).unwrap_err()).contains("stale"));
    let forged = AdapterBinding {
        target_fingerprint: other.fingerprint.clone(),
        ..adapter.clone()
    };
    assert!(adapter_files(&e.ws, &forged, &other).is_err());

    // Content tampering and extra files are caught.
    let root = e.ws.join("adapters").join(&adapter.id);
    let file = root.join("files/script/campaign.lua");
    unlock(&file);
    fs::write(&file, b"LUA").unwrap();
    assert!(adapter_files(&e.ws, &adapter, &binding).is_err());
    fs::write(&file, b"lua").unwrap();
    adapter_files(&e.ws, &adapter, &binding).unwrap();
    fs::write(root.join("files/script/extra.lua"), b"x").unwrap();
    assert!(adapter_files(&e.ws, &adapter, &binding).is_err());
}

#[test]
fn adapters_refuse_database_text_packs_and_foreign_paths() {
    let e = env();
    let binding = publish_default(&e);
    for (bad, why) in [
        ("db/units_tables/x", "database"),
        ("text/ui.loc", "text"),
        ("data.pack", "pack"),
        ("ui/inner.pack", "nested pack"),
        (
            "campaigns/eur_napoleon/scripting.lua",
            "other campaign script",
        ),
        ("campaigns/wld_napoleon/startpos.esf", "generated startpos"),
    ] {
        let folder = e
            .dir
            .path()
            .join(format!("adapter_{}", why.replace(' ', "_")));
        adapter_folder(&folder);
        let p = folder.join(bad);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, b"x").unwrap();
        let err = format!(
            "{:#}",
            import_adapter(&e.ws, &binding, &folder, &quiet()).unwrap_err()
        );
        assert!(err.contains("not an adapter file"), "{why}: {err}");
    }
    let empty = e.dir.path().join("empty");
    fs::create_dir_all(&empty).unwrap();
    assert!(import_adapter(&e.ws, &binding, &empty, &quiet()).is_err());
    assert!(import_adapter(&e.ws, &binding, &e.dir.path().join("missing"), &quiet()).is_err());
    assert!(
        !e.ws.join("adapters").exists()
            || fs::read_dir(e.ws.join("adapters")).unwrap().count() == 0
    );
}

fn requirement(table: &str) -> Requirement {
    Requirement {
        table: table.into(),
        reason: "test".into(),
    }
}

fn provided(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_owned()).collect()
}

#[test]
fn required_localisation_is_always_added_and_the_list_is_sorted() {
    let mut required = vec![requirement("units_tables"), requirement("factions_tables")];
    add_required_loc(&mut required);
    let names: Vec<_> = required.iter().map(|r| r.table.as_str()).collect();
    assert_eq!(
        names,
        [
            "factions_tables",
            "text/localisation.loc",
            "text/ui.loc",
            "units_tables"
        ]
    );
}

#[test]
fn supplied_tables_outside_the_requirements_are_optional_extras() {
    let mut preview = Preview::default();
    preview.required = vec![requirement("Units_Tables")];
    add_required_loc(&mut preview.required);
    preview.tables = vec![
        "preview_a_tables".into(),
        "text/localisation.loc".into(),
        "text/ui.loc".into(),
        "units_tables".into(),
    ];
    // Extras are reported but neither missing nor required.
    assert_eq!(preview.extras(), ["preview_a_tables"]);
    let present: BTreeSet<String> = preview.tables.iter().cloned().collect();
    assert!(missing_tables(&preview.required, &present).is_empty());
}

#[test]
fn missing_tables_never_demand_extras_or_their_references() {
    // A provided table that references tables outside the requirement list neither adds requirements nor
    // makes the folder incomplete; extra tables are ignored, and absence of a referenced table is not reported.
    let mut required = vec![requirement("Units_Tables")];
    add_required_loc(&mut required);
    let folder = provided(&[
        "units_tables",
        "preview_a_tables",
        "text/localisation.loc",
        "text/ui.loc",
    ]);
    assert!(missing_tables(&required, &folder).is_empty());
    let without_loc = provided(&["units_tables", "preview_a_tables"]);
    assert_eq!(
        missing_tables(&required, &without_loc),
        ["text/localisation.loc", "text/ui.loc"]
    );
}

#[test]
fn required_only_imports_drop_optional_tables_and_ignore_them_as_drift() {
    let mut list = entries();
    let ignored = keep_required(&mut list, &[requirement("Factions_Tables")]);
    assert_eq!(ignored, ["units_tables"]);
    let members: Vec<&str> = list.iter().map(|e| e.member.as_str()).collect();
    assert_eq!(
        members,
        ["db/factions_tables/mod_factions", "text/localisation.loc", "text/ui.loc"]
    );

    // A dataset published without extras does not report the folder's optional tables as drift, but a new
    // file of an imported table still is.
    let e = env();
    let mut spec = spec("Mod", &e.input);
    spec.extras = false;
    let binding = publish(&e.ws, &spec, &analysis(list, &e.input), &quiet()).unwrap();
    let dataset = load(&e.ws, &binding).unwrap();
    assert!(dataset.manifest.required_only);
    assert!(dataset.changes().unwrap().is_empty(), "{:?}", dataset.changes().unwrap());
    write_folder(
        &e.input,
        &[db_entry("db/factions_tables/more.tsv", "db/factions_tables/more", 1)],
    );
    let changes = dataset.changes().unwrap();
    assert!(changes.iter().any(|c| c.contains("more")), "{changes:?}");
}

const DEMO_SCHEMA: &str = r#"(
    version: 1,
    definitions: {
        "demo_tables": [(
            version: 1,
            fields: [
                (name: "key", field_type: StringU8, is_key: true, is_reference: None),
                (name: "n", field_type: I32, is_key: false, is_reference: None),
            ],
        )],
        "other_tables": [(
            version: 1,
            fields: [
                (name: "key", field_type: StringU8, is_key: true, is_reference: None),
                (name: "n", field_type: I32, is_key: false, is_reference: None),
            ],
        )],
    },
)"#;

fn raw(rel: &str, text: &str) -> RawFile {
    RawFile {
        rel: rel.to_owned(),
        bytes: text.as_bytes().to_vec(),
    }
}

#[test]
fn unreadable_or_repeated_files_of_tables_the_campaign_does_not_read_never_block_the_import() {
    let schema = db::Schema::parse(DEMO_SCHEMA).unwrap();
    let required = BTreeSet::from(["demo_tables".to_owned()]);
    let shared = "key\tn\n#other_tables;1;db/other_tables/shared\na\t1\n";
    let files = vec![
        raw("demo_tables/m.tsv", "key\tn\n#demo_tables;1;db/demo_tables/m\na\t1\n"),
        raw(
            "other_tables/broken.tsv",
            "key\tn\n#other_tables;1;db/other_tables/broken\na\tnot a number\n",
        ),
        raw("other_tables/one.tsv", shared),
        raw("other_tables/two.tsv", shared),
    ];
    let (mut problems, mut notes) = (Vec::new(), Vec::new());
    let kept = interpret(files, &schema, &required, &mut problems, &mut notes);
    assert!(problems.is_empty(), "{problems:?}");
    assert_eq!(
        kept.iter().map(|e| e.member.as_str()).collect::<Vec<_>>(),
        ["db/demo_tables/m"]
    );
    assert_eq!(notes.len(), 2, "{notes:?}");
}

#[test]
fn unreadable_or_repeated_files_of_required_tables_still_block_the_import() {
    let schema = db::Schema::parse(DEMO_SCHEMA).unwrap();
    let required = BTreeSet::from(["demo_tables".to_owned()]);
    let (mut problems, mut notes) = (Vec::new(), Vec::new());
    let kept = interpret(
        vec![raw(
            "demo_tables/m.tsv",
            "key\tn\n#demo_tables;1;db/demo_tables/m\na\tnot a number\n",
        )],
        &schema,
        &required,
        &mut problems,
        &mut notes,
    );
    assert!(kept.is_empty());
    assert_eq!(problems.len(), 1, "{problems:?}");

    let same = "key\tn\n#demo_tables;1;db/demo_tables/m\na\t1\n";
    let mut problems = Vec::new();
    interpret(
        vec![raw("demo_tables/m.tsv", same), raw("demo_tables/copy.tsv", same)],
        &schema,
        &required,
        &mut problems,
        &mut notes,
    );
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(notes.is_empty());
}
