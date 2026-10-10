use super::effective::{self, MemberRows};
use super::*;
use crate::{
    db::{Field, FieldType, Schema, Table},
    model::Resolution,
};
use serde_json::json;

const SCHEMA: &str = r#"(
    version: 1,
    definitions: {
        "demo_tables": [
            (
                version: 1,
                fields: [
                    (name: "key", field_type: StringU8, is_key: true, is_reference: None),
                    (name: "n", field_type: I32, is_key: false, is_reference: None),
                ],
            ),
        ],
    },
)"#;

fn demo_table(rows: &[(&str, i32)]) -> Table {
    let field = |name: &str, ty: FieldType, is_key: bool| Field {
        name: name.to_owned(),
        ty,
        is_key,
        reference: None,
    };
    Table {
        name: "demo_tables".into(),
        guid: None,
        version: 1,
        version_marker: true,
        flag: 1,
        fields: vec![
            field("key", FieldType::StringU8, true),
            field("n", FieldType::I32, false),
        ],
        rows: rows
            .iter()
            .map(|(k, n)| vec![Cell::Str((*k).to_owned()), Cell::I32(*n)])
            .collect(),
    }
}

fn member(path: &str, rows: &[(&str, i32)]) -> MemberRows {
    MemberRows {
        path: path.to_owned(),
        origin: format!("TSV {path}"),
        sha256: format!("sha-{path}"),
        table: demo_table(rows),
        removed: BTreeSet::new(),
    }
}

#[test]
fn napoleon_unit_keys_without_schema_key_markers_still_conflict() {
    let mut members = vec![
        member(
            "db/unit_stats_land_tables/data",
            &[("fusiliers", 211), ("guards", 160)],
        ),
        member("db/unit_stats_land_tables/mod", &[("fusiliers", 223)]),
    ];
    for member in &mut members {
        member.table.name = "unit_stats_land_tables".into();
        member.table.fields[0].is_key = false;
        member.table.fields[1].name = "men".into();
    }
    let duplicates = effective::duplicates("unit_stats_land_tables", &members).unwrap();
    let conflicts = duplicates
        .cross
        .values()
        .next()
        .expect("changed men must conflict by unit key");
    effective::keep_member(&mut members, conflicts, 1);
    assert_eq!(
        members[0]
            .live_rows()
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>(),
        vec![vec![Cell::Str("guards".into()), Cell::I32(160)]]
    );
    assert_eq!(
        members[1]
            .live_rows()
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>(),
        vec![vec![Cell::Str("fusiliers".into()), Cell::I32(223)]]
    );
}

#[test]
fn regiment_name_pools_keep_distinct_names_and_groups_but_detect_changed_order() {
    let mut pool = member("db/unit_regiment_names_tables/data", &[]);
    pool.table.name = "unit_regiment_names_tables".into();
    pool.table.fields = ["unit_caste", "name_group", "unit_name", "name_order"]
        .into_iter()
        .enumerate()
        .map(|(i, name)| Field {
            name: name.into(),
            ty: if i == 3 {
                FieldType::I32
            } else {
                FieldType::StringU8
            },
            is_key: i == 0,
            reference: None,
        })
        .collect();
    let row = |group: &str, name: &str, order| {
        vec![
            Cell::Str("infantry_line".into()),
            Cell::Str(group.into()),
            Cell::Str(name.into()),
            Cell::I32(order),
        ]
    };
    pool.table.rows = vec![
        row("english", "first", 1),
        row("english", "second", 2),
        row("french", "first", 1),
    ];
    let mut override_ = pool.clone();
    override_.path = "db/unit_regiment_names_tables/override".into();
    override_.table.rows = vec![row("english", "first", 9)];
    let mut members = vec![pool, override_];
    let duplicates = effective::duplicates("unit_regiment_names_tables", &members).unwrap();
    assert!(duplicates.within.is_empty());
    let conflicts = &duplicates.cross[&vec![0, 1]];
    assert_eq!(conflicts.len(), 1);
    effective::keep_member(&mut members, conflicts, 1);
    assert_eq!(
        members[0]
            .live_rows()
            .map(|(_, r)| r.clone())
            .collect::<Vec<_>>(),
        vec![row("english", "second", 2), row("french", "first", 1)]
    );
    assert_eq!(
        members[1].live_rows().next().unwrap().1,
        &row("english", "first", 9)
    );
}

#[test]
fn identical_duplicates_across_members_are_harmless() {
    let members = vec![
        member("db/demo_tables/data", &[("a", 1), ("b", 2)]),
        member("db/demo_tables/mod", &[("a", 1), ("c", 3)]),
    ];
    let dup = effective::duplicates("demo_tables", &members).unwrap();
    assert!(dup.cross.is_empty());
    assert!(dup.within.is_empty());
}

#[test]
fn conflicting_keys_group_by_member_set_and_a_choice_removes_only_the_losers() {
    let mut members = vec![
        member("db/demo_tables/data", &[("a", 1), ("b", 2), ("c", 3)]),
        member("db/demo_tables/mod_a", &[("a", 10), ("b", 2), ("d", 4)]),
        member("db/demo_tables/mod_b", &[("c", 30)]),
    ];
    let dup = effective::duplicates("demo_tables", &members).unwrap();
    // `a` is held by data + mod_a, `c` by data + mod_b: two different member sets
    assert_eq!(dup.cross.len(), 2);
    let ab = dup.cross.get(&vec![0, 1]).unwrap();
    assert_eq!(ab.len(), 1);
    assert_eq!(ab[0].id, "a");
    let cb = dup.cross.get(&vec![0, 2]).unwrap();
    assert_eq!(cb[0].id, "c");

    // the mod's definition wins for `a`; the vanilla definition wins for `c`
    effective::keep_member(&mut members, ab, 1);
    effective::keep_member(&mut members, cb, 0);
    assert_eq!(members[0].removed, BTreeSet::from([0usize]));
    assert!(members[1].removed.is_empty());
    assert_eq!(members[2].removed, BTreeSet::from([0usize]));
    // every key now has exactly one live definition
    let again = effective::duplicates("demo_tables", &members).unwrap();
    assert!(again.cross.is_empty() && again.within.is_empty());
}

#[test]
fn a_key_held_twice_inside_one_member_cannot_be_resolved_by_choosing_a_member() {
    let members = vec![
        member("db/demo_tables/data", &[("a", 1)]),
        member("db/demo_tables/mod", &[("a", 5), ("a", 6)]),
    ];
    let dup = effective::duplicates("demo_tables", &members).unwrap();
    assert_eq!(dup.within.len(), 1);
    assert_eq!(dup.within[0].id, "a");
    assert!(dup.cross.is_empty());
}

#[test]
fn rewritten_members_decode_without_the_removed_rows_and_keep_unrelated_rows() {
    let schema = Schema::parse(SCHEMA).unwrap();
    let mut members = vec![member(
        "db/demo_tables/mod",
        &[("a", 1), ("b", 2), ("c", 3)],
    )];
    members[0].removed.insert(1);
    let rewritten = effective::rewrite(&members).unwrap();
    assert_eq!(rewritten.len(), 1);
    assert_eq!(rewritten[0].0, "db/demo_tables/mod");
    assert_eq!(rewritten[0].2, 1);
    let back = db::decode("demo_tables", &rewritten[0].1, &schema).unwrap();
    assert_eq!(
        back.rows,
        vec![
            vec![Cell::Str("a".into()), Cell::I32(1)],
            vec![Cell::Str("c".into()), Cell::I32(3)]
        ]
    );
    // untouched members are not rewritten at all
    assert!(
        effective::rewrite(&[member("db/demo_tables/other", &[("x", 1)])])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn port_collisions_distinguish_identical_from_conflicting_rows() {
    let members = vec![member("db/demo_tables/mod", &[("a", 1), ("b", 2)])];
    let port = content::PortTable::from_table(demo_table(&[("a", 1), ("b", 99), ("z", 0)]), "seed")
        .unwrap();
    assert!(effective::same_layout(&port, &members));
    let collisions = effective::port_collisions(&port, &members);
    assert_eq!(collisions.len(), 2);
    let a = collisions.iter().find(|c| c.id == "a").unwrap();
    let b = collisions.iter().find(|c| c.id == "b").unwrap();
    assert!(a.identical);
    assert!(!b.identical);
    assert_eq!(b.target, vec![(0, 1)]);
    // a removed target row no longer collides
    let mut members = members;
    members[0].removed.insert(1);
    assert_eq!(effective::port_collisions(&port, &members).len(), 1);
    // a different table version is not comparable
    let mut other = members.clone();
    other[0].table.version = 2;
    assert!(!effective::same_layout(&port, &other));
}

fn campaign_members(table: &str, a: &[(&str, i32)], b: &[(&str, i32)]) -> Vec<MemberRows> {
    let mut members = vec![
        member(&format!("db/{table}/a"), a),
        member(&format!("db/{table}/b"), b),
    ];
    for m in &mut members {
        m.table.name = table.into();
    }
    members
}

#[test]
fn conflicts_in_tables_the_campaign_does_not_read_only_warn_and_change_nothing() {
    let project = Project::default();
    let mut st = State::new(&project);
    let members = vec![
        member(
            "db/demo_tables/a",
            &[("within", 1), ("within", 2), ("cross", 3)],
        ),
        member("db/demo_tables/b", &[("cross", 4), ("unrelated", 5)]),
    ];
    let original: Vec<_> = members
        .iter()
        .map(|m| db::encode(&m.table).unwrap())
        .collect();
    note_unread_duplicates(&mut st, "demo_tables", &members);
    assert!(st.drafts.is_empty(), "nothing to decide, nothing to block");
    assert!(st.warnings.iter().any(|w| w.starts_with("demo_tables:")));
    assert_eq!(
        members
            .iter()
            .map(|m| db::encode(&m.table).unwrap())
            .collect::<Vec<_>>(),
        original
    );
    assert!(effective::rewrite(&members).unwrap().is_empty());
}

#[test]
fn a_conflict_inside_one_member_on_a_key_the_campaign_reads_blocks_without_a_choice() {
    const TABLE: &str = "unit_stats_land_tables";
    let consumed = Consumed {
        units: BTreeSet::from(["unit".to_owned()]),
        ..Consumed::default()
    };
    let project = Project::default();
    let mut st = State::new(&project);
    let mut members = vec![member(
        "db/unit_stats_land_tables/data",
        &[("unit", 160), ("unit", 200), ("spare", 1), ("spare", 2)],
    )];
    members[0].table.name = TABLE.into();
    reconcile_duplicates(&mut st, TABLE, &mut members, &|id| consumed.reads(TABLE, id)).unwrap();
    assert_eq!(st.drafts.len(), 1);
    assert_eq!(st.drafts[0].severity, "blocker");
    assert!(!st.drafts[0].resolved);
    assert!(st.drafts[0].choices.is_empty());
    assert!(effective::rewrite(&members).unwrap().is_empty());
}

#[test]
fn a_member_choice_removes_only_the_losing_rows_of_keys_the_campaign_reads() {
    const TABLE: &str = "unit_stats_land_tables";
    let consumed = Consumed {
        units: BTreeSet::from(["fusiliers".to_owned()]),
        ..Consumed::default()
    };
    let project = Project::default();
    let mut members = campaign_members(
        TABLE,
        &[("fusiliers", 160), ("guards", 80)],
        &[("fusiliers", 200), ("guards", 90)],
    );
    let mut st = State::new(&project);
    reconcile_duplicates(&mut st, TABLE, &mut members, &|id| consumed.reads(TABLE, id)).unwrap();
    // `guards` conflicts as well, but the conversion never reads it: only `fusiliers` needs a decision
    assert_eq!(st.drafts.len(), 1);
    let draft = &st.drafts[0];
    assert_eq!(draft.severity, "blocker");
    assert!(!draft.resolved);
    assert!(draft.detail.contains("fusiliers") && !draft.detail.contains("guards"));
    assert!(effective::rewrite(&members).unwrap().is_empty());

    let mut decided = project.clone();
    decided.resolutions.insert(
        draft.id.clone(),
        Resolution {
            input_hash: draft.hash.clone(),
            choice: "member:db/unit_stats_land_tables/b".into(),
        },
    );
    let mut resolved = State::new(&decided);
    reconcile_duplicates(&mut resolved, TABLE, &mut members, &|id| consumed.reads(TABLE, id)).unwrap();
    assert!(resolved.drafts[0].resolved);
    assert_eq!(members[0].removed, BTreeSet::from([0]));
    assert!(members[1].removed.is_empty());
    assert_eq!(
        members[0]
            .live_rows()
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>(),
        vec![vec![Cell::Str("guards".into()), Cell::I32(80)]]
    );
}

#[test]
fn only_conflicts_on_keys_of_generated_rows_need_a_decision_in_other_tables() {
    let project = Project::default();
    let members = vec![
        member("db/demo_tables/a", &[("key", 1), ("spare", 1)]),
        member("db/demo_tables/b", &[("key", 2), ("spare", 2)]),
    ];
    let mut generated = State::new(&project);
    reconcile_duplicates(
        &mut generated,
        "demo_tables",
        &mut members.clone(),
        &|id| id == "key",
    )
    .unwrap();
    assert_eq!(generated.drafts.len(), 1);
    assert_eq!(generated.drafts[0].severity, "blocker");
    assert_eq!(generated.drafts[0].choices.len(), 2);

    let mut elsewhere = State::new(&project);
    let mut kept = members.clone();
    reconcile_duplicates(&mut elsewhere, "demo_tables", &mut kept, &|id| id == "new").unwrap();
    assert!(elsewhere.drafts.is_empty());
    assert!(effective::rewrite(&kept).unwrap().is_empty());
}

#[test]
fn the_campaign_reads_only_the_rows_the_resolver_actually_uses_after_mappings_and_owners() {
    let mut rules = recycle_rules(
        json!({"france": {}, "prussia": {}}),
        json!({"paris": {"owner": "france", "rebel_faction": "french_rebels"}}),
        json!({
            "france": [{"kind": "army", "region": "paris", "units": ["Stand_In"]}],
            "prussia": [{"kind": "army", "region": "paris", "units": ["Prussian_Only"]}]
        }),
    );
    rules.slot_types = json!({"a": "settlement", "b": "settlement_fortification"});
    rules.buildings = json!({"etw_building": "level_1"});
    rules.buildings_by_slot_type = json!({
        "etw_building": {"settlement": "slot_default", "settlement_fortification": "fort_level"},
        "etw_other": {"settlement": "other_default"}
    });
    let mut project = Project::default();
    project.owners.insert("delhi".into(), "hessen".into());
    project.mappings.factions.insert("france".into(), "gaul".into());
    project.mappings.units.insert("Stand_In".into(), "Target_Unit".into());
    project.mappings.buildings.insert("other_default".into(), "mapped_level".into());
    // the target defines gaul, hessen and (as an unrelated faction) france, but not prussia
    let existing = BTreeSet::from(["gaul".to_owned(), "hessen".to_owned(), "france".to_owned()]);
    let consumed = Consumed::new(&rules, &project, &existing);
    for (table, id, read) in [
        ("factions_tables", "gaul", true),
        ("factions_tables", "hessen", true),
        // the default replaced by the mapping, and a faction nobody uses, are not read
        ("factions_tables", "france", false),
        ("factions_tables", "unrelated", false),
        ("regions_tables", "paris", true),
        ("regions_tables", "unrelated", false),
        ("unit_stats_land_tables", "Target_Unit", true),
        ("unit_stats_naval_tables", "Target_Unit", true),
        ("unit_stats_land_tables", "Stand_In", false),
        // the army of a faction the target does not define is omitted by the resolver
        ("unit_stats_land_tables", "Prussian_Only", false),
        ("units_to_exclusive_faction_permissions_tables", "Target_Unit\u{1f}gaul", true),
        ("units_to_exclusive_faction_permissions_tables", "Stand_In\u{1f}gaul", false),
        ("building_levels_tables", "slot_default", true),
        ("building_levels_tables", "mapped_level", true),
        ("building_levels_tables", "other_default", false),
        // fortification slots never receive buildings, and rules.buildings values are only existence-checked
        ("building_levels_tables", "fort_level", false),
        ("building_levels_tables", "level_1", false),
        // unit, technology and building-chain rows are only tested for existence
        ("units_tables", "Target_Unit", false),
        ("technologies_tables", "gaul", false),
        ("building_chain_to_slots_tables", "slot_default", false),
    ] {
        assert_eq!(consumed.reads(table, id), read, "{table} {id}");
    }
}

#[test]
fn a_conflict_on_a_default_the_project_mapped_away_does_not_block() {
    const TABLE: &str = "unit_stats_land_tables";
    let rules = recycle_rules(
        json!({"france": {}}),
        json!({}),
        json!({"france": [{"kind": "army", "region": "paris", "units": ["Stand_In"]}]}),
    );
    let existing = BTreeSet::from(["france".to_owned()]);
    let conflicting = || {
        campaign_members(TABLE, &[("Stand_In", 160), ("Target_Unit", 80)], &[("Stand_In", 200), ("Target_Unit", 90)])
    };

    let project = Project::default();
    let consumed = Consumed::new(&rules, &project, &existing);
    let mut unmapped = State::new(&project);
    reconcile_duplicates(&mut unmapped, TABLE, &mut conflicting(), &|id| consumed.reads(TABLE, id)).unwrap();
    assert_eq!(unmapped.drafts.len(), 1, "the stand-in is read by the starting army");
    assert!(unmapped.drafts[0].detail.contains("Stand_In"));

    let mut mapped = project.clone();
    mapped.mappings.units.insert("Stand_In".into(), "Target_Unit".into());
    let consumed = Consumed::new(&rules, &mapped, &existing);
    let mut st = State::new(&mapped);
    let mut members = conflicting();
    reconcile_duplicates(&mut st, TABLE, &mut members, &|id| consumed.reads(TABLE, id)).unwrap();
    assert_eq!(st.drafts.len(), 1, "only the mapped target is read now");
    assert!(st.drafts[0].detail.contains("Target_Unit") && !st.drafts[0].detail.contains("Stand_In"));
    assert!(effective::rewrite(&members).unwrap().is_empty());
}

#[test]
fn input_hashes_bind_every_part_and_part_boundaries() {
    let digest = |parts: &[&str]| {
        let mut h = Hasher::new("t");
        for p in parts {
            h.part(p);
        }
        h.finish()
    };
    assert_eq!(digest(&["ab", "c"]), digest(&["ab", "c"]));
    assert_ne!(digest(&["ab", "c"]), digest(&["a", "bc"]));
    assert_ne!(digest(&["ab", "c"]), digest(&["ab", "d"]));
    assert_ne!(Hasher::new("x").finish(), Hasher::new("y").finish());
}

fn draft(id: &str, hash: &str) -> Draft {
    Draft::blocker(
        id.into(),
        hash.into(),
        "db_key",
        "title".into(),
        "detail".into(),
    )
    .choices(vec!["keep_target".into(), "use_port".into()])
}

#[test]
fn a_stored_choice_applies_only_to_the_inputs_it_was_made_for() {
    let mut project = Project::default();
    project.resolutions.insert(
        "port_row:x_tables".into(),
        Resolution {
            input_hash: "h1".into(),
            choice: "use_port".into(),
        },
    );
    project.resolutions.insert(
        "port_row:y_tables".into(),
        Resolution {
            input_hash: "h2".into(),
            choice: "member:db/y/other".into(),
        },
    );
    let mut st = State::new(&project);
    assert_eq!(
        st.decide(draft("port_row:x_tables", "h1")).as_deref(),
        Some("use_port")
    );
    // same id, but the target changed since the choice was made
    assert_eq!(st.decide(draft("port_row:x_tables", "h1-changed")), None);
    // the stored value is not one of the offered choices
    assert_eq!(st.decide(draft("port_row:y_tables", "h2")), None);
    // no stored choice at all
    assert_eq!(st.decide(draft("port_row:z_tables", "h3")), None);
    let resolved: Vec<bool> = st.drafts.iter().map(|d| d.resolved).collect();
    assert_eq!(resolved, [true, false, false, false]);
    assert_eq!(
        st.warnings.iter().filter(|w| w.contains("ignored")).count(),
        1
    );
}

fn ownership_problem(region: &str) -> campaign_profile::MappingProblem {
    campaign_profile::MappingProblem {
        field: "owners".into(),
        key: region.into(),
        kind: "existing_owner_required".into(),
        message: format!("{region} needs an existing target owner"),
    }
}

#[test]
fn absent_original_owners_are_pending_without_faction_mapping_requirements() {
    let assets = crate::pack::test_util::TempDir::new("pending_ownership");
    for sub in ["rules", "ui"] {
        fs::create_dir_all(assets.path().join(sub)).unwrap();
    }
    let project = Project::default();
    let mut st = State::new(&project);
    st.catalog.factions = vec!["britain".into(), "france".into()];
    st.profile_problem(ownership_problem("delhi"));
    st.profile_problem(ownership_problem("algiers"));
    let analysis = st.conclude(assets.path(), None).unwrap().analysis;
    assert_eq!(analysis.unassigned_regions, ["algiers", "delhi"]);
    assert!(analysis.issues.is_empty(), "absent originals are not invalid mappings");
    assert!(!analysis.ready, "owners must be selected before campaign export");
    assert_eq!(analysis.catalog.factions, ["britain", "france"], "available recipients remain selectable");
}

#[test]
fn invalid_explicit_owners_and_unit_mappings_remain_errors() {
    let assets = crate::pack::test_util::TempDir::new("invalid_ownership");
    for sub in ["rules", "ui"] {
        fs::create_dir_all(assets.path().join(sub)).unwrap();
    }
    let mut project = Project::default();
    project.owners.insert("delhi".into(), "not_a_target_faction".into());
    let mut st = State::new(&project);
    st.profile_problem(ownership_problem("delhi"));
    st.profile_problem(campaign_profile::MappingProblem {
        field: "units".into(), key: "missing_unit".into(), kind: "unmapped_missing".into(),
        message: "starting unit has no target definition".into(),
    });
    let analysis = st.conclude(assets.path(), None).unwrap().analysis;
    assert!(analysis.unassigned_regions.is_empty(), "an invalid explicit choice is not an untouched original owner");
    assert!(!analysis.ready);
    assert_eq!(analysis.blockers().count(), 2);
    let owner = analysis.issues.iter().find(|i| i.category == "ownership").unwrap();
    assert!(owner.mapping.is_none(), "ownership choices do not become faction mapping requirements");
    assert!(analysis.issues.iter().any(|i| i.category == "mapping" && i.mapping.as_ref().is_some_and(|m| m.field == "units")));
}

#[test]
fn the_project_digest_follows_decisions_but_not_the_name() {
    let base = Project::default();
    let mut renamed = base.clone();
    renamed.name = "Another".into();
    assert_eq!(project_digest(&base), project_digest(&renamed));
    let mut mapped = base.clone();
    mapped
        .mappings
        .factions
        .insert("france".into(), "gaul".into());
    assert_ne!(project_digest(&base), project_digest(&mapped));
    let mut resolved = base.clone();
    resolved.resolutions.insert(
        "id".into(),
        Resolution {
            input_hash: "h".into(),
            choice: "keep_target".into(),
        },
    );
    assert_ne!(project_digest(&base), project_digest(&resolved));
    let mut owned = base.clone();
    owned.owners.insert("france".into(), "britain".into());
    assert_ne!(project_digest(&base), project_digest(&owned));
}

#[test]
fn a_stale_dataset_binding_does_not_change_the_vanilla_conversion_but_binds_tsv_output() {
    let binding = crate::target::TargetBinding {
        id: "t1".into(),
        fingerprint: "f1".into(),
    };
    let vanilla = Project::default();
    let mut stale = vanilla.clone();
    stale.target = Some(binding.clone());
    assert_eq!(project_digest(&vanilla), project_digest(&stale));
    let mut tsv = stale.clone();
    tsv.target_mode = TargetMode::Tsv;
    assert_ne!(project_digest(&stale), project_digest(&tsv));
    let mut other = tsv.clone();
    other.target = Some(crate::target::TargetBinding {
        id: "t2".into(),
        fingerprint: "f2".into(),
    });
    assert_ne!(project_digest(&tsv), project_digest(&other));
}

#[test]
fn text_keys_are_grouped_by_the_members_that_define_them_differently() {
    let def = |m: &str, text: &str, tip: u8| (m.to_owned(), (text.to_owned(), tip));
    let mut by_key: BTreeMap<String, LocDefs> = BTreeMap::new();
    by_key.insert(
        "same".into(),
        vec![def("text/a.loc", "x", 0), def("text/b.loc", "x", 0)],
    );
    by_key.insert(
        "tip".into(),
        vec![def("text/a.loc", "x", 0), def("text/b.loc", "x", 1)],
    );
    by_key.insert(
        "text".into(),
        vec![def("text/a.loc", "x", 0), def("text/b.loc", "y", 0)],
    );
    by_key.insert(
        "three".into(),
        vec![
            def("text/a.loc", "x", 0),
            def("text/b.loc", "x", 0),
            def("text/c.loc", "z", 0),
        ],
    );
    by_key.insert("alone".into(), vec![def("text/a.loc", "x", 0)]);
    let groups = loc_groups(by_key);
    assert_eq!(groups.len(), 2);
    let ab = &groups[&vec!["text/a.loc".to_owned(), "text/b.loc".to_owned()]];
    assert_eq!(
        ab.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        ["text", "tip"]
    );
    let abc = &groups[&vec![
        "text/a.loc".to_owned(),
        "text/b.loc".to_owned(),
        "text/c.loc".to_owned(),
    ]];
    assert_eq!(abc[0].0, "three");
}

#[test]
fn a_rewritten_member_keeps_its_original_path_and_every_untouched_row() {
    let schema = Schema::parse(SCHEMA).unwrap();
    let mut members = vec![
        member("db/demo_tables/Mod_Part_1", &[("a", 1), ("b", 2)]),
        member("db/demo_tables/mod_part_2", &[("a", 9), ("c", 3)]),
    ];
    let dup = effective::duplicates("demo_tables", &members).unwrap();
    let conflicts = dup.cross.get(&vec![0, 1]).unwrap();
    effective::keep_member(&mut members, conflicts, 1);
    let rewritten = effective::rewrite(&members).unwrap();
    assert_eq!(rewritten.len(), 1);
    assert_eq!(rewritten[0].0, "db/demo_tables/Mod_Part_1");
    let back = db::decode("demo_tables", &rewritten[0].1, &schema).unwrap();
    assert_eq!(back.rows, vec![vec![Cell::Str("b".into()), Cell::I32(2)]]);
}

#[test]
fn unit_mapping_sources_are_the_ntw_stand_ins_not_etw_keys() {
    let rules = Rules {
        root: std::path::PathBuf::new(),
        regions: json!({}),
        factions: json!({"france": {}}),
        units: json!({"etw_a": "Stand_In", "etw_b": "Stand_In", "etw_c": null}),
        buildings: json!({"etw_building": "level_1"}),
        buildings_by_slot_type: json!({}),
        slot_types: json!({}),
        armies: json!({"france": [{"region": "x", "units": ["Army_Only", "Stand_In"]}]}),
    };
    let (factions, units, buildings) = source_catalog(&rules);
    assert_eq!(factions, ["france"]);
    // stand-ins from rules.units values and from the start armies; ETW keys are not mapping sources
    assert_eq!(units, ["Army_Only", "Stand_In"]);
    assert_eq!(buildings, ["etw_building", "level_1"]);
}

#[test]
fn adapters_may_only_supply_ui_scripts_and_the_template() {
    for ok in [
        "ui/campaign ui/layout_scripts/campaign_hud.luac",
        "script/campaign/mod.lua",
        "campaigns/wld_napoleon/scripting.lua",
        "campaigns/eur_napoleon/startpos.esf",
    ] {
        assert!(resources::adapter_path_allowed(ok), "{ok}");
    }
    for bad in [
        "db/units_tables/units",
        "text/localisation.loc",
        "campaigns/wld_napoleon/startpos.esf",
        "campaign_maps/nap_world/regions.esf",
        "campaigns/eur_napoleon/scripting.lua",
        "models/x.cs2",
    ] {
        assert!(!resources::adapter_path_allowed(bad), "{bad}");
    }
}

#[test]
fn an_unreadable_start_position_template_is_rejected() {
    assert!(resources::check_startpos(b"not an esf file").is_err());
    assert!(resources::check_startpos(&[]).is_err());
}

fn recycle_rules(factions: Value, regions: Value, armies: Value) -> Rules {
    Rules {
        root: std::path::PathBuf::new(),
        regions,
        factions,
        units: json!({}),
        buildings: json!({}),
        buildings_by_slot_type: json!({}),
        slot_types: json!({}),
        armies,
    }
}

#[test]
fn recycling_review_tracks_omitted_identities_and_changed_ownership() {
    let base = recycle_rules(
        json!({"france": {"source": "ntw"}, "mughal": {"source": "etw"}}),
        json!({
            "paris": {"owner": "france", "rebel_faction": "french_rebels"},
            "delhi": {"owner": "mughal", "rebel_faction": "mughal"},
            "agra": {"owner": "mughal", "rebel_faction": "mughal"}
        }),
        json!({
            "france": [{"kind": "army", "region": "paris", "units": []}],
            "mughal": [
                {"kind": "army", "region": "delhi", "units": []},
                {"kind": "navy", "region": "agra", "units": []}
            ]
        }),
    );
    // the user gave delhi to the existing target faction hessen, agra to france; mughal is gone
    let resolved = recycle_rules(
        json!({
            "france": {"source": "ntw", "etw_source": "france"},
            "hessen": {"source": "ntw", "etw_source": "hessen"}
        }),
        json!({
            "paris": {"owner": "france", "rebel_faction": "french_rebels"},
            "delhi": {"owner": "hessen", "rebel_faction": "hessen_rebels"},
            "agra": {"owner": "france", "rebel_faction": "french_rebels"}
        }),
        json!({"france": [{"kind": "army", "region": "paris", "units": []}]}),
    );
    let project = Project::default();
    let mut st = State::new(&project);
    recycle_report(&mut st, &base, &resolved);
    let omitted: Vec<_> = st
        .changes
        .iter()
        .filter(|c| c.operation == "omit_faction")
        .collect();
    assert_eq!(omitted.len(), 1);
    assert_eq!(omitted[0].key, "mughal");
    let assigned: Vec<&str> = st
        .changes
        .iter()
        .filter(|c| c.operation == "assign_owner")
        .map(|c| c.key.as_str())
        .collect();
    assert_eq!(assigned, ["france", "hessen"]);
}

fn reference_scope_fixture() -> (crate::snapshot::tests::Installs, Rules, Snapshot, db::Table) {
    let installs = crate::snapshot::tests::installs();
    crate::snapshot::import(&installs.etw, &installs.ntw, &installs.dest, &|_| {}).unwrap();
    let mut rules = recycle_rules(json!({}), json!({}), json!({}));
    rules.root = installs.dir.path().join("rules");
    std::fs::create_dir_all(&rules.root).unwrap();
    let schema_text = r#"(
        version: 1,
        definitions: {
            "demo_tables": [(
                version: 1,
                fields: [(name: "key", field_type: StringU8, is_key: true,
                    is_reference: Some(("provider", "key")))],
            )],
            "provider_tables": [(
                version: 1,
                fields: [(name: "key", field_type: StringU8, is_key: true, is_reference: None)],
            )],
        },
    )"#;
    std::fs::write(rules.root.join("schema_nap.ron"), schema_text).unwrap();
    let schema = Schema::parse(schema_text).unwrap();
    let make = |name: &str| {
        Table::from_json_rows(
            name,
            1,
            &schema.definition(name, 1).unwrap().fields,
            &[json!({"key": "known"})],
        )
        .unwrap()
    };
    let demo = make("demo_tables");
    let provider = make("provider_tables");
    let original = Snapshot::open(&installs.dest).unwrap();
    let base = original
        .with_ntw_members(
            BTreeMap::from([
                (
                    "db/demo_tables/data".into(),
                    (db::encode(&demo).unwrap(), "fixture".into()),
                ),
                (
                    "db/provider_tables/data".into(),
                    (db::encode(&provider).unwrap(), "fixture".into()),
                ),
            ]),
            original.names(Game::Ntw, "db/"),
        )
        .unwrap();
    (installs, rules, base, demo)
}

#[test]
fn unchanged_target_references_do_not_expand_imports_but_generated_rows_require_their_provider() {
    let (_installs, rules, base, demo) = reference_scope_fixture();
    let target = base
        .with_ntw_members(BTreeMap::new(), vec!["db/provider_tables/data".into()])
        .unwrap();
    let scan = effective::scan(&target, &base, &rules).unwrap();
    let project = Project::default();
    let mut st = State::new(&project);
    target_references(&mut st, &target, &base, &rules, &scan);
    assert!(st.drafts.iter().all(|d| d.severity != "blocker"));
    assert!(!target.exists(Game::Ntw, "db/provider_tables/data"));

    // The same reference in generated rows is a real conversion dependency, not an optional mod reference.
    let generated = BTreeMap::from([("demo_tables".into(), demo)]);
    let missing = missing_dependencies(
        &generated,
        &BTreeSet::from(["demo_tables".into()]),
        &base,
        &rules,
    );
    st.block_missing_dependencies(&missing);
    assert!(
        st.drafts
            .iter()
            .any(|d| d.id == "target:missing_table:provider_tables" && d.severity == "blocker")
    );
}

#[test]
fn a_dangling_reference_in_a_supplied_table_is_reported_but_never_blocks() {
    let (_installs, rules, base, _demo) = reference_scope_fixture();
    let mut provider = (*db::load_table(&base, Game::Ntw, &rules, "provider").unwrap()).clone();
    provider.rows[0][0] = Cell::Str("different".into());
    let target = base
        .with_ntw_overlay(
            BTreeMap::from([(
                "db/provider_tables/data".into(),
                db::encode(&provider).unwrap(),
            )]),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
    let scan = effective::scan(&target, &base, &rules).unwrap();
    let project = Project::default();
    let mut st = State::new(&project);
    target_references(&mut st, &target, &base, &rules, &scan);
    assert!(st.drafts.is_empty(), "no decision or blocker for a record the mod itself lacks");
    assert!(st.warnings.iter().any(|w| w.contains("demo_tables")));
}

/// A TSV-dataset target with unchanged installed tables (unit stats, factions, an optional table) and two
/// members of `demo_tables` that a collision resolution rewrote: one keeps a row, one lost every row.
struct ExportFixture {
    _installs: crate::snapshot::tests::Installs,
    prepared: Prepared,
    /// What `content::build_reconciled` hands to `complete_members`.
    generated: BTreeMap<String, Vec<u8>>,
}

const RESOURCE: &str = "ui\\campaign ui\\layout_scripts\\campaign_hud.luac";
const EXTRA_LOC: &str = "text\\extra.loc";
const KEPT_ROW_MEMBER: &str = "db\\demo_tables\\data";
const TOMBSTONE_MEMBER: &str = "db\\demo_tables\\mod";
const UNCHANGED_TABLES: [&str; 3] = ["unit_stats_land_tables", "factions_tables", "ancillaries_tables"];

fn export_fixture() -> ExportFixture {
    let (installs, rules, base, _demo) = reference_scope_fixture();
    let mut installed: BTreeMap<String, (Vec<u8>, String)> = BTreeMap::new();
    for table in UNCHANGED_TABLES {
        let mut t = demo_table(&[("row", 1)]);
        t.name = table.into();
        installed.insert(
            format!("db/{table}/data"),
            (db::encode(&t).unwrap(), "fixture".into()),
        );
    }
    let loc = db::encode_loc(&[], 1).unwrap();
    for text in ["ui", "localisation", "extra"] {
        installed.insert(format!("text/{text}.loc"), (loc.clone(), "fixture".into()));
    }
    let target = base.with_ntw_members(installed, Vec::new()).unwrap();

    // the rewrite the analysis would plan: row `a` removed from both members
    let mut members = vec![
        member("db/demo_tables/data", &[("a", 1), ("b", 2)]),
        member("db/demo_tables/mod", &[("a", 10)]),
    ];
    for m in &mut members {
        m.removed.insert(0);
    }
    let shadow: BTreeMap<String, Vec<u8>> = effective::rewrite(&members)
        .unwrap()
        .into_iter()
        .map(|(path, bytes, _)| (path.replace('/', "\\"), bytes))
        .collect();
    assert_eq!(
        shadow.keys().map(String::as_str).collect::<Vec<_>>(),
        [KEPT_ROW_MEMBER, TOMBSTONE_MEMBER]
    );

    let mut generated: BTreeMap<String, Vec<u8>> = content::port_table_names()
        .into_iter()
        .map(|table| {
            let mut t = demo_table(&[("row", 1)]);
            t.name = table.clone();
            (
                format!("db\\{table}\\{}", content::TABLE_FILE),
                db::encode(&t).unwrap(),
            )
        })
        .collect();
    generated.insert("text\\ui.loc".into(), loc.clone());
    generated.insert("text\\localisation.loc".into(), loc);

    let prepared = Prepared {
        snapshot: target.clone(),
        rules,
        analysis: Analysis::default(),
        resources: BTreeMap::from([(RESOURCE.to_owned(), b"hud".to_vec())]),
        pack_name: "export_fixture.pack".into(),
        dependencies: json!({}),
        plan: Plan {
            modded: true,
            keep_target_tables: BTreeSet::new(),
            loc_keep_target: BTreeSet::new(),
            loc_strip: BTreeMap::new(),
            loc_decided: BTreeSet::new(),
            loc_vanilla_english: false,
            shadow,
            dataset_loc_members: ["text\\ui.loc", "text\\localisation.loc", EXTRA_LOC]
                .map(String::from)
                .into(),
            base: target,
            project_digest: String::new(),
        },
    };
    ExportFixture {
        _installs: installs,
        prepared,
        generated,
    }
}

#[test]
fn the_pack_holds_only_generated_campaign_tables_and_planned_rewrites() {
    let fx = export_fixture();
    let out = complete_members(&fx.prepared, fx.generated.clone()).unwrap();

    // nothing the installed target already has is exported unless it was rewritten, and a member that lost
    // every row stays in the pack as an empty replacement
    let allowed: BTreeSet<&str> = fx
        .generated
        .keys()
        .map(String::as_str)
        .filter(|name| name.starts_with("db\\"))
        .chain([KEPT_ROW_MEMBER, TOMBSTONE_MEMBER])
        .collect();
    let exported: BTreeSet<&str> = out
        .keys()
        .map(String::as_str)
        .filter(|name| name.starts_with("db\\"))
        .collect();
    assert_eq!(exported, allowed);
    for table in UNCHANGED_TABLES {
        assert!(!out.keys().any(|name| name.contains(table)), "{table}");
    }
    // text stays complete: the merged files plus the dataset text the build did not regenerate
    for name in ["text\\ui.loc", "text\\localisation.loc", EXTRA_LOC, RESOURCE] {
        assert!(out.contains_key(name), "{name}");
    }
}

#[test]
fn a_planned_member_that_is_not_produced_is_rejected() {
    let fx = export_fixture();
    let out = complete_members(&fx.prepared, fx.generated.clone()).unwrap();
    validate_members(&fx.prepared, &out).unwrap();

    let planned = fx
        .prepared
        .shadowed_members()
        .chain(fx.prepared.resources.keys())
        .chain(fx.prepared.plan.dataset_loc_members.iter());
    for name in planned {
        let mut short = out.clone();
        short.remove(name).unwrap();
        assert!(
            validate_members(&fx.prepared, &short).is_err(),
            "dropping {name} was accepted"
        );
    }
}

#[test]
fn database_members_nobody_planned_are_rejected() {
    let fx = export_fixture();
    let unplanned = [
        // an installed table copied along with the rest
        "db\\unit_stats_land_tables\\data",
        "db\\ancillaries_tables\\data",
        // a faction-creation table, even under the campaign member name
        "db\\factions_tables\\data",
        "db\\factions_tables\\nap_world",
        // a campaign table under another member name than the generated one
        "db\\regions_tables\\data",
    ];
    for name in unplanned {
        let mut generated = fx.generated.clone();
        generated.insert(name.to_owned(), b"unplanned".to_vec());
        assert!(
            complete_members(&fx.prepared, generated).is_err(),
            "{name} survived final assembly"
        );

        let mut out = complete_members(&fx.prepared, fx.generated.clone()).unwrap();
        out.insert(name.to_owned(), b"unplanned".to_vec());
        assert!(
            validate_members(&fx.prepared, &out).is_err(),
            "{name} passed validation"
        );
    }
}

#[test]
fn a_generated_member_cannot_silently_replace_a_planned_rewrite() {
    let mut fx = export_fixture();
    let clash = format!("db\\regions_tables\\{}", content::TABLE_FILE);
    fx.prepared.plan.shadow.insert(clash, vec![1]);
    assert!(complete_members(&fx.prepared, fx.generated.clone()).is_err());
}
