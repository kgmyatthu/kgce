use super::*;
use crate::db;

fn field(name: &str, ty: &str, key: bool, colour: Option<u8>) -> String {
    let colour = colour.map_or("None".to_owned(), |g| format!("Some({g})"));
    format!(
        r#"(name: "{name}", field_type: {ty}, is_key: {key}, default_value: None, is_filename: false, filename_relative_path: None, is_reference: None, lookup: None, description: "", ca_order: -1, is_bitwise: 0, enum_values: {{}}, is_part_of_colour: {colour}),"#
    )
}

fn definition(version: i32, fields: &[String]) -> String {
    format!(
        "(version: {version}, fields: [{}], localised_fields: [], localised_key_order: []),",
        fields.concat()
    )
}

fn schema() -> Schema {
    let demo_v2 = definition(
        2,
        &[
            field("key", "StringU8", true, None),
            field("flag", "Boolean", false, None),
            field("n", "I32", false, None),
            field("f", "F32", false, None),
            field("wide", "StringU16", false, None),
            field("opt", "OptionalStringU8", false, None),
            field("col_r", "I32", false, Some(1)),
            field("col_g", "I32", false, Some(1)),
            field("col_b", "I32", false, Some(1)),
            field("small", "I16", false, None),
            field("big", "I64", false, None),
        ],
    );
    let demo_v1 = definition(1, &[field("key", "StringU8", true, None)]);
    let palette = definition(
        0,
        &[
            field("id", "StringU8", true, None),
            field("r", "F32", false, Some(1)),
            field("g", "F32", false, Some(1)),
            field("b", "F32", false, Some(1)),
        ],
    );
    let mini = definition(
        1,
        &[
            field("key", "StringU8", true, None),
            field("n", "I32", false, None),
            field("s16", "I16", false, None),
            field("f", "F32", false, None),
            field("b", "Boolean", false, None),
            field("o", "OptionalStringU16", false, None),
            field("w", "StringU16", false, None),
        ],
    );
    let single = definition(0, &[field("only", "StringU8", true, None)]);
    let text = format!(
        r#"(version: 5, definitions: {{ "demo_tables": [{demo_v1} {demo_v2}], "palette_tables": [{palette}], "mini_tables": [{mini}], "single_tables": [{single}] }}, patches: {{}})"#
    );
    Schema::parse(&text).unwrap()
}

fn table(schema: &Schema, name: &str, version: i64, rows: Vec<Vec<Cell>>) -> Table {
    let def = schema.definition(name, version).unwrap();
    Table {
        name: name.to_owned(),
        guid: None,
        version: version as u32,
        version_marker: version != 0,
        flag: 1,
        fields: def.fields.clone(),
        rows,
    }
}

fn demo_table(schema: &Schema) -> Table {
    let row1 = vec![
        Cell::Str("alpha".into()),
        Cell::Bool(true),
        Cell::I32(-7),
        Cell::F32(0.5),
        Cell::Str("Gr\u{fc}\u{df}e \u{1f600}".into()),
        Cell::Null,
        Cell::I32(255),
        Cell::I32(128),
        Cell::I32(0),
        Cell::I16(3),
        Cell::I64(-9_000_000_000_000),
    ];
    let row2 = vec![
        Cell::Str("beta\tline\nbreak".into()),
        Cell::Bool(false),
        Cell::I32(i32::MAX),
        Cell::F32(0.00001),
        Cell::Str(String::new()),
        Cell::Str("x".into()),
        Cell::I32(0),
        Cell::I32(1),
        Cell::I32(2),
        Cell::I16(i16::MIN),
        Cell::I64(i64::MIN),
    ];
    table(schema, "demo_tables", 2, vec![row1, row2])
}

fn mini_text(rows: &[&str]) -> String {
    let mut lines = vec![
        "key\tn\ts16\tf\tb\to\tw".to_owned(),
        "#mini_tables;1;db/mini_tables/m\t\t\t\t\t\t".to_owned(),
    ];
    lines.extend(rows.iter().map(|r| (*r).to_owned()));
    lines.join("\n") + "\n"
}

const MINI_OK: &str = "k\t1\t2\t0.5\ttrue\topt\ttext";

fn decode_mini(text: &str) -> Result<Document> {
    decode("db/mini_tables/m.tsv", text.as_bytes(), &schema())
}

fn err(r: Result<Document>) -> String {
    format!("{:#}", r.unwrap_err())
}

fn db_doc(doc: Document) -> (String, Table) {
    match doc {
        Document::Db { member, table } => (member, table),
        other => panic!("expected a DB document, got {other:?}"),
    }
}

#[test]
fn export_has_the_rpfm_shape_and_roundtrips_into_identical_binary() {
    let schema = schema();
    let original = demo_table(&schema);
    let bytes = encode_db("db/demo_tables/Main", &original, &schema).unwrap();
    let expected = [
        "key\tflag\tn\tf\twide\topt\tsmall\tbig\tcol_hex",
        "#demo_tables;2;db/demo_tables/main\t\t\t\t\t\t\t\t",
        "alpha\ttrue\t-7\t0.5000\tGr\u{fc}\u{df}e \u{1f600}\t\t3\t-9000000000000\tFF8000",
        "beta\\\\tline\\\\nbreak\tfalse\t2147483647\t0.00001\t\tx\t-32768\t-9223372036854775808\t000102",
        "",
    ]
    .join("\n");
    assert_eq!(String::from_utf8(bytes.clone()).unwrap(), expected);

    let (member, back) = db_doc(decode("demo_tables/main.tsv", &bytes, &schema).unwrap());
    assert_eq!(member, "db/demo_tables/main");
    assert_eq!(back.rows, original.rows);
    assert_eq!(back.version, 2);
    assert_eq!(db::encode(&back).unwrap(), db::encode(&original).unwrap());
}

#[test]
fn reordered_header_bom_and_crlf_are_accepted() {
    let schema = schema();
    let original = demo_table(&schema);
    let canonical =
        String::from_utf8(encode_db("db/demo_tables/main", &original, &schema).unwrap()).unwrap();
    // Reverse the cell order of the header and every data row; the metadata row keeps its text first.
    let reversed: Vec<String> = canonical
        .lines()
        .enumerate()
        .map(|(i, l)| {
            if i == 1 {
                l.to_owned()
            } else {
                l.split('\t').rev().collect::<Vec<_>>().join("\t")
            }
        })
        .collect();
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend_from_slice((reversed.join("\r\n") + "\r\n").as_bytes());
    let (_, back) = db_doc(decode("x.tsv", &bytes, &schema).unwrap());
    assert_eq!(back.rows, original.rows);
    assert_eq!(back.fields[0].name, "key");
}

#[test]
fn optional_empty_is_absent_but_plain_strings_stay_empty() {
    let (_, t) = db_doc(decode_mini(&mini_text(&["k\t1\t2\t0.5\ttrue\t\t"])).unwrap());
    assert_eq!(t.rows[0][5], Cell::Null);
    assert_eq!(t.rows[0][6], Cell::Str(String::new()));
    assert_eq!(t.rows[0][0], Cell::Str("k".into()));
}

#[test]
fn escapes_are_reversed_and_unrepresentable_text_is_refused_on_export() {
    let (_, t) = db_doc(decode_mini(&mini_text(&["a\\\\tb\t1\t2\t0.5\ttrue\t\tx\\\\ny"])).unwrap());
    assert_eq!(t.rows[0][0], Cell::Str("a\tb".into()));
    assert_eq!(t.rows[0][6], Cell::Str("x\ny".into()));

    let schema = schema();
    for bad in ["has\rcr", "literal \\\\n inside", "literal \\\\t inside"] {
        let mut t = table(
            &schema,
            "mini_tables",
            1,
            vec![vec![
                Cell::Str(bad.into()),
                Cell::I32(1),
                Cell::I16(1),
                Cell::F32(1.0),
                Cell::Bool(true),
                Cell::Null,
                Cell::Str(String::new()),
            ]],
        );
        assert!(
            encode_db("db/mini_tables/m", &t, &schema).is_err(),
            "{bad:?}"
        );
        t.rows[0][0] = Cell::Str("fine".into());
        assert!(encode_db("db/mini_tables/m", &t, &schema).is_ok());
    }
}

#[test]
fn numeric_and_boolean_boundaries() {
    let cases = [
        ("k\t1\t32768\t0.5\ttrue\t\t", "line 3, column 3 (`s16`)"),
        ("k\t2147483648\t1\t0.5\ttrue\t\t", "column 2 (`n`)"),
        ("k\t1.0\t1\t0.5\ttrue\t\t", "column 2 (`n`)"),
        ("k\t\t1\t0.5\ttrue\t\t", "column 2 (`n`)"),
        ("k\t1\t1\tinf\ttrue\t\t", "column 4 (`f`)"),
        ("k\t1\t1\tNaN\ttrue\t\t", "column 4 (`f`)"),
        ("k\t1\t1\t1e39\ttrue\t\t", "column 4 (`f`)"),
        ("k\t1\t1\t0,5\ttrue\t\t", "column 4 (`f`)"),
        ("k\t1\t1\t0.5\tyes\t\t", "column 5 (`b`)"),
        ("k\t1\t1\t0.5\t2\t\t", "column 5 (`b`)"),
        ("k\t1\t1\t 0.5\ttrue\t\t", "column 4 (`f`)"),
    ];
    for (row, want) in cases {
        let e = err(decode_mini(&mini_text(&[row])));
        assert!(e.contains(want), "{row:?}: {e}");
        assert!(e.starts_with("db/mini_tables/m.tsv:"), "{e}");
    }
    for ok in [
        "k\t-2147483648\t-32768\t1e3\tTRUE\t\t",
        "k\t+5\t32767\t.5\t0\t\t",
        "k\t0\t0\t-0.0\tFalse\t\t",
        "k\t0\t0\t1\t1\t\t",
    ] {
        decode_mini(&mini_text(&[ok])).unwrap_or_else(|e| panic!("{ok:?}: {e:#}"));
    }
    let long = "x".repeat(0x1_0000);
    let e = err(decode_mini(&mini_text(&[&format!(
        "{long}\t1\t1\t1\ttrue\t\t"
    )])));
    assert!(e.contains("column 1 (`key`)") && e.contains("limit"), "{e}");
}

#[test]
fn malformed_structure_is_refused() {
    let ok = mini_text(&[MINI_OK]);
    decode_mini(&ok).unwrap();
    // zero data rows is a valid empty table
    let (_, empty) = db_doc(decode_mini(&mini_text(&[])).unwrap());
    assert!(empty.rows.is_empty());
    // trailing blank lines are fine, interior ones are not
    decode_mini(&format!("{ok}\n\n")).unwrap();
    assert!(err(decode_mini(&mini_text(&[MINI_OK, "", MINI_OK]))).contains("line 4: blank line"));
    assert!(
        err(decode_mini(&mini_text(&["k\t1\t2\t0.5\ttrue\topt"])))
            .contains("line 3: 6 cells but the table has 7")
    );
    assert!(
        err(decode_mini(&mini_text(&[
            "k\t1\t2\t0.5\ttrue\topt\ttext\textra"
        ])))
        .contains("8 cells")
    );
    assert!(
        err(decode_mini(&mini_text(&[
            "k\t1\t2\t0.5\ttrue\topt\tte\rxt"
        ])))
        .contains("line 3: bare carriage return")
    );
    assert!(decode("f.tsv", b"key\n", &schema()).is_err());
    assert!(decode("f.tsv", b"", &schema()).is_err());
    // invalid UTF-8 reports its line, UTF-16 is called out
    let mut bad = mini_text(&[MINI_OK]).into_bytes();
    bad.extend_from_slice(b"k\xff\n");
    assert!(err(decode("f.tsv", &bad, &schema())).contains("line 4: invalid UTF-8"));
    assert!(err(decode("f.tsv", &[0xff, 0xfe, b'k', 0], &schema())).contains("UTF-16"));
}

#[test]
fn header_must_match_one_definition_exactly() {
    let base = |header: &str| format!("{header}\n#mini_tables;1;db/mini_tables/m\t\t\t\t\t\n");
    for (header, want) in [
        ("key\tn\ts16\tf\tb\to", "missing columns [\"w\"]"),
        ("key\tn\ts16\tf\tb\to\tw\tz", "unexpected columns [\"z\"]"),
        ("key\tn\ts16\tf\tb\to\tW", "missing columns [\"w\"]"),
    ] {
        let e = err(decode_mini(&base(header)));
        assert!(e.contains("line 1") && e.contains(want), "{header}: {e}");
    }
    let e = err(decode_mini(&base("key\tn\tn\tf\tb\to\tw")));
    assert!(e.contains("`n` appears twice"), "{e}");
    // permuted header (and identical data order) is fine
    // (the metadata row is not reordered: its text is always the first cell)
    let (_, t) = db_doc(
        decode_mini(
            "w\to\tb\tf\ts16\tn\tkey\n#mini_tables;1;db/mini_tables/m\ntext\topt\ttrue\t0.5\t2\t1\tk\n",
        )
        .unwrap(),
    );
    assert_eq!(t.rows[0][0], Cell::Str("k".into()));
    assert_eq!(t.rows[0][1], Cell::I32(1));
    assert_eq!(t.rows[0][6], Cell::Str("text".into()));
}

#[test]
fn colour_columns_follow_rpfm_merging() {
    let schema = schema();
    // Raw r/g/b columns are what an RPFM-less converter would write; RPFM never does, so refuse with a hint.
    let raw = "key\tflag\tn\tf\twide\topt\tcol_r\tcol_g\tcol_b\tsmall\tbig\n#demo_tables;2;db/demo_tables/m\n";
    let e = format!(
        "{:#}",
        decode("m.tsv", raw.as_bytes(), &schema).unwrap_err()
    );
    assert!(e.contains("`*_hex`") && e.contains("col_hex"), "{e}");

    let head = "key\tflag\tn\tf\twide\topt\tsmall\tbig\tcol_hex\n#demo_tables;2;db/demo_tables/m\n";
    let row = |hex: &str| format!("{head}k\ttrue\t1\t1\tw\t\t1\t1\t{hex}\n");
    let (_, t) = db_doc(decode("m.tsv", row("0a1B2c").as_bytes(), &schema).unwrap());
    assert_eq!(
        t.rows[0][6..9],
        [Cell::I32(10), Cell::I32(27), Cell::I32(44)]
    );
    for bad in ["FF80", "#FF8000", "FF800000", "GG0000", "", "+F8000"] {
        let e = format!(
            "{:#}",
            decode("m.tsv", row(bad).as_bytes(), &schema).unwrap_err()
        );
        assert!(e.contains("column 9 (`col_hex`)"), "{bad}: {e}");
    }

    // Channels without an `_` prefix merge into RPFM's "unnamed colour group_<n>" column; float channels.
    let palette = "id\tunnamed colour group_1\n#palette_tables;0;db/palette_tables/p\nx\t0000FF\n";
    let (_, t) = db_doc(decode("p.tsv", palette.as_bytes(), &schema).unwrap());
    assert_eq!(
        t.rows[0][1..],
        [Cell::F32(0.0), Cell::F32(0.0), Cell::F32(255.0)]
    );
    let out = encode_db("db/palette_tables/p", &t, &schema).unwrap();
    assert_eq!(
        String::from_utf8(out).unwrap().lines().last(),
        Some("x\t0000FF")
    );

    // A channel RPFM would silently truncate must not be exported.
    for bad in [
        Cell::F32(1.5),
        Cell::F32(256.0),
        Cell::F32(-1.0),
        Cell::F32(f32::NAN),
    ] {
        let mut broken = t.clone();
        broken.rows[0][1] = bad;
        assert!(encode_db("db/palette_tables/p", &broken, &schema).is_err());
    }
    for bad in [256, -1] {
        let mut demo = demo_table(&schema);
        demo.rows[0][6] = Cell::I32(bad);
        assert!(encode_db("db/demo_tables/m", &demo, &schema).is_err());
    }
}

#[test]
fn identity_and_version_are_never_guessed() {
    let schema = schema();
    let body = |meta: &str| format!("key\n{meta}\nk\n");
    let id = |hint: &str, meta: &str| decode(hint, body(meta).as_bytes(), &schema);

    // version 1 of demo_tables is just `key`; exact version selection
    let (member, t) = db_doc(id("m.tsv", "#demo_tables;1;db\\demo_tables\\Mixed.Case").unwrap());
    assert_eq!(
        (member.as_str(), t.version),
        ("db/demo_tables/mixed.case", 1)
    );
    assert!(t.version_marker);

    for (meta, want) in [
        (
            "#demo_tables;3;db/demo_tables/m",
            "version 3 is not in the schema",
        ),
        (
            "#demo_tables;;db/demo_tables/m",
            "not a non-negative integer",
        ),
        (
            "#demo_tables;-1;db/demo_tables/m",
            "not a non-negative integer",
        ),
        (
            "#demo_tables;1.5;db/demo_tables/m",
            "not a non-negative integer",
        ),
        ("#demo_tables;1", "exactly"),
        ("#demo_tables;1;db/demo_tables/m;extra", "exactly"),
        ("demo_tables;1;db/demo_tables/m", "metadata row"),
        ("#nope_tables;1;db/nope_tables/m", "no table `nope_tables`"),
        (
            "#demo_tables;1;db/other_tables/m",
            "must be `db/demo_tables/<name>`",
        ),
        (
            "#demo_tables;1;db/demo_tables/a/b",
            "must be `db/demo_tables/<name>`",
        ),
        ("#demo_tables;1;/db/demo_tables/m", "empty path component"),
        ("#demo_tables;1;db/demo_tables/../m", "`..`"),
        ("#demo_tables;1;C:/db/demo_tables/m", "not allowed"),
        ("#demo_tables;1;db/demo_tables/NUL", "reserved device"),
        ("#demo_tables;1;db/demo_tables/m.", "ends with a dot"),
        ("#demo_tables;1;", "empty table name or path"),
    ] {
        let e = format!("{:#}", id("m.tsv", meta).unwrap_err());
        assert!(e.contains(want), "{meta}: {e}");
        assert!(e.contains("m.tsv"), "{e}");
    }
    // Metadata may not carry data in later cells.
    let e = err(decode(
        "m.tsv",
        b"key\n#demo_tables;1;db/demo_tables/m\tx\nk\n",
        &schema,
    ));
    assert!(e.contains("line 2, column 2"), "{e}");
    // The folder of the file must agree with the metadata table.
    let e = format!(
        "{:#}",
        id("db/palette_tables/m.tsv", "#demo_tables;1;db/demo_tables/m").unwrap_err()
    );
    assert!(e.contains("ambiguous identity"), "{e}");
    id("db\\demo_tables\\m.tsv", "#demo_tables;1;db/demo_tables/m").unwrap();
}

#[test]
fn single_column_empty_cells_cannot_roundtrip() {
    let schema = schema();
    let t = table(
        &schema,
        "single_tables",
        0,
        vec![vec![Cell::Str(String::new())]],
    );
    assert!(encode_db("db/single_tables/s", &t, &schema).is_err());
    let t = table(
        &schema,
        "single_tables",
        0,
        vec![vec![Cell::Str("v".into())]],
    );
    let bytes = encode_db("db/single_tables/s", &t, &schema).unwrap();
    let (_, back) = db_doc(decode("s.tsv", &bytes, &schema).unwrap());
    assert_eq!(back.rows, t.rows);
    assert!(!back.version_marker);
}

#[test]
fn export_refuses_values_that_do_not_fit_the_schema() {
    let schema = schema();
    let mut t = demo_table(&schema);
    t.rows[0][3] = Cell::F32(f32::NAN);
    assert!(
        format!(
            "{:#}",
            encode_db("db/demo_tables/m", &t, &schema).unwrap_err()
        )
        .contains("row 1 column `f`")
    );
    t.rows[0][3] = Cell::F32(f32::INFINITY);
    assert!(encode_db("db/demo_tables/m", &t, &schema).is_err());
    let mut t = demo_table(&schema);
    t.rows[0][2] = Cell::Str("not an int".into());
    assert!(encode_db("db/demo_tables/m", &t, &schema).is_err());
    let mut t = demo_table(&schema);
    t.rows[1].pop();
    assert!(encode_db("db/demo_tables/m", &t, &schema).is_err());
    let mut t = demo_table(&schema);
    t.rows[0][0] = Cell::Null;
    assert!(encode_db("db/demo_tables/m", &t, &schema).is_err());
    // member must be the table's own folder; version/fields must exist in the schema
    let t = demo_table(&schema);
    assert!(encode_db("db/other_tables/m", &t, &schema).is_err());
    let mut wrong = demo_table(&schema);
    wrong.version = 9;
    assert!(encode_db("db/demo_tables/m", &wrong, &schema).is_err());
    let mut wrong = demo_table(&schema);
    wrong.fields.pop();
    assert!(encode_db("db/demo_tables/m", &wrong, &schema).is_err());
}

#[test]
fn floats_survive_export_bit_exactly() {
    let schema = schema();
    let values = [
        0.0f32,
        -0.0,
        0.5,
        0.1,
        1.0 / 3.0,
        0.000_012_345,
        123_456.79,
        f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
    ];
    let rows = values
        .iter()
        .map(|v| {
            vec![
                Cell::Str("k".into()),
                Cell::I32(0),
                Cell::I16(0),
                Cell::F32(*v),
                Cell::Bool(false),
                Cell::Null,
                Cell::Str(String::new()),
            ]
        })
        .collect();
    let t = table(&schema, "mini_tables", 1, rows);
    let bytes = encode_db("db/mini_tables/m", &t, &schema).unwrap();
    let (_, back) = db_doc(decode("m.tsv", &bytes, &schema).unwrap());
    for (a, b) in t.rows.iter().zip(&back.rows) {
        match (&a[3], &b[3]) {
            (Cell::F32(x), Cell::F32(y)) => assert_eq!(x.to_bits(), y.to_bits()),
            other => panic!("{other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Localisation
// ---------------------------------------------------------------------------------------------

fn loc_rows() -> Vec<LocEntry> {
    vec![
        ("a_key".into(), "Hello".into(), 0),
        (
            "b_key".into(),
            "Gr\u{fc}\u{df}e \u{1f600}\u{4e2d} multi\nline\ttab".into(),
            1,
        ),
        ("empty".into(), String::new(), 0),
    ]
}

#[test]
fn loc_export_shape_and_roundtrip_into_identical_binary() {
    let bytes = encode_loc("text\\Localisation.loc", 1, &loc_rows()).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert_eq!(
        text,
        "key\ttext\ttooltip\n#Loc;1;text/localisation.loc\t\t\na_key\tHello\tfalse\nb_key\tGr\u{fc}\u{df}e \u{1f600}\u{4e2d} multi\\\\nline\\\\ttab\ttrue\nempty\t\tfalse\n"
    );
    match decode("text/localisation.loc.tsv", &bytes, &schema()).unwrap() {
        Document::Loc {
            member,
            version,
            rows,
        } => {
            assert_eq!((member.as_str(), version), ("text/localisation.loc", 1));
            assert_eq!(rows, loc_rows());
            assert_eq!(
                db::encode_loc(&rows, version).unwrap(),
                db::encode_loc(&loc_rows(), 1).unwrap()
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn loc_refuses_bad_identity_values_and_unwritable_rows() {
    let loc = |text: &str| decode("t.tsv", text.as_bytes(), &schema());
    let head = "key\ttext\ttooltip\n#Loc;1;text/ui.loc\t\t\n";
    loc(&format!("{head}k\tt\ttrue\n")).unwrap();
    // the older RPFM metadata name and a reordered header are both accepted
    match loc("tooltip\ttext\tkey\n#Loc PackedFile;1;text/ui.loc\t\t\n1\tt\tk\n").unwrap() {
        Document::Loc { rows, .. } => assert_eq!(rows, vec![("k".into(), "t".into(), 1)]),
        other => panic!("{other:?}"),
    }
    for (text, want) in [
        (
            format!("{head}k\tt\tmaybe\n"),
            "line 3, column 3 (`tooltip`)",
        ),
        (
            format!("{head}k\tt\n"),
            "2 cells but a localisation row has 3",
        ),
        (format!("{head}k\tt\ttrue\textra\n"), "4 cells"),
        (
            "key\ttext\n#Loc;1;text/ui.loc\t\n".to_owned(),
            "missing columns [\"tooltip\"]",
        ),
        (
            "key\ttext\ttooltip\n#Loc;2;text/ui.loc\t\t\n".to_owned(),
            "version 2 is not supported",
        ),
        (
            "key\ttext\ttooltip\n#Loc;1;db/units_tables/x\t\t\n".to_owned(),
            "text/....loc",
        ),
        (
            "key\ttext\ttooltip\n#Loc;1;text/../ui.loc\t\t\n".to_owned(),
            "`..`",
        ),
        (
            "key\ttext\ttooltip\n#Loc;1;text/ui.txt\t\t\n".to_owned(),
            "text/....loc",
        ),
        (
            format!("{head}k\t{}\ttrue\n", "x".repeat(0x1_0000)),
            "limit",
        ),
    ] {
        let e = format!("{:#}", loc(&text).unwrap_err());
        assert!(e.contains(want), "{want}: {e}");
    }
    // A localisation file lying in a table folder is ambiguous.
    assert!(
        decode(
            "db/units_tables/x.tsv",
            format!("{head}k\tt\ttrue\n").as_bytes(),
            &schema()
        )
        .is_err()
    );

    assert!(encode_loc("text/ui.loc", 2, &loc_rows()).is_err());
    assert!(encode_loc("text/ui.txt", 1, &loc_rows()).is_err());
    assert!(encode_loc("text/ui.loc", 1, &[("k".into(), "t".into(), 2)]).is_err());
    assert!(encode_loc("text/ui.loc", 1, &[("k".into(), "a\rb".into(), 0)]).is_err());
    assert!(encode_loc("text/ui.loc", 1, &[("k".into(), "x".repeat(0x1_0000), 0)]).is_err());
}
