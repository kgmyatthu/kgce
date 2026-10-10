# KGC Port Editor

Native Rust desktop editor and ETW world-campaign converter for Napoleon: Total War. Version 0.7.3 corrects port/town selection geometry and includes the **1.3516979217529297× connected atlas**, **half-size campaign-building shader pack**, and **click-drag settlement movement**. Region splitting, wilderness activation, terrain relief and existing-faction ownership remain available. Modded targets use **RPFM-exported TSV datasets**. Except for the two bundled campaign-building shader inputs, game-derived output is rebuilt locally from the user's installed originals; the editor needs no Python, Java or RPFM runtime and never writes into game installations.

## Run and workflow

Open `KgcPortEditor.exe` with the shipped `assets/` directory beside it. Select your **Empire** and **Napoleon** installations (either the installation roots or their `data/` folders), then click **Import originals & rebuild resources**. Both games are required. The editor reads the installations without modifying them, creates a verified snapshot under `workspace/sources/`, and natively derives the conversion rules and UI from that snapshot. No game-source workspace is distributed. The two campaign-building shader inputs ship with the editor and do not require a shader import.

Editor content uses dark mode independently of the system preference. Main text and buttons are 16 pt, headings 22 pt, monospace text 15 pt and helper text 13 pt; map labels use the same styles. Already-running windows keep their old appearance until relaunched.

1. **Import TSV:** choose original Napoleon data or a modded TSV target. For a modded target, supply an RPFM export folder, name and localisation language; preview the required/missing table checklist and field errors, then import and pin the dataset.
2. **Mappings:** map source factions, Napoleon stand-in units and building levels to keys in the target dataset. A faction without its own start-position record needs a structural donor. Wrong unit kinds and building-slot incompatibilities block export.
3. **Compatibility:** review campaign-map blockers and advisory warnings. The editor is not a general mod-database validator: unrelated duplicate keys, missing uniforms and dangling references in unchanged mod rows do not block export. Ambiguous inputs actually consumed by the campaign and conflicts with generated rows still require resolution. DB/localisation decisions are bound to the exact issue inputs, not just a reusable issue label.
4. **Ownership map:** assign regions to existing target factions, split or activate regions, select **Campaign scale & layout → Apply proven setup**, and use **Move settlements** to drag capitals and towns inside their regions. Ctrl-click selects several regions; scroll zooms, right/middle-drag pans and **Fit map** resets the view.
5. **Export review:** inspect the plan and warnings, then **Save & Export ZIP**. The only deliverable game payload is `workspace/generated/ntw/GrandCampaign-NTW.zip`.

**Save project** stores the pinned dataset, mappings, conflict decisions, adapter, ownership, terrain, activations, splits, complete atlas plan, building factor and settlement destinations in `workspace/project.json`. Reset/restore ownership actions do not change original game files. An unbound TSV target never silently exports vanilla.

**Port/town selection correction (0.7.3):** re-export an existing project and install the new ZIP, then restart Napoleon. No source or TSV re-import is needed. Earlier exports combined clockwise Empire footprints with Napoleon artwork templates, producing inverted selection volumes: clicking a port could be interpreted as a fleet movement order onto land. The converter now uses Napoleon's counter-clockwise ordering for template-backed ports/towns while retaining legacy ordering for resources and settlement slots. Positions, docking points, polygon shapes and navigation data are unchanged by this correction; changing building size is not a workaround.

### Local resources and distribution

The source distribution contains Rust code, authored conversion policies under `assets/policy/`, community DB schemas, public elevation data and the two explicitly bundled, game-derived campaign-building shader inputs—not extracted game tables, localisation, UI, textures or heightmaps. The schemas are from [RPFM schemas](https://github.com/Frodo45127/rpfm-schemas), under its [MIT licence](https://github.com/Frodo45127/rpfm-schemas/blob/master/LICENSE); the licence is retained in `assets/rules/LICENSE.rpfm-schemas`. That schema licence does not cover the game shaders. Terrain uses the public NOAA ETOPO 2022 60-arc-second surface model ([dataset DOI](https://doi.org/10.25921/fd45-gt74)); its provenance and lossless encoding are documented in `assets/policy/terrain/README.txt`.

Derived rules and UI are stored in `workspace/derived/<fingerprint>/assets/`, keyed by the verified source manifest, language, schemas and recipes. Published files are hash-checked. If those rules/UI files are damaged, the error identifies the cache; close the editor, move that fingerprint directory aside and reopen to rebuild it. The first export also paints the fixed-theatre terrain locally; subsequent exports reuse its independently verified terrain cache. Changed or damaged terrain cache entries are regenerated, never silently reused. No Python, Java, RPFM or network access is required at runtime.

Existing intact snapshots and projects remain usable; derived resources are built on demand. Ownership, activations, splits and target bindings remain user decisions. `workspace/`, derived caches and exported campaigns contain game-derived material: keep them local rather than adding them to a source distribution. The two bundled building shaders are the explicit packaging exception; this does not authorize publishing other game-derived assets through Git history, tags or release archives.

## Existing factions only

Every export uses existing target faction keys. New-faction creation, its bundled DB/identity recipes, the recycling checkbox and `project recycling --enabled` have been removed. There is no alternate creation mode.

- **You choose the owners.** No territory is automatically assigned to a template or replacement faction, and no faction is created. A region whose original ETW owner is absent from the target and that has no assignment yet is **pending ownership**: it is a to-do for you, not a compatibility or mapping issue, and it is not counted as a blocker. Use **Assign ownership** in Mappings, Compatibility or Export review to select the pending regions in the Ownership map, or **Select regions awaiting an owner** on the map. Choose an existing faction and assign, or select smaller groups to give them different recipients. The recipient is not preselected. In the CLI, `analyze` and `resolve list` report them as `unassigned_regions`, and `assign --region KEY --faction EXISTING_KEY` selects the recipient.
- **Pending ownership versus an invalid override.** While any region is pending, the analysis is not export-ready and `build` refuses; `analyze` exits 3. The owner-dependent compatibility checks have not run yet: they resume after the recipients are selected, so further blockers can appear then. A pending region is never filled in automatically. By contrast, an owner you explicitly saved that is not an existing target faction is an actual error (issue `ownership:<region>:invalid_owner`), repaired by assigning an existing faction.
- The selector includes every faction in the effective target database, including factions not previously represented in the curated world. Faction mappings are optional; region ownership can be assigned directly.
- A selected existing faction without a start-position record still needs a live **Faction donor** in **Mappings**. Both the recipient and donor must already exist in the imported target; absent original factions are not offered as recipients. Its identity comes from the target DB, not the donor. Receiving land does not automatically make it playable.
- Missing world factions and their starting forces are omitted. Their armies/fleets are **not transferred** to whoever receives the land, and target-only recipients get no invented starting army. Retained forces keep the usual relocation rules.
- Existing valid revolt identities remain unchanged. If a region's Empire revolt faction is absent, the converter uses its selected owner's existing target `settler_rebellion_faction`. Missing or invalid rebellion configuration blocks export rather than inventing a faction.
- Older projects retain ownership choices and mappings, regardless of their former recycling setting. Invalid existing choices remain visible blockers; migration never redistributes territory. Changed output can require renewed compatibility decisions, but removing the old mode does not require reimporting an intact dataset.
- The direct start-position generator also rejects faction keys absent from the imported database. Generated or curated DB rows cannot introduce faction identities or make an invented key pass that check.

### Ownership workflow verification

- Release suite: 256 tests passed.
- An isolated copy of the imported dataset with no owner overrides reported 47 pending regions, zero compatibility issues and a refused export. Invented-owner assignment was refused without modifying the project; a saved invalid override produced a real ownership error, not a pending task.
- Native GUI: **Assign ownership (47 region(s))** selected all pending regions without choosing a recipient. Explicitly selecting imported `britain`, assigning and saving produced a ready analysis with zero pending regions.
- Full export: 137 regions, 41 represented factions and 47 explicit ownership choices. Decoding the exported start position confirmed every region owner existed in the imported catalog and every selected recipient was preserved. The pack contained 10 campaign DB members and no faction-identity or uniform table.
- The direct start-position API rejected an invented key even when generated DB rows claimed to define it. Donor mapping to an absent original faction was refused without changing the project.
- No game or user-project files were changed. The export still reported 54 pre-existing map/start-position consistency warnings in the unchanged conversion; the campaign was not installed or tested in Napoleon.

## Preparing RPFM exports

- Export the target mod's **effective, complete versions of the required tables**, plus localisation, through RPFM's TSV export. A patch containing only changed rows is not a complete target dataset: the editor does not fill missing DB rows from vanilla or infer pack precedence.
- You do **not** need to export all readable DB tables. **Scan folder** shows the campaign's required inputs and why each is needed. By default only those tables (and the two localisation files) are imported; every other table in the folder is ignored and reported. Tick **Also import the optional tables** (CLI `--with-extras`) to retain them as supplied for analysis; they never become required and are not forced into the export, and their references do not make more tables mandatory.
- Include both full `text/localisation.loc` and `text/ui.loc` exports, normally named `localisation.loc.tsv` and `ui.loc.tsv`. Additional localisation members are supported.
- Keep RPFM's header row and second metadata row (`#<table>;<version>;<member path>`). They determine the schema version and output member identity. Preserve processed colour columns such as `primary_colour_hex`; do not replace them with guessed raw RGB columns.
- UTF-8, optional UTF-8 BOM, and CRLF/LF exports are accepted. Required tables must have readable columns, values, versions and unambiguous member identities. Unreadable or repeated files for unrelated optional tables are skipped with warnings rather than blocking the campaign import. Unsafe member paths are not accepted. Non-TSV files are ignored with a warning.
- Different files for one table retain their separate member identities. Identical duplicate rows are accepted. Conflicting duplicates follow the campaign-aware policy below; the editor never guesses which row the engine uses.
- Imports are copied, encoded and hash-verified under `workspace/targets/`. Later edits to the external TSV folder do not mutate the pinned dataset. **Check folder** / `tables check` reports drift; import again to use changed inputs.
- `tables requirements`, `tables preview --folder DIR --language CODE`, GUI preview and import use the same minimum contract below. Missing required inputs block import. Optional tables never expand this list; without `--with-extras` they are not even imported, and **Check folder** does not count them as drift. You do not need to re-import an existing dataset for the export behaviour described next.
- References required by generated campaign rows remain checked. Dangling references in unchanged supplied target rows are advisory even when the provider table is present; missing uniform entries are not the editor's repair task. Absent optional providers do not become import requirements. The installed target game/mod must supply its own records. Successful preview does not certify conflict-free or gameplay-compatible data.

**Imported is not exported.** Imported tables, required or optional, are conversion inputs and analysis catalogs (for example land/naval unit stats are needed to generate starting forces), not an export list. The ZIP contains only the campaign tables the build adds or overrides, plus explicit conflict-resolution DB replacements (members that lose conflicting rows are replaced at their original paths, including empty replacements). Unchanged imported tables, such as unit stats and existing faction identities, are **not** copied: the installed target mod must supply them. Full target localisation is retained and merged with converted text. New port text can remain English; the analysis reports this rather than claiming complete translation.

Verified on the existing TSV-target campaign with Indochina activation and Mandalay split: the generated pack went from 30 DB members to 10; all retained DB bytes and every non-DB pack member were unchanged. The campaign exported 139 regions and 47 ownership overrides. All 249 tests passed, including omitted input-only tables, retained changed/empty replacement members, and rejection of unplanned DB exports. Import and Export review text were exercised in the native GUI. This was not an in-game test.

### Audited minimum imports

The shipped campaign requires these **20 DB tables**. The contract is the union of typed campaign-state consumers, tables receiving generated rows, and readable providers referenced by those generated rows—not a recursive dependency closure over supplied target data.

| Table | Campaign use |
| --- | --- |
| `building_chain_to_slots_tables` | Validate building chains against slot types. |
| `building_levels_tables` | Validate and map building levels. |
| `campaign_map_playable_areas_tables` | Add converted playable-area rows. |
| `campaign_map_settlements_tables` | Add converted settlement rows. |
| `campaign_map_slots_tables` | Add converted slot rows. |
| `campaign_map_towns_and_ports_tables` | Add converted town and port rows. |
| `campaign_variables_tables` | Resolve variable keys referenced by generated campaign settings. |
| `campaigns_campaign_variables_junctions_tables` | Add converted campaign settings. |
| `cdir_campaign_junctions_tables` | Add converted campaign-AI configuration. |
| `factions_tables` | Existing owner keys, faction identity, mappings and rebellion configuration. |
| `regions_continents_tables` | Resolve continents referenced by generated regions. |
| `regions_tables` | Add converted region rows and validate start-position region references. |
| `slots_art_tables` | Read artwork templates for the generated campaign map. |
| `technologies_tables` | Validate technology keys in the generated start position. |
| `trade_node_groups_tables` | Add converted trade-node groups. |
| `trade_nodes_tables` | Add converted trade nodes. |
| `unit_stats_land_tables` | Write target land-unit strengths into starting armies. |
| `unit_stats_naval_tables` | Write target crew strengths into starting fleets. |
| `units_tables` | Validate/map unit keys used by starting forces and generated permissions. |
| `units_to_exclusive_faction_permissions_tables` | Read target availability and derive starting-unit permissions. |

Both complete `text/localisation.loc` and `text/ui.loc` TSV exports are also required.

These **16 tables are no longer mandatory**:

```text
cdir_faction_junctions_tables
cultures_subcultures_tables
faction_rebellion_units_junctions_tables
faction_uniform_colours_tables
government_types_tables
groupings_military_tables
ministerial_positions_by_gov_types_tables
ministerial_positions_tables
names_groups_tables
names_royalty_tables
ship_names_tables
technology_faction_junctions_tables
uniform_to_faction_colours_tables
uniforms_tables
unit_regiment_names_tables
unit_stats_naval_crew_to_factions_tables
```

They served removed faction-creation recipes or references outside the remaining campaign consumers. Supplying them is still allowed and does not discard them from the output.

Likewise, optional target rows no longer demand `agent_attributes_tables`, `ancillary_info_tables`, `cursors_tables`, `governorships_tables`, `taxes_keys_tables` or `trait_categories_tables`. This relaxes the **import checklist**, not the installed mod's own runtime dependencies.

### Duplicate-key policy

- **Warn and preserve unrelated keys:** conflicting duplicates that cannot affect values consumed or rows written by the campaign are advisory, even inside a required table or one TSV member. Unchanged imported rows retain their bytes, order and member identities and are not exported as DB replacements.
- **Require resolution only for campaign dependencies:** conflicts on consumed faction identities, army strengths/permissions, named building levels, template slot art or generated keys can change the campaign and remain protected. Cross-member conflicts offer an explicit winning member; conflicting consumed rows inside one member must be repaired in the source TSV.
- Classification uses campaign consumers and generated rows, not table presence alone. Unit/technology key-existence checks do not require unique row contents. Uniform and faction-palette rows remain mod-supplied data; neither missing entries nor duplicate definitions make them campaign blockers.
- Unrelated repeated localisation keys are preserved. Only text keys written by the conversion must be unambiguous. Existing full-localisation export behavior is unchanged.
- Duplicate/reference warnings appear in the analysis and export reports. Required campaign inputs must still be readable; owner/mapping errors, missing generated-row dependencies and unsafe output collisions still block. Preserving mod data is not a claim about engine load order or gameplay compatibility.


### Resources TSV cannot describe

TSVs do not establish installed pack order, UI/script compatibility, model/texture availability or campaign-template compatibility. The resource-review checkbox records your review but is **not an export prerequisite**; leaving it unchecked produces a warning. Without an adapter, the converter uses its original campaign/UI/script templates. Missing or corrupt resources actually required to generate the campaign still fail normally.

An optional **resource adapter** is a folder of already merged files, bound to the exact dataset fingerprint. Allowed paths:

- `ui/...`
- `script/...`
- `campaigns/wld_napoleon/scripting.lua`
- `campaigns/eur_napoleon/startpos.esf` — structural donor input only; this file is not shipped.

A supplied campaign script replaces the generated **loose** script, so it is not shadowed by another copy. Adapter import does not certify gameplay compatibility. Opaque/schema-unsupported resource tables cannot be checked as TSV; unresolved resource references are explicitly reported as unverified. The matching installed mod must supply the resources, without a vanilla-row fallback in the generated dataset.

## ZIP layout and manual installation

Baseline and tested modded exports contain 674 game files under `data/`, plus export metadata under `kgc-export/`:

```text
data/
  nap_world.pack                 # vanilla mode
  # or zz_kgc_world_<hash>.pack  # TSV mode
  campaign_maps/nap_world/...
  campaigns/wld_napoleon/
    startpos.esf
    scripting.lua
kgc-export/
  ...
```

Install manually by dragging **only the ZIP's `data` folder** into the Napoleon folder containing `Napoleon.exe`. Do not put it inside the existing data folder. Keep `kgc-export/` outside the game.

For TSV targets, install the matching mod assets and arrange for the generated tables to override their corresponding tables. TSVs cannot prove real engine load order. Do not combine exports made for different datasets. Replacing an earlier installation is your file-manager decision, never an application action.

**The app never installs the export or overwrites game files.** Game directories are rejected as write destinations. Executables, preferences and user scripts are not edited. Source import only reads installations; generation uses copied snapshots.

Every ZIP entry is read back and CRC-checked before publication. Loose staging files are removed after packaging. The saved project and build report remain beside the ZIP. Earlier exports stay under `workspace/generated/previous-*`; failed builds leave the last successful export intact.

## Original sources and campaign behavior

- Original ETW/NTW snapshots are byte-for-byte, read-only copies with SHA-256, size and provenance manifests. **Verify originals** detects changed, missing or unlisted files. Use a new workspace to import different originals.
- The curated world starts with 137 regions and 57 represented faction definitions. Output faction keys must exist in the target DB; absent world factions and their forces are omitted. You explicitly choose existing owners for affected regions, including target factions not previously represented in the world.
- Ownership changes update holdings, capitals, buildings, governorships, deployed forces and campaign AI references. Lost capitals relocate; landless factions become inactive. Reviving a faction does not automatically add it to the playable roster.
- Displaced armies move to owned territory, preferring the same theatre. Fleets require an owned port; impossible placements block export.
- Starting forces use Napoleon stand-ins or explicit target-unit mappings. Target men/crew strengths are written into deployed units. This is not an Empire model port.
- Generated starts affect **new campaigns**, not existing saves. Static validation does not establish gameplay compatibility.
- Game-derived assets, including the two bundled building shaders, do not acquire independent redistribution rights by being included in the tool.

## Connected atlas, building size and settlement movement (0.7.0)

### Reproduce the accepted map arrangement

On **Ownership map**, open **Campaign scale & layout** and click **Apply proven setup**. This selects the authored `assets/policy/connected_atlas.json` plan and building factor **0.5**. Save the project, review compatibility, and export a **new campaign**.

- Exact map scale: **1417358 / 2²⁰ = 1.3516979217529297**. Six charts place seven theatres; **Europe and India share one affine transform and remain connected**. This is the accepted stock-executable experiment, not an arbitrary scale/theatre-placement editor and not proof of an absolute engine maximum.
- The preset reproduces the experiment's source crops. In particular, America's source theatre is **x = −760…−340**, not the unscaled layout's **−888…−340** Western Plains extension. Switching layouts can therefore reject an activation, split or settlement outside the selected coverage; the previous project remains intact.
- Artwork follows those source bounds: the atlas keeps original America's frame, colour and raw heights and generates only the India extension before transforming the map. The unscaled layout still generates both extensions. Their terrain caches are separate and selection-bound, preventing the wider Western Plains artwork from leaking into the narrower atlas.
- Geography, navigation, display rasters/vectors, start-position coordinates, visibility caches, campaign-AI indexes, script camera bounds and coordinate-bearing database rows use the same plan. Geometry and settlement edits happen in source space first, followed by one atlas transform.
- Only the three stored copies of theatre rectangle corners are snapped to **1/4096**: both `regions.esf` copies and `sea_grids.esf`. Global coordinates are not camera-snapped. This preserves the exact-horizontal India north edge required by the stock camera's pan limiter.
- The export records its plan, source/output area contract, navigation Q20 overrides, geometry checks, display hashes and campaign/table reports in **`kgc-export/atlas.json`**. The saved project carries the full plan, rather than depending on a hidden editor toggle.
- **Use original map layout** disables the atlas only; it returns to the unscaled fixed-bounds policy below and does not reset building size. Older projects open with no atlas and original building size until explicitly changed.

No executable patch is generated or installed. Larger arbitrary factors/offsets are rejected rather than risking the stock signed-coordinate arithmetic.

### Campaign-building scale

The **Building size** control accepts **0.1–1.0**; **0.5** is the in-game-approved half-size setting, and **1.0** emits no override pack. Both shader inputs are bundled with the editor: older snapshots need no shader re-import, and installed mod shader overrides are never used as the scaling source.

For an active factor, export creates **`data/zz_campaign_buildings_scale.pack`**, a separate PFH0 movie pack that auto-loads without a mod-list entry. Its only members are modified versions of the bundled `fx/campaignbuilding.fx` and `fx/campaignbuilding_vertex.fx_fragment`. The local model position is scaled before the instance transform in visible, shadow and depth paths; world translations, character models, databases and saves are unchanged. This controls geometry using the campaign-building shader, not necessarily ground underlays, every composite piece or CPU selection bounds. Other mods overriding these shaders can conflict.

**`kgc-export/building-scale.json`** records bundled-source/override hashes, provenance, factor and pack hash. The accepted 0.5 payload remains reproducible without obtaining shader files from the user's snapshot.

### Move an existing settlement

Click **Move settlements**, zoom in, then primary-button-drag a **capital circle** or **town square**. A capital's complete building-slot cluster moves with it. The preview reports invalid placement; release commits only a valid destination. **Ctrl+Z / Undo** reverses the last move, **Reset** restores an individual settlement, and **Esc** cancels a drag or leaves the tool. Right/middle-drag still pans.

The destination and full footprint must remain inside the settlement's own passable area/landmass, outside holes, with boundary clearance and spacing from other settlements/ports. Green preview means these region/spacing checks passed; **export additionally checks the navigation-node footprint** against roads, coasts, borders and other nodes. A navigation collision blocks export without replacing the last good ZIP; choose another position. **Ports remain locked** because relocating them requires coastal landing/dock regeneration. Invalid or off-viewport releases change nothing. The project stores stable region/slot IDs and **source-space coordinates**, so reload/export cannot repeatedly rescale a destination.

Export updates settlement/slot geometry, city navigation nodes, POIs and distances, trade-route endpoints and curves, start-position residence/slot/CAI/HLCI records and garrison references. Characters exactly at the original capital move with it; unrelated armies do not. Static decorative road splines are not a road-rerouting tool. Removing a split/activation also removes moves belonging to its deleted regions.

### Install or roll back manually

The editor exports a ZIP only. Close NTW, back up the previous generated world/campaign directories **outside `data`**, and move the old generated directories out before copying the new ones; merging can leave obsolete geometry. Remove any previous **`zz_campaign_buildings_half_size.pack`** and **`zz_campaign_buildings_scale.pack`** before copying the ZIP's `data` contents. Keep `kgc-export` outside the game.

Restart and start a **new campaign** for map/settlement changes. The shader pack affects campaign buildings across this installation, including existing campaigns, but never rewrites saves. Remove it and restart to restore original building size; exporting factor 1.0 does not automatically remove an already-installed pack. Test the exact mod setup in game: static/export checks are not gameplay validation.

Reproducible CLI equivalents:

```powershell
.\KgcPortEditor.exe atlas proven --workspace "F:\path\to\workspace"
.\KgcPortEditor.exe atlas show --workspace "F:\path\to\workspace"
.\KgcPortEditor.exe theatre list --workspace "F:\path\to\workspace"
.\KgcPortEditor.exe settlement list --workspace "F:\path\to\workspace"
.\KgcPortEditor.exe settlement set --region france --slot settlement:france:paris --center "23.805769,338.7181" --workspace "F:\path\to\workspace"
.\KgcPortEditor.exe build --workspace "F:\path\to\workspace"
```

`settlement set` coordinates are source-space and validated against the current project's geometry; the example is not a universal valid position after arbitrary region cuts. `settlement remove --region KEY --slot ID` removes a move. `atlas buildings --building-scale 1.0` restores the export's original building setting; `atlas off` disables only the atlas.

Integration verification: **432 release tests**, a release build and Clippy passed (existing warnings remain). Native GUI checks exercised dragging, rejection, undo and reload; independent export decoding passed all 28 stock-camera pan checks and reproduced the approved half-size shader pack byte-for-byte. Two exports of the same saved project, including Indochina activation and moved Paris/Dijon, reproduced all **665 game files byte-for-byte**. Only the atlas report's navigation timing changed among the 670 ZIP members, so compare payload hashes rather than requiring identical ZIP hashes. The integrated campaign still needs a new-campaign test in NTW; no game installation or existing save was modified.

## Region splits and activations

A region can be cut in two. Draw a cut line across the region on the Ownership map, name the new region and its settlement, and place the settlement; the project (format 6) stores the split as `splits[]` (source region, outline cut positions, interior cut points, key, names, settlement centre, lookup colour). Splits are applied in project order, and a later split may cut an earlier split's region.

`build` applies every split to the generated campaign: the map files (`regions.esf`, pathfinding, points of interest, sea grids, trade routes, lookup bitmap, border splines, database rows and strings), the start position (a new `REGION` cloned from the source with its own settlement and slots, governorship and ownership lists, trade node numbering with one new settlement node, and the campaign AI graph: region, settlement, building slots, boundaries) and the database/localisation rows (`regions_tables`, `regions_onscreen_<key>`, `regions_battle_name_<key>`, rebel faction name, settlement name). The new region starts with the owner the project assigns to its key (default: the source's owner at the time the rules were derived). Another owner must already hold land in the source's theatre.

Checks run during the build: every addition to the start position is re-read from its serialised bytes and verified (world ids, ownership and governorship lists, trade node ranges, CAI references), and the generated ownership must equal the project's. A failed check aborts the build. Static checks only: the new regions' campaign AI (it has no per-region history or analyser beliefs, the engine creates them lazily), battle maps, and turn processing have not been played in game.

**Activate wilderness region.** ETW's world map contains land polygons inside the playable theatres that carry no campaign data: `wilderness_*` regions, `alps`, `central_italy`. An *activation* turns one into a normal campaign region. The project stores it as `activations[]` (the region key from `regions.esf`, a `donor` region of the same theatre, region and settlement names, settlement centre, lookup colour). Activations apply in project order and before splits, so a split may cut an activated region, while a split must not target a wilderness key that was not activated first. The key of a region that is already a campaign region, or an unknown donor, is refused.

The donor is the template: the activated region takes its owner, theatre, culture, religion, rebels and resources from the donor's rules, and its settlement slots (road, fortification and `settlement_<k>_slot` positions are translated around the chosen centre), `REGION` record, governorship and campaign-AI entries are cloned from the donor in the start position. The wilderness polygon keeps its `regions.esf` index, its pathfinding ids and its neighbours; the map side (settlement and slots in `regions.esf`, points of interest, lookup colour, border splines, trade route node) is written like a split. The start position already holds a campaign-AI region record for every wilderness polygon (empty: no settlement, slots, owner or governorship, usually with its boundaries, HLCIs and theatre entry); that record is updated in place, missing boundaries are added and existing ones keep their ETW lengths. The database and localisation rows (`regions_tables`, `regions_onscreen_<key>`, `regions_battle_name_<key>`, rebel faction name, settlement name) are derived from the donor's row exactly as for a split. The owner follows the ownership map like any region; it must be a faction that also holds a region of the ETW world. Activated regions are static-checked only, like splits: `alps` is mostly impassable mountain, and nothing was played in game.

**Ownership map tool.** *Show borders & towns* (default on) draws every region outline with its city (circle, name), towns (square), ports (blue triangle) and resource slots (orange diamond), labelled as you zoom in. *Split region*: click the region, click one of its border vertices (the cut snaps to the nearest outline vertex within 2% of the view width), add free interior points, finish on another border vertex. The cyan half is the new region, the orange half keeps the source key; *Swap halves* exchanges them. Fill the names (key and settlement slug are derived), pick a lookup colour (one that no region of the theatre uses is proposed), *Place settlement* with a click inside the cyan half, then *Apply split*. The region list shows the new region with a *(split)* suffix and it can be assigned an owner like any other. *Remove split* drops a split and, after confirmation, every later split that was cut from it. Backspace undoes the last cut point, Esc leaves the tool. The source's own settlement must stay in the orange half and the cut may touch the outline only at its two end vertices; areas with inner holes (lakes) cannot be cut yet. CLI equivalents: `split add --source K --area N --a I --b J --interior "x,y;x,y" --name NAME --settlement-name NAME --center x,y [--key K2 --settlement slug --colour rrggbb]`, `split list`, `split remove --key K2`.

## Unscaled fixed theatre bounds, activation options and carving

With the atlas disabled, preview and export use the fixed source rectangles below. The connected atlas instead uses its saved, validated experimental plan and source coverage. Neither mode permits arbitrary theatre-rectangle editing. Activation, carving, splits and ownership remain user-driven; selecting a layout does not automatically activate land or assign a faction.

### Fixed theatre rectangles

Coordinates are `[[min x, min y], [max x, max y]]` in ETW world units:

| Theatre | Fixed rectangle | Coverage |
|---|---|---|
| Europe (`2`) | `[[-180, 140], [440, 520]]` | Original rectangle, unchanged |
| India (`3`) | `[[440, -80], [790, 300]]` | Known-working Thailand/Indonesia extent; 10 u gap to East Indies |
| America (`1`) | `[[-888, 0], [-340, 460]]` | Includes all of Western Plains (`western_plain`); only the west edge changes |

Western Plains is not Great Plains or `wilderness_great_plains`. Its 331 outline vertices reach x = -886.510193, so -888 is the nearest enclosing boundary on the original 2 u pathfinding lattice. The other four theatre rectangles remain unchanged.

The old **Extend theatre** tool, arbitrary rectangle fields and CLI set/reset commands remain removed. In unscaled mode the Ownership map shows **Theatre bounds are fixed**; hover to inspect the three rectangles. `KgcPortEditor.exe theatre list` is read-only and reports the selected layout. Older projects' obsolete `theatres` entries are discarded in memory, including unsafe or off-grid choices; opening a project does not rewrite its file. Saving writes format 10, with the separate `atlas`, `building_scale` and `settlement_moves` fields.

With the atlas disabled, every build applies these bounds, even without activations or splits: both theatre-record copies and moved transition areas, climate/wind canvases, theatre TGAs, pathfinding grids, sea-grid bounds, sea/backdrop clipping, and dependent start-position references. Unchanged Europe is not re-canvassed. India uses a 175x190 pathfinding grid; America uses 274x230. Existing sea-cell counts are preserved. New interior overlaps or new theatre contacts are rejected by the geometry primitive; the original Europe/India shared edge is retained.

India's rectangle matches the prior in-game experiment. The visible decorative frames and matching terrain of the two enlarged theatres (America west, India east) are produced **automatically** from the user's original ETW artwork and the shipped public elevation data (see *Fixed theatre artwork* below); no external painting or `painted_terrain` is needed. The combined fixed map is statically checked and exported/decoded here, but the new artwork has **not been tested in game**: an earlier build with these bounds loaded and the camera reached the extended area, while the old frames were still visible there. Activation and carving remain explicit choices.

### Navigation consistency (0.7.2)

- Fleets are assigned to the **finished sea grid**, not the original ETW grid. Each fleet's cell reference and that cell's fleet list must agree with its position, before and after atlas conversion.
- Sea-grid connections contain exactly one pair for every two sea-bearing cells. Newly flooded cells receive their connections; cells that lose their sea retain no stale connections.
- HLCIs use the stock structure: **one record per connected component, listing all areas of that component**. Valid multi-area groups remain intact. Duplicate components, missing/duplicate area coverage, stale component ids and out-of-range indices are rejected. This replaces the incorrect one-HLCI-per-area rule.
- Enlarging a theatre removes the old impassable sea frame from the moved edge, so it cannot wall off a relocated exit. Detection follows the source frame rather than a fitted width; unchanged edges, land and isolated impassable areas are preserved. Type-2 pathfinding pieces are not universal exit markers.

Verification: **459 release tests passed**. Portable regressions cover the historical 9-to-12 sea-cell expansion (36 to 66 connection pairs), fleet-cell boundaries and atlas scaling, grouped HLCIs, and a moved exit behind a frame wider than 12 units. Real normal and connected-atlas exports were independently decoded: all 21 retained fleets had correct cells, and all 205 regions had valid HLCI coverage. The normal export's moved western America exit now connects to the existing Pacific sea and southern exit; the Atlantic remains separate. Of 680 game files, only `pathfinding.esf`, `sea_grids.esf` and `startpos.esf` changed relative to the same project's earlier normal export.

These are file-level navigation checks, not an NTW gameplay test. The reported ship-selection crash and exit commands in the installed atlas build still need in-game confirmation. Re-export and start a **new campaign** to exercise the rebuilt start-position AI; existing saves are not repaired.

### Activation options

An activation (`activations[]`, see the wilderness section) gains three optional fields; format-7 activations load unchanged:

| Field | Meaning |
|---|---|
| `areas` | which polygons (*areas*) of the region become passable: `"centre"` (default; the area holding the settlement, which must be passable already), `"all"`, or `{"indices": [0,1,2]}` (positions in the region's `areas` list as the map has them after the theatre extension and clipping) |
| `merge` | `true` unites the selected areas that share a border into **one** passable area (areas touching nothing, such as offshore islets, are left as they are and stay impassable). The union cancels the shared internal edges, keeps the vertex ids, and the component id is the one of a passable land neighbour it borders, else a fresh one |
| `carve_from` | `Some(source)`: `key` is a **new** region created from the single area `areas = {"indices": [n]}` of the backdrop region `source` (for example an island of `unexplorable`); see below |

Command line: `activate add ... --areas all|centre|I,J,K [--merge] [--carve-from SOURCE]`. In the editor, the activation panel lists the areas of the picked region (tick them, or click them on the map; the map numbers them), *Merge the selected areas into one*, and *Carve into new region*. Selecting areas that need a pathfinding path (nothing in ETW's grid yet) makes the build **synthesise** it: land zone, coast band, region border bands with group paths, landing crossings (about one per 7 u of coast, two congruent hexagons) and a settlement node (city: regular octagon, town: diamond). The settlement point must stay at least **2.1 u** from the outline of its area (the editor and `activate add` refuse it otherwise: the node must fit inside the region).

### Carve into a new region

*Carve into new region* (or `--carve-from`) makes `key` a new land region that is appended **last** in `regions.esf`: the chosen area moves verbatim out of the backdrop region (its later areas shift down by one, every neighbour code and query leaf is re-pointed), becomes passable, and gets a settlement exactly like any activated region (`donor` supplies the rules and slot layout). The start position gets a new campaign-AI region (index = `regions.esf` index) with the usual records; an island with no land neighbour gets boundaries to the adjacent sea regions. The theatre's region list gains the key when the settlement is attached.

### What is derived and what you choose

| You choose (project) | The build derives |
|---|---|
| which areas; merge; carve source and area; donor, names, settlement slug and centre, lookup colour | fixed theatre rectangles; component ids, impassable ids, island ids and bounding boxes of the areas; polygon clipping and connectivity codes; the spatial index; pathfinding paths, group paths, landing crossings, settlement node; sea-grid cell lists (lands, seas, ports, trade-route splines) and connections; trade routes of the new settlement (an island gets a route to the nearest settlement); lookup bitmap pixels; border splines; database rows, strings; start-position REGION, settlement, slots, campaign-AI region and boundaries, **one HLCI per connected component containing all its areas**, camera bounds from the fixed rectangles |

After the files are written a **validation pass** runs and aborts the build on any problem: the regions topology; every settled region present in its theatre's pathfinding region list (with a single path) and the pathfinding rule checker on the changed theatres; the sea-grid lists reproducing from the geometry and complete connections between sea-bearing cells; no trade node without a route; the start position's region list equal to `regions.esf`, valid component-grouped HLCIs covering every area exactly once, and matching sea-cell counts and fleet membership. Cross-file navigation checks also run after atlas conversion. Review warnings (node clipped to its land zone, transition areas moved, ...) appear in the build report.

### Splitting a chain-activated region

A region activated with `areas`/`merge` (or carved) can be cut by a region split like any other. Because such a region has no pathfinding paths when the split runs, `build` synthesises **both halves** together after the splits (land zones, the border band and group path between them, landings of both coasts, a settlement node for each settlement), then runs the rule checker and the strict cross-file checks (exercised with a merged Burma cut at ring vertices 70 and 90). Splits of regions that already have ETW paths relabel their pieces to the new region and then run the split synthesis: pieces are assigned to the geometric side of the cut, the border band with the group path of the two halves is created and the new settlement gets its own type-7 node (a settlement point closer than about 2.1 u to the border or coast is refused). Where the untouched ETW grid already fails the rule checker (America, Europe), existing problems are retained as a baseline and every new problem blocks export. Verified on a merged Burma cut and on `orissa` (India grid): nodes and group paths present in the output, validator 0 problems.

When several areas of one region touch the other half of a split, their border strips are united by region membership before pathfinding cells are generated. Shared endpoints therefore cannot create overlapping pieces or duplicate group paths; merging the source region's areas is not required.

### Fixed theatre artwork (automatic) and optional painted override

The frames of the original theatres (wooden border and parchment margin) are baked into the ETW supertexture; neither the ESF bounds nor the camera limits can move them. Every build therefore repairs, **without any project setting**, the supertexture and raw heights around America `[[-888,0],[-340,460]]` and India `[[440,-80],[790,300]]`. The native painter samples the user's ETW palette, texture and frame, uses real ETOPO elevation with the authored per-theatre calibration, moves the frame outward, and erases the old frame/vignette without stretching or moving already-playable land. The verified snapshot repairs 667 America tiles and 466 India tiles over all seven mip levels. Europe's raw heights are unchanged; its shared frame seam with India and coarse DXT blocks straddling that seam can change.

`src/terrain/art/` performs the painting natively. `src/terrain/fixed_art.rs` caches the resulting sparse tiles and raw height runs under `workspace/derived/<fingerprint>/assets/terrain/fixed_art/`. Its manifest binds the recipe, fixed rectangles, original supertexture and heightmap hashes, public elevation data, and every generated file. Cache publication is atomic; invalid cache contents are regenerated from the snapshot. There are no embedded replacement-art ZIPs or Python runtime fallback. The only terrain-preparation developer tool retained is `tools/terrain_paint/prepare_dem.py`, which regenerates the shipped **public elevation data**, not game artwork.

Heights are patched **before** the relief curve and the fog mask is derived from the patched, shaped heights, so coast, relief and fog agree with the new land. `terrain original` skips only the relief shaping: the result equals the patched raw heightmap exactly (the fixed terrain is still included); it is no longer the unchanged ETW heightmap.

America's former interior edge is joined using a smooth source-minus-generated correction, rather than copying the nearest height or matching the dark parchment rim. The correction fades over 40 world units into generated land; original land is feathered over 20 units, excluding the first 8 units as reference material. Colour references also exclude black rock/underlay patches. Height blending preserves the land/sea mask. RGB-only repairs preserve the original DXT5 alpha blocks and artist-authored shoreline mip coverage instead of rebuilding them from colour changes; India's existing repaint policy is unchanged.

Winter snow continues the source latitude profile westwards, with an elevation-dependent snowline from the patched **raw** heights, independent of the relief setting. Both winter snow DDS paths are updated, including their mip tails; summer is unchanged. The source's clipped northern strip east of the bounded feather is retained, with a smooth transition rather than changing all original America. Both America climate rasters now continue their own source categories by latitude and coastal/inland class, not India's climate IDs. Existing climate pixels, wind maps and theatre geometry are preserved.

Older generated-art caches are automatically invalidated by the recipe version. Export again with the updated editor to regenerate the repair; no manual cache deletion or project edit is needed.

`painted_terrain` (`terrain set --painted DIR|none`) remains an **optional advanced override**: when set, its supertexture, heightmap and fog mask are copied over the generated assets after validating sizes and tile grids, replacing the automatic art. It must match the fixed geometry and supply matching height and fog data.

The **initial native port, before the America continuity repair**, was independently compared against the prior embedded build: the same 1,064 repaired tiles out of 10,922; all 9,858 untouched tiles byte-identical and every alpha value preserved. Native patched raw heights differed at three pixels by one byte value, with no sea/land flips; after default relief only one exported height pixel differed by one. America's fine texture differed because deterministic native patch selection resolves equal-score candidates differently from NumPy's unstable sorting: mean absolute RGB channel error 0.970/255 across the repaired tiles and mip levels (India: 0.000025/255). Compressed tile bytes also differed with the zlib encoder. That comparison is historical, not a claim that the later continuity repair preserves the old seam.


### Verification state

The validated experiment (India extended to `(440,-80)..(790,300)`, Burma merged from five areas with Rangoon, Sumatra carved from `unexplorable` with Aceh) loaded in game with the Burma union (armies walk to Calcutta, landings work). The Rust port is checked statically against that golden output (see *Verification performed for theatre extension*); nothing of the Sumatra carve has been played.

## Terrain relief

The ETW world heightmap is kept at its full 8192x4096 resolution, but ETW squeezes each continent into about 2.6 times fewer map units than Napoleon's own maps and paints uplands about three times higher (Alps: land mean 68, roughness 14.6 per pixel, against 20 and 3.0 for Switzerland on Napoleon's Europe map). Rendered by Napoleon that reads as noisy, bulging terrain. The project's **terrain relief** setting (Export review, or `terrain set|default|original`) shapes the exported `display/heightmap/heightmap.tga`: heights up to the *knee* are kept, everything above is scaled by the *upland scale*, then a land-only Gaussian blur smooths the result. Sea pixels and coastlines never change and land never drops to sea level; the fog mask derives from the shaped heights of the fixed-theatre-patched map, `metadata.dat` (battle terrain classes), regions, pathfinding and settlement heights are unaffected. The default (knee 24, uplands x0.4, blur 2 px) brings the Alps to mean 40 / p90 63 / roughness 1.8 and France to mean 24 / roughness 1.2, in the range of Napoleon's own maps. `terrain original` keeps the original relief curve (no shaping) but still includes the fixed-theatre terrain; the fog mask derives from those heights. Existing projects pick up the default on their next build.

## Command line

Run in the editor folder; quote paths containing spaces.

```powershell
.\KgcPortEditor.exe --help
.\KgcPortEditor.exe verify-sources
.\KgcPortEditor.exe tables requirements
.\KgcPortEditor.exe tables preview --folder "F:\mod-tsv" --language en
.\KgcPortEditor.exe tables import --folder "F:\mod-tsv" --language en --name "My target" --resources-reviewed --pin
.\KgcPortEditor.exe tables list
.\KgcPortEditor.exe tables check
.\KgcPortEditor.exe mapping search --kind units --side target --query Fusiliers
.\KgcPortEditor.exe mapping set --kind factions --from france --to mod_france
.\KgcPortEditor.exe mapping set --kind faction_templates --from mod_france --to france
.\KgcPortEditor.exe adapter import --folder "F:\merged-resources"
.\KgcPortEditor.exe analyze
.\KgcPortEditor.exe resolve list
.\KgcPortEditor.exe resolve set --issue "<reported id>" --choice "<offered choice>"
.\KgcPortEditor.exe assign --region alsace --faction britain
.\KgcPortEditor.exe theatre list
.\KgcPortEditor.exe activate add --key indochina --donor bengal --name Burma --settlement-name Rangoon --center 683.9,119.8 --colour d250be --areas 0,1,2,6,7 --merge
.\KgcPortEditor.exe activate add --key sumatra --carve-from unexplorable --areas 31 --donor ceylon --name Sumatra --settlement-name Aceh --center 724.612,-1.597
.\KgcPortEditor.exe terrain set --painted "F:\painted\nap_world"   # optional advanced override of the automatic fixed artwork
.\KgcPortEditor.exe project show
.\KgcPortEditor.exe build
```

`tables use --target ID` pins an imported dataset; `tables use --vanilla` explicitly selects original Napoleon data. Mappings/resolutions can be imported from JSON; `--help` documents their shapes. Preview/analyze exit 3 for blockers; `tables check` exits 4 for folder drift.

Use `--workspace DIR` for another workspace, `--project FILE` for another project on commands supporting it, and `--assets DIR` for the shipped schemas and recipes. Write destinations must not contain `..`. `inspect --startpos FILE` prints campaign owners.

To import untouched originals into a new workspace:

```powershell
.\KgcPortEditor.exe import --etw "F:\NTW2\Empire Total War" --ntw "F:\NTW2\Napoleon Total War" --language en --workspace "F:\NTW2\another_editor_workspace"
```

## Cutover from older projects

The mod-pack/profile/layer importer, profile UI, lazy external-pack overlays, `profiles` commands and `--profile` option have been removed. Original-game PFH0 import, native DB/LOC encoding and the campaign/map/terrain/CAI converter remain required foundations.

Project format is **10**. Versions 5–9 preserve ownership, target bindings, mappings, resolutions, activations, splits, terrain and painted-terrain choices; obsolete custom `theatres` rectangles are discarded. Earlier projects default to `atlas: null`, `building_scale: 1.0` and `settlement_moves: []`; they are not silently converted to the experimental layout. Migration is in memory until saving. Older editors refuse format 10 rather than exporting a different campaign. Version 1 preserves ownership. Version 2 preserves ownership and mappings; an old pack-profile target becomes an **unbound TSV target** requiring a fresh import, with obsolete bindings/adapters/decisions removed. It never becomes vanilla implicitly. Versions 3 and 4 preserve TSV bindings, adapters, mappings, decisions and ownership; the old recycling field is removed whether it was true or false. Regions whose original owner is absent from the target await explicit ownership assignment rather than being automatically reassigned or reported as missing-faction mappings. Invalid saved owners and future project versions remain errors.

## Build from Rust source

Rust 1.94.1 was used. On this workstation, keep build and temporary files on F:

```powershell
$env:CARGO_HOME = 'F:\NTW2\work\cargo-home'
$env:CARGO_TARGET_DIR = 'F:\NTW2\kgc_port_editor\target'
$env:CARGO_BUILD_JOBS = '1'
$env:TEMP = $env:TMP = 'F:\NTW2\work\tmp'
cargo clippy --release --locked --all-targets
cargo test --release --locked --all-targets
cargo build --release --locked -j1
```

The executable is `target/release/KgcPortEditor.exe`. Ship it with `assets/` from this source tree; do not include local workspaces or generated resources. The source-only release build requires the two community schemas and `assets/policy/` (including public DEM binaries), but no game installation or derived game assets. `-j1` avoids concurrent LLVM release compilation exhausting memory on this workstation. The application is entirely native Rust; RPFM is only needed by the user to prepare modded TSV exports. Python, NumPy and SciPy are optional developer dependencies only for regenerating the public DEM files.

`.github/workflows/pr.yml` runs independent **clippy (Windows)** and **test (Windows)** checks on every pull request and push to `main`, using Rust 1.94.1, locked dependencies and all Cargo targets. Clippy uses its default lint levels: errors fail the check, while existing style warnings remain visible and non-blocking. The workflow uses read-only repository permissions, commit-pinned actions and dependency caching; only pushes to `main` save the cache. Neither check requires game installations or a local workspace.

Release builds retain optimization level 3, but disable cross-crate LTO, use 16 code-generation units and enable incremental compilation. Keep the same Cargo home/target directory between edits so artifacts can be reused. A clean build or a profile/toolchain change must rebuild dependencies once. For type checking without producing an executable, use `cargo check --release --locked`.

On this workstation the one-time profile/dependency rebuild compiled in 49.34 seconds. A subsequent real source-edit release rebuild completed in 9.69 seconds. These are observed local timings, not guarantees for other machines.

## Verification performed for western America continuity

- **285 release tests passed** (283 library, 2 CLI), including damaged-rim/slope continuity, exclusion of invalid colour references, snow mip tails, latitude/coast climate continuation, and colour-only versus coverage-only edits across all seven supertexture mip levels.
- A full native CLI export succeeded from an isolated copy of the current 138-region project. Recipe changes regenerated the previous cache; a subsequent cached export also succeeded. Decoded colour, hillshade and winter-mask images were inspected.
- The 198,323-pixel land boundary sample's 95th-percentile height gradient fell from **4.03 to 2.55**; the 99th percentile fell from **6.02 to 3.91**. There were **zero land/sea height-mask changes** across the entire map.
- Compared with the previous native export, 378 texture tiles changed, all inside America's tile ancestry; **10,544 tiles are byte-identical**, including India. The file set is unchanged. Changed game files are limited to the supertexture pair, heightmap, fog mask, two winter DDS files and `regions.esf`.
- Winter's old 0-to-255 cutoff was replaced by a bounded feather: the maximum adjacent-pixel snow-mask step in the measured join fell from **255 to 49**. Summer, winter DDS headers and all winter-mask alpha blocks remain unchanged.
- Independently decoded America climate rasters preserve every original cell and use source-valid land/coast/ocean IDs. Both match across **99.91%** of the old boundary; the remaining differences are actual land/coast class boundaries. All other `regions.esf` records, including winds, theatre geometry and other theatres, match the prior export.
- All work stayed on F:, without installing the generated campaign or launching the game. These are exported-data and decoded-image checks, **not in-game rendering or gameplay verification**.

## Verification performed for native local rebuilding

- Built and tested from a separate source-only tree containing no prior extracted rules, UI or terrain bundles: **277 release tests passed** (275 library, 2 CLI).
- Cold import from both actual installation roots produced 4,391 snapshot files identical to the preserved original snapshot. All consumed rule data matched the embedded baseline; two JSON files differed only in formatting, and two unused notes files were removed.
- Full cold native CLI export: **679 game files**, the same set as the old executable; **675 byte-identical**. Differences are limited to the heightmap, supertexture data/index and campaign pack. Inside that pack, **17 of 21 members are byte-identical**, including every DB/localisation member and binary UI layout; the other four are locally generated Lua scripts.
- All six native decompilations were independently recompiled and compared to their original Lua 5.1 bytecode, including exact float32 constants. Generated scripts matched the prior UI behaviour in ten executable scenarios: campaign entry success/failure, travel accept/cancel/toggle-off, and faction-grid boundaries at 1, 9, 10, 18 and 19 factions. Development logging is removed; protected map opening is retained.
- Actual GUI cold import reached the 138-region ownership map and ready analysis, then **Save & Export ZIP** completed. All 679 game files matched the cold CLI export exactly. A separate cached `terrain original` export equalled the native patched raw heightmap byte-for-byte; default-relief coast and fog matched the prior build.
- Public DEM regeneration reproduced both shipped binaries and the authored calibration file byte-for-byte without any retired painter dependencies.
- Observed timings on this workstation: 45.0 s for the prior embedded export, 83.8 s for the native first export including painting, 34.4 s for the cached native original-relief export. These are different runs/settings, not a controlled speed benchmark.
- All verification work stayed on F:. The user's project hash and both installed-game file inventories were unchanged. No generated campaign was installed or launched; in-game appearance and gameplay remain unverified.

## Verification performed for fixed theatre bounds (format 9)

- `cargo test --release --locked`: 260 tests passed; release executable built.
- Actual CLI: read-only `theatre list` reports the fixed rectangles; `theatre set` and `theatre reset` refuse without changing the project. Loading the existing format-8 project drops its India x=800 choice, preserves every other project field, and leaves its original bytes unchanged.
- Native GUI on an isolated copy: loaded the 138-region project, rendered Western Plains, displayed the read-only bounds tooltip, and exposed no extension tool. Explicit **Save project** wrote format 9 without `theatres`, preserving all other choices.
- Full exports with the existing all-area Indochina activation and without activations passed the pipeline's cross-file checks. Both `regions.esf` theatre arrays contain the fixed rectangles; the other four theatres and Europe retain their source bounds. All 331 Western Plains outline vertices are preserved inside America.
- These automated checks used F: only, with no game installation or launch. The user subsequently confirmed that the combined fixed bounds load and allow the expanded camera movement. The new automatic artwork and broader gameplay still need manual in-game verification.

## Earlier verification of editable theatre extension (format 8)

Historical evidence below predates removal of the extension controls; its set/reset commands are no longer available.

- `cargo test --release`: 245 tests pass, including off-grid extension rejection without mutation, preservation of mixed fixed/corner-relative vertices, outward snapping and transition placement, same-region border-strip union, and canonical group-path remapping.
- CLI acceptance on a hard-linked copy of the imported sources: `theatre set --theatre 3 --rect 440,-80,790,300`; `activate add --key indochina --donor bengal --name Burma --settlement-name Rangoon --center 683.9,119.8 --colour d250be --areas 0,1,2,6,7 --merge`; `activate add --key sumatra --carve-from unexplorable --areas 31 --donor ceylon --name Sumatra --settlement-name Aceh --center 724.612,-1.597`; `build`.
- `cargo run --release --example compare_golden -- --generated GrandCampaign-NTW.zip --golden <golden data dir>`: Burma only against the in-game-validated variant B: 205 regions / 1023 areas equal in flags, bounding boxes and rings (1e-3), pathfinding 175x190 with validator 0 problems, identical path table and 305 landing halves, sea-grid boxes, lists and 66 connections identical, start-position check 0 problems (the then-used HLCI count check, since corrected in 0.7.2; region list; CAI sea cells), no routeless trade node. The one hard deviation of a raw comparison is `unexplorable` (63 areas against 61): variant B did not clip the backdrop islands the full experiment clips. Full (Burma + Sumatra) against the experiment's full output: 206 regions / 1086 areas, 428 landing halves equal, sea lists identical, 0 hard deviations.
- Deliberate differences: the merged Burma area takes component id **13** (the connected Bengal land) instead of the experiment's fresh 857; Sumatra and the impassable ids of clipped areas get fresh ids in a different order. That version also expanded HLCIs in 25 further regions. This was an incorrect one-per-area policy: 0.7.2 instead preserves valid stock component groups and repairs their area coverage.
- Native GUI smoke (isolated project copy, no game files touched): numeric and dragged theatre rectangle `[[440,-80],[790,300]]` saved exactly; Burma areas 0,1,2,6,7 chosen on the map and in the list (the list keeps the pre-merge indices), merged, Sumatra carved from `unexplorable` area 31; the exported campaign built (139 regions, 73 Burma and 34 Sumatra landings); a split of the merged Burma drawn with the mouse (snapped ring vertices 70 and 90) built too (140 regions; 69 + 4 landings, node and border group present).
- Theatres 1 (America) and 2 (Europe): extensions build and pass every cross-file check. Their untouched ETW pathfinding grids already fail the rule checker (2 and 28 problems), so the checker is applied differentially: problems are compared by world-cell identity against the untouched grid and any NEW problem aborts the build (India's baseline is empty, so it is checked absolutely). No experiment backs these two theatres.
- Off-grid regression: the unchanged saved TSV-target project with India `[[440,-83],[801,280]]`, America `[[-890,0],[-340,460]]`, all Indochina areas activated without merging, and Mandalay split from area 2 (vertices 6 to 78) now exports successfully: 139 regions, 41 factions, 47 ownership overrides, 680 files. India becomes `[[440,-84],[802,280]]`; its rule checker and an independent piece-overlap diagnostic report no problems. Native GUI numeric input and edge dragging were exercised on an isolated copy; applied bounds were saved on the 2-unit lattice. CLI fractional bounds snap outward and a shrinking request is refused without changing the saved rectangle.
- Not verified: the full Burma + Sumatra build and the splits have not been played in the game (only Burma-only variant B was).

## Verification performed for region splitting (0.6.0)

- `cargo test --release --locked`: 174 tests passed (unit tests cover the cut half a drawn line produces, crossed-cut rejection, cut validation, dependent-split removal, project v5 loading with empty splits, split rule derivation and owner reassignment).
- Format work was validated on the vanilla NTW Europe map before the editor integration: every regions/pathfinding/poi/sea_grids/trade_routes/startpos model round-trips the vanilla files byte-for-byte; the regions.esf spatial index rebuilt from geometry reproduces all 1453 vanilla cells' edge sets and probe regions; the sea-grid land lists reproduce vanilla exactly. The same models round-trip all seven theatres of the ETW world map.
- CLI smoke on the existing workspace: `split add` of Alsace (`alsace_southern_alsace`, settlement Mulhouse), `assign` of the new region to Britain, full `build`: 138 regions in the exported start position (Alsace France, the new region Britain), start-position consistency check passed, CAI check 0 problems, new rows and strings present once. A three-split chain (a split of a split plus Spain) also builds. A build without splits still yields 137 regions.
- The real native editor was driven through the whole flow (borders/towns overlay, source selection, snapped cut, halves preview, settlement placement, apply, list entry, removal) with read-only screenshot and accessibility inspection in a temporary workspace; `workspace/project.json` was not modified.
- Terrain relief: the default setting was measured on the exported heightmap (Alps mean 68 -> 40, roughness 14.6 -> 1.8; France 33 -> 24, 5.8 -> 1.2; every sea pixel unchanged, every land pixel >= 1) and a unit test pins the coastline/lowland/upland invariants. Its look in game is not verified.
- Wilderness activation, CLI on the existing workspace with temporary project files: `activate add` of `wilderness_great_plains` (donor `upper_louisiana`) plus `assign` to Britain builds with 138 regions in the exported start position. `wilderness_tejas` (donor `upper_louisiana`) activated and then cut by a split (`wilderness_tejas_tejas_east`) builds with 139 regions; every start-position addition passed its consistency check (0 problems), the `regions_tables` rows and the region, battle, settlement and rebel names are present in the pack. `wilderness_great_plains` itself cannot be split: its passable area has an inner hole (lake), which the cut tool does not support yet. The generated start position already holds the campaign-AI region record, boundaries, HLCIs, theatre entry and root beliefs of each wilderness polygon; an activation fills in the settlement, slots, world region id, governorship and owner of that record. Neighbouring regions without a settlement node (e.g. `great_plains`, `new_mexico`) get no trade route to an activated settlement.
- Nothing was played in game: the split regions, their pathfinding re-labelling and campaign-AI entries are static-checked only. A vanilla-map experiment (eur_lithuania -> eur_samogitia on the stock Europe campaign) is staged for a manual in-game test.

## Verification performed for minimum imports and existing-only conversion (0.5.0)

- `cargo test --release --locked`: 163 tests passed (161 library, 2 CLI); release executable built successfully.
- Native CLI requirements, preview, import and analysis succeeded with exactly 20 DB tables and two LOC files. Full export completed in 29.5 seconds with 674 game files, 137 regions, 41 represented factions and zero campaign-AI problems.
- Decoding the generated start position verified all 137 owners were existing target keys and all 47 explicit choices were preserved. All 20 supplied DB members matched their immutable dataset hashes.
- Removing land-unit strengths or the campaign-variable reference provider each produced exactly the missing required table. Import with missing strengths was refused without changing the project.
- A 278-member optional-input fixture omitted the six unrelated dependency tables named above. Requirements remained 22; analysis stayed ready despite within-member and cross-member uniform conflicts. Full export succeeded, preserving all 276 supplied DB members byte-for-byte, including 256 optional members.
- An existing target DB faction absent from the original start position blocked without a donor. Selecting an explicit French structural donor enabled full export; decoded ownership assigned Alsace to the existing fixture key.
- A version-4 project with recycling disabled migrated to version 5 without the old field and retained all 47 ownership choices. The removed CLI mode command was refused.
- Read-only native screenshot/accessibility inspection showed **Existing factions only**, no recycling checkbox, dark styling, larger text and a ready minimal-input dataset. Desktop focus and existing windows were preserved. All fixtures and exports stayed in an isolated workspace; user mod data, projects and game installations were untouched.
- No generated campaign was installed or launched in Napoleon. Export validation is not proof of gameplay compatibility.

The following sections record historical releases; their removed mode switches and broader import requirements do not describe 0.5.0.

## Verification performed for faction recycling (0.4)

- `cargo test --release --locked`: 154 tests passed (151 library, 3 CLI).
- Native CLI analysis reported 47 regions needing an explicitly chosen existing owner. An invalid owner was refused without changing the project file; a valid assignment succeeded while other ownership blockers remained.
- Vanilla recycling preserved all 137 region owners and 47 explicit choices, omitted 16 absent factions and their forces, and emitted no new faction DB rows.
- TSV recycling preserved 48 explicit choices, including a target-only faction absent from the original start position. Missing donor configuration blocked analysis; an explicit donor produced the target's own identity. All 78 emitted faction rows matched existing target rows, with zero new keys.
- Both recycling exports contained 674 game files; native output decoding confirmed existing-only campaign faction keys and campaign-AI consistency reported zero problems. Displaced French armies relocated after the ownership edit.
- With recycling off, all 674 game files were byte-for-byte identical to the previous release's output. The full conversion smoke took 16.9 seconds with the revised build profile.
- Read-only native inspection showed the checked checkbox, missing-owner warnings and disabled export. Desktop focus and existing application windows were preserved. No generated campaign was installed or launched in Napoleon.

## Verification performed for the TSV cutover

- `cargo test --release --locked`: 130 tests passed, covering codecs, immutable inputs, migrations, mappings, conflict decisions, ownership and write-safety boundaries.
- Real RPFM 4.7.4 exported 283 DB/LOC TSV members from an isolated original-table fixture. Import and analysis succeeded. Faction RGB values and both complete LOC files round-tripped to the original decoded rows.
- The vanilla ZIP retained the exact 674-file game set and identical bytes, with semantic comparison for startpos.
- A 284-member modded dataset exercised faction/unit/building mappings, donor identity, duplicate-key resolution, French/Unicode text and merged UI/scripts. All supplied members reached the pack. Upstream RPFM independently decoded the generated strengths, colours and text.
- Native output inspection confirmed nine mapped starting units at 217 men, one winning 223-men DB row for the conflicting source unit, 76 mapped building strings, target RGB 12/34/56 and 137 region owners.
- Assigning Alsace to Britain survived full regeneration and relocated the displaced French armies. CAI verification reported zero problems.
- Windows `Expand-Archive` extracted vanilla and modded ZIPs. Malformed/incomplete inputs, duplicate members, unsafe paths, missing dataset bindings and unreviewed resources were refused. External folder drift did not alter pinned inputs or overwrite the last ZIP.
- Read-only inspection verified the native workflow tabs and resolved ownership map without mouse, keyboard or focus actions. No generated campaign was installed or launched in Napoleon.

## Verification performed for dependency preview (0.4.1)

- 156 tests passed, including populated versus empty reference cells, empty tables, dependencies through selected tables, and absent opaque resources.
- An isolated copy of the 123-member mod dataset produced 38 baseline requirements plus 11 discovered dependencies. All six missing dependencies previously shown only during Compatibility were reported before import, with their referring fields.
- Import refused that incomplete fixture without creating a dataset. A complete 283-member RPFM dataset imported and analyzed successfully.
- The actual native import panel rendered the expanded checklist and disabled Import for missing dependencies. Read-only accessibility and screenshot inspection preserved desktop focus; no user inputs, datasets, projects or game files were modified.

## Verification performed for duplicate policy and styling (0.4.2)

- `cargo test --release --locked`: 160 tests passed. New regressions cover untouched-row preservation, critical within-member conflicts, explicit critical member winners, and empty/non-empty staged-table transitions with saved choices.
- An isolated 284-member TSV fixture contained both within-member and cross-member uniform conflicts. Version 0.4.1 blocked it; 0.4.2 reported both as warnings with recycling enabled and completed a real campaign ZIP export.
- All 282 imported DB members in that generated pack matched their immutable dataset SHA-256 hashes, including both conflicting uniform members. No rows or member identities were discarded.
- Turning recycling off made both uniform conflicts blocking again because converted uniform rows were generated. A conflicting land-unit strength remained blocking even with recycling on; an explicit member choice resolved it.
- Export with the unresolved strength conflict was refused, leaving the previous ZIP byte-for-byte unchanged.
- Read-only screenshot/accessibility inspection of the actual native editor verified dark styling, larger text, the ownership controls and export availability. Desktop focus and existing windows were preserved. All fixture changes and exports stayed in an isolated workspace; user projects, mod data and game installations were not modified.
