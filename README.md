# Issaquah Development Agent — D2 Pipeline

Builds the parcel, capacity, and constraint digests that ground the Issaquah
Development Agent's feasibility and forecast answers.

**Inputs:** King County Assessor extracts, City of Issaquah GIS services,
FEMA NFHL, WA DNR, King County SAO, and one hand-keyed zone standards matrix.
**Outputs:** four markdown files for the knowledge base, plus working data that
stays in the repo.

Run time, end to end: roughly 45–120 minutes on ~15,000 parcels, almost all of
it in constraint clipping.

---

## What this produces

| File | Goes in the knowledge base | Contents |
|---|---|---|
| `out/D2A_headroom.md` | **yes** | Development capacity by zone and subarea |
| `out/D2B_constraints.md` | **yes** | Constraint incidence, cost, schedule, overlap QA |
| `out/D2C_candidates.md` | **yes** | Ranked redevelopment candidates by subarea |
| `out/D2_manifest.md` | **yes** | Field dictionary, coverage, provenance, regeneration |
| `out/parcels_master.parquet` | no | S1 working output |
| `out/parcels_capacity.parquet` | no | S2 working output, feeds the forecast notebook |
| `out/S1_coverage.json` | no | Join coverage report |
| `out/S2_qa.json` | no | QA report — read it every run |

**Upload the four markdown files and nothing else.** The parquet files are the
full per-parcel join; the digests exist precisely so the knowledge base does not
have to carry 15,000 rows. Volume dilutes retrieval, and a bigger upload can
produce a worse agent.

Replace all four together as one set. A digest from one run sitting beside a
digest from another makes the agent hedge, and it is right to.

---

## Prerequisites

### Python environment

GDAL and PROJ are easier through conda than pip.

```bash
conda create -n issaquah python=3.11
conda activate issaquah
conda install -c conda-forge geopandas'>=1.0' shapely'>=2.0' pandas pyarrow \
    requests pyyaml openpyxl tabulate
```

Shapely 2.x is required — the combination functions in `d2_constraints.py` use the 2.x geometry API. GeoPandas 1.0+ is required for `read_parquet` / `to_parquet` round-tripping of geometry.

Verify:

```bash
python -c "import geopandas, shapely; print(geopandas.__version__, shapely.__version__)"
```

### Repository layout

```
issaquah-forecast/
├── config.yaml                    # endpoints, product forms, explicit defaults
├── d2_constraints.py              # constraint registry + combination rules
├── s1_parcel_master.py            # assessor + GIS → parcel master
├── s2_capacity.py                 # A1 standards + constraints → capacity
├── s2_digests.py                  # capacity → the four markdown digests
├── kb/
│   └── A1_zone_standards.csv      # HAND-KEYED — see below
├── data/
│   ├── assessor/2026-09-01/       # five CSVs, one directory per vintage
│   └── gis/                       # optional parcel shapefile fallback
└── out/                           # all generated output
```

### A1 zone standards matrix — build this first

`kb/A1_zone_standards.csv` is hand-keyed from IMC Title 18 and cannot be scripted. The pipeline will not run without it.

Required columns, exactly these names:

```
zone, max_du_per_acre, max_du_per_lot, min_lot_area_sf, max_far,
max_height_ft, max_lot_coverage, parking_min_per_unit, setback_min_ft,
mh_eligible, densest_product, process_type
```

Two columns are additions you will not find in the code and have to decide:

- `densest_product` must key to a name in `config.yaml → product_forms`(`adu`, `middle_housing`, `cottage`, `townhouse`, `stacked_flats`, `podium`, `tod_midrise`). Without it no building footprint exists, no deep-foundation adder fires, and Central Issaquah comes back looking cheap to build on.
- `process_type` keys to the schedule table in A2 Part 2.

`mh_eligible` is a boolean: does HB 1110 middle housing apply in this zone.

Blank cells are read as "no such standard" and that standard is skipped — it does **not** mean zero. A blank `max_du_per_acre` with a populated `min_lot_area_sf` is a valid row.

---

## One-time setup

### 1 · Download the assessor extracts

From the King County Assessor Data Download page, pull these five into `data/assessor/<YYYY-MM-DD>/`, named exactly:

```
EXTR_RPAcct_NoName.csv
EXTR_Parcel.csv
EXTR_ResBldg.csv
EXTR_AptComplex.csv
EXTR_CondoUnit.csv
```

Use the download date as the directory name. That date becomes `--parcel-vintage` and is stamped into every digest section.

These are large (the account file runs to hundreds of MB) and encoded latin-1. Do not open and re-save them in Excel — it will mangle the leading-zero Major/Minor fields that form the PIN.

### 2 · Decide the parcel geometry source

`config.yaml → parcels.mode`:

- `service` (default) pulls from the city's ParcelsRelated service. Clipped to Issaquah, already EPSG:2926, no download. May lag the county on very recent splits.
- `shapefile` reads `data/gis/parcel_extr.shp`. Download `parcel_address_area` from the King County SDC. 636,698 features countywide — use the bulk download, never REST.

Start with `service`. Switch if S1 coverage drops or a known split parcel is missing.

### 3 · Resolve the config VERIFY markers

`config.yaml` ships with five field names marked `VERIFY`. Each is one probe call, no data fetched:

```bash
python s1_parcel_master.py --probe zoning
python s1_parcel_master.py --probe subareas
python s1_parcel_master.py --probe soil_infiltration
python s1_parcel_master.py --probe stream_buffer
python s1_parcel_master.py --probe wetland_buffer
```

Each prints the layer name, geometry type, and every field with its coded values. Fill the matching `field:` / `buffer_field:` / `feasible_values:` entry in `config.yaml`.

**Check the zoning probe hardest.** If the returned values do not key to A1's `zone` column, you have the Comprehensive Plan land use designation layer rather than the zoning layer. They look similar and would silently mis-key every parcel. S2 raises on this, but catching it at the probe saves an hour.

`soil_infiltration.feasible_values` **is the highest-value five minutes in setup.** While it is empty the CARA detention-vault adder fires on every CARA parcel across the valley, which overstates cost citywide.

### 4 · Fix the one known config/registry conflict

Delete `flag_only: true` from the `landslide_affected_parcels` entry in `config.yaml`. The registry field of the same name now governs and is enforced by `validate()`; leaving the config line in makes S2 skip the layer entirely, so the flag never reaches D2B.

### 5 · Inspect the registry before the first run

```bash
python d2_constraints.py
```

Prints every layer with its capacity mode, fraction, cost rate, denominator, group, and schedule effect, plus warnings for proxies and unresolved references. Scan two things:

- All four `deep_foundation` members read `footprint_sf`. A member on `lot_sf`would dwarf the others and win the group for the wrong reason.
- Nothing in `earthwork` shares a denominator with something in `stormwater`by accident.

---

## Running the pipeline

Three commands, in order. Each reads the previous one's output.

```bash
# 1 · parcel master: assessor + GIS, clipped to city limits
python s1_parcel_master.py --vintage 2026-09-01

# 2 · capacity: A1 standards + constraint registry
python s2_capacity.py --acknowledge-unverified

# 3 · digests: the four markdown files
python s2_digests.py \
    --run-id D2-2026-10-01 \
    --parcel-vintage 2026-09-01 \
    --zoning-vintage 2026-10-01 \
    --a1-version 1.0 \
    --acknowledge-unverified
```

Use a `--run-id` that sorts and is unique per run. It appears in every digest section and is what the agent cites.

`--parcel-vintage` must match the assessor directory name. `--zoning-vintage` is the date you pulled the city GIS layers — `config.yaml → zoning_vintage`.

### About `--acknowledge-unverified`

Eight constants in `VERIFY_BEFORE_RUN` are unverified assumptions. Without the flag, every script refuses to run. With it, every script runs and stamps a banner into all four digests saying the figures are estimates, not findings.

That banner is load-bearing. Per A2 Part 2 failure behavior, a verdict resting on an unverified constraint is an upper bound and must say so. Drop the flag only after setting `verified=True` on the corresponding registry entries, and only once each has a real local source behind it.

---

## The three hard gates

These stop a run rather than warn. Each is the gate working, not a bug — unless the troubleshooting table says otherwise.

| Gate | Where | Why it is a stop |
| --- | --- | --- |
| Unverified constants | all three scripts | Prevents a reportable number resting on a guess without the estimate label |
| Zone coverage &lt; 95% | `s2_digests.check_coverage` | Partial-coverage aggregates read as complete while silently understating capacity. Nothing in the agent's answer would reveal it |
| Zone on a parcel absent from A1 | `s2_capacity.main` | Produces zero allowed units, which reads as "no headroom" rather than "no standard keyed" |

---

## Reading the QA output

### `out/S1_coverage.json`

| Field | What to expect |
| --- | --- |
| `zone_resolved` | ≥ 0.95, or the digests will refuse |
| `subarea_resolved` | ≥ 0.98. Lower means the subarea layer does not cover the whole city |
| `near_transit` vs `near_transit_centroid_basis` | A spread of a few hundred parcels is normal. A spread of thousands means the park-and-ride parcel is large enough that the basis choice matters materially |
| `in_development_agreement` | `null` means the DA layer did not fetch and `VESTED_OVERRIDE` has no source |
| `zone_split` | Parcels spanning two zones, assigned by dominant overlap. A large count is worth a look |

### `out/S2_qa.json` — four numbers first

`union_vs_naive_units`**.** Capacity the naive per-layer sum would have removed beyond what the geometric union removes. Near zero means steep slope, landslide and erosion are not actually overlapping and the union machinery is doing nothing. Above \~30% of the union figure means those three layers are close to redundant and should probably be collapsed into one registry entry. This is also the figure D2B publishes as evidence the double-count was avoided.

`layers_zero_hits`**.** A joined layer hitting no parcels either did not join correctly or genuinely is not present in Issaquah. `layers_skipped` separates those two cases.

`no_product_assigned`**.** Parcels whose zone has no `densest_product` in A1. Each one gets `_default` form factors, so its site adders are wrong. Should be zero.

`mh_units_from_transit_floor`**.** Citywide units attributable to the 4-unit middle-housing floor near a major transit stop. This rests on an unsettled definition — see below — and belongs in any forecast answer leaning on middle-housing capacity.

---

## Troubleshooting

| Message or symptom | Cause | Fix |
| --- | --- | --- |
| `Zones on parcels but absent from A1: [...]` | Either A1 is missing rows, or `layers.zoning` points at the Comprehensive Plan land use layer | If the listed values look like land use designations, re-probe and repoint. Otherwise add the rows to A1 |
| `Zone join coverage is NN%, below the 95% floor` | Wrong `zoning_field`; unpaged layer fetch truncating; parcel geometry from a different source than the zoning layer | Re-probe zoning. Check the fetch printed the full feature count |
| `... ignores resultOffset — paginate by OBJECTID instead` | Older ArcGIS Server build | Not auto-handled. Download the layer manually and point at a local file, or add OBJECTID windowing to `fetch_layer` |
| `buffer_field 'WATER_TYPE' absent` | The configured stream layer is cartographic and carries no type classification | Switch `layers.stream_buffer` to the King County SAO streams layer or WDFW SWIFD. Do not buffer every stream at one width — the raise is deliberate |
| `unmapped CATEGORY values ['...']` | The layer uses a different coding than the registry map | Read IMC 18.802, add the values to `buffer_by_field` in `d2_constraints.py` |
| `No configured transit-stop parcel resolved` | PIN not zero-padded to 10, or the parcel layer excludes those parcels | Check the PINs against the parcel layer. Capacity on every middle-housing parcel would otherwise drop from 4 units to 2 |
| `flag_only layer carries capacity_mode ...` | A registry edit gave a parcel-polygon layer a capacity effect | Remove the capacity mode. The validator is preventing a layer that would zero every flagged parcel |
| Several layers in `layers_skipped` with fetch errors | Service moved, or countywide layer timing out | Check `use_envelope: true` is set on all King County, DNR and FEMA entries |
| S2 runs for hours | Constraint clipping on a large feature set | Confirm the envelope filter is applied. Without it the SAO and DNR layers return tens of thousands of features |
| A digest section reads `—` where a number belongs | Missing data, rendering correctly | Check the coverage row in the same section. Missing never renders as zero, by design |

---

## Refresh cadence

| Trigger | Re-run | Notes |
| --- | --- | --- |
| New assessor extract | Full pipeline | Annual. New `--vintage` directory |
| Title 18 amendment | A1 by hand, then full pipeline | A1 is the only hand-keyed input |
| City GIS layer update | Full pipeline | Bump `zoning_vintage` |
| New FEMA FIRM | Full pipeline | The city flood layers are a 08/19/2020 snapshot and go stale silently. Compare against the NFHL cross-check annually |
| A2 constants reach a review date | Forecast only | Does not affect D2 |

Run the full pipeline rather than a partial one. The four digests are a set and their stamps must agree.

---

## Known gaps carried into every run

These are stated in the digests, not hidden. Listed here so whoever inherits the repo knows they are decisions, not oversights.

- `high_groundwater` **has no published layer.** The dewatering adder never fires. Deriving it from Ecology well logs is a project, not a join. Every below-grade verdict is an upper bound.
- `frontage_improved` **is a config default, not data.** No published layer separates sidewalk from road pavement. Shipped as `true`, which never charges the $1,200/LF adder and understates greenfield cost. S2 reports the parcel count affected. Changing it to `false` manufactures a "does not pencil" conclusion — do not flip it without saying so in the manifest.
- **Three constraints are King County 1990 SAO proxies**, not Issaquah's own IMC 18.802 designations: seismic hazard, erosion hazard, coal mine hazard. The coal mine layer has no planned update. Issaquah publishes its own `coal_mines` service — probe it and prefer it if it carries a hazard polygon.
- `s2_digests` **does not yet render** `proxy_for`**.** Those three layers appear in D2B as though they were the city's own designations. Add a Proxy column when that file is next touched.
- **"Major transit stop" is unsettled.** Washington carries more than one statutory definition, Issaquah has adopted no map, and the determination currently happens at permit intake. The two configured parcels — Issaquah Transit Center (2924069143) and Issaquah Highlands Park and Ride (2224069158) — are the robust cases because they are fixed facilities rather than frequency-dependent stops, but the list could grow without any Council action if bus service improves on a corridor. The default buffers radially from the parcel boundary, which over-counts against a walking-distance standard: treat transit-floor capacity as an upper bound.

---

## Downstream

`out/parcels_capacity.parquet` is the input to the forecast notebook (S5), which produces `D5_baseline.md`, `D5_scenarios.md`, and `D5_elasticity.md`. Those are a separate run on a separate cadence — quarterly rather than annual — and are not produced by this pipeline.


