"""
d2_constraints.py — Constraint layer registry for the D2 capacity & constraint digest.

Companion to A2 Method Spec v0.2 (site adder table). Consumed by s2_capacity.py
and reported by s2_digests.py.

Version 0.3 · 2026-10-01. Changes from 0.2:
  - encumbered_fraction() honors capacity_fraction below 1.0. The prior version
    appended a graded layer's whole overlap to the union, removing 100% of
    floodplain area on a layer registered at 25%.
  - capacity_factor() applies PROHIBIT to the area it covers, not the whole
    parcel. A parcel half in the floodway keeps the other half.
  - encumbered_fraction_naive() added to feed the D2B overlap QA line.
  - seismic_hazard switched off DNR NEHRP site class to the King County SAO
    hazard polygon; erosion and coal mine likewise sourced from the county.
    All three are marked proxy_for the city's own IMC 18.802 designation.
  - landslide_hazard repointed at the WADNR inventory sublayers; the city's
    "landslide affected parcels" layer registered separately as flag_only.
  - layer_ref now names a config.yaml key rather than a guessed URL. Endpoints
    live in config; semantics live here.

Tagging: [M] market-sourced, [B] regional benchmark, [A] assumption.
All seven site adders remain [A]. The validator refuses to pass unverified
values unless explicitly acknowledged, by design.

Buffer widths and capacity fractions must be checked against IMC 18.802 before
any run that produces a reportable number. They are placeholders chosen to be
plausible, not readings of the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

BASIS_DATE = "2026-10-01"
SPEC_VERSION = "A2 v0.2"
REGISTRY_VERSION = "0.3"


# ──────────────────────────────────────────────────────────────────────────────
# Semantics
# ──────────────────────────────────────────────────────────────────────────────

class Basis(str, Enum):
    """Denominator for a cost adder.

    Getting this wrong was the v0.1 bug: foundation work scales with building
    footprint, not lot area, and frontage improvements scale with linear feet
    of street. One adder applied to the wrong denominator is off by an order
    of magnitude on a large lot.
    """
    LOT_SF = "lot_sf"
    FOOTPRINT_SF = "building_footprint_sf"
    BELOW_GRADE_SF = "below_grade_footprint_sf"
    FRONTAGE_LF = "street_frontage_lf"
    ENCUMBERED_SF = "constraint_area_within_parcel_sf"
    SLOPED_SF = "sloped_area_developed_sf"


class CapacityMode(str, Enum):
    NONE = "none"              # cost and/or schedule only; capacity untouched
    AREA_SHARE = "area_share"  # removal proportional to measured overlap
    FLAT = "flat"              # fixed fraction on any intersection (siting penalty)
    PROHIBIT = "prohibit"      # capacity -> 0 on the area covered


class Op(str, Enum):
    INTERSECT = "intersect"          # parcel touches the polygon
    CENTROID = "centroid_within"     # parcel centroid inside (zoning-like layers)
    BUFFER = "buffer_then_clip"      # buffer the feature, then measure overlap


class Tag(str, Enum):
    MARKET = "M"
    BENCHMARK = "B"
    ASSUMPTION = "A"


# ──────────────────────────────────────────────────────────────────────────────
# Product predicates — a constraint can be conditional on what is being built
# ──────────────────────────────────────────────────────────────────────────────

PREDICATES: Mapping[str, Callable[[Mapping], bool]] = {
    "always": lambda p: True,
    "has_below_grade": lambda p: p.get("below_grade_levels", 0) > 0,
    "lot_line_excavation": lambda p: (
        p.get("below_grade_levels", 0) > 0 and p.get("setback_min_ft", 99) < 10
    ),
    # infiltration_feasible is joined from the city's Soil Infiltration
    # Categories layer in S1. Where it is unavailable the config default is
    # False, so the vault adder fires everywhere — an overstatement, reported
    # by S1 rather than silently absorbed.
    "no_infiltration": lambda p: not p.get("infiltration_feasible", False),
    "frontage_unimproved": lambda p: p.get("frontage_improved", True) is False,
}


# Table 18.802.220(B), Wetland Buffer Ratings. Keyed (category_group,
# habitat_band) -> width by intensity of impact of adjacent land use.
#
# Categories I and II share every row — the code groups them. Categories
# I, II and III are identical at habitat >= 6 and differ by 20 ft at high
# intensity in the low band. Category IV is the only classification that
# materially changes a buffer. Any classification dispute that does not
# turn on "IV or not IV" moves a buffer by at most 20 ft.
WETLAND_BUFFER_TABLE: Mapping[tuple[str, str], Mapping[str, float]] = {
    ("I_II", "bog_hcv"): {"low": 125, "moderate": 190, "high": 250},
    ("I_II", "hab_8_9"): {"low": 150, "moderate": 225, "high": 300},
    ("I_II", "hab_6_7"): {"low":  75, "moderate": 110, "high": 150},
    ("I_II", "hab_lt6"): {"low":  50, "moderate":  75, "high": 100},
    ("III",  "hab_8_9"): {"low": 150, "moderate": 225, "high": 300},
    ("III",  "hab_6_7"): {"low":  75, "moderate": 110, "high": 150},
    ("III",  "hab_3_5"): {"low":  40, "moderate":  60, "high":  80},
    ("IV",   "any"):     {"low":  25, "moderate":  40, "high":  50},
}

# Category -> group used by the table.
WETLAND_CATEGORY_GROUP = {"I": "I_II", "II": "I_II", "III": "III", "IV": "IV"}

# Habitat bands available per group. Category III has no row below habitat 3,
# and the I/II low-band row additionally requires a water quality score of
# 8-9. Neither gap is covered by the published table; resolve_wetland_width
# raises rather than interpolating.
WETLAND_HABITAT_BANDS = {
    "I_II": {"low": "hab_lt6", "moderate": "hab_6_7", "high": "hab_8_9"},
    "III":  {"low": "hab_3_5", "moderate": "hab_6_7", "high": "hab_8_9"},
    "IV":   {"low": "any",     "moderate": "any",     "high": "any"},
}

# Shifting a category up one is the legacy Class 1-3 -> Category II-IV
# crosswalk. Only III -> IV changes a buffer row meaningfully.
WETLAND_LEGACY_SHIFT = {"I": "II", "II": "III", "III": "IV", "IV": "IV"}


# ──────────────────────────────────────────────────────────────────────────────
# Classification vocabularies
#
# A lookup table keyed to one vocabulary and normalized under another is the
# failure that assigned Category I buffers to legacy Class 1 wetlands. Declaring
# the scheme makes the mismatch checkable instead of silent.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassScheme:
    """A classification vocabulary, its token aliases, and its ambiguity rule.

    Lives here, not in s2_capacity, because validate() and the runtime
    normalizer must agree on the scheme name — and they have now drifted
    twice. One definition, both readers.
    """
    name: str
    classes: tuple[str, ...]
    aliases: Mapping[str, str]
    numerals_ambiguous: bool = False
    ambiguity_note: str = ""


# Every state that means "no classification." A 0 is not a category under any
# scheme, and an empty string is not a null — both must normalize to missing,
# or a feature gets a buffer it was never rated for.
MISSING_CLASS_TOKENS = frozenset(
    {"", "NAN", "NONE", "NULL", "NA", "0", "UNK", "UNKNOWN", "-", "TBD"})

_ECOLOGY_ALIASES = {
    "I": "I", "1": "I", "CATEGORYI": "I", "CATI": "I", "CLASS1": "I",
    "II": "II", "2": "II", "CATEGORYII": "II", "CATII": "II", "CLASS2": "II",
    "III": "III", "3": "III", "CATEGORYIII": "III", "CATIII": "III", "CLASS3": "III",
    "IV": "IV", "4": "IV", "CATEGORYIV": "IV", "CATIV": "IV", "CLASS4": "IV",
}

_WDFW_ALIASES = {
    "S": "S", "TYPES": "S", "F": "F", "TYPEF": "F",
    "NP": "Np", "TYPENP": "Np", "NS": "Ns", "TYPENS": "Ns",
}

CLASS_SCHEMES: Mapping[str, ClassScheme] = {
    "ecology_category": ClassScheme(
        name="ecology_category",
        classes=("I", "II", "III", "IV"),
        aliases=_ECOLOGY_ALIASES,
        numerals_ambiguous=True,
        ambiguity_note=(
            "A bare 1-4 may be an Ecology 2014 Category or a legacy Issaquah "
            "Class 1-3. The legacy reading shifts every value up one category, "
            "which for Class 3 -> Category IV is a 30-100 ft buffer swing. "
            "Set numeral_meaning: category in config to declare the reading."),
    ),
    "wdfw_water_type": ClassScheme(
        name="wdfw_water_type",
        classes=("S", "F", "Np", "Ns"),
        aliases=_WDFW_ALIASES,
    ),
    # Vocabulary unread — probe IsqRating's coded values before populating.
    # An empty aliases map makes normalize_class raise rather than silently
    # mapping nothing.
    "issaquah_stream_rating": ClassScheme(
        name="issaquah_stream_rating", classes=(), aliases={}),
}

# Multi-dimensional buffer tables, registered by name so a layer can reference
# one without the registry importing it at definition time.
BUFFER_TABLES: Mapping[str, Mapping[tuple[str, str], Mapping[str, float]]] = {
    "WETLAND_BUFFER_TABLE": WETLAND_BUFFER_TABLE,
}

BUFFER_TABLE_AXES: Mapping[str, dict] = {
    "WETLAND_BUFFER_TABLE": {
        "category_group": WETLAND_CATEGORY_GROUP,
        "bands": WETLAND_HABITAT_BANDS,
        "scheme": "ecology_category",
        "intensities": ("low", "moderate", "high"),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Records
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostAdder:
    rate: float
    basis: Basis
    tag: Tag
    group: str | None = None      # adders sharing a group do not stack; max applies
    applies_if: str = "always"    # key into PREDICATES
    note: str = ""

@dataclass(frozen=True)
class ConstraintLayer:
    key: str
    label: str
    imc: str
    publisher: str
    layer_ref: str
    op: Op

    capacity_mode: CapacityMode = CapacityMode.NONE
    capacity_fraction: float = 0.0

    # Width resolution, in descending precedence. A layer may declare several;
    # width_sources() states the order and validate() reports it.
    buffer_width_field: str | None = None          # published per-feature width
    buffer_table: str | None = None                # multi-axis table by name
    buffer_by_field: Mapping[str, float] | None = None   # flat class lookup
    buffer_ft: float | None = None                 # single fixed width

    buffer_field: str | None = None                # primary classification field
    buffer_field_fallback: str | None = None       # per-feature fallback field
    buffer_class_scheme: str = ""                  # key into CLASS_SCHEMES
    buffer_width_range_ft: tuple[float, float] = (5.0, 400.0)
    setback_beyond_buffer_ft: float = 0.0          # IMC setback outside the edge

    cost: CostAdder | None = None
    predev_months_add: float = 0.0

    flag_only: bool = False
    proxy_for: str = ""
    verified: bool = False
    note: str = ""


def width_sources(layer: ConstraintLayer) -> list[str]:
    """Declared width sources in the order apply_buffer() consults them."""
    order = []
    if layer.buffer_width_field:
        order.append(f"published field {layer.buffer_width_field!r}")
    if layer.buffer_table:
        order.append(f"table {layer.buffer_table}")
    if layer.buffer_by_field:
        order.append(f"class lookup on {layer.buffer_field!r}")
    if layer.buffer_ft:
        order.append(f"fixed {layer.buffer_ft:g} ft")
    return order


def uses_classification(layer: ConstraintLayer) -> bool:
    return bool(layer.buffer_table or layer.buffer_by_field)


# ──────────────────────────────────────────────────────────────────────────────
# The registry
# ──────────────────────────────────────────────────────────────────────────────

CONSTRAINT_LAYERS: Sequence[ConstraintLayer] = (

    # ── Geotechnical: cost, not capacity ──────────────────────────────────────
    ConstraintLayer(
        key="liquefaction",
        label="Liquefaction susceptibility (moderate to high)",
        imc="18.802 Art. II — seismic hazard areas",
        publisher="WA DNR, Palmer et al. OFR 2004-20",
        layer_ref="config: layers.liquefaction",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.NONE,
        cost=CostAdder(
            rate=40.00, basis=Basis.FOOTPRINT_SF, tag=Tag.ASSUMPTION,
            group="deep_foundation",
            note="Highest-consequence number in the model. One Central "
                 "Issaquah geotech bid replaces it.",
        ),
        note="Drives the Central Issaquah 'pilings problem'. Does not remove "
             "capacity — it removes feasibility, which is forecast step 4, "
             "not step 2. Regional screening at roughly 1:100,000; never a "
             "substitute for a site-specific geotechnical report.",
    ),
    ConstraintLayer(
        key="peat_soft_soils",
        label="Peat / organic / soft compressible soils",
        imc="18.802 Art. II",
        publisher="WA DNR, same dataset as liquefaction",
        layer_ref="config: layers.peat_soft_soils",
        op=Op.INTERSECT,
        cost=CostAdder(
            rate=40.00, basis=Basis.FOOTPRINT_SF, tag=Tag.ASSUMPTION,
            group="deep_foundation",
            note="Same scope as liquefaction — grouped so a parcel hitting "
                 "both is charged once.",
        ),
        note="DNR maps peat separately within the liquefaction dataset "
             "because peat is not liquefiable, but it remains subject to "
             "permanent ground deformation and requires site-specific "
             "analysis under the IBC. SSURGO is no longer needed: one "
             "publisher, one attribute filter.",
    ),
    ConstraintLayer(
        key="seismic_hazard",
        label="Seismic hazard area",
        imc="18.802 Art. II",
        publisher="King County DPER, 1990 Sensitive Areas Ordinance",
        layer_ref="config: layers.seismic_hazard",
        op=Op.INTERSECT,
        cost=CostAdder(
            rate=28.00, basis=Basis.FOOTPRINT_SF, tag=Tag.ASSUMPTION,
            group="deep_foundation",
            note="Frequently coincident with liquefaction. Grouped; the "
                 "lower rate loses to the $40 layers rather than adding.",
        ),
        proxy_for="City of Issaquah IMC 18.802 seismic hazard designation",
        note="SWITCHED from DNR NEHRP site class, which is a soil "
             "classification rather than a hazard area: it required an "
             "unverified attribute filter and unfiltered would hit every "
             "parcel and silently win the deep_foundation group. This layer "
             "is a hazard polygon with the right semantics, but it carries "
             "the county's 1990 definition, not the city's.",
    ),
    ConstraintLayer(
        key="coal_mine_hazard",
        label="Coal mine hazard area",
        imc="18.802 Art. II — coal mine hazard",
        publisher="King County DPER, 1990 SAO (67 polygons countywide)",
        layer_ref="config: layers.coal_mine_hazard",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.FLAT,
        capacity_fraction=0.15,
        cost=CostAdder(
            rate=55.00, basis=Basis.FOOTPRINT_SF, tag=Tag.ASSUMPTION,
            group="deep_foundation",
            note="Grouting or void remediation. Beats the other "
                 "deep-foundation rates when present.",
        ),
        predev_months_add=4.0,
        proxy_for="City of Issaquah IMC 18.802 coal mine hazard designation",
        note="Issaquah-specific and materially underweighted in generic "
             "models. Carries both a siting penalty and a study period. The "
             "county source covers abandoned workings — adits, drifts, "
             "tunnels, air shafts — on a 1990 basis with no planned update. "
             "Issaquah publishes its own coal_mines service; probe it, and "
             "if it carries a hazard polygon prefer it and demote this to a "
             "cross-check.",
    ),
    ConstraintLayer(
        key="erosion_hazard",
        label="Erosion hazard area",
        imc="18.802 Art. II",
        publisher="King County DPER, 1990 SAO",
        layer_ref="config: layers.erosion_hazard",
        op=Op.INTERSECT,
        cost=CostAdder(
            rate=2.50, basis=Basis.LOT_SF, tag=Tag.ASSUMPTION,
            group="earthwork",
            note="Enhanced TESC, seasonal work restrictions.",
        ),
        proxy_for="City of Issaquah IMC 18.802 erosion hazard designation",
        note="1990 vintage, no planned update.",
    ),

    # ── Topography: capacity and cost ─────────────────────────────────────────
    ConstraintLayer(
        key="steep_slope",
        label="Steep slope (40%+)",
        imc="18.802 Art. II — steep slope development standards",
        publisher="City of Issaquah (LiDAR derivative)",
        layer_ref="config: layers.steep_slope",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.AREA_SHARE,
        capacity_fraction=1.00,
        cost=CostAdder(
            rate=22.00, basis=Basis.SLOPED_SF, tag=Tag.ASSUMPTION,
            group="earthwork",
            note="Retaining and export. Beats erosion within the group.",
        ),
        predev_months_add=2.0,
        note="capacity_fraction 1.00 means the overlapping area is fully "
             "removed from developable land, not that the parcel loses all "
             "capacity. Matches the IMC 18.802 steep slope threshold. The "
             "city also publishes 15-40% slopes; that layer is deliberately "
             "not registered — add it only with a cost basis, not by analogy.",
    ),
    ConstraintLayer(
        key="landslide_hazard",
        label="Landslide hazard (WADNR inventory: scarps/flanks, deposits)",
        imc="18.802 Art. II",
        publisher="WA DNR landslide inventory, republished by City of Issaquah",
        layer_ref="config: layers.landslide_hazard (sublayers 3, 4)",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.AREA_SHARE,
        capacity_fraction=1.00,
        predev_months_add=3.0,
        proxy_for="City of Issaquah IMC 18.802 landslide hazard designation",
        note="Geomorphic inventory, not a regulatory hazard designation. "
             "NEVER include the service's Study_Area sublayer — it is the "
             "extent of the mapping effort and covers everything, so joining "
             "it flags every parcel in the city. Near-coincident with "
             "steep_slope; the union rule in encumbered_fraction() is what "
             "keeps the pair from double-removing the same ground.",
    ),
    ConstraintLayer(
        key="landslide_affected_parcels",
        label="Known landslide-affected parcels (city inventory)",
        imc="",
        publisher="City of Issaquah",
        layer_ref="config: layers.landslide_affected_parcels",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.NONE,
        flag_only=True,
        note="Publisher labels this '[incomplete!]'. These are PARCEL "
             "polygons, not hazard polygons: an AREA_SHARE join against them "
             "would remove 100% of every flagged parcel's capacity, because "
             "the polygon IS the parcel. Registered flag_only so the "
             "information reaches the digest while the arithmetic cannot "
             "touch it. validate() enforces that.",
    ),

    # ── Water: capacity, cost, and time ───────────────────────────────────────
    ConstraintLayer(
        key="stream_buffer",
        label="Stream buffer (published per-reach width)",
        imc="18.802 — streams and buffers",
        publisher="City of Issaquah",
        layer_ref="config: layers.stream_buffer",
        op=Op.BUFFER,
        buffer_width_field="BufferWidthFt",
        buffer_field="IsqRating",
        buffer_by_field=None,          # [FILL] from the IsqRating coded values
        buffer_width_range_ft=(5.0, 400.0),
        setback_beyond_buffer_ft=15.0,
        capacity_mode=CapacityMode.AREA_SHARE,
        capacity_fraction=1.00,
        cost=CostAdder(
            rate=6.00, basis=Basis.ENCUMBERED_SF, tag=Tag.ASSUMPTION,
            group="critical_area_mitigation",
            note="Planting, fencing, monitoring on the disturbed portion only.",
        ),
        predev_months_add=5.0,
        note="Buffer width is PUBLISHED PER REACH in BufferWidthFt — the city's "
             "own determination, which supersedes the hand-keyed WDFW water-type "
             "table carried in v0.2. buffer_by_field is retained as a fallback "
             "keyed on IsqRating (the city's classification) and is unpopulated "
             "until those coded values are read. Two cautions that survive into "
             "every answer: the source is CENTERLINE geometry, while IMC buffers "
             "run from the ordinary high water mark, so a centerline buffer "
             "under-covers by about half the channel width — see buffer_add_ft "
             "in config. And a zero or null published width is not a zero-foot "
             "buffer; zero_width_policy decides, and 'fail' is the default. "
             "Issaquah Creek and its tributaries encumber a large share of the "
             "valley floor; this layer moves more capacity than any other.",
    ),
    ConstraintLayer(
        key="wetland_buffer",
        label="Wetland buffer (Table 18.802.220(B), high-intensity land use)",
        imc="18.802.220(B); building setback 18.802.450(F)",
        publisher="City of Issaquah",
        layer_ref="config: layers.wetland_buffer",
        op=Op.BUFFER,
        buffer_field="Iss_WetlandClass",
        buffer_field_fallback="DOE_WetlandClass",
        buffer_class_scheme="ecology_category",
        buffer_table="WETLAND_BUFFER_TABLE",
        setback_beyond_buffer_ft=15.0,
        buffer_width_range_ft=(25.0, 300.0),
        capacity_mode=CapacityMode.AREA_SHARE,
        capacity_fraction=1.00,
        cost=CostAdder(
            rate=6.00, basis=Basis.ENCUMBERED_SF, tag=Tag.ASSUMPTION,
            group="critical_area_mitigation",
        ),
        predev_months_add=6.0,
        note="Width keys on category, habitat score and adjacent land use "
             "intensity. The layer carries category only, so habitat_band is "
             "a stated config assumption and land_use_intensity is 'high' "
             "(correct for residential, not an assumption). Iss_WetlandClass "
             "is primary on COVERAGE grounds (63 valid vs 29) — not "
             "authority: none of the 11 Iss/DOE disagreements involves "
             "Category IV, so the field choice moves at most 20 ft on 6 "
             "features. The live exposure is vintage: 21 features read "
             "Iss=3, which is Category III as written and Category IV under "
             "the legacy Class 1-3 scheme, a 30-100 ft swing. Geometry is "
             "polygon and the code measures perpendicular from the "
             "delineated boundary, so no centerline allowance applies. "
             "UNMODELED: the 25 ft steep-slope buffer extension, and "
             "18.802.220(G), which directs using the next-higher intensity "
             "column where the existing buffer is degraded — both widen "
             "buffers, so capacity near wetlands is an upper bound.",
    ),
    ConstraintLayer(
        key="floodway",
        label="FEMA regulatory floodway",
        imc="18.802 — frequently flooded areas",
        publisher="City of Issaquah (FIRM effective 08/19/2020)",
        layer_ref="config: layers.floodway",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.PROHIBIT,
        note="No residential capacity on the area covered. A partially "
             "encumbered parcel keeps its unencumbered portion — "
             "capacity_factor() zeroes the parcel outright only at full "
             "coverage. The city layer is locally clipped and already in "
             "EPSG:2926; it goes stale silently if FEMA issues a new FIRM, "
             "so the live NFHL is retained as an annual cross-check.",
    ),
    ConstraintLayer(
        key="floodplain_100yr",
        label="100-year floodplain (Zone A / AE)",
        imc="18.802 and IMC Title 16",
        publisher="City of Issaquah (FIRM effective 08/19/2020)",
        layer_ref="config: layers.floodplain_100yr",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.AREA_SHARE,
        capacity_fraction=0.25,
        cost=CostAdder(
            rate=14.00, basis=Basis.FOOTPRINT_SF, tag=Tag.ASSUMPTION,
            note="Freeboard, flood-resistant construction, compensatory "
                 "storage.",
        ),
        predev_months_add=3.0,
        note="The only graded AREA_SHARE layer in the registry: building in "
             "the floodplain is permitted with standards, so a quarter of "
             "the overlap is treated as lost rather than all of it. "
             "encumbered_fraction() applies the 0.25 to whatever area the "
             "full-removal layers and the floodway have not already taken — "
             "the v0.2 bug was taking 100% here.",
    ),
    ConstraintLayer(
        key="high_groundwater",
        label="High groundwater table",
        imc="",
        publisher="(none — no published layer)",
        layer_ref="config: layers.high_groundwater (absent by decision)",
        op=Op.INTERSECT,
        cost=CostAdder(
            rate=18.00, basis=Basis.BELOW_GRADE_SF, tag=Tag.ASSUMPTION,
            applies_if="has_below_grade",
            note="Fires only on products with below-grade construction. A "
                 "townhouse on the same parcel pays nothing.",
        ),
        note="CONFIRMED UNPUBLISHED. No endpoint is configured, so S2 skips "
             "this layer and reports the omission, and D2 propagates it as "
             "an upper-bound caveat. Deriving it from Ecology well logs is a "
             "project, not a join. Kept in the registry so the gap is "
             "visible rather than forgotten. The stormwater question it "
             "partly covered is now answered by the soil infiltration join "
             "in S1 instead.",
    ),
    ConstraintLayer(
        key="shoreline_jurisdiction",
        label="Shoreline jurisdiction",
        imc="Shoreline Master Program",
        publisher="City of Issaquah (Shoreline Environment Designations)",
        layer_ref="config: layers.shoreline_jurisdiction",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.FLAT,
        capacity_fraction=0.20,
        predev_months_add=8.0,
        proxy_for="200 ft from OHWM plus associated wetlands and floodplain",
        note="Primarily a schedule constraint, and the longest one in the "
             "registry. Eight months of carry is the real cost; A2 step 4 "
             "picks it up through PREDEV_BURN_MONTHLY, which at the 41+ unit "
             "rate is roughly $680,000 before a shovel moves. The configured "
             "source is the environment designation layer, which understates "
             "true jurisdiction.",
    ),

    # ── Aquifer protection: Issaquah's signature overlay ──────────────────────
    ConstraintLayer(
        key="cara",
        label="Critical Aquifer Recharge Area",
        imc="18.802 — critical aquifer recharge areas",
        publisher="City of Issaquah (2023 edition)",
        layer_ref="config: layers.cara",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.NONE,
        cost=CostAdder(
            rate=9.00, basis=Basis.LOT_SF, tag=Tag.ASSUMPTION,
            applies_if="no_infiltration",
            group="stormwater",
            note="Detention vault where infiltration is disallowed.",
        ),
        predev_months_add=2.0,
        note="Broad coverage across the valley. Lot-based, so it penalizes "
             "large-lot low-density product hardest — a real effect, not an "
             "artifact. The city publishes a superseded edition of this "
             "layer in the same service; config pins the 2023 one, and the "
             "manifest should say both exist.",
    ),
    ConstraintLayer(
        key="wellhead_1yr",
        label="Wellhead protection, 1-year time of travel",
        imc="18.802",
        publisher="Sammamish Plateau Water, via City of Issaquah",
        layer_ref="config: layers.wellhead_1yr",
        op=Op.INTERSECT,
        capacity_mode=CapacityMode.FLAT,
        capacity_fraction=0.10,
        cost=CostAdder(
            rate=11.00, basis=Basis.LOT_SF, tag=Tag.ASSUMPTION,
            group="stormwater",
            note="Beats the CARA vault rate within the group.",
        ),
        predev_months_add=2.0,
        note="PARTIAL COVERAGE. Sammamish Plateau Water zones only; does not "
             "cover City of Issaquah utility wells. A no-hit is not evidence "
             "of no wellhead constraint, and any answer touching this layer "
             "says so.",
    ),

    # ── Parcel-attribute constraints (not polygon joins) ──────────────────────
    ConstraintLayer(
        key="frontage_unimproved",
        label="Unimproved street frontage",
        imc="IMC Title 13 / development standards",
        publisher="Derived in S1 — street centerlines vs. parcel boundary",
        layer_ref="derived: s1_parcel_master.derive_frontage",
        op=Op.INTERSECT,
        cost=CostAdder(
            rate=1200.00, basis=Basis.FRONTAGE_LF, tag=Tag.ASSUMPTION,
            applies_if="frontage_unimproved",
            note="Curb, gutter, sidewalk, planter, illumination.",
        ),
        note="Frontage LENGTH is measured; frontage IMPROVEMENT STATUS is "
             "not derivable — the city publishes impervious surface with "
             "sidewalk merged into a combined road-pavement-sidewalk class. "
             "The predicate therefore reads an explicit config default, "
             "shipped as 'improved' so the adder does not fire. That "
             "understates greenfield cost; S2 reports how many parcels the "
             "choice touches. Disproportionate on small infill lots — 60 LF "
             "on a 6,000 SF lot is $72,000 against a single-digit unit count.",
    ),
    ConstraintLayer(
        key="lot_line_shoring",
        label="Shoring for excavation at property line",
        imc="",
        publisher="Derived in S2 — product attributes, not a mapped layer",
        layer_ref="derived: s2_capacity.dimension_product",
        op=Op.INTERSECT,
        cost=CostAdder(
            rate=12.00, basis=Basis.LOT_SF, tag=Tag.ASSUMPTION,
            applies_if="lot_line_excavation",
        ),
        note="Registered so every A2 site adder has exactly one home. The "
             "geometric condition is universally true, so this never fires "
             "on its own — the predicate does all the work.",
    ),
)

LAYERS_BY_KEY = {c.key: c for c in CONSTRAINT_LAYERS}


# ──────────────────────────────────────────────────────────────────────────────
# Combination rules — where the double-counting bugs live
# ──────────────────────────────────────────────────────────────────────────────

def encumbered_fraction(parcel_geom, hits, exclude=None) -> float:
    """Share of lot area removed by AREA_SHARE constraints.

    Full-removal layers (capacity_fraction >= 1.0) are unioned, then graded
    layers contribute their fraction of whatever area the union did not
    already take.

    Two bugs this shape avoids. Multiplying per-layer factors removes the same
    ground once per layer, and steep slope, landslide and erosion are
    frequently the same polygon in Issaquah's data. Appending a graded layer's
    whole overlap to the union removes 100% of floodplain area on a layer
    registered at 25%.

    `exclude` removes already-accounted geometry — the PROHIBIT area — from
    the base so capacity_factor() does not subtract it twice.

    `hits` maps layer key -> the clipped overlap geometry for this parcel.
    """
    from shapely.ops import unary_union

    if parcel_geom is None or parcel_geom.is_empty or parcel_geom.area <= 0:
        return 0.0

    full, graded = [], []
    for key, geom in hits.items():
        layer = LAYERS_BY_KEY[key]
        if layer.flag_only or layer.capacity_mode is not CapacityMode.AREA_SHARE:
            continue
        if geom is None or geom.is_empty:
            continue
        clipped = geom.buffer(0).intersection(parcel_geom)
        if exclude is not None and not exclude.is_empty:
            clipped = clipped.difference(exclude)
        if clipped.is_empty or clipped.area <= 0:
            continue
        (full if layer.capacity_fraction >= 1.0
         else graded).append((layer.capacity_fraction, clipped))

    full_union = unary_union([g for _, g in full]) if full else None
    removed = full_union.area if full_union is not None else 0.0

    # Graded layers take their fraction of the residual. Where two graded
    # layers overlap each other the MAX fraction applies, not the sum. Only
    # floodplain_100yr is graded today; the max rule keeps this correct if a
    # second one is registered.
    if graded:
        graded.sort(key=lambda t: t[0], reverse=True)
        claimed = full_union
        for frac, geom in graded:
            residual = geom.difference(claimed) if claimed is not None else geom
            if residual.is_empty:
                continue
            removed += frac * residual.area
            claimed = residual if claimed is None else unary_union([claimed, residual])

    return min(1.0, removed / parcel_geom.area)


def capacity_factor(parcel_geom, hits) -> float:
    """Multiplier on allowed units.

    PROHIBIT zeroes only the area it covers — a parcel half in the floodway
    keeps the other half, and the registry note for `floodway` says so. It
    zeroes the parcel outright only at effectively full coverage. FLAT
    penalties compound on whatever area-based removal leaves.
    """
    from shapely.ops import unary_union

    if parcel_geom is None or parcel_geom.is_empty or parcel_geom.area <= 0:
        return 0.0

    proh = []
    for key, geom in hits.items():
        layer = LAYERS_BY_KEY[key]
        if layer.flag_only or layer.capacity_mode is not CapacityMode.PROHIBIT:
            continue
        if geom is None or geom.is_empty:
            continue
        clipped = geom.buffer(0).intersection(parcel_geom)
        if not clipped.is_empty and clipped.area > 0:
            proh.append(clipped)

    proh_union = unary_union(proh) if proh else None
    proh_share = (proh_union.area / parcel_geom.area) if proh_union is not None else 0.0
    if proh_share >= 0.995:
        return 0.0

    factor = 1.0 - proh_share - encumbered_fraction(parcel_geom, hits,
                                                    exclude=proh_union)
    for key in hits:
        layer = LAYERS_BY_KEY[key]
        if layer.flag_only:
            continue
        if layer.capacity_mode is CapacityMode.FLAT:
            factor *= (1.0 - layer.capacity_fraction)

    return max(0.0, factor)


def encumbered_fraction_naive(parcel_geom, hits) -> float:
    """Per-layer fractions summed independently, with no union.

    Exists only to feed the D2B overlap QA line, which publishes this beside
    the union figure as evidence the double-count was avoided. Never read it
    into a capacity figure.
    """
    if parcel_geom is None or parcel_geom.is_empty or parcel_geom.area <= 0:
        return 0.0
    total = 0.0
    for key, geom in hits.items():
        layer = LAYERS_BY_KEY[key]
        if layer.flag_only or layer.capacity_mode is not CapacityMode.AREA_SHARE:
            continue
        if geom is None or geom.is_empty:
            continue
        clipped = geom.buffer(0).intersection(parcel_geom)
        total += layer.capacity_fraction * clipped.area / parcel_geom.area
    return total


def cost_adders(hits: Mapping[str, object],
                parcel: Mapping,
                product: Mapping) -> dict[str, float]:
    """Dollar adders by group. Within a group the MAX applies, not the sum —
    liquefaction, peat, seismic hazard and coal mine hazard all buy one
    foundation scope, and charging four would make every Central Issaquah
    parcel structurally infeasible for the wrong reason.

    Returns {group_or_key: dollars}. The caller adds the total to A2 step 3.
    """
    quantities = {
        Basis.LOT_SF: parcel.get("lot_sf", 0.0),
        Basis.FOOTPRINT_SF: product.get("footprint_sf", 0.0),
        Basis.BELOW_GRADE_SF: product.get("below_grade_sf", 0.0),
        Basis.FRONTAGE_LF: parcel.get("frontage_lf", 0.0),
        Basis.SLOPED_SF: parcel.get("sloped_developed_sf", 0.0),
    }
    ctx = {**parcel, **product}
    out: dict[str, float] = {}

    for key, geom in hits.items():
        layer = LAYERS_BY_KEY[key]
        if layer.flag_only:
            continue
        adder = layer.cost
        if adder is None or not PREDICATES[adder.applies_if](ctx):
            continue
        if adder.basis is Basis.ENCUMBERED_SF:
            qty = 0.0 if geom is None or geom.is_empty else geom.area
        else:
            qty = float(quantities.get(adder.basis, 0.0) or 0.0)
        bucket = adder.group or key
        out[bucket] = max(out.get(bucket, 0.0), adder.rate * qty)

    return out


def predev_months_add(hits: Mapping[str, object]) -> float:
    """Schedule impact.

    Reviews run concurrently, so the MAX applies, plus a quarter of the
    remainder for coordination friction. Summing would put a creekside
    wetland parcel at two extra years, which overstates it.
    """
    months = sorted((LAYERS_BY_KEY[k].predev_months_add
                     for k in hits if not LAYERS_BY_KEY[k].flag_only),
                    reverse=True)
    months = [m for m in months if m > 0]
    return 0.0 if not months else months[0] + 0.25 * sum(months[1:])




# ──────────────────────────────────────────────────────────────────────────────
# Gate: a value belongs here only if a WRONG VALUE CHANGES A VERDICT, not merely
# a decimal. Without that rule the list grows until acknowledge_unverified=True
# becomes permanent and the gate stops meaning anything.
# ──────────────────────────────────────────────────────────────────────────────

VERIFY_BEFORE_RUN = {
    "stream_buffer":
        "Confirm BufferWidthFt reflects IMC 18.802 standards and not approved "
        "project-specific reductions under 18.802.330(K). Check whether width "
        "varies within a single IsqRating class — if it does, those are "
        "reductions and the layer is not a standards table.",
    "wetland_buffer":
        "habitat_band is an assumption and Categories I, II and III take "
        "identical widths at moderate and high — the band choice moves more "
        "capacity than any classification dispute. Also iss_only_vintage, "
        "governing 21 features where Cat III vs IV is a 30-100 ft swing.",
    "liquefaction":
        "Deep foundation rate ($40/SF footprint) — replace with a real Central "
        "Issaquah geotechnical bid.",
    "peat_soft_soils":
        "Same bid as liquefaction, plus the DNR attribute encoding the config "
        "LIKE clause guesses at.",
    "coal_mine_hazard":
        "Remediation rate ($55/SF) and 15% siting penalty — no local source. "
        "Check Issaquah's own coal_mines service before trusting the county "
        "1990 layer for a Central or Olde Town verdict.",
    "seismic_hazard":
        "Rate ($28/SF), and whether the county's 1990 SAO polygon is an "
        "acceptable proxy for the city's IMC 18.802 designation.",
    "cara":
        "Vault rate ($9/SF lot) and the soil infiltration coded values that "
        "gate it. Until those are set the adder fires on every CARA parcel.",
    "shoreline_jurisdiction":
        "Eight added predevelopment months is the largest schedule figure in "
        "the registry, roughly $680,000 of carry on a 41+ unit project.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _check_buffer_table(name: str, problems: list, warnings: list) -> None:
    """Structural checks on a multi-axis buffer table.

    A transcribed table is the one artifact where a single misplaced cell
    produces a plausible-looking wrong buffer on every feature of a class.
    """
    table = BUFFER_TABLES.get(name)
    axes = BUFFER_TABLE_AXES.get(name)
    if table is None or axes is None:
        problems.append(f"buffer_table {name!r} is not registered in BUFFER_TABLES")
        return

    intensities = axes["intensities"]
    groups, bands = axes["category_group"], axes["bands"]

    for coord, row in table.items():
        missing = [i for i in intensities if i not in row]
        if missing:
            problems.append(f"{name}{coord}: missing intensity columns {missing}")
            continue
        vals = [float(row[i]) for i in intensities]
        if vals != sorted(vals):
            problems.append(
                f"{name}{coord}: widths {vals} are not non-decreasing across "
                f"{intensities}. A higher intensity of adjacent land use cannot "
                f"require a narrower buffer — likely a transposed row.")

    # Every category must reach a group, every group/band pair must exist.
    for cat, group in groups.items():
        if group not in bands:
            problems.append(f"{name}: category {cat!r} maps to group {group!r} "
                            f"with no band mapping")
            continue
        for band, row_key in bands[group].items():
            if (group, row_key) not in table:
                problems.append(
                    f"{name}: band {band!r} for group {group!r} points at row "
                    f"({group}, {row_key}) which is absent from the table")

    # Rows the band map can never select. Unreachable rows are dead weight at
    # best and a missing selection axis at worst.
    reachable = {(g, r) for g, m in bands.items() for r in m.values()}
    for coord in sorted(set(table) - reachable, key=str):
        warnings.append(
            f"{name}{coord}: row is UNREACHABLE through the band map — no "
            f"combination of category and band selects it. Either the axis "
            f"that selects it is unmodeled, or the row is surplus.")


def _check_layer(c: ConstraintLayer, problems: list, warnings: list) -> None:
    # ── capacity semantics ────────────────────────────────────────────────
    if c.capacity_mode is CapacityMode.NONE and c.capacity_fraction:
        problems.append(f"{c.key}: capacity_fraction set but mode is NONE")
    if c.capacity_mode in (CapacityMode.AREA_SHARE, CapacityMode.FLAT) \
            and not 0 < c.capacity_fraction <= 1:
        problems.append(f"{c.key}: capacity_fraction out of range")
    if c.capacity_mode is CapacityMode.PROHIBIT and c.capacity_fraction:
        problems.append(f"{c.key}: PROHIBIT takes no capacity_fraction")

    # ── flag_only is a safety property, not a convention ──────────────────
    # A layer whose polygons are parcels must never reach the capacity or cost
    # math: an AREA_SHARE join against it removes 100% of every flagged parcel.
    if c.flag_only:
        if c.capacity_mode is not CapacityMode.NONE:
            problems.append(f"{c.key}: flag_only layer carries capacity_mode "
                            f"{c.capacity_mode.value} — it would remove capacity")
        if c.cost is not None:
            problems.append(f"{c.key}: flag_only layer carries a cost adder")
        if c.predev_months_add:
            problems.append(f"{c.key}: flag_only layer carries a schedule effect")
        if c.setback_beyond_buffer_ft:
            problems.append(f"{c.key}: flag_only layer carries a setback")

    # ── buffering ─────────────────────────────────────────────────────────
    sources = width_sources(c)
    if c.op is Op.BUFFER:
        if not sources:
            problems.append(f"{c.key}: BUFFER op with no width source")
        if len(sources) > 1:
            warnings.append(f"{c.key}: {len(sources)} width sources, "
                            f"precedence {' -> '.join(sources)}")
        lo, hi = c.buffer_width_range_ft
        if not 0 < lo < hi:
            problems.append(f"{c.key}: buffer_width_range_ft {(lo, hi)} invalid")
        if c.buffer_ft and not lo <= c.buffer_ft <= hi:
            problems.append(f"{c.key}: buffer_ft {c.buffer_ft:g} outside its "
                            f"own plausible range {lo:g}-{hi:g}")
    else:
        if sources:
            problems.append(f"{c.key}: width source declared on a non-BUFFER "
                            f"layer — it would never be applied")
        if c.setback_beyond_buffer_ft:
            problems.append(f"{c.key}: setback_beyond_buffer_ft on a "
                            f"non-BUFFER layer has no edge to sit outside of")

    # ── classification ────────────────────────────────────────────────────
    if uses_classification(c) and not c.buffer_field:
        problems.append(f"{c.key}: class-keyed width with no buffer_field")
    if c.buffer_field_fallback and not c.buffer_field:
        problems.append(f"{c.key}: buffer_field_fallback without a primary "
                        f"buffer_field — nothing to fall back from")
    if c.buffer_field_fallback == c.buffer_field and c.buffer_field:
        problems.append(f"{c.key}: fallback field is the primary field")

    if uses_classification(c):
        scheme = c.buffer_class_scheme
        if not scheme:
            problems.append(
                f"{c.key}: class-keyed width with no buffer_class_scheme. "
                f"Declare the vocabulary before mapping tokens onto it — an "
                f"undeclared scheme is how a legacy class is read as a category.")
        elif scheme not in CLASS_SCHEMES:
            problems.append(f"{c.key}: unknown buffer_class_scheme {scheme!r}")
        else:
            vocab = CLASS_SCHEMES[scheme]
            vocab = frozenset(CLASS_SCHEMES[scheme].classes)
            if c.buffer_by_field and vocab:
                stray = sorted(set(c.buffer_by_field) - vocab)
                if stray:
                    problems.append(
                        f"{c.key}: buffer_by_field keys {stray} are not in "
                        f"scheme {scheme!r}. The table and the normalizer speak "
                        f"different vocabularies and every lookup would miss.")
            if c.buffer_table:
                axes = BUFFER_TABLE_AXES.get(c.buffer_table, {})
                if axes.get("scheme") != scheme:
                    problems.append(
                        f"{c.key}: declares scheme {scheme!r} but table "
                        f"{c.buffer_table} is keyed to {axes.get('scheme')!r}")
                uncovered = sorted(vocab - set(axes.get("category_group", {})))
                if uncovered:
                    problems.append(
                        f"{c.key}: scheme classes {uncovered} have no row in "
                        f"{c.buffer_table} — those features get no width")

    # ── width plausibility against the layer's own range ──────────────────
    lo, hi = c.buffer_width_range_ft
    widths: list[float] = list(c.buffer_by_field.values()) if c.buffer_by_field else []
    if c.buffer_table and c.buffer_table in BUFFER_TABLES:
        widths += [float(v) for row in BUFFER_TABLES[c.buffer_table].values()
                   for v in row.values()]
    out = sorted({w for w in widths if not lo <= w <= hi})
    if out:
        problems.append(
            f"{c.key}: widths {out} fall outside buffer_width_range_ft "
            f"{lo:g}-{hi:g}. Either the range is wrong or a cell is.")

    # ── provenance ────────────────────────────────────────────────────────
    if c.cost and c.cost.applies_if not in PREDICATES:
        problems.append(f"{c.key}: unknown predicate {c.cost.applies_if!r}")
    if c.layer_ref.startswith("<") or "UNRESOLVED" in c.layer_ref:
        warnings.append(f"{c.key}: layer_ref unresolved — {c.layer_ref}")
    if c.proxy_for:
        warnings.append(f"{c.key}: PROXY for {c.proxy_for} — label downstream "
                        f"figures accordingly")

    # ── inert layers ──────────────────────────────────────────────────────
    inert = (c.capacity_mode is CapacityMode.NONE and c.cost is None
             and not c.predev_months_add and not c.setback_beyond_buffer_ft
             and not c.flag_only)
    if inert:
        warnings.append(f"{c.key}: layer has no capacity, cost, schedule or "
                        f"setback effect — it is fetched and ignored")


def validate(acknowledge_unverified: bool = False) -> list[str]:
    """Raise on structural errors and unverified values; return warnings.

    The unverified gate is deliberately a hard stop. Per A2 Part 2 failure
    behavior, a verdict resting on an unverified constraint is an upper bound,
    and that has to be stated rather than discovered.
    """
    problems: list[str] = []
    warnings: list[str] = []

    seen: set[str] = set()
    for c in CONSTRAINT_LAYERS:
        if c.key in seen:
            problems.append(f"{c.key}: duplicate registry key")
        seen.add(c.key)
        _check_layer(c, problems, warnings)

    for name in sorted({c.buffer_table for c in CONSTRAINT_LAYERS if c.buffer_table}):
        _check_buffer_table(name, problems, warnings)

    # ── cross-layer: the IMC building setback applies to ALL buffers ──────
    buffered = [c for c in CONSTRAINT_LAYERS
                if c.op is Op.BUFFER and not c.flag_only]
    setbacks = {c.setback_beyond_buffer_ft for c in buffered}
    if len(setbacks) > 1:
        detail = ", ".join(f"{c.key}={c.setback_beyond_buffer_ft:g}" for c in buffered)
        problems.append(
            f"Buffered layers disagree on setback_beyond_buffer_ft ({detail}). "
            f"IMC 18.802.450(F) establishes the building setback from the outer "
            f"edge of ALL buffers — a layer at 0 while another carries 15 is an "
            f"omission, not a distinction. Set them deliberately or equally.")

    # Mixed denominators inside a cost group are legal — the max dollar wins —
    # but they are rarely intended, so say so once.
    groups: dict[str, set] = {}
    for c in CONSTRAINT_LAYERS:
        if c.cost and c.cost.group:
            groups.setdefault(c.cost.group, set()).add(c.cost.basis.value)
    for g, bases in groups.items():
        if len(bases) > 1:
            warnings.append(
                f"cost group {g!r} mixes denominators ({', '.join(sorted(bases))}) "
                f"— the largest dollar amount wins, which may not be the "
                f"largest scope")

    unknown_gate = sorted(set(VERIFY_BEFORE_RUN) - set(LAYERS_BY_KEY))
    if unknown_gate:
        problems.append(f"VERIFY_BEFORE_RUN names layers absent from the "
                        f"registry: {', '.join(unknown_gate)}")

    unverified = [k for k in VERIFY_BEFORE_RUN
                  if k in LAYERS_BY_KEY and not LAYERS_BY_KEY[k].verified]
    if unverified and not acknowledge_unverified:
        problems.append(
            "Unverified values: " + ", ".join(unverified) +
            ". Clear them, or pass acknowledge_unverified=True and label every "
            "downstream figure an estimate per A2 Part 2 failure behavior.")

    if problems:
        raise ValueError("D2 constraint config invalid:\n  " + "\n  ".join(problems))
    return warnings


def registry_summary() -> str:
    """One line per layer. Print before a run to see the registry as it will
    behave, rather than as the notes describe it."""
    rows = []
    for c in CONSTRAINT_LAYERS:
        cost = ("—" if c.cost is None
                else f"${c.cost.rate:,.0f}/{c.cost.basis.value}"
                     f"{' [' + c.cost.group + ']' if c.cost.group else ''}")
        marks = "".join(["F" if c.flag_only else " ",
                         "P" if c.proxy_for else " ",
                         "V" if c.verified else " "])
        width = " + ".join(width_sources(c)) or "—"
        setback = f"+{c.setback_beyond_buffer_ft:g}" if c.setback_beyond_buffer_ft else ""
        rows.append(f"  {marks}  {c.key:<28} {c.capacity_mode.value:<11} "
                    f"{c.capacity_fraction or '':<5} {cost:<34} "
                    f"{c.predev_months_add or '':>4} {setback:<4} {width}")
    header = (f"D2 constraint registry v{REGISTRY_VERSION} "
              f"({len(CONSTRAINT_LAYERS)} layers)  "
              f"[F]lag-only  [P]roxy  [V]erified\n"
              f"      {'key':<28} {'mode':<11} {'frac':<5} {'cost':<34} "
              f"{'mo':>4} {'sb':<4} width source")
    return "\n".join([header, *rows])

if __name__ == "__main__":
    print(registry_summary())
    print()
    for w in validate(acknowledge_unverified=True):
        print(f"  warning: {w}")
