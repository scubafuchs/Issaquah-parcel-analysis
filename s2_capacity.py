"""
s2_capacity.py — apply A1 zone standards and the D2 constraint registry.

Reads  out/parcels_master.parquet  (s1_parcel_master.py)
       kb/A1_zone_standards.csv    (hand-keyed matrix, A1)
       config.yaml                 (resolved endpoints, product form factors)
Writes out/parcels_capacity.parquet  — the frame s2_digests.py expects
       out/S2_qa.json

Transit proximity is computed in S1 and arrives as a column. S2 owns
standards and constraints only.
"""

from __future__ import annotations

import argparse
import json
import time
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.ops import unary_union

from d2_constraints import (
    CONSTRAINT_LAYERS, LAYERS_BY_KEY, CapacityMode, Op,
    capacity_factor, cost_adders, encumbered_fraction,
    encumbered_fraction_naive, predev_months_add, validate,
)
from s1_parcel_master import fetch_layer, fetch_multi, CRS_WORK

SQFT_PER_ACRE = 43_560.0

A1_COLUMNS = [
    "zone", "max_du_per_acre", "max_du_per_lot", "min_lot_area_sf", "max_far",
    "max_height_ft", "max_lot_coverage", "parking_min_per_unit",
    "setback_min_ft", "mh_eligible", "densest_product", "process_type",
]

# Columns A1 v1.0 adds. Absent in an older A1, so read defensively.
A1_OPTIONAL = [
    "max_far_bonus", "max_height_bonus_ft", "max_impervious_pct",
    "setback_front_ft", "setback_side_ft", "setback_rear_ft",
    "da_governed", "imc_citation", "confidence", "fill_needed",
]

NUMERIC_A1 = [
    "max_du_per_acre", "max_du_per_lot", "min_lot_area_sf", "max_far",
    "max_far_bonus", "max_height_ft", "max_height_bonus_ft",
    "max_lot_coverage", "max_impervious_pct", "parking_min_per_unit",
    "setback_front_ft", "setback_side_ft", "setback_rear_ft", "setback_min_ft",
]


def load_a1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in A1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"A1 is missing columns: {', '.join(missing)}. densest_product and "
            "process_type are hand-keyed additions — without them no product "
            "can be dimensioned and no site adder will fire."
        )
    df["zone"] = df["zone"].astype(str).str.strip()
    for col in A1_OPTIONAL:
        if col not in df.columns:
            df[col] = pd.NA

    # Any non-numeric token in a numeric column turns the whole column to
    # object dtype, after which pd.notna("[FILL]") is True and
    # allowed_units_base() does arithmetic on a string. Coerce and report.
    for col in NUMERIC_A1:
        raw = df[col]
        df[col] = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & df[col].isna() & (raw.astype(str).str.strip() != "")
        if bad.any():
            raise ValueError(
                f"A1 column {col!r} holds non-numeric values in zones "
                f"{sorted(df.loc[bad, 'zone'])}. Leave an unkeyed standard "
                f"EMPTY and name it in fill_needed — empty means 'no such "
                f"standard' and is skipped; a token would be read as a value."
            )

    for col in ("mh_eligible", "da_governed"):
        df[col] = (df[col].astype(str).str.lower()
                   .isin(("1", "true", "yes", "y")))

    da = df[df["da_governed"]]
    if len(da):
        print(f"  A1: {len(da)} DA-governed zones "
              f"({', '.join(da['zone'])}) — standards come from executed "
              f"agreements, not Title 18. Headroom for these parcels is taken "
              f"from the pipeline, not computed.")
    gaps = df.loc[df["fill_needed"].notna(), ["zone", "fill_needed"]]
    if len(gaps):
        print(f"  A1: {len(gaps)} zones carry unkeyed standards — see "
              f"fill_needed. Capacity for those standards is skipped, not zero.")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Constraint layers
# ──────────────────────────────────────────────────────────────────────────────

# Records which width basis each buffered layer actually used, for QA and the
# D2 manifest. A buffer applied on an unstated basis is an untraceable number.
BUFFER_PROVENANCE: dict[str, dict] = {}


def _classification_widths(gdf: gpd.GeoDataFrame, layer, field: str) -> pd.Series:
    """Fallback: width from a classification lookup."""
    if not layer.buffer_by_field:
        raise ValueError(
            f"{layer.key}: no buffer_by_field table to fall back on. Populate "
            f"it from the {field!r} coded values, or set zero_width_policy."
        )
    if field not in gdf.columns:
        raise ValueError(
            f"{layer.key}: classification field {field!r} absent. Available: "
            f"{sorted(c for c in gdf.columns if c != 'geometry')}"
        )
    w = gdf[field].astype(str).str.strip().map(layer.buffer_by_field)
    if w.isna().any():
        unknown = sorted(gdf.loc[w.isna(), field].astype(str).unique())
        raise ValueError(f"{layer.key}: unmapped {field} values {unknown}")
    return w

def _published_widths(gdf, layer, entry, width_field, cls_field, prov):
    """Width from a publisher's per-feature field.

    Returns (gdf, widths). The frame comes back because zero_width_policy
    'drop' filters it, and the caller must buffer the same rows it got back.

    Does NOT apply buffer_add_ft, the IMC setback, or the plausibility range —
    apply_buffer and resolve_layers own those so every width path behaves
    identically.
    """
    n = len(gdf)
    raw = gdf[width_field]
    w = pd.to_numeric(raw, errors="coerce")

    # Three distinct failures that all end up as NaN, and they need different
    # fixes: a field of strings like "150'" is a parsing problem, a genuinely
    # absent width is a data gap, and a zero is a deliberate publisher entry
    # that may mean "piped, no buffer" or may mean "not yet determined."
    nonnumeric = int((raw.notna() & w.isna()).sum())
    null = int(raw.isna().sum())
    zero = int((w == 0).sum())
    negative = int((w < 0).sum())
    bad = w.isna() | (w <= 0)

    prov.update({"width_field": width_field, "width_nonnumeric": nonnumeric,
                 "width_null": null, "width_zero": zero,
                 "width_negative": negative})

    if nonnumeric and nonnumeric == n:
        raise ValueError(
            f"{layer.key}: no value in {width_field!r} parses as a number "
            f"(sample: {raw.dropna().astype(str).head(5).tolist()}). The field "
            f"is probably formatted text, not a numeric width. Parse it before "
            f"buffering rather than letting every feature fall through to the "
            f"zero-width policy.")

    if negative:
        raise ValueError(
            f"{layer.key}: {negative} features carry a negative {width_field}. "
            f"A negative buffer erodes the feature instead of extending it. "
            f"Fix the source or exclude those rows explicitly.")

    policy = str(entry.get("zero_width_policy", "fail")).lower()
    if policy not in ("fail", "fallback", "drop"):
        raise ValueError(f"{layer.key}: zero_width_policy {policy!r} is not "
                         f"one of fail / fallback / drop")

    if bad.any():
        if policy == "fail":
            cols = [c for c in (cls_field, "StreamID", "NameLabel", "OBJECTID",
                                "DataSource", "Datasource")
                    if c in gdf.columns]
            sample = (gdf.loc[bad, cols].head(8).to_string(index=False)
                      if cols else f"{int(bad.sum())} rows")
            raise ValueError(
                f"{layer.key}: {int(bad.sum())} of {n} features have a zero, "
                f"null or unparseable {width_field} "
                f"(null {null}, zero {zero}, unparseable {nonnumeric}). "
                f"Buffering those at zero removes the constraint on those "
                f"reaches, which reads identically to no constraint. Set "
                f"zero_width_policy to 'fallback' (classification table) or "
                f"'drop' (exclude and state the omission) once you know "
                f"whether they are piped segments or missing data.\n{sample}")

        if policy == "drop":
            gdf = gdf.loc[~bad].copy()
            w = w.loc[~bad]
            prov["zero_width_handling"] = (
                f"dropped {int(bad.sum())} features — omission stated in D2")
            if gdf.empty:
                raise ValueError(
                    f"{layer.key}: zero_width_policy 'drop' removed every "
                    f"feature. The layer would silently contribute nothing.")

        elif policy == "fallback":
            sub, _ = _classification_widths(gdf.loc[bad], layer, entry)
            w = w.copy()
            w.loc[bad] = sub
            prov["zero_width_handling"] = (
                f"classification fallback on {cls_field!r} for "
                f"{int(bad.sum())} features")

    # ── Does this field publish standards, or approved reductions? ──────────
    # IMC 18.802.330(K) allows a buffer to be reduced where the on-site buffer
    # provides insignificant function. A reduction is project-specific: it is
    # correct for parcels already through review and wrong for everything else.
    # Width varying WITHIN a single classification is the signature.
    if cls_field and cls_field in gdf.columns:
        by_class = {
            str(k): sorted(float(v) for v in grp.dropna().unique())
            for k, grp in w.groupby(gdf[cls_field].astype(str).str.strip())
        }
        prov["width_by_classification"] = by_class
        varies = {k: v for k, v in by_class.items() if len(v) > 1}
        prov["classes_with_varying_width"] = sorted(varies)

        if varies:
            detail = "; ".join(f"{k}: {v}" for k, v in sorted(varies.items()))
            msg = (f"{layer.key}: {len(varies)} classification(s) carry more "
                   f"than one width ({detail}). Published widths may encode "
                   f"approved project-specific reductions rather than code "
                   f"standards — correct for parcels already through review, "
                   f"wrong for every other parcel in that class. Record the "
                   f"basis in the manifest.")
            vpol = str(entry.get("width_variance_policy", "warn")).lower()
            if vpol == "fail":
                raise ValueError(msg)
            if vpol != "ignore":
                print(f"      WARNING: {msg}")

    # Flag column so clip_hits can propagate per-parcel provenance, matching
    # what the buffer_table path sets. Without it the two paths produce frames
    # with different columns and clip_hits behaves differently per layer.
    flag = pd.Series("", index=gdf.index)
    if cls_field in gdf.columns and prov.get("classes_with_varying_width"):
        flag = flag.mask(
            gdf[cls_field].astype(str).str.strip()
              .isin(prov["classes_with_varying_width"]),
            "width_may_be_reduced")
    if policy == "fallback":
        flag = flag.mask(bad.reindex(gdf.index, fill_value=False),
                         "width_from_classification")
    gdf["buffer_flag"] = flag.values

    return gdf, w

def apply_buffer(gdf, layer, entry=None):
    """Buffer features to their regulatory extent.

    Width basis, in descending preference — matching width_sources():
      1. buffer_width_field  — publisher's own per-feature width
      2. buffer_table        — multi-axis lookup via a registered resolver
      3. buffer_by_field     — flat classification lookup
      4. buffer_ft           — single fixed width

    The IMC 18.802.450(F) building setback is NOT applied here; resolve_layers
    applies it uniformly to all four critical-area-edge layers, including the
    two that are not Op.BUFFER.
    """
    entry = entry or {}
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    n = len(gdf)
    prov = {"layer": layer.key, "features": n}

    cls_field = entry.get("buffer_field") or layer.buffer_field
    alt_field = entry.get("buffer_field_fallback") or layer.buffer_field_fallback
    width_field = entry.get("buffer_width_field") or layer.buffer_width_field
    add_ft = float(entry.get("buffer_add_ft", 0.0) or 0.0)

    if width_field and width_field in gdf.columns:
        prov["basis"] = f"published per-feature width ({width_field})"
        gdf, w = _published_widths(gdf, layer, entry, width_field, cls_field, prov)

    elif layer.buffer_table:
        resolve_cls, widths_from_cls = BUFFER_TABLE_RESOLVERS[layer.buffer_table]
        for f in (cls_field, alt_field):
            if f and f not in gdf.columns:
                raise ValueError(
                    f"{layer.key}: classification field {f!r} absent. "
                    f"Available: "
                    f"{sorted(c for c in gdf.columns if c != 'geometry')}")
        cls, flag, tprov = resolve_cls(gdf, layer, entry)
        w = widths_from_cls(gdf, cls, entry, tprov)
        # Per-feature provenance flag rides onto the frame so clip_hits can
        # propagate the worst flag per parcel into D2B.
        gdf["buffer_flag"] = flag.values
        prov.update(tprov)
        prov["basis"] = f"table {layer.buffer_table} keyed on {cls_field}"

    elif layer.buffer_by_field:
        prov["basis"] = f"classification lookup ({cls_field})"
        w, cprov = _classification_widths(gdf, layer, entry)
        prov.update(cprov)

    elif layer.buffer_ft:
        prov["basis"] = f"fixed width ({layer.buffer_ft:g} ft)"
        w = pd.Series(float(layer.buffer_ft), index=gdf.index)

    else:
        declared = width_sources(layer) or ["none"]
        raise ValueError(
            f"{layer.key}: no usable width source at runtime. Registry "
            f"declares: {', '.join(declared)}. Either the declared field is "
            f"absent from the fetched layer, or the table has no resolver.")

    # Range check on every path — a resolver bug can produce NaN or a width
    # the static table check in validate() never sees.
    lo, hi = layer.buffer_width_range_ft
    bad = w.isna() | (w < lo) | (w > hi)
    if bad.any():
        raise ValueError(
            f"{layer.key}: {int(bad.sum())} of {len(w)} resolved widths are "
            f"null or outside {lo:g}-{hi:g} ft (min {w.min()}, max {w.max()}). "
            f"Basis: {prov['basis']}.")

    prov["width_ft_min"] = float(w.min())
    prov["width_ft_max"] = float(w.max())
    prov["ohwm_add_ft"] = add_ft
    gdf["geometry"] = gdf.geometry.buffer(w.values + add_ft)

    BUFFER_PROVENANCE[layer.key] = prov
    print(f"      buffer: {prov['basis']}, "
          f"widths {prov['width_ft_min']:g}-{prov['width_ft_max']:g} ft"
          + (f", +{add_ft:g} ft OHWM allowance" if add_ft else ""))
    return gdf


from d2_constraints import (
    WETLAND_BUFFER_TABLE, WETLAND_CATEGORY_GROUP,
    WETLAND_HABITAT_BANDS, WETLAND_LEGACY_SHIFT,
)


def wetland_width(category: str, habitat_band: str, intensity: str) -> float:
    """One width from Table 18.802.220(B). Raises on uncovered combinations."""
    group = WETLAND_CATEGORY_GROUP[category]
    row = WETLAND_HABITAT_BANDS[group][habitat_band]
    try:
        return float(WETLAND_BUFFER_TABLE[(group, row)][intensity])
    except KeyError:
        raise ValueError(
            f"Table 18.802.220(B) has no width for Category {category}, "
            f"habitat band {habitat_band!r}, intensity {intensity!r}. The "
            f"published table does not cover Category III below habitat 3, "
            f"nor Category I/II below habitat 6 without a water quality "
            f"score of 8-9. Do not interpolate — flag for delineation review."
        )

# Datasource -> authority tier. ORDER IS PRECEDENCE: first match wins, most
# specific first. "Port Blakely, IH Critical Areas .dwg" matches both
# "port blakely" (da_area) and "critical areas" (delineation); da_area must
# win, or 71 master-plan features lose their da_governed flag and land in
# the review queue.
_WETLAND_TIER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("da_area",     ("port blakely", "talus", "issaquah highlands")),
    ("legacy",      ("coot company", "2004 wetlands")),
    ("advisory",    ("nwi",)),
    ("delineation", ("wetland report", "rating summary", "critical areas",
                     "fieldwork", "survey", "plat", "site plan", "plan set",
                     "ngpe", "wetland.dx", "dxf", "gateway", "watershed company",
                     "memory care", "pln", "sdp", "pp1", "pp2", "lla")),
)


def wetland_source_tier(datasource: pd.Series, entry: dict) -> pd.Series:
    """Assign an authority tier per feature, first match wins.

    A project-specific delineation is a regulatory determination. NWI is
    advisory screening. A 2004 inventory predates the 2014 rating system.
    DA-area features never read A1 standards at all, so their classification
    is irrelevant to capacity and they must not enter the review queue.
    """
    patterns = entry.get("source_tiers") or _WETLAND_TIER_PATTERNS
    if isinstance(patterns, dict):          # config supplies a mapping
        patterns = tuple(patterns.items())  # relies on YAML order — keep it explicit
    low = datasource.fillna("").str.lower()
    tier = pd.Series("unclassified", index=datasource.index)
    assigned = pd.Series(False, index=datasource.index)
    for name, pats in patterns:
        hit = pd.Series(False, index=datasource.index)
        for p in pats:
            hit |= low.str.contains(p, regex=False)
        tier = tier.mask(hit & ~assigned, name)
        assigned |= hit
    return tier

_CLASS_TOKENS = {
    "I": "I", "1": "I", "CATEGORYI": "I", "CATI": "I",
    "II": "II", "2": "II", "CATEGORYII": "II", "CATII": "II",
    "III": "III", "3": "III", "CATEGORYIII": "III", "CATIII": "III",
    "IV": "IV", "4": "IV", "CATEGORYIV": "IV", "CATIV": "IV",
}


from d2_constraints import CLASS_SCHEMES, MISSING_CLASS_TOKENS

def normalize_class(s: pd.Series, layer, entry: dict) -> pd.Series:
    """Map raw classification tokens onto the scheme's vocabulary.

    Returns NA for every missing state. Raises on a token that is neither
    missing nor mappable — an unrecognized value is a data surprise, and
    letting it fall through to default_class assigns a buffer by accident.
    """
    scheme = CLASS_SCHEMES.get(layer.buffer_class_scheme)
    if scheme is None:
        raise ValueError(
            f"{layer.key}: buffer_class_scheme "
            f"{layer.buffer_class_scheme!r} is not registered in "
            f"CLASS_SCHEMES. Known: {sorted(CLASS_SCHEMES)}")
    if not scheme.aliases:
        raise ValueError(
            f"{layer.key}: scheme {scheme.name!r} has no alias map — its "
            f"coded values have not been read yet. Probe the layer and "
            f"populate CLASS_SCHEMES before buffering from it.")

    key = (s.astype(str).str.upper()
             .str.replace(r"[^A-Z0-9]", "", regex=True))
    key = key.where(~key.isin(MISSING_CLASS_TOKENS))

    if scheme.numerals_ambiguous and key.str.fullmatch(r"[0-9]+").fillna(False).any():
        declared = str(entry.get("numeral_meaning", "")).lower()
        if declared not in ("category", "legacy"):
            raise ValueError(
                f"{layer.key}: field carries bare numerals. "
                f"{scheme.ambiguity_note}")

    out = key.map(scheme.aliases)
    unknown = sorted(key[key.notna() & out.isna()].unique())
    if unknown:
        raise ValueError(
            f"{layer.key}: tokens {unknown} are neither missing nor in scheme "
            f"{scheme.name!r} (valid: {list(scheme.classes)}). Add them to the "
            f"alias map if they are real, or to MISSING_CLASS_TOKENS if they "
            f"mean 'unrated'. Do not let them fall through to default_class.")
    return out


def resolve_wetland_class(gdf, layer, entry):
    """Resolve category per feature and report what the choices cost.

    Iss_WetlandClass is primary on coverage, not authority. Every observed
    Iss/DOE disagreement is Category I-vs-II (identical buffer) or
    II-vs-III (20 ft, low habitat band only); none involves Category IV,
    the one classification the table treats differently.
    """
    n = len(gdf)
    ds = gdf.get("Datasource", pd.Series("", index=gdf.index)).fillna("").str.lower()
    tier = pd.Series("unclassified", index=gdf.index)
    for name, pats in _WETLAND_TIER_PATTERNS:
        for p in pats:
            tier = wetland_source_tier(
                gdf.get("Datasource", pd.Series("", index=gdf.index)), entry)

    iss = normalize_class(gdf[layer.buffer_field], layer, entry)
    doe = normalize_class(gdf[layer.buffer_field_fallback], layer, entry)

    cls = pd.Series(pd.NA, index=gdf.index, dtype="object")
    flag = pd.Series("", index=gdf.index)

    ov_path = entry.get("class_override_file")
    if ov_path and Path(ov_path).exists() and "OBJECTID" in gdf.columns:
        ov = pd.read_csv(ov_path).dropna(subset=["reviewed_class"])
        m = gdf["OBJECTID"].map(dict(zip(ov["OBJECTID"], ov["reviewed_class"])))
        cls, tier = cls.mask(m.notna(), m), tier.mask(m.notna(), "override")

    usable = tier.isin(("delineation", "advisory", "inventory"))
    hit = cls.isna() & usable & iss.notna()
    cls = cls.mask(hit, iss)
    iss_only = hit & doe.isna()
    hit = cls.isna() & usable & doe.notna()
    cls = cls.mask(hit, doe)

    # Vintage flag only where the alternate reading changes a buffer row.
    # Iss=1 shifts I->II, which is the same row: not flagged.
    material = iss_only & cls.isin(("II", "III"))
    flag = flag.mask(material, "vintage_unknown")
    flag = flag.mask(tier == "advisory", "advisory_source")
    flag = flag.mask(tier == "legacy", "pre_2014_rating")
    flag = flag.mask(tier == "da_area", "da_governed")

    policy = str(entry.get("iss_only_vintage", "")).lower()
    if material.any() and policy not in ("category", "legacy"):
        raise ValueError(
            f"{layer.key}: {int(material.sum())} features carry a city-field "
            f"class with no cross-check, and the legacy Class 1-3 reading "
            f"shifts each one category higher. For Iss=3 that is Category "
            f"III vs IV — a 30 ft difference at low habitat, 100 ft at "
            f"moderate. Set iss_only_vintage to 'category' (as written; "
            f"wider buffer, no inference) or 'legacy' (shift up one)."
        )
    if policy == "legacy":
        cls = cls.mask(material, cls.map(WETLAND_LEGACY_SHIFT))

    default = entry.get("default_class")
    unresolved = cls.isna()
    if unresolved.any():
        if not default:
            raise ValueError(
                f"{layer.key}: {int(unresolved.sum())} features have no "
                f"usable class and no default_class is set. An unbuffered "
                f"wetland reads identically to no wetland.")
        cls = cls.fillna(default)
        flag = flag.mask(unresolved & (flag == ""), "needs_review")

    prov = {
        "features": n,
        "by_tier": tier.value_counts().to_dict(),
        "by_flag": flag[flag != ""].value_counts().to_dict(),
        "class_counts": cls.value_counts().to_dict(),
        "from_iss": int((usable & iss.notna()).sum()),
        "from_doe_only": int((usable & iss.isna() & doe.notna()).sum()),
        "fields_disagree": int((iss.notna() & doe.notna() & (iss != doe)).sum()),
        "disagreements_involving_cat_iv": int(
            (iss.notna() & doe.notna() & (iss != doe)
             & ((iss == "IV") | (doe == "IV"))).sum()),
        "iss_only_vintage_material": int(material.sum()),
        "iss_only_vintage_policy": policy,
        "default_class_applied": int(unresolved.sum()),
    }
    return cls, flag, prov


def wetland_buffer_widths(gdf, cls, entry, prov):
    """Apply the table, and price both the habitat and vintage assumptions."""
    band = str(entry.get("habitat_band", "")).lower()
    intensity = str(entry.get("land_use_intensity", "high")).lower()
    if band not in ("low", "moderate", "high"):
        raise ValueError(
            "wetland_buffer: habitat_band must be low / moderate / high. No "
            "habitat score is published per feature, so the band is a stated "
            "assumption. Note that Categories I, II and III are IDENTICAL at "
            "moderate and high bands — the band choice matters more than the "
            "classification does."
        )
    w = cls.map(lambda c: wetland_width(c, band, intensity))

    alt = cls.map(WETLAND_LEGACY_SHIFT)
    prov["width_ft_mean"] = round(float(w.mean()), 1)
    prov["width_ft_distinct"] = sorted(float(v) for v in w.unique())
    prov["habitat_band"] = band
    prov["land_use_intensity"] = intensity
    prov["mean_width_under_alt_vintage"] = round(float(
        alt.map(lambda c: wetland_width(c, band, intensity)).mean()), 1)
    return w

# Multi-axis buffer tables need a pair of functions: one that resolves a
# category per feature, one that turns categories into widths. Registered by
# table name so apply_buffer stays general rather than hardcoding wetlands.
BUFFER_TABLE_RESOLVERS = {
    "WETLAND_BUFFER_TABLE": (resolve_wetland_class, wetland_buffer_widths),
}


def check_resolvers() -> None:
    """Every registry layer declaring a buffer_table must have a runtime
    resolver. This is the check that would have caught the present error:
    validate() lives in d2_constraints and cannot see this module."""
    missing = [c.key for c in CONSTRAINT_LAYERS
               if c.buffer_table and c.buffer_table not in BUFFER_TABLE_RESOLVERS]
    if missing:
        raise ValueError(
            f"Layers declare a buffer_table with no resolver in "
            f"BUFFER_TABLE_RESOLVERS: {missing}. validate() counts the table "
            f"as a width source, so this fails at buffering rather than at "
            f"config load.")

def resolve_layers(cfg: dict) -> tuple[dict[str, gpd.GeoDataFrame], list[str]]:
    """Fetch each registry layer with an endpoint in config.

    Keys absent from config, disabled, or marked flag_only are skipped and
    reported. s2_digests prints them as absent so the omission reaches the
    knowledge base rather than disappearing.
    """
    spec = cfg.get("layers", {})
    resolved, skipped = {}, []
    for layer in CONSTRAINT_LAYERS:
        entry = spec.get(layer.key)
        if not entry or not entry.get("url"):
            skipped.append(f"{layer.key} (no endpoint)")
            continue
        if entry.get("enabled") is False:
            skipped.append(f"{layer.key} (disabled)")
            continue
        if entry.get("flag_only"):
            skipped.append(f"{layer.key} (flag only, not joined)")
            continue
        try:
            gdf = fetch_multi(entry, cfg) if "layers" in entry else fetch_layer(entry, cfg)
        except Exception as exc:                            # noqa: BLE001
            skipped.append(f"{layer.key} ({type(exc).__name__}: {exc})")
            continue
        if layer.op is Op.BUFFER:
            gdf = apply_buffer(gdf, layer, entry)
        resolved[layer.key] = gdf[["geometry"]]
        print(f"    {layer.key:<28} {len(gdf):>7,} features")
        time.sleep(0.1)
    for s in skipped:
        print(f"    skipped: {s}")
    return resolved, skipped


def clip_hits(parcels: gpd.GeoDataFrame,
              layers: dict[str, gpd.GeoDataFrame]) -> dict[str, dict]:
    """hits[PIN][layer_key] = the portion of that layer inside that parcel.

    One sjoin per layer, then dissolve per PIN. Looping parcel-by-parcel over
    every layer is hours instead of minutes on 15k parcels.
    """
    hits: dict[str, dict] = {pin: {} for pin in parcels["PIN"]}
    idx = parcels.set_index("PIN").geometry
    for key, gdf in layers.items():
        pairs = gpd.sjoin(parcels[["PIN", "geometry"]], gdf,
                          how="inner", predicate="intersects")
        if pairs.empty:
            continue
        joined = gdf.loc[pairs["index_right"]].assign(PIN=pairs["PIN"].values)
        for pin, grp in joined.groupby("PIN"):
            clipped = unary_union(grp.geometry.values).intersection(idx.loc[pin])
            if not clipped.is_empty and clipped.area > 0:
                hits[pin][key] = clipped
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Allowed units and product dimensioning
# ──────────────────────────────────────────────────────────────────────────────

def allowed_units_base(row: pd.Series, near_transit: bool) -> float:
    """Binding constraint across density, lot area, per-lot cap, and FAR,
    then floored at the middle-housing minimum. The binding standard is
    almost never the nominal density figure, which is why all four are tested.
    """
    caps = []
    if pd.notna(row.get("max_du_per_acre")) and pd.notna(row.get("lot_sf")):
        caps.append(row["lot_sf"] / SQFT_PER_ACRE * row["max_du_per_acre"])
    if pd.notna(row.get("min_lot_area_sf")) and (row.get("min_lot_area_sf") or 0) > 0:
        caps.append(row["lot_sf"] / row["min_lot_area_sf"])
    if pd.notna(row.get("max_du_per_lot")):
        caps.append(row["max_du_per_lot"])
    if pd.notna(row.get("max_far")) and pd.notna(row.get("_avg_unit_gsf")):
        caps.append(row["lot_sf"] * row["max_far"] / row["_avg_unit_gsf"])
    base = min(caps) if caps else 0.0

    # Tier B middle-housing floor: 2 per lot, 4 within a quarter mile of a
    # major transit stop. A floor, never a reduction. near_transit arrives
    # from S1 and rests on a working definition — see the method notes.
    floor = (2.0 + 2.0 * near_transit) if row.get("mh_eligible") else 0.0
    return max(base, floor)


def dimension_product(row: pd.Series, forms: dict, cfg: dict) -> dict:
    """Physical attributes cost_adders() reads.

    Issaquah regulates impervious surface, not lot coverage, and impervious
    includes driveways, parking and walkways. Letting a building occupy the
    whole impervious allowance overstates the footprint — which is the
    denominator on the deep-foundation adder, the most consequential number
    in the model.
    """
    form = forms.get(row.get("densest_product"), forms["_default"])
    cov = form["site_coverage"]

    if pd.notna(row.get("max_lot_coverage")):          # not used in Issaquah
        cov = min(cov, float(row["max_lot_coverage"]))
    elif pd.notna(row.get("max_impervious_pct")):
        share = float(cfg.get("building_share_of_impervious", 0.70))   # [A]
        cov = min(cov, float(row["max_impervious_pct"]) * share)

    footprint = float(row.get("lot_sf") or 0) * cov
    levels = int(form.get("below_grade_levels", 0))
    return {
        "tested_product": row.get("densest_product") or "unknown",
        "footprint_sf": footprint,
        "below_grade_levels": levels,
        "below_grade_sf": footprint * levels,
        "setback_min_ft": (row["setback_min_ft"]
                           if pd.notna(row.get("setback_min_ft")) else 99),
        "_avg_unit_gsf": form["avg_unit_gsf"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main pass
# ──────────────────────────────────────────────────────────────────────────────

def compute(parcels: gpd.GeoDataFrame, a1: pd.DataFrame,
            cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    forms = cfg["product_forms"]

    print("  resolving constraint layers")
    layers, skipped = resolve_layers(cfg)

    print("  clipping constraint overlaps")
    hits = clip_hits(parcels, layers)

    df = parcels.merge(a1, left_on="ZONE", right_on="zone", how="left")

    rows = []
    for _, r in df.iterrows():
        pin = r["PIN"]
        h = hits.get(pin, {})
        geom = r.geometry
        product = dimension_product(r, forms, cfg)

        allowed = allowed_units_base(pd.concat([r, pd.Series(product)]),
                                     bool(r.get("near_transit", False)))
        if math.isnan(allowed): allowed = 0
        parcel_ctx = {
            "lot_sf": float(r["lot_sf"] or 0.0),
            "frontage_lf": float(r["frontage_lf"]) if pd.notna(r["frontage_lf"]) else 0.0,
            "frontage_improved": (bool(r["frontage_improved"])
                                  if pd.notna(r["frontage_improved"]) else True),
            "infiltration_feasible": bool(r["infiltration_feasible"]),
            "sloped_developed_sf": product["footprint_sf"],
        }

        enc = encumbered_fraction(geom, h)
        cap = capacity_factor(geom, h)
        if math.isnan(cap): cap = 0
        adders = cost_adders(h, parcel_ctx, product)


        # ── Capacity source ───────────────────────────────────────────────
        # Where capacity comes from decides whether it can be computed at all.
        # Order is most-specific first: a common tract inside a development
        # agreement is still a tract — the DA's units sit on the complex
        # parcels, not on the shared driveway.
        role = r.get("parcel_role")
        da_gov = bool(r.get("da_governed", False))
        prod = str(r.get("densest_product") or "").strip()
        keyed = any(pd.notna(r.get(c)) for c in
                    ("max_du_per_acre", "max_du_per_lot",
                     "min_lot_area_sf", "max_far"))

        if role == "tract":
            capacity_source = "none_tract"
        elif da_gov:
            capacity_source = "development_agreement"
        elif not prod:
            capacity_source = "none_non_residential"
        elif not keyed:
            capacity_source = "unkeyed"
        else:
            capacity_source = "code"

        computable = capacity_source == "code"

        # ── Netting ───────────────────────────────────────────────────────
        # Only HOUSING units are netted against allowed housing units. A5's
        # rule: G-2044-BEDS is counted in beds and is never added to or
        # subtracted from a unit count. A 120-bed nursing home netted as 120
        # units would zero out the headroom on a parcel that holds no
        # dwellings at all.
        units_for_capacity = (int(r["existing_units"])
                              if r.get("unit_class") == "housing_unit" else 0)

        rec = {
            "PIN": pin,
            "ZONE": r["ZONE"],
            "SUBAREA": r["SUBAREA"],
            "lot_sf": r["lot_sf"],
            "existing_units": int(r["existing_units"]),
            "yr_built": r["yr_built"],
            "ApprLandVal": r["ApprLandVal"],
            "ApprImpsVal": r["ApprImpsVal"],
            "i_l_ratio": r["i_l_ratio"],

            "allowed_units_base": allowed if computable else pd.NA,
            "capacity_factor": cap,
            "encumbered_fraction": enc,
            "encumbered_fraction_naive": encumbered_fraction_naive(geom, h),
            "net_capacity": (round(allowed * cap) - units_for_capacity
                             if computable else pd.NA),
            "capacity_source": capacity_source,

            "cost_adder_total": float(sum(adders.values())),
            "predev_months_add": predev_months_add(h),
            "tested_product": product["tested_product"],
            "footprint_sf": product["footprint_sf"],

            # Provenance carried from S1. These answer "why is this parcel's
            # number what it is" inside the digest, rather than requiring a
            # join back to the parcel master.
            "parcel_role": role,
            "unit_class": r.get("unit_class"),
            "units_basis": r.get("units_basis"),
            "units_netted": units_for_capacity,
            "condo_value_rollup": bool(r.get("condo_value_rollup", False)),

            "near_transit": bool(r.get("near_transit", False)),
            "near_transit_centroid_basis": bool(
                r.get("near_transit_centroid_basis", False)),
            "in_development_agreement": r.get("in_development_agreement"),
            "zone_split": bool(r.get("zone_split", False)),
            "in_paa": bool(r.get("in_paa", False)),
        }

        rec.update({f"adder_{k}": v for k, v in adders.items()})
        for layer in CONSTRAINT_LAYERS:
            if layer.key in layers:
                rec[f"c_{layer.key}"] = layer.key in h
        rows.append(rec)

    out = pd.DataFrame(rows)

    # pd.NA mixed with ints yields object dtype, and every downstream
    # comparison then behaves differently than it reads. Coerce once.
    for c in ("allowed_units_base", "net_capacity"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float64")

    # Decile 1 = lowest improvement-to-land ratio, matching
    # HAZARD_DECILE_WEIGHTS in A2 Part 3. Tracts carry NA i_l_ratio from S1
    # and drop out here by construction — a shared driveway with zero
    # improvement value would otherwise take decile 1 and the highest
    # redevelopment hazard weight in the model.
    valid = out["i_l_ratio"].notna() & out["capacity_source"].ne("none_tract")
    out["redev_decile"] = pd.NA
    if valid.sum() >= 10:
        out.loc[valid, "redev_decile"] = (
            pd.qcut(out.loc[valid, "i_l_ratio"].rank(method="first", pct=True),
                    10, labels=False) + 1)
    return out, skipped


def qa(out: pd.DataFrame, skipped: list[str], cfg: dict) -> dict:
    joined = [l.key for l in CONSTRAINT_LAYERS if f"c_{l.key}" in out.columns]
    mh_delta = int((out["near_transit"] != out["near_transit_centroid_basis"]).sum())
    da = out["in_development_agreement"]
    return {
        "parcels": int(len(out)),
        "zone_resolved": round(float(out["ZONE"].notna().mean()), 4),
        "layers_joined": len(joined),
        "layers_in_registry": len(CONSTRAINT_LAYERS),
        "layers_skipped": skipped,
        "layers_zero_hits": [k for k in joined if not out[f"c_{k}"].any()],
        "allowed_units_total": int(out["allowed_units_base"].sum()),
        "net_capacity_total": int(out["net_capacity"].clip(lower=0).sum()),
        "parcels_any_constraint": (int(out[[f"c_{k}" for k in joined]].any(axis=1).sum())
                                   if joined else 0),
        "union_vs_naive_units": round(float(
            (out["allowed_units_base"] * out["encumbered_fraction_naive"].clip(upper=1)).sum()
            - (out["allowed_units_base"] * out["encumbered_fraction"]).sum()), 1),
        "no_product_assigned": int((out["tested_product"] == "unknown").sum()),
        # transit sensitivity: how much capacity rests on the unsettled definition
        "near_transit_parcels": int(out["near_transit"].sum()),
        "near_transit_basis_disagreement_parcels": mh_delta,
        "mh_units_from_transit_floor": int(
            (out.loc[out["near_transit"], "allowed_units_base"].clip(upper=4) - 2)
            .clip(lower=0).sum()),
        # vested
        "parcels_in_development_agreement": (None if da.isna().all()
                                             else int(da.fillna(False).sum())),
        # frontage default exposure
        "frontage_default_used": bool(cfg.get("frontage_improved_default", True)),
        "parcels_with_frontage_measured": int(out["cost_adder_total"].notna().sum()),
        "buffer_provenance": BUFFER_PROVENANCE,
        "by_capacity_source": out["capacity_source"].value_counts().to_dict(),
        "by_parcel_role": out["parcel_role"].value_counts(dropna=False).to_dict(),
        "by_units_basis": out["units_basis"].value_counts(dropna=False).to_dict(),
        "units_housing": int(out.loc[out["unit_class"].eq("housing_unit"),
                                     "existing_units"].sum()),
        "units_group_quarters": int(out.loc[out["unit_class"].ne("housing_unit"),
                                            "existing_units"].sum()),
        "units_not_netted": int(
            (out["existing_units"] - out["units_netted"]).clip(lower=0).sum()),
        "parcels_capacity_not_computable": int(
            out["capacity_source"].ne("code").sum()),
    }

CROSSWALK_BASES = {
    "identical", "normalization", "granularity", "overlay_suffix",
    "non_residential", "excluded", "unzoned", "unmapped",
}


def load_crosswalk(path: Path, a1: pd.DataFrame,
                   observed: pd.Series) -> pd.DataFrame:
    """Map GIS zone strings to A1 rows.

    The GIS layer is the join key because it is what parcels actually carry;
    A1 is the standards authority. Neither is renamed to match the other —
    the mapping is recorded here, where it can be reviewed, rather than
    performed by string normalization nobody reads.
    """
    cw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    cw["gis_zone"] = cw["gis_zone"].astype(str).str.strip()
    cw["a1_zone"] = cw["a1_zone"].astype(str).str.strip().replace("nan", pd.NA)

    bad = sorted(set(cw["basis"].dropna()) - CROSSWALK_BASES)
    if bad:
        raise ValueError(f"Crosswalk uses unknown basis values: {bad}")

    dupes = cw["gis_zone"][cw["gis_zone"].duplicated()].unique()
    if len(dupes):
        raise ValueError(
            f"Crosswalk maps {sorted(dupes)} to more than one A1 row. "
            "Standards would be ambiguous and S2 would pick one silently."
        )

    joinable = cw[cw["basis"].isin(
        {"identical", "normalization", "granularity",
         "overlay_suffix", "non_residential"})]
    orphan = sorted(set(joinable["a1_zone"].dropna()) - set(a1["zone"]))
    if orphan:
        raise ValueError(
            f"Crosswalk points at A1 rows that do not exist: {orphan}."
        )

    seen = {z for z in observed.dropna().astype(str).str.strip().unique()}
    unmapped = sorted(seen - set(cw["gis_zone"]))
    if unmapped:
        raise ValueError(
            f"Zone values present on parcels but absent from the crosswalk: "
            f"{unmapped}. Each needs a row and a basis. An unmapped value "
            f"produces zero allowed units, which reads as 'no headroom' "
            f"rather than 'no standard keyed'."
        )

    gran = joinable[joinable["basis"] == "granularity"]
    if len(gran):
        pairs = ", ".join(f"{r.gis_zone}->{r.a1_zone}" for r in gran.itertuples())
        print(f"  crosswalk: {len(gran)} GIS values are finer than A1 "
              f"({pairs}). A1 is probably under-specified — splitting those "
              f"rows is the fix, not collapsing the GIS value.")

    unused = sorted(set(a1["zone"]) - set(cw["a1_zone"].dropna()))
    if unused:
        print(f"  crosswalk: {len(unused)} A1 zones are not mapped on any "
              f"parcel ({', '.join(unused)}). Legal — the zone exists in code "
              f"but is unmapped in the city.")
    return cw

def main() -> None:
    p = argparse.ArgumentParser(description="Apply A1 standards and the D2 registry.")
    p.add_argument("--input", default="out/parcels_master.parquet")
    p.add_argument("--a1", default="kb/A1_zone_standards.csv")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--outdir", default="out")
    p.add_argument("--crosswalk", default="kb/A1_zone_crosswalk.csv", action="store_true")
    p.add_argument("--acknowledge-unverified", action="store_true")
    a = p.parse_args()

    for w in validate(acknowledge_unverified=a.acknowledge_unverified):
        print(f"  warning: {w}")

    check_resolvers()

    cfg = yaml.safe_load(Path(a.config).read_text())
    parcels = gpd.read_parquet(a.input).to_crs(CRS_WORK)
    a1 = load_a1(Path(a.a1))

    cw = load_crosswalk(Path(a.crosswalk), a1, parcels["ZONE"])

    parcels["GIS_ZONE"] = parcels["ZONE"].astype(str).str.strip()
    parcels = parcels.merge(
        cw[["gis_zone", "a1_zone", "basis"]].rename(
            columns={"gis_zone": "GIS_ZONE", "a1_zone": "A1_ZONE",
                     "basis": "zone_basis"}),
        on="GIS_ZONE", how="left")

    excluded = parcels["zone_basis"].isin({"excluded", "unzoned"})
    if excluded.any():
        print(f"  {int(excluded.sum()):,} parcels excluded from capacity "
              f"(right-of-way, water, unzoned) — reported, not silently dropped")
    parcels = parcels[~excluded].copy()

    # A1_ZONE is the standards key from here. ZONE stays as the label a
    # reader sees on the city's map, and both reach the digests.
    parcels["ZONE"] = parcels["A1_ZONE"]

    out, skipped = compute(parcels, a1, cfg)
    dest = Path(a.outdir); dest.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest / "parcels_capacity.parquet", index=False)

    rep = qa(out, skipped, cfg)
    (dest / "S2_qa.json").write_text(json.dumps(rep, indent=2, default=str))
    print("\n  QA")
    for k, v in rep.items():
        print(f"    {k:<42} {v}")
    print(f"\n  wrote {dest/'parcels_capacity.parquet'}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
