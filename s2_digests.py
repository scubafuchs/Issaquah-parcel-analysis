"""
s2_digests.py — emit the D2 markdown digests for the Issaquah Development Agent.

Consumes the per-parcel capacity/constraint join produced by s2_capacity.py.
Emits four markdown files into out/:

    D2A_headroom.md      capacity by zone and subarea
    D2B_constraints.md   constraint incidence, cost, schedule, overlap
    D2C_candidates.md    ranked redevelopment candidates by subarea
    D2_manifest.md       field dictionary, coverage, provenance, regeneration

Companion to A2 Method Spec v0.2 and d2_constraints.py.

Design rules enforced here, not left to the caller:
  1. Every section restamps its own as-of date, units, and caveat. A retrieved
     section arrives alone.
  2. Missing renders as "—", never 0. Coverage is reported beside every aggregate.
  3. Row counts are bounded. A digest that dumps 10,000 rows dilutes retrieval
     and is worse than the parcel table it was meant to summarize.
  4. Unverified constraint values propagate into every file as a banner, so the
     agent labels downstream figures estimates per A2 Part 2 failure behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from d2_constraints import (
    BASIS_DATE,
    SPEC_VERSION,
    CONSTRAINT_LAYERS,
    LAYERS_BY_KEY,
    VERIFY_BEFORE_RUN,
    CapacityMode,
    validate,
)

MISSING = "—"
SQFT_PER_ACRE = 43_560.0

# Retrieval budgets. Past these the digest stops summarizing and starts dumping.
MAX_CANDIDATES_PER_SUBAREA = 40
MAX_ZONE_SUBAREA_CELLS = 120
MAX_COOCCURRENCE_PAIRS = 15
MIN_COVERAGE = 0.90          # share of parcels with a resolved zone
MIN_CELL_PARCELS = 3         # below this a zone×subarea cell is noise, not signal


# ──────────────────────────────────────────────────────────────────────────────
# Input contract
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: dict[str, str] = {
    "PIN":                 "King County parcel identifier, major+minor, 10 char",
    "ZONE":                "Issaquah zone code, spatial join from city GIS",
    "SUBAREA":             "Issaquah subarea or neighborhood name",
    "lot_sf":              "Lot area, square feet, from EXTR_Parcel.SqFtLot",
    "existing_units":      "Existing dwelling units, integer",
    "yr_built":            "Earliest year built across buildings on parcel",
    "ApprLandVal":         "Appraised land value, dollars",
    "ApprImpsVal":         "Appraised improvement value, dollars",
    "i_l_ratio":           "Improvement value divided by land value",
    "allowed_units_base":  "Allowed units before constraints, A1 standards",
    "capacity_factor":     "Multiplier on allowed units, 0.0 to 1.0",
    "encumbered_fraction": "Share of lot removed by AREA_SHARE layers, union basis",
    "encumbered_fraction_naive": "Same, summed per layer — for the overlap QA line",
    "net_capacity":        "Allowed units after constraints minus existing units",
    "cost_adder_total":    "Total site adder for the tested product, dollars",
    "predev_months_add":   "Additional predevelopment months from constraints",
    "tested_product":      "Product type the adders and capacity were computed for",
    "capacity_source":  "code | development_agreement | none_tract | none_non_residential | unkeyed",
    "parcel_role":      "parcel | condo_complex | tract (EXTR_Parcel PropType, NbrBldgSites, tract flags)",
    "unit_class":       "housing_unit | group_quarters | mixed_review (PresentUse)",
    "units_basis":      "extract | presentuse_floor | manual_override | tract | none",
    "units_netted":     "Housing units subtracted from allowed units; 0 for group quarters",
}


def constraint_flag_columns(df: pd.DataFrame) -> list[str]:
    """c_<key> boolean columns present in the frame, in registry order."""
    return [f"c_{c.key}" for c in CONSTRAINT_LAYERS if f"c_{c.key}" in df.columns]


def check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "s2_capacity output is missing required columns: "
            + ", ".join(missing)
            + ". Emitting digests from a partial frame produces tables that look "
              "complete and understate capacity."
        )
    flags = constraint_flag_columns(df)
    absent = [c.key for c in CONSTRAINT_LAYERS if f"c_{c.key}" not in df.columns]
    if not flags:
        raise ValueError("No c_<key> constraint flag columns found.")
    if absent:
        print(f"  note: {len(absent)} registry layers absent from the join "
              f"(likely unresolved layer_ref): {', '.join(absent)}")


# ──────────────────────────────────────────────────────────────────────────────
# Run metadata
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunMeta:
    run_id: str
    parcel_vintage: str                 # as-of date of the assessor extract
    zoning_vintage: str                 # as-of date of the city GIS pull
    a1_version: str                     # zone standards matrix version keyed
    generated: str = date.today().isoformat()
    basis_date: str = BASIS_DATE
    spec_version: str = SPEC_VERSION
    config_hash: str = ""
    unverified: tuple[str, ...] = ()

    @classmethod
    def build(cls, *, run_id: str, parcel_vintage: str,
              zoning_vintage: str, a1_version: str) -> "RunMeta":
        payload = json.dumps(
            [[c.key, c.capacity_mode.value, c.capacity_fraction,
              c.buffer_ft, c.buffer_by_field,
              None if c.cost is None else [c.cost.rate, c.cost.basis.value,
                                           c.cost.group, c.cost.applies_if],
              c.predev_months_add]
             for c in CONSTRAINT_LAYERS],
            sort_keys=True, default=str,
        )
        return cls(
            run_id=run_id,
            parcel_vintage=parcel_vintage,
            zoning_vintage=zoning_vintage,
            a1_version=a1_version,
            config_hash=hashlib.sha256(payload.encode()).hexdigest()[:12],
            unverified=tuple(k for k in VERIFY_BEFORE_RUN
                             if not LAYERS_BY_KEY[k].verified),
        )


def stamp(meta: RunMeta) -> str:
    """One-line provenance. Goes at the top of every section, not just every file."""
    return (f"*Run `{meta.run_id}` · generated {meta.generated} · "
            f"parcels as of {meta.parcel_vintage} · zoning as of {meta.zoning_vintage} · "
            f"A1 {meta.a1_version} · constraint config `{meta.config_hash}`*")


def caveat(meta: RunMeta) -> str:
    """Propagates unverified constraint values into every emitted file."""
    if not meta.unverified:
        return ("> All constraint values verified against IMC 18.802 and local "
                "cost evidence as of this run.")
    lines = [
        "> **Estimates, not findings.** The following constraint values are "
        "unverified assumptions; every capacity, cost, and schedule figure in "
        "this file inherits that status and is to be labeled an estimate:",
        ">",
    ]
    lines += [f"> - `{k}` — {VERIFY_BEFORE_RUN[k]}" for k in meta.unverified]
    lines += [">", "> Per A2 Part 2 failure behavior, a verdict resting on any of "
                   "these is an upper bound, and the omission is stated in the answer."]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Formatting — deterministic, so a spot-check against source never drifts
# ──────────────────────────────────────────────────────────────────────────────

def fmt_int(v) -> str:
    return MISSING if pd.isna(v) else f"{int(round(float(v))):,}"


def fmt_money(v) -> str:
    return MISSING if pd.isna(v) else f"${float(v):,.0f}"


def fmt_acres(v) -> str:
    return MISSING if pd.isna(v) else f"{float(v):,.1f}"


def fmt_pct(v, digits: int = 0) -> str:
    return MISSING if pd.isna(v) else f"{float(v) * 100:.{digits}f}%"


def fmt_num(v, digits: int = 1) -> str:
    return MISSING if pd.isna(v) else f"{float(v):,.{digits}f}"


def md_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    """Markdown table with no dependency on to_markdown's version quirks."""
    if not rows:
        return "_No rows meet the inclusion rule for this section._"
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(c, MISSING)) for c in columns) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])


def coverage(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["ZONE"].notna().mean())


def check_coverage(df: pd.DataFrame) -> None:
    cov = coverage(df)
    if cov < MIN_COVERAGE:
        raise ValueError(
            f"Zone join coverage is {cov:.1%}, below the {MIN_COVERAGE:.0%} floor. "
            "Emitting now would publish aggregates that read as complete while "
            "silently omitting unzoned parcels. Fix the spatial join first."
        )


# ──────────────────────────────────────────────────────────────────────────────
# D2A — headroom
# ──────────────────────────────────────────────────────────────────────────────

_HEADROOM_COLS = [
    "Group", "Parcels", "Lot acres", "Existing units",
    "Allowed (pre-constraint)", "Allowed (post-constraint)", "Net capacity",
    "Parcels encumbered", "Median encumbered share", "Zone resolved",
]


def _headroom_rows(df: pd.DataFrame, by: str | None, label: str | None = None) -> list[dict]:
    groups: Iterable[tuple[str, pd.DataFrame]]
    groups = [(label or "Citywide", df)] if by is None else sorted(
        ((str(k), g) for k, g in df.groupby(by, dropna=False)), key=lambda t: t[0]
    )
    rows = []
    for name, g in groups:
        post = g["allowed_units_base"] * g["capacity_factor"]
        rows.append({
            "Group": name,
            "Parcels": fmt_int(len(g)),
            "Lot acres": fmt_acres(g["lot_sf"].sum() / SQFT_PER_ACRE),
            "Existing units": fmt_int(
                g.loc[g["unit_class"].eq("housing_unit"), "existing_units"].sum()),
            "Allowed (pre-constraint)": fmt_int(g["allowed_units_base"].sum()),
            "Allowed (post-constraint)": fmt_int(post.sum()),
            "Net capacity": fmt_int(g["net_capacity"].clip(lower=0).sum()),
            "Parcels encumbered": fmt_pct((g["encumbered_fraction"] > 0).mean()),
            "Median encumbered share": fmt_pct(
                g.loc[g["encumbered_fraction"] > 0, "encumbered_fraction"].median()),
            "Zone resolved": fmt_pct(coverage(g)),
        })
    return rows


_SOURCE_LABEL = {
    "code":                 "Code (A1 standards applied)",
    "development_agreement": "Development agreement",
    "none_tract":           "Common tract — no building site",
    "none_non_residential": "No residential use permitted",
    "unkeyed":              "Standards not yet keyed in A1",
}


def _capacity_source_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for src, g in df.groupby("capacity_source", dropna=False):
        computable = src == "code"
        rows.append({
            "Capacity source": _SOURCE_LABEL.get(str(src), str(src)),
            "Parcels": fmt_int(len(g)),
            "Lot acres": fmt_acres(g["lot_sf"].sum() / SQFT_PER_ACRE),
            "Existing housing units": fmt_int(
                g.loc[g["unit_class"].eq("housing_unit"), "existing_units"].sum()),
            "Group quarters": fmt_int(
                g.loc[g["unit_class"].ne("housing_unit"), "existing_units"].sum()),
            "Net capacity": (fmt_int(g["net_capacity"].clip(lower=0).sum())
                             if computable else MISSING),
        })
    return sorted(rows, key=lambda r: r["Capacity source"])


def digest_headroom(df: pd.DataFrame, meta: RunMeta) -> str:
    s, c = stamp(meta), caveat(meta)
    parts: list[str] = [
        "# D2A — Development capacity by zone and subarea",
        "",
        s,
        "",
        c,
        "",
        "**Units of measure.** Counts are dwelling units. *Net capacity* is "
        "post-constraint allowed units minus existing units, floored at zero — "
        "a parcel built beyond current allowance contributes zero, not a negative. "
        "Acres are lot area, not developable area. *Encumbered share* is the "
        "union-basis fraction of lot area removed by area-share constraints.",
        "",
        "---",
        "",
    ]

    parts += ["## Citywide totals", "", s, "",
              md_table(_headroom_rows(df, None), _HEADROOM_COLS), "", "---", ""]

    parts += [
        "## Capacity by source",
        "",
        s,
        "",
        "*Only parcels under **Code** carry a computed net capacity. The dash "
        "elsewhere is deliberate — those parcels are not low-capacity, they "
        "are parcels the headroom question does not apply to. Development "
        "agreement parcels take their units from the executed agreement and "
        "the D7 pipeline. Common tracts are shared driveways, open space and "
        "protected easements with no building site. Group quarters are "
        "counted separately and never netted against housing units.*",
        "",
        md_table(_capacity_source_rows(df),
                 ["Capacity source", "Parcels", "Lot acres",
                  "Existing housing units", "Group quarters", "Net capacity"]),
        "", "---", "",
    ]

    parts += ["## Capacity by zone", "",
              s, "",
              "*One row per zone. Net capacity is the figure the forecast's step 1 "
              "reads.*", "",
              md_table(_headroom_rows(df, "ZONE"), _HEADROOM_COLS), "", "---", ""]

    parts += ["## Capacity by subarea", "", s, "",
              md_table(_headroom_rows(df, "SUBAREA"), _HEADROOM_COLS), "", "---", ""]

    # zone × subarea, bounded and thresholded
    cell = df.copy()
    cell["_k"] = cell["SUBAREA"].astype(str) + " · " + cell["ZONE"].astype(str)
    sizes = cell.groupby("_k")["net_capacity"].agg(["count", "sum"])
    keep = sizes[sizes["count"] >= MIN_CELL_PARCELS] \
        .sort_values("sum", ascending=False).head(MAX_ZONE_SUBAREA_CELLS).index
    sub = cell[cell["_k"].isin(keep)]
    omitted = len(sizes) - len(keep)

    parts += [
        "## Capacity by zone within subarea",
        "",
        s,
        "",
        f"*Ranked by net capacity. Cells with fewer than {MIN_CELL_PARCELS} parcels "
        f"are excluded as noise; {omitted} of {len(sizes)} cells omitted on that "
        f"basis or by the {MAX_ZONE_SUBAREA_CELLS}-row cap. Omitted cells are "
        f"included in the citywide and per-zone totals above — do not sum this "
        f"table to a citywide figure.*",
        "",
        md_table(_headroom_rows(sub, "_k"), _HEADROOM_COLS),
        "",
    ]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# D2B — constraints
# ──────────────────────────────────────────────────────────────────────────────

_INCIDENCE_COLS = [
    "Constraint", "Mode", "Parcels hit", "Share of parcels",
    "Capacity units removed", "Median cost adder", "Total cost adder",
    "Predev months added", "Tag", "Verified", "Citation",
]


def _incidence_rows(df: pd.DataFrame, meta: RunMeta) -> list[dict]:
    rows = []
    for layer in CONSTRAINT_LAYERS:
        col = f"c_{layer.key}"
        if col not in df.columns:
            continue
        hit = df[df[col].fillna(False)]
        if hit.empty:
            continue
        # capacity attributable to this layer, computed by removal not by share
        if layer.capacity_mode is CapacityMode.PROHIBIT:
            removed = hit["allowed_units_base"].sum()
        elif layer.capacity_mode is CapacityMode.FLAT:
            removed = (hit["allowed_units_base"] * layer.capacity_fraction).sum()
        elif layer.capacity_mode is CapacityMode.AREA_SHARE:
            removed = (hit["allowed_units_base"] * hit["encumbered_fraction"]).sum()
        else:
            removed = None
        rows.append({
            "Constraint": f"`{layer.key}`",
            "Mode": layer.capacity_mode.value,
            "Parcels hit": fmt_int(len(hit)),
            "Share of parcels": fmt_pct(len(hit) / len(df), 1),
            "Capacity units removed": (MISSING if removed is None
                                       else fmt_int(removed)),
            "Median cost adder": fmt_money(hit["cost_adder_total"].median()),
            "Total cost adder": fmt_money(hit["cost_adder_total"].sum()),
            "Predev months added": fmt_num(layer.predev_months_add, 1),
            "Tag": ("—" if layer.cost is None else layer.cost.tag.value),
            "Verified": "yes" if layer.verified else "**no**",
            "Citation": layer.imc or MISSING,
        })
    return rows


def _overlap_rows(df: pd.DataFrame) -> list[dict]:
    flags = [c for c in constraint_flag_columns(df)
             if LAYERS_BY_KEY[c[2:]].capacity_mode is CapacityMode.AREA_SHARE]
    pairs = []
    for a, b in combinations(flags, 2):
        n = int((df[a].fillna(False) & df[b].fillna(False)).sum())
        if n:
            pairs.append((n, a[2:], b[2:]))
    pairs.sort(reverse=True)
    return [{
        "Layer A": f"`{a}`", "Layer B": f"`{b}`",
        "Parcels hit by both": fmt_int(n),
        "Share of A also hit by B": fmt_pct(
            n / max(1, int(df[f"c_{a}"].fillna(False).sum())), 1),
    } for n, a, b in pairs[:MAX_COOCCURRENCE_PAIRS]]


def digest_constraints(df: pd.DataFrame, meta: RunMeta) -> str:
    s, c = stamp(meta), caveat(meta)
    union = (df["allowed_units_base"] * df["encumbered_fraction"]).sum()
    naive = (df["allowed_units_base"]
             * df["encumbered_fraction_naive"].clip(upper=1.0)).sum()

    parts: list[str] = [
        "# D2B — Constraint incidence, cost, and schedule effect",
        "",
        s,
        "",
        c,
        "",
        "**How to read this.** A constraint either removes capacity, adds cost, "
        "adds schedule, or some combination. *Mode* states which: `none` means "
        "cost or schedule only and capacity is untouched; `area_share` removes "
        "the measured overlap; `flat` applies a fixed siting penalty; `prohibit` "
        "zeroes capacity on the overlapping area. Cost adders are for the "
        f"product tested in this run ({df['tested_product'].mode().iat[0] if not df.empty else MISSING}) "
        "and change with product type — a townhouse and a podium on the same "
        "parcel do not carry the same adders.",
        "",
        "---",
        "",
        "## Incidence by constraint layer",
        "",
        s,
        "",
        "*Capacity removed is attributable to this layer in isolation. These "
        "figures do not sum to total capacity removed — see the overlap section "
        "below for why.*",
        "",
        md_table(_incidence_rows(df, meta), _INCIDENCE_COLS),
        "",
        "---",
        "",
        "## Overlap check — union basis versus naive sum",
        "",
        s,
        "",
        md_table([{
            "Basis": "Union of overlapping area-share constraints",
            "Capacity units removed": fmt_int(union),
            "Method": "Geometric union, then a single fraction per parcel",
        }, {
            "Basis": "Naive per-layer sum",
            "Capacity units removed": fmt_int(naive),
            "Method": "Each layer's fraction applied independently",
        }, {
            "Basis": "Difference (double-count avoided)",
            "Capacity units removed": fmt_int(naive - union),
            "Method": "Union basis governs",
        }], ["Basis", "Capacity units removed", "Method"]),
        "",
        f"The naive sum overstates removal by {fmt_int(naive - union)} units "
        f"({fmt_pct((naive - union) / union, 1) if union else MISSING} of the union "
        "figure). Steep slope, landslide hazard, and erosion hazard are frequently "
        "the same polygon in Issaquah's data. The union basis is what the forecast "
        "reads; the naive figure appears here only as evidence the double-count "
        "was avoided.",
        "",
        "---",
        "",
        "## Constraint co-occurrence",
        "",
        s,
        "",
        "*Area-share layers only. High co-occurrence is the reason capacity "
        "removal is computed once off a union rather than layer by layer.*",
        "",
        md_table(_overlap_rows(df),
                 ["Layer A", "Layer B", "Parcels hit by both",
                  "Share of A also hit by B"]),
        "",
        "---",
        "",
        "## Parcels carrying no constraint",
        "",
        s,
        "",
    ]

    flags = constraint_flag_columns(df)
    clean = df[~df[flags].fillna(False).any(axis=1)] if flags else df.iloc[0:0]
    parts += [md_table([{
        "Measure": "Unconstrained parcels",
        "Value": fmt_int(len(clean)),
        "Share": fmt_pct(len(clean) / len(df), 1) if len(df) else MISSING,
    }, {
        "Measure": "Net capacity on unconstrained parcels",
        "Value": fmt_int(clean["net_capacity"].clip(lower=0).sum()),
        "Share": fmt_pct(
            clean["net_capacity"].clip(lower=0).sum()
            / max(1.0, df["net_capacity"].clip(lower=0).sum()), 1),
    }], ["Measure", "Value", "Share"]),
        "",
        "Unconstrained capacity is where feasibility turns on cost of capital "
        "alone, with no site adder and no added review time. It is the cleanest "
        "test of whether the rate environment or the geology is binding.",
        ""]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# D2C — redevelopment candidates
# ──────────────────────────────────────────────────────────────────────────────

_CANDIDATE_COLS = [
    "PIN", "Zone", "Lot acres", "Existing units", "Units basis",
    "Net capacity", "I/L ratio", "Decile", "Land value", "Site adders",
    "Predev months added", "Constraints",
]


def _candidate_rows(g: pd.DataFrame) -> list[dict]:
    flags = constraint_flag_columns(g)
    rows = []
    for _, r in g.iterrows():
        hits = [c[2:] for c in flags if bool(r.get(c, False))]
        rows.append({
            "PIN": f"`{r['PIN']}`",
            "Zone": r["ZONE"] if pd.notna(r["ZONE"]) else MISSING,
            "Lot acres": fmt_acres(r["lot_sf"] / SQFT_PER_ACRE),
            "Existing units": fmt_int(r["existing_units"]),
            "Units basis": str(r.get("units_basis") or MISSING),
            "Net capacity": fmt_int(r["net_capacity"]),
            "I/L ratio": fmt_num(r["i_l_ratio"], 2),
            "Decile": fmt_int(r.get("redev_decile")),
            "Land value": fmt_money(r["ApprLandVal"]),
            "Site adders": fmt_money(r["cost_adder_total"]),
            "Predev months added": fmt_num(r["predev_months_add"], 1),
            "Constraints": ", ".join(f"`{h}`" for h in hits) if hits else "none",
        })
    return rows


def digest_candidates(df: pd.DataFrame, meta: RunMeta,
                      top_n: int = MAX_CANDIDATES_PER_SUBAREA) -> str:
    s, c = stamp(meta), caveat(meta)
    parts: list[str] = [
        "# D2C — Redevelopment candidates by subarea",
        "",
        s,
        "",
        c,
        "",
        "**Ranking basis.** Net capacity descending, then improvement-to-land "
        "ratio ascending. A low I/L ratio means the building is worth little "
        "relative to the land under it, which is the condition signal the "
        "forecast's step 3 reads. Parcels with net capacity of zero or less are "
        "excluded.",
        "",
        "**This is a capacity and condition ranking, not a feasibility ranking.** "
        "No parcel on this list has been tested against A2 Part 2. A high-capacity "
        "parcel carrying a large site adder can rank at the top here and fail "
        "feasibility outright. Do not describe a parcel as a development "
        "opportunity on the strength of this table alone.",
        "",
        "---",
        "",
    ]

    cand = df[df["net_capacity"].fillna(-1) > 0].copy()
    cand = cand.sort_values(["net_capacity", "i_l_ratio"], ascending=[False, True])

    for sub, g in sorted(cand.groupby("SUBAREA", dropna=False),
                         key=lambda t: str(t[0])):
        shown = g.head(top_n)
        parts += [
            f"## {sub} — top {len(shown)} candidates",
            "",
            s,
            "",
            f"*{fmt_int(len(g))} parcels in {sub} carry positive net capacity, "
            f"totaling {fmt_int(g['net_capacity'].sum())} units. "
            f"The {len(shown)} shown carry "
            f"{fmt_int(shown['net_capacity'].sum())} units "
            f"({fmt_pct(shown['net_capacity'].sum() / g['net_capacity'].sum(), 1)} "
            f"of the subarea total). Counts are dwelling units; dollars are "
            f"{meta.parcel_vintage} appraised values.*",
            "",
            md_table(_candidate_rows(shown), _CANDIDATE_COLS),
            "",
            "---",
            "",
        ]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# D2 manifest
# ──────────────────────────────────────────────────────────────────────────────

def digest_manifest(df: pd.DataFrame, meta: RunMeta,
                    warnings: Sequence[str]) -> str:
    flags = constraint_flag_columns(df)
    absent = [c.key for c in CONSTRAINT_LAYERS if f"c_{c.key}" not in df.columns]

    parts: list[str] = [
        "# D2 — Manifest, field dictionary, and provenance",
        "",
        stamp(meta),
        "",
        caveat(meta),
        "",
        "This file answers *where did that come from* for every figure in D2A, "
        "D2B, and D2C. Cite it when asked to source a capacity or site-cost number.",
        "",
        "---",
        "",
        "## Run",
        "",
        md_table([{"Field": k, "Value": f"`{v}`" if k in
                   ("run_id", "config_hash") else str(v)}
                  for k, v in asdict(meta).items() if k != "unverified"],
                 ["Field", "Value"]),
        "",
        "---",
        "",
        "## Coverage",
        "",
        stamp(meta),
        "",
        md_table([
            {"Measure": "Parcels in scope", "Value": fmt_int(len(df))},
            {"Measure": "Zone resolved by spatial join",
             "Value": fmt_pct(coverage(df), 2)},
            {"Measure": "Parcels with any constraint flag",
             "Value": fmt_pct(df[flags].fillna(False).any(axis=1).mean(), 1)
             if flags else MISSING},
            {"Measure": "Parcels with positive net capacity",
             "Value": fmt_int(int((df['net_capacity'] > 0).sum()))},
            {"Measure": "Constraint layers joined",
             "Value": f"{len(flags)} of {len(CONSTRAINT_LAYERS)}"},
            {"Measure": "Layers absent from the join",
             "Value": ", ".join(f"`{k}`" for k in absent) if absent else "none"},
            {"Measure": "Housing units", "Value": fmt_int(
                df.loc[df["unit_class"].eq("housing_unit"), "existing_units"].sum())},
            {"Measure": "Group quarters (beds, not netted)", "Value": fmt_int(
                df.loc[df["unit_class"].ne("housing_unit"), "existing_units"].sum())},
            {"Measure": "Unit counts by basis",
             "Value": ", ".join(f"{k}: {v:,}" for k, v in
                                df["units_basis"].value_counts().items())},
            {"Measure": "Condominium value rollups applied",
             "Value": fmt_int(df["condo_value_rollup"].sum())},
        ], ["Measure", "Value"]),
        "",
        "Zoning comes from the City of Issaquah GIS layer by spatial join, never "
        "from the assessor's `CurrentZoning` field, which is unreliable for "
        "incorporated parcels. Scope is the city limits polygon, not "
        "`DistrictName == 'ISSAQUAH'`, because that field lags annexations by "
        "months. Parcels in the PAA are flagged separately and are outside this "
        "digest.",
        "",
        "---",
        "",
        "## Field dictionary",
        "",
        stamp(meta),
        "",
        md_table([{"Field": f"`{k}`", "Definition": v}
                  for k, v in REQUIRED_COLUMNS.items()],
                 ["Field", "Definition"]),
        "",
        "---",
        "",
        "## Constraint registry as run",
        "",
        stamp(meta),
        "",
        md_table([{
            "Layer": f"`{c.key}`",
            "Label": c.label,
            "Mode": c.capacity_mode.value,
            "Fraction": fmt_num(c.capacity_fraction, 2) if c.capacity_fraction else MISSING,
            "Cost rate": (MISSING if c.cost is None
                          else f"{fmt_money(c.cost.rate)} / {c.cost.basis.value}"),
            "Group": (c.cost.group if c.cost and c.cost.group else MISSING),
            "Fires when": (c.cost.applies_if if c.cost else MISSING),
            "Predev months": fmt_num(c.predev_months_add, 1),
            "Tag": (MISSING if c.cost is None else c.cost.tag.value),
            "Verified": "yes" if c.verified else "**no**",
            "Citation": c.imc or MISSING,
        } for c in CONSTRAINT_LAYERS],
            ["Layer", "Label", "Mode", "Fraction", "Cost rate", "Group",
             "Fires when", "Predev months", "Tag", "Verified", "Citation"]),
        "",
        "Adders sharing a `Group` do not stack — the largest applicable adder in "
        "the group applies once. Liquefaction, peat, seismic hazard, and coal "
        "mine hazard all buy a single deep-foundation scope.",
        "",
        "---",
        "",
        "## Regeneration",
        "",
        stamp(meta),
        "",
        "```",
        "python s1_parcel_master.py --vintage <YYYY-MM-DD>",
        "python s2_capacity.py     --vintage <YYYY-MM-DD>",
        "python s2_digests.py      --run-id <id> --parcel-vintage <YYYY-MM-DD> \\",
        "                          --zoning-vintage <YYYY-MM-DD> --a1-version <ver>",
        "```",
        "",
        "Replace D2A, D2B, D2C and this manifest together as one set. A digest "
        "from one run sitting beside a digest from another makes the agent hedge, "
        "and it is right to.",
        "",
    ]
    if warnings:
        parts += ["---", "", "## Emitter warnings", "", stamp(meta), "",
                  *[f"- {w}" for w in warnings], ""]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def emit_all(df: pd.DataFrame, outdir: Path, meta: RunMeta,
             *, acknowledge_unverified: bool = False) -> dict[str, Path]:
    warnings = list(validate(acknowledge_unverified=acknowledge_unverified))
    check_schema(df)
    check_coverage(df)

    outdir.mkdir(parents=True, exist_ok=True)
    files = {
        "D2A_headroom.md":    digest_headroom(df, meta),
        "D2B_constraints.md": digest_constraints(df, meta),
        "D2C_candidates.md":  digest_candidates(df, meta),
        "D2_manifest.md":     digest_manifest(df, meta, warnings),
    }
    written = {}
    for name, text in files.items():
        path = outdir / name
        path.write_text(text, encoding="utf-8")
        written[name] = path
        print(f"  wrote {path}  ({len(text):,} chars, "
              f"{text.count(chr(10)) + 1:,} lines)")
    for w in warnings:
        print(f"  warning: {w}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Emit the D2 markdown digests.")
    p.add_argument("--input", default="out/parcels_capacity.parquet")
    p.add_argument("--outdir", default="out")
    p.add_argument("--run-id", required=True)
    p.add_argument("--parcel-vintage", required=True)
    p.add_argument("--zoning-vintage", required=True)
    p.add_argument("--a1-version", required=True)
    p.add_argument("--acknowledge-unverified", action="store_true",
                   help="Emit despite unverified constraint values. Every "
                        "downstream figure must then be labeled an estimate.")
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    meta = RunMeta.build(run_id=args.run_id,
                         parcel_vintage=args.parcel_vintage,
                         zoning_vintage=args.zoning_vintage,
                         a1_version=args.a1_version)
    emit_all(df, Path(args.outdir), meta,
             acknowledge_unverified=args.acknowledge_unverified)


if __name__ == "__main__":
    main()
