"""
s3_parcel_pages.py — emit retrieval-optimized parcel pages for the knowledge base.

Reads   out/parcels_master.parquet        (s1_parcel_master.py)        REQUIRED
        out/parcels_capacity.parquet      (s2_capacity.py)             strongly advised
        kb/parcel_watchlist.csv           PIN,name — landmark aliases  optional
        data/parcel_addresses.csv         Issaquah Open Data address points  optional

Writes  out/kb/P1_dossiers_<subarea>_<nn>.md
        out/kb/P2_index_<subarea>_<nn>.md
        out/kb/P3_addresses_<nn>.md
        out/kb/P4_zone_profiles.md
        out/kb/P5_parcel_lookup_guide.md
        out/kb/P_manifest.md
        out/kb/P_summary.json

Design rules, each reversing a failure of the sharded parcel table:

  1. One SECTION per parcel in the dossier tier, not one row. A retrieved
     section arrives by itself, so it restates the PIN, address, zone and
     as-of date rather than relying on a header above it.
  2. The index tier carries enough to answer a small-parcel question without
     a dossier: zone, lot area, existing units, net capacity, land value,
     transit proximity, buildable share, and a packed constraint summary.
     Issaquah redevelopment is a few big projects plus many small ones, and
     the small ones live here.
  3. Constraint codes are abbreviated, and each section carries a legend for
     ONLY the codes appearing in it. A legend per row would triple the index;
     a legend only in the manifest would not retrieve with the section.
  4. Facts as statements with their conditions attached. Missing renders as an
     em dash, never as zero.
  5. No verdicts. These pages carry feasibility INPUTS; the pencils / marginal
     / does-not-pencil call is computed at answer time from A2 Part 2, whose
     constants change faster than the parcel table does.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

MISSING = "—"
SQFT_PER_ACRE = 43_560.0

# Retrieval budgets. Past these, sections stop being findable because they
# compete with their own near-identical siblings.
DEFAULT_DOSSIER_BUDGET = 1_200
DOSSIERS_PER_FILE = 60
INDEX_ROWS_PER_SECTION = 60
INDEX_SECTIONS_PER_FILE = 12
MIN_STREET_SECTION = 3

# Thresholds for the index prose, so a section can say what it contains.
SIGNIFICANT_AREA_LOSS = 0.20      # 20%+ of allowed units lost to constraints
SIGNIFICANT_COST_ADDER = 100_000  # dollars


# ──────────────────────────────────────────────────────────────────────────────
# Constraint codes
#
# Short, stable, and paired with their EFFECT, because the whole point of the
# summary is to let the agent guess at a significant buildable-area reduction
# or a significant cost increase without opening a dossier.
# ──────────────────────────────────────────────────────────────────────────────

CONSTRAINT_CODES: dict[str, tuple[str, str]] = {
    # geotechnical — cost, not capacity
    "liquefaction":               ("liq",   "liquefaction — deep foundations, adds cost"),
    "peat_soft_soils":            ("peat",  "peat or soft soils — deep foundations, adds cost"),
    "seismic_hazard":             ("seis",  "seismic hazard — foundation work, adds cost"),
    "coal_mine_hazard":           ("coal",  "coal mine hazard — remediation cost, siting penalty, adds review time"),
    "erosion_hazard":             ("eros",  "erosion hazard — erosion control, adds cost"),
    # topography — removes area and adds cost
    "steep_slope":                ("slope", "steep slope 40%+ — removes buildable area, adds earthwork cost"),
    "landslide_hazard":           ("slide", "landslide hazard — removes buildable area, adds review time"),
    "landslide_affected_parcels": ("slide?", "flagged landslide-affected parcel — informational only"),
    # water — removes area, adds cost and time
    "stream_buffer":              ("strm",  "stream buffer — removes buildable area, adds mitigation cost and review time"),
    "wetland_buffer":             ("wet",   "wetland buffer — removes buildable area, adds mitigation cost and review time"),
    "floodway":                   ("fldwy", "regulatory floodway — no residential capacity on the area covered"),
    "floodplain_100yr":           ("flood", "100-year floodplain — partial area loss, flood-resistant construction cost"),
    "high_groundwater":           ("gw",    "high groundwater — dewatering cost on below-grade construction"),
    "shoreline_jurisdiction":     ("shore", "shoreline jurisdiction — siting penalty and the longest added review time"),
    # aquifer
    "cara":                       ("cara",  "critical aquifer recharge area — stormwater vault cost"),
    "wellhead_1yr":               ("well",  "wellhead protection — siting penalty and stormwater cost"),
    # derived
    "frontage_unimproved":        ("front", "unimproved street frontage — frontage improvement cost"),
    "lot_line_shoring":           ("shor",  "lot-line excavation — shoring cost on below-grade construction"),
}


def code_for(key: str) -> str:
    return CONSTRAINT_CODES.get(key, (key[:5], key.replace("_", " ")))[0]


def legend_for(key: str) -> str:
    return CONSTRAINT_CODES.get(key, (key[:5], key.replace("_", " ")))[1]


def flag_cols(df: pd.DataFrame) -> list[str]:
    """c_<key> columns present, in registry order where known."""
    known = [f"c_{k}" for k in CONSTRAINT_CODES if f"c_{k}" in df.columns]
    other = sorted(c for c in df.columns
                   if c.startswith("c_") and c not in known)
    return known + other


def codes_present(g: pd.DataFrame, flags: Sequence[str]) -> list[str]:
    """Constraint keys with at least one hit in this slice of the frame."""
    return [c[2:] for c in flags if g[c].fillna(False).any()]


def legend_line(keys: Sequence[str]) -> str:
    if not keys:
        return ""
    parts = " · ".join(f"`{code_for(k)}` = {legend_for(k)}" for k in keys)
    return f"**Constraint codes used in this section.** {parts}"


# ──────────────────────────────────────────────────────────────────────────────
# Input contract
# ──────────────────────────────────────────────────────────────────────────────

MASTER_REQUIRED = ["PIN", "ZONE", "SUBAREA", "lot_sf", "existing_units",
                   "ApprLandVal", "ApprImpsVal", "i_l_ratio"]

CAPACITY_FIELDS = ["allowed_units_base", "capacity_factor", "net_capacity",
                   "capacity_source", "encumbered_fraction", "cost_adder_total",
                   "predev_months_add", "tested_product", "footprint_sf",
                   "redev_decile"]

FIELD_DEFINITIONS = {
    "PIN": "King County parcel identifier, Major (6) + Minor (4), 10 characters",
    "ZONE": "Issaquah zone code, spatial join from the city planning service",
    "SUBAREA": "Issaquah subarea, spatial join",
    "site_address": "Situs address where published; blank where none is assigned",
    "lot_sf": "Lot area in square feet, assessor EXTR_Parcel.SqFtLot",
    "existing_units": "Existing dwelling units; group quarters counted separately",
    "units_basis": "extract | presentuse_floor | manual_override | tract | none",
    "ApprLandVal": "Assessed land value, dollars, summed across tax accounts",
    "allowed_units_base": "Allowed units under A1 zone standards, before constraints",
    "capacity_factor": "Share of allowed units surviving site constraints, 0.0-1.0",
    "net_capacity": "Allowed units after constraints minus existing housing units",
    "capacity_source": "code | development_agreement | none_tract | "
                       "none_non_residential | unkeyed",
    "cost_adder_total": "Site cost adders for the tested product, dollars; "
                        "unverified rates, so an estimate rather than a finding",
    "predev_months_add": "Added predevelopment months from constraint review",
    "near_transit": "Within a quarter mile of a parcel treated as a major "
                    "transit stop; raises the middle-housing floor from 2 to 4",
    "i_l_ratio": "Improvement value divided by land value",
    "redev_decile": "1 = lowest improvement-to-land ratio, the strongest "
                    "redevelopment condition signal",
}

ADDRESS_PIN_FIELD = "ParcelPIN"
ADDRESS_PIN_FALLBACK = "ParcelOwnerPIN"
ADDRESS_EXCLUDE_FLAGS = ("IsWaterMeterAddress", "IsIrrigationWaterMeter")


# ──────────────────────────────────────────────────────────────────────────────
# Run metadata
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunMeta:
    run_id: str
    parcel_vintage: str
    zoning_vintage: str
    a1_version: str
    capacity_joined: bool
    addresses_joined: bool
    generated: str = date.today().isoformat()


def stamp(m: RunMeta) -> str:
    """Provenance line, repeated in EVERY section — a retrieved section
    arrives without the file header above it."""
    return (f"*Parcel page · run `{m.run_id}` · parcels as of {m.parcel_vintage} "
            f"· zoning as of {m.zoning_vintage} · A1 {m.a1_version}*")


# ──────────────────────────────────────────────────────────────────────────────
# NA-safe scalars
#
# pandas NA raises TypeError on bool(), so every `if value:` over a column that
# can be missing is a latent crash. Ordering matters too: `not v or _na(v)`
# evaluates the bool first and raises before the guard runs.
# ──────────────────────────────────────────────────────────────────────────────

def _na(v) -> bool:
    if v is None:
        return True
    try:
        res = pd.isna(v)
    except (TypeError, ValueError):
        return False
    return bool(res) if isinstance(res, (bool, np.bool_)) else False


def truthy(v) -> bool:
    if _na(v):
        return False
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "nan", "none", "0")
    try:
        return bool(v)
    except (TypeError, ValueError):
        return False


def text(v) -> str:
    return "" if _na(v) else str(v).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────────────

def f_int(v) -> str:
    return MISSING if _na(v) else f"{int(round(float(v))):,}"


def f_money(v) -> str:
    return MISSING if _na(v) else f"${float(v):,.0f}"


def f_money_short(v) -> str:
    """Compact dollars for table cells: $412k, $1.2M."""
    if _na(v):
        return MISSING
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}k"
    return f"${v:,.0f}"


def f_sf(v) -> str:
    if _na(v):
        return MISSING
    v = float(v)
    return f"{v:,.0f} sq ft ({v / SQFT_PER_ACRE:,.2f} acres)"


def f_num(v, d: int = 2) -> str:
    return MISSING if _na(v) else f"{float(v):,.{d}f}"


def f_pct(v, d: int = 0) -> str:
    return MISSING if _na(v) else f"{float(v) * 100:.{d}f}%"


def md_table(rows: Sequence[dict], cols: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, MISSING)) for c in cols) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])


def slug(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")
    return out or "unassigned"


# ──────────────────────────────────────────────────────────────────────────────
# Aliases — the single biggest retrieval lever
# ──────────────────────────────────────────────────────────────────────────────

def pin_forms(pin: str) -> list[str]:
    """Every way a PIN gets written. Someone pasting from the Assessor site,
    from a staff report and from a spreadsheet produces three strings."""
    p = str(pin).strip().zfill(10)
    return [p, f"{p[:6]}-{p[6:]}", f"Major {p[:6]} Minor {p[6:]}"]


_STREET_EXPAND = {
    " St": " Street", " Ave": " Avenue", " Rd": " Road", " Blvd": " Boulevard",
    " Dr": " Drive", " Ln": " Lane", " Pl": " Place", " Ct": " Court",
    " Ter": " Terrace", " Cir": " Circle", " Pkwy": " Parkway",
    " N": " North", " S": " South", " E": " East", " W": " West",
    " Ne": " Northeast", " Nw": " Northwest",
    " Se": " Southeast", " Sw": " Southwest",
}


def address_forms(addr) -> list[str]:
    """Abbreviated and spelled-out street types. A query says 'Front Street'
    where the record says 'FRONT ST'. Guard runs BEFORE any truthiness test."""
    a = " ".join(text(addr).split()).title()
    if not a:
        return []
    long = a
    for k, v in _STREET_EXPAND.items():
        long = re.sub(re.escape(k) + r"\b", v, long)
    return [a] if long == a else [a, long]


def street_name(addr) -> str | None:
    a = " ".join(text(addr).split()).title()
    if not a:
        return None
    m = re.match(r"^\s*\d+[A-Za-z]?\s+(.*)$", a)
    return (m.group(1) if m else a).strip() or None


def alias_line(r: pd.Series) -> str:
    parts = pin_forms(r["PIN"]) + address_forms(r.get("site_address"))
    for extra in (r.get("alt_addresses") or []):
        if text(extra) and text(extra) != text(r.get("site_address")):
            parts += address_forms(extra)
    name = text(r.get("watchlist_name"))
    if name:
        parts.append(name)
    return " · ".join(dict.fromkeys(p for p in parts if p))


# ──────────────────────────────────────────────────────────────────────────────
# Tier selection
# ──────────────────────────────────────────────────────────────────────────────

def dossier_score(df: pd.DataFrame) -> pd.Series:
    """Rank parcels by how likely someone is to ask about this one by name.

    A full page is a retrieval slot: spending one on a parcel nobody asks
    about makes every other page harder to find. Parcels not selected are not
    excluded — they get an index row carrying zone, lot area, units, capacity,
    land value, transit and constraints, which answers most questions about a
    small parcel on its own.
    """
    def z(col, log=False):
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        v = pd.to_numeric(df[col], errors="coerce")
        if log:
            v = np.log1p(v.clip(lower=0))
        return v.rank(pct=True).fillna(0.0)

    s = (
        3.0 * z("net_capacity", log=True)
        + 2.0 * z("lot_sf", log=True)
        + 1.5 * (11 - pd.to_numeric(df.get("redev_decile"), errors="coerce")
                 .fillna(11)) / 10.0
        + 1.0 * z("ApprLandVal", log=True)
    )
    s += 2.0 * df.get("in_development_agreement",
                      pd.Series(False, index=df.index)).map(truthy).astype(float)
    s += 1.5 * df.get("near_transit",
                      pd.Series(False, index=df.index)).map(truthy).astype(float)
    s += 1.0 * df["ZONE"].astype(str).str.startswith(
        ("MU", "UC", "VR", "CBD")).astype(float)
    s += 1.0 * (pd.to_numeric(df.get("existing_units"), errors="coerce")
                .fillna(0) >= 10).astype(float)
    s -= 2.0 * df.get("capacity_source", pd.Series("", index=df.index)).isin(
        ["none_tract", "none_non_residential"]).astype(float)
    return s


def assign_tiers(df: pd.DataFrame, budget: int) -> pd.DataFrame:
    df = df.copy()
    df["dossier_score"] = dossier_score(df)
    forced = df.get("watchlist_name", pd.Series(pd.NA, index=df.index)).notna()
    ranked = df.loc[~forced, "dossier_score"].nlargest(
        max(0, budget - int(forced.sum()))).index
    df["tier"] = np.where(forced | df.index.isin(ranked), "dossier", "index")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# The packed constraint summary — the new index field
# ──────────────────────────────────────────────────────────────────────────────

def constraint_cell(r: pd.Series, flags: Sequence[str],
                    capacity_joined: bool, max_codes: int = 4) -> str:
    """One cell carrying cost, added review time, and which constraints hit.

    Omits any component that is zero, so an unconstrained parcel reads as
    "none mapped" rather than as three zeros. Dollars are absolute rather than
    banded because the land value sits in the same row, and the comparison
    between the two is most of the small-parcel feasibility intuition.
    """
    if not capacity_joined:
        return MISSING
    src = text(r.get("capacity_source"))
    if src == "none_tract":
        return "no building site"

    hits = [c[2:] for c in flags if truthy(r.get(c))]
    cost = r.get("cost_adder_total")
    months = r.get("predev_months_add")

    bits = []
    if not _na(cost) and float(cost) > 0:
        bits.append(f_money_short(cost))
    if not _na(months) and float(months) > 0:
        bits.append(f"+{float(months):.0f}mo")
    if hits:
        shown = [code_for(h) for h in hits[:max_codes]]
        more = len(hits) - len(shown)
        bits.append(", ".join(shown) + (f" +{more}" if more > 0 else ""))

    return " · ".join(bits) if bits else "none mapped"


def buildable_cell(r: pd.Series, capacity_joined: bool) -> str:
    """Share of allowed units surviving site constraints.

    100% means unconstrained. Expressed as the share retained rather than the
    reduction, so the scale is monotone and there is no sign to misread.
    """
    if not capacity_joined:
        return MISSING
    src = text(r.get("capacity_source"))
    if src == "none_tract":
        return "0%"
    if src in ("development_agreement", "none_non_residential", "unkeyed"):
        return MISSING
    return f_pct(r.get("capacity_factor"), 0)


def transit_cell(r: pd.Series) -> str:
    return "yes" if truthy(r.get("near_transit")) else MISSING


# ──────────────────────────────────────────────────────────────────────────────
# P2 — the enriched index
# ──────────────────────────────────────────────────────────────────────────────

INDEX_COLS = ["PIN", "Address", "Zone", "Lot area", "Existing units",
              "Net capacity", "Land value", "¼-mi transit", "Buildable",
              "Site constraints", "I/L ratio"]


def _index_rows(g: pd.DataFrame, flags: Sequence[str],
                capacity_joined: bool) -> list[dict]:
    return [{
        "PIN": f"`{r['PIN']}`",
        "Address": (address_forms(r.get("site_address")) or [MISSING])[0],
        "Zone": r["ZONE"] if not _na(r.get("ZONE")) else MISSING,
        "Lot area": (f"{int(round(float(r['lot_sf']))):,}"
                     if not _na(r.get("lot_sf")) else MISSING),
        "Existing units": f_int(r.get("existing_units")),
        "Net capacity": f_int(r.get("net_capacity")),
        "Land value": f_money_short(r.get("ApprLandVal")),
        "¼-mi transit": transit_cell(r),
        "Buildable": buildable_cell(r, capacity_joined),
        "Site constraints": constraint_cell(r, flags, capacity_joined),
        "I/L ratio": f_num(r.get("i_l_ratio")),
    } for _, r in g.iterrows()]


def _index_section_notes(sec: pd.DataFrame, flags: Sequence[str],
                         capacity_joined: bool) -> list[str]:
    """Reading guidance and a legend covering only the codes in this section.

    Conditional by design. A legend repeated per row would triple the index;
    a legend only in the manifest would not retrieve alongside the section.
    """
    out = []
    if capacity_joined:
        out.append(
            "**How to read this.** *Lot area* is square feet. *Net capacity* is "
            "allowed units after site constraints minus existing housing units. "
            "*Buildable* is the share of allowed units surviving those "
            "constraints, so 100% means nothing mapped reduces it. *Site "
            "constraints* lists estimated added site cost, added "
            "predevelopment months, and the constraints that apply. Compare "
            "the site cost against the land value in the same row: an added "
            "cost approaching or exceeding land value is a strong signal the "
            "parcel will not pencil, before any pro forma is run. Constraint "
            "cost rates are unverified assumptions, so these are estimates "
            "rather than findings.")
    else:
        out.append(
            "**How to read this.** Capacity and constraint columns are not "
            "available in this run, so they show a dash. Lot area, existing "
            "units and land value are measured.")

    keys = codes_present(sec, flags)
    if keys:
        out.append(legend_for_section := legend_line(keys))
    if sec.apply(lambda r: truthy(r.get("near_transit")), axis=1).any():
        out.append(
            "**Transit proximity.** A parcel marked yes under ¼-mi transit lies "
            "within a quarter mile of a parcel treated as a major transit stop, "
            "which raises the state middle-housing floor from two units per lot "
            "to four. Issaquah has not adopted a major transit stop map, so "
            "that proximity is a working assumption and the four-unit floor is "
            "an upper bound.")
    return [o for o in out if o]


def emit_index(df: pd.DataFrame, meta: RunMeta, outdir: Path,
               flags: Sequence[str]) -> list[Path]:
    written = []
    d = df[df["tier"] == "index"].copy()
    sort_key = "site_address" if "site_address" in d.columns else "PIN"
    for sub, grp in d.groupby(d["SUBAREA"].fillna("Unassigned")):
        grp = grp.sort_values([sort_key, "PIN"])
        secs = [grp.iloc[i:i + INDEX_ROWS_PER_SECTION]
                for i in range(0, len(grp), INDEX_ROWS_PER_SECTION)]
        files = [secs[i:i + INDEX_SECTIONS_PER_FILE]
                 for i in range(0, len(secs), INDEX_SECTIONS_PER_FILE)]
        for fn, group in enumerate(files, 1):
            body = [f"# Parcel index — {sub} ({fn} of {len(files)})", "",
                    stamp(meta), "",
                    f"Development facts for parcels in the {sub} subarea of "
                    f"Issaquah that do not have a full parcel page. Each row "
                    f"carries that parcel's zone, lot area, existing units, net "
                    f"capacity, assessed land value, transit proximity, "
                    f"buildable share after site constraints, and the "
                    f"constraints that apply — enough to answer what can be "
                    f"built on a small parcel without a full page.",
                    "", "---", ""]
            for sec in group:
                first, last = sec.iloc[0], sec.iloc[-1]
                a0 = (address_forms(first.get("site_address")) or [None])[0]
                a1 = (address_forms(last.get("site_address")) or [None])[0]
                span = (f"{a0} through {a1}" if a0 and a1
                        else f"parcel {first['PIN']} through {last['PIN']}")
                zones = sorted(set(sec["ZONE"].dropna().astype(str)))
                zone_txt = (f" Zones in this section: {', '.join(zones[:6])}"
                            + ("." if len(zones) <= 6 else ", and others.")
                            if zones else "")
                body += [
                    f"## {sub} parcels — {span}", "", stamp(meta), "",
                    f"Parcel numbers {first['PIN']} through {last['PIN']} in the "
                    f"{sub} subarea of Issaquah.{zone_txt}", "",
                    md_table(_index_rows(sec, flags, meta.capacity_joined),
                             INDEX_COLS), "",
                ]
                body += [n + "\n" for n in
                         _index_section_notes(sec, flags, meta.capacity_joined)]
                body += ["---", ""]
            p = outdir / f"P2_index_{slug(sub)}_{fn:02d}.md"
            p.write_text("\n".join(body), encoding="utf-8")
            written.append(p)
    return written


# ──────────────────────────────────────────────────────────────────────────────
# P1 — dossiers
# ──────────────────────────────────────────────────────────────────────────────

def _capacity_prose(r: pd.Series, capacity_joined: bool) -> list[str]:
    if not capacity_joined:
        return ["**Development capacity.** Capacity for this parcel is not "
                "carried in this run. See the D2A headroom digest for capacity "
                "by zone and subarea, and apply the A1 zone standards matrix to "
                "the lot area above."]

    src = text(r.get("capacity_source"))
    if src == "development_agreement":
        return ["**Development capacity.** This parcel's development standards "
                "come from an executed development agreement, not from Title 18. "
                "Allowed units are not computed from the zone standards here — "
                "take them from the agreement and the D7 pipeline table. Treat "
                "any capacity figure derived from zoning alone as inapplicable "
                "to this parcel."]
    if src == "none_tract":
        return ["**Development capacity.** This parcel is a common tract — "
                "shared driveway, open space, or protected easement — with no "
                "building site. It carries no development capacity, and that is "
                "a property of the parcel rather than a low score. Do not list "
                "it as a redevelopment candidate."]
    if src == "none_non_residential":
        return ["**Development capacity.** This parcel's zone permits no "
                "residential use, so no housing capacity is computed. "
                "Non-residential development may still be permitted; see the "
                "zone standards in A1."]
    if src == "unkeyed":
        return ["**Development capacity.** The standards for this parcel's zone "
                "are not yet keyed into the A1 matrix, so capacity is not "
                "computed. That is a gap in the knowledge base, not a finding "
                "that the parcel has no capacity."]

    allowed, net = r.get("allowed_units_base"), r.get("net_capacity")
    cap_f, existing = r.get("capacity_factor"), r.get("existing_units")
    after = (None if _na(allowed) or _na(cap_f)
             else round(float(allowed) * float(cap_f)))

    out = [
        f"**Development capacity.** Under current zoning this parcel allows up "
        f"to {f_int(allowed)} dwelling units before site constraints and "
        f"{f_int(after)} after. With {f_int(existing)} existing housing units, "
        f"net capacity is {f_int(net)} additional units. Capacity is computed "
        f"from the A1 zone standards matrix applied to this parcel's lot "
        f"geometry; it is a zoning calculation, not a determination that a "
        f"project of that size would be permitted or would pencil."
    ]
    if truthy(r.get("near_transit")):
        out.append(
            "**Transit proximity.** This parcel lies within a quarter mile of a "
            "parcel treated as a major transit stop, which raises the state "
            "middle-housing floor from two units per lot to four. Issaquah has "
            "not adopted a major transit stop map, so that proximity is a "
            "working assumption and the four-unit floor on this parcel is an "
            "upper bound.")
    return out


def _constraint_prose(r: pd.Series, flags: Sequence[str]) -> str:
    hits = [c[2:] for c in flags if truthy(r.get(c))]
    enc, cost = r.get("encumbered_fraction"), r.get("cost_adder_total")
    months = r.get("predev_months_add")

    if not hits:
        return ("**Site constraints.** No mapped critical area, geologic hazard "
                "or aquifer overlay intersects this parcel. Absence of a mapped "
                "constraint is not a determination that none exists — the "
                "wetland layer is publisher-labeled incomplete, and no "
                "high-groundwater layer is published at all.")

    named = ", ".join(f"{legend_for(h)} (`{code_for(h)}`)" for h in hits)
    bits = [f"**Site constraints.** {len(hits)} mapped constraint"
            f"{'s' if len(hits) > 1 else ''} intersect this parcel: {named}."]
    if not _na(enc):
        bits.append(f"Together they encumber {f_pct(enc, 1)} of lot area, "
                    f"computed as a geometric union so overlapping layers are "
                    f"not counted twice.")
    if not _na(cost) and float(cost) > 0:
        bits.append(f"They add an estimated {f_money(cost)} of site cost for "
                    f"the tested product.")
    if not _na(months) and float(months) > 0:
        bits.append(f"They add about {f_num(months, 1)} months of "
                    f"predevelopment review.")
    bits.append("Constraint cost rates are unverified assumptions; treat these "
                "figures as estimates, not findings.")
    return " ".join(bits)


def render_dossier(r: pd.Series, meta: RunMeta, flags: Sequence[str],
                   peers: int) -> str:
    pin = str(r["PIN"])
    addr_txt = (address_forms(r.get("site_address")) or [None])[0]
    zone = r.get("ZONE") if not _na(r.get("ZONE")) else "an unassigned zone"
    sub = text(r.get("SUBAREA")) or "Unassigned"
    name = text(r.get("watchlist_name"))

    title = f"### Parcel {pin}"
    if addr_txt:
        title += f" — {addr_txt}"
    if name:
        title += f" ({name})"
    title += f", {sub}"

    where = f" at {addr_txt}" if addr_txt else ""
    lead = (
        f"Parcel number {pin}{where} is a {f_sf(r.get('lot_sf'))} lot in the "
        f"{sub} subarea of Issaquah, zoned {zone}. This section carries the "
        f"development capacity, site constraints and feasibility inputs for this "
        f"specific parcel — use it to answer what can be built here, what the "
        f"highest and best use is, and whether a project on this parcel pencils."
    )

    facts = [
        {"Measure": "Parcel number (PIN)", "Value": f"`{pin}`"},
        {"Measure": "Site address", "Value": addr_txt or MISSING},
        {"Measure": "Zone", "Value": str(zone)},
        {"Measure": "Subarea", "Value": sub},
        {"Measure": "Lot area", "Value": f_sf(r.get("lot_sf"))},
        {"Measure": "Existing housing units",
         "Value": f_int(r.get("existing_units"))
                  + (f" (basis: {text(r.get('units_basis'))})"
                     if text(r.get("units_basis")) else "")},
        {"Measure": "Year built", "Value": f_int(r.get("yr_built"))},
        {"Measure": "Assessed land value", "Value": f_money(r.get("ApprLandVal"))},
        {"Measure": "Assessed improvement value",
         "Value": f_money(r.get("ApprImpsVal"))},
        {"Measure": "Improvement-to-land ratio",
         "Value": f_num(r.get("i_l_ratio")) + (
             f" (redevelopment decile {f_int(r.get('redev_decile'))} of 10)"
             if not _na(r.get("redev_decile")) else "")},
        {"Measure": "Within a quarter mile of a major transit stop",
         "Value": "yes" if truthy(r.get("near_transit")) else "no"},
    ]
    if not _na(r.get("allowed_units_base")):
        facts += [
            {"Measure": "Allowed units before constraints",
             "Value": f_int(r.get("allowed_units_base"))},
            {"Measure": "Buildable share after constraints",
             "Value": f_pct(r.get("capacity_factor"), 0)},
            {"Measure": "Net capacity", "Value": f_int(r.get("net_capacity"))},
        ]

    parts = [title, "", stamp(meta), "", lead, "",
             f"**Also written:** {alias_line(r)}", "",
             md_table(facts, ["Measure", "Value"]), ""]
    parts += [p + "\n" for p in _capacity_prose(r, meta.capacity_joined)]
    parts += [_constraint_prose(r, flags), ""]

    if text(r.get("tested_product")):
        parts += [
            f"**Feasibility inputs.** The product tested on this parcel is "
            f"{text(r['tested_product'])}, with a modeled building footprint of "
            f"{f_sf(r.get('footprint_sf'))}. Site adders total "
            f"{f_money(r.get('cost_adder_total'))} and added predevelopment "
            f"time is {f_num(r.get('predev_months_add'), 1)} months. Apply the "
            f"A2 method spec Part 2 sequence to these inputs together with the "
            f"current cost, rent and rate constants. **This page carries no "
            f"verdict.** Whether a project here pencils depends on constants "
            f"that change faster than this file does, so the pencils / marginal "
            f"/ does-not-pencil call is computed when the question is asked, not "
            f"stored here.", ""]

    caveats = []
    if truthy(r.get("zone_split")):
        caveats.append("This parcel spans more than one zone; the zone above is "
                       "the one covering the largest share of its area.")
    if truthy(r.get("in_paa")):
        caveats.append("This parcel is in the Potential Annexation Area and is "
                       "governed by King County, not the City of Issaquah.")
    if text(r.get("units_basis")) == "presentuse_floor":
        caveats.append("Existing units were not available from assessor records "
                       "and a conservative minimum was applied from the property "
                       "class, so the unit count is an undercount and net "
                       "capacity is an upper bound.")
    if caveats:
        parts += ["**Caveats.** " + " ".join(caveats), ""]

    parts += [
        f"**Related.** {peers:,} other parcels share this zone and subarea; see "
        f"the {zone} zone profile for what is typical there, and the D2A "
        f"headroom digest for subarea totals. Figures on this page come from a "
        f"citizen-built model with no official standing and are not a permit "
        f"determination.", "", "---", ""]
    return "\n".join(parts)


def emit_dossiers(df: pd.DataFrame, meta: RunMeta, outdir: Path,
                  flags: Sequence[str]) -> list[Path]:
    peer_counts = df.groupby(["ZONE", "SUBAREA"], dropna=False).size()
    written = []
    d = df[df["tier"] == "dossier"].copy()
    sort_key = "site_address" if "site_address" in d.columns else "PIN"
    for sub, grp in d.groupby(d["SUBAREA"].fillna("Unassigned")):
        grp = grp.sort_values([sort_key, "PIN"])
        chunks = [grp.iloc[i:i + DOSSIERS_PER_FILE]
                  for i in range(0, len(grp), DOSSIERS_PER_FILE)]
        for n, chunk in enumerate(chunks, 1):
            body = [f"# Parcel pages — {sub} ({n} of {len(chunks)})", "",
                    stamp(meta), "",
                    f"One section per parcel in the {sub} subarea of Issaquah. "
                    f"Each section stands alone and carries that parcel's zone, "
                    f"lot area, existing units, development capacity, site "
                    f"constraints and feasibility inputs.", "", "---", ""]
            for _, r in chunk.iterrows():
                peers = int(peer_counts.get((r["ZONE"], r["SUBAREA"]), 1)) - 1
                body.append(render_dossier(r, meta, flags, max(peers, 0)))
            p = outdir / f"P1_dossiers_{slug(sub)}_{n:02d}.md"
            p.write_text("\n".join(body), encoding="utf-8")
            written.append(p)
    return written


# ──────────────────────────────────────────────────────────────────────────────
# P3 — address crosswalk
# ──────────────────────────────────────────────────────────────────────────────

def emit_addresses(df: pd.DataFrame, meta: RunMeta, outdir: Path) -> list[Path]:
    if "site_address" not in df.columns or df["site_address"].isna().all():
        return []
    base = df[df["site_address"].notna()].copy()

    # One row per (parcel, address) pair, so a corner parcel appears under both
    # streets. Drop the source columns BEFORE renaming onto that name — rename
    # duplicates rather than replaces, and a duplicated label makes every later
    # df["site_address"] return a DataFrame.
    base["_all"] = base.apply(
        lambda r: list(dict.fromkeys(
            [text(r["site_address"])]
            + [text(x) for x in (r.get("alt_addresses") or [])])), axis=1)
    d = (base.drop(columns=["site_address", "alt_addresses"], errors="ignore")
             .explode("_all").rename(columns={"_all": "site_address"}))
    d["site_address"] = d["site_address"].map(text)
    d = d[d["site_address"].str.len() > 0]
    d["_street"] = d["site_address"].map(street_name)
    d = d[d["_street"].notna()]

    sizes = d["_street"].value_counts()
    d["_group"] = np.where(d["_street"].map(sizes) >= MIN_STREET_SECTION,
                           d["_street"], "Other streets")

    written, buf, fileno = [], [], 1

    def flush(buf, fileno):
        p = outdir / f"P3_addresses_{fileno:02d}.md"
        p.write_text("\n".join([f"# Address to parcel number — Issaquah "
                                f"({fileno})", "", stamp(meta), "", *buf]),
                     encoding="utf-8")
        return p

    for st, grp in sorted(d.groupby("_group"), key=lambda t: str(t[0])):
        grp = grp.assign(_num=pd.to_numeric(
            grp["site_address"].astype(str).str.extract(r"^\s*(\d+)")[0],
            errors="coerce")).sort_values(["_num", "PIN"])
        rows = [{"Address": (address_forms(r["site_address"]) or [MISSING])[0],
                 "Parcel number": f"`{r['PIN']}`",
                 "Zone": r["ZONE"] if not _na(r.get("ZONE")) else MISSING,
                 "Subarea": text(r.get("SUBAREA")) or MISSING,
                 "Net capacity": f_int(r.get("net_capacity")),
                 "Detail": "full page" if r["tier"] == "dossier" else "index"}
                for _, r in grp.iterrows()]
        buf += [f"## Addresses on {st}", "", stamp(meta), "",
                f"Parcel numbers for addresses on {st} in Issaquah. Use this to "
                f"turn a street address into a parcel number, then look up that "
                f"parcel number for its development capacity and constraints.",
                "", md_table(rows, ["Address", "Parcel number", "Zone",
                                    "Subarea", "Net capacity", "Detail"]),
                "", "---", ""]
        if len(buf) > 400:
            written.append(flush(buf, fileno)); buf, fileno = [], fileno + 1
    if buf:
        written.append(flush(buf, fileno))
    return written


# ──────────────────────────────────────────────────────────────────────────────
# P4 — zone profiles
# ──────────────────────────────────────────────────────────────────────────────

def emit_zone_profiles(df: pd.DataFrame, meta: RunMeta, outdir: Path,
                       flags: Sequence[str]) -> Path:
    body = ["# Typical parcel by zone — Issaquah", "", stamp(meta), "",
            "What a typical parcel looks like in each Issaquah zone. Use this "
            "to reason about a parcel that has no page and no index row of its "
            "own: find its zone, read the profile, and say plainly that the "
            "figures are typical for the zone rather than measured for that "
            "parcel.", "", "---", ""]
    for zone, g in sorted(df.groupby(df["ZONE"].fillna("(unassigned)")),
                          key=lambda t: str(t[0])):
        inc = sorted(
            [(c[2:], float(g[c].fillna(False).mean())) for c in flags
             if c in g.columns and float(g[c].fillna(False).mean()) > 0.02],
            key=lambda t: -t[1])[:5]
        rows = [
            {"Measure": "Parcels in this zone", "Value": f_int(len(g))},
            {"Measure": "Median lot area", "Value": f_sf(g["lot_sf"].median())},
            {"Measure": "Median existing units",
             "Value": f_num(g["existing_units"].median(), 1)},
            {"Measure": "Median allowed units",
             "Value": f_num(g["allowed_units_base"].median(), 1)
             if "allowed_units_base" in g else MISSING},
            {"Measure": "Median net capacity",
             "Value": f_num(g["net_capacity"].median(), 1)
             if "net_capacity" in g else MISSING},
            {"Measure": "Median buildable share after constraints",
             "Value": f_pct(g["capacity_factor"].median(), 0)
             if "capacity_factor" in g else MISSING},
            {"Measure": "Median site cost adder",
             "Value": f_money(g["cost_adder_total"].median())
             if "cost_adder_total" in g else MISSING},
            {"Measure": "Median assessed land value",
             "Value": f_money(g["ApprLandVal"].median())},
            {"Measure": "Median improvement-to-land ratio",
             "Value": f_num(g["i_l_ratio"].median())},
            {"Measure": "Parcels within a quarter mile of a major transit stop",
             "Value": f_int(g.get("near_transit",
                                  pd.Series(dtype=bool)).map(truthy).sum())},
            {"Measure": "Parcels with a full page",
             "Value": f_int((g["tier"] == "dossier").sum())},
        ]
        prose = f"The {zone} zone contains {len(g):,} parcels in Issaquah. "
        if "tested_product" in g and not g["tested_product"].dropna().empty:
            prose += (f"The densest product tested in this zone is "
                      f"{text(g['tested_product'].mode().iat[0])}. ")
        if inc:
            prose += ("The constraints most often found on parcels in this zone "
                      "are " + ", ".join(
                          f"{legend_for(k)} ({v * 100:.0f}% of parcels)"
                          for k, v in inc) + ". ")
        prose += ("These are medians across the zone, not standards. The "
                  "governing standards are in the A1 zone standards matrix.")
        body += [f"## {zone} — typical parcel", "", stamp(meta), "", prose, "",
                 md_table(rows, ["Measure", "Value"]), "", "---", ""]
    p = outdir / "P4_zone_profiles.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# P5 — lookup guide
# ──────────────────────────────────────────────────────────────────────────────

def emit_lookup_guide(df: pd.DataFrame, meta: RunMeta, outdir: Path) -> Path:
    n = len(df)
    nd = int((df["tier"] == "dossier").sum())
    na = int(df["site_address"].notna().sum()) if "site_address" in df else 0
    nt = int(df.get("near_transit", pd.Series(dtype=bool)).map(truthy).sum())

    body = [
        "# How parcels are identified and what is covered", "", stamp(meta), "",
        "## Parcel coverage in this knowledge base", "", stamp(meta), "",
        f"This knowledge base covers {n:,} parcels inside Issaquah city limits, "
        f"as of {meta.parcel_vintage}. Every one of them is present. "
        f"{nd:,} have a full parcel page with capacity, constraints and "
        f"feasibility inputs written out; the remaining {n - nd:,} appear in the "
        f"parcel index, where each row carries zone, lot area, existing units, "
        f"net capacity, assessed land value, transit proximity, buildable share "
        f"after site constraints, and the constraints that apply. "
        f"{na:,} parcels have a site address recorded and {nt:,} lie within a "
        f"quarter mile of a parcel treated as a major transit stop.", "",
        "An index row is sufficient to answer most questions about a small "
        "parcel: what the zone allows, how many units of headroom remain, "
        "whether constraints materially reduce the buildable area or add cost, "
        "and how that added cost compares to the land value. A full page adds "
        "reasoning and caveats, not new measurements. A parcel without a full "
        "page is not a parcel without capacity — it is a parcel judged less "
        "likely to be asked about by name.", "",
        "Parcels in the Potential Annexation Area are flagged but are governed "
        "by King County, not the City of Issaquah. Parcels outside the city "
        "limits and its PAA are not covered at all.", "", "---", "",
        "## How to identify a parcel", "", stamp(meta), "",
        "An Issaquah parcel is identified by its 10-digit King County parcel "
        "number, made of a 6-digit Major and a 4-digit Minor. The same parcel "
        "may be written `1234567890`, `123456-7890`, or as Major 123456 Minor "
        "7890 — all three refer to the same parcel.", "",
        "A street address can be turned into a parcel number using the address "
        "crosswalk. A condominium complex is a single parcel whose Minor is "
        "`0000`; its individual units do not have parcels of their own.", "",
        "---", "",
        "## Reading a parcel's constraint summary", "", stamp(meta), "",
        "Index rows carry a packed site-constraint summary: estimated added "
        "site cost, added predevelopment months, and short codes for the "
        "constraints that apply. A row reading `$412k · +6mo · liq, slope, "
        "strm` means roughly $412,000 of added site cost, about six additional "
        "months of predevelopment review, and liquefaction, steep slope and "
        "stream buffer all present. `none mapped` means no constraint layer "
        "intersects the parcel. `no building site` means the parcel is a common "
        "tract.", "",
        "The full code list is in the parcel pages manifest, and each index "
        "section repeats the codes it uses. Constraint cost rates are "
        "unverified assumptions, so every cost and month figure is an estimate "
        "rather than a finding, and absence of a mapped constraint is not a "
        "determination that none exists.", "", "---", "",
        "## When a parcel is not found", "", stamp(meta), "",
        "If a parcel number or address does not appear anywhere in this "
        "knowledge base, say that it is not in the knowledge base rather than "
        "estimating from a neighboring parcel. Ask for the parcel number or the "
        "full street address. If only the zone is known, the zone profile gives "
        "what is typical there — label any such figure as typical for the zone, "
        "not measured for that parcel.", "", "---", "",
        "## What a parcel page or index row does not establish", "",
        stamp(meta), "",
        "These pages carry measured and modeled inputs: zone, lot area, "
        "existing units, allowed units, constraints, site cost adders and added "
        "review time. They do not carry a feasibility verdict, because whether "
        "a project pencils depends on cost, rent and interest-rate constants "
        "that change faster than the parcel data does — that calculation is run "
        "when the question is asked.", "",
        "No figure here is a permit determination, a legal opinion, or an "
        "appraisal. This is a citizen-built model with no official standing. "
        "Zoning and overlay boundaries come from the city's GIS services, which "
        "are a mapping convenience rather than the adopted zoning map.", "",
    ]
    p = outdir / "P5_parcel_lookup_guide.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────

def emit_manifest(df: pd.DataFrame, meta: RunMeta, outdir: Path,
                  files: dict[str, int], flags: Sequence[str]) -> Path:
    used = codes_present(df, flags)
    body = [
        "# Parcel pages — manifest, field dictionary and constraint codes", "",
        stamp(meta), "",
        md_table([{"Field": k, "Value": str(v)} for k, v in asdict(meta).items()],
                 ["Field", "Value"]), "", "---", "",
        "## Constraint code legend", "", stamp(meta), "",
        "Short codes used in the site-constraint summary on every parcel index "
        "row. The effect column states whether the constraint removes buildable "
        "area, adds cost, adds review time, or some combination — which is what "
        "the summary exists to let a reader judge without opening a full page.",
        "",
        md_table([{"Code": f"`{code_for(k)}`",
                   "Constraint": k.replace("_", " "),
                   "Effect": legend_for(k).split("—", 1)[-1].strip(),
                   "Present in this run": "yes" if k in used else "no"}
                  for k in CONSTRAINT_CODES],
                 ["Code", "Constraint", "Effect", "Present in this run"]),
        "", "---", "",
        "## Field dictionary", "", stamp(meta), "",
        md_table([{"Field": f"`{k}`", "Definition": v}
                  for k, v in FIELD_DEFINITIONS.items()],
                 ["Field", "Definition"]), "", "---", "",
        "## Which parcels get a full page", "", stamp(meta), "",
        f"{int((df['tier'] == 'dossier').sum()):,} of {len(df):,} parcels have a "
        f"full page. Selection is by a composite score favouring net capacity, "
        f"lot area, low improvement-to-land ratio, land value, development "
        f"agreement status, transit proximity and mixed-use zoning, plus every "
        f"parcel named in the watchlist. Common tracts and parcels in zones "
        f"permitting no residential use are ranked down.", "",
        "Every other parcel has an index row carrying zone, lot area, existing "
        "units, net capacity, land value, transit proximity, buildable share "
        "and a constraint summary. Pages are a retrieval budget: spending one "
        "on a parcel nobody asks about makes every other page harder to find.",
        "", "---", "",
        "## Files emitted", "", stamp(meta), "",
        md_table([{"File group": k, "Files": f_int(v)} for k, v in files.items()],
                 ["File group", "Files"]), "",
    ]
    p = outdir / "P_manifest.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Addresses
# ──────────────────────────────────────────────────────────────────────────────

def load_addresses(path: Path, master_pins: set[str]) -> tuple[pd.DataFrame, dict]:
    """Site addresses keyed to PIN, with one primary per parcel.

    Two PIN columns, and the distinction matters: ParcelPIN is the parcel the
    structure physically sits on; ParcelOwnerPIN is the parcel the owner holds.
    They diverge on condo complexes and lot-line-straddling buildings. Parcel
    pages are keyed to physical location, so ParcelPIN governs — but where a
    ParcelPIN resolves to nothing in the parcel master and the owner PIN does,
    the owner PIN is used, because that is the condo-unit case and the complex
    is the parcel carrying geometry.
    """
    a = pd.read_csv(path, low_memory=False)
    a.columns = [c.strip() for c in a.columns]
    stats = {"rows_in": len(a)}

    def flag(col):
        if col not in a.columns:
            return pd.Series(False, index=a.index)
        s = a[col]
        if s.dtype == object:
            return s.astype(str).str.strip().str.lower().isin(
                ("y", "yes", "true", "t", "1"))
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)

    def clean_pin(col):
        if col not in a.columns:
            return pd.Series(pd.NA, index=a.index, dtype="object")
        s = (a[col].astype(str).str.strip()
             .str.replace(r"\D", "", regex=True))
        return s.where(s.str.len() == 10)

    pin, alt = clean_pin(ADDRESS_PIN_FIELD), clean_pin(ADDRESS_PIN_FALLBACK)
    stats["pin_fields_disagree"] = int(
        (pin.notna() & alt.notna() & (pin != alt)).sum())
    filled = pin.isna() & alt.notna()
    stats["pin_from_owner_fallback"] = int(filled.sum())
    pin = pin.where(~filled, alt)
    orphan = pin.notna() & ~pin.isin(master_pins) & alt.isin(master_pins)
    stats["pin_rescued_by_owner"] = int(orphan.sum())
    pin = pin.where(~orphan, alt)
    a["PIN"] = pin
    stats["no_pin"] = int(a["PIN"].isna().sum())

    keep = a["PIN"].notna()
    if "Status" in a.columns:
        st = a["Status"].astype(str).str.strip().str.lower()
        bad = st.isin(("retired", "inactive", "deleted", "void", "pending"))
        stats["dropped_status"] = int((keep & bad).sum()); keep &= ~bad
    issue = flag("AddressHasIssue")
    stats["dropped_has_issue"] = int((keep & issue).sum()); keep &= ~issue
    meter = flag(ADDRESS_EXCLUDE_FLAGS[0]) | flag(ADDRESS_EXCLUDE_FLAGS[1])
    stats["dropped_meter"] = int((keep & meter).sum()); keep &= ~meter
    stats["flagged_outside_city"] = int((keep & flag("AddressOutsideCity")).sum())

    a = a[keep].copy()
    stats["rows_kept"] = len(a)

    for col in ("BaseAddress", "FullAddress", "Address"):
        if col in a.columns and a[col].notna().any():
            a["_addr"] = a[col].astype(str).str.strip()
            stats["address_field"] = col
            break
    else:
        raise ValueError(
            f"No usable address column. Expected one of BaseAddress, "
            f"FullAddress, Address. Available: {sorted(a.columns)[:30]}")
    a = a[a["_addr"].str.len() > 0]

    # Parcel address beats building beats unit. A 60-unit condo publishes 60
    # unit addresses; taking them all makes the heading unreadable and the
    # street crosswalk 60x too long.
    is_unit = flag("IsUnitAddress")
    a["_rank"] = np.select(
        [flag("IsParcelAddress"), flag("IsBldgAddress"), ~is_unit], [0, 1, 2], 3)
    stats["unit_addresses"] = int(is_unit.sum())

    primary = (a.sort_values(["PIN", "_rank", "_addr"]).drop_duplicates("PIN")
               [["PIN", "_addr"]].rename(columns={"_addr": "site_address"}))
    alts = (a.sort_values(["PIN", "_rank", "_addr"]).groupby("PIN")["_addr"]
            .apply(lambda s: list(dict.fromkeys(s))[:6])
            .rename("alt_addresses").reset_index())

    out = primary.merge(alts, on="PIN", how="left").set_index("PIN")
    stats["parcels_with_address"] = len(out)
    stats["multi_address_parcels"] = int(
        out["alt_addresses"].map(lambda v: len(v) if isinstance(v, list) else 0)
        .gt(1).sum())
    return out, stats


# ──────────────────────────────────────────────────────────────────────────────
# Load and orchestrate
# ──────────────────────────────────────────────────────────────────────────────

def load(master: Path, capacity: Path | None, addresses: Path | None,
         watchlist: Path | None) -> tuple[pd.DataFrame, bool, bool]:
    df = pd.read_parquet(master)
    if hasattr(df, "geometry"):
        df = pd.DataFrame(df.drop(columns="geometry"))
    miss = [c for c in MASTER_REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"parcels_master is missing {miss}")
    df["PIN"] = df["PIN"].astype(str).str.strip().str.zfill(10)

    has_cap = False
    if capacity and Path(capacity).exists():
        cap = pd.read_parquet(capacity)
        cap["PIN"] = cap["PIN"].astype(str).str.strip().str.zfill(10)
        cols = ["PIN"] + [c for c in cap.columns
                          if (c in CAPACITY_FIELDS or c.startswith("c_"))
                          and c not in df.columns]
        df = df.merge(cap[cols], on="PIN", how="left")
        has_cap = "capacity_factor" in df.columns
    if not has_cap:
        print("  WARNING: capacity not joined. Index rows will carry zone, lot "
              "area, units and land value but NO capacity, buildable share or "
              "constraint summary — which is most of what a 'what can be built "
              "here' question needs. Run s2_capacity.py first.")

    # Known booleans normalized once, so a later `if r[...]` cannot reintroduce
    # the NA-ambiguity crash. in_development_agreement stays out: NA there means
    # the DA layer did not fetch, which is not the same as "not in a DA".
    for c in (["zone_split", "in_paa", "near_transit",
               "near_transit_centroid_basis", "condo_value_rollup"]
              + [c for c in df.columns if c.startswith("c_")]):
        if c in df.columns:
            df[c] = df[c].map(truthy).astype(bool)

    has_addr = False
    df["alt_addresses"] = pd.Series([[] for _ in range(len(df))], index=df.index)
    for c in ("site_address", "Address", "SITUS_ADDRESS", "ADDR_FULL"):
        if c in df.columns and df[c].notna().any():
            df["site_address"] = df[c]; has_addr = True; break
    else:
        df["site_address"] = pd.NA

    if addresses and Path(addresses).exists():
        addr, astats = load_addresses(Path(addresses), set(df["PIN"]))
        df = df.join(addr, on="PIN", rsuffix="_src")
        if "site_address_src" in df.columns:
            df["site_address"] = df["site_address"].fillna(
                df.pop("site_address_src"))
        if "alt_addresses_src" in df.columns:
            src = df.pop("alt_addresses_src")
            df["alt_addresses"] = [b if isinstance(b, list) and b else
                                   (a if isinstance(a, list) else [])
                                   for a, b in zip(src, df["alt_addresses"])]
        df["alt_addresses"] = df["alt_addresses"].map(
            lambda v: v if isinstance(v, list) else [])
        has_addr = df["site_address"].notna().any()

        matched = int(df["site_address"].notna().sum())
        print(f"  addresses: {astats['rows_in']:,} points -> "
              f"{astats['rows_kept']:,} after quality filters -> "
              f"{astats['parcels_with_address']:,} parcels; {matched:,} of "
              f"{len(df):,} parcels matched ({matched / max(len(df), 1):.1%})")
        for k in ("dropped_status", "dropped_has_issue", "dropped_meter",
                  "pin_from_owner_fallback", "pin_rescued_by_owner",
                  "pin_fields_disagree", "flagged_outside_city",
                  "unit_addresses", "multi_address_parcels"):
            if astats.get(k):
                print(f"    {k}: {astats[k]:,}")
        if matched / max(len(df), 1) < 0.5:
            print("    WARNING: under half of parcels matched an address. "
                  "People ask about parcels by address far more often than by "
                  "parcel number — check the PIN format before judging "
                  "retrieval quality.")
    elif not has_addr:
        print("  WARNING: no site addresses joined. The address crosswalk "
              "cannot be built and every parcel page falls back to a bare "
              "parcel number in its heading.")

    df["watchlist_name"] = pd.NA
    if watchlist and Path(watchlist).exists():
        w = pd.read_csv(watchlist, dtype={"PIN": str})
        w["PIN"] = w["PIN"].str.strip().str.zfill(10)
        df["watchlist_name"] = df["PIN"].map(dict(zip(w["PIN"], w["name"])))
        print(f"  {int(df['watchlist_name'].notna().sum()):,} parcels named in "
              f"the watchlist — each forced to a full page")

    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        raise ValueError(
            f"Duplicate column labels after joins: {dupes}. A duplicated label "
            f"makes df[name] return a DataFrame, and every Series method on it "
            f"fails somewhere far from the join that caused it.")
    return df, has_cap, has_addr


def main() -> None:
    p = argparse.ArgumentParser(description="Emit parcel pages for the KB.")
    p.add_argument("--master", default="out/parcels_master.parquet")
    p.add_argument("--capacity", default="out/parcels_capacity.parquet")
    p.add_argument("--addresses", default="data/parcel_addresses.csv")
    p.add_argument("--watchlist", default="kb/parcel_watchlist.csv")
    p.add_argument("--outdir", default="out/kb")
    p.add_argument("--run-id", required=True)
    p.add_argument("--parcel-vintage", required=True)
    p.add_argument("--zoning-vintage", required=True)
    p.add_argument("--a1-version", required=True)
    p.add_argument("--budget", type=int, default=DEFAULT_DOSSIER_BUDGET)
    a = p.parse_args()

    df, has_cap, has_addr = load(Path(a.master), a.capacity, a.addresses,
                                 a.watchlist)
    meta = RunMeta(a.run_id, a.parcel_vintage, a.zoning_vintage, a.a1_version,
                   has_cap, has_addr)
    df = assign_tiers(df, a.budget)
    flags = flag_cols(df)

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    d1 = emit_dossiers(df, meta, out, flags)
    d2 = emit_index(df, meta, out, flags)
    d3 = emit_addresses(df, meta, out)
    d4 = [emit_zone_profiles(df, meta, out, flags)]
    d5 = [emit_lookup_guide(df, meta, out)]
    mf = emit_manifest(df, meta, out,
                       {"P1 parcel pages": len(d1), "P2 parcel index": len(d2),
                        "P3 address crosswalk": len(d3),
                        "P4 zone profiles": len(d4),
                        "P5 lookup guide": len(d5), "Manifest": 1}, flags)

    all_files = [*d1, *d2, *d3, *d4, *d5, mf]
    chars = sum(len(f.read_text(encoding="utf-8")) for f in all_files)
    idx_chars = sum(len(f.read_text(encoding="utf-8")) for f in d2)
    n_idx = int((df["tier"] == "index").sum())

    big_loss = int((pd.to_numeric(df.get("capacity_factor"), errors="coerce")
                    < (1 - SIGNIFICANT_AREA_LOSS)).sum()) if has_cap else 0
    big_cost = int((pd.to_numeric(df.get("cost_adder_total"), errors="coerce")
                    >= SIGNIFICANT_COST_ADDER).sum()) if has_cap else 0

    print(f"\n  {len(all_files)} files, {chars:,} characters")
    print(f"  {int((df['tier'] == 'dossier').sum()):,} full pages · "
          f"{n_idx:,} index rows "
          f"({idx_chars / max(n_idx, 1):.0f} chars per index row)")
    if has_cap:
        print(f"  {big_loss:,} parcels lose >{SIGNIFICANT_AREA_LOSS:.0%} of "
              f"allowed units to constraints · {big_cost:,} carry a site adder "
              f"of {SIGNIFICANT_COST_ADDER:,}+")
    (out / "P_summary.json").write_text(json.dumps({
        "files": len(all_files), "characters": chars,
        "index_characters": idx_chars,
        "dossiers": int((df["tier"] == "dossier").sum()), "index_rows": n_idx,
        "addresses": int(df["site_address"].notna().sum()),
        "transit_parcels": int(df.get("near_transit",
                                      pd.Series(dtype=bool)).map(truthy).sum()),
        "capacity_joined": has_cap,
        "parcels_significant_area_loss": big_loss,
        "parcels_significant_cost_adder": big_cost,
        "constraint_codes_present": codes_present(df, flags),
    }, indent=2))


if __name__ == "__main__":
    main()
