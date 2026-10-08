"""
s3_parcel_pages.py — emit retrieval-optimized parcel pages for the knowledge base.

Reads   out/parcels_master.parquet        (s1_parcel_master.py)        REQUIRED
        out/parcels_capacity.parquet      (s2_capacity.py)             optional
        kb/parcel_watchlist.csv           hand-maintained names        optional
        data/parcel_addresses.csv         PIN -> site address          optional

Writes  out/kb/P1_dossiers_<subarea>_<nn>.md
        out/kb/P2_index_<subarea>_<nn>.md
        out/kb/P3_addresses_<street>.md
        out/kb/P4_zone_profiles.md
        out/kb/P5_parcel_lookup_guide.md
        out/kb/P_manifest.md

Design rules, each reversing a failure of the sharded parcel table:

  1. One SECTION per parcel, not one row. A retrieved section arrives by
     itself, so every section restates the PIN, the address, the zone, the
     as-of date, and enough prose to be semantically findable.
  2. Facts as statements with their conditions attached, not bare cells.
     "This parcel allows up to 4 dwelling units" beats a column headed
     "allowed_units" that arrives detached from its header.
  3. Aliases in the heading AND the first sentence. PINs get written three
     ways and addresses two, because people paste what they have.
  4. Tiered by likelihood of being asked about. Full pages for every parcel
     would be a document dump, and volume dilutes retrieval.
  5. No verdicts. This file carries feasibility INPUTS; the pencils/marginal
     /no call is computed at answer time from A2 Part 2, whose constants
     change faster than the parcel table does.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

MISSING = "—"
SQFT_PER_ACRE = 43_560.0

# Retrieval budgets. Past these, sections stop being findable because they
# compete with their own siblings.
DEFAULT_DOSSIER_BUDGET = 1_200      # parcels earning a full page
DOSSIERS_PER_FILE = 60
INDEX_ROWS_PER_SECTION = 60
INDEX_SECTIONS_PER_FILE = 12
MIN_STREET_SECTION = 3              # streets with fewer parcels fold into "other"


# ──────────────────────────────────────────────────────────────────────────────
# Input contract
# ──────────────────────────────────────────────────────────────────────────────

MASTER_REQUIRED = ["PIN", "ZONE", "SUBAREA", "lot_sf", "existing_units",
                   "ApprLandVal", "ApprImpsVal", "i_l_ratio"]

# Present only when parcels_capacity.parquet is joined. Without them a dossier
# can describe a parcel but cannot state its capacity — which is most of the
# value, so the script says so loudly rather than quietly emitting less.
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
    "allowed_units_base": "Allowed units under A1 zone standards, before constraints",
    "capacity_factor": "Multiplier on allowed units from site constraints, 0.0-1.0",
    "net_capacity": "Allowed units after constraints minus existing housing units",
    "capacity_source": "code | development_agreement | none_tract | "
                       "none_non_residential | unkeyed",
    "cost_adder_total": "Site cost adders for the tested product, dollars",
    "predev_months_add": "Added predevelopment months from constraint review",
    "i_l_ratio": "Improvement value divided by land value",
    "redev_decile": "1 = lowest improvement-to-land ratio, the strongest "
                    "redevelopment condition signal",
}

ZONE_NAMES = {}   # optional: loaded from A1 if the matrix carries zone_name


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
    """Provenance line. Repeated in EVERY section, not once per file — a
    retrieved section arrives without the file header above it."""
    return (f"*Parcel page · run `{m.run_id}` · parcels as of {m.parcel_vintage} "
            f"· zoning as of {m.zoning_vintage} · A1 {m.a1_version}*")


# ──────────────────────────────────────────────────────────────────────────────
# Formatting — missing renders as an em dash, never as zero
# ──────────────────────────────────────────────────────────────────────────────


# ── NA-safe scalars ───────────────────────────────────────────────────────
# pandas NA raises TypeError on bool(), so every `if value:` over a column
# that can be missing is a latent crash rather than a missing-value path.
# Ordering matters too: `not value or _na(value)` evaluates the bool first
# and raises before the guard runs.

def _na(v) -> bool:
    if v is None:
        return True
    try:
        res = pd.isna(v)
    except (TypeError, ValueError):
        return False
    return bool(res) if isinstance(res, (bool, np.bool_)) else False


def truthy(v) -> bool:
    """Boolean-ish value that may be NA. Missing reads as False, not a crash."""
    if _na(v):
        return False
    if isinstance(v, str):
        return v.strip().lower() not in ("", "false", "nan", "none", "0")
    try:
        return bool(v)
    except (TypeError, ValueError):
        return False


def text(v) -> str:
    """String-ish value that may be NA. Missing reads as the empty string."""
    return "" if _na(v) else str(v).strip()


def f_int(v) -> str:
    return MISSING if _na(v) else f"{int(round(float(v))):,}"


def f_money(v) -> str:
    return MISSING if _na(v) else f"${float(v):,.0f}"


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
    from a staff report, and from a spreadsheet produces three strings."""
    p = str(pin).strip().zfill(10)
    return [p, f"{p[:6]}-{p[6:]}", f"Major {p[:6]} Minor {p[6:]}"]


def address_forms(addr) -> list[str]:
    a = " ".join(text(addr).split()).title()
    if not a:
        return []
    a = " ".join(str(addr).split()).title()
    expand = {" St": " Street", " Ave": " Avenue", " Rd": " Road",
              " Blvd": " Boulevard", " Dr": " Drive", " Ln": " Lane",
              " Pl": " Place", " Ct": " Court", " Way": " Way",
              " N": " North", " S": " South", " E": " East", " W": " West",
              " Ne": " Northeast", " Nw": " Northwest",
              " Se": " Southeast", " Sw": " Southwest"}
    long = a
    for k, v in expand.items():
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
    """Rank parcels by how likely someone is to ask about this one specifically.

    Not by size, and not by capacity alone. A full page is a retrieval slot:
    spending one on a parcel nobody asks about makes every other page harder
    to find.
    """
    def z(col, log=False):
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        v = pd.to_numeric(df[col], errors="coerce")
        if log:
            v = np.log1p(v.clip(lower=0))
        rank = v.rank(pct=True)
        return rank.fillna(0.0)

    s = (
        3.0 * z("net_capacity", log=True)          # the development question
        + 2.0 * z("lot_sf", log=True)              # big sites get asked about
        + 1.5 * (11 - pd.to_numeric(df.get("redev_decile"), errors="coerce")
                 .fillna(11)) / 10.0               # low I/L = turnover candidate
        + 1.0 * z("ApprLandVal", log=True)
    )
    # Categorical boosts: parcels that generate questions regardless of size.
    s += 2.0 * df.get("in_development_agreement", pd.Series(False, index=df.index)) \
        .fillna(False).astype(float)
    s += 1.5 * df.get("near_transit", pd.Series(False, index=df.index)) \
        .fillna(False).astype(float)
    s += 1.0 * df["ZONE"].astype(str).str.startswith(("MU", "UC", "VR", "CBD")) \
        .astype(float)
    s += 1.0 * (pd.to_numeric(df.get("existing_units"), errors="coerce")
                .fillna(0) >= 10).astype(float)
    # Tracts and non-residential parcels still get pages when large, but they
    # rank below anything with real capacity.
    s -= 2.0 * df.get("capacity_source", pd.Series("", index=df.index)) \
        .isin(["none_tract", "none_non_residential"]).astype(float)
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
# P1 — dossiers
# ──────────────────────────────────────────────────────────────────────────────

def _capacity_prose(r: pd.Series, has_cap: bool) -> list[str]:
    if not has_cap:
        return ["**Development capacity.** Capacity for this parcel is not "
                "carried on this page. See the D2A headroom digest for capacity "
                "by zone and subarea, and apply the A1 zone standards matrix to "
                "the lot area above."]

    src = str(r.get("capacity_source") or "")
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
                "computed. This is a gap in the knowledge base, not a finding "
                "that the parcel has no capacity."]

    allowed = r.get("allowed_units_base")
    net = r.get("net_capacity")
    cap_f = r.get("capacity_factor")
    existing = r.get("existing_units")
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
            "**Middle housing.** This parcel lies within a quarter mile of a "
            "parcel treated as a major transit stop, which raises the state "
            "middle-housing floor from two units per lot to four. Issaquah has "
            "not adopted a major transit stop map, so that proximity is a "
            "working assumption and the four-unit floor on this parcel is an "
            "upper bound.")
    return out


def _constraint_prose(r: pd.Series, flags: Sequence[str]) -> str:
    hits = [c[2:] for c in flags if truthy(r.get(c))]
    enc = r.get("encumbered_fraction")
    cost = r.get("cost_adder_total")
    months = r.get("predev_months_add")

    if not hits:
        return ("**Site constraints.** No mapped critical area, geologic hazard "
                "or aquifer overlay intersects this parcel. Absence of a mapped "
                "constraint is not a determination that none exists — the "
                "wetland layer is publisher-labeled incomplete, and no "
                "high-groundwater layer is published at all.")

    named = ", ".join(f"`{h}`" for h in hits)
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
    addr = r.get("site_address")
    addr_txt = (address_forms(addr) or [None])[0]
    zone = r.get("ZONE") if not _na(r.get("ZONE")) else "an unassigned zone"
    zone_full = (f"{zone} ({ZONE_NAMES[zone]})" if zone in ZONE_NAMES else zone)
    sub = r.get("SUBAREA") or "Unassigned"
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
        f"{sub} subarea of Issaquah, zoned {zone_full}. This section carries the "
        f"development capacity, site constraints and feasibility inputs for this "
        f"specific parcel — use it to answer what can be built here, what the "
        f"highest and best use is, and whether a project on this parcel pencils."
    )

    facts = [
        {"Measure": "Parcel number (PIN)", "Value": f"`{pin}`"},
        {"Measure": "Site address", "Value": addr_txt or MISSING},
        {"Measure": "Zone", "Value": str(zone)},
        {"Measure": "Subarea", "Value": str(sub)},
        {"Measure": "Lot area", "Value": f_sf(r.get("lot_sf"))},
        {"Measure": "Existing housing units",
         "Value": f"{f_int(r.get('existing_units'))}"
                  + (f" (basis: {r['units_basis']})"
                     if not _na(r.get("units_basis")) else "")},
        {"Measure": "Year built", "Value": f_int(r.get("yr_built"))},
        {"Measure": "Assessed land value", "Value": f_money(r.get("ApprLandVal"))},
        {"Measure": "Assessed improvement value",
         "Value": f_money(r.get("ApprImpsVal"))},
        {"Measure": "Improvement-to-land ratio",
         "Value": f_num(r.get("i_l_ratio")) + (
             f" (redevelopment decile {f_int(r.get('redev_decile'))} of 10)"
             if not _na(r.get("redev_decile")) else "")},
    ]
    if not _na(r.get("allowed_units_base")):
        facts += [
            {"Measure": "Allowed units before constraints",
             "Value": f_int(r.get("allowed_units_base"))},
            {"Measure": "Net capacity", "Value": f_int(r.get("net_capacity"))},
        ]

    parts = [title, "", stamp(meta), "", lead, "",
             f"**Also written:** {alias_line(r)}", "",
             md_table(facts, ["Measure", "Value"]), ""]

    parts += [p + "\n" for p in _capacity_prose(r, meta.capacity_joined)]
    parts += [_constraint_prose(r, flags), ""]

    if not _na(r.get("tested_product")):
        parts += [
            f"**Feasibility inputs.** The product tested on this parcel is "
            f"{r['tested_product']}, with a modeled building footprint of "
            f"{f_sf(r.get('footprint_sf'))}. Site adders total "
            f"{f_money(r.get('cost_adder_total'))} and added predevelopment "
            f"time is {f_num(r.get('predev_months_add'), 1)} months. Apply the "
            f"A2 method spec Part 2 sequence to these inputs together with the "
            f"current cost, rent and rate constants. **This page carries no "
            f"verdict.** Whether a project here pencils depends on constants "
            f"that change faster than this file does, so the pencils / marginal "
            f"/ does-not-pencil call is computed at the time the question is "
            f"asked, not stored here.", ""]

    flags_txt = []
    if truthy(r.get("zone_split")):
        flags_txt.append("This parcel spans more than one zone; the zone above "
                         "is the one covering the largest share of its area.")
    if truthy(r.get("in_paa")):
        flags_txt.append("This parcel is in the Potential Annexation Area and "
                         "is governed by King County, not the City of Issaquah.")
    if str(r.get("units_basis")) == "presentuse_floor":
        flags_txt.append("Existing units were not available from assessor "
                         "records and a conservative minimum was applied from "
                         "the property class, so the unit count is an "
                         "undercount and net capacity is an upper bound.")
    if flags_txt:
        parts += ["**Caveats.** " + " ".join(flags_txt), ""]

    parts += [
        f"**Related.** {peers:,} other parcels share this zone and subarea; see "
        f"the {zone} zone profile for what is typical there, and the D2A "
        f"headroom digest for subarea totals. Figures on this page are from a "
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
# P2 — compact index for every parcel without a dossier
# ──────────────────────────────────────────────────────────────────────────────

INDEX_COLS = ["PIN", "Address", "Zone", "Lot area", "Existing units",
              "Net capacity", "I/L ratio"]


def _index_rows(g: pd.DataFrame) -> list[dict]:
    return [{
        "PIN": f"`{r['PIN']}`",
        "Address": (address_forms(r.get("site_address")) or [MISSING])[0],
        "Zone": r["ZONE"] if not _na(r.get("ZONE")) else MISSING,
        "Lot area": f_int(r.get("lot_sf")) + " sq ft" if not _na(r.get("lot_sf"))
                    else MISSING,
        "Existing units": f_int(r.get("existing_units")),
        "Net capacity": f_int(r.get("net_capacity")),
        "I/L ratio": f_num(r.get("i_l_ratio")),
    } for _, r in g.iterrows()]


def emit_index(df: pd.DataFrame, meta: RunMeta, outdir: Path) -> list[Path]:
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
                    f"Summary facts for parcels in the {sub} subarea that do "
                    f"not have a full parcel page. Look up a parcel number here "
                    f"to confirm it exists and get its zone, lot area and "
                    f"capacity; a full page exists only for parcels most likely "
                    f"to be asked about.", "", "---", ""]
            for sec in group:
                first, last = sec.iloc[0], sec.iloc[-1]
                a0 = (address_forms(first.get("site_address")) or [None])[0]
                a1 = (address_forms(last.get("site_address")) or [None])[0]
                span = (f"{a0} through {a1}" if a0 and a1
                        else f"PIN {first['PIN']} through {last['PIN']}")
                body += [f"## {sub} parcels — {span}", "", stamp(meta), "",
                         f"Parcel numbers {first['PIN']} through "
                         f"{last['PIN']} in the {sub} subarea.", "",
                         md_table(_index_rows(sec), INDEX_COLS), "", "---", ""]
            p = outdir / f"P2_index_{slug(sub)}_{fn:02d}.md"
            p.write_text("\n".join(body), encoding="utf-8")
            written.append(p)
    return written


# ──────────────────────────────────────────────────────────────────────────────
# P3 — address crosswalk, the file that makes natural-language lookup work
# ──────────────────────────────────────────────────────────────────────────────

def emit_addresses(df: pd.DataFrame, meta: RunMeta, outdir: Path) -> list[Path]:
    if "site_address" not in df.columns or df["site_address"].isna().all():
        return []

    base = df[df["site_address"].notna()].copy()

    # One row per (parcel, address) pair, so a corner parcel appears under
    # both streets. Drop the source columns BEFORE renaming onto that name —
    # rename() does not replace, it duplicates, and a duplicated label makes
    # every later df["site_address"] return a DataFrame.
    base["_all"] = base.apply(
        lambda r: list(dict.fromkeys(
            [text(r["site_address"])]
            + [text(x) for x in (r.get("alt_addresses") or [])])), axis=1)

    d = (base.drop(columns=["site_address", "alt_addresses"], errors="ignore")
             .explode("_all")
             .rename(columns={"_all": "site_address"}))
    d["site_address"] = d["site_address"].map(text)
    d = d[d["site_address"].str.len() > 0]
    d["_street"] = d["site_address"].map(street_name)
    d = d[d["_street"].notna()]

    sizes = d["_street"].value_counts()
    d["_group"] = np.where(d["_street"].map(sizes) >= MIN_STREET_SECTION,
                           d["_street"], "Other streets")

    written, buf, fileno = [], [], 1
    for st, grp in sorted(d.groupby("_group"), key=lambda t: str(t[0])):
        grp = grp.assign(_num=pd.to_numeric(
            grp["site_address"].astype(str).str.extract(r"^\s*(\d+)")[0],
            errors="coerce")).sort_values(["_num", "PIN"])
        rows = [{"Address": (address_forms(r["site_address"]) or [MISSING])[0],
                 "Parcel number": f"`{r['PIN']}`",
                 "Zone": r["ZONE"] if not _na(r.get("ZONE")) else MISSING,
                 "Subarea": r.get("SUBAREA") or MISSING,
                 "Full page": "yes" if r["tier"] == "dossier" else "index only"}
                for _, r in grp.iterrows()]
        buf += [f"## Addresses on {st}", "", stamp(meta), "",
                f"Parcel numbers for addresses on {st} in Issaquah. Use this to "
                f"turn a street address into a parcel number, then look up that "
                f"parcel number for its development capacity.", "",
                md_table(rows, ["Address", "Parcel number", "Zone", "Subarea",
                                "Full page"]), "", "---", ""]
        if len(buf) > 400:
            p = outdir / f"P3_addresses_{fileno:02d}.md"
            p.write_text("\n".join([f"# Address to parcel number — Issaquah "
                                    f"({fileno})", "", stamp(meta), "", *buf]),
                         encoding="utf-8")
            written.append(p); buf, fileno = [], fileno + 1
    if buf:
        p = outdir / f"P3_addresses_{fileno:02d}.md"
        p.write_text("\n".join([f"# Address to parcel number — Issaquah "
                                f"({fileno})", "", stamp(meta), "", *buf]),
                     encoding="utf-8")
        written.append(p)
    return written


# ──────────────────────────────────────────────────────────────────────────────
# P4 — zone profiles, the fallback for parcels with no page
# ──────────────────────────────────────────────────────────────────────────────

def emit_zone_profiles(df: pd.DataFrame, meta: RunMeta, outdir: Path,
                       flags: Sequence[str]) -> Path:
    body = ["# Typical parcel by zone — Issaquah", "", stamp(meta), "",
            "What a typical parcel looks like in each Issaquah zone. Use this "
            "to reason about a parcel that has no page of its own: find its "
            "zone, read the profile, and say plainly that the figures are "
            "typical for the zone rather than measured for that parcel.",
            "", "---", ""]
    for zone, g in sorted(df.groupby(df["ZONE"].fillna("(unassigned)")),
                          key=lambda t: str(t[0])):
        inc = [(c[2:], float(g[c].fillna(False).mean()))
               for c in flags if c in g.columns]
        inc = sorted([x for x in inc if x[1] > 0.02],
                     key=lambda t: -t[1])[:4]
        rows = [
            {"Measure": "Parcels in this zone", "Value": f_int(len(g))},
            {"Measure": "Median lot area",
             "Value": f_sf(g["lot_sf"].median())},
            {"Measure": "Median existing units",
             "Value": f_num(g["existing_units"].median(), 1)},
            {"Measure": "Median allowed units",
             "Value": f_num(g["allowed_units_base"].median(), 1)
             if "allowed_units_base" in g else MISSING},
            {"Measure": "Median net capacity",
             "Value": f_num(g["net_capacity"].median(), 1)
             if "net_capacity" in g else MISSING},
            {"Measure": "Median assessed land value",
             "Value": f_money(g["ApprLandVal"].median())},
            {"Measure": "Median improvement-to-land ratio",
             "Value": f_num(g["i_l_ratio"].median())},
            {"Measure": "Parcels with a full page",
             "Value": f_int((g["tier"] == "dossier").sum())},
        ]
        prod = text(g["tested_product"].mode().iat[0]
                if "tested_product" in g and not g["tested_product"].dropna().empty
                else "")
        prose = (f"The {zone} zone contains {len(g):,} parcels in Issaquah. ")
        if prod:
            prose += (f"The densest product tested in this zone is {prod}. ")
        if inc:
            prose += ("The constraints most often found on parcels in this zone "
                      "are " + ", ".join(f"{k} ({v*100:.0f}% of parcels)"
                                         for k, v in inc) + ". ")
        prose += ("These are medians across the zone, not standards. The "
                  "governing standards are in the A1 zone standards matrix.")
        body += [f"## {zone} — typical parcel", "", stamp(meta), "", prose, "",
                 md_table(rows, ["Measure", "Value"]), "", "---", ""]
    p = outdir / "P4_zone_profiles.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# P5 — lookup guide and coverage statement
# ──────────────────────────────────────────────────────────────────────────────

def emit_lookup_guide(df: pd.DataFrame, meta: RunMeta, outdir: Path) -> Path:
    n = len(df)
    nd = int((df["tier"] == "dossier").sum())
    na = int(df["site_address"].notna().sum()) if "site_address" in df else 0
    body = [
        "# How parcels are identified and what is covered", "", stamp(meta), "",
        "## Parcel coverage in this knowledge base", "", stamp(meta), "",
        f"This knowledge base covers {n:,} parcels inside Issaquah city limits, "
        f"as of {meta.parcel_vintage}. Of those, {nd:,} have a full parcel page "
        f"carrying capacity, constraints and feasibility inputs; the remaining "
        f"{n - nd:,} appear in the parcel index with summary facts only. "
        f"{na:,} parcels have a site address recorded.", "",
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
        "## When a parcel is not found", "", stamp(meta), "",
        "If a parcel number or address does not appear anywhere in this "
        "knowledge base, say that it is not in the knowledge base rather than "
        "estimating from a neighboring parcel. Ask for the parcel number or the "
        "full street address. If only the zone is known, the zone profile gives "
        "what is typical there — label any such figure as typical for the zone, "
        "not measured for that parcel.", "", "---", "",
        "## What a parcel page does and does not establish", "", stamp(meta), "",
        "A parcel page carries measured and modeled inputs: zone, lot area, "
        "existing units, allowed units, constraints, site cost adders and added "
        "review time. It does not carry a feasibility verdict, because whether "
        "a project pencils depends on cost, rent and interest-rate constants "
        "that change faster than the parcel data does — that calculation is run "
        "when the question is asked.", "",
        "No figure on a parcel page is a permit determination, a legal opinion, "
        "or an appraisal. This is a citizen-built model with no official "
        "standing. Zoning and overlay boundaries come from the city's GIS "
        "services, which are a mapping convenience rather than the adopted "
        "zoning map.", "",
    ]
    p = outdir / "P5_parcel_lookup_guide.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────

def emit_manifest(df: pd.DataFrame, meta: RunMeta, outdir: Path,
                  files: dict[str, int]) -> Path:
    body = [
        "# Parcel pages — manifest, field dictionary and selection rule", "",
        stamp(meta), "",
        md_table([{"Field": k, "Value": str(v)} for k, v in asdict(meta).items()],
                 ["Field", "Value"]), "", "---", "",
        "## Field dictionary", "", stamp(meta), "",
        md_table([{"Field": f"`{k}`", "Definition": v}
                  for k, v in FIELD_DEFINITIONS.items()], ["Field", "Definition"]),
        "", "---", "",
        "## Which parcels get a full page", "", stamp(meta), "",
        f"{int((df['tier']=='dossier').sum()):,} of {len(df):,} parcels have a "
        f"full page. Selection is by a composite score favouring net capacity, "
        f"lot area, low improvement-to-land ratio, land value, development "
        f"agreement status, transit proximity and mixed-use zoning, plus every "
        f"parcel named in the watchlist. Common tracts and parcels in zones "
        f"permitting no residential use are ranked down.", "",
        "A parcel without a full page is not a parcel without capacity. It is a "
        "parcel judged less likely to be asked about by name. Pages are a "
        "retrieval budget: spending one on a parcel nobody asks about makes "
        "every other page harder to find.", "", "---", "",
        "## Files emitted", "", stamp(meta), "",
        md_table([{"File group": k, "Files": f_int(v)}
                  for k, v in files.items()], ["File group", "Files"]), "",
    ]
    p = outdir / "P_manifest.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p



# Issaquah Open Data address points. Two PIN columns, and the distinction
# matters: ParcelPIN is the parcel the structure physically sits on;
# ParcelOwnerPIN is the parcel the owner holds. They diverge on condo
# complexes and lot-line-straddling buildings. Parcel pages are keyed to
# physical location, so ParcelPIN governs.
ADDRESS_PIN_FIELD = "ParcelPIN"
ADDRESS_PIN_FALLBACK = "ParcelOwnerPIN"

# One address per parcel for the page heading. A 60-unit condo publishes 60
# unit addresses; loading them all makes the heading unreadable and the
# street crosswalk 60x too long.
ADDRESS_PRIMARY_FLAGS = ("IsParcelAddress", "IsBldgAddress")

# Not addresses of a place anyone asks about.
ADDRESS_EXCLUDE_FLAGS = ("IsWaterMeterAddress", "IsIrrigationWaterMeter")


def load_addresses(path: Path) -> tuple[pd.DataFrame, dict]:
    """Site addresses keyed to PIN, with one primary per parcel.

    Returns (frame indexed by PIN with site_address + alt_addresses, stats).
    Quality filtering is explicit rather than silent: a retired or disputed
    address pointing at a parcel page is worse than no address at all.
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

    # ── PIN, with a fallback that is reported, not silent ────────────────
    pin = a.get(ADDRESS_PIN_FIELD, pd.Series(pd.NA, index=a.index))
    pin = pin.astype(str).str.strip().str.replace(r"\D", "", regex=True)
    pin = pin.where(pin.str.len() == 10)
    if ADDRESS_PIN_FALLBACK in a.columns:
        alt = (a[ADDRESS_PIN_FALLBACK].astype(str).str.strip()
               .str.replace(r"\D", "", regex=True))
        alt = alt.where(alt.str.len() == 10)
        filled = pin.isna() & alt.notna()
        stats["pin_from_owner_fallback"] = int(filled.sum())
        stats["pin_fields_disagree"] = int(
            (pin.notna() & alt.notna() & (pin != alt)).sum())
        pin = pin.where(~filled, alt)
    a["PIN"] = pin
    stats["no_pin"] = int(a["PIN"].isna().sum())

    # ── Quality filters, each counted ────────────────────────────────────
    keep = a["PIN"].notna()
    if "Status" in a.columns:
        st = a["Status"].astype(str).str.strip().str.lower()
        bad = st.isin(("retired", "inactive", "deleted", "void", "pending"))
        stats["dropped_status"] = int((keep & bad).sum())
        keep &= ~bad
    issue = flag("AddressHasIssue")
    stats["dropped_has_issue"] = int((keep & issue).sum())
    keep &= ~issue
    meter = flag(ADDRESS_EXCLUDE_FLAGS[0]) | flag(ADDRESS_EXCLUDE_FLAGS[1])
    stats["dropped_meter"] = int((keep & meter).sum())
    keep &= ~meter
    outside = flag("AddressOutsideCity")
    stats["flagged_outside_city"] = int((keep & outside).sum())

    a = a[keep].copy()
    stats["rows_kept"] = len(a)

    # ── Address text. FullAddress may carry city/state; BaseAddress is the
    # street portion and is what the crosswalk should group on. ──────────
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

    # ── Rank: parcel address beats building beats unit ───────────────────
    is_unit = flag("IsUnitAddress")
    rank = np.select(
        [flag("IsParcelAddress"), flag("IsBldgAddress"), ~is_unit],
        [0, 1, 2], 3)
    a["_rank"] = rank
    a["_unit"] = is_unit
    stats["unit_addresses"] = int(is_unit.sum())

    primary = (a.sort_values(["PIN", "_rank", "_addr"])
               .drop_duplicates("PIN")[["PIN", "_addr"]]
               .rename(columns={"_addr": "site_address"}))

    # Alternates feed the alias line and the street crosswalk, so a parcel
    # with addresses on two streets is findable from either. Capped — a
    # 200-unit building would otherwise bury its own heading.
    alts = (a[~a["PIN"].isin([])].sort_values(["PIN", "_rank", "_addr"])
            .groupby("PIN")["_addr"]
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
    if capacity and capacity.exists():
        cap = pd.read_parquet(capacity)
        cap["PIN"] = cap["PIN"].astype(str).str.strip().str.zfill(10)
        cols = ["PIN"] + [c for c in cap.columns
                          if c in CAPACITY_FIELDS or c.startswith("c_")]
        df = df.merge(cap[cols], on="PIN", how="left", suffixes=("", "_cap"))
        has_cap = True
    else:
        print("  WARNING: parcels_capacity.parquet not joined. Pages will carry "
              "zone, lot area and values but NO development capacity — which is "
              "most of what a 'what can be built here' question needs. Run "
              "s2_capacity.py first.")


    df["alt_addresses"] = pd.Series([[] for _ in range(len(df))], index=df.index)
    has_addr = False
    if addresses and addresses.exists():
        addr, astats = load_addresses(addresses)
        df = df.join(addr, on="PIN", rsuffix="_src")
        if "site_address_src" in df.columns:
            df["site_address"] = df["site_address"].fillna(
                df.pop("site_address_src"))
        df["alt_addresses"] = df["alt_addresses"].map(
            lambda v: v if isinstance(v, list) else [])
        has_addr = df["site_address"].notna().any()

        matched = int(df["site_address"].notna().sum())
        print(f"  addresses: {astats['rows_in']:,} points -> "
              f"{astats['rows_kept']:,} after quality filters -> "
              f"{astats['parcels_with_address']:,} parcels; "
              f"{matched:,} of {len(df):,} parcels matched "
              f"({matched / max(len(df), 1):.1%})")
        for k in ("dropped_status", "dropped_has_issue", "dropped_meter",
                  "pin_from_owner_fallback", "pin_fields_disagree",
                  "flagged_outside_city", "unit_addresses",
                  "multi_address_parcels"):
            if astats.get(k):
                print(f"    {k}: {astats[k]:,}")

        orphan = astats["parcels_with_address"] - matched
        if orphan > 0:
            print(f"    {orphan:,} addressed PINs are not in the parcel "
                  f"master — PAA parcels, recent splits, or a vintage "
                  f"mismatch between the two sources")
        if matched / max(len(df), 1) < 0.5:
            print("    WARNING: under half of parcels matched an address. "
                  "People ask about parcels by address far more often than by "
                  "parcel number — check the PIN format before judging "
                  "retrieval quality.")
    else:
        df["site_address"] = pd.NA
        print("  WARNING: no site addresses joined. The address crosswalk "
              "cannot be built and every parcel page falls back to a bare "
              "parcel number in its heading.")

    df["watchlist_name"] = pd.NA
    if watchlist and watchlist.exists():
        w = pd.read_csv(watchlist, dtype={"PIN": str})
        w["PIN"] = w["PIN"].str.strip().str.zfill(10)
        df["watchlist_name"] = df["PIN"].map(dict(zip(w["PIN"], w["name"])))
        print(f"  {int(df['watchlist_name'].notna().sum()):,} parcels named in "
              f"the watchlist — each forced to a full page")

    for c in ["zone_split", "in_paa", "near_transit",
              "near_transit_centroid_basis", "condo_value_rollup"] + \
             [c for c in df.columns if c.startswith("c_")]:
        if c in df.columns:
            df[c] = df[c].map(truthy).astype(bool)

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

    df, has_cap, has_addr = load(Path(a.master), Path(a.capacity),
                                 Path(a.addresses), Path(a.watchlist))
    meta = RunMeta(a.run_id, a.parcel_vintage, a.zoning_vintage, a.a1_version,
                   has_cap, has_addr)

    # Tracts and PAA parcels stay in the index but never earn a page.
    df = assign_tiers(df, a.budget)
    flags = sorted(c for c in df.columns if c.startswith("c_"))

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    d1 = emit_dossiers(df, meta, out, flags)
    d2 = emit_index(df, meta, out)
    d3 = emit_addresses(df, meta, out)
    d4 = [emit_zone_profiles(df, meta, out, flags)]
    d5 = [emit_lookup_guide(df, meta, out)]
    mf = emit_manifest(df, meta, out,
                       {"P1 parcel pages": len(d1), "P2 parcel index": len(d2),
                        "P3 address crosswalk": len(d3),
                        "P4 zone profiles": len(d4),
                        "P5 lookup guide": len(d5), "Manifest": 1})

    total = len(d1) + len(d2) + len(d3) + len(d4) + len(d5) + 1
    chars = sum(len(f.read_text(encoding="utf-8"))
                for f in [*d1, *d2, *d3, *d4, *d5, mf])
    print(f"\n  {total} files, {chars:,} characters")
    print(f"  {int((df['tier']=='dossier').sum()):,} full pages · "
          f"{int((df['tier']=='index').sum()):,} index entries")
    (out / "P_summary.json").write_text(json.dumps(
        {"files": total, "characters": chars,
         "dossiers": int((df["tier"] == "dossier").sum()),
         "index": int((df["tier"] == "index").sum()),
         "addresses": int(df["site_address"].notna().sum()),
         "capacity_joined": has_cap}, indent=2))


if __name__ == "__main__":
    main()
