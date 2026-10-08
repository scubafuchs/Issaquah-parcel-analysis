"""
s1_parcel_master.py — Issaquah parcel master table.

Joins King County Assessor extracts to city GIS, clips to city limits,
assigns zone / subarea / PAA / transit proximity, and derives the parcel-level
fields s2_capacity.py consumes.

Emits:
    out/parcels_master.parquet   geo-parquet, EPSG:2926, one row per PIN
    out/S1_coverage.json         join coverage, for the D2 manifest

Endpoints resolved 2026-10-01 and held in config.yaml, not here.

Two data traps this script exists to avoid:
  1. The assessor's CurrentZoning field is unreliable for incorporated
     parcels. Zoning comes from the city planning service by dominant-overlap
     spatial join, never from the extract.
  2. DistrictName lags annexations by months. Scope is the city limits
     polygon, not a string filter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import yaml
from shapely.geometry import box
from shapely.ops import unary_union

CRS_WORK = 2926          # NAD83(HARN) / Washington North, ftUS
ARCGIS_PAGE = 1000
FRONTAGE_BUFFER_FT = 15


# ──────────────────────────────────────────────────────────────────────────────
# ArcGIS REST
# ──────────────────────────────────────────────────────────────────────────────

def layer_url(entry: dict) -> str:
    """Config entries carry a service root plus a layer id."""
    base = str(entry["url"]).rstrip("/")
    lyr = entry.get("layer")
    return base if lyr is None else f"{base}/{lyr}"


def envelope_params(cfg: dict, entry: dict) -> dict:
    """Server-side bbox filter for countywide and statewide sources.

    Without it, the King County SAO and DNR layers return tens of thousands
    of features and the run takes hours instead of minutes.
    """
    if not entry.get("use_envelope"):
        return {}
    xmin, ymin, xmax, ymax = cfg["query_envelope_2926"]
    return {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": CRS_WORK,
        "spatialRel": "esriSpatialRelIntersects",
    }


def fetch_layer(entry, cfg: dict | None = None,
                crs: int = CRS_WORK) -> gpd.GeoDataFrame:
    """Page through a MapServer or FeatureServer layer.

    Older ArcGIS Server builds ignore resultOffset. When page two returns the
    same first feature, stop loudly — accepting it would duplicate every
    feature, and a silent duplicate inflates every area-based figure
    downstream.
    """
    if isinstance(entry, str):
        entry = {"url": entry}
    if not entry.get("url"):
        raise ValueError("Layer has no url in config")

    url = layer_url(entry)
    where = entry.get("where", "1=1")
    extra = envelope_params(cfg, entry) if cfg else {}

    frames, offset, first_seen = [], 0, None
    while True:
        params = {"where": where, "outFields": "*", "f": "geojson",
                  "resultOffset": offset, "resultRecordCount": ARCGIS_PAGE,
                  "outSR": 4326, **extra}
        r = requests.get(f"{url}/query", params=params, timeout=240)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload:
            raise RuntimeError(f"{url}: server returned an error for where={where!r}: {payload=}, {params=}")

        feats = payload.get("features", [])
        if not feats:
            break

        fp = json.dumps(feats[0].get("properties", {}), sort_keys=True)
        if offset and fp == first_seen:
            raise RuntimeError(
                f"{url} ignores resultOffset — paginate by OBJECTID instead."
            )
        if offset == 0:
            first_seen = fp

        frames.append(gpd.GeoDataFrame.from_features(feats, crs=4326))
        if len(feats) < ARCGIS_PAGE and not payload.get("exceededTransferLimit"):
            break
        offset += len(feats)
        time.sleep(0.2)

    if not frames:
        raise RuntimeError(f"No features: {url} where {where}")
    out = pd.concat(frames, ignore_index=True)
    g = gpd.GeoDataFrame(out, geometry="geometry", crs=4326).to_crs(crs)
    g["geometry"] = g.geometry.buffer(0)
    return g


def fetch_multi(entry: dict, cfg: dict) -> gpd.GeoDataFrame:
    """Union several sublayers of one service into a single frame.

    Used by landslide_hazard, which needs scarps/flanks plus deposits but
    must not include the Study_Area sublayer.
    """
    if "layers" not in entry:
        return fetch_layer(entry, cfg)
    parts = []
    for lid in entry["layers"]:
        sub = {**entry, "layer": lid}
        sub.pop("layers", None)
        parts.append(fetch_layer(sub, cfg)[["geometry"]])
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                            geometry="geometry", crs=CRS_WORK)


def probe(entry: dict) -> None:
    """Print a layer's name, geometry type, and field list.

    Every VERIFY marker in config.yaml is answerable by one probe call.
    """
    r = requests.get(f"{layer_url(entry)}?f=json", timeout=60)
    r.raise_for_status()
    meta = r.json()
    print(f"  name        : {meta.get('name')}")
    print(f"  geometry    : {meta.get('geometryType')}")
    print(f"  maxRecords  : {meta.get('maxRecordCount')}")
    for f in meta.get("fields", []):
        cv = f.get("domain", {}).get("codedValues") if f.get("domain") else None
        vals = ("  values=" + ", ".join(str(c["code"]) for c in cv[:12])) if cv else ""
        print(f"    {f['name']:<24} {f['type'][13:]:<12}{vals}")


import numpy as np

# ── Unit attribution ──────────────────────────────────────────────────────
# King County splits dwelling counts across four extracts by property class,
# and that split is the entire bug. A condominium complex has no ResBldg
# record, no AptComplex record, and its units live on sibling PINs that carry
# no geometry — so grouping on the full PIN sends every unit, and every dollar
# of improvement value, to PINs the parcel layer does not contain.
#
#   EXTR_ResBldg       1-3 living units ONLY — 4-plex and up are absent
#   EXTR_AptComplex    NbrUnits per PIN
#   EXTR_CondoComplex  one row per complex, NbrUnits, Major -> PIN + '0000'
#   EXTR_CondoUnit     one row per unit RECORD — parking, storage, dock, moorage
#   EXTR_CommBldg      "commercial buildings include apartments"; no unit count

CONDO_DWELLING_UNITTYPES = {1, 2, 3, 4, 16, 26}
# 1 flat · 2 townhouse · 3 penthouse flat · 4 penthouse townhouse
# 16 mobile home · 26 floating home townhouse
# Excludes 5 parking, 6 storage, 7 dock, 8 moorage, 9 land only, 11 development
# rights, 12-15 unassigned. Counting rows instead inflates a 60-home building
# to 200+: the opposite error, and just as silent.

UNIT_FLOOR_BY_PRESENTUSE = {
    3: 2, 4: 3, 5: 4,              # duplex, triplex, 4-plex — confirmed non-firing
    11: 4, 16: 4, 17: 4, 18: 4,    # apartment variants: 4+ by definition
    20: 2, 25: 2, 48: 2,           # condominium residential / mixed use / mobile home park
}
# Floors fire only when no extract supplies a count. Codes 3/4/5 verified
# clean (median 2/3/4; 28 of 28 four-plexes exact) and stay as a safety net.
#
# DELIBERATELY NOT FLOORED:
#   29 Townhouse plat   — the 7 zeros are common tracts; a floor invents units
#                         AND puts tracts into D2C as redevelopment candidates
#   38 Mobile home park — no extract carries a count. A floor of 1 is worse
#                         than the zero it replaces because it looks resolved.
#                         Manual count -> UNIT_OVERRIDE_FILE.
#   49/55/56/57/59      — group quarters; classified, not floored

GROUP_QUARTERS_PRESENTUSE = {55, 56, 57, 59}   # rehab, dorm, group home, nursing
MIXED_REVIEW_PRESENTUSE = {49}                 # retirement facility: independent
                                               # living units are housing units,
                                               # skilled nursing beds are not
UNIT_OVERRIDE_FILE = "kb/unit_count_override.csv"


def _col(df: pd.DataFrame, *names: str) -> str | None:
    """Find a column under any of several spellings. Assessor extracts vary in
    casing and underscoring between the bulk CSVs and GIS-published copies."""
    lut = {c.lower().replace("_", ""): c for c in df.columns}
    for n in names:
        hit = lut.get(n.lower().replace("_", ""))
        if hit:
            return hit
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Assessor extracts
# ──────────────────────────────────────────────────────────────────────────────

# EXTR_CondoUnit was superseded by EXTR_CondoUnit2 (schema change: MOBHOME*
# fields collapsed to Length / Width / MHomeDescr). Both are tried so an
# older vintage directory still loads.
CONDO_UNIT_FILES    = ("EXTR_CondoUnit2.csv", "EXTR_CondoUnit.csv")
CONDO_COMPLEX_FILES = ("EXTR_CondoComplex2.csv", "EXTR_CondoComplex.csv")

CONDO_COMPLEX_MINOR = "0000"   # complexes are always minor 0000


def _pin_part(s: pd.Series, width: int) -> pd.Series:
    """Zero-pad a Major or Minor to fixed width.

    Numeric-coerce first: pandas reads a Minor column containing any null as
    float, so a plain str().zfill() yields '300.0' -> '300.0' and the PIN
    silently fails to join. That failure looks exactly like a parcel the
    assessor does not cover.
    """
    v = pd.to_numeric(s, errors="coerce")
    padded = v.astype("Int64").astype(str).str.zfill(width)
    return padded.where(v.notna(), s.astype(str).str.strip().str.zfill(width))


def _pin(df: pd.DataFrame, minor_default: str | None = None) -> pd.Series:
    """Major+Minor, 10 characters.

    EXTR_CondoComplex carries no Minor field — there is one row per complex,
    and the complex parcel is always minor 0000. That absence is not a schema
    gap; it is the confirmation that complex rows key to the PIN the parcel
    geometry layer holds, while unit rows key to PINs it does not.
    """
    major = _pin_part(df["Major"], 6)
    if "Minor" in df.columns:
        return major + _pin_part(df["Minor"], 4)
    if minor_default is not None:
        return major + minor_default
    raise ValueError(
        f"Extract has no Minor column and no default was supplied. "
        f"Columns: {sorted(df.columns)[:25]}")

def condo_units(ccx: pd.DataFrame | None, cu: pd.DataFrame) -> pd.Series:
    """Dwellings per complex, keyed to the PIN that carries geometry.

    CondoComplex gives a direct count and needs no unit-type filtering. Where
    it is absent, zero or null, fall back to counting dwelling-type unit
    records — EXTR_CondoUnit2 holds one row per unit RECORD, including
    parking, storage, dock and moorage, so counting rows unfiltered inflates
    a 60-home building to 200+. That is the opposite error from the one being
    fixed, and just as silent.
    """
    ut = _col(cu, "UnitType")
    if ut is None:
        raise ValueError(
            f"{'EXTR_CondoUnit2'} has no UnitType column. Counting rows would "
            f"treat parking, storage and moorage as dwellings. "
            f"Available: {sorted(cu.columns)[:25]}")
    dwell = cu[pd.to_numeric(cu[ut], errors="coerce")
               .isin(CONDO_DWELLING_UNITTYPES)]
    fallback = ((_pin_part(dwell["Major"], 6) + CONDO_COMPLEX_MINOR)
                .value_counts().rename_axis("PIN"))

    primary = pd.Series(dtype="Float64", name="u")
    if ccx is not None:
        n = _col(ccx, "NbrUnits", "NbrLivingUnits", "NumberOfUnits", "UnitCount")
        if n is None:
            print(f"  WARNING: EXTR_CondoComplex has no recognizable unit-count "
                  f"column — every complex falls back to filtered unit records. "
                  f"Available: {sorted(ccx.columns)[:25]}")
        else:
            primary = (pd.DataFrame({"PIN": ccx["PIN"],
                                     "u": pd.to_numeric(ccx[n], errors="coerce")})
                       .groupby("PIN")["u"].max())
            primary = primary.where(primary > 0)

    out = primary.combine_first(fallback).rename("u_condo")
    if out.empty:
        raise ValueError(
            "No condominium units resolved from either extract. 115 complexes "
            "would read as zero-unit vacant land.")
    print(f"  condo: {int(primary.notna().sum()):,} complexes from "
          f"CondoComplex, "
          f"{int(out.notna().sum() - primary.notna().sum()):,} from "
          f"unit-record fallback, {int(out.sum()):,} dwellings; "
          f"{len(cu) - len(dwell):,} parking/storage/dock/moorage records "
          f"excluded")
    return out


def condo_values(cu: pd.DataFrame, acct: pd.DataFrame) -> pd.DataFrame:
    """Land and improvement value per complex, summed off the unit accounts.

    Fixing units without fixing value leaves the ranking wrong: near-zero
    ApprImpsVal gives i_l_ratio ~0, decile 1, and HAZARD_DECILE_WEIGHTS[0] =
    3.0 — the highest redevelopment likelihood in the model.
    """
    v = acct[acct["PIN"].isin(set(cu["PIN"]))].copy()
    if v.empty:
        print("  note: no unit-level accounts matched — condo value rollup is "
              "a no-op, and those complexes keep a near-zero i_l_ratio")
        return pd.DataFrame(columns=["ApprLandVal", "ApprImpsVal"])
    v["PIN_cx"] = v["PIN"].str[:6] + CONDO_COMPLEX_MINOR
    return v.groupby("PIN_cx")[["ApprLandVal", "ApprImpsVal"]].sum(min_count=1)

def condo_year_built(ccx: pd.DataFrame | None, cu: pd.DataFrame) -> pd.Series:
    """Earliest year built per complex, keyed to the complex PIN.

    CondoComplex is primary — year built is a complex-level attribute, and
    the per-unit field is often blank or zero. Both domains include 0 as
    'unknown', so zero is missing rather than a year, and a plausibility
    window catches a stray 20 or 202 from a truncated field.
    """
    def roll(df):
        if df is None:
            return pd.Series(dtype="Float64")
        col = _col(df, "YrBuilt")
        if col is None:
            return pd.Series(dtype="Float64")
        d = pd.DataFrame({"PIN": _pin_part(df["Major"], 6) + CONDO_COMPLEX_MINOR,
                          "yr": pd.to_numeric(df[col], errors="coerce")})
        return d[d["yr"].between(1850, 2030)].groupby("PIN")["yr"].min()

    primary, fallback = roll(ccx), roll(cu)
    out = primary.combine_first(fallback)
    print(f"  condo yr_built: {int(primary.notna().sum()):,} from CondoComplex, "
          f"{int(out.notna().sum() - primary.notna().sum()):,} from unit records")
    return out

def load_assessor(src: Path) -> pd.DataFrame:
    """Build the parcel attribute table with correct unit and value attribution.

    Order matters: parcel role is classified BEFORE the condo rollup, because
    a common tract sharing a Major with its complex would otherwise receive
    the complex's units.
    """
    def read(names, required: bool = True, minor_default: str | None = None):
        names = (names,) if isinstance(names, str) else names
        for name in names:
            path = src / name
            if path.exists():
                df = pd.read_csv(path, low_memory=False, encoding="latin-1")
                df.columns = [c.strip() for c in df.columns]
                df["PIN"] = _pin(df, minor_default=minor_default)
                if name != names[0]:
                    print(f"  note: using {name} (preferred {names[0]} absent)")
                return df
        if required:
            raise FileNotFoundError(
                f"None of {list(names)} found in {src}. Condominium units "
                f"cannot be attributed without them, and 115 complexes read "
                f"as zero-unit vacant land at redevelopment decile 1.")
        print(f"  note: {names[0]} absent — skipping")
        return None

    acct   = read("EXTR_RPAcct_NoName.csv")
    parcel = read("EXTR_Parcel.csv")
    res    = read("EXTR_ResBldg.csv")
    apt    = read("EXTR_AptComplex.csv")
    cu     = read(CONDO_UNIT_FILES)
    ccx    = read(CONDO_COMPLEX_FILES, required=False,
                  minor_default=CONDO_COMPLEX_MINOR)

    # ── 1 · Account values, summed not deduplicated ───────────────────────
    acct_vals = (acct.groupby("PIN")[["ApprLandVal", "ApprImpsVal"]]
                 .sum(min_count=1).reset_index())
    multi = int((acct.groupby("PIN").size() > 1).sum())
    if multi:
        print(f"  {multi:,} PINs carry multiple tax accounts — values summed")

    # ── 2 · Base attributes, including role-detection fields ──────────────
    keep = ["PIN", "SqFtLot", "PresentUse", "DistrictName", "PropType",
            "NbrBldgSites", "SensitiveAreaTract", "NativeGrowthProtEsmt"]
    present = [c for c in keep if c in parcel.columns]
    for c in set(keep) - set(present):
        print(f"  note: EXTR_Parcel has no {c!r} — role detection degraded")
    base = (parcel[present].drop_duplicates("PIN")
            .merge(acct_vals, on="PIN", how="left")
            .rename(columns={"SqFtLot": "lot_sf"}))
    base["lot_sf"] = pd.to_numeric(base["lot_sf"], errors="coerce")
    base.loc[base["lot_sf"] <= 0, "lot_sf"] = pd.NA
    base["PresentUse"] = pd.to_numeric(base["PresentUse"], errors="coerce")

    # ── 3 · Parcel role, BEFORE any rollup ────────────────────────────────
    base["parcel_role"] = classify_parcel_role(base)
    print(f"  parcel roles: {base['parcel_role'].value_counts().to_dict()}")

    # ── 4 · Per-source unit counts ────────────────────────────────────────
    nlu = _col(res, "NbrLivingUnits")
    if nlu is None:
        raise ValueError(
            "EXTR_ResBldg has no NbrLivingUnits column. Defaulting to one "
            "unit per building reads every duplex and triplex as "
            "single-family and inflates middle-housing headroom at the same "
            f"time. Available: {sorted(res.columns)[:25]}")
    res_u = (pd.to_numeric(res[nlu], errors="coerce").fillna(1).clip(lower=1)
             .groupby(res["PIN"]).sum())

    apt_n = _col(apt, "NbrUnits")
    apt_u = pd.to_numeric(apt[apt_n], errors="coerce").groupby(apt["PIN"]).sum()

    condo_u = condo_units(ccx, cu)
    condo_v = condo_values(cu, acct)
    condo_yr = condo_year_built(ccx, cu)

    # ── 5 · Move complex value onto the complex parcel ────────────────────
    tgt = base["PIN"].isin(condo_v.index) & base["parcel_role"].ne("tract")
    for col in ("ApprLandVal", "ApprImpsVal"):
        rolled = base.loc[tgt, "PIN"].map(condo_v[col])
        base.loc[tgt, col] = np.fmax(
            pd.to_numeric(base.loc[tgt, col], errors="coerce").fillna(0.0),
            rolled.fillna(0.0))
    base["condo_value_rollup"] = tgt
    print(f"  condo value rollup applied to {int(tgt.sum()):,} complexes")

    # ── 6 · Units by precedence, then floors, then overrides ──────────────
    base["existing_units"] = resolve_units(base, res_u, apt_u, condo_u)
    base = apply_unit_overrides(base)

    # ── 7 · i_l_ratio LAST — stale on every row the rollup touched ────────
    land = pd.to_numeric(base["ApprLandVal"], errors="coerce").replace(0, pd.NA)
    base["i_l_ratio"] = pd.to_numeric(base["ApprImpsVal"], errors="coerce") / land
    base.loc[base["parcel_role"].eq("tract"), "i_l_ratio"] = pd.NA

    # ── 8 · Year built, with the condo fallback ───────────────────────────
    # Condo complexes have no ResBldg record, so yr_built has been NaN on all
    # of them — the same root cause as the zero units, showing as a dash in
    # D2C right beside it. EXTR_CondoUnit2 carries YrBuilt per unit.
    ryr = _col(res, "YrBuilt")
    res_yr = res.groupby("PIN")[ryr].min()
    res_yr = res_yr.where(res_yr.between(1850, 2030)).rename("yr_built")
    base = base.merge(res_yr, on="PIN", how="left")
    fill = base["yr_built"].isna() & base["PIN"].isin(condo_yr.index)
    base.loc[fill, "yr_built"] = base.loc[fill, "PIN"].map(condo_yr)
    if fill.any():
        print(f"  {int(fill.sum()):,} complexes took yr_built from condo units")

    # ── 9 · Housing units vs group quarters ───────────────────────────────
    base["unit_class"] = np.select(
        [base["PresentUse"].isin(GROUP_QUARTERS_PRESENTUSE),
         base["PresentUse"].isin(MIXED_REVIEW_PRESENTUSE)],
        ["group_quarters", "mixed_review"], "housing_unit")

    return base


# ──────────────────────────────────────────────────────────────────────────────
# Spatial assignment
# ──────────────────────────────────────────────────────────────────────────────

def dominant_overlap(parcels: gpd.GeoDataFrame, polys: gpd.GeoDataFrame,
                     field: str, out_col: str) -> pd.DataFrame:
    """Assign each parcel the polygon attribute covering the most of its area.

    An `intersects` sjoin duplicates split-zoned parcels and every downstream
    aggregate then counts them twice.
    """
    low = out_col.lower()
    cols = ["PIN", out_col, f"{low}_overlap_share", f"{low}_split"]
    if field not in polys.columns:
        raise ValueError(
            f"Field {field!r} not found in the {out_col} layer. Available: "
            f"{sorted(c for c in polys.columns if c != 'geometry')}. "
            f"Run --probe to confirm, then fix config.yaml."
        )
    pairs = gpd.overlay(parcels[["PIN", "geometry"]], polys[[field, "geometry"]],
                        how="intersection", keep_geom_type=False)
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    pairs["part_area"] = pairs.geometry.area
    lot = parcels.set_index("PIN").geometry.area
    grp = pairs.groupby(["PIN", field], as_index=False)["part_area"].sum()
    grp["share"] = grp["part_area"] / grp["PIN"].map(lot)
    grp = grp.sort_values("part_area", ascending=False)
    win = grp.drop_duplicates("PIN").copy()
    nparts = grp[grp["share"] > 0.05].groupby("PIN").size()
    win["split"] = win["PIN"].map(nparts).fillna(1) > 1
    return win.rename(columns={field: out_col, "share": f"{low}_overlap_share",
                               "split": f"{low}_split"})[cols]


def transit_proximity(parcels: gpd.GeoDataFrame, cfg: dict) -> pd.DataFrame:
    """Quarter-mile proximity to a major transit stop.

    Issaquah has not adopted a major transit stop map, so the stops are
    defined by parcel in config.yaml. See the method notes: this is a working
    assumption, not a published designation, and it decides whether a parcel
    carries a 4-unit middle-housing floor or a 2-unit one.

    Both the boundary basis and the centroid basis are computed so the
    sensitivity is reportable rather than buried in one number.
    """
    spec = cfg.get("major_transit_stops", {})
    out = pd.DataFrame({"PIN": parcels["PIN"],
                        "near_transit": False,
                        "near_transit_centroid_basis": False})
    pins = [str(s["pin"]).zfill(10) for s in spec.get("pins", [])]
    if not pins:
        print("  note: no major transit stops configured — "
              "every mh floor is 2 units, which understates capacity")
        return out

    stops = parcels[parcels["PIN"].isin(pins)]
    found = set(stops["PIN"])
    for s in spec["pins"]:
        pin = str(s["pin"]).zfill(10)
        mark = "ok" if pin in found else "NOT FOUND in parcel layer"
        print(f"    transit stop {pin}  {s.get('name','')}  [{mark}]")
    if stops.empty:
        raise ValueError(
            "No configured transit-stop parcel resolved. Capacity on every "
            "middle-housing parcel would silently drop from 4 units to 2. "
            "Fix the PINs before running."
        )

    r = float(spec.get("buffer_ft", 1320))
    pt = parcels.geometry.representative_point()

    band_edge = unary_union(stops.geometry.buffer(r).values)
    band_ctr = unary_union(stops.geometry.representative_point().buffer(r).values)
    out["near_transit"] = pt.within(band_edge).values
    out["near_transit_centroid_basis"] = pt.within(band_ctr).values

    if spec.get("buffer_from") == "parcel_centroid":
        out["near_transit"] = out["near_transit_centroid_basis"]

    n_e = int(out["near_transit"].sum())
    n_c = int(out["near_transit_centroid_basis"].sum())
    print(f"  transit proximity: {n_e:,} parcels on the configured basis, "
          f"{n_c:,} measured from stop centroids (spread {abs(n_e - n_c):,})")
    return out


def derive_frontage(parcels: gpd.GeoDataFrame, streets: gpd.GeoDataFrame | None,
                    cfg: dict) -> pd.DataFrame:
    """Street frontage in linear feet, and whether it is improved.

    Frontage length is derived from street centerlines. Improvement status is
    NOT derivable — no published layer separates sidewalk from road pavement
    — so it comes from an explicit config default, never a silent one.
    """
    out = pd.DataFrame({"PIN": parcels["PIN"], "frontage_lf": pd.NA})
    if streets is not None and not streets.empty:
        band = unary_union(streets.geometry.buffer(FRONTAGE_BUFFER_FT).values)
        out["frontage_lf"] = (parcels.geometry.boundary.intersection(band)
                              .length.replace(0, pd.NA).values)

    default = bool(cfg.get("frontage_improved_default", True))
    out["frontage_improved"] = default

    h = cfg.get("frontage_improved_heuristic", {})
    if h.get("enabled"):
        out["frontage_improved"] = (
            (parcels["existing_units"].values > 0)
            & (pd.to_numeric(parcels["yr_built"], errors="coerce").fillna(0).values >= 1990)
        )
        print("  frontage improvement inferred by heuristic [A] — label it in the manifest")
    else:
        n = int(out["frontage_lf"].notna().sum())
        print(f"  frontage improvement set to {default} for all {n:,} parcels "
              f"with measured frontage (no published sidewalk layer)")
    return out


def derive_infiltration(parcels: gpd.GeoDataFrame, cfg: dict) -> pd.Series:
    spec = cfg.get("layers", {}).get("soil_infiltration")
    vals = (spec or {}).get("feasible_values") or []
    if not spec or not spec.get("url") or not vals:
        d = bool(cfg.get("infiltration_feasible_default", False))
        print(f"  note: infiltration not joined (feasible_values empty) — "
              f"CARA vault adder fires on every CARA parcel")
        return pd.Series(d, index=parcels.index)
    cats = fetch_layer(spec, cfg)
    field = spec.get("field", "CATEGORY")
    ok = unary_union(cats[cats[field].isin(vals)].geometry.values)
    return pd.Series(parcels.geometry.representative_point().within(ok).values,
                     index=parcels.index)

def classify_parcel_role(p: pd.DataFrame) -> pd.Series:
    """Separate 'zero units because they live elsewhere' from 'zero units
    because nothing can ever be built here.'

    Both currently read as zero and both rank at the top of D2C — the first
    with the complex's real headroom, the second with headroom it can never
    use. EXTR_Parcel carries the fields that tell them apart:
      PropType  K = condominium, U = undivided interest
      NbrBldgSites  0 = tract
      SensitiveAreaTract / NativeGrowthProtEsmt  Y = protected tract
    """
    idx = p.index
    yes = lambda c: (p[c].astype(str).str.strip().str.upper().eq("Y")
                     if c in p.columns else pd.Series(False, index=idx))
    pt = (p["PropType"].astype(str).str.strip().str.upper()
          if "PropType" in p.columns else pd.Series("", index=idx))
    sites = (pd.to_numeric(p["NbrBldgSites"], errors="coerce")
             if "NbrBldgSites" in p.columns else pd.Series(pd.NA, index=idx))

    # Zero improvement value WITH an account present (0.0, not NaN) on a parcel
    # with no building sites is the common-tract signature. NaN improvement
    # value means no account at that PIN at all — that is the condo case.
    imps = pd.to_numeric(p["ApprImpsVal"], errors="coerce")
    tract = (pt.eq("U") | yes("SensitiveAreaTract") | yes("NativeGrowthProtEsmt")
             | ((sites == 0) & imps.eq(0.0)))
    condo = (pt.eq("K") | p["PIN"].str.endswith("0000")) & ~tract
    return pd.Series(np.select([tract, condo], ["tract", "condo_complex"],
                               "parcel"), index=idx)


def resolve_units(p: pd.DataFrame, res_u, apt_u, condo_u) -> pd.Series:
    """Precedence by source authority, then a PresentUse floor.

    MAX degraded silently: when the authoritative source was null it fell
    through to the weakest one, so an apartment with no AptComplex record read
    as a single-family house — or, with no ResBldg record either, as vacant
    land. The 7 PresentUse-11 parcels are in EXTR_CommBldg, which carries no
    unit count, so they reach the floor.
    """
    u = pd.Series(pd.NA, index=p.index, dtype="Float64")
    for src in (condo_u, apt_u, res_u):
        cand = p["PIN"].map(src)
        u = u.mask(u.isna() & cand.notna() & (cand > 0), cand)

    is_tract = p["parcel_role"].eq("tract")
    floor = p["PresentUse"].map(UNIT_FLOOR_BY_PRESENTUSE)
    rescued = (u.isna() | u.eq(0)) & floor.notna() & ~is_tract
    if rescued.any():
        by_use = p.loc[rescued, "PresentUse"].value_counts().to_dict()
        print(f"  {int(rescued.sum()):,} parcels had no unit record but a "
              f"unit-bearing PresentUse {by_use} — floored, not zeroed. "
              f"These are UNDERCOUNTS and belong in the D2 manifest.")
    u = u.mask(rescued, floor)

    p["units_basis"] = np.select(
        [is_tract, rescued, u.notna()],
        ["tract", "presentuse_floor", "extract"], "none")
    return u.fillna(0).round().astype(int)


def apply_unit_overrides(p: pd.DataFrame) -> pd.DataFrame:
    """Hand-counted units for classes no extract can supply.

    Its own file because it expires on its own schedule: it changes when
    somebody counts a mobile home park, not when the code or the rates change.
    """
    path = Path(UNIT_OVERRIDE_FILE)
    if not path.exists():
        missing = p[p["PresentUse"].eq(38) & p["existing_units"].eq(0)]
        if len(missing):
            print(f"  WARNING: {len(missing)} mobile home park parcel(s) read "
                  f"zero and no override file exists. A park is the highest "
                  f"units-per-parcel class in the county. PINs: "
                  f"{list(missing['PIN'])}")
        return p
    ov = pd.read_csv(path, dtype={"PIN": str}).dropna(subset=["units"])
    ov["PIN"] = ov["PIN"].str.strip().str.zfill(10)
    m = p["PIN"].map(dict(zip(ov["PIN"], pd.to_numeric(ov["units"]))))
    hit = m.notna()
    p.loc[hit, "existing_units"] = m[hit].round().astype(int)
    p.loc[hit, "units_basis"] = "manual_override"
    print(f"  {int(hit.sum()):,} parcels took a manual unit override "
          f"({int(m[hit].sum()):,} units)")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────────

def load_parcel_geometry(cfg: dict, gis_dir: Path) -> gpd.GeoDataFrame:
    spec = cfg["parcels"]
    if spec.get("mode", "service") == "service":
        g = fetch_layer({"url": spec["url"], "layer": spec["layer"]}, cfg)
        pin_field = spec.get("pin_field", "ParcelPIN")
    else:
        g = gpd.read_file(gis_dir / Path(spec["shapefile_path"]).name).to_crs(CRS_WORK)
        pin_field = spec.get("shapefile_pin_field", "PIN")
    if pin_field not in g.columns:
        raise ValueError(
            f"PIN field {pin_field!r} absent from the parcel layer. "
            f"Available: {sorted(c for c in g.columns if c != 'geometry')}"
        )
    g["PIN"] = g[pin_field].astype(str).str.strip().str.zfill(10)
    g = g[["PIN", "geometry"]].drop_duplicates("PIN")
    g["geometry"] = g.geometry.buffer(0)
    return g

def check_unit_totals(g: gpd.GeoDataFrame, cfg: dict) -> None:
    """Citywide housing units against the adopted baseline.

    A5 records 18,000 housing units as of 2023, so a 2026 run should exceed it.
    This failure is invisible by construction — nothing in D2C says "these
    units are missing," it lists occupied housing as prime redevelopment
    candidates.

    Group quarters are excluded from the test, or they quietly pad the total
    past the floor. A5's rule: beds are never added to a unit count.
    """
    housing = int(g.loc[g["unit_class"].eq("housing_unit"), "existing_units"].sum())
    floor = int(cfg.get("existing_units_floor", 17_000))
    gq = int(g.loc[g["unit_class"].ne("housing_unit"), "existing_units"].sum())
    print(f"  housing units {housing:,} · group quarters {gq:,}")
    if housing < floor:
        raise ValueError(
            f"Citywide housing units {housing:,}, below the {floor:,} floor "
            f"(A5: 18,000 as of 2023). A dropped property class does not read "
            f"as missing data in D2C — it reads as vacant land with full "
            f"headroom at decile 1. Check condominium and apartment "
            f"attribution before emitting digests.")


def build(cfg: dict, vintage: str, assessor_dir: Path, gis_dir: Path) -> gpd.GeoDataFrame:
    L = cfg["layers"]

    print("  loading assessor extracts")
    attrs = load_assessor(assessor_dir)

    print("  loading parcel geometry")
    geom = load_parcel_geometry(cfg, gis_dir)

    print("  fetching framework layers")
    limits = fetch_layer(L["city_limits"], cfg)
    zoning = fetch_layer(L["zoning"], cfg)
    subarea = fetch_layer(L["subareas"], cfg)
    streets = fetch_layer(L["streets"], cfg) if L.get("streets", {}).get("url") else None

    paa = None
    paa_spec = L.get("paa") or {}
    if paa_spec.get("shapefile_path"):
        paa = gpd.read_file(paa_spec["shapefile_path"]).to_crs(CRS_WORK)
        if paa_spec.get("where"):
            paa = paa.query("JURIS == 'IS'")
    elif paa_spec.get("url") and paa_spec.get("layer") is not None:
        try:
            paa = fetch_layer(paa_spec, cfg)
        except Exception as exc:                            # noqa: BLE001
            print(f"  note: PAA layer unavailable ({type(exc).__name__}) — "
                  f"in_paa will be False for every parcel")

    das = None
    if (L.get("development_agreements") or {}).get("url"):
        try:
            das = fetch_layer(L["development_agreements"], cfg)
        except Exception as exc:                            # noqa: BLE001
            print(f"  note: development agreements unavailable "
                  f"({type(exc).__name__}) — VESTED_OVERRIDE has no source")

    g = gpd.GeoDataFrame(geom.merge(attrs, on="PIN", how="inner"),
                         geometry="geometry", crs=CRS_WORK)
    n_all = len(g)

    city = unary_union(limits.geometry.values)
    g["in_city"] = g.geometry.representative_point().within(city)
    g["in_paa"] = (g.geometry.representative_point()
                   .within(unary_union(paa.geometry.values)) if paa is not None else False)
    g = g[g["in_city"]].drop(columns="in_city").reset_index(drop=True)
    print(f"  {len(g):,} of {n_all:,} parcels inside city limits")

    g = g.merge(dominant_overlap(g, zoning, L["zoning"].get("field", "ZONE"), "ZONE"),
                on="PIN", how="left")
    g = g.merge(dominant_overlap(g, subarea, L["subareas"].get("field", "NAME"), "SUBAREA"),
                on="PIN", how="left")
    g["SUBAREA"] = g["SUBAREA"].fillna("Unassigned")

    # Vested: parcels under an executed development agreement bypass forecast
    # steps 3-4 per A2 Part 3. Boundary only — unit counts still come from D7.
    if das is not None:
        in_da = unary_union(das.geometry.values)
        g["in_development_agreement"] = g.geometry.representative_point().within(in_da)
        print(f"  {int(g['in_development_agreement'].sum()):,} parcels inside a "
              f"development agreement boundary")
    else:
        g["in_development_agreement"] = pd.NA

    g = g.merge(transit_proximity(g, cfg), on="PIN", how="left")
    g = g.merge(derive_frontage(g, streets, cfg), on="PIN", how="left")
    g["infiltration_feasible"] = derive_infiltration(g, cfg).values

    g["parcel_vintage"] = vintage
    g["zoning_vintage"] = cfg.get("zoning_vintage", vintage)

    check_unit_totals(g, cfg)

    return g


def coverage_report(g: gpd.GeoDataFrame) -> dict:
    da = g["in_development_agreement"]
    return {
        "parcels": int(len(g)),
        "zone_resolved": round(float(g["ZONE"].notna().mean()), 4),
        "zone_split": int(g.get("zone_split", pd.Series(False, index=g.index)).sum()),
        "subarea_resolved": round(float((g["SUBAREA"] != "Unassigned").mean()), 4),
        "lot_sf_present": round(float(g["lot_sf"].notna().mean()), 4),
        "land_value_present": round(float(g["i_l_ratio"].notna().mean()), 4),
        "frontage_present": round(float(g["frontage_lf"].notna().mean()), 4),
        "near_transit": int(g["near_transit"].sum()),
        "near_transit_centroid_basis": int(g["near_transit_centroid_basis"].sum()),
        "in_development_agreement": (None if da.isna().all() else int(da.fillna(False).sum())),
        "in_paa": int(pd.Series(g["in_paa"]).fillna(False).sum()),
        "existing_units_total": int(g["existing_units"].sum()),
        "units_housing": int(g.loc[g["unit_class"].eq("housing_unit"),
                                   "existing_units"].sum()),
        "units_group_quarters": int(g.loc[g["unit_class"].ne("housing_unit"),
                                          "existing_units"].sum()),
        "units_basis": g["units_basis"].value_counts().to_dict(),
        "parcel_role": g["parcel_role"].value_counts().to_dict(),
        "condo_value_rollup": int(g["condo_value_rollup"].fillna(False).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build the Issaquah parcel master.")
    p.add_argument("--vintage", help="assessor extract date, YYYY-MM-DD")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--assessor-dir", default=None)
    p.add_argument("--gis-dir", default="data/gis")
    p.add_argument("--outdir", default="out")
    p.add_argument("--probe", metavar="KEY", default=None,
                   help="print field names for a config layer key, then exit")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())

    if a.probe:
        entry = cfg["layers"].get(a.probe) or cfg.get(a.probe) or cfg["parcels"]
        print(f"\nprobing {a.probe}: {layer_url(entry)}")
        probe(entry)
        return

    if not a.vintage:
        p.error("--vintage is required unless --probe is used")

    src = Path(a.assessor_dir) if a.assessor_dir else Path("data/assessor") / a.vintage
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    g = build(cfg, a.vintage, src, Path(a.gis_dir))
    g.to_parquet(out / "parcels_master.parquet", index=False)

    cov = coverage_report(g)
    (out / "S1_coverage.json").write_text(json.dumps(cov, indent=2))
    print("\n  coverage")
    for k, v in cov.items():
        print(f"    {k:<32} {v}")
    if cov["zone_resolved"] < 0.95:
        print("\n  WARNING: zone coverage below 0.95 — s2_digests will refuse to emit.")


if __name__ == "__main__":
    main()
