import geopandas as gpd, pandas as pd

g = gpd.read_parquet("out/parcels_master.parquet")
a1 = set(pd.read_csv("kb/A1_zone_standards.csv")["zone"].str.strip())

obs = (g["ZONE"].astype(str).str.strip()
       .value_counts(dropna=False).rename_axis("gis_zone")
       .reset_index(name="parcels"))
obs["a1_zone"] = obs["gis_zone"].where(obs["gis_zone"].isin(a1), "")
obs["basis"] = obs["a1_zone"].map(lambda v: "identical" if v else "unmapped")
obs["note"] = ""
obs[["gis_zone", "a1_zone", "basis", "note", "parcels"]].to_csv(
    "kb/A1_zone_crosswalk_DRAFT.csv", index=False)
print(obs.to_string(index=False))
