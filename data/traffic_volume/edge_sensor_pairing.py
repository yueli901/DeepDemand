import pandas as pd
import geopandas as gpd
from shapely.geometry import box, LineString
from tqdm import tqdm
import json
import numpy as np
import os
import math

# === 0. Setup ===
TEMP_DIR = "temp_sensor_matches"
os.makedirs(TEMP_DIR, exist_ok=True)

# === 1. Load sensor CSV ===
sensor_df = pd.read_csv("sensor_quality_2015-2024.csv")

# Drop sensors with all 0 data quality AND Inactive status ===
quality_cols = [c for c in sensor_df.columns if "_data_quality" in c]
mask_all_zero = (sensor_df[quality_cols].sum(axis=1) == 0)
mask_inactive = (sensor_df["Status"] == "Inactive")
sensor_df = sensor_df[~(mask_all_zero & mask_inactive)].copy()

sensor_gdf = gpd.GeoDataFrame(
    sensor_df,
    geometry=gpd.points_from_xy(sensor_df["Longitude"], sensor_df["Latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=27700)

# === 2. Load edge GeoJSON with only necessary columns ===
edge_gdf = gpd.read_file("../highway_network/uk_driving_edges_simplified.geojson").to_crs(epsg=27700)
edge_gdf = edge_gdf[[c for c in ["u", "v", "key", "geometry", "highway"] if c in edge_gdf.columns]]
edge_gdf = edge_gdf[edge_gdf["highway"].isin({"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"})].copy()
edge_gdf = edge_gdf[edge_gdf.geometry.type == "LineString"].reset_index(drop=True)

# === 3. Build spatial index ===
edges_sindex = edge_gdf.sindex

# === 4. Nearest-neighbor match with 5 km buffer ===
for sensor in tqdm(sensor_gdf.itertuples(index=False), total=len(sensor_gdf), desc="Assigning sensors to edges"):
    sensor_id = sensor.Id
    temp_path = os.path.join(TEMP_DIR, f"{sensor_id}.json")
    if os.path.exists(temp_path):
        continue  # Skip if already processed

    pt = sensor.geometry
    bbox = box(pt.x - 5000, pt.y - 5000, pt.x + 5000, pt.y + 5000)
    candidate_idxs = list(edges_sindex.intersection(bbox.bounds))

    if not candidate_idxs:
        continue

    candidate = edge_gdf.iloc[candidate_idxs]
    candidate = candidate[candidate.geometry.is_valid & ~candidate.geometry.is_empty]
    if candidate.empty:
        continue

    geoms = candidate.geometry.values
    dists = np.array([pt.distance(geom) for geom in geoms])
    min_idx = int(np.argmin(dists))
    min_row = candidate.iloc[min_idx]
    min_dist = dists[min_idx]

    edge_id = f"{min_row.u}_{min_row.v}_{min_row.key if pd.notna(min_row.key) else 0}"
    result = {
        "sensor_id": sensor_id,
        "edge_id": edge_id,
        "min_distance": min_dist
    }

    with open(temp_path, "w") as f:
        json.dump(result, f)

# === 5. Merge all results ===
edge_to_sensors = {}
matched_sensor_ids = []
matched_min_dists = []

for fname in tqdm(os.listdir(TEMP_DIR), desc="Merging results"):
    with open(os.path.join(TEMP_DIR, fname)) as f:
        res = json.load(f)
        sid = res["sensor_id"]
        eid = res["edge_id"]
        dist = res["min_distance"]

        edge_to_sensors.setdefault(eid, []).append(sid)
        matched_sensor_ids.append(sid)
        matched_min_dists.append(dist)

with open("edge_to_sensor.json", "w") as f:
    json.dump(edge_to_sensors, f, indent=2)

pd.DataFrame({
    "sensor_id": matched_sensor_ids,
    "min_distance": matched_min_dists
}).sort_values(by="min_distance", ascending=False).to_csv("sensor_edge_distances.csv", index=False)

print("✅ Final results saved.")