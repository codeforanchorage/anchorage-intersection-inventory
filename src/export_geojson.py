"""Phase 5: Assemble per-intersection inventories into GeoJSON + summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from geojson import Feature, FeatureCollection, Point, dump

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# OSM tags worth keeping on the output (drop the noisy/internal ones).
OSM_TAGS_TO_KEEP = {
    "name", "ref", "operator", "traffic_signals", "traffic_signals:direction",
    "crossing", "crossing:markings", "button_operated", "highway",
}


def flatten_assets(assets: list[dict]) -> dict:
    """Flatten the assets array into ArcGIS-friendly columns: <asset>_count, <asset>_condition, etc."""
    flat: dict = {}
    for asset_type in config.ASSET_TYPES:
        flat[f"{asset_type}_count"] = 0
        flat[f"{asset_type}_avg_score"] = None
        flat[f"{asset_type}_worst_condition"] = None
    for a in assets:
        t = a["asset_type"]
        if t not in config.ASSET_TYPES:
            t = "other"
        flat[f"{t}_count"] = flat.get(f"{t}_count", 0) + a["count"]
        flat[f"{t}_avg_score"] = a["avg_condition_score"]
        flat[f"{t}_worst_condition"] = a["worst_condition"]
    return flat


def load_signal_props(path: Path) -> dict[str, dict]:
    """Map osm_id -> original Phase 1 properties (for OSM tags + lat/lon fallback)."""
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for feat in json.load(f).get("features", []):
            out[feat["properties"]["osm_id"]] = feat["properties"]
    return out


def build_feature(record: dict, signal_props: dict) -> Feature:
    osm_id = record["intersection_id"]
    lat, lon = record["lat"], record["lon"]
    tags = signal_props.get(osm_id, {})
    osm_tag_props = {f"osm_{k}": v for k, v in tags.items() if k in OSM_TAGS_TO_KEEP}

    properties = {
        "intersection_id": osm_id,
        "lat": lat,
        "lon": lon,
        "cluster_size": record.get("cluster_size", 1),
        "gsv_coverage": record.get("gsv_coverage", False),
        "gsv_date": record.get("gsv_date", ""),
        "gsv_pano_id": record.get("gsv_pano_id", ""),
        "overall_condition_score": record.get("overall_condition_score"),
        "priority_issues": "; ".join(record.get("priority_issues", []) or []),
        "image_count": len(record.get("images", [])),
        **flatten_assets(record.get("assets", [])),
        **osm_tag_props,
    }
    return Feature(geometry=Point((lon, lat)), properties=properties)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5: export GeoJSON + CSV.")
    args = parser.parse_args(argv)

    if not config.SIGNALS_GEOJSON.exists():
        print(f"ERROR: {config.SIGNALS_GEOJSON} not found.")
        return 1

    signal_props = load_signal_props(config.SIGNALS_GEOJSON)
    inventory_files = sorted(config.RESULTS_DIR.glob("*_inventory.json"))
    if not inventory_files:
        print("ERROR: no *_inventory.json files in results dir. Run Phase 4 first.")
        return 1

    features: list[Feature] = []
    csv_rows: list[dict] = []
    for path in inventory_files:
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        feat = build_feature(record, signal_props)
        features.append(feat)
        csv_rows.append(feat["properties"])

    config.INVENTORY_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    with config.INVENTORY_GEOJSON.open("w", encoding="utf-8") as f:
        dump(FeatureCollection(features), f, indent=2)

    fieldnames: list[str] = []
    for row in csv_rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with config.INVENTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {len(features)} features to {config.INVENTORY_GEOJSON}")
    print(f"Wrote summary CSV to {config.INVENTORY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
