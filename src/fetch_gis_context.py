"""Phase 7: Pull Anchorage GIS context per intersection.

Adds three signals the GSV-only pipeline cannot produce:

  1. Ground elevation in meters from the 2025 LiDAR-derived DEM
     (Map Service ``DEM_2025`` / ``identify`` endpoint). A single point
     sample at the intersection centroid; useful for ADA cross-slope
     screening, drainage / flooding context, and sight-distance work.

  2. A 2024 EagleView orthophoto JPG centered on the intersection
     (60m × 60m footprint at 512 px = ~12 cm/pixel). Stored as
     ``data/gis/{osm_id}_aerial.jpg``.

  3. Top-down SAM 3 detections on the aerial — crosswalks (for
     calibrating ground SAM 3's known zebra-stripe fragmentation) and
     vehicles (sanity check that the model is engaging). Written to
     ``data/results/{osm_id}_aerial_detections.json`` in the same
     schema Phase 4 already understands.

Re-runs are idempotent: if the aerial JPG and the per-intersection
detections file already exist, the row is preserved and the network /
GPU work is skipped. ``--refresh`` forces a re-fetch of the aerial and
re-run of SAM 3.

Phase 4 ingests the gis_context.csv + the aerial detection files and
surfaces aerial counts + elevation as additional inventory columns.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import detect_assets


_LOG = logging.getLogger("fetch_gis_context")


# Aerial-specific SAM 3 prompts. Top-down view is a different visual signature
# than ground GSV; the PoC sanity check showed signal heads / poles are below
# SAM 3's effective resolution at 60m / 512px (~3-pixel asset). Crosswalks and
# vehicles are reliably detectable; we keep both — crosswalk for the count
# comparison, vehicle as a sanity signal that the model engaged.
AERIAL_PROMPTS = [
    {"text": "crosswalk", "asset_type": "crosswalk_marking", "confidence": 0.20},
    {"text": "vehicle", "asset_type": "vehicle", "confidence": 0.30},
]


def bbox_around(lat: float, lon: float, half_extent_m: float) -> str:
    """ArcGIS bbox string in EPSG:4326 (xmin,ymin,xmax,ymax)."""
    deg_lat = half_extent_m / 111111.0
    deg_lon = deg_lat / abs(math.cos(math.radians(lat)))
    return f"{lon - deg_lon},{lat - deg_lat},{lon + deg_lon},{lat + deg_lat}"


def fetch_aerial(lat: float, lon: float) -> bytes:
    bbox = bbox_around(lat, lon, config.AERIAL_HALF_EXTENT_M)
    r = requests.get(
        f"{config.ANCHORAGE_PHOTO_2024}/export",
        params={
            "bbox": bbox, "bboxSR": 4326, "imageSR": 4326,
            "size": f"{config.AERIAL_TILE_PX},{config.AERIAL_TILE_PX}",
            "format": "jpg", "f": "image",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.content


def fetch_dem_elevation(lat: float, lon: float) -> float | None:
    """Single-point elevation in meters from DEM_2025. None on miss/error."""
    bbox = bbox_around(lat, lon, half_extent_m=5.0)
    r = requests.get(
        f"{config.ANCHORAGE_DEM_2025}/identify",
        params={
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "sr": 4326,
            "tolerance": 1,
            "mapExtent": bbox,
            "imageDisplay": "512,512,96",
            "f": "json",
        },
        timeout=30,
    )
    r.raise_for_status()
    doc = r.json()
    for result in doc.get("results", []):
        attrs = result.get("attributes", {})
        for key in ("Stretch.Pixel Value", "Pixel Value", "PIXEL VALUE"):
            if key in attrs:
                try:
                    return float(attrs[key])
                except (TypeError, ValueError):
                    return None
    return None


def aerial_detections_path(osm_id: str) -> Path:
    return config.RESULTS_DIR / f"{osm_id}_aerial_detections.json"


def aerial_image_path(osm_id: str) -> Path:
    return config.GIS_DIR / f"{osm_id}_aerial.jpg"


def already_done(osm_id: str) -> bool:
    return aerial_image_path(osm_id).exists() and aerial_detections_path(osm_id).exists()


def load_existing_rows(csv_path: Path) -> dict[str, dict]:
    """Map intersection_id -> existing csv row, for carry-over on partial runs."""
    out: dict[str, dict] = {}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["intersection_id"]] = row
    return out


def write_aerial_detections(osm_id: str, image_path: Path, detections: list[dict]) -> None:
    doc = {
        "intersection_id": osm_id,
        "images": [{
            "image": image_path.name,
            "heading": None,
            "pitch": None,
            "source": "aerial_photo_2024",
            "detections": detections,
        }],
    }
    aerial_detections_path(osm_id).parent.mkdir(parents=True, exist_ok=True)
    with aerial_detections_path(osm_id).open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def process_intersections(
    features: Iterable[dict],
    refresh: bool,
    skip_sam3: bool,
) -> tuple[int, int, int, int]:
    """Returns (processed, skipped_existing, fetch_errors, sam3_failures)."""
    config.GIS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_list = list(features)
    existing_rows = load_existing_rows(config.GIS_CONTEXT_CSV)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    processed = skipped_existing = fetch_errors = sam3_failures = 0

    predictor = None
    if not skip_sam3:
        print("Loading SAM 3 (one-time)...")
        predictor = detect_assets.init_sam3_predictor(default_confidence=0.20)
        print(f"  {len(AERIAL_PROMPTS)} aerial prompts ready")

    fieldnames = [
        "intersection_id", "lat", "lon",
        "ground_elevation_m", "aerial_image_path",
        "aerial_crosswalk_count", "aerial_vehicle_count",
        "fetched_at", "status",
    ]
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=config.GIS_CONTEXT_CSV.stem + ".",
        suffix=".csv.tmp", dir=str(config.GIS_CONTEXT_CSV.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    processed_ids: set[str] = set()

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for feat in feature_list:
            osm_id = feat["properties"]["osm_id"]
            lon, lat = feat["geometry"]["coordinates"]
            processed_ids.add(osm_id)

            if not refresh and already_done(osm_id) and osm_id in existing_rows:
                writer.writerow({k: existing_rows[osm_id].get(k, "") for k in fieldnames})
                skipped_existing += 1
                continue

            elev: float | None = None
            try:
                elev = fetch_dem_elevation(lat, lon)
            except Exception as exc:
                _LOG.warning("DEM elevation failed for %s: %s", osm_id, exc)
                fetch_errors += 1
            time.sleep(config.ANCHORAGE_GIS_RATE_LIMIT_SEC)

            img_path = aerial_image_path(osm_id)
            try:
                aerial_bytes = fetch_aerial(lat, lon)
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.write_bytes(aerial_bytes)
            except Exception as exc:
                _LOG.warning("aerial fetch failed for %s: %s", osm_id, exc)
                fetch_errors += 1
                writer.writerow({
                    "intersection_id": osm_id, "lat": lat, "lon": lon,
                    "ground_elevation_m": elev if elev is not None else "",
                    "aerial_image_path": "",
                    "aerial_crosswalk_count": "",
                    "aerial_vehicle_count": "",
                    "fetched_at": now_iso, "status": "AERIAL_ERROR",
                })
                continue
            time.sleep(config.ANCHORAGE_GIS_RATE_LIMIT_SEC)

            crosswalk_n = vehicle_n = 0
            if predictor is not None:
                try:
                    detections = detect_assets.detect_with_sam3(
                        predictor, img_path, AERIAL_PROMPTS,
                    )
                    write_aerial_detections(osm_id, img_path, detections)
                    for det in detections:
                        atype = det.get("asset_type", "")
                        if atype == "crosswalk_marking":
                            crosswalk_n += 1
                        elif atype == "vehicle":
                            vehicle_n += 1
                except Exception as exc:
                    _LOG.warning("SAM 3 failed for %s aerial: %s", osm_id, exc)
                    sam3_failures += 1

            writer.writerow({
                "intersection_id": osm_id, "lat": lat, "lon": lon,
                "ground_elevation_m": f"{elev:.3f}" if elev is not None else "",
                "aerial_image_path": str(img_path.relative_to(config.PROJECT_ROOT)),
                "aerial_crosswalk_count": crosswalk_n,
                "aerial_vehicle_count": vehicle_n,
                "fetched_at": now_iso, "status": "OK",
            })
            processed += 1
            if processed % 25 == 0:
                print(f"  ...{processed} processed, {skipped_existing} skipped, "
                      f"{fetch_errors} fetch errors")

        # Carry over un-touched intersections so --limit doesn't truncate.
        for xid, row in existing_rows.items():
            if xid in processed_ids:
                continue
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    os.replace(tmp_path, config.GIS_CONTEXT_CSV)
    return processed, skipped_existing, fetch_errors, sam3_failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7: pull Anchorage GIS context.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N intersections.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch aerial + re-run SAM 3 even when outputs exist.")
    parser.add_argument("--skip-sam3", action="store_true",
                        help="Fetch aerial + DEM only; skip SAM 3 detection on aerials.")
    args = parser.parse_args(argv)

    if not config.SIGNALS_GEOJSON.exists():
        print(f"ERROR: {config.SIGNALS_GEOJSON} not found. Run Phase 1 first.")
        return 1

    with config.SIGNALS_GEOJSON.open("r", encoding="utf-8") as f:
        features = json.load(f).get("features", [])
    if args.limit is not None:
        features = features[: args.limit]

    print(f"Processing {len(features)} intersections "
          f"(refresh={args.refresh}, skip_sam3={args.skip_sam3})...")
    processed, skipped, fetch_errs, sam3_errs = process_intersections(
        features, refresh=args.refresh, skip_sam3=args.skip_sam3,
    )
    print(f"  Processed: {processed}")
    print(f"  Skipped (already on disk): {skipped}")
    print(f"  Fetch errors: {fetch_errs}")
    print(f"  SAM 3 failures: {sam3_errs}")
    print(f"GIS context CSV: {config.GIS_CONTEXT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
