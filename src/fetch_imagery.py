"""Phase 2: Download Google Street View imagery for each intersection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def load_intersections(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    return fc.get("features", [])


def metadata_for(lat: float, lon: float, key: str) -> dict:
    """Hit the GSV metadata endpoint to confirm imagery exists. Free and unmetered."""
    params = {"location": f"{lat},{lon}", "key": key}
    response = requests.get(config.GSV_METADATA_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def download_image(lat: float, lon: float, heading: int, key: str, out_path: Path) -> None:
    params = {
        "size": config.GSV_SIZE,
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": config.GSV_PITCH,
        "fov": config.GSV_FOV,
        "key": key,
    }
    response = requests.get(config.GSV_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)


def request_with_backoff(callable_, *args, **kwargs):
    """Retry on 429/5xx with exponential backoff. Up to 5 attempts."""
    delay = 1.0
    for attempt in range(5):
        try:
            return callable_(*args, **kwargs)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 or (status and 500 <= status < 600):
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("max retries exceeded")


def process_intersections(
    features: Iterable[dict],
    api_key: str,
    dry_run: bool,
    manifest_path: Path,
) -> tuple[int, int, int]:
    """Returns (covered, missing, downloaded)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    covered = missing = downloaded = 0

    fieldnames = [
        "intersection_id", "lat", "lon", "heading",
        "image_path", "gsv_pano_id", "gsv_date", "status",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for feat in features:
            props = feat["properties"]
            osm_id = props["osm_id"]
            lon, lat = feat["geometry"]["coordinates"]

            try:
                meta = request_with_backoff(metadata_for, lat, lon, api_key)
            except Exception as exc:
                print(f"  {osm_id}: metadata error: {exc}")
                meta = {"status": "ERROR"}
            time.sleep(config.GSV_RATE_LIMIT_SEC)

            status = meta.get("status", "UNKNOWN")
            pano_id = meta.get("pano_id", "")
            date = meta.get("date", "")

            if status != "OK":
                missing += 1
                writer.writerow({
                    "intersection_id": osm_id, "lat": lat, "lon": lon,
                    "heading": "", "image_path": "", "gsv_pano_id": "",
                    "gsv_date": "", "status": status,
                })
                continue

            covered += 1
            for heading in config.GSV_HEADINGS:
                image_path = config.IMAGE_DIR / f"{osm_id}_h{heading}.jpg"
                if dry_run:
                    img_str = ""
                    img_status = "DRY_RUN"
                else:
                    try:
                        request_with_backoff(
                            download_image, lat, lon, heading, api_key, image_path,
                        )
                        downloaded += 1
                        img_str = str(image_path.relative_to(config.PROJECT_ROOT))
                        img_status = "OK"
                    except Exception as exc:
                        print(f"  {osm_id} h{heading}: download error: {exc}")
                        img_str = ""
                        img_status = "DOWNLOAD_ERROR"
                    time.sleep(config.GSV_RATE_LIMIT_SEC)

                writer.writerow({
                    "intersection_id": osm_id, "lat": lat, "lon": lon,
                    "heading": heading, "image_path": img_str,
                    "gsv_pano_id": pano_id, "gsv_date": date, "status": img_status,
                })

    return covered, missing, downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: download GSV imagery.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check coverage only; do not download images.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N intersections.")
    args = parser.parse_args(argv)

    load_dotenv(config.PROJECT_ROOT / ".env")
    api_key = os.environ.get("GSV_API_KEY")
    if not api_key:
        print("ERROR: GSV_API_KEY not set in environment or .env file.")
        return 1

    if not config.SIGNALS_GEOJSON.exists():
        print(f"ERROR: {config.SIGNALS_GEOJSON} not found. Run Phase 1 first.")
        return 1

    features = load_intersections(config.SIGNALS_GEOJSON)
    if args.limit is not None:
        features = features[: args.limit]

    print(f"Processing {len(features)} intersections (dry_run={args.dry_run})...")
    covered, missing, downloaded = process_intersections(
        features, api_key, args.dry_run, config.IMAGERY_MANIFEST,
    )
    print(f"  GSV coverage OK: {covered}")
    print(f"  GSV no coverage: {missing}")
    print(f"  Images downloaded: {downloaded}")
    print(f"Manifest: {config.IMAGERY_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
