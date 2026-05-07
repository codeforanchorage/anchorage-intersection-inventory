"""Phase 2: Download Google Street View imagery for each intersection.

Files are stored as ``{osm_id}_h{heading}_p{pitch}.jpg`` so multiple pitches
per heading can coexist. Re-runs are idempotent: if a file already exists on
disk it is not re-downloaded, and the manifest row is preserved with status
``SKIP_EXISTS``. Pano-id and date are cached from the prior manifest where
available so we skip the metadata round-trip for already-covered intersections.

Pass ``--refresh-metadata`` to re-query GSV's metadata endpoint for every
intersection (free) and auto-replace images for any intersection where the
pano_id has changed since the last run (Google publishes fresher coverage
periodically; without this flag stale panos are kept indefinitely).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


METADATA_STALENESS_WARN_DAYS = 180


def load_intersections(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    return fc.get("features", [])


def load_existing_metadata(manifest_path: Path) -> dict[str, dict]:
    """Return {intersection_id: {pano_id, date, status, fetched_at}} from a prior manifest.

    Lets reruns avoid hitting the GSV metadata endpoint for intersections we
    already know are covered (or known-missing). ``fetched_at`` lets the caller
    detect stale metadata; missing values are returned as empty strings for
    backward compatibility with older manifests that didn't track it.
    """
    out: dict[str, dict] = {}
    if not manifest_path.exists():
        return out
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xid = row["intersection_id"]
            if xid in out:
                # Prefer the first OK row's metadata; otherwise keep what's there.
                if out[xid].get("status") == "OK":
                    continue
            out[xid] = {
                "pano_id": row.get("gsv_pano_id", "") or "",
                "date": row.get("gsv_date", "") or "",
                "status": row.get("status", "") or "",
                "fetched_at": row.get("metadata_fetched_at", "") or "",
            }
    return out


def load_existing_rows(manifest_path: Path) -> dict[str, list[dict]]:
    """Group every row in a prior manifest by intersection_id.

    Used to preserve un-touched intersections when a run only processes a
    subset (e.g. ``--limit N``). Without this, every limited run would
    truncate the manifest to just the processed intersections.
    """
    out: dict[str, list[dict]] = {}
    if not manifest_path.exists():
        return out
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["intersection_id"], []).append(row)
    return out


def median_metadata_age_days(cached_meta: dict[str, dict]) -> float | None:
    """Median age in days of cached `metadata_fetched_at` timestamps. None if absent."""
    now = dt.datetime.now(dt.timezone.utc)
    ages: list[float] = []
    for entry in cached_meta.values():
        ts = entry.get("fetched_at")
        if not ts:
            continue
        try:
            fetched = dt.datetime.fromisoformat(ts)
        except ValueError:
            continue
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=dt.timezone.utc)
        ages.append((now - fetched).total_seconds() / 86400.0)
    if not ages:
        return None
    ages.sort()
    n = len(ages)
    return ages[n // 2] if n % 2 else 0.5 * (ages[n // 2 - 1] + ages[n // 2])


def metadata_for(lat: float, lon: float, key: str) -> dict:
    """Hit the GSV metadata endpoint to confirm imagery exists. Free and unmetered."""
    params = {"location": f"{lat},{lon}", "key": key}
    response = requests.get(config.GSV_METADATA_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def download_image(lat: float, lon: float, heading: int, pitch: int,
                   key: str, out_path: Path) -> None:
    params = {
        "size": config.GSV_SIZE,
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
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
    refresh_metadata: bool = False,
) -> tuple[int, int, int, int, int]:
    """Returns (covered, missing, downloaded, skipped_existing, panos_changed)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Read prior manifest BEFORE any writes touch the target file.
    cached_meta = load_existing_metadata(manifest_path)
    existing_rows = load_existing_rows(manifest_path)
    covered = missing = downloaded = skipped_existing = panos_changed = 0
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    fieldnames = [
        "intersection_id", "lat", "lon", "heading", "pitch",
        "image_path", "gsv_pano_id", "gsv_date", "metadata_fetched_at", "status",
    ]
    processed_ids: set[str] = set()
    # Write to a temp file in the same directory; rename on success so a mid-run
    # crash doesn't destroy the existing manifest (and its cached pano metadata).
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=manifest_path.stem + ".", suffix=".csv.tmp",
        dir=str(manifest_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for feat in features:
            props = feat["properties"]
            osm_id = props["osm_id"]
            processed_ids.add(osm_id)
            lon, lat = feat["geometry"]["coordinates"]

            # Reuse prior manifest metadata when present; otherwise hit the API.
            cached = cached_meta.get(osm_id)
            use_cache = (
                cached is not None
                and cached.get("status") in {"OK", "ZERO_RESULTS", "NOT_FOUND"}
                and not refresh_metadata
            )
            if use_cache:
                meta = {
                    "status": cached["status"],
                    "pano_id": cached["pano_id"],
                    "date": cached["date"],
                }
                fetched_at = cached.get("fetched_at") or ""
            else:
                try:
                    meta = request_with_backoff(metadata_for, lat, lon, api_key)
                except Exception as exc:
                    print(f"  {osm_id}: metadata error: {exc}")
                    meta = {"status": "ERROR"}
                fetched_at = now_iso
                time.sleep(config.GSV_RATE_LIMIT_SEC)

            status = meta.get("status", "UNKNOWN")
            pano_id = meta.get("pano_id", "")
            date = meta.get("date", "")

            # Detect a pano change on refresh: if the fresh metadata returned a
            # different pano_id than the cache had, the existing image files on
            # disk were captured from the now-stale pano. Delete them so the
            # download loop below grabs fresh ones.
            pano_changed = (
                refresh_metadata
                and cached is not None
                and cached.get("pano_id")
                and pano_id
                and cached["pano_id"] != pano_id
            )
            if pano_changed:
                panos_changed += 1
                print(
                    f"  {osm_id}: pano changed "
                    f"({cached['pano_id'][:10]}.../{cached.get('date') or '?'}"
                    f" -> {pano_id[:10]}.../{date or '?'}) — replacing images"
                )
                for h in config.GSV_HEADINGS:
                    for p in config.GSV_PITCHES:
                        stale = config.IMAGE_DIR / f"{osm_id}_h{h}_p{p}.jpg"
                        if stale.exists():
                            stale.unlink()

            if status != "OK":
                missing += 1
                writer.writerow({
                    "intersection_id": osm_id, "lat": lat, "lon": lon,
                    "heading": "", "pitch": "", "image_path": "",
                    "gsv_pano_id": "", "gsv_date": "",
                    "metadata_fetched_at": fetched_at, "status": status,
                })
                continue

            covered += 1
            for heading in config.GSV_HEADINGS:
                for pitch in config.GSV_PITCHES:
                    image_path = config.IMAGE_DIR / f"{osm_id}_h{heading}_p{pitch}.jpg"
                    if dry_run:
                        img_str = ""
                        img_status = "DRY_RUN"
                    elif image_path.exists():
                        img_str = str(image_path.relative_to(config.PROJECT_ROOT))
                        img_status = "SKIP_EXISTS"
                        skipped_existing += 1
                    else:
                        try:
                            request_with_backoff(
                                download_image, lat, lon, heading, pitch,
                                api_key, image_path,
                            )
                            downloaded += 1
                            img_str = str(image_path.relative_to(config.PROJECT_ROOT))
                            img_status = "OK"
                        except Exception as exc:
                            print(f"  {osm_id} h{heading} p{pitch}: download error: {exc}")
                            img_str = ""
                            img_status = "DOWNLOAD_ERROR"
                        time.sleep(config.GSV_RATE_LIMIT_SEC)

                    writer.writerow({
                        "intersection_id": osm_id, "lat": lat, "lon": lon,
                        "heading": heading, "pitch": pitch, "image_path": img_str,
                        "gsv_pano_id": pano_id, "gsv_date": date,
                        "metadata_fetched_at": fetched_at, "status": img_status,
                    })

        # Carry over un-touched intersections so --limit doesn't truncate.
        for xid, rows in existing_rows.items():
            if xid in processed_ids:
                continue
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Atomic publish — replace the old manifest only after the new one is fully
    # written. os.replace is atomic on Windows when source and destination are
    # in the same directory.
    os.replace(tmp_path, manifest_path)
    return covered, missing, downloaded, skipped_existing, panos_changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: download GSV imagery.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check coverage only; do not download images.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N intersections.")
    parser.add_argument("--refresh-metadata", action="store_true",
                        help="Bypass the cache and re-query GSV's metadata "
                             "endpoint for every intersection (free). When a "
                             "pano_id has changed, stale image files for that "
                             "intersection are deleted so they re-download.")
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

    # Stale-metadata heads-up: the median age of cached metadata in the manifest.
    if config.IMAGERY_MANIFEST.exists() and not args.refresh_metadata:
        existing = load_existing_metadata(config.IMAGERY_MANIFEST)
        age_days = median_metadata_age_days(existing)
        if age_days is not None and age_days >= METADATA_STALENESS_WARN_DAYS:
            print(f"WARNING: cached GSV metadata is ~{age_days:.0f} days old "
                  f"(threshold {METADATA_STALENESS_WARN_DAYS}). "
                  f"Consider rerunning with --refresh-metadata to detect new panos.")

    print(f"Processing {len(features)} intersections "
          f"(dry_run={args.dry_run}, headings={config.GSV_HEADINGS}, "
          f"pitches={config.GSV_PITCHES}, refresh_metadata={args.refresh_metadata})...")
    covered, missing, downloaded, skipped, panos_changed = process_intersections(
        features, api_key, args.dry_run, config.IMAGERY_MANIFEST,
        refresh_metadata=args.refresh_metadata,
    )
    print(f"  GSV coverage OK: {covered}")
    print(f"  GSV no coverage: {missing}")
    print(f"  Images downloaded: {downloaded}")
    print(f"  Images skipped (already on disk): {skipped}")
    if args.refresh_metadata:
        print(f"  Panos changed (images re-downloaded): {panos_changed}")
    print(f"Manifest: {config.IMAGERY_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
