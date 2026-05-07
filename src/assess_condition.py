"""Phase 4: Aggregate per-image detections into per-intersection asset inventory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


CONDITION_RANK = {"good": 3, "fair": 2, "poor": 1, "not_assessable": 0}
RANK_TO_LABEL = {3: "good", 2: "fair", 1: "poor", 0: "not_assessable"}


def score(condition: str) -> float | None:
    return config.CONDITION_SCORES.get(condition)


def worst_condition(conditions: list[str]) -> str:
    ranked = [c for c in conditions if c in CONDITION_RANK and c != "not_assessable"]
    if not ranked:
        return "not_assessable"
    worst_rank = min(CONDITION_RANK[c] for c in ranked)
    return RANK_TO_LABEL[worst_rank]


def load_imagery_manifest() -> dict[str, dict]:
    """Map intersection_id -> {pano_id, gsv_date, covered: bool, images: [paths]}."""
    out: dict[str, dict] = {}
    if not config.IMAGERY_MANIFEST.exists():
        return out
    with config.IMAGERY_MANIFEST.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xid = row["intersection_id"]
            entry = out.setdefault(xid, {
                "gsv_pano_id": "", "gsv_date": "",
                "covered": False, "images": [],
            })
            if row["status"] == "OK" or row["status"] == "DRY_RUN":
                entry["covered"] = True
            if row["gsv_pano_id"]:
                entry["gsv_pano_id"] = row["gsv_pano_id"]
            if row["gsv_date"]:
                entry["gsv_date"] = row["gsv_date"]
            if row["image_path"]:
                entry["images"].append(row["image_path"])
    return out


def aggregate_intersection(detections_doc: dict) -> dict:
    """Collapse per-image detections into per-asset summaries.

    Multi-pitch dedup: the same physical asset (e.g. a mast arm) appears in
    every pitch image of a given heading. Counting naively triples the count.
    Strategy: group by (heading, pitch, asset_type), then for each
    (heading, asset_type) take the pitch with the max count as the canonical
    observation; sum canonical counts across headings. Conditions, scores, and
    notes are unioned across all pitches for an asset so we don't lose a
    'poor' observation that only one pitch caught.
    """
    # Per (heading, pitch, asset_type) bucket.
    by_h_p_a: dict[tuple, dict] = defaultdict(lambda: {
        "count": 0,
        "scores": [],
        "conditions": [],
        "notes": [],
    })
    for img in detections_doc.get("images", []):
        heading = img.get("heading", 0)
        pitch = img.get("pitch", 10)
        for det in img.get("detections", []):
            asset = det.get("asset_type", "other") or "other"
            cond = det.get("condition", "not_assessable")
            note = det.get("condition_notes", "")
            entry = by_h_p_a[(heading, pitch, asset)]
            entry["count"] += 1
            entry["conditions"].append(cond)
            s = score(cond)
            if s is not None:
                entry["scores"].append(s)
            if note and note.lower() not in {"none", "no issues observed", ""}:
                entry["notes"].append(note)

    # Reshape: per (asset, heading) -> list of (pitch, entry) across all pitches.
    by_a_h: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for (heading, pitch, asset), entry in by_h_p_a.items():
        by_a_h[(asset, heading)].append((pitch, entry))

    # Aggregate per asset: canonical count per heading, union conditions.
    by_asset: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "scores": [],
        "conditions": [],
        "notes": [],
        "headings_seen": set(),
        "pitches_seen": set(),
        "worst_heading": None,
        "worst_heading_rank": 4,  # higher than any CONDITION_RANK + 1
        "best_pitch": None,
        "best_pitch_count": -1,
    })
    for (asset, heading), pitch_entries in by_a_h.items():
        canonical_pitch, canonical_entry = max(pitch_entries, key=lambda pe: pe[1]["count"])
        canonical_count = canonical_entry["count"]
        out = by_asset[asset]
        out["count"] += canonical_count
        out["headings_seen"].add(heading)
        for pitch, entry in pitch_entries:
            out["pitches_seen"].add(pitch)
            out["scores"].extend(entry["scores"])
            out["conditions"].extend(entry["conditions"])
            out["notes"].extend(entry["notes"])
        # Track which heading has the worst-condition observation for this asset.
        heading_worst = worst_condition(
            [c for _, e in pitch_entries for c in e["conditions"]]
        )
        rank = CONDITION_RANK.get(heading_worst, 0)
        # Lower rank = worse condition (1=poor); track the heading with the worst.
        # Skip not_assessable (rank 0) unless nothing else has been seen.
        is_real = heading_worst in CONDITION_RANK and heading_worst != "not_assessable"
        if is_real and rank < out["worst_heading_rank"]:
            out["worst_heading_rank"] = rank
            out["worst_heading"] = heading
        elif out["worst_heading"] is None:
            out["worst_heading"] = heading
        # Best pitch for this asset = pitch that produced the most detections at
        # any heading (ties broken by larger pitch — 25/45 imply discovery).
        if (canonical_count > out["best_pitch_count"]
                or (canonical_count == out["best_pitch_count"]
                    and (out["best_pitch"] or 0) < canonical_pitch)):
            out["best_pitch_count"] = canonical_count
            out["best_pitch"] = canonical_pitch

    assets = []
    weighted_scores: list[float] = []
    priority_issues: list[str] = []
    for asset_type, info in by_asset.items():
        avg = sum(info["scores"]) / len(info["scores"]) if info["scores"] else None
        worst = worst_condition(info["conditions"])
        # Dedup notes while preserving order.
        seen, deduped = set(), []
        for n in info["notes"]:
            if n not in seen:
                seen.add(n)
                deduped.append(n)
        assets.append({
            "asset_type": asset_type,
            "count": info["count"],
            "avg_condition_score": round(avg, 3) if avg is not None else None,
            "worst_condition": worst,
            "worst_heading": info["worst_heading"],
            "best_pitch": info["best_pitch"],
            "headings_seen": sorted(info["headings_seen"]),
            "pitches_seen": sorted(info["pitches_seen"]),
            "notes": deduped,
        })
        if avg is not None:
            weighted_scores.append(avg)
        if worst == "poor":
            priority_issues.append(f"poor {asset_type}")

    overall = round(sum(weighted_scores) / len(weighted_scores), 3) if weighted_scores else None
    return {
        "assets": assets,
        "overall_condition_score": overall,
        "priority_issues": priority_issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4: aggregate and score conditions.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if not config.SIGNALS_GEOJSON.exists():
        print(f"ERROR: {config.SIGNALS_GEOJSON} not found.")
        return 1
    with config.SIGNALS_GEOJSON.open("r", encoding="utf-8") as f:
        features = json.load(f).get("features", [])
    if args.limit is not None:
        features = features[: args.limit]

    manifest = load_imagery_manifest()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_written = n_no_detect = 0

    for feat in features:
        osm_id = feat["properties"]["osm_id"]
        det_path = config.RESULTS_DIR / f"{osm_id}_detections.json"
        if not det_path.exists():
            n_no_detect += 1
            continue
        with det_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        agg = aggregate_intersection(doc)
        manifest_entry = manifest.get(osm_id, {})
        record = {
            "intersection_id": osm_id,
            "lat": feat["properties"]["lat"],
            "lon": feat["properties"]["lon"],
            "cluster_size": feat["properties"].get("cluster_size", 1),
            "gsv_coverage": manifest_entry.get("covered", False),
            "gsv_date": manifest_entry.get("gsv_date", ""),
            "gsv_pano_id": manifest_entry.get("gsv_pano_id", ""),
            "images": [Path(p).name for p in manifest_entry.get("images", [])],
            **agg,
        }
        out_path = config.RESULTS_DIR / f"{osm_id}_inventory.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        n_written += 1

    print(f"Inventory records written: {n_written}  missing detections: {n_no_detect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
