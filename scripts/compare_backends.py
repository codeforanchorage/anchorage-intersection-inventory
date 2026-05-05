"""Run multiple Phase-3 backends on the same intersection(s) and produce a
side-by-side report.

Usage:
  python scripts/compare_backends.py --intersection osm_31329116 \\
      --backends vision-only,roboflow,sam3 [--rerun] [--with-condition]

Backends that aren't configured (missing API key, missing workspace, SAM 3
license not granted, etc.) are skipped with a warning rather than failing the
whole run. Output:

  data/results/comparison/{intersection_id}/
      {backend}_detections.json     # per-backend raw detections
      {backend}_h{heading}.jpg      # annotated image (bboxes for sam3/roboflow,
                                    # text-overlay for vision-only)
      report.md                     # markdown comparison table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

import config
from src import detect_assets


# --- Per-backend runners ----------------------------------------------------

def run_vision_only(intersection_id: str, claude_model: str, claude_client) -> tuple[list[dict], float]:
    images = detect_assets.list_images_for(intersection_id)
    start = time.time()
    out = []
    for img_path in images:
        detections = detect_assets.process_image_vision_only(img_path, claude_client, claude_model)
        out.append({
            "image": img_path.name,
            "heading": detect_assets.heading_from_path(img_path),
            "detections": detections,
        })
    return out, time.time() - start


def run_roboflow(intersection_id: str, rf_model, confidence: int, overlap: int,
                 with_condition: bool, claude_model: str, claude_client) -> tuple[list[dict], float]:
    images = detect_assets.list_images_for(intersection_id)
    start = time.time()
    out = []
    for img_path in images:
        raw = detect_assets.detect_with_roboflow(rf_model, img_path, confidence=confidence, overlap=overlap)
        if with_condition and claude_client is not None:
            raw = [detect_assets.score_detection_with_claude(img_path, det, claude_client, claude_model)
                   for det in raw]
        for det in raw:
            det.setdefault("source", "roboflow+claude_crop" if with_condition else "roboflow")
            det.setdefault("condition", "not_assessable")
            det.setdefault("condition_notes", "")
        out.append({
            "image": img_path.name,
            "heading": detect_assets.heading_from_path(img_path),
            "detections": raw,
        })
    return out, time.time() - start


def run_sam3(intersection_id: str, predictor, prompts: list[dict],
             with_condition: bool, claude_model: str, claude_client) -> tuple[list[dict], float]:
    images = detect_assets.list_images_for(intersection_id)
    start = time.time()
    out = []
    for img_path in images:
        raw = detect_assets.detect_with_sam3(predictor, img_path, prompts)
        if with_condition and claude_client is not None:
            raw = [detect_assets.score_detection_with_claude(img_path, det, claude_client, claude_model)
                   for det in raw]
        for det in raw:
            det.setdefault("source", "sam3+claude_crop" if with_condition else "sam3")
            det.setdefault("condition", "not_assessable")
            det.setdefault("condition_notes", "")
        out.append({
            "image": img_path.name,
            "heading": detect_assets.heading_from_path(img_path),
            "detections": raw,
        })
    return out, time.time() - start


# --- Annotation -------------------------------------------------------------

PALETTE = {
    "traffic_signal_head": "#FF3B30",
    "pedestrian_signal":   "#FF9500",
    "signal_pole":         "#5856D6",
    "mast_arm":            "#34C759",
    "signal_cabinet":      "#AF52DE",
    "crosswalk_marking":   "#00C7BE",
    "curb_ramp":           "#FFCC00",
    "push_button":         "#BF5AF2",
    "street_light":        "#64D2FF",
    "road_sign":           "#FF2D55",
    "lane_marking":        "#30B0C7",
    "other":               "#8E8E93",
}


def _font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate_image(img_path: Path, image_entry: dict, out_path: Path, backend: str) -> None:
    """Overlay detections on the source image. Bboxes for sam3/roboflow,
    a text panel for vision-only."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(14)
    small = _font(11)

    detections = image_entry.get("detections", [])
    has_bbox = any("bbox" in det for det in detections)

    if has_bbox:
        for det in detections:
            bbox = det.get("bbox") or {}
            if "x1" in bbox:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            elif "x" in bbox:
                cx, cy = bbox["x"], bbox["y"]
                bw, bh = bbox["width"], bbox["height"]
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            else:
                continue
            color = PALETTE.get(det.get("asset_type", "other"), "#FFFFFF")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{det.get('asset_type', '?')} {det.get('confidence', 0):.2f}"
            tw = draw.textlength(label, font=small)
            draw.rectangle([x1, max(0, y1 - 16), x1 + tw + 6, y1], fill=color)
            draw.text((x1 + 3, max(0, y1 - 15)), label, fill="white", font=small)
    else:
        # Vision-only: no bboxes, draw a sidebar.
        sidebar_w = 220
        new = Image.new("RGB", (img.width + sidebar_w, img.height), (20, 20, 20))
        new.paste(img, (0, 0))
        draw = ImageDraw.Draw(new)
        y = 8
        draw.text((8 + img.width, y), f"{backend}", fill="white", font=font)
        y += 20
        for det in detections:
            atype = det.get("asset_type", "other")
            cond = det.get("condition", "?")
            loc = det.get("location_in_image", "")
            color = PALETTE.get(atype, "#FFFFFF")
            draw.rectangle([8 + img.width, y, 14 + img.width, y + 12], fill=color)
            txt = f"{atype}  ({cond})  {loc}"
            draw.text((20 + img.width, y - 1), txt[:30], fill="white", font=small)
            y += 14
            if y > img.height - 14:
                break
        img = new

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=85)


# --- Aggregation + report ---------------------------------------------------

def asset_counts(per_image: list[dict]) -> Counter:
    c: Counter = Counter()
    for img in per_image:
        for det in img.get("detections", []):
            c[det.get("asset_type", "other")] += 1
    return c


def write_report(intersection_id: str, results: dict[str, dict], out_dir: Path) -> Path:
    """results: {backend: {"per_image": [...], "elapsed_s": float, "skipped": str|None}}"""
    backends = list(results.keys())
    asset_types: list[str] = []
    counts_by_backend: dict[str, Counter] = {}
    for bk in backends:
        per_image = results[bk].get("per_image") or []
        counts = asset_counts(per_image)
        counts_by_backend[bk] = counts
        for at in counts:
            if at not in asset_types:
                asset_types.append(at)
    asset_types.sort()

    lines: list[str] = []
    lines.append(f"# Backend comparison — `{intersection_id}`\n")
    lines.append("## Run summary\n")
    lines.append("| Backend | Status | Total detections | Elapsed |")
    lines.append("|---|---|---|---|")
    for bk in backends:
        r = results[bk]
        if r.get("skipped"):
            lines.append(f"| {bk} | SKIPPED — {r['skipped']} | — | — |")
            continue
        total = sum(counts_by_backend[bk].values())
        lines.append(f"| {bk} | ok | {total} | {r['elapsed_s']:.1f}s |")
    lines.append("")

    if any(not r.get("skipped") for r in results.values()):
        lines.append("## Per-asset detection counts\n")
        header = "| asset_type | " + " | ".join(backends) + " |"
        sep = "|---|" + "|".join("---" for _ in backends) + "|"
        lines.append(header)
        lines.append(sep)
        for at in asset_types:
            row = [at] + [
                str(counts_by_backend[bk].get(at, 0)) if not results[bk].get("skipped") else "—"
                for bk in backends
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Per-image breakdown\n")
    headings = sorted({img["heading"]
                       for r in results.values() if not r.get("skipped")
                       for img in r["per_image"]})
    for h in headings:
        lines.append(f"### Heading {h}°\n")
        lines.append("| Backend | Detections | Top assets |")
        lines.append("|---|---|---|")
        for bk in backends:
            r = results[bk]
            if r.get("skipped"):
                continue
            for img in r["per_image"]:
                if img["heading"] == h:
                    n = len(img["detections"])
                    top = Counter(d.get("asset_type", "other") for d in img["detections"]).most_common(3)
                    top_str = ", ".join(f"{a}×{c}" for a, c in top)
                    lines.append(f"| {bk} | {n} | {top_str} |")
                    break
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# --- CLI --------------------------------------------------------------------

def maybe_load_anthropic():
    load_dotenv(config.PROJECT_ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None, "ANTHROPIC_API_KEY missing"
    from anthropic import Anthropic
    return Anthropic(api_key=key), None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Phase-3 backends on one or more intersections.")
    parser.add_argument("--intersection", required=True,
                        help="osm_id, e.g. osm_31329116")
    parser.add_argument("--backends", default="vision-only",
                        help="Comma-separated list: sam3,roboflow,vision-only")
    parser.add_argument("--with-condition", action="store_true",
                        help="For sam3/roboflow: send each crop to Claude for condition.")
    parser.add_argument("--rerun", action="store_true",
                        help="Reprocess even if a cached comparison file exists.")
    parser.add_argument("--annotate", action="store_true",
                        help="Save annotated overlay images per backend per heading.")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--rf-overlap", type=int, default=30)
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not detect_assets.list_images_for(args.intersection):
        print(f"ERROR: no images found for {args.intersection}. Run Phase 2 first.")
        return 1

    out_dir = config.RESULTS_DIR / "comparison" / args.intersection
    out_dir.mkdir(parents=True, exist_ok=True)

    claude_client, claude_err = maybe_load_anthropic()
    needs_claude = "vision-only" in backends or args.with_condition
    if needs_claude and claude_client is None:
        print(f"ERROR: Claude needed but {claude_err}")
        return 1

    results: dict[str, dict] = {}

    for bk in backends:
        cache_path = out_dir / f"{bk}_detections.json"
        if cache_path.exists() and not args.rerun:
            with cache_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            results[bk] = {
                "per_image": doc["images"],
                "elapsed_s": doc.get("elapsed_s", 0.0),
                "skipped": None,
                "from_cache": True,
            }
            print(f"[{bk}] cached ({sum(len(i['detections']) for i in doc['images'])} detections)")
            continue

        try:
            if bk == "vision-only":
                per_image, elapsed = run_vision_only(args.intersection, args.claude_model, claude_client)
            elif bk == "roboflow":
                rf_key = os.environ.get("ROBOFLOW_API_KEY")
                rf_ws = os.environ.get("ROBOFLOW_WORKSPACE", "")
                rf_proj = os.environ.get("ROBOFLOW_PROJECT", "")
                rf_ver = int(os.environ.get("ROBOFLOW_VERSION", "0") or 0)
                if not (rf_key and rf_ws and rf_proj and rf_ver):
                    raise RuntimeError(
                        "Roboflow needs ROBOFLOW_API_KEY + WORKSPACE + PROJECT + VERSION in .env"
                    )
                rf_model = detect_assets.load_roboflow_model(rf_key, rf_ws, rf_proj, rf_ver)
                conf_pct = int(args.confidence * 100) if args.confidence < 1 else int(args.confidence)
                per_image, elapsed = run_roboflow(
                    args.intersection, rf_model, conf_pct, args.rf_overlap,
                    args.with_condition, args.claude_model, claude_client,
                )
            elif bk == "sam3":
                predictor = detect_assets.init_sam3_predictor(default_confidence=args.confidence)
                prompts = detect_assets.load_prompts(detect_assets.DEFAULT_PROMPTS_PATH)
                per_image, elapsed = run_sam3(
                    args.intersection, predictor, prompts,
                    args.with_condition, args.claude_model, claude_client,
                )
            else:
                raise ValueError(f"unknown backend: {bk}")
        except Exception as exc:
            print(f"[{bk}] SKIPPED: {exc}")
            results[bk] = {"per_image": [], "elapsed_s": 0.0, "skipped": str(exc)}
            continue

        results[bk] = {"per_image": per_image, "elapsed_s": elapsed, "skipped": None, "from_cache": False}
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump({
                "intersection_id": args.intersection,
                "backend": bk,
                "elapsed_s": elapsed,
                "images": per_image,
            }, f, indent=2)
        print(f"[{bk}] ran fresh in {elapsed:.1f}s "
              f"({sum(len(i['detections']) for i in per_image)} detections)")

    if args.annotate:
        for bk, r in results.items():
            if r.get("skipped"):
                continue
            for img_entry in r["per_image"]:
                src = config.IMAGE_DIR / img_entry["image"]
                if not src.exists():
                    continue
                heading = img_entry["heading"]
                dst = out_dir / f"{bk}_h{heading}.jpg"
                annotate_image(src, img_entry, dst, bk)
        print(f"Annotated images written to {out_dir}")

    report_path = write_report(args.intersection, results, out_dir)
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
