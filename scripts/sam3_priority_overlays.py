"""Run SAM 3 on the top-N priority intersections and write bbox-annotated
overlays alongside the existing Phase 6 priority images.

Vision-only Phase 6 produces sidebar-style visualizations because vision-only
doesn't return bboxes. This script supplements that by running SAM 3 on the
same priority headings and writing per-asset bbox overlays — the kind of
visual evidence Traffic Engineering will reach for first.

Output:
  data/results/priority/{osm_id}_h{heading}_sam3.jpg  # bbox overlay alongside
                                                      # the existing _sidebar
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

import config
from src import detect_assets, visualize_priority


PRIORITY_DIR = config.RESULTS_DIR / "priority"
INDEX_PATH = PRIORITY_DIR / "index.md"


def parse_top_n(index_path: Path, n: int) -> list[tuple[str, int]]:
    """Pull (osm_id, heading) pairs in priority order from the Phase 6 index."""
    if not index_path.exists():
        raise FileNotFoundError(f"Phase 6 index not found at {index_path}. Run Phase 6 first.")
    text = index_path.read_text(encoding="utf-8")
    pattern = re.compile(r"!\[(?P<osm>osm_\d+)_h(?P<h>\d+)\.jpg\]")
    seen: set[str] = set()
    out: list[tuple[str, int]] = []
    for m in pattern.finditer(text):
        osm = m.group("osm")
        if osm in seen:
            continue
        seen.add(osm)
        out.append((osm, int(m.group("h"))))
        if len(out) >= n:
            break
    return out


def annotate_sam3_overlay(src: Path, detections: list[dict], out: Path,
                          intersection_id: str, heading: int) -> None:
    """Draw bboxes color-coded by asset_type. No banner — companion to the
    existing Phase 6 sidebar image which already lists findings."""
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    palette = {
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

    for det in detections:
        bb = det.get("bbox") or {}
        if "x1" not in bb:
            continue
        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        atype = det.get("asset_type", "other")
        color = palette.get(atype, "#FFFFFF")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{atype} {det.get('confidence', 0):.2f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x1, max(0, y1 - 16), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - 15)), label, fill="white", font=font)

    title_h = 22
    composed = Image.new("RGB", (img.width, img.height + title_h), (20, 20, 20))
    composed.paste(img, (0, title_h))
    tdraw = ImageDraw.Draw(composed)
    title = f"SAM 3 bbox overlay  ·  {intersection_id}  ·  heading {heading}°"
    tdraw.text((8, 4), title, fill="white", font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    composed.save(out, format="JPEG", quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 3 overlays for top-N priority intersections.")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of priority intersections to overlay (default 10).")
    args = parser.parse_args()

    targets = parse_top_n(INDEX_PATH, args.top)
    if not targets:
        print("ERROR: no targets parsed from priority/index.md")
        return 1
    print(f"Targeting top {len(targets)} priority intersections")

    print("Loading SAM 3 (one-time)...")
    predictor = detect_assets.init_sam3_predictor(default_confidence=0.25)
    prompts = detect_assets.load_prompts(detect_assets.DEFAULT_PROMPTS_PATH)
    print(f"  {len(prompts)} prompts loaded")

    n_done = 0
    n_skipped = 0
    started = time.time()
    for osm_id, heading in targets:
        src = config.IMAGE_DIR / f"{osm_id}_h{heading}.jpg"
        if not src.exists():
            print(f"  [{osm_id} h{heading}] no GSV image — skipping")
            n_skipped += 1
            continue
        out = PRIORITY_DIR / f"{osm_id}_h{heading}_sam3.jpg"
        t0 = time.time()
        detections = detect_assets.detect_with_sam3(predictor, src, prompts)
        annotate_sam3_overlay(src, detections, out, osm_id, heading)
        elapsed = time.time() - t0
        print(f"  [{osm_id} h{heading}] {len(detections)} detections in {elapsed:.1f}s -> {out.name}")
        n_done += 1

    print(f"\nDone. {n_done} overlays written, {n_skipped} skipped, total {time.time() - started:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
