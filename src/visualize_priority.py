"""Phase 6: Generate visual evidence of each intersection's worst heading.

For every intersection with detection data, this picks the heading with the
most safety-relevant findings (poor + fair on critical assets) and writes:

  data/results/priority/
      {osm_id}_h{heading}_p{pitch}.jpg     # annotated GSV with severity-coded labels
      index.md                             # one-page ranked report linking each
                                           # intersection to its visualization

Severity tiers:
  poor             → red highlight  (immediate attention)
  fair on critical → orange         (crosswalks, curb ramps, push buttons —
                                     safety items that fade to dangerous)
  not_assessable   → blue           (visibility issue — may need closer survey)
  good / fair other → muted gray    (context only)

Critical asset types (where 'fair' still warrants surfacing):
  crosswalk_marking, lane_marking, curb_ramp, push_button, pedestrian_signal
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


PRIORITY_DIR = config.RESULTS_DIR / "priority"

CRITICAL_ASSETS = {
    "crosswalk_marking", "lane_marking", "curb_ramp",
    "push_button", "pedestrian_signal",
}

# Severity weights — higher = ranks higher in the priority list.
SEVERITY_WEIGHTS = {
    "poor": 10,
    "fair_critical": 3,
    "not_assessable_critical": 1,
}

# Color palette per severity (RGB).
COLOR_POOR = (255, 59, 48)
COLOR_FAIR_CRIT = (255, 149, 0)
COLOR_NA_CRIT = (90, 200, 250)
COLOR_MUTED = (140, 140, 140)
COLOR_BG = (20, 20, 20)
COLOR_FG = (255, 255, 255)


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def severity_of(det: dict) -> str:
    """Return one of: 'poor', 'fair_critical', 'not_assessable_critical', 'context'."""
    cond = det.get("condition", "not_assessable")
    atype = det.get("asset_type", "other")
    if cond == "poor":
        return "poor"
    if cond == "fair" and atype in CRITICAL_ASSETS:
        return "fair_critical"
    if cond == "not_assessable" and atype in CRITICAL_ASSETS:
        return "not_assessable_critical"
    return "context"


SEV_LABEL_SHORT = {
    "poor": "POOR",
    "fair_critical": "FAIR_CRIT",
    "not_assessable_critical": "NAc",
}


def color_of(severity: str) -> tuple[int, int, int]:
    return {
        "poor": COLOR_POOR,
        "fair_critical": COLOR_FAIR_CRIT,
        "not_assessable_critical": COLOR_NA_CRIT,
        "context": COLOR_MUTED,
    }[severity]


def heading_severity_score(detections: list[dict]) -> tuple[int, Counter]:
    """Return (priority_score, severity_counts) for one heading's detections."""
    counts: Counter = Counter()
    score = 0
    for det in detections:
        sev = severity_of(det)
        counts[sev] += 1
        if sev != "context":
            score += SEVERITY_WEIGHTS.get(
                "poor" if sev == "poor"
                else "fair_critical" if sev == "fair_critical"
                else "not_assessable_critical", 0,
            )
    return score, counts


def _bbox_xyxy(det: dict) -> tuple[float, float, float, float] | None:
    bbox = det.get("bbox")
    if not bbox:
        return None
    if "x1" in bbox:
        return bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    if "x" in bbox:
        cx, cy = bbox["x"], bbox["y"]
        bw, bh = bbox["width"], bbox["height"]
        return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    return None


def annotate_heading(src_path: Path, detections: list[dict], out_path: Path,
                     intersection_id: str, heading: int, pitch: int,
                     severity_counts: Counter) -> None:
    img = Image.open(src_path).convert("RGB")
    has_bbox = any(_bbox_xyxy(d) is not None for d in detections)

    if has_bbox:
        draw = ImageDraw.Draw(img, "RGBA")
        # Draw context detections first (faint), then critical ones on top.
        ordered = sorted(detections, key=lambda d: 0 if severity_of(d) == "context" else 1)
        for det in ordered:
            xy = _bbox_xyxy(det)
            if xy is None:
                continue
            sev = severity_of(det)
            color = color_of(sev)
            x1, y1, x2, y2 = xy
            # Clamp to image bounds — SAM 3 occasionally returns bboxes that
            # extend past the visible frame (overhead assets at high pitch).
            x1 = max(0, min(x1, img.width - 1))
            x2 = max(x1 + 1, min(x2, img.width))
            y1 = max(0, min(y1, img.height - 1))
            y2 = max(y1 + 1, min(y2, img.height))
            xy_clamped = (x1, y1, x2, y2)
            if sev == "context":
                draw.rectangle(xy_clamped, outline=color + (140,), width=1)
            else:
                draw.rectangle(xy_clamped, outline=color, width=4)
                label = f"{SEV_LABEL_SHORT.get(sev, sev.upper())} · {det.get('asset_type', '?')}"
                font = _font(13)
                tw = draw.textlength(label, font=font)
                # Place the label above the bbox if there's room, else below.
                if y1 >= 18:
                    label_top, label_bot, text_y = y1 - 18, y1, y1 - 17
                else:
                    label_top, label_bot, text_y = y2, min(img.height, y2 + 18), y2 + 1
                draw.rectangle([x1, label_top, x1 + tw + 8, label_bot], fill=color)
                draw.text((x1 + 4, text_y), label, fill="white", font=font)

    # Banner along the bottom: title + severity-coded findings list.
    notable = [d for d in detections if severity_of(d) != "context"]
    banner_h = 48 + 18 * max(1, len(notable))
    banner_h = min(banner_h, max(64, img.height // 2))
    new = Image.new("RGB", (img.width, img.height + banner_h), COLOR_BG)
    new.paste(img, (0, 0))
    bdraw = ImageDraw.Draw(new)

    title = f"{intersection_id}  ·  heading {heading}°  ·  pitch {pitch}°"
    bdraw.text((8, img.height + 4), title, fill=COLOR_FG, font=_font(15))
    summary_parts = []
    if severity_counts.get("poor"):
        summary_parts.append(f"{severity_counts['poor']} poor")
    if severity_counts.get("fair_critical"):
        summary_parts.append(f"{severity_counts['fair_critical']} fair-critical")
    if severity_counts.get("not_assessable_critical"):
        summary_parts.append(f"{severity_counts['not_assessable_critical']} unassessable-critical")
    summary = "  ·  ".join(summary_parts) or "no priority findings"
    bdraw.text((8, img.height + 22), summary, fill=COLOR_MUTED, font=_font(12))

    y = img.height + 44
    max_y = img.height + banner_h - 4
    for det in notable:
        sev = severity_of(det)
        atype = det.get("asset_type", "?")
        loc = det.get("location_in_image", "")
        loc_str = f" [{loc}]" if loc else ""
        notes = det.get("condition_notes") or ""
        text = f"• {atype}{loc_str}: {notes}"
        max_chars = max(40, img.width // 7)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        # Color swatch for the severity.
        bdraw.rectangle([8, y + 2, 14, y + 12], fill=color_of(sev))
        bdraw.text((20, y - 1), text, fill=COLOR_FG, font=_font(11))
        y += 18
        if y > max_y:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new.save(out_path, format="JPEG", quality=88)


def write_index(rankings: list[dict], out_path: Path) -> None:
    lines = ["# Priority findings — Anchorage intersection inventory", ""]
    if not rankings:
        lines.append("_No detection data available._")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    n_with_priority = sum(1 for r in rankings if r["score"] > 0)
    lines.append(f"**{n_with_priority}** of {len(rankings)} intersections have at least "
                 f"one safety-relevant finding worth Traffic Engineering review.\n")
    lines.append("Sorted by priority score (poor=10, fair-on-critical=3, "
                 "unassessable-on-critical=1).\n")

    lines.append("## Ranked findings\n")
    for i, r in enumerate(rankings, 1):
        osm_id = r["intersection_id"]
        score = r["score"]
        heading = r["heading"]
        pitch = r.get("pitch", 10)
        counts = r["severity_counts"]
        img_name = f"{osm_id}_h{heading}_p{pitch}.jpg"
        lat = r.get("lat")
        lon = r.get("lon")
        gmap = (f"[Open in Google Maps](https://www.google.com/maps?q={lat},{lon})"
                if lat is not None and lon is not None else "")
        lines.append(f"### {i}. {osm_id}  ·  score {score}")
        if gmap:
            lines.append(f"`{lat:.5f}, {lon:.5f}` — {gmap}")
        breakdown = []
        if counts.get("poor"):
            breakdown.append(f"{counts['poor']} poor")
        if counts.get("fair_critical"):
            breakdown.append(f"{counts['fair_critical']} fair-critical")
        if counts.get("not_assessable_critical"):
            breakdown.append(f"{counts['not_assessable_critical']} unassessable-critical")
        if breakdown:
            lines.append("Severity: " + " · ".join(breakdown))
        lines.append("")
        lines.append(f"![{img_name}]({img_name})")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6: priority visualization.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top", type=int, default=None,
                        help="Only render the top N intersections by priority score.")
    args = parser.parse_args(argv)

    if not config.SIGNALS_GEOJSON.exists():
        print(f"ERROR: {config.SIGNALS_GEOJSON} not found.")
        return 1
    with config.SIGNALS_GEOJSON.open("r", encoding="utf-8") as f:
        features = json.load(f).get("features", [])
    if args.limit is not None:
        features = features[: args.limit]

    PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
    rankings = []

    for feat in features:
        osm_id = feat["properties"]["osm_id"]
        det_path = config.RESULTS_DIR / f"{osm_id}_detections.json"
        if not det_path.exists():
            continue
        with det_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        # Score each (heading, pitch) image; pick the worst.
        worst_heading = None
        worst_pitch = None
        worst_score = -1
        worst_counts: Counter = Counter()
        for img_entry in doc.get("images", []):
            score, counts = heading_severity_score(img_entry.get("detections", []))
            if score > worst_score:
                worst_score = score
                worst_heading = img_entry.get("heading")
                worst_pitch = img_entry.get("pitch", 10)
                worst_counts = counts
        if worst_heading is None:
            continue

        rankings.append({
            "intersection_id": osm_id,
            "heading": worst_heading,
            "pitch": worst_pitch,
            "score": worst_score,
            "severity_counts": worst_counts,
            "lat": feat["properties"].get("lat"),
            "lon": feat["properties"].get("lon"),
            "_doc": doc,
        })

    rankings.sort(key=lambda r: -r["score"])
    if args.top is not None:
        rankings = rankings[: args.top]

    for r in rankings:
        osm_id = r["intersection_id"]
        heading = r["heading"]
        pitch = r.get("pitch", 10)
        img_name = f"{osm_id}_h{heading}_p{pitch}.jpg"
        src = config.IMAGE_DIR / img_name
        if not src.exists():
            continue
        # Find the detections for this (heading, pitch).
        heading_dets: list[dict] = []
        for img_entry in r["_doc"].get("images", []):
            if (img_entry.get("heading") == heading
                    and img_entry.get("pitch", 10) == pitch):
                heading_dets = img_entry.get("detections", [])
                break
        out = PRIORITY_DIR / img_name
        annotate_heading(src, heading_dets, out, osm_id, heading, pitch, r["severity_counts"])

    # Drop the in-memory doc reference before writing the index.
    for r in rankings:
        r.pop("_doc", None)

    index_path = PRIORITY_DIR / "index.md"
    write_index(rankings, index_path)

    n_with = sum(1 for r in rankings if r["score"] > 0)
    print(f"Rendered {len(rankings)} intersections "
          f"({n_with} with at least one safety-relevant finding).")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
