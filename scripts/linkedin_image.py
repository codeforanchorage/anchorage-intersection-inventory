"""Render a clean LinkedIn-ready composite for the Anchorage intersection
inventory project.

Picks a (heading, pitch) where the most pole-mounted assets are visible
*outside* the top-right aerial-inset zone, draws colored bboxes with
asset-type-only labels (no confidence scores — readable to non-engineers),
and overlays a 2024 EagleView aerial thumbnail in the top-right corner.

Output: data/results/linkedin/{osm_id}_h{H}_p{P}_linkedin.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.visualize_priority import _paste_aerial_inset


POLE_TYPES = {
    "traffic_signal_head", "signal_pole", "mast_arm",
    "signal_cabinet", "street_light", "road_sign",
}

# Apple system-color palette — high contrast, distinguishable for color-blind
# viewers, reads well against varied GSV backgrounds.
PALETTE = {
    "traffic_signal_head": "#FF3B30",
    "signal_pole":         "#5856D6",
    "mast_arm":            "#34C759",
    "signal_cabinet":      "#AF52DE",
    "street_light":        "#64D2FF",
    "road_sign":           "#FF9500",
}

# Human-readable labels (no underscores, no confidence).
LABELS = {
    "traffic_signal_head": "SIGNAL HEAD",
    "signal_pole":         "POLE",
    "mast_arm":            "MAST ARM",
    "signal_cabinet":      "CABINET",
    "street_light":        "STREET LIGHT",
    "road_sign":           "SIGN",
}

# Inset-occlusion zone (top-right of GSV image). Detections whose bbox center
# falls inside are dropped from the render so labels don't sit under the inset.
INSET_X_MIN, INSET_Y_MAX = 460, 180


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def in_inset_zone(det: dict) -> bool:
    bb = det.get("bbox") or {}
    if "x1" not in bb:
        return False
    cx = (bb["x1"] + bb["x2"]) / 2
    cy = (bb["y1"] + bb["y2"]) / 2
    return cx >= INSET_X_MIN and cy <= INSET_Y_MAX


def _aerial_vehicle_counts() -> dict[str, int]:
    """Per-intersection vehicle counts from the Phase 7 GIS context CSV.

    Used as a "how busy is this scene" signal so the picker can prefer
    infrastructure-heavy intersections over parking-lot-looking ones.
    """
    out: dict[str, int] = {}
    if not config.GIS_CONTEXT_CSV.exists():
        return out
    import csv as _csv
    with config.GIS_CONTEXT_CSV.open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            v = row.get("aerial_vehicle_count", "")
            if v:
                try:
                    out[row["intersection_id"]] = int(v)
                except ValueError:
                    pass
    return out


def pick_best_view(pitch: int = 25, vehicle_penalty: float = 2.0) -> tuple[str, int, int]:
    """Pick (osm_id, heading, pitch) maximizing visible pole assets minus a
    vehicle penalty so we prefer scenes with infrastructure over scenes with
    cars. ``vehicle_penalty`` weights one detected vehicle as N pole assets."""
    veh = _aerial_vehicle_counts()
    best = (-1e9, "", 0, pitch)
    for det_path in sorted(config.RESULTS_DIR.glob("osm_*_detections.json")):
        osm_id = det_path.stem.replace("_detections", "")
        with det_path.open() as f:
            doc = json.load(f)
        for img in doc.get("images", []):
            if img.get("pitch", 10) != pitch:
                continue
            visible = sum(
                1 for det in img.get("detections", [])
                if det.get("asset_type") in POLE_TYPES and not in_inset_zone(det)
            )
            score = visible - vehicle_penalty * veh.get(osm_id, 0)
            if score > best[0]:
                best = (score, osm_id, img.get("heading"), pitch)
    return best[1], best[2], best[3]


def load_detections(osm_id: str, heading: int, pitch: int) -> list[dict]:
    det_path = config.RESULTS_DIR / f"{osm_id}_detections.json"
    with det_path.open() as f:
        doc = json.load(f)
    for img in doc.get("images", []):
        if img.get("heading") == heading and img.get("pitch", 10) == pitch:
            return [
                d for d in img.get("detections", [])
                if d.get("asset_type") in POLE_TYPES and not in_inset_zone(d)
            ]
    return []


def render(osm_id: str, heading: int, pitch: int, out_dir: Path) -> tuple[Path, Path]:
    """Returns (annotated_path, original_path). Both saved in out_dir."""
    src = config.IMAGE_DIR / f"{osm_id}_h{heading}_p{pitch}.jpg"
    aerial = config.GIS_DIR / f"{osm_id}_aerial.jpg"
    if not src.exists():
        raise FileNotFoundError(f"GSV not on disk: {src}")
    if not aerial.exists():
        raise FileNotFoundError(f"Aerial not on disk: {aerial}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the unannotated GSV alongside the annotated version. Useful as a
    # before/after pair for social posts.
    original_path = out_dir / f"{osm_id}_h{heading}_p{pitch}_original.jpg"
    Image.open(src).convert("RGB").save(original_path, format="JPEG", quality=92)

    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    detections = load_detections(osm_id, heading, pitch)
    # Draw signal heads LAST so the red labels render on top of any
    # neighbouring pole / mast-arm labels they overlap.
    detections = sorted(
        detections,
        key=lambda d: 1 if d.get("asset_type") == "traffic_signal_head" else 0,
    )
    font = _font(13)

    for det in detections:
        bb = det.get("bbox") or {}
        x1 = max(0, min(bb["x1"], img.width - 1))
        y1 = max(0, min(bb["y1"], img.height - 1))
        x2 = max(x1 + 1, min(bb["x2"], img.width))
        y2 = max(y1 + 1, min(bb["y2"], img.height))
        atype = det.get("asset_type", "other")
        color = PALETTE.get(atype, "#FFFFFF")
        label = LABELS.get(atype, atype.upper())

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tw = draw.textlength(label, font=font)
        # Place above bbox if room, else below — same logic as Phase 6.
        if y1 >= 18:
            top, bot, ty = y1 - 18, y1, y1 - 17
        else:
            top, bot, ty = y2, min(img.height, y2 + 18), y2 + 1
        draw.rectangle([x1, top, x1 + tw + 8, bot], fill=color)
        draw.text((x1 + 4, ty), label, fill="white", font=font)

    _paste_aerial_inset(img, aerial, img.width, img.height)

    out_path = out_dir / f"{osm_id}_h{heading}_p{pitch}_linkedin.jpg"
    img.save(out_path, format="JPEG", quality=92)
    return out_path, original_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a LinkedIn-ready composite.")
    parser.add_argument("--osm-id", default=None,
                        help="Specific intersection (e.g. osm_31329116). "
                             "If omitted, auto-picks based on pole-asset density.")
    parser.add_argument("--heading", type=int, default=None)
    parser.add_argument("--pitch", type=int, default=25,
                        choices=[10, 25, 45],
                        help="GSV pitch (default 25 — balanced view).")
    args = parser.parse_args()

    if args.osm_id and args.heading is not None:
        osm_id, heading, pitch = args.osm_id, args.heading, args.pitch
    else:
        osm_id, heading, pitch = pick_best_view(pitch=args.pitch)
        print(f"Auto-picked: {osm_id} heading={heading}° pitch={pitch}°")

    out_dir = config.RESULTS_DIR / "linkedin"
    out_path, original_path = render(osm_id, heading, pitch, out_dir)
    n_dets = len(load_detections(osm_id, heading, pitch))
    print(f"Rendered {n_dets} pole-asset bboxes")
    print(f"  annotated: {out_path}")
    print(f"  original:  {original_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
