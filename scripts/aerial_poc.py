"""Proof-of-concept: pull MOA's 2024 EagleView aerial + 2025 LiDAR hillshade
+ 2025 DEM elevation for the top-N priority intersections, run SAM 3 on the
aerial as a top-down "13th view," and compose a side-by-side image with the
existing GSV priority view.

Goal: see whether top-down counting genuinely contradicts SAM 3's known
ground-view overcounts (e.g. mast_arm 25/intersection from ground vs. ~4
expected from above), and whether the DEM elevation + hillshade provide
useful context Traffic Engineering would actually act on.

Output:
  data/results/aerial_poc/
      {osm_id}_composite.jpg     # 3-panel: GSV worst view | aerial | hillshade
      summary.md                 # ground vs aerial counts + elevation per id

Run:  python scripts/aerial_poc.py --top 5
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import detect_assets


PHOTO_2024 = "https://www.ancgis.com/arcgis/rest/services/imagery_public/Photo_2024/MapServer"
DEM_2025 = "https://www.ancgis.com/arcgis/rest/services/elevation_dem/DEM_2025/MapServer"
HILLSHADE_2025 = "https://www.ancgis.com/arcgis/rest/services/elevation_dem/hillshade_2025/MapServer"

PRIORITY_DIR = config.RESULTS_DIR / "priority"
INDEX_PATH = PRIORITY_DIR / "index.md"
OUT_DIR = config.RESULTS_DIR / "aerial_poc"

# Aerial-specific SAM 3 prompts. Top-down view is a different visual signature
# than ground GSV, and the PoC sanity check showed signal heads / poles are
# below SAM 3's effective resolution at 60m × 60m / 512px (~3-pixel asset).
# What DOES work top-down: crosswalks (zebra patterns are large + high contrast)
# and vehicles (good context, not a project asset). We keep just crosswalk for
# the count comparison; vehicle is a sanity check that SAM 3 is engaging.
AERIAL_PROMPTS = [
    {"text": "crosswalk", "asset_type": "crosswalk_marking", "confidence": 0.20},
    {"text": "vehicle", "asset_type": "vehicle", "confidence": 0.30},
]

EXTENT_M = 30.0  # half-extent → 60m × 60m footprint per intersection
TILE_PX = 512


def parse_top_n(index_path: Path, n: int) -> list[tuple[str, int, int]]:
    """(osm_id, heading, pitch) tuples from the Phase 6 priority index."""
    text = index_path.read_text(encoding="utf-8")
    pat = re.compile(r"!\[(?P<osm>osm_\d+)_h(?P<h>\d+)(?:_p(?P<p>\d+))?\.jpg\]")
    seen: set[str] = set()
    out: list[tuple[str, int, int]] = []
    for m in pat.finditer(text):
        osm = m.group("osm")
        if osm in seen:
            continue
        seen.add(osm)
        out.append((osm, int(m.group("h")), int(m.group("p")) if m.group("p") else 10))
        if len(out) >= n:
            break
    return out


def lookup_latlon(osm_id: str) -> tuple[float, float]:
    """Read intersection lat/lon from the per-intersection inventory JSON."""
    import json
    path = config.RESULTS_DIR / f"{osm_id}_inventory.json"
    with path.open() as f:
        rec = json.load(f)
    return rec["lat"], rec["lon"]


def bbox_around(lat: float, lon: float, half_extent_m: float = EXTENT_M) -> str:
    """ArcGIS bbox string in EPSG:4326 (xmin,ymin,xmax,ymax)."""
    deg_lat = half_extent_m / 111111.0
    deg_lon = deg_lat / abs(math.cos(math.radians(lat)))
    return f"{lon - deg_lon},{lat - deg_lat},{lon + deg_lon},{lat + deg_lat}"


def fetch_image(service_base: str, lat: float, lon: float) -> bytes:
    bbox = bbox_around(lat, lon)
    r = requests.get(
        f"{service_base}/export",
        params={
            "bbox": bbox, "bboxSR": 4326, "imageSR": 4326,
            "size": f"{TILE_PX},{TILE_PX}",
            "format": "jpg", "f": "image",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.content


def fetch_dem_elevation(lat: float, lon: float) -> float | None:
    """Single-point elevation in meters. None if the identify call fails."""
    bbox = bbox_around(lat, lon, half_extent_m=5.0)
    r = requests.get(
        f"{DEM_2025}/identify",
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
        for k in ("Stretch.Pixel Value", "Pixel Value", "PIXEL VALUE"):
            if k in attrs:
                try:
                    return float(attrs[k])
                except (TypeError, ValueError):
                    return None
    return None


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _aerial_palette() -> dict[str, tuple[int, int, int]]:
    return {
        "crosswalk_marking": (0, 199, 190),
        "vehicle": (200, 200, 200),
    }


def annotate_aerial(jpeg_bytes: bytes, detections: list[dict]) -> Image.Image:
    img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    pal = _aerial_palette()
    for det in detections:
        bb = det.get("bbox") or {}
        if "x1" not in bb:
            continue
        atype = det.get("asset_type", "other")
        color = pal.get(atype, (200, 200, 200))
        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        # Clamp to canvas
        x1 = max(0, min(x1, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(x1 + 1, min(x2, img.width))
        y2 = max(y1 + 1, min(y2, img.height))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return img


def compose(gsv_path: Path, aerial_img: Image.Image, hillshade_bytes: bytes,
            osm_id: str, heading: int, pitch: int,
            ground_counts: dict[str, int], aerial_counts: dict[str, int],
            elevation_m: float | None) -> Image.Image:
    """3-panel composite: GSV | aerial | hillshade, with caption banner."""
    gsv = Image.open(gsv_path).convert("RGB").resize((TILE_PX, TILE_PX))
    hill = Image.open(BytesIO(hillshade_bytes)).convert("RGB")
    panel_w, panel_h = TILE_PX, TILE_PX
    pad = 6
    banner_h = 90
    total_w = panel_w * 3 + pad * 4
    total_h = panel_h + banner_h + pad * 2
    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
    canvas.paste(gsv,    (pad,                       banner_h + pad))
    canvas.paste(aerial_img, (pad * 2 + panel_w,     banner_h + pad))
    canvas.paste(hill,   (pad * 3 + panel_w * 2,     banner_h + pad))

    d = ImageDraw.Draw(canvas)
    title = f"{osm_id}  ·  GSV worst view: heading {heading}° pitch {pitch}°"
    d.text((pad + 2, 6), title, fill="white", font=font(16))
    elev_str = f"DEM elevation: {elevation_m:.1f} m AGL" if elevation_m is not None else "DEM elevation: n/a"
    d.text((pad + 2, 28), elev_str, fill=(180, 220, 255), font=font(13))
    line2 = (
        f"ground SAM3 crosswalks: {ground_counts.get('crosswalk_marking', 0)}    |    "
        f"aerial SAM3 crosswalks: {aerial_counts.get('crosswalk_marking', 0)}    "
        f"(aerial vehicles: {aerial_counts.get('vehicle', 0)} — sanity check)"
    )
    d.text((pad + 2, 50), line2, fill="white", font=font(12))
    d.text((pad + 2, 70),
           "GSV (Phase 6 priority view)         |         "
           "2024 EagleView aerial (3-in or 6-in res)        |        "
           "2025 LiDAR hillshade",
           fill=(160, 160, 160), font=font(11))
    return canvas


def ground_counts_from_detections(osm_id: str, heading: int, pitch: int) -> dict[str, int]:
    """Count detections per asset_type for the matching (heading, pitch) image."""
    import json
    from collections import Counter
    path = config.RESULTS_DIR / f"{osm_id}_detections.json"
    if not path.exists():
        return {}
    with path.open() as f:
        doc = json.load(f)
    for img in doc.get("images", []):
        if img.get("heading") == heading and img.get("pitch", 10) == pitch:
            counts: Counter = Counter()
            for det in img.get("detections", []):
                counts[det.get("asset_type", "other")] += 1
            return dict(counts)
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LiDAR + aerial PoC: compare ground vs top-down SAM 3 counts."
    )
    parser.add_argument("--top", type=int, default=5,
                        help="Number of priority intersections (default 5).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = parse_top_n(INDEX_PATH, args.top)
    if not targets:
        print("ERROR: could not parse priority index.")
        return 1
    print(f"Targeting top {len(targets)} priority intersections")

    print("Loading SAM 3 (one-time)...")
    predictor = detect_assets.init_sam3_predictor(default_confidence=0.25)
    print(f"  {len(AERIAL_PROMPTS)} aerial prompts")

    summary_lines = [
        "# Aerial-vs-ground SAM 3 PoC",
        "",
        "Side-by-side of the GSV priority view, the 2024 EagleView aerial, and ",
        "the 2025 LiDAR hillshade for the top-N priority intersections. ",
        "Compares ground SAM 3 crosswalk fragmentation against top-down counts.",
        "",
        "**Note on signal heads:** SAM 3 cannot see traffic signal heads from ",
        "above at 60 m × 60 m / 512 px — each signal head is roughly 3 pixels ",
        "wide, below the model's effective resolution. Counting signals from ",
        "aerial would require a tighter crop (30 m extent at 1024 px) or a ",
        "model trained on aerial imagery.",
        "",
        "| Intersection | Elev (m) | Ground crosswalks | Aerial crosswalks | Δ | Aerial vehicles |",
        "|---|---|---|---|---|---|",
    ]

    for osm_id, heading, pitch in targets:
        try:
            lat, lon = lookup_latlon(osm_id)
        except Exception as exc:
            print(f"  {osm_id}: lookup failed ({exc})")
            continue
        gsv_path = config.IMAGE_DIR / f"{osm_id}_h{heading}_p{pitch}.jpg"
        if not gsv_path.exists():
            print(f"  {osm_id}: GSV {gsv_path.name} missing — skip")
            continue
        print(f"  {osm_id} ({lat:.5f}, {lon:.5f}): fetching aerial+hillshade+DEM...")

        aerial_bytes = fetch_image(PHOTO_2024, lat, lon)
        hillshade_bytes = fetch_image(HILLSHADE_2025, lat, lon)
        elev = fetch_dem_elevation(lat, lon)

        # Save aerial to temp path so SAM 3 can ingest it (it expects a file).
        aerial_tmp = OUT_DIR / f"{osm_id}_aerial.jpg"
        aerial_tmp.write_bytes(aerial_bytes)

        t0 = time.time()
        aerial_dets = detect_assets.detect_with_sam3(predictor, aerial_tmp, AERIAL_PROMPTS)
        t1 = time.time()
        from collections import Counter
        ac: Counter = Counter()
        for d in aerial_dets:
            ac[d.get("asset_type", "other")] += 1
        aerial_counts = dict(ac)
        gc = ground_counts_from_detections(osm_id, heading, pitch)

        annotated_aerial = annotate_aerial(aerial_bytes, aerial_dets)
        composite = compose(
            gsv_path, annotated_aerial, hillshade_bytes,
            osm_id, heading, pitch, gc, aerial_counts, elev,
        )
        out = OUT_DIR / f"{osm_id}_composite.jpg"
        composite.save(out, format="JPEG", quality=88)
        print(f"    DEM={elev}  ground={gc}  aerial={aerial_counts}  "
              f"({len(aerial_dets)} aerial detections in {t1 - t0:.1f}s) -> {out.name}")

        gx = gc.get("crosswalk_marking", 0)
        ax_ = aerial_counts.get("crosswalk_marking", 0)
        veh = aerial_counts.get("vehicle", 0)
        elev_str = f"{elev:.1f}" if elev is not None else "n/a"
        summary_lines.append(
            f"| {osm_id} | {elev_str} | {gx} | {ax_} | {ax_ - gx:+d} | {veh} |"
        )

    summary_lines += [
        "",
        "Composites: `data/results/aerial_poc/{osm_id}_composite.jpg`. ",
        "Each composite shows the GSV priority view, the 2024 EagleView aerial "
        "(with top-down SAM 3 bboxes), and the 2025 LiDAR hillshade side-by-side.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nSummary: {OUT_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
