"""Phase 3: Detect intersection assets and (optionally) assess condition.

Three pluggable backends, selected with --backend:

  sam3        Local Meta SAM 3 (Ultralytics). Requires CUDA GPU + HuggingFace
              license accepted for facebook/sam3. Open-vocabulary text prompts
              from config/asset_prompts.yaml. Fast and free at runtime.

  roboflow    Roboflow hosted inference. Requires ROBOFLOW_API_KEY plus
              workspace / project / version pointing at a road-asset model.

  vision-only Claude Vision on full images. No detector — Claude inventories
              and condition-scores everything in one call. Slowest per-image,
              simplest setup.

All backends emit the same per-intersection JSON schema so Phase 4 stays
unchanged regardless of which detector was used:

  {"intersection_id": ..., "images": [
     {"image": ..., "heading": ..., "detections": [
        {"asset_type": ..., "confidence": ..., "bbox": ...,
         "condition": ..., "condition_notes": ..., "source": ...}, ...]}, ...]}

For sam3/roboflow runs, condition fields are populated only when
`--with-condition` is set (each detection's crop is sent to Claude Vision for
a focused good/fair/poor scoring). Without that flag, Phase 4 will produce
counts but null condition scores.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


_LOG = logging.getLogger("detect_assets")


# --- Roboflow detection -----------------------------------------------------

ROBOFLOW_CLASS_MAP = {
    "traffic light": "traffic_signal_head",
    "traffic-light": "traffic_signal_head",
    "signal": "traffic_signal_head",
    "pedestrian signal": "pedestrian_signal",
    "ped signal": "pedestrian_signal",
    "pole": "signal_pole",
    "mast arm": "mast_arm",
    "cabinet": "signal_cabinet",
    "crosswalk": "crosswalk_marking",
    "curb ramp": "curb_ramp",
    "ramp": "curb_ramp",
    "push button": "push_button",
    "street light": "street_light",
    "streetlight": "street_light",
    "sign": "road_sign",
    "stop sign": "road_sign",
    "lane marking": "lane_marking",
    "stop line": "lane_marking",
}


def normalize_asset_class(raw: str) -> str:
    raw_lc = raw.strip().lower()
    if raw_lc in ROBOFLOW_CLASS_MAP:
        return ROBOFLOW_CLASS_MAP[raw_lc]
    for key, val in ROBOFLOW_CLASS_MAP.items():
        if key in raw_lc:
            return val
    return "other"


def load_roboflow_model(api_key: str, workspace: str, project: str, version: int):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    return rf.workspace(workspace).project(project).version(version).model


def detect_with_roboflow(model, image_path: Path, confidence: int, overlap: int) -> list[dict]:
    result = model.predict(str(image_path), confidence=confidence, overlap=overlap).json()
    detections = []
    for pred in result.get("predictions", []):
        detections.append({
            "raw_class": pred.get("class", ""),
            "asset_type": normalize_asset_class(pred.get("class", "")),
            "confidence": float(pred.get("confidence", 0.0)),
            "bbox": {
                "x": float(pred.get("x", 0.0)),
                "y": float(pred.get("y", 0.0)),
                "width": float(pred.get("width", 0.0)),
                "height": float(pred.get("height", 0.0)),
            },
        })
    return detections


# --- SAM 3 detection --------------------------------------------------------

DEFAULT_PROMPTS_PATH = config.PROJECT_ROOT / "config" / "asset_prompts.yaml"


def load_prompts(path: Path) -> list[dict]:
    """Load the prompt list from YAML. Returns list of {text, asset_type, confidence, category}."""
    import yaml
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc.get("prompts", [])


def ensure_sam3_weights() -> str:
    """Make sure facebook/sam3's sam3.pt is on disk and return the absolute path.

    Ultralytics 8.4.x ships SAM 3 code but doesn't include sam3.pt in its
    GITHUB_ASSETS auto-download list, so we fetch it ourselves from the gated
    HuggingFace repo. HF_TOKEN must be set in env or .env (and the user must
    have been granted access at huggingface.co/facebook/sam3).
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download SAM 3 weights. "
            "Install with: pip install huggingface_hub"
        ) from exc
    return hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        token=token,
    )


def init_sam3_predictor(default_confidence: float = 0.25, half: bool = True):
    """Lazy-import ultralytics and return a SAM 3 predictor."""
    from ultralytics.models.sam import SAM3SemanticPredictor
    weights_path = ensure_sam3_weights()
    overrides = dict(
        conf=default_confidence,
        task="segment",
        mode="predict",
        model=weights_path,
        half=half,
        verbose=False,
    )
    return SAM3SemanticPredictor(overrides=overrides)


def _bbox_iou(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    """Standard intersection-over-union for (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def nms_by_asset_type(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Suppress overlapping detections within each asset_type, keeping highest confidence.

    SAM 3 with multiple prompt variants targeting the same asset_type (e.g. three
    phrasings for 'mast_arm') will segment the same physical object multiple
    times. Single-prompt runs also produce overlap when SAM 3 segments parts of
    the same object. This collapses both into one detection per physical asset.
    """
    grouped: dict[str, list[dict]] = {}
    for det in detections:
        grouped.setdefault(det.get("asset_type", "other"), []).append(det)

    kept_all: list[dict] = []
    for atype, dets in grouped.items():
        dets_sorted = sorted(dets, key=lambda d: -d.get("confidence", 0.0))
        kept_local: list[tuple[float, float, float, float]] = []
        for det in dets_sorted:
            bb = det.get("bbox") or {}
            box = (bb.get("x1", 0.0), bb.get("y1", 0.0),
                   bb.get("x2", 0.0), bb.get("y2", 0.0))
            if any(_bbox_iou(box, k) >= iou_threshold for k in kept_local):
                continue
            kept_local.append(box)
            kept_all.append(det)
    return kept_all


def detect_with_sam3(predictor, image_path: Path, prompts: list[dict],
                     nms_iou: float = 0.5) -> list[dict]:
    """Run every prompt against a single image, then suppress duplicates by asset_type."""
    predictor.set_image(str(image_path))
    detections: list[dict] = []
    for spec in prompts:
        text = spec["text"]
        asset_type = spec.get("asset_type", text.replace(" ", "_"))
        # Per-prompt confidence override.
        if "confidence" in spec and hasattr(predictor, "args"):
            try:
                predictor.args.conf = float(spec["confidence"])
            except Exception:
                pass

        results = predictor(text=[text])
        if not results:
            continue
        result = results[0]
        if result.masks is None or result.boxes is None:
            continue
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            detections.append({
                "raw_class": text,
                "asset_type": asset_type,
                "confidence": float(scores[i]),
                "bbox": {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                },
                "mask_pixel_count": int(masks[i].sum()),
            })
    return nms_by_asset_type(detections, iou_threshold=nms_iou)


# --- Claude Vision (full image + crop modes) --------------------------------

CONDITION_SYSTEM_PROMPT = """You are a civil-infrastructure inspector reviewing
crops from Google Street View images of signalized intersections in Anchorage,
Alaska. For each image you receive, identify the primary intersection asset
visible and assess its apparent condition.

Respond ONLY with a JSON object (no prose, no markdown fences) of the form:
{
  "asset_type": "traffic_signal_head|pedestrian_signal|signal_pole|mast_arm|signal_cabinet|crosswalk_marking|curb_ramp|push_button|street_light|road_sign|lane_marking|other",
  "condition": "good|fair|poor|not_assessable",
  "condition_notes": "<short description of any visible damage, fading, rust, missing covers, or 'no issues observed'>",
  "approximate_location_in_image": "left|center|right"
}

If the image is too blurry, occluded, or shows nothing recognizable, set
condition to "not_assessable" and asset_type to "other".
"""

FULL_IMAGE_PROMPT = """Analyze this street-view image of a signalized
intersection. Identify every visible piece of intersection infrastructure and
assess its apparent condition.

Respond ONLY with a JSON array (no prose, no markdown fences). Each element:
{
  "asset_type": "traffic_signal_head|pedestrian_signal|signal_pole|mast_arm|signal_cabinet|crosswalk_marking|curb_ramp|push_button|street_light|road_sign|lane_marking|other",
  "condition": "good|fair|poor|not_assessable",
  "condition_notes": "<short description>",
  "approximate_location_in_image": "left|center|right"
}
"""


def _extract_json(text: str) -> Any:
    """Extract a JSON array or object from a possibly fenced/prose-wrapped string."""
    s = text.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def claude_assess(client, image_bytes: bytes, prompt: str, system: str | None,
                  model: str) -> Any:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    kwargs = dict(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    parsed = _extract_json(text)
    if parsed is None:
        _LOG.warning(
            "claude_assess: could not parse response (stop_reason=%s, len=%d). Preview: %s",
            getattr(response, "stop_reason", "?"), len(text), text[:200].replace("\n", " "),
        )
    elif getattr(response, "stop_reason", None) == "max_tokens":
        _LOG.warning("claude_assess: hit max_tokens — output may be truncated")
    return parsed


async def claude_assess_async(async_client, image_bytes: bytes, prompt: str,
                              system: str | None, model: str) -> Any:
    """Async equivalent of claude_assess for use with AsyncAnthropic."""
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    kwargs = dict(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    if system:
        kwargs["system"] = system
    response = await async_client.messages.create(**kwargs)
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    parsed = _extract_json(text)
    if parsed is None:
        _LOG.warning(
            "claude_assess_async: could not parse response (stop_reason=%s, len=%d). Preview: %s",
            getattr(response, "stop_reason", "?"), len(text), text[:200].replace("\n", " "),
        )
    elif getattr(response, "stop_reason", None) == "max_tokens":
        _LOG.warning("claude_assess_async: hit max_tokens — output may be truncated")
    return parsed


def crop_for_detection(image_path: Path, det: dict, padding: float = 0.1) -> bytes:
    """Crop a padded region around a bbox. Returns JPEG bytes.

    Accepts both center-format bboxes (x,y,width,height — Roboflow) and
    corner-format bboxes (x1,y1,x2,y2 — SAM 3).
    """
    img = Image.open(image_path).convert("RGB")
    bbox = det["bbox"]
    if "x1" in bbox:
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    else:
        cx, cy = bbox["x"], bbox["y"]
        bw, bh = bbox["width"], bbox["height"]
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    pad_w = (x2 - x1) * padding
    pad_h = (y2 - y1) * padding
    left = max(0, int(x1 - pad_w))
    top = max(0, int(y1 - pad_h))
    right = min(img.width, int(x2 + pad_w))
    bottom = min(img.height, int(y2 + pad_h))
    crop = img.crop((left, top, right, bottom))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# --- Per-intersection orchestration -----------------------------------------

def list_images_for(intersection_id: str) -> list[Path]:
    return sorted(config.IMAGE_DIR.glob(f"{intersection_id}_h*.jpg"))


def heading_from_path(path: Path) -> int:
    return int(path.stem.split("_h")[-1])


def detections_path(intersection_id: str) -> Path:
    return config.RESULTS_DIR / f"{intersection_id}_detections.json"


def already_done(intersection_id: str) -> bool:
    return detections_path(intersection_id).exists()


def process_image_vision_only(img_path: Path, claude_client, claude_model: str) -> list[dict]:
    with img_path.open("rb") as f:
        img_bytes = f.read()
    result = claude_assess(claude_client, img_bytes, FULL_IMAGE_PROMPT, None, claude_model)
    return _normalize_full_image_result(result)


def _normalize_full_image_result(result: Any) -> list[dict]:
    out: list[dict] = []
    if isinstance(result, list):
        for item in result:
            out.append({
                "asset_type": item.get("asset_type", "other"),
                "condition": item.get("condition", "not_assessable"),
                "condition_notes": item.get("condition_notes", ""),
                "location_in_image": item.get("approximate_location_in_image", ""),
                "source": "claude_vision_full",
            })
    return out


async def _process_image_vision_only_async(img_path: Path, async_client, claude_model: str,
                                           semaphore) -> tuple[Path, list[dict]]:
    async with semaphore:
        img_bytes = img_path.read_bytes()
        result = await claude_assess_async(
            async_client, img_bytes, FULL_IMAGE_PROMPT, None, claude_model,
        )
    return img_path, _normalize_full_image_result(result)


async def _run_vision_only_concurrent(
    features: list[dict],
    claude_model: str,
    concurrency: int,
    skip_existing: bool,
) -> tuple[int, int, int]:
    """Process all images for given intersections concurrently. Writes JSON per intersection.

    Returns (n_processed, n_skipped, n_no_images).
    """
    from anthropic import AsyncAnthropic  # imported here so sync paths don't need it
    async_client = AsyncAnthropic()

    plan: list[tuple[str, list[Path]]] = []
    n_skipped = n_no_images = 0
    for feat in features:
        osm_id = feat["properties"]["osm_id"]
        if skip_existing and already_done(osm_id):
            n_skipped += 1
            continue
        images = list_images_for(osm_id)
        if not images:
            n_no_images += 1
            continue
        plan.append((osm_id, images))

    if not plan:
        return 0, n_skipped, n_no_images

    semaphore = asyncio.Semaphore(concurrency)
    path_to_osm: dict[Path, str] = {}
    tasks = []
    for osm_id, images in plan:
        for img_path in images:
            path_to_osm[img_path] = osm_id
            tasks.append(_process_image_vision_only_async(
                img_path, async_client, claude_model, semaphore,
            ))

    total = len(tasks)
    print(f"  dispatching {total} image calls across {len(plan)} intersections "
          f"at concurrency={concurrency}")

    results_by_osm: dict[str, list[dict]] = {osm_id: [] for osm_id, _ in plan}
    completed = 0
    for fut in asyncio.as_completed(tasks):
        img_path, detections = await fut
        osm_id = path_to_osm[img_path]
        results_by_osm[osm_id].append({
            "image": img_path.name,
            "heading": heading_from_path(img_path),
            "detections": detections,
        })
        completed += 1
        if completed % max(1, total // 20) == 0 or completed == total:
            print(f"    [{completed}/{total}] images complete")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for osm_id, image_entries in results_by_osm.items():
        image_entries.sort(key=lambda e: e["heading"])
        with detections_path(osm_id).open("w", encoding="utf-8") as f:
            json.dump({"intersection_id": osm_id, "images": image_entries}, f, indent=2)

    return len(plan), n_skipped, n_no_images


def score_detection_with_claude(img_path: Path, det: dict, claude_client, claude_model: str) -> dict:
    """Add condition fields to a single detection by sending its crop to Claude."""
    crop_bytes = crop_for_detection(img_path, det)
    assess = claude_assess(
        claude_client, crop_bytes,
        "Assess this cropped intersection asset.",
        CONDITION_SYSTEM_PROMPT, claude_model,
    )
    det = dict(det)
    det["condition"] = "not_assessable"
    det["condition_notes"] = ""
    det["location_in_image"] = ""
    if isinstance(assess, dict):
        det["condition"] = assess.get("condition", "not_assessable")
        det["condition_notes"] = assess.get("condition_notes", "")
        det["location_in_image"] = assess.get("approximate_location_in_image", "")
        # Trust Claude's asset_type only if the detector wasn't confident about it.
        if det.get("asset_type") in (None, "", "other") and assess.get("asset_type"):
            det["asset_type"] = assess["asset_type"]
    return det


def process_image_with_detector(
    img_path: Path,
    backend: str,
    detector_state: dict,
    with_condition: bool,
    claude_client,
    claude_model: str,
) -> list[dict]:
    if backend == "sam3":
        raw = detect_with_sam3(detector_state["predictor"], img_path, detector_state["prompts"])
        source = "sam3"
    elif backend == "roboflow":
        raw = detect_with_roboflow(
            detector_state["model"], img_path,
            confidence=detector_state["confidence"],
            overlap=detector_state["overlap"],
        )
        source = "roboflow"
    else:
        raise ValueError(f"unknown detector backend: {backend}")

    out: list[dict] = []
    for det in raw:
        det["source"] = source
        if with_condition:
            det = score_detection_with_claude(img_path, det, claude_client, claude_model)
            det["source"] = f"{source}+claude_crop"
        else:
            det.setdefault("condition", "not_assessable")
            det.setdefault("condition_notes", "")
            det.setdefault("location_in_image", "")
        out.append(det)
    return out


def extract_crops_from_results(crops_dir: Path, padding: int = 20) -> int:
    """Walk the per-intersection detection JSONs and write JPEG crops to disk."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for det_path in sorted(config.RESULTS_DIR.glob("*_detections.json")):
        with det_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        for img_entry in doc.get("images", []):
            img_path = config.IMAGE_DIR / img_entry["image"]
            if not img_path.exists():
                continue
            for i, det in enumerate(img_entry.get("detections", [])):
                if "bbox" not in det:
                    continue
                try:
                    crop_bytes = crop_for_detection(img_path, det, padding=padding / 100.0)
                except Exception as exc:
                    _LOG.warning("crop failed for %s det %d: %s", img_path.name, i, exc)
                    continue
                slug = (det.get("asset_type") or "asset").replace(" ", "_")
                out = crops_dir / f"{doc['intersection_id']}_{slug}_{i:03d}.jpg"
                out.write_bytes(crop_bytes)
                written += 1
    return written


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: detect assets and assess condition.")
    parser.add_argument("--backend", choices=["sam3", "roboflow", "vision-only"],
                        default="sam3",
                        help="Detection backend (default: sam3).")
    # Backwards-compat: map --vision-only to --backend vision-only.
    parser.add_argument("--vision-only", action="store_true",
                        help="Deprecated alias for --backend vision-only.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--with-condition", action="store_true",
                        help="For sam3/roboflow: also send each crop to Claude for condition scoring.")
    parser.add_argument("--extract-crops", action="store_true",
                        help="After detection, write per-asset crops to data/results/crops/.")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Default detection-confidence threshold (sam3) or 0-100 cutoff (roboflow).")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH),
                        help="Path to SAM 3 prompts YAML.")
    parser.add_argument("--rf-workspace", default=os.environ.get("ROBOFLOW_WORKSPACE", ""))
    parser.add_argument("--rf-project", default=os.environ.get("ROBOFLOW_PROJECT", ""))
    parser.add_argument("--rf-version", type=int,
                        default=int(os.environ.get("ROBOFLOW_VERSION", "0") or 0))
    parser.add_argument("--rf-overlap", type=int, default=30)
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent in-flight Claude calls. Vision-only path only. "
                             "Recommend 8-10 for full Anchorage; 1 = sequential.")
    args = parser.parse_args(argv)

    if args.vision_only:
        args.backend = "vision-only"

    load_dotenv(config.PROJECT_ROOT / ".env")

    # Anthropic is required for vision-only and --with-condition; otherwise optional.
    claude_client = None
    needs_claude = args.backend == "vision-only" or args.with_condition
    if needs_claude:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            return 1
        from anthropic import Anthropic
        claude_client = Anthropic(api_key=anthropic_key)

    detector_state: dict = {}
    if args.backend == "sam3":
        try:
            predictor = init_sam3_predictor(default_confidence=args.confidence)
        except ImportError as exc:
            print(f"ERROR: SAM 3 backend requires `ultralytics` and `torch`: {exc}")
            print("Install with: pip install ultralytics torch torchvision einops")
            return 1
        prompts = load_prompts(Path(args.prompts))
        detector_state = {"predictor": predictor, "prompts": prompts}
        print(f"SAM 3 ready ({len(prompts)} prompts from {args.prompts})")
    elif args.backend == "roboflow":
        rf_key = os.environ.get("ROBOFLOW_API_KEY")
        if not rf_key or not args.rf_workspace or not args.rf_project or not args.rf_version:
            print("ERROR: Roboflow backend needs ROBOFLOW_API_KEY + workspace/project/version.")
            return 1
        rf_model = load_roboflow_model(rf_key, args.rf_workspace, args.rf_project, args.rf_version)
        detector_state = {
            "model": rf_model,
            "confidence": int(args.confidence * 100) if args.confidence < 1 else int(args.confidence),
            "overlap": args.rf_overlap,
        }

    if not config.SIGNALS_GEOJSON.exists():
        print(f"ERROR: {config.SIGNALS_GEOJSON} not found. Run Phase 1 first.")
        return 1
    with config.SIGNALS_GEOJSON.open("r", encoding="utf-8") as f:
        features = json.load(f).get("features", [])
    if args.limit is not None:
        features = features[: args.limit]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.backend == "vision-only" and args.concurrency > 1:
        n_processed, n_skipped, n_no_images = asyncio.run(_run_vision_only_concurrent(
            features=features,
            claude_model=args.claude_model,
            concurrency=args.concurrency,
            skip_existing=args.skip_existing,
        ))
    else:
        n_processed = n_skipped = n_no_images = 0
        for feat in features:
            osm_id = feat["properties"]["osm_id"]
            if args.skip_existing and already_done(osm_id):
                n_skipped += 1
                continue
            images = list_images_for(osm_id)
            if not images:
                n_no_images += 1
                continue
            print(f"  {osm_id}: {len(images)} images")

            per_image: list[dict] = []
            for img_path in images:
                heading = heading_from_path(img_path)
                if args.backend == "vision-only":
                    detections = process_image_vision_only(img_path, claude_client, args.claude_model)
                else:
                    detections = process_image_with_detector(
                        img_path, args.backend, detector_state,
                        args.with_condition, claude_client, args.claude_model,
                    )
                per_image.append({"image": img_path.name, "heading": heading, "detections": detections})

            with detections_path(osm_id).open("w", encoding="utf-8") as f:
                json.dump({"intersection_id": osm_id, "images": per_image}, f, indent=2)
            n_processed += 1

    print(f"Processed: {n_processed}  skipped: {n_skipped}  no_images: {n_no_images}")

    if args.extract_crops:
        crops_dir = config.RESULTS_DIR / "crops"
        n_crops = extract_crops_from_results(crops_dir)
        print(f"Wrote {n_crops} crops to {crops_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
