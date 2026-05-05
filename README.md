# Anchorage Intersection Asset Inventory

Using open data and computer vision to build a composable infrastructure audit
system for the Municipality of Anchorage. Built for the MOA i-team in the
Bloomberg Center for Government Excellence / civic-innovation tradition: take
public data sources (OpenStreetMap, Google Street View) and modern foundation
models, and turn them into something Traffic Engineering can actually act on.

The pipeline produces a GeoJSON feature layer of every signalized intersection
in Anchorage, annotated with the assets present (signal heads, mast arms,
pedestrian signals, crosswalk markings, curb ramps, push buttons, signs, and
more) and an apparent-condition score for each. The output is suitable for
import to ArcGIS Online and for sharing with MOA Traffic Engineering or
publishing as a public feature layer.

## Pipeline phases

1. **Fetch signals** (`src/fetch_signals.py`) — Overpass API → all
   `highway=traffic_signals` nodes inside Anchorage. Clusters nodes within
   30 m so each physical intersection is one feature. Outputs
   `data/signals.geojson`.

2. **Fetch imagery** (`src/fetch_imagery.py`) — for each intersection, checks
   the Google Street View metadata endpoint (free) and downloads four
   headings (N/E/S/W) at 640×640. Writes `data/images/<osm_id>_h<heading>.jpg`
   and a manifest CSV with `gsv_pano_id` and `gsv_date` per image.

3. **Detect assets** (`src/detect_assets.py`) — three pluggable backends
   selected with `--backend`:
   - **`sam3`** *(default, GPU)* — Meta's SAM 3 running locally via
     Ultralytics. Open-vocabulary text prompts from
     `config/asset_prompts.yaml`. Free at runtime, ~3.4 GB VRAM.
   - **`roboflow`** — Roboflow hosted inference. Requires a workspace /
     project / version pointing at a road-asset model.
   - **`vision-only`** — Claude Vision on full images. No detector, no GPU,
     no Roboflow account; slowest per-image but simplest setup.

   Add `--with-condition` to any backend (default for `vision-only`) to send
   each detected asset's crop to Claude Vision for a focused condition
   score. Add `--extract-crops` to also write per-asset JPEGs under
   `data/results/crops/` for review.

4. **Assess condition** (`src/assess_condition.py`) — aggregates per-image
   detections into a per-intersection inventory: count, average condition
   score, worst observed condition, and free-text notes per asset class.

5. **Export GeoJSON** (`src/export_geojson.py`) — assembles the inventories
   into `data/results/intersection_inventory.geojson` (one Point feature per
   intersection, with flattened `<asset>_count` / `<asset>_avg_score` /
   `<asset>_worst_condition` columns for ArcGIS) plus a parallel CSV.

6. **Visualize priority findings** (`src/visualize_priority.py`) — picks each
   intersection's worst heading, generates an annotated GSV image with
   severity-coded labels (poor=red, fair-on-critical=orange,
   unassessable-on-critical=blue), and writes a ranked
   `data/results/priority/index.md` with Google Maps deep-links to each
   intersection. Critical asset types — those where "fair" is still a safety
   concern — are: crosswalk markings, lane markings, curb ramps, push
   buttons, and pedestrian signals.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

API keys you'll need:

- `GSV_API_KEY` — Google Cloud console; enable the Street View Static API.
- `ANTHROPIC_API_KEY` — console.anthropic.com (used for condition scoring via
  Claude Vision; required for `--backend vision-only` or `--with-condition`).
- `HF_TOKEN` — *only for `--backend sam3`*. Accept the model license at
  https://huggingface.co/facebook/sam3, then create a token at
  https://huggingface.co/settings/tokens and run `huggingface-cli login`.
  The first SAM 3 run downloads ~3.4 GB of weights from Hugging Face.
- `ROBOFLOW_API_KEY` *(plus `ROBOFLOW_WORKSPACE` / `_PROJECT` / `_VERSION`)*
  — *only for `--backend roboflow`*.

## Running

```bash
python scripts/run_pipeline.py --phase 1                            # fetch signals
python scripts/run_pipeline.py --phase 2 --dry-run                  # check GSV coverage only
python scripts/run_pipeline.py --phase 2                            # download images
python scripts/run_pipeline.py --phase 3 --backend sam3             # GPU detection (default)
python scripts/run_pipeline.py --phase 3 --backend sam3 --with-condition   # + Claude crop scoring
python scripts/run_pipeline.py --phase 3 --backend vision-only      # CPU/no-GPU fallback
python scripts/run_pipeline.py --phase 3 --backend roboflow         # Roboflow API path
python scripts/run_pipeline.py --phase 4                            # aggregate
python scripts/run_pipeline.py --phase 5                            # export
python scripts/run_pipeline.py --phase 6                            # priority visualizations
python scripts/run_pipeline.py --all --backend vision-only --limit 10
```

`--limit N` runs against only the first N intersections — use it heavily while
iterating on Phase 3 prompts before scaling out.

## Cost estimate

At current API rates and ~321 signalized intersections × 4 headings = ~1,284
images:

- **Google Street View Static API** — ~$7 per 1,000 images → **~$8.40**.
  The metadata endpoint (used in `--dry-run`) is free.
- **`--backend sam3`** — $0 at runtime. One-time ~3.4 GB model download.
- **`--backend vision-only`** (Claude Sonnet on full images) —
  ~$0.02/image → **~$25** for a full Anchorage pass.
- **`--with-condition` add-on** for sam3 / roboflow — sends one Claude call
  per detected asset crop (~10 detections per image). Crops are smaller and
  responses are tiny, so cost is **~$5–10** for the full dataset.
- **Roboflow** — free tier covers evaluation; paid tier ~$0.002–0.004 per
  inference (~$3-6 for the full dataset).

| Configuration | Total cost | Notes |
|---|---|---|
| sam3 only (inventory, no condition) | **~$8.40** | GSV alone |
| sam3 + `--with-condition` | **~$13–18** | recommended for v1 |
| vision-only | **~$33** | simplest, no GPU needed |

**Vision-only run time** scales with `--concurrency`. Sequential is ~11s per
image (~4 hours for full Anchorage). Measured 8× speedup at `--concurrency 8`
(40 images in 54s on the 10-intersection sample), so a full Anchorage pass is
**~30 minutes** at `--concurrency 8`. Anthropic's per-account rate limits
comfortably allow 8-10 concurrent in-flight calls on Sonnet.

```bash
python scripts/run_pipeline.py --phase 3 --backend vision-only --concurrency 8
```

## Notes

- Anchorage has long winters; Street View imagery captured under snow makes
  pavement-marking and curb-ramp scoring unreliable. The `gsv_date` field is
  carried through the pipeline so you can filter for summer imagery in
  ArcGIS.
- Some signals in Anchorage are on state routes managed by ADOT&PF rather
  than MOA. Where OSM tags it, the road operator is preserved on the output
  feature (`osm_operator`).
- Each phase reads from and writes to `data/` so you can re-run any phase
  independently. Detection JSON is per intersection so partial progress is
  cheap to resume.
