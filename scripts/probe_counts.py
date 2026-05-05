"""Show detection counts per heading for a given intersection."""

import json
import sys
from pathlib import Path

osm_id = sys.argv[1] if len(sys.argv) > 1 else "osm_31329116"
path = Path(__file__).resolve().parents[1] / "data" / "results" / f"{osm_id}_detections.json"
doc = json.loads(path.read_text(encoding="utf-8"))
print(f"{osm_id}:")
for img in doc["images"]:
    print(f"  h{img['heading']:3d}  {img['image']}: {len(img['detections'])} detections")
