"""Print detection counts per heading for every intersection that has results."""

import json
from pathlib import Path

results = Path(__file__).resolve().parents[1] / "data" / "results"
total = 0
for path in sorted(results.glob("*_detections.json")):
    doc = json.loads(path.read_text(encoding="utf-8"))
    counts = [len(img["detections"]) for img in doc["images"]]
    print(f"{doc['intersection_id']}: {counts}  total={sum(counts)}")
    total += sum(counts)
print(f"\nGrand total: {total} detections")
