"""Tally condition labels across the current detection JSONs."""

import json
from collections import Counter
from pathlib import Path

results = Path(__file__).resolve().parents[1] / "data" / "results"
overall = Counter()
poor_by_intersection = {}

for path in sorted(results.glob("*_detections.json")):
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    poor_count = 0
    for img in doc.get("images", []):
        for det in img.get("detections", []):
            cond = det.get("condition", "missing")
            overall[cond] += 1
            if cond == "poor":
                poor_count += 1
    if poor_count:
        poor_by_intersection[doc["intersection_id"]] = poor_count

print("Condition tally across all current detections:")
for cond, n in overall.most_common():
    print(f"  {cond:20s}  {n}")
print()
print(f"Intersections with 'poor' findings: {len(poor_by_intersection)}")
for i, n in poor_by_intersection.items():
    print(f"  {i}: {n}")
