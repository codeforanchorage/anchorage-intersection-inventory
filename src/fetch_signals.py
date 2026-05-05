"""Phase 1: Fetch signalized intersections from OpenStreetMap via Overpass API."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import requests
from geojson import Feature, FeatureCollection, Point, dump

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def query_overpass(query: str) -> dict:
    """POST a query to the Overpass API and return parsed JSON."""
    response = requests.post(
        config.OVERPASS_ENDPOINT,
        data={"data": query},
        headers={"User-Agent": "anchorage-intersection-inventory/0.1"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def fetch_signal_nodes() -> list[dict]:
    """Fetch traffic_signals nodes for Anchorage. Falls back to bbox if area query is empty."""
    data = query_overpass(config.OVERPASS_QUERY)
    elements = data.get("elements", [])
    if not elements:
        print("Named-area query returned 0 nodes — falling back to bounding box query.")
        data = query_overpass(config.OVERPASS_QUERY_BBOX)
        elements = data.get("elements", [])
    return [el for el in elements if el.get("type") == "node"]


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cluster_nodes(nodes: list[dict], radius_m: float) -> list[list[dict]]:
    """Greedy single-link clustering: any node within radius_m of an existing cluster joins it."""
    clusters: list[list[dict]] = []
    for node in nodes:
        lat, lon = node["lat"], node["lon"]
        placed = False
        for cluster in clusters:
            for member in cluster:
                if haversine_meters(lat, lon, member["lat"], member["lon"]) <= radius_m:
                    cluster.append(node)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([node])
    return clusters


def cluster_centroid(cluster: list[dict]) -> tuple[float, float]:
    lat = sum(n["lat"] for n in cluster) / len(cluster)
    lon = sum(n["lon"] for n in cluster) / len(cluster)
    return lat, lon


def merge_tags(cluster: list[dict]) -> dict:
    """Union of OSM tags across cluster members. Conflicts: keep first non-empty value."""
    merged: dict = {}
    for node in cluster:
        for k, v in (node.get("tags") or {}).items():
            merged.setdefault(k, v)
    return merged


def build_features(clusters: Iterable[list[dict]]) -> list[Feature]:
    features = []
    for cluster in clusters:
        lat, lon = cluster_centroid(cluster)
        ids = sorted(n["id"] for n in cluster)
        primary_id = ids[0]
        tags = merge_tags(cluster)
        properties = {
            "osm_id": f"osm_{primary_id}",
            "osm_node_ids": ids,
            "lat": lat,
            "lon": lon,
            "cluster_size": len(cluster),
            **tags,
        }
        features.append(Feature(geometry=Point((lon, lat)), properties=properties))
    return features


def write_geojson(features: list[Feature], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        dump(FeatureCollection(features), f, indent=2)


def main() -> int:
    print("Querying Overpass API for Anchorage traffic signals...")
    nodes = fetch_signal_nodes()
    print(f"  raw signal nodes: {len(nodes)}")

    if not nodes:
        print("ERROR: no signal nodes returned — check connectivity or query.")
        return 1

    clusters = cluster_nodes(nodes, config.CLUSTER_RADIUS_METERS)
    print(f"  clustered intersections (radius {config.CLUSTER_RADIUS_METERS} m): {len(clusters)}")

    features = build_features(clusters)
    write_geojson(features, config.SIGNALS_GEOJSON)
    print(f"Wrote {len(features)} features to {config.SIGNALS_GEOJSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
