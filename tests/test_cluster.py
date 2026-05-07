"""Tests for src.fetch_signals.cluster_nodes — single-link clustering of
OSM signal nodes. Anchorage produces ~574 raw nodes that collapse to ~321
clustered intersections; misclustering shows up as a single ArcGIS feature
where MOA expects two (or vice versa)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fetch_signals import cluster_nodes, haversine_meters


def node(node_id: int, lat: float, lon: float, **tags) -> dict:
    return {"id": node_id, "lat": lat, "lon": lon, "type": "node", "tags": tags or {}}


def test_haversine_known_anchorage_distance():
    # 1st & C St (61.2181, -149.8990) to 1st & D St (61.2181, -149.9013) — about 124m.
    d = haversine_meters(61.2181, -149.8990, 61.2181, -149.9013)
    assert 100 <= d <= 150, f"expected ~124m got {d:.1f}m"


def test_isolated_node_is_its_own_cluster():
    nodes = [node(1, 61.2181, -149.8990)]
    clusters = cluster_nodes(nodes, radius_m=30.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_two_close_nodes_cluster_together():
    # ~10 m apart — well within 30 m radius.
    nodes = [node(1, 61.2181, -149.8990), node(2, 61.21819, -149.8990)]
    clusters = cluster_nodes(nodes, radius_m=30.0)
    assert len(clusters) == 1
    assert {n["id"] for n in clusters[0]} == {1, 2}


def test_two_far_nodes_stay_separate():
    # ~150 m apart — way past 30 m.
    nodes = [node(1, 61.2181, -149.8990), node(2, 61.2181, -149.8975)]
    clusters = cluster_nodes(nodes, radius_m=30.0)
    assert len(clusters) == 2


def test_chain_of_three_nodes_is_known_limitation():
    # Greedy single-link: A-B = 25 m, B-C = 25 m, A-C = 50 m. With radius=30,
    # A and C end up in the same cluster via B. This is the documented
    # behavior; documenting it here so a future change to centroid-link or
    # DBSCAN flags this test as needing review.
    nodes = [
        node(1, 61.21810, -149.8990),
        node(2, 61.21833, -149.8990),  # ~25 m from #1
        node(3, 61.21856, -149.8990),  # ~25 m from #2, ~50 m from #1
    ]
    clusters = cluster_nodes(nodes, radius_m=30.0)
    assert len(clusters) == 1
    assert {n["id"] for n in clusters[0]} == {1, 2, 3}


def test_radius_zero_keeps_each_node_separate():
    nodes = [node(i, 61.2181 + i * 0.0001, -149.8990) for i in range(5)]
    clusters = cluster_nodes(nodes, radius_m=0.0)
    assert len(clusters) == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
