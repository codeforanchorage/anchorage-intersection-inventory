"""Tests for src.fetch_imagery.median_metadata_age_days — the helper that
decides whether to print a stale-metadata warning at startup."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fetch_imagery import median_metadata_age_days


def _entry(days_ago: int | None) -> dict:
    if days_ago is None:
        return {"fetched_at": ""}
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago))
    return {"fetched_at": ts.isoformat(timespec="seconds")}


def test_empty_cache_returns_none():
    assert median_metadata_age_days({}) is None


def test_all_missing_timestamps_returns_none():
    cache = {"a": _entry(None), "b": _entry(None)}
    assert median_metadata_age_days(cache) is None


def test_single_entry_returns_its_age():
    cache = {"a": _entry(42)}
    age = median_metadata_age_days(cache)
    assert age is not None
    assert 41.5 <= age <= 42.5


def test_odd_count_returns_middle():
    cache = {
        "a": _entry(10),
        "b": _entry(50),
        "c": _entry(200),
    }
    age = median_metadata_age_days(cache)
    assert age is not None
    assert 49.5 <= age <= 50.5


def test_even_count_returns_average_of_two_middles():
    cache = {
        "a": _entry(10),
        "b": _entry(20),
        "c": _entry(30),
        "d": _entry(40),
    }
    age = median_metadata_age_days(cache)
    assert age is not None
    assert 24.5 <= age <= 25.5


def test_naive_timestamps_are_treated_as_utc():
    # An old manifest row written by an older tool may have a naive ISO ts.
    cache = {"a": {"fetched_at": "2025-01-01T00:00:00"}}
    age = median_metadata_age_days(cache)
    assert age is not None
    assert age > 365  # somewhere over a year ago, exact value depends on test run date


def test_garbage_timestamp_skipped_not_crashed():
    cache = {
        "a": {"fetched_at": "not a real timestamp"},
        "b": _entry(100),
    }
    age = median_metadata_age_days(cache)
    assert age is not None
    assert 99.5 <= age <= 100.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
