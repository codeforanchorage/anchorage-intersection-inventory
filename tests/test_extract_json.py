"""Tests for src.detect_assets._extract_json — Claude Vision occasionally
wraps its JSON in markdown fences or adds prose. This scraper is the only
thing standing between Phase 3 and a parse error, so its corner cases
deserve to be pinned."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detect_assets import _extract_json


def test_bare_json_object():
    assert _extract_json('{"asset_type": "mast_arm"}') == {"asset_type": "mast_arm"}


def test_bare_json_array():
    assert _extract_json('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]


def test_fenced_json_block():
    text = '```json\n{"asset_type": "signal_pole"}\n```'
    assert _extract_json(text) == {"asset_type": "signal_pole"}


def test_fenced_block_no_language():
    text = '```\n[1, 2, 3]\n```'
    assert _extract_json(text) == [1, 2, 3]


def test_prose_wrapped_object():
    text = "Here's what I found: {\"asset_type\": \"crosswalk_marking\", \"condition\": \"poor\"}. Hope this helps!"
    result = _extract_json(text)
    assert result == {"asset_type": "crosswalk_marking", "condition": "poor"}


def test_prose_wrapped_array():
    text = "Found these:\n[{\"asset_type\": \"mast_arm\"}, {\"asset_type\": \"signal_pole\"}]\nThat's all."
    result = _extract_json(text)
    assert result == [{"asset_type": "mast_arm"}, {"asset_type": "signal_pole"}]


def test_nested_braces_in_string_value():
    # The depth tracker must not be fooled by braces inside string literals.
    text = '{"notes": "saw {something} weird", "ok": true}'
    assert _extract_json(text) == {"notes": "saw {something} weird", "ok": True}


def test_returns_none_on_garbage():
    assert _extract_json("not json at all") is None


def test_returns_none_on_unbalanced():
    assert _extract_json('{"asset_type":') is None


def test_handles_escaped_quotes_in_string():
    text = '{"notes": "tag reads \\"STOP\\""}'
    result = _extract_json(text)
    assert result == {"notes": 'tag reads "STOP"'}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
