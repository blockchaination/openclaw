#!/usr/bin/env python3
"""
Tests for signal outcome tracking (append_signal_outcome, JSONL format).

Run: python -m pytest test_signal_outcomes.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from quant_engine import append_signal_outcome, default_signal_outcomes_path


def test_signal_event_written() -> None:
    """Signal event is written to JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        record = {
            "timestamp": "2026-03-15T12:00:00Z",
            "pair": "XBTUSD",
            "signal": "buy",
            "price_at_signal": 62350.21,
        }
        append_signal_outcome(path, record)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["timestamp"] == "2026-03-15T12:00:00Z"
        assert parsed["pair"] == "XBTUSD"
        assert parsed["signal"] == "buy"
        assert parsed["price_at_signal"] == 62350.21
    finally:
        path.unlink(missing_ok=True)


def test_outcome_appended_later() -> None:
    """Outcome record with price_after_* can be appended."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        partial = {"timestamp": "2026-03-15T12:00:00Z", "pair": "XBTUSD", "signal": "sell", "price_at_signal": 62350.21}
        append_signal_outcome(path, partial)
        complete = {
            **partial,
            "price_after_30s": 62380.10,
            "price_after_60s": 62410.44,
            "price_after_300s": 62500.12,
        }
        append_signal_outcome(path, complete)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        second = json.loads(lines[1])
        assert second["price_after_30s"] == 62380.10
        assert second["price_after_60s"] == 62410.44
        assert second["price_after_300s"] == 62500.12
    finally:
        path.unlink(missing_ok=True)


def test_jsonl_format_valid() -> None:
    """Written records are valid JSONL (one JSON object per line)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        for record in [
            {"timestamp": "2026-03-15T12:00:00Z", "pair": "XBTUSD", "signal": "buy", "price_at_signal": 100.0},
            {"timestamp": "2026-03-15T12:01:00Z", "pair": "XBTUSD", "signal": "sell", "price_at_signal": 101.0},
        ]:
            append_signal_outcome(path, record)
        content = path.read_text(encoding="utf-8")
        for line in content.strip().split("\n"):
            obj = json.loads(line)
            assert "timestamp" in obj and "pair" in obj and "signal" in obj and "price_at_signal" in obj
    finally:
        path.unlink(missing_ok=True)


def test_default_signal_outcomes_path() -> None:
    """default_signal_outcomes_path returns logs/signal_outcomes.jsonl under repo root."""
    p = default_signal_outcomes_path()
    assert p.name == "signal_outcomes.jsonl"
    assert "logs" in str(p)
