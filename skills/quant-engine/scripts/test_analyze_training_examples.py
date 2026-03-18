#!/usr/bin/env python3
"""
Tests for analyze_training_examples.py.

Run: python -m pytest test_analyze_training_examples.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_training_examples import (
    _compute_metrics,
    _load_completed_records,
)


def test_empty_file_safe() -> None:
    """Empty file returns empty metrics."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        records = _load_completed_records(path)
        assert records == []
        m = _compute_metrics(records)
        assert m["total"] == 0
        assert m["buy"] == 0
        assert m["sell"] == 0
        assert m["flat"] == 0
        assert m["long"] == 0
        assert m["label_30s_positive_rate"] == 0.0
        assert m["label_300s_positive_rate"] == 0.0
    finally:
        path.unlink(missing_ok=True)


def test_incomplete_rows_ignored() -> None:
    """Records without price_after_300s or label_300s are ignored."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        path.write_text(
            json.dumps({
                "timestamp": "2026-03-15T12:00:00Z",
                "pair": "XBTUSD",
                "candidate_side": "buy",
                "spot_state": "FLAT",
                "mid_price": 100.0,
                "price_after_30s": 101.0,
                "price_after_60s": 102.0,
                "label_30s": 1,
                "label_60s": 1,
            }) + "\n",
            encoding="utf-8",
        )
        records = _load_completed_records(path)
        assert len(records) == 0
    finally:
        path.unlink(missing_ok=True)


def test_incomplete_missing_label_300s_ignored() -> None:
    """Records with price_after_300s but no label_300s are ignored."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        path.write_text(
            json.dumps({
                "timestamp": "2026-03-15T12:00:00Z",
                "pair": "XBTUSD",
                "candidate_side": "buy",
                "spot_state": "FLAT",
                "mid_price": 100.0,
                "price_after_30s": 101.0,
                "price_after_60s": 102.0,
                "price_after_300s": 105.0,
                "label_30s": 1,
                "label_60s": 1,
            }) + "\n",
            encoding="utf-8",
        )
        records = _load_completed_records(path)
        assert len(records) == 0
    finally:
        path.unlink(missing_ok=True)


def test_counts_and_positive_rate_correct() -> None:
    """Counts and positive-rate calculations are correct."""
    records = [
        {
            "candidate_side": "buy",
            "spot_state": "FLAT",
            "signal_strength": 3.0,
            "label_30s": 1,
            "label_60s": 1,
            "label_300s": 1,
        },
        {
            "candidate_side": "buy",
            "spot_state": "FLAT",
            "signal_strength": 2.5,
            "label_30s": 0,
            "label_60s": 0,
            "label_300s": 0,
        },
        {
            "candidate_side": "sell",
            "spot_state": "LONG",
            "signal_strength": -3.0,
            "label_30s": 1,
            "label_60s": 1,
            "label_300s": 1,
        },
    ]
    m = _compute_metrics(records)
    assert m["total"] == 3
    assert m["buy"] == 2
    assert m["sell"] == 1
    assert m["flat"] == 2
    assert m["long"] == 1
    assert m["buy_in_flat"] == 2
    assert m["buy_in_long"] == 0
    assert m["sell_in_flat"] == 0
    assert m["sell_in_long"] == 1
    assert abs(m["avg_signal_strength_buy"] - 2.75) < 1e-9
    assert abs(m["avg_signal_strength_sell"] - (-3.0)) < 1e-9
    assert abs(m["label_30s_positive_rate"] - (2 / 3 * 100)) < 1e-6
    assert abs(m["label_60s_positive_rate"] - (2 / 3 * 100)) < 1e-6
    assert abs(m["label_300s_positive_rate"] - (2 / 3 * 100)) < 1e-6
