#!/usr/bin/env python3
"""
Tests for analyze_signal_outcomes.py.

Run: python -m pytest test_analyze_signal_outcomes.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_signal_outcomes import (
    _compute_metrics,
    _compute_return,
    _load_completed_records,
)


def test_empty_file_safe() -> None:
    """Empty file returns empty metrics."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        records = _load_completed_records(path)
        assert records == []
        metrics = _compute_metrics(records)
        assert metrics["total_signals"] == 0
        assert metrics["buy_signals"] == 0
        assert metrics["sell_signals"] == 0
        assert metrics["average_return_30s"] == 0.0
        assert metrics["win_rate_300s"] == 0.0
    finally:
        path.unlink(missing_ok=True)


def test_incomplete_signals_ignored() -> None:
    """Records without price_after_300s are ignored."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        # Incomplete: no price_after_300s
        path.write_text(
            json.dumps({
                "timestamp": "2026-03-15T12:00:00Z",
                "pair": "XBTUSD",
                "signal": "buy",
                "price_at_signal": 100.0,
                "price_after_30s": 101.0,
            }) + "\n",
            encoding="utf-8",
        )
        records = _load_completed_records(path)
        assert len(records) == 0
    finally:
        path.unlink(missing_ok=True)


def test_metrics_calculation() -> None:
    """Metrics are computed correctly from completed records."""
    # Buy: price 100 -> 102 at 30s, 103 at 60s, 105 at 300s
    #   return_30 = (102-100)/100 = 0.02, return_60 = 0.03, return_300 = 0.05
    # Sell: price 100 -> 98 at 30s, 97 at 60s, 95 at 300s
    #   return_30 = (100-98)/100 = 0.02, return_60 = 0.03, return_300 = 0.05
    records = [
        {
            "timestamp": "2026-03-15T12:00:00Z",
            "pair": "XBTUSD",
            "signal": "buy",
            "price_at_signal": 100.0,
            "price_after_30s": 102.0,
            "price_after_60s": 103.0,
            "price_after_300s": 105.0,
        },
        {
            "timestamp": "2026-03-15T12:01:00Z",
            "pair": "XBTUSD",
            "signal": "sell",
            "price_at_signal": 100.0,
            "price_after_30s": 98.0,
            "price_after_60s": 97.0,
            "price_after_300s": 95.0,
        },
    ]
    metrics = _compute_metrics(records)
    assert metrics["total_signals"] == 2
    assert metrics["buy_signals"] == 1
    assert metrics["sell_signals"] == 1
    assert abs(metrics["average_return_30s"] - 0.02) < 1e-9
    assert abs(metrics["average_return_60s"] - 0.03) < 1e-9
    assert abs(metrics["average_return_300s"] - 0.05) < 1e-9
    assert metrics["win_rate_300s"] == 100.0  # both positive


def test_compute_return_buy() -> None:
    """Buy return = (future - price_at_signal) / price_at_signal."""
    rec = {"signal": "buy", "price_at_signal": 100.0, "price_after_30s": 102.0}
    r = _compute_return(rec, "price_after_30s")
    assert r is not None
    assert abs(r - 0.02) < 1e-9


def test_compute_return_sell() -> None:
    """Sell return = (price_at_signal - future) / price_at_signal."""
    rec = {"signal": "sell", "price_at_signal": 100.0, "price_after_30s": 98.0}
    r = _compute_return(rec, "price_after_30s")
    assert r is not None
    assert abs(r - 0.02) < 1e-9


def test_win_rate_mixed() -> None:
    """Win rate counts signals with return_300s > 0."""
    records = [
        {"signal": "buy", "price_at_signal": 100.0, "price_after_30s": 101.0, "price_after_60s": 101.0, "price_after_300s": 102.0},
        {"signal": "buy", "price_at_signal": 100.0, "price_after_30s": 99.0, "price_after_60s": 99.0, "price_after_300s": 98.0},
    ]
    metrics = _compute_metrics(records)
    assert metrics["win_rate_300s"] == 50.0
