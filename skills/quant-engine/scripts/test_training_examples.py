#!/usr/bin/env python3
"""
Tests for training example logging (append_training_example, label computation).

Run: python -m pytest test_training_examples.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from quant_engine import (
    _compute_training_labels,
    _is_directional_candidate,
    append_training_example,
    default_training_examples_path,
)


def test_neutral_hold_not_written() -> None:
    """Neutral hold (no mean-reversion, invalid input) is not a directional candidate."""
    assert _is_directional_candidate({
        "strategy": {"action": "hold"},
        "decision_reason": "no long-entry signal",
    }) is False
    assert _is_directional_candidate({
        "strategy": {"action": "hold"},
        "decision_reason": "no exit signal",
    }) is False
    assert _is_directional_candidate({
        "strategy": {"action": "hold"},
        "decision_reason": "invalid mid_price or spread",
    }) is False
    assert _is_directional_candidate({
        "strategy": {"action": "hold"},
        "decision_reason": "volatility < 0",
    }) is False


def test_directional_candidate_detected() -> None:
    """Buy, sell, and weak_signal_filtered are directional candidates."""
    assert _is_directional_candidate({
        "strategy": {"action": "buy"},
        "decision_reason": "buy mean-reversion entry",
    }) is True
    assert _is_directional_candidate({
        "strategy": {"action": "sell"},
        "decision_reason": "sell mean-reversion exit",
    }) is True
    assert _is_directional_candidate({
        "strategy": {"action": "hold"},
        "decision_reason": "weak_signal_filtered",
    }) is True


def test_directional_candidate_row_written() -> None:
    """A complete training example row can be written with required schema fields."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        record = {
            "timestamp": "2026-03-15T12:00:00Z",
            "pair": "XBTUSD",
            "runtime_mode": "paper",
            "spot_state": "FLAT",
            "candidate_side": "buy",
            "decision_reason": "buy mean-reversion entry",
            "signal_strength": 3.0,
            "mid_price": 62350.0,
            "spread": 10.0,
            "book_imbalance": 0.05,
            "momentum": -3.0,
            "volatility": 5.0,
            "momentum_threshold": 1.0,
            "price_after_30s": 62400.0,
            "price_after_60s": 62450.0,
            "price_after_300s": 62500.0,
            "label_30s": 1,
            "label_60s": 1,
            "label_300s": 1,
        }
        append_training_example(path, record)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["timestamp"] == "2026-03-15T12:00:00Z"
        assert parsed["pair"] == "XBTUSD"
        assert parsed["spot_state"] == "FLAT"
        assert parsed["candidate_side"] == "buy"
        assert parsed["signal_strength"] == 3.0
        assert parsed["label_30s"] == 1
        assert parsed["label_300s"] == 1
    finally:
        path.unlink(missing_ok=True)


def test_labels_computed_correctly_for_buy() -> None:
    """Buy candidate: label=1 when future > price_at_signal, else 0."""
    labels = _compute_training_labels(
        price_at_signal=100.0,
        candidate_side="buy",
        price_after_30s=105.0,
        price_after_60s=95.0,
        price_after_300s=100.0,
    )
    assert labels["label_30s"] == 1
    assert labels["label_60s"] == 0
    assert labels["label_300s"] == 0


def test_labels_computed_correctly_for_sell() -> None:
    """Sell candidate: label=1 when future < price_at_signal, else 0."""
    labels = _compute_training_labels(
        price_at_signal=100.0,
        candidate_side="sell",
        price_after_30s=95.0,
        price_after_60s=105.0,
        price_after_300s=90.0,
    )
    assert labels["label_30s"] == 1
    assert labels["label_60s"] == 0
    assert labels["label_300s"] == 1


def test_labels_none_when_price_missing() -> None:
    """Labels are None when future price is missing."""
    labels = _compute_training_labels(
        price_at_signal=100.0,
        candidate_side="buy",
        price_after_30s=105.0,
        price_after_60s=None,
        price_after_300s=None,
    )
    assert labels["label_30s"] == 1
    assert labels["label_60s"] is None
    assert labels["label_300s"] is None


def test_delayed_outcome_fields_appended() -> None:
    """Complete record with price_after and labels can be appended."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        base = {
            "timestamp": "2026-03-15T12:00:00Z",
            "pair": "XBTUSD",
            "runtime_mode": "paper",
            "spot_state": "LONG",
            "candidate_side": "sell",
            "decision_reason": "sell mean-reversion exit",
            "signal_strength": -3.0,
            "mid_price": 62350.0,
            "spread": 10.0,
            "book_imbalance": -0.05,
            "momentum": 3.0,
            "volatility": 5.0,
            "momentum_threshold": 1.0,
        }
        labels = _compute_training_labels(
            62350.0, "sell",
            price_after_30s=62300.0,
            price_after_60s=62250.0,
            price_after_300s=62200.0,
        )
        record = {
            **base,
            "price_after_30s": 62300.0,
            "price_after_60s": 62250.0,
            "price_after_300s": 62200.0,
            **labels,
        }
        append_training_example(path, record)
        parsed = json.loads(path.read_text(encoding="utf-8").strip())
        assert parsed["price_after_30s"] == 62300.0
        assert parsed["price_after_300s"] == 62200.0
        assert parsed["label_30s"] == 1
        assert parsed["label_300s"] == 1
    finally:
        path.unlink(missing_ok=True)


def test_jsonl_remains_valid() -> None:
    """Multiple training examples produce valid JSONL."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        for record in [
            {
                "timestamp": "2026-03-15T12:00:00Z",
                "pair": "XBTUSD",
                "runtime_mode": "paper",
                "spot_state": "FLAT",
                "candidate_side": "buy",
                "decision_reason": "buy mean-reversion entry",
                "signal_strength": 3.0,
                "mid_price": 100.0,
                "spread": 1.0,
                "book_imbalance": 0.05,
                "momentum": -3.0,
                "volatility": 5.0,
                "momentum_threshold": 1.0,
                "price_after_30s": 101.0,
                "price_after_60s": 102.0,
                "price_after_300s": 103.0,
                "label_30s": 1,
                "label_60s": 1,
                "label_300s": 1,
            },
            {
                "timestamp": "2026-03-15T12:01:00Z",
                "pair": "XBTUSD",
                "runtime_mode": "paper",
                "spot_state": "LONG",
                "candidate_side": "sell",
                "decision_reason": "weak_signal_filtered",
                "signal_strength": -2.0,
                "mid_price": 100.0,
                "spread": 1.0,
                "book_imbalance": -0.03,
                "momentum": 2.0,
                "volatility": 5.0,
                "momentum_threshold": 1.0,
                "price_after_30s": 99.0,
                "price_after_60s": 98.0,
                "price_after_300s": 97.0,
                "label_30s": 1,
                "label_60s": 1,
                "label_300s": 1,
            },
        ]:
            append_training_example(path, record)
        content = path.read_text(encoding="utf-8")
        for line in content.strip().split("\n"):
            obj = json.loads(line)
            assert "timestamp" in obj and "pair" in obj
            assert "spot_state" in obj and "candidate_side" in obj
            assert "label_30s" in obj and "label_300s" in obj
    finally:
        path.unlink(missing_ok=True)


def test_default_training_examples_path() -> None:
    """default_training_examples_path returns logs/training_examples.jsonl under repo root."""
    p = default_training_examples_path()
    assert p.name == "training_examples.jsonl"
    assert "logs" in str(p)
