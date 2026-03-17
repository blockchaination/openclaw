#!/usr/bin/env python3
"""
Tests for minimum signal-strength gate in quant engine.

Run: python -m pytest test_signal_strength_gate.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from strategy import maker_first_mean_reversion, MIN_SIGNAL_STRENGTH
from quant_engine import _build_status
from operator_status import _format_signal_strength


def test_strong_buy_signal_passes_through() -> None:
    """Strong buy signal (abs(signal_strength) >= 3.0) passes through as buy."""
    # momentum=-3.0, momentum_threshold=1.0 => signal_strength=3.0
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-3.0,
        volatility=20.0 / 3.0,
    )
    assert decision["action"] == "buy"
    assert decision["signal_strength"] == 3.0
    assert abs(decision["signal_strength"]) >= MIN_SIGNAL_STRENGTH


def test_strong_sell_signal_passes_through() -> None:
    """Strong sell signal (abs(signal_strength) >= 3.0) passes through as sell."""
    # momentum=3.0, momentum_threshold=1.0 => signal_strength=-3.0
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=-0.05,
        momentum=3.0,
        volatility=20.0 / 3.0,
    )
    assert decision["action"] == "sell"
    assert decision["signal_strength"] == -3.0
    assert abs(decision["signal_strength"]) >= MIN_SIGNAL_STRENGTH


def test_weak_buy_candidate_becomes_hold() -> None:
    """Weak buy candidate (abs(signal_strength) < 3.0) becomes hold."""
    # momentum=-2.0 < -1.0 (threshold), book_imbalance>0.02 => buy branch
    # signal_strength=2.0 < 3.0 => filtered to hold
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-2.0,
        volatility=20.0 / 3.0,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "weak_signal_filtered"
    assert decision["signal_strength"] == 2.0
    assert abs(decision["signal_strength"]) < MIN_SIGNAL_STRENGTH


def test_weak_sell_candidate_becomes_hold() -> None:
    """Weak sell candidate (abs(signal_strength) < 3.0) becomes hold."""
    # momentum=2.0 > 1.0 (threshold), book_imbalance<-0.02 => sell branch
    # signal_strength=-2.0, abs=2.0 < 3.0 => filtered to hold
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=-0.05,
        momentum=2.0,
        volatility=20.0 / 3.0,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "weak_signal_filtered"
    assert decision["signal_strength"] == -2.0
    assert abs(decision["signal_strength"]) < MIN_SIGNAL_STRENGTH


def test_no_mean_reversion_signal_produces_none() -> None:
    """No mean-reversion signal -> signal_strength is None."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.01,
        momentum=0.0,
        volatility=5.0,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "no mean-reversion signal"
    assert decision["signal_strength"] is None


def test_invalid_input_produces_none() -> None:
    """Invalid mid_price or spread -> signal_strength is None."""
    decision = maker_first_mean_reversion(
        mid_price=0.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-2.0,
        volatility=5.0,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "invalid mid_price or spread"
    assert decision["signal_strength"] is None


def test_volatility_negative_produces_none() -> None:
    """Volatility < 0 -> signal_strength is None."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-2.0,
        volatility=-0.5,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "volatility < 0"
    assert decision["signal_strength"] is None


def test_signal_strength_none_in_status_shows_dash() -> None:
    """Status with signal_strength=None: key present, value None (operator shows '-')."""
    result = {
        "strategy": {"action": "hold", "signal_strength": None},
        "order": {"submitted": False, "skipped_reason": None},
        "raw_signal": "hold",
        "final_action": "hold",
        "decision_reason": "no mean-reversion signal",
        "signal_strength": None,
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert "signal_strength" in status
    assert status["signal_strength"] is None


def test_signal_strength_present_when_filtered() -> None:
    """signal_strength present in status when weak signal filtered."""
    result = {
        "strategy": {
            "action": "hold",
            "reason": "weak_signal_filtered",
            "signal_strength": 2.0,
        },
        "order": {"submitted": False, "skipped_reason": None},
        "raw_signal": "hold",
        "final_action": "hold",
        "decision_reason": "weak_signal_filtered",
        "signal_strength": 2.0,
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status["signal_strength"] == 2.0
    assert status["decision_reason"] == "weak_signal_filtered"


def test_no_mean_reversion_runtime_status_signal_strength_none() -> None:
    """No mean-reversion signal -> runtime/status signal_strength is None (even with stale -0.0)."""
    result = {
        "strategy": {"action": "hold", "reason": "no mean-reversion signal", "signal_strength": None},
        "order": {"submitted": False, "skipped_reason": None},
        "raw_signal": "hold",
        "final_action": "hold",
        "decision_reason": "no mean-reversion signal",
        "signal_strength": -0.0,
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status["signal_strength"] is None


def test_invalid_input_runtime_status_signal_strength_none() -> None:
    """Invalid input -> runtime/status signal_strength is None."""
    result = {
        "strategy": {"action": "hold", "reason": "invalid mid_price or spread", "signal_strength": None},
        "order": {"submitted": False, "skipped_reason": None},
        "raw_signal": "hold",
        "final_action": "hold",
        "decision_reason": "invalid mid_price or spread",
        "signal_strength": 0.0,
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status["signal_strength"] is None


def test_operator_format_signal_strength_none_shows_dash() -> None:
    """Operator formatting: None -> '-'."""
    assert _format_signal_strength(None) == "-"


def test_operator_format_signal_strength_negative_zero_shows_zero() -> None:
    """Operator formatting: -0.0 -> '0.0' (never '-0.0')."""
    assert _format_signal_strength(-0.0) == "0.0"
