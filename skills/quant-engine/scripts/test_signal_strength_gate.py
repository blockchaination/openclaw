#!/usr/bin/env python3
"""
Tests for signal_strength contract enforcement in quant engine.

Run: python -m pytest test_signal_strength_gate.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from strategy import maker_first_mean_reversion, MIN_SIGNAL_STRENGTH
from quant_engine import _build_status, _resolve_signal_strength
from operator_status import _format_signal_strength


def test_no_mean_reversion_signal_strength_none_in_result_and_status() -> None:
    """No mean-reversion signal -> signal_strength is None (in result AND status)."""
    decision = {"action": "hold", "reason": "no mean-reversion signal", "signal_strength": -0.0}
    assert _resolve_signal_strength(decision) is None
    result = {
        "strategy": decision,
        "order": {},
        "raw_signal": "hold",
        "final_action": "hold",
        "decision_reason": "no mean-reversion signal",
        "signal_strength": _resolve_signal_strength(decision),
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status["signal_strength"] is None


def test_weak_signal_filtered_retains_numeric() -> None:
    """weak_signal_filtered -> retains numeric value."""
    decision = {"action": "hold", "reason": "weak_signal_filtered", "signal_strength": 2.0}
    assert _resolve_signal_strength(decision) == 2.0
    result = {
        "strategy": decision,
        "order": {},
        "raw_signal": "hold",
        "final_action": "hold",
        "decision_reason": "weak_signal_filtered",
        "signal_strength": 2.0,
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status["signal_strength"] == 2.0


def test_strong_buy_retains_numeric() -> None:
    """Strong buy -> retains numeric value."""
    decision = {"action": "buy", "reason": "buy mean-reversion signal", "signal_strength": 3.0}
    assert _resolve_signal_strength(decision) == 3.0


def test_strong_sell_retains_numeric() -> None:
    """Strong sell -> retains numeric value."""
    decision = {"action": "sell", "reason": "sell mean-reversion signal", "signal_strength": -3.0}
    assert _resolve_signal_strength(decision) == -3.0


def test_operator_prints_dash_for_none() -> None:
    """operator_status prints '-' for None."""
    assert _format_signal_strength(None) == "-"


def test_operator_never_prints_negative_zero() -> None:
    """operator_status NEVER prints '-0.0'."""
    assert _format_signal_strength(-0.0) == "0.0"
    assert "-0.0" not in _format_signal_strength(-0.0)


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


