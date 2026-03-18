#!/usr/bin/env python3
"""
Tests for signal_strength contract and two-state spot strategy.

Run: python -m pytest test_signal_strength_gate.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from strategy import maker_first_mean_reversion, MIN_SIGNAL_STRENGTH, MIN_XBT_TO_SELL
from quant_engine import _build_status, _resolve_signal_strength
from operator_status import _format_signal_strength

# FLAT: xbt < MIN_XBT_TO_SELL. LONG: xbt >= MIN_XBT_TO_SELL
XBT_FLAT = 0.0
XBT_LONG = MIN_XBT_TO_SELL


def test_flat_state_cannot_produce_sell() -> None:
    """FLAT state cannot produce sell; sell conditions become hold with no long-entry signal."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=-0.05,
        momentum=3.0,
        volatility=20.0 / 3.0,
        xbt_inventory=XBT_FLAT,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "no long-entry signal"
    assert decision["signal_strength"] is None


def test_long_state_cannot_produce_buy() -> None:
    """LONG state cannot produce buy; buy conditions become hold with no exit signal."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-3.0,
        volatility=20.0 / 3.0,
        xbt_inventory=XBT_LONG,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "no exit signal"
    assert decision["signal_strength"] is None


def test_flat_strong_entry_produces_buy() -> None:
    """FLAT state + strong entry conditions -> buy."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-3.0,
        volatility=20.0 / 3.0,
        xbt_inventory=XBT_FLAT,
    )
    assert decision["action"] == "buy"
    assert decision["reason"] == "buy mean-reversion entry"
    assert decision["signal_strength"] == 3.0
    assert abs(decision["signal_strength"]) >= MIN_SIGNAL_STRENGTH


def test_long_strong_exit_produces_sell() -> None:
    """LONG state + strong exit conditions -> sell."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=-0.05,
        momentum=3.0,
        volatility=20.0 / 3.0,
        xbt_inventory=XBT_LONG,
    )
    assert decision["action"] == "sell"
    assert decision["reason"] == "sell mean-reversion exit"
    assert decision["signal_strength"] == -3.0
    assert abs(decision["signal_strength"]) >= MIN_SIGNAL_STRENGTH


def test_flat_weak_entry_becomes_hold() -> None:
    """FLAT state + weak entry -> hold (weak_signal_filtered)."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-2.0,
        volatility=20.0 / 3.0,
        xbt_inventory=XBT_FLAT,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "weak_signal_filtered"
    assert decision["signal_strength"] == 2.0


def test_long_weak_exit_becomes_hold() -> None:
    """LONG state + weak exit -> hold (weak_signal_filtered)."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=-0.05,
        momentum=2.0,
        volatility=20.0 / 3.0,
        xbt_inventory=XBT_LONG,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "weak_signal_filtered"
    assert decision["signal_strength"] == -2.0


def test_no_mean_reversion_signal_strength_none_in_result_and_status() -> None:
    """No long-entry / no exit signal -> signal_strength is None (in result AND status)."""
    decision = {"action": "hold", "reason": "no long-entry signal", "signal_strength": -0.0}
    assert _resolve_signal_strength(decision) is None
    result = {
        "strategy": decision,
        "order": {},
        "raw_signal": "hold",
        "final_action": "hold",
        "decision_reason": "no long-entry signal",
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
    decision = {"action": "buy", "reason": "buy mean-reversion entry", "signal_strength": 3.0}
    assert _resolve_signal_strength(decision) == 3.0


def test_strong_sell_retains_numeric() -> None:
    """Strong sell -> retains numeric value."""
    decision = {"action": "sell", "reason": "sell mean-reversion exit", "signal_strength": -3.0}
    assert _resolve_signal_strength(decision) == -3.0


def test_operator_prints_dash_for_none() -> None:
    """operator_status prints '-' for None."""
    assert _format_signal_strength(None) == "-"


def test_operator_never_prints_negative_zero() -> None:
    """operator_status NEVER prints '-0.0'."""
    assert _format_signal_strength(-0.0) == "0.0"
    assert "-0.0" not in _format_signal_strength(-0.0)


def test_invalid_input_produces_none() -> None:
    """Invalid mid_price or spread -> signal_strength is None."""
    decision = maker_first_mean_reversion(
        mid_price=0.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-2.0,
        volatility=5.0,
        xbt_inventory=XBT_FLAT,
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
        xbt_inventory=XBT_FLAT,
    )
    assert decision["action"] == "hold"
    assert decision["reason"] == "volatility < 0"
    assert decision["signal_strength"] is None
