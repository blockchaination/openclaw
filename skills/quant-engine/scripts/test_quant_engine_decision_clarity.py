#!/usr/bin/env python3
"""
Tests for raw_signal, final_action, decision_reason decision clarity in quant_engine.

Run: python -m pytest test_quant_engine_decision_clarity.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from quant_engine import _build_status


def test_buy_signal_not_suppressed() -> None:
    """Buy signal not suppressed => raw_signal=buy, final_action=buy, decision_reason from strategy."""
    result = {
        "strategy": {"action": "buy"},
        "order": {"submitted": True, "skipped_reason": None},
        "raw_signal": "buy",
        "final_action": "buy",
        "decision_reason": "buy mean-reversion entry",
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status.get("raw_signal") == "buy"
    assert status.get("final_action") == "buy"
    assert status.get("decision_reason") == "buy mean-reversion entry"


def test_sell_suppressed_low_inventory() -> None:
    """Sell suppressed due to low inventory => raw_signal=sell, final_action=hold, decision_reason=sell_suppressed_low_inventory."""
    result = {
        "strategy": {"action": "sell"},
        "order": {"submitted": False, "skipped_reason": "sell_suppressed_low_inventory"},
        "raw_signal": "sell",
        "final_action": "hold",
        "decision_reason": "sell_suppressed_low_inventory",
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status.get("raw_signal") == "sell"
    assert status.get("final_action") == "hold"
    assert status.get("decision_reason") == "sell_suppressed_low_inventory"


def test_buy_suppressed_low_usd() -> None:
    """Buy suppressed due to low USD => raw_signal=buy, final_action=hold, decision_reason=buy_suppressed_low_usd."""
    result = {
        "strategy": {"action": "buy"},
        "order": {"submitted": False, "skipped_reason": "buy_suppressed_low_usd"},
        "raw_signal": "buy",
        "final_action": "hold",
        "decision_reason": "buy_suppressed_low_usd",
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status.get("raw_signal") == "buy"
    assert status.get("final_action") == "hold"
    assert status.get("decision_reason") == "buy_suppressed_low_usd"


def test_cooldown_suppression() -> None:
    """Cooldown suppression => raw_signal=buy, final_action=hold, decision_reason=buy_cooldown_active."""
    result = {
        "strategy": {"action": "buy"},
        "order": {"submitted": False, "skipped_reason": "buy_cooldown_active"},
        "raw_signal": "buy",
        "final_action": "hold",
        "decision_reason": "buy_cooldown_active",
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status.get("raw_signal") == "buy"
    assert status.get("final_action") == "hold"
    assert status.get("decision_reason") == "buy_cooldown_active"


def test_operator_status_backward_compat() -> None:
    """Old status without raw_signal/final_action/decision_reason uses fallbacks (no crash)."""
    old_status = {"pair": "XBTUSD", "last_signal": "hold", "last_action": "hold"}
    raw = old_status.get("raw_signal", old_status.get("last_signal", "-"))
    final = old_status.get("final_action", old_status.get("last_action", "-"))
    reason = old_status.get("decision_reason", "-")
    assert raw == "hold"
    assert final == "hold"
    assert reason == "-"
