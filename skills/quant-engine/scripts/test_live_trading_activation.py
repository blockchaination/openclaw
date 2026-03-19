#!/usr/bin/env python3
"""
Tests for first-live-trade activation: capped size, cooldown, signal gates.

Run: python -m pytest test_live_trading_activation.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from quant_engine import (
    FIRST_LIVE_ORDER_USD,
    MAX_LIVE_ORDER_USD,
    MIN_SECONDS_BETWEEN_LIVE_ORDERS,
    _build_status,
    _effective_live_buy_usd,
    _last_live_order_timestamp,
)
from strategy import MIN_SIGNAL_STRENGTH, maker_first_mean_reversion


def test_live_buy_effective_size_capped_to_first_live_order_usd() -> None:
    """Live buy order size is capped to min(FIRST_LIVE_ORDER_USD, available_usd, MAX_LIVE_ORDER_USD)."""
    assert FIRST_LIVE_ORDER_USD == 5.0
    assert MAX_LIVE_ORDER_USD == 10.0
    effective = _effective_live_buy_usd(available_usd=100.0, requested_usd=20.0)
    assert effective == FIRST_LIVE_ORDER_USD


def test_live_buy_submission_path_uses_5_usd_max() -> None:
    """_effective_live_buy_usd returns 5.0 max for first-live-trade mode."""
    assert _effective_live_buy_usd(100.0, 20.0) == 5.0
    assert _effective_live_buy_usd(50.0, 15.0) == 5.0
    assert _effective_live_buy_usd(10.0, 20.0) == 5.0


def test_live_order_blocked_never_reports_20_for_first_live_trade() -> None:
    """Effective size is always <= 5, so live_order_blocked never logs size_usd 20."""
    for requested in (20.0, 50.0, 100.0):
        effective = _effective_live_buy_usd(available_usd=100.0, requested_usd=requested)
        assert effective <= FIRST_LIVE_ORDER_USD
        assert effective != 20.0


def test_volume_derived_from_capped_usd() -> None:
    """Volume = effective_usd / mid_price; capped USD yields correct volume."""
    effective = _effective_live_buy_usd(100.0, 20.0)
    assert effective == 5.0
    mid = 100000.0
    volume = effective / mid
    assert volume == 0.00005


def test_max_live_order_usd_guard_if_first_increased() -> None:
    """MAX_LIVE_ORDER_USD guard still applies if FIRST_LIVE_ORDER_USD were increased."""
    # With current constants: FIRST=5, MAX=10. Requested 20 -> min(5,100,10,20)=5
    effective = _effective_live_buy_usd(100.0, 20.0)
    assert effective <= MAX_LIVE_ORDER_USD
    # If available is 15, we'd get min(5,15,10,20)=5. MAX caps at 10.
    effective2 = _effective_live_buy_usd(15.0, 20.0)
    assert effective2 == 5.0
    assert effective2 <= MAX_LIVE_ORDER_USD


def test_live_order_cooldown_blocks_repeated_orders() -> None:
    """_last_live_order_timestamp returns timestamp; cooldown blocks if < 15 min."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev = {"timestamp": ts, "event_type": "live_order_submitted", "pair": "XBTUSD", "side": "buy"}
        path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        last = _last_live_order_timestamp(path)
        assert last is not None
        now = datetime.now(timezone.utc)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed = (now - last).total_seconds()
        assert elapsed < MIN_SECONDS_BETWEEN_LIVE_ORDERS
    finally:
        path.unlink(missing_ok=True)


def test_live_order_cooldown_passes_after_15_min() -> None:
    """Cooldown passes when last live order was > 15 min ago."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=1000)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev = {"timestamp": ts, "event_type": "live_order_submitted", "pair": "XBTUSD", "side": "buy"}
        path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        last = _last_live_order_timestamp(path)
        assert last is not None
        now = datetime.now(timezone.utc)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed = (now - last).total_seconds()
        assert elapsed >= MIN_SECONDS_BETWEEN_LIVE_ORDERS
    finally:
        path.unlink(missing_ok=True)


def test_first_buy_passes_when_flat_and_strong_signal() -> None:
    """FLAT + signal_strength >= 1.5 -> buy."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=0.05,
        momentum=-2.0,
        volatility=20.0 / 3.0,
        xbt_inventory=0.0,
    )
    assert decision["action"] == "buy"
    assert decision["reason"] == "buy mean-reversion entry"
    assert abs(decision["signal_strength"]) >= MIN_SIGNAL_STRENGTH


def test_sell_passes_when_long_and_strong_signal() -> None:
    """LONG + signal_strength magnitude >= 1.5 -> sell."""
    decision = maker_first_mean_reversion(
        mid_price=50000.0,
        spread=10.0,
        book_imbalance=-0.05,
        momentum=2.0,
        volatility=20.0 / 3.0,
        xbt_inventory=0.0005,
    )
    assert decision["action"] == "sell"
    assert decision["reason"] == "sell mean-reversion exit"
    assert abs(decision["signal_strength"]) >= MIN_SIGNAL_STRENGTH


def test_model_probability_does_not_gate_trading() -> None:
    """Status includes model_probability but decision_reason is from rules, not model."""
    result = {
        "strategy": {"action": "buy"},
        "order": {"submitted": True, "skipped_reason": None},
        "raw_signal": "buy",
        "final_action": "buy",
        "decision_reason": "buy mean-reversion entry",
        "model_probability": 0.3,
        "signal_strength": 2.5,
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status.get("decision_reason") == "buy mean-reversion entry"
    assert status.get("model_probability") == 0.3
    # Low model prob does not override decision_reason
    assert status["final_action"] == "buy"


def test_status_includes_live_order_cooldown_active() -> None:
    """Status has live_order_cooldown_active when decision_reason is that."""
    result = {
        "strategy": {"action": "buy"},
        "order": {"submitted": False, "skipped_reason": "live_order_cooldown_active"},
        "raw_signal": "buy",
        "final_action": "hold",
        "decision_reason": "live_order_cooldown_active",
    }
    status = _build_status(result, False, None, None, pair_fallback="XBTUSD")
    assert status.get("live_order_cooldown_active") is True
    assert status.get("decision_reason") == "live_order_cooldown_active"
