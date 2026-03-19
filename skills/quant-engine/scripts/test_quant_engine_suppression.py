#!/usr/bin/env python3
"""
Minimal tests for buy/sell suppression and cooldown in quant_engine.

Run: python -m pytest skills/quant-engine/scripts/test_quant_engine_suppression.py -v
Or from scripts dir: python -m pytest test_quant_engine_suppression.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add scripts dir to path so we can import quant_engine
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from quant_engine import (
    FIRST_LIVE_ORDER_USD,
    MIN_SECONDS_BETWEEN_SAME_SIDE_ACTIONS,
    MIN_XBT_TO_SELL,
    _last_same_side_action_timestamp,
)


def _action_eligible_and_skip(
    runtime_mode: str,
    live_account: dict | None,
    action: str,
    broker_position: float,
) -> tuple[bool, str | None]:
    """Replicate buy/sell eligibility logic from quant_engine._run_one_cycle for testing."""
    if action == "buy":
        if runtime_mode == "live" and live_account is not None:
            usd = live_account.get("usd", 0) or 0
            buy_eligible = usd >= FIRST_LIVE_ORDER_USD
            if not buy_eligible:
                return False, "buy_suppressed_low_usd"
            return True, None
        return True, None  # paper mode: no buy suppression
    if action == "sell":
        if runtime_mode == "live" and live_account is not None:
            xbt = live_account.get("xbt", 0) or 0
            sell_eligible = xbt >= MIN_XBT_TO_SELL
            if not sell_eligible:
                return False, "sell_suppressed_low_inventory"
            return True, None
        sell_eligible = broker_position > 0
        if not sell_eligible:
            return False, "no_inventory_to_sell"
        return True, None
    return True, None


# --- Sell suppression tests ---


def test_live_sell_low_xbt_suppressed() -> None:
    """live mode + live_account + sell + low XBT => hold with sell_suppressed_low_inventory."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="live",
        live_account={"usd": 100.0, "xbt": 0.0001},
        action="sell",
        broker_position=0.0,
    )
    assert not eligible
    assert skip == "sell_suppressed_low_inventory"


def test_live_sell_sufficient_xbt_unchanged() -> None:
    """live mode + live_account + sell + sufficient XBT => sell unchanged."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="live",
        live_account={"usd": 100.0, "xbt": 0.0005},
        action="sell",
        broker_position=0.0,
    )
    assert eligible
    assert skip is None


def test_paper_sell_mode_unchanged() -> None:
    """paper / non-live mode => sell uses broker position."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="paper",
        live_account=None,
        action="sell",
        broker_position=0.0,
    )
    assert not eligible
    assert skip == "no_inventory_to_sell"
    eligible2, skip2 = _action_eligible_and_skip(
        runtime_mode="paper",
        live_account=None,
        action="sell",
        broker_position=1.0,
    )
    assert eligible2
    assert skip2 is None


def test_min_xbt_threshold_boundary() -> None:
    """XBT exactly at MIN_XBT_TO_SELL is eligible."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="live",
        live_account={"usd": 0, "xbt": MIN_XBT_TO_SELL},
        action="sell",
        broker_position=0.0,
    )
    assert eligible
    assert skip is None


# --- Buy suppression tests ---


def test_live_buy_low_usd_suppressed() -> None:
    """live mode + live_account + buy + low USD => hold with buy_suppressed_low_usd."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="live",
        live_account={"usd": 3.0, "xbt": 0.001},
        action="buy",
        broker_position=0.0,
    )
    assert not eligible
    assert skip == "buy_suppressed_low_usd"


def test_live_buy_sufficient_usd_unchanged() -> None:
    """live mode + live_account + buy + sufficient USD => buy unchanged."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="live",
        live_account={"usd": 50.0, "xbt": 0.0},
        action="buy",
        broker_position=0.0,
    )
    assert eligible
    assert skip is None


def test_paper_buy_mode_unchanged() -> None:
    """paper / non-live mode => buy unchanged (no buy suppression)."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="paper",
        live_account=None,
        action="buy",
        broker_position=0.0,
    )
    assert eligible
    assert skip is None


def test_min_usd_threshold_boundary() -> None:
    """USD exactly at FIRST_LIVE_ORDER_USD is eligible."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="live",
        live_account={"usd": FIRST_LIVE_ORDER_USD, "xbt": 0.0},
        action="buy",
        broker_position=0.0,
    )
    assert eligible
    assert skip is None


# --- Cooldown tests ---


def _cooldown_skip_reason(
    trade_events_path: Path | None,
    action: str,
    runtime_mode: str,
) -> str | None:
    """Replicate cooldown check from quant_engine._run_one_cycle for testing."""
    if trade_events_path is None or runtime_mode != "live" or action not in ("buy", "sell"):
        return None
    now_utc = datetime.now(timezone.utc)
    last_ts = _last_same_side_action_timestamp(trade_events_path, action)
    if last_ts is None:
        return None
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    elapsed = (now_utc - last_ts).total_seconds()
    if elapsed < MIN_SECONDS_BETWEEN_SAME_SIDE_ACTIONS:
        return "buy_cooldown_active" if action == "buy" else "sell_cooldown_active"
    return None


def test_live_buy_repeated_within_cooldown() -> None:
    """live mode + repeated buy within cooldown => hold with buy_cooldown_active."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev = {"timestamp": ts, "event_type": "buy_suppressed_low_usd", "pair": "XBTUSD", "side": "buy"}
        path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        skip = _cooldown_skip_reason(path, "buy", "live")
        assert skip == "buy_cooldown_active"
    finally:
        path.unlink(missing_ok=True)


def test_live_sell_repeated_within_cooldown() -> None:
    """live mode + repeated sell within cooldown => hold with sell_cooldown_active."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev = {"timestamp": ts, "event_type": "sell_suppressed_low_inventory", "pair": "XBTUSD", "side": "sell"}
        path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        skip = _cooldown_skip_reason(path, "sell", "live")
        assert skip == "sell_cooldown_active"
    finally:
        path.unlink(missing_ok=True)


def test_action_outside_cooldown_window() -> None:
    """Action outside cooldown window => unchanged."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=400)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev = {"timestamp": ts, "event_type": "buy_suppressed_low_usd", "pair": "XBTUSD", "side": "buy"}
        path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        skip = _cooldown_skip_reason(path, "buy", "live")
        assert skip is None
    finally:
        path.unlink(missing_ok=True)


def test_paper_mode_cooldown_unchanged() -> None:
    """paper / non-live mode => cooldown not applied."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev = {"timestamp": ts, "event_type": "buy_suppressed_low_usd", "pair": "XBTUSD", "side": "buy"}
        path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        skip = _cooldown_skip_reason(path, "buy", "paper")
        assert skip is None
    finally:
        path.unlink(missing_ok=True)


def test_hold_action_cooldown_unchanged() -> None:
    """hold action => cooldown not applied."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev = {"timestamp": ts, "event_type": "buy_suppressed_low_usd", "pair": "XBTUSD", "side": "buy"}
        path.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        skip = _cooldown_skip_reason(path, "hold", "live")
        assert skip is None
    finally:
        path.unlink(missing_ok=True)


def test_last_same_side_timestamp_returns_most_recent() -> None:
    """_last_same_side_action_timestamp returns most recent matching event."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_ts = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        path.write_text(
            json.dumps({"timestamp": old_ts, "event_type": "buy_suppressed_low_usd", "side": "buy"}) + "\n"
            + json.dumps({"timestamp": new_ts, "event_type": "buy_suppressed_low_usd", "side": "buy"}) + "\n",
            encoding="utf-8",
        )
        last = _last_same_side_action_timestamp(path, "buy")
        assert last is not None
        last_utc = last if last.tzinfo else last.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last_utc).total_seconds()
        assert elapsed < 120  # newer event was 30s ago
    finally:
        path.unlink(missing_ok=True)
