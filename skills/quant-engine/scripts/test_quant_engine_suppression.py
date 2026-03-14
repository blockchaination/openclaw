#!/usr/bin/env python3
"""
Minimal tests for buy/sell suppression (MIN_USD_TO_BUY, MIN_XBT_TO_SELL) in quant_engine.

Run: python -m pytest skills/quant-engine/scripts/test_quant_engine_suppression.py -v
Or from scripts dir: python -m pytest test_quant_engine_suppression.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts dir to path so we can import quant_engine
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from quant_engine import MIN_USD_TO_BUY, MIN_XBT_TO_SELL


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
            buy_eligible = usd >= MIN_USD_TO_BUY
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
        live_account={"usd": 5.0, "xbt": 0.001},
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
    """USD exactly at MIN_USD_TO_BUY is eligible."""
    eligible, skip = _action_eligible_and_skip(
        runtime_mode="live",
        live_account={"usd": MIN_USD_TO_BUY, "xbt": 0.0},
        action="buy",
        broker_position=0.0,
    )
    assert eligible
    assert skip is None
