#!/usr/bin/env python3
"""
Decision snapshot logging for Trade Outcome Intelligence Layer.

Appends structured decision records to data/decision_events.jsonl.
Additive; does not replace trade_events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe(v: Any, default: Any = None) -> Any:
    if v is None:
        return default
    return v


def log_decision_snapshot(
    result: dict,
    path: Path,
    *,
    cooldown_active: bool = False,
    kill_switch_active: bool = False,
    order_size_usd_candidate: float | None = None,
    trade_submitted: bool = False,
    decision_features: dict | None = None,
    expectancy_gate_mode: str | None = None,
    expectancy_gate_decision: str | None = None,
    expectancy_gate_reason: str | None = None,
    expectancy_sample_count: int | None = None,
    expectancy_mean_return_15m: float | None = None,
    expectancy_win_rate: float | None = None,
    expectancy_feature_bucket_summary: str | None = None,
    expectancy_counterfactual_blocked: bool | None = None,
) -> None:
    """
    Append one decision snapshot to JSONL. No-op on error (degrade safely).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        market = result.get("market") or {}
        strategy = result.get("strategy") or {}
        order = result.get("order") or {}
        broker = result.get("broker") or {}
        live_account = result.get("live_account")

        signal_direction = "none"
        raw = result.get("raw_signal", strategy.get("action", "hold"))
        if raw == "buy":
            signal_direction = "buy"
        elif raw == "sell":
            signal_direction = "sell"

        decision_action = "hold"
        if result.get("final_action") == "buy":
            decision_action = "buy"
        elif result.get("final_action") == "sell":
            decision_action = "sell"
        elif order.get("skipped_reason"):
            decision_action = "blocked"

        order_size = order_size_usd_candidate
        if order_size is None and result.get("live_order_ready"):
            order_size = result["live_order_ready"].get("size_usd")

        rec: dict = {
            "timestamp_utc": result.get("timestamp_utc", ""),
            "pair": result.get("pair", ""),
            "runtime_mode": result.get("runtime_mode", "paper"),
            "execution_mode": result.get("execution_mode", "taker"),
            "iteration": result.get("iteration", 0),
            "mid_price": _safe(market.get("mid_price")),
            "signal_direction": signal_direction,
            "signal_strength": _safe(result.get("signal_strength")),
            "spread": _safe(market.get("spread")),
            "volatility": _safe(market.get("volatility")),
            "momentum": _safe(market.get("momentum")),
            "book_imbalance": _safe(market.get("book_imbalance")),
            "inventory": _safe(broker.get("position_units")),
            "cash_quote_balance": None,
            "cooldown_active": cooldown_active,
            "kill_switch_active": kill_switch_active,
            "order_size_usd_candidate": order_size,
            "decision_action": decision_action,
            "decision_reason": result.get("decision_reason", ""),
            "trade_submitted": trade_submitted,
        }
        if live_account is not None:
            rec["cash_quote_balance"] = live_account.get("usd")

        if decision_features:
            for k, v in decision_features.items():
                rec[k] = v

        if expectancy_gate_mode is not None:
            rec["expectancy_gate_mode"] = expectancy_gate_mode
        if expectancy_gate_decision is not None:
            rec["expectancy_gate_decision"] = expectancy_gate_decision
        if expectancy_gate_reason is not None:
            rec["expectancy_gate_reason"] = expectancy_gate_reason
        if expectancy_sample_count is not None:
            rec["expectancy_sample_count"] = expectancy_sample_count
        if expectancy_mean_return_15m is not None:
            rec["expectancy_mean_return_15m"] = expectancy_mean_return_15m
        if expectancy_win_rate is not None:
            rec["expectancy_win_rate"] = expectancy_win_rate
        if expectancy_feature_bucket_summary is not None:
            rec["expectancy_feature_bucket_summary"] = expectancy_feature_bucket_summary
        if expectancy_counterfactual_blocked is not None:
            rec["expectancy_counterfactual_blocked"] = expectancy_counterfactual_blocked

        inputs = strategy.get("inputs", {})
        if isinstance(inputs, dict):
            rec["momentum_threshold"] = inputs.get("momentum_threshold")
        mom = rec.get("momentum")
        thresh = rec.get("momentum_threshold")
        rec["zscore"] = (
            mom / thresh if mom is not None and thresh and thresh != 0 else None
        )

        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except (OSError, TypeError, ValueError):
        pass
