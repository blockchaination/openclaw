#!/usr/bin/env python3
"""
Decision-time feature computation for Trade Outcome Intelligence Layer.

Computes higher-quality market-context features from price history,
decision history, and trade events. All functions degrade safely (return null on error).
"""

from __future__ import annotations

import datetime
import json
import statistics
from pathlib import Path
from typing import Any


def _safe_float(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _parse_ts(ts_str: str) -> datetime.datetime | None:
    if not ts_str:
        return None
    s = ts_str.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _realized_vol(prices: list[float]) -> float | None:
    """Sample std dev of returns. Returns None if insufficient data."""
    if len(prices) < 2:
        return None
    returns: list[float] = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            r = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(r)
    if len(returns) < 2:
        return None
    try:
        return statistics.stdev(returns)
    except statistics.StatisticsError:
        return None


def _regime_tag(
    momentum_60s: float | None,
    distance_short_bps: float | None,
    distance_medium_bps: float | None,
) -> str | None:
    """
    Lightweight heuristic regime: trend, mean_revert, or neutral.
    trend: strong momentum, price far from mean
    mean_revert: weak momentum, price near mean
    neutral: otherwise
    Returns None if insufficient data.
    """
    mom = _safe_float(momentum_60s)
    d_short = _safe_float(distance_short_bps)
    d_med = _safe_float(distance_medium_bps)
    if mom is None and d_short is None and d_med is None:
        return None
    mom_val = mom if mom is not None else 0.0
    dist = max(
        abs(d_short) if d_short is not None else 0,
        abs(d_med) if d_med is not None else 0,
    )
    if abs(mom_val) > 0.0005 and dist > 20:
        return "trend"
    if abs(mom_val) < 0.0002 and dist < 15:
        return "mean_revert"
    return "neutral"


def compute_decision_features(
    result: dict,
    timestamp_utc: str,
    pair: str,
    *,
    price_history_path: Path | None = None,
    decision_events_path: Path | None = None,
    trade_events_path: Path | None = None,
) -> dict[str, Any]:
    """
    Compute higher-quality decision features. Returns dict with null for unavailable.
    """
    out: dict[str, Any] = {
        "return_30s": None,
        "return_60s": None,
        "return_180s": None,
        "momentum_30s": None,
        "momentum_60s": None,
        "momentum_180s": None,
        "distance_from_short_ma_bps": None,
        "distance_from_medium_ma_bps": None,
        "realized_vol_1m": None,
        "realized_vol_5m": None,
        "spread_bps": None,
        "price_vs_recent_high_bps": None,
        "price_vs_recent_low_bps": None,
        "signal_persistence_count": None,
        "same_direction_signal_streak": None,
        "recent_trade_direction": None,
        "recent_trade_count_15m": None,
        "inventory_notional_usd": None,
        "regime_tag": None,
    }
    try:
        now = _parse_ts(timestamp_utc)
        if now is None:
            return out
        if now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)

        market = result.get("market") or {}
        broker = result.get("broker") or {}
        mid = _safe_float(market.get("mid_price"))
        spread_val = _safe_float(market.get("spread"))
        inv_units = _safe_float(broker.get("position_units"), 0.0) or 0.0

        if mid is not None and mid > 0:
            out["inventory_notional_usd"] = round(inv_units * mid, 2)
        if spread_val is not None and mid is not None and mid > 0:
            out["spread_bps"] = round(10000 * spread_val / mid, 2)

        if price_history_path and price_history_path.exists() and mid is not None and mid > 0:
            prices_chrono: list[tuple[datetime.datetime, float]] = []
            try:
                with price_history_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("pair") != pair:
                            continue
                        ts = _parse_ts(rec.get("timestamp_utc", ""))
                        p = rec.get("mid_price")
                        if ts is not None and isinstance(p, (int, float)) and p > 0:
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=datetime.timezone.utc)
                            prices_chrono.append((ts, float(p)))
            except OSError:
                pass

            if prices_chrono:
                prices_chrono.sort(key=lambda x: x[0])
                prices = [p for _, p in prices_chrono]

                def _price_at_or_before(target: datetime.datetime) -> float | None:
                    last_p: float | None = None
                    for t, p in prices_chrono:
                        if t <= target:
                            last_p = p
                        else:
                            break
                    return last_p

                for delta_s, key_ret, key_mom in [
                    (30, "return_30s", "momentum_30s"),
                    (60, "return_60s", "momentum_60s"),
                    (180, "return_180s", "momentum_180s"),
                ]:
                    past_ts = now - datetime.timedelta(seconds=delta_s)
                    past_price = _price_at_or_before(past_ts)
                    if past_price is not None and past_price > 0:
                        ret = (mid - past_price) / past_price
                        out[key_ret] = round(ret, 6)
                        out[key_mom] = round(mid - past_price, 4)

                window_1m = [(t, p) for t, p in prices_chrono if (now - t).total_seconds() <= 60]
                window_5m = [(t, p) for t, p in prices_chrono if (now - t).total_seconds() <= 300]
                if len(window_1m) >= 2:
                    p1 = [p for _, p in window_1m]
                    vol = _realized_vol(p1)
                    if vol is not None:
                        out["realized_vol_1m"] = round(vol, 6)
                if len(window_5m) >= 2:
                    p5 = [p for _, p in window_5m]
                    vol = _realized_vol(p5)
                    if vol is not None:
                        out["realized_vol_5m"] = round(vol, 6)

                if len(prices) >= 2:
                    high = max(prices[-min(20, len(prices)):])
                    low = min(prices[-min(20, len(prices)):])
                    if high > 0:
                        out["price_vs_recent_high_bps"] = round(
                            10000 * (mid - high) / high, 2
                        )
                    if low > 0:
                        out["price_vs_recent_low_bps"] = round(
                            10000 * (mid - low) / low, 2
                        )

                short_ma_n = min(5, len(prices))
                med_ma_n = min(20, len(prices))
                if short_ma_n >= 1:
                    short_ma = sum(prices[-short_ma_n:]) / short_ma_n
                    if short_ma > 0:
                        out["distance_from_short_ma_bps"] = round(
                            10000 * (mid - short_ma) / short_ma, 2
                        )
                if med_ma_n >= 1:
                    med_ma = sum(prices[-med_ma_n:]) / med_ma_n
                    if med_ma > 0:
                        out["distance_from_medium_ma_bps"] = round(
                            10000 * (mid - med_ma) / med_ma, 2
                        )

                out["regime_tag"] = _regime_tag(
                    out.get("momentum_60s"),
                    out.get("distance_from_short_ma_bps"),
                    out.get("distance_from_medium_ma_bps"),
                )

        if decision_events_path and decision_events_path.exists():
            try:
                decisions: list[dict] = []
                with decision_events_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("pair") != pair:
                            continue
                        decisions.append(rec)
                if decisions:
                    sig_dir = result.get("raw_signal") or result.get("strategy", {}).get("action", "hold")
                    if sig_dir not in ("buy", "sell"):
                        sig_dir = None
                    if sig_dir:
                        same_count = 0
                        streak = 0
                        for d in reversed(decisions[-20:]):
                            sd = d.get("signal_direction") or d.get("decision_action")
                            if sd == sig_dir:
                                same_count += 1
                                streak += 1
                            else:
                                break
                        out["signal_persistence_count"] = same_count
                        out["same_direction_signal_streak"] = streak
            except OSError:
                pass

        if trade_events_path and trade_events_path.exists():
            try:
                cutoff = now - datetime.timedelta(minutes=15)
                trade_sides: list[str] = []
                with trade_events_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if ev.get("pair") != pair:
                            continue
                        etype = ev.get("event_type", "")
                        if etype not in ("live_order_submitted", "forced_live_test_buy_submitted"):
                            continue
                        ts = _parse_ts(ev.get("timestamp", ""))
                        if ts is None:
                            continue
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=datetime.timezone.utc)
                        if ts >= cutoff:
                            side = ev.get("side", "")
                            if side in ("buy", "sell"):
                                trade_sides.append(side)
                out["recent_trade_count_15m"] = len(trade_sides)
                if trade_sides:
                    out["recent_trade_direction"] = trade_sides[-1]
            except OSError:
                pass

    except (OSError, TypeError, ValueError, KeyError):
        pass
    return out
