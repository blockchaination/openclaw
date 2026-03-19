#!/usr/bin/env python3
"""
Expectancy gate: uses historical labeled decisions to allow/block trades.

Heuristic model: same direction, bucket by features, require min samples,
mean return_15m, and win rate above thresholds.
Supports shadow mode: evaluate and log without blocking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from expectancy_config import (
    EXPECTANCY_ALLOW_IF_INSUFFICIENT_DATA,
    EXPECTANCY_MIN_MEAN_RETURN_15M,
    EXPECTANCY_MIN_SAMPLES,
    EXPECTANCY_MIN_WIN_RATE,
)


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _momentum_band(momentum: float | None) -> str:
    if momentum is None:
        return "unknown"
    if momentum > 0:
        return "pos"
    if momentum < 0:
        return "neg"
    return "zero"


def _volatility_band(vol: float | None) -> str:
    if vol is None or vol < 0:
        return "unknown"
    if vol < 5:
        return "low"
    if vol < 15:
        return "med"
    return "high"


def _realized_vol_band(vol: float | None) -> str:
    """Band for realized_vol_1m/5m (typically 1e-4 to 1e-2)."""
    if vol is None or vol < 0:
        return "unknown"
    if vol < 0.0005:
        return "low"
    if vol < 0.002:
        return "med"
    return "high"


def _spread_band(spread: float | None) -> str:
    if spread is None or spread < 0:
        return "unknown"
    if spread < 1:
        return "low"
    if spread < 5:
        return "med"
    return "high"


def _spread_bps_band(spread_bps: float | None) -> str:
    if spread_bps is None or spread_bps < 0:
        return "unknown"
    if spread_bps < 2:
        return "low"
    if spread_bps < 10:
        return "med"
    return "high"


def _zscore_band(zscore: float | None) -> str:
    if zscore is None:
        return "unknown"
    z = float(zscore)
    if z < -1.5:
        return "low"
    if z < 0:
        return "med_low"
    if z < 1.5:
        return "med_high"
    return "high"


def _distance_bps_band(dist_bps: float | None) -> str:
    if dist_bps is None:
        return "unknown"
    d = abs(float(dist_bps))
    if d < 5:
        return "low"
    if d < 20:
        return "med"
    return "high"


def _build_feature_bucket_summary(
    direction: str,
    regime_tag: str | None,
    zscore: float | None,
    realized_vol: float | None,
    spread_bps: float | None,
    momentum_60s: float | None,
    distance_short_bps: float | None,
) -> str:
    """Deterministic string summary of feature buckets for logging."""
    parts = [f"dir={direction}"]
    parts.append(f"reg={regime_tag or 'unknown'}")
    parts.append(f"zs={_zscore_band(zscore)}")
    parts.append(f"vol={_realized_vol_band(realized_vol)}")
    parts.append(f"spread_bps={_spread_bps_band(spread_bps)}")
    parts.append(f"mom60={_momentum_band(momentum_60s)}")
    parts.append(f"dist={_distance_bps_band(distance_short_bps)}")
    return "|".join(parts)


def _load_labeled(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("return_15m") is None:
                    continue
                out.append(rec)
    except OSError:
        pass
    return out


def evaluate_expectancy(
    decision: dict,
    labeled_path: Path,
    *,
    min_samples: int = EXPECTANCY_MIN_SAMPLES,
    min_mean_return: float = EXPECTANCY_MIN_MEAN_RETURN_15M,
    min_win_rate: float = EXPECTANCY_MIN_WIN_RATE,
    allow_if_insufficient: bool = EXPECTANCY_ALLOW_IF_INSUFFICIENT_DATA,
) -> tuple[bool, str, dict]:
    """
    Evaluate whether trade should be allowed based on historical expectancy.
    Returns (allowed, reason, stats_dict).
    stats_dict includes: sample_count, mean_return_15m, win_rate, blocked,
    expectancy_gate_decision (allow/block/insufficient_data/error),
    expectancy_gate_reason, expectancy_feature_bucket_summary.
    """
    stats: dict = {
        "sample_count": 0,
        "mean_return_15m": None,
        "win_rate": None,
        "blocked": False,
        "expectancy_gate_decision": "allow",
        "expectancy_gate_reason": "",
        "expectancy_feature_bucket_summary": "",
    }
    direction = decision.get("signal_direction") or decision.get("raw_signal", "")
    market = decision.get("market") or {}
    momentum = decision.get("momentum") if "momentum" in decision else market.get("momentum")
    vol = decision.get("volatility") if "volatility" in decision else market.get("volatility")
    spread_val = decision.get("spread") if "spread" in decision else market.get("spread")
    regime_tag = decision.get("regime_tag")
    zscore = decision.get("zscore")
    realized_vol_1m = decision.get("realized_vol_1m")
    realized_vol_5m = decision.get("realized_vol_5m") or realized_vol_1m
    spread_bps = decision.get("spread_bps")
    momentum_60s = decision.get("momentum_60s")
    distance_short_bps = decision.get("distance_from_short_ma_bps")

    if direction not in ("buy", "sell"):
        stats["expectancy_gate_decision"] = "allow"
        stats["expectancy_gate_reason"] = "hold_no_gate"
        return True, "hold_no_gate", stats

    bucket_summary = _build_feature_bucket_summary(
        direction, regime_tag, zscore, realized_vol_5m,
        spread_bps, momentum_60s, distance_short_bps,
    )
    stats["expectancy_feature_bucket_summary"] = bucket_summary

    mb = _momentum_band(_safe_float(momentum) if momentum is not None else None)
    vb = _volatility_band(_safe_float(vol) if vol is not None else None)
    sb = _spread_band(_safe_float(spread_val) if spread_val is not None else None)
    sb_bps = _spread_bps_band(spread_bps)
    rvb = _realized_vol_band(realized_vol_5m)
    mom60b = _momentum_band(_safe_float(momentum_60s) if momentum_60s is not None else None)
    distb = _distance_bps_band(distance_short_bps)
    zsb = _zscore_band(zscore)
    reg = regime_tag or "unknown"

    labeled = _load_labeled(labeled_path)
    matches: list[dict] = []
    for rec in labeled:
        if rec.get("signal_direction") != direction and rec.get("decision_action") != direction:
            continue
        rec_mb = _momentum_band(rec.get("momentum"))
        rec_vb = _volatility_band(rec.get("volatility"))
        rec_sb = _spread_band(rec.get("spread"))
        rec_sb_bps = _spread_bps_band(rec.get("spread_bps"))
        rec_rvb = _realized_vol_band(rec.get("realized_vol_5m") or rec.get("realized_vol_1m"))
        rec_mom60b = _momentum_band(rec.get("momentum_60s"))
        rec_distb = _distance_bps_band(rec.get("distance_from_short_ma_bps"))
        rec_zsb = _zscore_band(rec.get("zscore"))
        rec_reg = rec.get("regime_tag") or "unknown"

        if mb != "unknown" and rec_mb != mb:
            continue
        if vb != "unknown" and rec_vb != vb:
            continue
        if sb != "unknown" and rec_sb != sb:
            continue
        if sb_bps != "unknown" and rec_sb_bps != "unknown" and sb_bps != rec_sb_bps:
            continue
        if rvb != "unknown" and rec_rvb != "unknown" and rvb != rec_rvb:
            continue
        if mom60b != "unknown" and rec_mom60b != "unknown" and mom60b != rec_mom60b:
            continue
        if distb != "unknown" and rec_distb != "unknown" and distb != rec_distb:
            continue
        if zsb != "unknown" and rec_zsb != "unknown" and zsb != rec_zsb:
            continue
        if reg != "unknown" and rec_reg != "unknown" and reg != rec_reg:
            continue

        r = rec.get("return_15m")
        if r is not None:
            matches.append(rec)

    if not matches:
        stats["expectancy_gate_decision"] = "insufficient_data"
        stats["expectancy_gate_reason"] = "no_matching_labeled_decisions"
        return (
            (True, "insufficient_data_allow", stats)
            if allow_if_insufficient
            else (False, "expectancy_gate_blocked", {
                **stats,
                "blocked": True,
                "expectancy_gate_decision": "block",
                "expectancy_gate_reason": "no_matching_labeled_decisions",
            })
        )

    returns = [_safe_float(m.get("return_15m")) for m in matches]
    n = len(returns)
    mean_ret = sum(returns) / n if n > 0 else 0.0
    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / n if n > 0 else 0.0

    stats["sample_count"] = n
    stats["mean_return_15m"] = round(mean_ret, 6)
    stats["win_rate"] = round(win_rate, 4)

    if n < min_samples:
        stats["expectancy_gate_decision"] = "insufficient_data"
        stats["expectancy_gate_reason"] = f"n={n}<min={min_samples}"
        return (
            (True, "insufficient_samples_allow", stats)
            if allow_if_insufficient
            else (False, "expectancy_gate_blocked", {
                **stats,
                "blocked": True,
                "expectancy_gate_decision": "block",
                "expectancy_gate_reason": f"n={n}<min={min_samples}",
            })
        )
    if mean_ret < min_mean_return:
        stats["expectancy_gate_decision"] = "block"
        stats["expectancy_gate_reason"] = f"mean_ret={mean_ret:.4f}<min={min_mean_return}"
        return False, "expectancy_gate_blocked", {**stats, "blocked": True}
    if win_rate < min_win_rate:
        stats["expectancy_gate_decision"] = "block"
        stats["expectancy_gate_reason"] = f"win_rate={win_rate:.2%}<min={min_win_rate:.2%}"
        return False, "expectancy_gate_blocked", {**stats, "blocked": True}

    stats["expectancy_gate_decision"] = "allow"
    stats["expectancy_gate_reason"] = "expectancy_gate_passed"
    return True, "expectancy_gate_passed", stats
