#!/usr/bin/env python3
"""
Price history for outcome labeling.

Appends (timestamp, pair, mid_price) each cycle.
Used by outcome_labeler to compute forward returns.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path


def append_price(path: Path, timestamp_utc: str, pair: str, mid_price: float) -> None:
    """Append one price observation. No-op on error."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"timestamp_utc": timestamp_utc, "pair": pair, "mid_price": mid_price}
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except (OSError, TypeError, ValueError):
        pass


def _parse_ts(ts_str: str) -> datetime.datetime | None:
    if not ts_str:
        return None
    s = ts_str.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def get_price_at_or_before(
    path: Path, pair: str, target_ts: datetime.datetime
) -> float | None:
    """
    Find last price for pair at or before target_ts. Returns mid_price or None.
    Used for backward-looking returns (e.g. return_30s).
    """
    if not path.exists():
        return None
    last_price: float | None = None
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
                if rec.get("pair") != pair:
                    continue
                ts = _parse_ts(rec.get("timestamp_utc", ""))
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                if target_ts.tzinfo is None:
                    target_ts = target_ts.replace(tzinfo=datetime.timezone.utc)
                if ts <= target_ts:
                    price = rec.get("mid_price")
                    if isinstance(price, (int, float)) and price > 0:
                        last_price = float(price)
                else:
                    break
    except OSError:
        pass
    return last_price


def get_price_at_or_after(
    path: Path, pair: str, target_ts: datetime.datetime
) -> float | None:
    """
    Find first price for pair at or after target_ts. Returns mid_price or None.
    """
    if not path.exists():
        return None
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
                if rec.get("pair") != pair:
                    continue
                ts = _parse_ts(rec.get("timestamp_utc", ""))
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                if target_ts.tzinfo is None:
                    target_ts = target_ts.replace(tzinfo=datetime.timezone.utc)
                if ts >= target_ts:
                    price = rec.get("mid_price")
                    if isinstance(price, (int, float)) and price > 0:
                        return float(price)
                    return None
    except OSError:
        pass
    return None
