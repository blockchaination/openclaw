#!/usr/bin/env python3
"""
Analyze signal_outcomes.jsonl and report signal performance.

Run: python analyze_signal_outcomes.py [--file PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return repo root (parent of skills/)."""
    return Path(__file__).resolve().parents[3]


def _default_path() -> Path:
    return _repo_root() / "logs" / "signal_outcomes.jsonl"


def _load_completed_records(path: Path) -> list[dict]:
    """Load JSONL, return only records with price_after_300s."""
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("price_after_300s") is None:
                    continue
                price_at = rec.get("price_at_signal")
                if price_at is None or float(price_at) <= 0:
                    continue
                records.append(rec)
    except OSError:
        pass
    return records


def _compute_return(rec: dict, horizon_key: str) -> float | None:
    """Compute return for one record at given horizon. Returns None if invalid."""
    price_at = rec.get("price_at_signal")
    future = rec.get(horizon_key)
    if price_at is None or future is None:
        return None
    try:
        p0 = float(price_at)
        p1 = float(future)
    except (TypeError, ValueError):
        return None
    if p0 <= 0:
        return None
    signal = (rec.get("signal") or "").lower()
    if signal == "buy":
        return (p1 - p0) / p0
    if signal == "sell":
        return (p0 - p1) / p0
    return None


def _compute_metrics(records: list[dict]) -> dict:
    """Compute summary metrics from completed records."""
    if not records:
        return {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "average_return_30s": 0.0,
            "average_return_60s": 0.0,
            "average_return_300s": 0.0,
            "win_rate_300s": 0.0,
        }
    buy_count = sum(1 for r in records if (r.get("signal") or "").lower() == "buy")
    sell_count = sum(1 for r in records if (r.get("signal") or "").lower() == "sell")
    returns_30: list[float] = []
    returns_60: list[float] = []
    returns_300: list[float] = []
    for rec in records:
        r30 = _compute_return(rec, "price_after_30s")
        r60 = _compute_return(rec, "price_after_60s")
        r300 = _compute_return(rec, "price_after_300s")
        if r30 is not None:
            returns_30.append(r30)
        if r60 is not None:
            returns_60.append(r60)
        if r300 is not None:
            returns_300.append(r300)
    n = len(returns_300)
    avg_30 = sum(returns_30) / len(returns_30) if returns_30 else 0.0
    avg_60 = sum(returns_60) / len(returns_60) if returns_60 else 0.0
    avg_300 = sum(returns_300) / len(returns_300) if returns_300 else 0.0
    wins = sum(1 for r in returns_300 if r > 0)
    win_rate = (wins / n * 100.0) if n > 0 else 0.0
    return {
        "total_signals": len(records),
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "average_return_30s": avg_30,
        "average_return_60s": avg_60,
        "average_return_300s": avg_300,
        "win_rate_300s": win_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze signal_outcomes.jsonl and report signal performance.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help=f"Path to signal outcomes JSONL (default: {_default_path()})",
    )
    args = parser.parse_args()
    path = args.file or _default_path()

    records = _load_completed_records(path)
    metrics = _compute_metrics(records)

    print("OpenClaw Signal Analysis")
    print("-----------------------")
    print()
    print("Completed evaluated signals:")
    print(f"  total:  {metrics['total_signals']}")
    print(f"  buy:    {metrics['buy_signals']}")
    print(f"  sell:   {metrics['sell_signals']}")
    print()
    print(f"Average return 30s:  {metrics['average_return_30s']:.5f}")
    print(f"Average return 60s:  {metrics['average_return_60s']:.5f}")
    print(f"Average return 300s: {metrics['average_return_300s']:.5f}")
    print()
    print(f"Win rate (300s): {metrics['win_rate_300s']:.1f}%")
    print()
    print(f"file: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
