#!/usr/bin/env python3
"""
Analyze JSONL decision log produced by quant_engine.py.

Computes basic trading metrics: signal counts, order stats, PnL, max drawdown.
"""

import argparse
import json
from pathlib import Path


def load_runs(path: Path) -> list[dict]:
    """
    Read JSONL file line by line. Return list of parsed run dicts.
    Skip empty lines.
    """
    runs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def compute_metrics(runs: list[dict]) -> dict:
    """
    Compute metrics from run dicts.

    Returns dict with:
    - total_runs
    - buy_count, sell_count, hold_count
    - orders_submitted, orders_filled
    - realized_pnl_usd, unrealized_pnl_usd, total_pnl_usd (last seen)
    - max_drawdown
    """
    buy_count = 0
    sell_count = 0
    hold_count = 0
    orders_submitted = 0
    orders_filled = 0
    realized_pnl_usd = 0.0
    unrealized_pnl_usd = 0.0
    total_pnl_usd = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0

    for run in runs:
        strategy = run.get("strategy") or {}
        action = strategy.get("action", "hold")
        if action == "buy":
            buy_count += 1
        elif action == "sell":
            sell_count += 1
        else:
            hold_count += 1

        order = run.get("order") or {}
        if order.get("submitted"):
            orders_submitted += 1
        fills = order.get("fills") or []
        orders_filled += len(fills)

        broker = run.get("broker") or {}
        realized_pnl_usd = broker.get("realized_pnl_usd", realized_pnl_usd)
        unrealized_pnl_usd = broker.get("unrealized_pnl_usd", unrealized_pnl_usd)
        total_pnl_usd = broker.get("total_pnl_usd", total_pnl_usd)

        peak_pnl = max(peak_pnl, total_pnl_usd)
        drawdown = total_pnl_usd - peak_pnl
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    total_runs = len(runs)
    fill_rate = (
        orders_filled / orders_submitted if orders_submitted > 0 else 0.0
    )
    signal_rate = (
        (buy_count + sell_count) / total_runs if total_runs > 0 else 0.0
    )

    return {
        "total_runs": total_runs,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "orders_submitted": orders_submitted,
        "orders_filled": orders_filled,
        "fill_rate": fill_rate,
        "signal_rate": signal_rate,
        "realized_pnl_usd": realized_pnl_usd,
        "unrealized_pnl_usd": unrealized_pnl_usd,
        "total_pnl_usd": total_pnl_usd,
        "max_drawdown": max_drawdown,
    }


def print_report(metrics: dict) -> None:
    """Print a clean terminal report."""
    print(f"Runs analyzed: {metrics['total_runs']}")
    print()
    print("Orders")
    print("------")
    print(f"submitted: {metrics['orders_submitted']}")
    print(f"fills: {metrics['orders_filled']}")
    print(f"fill rate: {metrics['fill_rate']:.6f}")
    print()
    print("Signals")
    print("------")
    print(f"buy: {metrics['buy_count']}")
    print(f"sell: {metrics['sell_count']}")
    print(f"hold: {metrics['hold_count']}")
    print(f"signal rate: {metrics['signal_rate']:.6f}")
    print()
    print("PnL")
    print("------")
    print(f"realized pnl: {metrics['realized_pnl_usd']:.6f} USD")
    print(f"unrealized pnl: {metrics['unrealized_pnl_usd']:.6f} USD")
    print(f"total pnl: {metrics['total_pnl_usd']:.6f} USD")
    print(f"max drawdown: {metrics['max_drawdown']:.6f} USD")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze JSONL decision log from quant_engine.py.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/quant_engine_runs.jsonl",
        help="Path to JSONL log file (default: logs/quant_engine_runs.jsonl).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 1

    runs = load_runs(log_path)
    metrics = compute_metrics(runs)
    print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
