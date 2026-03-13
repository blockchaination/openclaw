#!/usr/bin/env python3
"""
Operator briefing generator for OpenClaw Kraken Quant.

Reads logs/status.json and logs/trade_events.jsonl, generates a compact
human-readable daily/weekly briefing. Read-only; does not affect trading logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _repo_root() -> Path:
    """Return repo root (parent of skills/)."""
    return Path(__file__).resolve().parents[3]


def _default_status_path() -> Path:
    return _repo_root() / "logs" / "status.json"


def _default_trade_events_path() -> Path:
    return _repo_root() / "logs" / "trade_events.jsonl"


def _parse_ts(s: str) -> datetime | None:
    """Parse ISO timestamp. Return None if invalid."""
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _read_json(path: Path) -> dict | None:
    """Read JSON file. Return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl_in_window(path: Path, cutoff: datetime) -> list[dict]:
    """Read JSONL, return events with timestamp >= cutoff (oldest first)."""
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(ev.get("timestamp", ""))
                if ts is not None and ts >= cutoff:
                    out.append(ev)
    except OSError:
        pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate operator briefing from quant engine logs.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Briefing window: last N hours (default if --days omitted: 24)",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Briefing window: last N days (e.g. 7 for weekly)",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help=f"Path to status JSON (default: {_default_status_path()})",
    )
    parser.add_argument(
        "--trade-events-file",
        type=Path,
        default=None,
        help=f"Path to trade events JSONL (default: {_default_trade_events_path()})",
    )
    args = parser.parse_args()

    status_path = args.status_file or _default_status_path()
    events_path = args.trade_events_file or _default_trade_events_path()

    # Window: default 24h if both omitted; --days overrides when --hours not set
    now = datetime.now(timezone.utc)
    if args.hours is not None:
        delta_hours = args.hours
    elif args.days is not None:
        delta_hours = args.days * 24
    else:
        delta_hours = 24
    cutoff = now - timedelta(hours=delta_hours)

    status = _read_json(status_path)
    events = _read_jsonl_in_window(events_path, cutoff)

    # Counts
    n_signals = sum(1 for e in events if e.get("event_type") == "signal_generated")
    n_buy = sum(1 for e in events if e.get("event_type") == "signal_generated" and e.get("side") == "buy")
    n_sell = sum(1 for e in events if e.get("event_type") == "signal_generated" and e.get("side") == "sell")
    n_hold = n_signals - n_buy - n_sell  # hold = signal that wasn't buy/sell (if any)
    if n_hold < 0:
        n_hold = 0
    n_live_mode_blocked = sum(1 for e in events if e.get("event_type") == "live_mode_blocked")
    n_live_order_submitted = sum(1 for e in events if e.get("event_type") == "live_order_submitted")
    n_live_order_failed = sum(1 for e in events if e.get("event_type") == "live_order_submission_failed")
    n_forced_buy = sum(1 for e in events if e.get("event_type") == "forced_live_test_buy_submitted")
    n_engine_error = sum(1 for e in events if e.get("event_type") == "engine_error")

    # Most recent engine_started / engine_stopped (from events in window)
    last_started = None
    last_stopped = None
    for e in reversed(events):
        ts = _parse_ts(e.get("timestamp", ""))
        if ts is None:
            continue
        if e.get("event_type") == "engine_started" and last_started is None:
            last_started = ts
        elif e.get("event_type") == "engine_stopped" and last_stopped is None:
            last_stopped = ts
        if last_started is not None and last_stopped is not None:
            break

    # Window label
    if args.hours is not None:
        window_label = f"last {int(args.hours)}h"
    elif args.days is not None:
        window_label = f"last {int(args.days)}d"
    else:
        window_label = "last 24h"

    lines: list[str] = []
    lines.append("=== OpenClaw Kraken Quant Operator Briefing ===")
    lines.append("")
    lines.append(f"Briefing window: {window_label}")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")

    # Current state from status
    pair = status.get("pair", "-") if status else "-"
    runtime = status.get("runtime_mode", "-") if status else "-"
    execution = status.get("execution_mode", "-") if status else "-"
    kill_active = status.get("kill_switch_active", False) if status else False

    lines.append("--- Current state ---")
    lines.append(f"pair:              {pair}")
    lines.append(f"runtime mode:      {runtime}")
    lines.append(f"execution mode:    {execution}")
    lines.append(f"kill switch:       {'ACTIVE' if kill_active else 'inactive'}")
    if status and status.get("live_account") is not None:
        la = status["live_account"]
        lines.append(f"live balances:     USD={la.get('usd', 0):.2f} XBT={la.get('xbt', 0):.6f}")
    lines.append("")

    # Event counts in window
    lines.append("--- Event counts (in window) ---")
    lines.append(f"signal_generated:           {n_signals}")
    lines.append(f"  buy signals:              {n_buy}")
    lines.append(f"  sell signals:             {n_sell}")
    if n_hold > 0:
        lines.append(f"  holds:                    {n_hold}")
    lines.append(f"live_mode_blocked:         {n_live_mode_blocked}")
    lines.append(f"live_order_submitted:      {n_live_order_submitted}")
    lines.append(f"live_order_submission_failed: {n_live_order_failed}")
    lines.append(f"forced_live_test_buy_submitted: {n_forced_buy}")
    lines.append(f"engine_error:               {n_engine_error}")
    if last_started is not None:
        lines.append(f"last engine_started:        {last_started.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if last_stopped is not None:
        lines.append(f"last engine_stopped:        {last_stopped.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")

    # Operator summary
    lines.append("--- Operator summary ---")
    summary_parts: list[str] = []
    summary_parts.append(f"Pair {pair} in {runtime}/{execution} mode.")
    if kill_active:
        summary_parts.append("Kill switch is ACTIVE; no orders will execute.")
    else:
        summary_parts.append("Kill switch inactive.")
    summary_parts.append(f"In the window: {n_signals} signals ({n_buy} buy, {n_sell} sell).")
    if n_live_order_submitted > 0:
        summary_parts.append(f"{n_live_order_submitted} live order(s) submitted.")
    if n_live_order_failed > 0:
        summary_parts.append(f"{n_live_order_failed} live order submission(s) failed.")
    if n_live_mode_blocked > 0:
        summary_parts.append(f"{n_live_mode_blocked} live_mode_blocked (e.g. no XBT balance).")
    if n_forced_buy > 0:
        summary_parts.append(f"{n_forced_buy} forced live test buy(s) submitted.")
    if n_engine_error > 0:
        summary_parts.append(f"WARNING: {n_engine_error} engine_error(s) in window.")
    if not summary_parts:
        summary_parts.append("No notable activity in window.")
    lines.append(" ".join(summary_parts))
    lines.append("")
    lines.append(f"files: {status_path} | {events_path}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
