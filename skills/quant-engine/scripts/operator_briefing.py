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
    n_sell_suppressed = sum(
        1 for e in events if e.get("event_type") == "sell_suppressed_low_inventory"
    )
    n_buy_suppressed = sum(
        1 for e in events if e.get("event_type") == "buy_suppressed_low_usd"
    )
    n_engine_error = sum(1 for e in events if e.get("event_type") == "engine_error")

    # Last buy/sell signal timestamps
    last_buy_ts = None
    last_sell_ts = None
    for e in reversed(events):
        if e.get("event_type") != "signal_generated":
            continue
        ts = _parse_ts(e.get("timestamp", ""))
        if ts is None:
            continue
        if e.get("side") == "buy" and last_buy_ts is None:
            last_buy_ts = ts
        elif e.get("side") == "sell" and last_sell_ts is None:
            last_sell_ts = ts
        if last_buy_ts is not None and last_sell_ts is not None:
            break

    # Blocked-reason summary (from live_mode_blocked, live_order_blocked, forced_live_test_buy_blocked, sell_suppressed_low_inventory, buy_suppressed_low_usd)
    blocked_events = [
        e for e in events
        if e.get("event_type")
        in (
            "live_mode_blocked",
            "live_order_blocked",
            "forced_live_test_buy_blocked",
            "sell_suppressed_low_inventory",
            "buy_suppressed_low_usd",
        )
    ]
    reason_counts: dict[str, int] = {}
    for e in blocked_events:
        r = e.get("reason") or "unknown"
        reason_counts[r] = reason_counts.get(r, 0) + 1
    blocked_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])  # most common first

    # Signal bias
    if n_signals == 0:
        signal_bias = "no signals"
    elif n_buy > 0 and n_sell == 0:
        signal_bias = "strongly buy-biased"
    elif n_sell > 0 and n_buy == 0:
        signal_bias = "strongly sell-biased"
    elif n_buy > 0 and n_sell > 0:
        ratio = max(n_buy, n_sell) / min(n_buy, n_sell)
        if ratio >= 2:
            signal_bias = "sell-biased" if n_sell > n_buy else "buy-biased"
        else:
            signal_bias = "balanced"
    else:
        signal_bias = "balanced"

    # Position/actionability hint (from live_account)
    live_account = status.get("live_account") if status else None
    if live_account is not None:
        xbt = live_account.get("xbt", 0) or 0
        has_xbt = xbt > 0.0001  # meaningful threshold
        actionability_hint = (
            "Bot currently has XBT inventory available for sells"
            if has_xbt
            else "Bot currently has no meaningful XBT inventory for sells"
        )
    else:
        actionability_hint = "Live balances unknown (no snapshot)"

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
    if last_buy_ts is not None:
        lines.append(f"  last buy signal:          {last_buy_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if last_sell_ts is not None:
        lines.append(f"  last sell signal:         {last_sell_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"signal bias:               {signal_bias}")
    lines.append(f"live_mode_blocked:         {n_live_mode_blocked}")
    lines.append(f"sell_suppressed_low_inventory: {n_sell_suppressed}")
    lines.append(f"buy_suppressed_low_usd:    {n_buy_suppressed}")
    lines.append(f"live_order_submitted:      {n_live_order_submitted}")
    lines.append(f"live_order_submission_failed: {n_live_order_failed}")
    lines.append(f"forced_live_test_buy_submitted: {n_forced_buy}")
    lines.append(f"engine_error:               {n_engine_error}")
    if last_started is not None:
        lines.append(f"last engine_started:        {last_started.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if last_stopped is not None:
        lines.append(f"last engine_stopped:        {last_stopped.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")

    # Blocked-reason summary
    if blocked_reasons:
        lines.append("--- Blocked reasons (in window) ---")
        for reason, count in blocked_reasons:
            lines.append(f"  {reason}: {count}")
        lines.append("")

    # Position/actionability hint
    lines.append("--- Actionability ---")
    lines.append(actionability_hint)
    lines.append("")

    # Operator summary (sharper)
    lines.append("--- Operator summary ---")
    summary_parts: list[str] = []
    summary_parts.append(f"Pair {pair} in {runtime}/{execution} mode.")
    if kill_active:
        summary_parts.append("Kill switch ACTIVE; no orders will execute.")
    else:
        summary_parts.append("Kill switch inactive.")
    summary_parts.append(f"Signal mix: {signal_bias} ({n_signals} signals: {n_buy} buy, {n_sell} sell).")
    if blocked_reasons:
        top = blocked_reasons[0]
        summary_parts.append(f"Bot was blocked: {top[1]}x {top[0]}.")
    else:
        summary_parts.append("No blocks in window.")
    summary_parts.append(actionability_hint + ".")
    if n_live_order_submitted > 0:
        summary_parts.append(f"{n_live_order_submitted} live order(s) submitted.")
    elif n_live_order_failed > 0:
        summary_parts.append(f"{n_live_order_failed} live order submission(s) failed.")
    if n_forced_buy > 0:
        summary_parts.append(f"{n_forced_buy} forced live test buy(s).")
    if n_engine_error > 0:
        summary_parts.append(f"WARNING: {n_engine_error} engine_error(s).")
    lines.append(" ".join(summary_parts))
    lines.append("")
    lines.append(f"files: {status_path} | {events_path}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
