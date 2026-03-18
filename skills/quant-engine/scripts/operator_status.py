#!/usr/bin/env python3
"""
Operator status/report for OpenClaw Kraken Quant.

Reads logs/status.json and logs/trade_events.jsonl, prints a compact
human-readable summary. Read-only; does not affect trading logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return repo root (parent of skills/)."""
    return Path(__file__).resolve().parents[3]


def _default_status_path() -> Path:
    return _repo_root() / "logs" / "status.json"


def _default_trade_events_path() -> Path:
    return _repo_root() / "logs" / "trade_events.jsonl"


def _read_json(path: Path) -> dict | None:
    """Read JSON file. Return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl_tail(path: Path, n: int) -> list[dict]:
    """Read last n lines from JSONL. Return list of parsed objects."""
    if not path.exists() or n <= 0:
        return []
    lines: list[str] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
                    if len(lines) > n:
                        lines.pop(0)
    except OSError:
        return []
    out: list[dict] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _format_signal_strength(ss: int | float | None) -> str:
    """Format signal_strength for display: None -> '-', normalize -0.0 to 0.0."""
    if ss is None:
        return "-"
    if isinstance(ss, (int, float)) and abs(ss) < 1e-9:
        return "0.0"
    return str(ss)


def _format_event(ev: dict) -> str:
    """Format a single trade event for display."""
    ts = ev.get("timestamp", "")
    etype = ev.get("event_type", "?")
    parts = [f"{ts} {etype}"]
    if etype == "signal_generated":
        parts.append(f"side={ev.get('side','?')} reason={ev.get('reason','')}")
    elif etype == "order_submitted":
        parts.append(f"side={ev.get('side','?')} size_usd={ev.get('size_usd','?')}")
    elif etype == "order_filled":
        parts.append(f"side={ev.get('side','?')} size_usd={ev.get('size_usd','?')} pos_usd={ev.get('position_usd','?')}")
    elif etype == "engine_started":
        iters = ev.get("iterations", ev.get("iteration", "?"))
        parts.append(f"iterations={iters}")
    elif etype == "engine_stopped":
        parts.append(f"iter={ev.get('iteration','?')} reason={ev.get('reason','') or '-'}")
    elif etype == "kill_switch_triggered":
        parts.append(f"reason={ev.get('reason','')}")
    elif etype == "weak_signal_filtered":
        parts.append(f"signal_strength={_format_signal_strength(ev.get('signal_strength'))}")
    elif "reason" in ev:
        parts.append(f"reason={ev.get('reason','')}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print operator status summary from quant engine logs.",
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
    parser.add_argument(
        "--tail",
        type=int,
        default=10,
        help="Number of recent trade events to show (default: 10)",
    )
    args = parser.parse_args()

    status_path = args.status_file or _default_status_path()
    events_path = args.trade_events_file or _default_trade_events_path()

    status = _read_json(status_path)
    events = _read_jsonl_tail(events_path, args.tail)

    lines: list[str] = []
    lines.append("--- OpenClaw Kraken Quant Operator Status ---")
    lines.append("")

    if status is None:
        lines.append("Status: (no status file or invalid JSON)")
        lines.append(f"  status file: {status_path}")
        lines.append(f"  trade events: {events_path}")
    else:
        lines.append(f"pair:            {status.get('pair', '-')}")
        lines.append(f"runtime mode:    {status.get('runtime_mode', '-')}")
        lines.append(f"execution mode:  {status.get('execution_mode', '-')}")
        lines.append(f"raw signal:      {status.get('raw_signal', status.get('last_signal', '-'))}")
        lines.append(f"final action:    {status.get('final_action', status.get('last_action', '-'))}")
        lines.append(f"decision reason: {status.get('decision_reason', '-')}")
        ss_str = _format_signal_strength(status.get("signal_strength"))
        lines.append(f"signal strength: {ss_str}")
        prob = status.get("model_probability")
        if prob is not None:
            lines.append(f"model probability: {prob:.4f}")
        lines.append(f"kill switch:     {'ACTIVE' if status.get('kill_switch_active') else 'inactive'}")
        if status.get("shutdown_reason"):
            lines.append(f"shutdown reason: {status['shutdown_reason']}")
        if status.get("error"):
            lines.append(f"error:           {status['error']}")
        if status.get("live_account") is not None:
            la = status["live_account"]
            usd = la.get("usd", 0)
            xbt = la.get("xbt", 0)
            lines.append(f"live balances:   USD={usd:.2f} XBT={xbt:.6f}")
        lines.append("")
        lines.append(f"recent events (last {args.tail}):")
        if not events:
            lines.append("  (none)")
        else:
            for ev in events:
                lines.append(f"  {_format_event(ev)}")

    lines.append("")
    lines.append(f"files: {status_path} | {events_path}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
