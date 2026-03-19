#!/usr/bin/env python3
"""
Minimal Telegram operator layer for OpenClaw quant engine.

Uses Bot API over HTTP. Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
If either is missing, integration stays inactive (no crash).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


TELEGRAM_API = "https://api.telegram.org/bot"


def _is_enabled() -> bool:
    """True if both token and chat_id are set."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    return bool(token and chat_id)


def send_telegram_message(text: str) -> None:
    """Send text to configured chat. No-op if env vars missing."""
    if not _is_enabled():
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    url = f"{TELEGRAM_API}{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
    except (urllib.error.URLError, OSError, TimeoutError):
        pass


def get_telegram_updates(offset: int | None = None) -> list[dict]:
    """Fetch updates. Returns list of update objects. Empty if disabled or error."""
    if not _is_enabled():
        return []
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    url = f"{TELEGRAM_API}{token}/getUpdates?timeout=5"
    if offset is not None:
        url += f"&offset={offset}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return []
    if not data.get("ok"):
        return []
    results = data.get("result") or []
    out: list[dict] = []
    for r in results:
        msg = r.get("message") or {}
        msg_chat_id = str(msg.get("chat", {}).get("id", ""))
        if msg_chat_id != chat_id:
            continue
        out.append(r)
    return out


def parse_telegram_command(update: dict) -> str | None:
    """
    Extract command from update if present. Returns /status, /stop, /start, /help, or None.
    """
    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    if not text or not text.startswith("/"):
        return None
    parts = text.split()
    cmd = parts[0].lower()
    if cmd in ("/status", "/stop", "/start", "/help"):
        return cmd
    return None


def format_trade_alert(ev: dict) -> str:
    """Format live_order_submitted as operator-friendly alert."""
    pair = ev.get("pair", "?")
    side = (ev.get("side") or "?").upper()
    price = ev.get("price")
    price_str = f"{price:.2f}" if isinstance(price, (int, float)) else str(price or "-")
    reason = ev.get("reason", ev.get("candidate_reason", "-"))
    ss = ev.get("signal_strength")
    ss_str = f"{ss:.2f}" if isinstance(ss, (int, float)) else (str(ss) if ss is not None else "-")
    runtime = ev.get("runtime_mode", "?")
    exec_mode = ev.get("execution_mode", "?")
    return (
        f"<b>OpenClaw Trade</b>\n"
        f"Pair: {pair}\n"
        f"Action: {side}\n"
        f"Price: {price_str}\n"
        f"Reason: {reason}\n"
        f"Signal strength: {ss_str}\n"
        f"Runtime: {runtime}/{exec_mode}"
    )


def format_error_alert(ev: dict) -> str:
    """Format engine_error as operator-friendly alert."""
    pair = ev.get("pair", "?")
    reason = ev.get("reason", "unexpected_error")
    err = ev.get("error", "")
    return (
        f"<b>OpenClaw Error</b>\n"
        f"Pair: {pair}\n"
        f"Reason: {reason}\n"
        f"Error: {err}"
    )


def format_submission_failed_alert(ev: dict) -> str:
    """Format live_order_submission_failed."""
    pair = ev.get("pair", "?")
    side = (ev.get("side") or "?").upper()
    reason = ev.get("reason", "unknown")
    return (
        f"<b>OpenClaw Order Failed</b>\n"
        f"Pair: {pair}\n"
        f"Side: {side}\n"
        f"Reason: {reason}"
    )


def format_engine_started(ev: dict) -> str:
    """Format engine_started."""
    pair = ev.get("pair", "?")
    runtime = ev.get("runtime_mode", "?")
    exec_mode = ev.get("execution_mode", "?")
    iters = ev.get("iterations", "?")
    return (
        f"<b>OpenClaw Engine Started</b>\n"
        f"Pair: {pair}\n"
        f"Runtime: {runtime}/{exec_mode}\n"
        f"Iterations: {iters}"
    )


def format_engine_stopped(ev: dict) -> str:
    """Format engine_stopped."""
    pair = ev.get("pair", "?")
    reason = ev.get("reason", "?")
    err = ev.get("error")
    lines = [
        "<b>OpenClaw Engine Stopped</b>",
        f"Pair: {pair}",
        f"Reason: {reason}",
    ]
    if err:
        lines.append(f"Error: {err}")
    return "\n".join(lines)


def format_kill_switch_triggered(ev: dict) -> str:
    """Format kill_switch_triggered."""
    pair = ev.get("pair", "?")
    reason = ev.get("reason", "kill switch")
    return (
        f"<b>OpenClaw Kill Switch</b>\n"
        f"Pair: {pair}\n"
        f"Status: ACTIVATED\n"
        f"Reason: {reason}"
    )


def format_kill_switch_deactivated(pair: str) -> str:
    """Format kill switch deactivated confirmation."""
    return (
        f"<b>OpenClaw Kill Switch</b>\n"
        f"Pair: {pair}\n"
        f"Status: DEACTIVATED"
    )


def format_kill_switch_activated(pair: str) -> str:
    """Format kill switch activated confirmation."""
    return (
        f"<b>OpenClaw Kill Switch</b>\n"
        f"Pair: {pair}\n"
        f"Status: ACTIVATED"
    )


def format_status_reply(status: dict) -> str:
    """Format /status reply from status dict."""
    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        return str(v)

    def _ss(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, (int, float)) and abs(v) < 1e-9:
            return "0.0"
        return str(v)

    lines = [
        "<b>OpenClaw Status</b>",
        f"pair: {_fmt(status.get('pair'))}",
        f"runtime mode: {_fmt(status.get('runtime_mode'))}",
        f"execution mode: {_fmt(status.get('execution_mode'))}",
        f"raw signal: {_fmt(status.get('raw_signal', status.get('last_signal')))}",
        f"final action: {_fmt(status.get('final_action', status.get('last_action')))}",
        f"decision reason: {_fmt(status.get('decision_reason'))}",
        f"signal strength: {_ss(status.get('signal_strength'))}",
        f"candidate side: {_fmt(status.get('candidate_side'))}",
        f"candidate reason: {_fmt(status.get('candidate_reason'))}",
        f"runtime reason: {_fmt(status.get('runtime_reason'))}",
    ]
    prob = status.get("model_probability")
    if prob is not None:
        lines.append(f"model probability: {prob:.4f}")
    lines.append(f"kill switch: {'ACTIVE' if status.get('kill_switch_active') else 'inactive'}")
    if status.get("live_order_cooldown_active"):
        lines.append("live order cooldown: ACTIVE (15 min)")
    la = status.get("live_account")
    if la is not None:
        usd = la.get("usd", 0) or 0
        xbt = la.get("xbt", 0) or 0
        usd_val = float(usd) if isinstance(usd, (int, float)) else 0.0
        xbt_val = float(xbt) if isinstance(xbt, (int, float)) else 0.0
        lines.append(f"live balances: USD={usd_val:.2f} XBT={xbt_val:.6f}")
    return "\n".join(lines)


def format_help_reply() -> str:
    """Format /help reply."""
    return (
        "<b>OpenClaw Commands</b>\n"
        "/status - current status\n"
        "/stop - activate kill switch\n"
        "/start - deactivate kill switch\n"
        "/help - this message"
    )


def send_alert_for_event(ev: dict) -> None:
    """Send Telegram alert for event type. No-op if disabled or unknown type."""
    if not _is_enabled():
        return
    etype = ev.get("event_type", "")
    if etype == "live_order_submitted":
        send_telegram_message(format_trade_alert(ev))
    elif etype == "live_order_submission_failed":
        send_telegram_message(format_submission_failed_alert(ev))
    elif etype == "engine_error":
        send_telegram_message(format_error_alert(ev))
    elif etype == "engine_started":
        send_telegram_message(format_engine_started(ev))
    elif etype == "engine_stopped":
        send_telegram_message(format_engine_stopped(ev))
    elif etype == "kill_switch_triggered":
        send_telegram_message(format_kill_switch_triggered(ev))
