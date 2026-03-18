#!/usr/bin/env python3
"""
Tests for Telegram notifier: env handling, command parsing, formatting, alerts.

Run: python -m pytest test_telegram_notifier.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from telegram_notifier import (
    format_error_alert,
    format_engine_started,
    format_engine_stopped,
    format_help_reply,
    format_kill_switch_activated,
    format_kill_switch_deactivated,
    format_kill_switch_triggered,
    format_status_reply,
    format_submission_failed_alert,
    format_trade_alert,
    get_telegram_updates,
    parse_telegram_command,
    send_alert_for_event,
    send_telegram_message,
)


def test_env_missing_no_crash() -> None:
    """With TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID unset, no crash."""
    with patch.dict(os.environ, {}, clear=False):
        for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            os.environ.pop(key, None)
        send_telegram_message("test")
        updates = get_telegram_updates()
        assert updates == []
        send_alert_for_event({"event_type": "engine_error", "pair": "XBTUSD", "error": "x"})


def test_parse_telegram_command_status() -> None:
    """parse_telegram_command returns /status for status command."""
    upd = {"update_id": 1, "message": {"chat": {"id": 123}, "text": "/status"}}
    assert parse_telegram_command(upd) == "/status"


def test_parse_telegram_command_stop() -> None:
    """parse_telegram_command returns /stop."""
    upd = {"update_id": 1, "message": {"chat": {"id": 123}, "text": "/stop"}}
    assert parse_telegram_command(upd) == "/stop"


def test_parse_telegram_command_start() -> None:
    """parse_telegram_command returns /start."""
    upd = {"update_id": 1, "message": {"chat": {"id": 123}, "text": "/start"}}
    assert parse_telegram_command(upd) == "/start"


def test_parse_telegram_command_help() -> None:
    """parse_telegram_command returns /help."""
    upd = {"update_id": 1, "message": {"chat": {"id": 123}, "text": "/help"}}
    assert parse_telegram_command(upd) == "/help"


def test_parse_telegram_command_with_args() -> None:
    """parse_telegram_command returns command when text has trailing args."""
    upd = {"update_id": 1, "message": {"chat": {"id": 123}, "text": "/status foo"}}
    assert parse_telegram_command(upd) == "/status"


def test_parse_telegram_command_none_for_invalid() -> None:
    """parse_telegram_command returns None for non-commands."""
    assert parse_telegram_command({"message": {"text": "hello"}}) is None
    assert parse_telegram_command({"message": {"text": "/unknown"}}) is None
    assert parse_telegram_command({"message": {}}) is None
    assert parse_telegram_command({}) is None


def test_format_status_reply() -> None:
    """format_status_reply produces operator-friendly status text."""
    status = {
        "pair": "XBTUSD",
        "runtime_mode": "live",
        "execution_mode": "taker",
        "raw_signal": "buy",
        "final_action": "submitted",
        "decision_reason": "buy mean-reversion entry",
        "signal_strength": 12.17,
        "candidate_side": "buy",
        "candidate_reason": "buy mean-reversion entry",
        "runtime_reason": "live_submitted",
        "kill_switch_active": False,
        "live_account": {"usd": 1000.5, "xbt": 0.05},
    }
    text = format_status_reply(status)
    assert "OpenClaw Status" in text
    assert "XBTUSD" in text
    assert "live" in text
    assert "taker" in text
    assert "buy" in text
    assert "12.17" in text
    assert "inactive" in text
    assert "USD=" in text
    assert "XBT=" in text


def test_format_status_reply_kill_switch_active() -> None:
    """format_status_reply shows ACTIVE when kill_switch_active is True."""
    status = {"pair": "XBTUSD", "kill_switch_active": True}
    text = format_status_reply(status)
    assert "ACTIVE" in text


def test_format_trade_alert() -> None:
    """format_trade_alert produces operator-friendly trade text."""
    ev = {
        "pair": "XBTUSD",
        "side": "buy",
        "price": 71623.65,
        "reason": "buy mean-reversion entry",
        "signal_strength": 12.17,
        "runtime_mode": "live",
        "execution_mode": "taker",
    }
    text = format_trade_alert(ev)
    assert "OpenClaw Trade" in text
    assert "XBTUSD" in text
    assert "BUY" in text
    assert "71623.65" in text
    assert "buy mean-reversion entry" in text
    assert "12.17" in text
    assert "live" in text
    assert "taker" in text


def test_format_error_alert() -> None:
    """format_error_alert produces operator-friendly error text."""
    ev = {"pair": "XBTUSD", "reason": "unexpected_error", "error": "Connection refused"}
    text = format_error_alert(ev)
    assert "OpenClaw Error" in text
    assert "XBTUSD" in text
    assert "Connection refused" in text


def test_format_submission_failed_alert() -> None:
    """format_submission_failed_alert produces expected text."""
    ev = {"pair": "XBTUSD", "side": "buy", "reason": "Insufficient funds"}
    text = format_submission_failed_alert(ev)
    assert "OpenClaw Order Failed" in text
    assert "BUY" in text
    assert "Insufficient funds" in text


def test_format_engine_started() -> None:
    """format_engine_started produces expected text."""
    ev = {"pair": "XBTUSD", "runtime_mode": "live", "execution_mode": "taker", "iterations": 100}
    text = format_engine_started(ev)
    assert "OpenClaw Engine Started" in text
    assert "100" in text


def test_format_engine_stopped() -> None:
    """format_engine_stopped produces expected text."""
    ev = {"pair": "XBTUSD", "reason": "kill_switch", "error": None}
    text = format_engine_stopped(ev)
    assert "OpenClaw Engine Stopped" in text
    assert "kill_switch" in text


def test_format_kill_switch_triggered() -> None:
    """format_kill_switch_triggered produces expected text."""
    ev = {"pair": "XBTUSD", "reason": "kill switch file exists"}
    text = format_kill_switch_triggered(ev)
    assert "OpenClaw Kill Switch" in text
    assert "ACTIVATED" in text


def test_format_kill_switch_activated_deactivated() -> None:
    """format_kill_switch_activated and deactivated produce expected text."""
    assert "ACTIVATED" in format_kill_switch_activated("XBTUSD")
    assert "DEACTIVATED" in format_kill_switch_deactivated("XBTUSD")


def test_format_help_reply() -> None:
    """format_help_reply lists all commands."""
    text = format_help_reply()
    assert "/status" in text
    assert "/stop" in text
    assert "/start" in text
    assert "/help" in text


def test_send_alert_for_event_unknown_type_no_crash() -> None:
    """send_alert_for_event with unknown event_type does not crash."""
    with patch.dict(os.environ, {}, clear=False):
        for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            os.environ.pop(key, None)
        send_alert_for_event({"event_type": "unknown_type"})


def test_process_telegram_commands_stop_creates_kill_switch() -> None:
    """_process_telegram_commands /stop creates kill switch file."""
    import tempfile

    from quant_engine import _process_telegram_commands

    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        kill_switch_path = Path(tmp) / "openclaw.kill"
        status_path.write_text('{"pair":"XBTUSD"}', encoding="utf-8")

        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "fake", "TELEGRAM_CHAT_ID": "123"},
            clear=False,
        ):
            with patch("quant_engine.get_telegram_updates") as mock_get:
                mock_get.return_value = [
                    {
                        "update_id": 1,
                        "message": {"chat": {"id": 123}, "text": "/stop"},
                    },
                ]
                _process_telegram_commands(
                    status_path, kill_switch_path, "XBTUSD", 0
                )

        assert kill_switch_path.exists()


def test_process_telegram_commands_start_removes_kill_switch() -> None:
    """_process_telegram_commands /start removes kill switch file."""
    import tempfile

    from quant_engine import _process_telegram_commands

    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.json"
        kill_switch_path = Path(tmp) / "openclaw.kill"
        kill_switch_path.touch()
        status_path.write_text('{"pair":"XBTUSD"}', encoding="utf-8")

        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "fake", "TELEGRAM_CHAT_ID": "123"},
            clear=False,
        ):
            with patch("quant_engine.get_telegram_updates") as mock_get:
                mock_get.return_value = [
                    {
                        "update_id": 1,
                        "message": {"chat": {"id": 123}, "text": "/start"},
                    },
                ]
                _process_telegram_commands(
                    status_path, kill_switch_path, "XBTUSD", 0
                )

        assert not kill_switch_path.exists()
