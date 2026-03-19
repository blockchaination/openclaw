#!/usr/bin/env python3
"""
Tests for kill-switch hotfix: stay alive, no spam, /stop then /start in-process.

Run: python -m pytest test_kill_switch_behavior.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_kill_switch_blocks_trading_without_exit() -> None:
    """Kill switch active: no trades, no process exit, exactly one kill_switch_triggered."""
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp) / "logs"
        logs.mkdir()
        status_path = logs / "status.json"
        trade_events_path = logs / "trade_events.jsonl"
        kill_switch_path = Path(tmp) / "openclaw.kill"
        kill_switch_path.touch()

        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "quant_engine.py"),
                "--pair",
                "XBTUSD",
                "--iterations",
                "3",
                "--sleep-seconds",
                "0",
                "--status-file",
                str(status_path),
                "--trade-events-file",
                str(trade_events_path),
                "--kill-switch-file",
                str(kill_switch_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_SCRIPTS),
        )
        assert result.returncode == 0
        events = [
            json.loads(line)
            for line in trade_events_path.read_text().strip().split("\n")
            if line.strip()
        ]
        kill_triggered = [e for e in events if e.get("event_type") == "kill_switch_triggered"]
        engine_stopped = [e for e in events if e.get("event_type") == "engine_stopped"]
        assert len(kill_triggered) == 1, "kill_switch_triggered should fire exactly once"
        assert not any(
            e.get("reason") == "kill_switch" for e in engine_stopped
        ), "no engine_stopped for kill_switch when staying alive"


def test_stop_then_start_works_in_process() -> None:
    """_process_telegram_commands: /stop creates file, /start removes it; trading can resume."""
    import os
    from unittest.mock import patch

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
                    {"update_id": 1, "message": {"chat": {"id": 123}, "text": "/stop"}},
                ]
                _process_telegram_commands(status_path, kill_switch_path, "XBTUSD", 0)
        assert kill_switch_path.exists()

        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "fake", "TELEGRAM_CHAT_ID": "123"},
            clear=False,
        ):
            with patch("quant_engine.get_telegram_updates") as mock_get:
                mock_get.return_value = [
                    {"update_id": 2, "message": {"chat": {"id": 123}, "text": "/start"}},
                ]
                _process_telegram_commands(status_path, kill_switch_path, "XBTUSD", 1)
        assert not kill_switch_path.exists()


def test_no_repeated_kill_switch_spam() -> None:
    """Multiple iterations with kill switch: only one kill_switch_triggered event."""
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp) / "logs"
        logs.mkdir()
        status_path = logs / "status.json"
        trade_events_path = logs / "trade_events.jsonl"
        kill_switch_path = Path(tmp) / "openclaw.kill"
        kill_switch_path.touch()

        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "quant_engine.py"),
                "--pair",
                "XBTUSD",
                "--iterations",
                "10",
                "--sleep-seconds",
                "0",
                "--status-file",
                str(status_path),
                "--trade-events-file",
                str(trade_events_path),
                "--kill-switch-file",
                str(kill_switch_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_SCRIPTS),
        )
        assert result.returncode == 0
        events = [
            json.loads(line)
            for line in trade_events_path.read_text().strip().split("\n")
            if line.strip()
        ]
        kill_triggered = [e for e in events if e.get("event_type") == "kill_switch_triggered"]
        engine_started = [e for e in events if e.get("event_type") == "engine_started"]
        assert len(kill_triggered) == 1
        assert len(engine_started) == 1
