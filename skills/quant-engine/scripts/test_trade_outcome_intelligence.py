#!/usr/bin/env python3
"""
Tests for Trade Outcome Intelligence Layer: decision logging, outcome labeling,
expectancy gate, and engine degradation on missing data.

Run: python -m pytest skills/quant-engine/scripts/test_trade_outcome_intelligence.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from decision_logger import log_decision_snapshot
from decision_features import _regime_tag
from outcome_labeler import label_pending_decisions
from expectancy_gate import evaluate_expectancy
from price_history import append_price


# --- Decision snapshot tests ---


def test_decision_snapshot_creation() -> None:
    """Decision snapshot is created with required fields."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decisions.jsonl"
        result = {
            "timestamp_utc": "2025-03-19T12:00:00Z",
            "pair": "XBTUSD",
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "iteration": 1,
            "raw_signal": "buy",
            "final_action": "buy",
            "decision_reason": "signal_buy",
            "signal_strength": 2.0,
            "market": {
                "mid_price": 100000.0,
                "spread": 1.0,
                "volatility": 5.0,
                "momentum": -10.0,
                "book_imbalance": 0.02,
            },
            "strategy": {"action": "buy", "inputs": {"momentum_threshold": 5.0}},
            "order": {"submitted": True, "skipped_reason": None},
            "broker": {"position_units": 0.0},
            "live_account": None,
        }
        log_decision_snapshot(
            result,
            path,
            cooldown_active=False,
            kill_switch_active=False,
            trade_submitted=True,
        )
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["timestamp_utc"] == "2025-03-19T12:00:00Z"
        assert rec["pair"] == "XBTUSD"
        assert rec["signal_direction"] == "buy"
        assert rec["decision_action"] == "buy"
        assert rec["mid_price"] == 100000.0
        assert rec["trade_submitted"] is True
        assert rec["cooldown_active"] is False
        assert rec["kill_switch_active"] is False


# --- Outcome labeling tests ---


def test_outcome_labeling_idempotent() -> None:
    """Labeling is idempotent: records with label_completed_at are not relabeled."""
    with tempfile.TemporaryDirectory() as tmp:
        decision_path = Path(tmp) / "decisions.jsonl"
        labeled_path = Path(tmp) / "labeled.jsonl"
        price_path = Path(tmp) / "prices.jsonl"

        base_ts = datetime(2025, 3, 19, 12, 0, 0, tzinfo=timezone.utc)
        rec = {
            "timestamp_utc": base_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pair": "XBTUSD",
            "iteration": 1,
            "mid_price": 100000.0,
            "signal_direction": "buy",
        }
        decision_path.write_text(json.dumps(rec, separators=(",", ":")) + "\n")

        # Append prices at 5m, 15m, 60m
        for m in [5, 15, 60]:
            ts = base_ts + timedelta(minutes=m)
            append_price(
                price_path,
                ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "XBTUSD",
                100000.0 + m * 10,
            )

        n1 = label_pending_decisions(
            decision_path, labeled_path, price_path, horizons_minutes=(5, 15, 60)
        )
        assert n1 == 1
        assert labeled_path.exists()
        lines1 = labeled_path.read_text().strip().split("\n")
        assert len(lines1) == 1

        n2 = label_pending_decisions(
            decision_path, labeled_path, price_path, horizons_minutes=(5, 15, 60)
        )
        assert n2 == 0
        lines2 = labeled_path.read_text().strip().split("\n")
        assert len(lines2) == 1


def test_outcome_labeling_sell_return_direction() -> None:
    """Sell decisions: return = (entry - future) / entry (profit when price drops)."""
    with tempfile.TemporaryDirectory() as tmp:
        decision_path = Path(tmp) / "decisions.jsonl"
        labeled_path = Path(tmp) / "labeled.jsonl"
        price_path = Path(tmp) / "prices.jsonl"

        base_ts = datetime(2025, 3, 19, 12, 0, 0, tzinfo=timezone.utc)
        rec = {
            "timestamp_utc": base_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pair": "XBTUSD",
            "iteration": 1,
            "mid_price": 100000.0,
            "signal_direction": "sell",
        }
        decision_path.write_text(json.dumps(rec, separators=(",", ":")) + "\n")

        # Price drops to 99000 at 15m (sell profit)
        ts_15 = base_ts + timedelta(minutes=15)
        append_price(
            price_path,
            ts_15.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "XBTUSD",
            99000.0,
        )
        ts_5 = base_ts + timedelta(minutes=5)
        append_price(price_path, ts_5.strftime("%Y-%m-%dT%H:%M:%SZ"), "XBTUSD", 99500.0)
        ts_60 = base_ts + timedelta(minutes=60)
        append_price(price_path, ts_60.strftime("%Y-%m-%dT%H:%M:%SZ"), "XBTUSD", 98500.0)

        label_pending_decisions(
            decision_path, labeled_path, price_path, horizons_minutes=(5, 15, 60)
        )
        lines = labeled_path.read_text().strip().split("\n")
        labeled = json.loads(lines[0])
        # Sell: return = (100000 - 99000) / 100000 = 0.01
        assert labeled["return_15m"] == 0.01
        assert labeled["future_price_15m"] == 99000.0


# --- Expectancy gate tests ---


def test_expectancy_gate_allows_when_thresholds_met() -> None:
    """Expectancy gate allows trade when min samples, mean return, win rate are met."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        # 25 samples, mean return 0.002, win rate 0.6
        records = []
        for i in range(25):
            r = 0.003 if i < 15 else -0.001
            records.append({
                "timestamp_utc": "2025-03-19T12:00:00Z",
                "pair": "XBTUSD",
                "signal_direction": "buy",
                "momentum": -5.0,
                "volatility": 10.0,
                "spread": 2.0,
                "return_15m": r,
            })
        labeled_path.write_text(
            "\n".join(json.dumps(x, separators=(",", ":")) for x in records)
        )
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        allowed, reason, stats = evaluate_expectancy(
            decision,
            labeled_path,
            min_samples=20,
            min_mean_return=0.001,
            min_win_rate=0.55,
            allow_if_insufficient=False,
        )
        assert allowed is True
        assert reason == "expectancy_gate_passed"
        assert stats["sample_count"] == 25
        assert stats["mean_return_15m"] is not None
        assert stats["win_rate"] >= 0.55


def test_expectancy_gate_blocks_when_thresholds_not_met() -> None:
    """Expectancy gate blocks when mean return or win rate below threshold."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        # 25 samples, mean return -0.001, win rate 0.4
        records = []
        for i in range(25):
            r = 0.002 if i < 10 else -0.003
            records.append({
                "timestamp_utc": "2025-03-19T12:00:00Z",
                "pair": "XBTUSD",
                "signal_direction": "buy",
                "momentum": -5.0,
                "volatility": 10.0,
                "spread": 2.0,
                "return_15m": r,
            })
        labeled_path.write_text(
            "\n".join(json.dumps(x, separators=(",", ":")) for x in records)
        )
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        allowed, reason, stats = evaluate_expectancy(
            decision,
            labeled_path,
            min_samples=20,
            min_mean_return=0.001,
            min_win_rate=0.55,
            allow_if_insufficient=False,
        )
        assert allowed is False
        assert reason == "expectancy_gate_blocked"
        assert stats.get("blocked") is True


def test_expectancy_gate_hold_no_gate() -> None:
    """Expectancy gate passes for hold signals without evaluation."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        labeled_path.write_text("")
        decision = {"signal_direction": "hold", "market": {}}
        allowed, reason, stats = evaluate_expectancy(
            decision, labeled_path, allow_if_insufficient=False
        )
        assert allowed is True
        assert reason == "hold_no_gate"


def test_expectancy_gate_insufficient_data_allow() -> None:
    """When allow_if_insufficient=True, allows trade with no labeled data."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        labeled_path.write_text("")
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        allowed, reason, stats = evaluate_expectancy(
            decision, labeled_path, allow_if_insufficient=True
        )
        assert allowed is True
        assert reason == "insufficient_data_allow"


def test_expectancy_gate_missing_file_no_crash() -> None:
    """Expectancy gate does not crash when labeled file is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "nonexistent.jsonl"
        assert not labeled_path.exists()
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        allowed, reason, stats = evaluate_expectancy(
            decision, labeled_path, allow_if_insufficient=False
        )
        assert allowed is False
        assert reason == "expectancy_gate_blocked"


def test_expectancy_gate_malformed_file_no_crash() -> None:
    """Expectancy gate degrades when labeled file has malformed lines."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        labeled_path.write_text('{"return_15m": 0.001}\nnot json\n{"return_15m": 0.002}\n')
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        allowed, reason, stats = evaluate_expectancy(
            decision, labeled_path, allow_if_insufficient=False
        )
        # Should still work with the 2 valid records (or block if insufficient)
        assert reason in ("expectancy_gate_passed", "expectancy_gate_blocked", "insufficient_samples_allow")


# --- New decision snapshot fields ---


def test_decision_snapshot_new_fields_present() -> None:
    """New decision snapshot fields (decision_features, expectancy) are present when passed."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decisions.jsonl"
        result = {
            "timestamp_utc": "2025-03-19T12:00:00Z",
            "pair": "XBTUSD",
            "market": {"mid_price": 100000.0, "spread": 1.0},
            "strategy": {"action": "buy"},
            "broker": {"position_units": 0.0},
        }
        features = {
            "return_30s": 0.0001,
            "return_60s": 0.0002,
            "momentum_60s": 20.0,
            "regime_tag": "trend",
            "spread_bps": 1.0,
        }
        log_decision_snapshot(
            result,
            path,
            decision_features=features,
            expectancy_gate_mode="shadow",
            expectancy_gate_decision="allow",
            expectancy_gate_reason="expectancy_gate_passed",
            expectancy_sample_count=25,
            expectancy_mean_return_15m=0.001,
            expectancy_win_rate=0.6,
            expectancy_feature_bucket_summary="dir=buy|reg=trend",
            expectancy_counterfactual_blocked=False,
        )
        rec = json.loads(path.read_text().strip().split("\n")[0])
        assert rec.get("return_30s") == 0.0001
        assert rec.get("regime_tag") == "trend"
        assert rec.get("expectancy_gate_mode") == "shadow"
        assert rec.get("expectancy_gate_decision") == "allow"
        assert rec.get("expectancy_counterfactual_blocked") is False
        assert "dir=buy|reg=trend" in (rec.get("expectancy_feature_bucket_summary") or "")


def test_regime_tag_deterministic() -> None:
    """regime_tag is computed deterministically from momentum and distance."""
    # trend: strong momentum, far from mean
    assert _regime_tag(0.001, 25.0, 30.0) == "trend"
    assert _regime_tag(-0.001, -25.0, -30.0) == "trend"

    # mean_revert: weak momentum, near mean
    assert _regime_tag(0.0001, 5.0, 10.0) == "mean_revert"

    # neutral: otherwise
    assert _regime_tag(0.0003, 10.0, 15.0) == "neutral"

    # None when all None
    assert _regime_tag(None, None, None) is None


def test_expectancy_gate_feature_bucket_summary_stable() -> None:
    """Feature bucket summary is deterministic and stable for same inputs."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        records = [
            {
                "timestamp_utc": "2025-03-19T12:00:00Z",
                "pair": "XBTUSD",
                "signal_direction": "buy",
                "momentum": -5.0,
                "volatility": 10.0,
                "spread": 2.0,
                "return_15m": 0.002,
            }
            for _ in range(25)
        ]
        labeled_path.write_text(
            "\n".join(json.dumps(x, separators=(",", ":")) for x in records)
        )
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        _, _, stats1 = evaluate_expectancy(
            decision, labeled_path, min_samples=20, min_mean_return=0.001, min_win_rate=0.55
        )
        _, _, stats2 = evaluate_expectancy(
            decision, labeled_path, min_samples=20, min_mean_return=0.001, min_win_rate=0.55
        )
        s1 = stats1.get("expectancy_feature_bucket_summary", "")
        s2 = stats2.get("expectancy_feature_bucket_summary", "")
        assert s1 == s2
        assert "dir=buy" in s1


def test_expectancy_gate_shadow_does_not_block() -> None:
    """Shadow mode: gate evaluates but caller does not block. Test via stats."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        # Data that would block (low win rate)
        records = [
            {
                "timestamp_utc": "2025-03-19T12:00:00Z",
                "pair": "XBTUSD",
                "signal_direction": "buy",
                "momentum": -5.0,
                "volatility": 10.0,
                "spread": 2.0,
                "return_15m": -0.002 if i < 15 else 0.001,
            }
            for i in range(25)
        ]
        labeled_path.write_text(
            "\n".join(json.dumps(x, separators=(",", ":")) for x in records)
        )
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        allowed, reason, stats = evaluate_expectancy(
            decision, labeled_path, min_samples=20, min_mean_return=0.001, min_win_rate=0.55
        )
        # Gate would block (allowed=False)
        assert allowed is False
        assert stats.get("expectancy_counterfactual_blocked") is None  # set by caller
        # In shadow mode, caller ignores allowed and does not set skipped_reason
        # This test verifies the gate returns blocked; integration tests verify shadow doesn't block


def test_expectancy_gate_active_blocks_when_thresholds_fail() -> None:
    """Active mode: gate blocks when thresholds not met (same as existing block test)."""
    with tempfile.TemporaryDirectory() as tmp:
        labeled_path = Path(tmp) / "labeled.jsonl"
        records = []
        for i in range(25):
            r = 0.002 if i < 10 else -0.003
            records.append({
                "timestamp_utc": "2025-03-19T12:00:00Z",
                "pair": "XBTUSD",
                "signal_direction": "buy",
                "momentum": -5.0,
                "volatility": 10.0,
                "spread": 2.0,
                "return_15m": r,
            })
        labeled_path.write_text(
            "\n".join(json.dumps(x, separators=(",", ":")) for x in records)
        )
        decision = {
            "signal_direction": "buy",
            "market": {"momentum": -5.0, "volatility": 10.0, "spread": 2.0},
        }
        allowed, reason, stats = evaluate_expectancy(
            decision, labeled_path, min_samples=20, min_mean_return=0.001, min_win_rate=0.55
        )
        assert allowed is False
        assert reason == "expectancy_gate_blocked"
        assert stats.get("expectancy_gate_decision") == "block"


def test_shadow_counterfactual_logged() -> None:
    """Shadow mode: expectancy_counterfactual_blocked is logged when gate would block."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decisions.jsonl"
        result = {
            "timestamp_utc": "2025-03-19T12:00:00Z",
            "pair": "XBTUSD",
            "final_action": "buy",
            "market": {"mid_price": 100000.0},
            "strategy": {"action": "buy"},
            "order": {},
            "broker": {},
        }
        log_decision_snapshot(
            result,
            path,
            expectancy_gate_mode="shadow",
            expectancy_gate_decision="block",
            expectancy_gate_reason="mean_ret<min",
            expectancy_counterfactual_blocked=True,
        )
        rec = json.loads(path.read_text().strip().split("\n")[0])
        assert rec["expectancy_counterfactual_blocked"] is True
        assert rec["expectancy_gate_decision"] == "block"
        assert rec["decision_action"] == "buy"  # real action taken, not blocked
