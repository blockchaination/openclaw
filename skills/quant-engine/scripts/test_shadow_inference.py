#!/usr/bin/env python3
"""
Tests for shadow-mode model inference.

Run: python -m pytest test_shadow_inference.py -v
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

from quant_engine import _apply_shadow_inference, _is_directional_candidate
from shadow_model import load_model, score_candidate


def _directional_result() -> dict:
    """Result representing a buy candidate."""
    return {
        "timestamp_utc": "2026-03-09T12:00:00Z",
        "iteration": 1,
        "pair": "XBTUSD",
        "strategy": {
            "action": "buy",
            "reason": "buy mean-reversion entry",
            "signal_strength": -3.5,
            "inputs": {
                "momentum_threshold": 1.0,
                "spread": 10.0,
                "book_imbalance": 0.05,
                "momentum": -3.5,
                "volatility": 5.0,
                "spot_state": "flat",
            },
        },
        "decision_reason": "buy mean-reversion entry",
        "candidate_side": "buy",
        "candidate_reason": "buy mean-reversion entry",
        "runtime_reason": "signal_buy",
        "signal_strength": -3.5,
        "market": {
            "spread": 10.0,
            "book_imbalance": 0.05,
            "momentum": -3.5,
            "volatility": 5.0,
        },
    }


def _neutral_hold_result() -> dict:
    """Result representing neutral hold (no directional signal)."""
    return {
        "timestamp_utc": "2026-03-09T12:00:00Z",
        "iteration": 1,
        "pair": "XBTUSD",
        "strategy": {
            "action": "hold",
            "reason": "no long-entry signal",
            "signal_strength": None,
            "inputs": {
                "momentum_threshold": 1.0,
                "spread": 10.0,
                "book_imbalance": 0.02,
                "momentum": -0.5,
                "volatility": 5.0,
            },
        },
        "decision_reason": "no long-entry signal",
        "signal_strength": None,
        "market": {
            "spread": 10.0,
            "book_imbalance": 0.02,
            "momentum": -0.5,
            "volatility": 5.0,
        },
    }


def test_candidate_gets_scored() -> None:
    """Directional candidate receives model_score and model_probability."""
    model = {
        "weights": [0.1, -0.01, 2.0, -0.5, 0.0, 0.2],
        "bias": -0.5,
        "feature_names": [
            "signal_strength",
            "spread",
            "book_imbalance",
            "momentum",
            "volatility",
            "momentum_threshold",
        ],
    }
    result = _directional_result()
    shadow_path = Path(tempfile.mkdtemp()) / "shadow.jsonl"

    _apply_shadow_inference(result, model, shadow_path)

    assert "model_score" in result
    assert "model_probability" in result
    assert isinstance(result["model_score"], (int, float))
    assert isinstance(result["model_probability"], (int, float))
    assert 0 <= result["model_probability"] <= 1
    assert shadow_path.exists()
    lines = [l.strip() for l in shadow_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "action" not in row, "shadow log must not use legacy action field"
    assert "prob" in row
    assert "score" in row
    assert row.get("candidate_side") == "buy"
    assert row.get("candidate_reason") == "buy mean-reversion entry"
    assert row.get("runtime_reason") == "signal_buy"
    assert "spot_state" in row
    assert "signal_strength" in row


def test_shadow_inference_includes_threshold_when_present() -> None:
    """Shadow log includes threshold when model has recommended_shadow_threshold."""
    model = {
        "weights": [0.0] * 6,
        "bias": 0.0,
        "feature_names": [
            "signal_strength",
            "spread",
            "book_imbalance",
            "momentum",
            "volatility",
            "momentum_threshold",
        ],
        "recommended_shadow_threshold": 0.65,
    }
    result = _directional_result()
    shadow_path = Path(tempfile.mkdtemp()) / "shadow.jsonl"
    _apply_shadow_inference(result, model, shadow_path)
    assert shadow_path.exists()
    row = json.loads(shadow_path.read_text().strip())
    assert row.get("threshold") == 0.65


def test_neutral_hold_no_bogus_score() -> None:
    """Neutral hold does not produce model_score or model_probability."""
    model = {
        "weights": [0.1, -0.01, 2.0, -0.5, 0.0, 0.2],
        "bias": -0.5,
        "feature_names": [
            "signal_strength",
            "spread",
            "book_imbalance",
            "momentum",
            "volatility",
            "momentum_threshold",
        ],
    }
    result = _neutral_hold_result()
    shadow_path = Path(tempfile.mkdtemp()) / "shadow.jsonl"

    _apply_shadow_inference(result, model, shadow_path)

    assert "model_score" not in result
    assert "model_probability" not in result
    assert not shadow_path.exists()


def test_model_none_skips_scoring() -> None:
    """When model is None, no scoring is applied."""
    result = _directional_result()
    shadow_path = Path(tempfile.mkdtemp()) / "shadow.jsonl"

    _apply_shadow_inference(result, None, shadow_path)

    assert "model_score" not in result
    assert "model_probability" not in result
    assert not shadow_path.exists()


def test_is_directional_candidate() -> None:
    """_is_directional_candidate identifies buy, sell, weak_signal_filtered."""
    assert _is_directional_candidate(_directional_result()) is True
    sell_result = _directional_result()
    sell_result["strategy"]["action"] = "sell"
    sell_result["decision_reason"] = "sell mean-reversion exit"
    assert _is_directional_candidate(sell_result) is True
    weak_result = _neutral_hold_result()
    weak_result["decision_reason"] = "weak_signal_filtered"
    weak_result["strategy"]["reason"] = "weak_signal_filtered"
    assert _is_directional_candidate(weak_result) is True
    assert _is_directional_candidate(_neutral_hold_result()) is False


def test_score_candidate() -> None:
    """score_candidate returns (score, probability) in valid range."""
    model = {
        "weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "bias": 0.0,
        "feature_names": [
            "signal_strength",
            "spread",
            "book_imbalance",
            "momentum",
            "volatility",
            "momentum_threshold",
        ],
    }
    result = _directional_result()
    score, prob = score_candidate(model, result)
    assert isinstance(score, (int, float))
    assert isinstance(prob, (int, float))
    assert 0 <= prob <= 1
    assert prob == 0.5  # sigmoid(0) = 0.5


def test_status_prints_probability_cleanly() -> None:
    """operator_status prints candidate/runtime/model fields when present."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        status_path = Path(f.name)
    try:
        status = {
            "pair": "XBTUSD",
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "raw_signal": "buy",
            "final_action": "buy",
            "decision_reason": "buy mean-reversion entry",
            "candidate_side": "buy",
            "candidate_reason": "buy mean-reversion entry",
            "runtime_reason": "signal_buy",
            "signal_strength": -3.5,
            "model_probability": 0.7234,
            "kill_switch_active": False,
        }
        status_path.write_text(json.dumps(status), encoding="utf-8")

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as ef:
            events_path = Path(ef.name)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPTS / "operator_status.py"),
                    "--status-file",
                    str(status_path),
                    "--trade-events-file",
                    str(events_path),
                    "--tail",
                    "0",
                ],
                capture_output=True,
                text=True,
                cwd=str(_SCRIPTS),
            )
            assert result.returncode == 0
            assert "model probability: 0.7234" in result.stdout
            assert "candidate side:" in result.stdout
            assert "candidate reason:" in result.stdout
            assert "runtime reason:" in result.stdout
        finally:
            events_path.unlink(missing_ok=True)
    finally:
        status_path.unlink(missing_ok=True)
