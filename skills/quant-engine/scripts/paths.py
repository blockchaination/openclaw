#!/usr/bin/env python3
"""
Canonical path helpers for OpenClaw quant-engine ML/AI files.

All paths resolve relative to repo root. Use these everywhere for consistency.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return repo root (parent of skills/)."""
    return Path(__file__).resolve().parents[3]


def default_training_examples_path() -> Path:
    """<repo_root>/logs/training_examples.jsonl"""
    return repo_root() / "logs" / "training_examples.jsonl"


def default_shadow_inference_path() -> Path:
    """<repo_root>/logs/shadow_inference.jsonl"""
    return repo_root() / "logs" / "shadow_inference.jsonl"


def default_model_artifact_path() -> Path:
    """<repo_root>/artifacts/first_model_metrics.json"""
    return repo_root() / "artifacts" / "first_model_metrics.json"


def default_artifacts_dir() -> Path:
    """<repo_root>/artifacts"""
    return repo_root() / "artifacts"


def default_signal_outcomes_path() -> Path:
    """<repo_root>/logs/signal_outcomes.jsonl"""
    return repo_root() / "logs" / "signal_outcomes.jsonl"


def default_status_path() -> Path:
    """<repo_root>/logs/status.json"""
    return repo_root() / "logs" / "status.json"


def default_trade_events_path() -> Path:
    """<repo_root>/logs/trade_events.jsonl"""
    return repo_root() / "logs" / "trade_events.jsonl"


def default_log_path() -> Path:
    """<repo_root>/logs/quant_engine_runs.jsonl"""
    return repo_root() / "logs" / "quant_engine_runs.jsonl"


def default_decision_events_path() -> Path:
    """<repo_root>/data/decision_events.jsonl"""
    return repo_root() / "data" / "decision_events.jsonl"


def default_labeled_decision_events_path() -> Path:
    """<repo_root>/data/labeled_decision_events.jsonl"""
    return repo_root() / "data" / "labeled_decision_events.jsonl"


def default_price_history_path() -> Path:
    """<repo_root>/data/price_history.jsonl"""
    return repo_root() / "data" / "price_history.jsonl"


def ensure_logs_and_artifacts_dirs() -> None:
    """Create logs/ and artifacts/ under repo root if they do not exist."""
    root = repo_root()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
