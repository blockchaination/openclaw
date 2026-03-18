#!/usr/bin/env python3
"""
Tests for train_first_model.py.

Run: python -m pytest test_train_first_model.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_first_model import (
    FEATURE_NAMES,
    _compute_metrics,
    _extract_features,
    _filter_buy_flat,
    _load_completed_records,
)


def test_dataset_filtering() -> None:
    """Buy+FLAT filter keeps only candidate_side=buy and spot_state=FLAT."""
    records = [
        {"candidate_side": "buy", "spot_state": "FLAT", "label_300s": 1},
        {"candidate_side": "buy", "spot_state": "LONG", "label_300s": 0},
        {"candidate_side": "sell", "spot_state": "FLAT", "label_300s": 1},
        {"candidate_side": "sell", "spot_state": "LONG", "label_300s": 0},
        {"candidate_side": "buy", "spot_state": "FLAT", "label_300s": 0},
    ]
    filtered = _filter_buy_flat(records)
    assert len(filtered) == 2
    assert all((r.get("candidate_side") or "").lower() == "buy" for r in filtered)
    assert all((r.get("spot_state") or "").upper() == "FLAT" for r in filtered)


def test_load_completed_ignores_incomplete() -> None:
    """Records without label_300s are ignored."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        path.write_text(
            json.dumps({
                "candidate_side": "buy",
                "spot_state": "FLAT",
                "price_after_300s": 105.0,
            }) + "\n",
            encoding="utf-8",
        )
        records = _load_completed_records(path)
        assert len(records) == 0
    finally:
        path.unlink(missing_ok=True)


def test_feature_extraction() -> None:
    """Features are extracted in correct order."""
    rec = {
        "signal_strength": 3.0,
        "spread": 10.0,
        "book_imbalance": 0.05,
        "momentum": -2.0,
        "volatility": 5.0,
        "momentum_threshold": 1.0,
    }
    features = _extract_features(rec)
    assert features == [3.0, 10.0, 0.05, -2.0, 5.0, 1.0]
    assert len(features) == len(FEATURE_NAMES)


def test_feature_extraction_handles_missing() -> None:
    """Missing features become 0.0."""
    rec = {}
    features = _extract_features(rec)
    assert features == [0.0] * len(FEATURE_NAMES)


def test_metrics_file_output() -> None:
    """Training produces valid metrics JSON with expected keys."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data_path = Path(f.name)
    artifacts_dir = Path(tempfile.mkdtemp())
    try:
        records = [
            {
                "candidate_side": "buy",
                "spot_state": "FLAT",
                "signal_strength": 3.0,
                "spread": 10.0,
                "book_imbalance": 0.05,
                "momentum": -2.0,
                "volatility": 5.0,
                "momentum_threshold": 1.0,
                "label_300s": 1,
            },
            {
                "candidate_side": "buy",
                "spot_state": "FLAT",
                "signal_strength": 2.0,
                "spread": 1.0,
                "book_imbalance": 0.03,
                "momentum": -1.0,
                "volatility": 4.0,
                "momentum_threshold": 1.0,
                "label_300s": 0,
            },
            {
                "candidate_side": "buy",
                "spot_state": "FLAT",
                "signal_strength": 2.5,
                "spread": 5.0,
                "book_imbalance": 0.04,
                "momentum": -1.5,
                "volatility": 4.5,
                "momentum_threshold": 1.0,
                "label_300s": 1,
            },
        ]
        data_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "train_first_model.py"),
                "--file",
                str(data_path),
                "--artifacts-dir",
                str(artifacts_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(_SCRIPTS),
        )
        assert result.returncode == 0, result.stderr

        metrics_path = artifacts_dir / "first_model_metrics.json"
        assert metrics_path.exists()
        with metrics_path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert "feature_names" in data
        assert data["feature_names"] == FEATURE_NAMES
        assert "train_row_count" in data
        assert "test_row_count" in data
        assert "metrics" in data
        assert "accuracy" in data["metrics"]
        assert "precision" in data["metrics"]
        assert "recall" in data["metrics"]
        assert "timestamp" in data
    finally:
        data_path.unlink(missing_ok=True)


def test_compute_metrics() -> None:
    """Accuracy, precision, recall computed correctly."""
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 1]
    m = _compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.8
    assert m["precision"] == 1.0
    assert m["recall"] == 2 / 3
