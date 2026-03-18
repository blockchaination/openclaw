#!/usr/bin/env python3
"""
Tests for path resolution and ML/AI file wiring.

Run: python -m pytest test_paths.py -v
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

from paths import (
    default_model_artifact_path,
    default_shadow_inference_path,
    default_signal_outcomes_path,
    default_training_examples_path,
    ensure_logs_and_artifacts_dirs,
    repo_root,
)


def test_all_default_paths_resolve_under_repo_root() -> None:
    """All default paths resolve under repo root."""
    root = repo_root().resolve()
    assert root.is_absolute()
    assert (root / "skills").exists(), "repo root should contain skills/"
    paths = [
        default_training_examples_path(),
        default_shadow_inference_path(),
        default_model_artifact_path(),
        default_signal_outcomes_path(),
    ]
    for p in paths:
        resolved = p.resolve()
        assert str(resolved).startswith(str(root)), f"{resolved} should be under {root}"


def test_parent_dirs_created_when_needed() -> None:
    """ensure_logs_and_artifacts_dirs creates logs/ and artifacts/ under repo root."""
    root = repo_root()
    ensure_logs_and_artifacts_dirs()
    assert (root / "logs").exists()
    assert (root / "artifacts").exists()


def test_trainer_writes_model_artifact_to_repo_root_artifacts() -> None:
    """train_first_model.py writes first_model_metrics.json to repo-root artifacts/."""
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "training.jsonl"
        artifacts_dir = Path(tmp) / "artifacts"
        artifacts_dir.mkdir()
        # Minimal valid training data
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
        ]
        data_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

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
        data = json.loads(metrics_path.read_text())
        assert "weights" in data
        assert "bias" in data
        assert "feature_names" in data


def test_shadow_loader_reads_from_repo_root_artifact_path() -> None:
    """shadow_model.load_model reads from repo-root artifact path when given explicit path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        metrics_path = Path(f.name)
    try:
        metrics_path.write_text(
            json.dumps({
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
            }),
            encoding="utf-8",
        )
        from shadow_model import load_model
        model = load_model(metrics_path)
        assert model is not None
        assert model["bias"] == 0.0
        assert len(model["weights"]) == 6
    finally:
        metrics_path.unlink(missing_ok=True)


def test_print_openclaw_paths_output() -> None:
    """print_openclaw_paths.py prints resolved paths."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / "print_openclaw_paths.py")],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPTS),
    )
    assert result.returncode == 0
    out = result.stdout
    assert "repo_root:" in out
    assert "training_examples_path:" in out
    assert "shadow_inference_path:" in out
    assert "model_artifact_path:" in out
    assert "signal_outcomes_path:" in out
