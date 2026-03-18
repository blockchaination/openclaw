#!/usr/bin/env python3
"""
Offline trainer for first buy-candidate classifier.

Reads training_examples.jsonl, trains a simple logistic regression on buy+FLAT
examples, saves metrics to artifacts/first_model_metrics.json.

Run: python train_first_model.py [--file PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


FEATURE_NAMES = [
    "signal_strength",
    "spread",
    "book_imbalance",
    "momentum",
    "volatility",
    "momentum_threshold",
]


def _repo_root() -> Path:
    """Return repo root (parent of skills/)."""
    return Path(__file__).resolve().parents[3]


def _default_path() -> Path:
    return _repo_root() / "logs" / "training_examples.jsonl"


def _default_artifacts_dir() -> Path:
    return _repo_root() / "artifacts"


def _load_completed_records(path: Path) -> list[dict]:
    """Load JSONL, return only records with label_300s present."""
    if not path.exists():
        return []
    records: list[dict] = []
    raw = path.read_bytes()
    # Try utf-8 first; if it yields null bytes (common with UTF-16 on Windows), try utf-16-le
    try:
        text = raw.decode("utf-8")
        if "\x00" in text[: min(200, len(text))]:
            text = raw.decode("utf-16-le")
    except (UnicodeDecodeError, LookupError):
        try:
            text = raw.decode("utf-16-le")
        except (UnicodeDecodeError, LookupError):
            return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("label_300s") is None:
            continue
        records.append(rec)
    return records


def _filter_buy_flat(records: list[dict]) -> list[dict]:
    """Filter to candidate_side==buy and spot_state==FLAT."""
    return [
        r
        for r in records
        if (r.get("candidate_side") or "").lower() == "buy"
        and (r.get("spot_state") or "").upper() == "FLAT"
    ]


def _safe_float(x: object) -> float:
    """Convert to float, return 0.0 if invalid."""
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _extract_features(rec: dict) -> list[float]:
    """Extract feature vector for one record."""
    return [_safe_float(rec.get(name)) for name in FEATURE_NAMES]


def _sigmoid(x: float) -> float:
    """Sigmoid function. Clamp to avoid overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _train_logistic_regression(
    X: list[list[float]],
    y: list[int],
    epochs: int = 100,
    lr: float = 0.1,
) -> tuple[list[float], float]:
    """
    Pure Python logistic regression. Returns (weights, bias).
    X: list of feature vectors, y: list of 0/1 labels.
    """
    n_features = len(FEATURE_NAMES)
    w = [0.0] * n_features
    b = 0.0
    n = len(X)
    if n == 0:
        return w, b

    for _ in range(epochs):
        dw = [0.0] * n_features
        db = 0.0
        for i, xi in enumerate(X):
            pred = _sigmoid(sum(w[j] * xi[j] for j in range(n_features)) + b)
            err = pred - y[i]
            for j in range(n_features):
                dw[j] += err * xi[j]
            db += err
        for j in range(n_features):
            w[j] -= lr * dw[j] / n
        b -= lr * db / n
    return w, b


def _predict_proba(w: list[float], b: float, x: list[float]) -> float:
    """Predict P(y=1) for one sample."""
    return _sigmoid(sum(w[j] * x[j] for j in range(len(x))) + b)


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute accuracy, precision, recall."""
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    correct = sum(1 for i in range(n) if y_true[i] == y_pred[i])
    accuracy = correct / n

    tp = sum(1 for i in range(n) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(n) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(n) if y_true[i] == 1 and y_pred[i] == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train first buy-candidate classifier from training_examples.jsonl.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help=f"Path to training examples JSONL (default: {_default_path()})",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=f"Directory for output artifacts (default: {_default_artifacts_dir()})",
    )
    args = parser.parse_args()
    path = args.file or _default_path()
    artifacts_dir = args.artifacts_dir or _default_artifacts_dir()

    records = _load_completed_records(path)
    buy_flat = _filter_buy_flat(records)

    if len(buy_flat) < 2:
        print("Insufficient buy+FLAT examples for training (need at least 2).")
        print(f"  completed: {len(records)}, buy+FLAT: {len(buy_flat)}")
        return 1

    split_idx = int(len(buy_flat) * 0.8)
    if split_idx < 1:
        split_idx = 1
    train_recs = buy_flat[:split_idx]
    test_recs = buy_flat[split_idx:]

    X_train = [_extract_features(r) for r in train_recs]
    y_train = [int(r.get("label_300s", 0)) for r in train_recs]
    X_test = [_extract_features(r) for r in test_recs]
    y_test = [int(r.get("label_300s", 0)) for r in test_recs]

    w, b = _train_logistic_regression(X_train, y_train)
    y_pred = [1 if _predict_proba(w, b, x) >= 0.5 else 0 for x in X_test]

    baseline_pos = sum(y_test) / len(y_test) if y_test else 0.0
    metrics = _compute_metrics(y_test, y_pred)

    print("OpenClaw First Model Training")
    print("------------------------------")
    print()
    print(f"Training rows:   {len(train_recs)}")
    print(f"Test rows:      {len(test_recs)}")
    print(f"Baseline positive rate: {baseline_pos:.2%}")
    print()
    print("Model metrics (test set):")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print()

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "first_model_metrics.json"
    output = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_names": FEATURE_NAMES,
        "train_row_count": len(train_recs),
        "test_row_count": len(test_recs),
        "baseline_positive_rate": baseline_pos,
        "metrics": metrics,
        "weights": w,
        "bias": b,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
