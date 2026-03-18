#!/usr/bin/env python3
"""
Shadow-mode model inference for quant engine.

Loads model metadata from artifacts/first_model_metrics.json and scores
directional candidates. Does NOT control trading; rules remain in control.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from paths import default_model_artifact_path


FEATURE_NAMES = [
    "signal_strength",
    "spread",
    "book_imbalance",
    "momentum",
    "volatility",
    "momentum_threshold",
]


def _sigmoid(x: float) -> float:
    """Sigmoid function. Clamp to avoid overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def load_model(metrics_path: Path | None = None) -> dict | None:
    """
    Load model metadata from JSON. Return dict with weights, bias, feature_names,
    or None if file missing/invalid.
    """
    path = metrics_path or default_model_artifact_path()
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    weights = data.get("weights")
    bias = data.get("bias")
    names = data.get("feature_names")
    if not isinstance(weights, list) or not isinstance(names, list):
        return None
    if len(weights) != len(names) or len(names) != len(FEATURE_NAMES):
        return None
    if bias is None or not isinstance(bias, (int, float)):
        return None
    out: dict = {"weights": weights, "bias": float(bias), "feature_names": list(names)}
    thresh = data.get("recommended_shadow_threshold")
    if thresh is not None and isinstance(thresh, (int, float)):
        out["recommended_shadow_threshold"] = float(thresh)
    return out


def _safe_float(x: object) -> float:
    """Convert to float, return 0.0 if invalid."""
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _extract_features_from_result(result: dict) -> list[float]:
    """Extract feature vector from quant engine result dict."""
    market = result.get("market") or {}
    strategy = result.get("strategy") or {}
    inputs = strategy.get("inputs") or {}
    rec = {
        "signal_strength": result.get("signal_strength"),
        "spread": market.get("spread"),
        "book_imbalance": market.get("book_imbalance"),
        "momentum": market.get("momentum"),
        "volatility": market.get("volatility"),
        "momentum_threshold": inputs.get("momentum_threshold"),
    }
    return [_safe_float(rec.get(name)) for name in FEATURE_NAMES]


def score_candidate(model: dict, result: dict) -> tuple[float, float]:
    """
    Compute model score (logit) and probability for a directional candidate.
    Returns (score, probability). score = sum(w*x) + bias; probability = sigmoid(score).
    """
    x = _extract_features_from_result(result)
    w = model["weights"]
    b = model["bias"]
    score = sum(w[j] * x[j] for j in range(len(x))) + b
    prob = _sigmoid(score)
    return score, prob
