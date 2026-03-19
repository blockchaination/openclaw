#!/usr/bin/env python3
"""
Config for expectancy gate and outcome labeling.

Uses env vars. Defaults are conservative.
"""

from __future__ import annotations

import os


def _bool_env(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _float_env(key: str, default: float) -> float:
    try:
        v = os.environ.get(key, "").strip()
        if v:
            return float(v)
    except (ValueError, TypeError):
        pass
    return default


def _int_env(key: str, default: int) -> int:
    try:
        v = os.environ.get(key, "").strip()
        if v:
            return int(v)
    except (ValueError, TypeError):
        pass
    return default


def _str_env(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip().lower()
    return v if v else default


EXPECTANCY_GATE_ENABLED = _bool_env("EXPECTANCY_GATE_ENABLED", False)
EXPECTANCY_GATE_MODE = _str_env("EXPECTANCY_GATE_MODE", "shadow")
if EXPECTANCY_GATE_MODE not in ("off", "shadow", "active"):
    EXPECTANCY_GATE_MODE = "shadow"
EXPECTANCY_MIN_SAMPLES = _int_env("EXPECTANCY_MIN_SAMPLES", 20)
EXPECTANCY_MIN_MEAN_RETURN_15M = _float_env("EXPECTANCY_MIN_MEAN_RETURN_15M", 0.001)
EXPECTANCY_MIN_WIN_RATE = _float_env("EXPECTANCY_MIN_WIN_RATE", 0.55)
EXPECTANCY_ALLOW_IF_INSUFFICIENT_DATA = _bool_env("EXPECTANCY_ALLOW_IF_INSUFFICIENT_DATA", False)


def _horizons_env() -> tuple[int, ...]:
    v = os.environ.get("OUTCOME_LABEL_HORIZONS_MINUTES", "5,15,60").strip()
    if not v:
        return (5, 15, 60)
    out: list[int] = []
    for part in v.split(","):
        try:
            out.append(int(part.strip()))
        except (ValueError, TypeError):
            pass
    return tuple(out) if out else (5, 15, 60)


OUTCOME_LABEL_HORIZONS_MINUTES = _horizons_env()
