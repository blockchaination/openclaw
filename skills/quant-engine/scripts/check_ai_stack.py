#!/usr/bin/env python3
"""
Check OpenClaw AI stack health: paths, file existence, row counts, model metrics.

Use for VPS diagnostics and operator verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from paths import (
    default_model_artifact_path,
    default_shadow_inference_path,
    default_signal_outcomes_path,
    default_training_examples_path,
    repo_root,
)


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty JSONL lines."""
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _count_completed_training(path: Path) -> int:
    """Count training rows with label_300s present."""
    if not path.exists():
        return 0
    count = 0
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        if "\x00" in text[: min(200, len(text))]:
            try:
                text = raw.decode("utf-16-le")
            except Exception:
                pass
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("label_300s") is not None:
                count += 1
    except OSError:
        pass
    return count


def main() -> int:
    root = repo_root().resolve()
    training_path = default_training_examples_path().resolve()
    shadow_path = default_shadow_inference_path().resolve()
    model_path = default_model_artifact_path().resolve()

    print("OpenClaw AI Stack Check")
    print("=======================")
    print()
    print("Resolved paths:")
    print(f"  repo_root:              {root}")
    print(f"  training_examples:      {training_path}")
    print(f"  shadow_inference:       {shadow_path}")
    print(f"  model_artifact:        {model_path}")
    print()

    print("File existence:")
    print(f"  logs/training_examples.jsonl:  {'exists' if training_path.exists() else 'MISSING'}")
    print(f"  logs/shadow_inference.jsonl:  {'exists' if shadow_path.exists() else 'MISSING'}")
    print(f"  artifacts/first_model_metrics.json: {'exists' if model_path.exists() else 'MISSING'}")
    print()

    completed = _count_completed_training(training_path)
    shadow_rows = _count_jsonl_lines(shadow_path)
    print("Row counts:")
    print(f"  completed training examples: {completed}")
    print(f"  shadow inference rows:      {shadow_rows}")
    print()

    if model_path.exists():
        try:
            with model_path.open(encoding="utf-8") as f:
                data = json.load(f)
            metrics = data.get("metrics") or {}
            print("Model metrics (from artifact):")
            acc = metrics.get("accuracy")
            prec = metrics.get("precision")
            rec = metrics.get("recall")
            print(f"  accuracy:   {acc:.4f}" if isinstance(acc, (int, float)) else "  accuracy:   -")
            print(f"  precision:  {prec:.4f}" if isinstance(prec, (int, float)) else "  precision:  -")
            print(f"  recall:     {rec:.4f}" if isinstance(rec, (int, float)) else "  recall:     -")
            thresh = data.get("recommended_shadow_threshold")
            if thresh is not None:
                print(f"  recommended_shadow_threshold: {thresh}")
            print()
        except (json.JSONDecodeError, OSError):
            print("Model metrics: (could not read artifact)")
            print()
    else:
        print("Model metrics: (artifact missing)")
        print()

    live_ready = (
        model_path.exists()
        and shadow_path.exists()
        and completed >= 2
    )
    print(f"Shadow inference live-ready: {'yes' if live_ready else 'no'}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
