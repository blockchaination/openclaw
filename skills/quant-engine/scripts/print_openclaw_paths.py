#!/usr/bin/env python3
"""
Print fully resolved absolute paths for OpenClaw quant-engine ML/AI files.

Use for diagnostics on VPS to verify path wiring.
"""

from __future__ import annotations

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


def main() -> int:
    root = repo_root()
    print("repo_root:              ", root.resolve())
    print("training_examples_path:  ", default_training_examples_path().resolve())
    print("shadow_inference_path:  ", default_shadow_inference_path().resolve())
    print("model_artifact_path:    ", default_model_artifact_path().resolve())
    print("signal_outcomes_path:   ", default_signal_outcomes_path().resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
