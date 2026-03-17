#!/usr/bin/env python3
"""
Lightweight checkpoint: count signals in signal_outcomes.jsonl.

Run: python analyze_signal_strengths.py [--file PATH]

Use to compare pre/post threshold runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return repo root (parent of skills/)."""
    return Path(__file__).resolve().parents[3]


def _default_path() -> Path:
    return _repo_root() / "logs" / "signal_outcomes.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count total, buy, and sell signals in signal_outcomes.jsonl.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help=f"Path to signal_outcomes JSONL (default: {_default_path()})",
    )
    args = parser.parse_args()
    path = args.file or _default_path()

    total = 0
    buy_count = 0
    sell_count = 0

    if not path.exists():
        print(f"total signals: {total}")
        print(f"buy signals:  {buy_count}")
        print(f"sell signals: {sell_count}")
        return 0

    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sig = rec.get("signal")
                if sig == "buy":
                    buy_count += 1
                    total += 1
                elif sig == "sell":
                    sell_count += 1
                    total += 1
    except OSError:
        pass

    print(f"total signals: {total}")
    print(f"buy signals:  {buy_count}")
    print(f"sell signals: {sell_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
