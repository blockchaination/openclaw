#!/usr/bin/env python3
"""
Outcome labeling for Trade Outcome Intelligence Layer.

Labels historical decision snapshots with forward returns at configurable horizons.
Idempotent: records with label_completed_at are skipped.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from price_history import get_price_at_or_after


def _parse_ts(ts_str: str) -> datetime.datetime | None:
    if not ts_str:
        return None
    s = ts_str.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _decision_key(rec: dict) -> str:
    return f"{rec.get('timestamp_utc','')}|{rec.get('pair','')}|{rec.get('iteration',0)}"


def _load_labeled_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("label_completed_at"):
                    keys.add(_decision_key(rec))
    except OSError:
        pass
    return keys


def label_pending_decisions(
    decision_path: Path,
    labeled_path: Path,
    price_history_path: Path,
    horizons_minutes: tuple[int, ...] = (5, 15, 60),
) -> int:
    """
    Label unlabeled decisions with forward returns. Returns count of newly labeled.
    Idempotent: skips records already in labeled_path.
    """
    labeled_keys = _load_labeled_keys(labeled_path)
    if not decision_path.exists():
        return 0

    count = 0
    try:
        with decision_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _decision_key(rec) in labeled_keys:
                    continue
                rec_ts = _parse_ts(rec.get("timestamp_utc", ""))
                if rec_ts is None:
                    continue
                if rec_ts.tzinfo is None:
                    rec_ts = rec_ts.replace(tzinfo=datetime.timezone.utc)
                pair = rec.get("pair", "")
                entry_price = rec.get("mid_price")
                if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                    continue

                outcome: dict = dict(rec)
                direction = rec.get("signal_direction", rec.get("decision_action", "buy"))
                for h in horizons_minutes:
                    target_ts = rec_ts + datetime.timedelta(minutes=h)
                    fp = get_price_at_or_after(price_history_path, pair, target_ts)
                    if fp is not None:
                        if direction == "sell":
                            ret = (entry_price - fp) / entry_price
                        else:
                            ret = (fp - entry_price) / entry_price
                        outcome[f"future_price_{h}m"] = fp
                        outcome[f"return_{h}m"] = round(ret, 6)
                    else:
                        outcome[f"future_price_{h}m"] = None
                        outcome[f"return_{h}m"] = None

                outcome["max_favorable_excursion"] = None
                outcome["max_adverse_excursion"] = None
                outcome["label_completed_at"] = (
                    datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                )

                labeled_path.parent.mkdir(parents=True, exist_ok=True)
                with labeled_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(outcome, separators=(",", ":")) + "\n")
                labeled_keys.add(_decision_key(rec))
                count += 1
    except (OSError, TypeError, ValueError):
        pass
    return count
