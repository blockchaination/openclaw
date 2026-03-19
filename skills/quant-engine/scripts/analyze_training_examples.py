#!/usr/bin/env python3
"""
Analyze training_examples.jsonl for dataset quality before model training.

Run: python analyze_training_examples.py [--file PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paths import default_training_examples_path


def _load_completed_records(path: Path) -> list[dict]:
    """Load JSONL, return only records with price_after_300s and label_300s."""
    if not path.exists():
        return []
    records: list[dict] = []
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
                if rec.get("price_after_300s") is None:
                    continue
                if rec.get("label_300s") is None:
                    continue
                records.append(rec)
    except OSError:
        pass
    return records


_RUNTIME_ONLY_REASONS = frozenset({
    "live_mode_blocked",
    "live_order_cooldown_active",
    "buy_cooldown_active",
    "sell_cooldown_active",
    "risk_blocked",
    "buy_suppressed_low_usd",
    "sell_suppressed_low_inventory",
    "no_inventory_to_sell",
})


def _get_candidate_reason(rec: dict) -> str:
    """Extract candidate_reason. Never use runtime-only values for candidate semantics."""
    cr = rec.get("candidate_reason")
    if cr and isinstance(cr, str) and cr.strip():
        return cr.strip()
    dr = rec.get("decision_reason")
    if dr and isinstance(dr, str) and dr.strip() and dr.strip() not in _RUNTIME_ONLY_REASONS:
        return dr.strip()
    return "-"


def _get_runtime_reason(rec: dict) -> str:
    """Extract runtime_reason with fallback for older rows."""
    rr = rec.get("runtime_reason")
    if rr and isinstance(rr, str) and rr.strip():
        return rr.strip()
    return rec.get("decision_reason") or "-"


def _compute_metrics(records: list[dict]) -> dict:
    """Compute summary metrics from completed training examples."""
    if not records:
        return {
            "total": 0,
            "buy": 0,
            "sell": 0,
            "flat": 0,
            "long": 0,
            "buy_in_flat": 0,
            "buy_in_long": 0,
            "sell_in_flat": 0,
            "sell_in_long": 0,
            "avg_signal_strength_buy": 0.0,
            "avg_signal_strength_sell": 0.0,
            "label_30s_positive_rate": 0.0,
            "label_60s_positive_rate": 0.0,
            "label_300s_positive_rate": 0.0,
            "candidate_reason_counts": {},
            "runtime_reason_counts": {},
            "v1_training_count": 0,
        }
    buy_recs = [r for r in records if (r.get("candidate_side") or "").lower() == "buy"]
    sell_recs = [r for r in records if (r.get("candidate_side") or "").lower() == "sell"]
    flat_recs = [r for r in records if (r.get("spot_state") or "").upper() == "FLAT"]
    long_recs = [r for r in records if (r.get("spot_state") or "").upper() == "LONG"]

    buy_in_flat = sum(1 for r in records if (r.get("candidate_side") or "").lower() == "buy" and (r.get("spot_state") or "").upper() == "FLAT")
    buy_in_long = sum(1 for r in records if (r.get("candidate_side") or "").lower() == "buy" and (r.get("spot_state") or "").upper() == "LONG")
    sell_in_flat = sum(1 for r in records if (r.get("candidate_side") or "").lower() == "sell" and (r.get("spot_state") or "").upper() == "FLAT")
    sell_in_long = sum(1 for r in records if (r.get("candidate_side") or "").lower() == "sell" and (r.get("spot_state") or "").upper() == "LONG")

    def _safe_float(x: object) -> float | None:
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    buy_strengths = [_safe_float(r.get("signal_strength")) for r in buy_recs]
    buy_strengths = [s for s in buy_strengths if s is not None]
    avg_signal_buy = sum(buy_strengths) / len(buy_strengths) if buy_strengths else 0.0

    sell_strengths = [_safe_float(r.get("signal_strength")) for r in sell_recs]
    sell_strengths = [s for s in sell_strengths if s is not None]
    avg_signal_sell = sum(sell_strengths) / len(sell_strengths) if sell_strengths else 0.0

    labels_30 = [r.get("label_30s") for r in records if r.get("label_30s") is not None]
    labels_60 = [r.get("label_60s") for r in records if r.get("label_60s") is not None]
    labels_300 = [r.get("label_300s") for r in records if r.get("label_300s") is not None]

    pos_30 = sum(1 for L in labels_30 if L == 1)
    pos_60 = sum(1 for L in labels_60 if L == 1)
    pos_300 = sum(1 for L in labels_300 if L == 1)

    rate_30 = (pos_30 / len(labels_30) * 100.0) if labels_30 else 0.0
    rate_60 = (pos_60 / len(labels_60) * 100.0) if labels_60 else 0.0
    rate_300 = (pos_300 / len(labels_300) * 100.0) if labels_300 else 0.0

    candidate_reason_counts: dict[str, int] = {}
    runtime_reason_counts: dict[str, int] = {}
    for r in records:
        cr = _get_candidate_reason(r)
        candidate_reason_counts[cr] = candidate_reason_counts.get(cr, 0) + 1
        rr = _get_runtime_reason(r)
        runtime_reason_counts[rr] = runtime_reason_counts.get(rr, 0) + 1

    valid_candidate_reasons = frozenset({
        "buy mean-reversion entry",
        "sell mean-reversion exit",
        "weak_signal_filtered",
    })
    v1_training_count = sum(
        1
        for r in records
        if (r.get("candidate_side") or "").lower() == "buy"
        and (r.get("spot_state") or "").upper() == "FLAT"
        and _get_candidate_reason(r) in valid_candidate_reasons
        and r.get("label_300s") is not None
    )

    return {
        "total": len(records),
        "buy": len(buy_recs),
        "sell": len(sell_recs),
        "flat": len(flat_recs),
        "long": len(long_recs),
        "buy_in_flat": buy_in_flat,
        "buy_in_long": buy_in_long,
        "sell_in_flat": sell_in_flat,
        "sell_in_long": sell_in_long,
        "avg_signal_strength_buy": avg_signal_buy,
        "avg_signal_strength_sell": avg_signal_sell,
        "label_30s_positive_rate": rate_30,
        "label_60s_positive_rate": rate_60,
        "label_300s_positive_rate": rate_300,
        "candidate_reason_counts": candidate_reason_counts,
        "runtime_reason_counts": runtime_reason_counts,
        "v1_training_count": v1_training_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze training_examples.jsonl for dataset quality.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help=f"Path to training examples JSONL (default: {default_training_examples_path()})",
    )
    args = parser.parse_args()
    path = args.file or default_training_examples_path()

    records = _load_completed_records(path)
    m = _compute_metrics(records)

    print("OpenClaw Training Examples Analysis")
    print("-----------------------------------")
    print()
    print("Completed examples:")
    print(f"  total:  {m['total']}")
    print(f"  buy:   {m['buy']}")
    print(f"  sell:  {m['sell']}")
    print(f"  FLAT:  {m['flat']}")
    print(f"  LONG:  {m['long']}")
    print()
    print("Side / state breakdown:")
    print(f"  buy in FLAT:  {m['buy_in_flat']}")
    print(f"  buy in LONG:  {m['buy_in_long']}")
    print(f"  sell in FLAT: {m['sell_in_flat']}")
    print(f"  sell in LONG: {m['sell_in_long']}")
    print()
    print("Average signal strength:")
    print(f"  buy:   {m['avg_signal_strength_buy']:.4f}")
    print(f"  sell:  {m['avg_signal_strength_sell']:.4f}")
    print()
    print("Label positive rate:")
    print(f"  30s:  {m['label_30s_positive_rate']:.1f}%")
    print(f"  60s:  {m['label_60s_positive_rate']:.1f}%")
    print(f"  300s: {m['label_300s_positive_rate']:.1f}%")
    print()
    cr_counts = m.get("candidate_reason_counts") or {}
    if cr_counts:
        print("Candidate reason breakdown:")
        for k, v in sorted(cr_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        print()
    rr_counts = m.get("runtime_reason_counts") or {}
    if rr_counts:
        print("Runtime reason breakdown:")
        for k, v in sorted(rr_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        print()
    v1 = m.get("v1_training_count", 0)
    print(f"Completed examples used for v1 training (buy+FLAT+valid candidate_reason+label_300s): {v1}")
    print()
    print(f"file: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
