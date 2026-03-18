#!/usr/bin/env python3
"""
Paper-trading quant engine orchestrator.

One-shot execution: fetch market data, compute features, run strategy,
check risk, simulate order placement. No live trading, no persistence.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from features import spread, mid_price, book_imbalance, short_momentum, volatility
from paper_broker import PaperBroker
from paths import (
    default_log_path,
    default_shadow_inference_path,
    default_signal_outcomes_path,
    default_status_path,
    default_trade_events_path,
    default_training_examples_path,
    ensure_logs_and_artifacts_dirs,
    repo_root,
)
from risk import allow_trade
from shadow_model import load_model as load_shadow_model
from shadow_model import score_candidate as shadow_score_candidate
from strategy import maker_first_mean_reversion

MAX_LIVE_ORDER_USD = 10
MIN_USD_TO_BUY = 10.0
MIN_XBT_TO_SELL = 0.0002
MIN_SECONDS_BETWEEN_SAME_SIDE_ACTIONS = 300


def _repo_root() -> Path:
    """Return repo root (parent of skills/). Kept for backward compat; use paths.repo_root()."""
    return repo_root()


def append_training_example(path: Path, record: dict) -> None:
    """Append one training example row as JSONL line. Create parent dirs if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        raise RuntimeError(f"Failed to append training example to {path}: {e}") from e


def append_signal_outcome(path: Path, record: dict) -> None:
    """Append one signal outcome record as JSONL line. Create parent dirs if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        raise RuntimeError(f"Failed to append signal outcome to {path}: {e}") from e


def append_jsonl(path: Path, row: dict) -> None:
    """
    Append one JSON object as a single line to a JSONL file.
    Create parent directories if needed.
    Raise RuntimeError on write failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        raise RuntimeError(f"Failed to append to log file {path}: {e}") from e


def write_status(path: Path, status: dict) -> None:
    """Overwrite status file with compact JSON. Create parent dirs if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(status, f, separators=(",", ":"))
    except OSError as e:
        raise RuntimeError(f"Failed to write status file {path}: {e}") from e


def append_trade_event(path: Path, event: dict) -> None:
    """Append one trade event as JSONL line. Create parent dirs if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        raise RuntimeError(f"Failed to append trade event to {path}: {e}") from e


def _last_same_side_action_timestamp(path: Path, side: str) -> datetime.datetime | None:
    """
    Return timestamp of most recent same-side actionability event from trade_events.jsonl.
    side: "buy" or "sell".
    Events that count: live_order_submitted, forced_live_test_buy_submitted (buy),
    buy_suppressed_low_usd (buy), sell_suppressed_low_inventory (sell).
    """
    if not path.exists():
        return None
    buy_events = ("live_order_submitted", "forced_live_test_buy_submitted", "buy_suppressed_low_usd")
    sell_events = ("live_order_submitted", "sell_suppressed_low_inventory")
    target = buy_events if side == "buy" else sell_events
    last_ts: datetime.datetime | None = None
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("event_type", "")
                if etype not in target:
                    continue
                if etype == "live_order_submitted" and ev.get("side") != side:
                    continue
                ts_str = ev.get("timestamp", "")
                if not ts_str:
                    continue
                ts_str = ts_str.replace("Z", "+00:00")
                try:
                    ts = datetime.datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    continue
                if last_ts is None or (ts > last_ts):
                    last_ts = ts
    except OSError:
        return None
    return last_ts


def _kraken_reader_path() -> Path:
    """Resolve path to kraken_reader.py relative to this file."""
    this_dir = Path(__file__).resolve().parent
    return this_dir.parent.parent / "kraken-reader" / "scripts" / "kraken_reader.py"


def _fetch_live_account_snapshot(pair: str) -> dict | None:
    """
    Fetch Kraken balance snapshot for live mode.
    Returns {"usd": float, "xbt": float} or None on failure.
    Uses Kraken asset names: XXBT, ZUSD.
    """
    try:
        kraken_auth = _kraken_auth_module()
        api_key, api_secret = kraken_auth.load_credentials()
        data = kraken_auth.get_account_balance(api_key, api_secret)
        result = data.get("result") or {}
        if not isinstance(result, dict):
            return None
        usd = _safe_float(result.get("ZUSD", 0))
        xbt = _safe_float(result.get("XXBT", 0))
        return {"usd": usd, "xbt": xbt}
    except RuntimeError:
        return None


def _kraken_auth_module():
    """Load kraken_auth from skills/kraken-reader/scripts. Returns the module."""
    import importlib.util
    this_dir = Path(__file__).resolve().parent
    auth_path = (this_dir.parent.parent / "kraken-reader" / "scripts").resolve()
    spec = importlib.util.spec_from_file_location(
        "kraken_auth", auth_path / "kraken_auth.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load kraken_auth from {auth_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kraken_auth"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_kraken_add_order_check(
    pair: str,
    order_side: str,
    order_type: str,
    order_volume: str,
    order_price: str | None,
    status_path: Path,
    trade_events_path: Path,
) -> None:
    """Run Kraken AddOrder validation check (validate=True), emit events, exit."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    append_trade_event(
        trade_events_path,
        {
            "timestamp": ts,
            "pair": pair,
            "event_type": "kraken_add_order_check_started",
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "side": order_side,
            "order_type": order_type,
            "volume": order_volume,
            "price": order_price,
        },
    )
    try:
        kraken_auth = _kraken_auth_module()
        api_key, api_secret = kraken_auth.load_credentials()
        price_arg: str | float | None = float(order_price) if order_price else None
        kraken_auth.add_order(
            api_key,
            api_secret,
            pair=pair,
            side=order_side,
            ordertype=order_type,
            volume=order_volume,
            price=price_arg,
            validate=True,
        )
        append_trade_event(
            trade_events_path,
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "pair": pair,
                "event_type": "kraken_add_order_check_succeeded",
                "runtime_mode": "paper",
                "execution_mode": "taker",
                "side": order_side,
                "order_type": order_type,
            },
        )
        status = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pair": pair,
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "iteration": 0,
            "kill_switch_active": False,
            "last_signal": "hold",
            "last_action": "hold",
            "last_mid_price": 0.0,
            "position_usd": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_orders": 0,
            "risk_ok": True,
            "shutdown_reason": "kraken_add_order_check",
            "error": None,
            "kraken_add_order_check_ok": True,
        }
        write_status(status_path, status)
        print(
            f"Kraken add-order check OK. pair={pair} side={order_side} type={order_type} "
            f"volume={order_volume} validate=true"
        )
    except RuntimeError as e:
        err_msg = str(e)
        append_trade_event(
            trade_events_path,
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "pair": pair,
                "event_type": "kraken_add_order_check_failed",
                "runtime_mode": "paper",
                "execution_mode": "taker",
                "reason": err_msg,
            },
        )
        status = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pair": pair,
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "iteration": 0,
            "kill_switch_active": False,
            "last_signal": "hold",
            "last_action": "hold",
            "last_mid_price": 0.0,
            "position_usd": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_orders": 0,
            "risk_ok": True,
            "shutdown_reason": "kraken_add_order_check",
            "error": err_msg,
            "kraken_add_order_check_ok": False,
        }
        write_status(status_path, status)
        print(f"Kraken add-order check: FAILED ({err_msg})", file=sys.stderr)
        sys.exit(1)


def _run_kraken_cancel_order_check(
    pair: str,
    cancel_txid: str,
    status_path: Path,
    trade_events_path: Path,
) -> None:
    """Run Kraken CancelOrder check, emit events, exit."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    append_trade_event(
        trade_events_path,
        {
            "timestamp": ts,
            "pair": pair,
            "event_type": "kraken_cancel_order_check_started",
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "txid": cancel_txid,
        },
    )
    print(
        "Warning: cancel-order check calls Kraken CancelOrder on a real txid.",
        file=sys.stderr,
    )
    try:
        kraken_auth = _kraken_auth_module()
        api_key, api_secret = kraken_auth.load_credentials()
        kraken_auth.cancel_order(api_key, api_secret, cancel_txid)
        append_trade_event(
            trade_events_path,
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "pair": pair,
                "event_type": "kraken_cancel_order_check_succeeded",
                "runtime_mode": "paper",
                "execution_mode": "taker",
                "txid": cancel_txid,
            },
        )
        status = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pair": pair,
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "iteration": 0,
            "kill_switch_active": False,
            "last_signal": "hold",
            "last_action": "hold",
            "last_mid_price": 0.0,
            "position_usd": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_orders": 0,
            "risk_ok": True,
            "shutdown_reason": "kraken_cancel_order_check",
            "error": None,
            "kraken_cancel_order_check_ok": True,
        }
        write_status(status_path, status)
        print(f"Kraken cancel-order check OK for txid={cancel_txid}")
    except RuntimeError as e:
        err_msg = str(e)
        append_trade_event(
            trade_events_path,
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "pair": pair,
                "event_type": "kraken_cancel_order_check_failed",
                "runtime_mode": "paper",
                "execution_mode": "taker",
                "reason": err_msg,
                "txid": cancel_txid,
            },
        )
        status = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pair": pair,
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "iteration": 0,
            "kill_switch_active": False,
            "last_signal": "hold",
            "last_action": "hold",
            "last_mid_price": 0.0,
            "position_usd": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_orders": 0,
            "risk_ok": True,
            "shutdown_reason": "kraken_cancel_order_check",
            "error": err_msg,
            "kraken_cancel_order_check_ok": False,
        }
        write_status(status_path, status)
        print(f"Kraken cancel-order check: FAILED ({err_msg})", file=sys.stderr)
        sys.exit(1)


def _run_kraken_auth_check(
    pair: str,
    status_path: Path,
    trade_events_path: Path,
) -> None:
    """Run Kraken auth/balance check, emit events, write status, exit."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "timestamp": ts,
        "pair": pair,
        "event_type": "kraken_auth_check_started",
    }
    append_trade_event(trade_events_path, base)

    try:
        kraken_auth = _kraken_auth_module()
        api_key, api_secret = kraken_auth.load_credentials()
        data = kraken_auth.get_account_balance(api_key, api_secret)
        result = data.get("result") or {}
        balance_count = len(result) if isinstance(result, dict) else 0
        append_trade_event(
            trade_events_path,
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "pair": pair,
                "event_type": "kraken_auth_check_succeeded",
                "balance_assets": balance_count,
            },
        )
        status = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pair": pair,
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "iteration": 0,
            "kill_switch_active": False,
            "last_signal": "hold",
            "last_action": "hold",
            "last_mid_price": 0.0,
            "position_usd": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_orders": 0,
            "risk_ok": True,
            "shutdown_reason": "kraken_auth_check",
            "error": None,
            "kraken_auth_ok": True,
        }
        write_status(status_path, status)
        print("Kraken auth check: OK (balance endpoint succeeded)")
    except RuntimeError as e:
        err_msg = str(e)
        append_trade_event(
            trade_events_path,
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "pair": pair,
                "event_type": "kraken_auth_check_failed",
                "reason": err_msg,
            },
        )
        status = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pair": pair,
            "runtime_mode": "paper",
            "execution_mode": "taker",
            "iteration": 0,
            "kill_switch_active": False,
            "last_signal": "hold",
            "last_action": "hold",
            "last_mid_price": 0.0,
            "position_usd": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "open_orders": 0,
            "risk_ok": True,
            "shutdown_reason": "kraken_auth_check",
            "error": err_msg,
            "kraken_auth_ok": False,
        }
        write_status(status_path, status)
        print(f"Kraken auth check: FAILED ({err_msg})", file=sys.stderr)
        sys.exit(1)


def run_kraken_reader(command: list[str]) -> dict:
    """
    Run kraken_reader.py via subprocess and parse stdout as JSON.

    Raises RuntimeError with stderr/stdout context if subprocess fails
    or if JSON parsing fails.
    """
    kraken_path = _kraken_reader_path()
    if not kraken_path.exists():
        raise RuntimeError(f"kraken_reader.py not found at {kraken_path}")

    cmd = [sys.executable, str(kraken_path)] + command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"kraken_reader failed (exit {result.returncode}): "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"kraken_reader output is not valid JSON: {e}. stdout={result.stdout[:500]!r}"
        ) from e


def _fetch_mid_price(pair: str) -> float | None:
    """Fetch current mid price from ticker. Returns None on failure."""
    try:
        ticker = run_kraken_reader(["--format", "json", "ticker", "--pair", pair])
        bid = _safe_float(ticker.get("bid", 0))
        ask = _safe_float(ticker.get("ask", 0))
        if bid <= 0 or ask <= 0:
            return None
        return (bid + ask) / 2.0
    except (RuntimeError, Exception):
        return None


def _safe_float(val: object, default: float = 0.0) -> float:
    """Parse value to float; return default if invalid."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    return default


def _normalize_bids_asks(raw_bids: object, raw_asks: object) -> tuple[list[dict], list[dict]]:
    """Normalize orderbook bids/asks into list of dicts with price, volume, timestamp."""

    def _parse_levels(raw: object) -> list[dict]:
        out: list[dict] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if isinstance(item, dict):
                out.append({
                    "price": str(item.get("price", "")),
                    "volume": str(item.get("volume", "")),
                    "timestamp": str(item.get("timestamp", "")),
                })
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append({
                    "price": str(item[0]),
                    "volume": str(item[1]),
                    "timestamp": str(item[2]) if len(item) > 2 else "",
                })
        return out

    return _parse_levels(raw_bids), _parse_levels(raw_asks)


def _extract_prices_from_trades(trades_data: object, last_price: float) -> list[float]:
    """Extract recent trade prices into list[float]."""
    prices: list[float] = []

    if isinstance(trades_data, dict) and "trades" in trades_data:
        raw_trades = trades_data.get("trades", [])
        if isinstance(raw_trades, list):
            for t in raw_trades:
                if isinstance(t, (list, tuple)) and len(t) >= 1:
                    p = _safe_float(t[0])
                    if p > 0:
                        prices.append(p)
                elif isinstance(t, dict) and "price" in t:
                    p = _safe_float(t.get("price"))
                    if p > 0:
                        prices.append(p)

    if not prices:
        if last_price > 0:
            prices = [last_price]
        else:
            prices = []

    return prices


def _is_directional_candidate(result: dict) -> bool:
    """True if result represents a real directional candidate (buy, sell, or weak_signal_filtered)."""
    action = result.get("strategy", {}).get("action", "hold")
    reason = result.get("decision_reason", "")
    return action in ("buy", "sell") or reason == "weak_signal_filtered"


def _apply_shadow_inference(
    result: dict,
    model: dict | None,
    shadow_path: Path,
) -> None:
    """
    For directional candidates only: compute model score and probability,
    add to result, append compact log row. Does nothing for neutral hold.
    """
    if model is None or not _is_directional_candidate(result):
        return
    score, prob = shadow_score_candidate(model, result)
    result["model_score"] = round(score, 6)
    result["model_probability"] = round(prob, 6)
    strategy_inputs = result.get("strategy", {}).get("inputs", {})
    spot_state_raw = strategy_inputs.get("spot_state", "flat")
    spot_state = "FLAT" if spot_state_raw == "flat" else "LONG"
    row: dict = {
        "ts": result.get("timestamp_utc", ""),
        "iter": result.get("iteration"),
        "candidate_side": result.get("candidate_side", ""),
        "candidate_reason": result.get("candidate_reason", ""),
        "runtime_reason": result.get("runtime_reason", ""),
        "score": round(score, 4),
        "prob": round(prob, 4),
        "spot_state": spot_state,
        "signal_strength": result.get("signal_strength"),
    }
    threshold = model.get("recommended_shadow_threshold")
    if threshold is not None:
        row["threshold"] = threshold
    append_jsonl(shadow_path, row)


def _compute_training_labels(
    price_at_signal: float,
    candidate_side: str,
    price_after_30s: float | None,
    price_after_60s: float | None,
    price_after_300s: float | None,
) -> dict[str, int | None]:
    """Compute label_30s, label_60s, label_300s for a training example."""
    out: dict[str, int | None] = {}
    for delta, price in [(30, price_after_30s), (60, price_after_60s), (300, price_after_300s)]:
        if price is not None and price_at_signal > 0:
            if candidate_side == "buy":
                out[f"label_{delta}s"] = 1 if price > price_at_signal else 0
            else:
                out[f"label_{delta}s"] = 1 if price < price_at_signal else 0
        else:
            out[f"label_{delta}s"] = None
    return out


def _resolve_signal_strength(decision: dict) -> int | float | None:
    """Enforce signal_strength contract: None for non-directional hold, else pass through."""
    if decision.get("action") == "hold" and decision.get("reason") != "weak_signal_filtered":
        return None
    return decision.get("signal_strength")


def _run_one_cycle(
    pair: str,
    usd_order_size: float,
    depth: int,
    execution_mode: str,
    runtime_mode: str,
    broker: PaperBroker,
    iteration: int,
    enable_live_orders: bool = False,
    live_account: dict | None = None,
    trade_events_path: Path | None = None,
) -> dict:
    """Run one paper-trading cycle. Returns the result dict."""
    ticker = run_kraken_reader(["--format", "json", "ticker", "--pair", pair])
    bid_str = ticker.get("bid", "")
    ask_str = ticker.get("ask", "")
    last_str = ticker.get("last", "")

    best_bid = _safe_float(bid_str)
    best_ask = _safe_float(ask_str)
    last_price = _safe_float(last_str)

    if best_bid <= 0 or best_ask <= 0:
        raise RuntimeError(
            f"Missing or invalid bid/ask from ticker: bid={bid_str!r} ask={ask_str!r}"
        )

    orderbook = run_kraken_reader(["--format", "json", "orderbook", "--pair", pair])
    raw_bids = orderbook.get("bids", [])
    raw_asks = orderbook.get("asks", [])

    bids, asks = _normalize_bids_asks(raw_bids, raw_asks)
    if not bids or not asks:
        raise RuntimeError("Orderbook missing bids or asks")

    try:
        trades_data = run_kraken_reader(["trades", "--format", "json", "--pair", pair])
    except RuntimeError:
        trades_data = {}

    prices = _extract_prices_from_trades(trades_data, last_price)
    if not prices:
        prices = [last_price]

    lookback = min(5, len(prices) - 1) if len(prices) > 1 else 0
    current_spread = spread(bids, asks)
    current_mid_price = mid_price(bids, asks)
    current_book_imbalance = book_imbalance(bids, asks, depth=depth)
    current_momentum = short_momentum(prices, lookback) if lookback >= 1 else 0.0
    current_volatility = volatility(prices, lookback) if lookback >= 2 else 0.0

    xbt_inventory = (
        (live_account.get("xbt", 0) or 0)
        if runtime_mode == "live" and live_account is not None
        else broker.position_units
    )

    decision = maker_first_mean_reversion(
        mid_price=current_mid_price,
        spread=current_spread,
        book_imbalance=current_book_imbalance,
        momentum=current_momentum,
        volatility=current_volatility,
        xbt_inventory=xbt_inventory,
        min_xbt_to_sell=MIN_XBT_TO_SELL,
    )

    action = decision.get("action", "hold")
    if action == "buy":
        proposed_order_usd = usd_order_size
    elif action == "sell":
        proposed_order_usd = -usd_order_size
    else:
        proposed_order_usd = 0.0

    risk_result = allow_trade(
        current_position_usd=broker.position_units * current_mid_price,
        proposed_order_usd=proposed_order_usd,
        daily_pnl_usd=broker.realized_pnl_usd,
        open_orders_count=len(broker.open_orders),
    )

    order_submitted = False
    order_details: dict | None = None
    fills: list[dict] = []
    skipped_reason: str | None = None
    live_mode_blocked = False
    live_order_ready: dict | None = None

    if action in ("buy", "sell"):
        if runtime_mode == "live" and live_account is not None:
            usd = live_account.get("usd", 0) or 0
            xbt = live_account.get("xbt", 0) or 0
            buy_eligible = usd >= MIN_USD_TO_BUY
            sell_eligible = xbt >= MIN_XBT_TO_SELL
            if action == "buy" and not buy_eligible:
                skipped_reason = "buy_suppressed_low_usd"
            elif action == "sell" and not sell_eligible:
                skipped_reason = "sell_suppressed_low_inventory"
        else:
            sell_eligible = broker.position_units > 0
            if action == "sell" and not sell_eligible:
                skipped_reason = "no_inventory_to_sell"
        if skipped_reason is None and runtime_mode == "live" and trade_events_path is not None:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            last_ts = _last_same_side_action_timestamp(trade_events_path, action)
            if last_ts is not None:
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=datetime.timezone.utc)
                elapsed = (now_utc - last_ts).total_seconds()
                if elapsed < MIN_SECONDS_BETWEEN_SAME_SIDE_ACTIONS:
                    skipped_reason = (
                        "buy_cooldown_active" if action == "buy" else "sell_cooldown_active"
                    )
        if skipped_reason is None and not risk_result.get("allowed"):
            skipped_reason = "risk_blocked"
        elif skipped_reason is None and runtime_mode == "live" and not enable_live_orders:
            live_mode_blocked = True
        elif skipped_reason is None and runtime_mode == "live" and enable_live_orders:
            ordertype = "limit" if execution_mode == "maker" else "market"
            if ordertype == "limit":
                price = best_bid if action == "buy" else best_ask
            else:
                price = None
            size_units = usd_order_size / current_mid_price
            live_order_ready = {
                "side": action,
                "ordertype": ordertype,
                "volume": size_units,
                "price": price,
                "size_usd": usd_order_size,
            }
        elif skipped_reason is None:
            size_units = usd_order_size / current_mid_price
            if execution_mode == "taker":
                if action == "buy":
                    price = best_ask
                else:
                    price = best_bid
            else:
                if action == "buy":
                    price = best_bid
                else:
                    price = best_ask

            place_result = broker.place_limit_order(action, price, size_units)
            if place_result.get("status") == "open":
                order_submitted = True
                order_details = place_result.get("order")
                fills = broker.process_market_tick(bid=best_bid, ask=best_ask)

                if execution_mode == "taker" and order_submitted and len(fills) == 0:
                    order_id = order_details.get("order_id") if order_details else None
                    if order_id is not None:
                        broker.cancel_order(order_id)

    pnl = broker.pnl_summary(current_mid_price)
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_signal = action
    final_action = "hold" if (skipped_reason or live_mode_blocked) else action
    candidate_reason = decision.get("reason", "hold")
    if skipped_reason:
        decision_reason = skipped_reason
        runtime_reason = skipped_reason
    elif live_mode_blocked:
        decision_reason = "live_mode_blocked"
        runtime_reason = "live_mode_blocked"
    else:
        decision_reason = candidate_reason
        runtime_reason = (
            "signal_buy" if action == "buy" else "signal_sell" if action == "sell" else candidate_reason
        )
    signal_strength = _resolve_signal_strength(decision)
    spot_state_raw = decision.get("inputs", {}).get("spot_state", "flat")
    is_directional = action in ("buy", "sell") or candidate_reason == "weak_signal_filtered"
    candidate_side = (
        (action if action in ("buy", "sell") else ("buy" if spot_state_raw == "flat" else "sell"))
        if is_directional
        else None
    )
    out: dict = {
        "timestamp_utc": timestamp_utc,
        "iteration": iteration,
        "pair": pair,
        "runtime_mode": runtime_mode,
        "execution_mode": execution_mode,
        "raw_signal": raw_signal,
        "final_action": final_action,
        "decision_reason": decision_reason,
        "runtime_reason": runtime_reason,
        "signal_strength": signal_strength,
        "market": {
            "bid": best_bid,
            "ask": best_ask,
            "last": last_price,
            "spread": current_spread,
            "mid_price": current_mid_price,
            "book_imbalance": current_book_imbalance,
            "momentum": current_momentum,
            "volatility": current_volatility,
        },
        "strategy": decision,
        "risk": risk_result,
        "order": {
            "submitted": order_submitted,
            "details": order_details,
            "fills": fills,
            "skipped_reason": skipped_reason,
        },
        "broker": {
            "cash_usd": pnl["cash_usd"],
            "position_units": pnl["position_units"],
            "average_entry_price": pnl["average_entry_price"],
            "open_orders_count": pnl["open_orders_count"],
            "realized_pnl_usd": pnl["realized_pnl_usd"],
            "unrealized_pnl_usd": pnl["unrealized_pnl_usd"],
            "total_pnl_usd": pnl["total_pnl_usd"],
        },
        "live_mode_blocked": live_mode_blocked,
        "live_order_ready": live_order_ready,
        "live_account": live_account,
    }
    if is_directional:
        out["candidate_side"] = candidate_side
        out["candidate_reason"] = candidate_reason
    return out


def _build_status(
    result: dict | None,
    kill_switch_active: bool,
    shutdown_reason: str | None,
    error: str | None,
    pair_fallback: str = "",
) -> dict:
    """Build compact operator status snapshot."""
    r = result or {}
    broker = r.get("broker") or {}
    strategy = r.get("strategy") or {}
    order = r.get("order") or {}
    risk = r.get("risk") or {}
    market = r.get("market") or {}
    pos_units = broker.get("position_units", 0.0)
    mid = _safe_float(market.get("mid_price"))
    position_usd = pos_units * mid if mid > 0 else 0.0
    forced_outcome = r.get("forced_live_test_buy_outcome")
    live_outcome = r.get("live_order_outcome")
    if forced_outcome:
        last_action = forced_outcome
        raw_signal = r.get("raw_signal", "buy")
        final_action = r.get("final_action", forced_outcome)
        decision_reason = r.get("decision_reason", forced_outcome)
    elif live_outcome:
        last_action = live_outcome
        raw_signal = r.get("raw_signal", strategy.get("action", "hold"))
        final_action = r.get("final_action", live_outcome)
        decision_reason = r.get("decision_reason", live_outcome)
    else:
        last_action = (
            "submitted" if order.get("submitted") else (order.get("skipped_reason") or "hold")
        )
        raw_signal = r.get("raw_signal", strategy.get("action", "hold"))
        final_action = r.get("final_action", last_action)
        decision_reason = r.get("decision_reason", order.get("skipped_reason") or "hold")
    out: dict = {
        "timestamp": r.get("timestamp_utc")
        or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pair": r.get("pair") or pair_fallback,
        "runtime_mode": r.get("runtime_mode", "paper"),
        "execution_mode": r.get("execution_mode", "taker"),
        "iteration": r.get("iteration", 0),
        "kill_switch_active": kill_switch_active,
        "last_signal": strategy.get("action", "hold"),
        "last_action": last_action,
        "raw_signal": raw_signal,
        "final_action": final_action,
        "decision_reason": decision_reason,
        "signal_strength": r.get("signal_strength"),
        "last_mid_price": mid,
        "position_usd": position_usd,
        "realized_pnl": broker.get("realized_pnl_usd", 0.0),
        "unrealized_pnl": broker.get("unrealized_pnl_usd", 0.0),
        "open_orders": broker.get("open_orders_count", 0),
        "risk_ok": risk.get("allowed", True),
        "shutdown_reason": shutdown_reason,
        "error": error,
        **({"live_account": r["live_account"]} if r.get("live_account") is not None else {}),
    }
    if r.get("model_probability") is not None:
        out["model_probability"] = r["model_probability"]
    if r.get("model_score") is not None:
        out["model_score"] = r["model_score"]
    if r.get("candidate_side") is not None:
        out["candidate_side"] = r["candidate_side"]
    if r.get("candidate_reason") is not None:
        out["candidate_reason"] = r["candidate_reason"]
    if r.get("runtime_reason") is not None:
        out["runtime_reason"] = r["runtime_reason"]
    return out


def main() -> int:
    """Run one or more paper-trading cycles."""
    parser = argparse.ArgumentParser(
        description="Paper-trading quant engine (one-shot).",
    )
    parser.add_argument("--pair", required=True, help="Trading pair (e.g. XBTUSD).")
    parser.add_argument(
        "--usd-order-size",
        type=float,
        default=20.0,
        help="Order size in USD (default: 20.0).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Order book depth for features (default: 5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of cycles to run (default: 1).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Seconds to sleep between iterations (default: 0.0).",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["maker", "taker"],
        default="taker",
        help="Execution mode for paper trading. maker posts passive orders; taker uses aggressive prices for immediate fills.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="JSONL log file path for decision logging (default: <repo_root>/logs/quant_engine_runs.jsonl).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Runtime mode: live. Requires --enable-live-orders for real execution.",
    )
    parser.add_argument(
        "--enable-live-orders",
        action="store_true",
        help="Enable real Kraken orders. Requires --live. Default: observation only.",
    )
    parser.add_argument(
        "--force-live-test-buy",
        action="store_true",
        help="One-off: force one BUY market order on iteration 1. Requires --live and --enable-live-orders.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Runtime mode: paper (default).",
    )
    parser.add_argument(
        "--kill-switch-file",
        type=str,
        default="/tmp/openclaw.kill",
        help="Path to kill switch file. If it exists, bot stops cleanly (default: /tmp/openclaw.kill).",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Path to runtime status JSON (default: <repo_root>/logs/status.json).",
    )
    parser.add_argument(
        "--trade-events-file",
        type=str,
        default=None,
        help="Path to trade events JSONL (default: <repo_root>/logs/trade_events.jsonl).",
    )
    parser.add_argument(
        "--check-kraken-auth",
        action="store_true",
        help="Check Kraken API credentials and balance, then exit. No trading loop.",
    )
    parser.add_argument(
        "--check-kraken-add-order",
        action="store_true",
        help="Validate Kraken AddOrder (validate=True), then exit. No trading loop.",
    )
    parser.add_argument(
        "--check-kraken-cancel-order",
        action="store_true",
        help="Test Kraken CancelOrder with given txid, then exit. No trading loop.",
    )
    parser.add_argument(
        "--order-side",
        type=str,
        choices=["buy", "sell"],
        help="Required for --check-kraken-add-order: buy or sell.",
    )
    parser.add_argument(
        "--order-type",
        type=str,
        choices=["limit", "market"],
        help="Required for --check-kraken-add-order: limit or market.",
    )
    parser.add_argument(
        "--order-volume",
        type=str,
        help="Required for --check-kraken-add-order: order size in base asset.",
    )
    parser.add_argument(
        "--order-price",
        type=str,
        default=None,
        help="Required for --check-kraken-add-order when --order-type=limit. Ignored for market.",
    )
    parser.add_argument(
        "--cancel-txid",
        type=str,
        help="Required for --check-kraken-cancel-order: order transaction ID to cancel.",
    )
    args = parser.parse_args()

    if args.live and args.paper:
        print("Error: cannot specify both --live and --paper", file=sys.stderr)
        return 1
    if args.enable_live_orders and not args.live:
        print(
            "Error: --enable-live-orders requires --live",
            file=sys.stderr,
        )
        return 1
    if args.force_live_test_buy and not (args.live and args.enable_live_orders):
        print(
            "Error: --force-live-test-buy requires both --live and --enable-live-orders",
            file=sys.stderr,
        )
        return 1
    runtime_mode = "live" if args.live else "paper"
    enable_live_orders = bool(args.enable_live_orders)
    force_live_test_buy = bool(args.force_live_test_buy)

    check_modes = sum(
        [bool(args.check_kraken_auth), bool(args.check_kraken_add_order), bool(args.check_kraken_cancel_order)]
    )
    if check_modes > 1:
        print(
            "Error: --check-kraken-auth, --check-kraken-add-order, and --check-kraken-cancel-order are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    pair = args.pair.strip()
    usd_order_size = args.usd_order_size
    depth = args.depth
    iterations = max(1, args.iterations)
    sleep_seconds = max(0.0, args.sleep_seconds)
    execution_mode = args.execution_mode
    log_path = default_log_path() if args.log_file is None else Path(args.log_file)
    kill_switch_path = Path(args.kill_switch_file)
    status_path = (
        default_status_path()
        if args.status_file is None
        else Path(args.status_file)
    )
    trade_events_path = (
        default_trade_events_path()
        if args.trade_events_file is None
        else Path(args.trade_events_file)
    )
    signal_outcomes_path = default_signal_outcomes_path()
    training_examples_path = default_training_examples_path()
    shadow_inference_path = default_shadow_inference_path()

    shadow_model_obj = load_shadow_model()

    ensure_logs_and_artifacts_dirs()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    trade_events_path.parent.mkdir(parents=True, exist_ok=True)
    signal_outcomes_path.parent.mkdir(parents=True, exist_ok=True)

    if args.check_kraken_auth:
        _run_kraken_auth_check(
            pair=pair,
            status_path=status_path,
            trade_events_path=trade_events_path,
        )
        return 0

    if args.check_kraken_add_order:
        if not args.order_side or not args.order_type or not args.order_volume:
            print(
                "Error: --check-kraken-add-order requires --order-side, --order-type, and --order-volume",
                file=sys.stderr,
            )
            return 1
        if args.order_type == "limit" and not args.order_price:
            print(
                "Error: --order-type=limit requires --order-price",
                file=sys.stderr,
            )
            return 1
        try:
            vol = float(args.order_volume)
        except (ValueError, TypeError):
            vol = -1.0
        if vol <= 0:
            print(
                "Error: --order-volume must be greater than 0",
                file=sys.stderr,
            )
            return 1
        if args.order_type == "limit":
            try:
                pr = float(args.order_price)
            except (ValueError, TypeError):
                pr = -1.0
            if pr <= 0:
                print(
                    "Error: --order-price must be greater than 0 for limit orders",
                    file=sys.stderr,
                )
                return 1
        _run_kraken_add_order_check(
            pair=pair,
            order_side=args.order_side,
            order_type=args.order_type,
            order_volume=args.order_volume,
            order_price=args.order_price if args.order_type == "limit" else None,
            status_path=status_path,
            trade_events_path=trade_events_path,
        )
        return 0

    if args.check_kraken_cancel_order:
        txid = args.cancel_txid.strip() if args.cancel_txid else ""
        if not txid:
            print(
                "Error: --check-kraken-cancel-order requires non-empty --cancel-txid",
                file=sys.stderr,
            )
            return 1
        _run_kraken_cancel_order_check(
            pair=pair,
            cancel_txid=txid,
            status_path=status_path,
            trade_events_path=trade_events_path,
        )
        return 0

    if runtime_mode == "live":
        try:
            _kraken_auth_module().load_credentials()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            print(
                "Live mode requires KRAKEN_API_KEY and KRAKEN_API_SECRET.",
                file=sys.stderr,
            )
            return 1

    broker = PaperBroker(starting_cash_usd=200.0)
    last_result: dict | None = None
    shutdown_reason: str | None = None
    exit_code = 0

    def _emit_stopped(reason: str, err: str | None = None) -> None:
        nonlocal shutdown_reason
        shutdown_reason = reason
        status = _build_status(last_result, False, reason, err, pair_fallback=pair)
        write_status(status_path, status)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        it = (last_result or {}).get("iteration", 0)
        base = {
            "timestamp": ts,
            "pair": pair,
            "runtime_mode": runtime_mode,
            "execution_mode": execution_mode,
            "iteration": it,
            "reason": reason,
        }
        if err:
            append_trade_event(
                trade_events_path,
                {**base, "event_type": "engine_error", "error": err},
            )
        append_trade_event(
            trade_events_path,
            {**base, "event_type": "engine_stopped", "error": err if err else None},
        )

    append_trade_event(
        trade_events_path,
        {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "event_type": "engine_started",
            "pair": pair,
            "runtime_mode": runtime_mode,
            "execution_mode": execution_mode,
            "iterations": iterations,
        },
    )

    pending_signal_outcomes: list[dict] = []
    pending_training_examples: list[dict] = []
    SIGNAL_OUTCOME_DELTAS = (30, 60, 300)

    try:
        for i in range(iterations):
            if kill_switch_path.exists():
                ts = datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": ts,
                        "event_type": "kill_switch_triggered",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "reason": "kill switch file exists",
                    },
                )
                status = _build_status(
                    last_result, True, "kill_switch", None, pair_fallback=pair
                )
                write_status(status_path, status)
                shutdown_reason = "kill_switch"
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": ts,
                        "event_type": "engine_stopped",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "reason": "kill_switch",
                    },
                )
                break

            # Best-effort: process one pending signal outcome per iteration (non-blocking)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            for idx, pending in enumerate(pending_signal_outcomes):
                signal_time = pending.get("_signal_time")
                if signal_time is None:
                    continue
                if signal_time.tzinfo is None:
                    signal_time = signal_time.replace(tzinfo=datetime.timezone.utc)
                elapsed = (now_utc - signal_time).total_seconds()
                updated = False
                for delta in SIGNAL_OUTCOME_DELTAS:
                    key = f"price_after_{delta}s"
                    if elapsed >= delta and pending.get(key) is None:
                        mid = _fetch_mid_price(pending["pair"])
                        if mid is not None:
                            pending[key] = round(mid, 2)
                            updated = True
                        break
                if updated:
                    if all(pending.get(f"price_after_{d}s") is not None for d in SIGNAL_OUTCOME_DELTAS):
                        record = {k: v for k, v in pending.items() if not k.startswith("_")}
                        append_signal_outcome(signal_outcomes_path, record)
                        pending_signal_outcomes.pop(idx)
                    break

            # Best-effort: process one pending training example per iteration (non-blocking)
            for idx, pending in enumerate(pending_training_examples):
                signal_time = pending.get("_signal_time")
                if signal_time is None:
                    continue
                if signal_time.tzinfo is None:
                    signal_time = signal_time.replace(tzinfo=datetime.timezone.utc)
                elapsed = (now_utc - signal_time).total_seconds()
                updated = False
                for delta in SIGNAL_OUTCOME_DELTAS:
                    key = f"price_after_{delta}s"
                    if elapsed >= delta and pending.get(key) is None:
                        mid = _fetch_mid_price(pending["pair"])
                        if mid is not None:
                            pending[key] = round(mid, 2)
                            updated = True
                        break
                if updated:
                    if all(pending.get(f"price_after_{d}s") is not None for d in SIGNAL_OUTCOME_DELTAS):
                        price_at = _safe_float(pending.get("mid_price"))
                        candidate_side = pending.get("candidate_side", "buy")
                        labels = _compute_training_labels(
                            price_at or 0.0,
                            candidate_side,
                            _safe_float(pending.get("price_after_30s")),
                            _safe_float(pending.get("price_after_60s")),
                            _safe_float(pending.get("price_after_300s")),
                        )
                        pending.update(labels)
                        record = {k: v for k, v in pending.items() if not k.startswith("_")}
                        append_training_example(training_examples_path, record)
                        pending_training_examples.pop(idx)
                    break

            if iterations > 1:
                print(f"--- iteration {i + 1}/{iterations} ---", file=sys.stderr)
            live_account = (
                _fetch_live_account_snapshot(pair)
                if runtime_mode == "live"
                else None
            )
            result = _run_one_cycle(
                pair,
                usd_order_size,
                depth,
                execution_mode,
                runtime_mode,
                broker,
                iteration=i + 1,
                enable_live_orders=enable_live_orders,
                live_account=live_account,
                trade_events_path=trade_events_path,
            )
            last_result = result

            _apply_shadow_inference(result, shadow_model_obj, shadow_inference_path)

            action = result.get("strategy", {}).get("action", "hold")
            order = result.get("order", {})
            skip = order.get("skipped_reason")
            live_blocked = result.get("live_mode_blocked", False)
            live_order_ready = result.get("live_order_ready")

            if force_live_test_buy and i == 0:
                ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                mid = _safe_float(result.get("market", {}).get("mid_price"))
                size_usd = usd_order_size
                volume = size_usd / mid if mid > 0 else 0.0
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": ts,
                        "event_type": "forced_live_test_buy_started",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": "taker",
                        "side": "buy",
                        "order_type": "market",
                        "size_usd": size_usd,
                        "volume": volume,
                    },
                )
                if not (isinstance(volume, (int, float)) and volume > 0):
                    append_trade_event(
                        trade_events_path,
                        {
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "event_type": "forced_live_test_buy_blocked",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "side": "buy",
                            "size_usd": size_usd,
                            "volume": volume,
                            "reason": "invalid_order_volume",
                        },
                    )
                    result["forced_live_test_buy_outcome"] = "forced_live_test_buy_blocked"
                elif size_usd > MAX_LIVE_ORDER_USD:
                    append_trade_event(
                        trade_events_path,
                        {
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "event_type": "forced_live_test_buy_blocked",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "side": "buy",
                            "size_usd": size_usd,
                            "reason": "max_live_order_usd_exceeded",
                        },
                    )
                    result["forced_live_test_buy_outcome"] = "forced_live_test_buy_blocked"
                else:
                    try:
                        kraken_auth = _kraken_auth_module()
                        api_key, api_secret = kraken_auth.load_credentials()
                        resp = kraken_auth.add_order(
                            api_key,
                            api_secret,
                            pair=pair,
                            side="buy",
                            ordertype="market",
                            volume=volume,
                            price=None,
                            validate=False,
                        )
                        txid = ""
                        res = resp.get("result") or {}
                        if isinstance(res, dict):
                            raw_txid = res.get("txid", [])
                            if isinstance(raw_txid, list) and raw_txid:
                                txid = raw_txid[0] if isinstance(raw_txid[0], str) else str(raw_txid[0])
                            elif isinstance(raw_txid, str):
                                txid = raw_txid
                            else:
                                txid = str(raw_txid) if raw_txid else ""
                        ev = {
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "event_type": "forced_live_test_buy_submitted",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "side": "buy",
                            "order_type": "market",
                            "size_usd": size_usd,
                            "volume": volume,
                        }
                        if txid:
                            ev["txid"] = txid
                        append_trade_event(trade_events_path, ev)
                        result["forced_live_test_buy_outcome"] = "forced_live_test_buy_submitted"
                    except RuntimeError as e:
                        err_msg = str(e)
                        append_trade_event(
                            trade_events_path,
                            {
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                                "event_type": "forced_live_test_buy_failed",
                                "pair": pair,
                                "runtime_mode": runtime_mode,
                                "side": "buy",
                                "size_usd": size_usd,
                                "volume": volume,
                                "reason": err_msg,
                            },
                        )
                        result["forced_live_test_buy_outcome"] = "forced_live_test_buy_failed"
                        result["forced_live_test_buy_error"] = err_msg
            elif live_order_ready and runtime_mode == "live" and enable_live_orders:
                ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                side = live_order_ready["side"]
                ordertype = live_order_ready["ordertype"]
                size_usd = live_order_ready["size_usd"]
                volume = live_order_ready["volume"]
                price = live_order_ready.get("price")
                if ordertype == "limit" and (price is None or price <= 0):
                    append_trade_event(
                        trade_events_path,
                        {
                            "timestamp": ts,
                            "event_type": "live_order_blocked",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "execution_mode": execution_mode,
                            "side": side,
                            "order_type": ordertype,
                            "size_usd": size_usd,
                            "reason": "no_limit_price",
                        },
                    )
                    result["live_order_outcome"] = "live_blocked"
                elif not (isinstance(volume, (int, float)) and volume > 0):
                    append_trade_event(
                        trade_events_path,
                        {
                            "timestamp": ts,
                            "event_type": "live_order_blocked",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "execution_mode": execution_mode,
                            "side": side,
                            "order_type": ordertype,
                            "size_usd": size_usd,
                            "volume": volume,
                            "reason": "invalid_order_volume",
                        },
                    )
                    result["live_order_outcome"] = "live_blocked"
                elif size_usd > MAX_LIVE_ORDER_USD:
                    append_trade_event(
                        trade_events_path,
                        {
                            "timestamp": ts,
                            "event_type": "live_order_blocked",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "execution_mode": execution_mode,
                            "side": side,
                            "order_type": ordertype,
                            "size_usd": size_usd,
                            "reason": "max_live_order_usd_exceeded",
                        },
                    )
                    result["live_order_outcome"] = "live_blocked"
                else:
                    try:
                        append_trade_event(
                            trade_events_path,
                            {
                                "timestamp": ts,
                                "event_type": "live_order_submission_started",
                                "pair": pair,
                                "runtime_mode": runtime_mode,
                                "execution_mode": execution_mode,
                                "side": side,
                                "order_type": ordertype,
                                "size_usd": size_usd,
                                "volume": volume,
                                "price": price,
                            },
                        )
                        kraken_auth = _kraken_auth_module()
                        api_key, api_secret = kraken_auth.load_credentials()
                        resp = kraken_auth.add_order(
                            api_key,
                            api_secret,
                            pair=pair,
                            side=side,
                            ordertype=ordertype,
                            volume=volume,
                            price=price if ordertype == "limit" else None,
                            validate=False,
                        )
                        txid = ""
                        res = resp.get("result") or {}
                        if isinstance(res, dict):
                            raw_txid = res.get("txid", [])
                            if isinstance(raw_txid, list) and raw_txid:
                                txid = raw_txid[0] if isinstance(raw_txid[0], str) else str(raw_txid[0])
                            elif isinstance(raw_txid, str):
                                txid = raw_txid
                            else:
                                txid = str(raw_txid) if raw_txid else ""
                        ev = {
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "event_type": "live_order_submitted",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "execution_mode": execution_mode,
                            "side": side,
                            "order_type": ordertype,
                            "size_usd": size_usd,
                            "volume": volume,
                            "price": price,
                        }
                        if txid:
                            ev["txid"] = txid
                        append_trade_event(trade_events_path, ev)
                        result["live_order_outcome"] = "live_submitted"
                    except RuntimeError as e:
                        err_msg = str(e)
                        append_trade_event(
                            trade_events_path,
                            {
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                                "event_type": "live_order_submission_failed",
                                "pair": pair,
                                "runtime_mode": runtime_mode,
                                "execution_mode": execution_mode,
                                "side": side,
                                "order_type": ordertype,
                                "size_usd": size_usd,
                                "volume": volume,
                                "reason": err_msg,
                            },
                        )
                        result["live_order_outcome"] = "live_failed"
                        result["live_order_error"] = err_msg

            if result.get("decision_reason") == "weak_signal_filtered":
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "weak_signal_filtered",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "reason": "weak_signal_filtered",
                        "signal_strength": result.get("signal_strength"),
                    },
                )

            is_directional_candidate = _is_directional_candidate(result)
            if is_directional_candidate:
                strategy_inputs = result.get("strategy", {}).get("inputs", {})
                spot_state_raw = strategy_inputs.get("spot_state", "flat")
                spot_state = "FLAT" if spot_state_raw == "flat" else "LONG"
                candidate_side = (
                    action
                    if action in ("buy", "sell")
                    else ("buy" if spot_state_raw == "flat" else "sell")
                )
                mid = _safe_float(result.get("market", {}).get("mid_price"))
                if mid and mid > 0:
                    ts_str = result.get("timestamp_utc", "")
                    ts_parsed = ts_str.replace("Z", "+00:00") if ts_str else ""
                    try:
                        signal_time = datetime.datetime.fromisoformat(ts_parsed) if ts_parsed else None
                    except (ValueError, TypeError):
                        signal_time = None
                    if signal_time is not None:
                        training_record = {
                            "timestamp": ts_str,
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "spot_state": spot_state,
                            "candidate_side": candidate_side,
                            "candidate_reason": result.get("candidate_reason", result.get("decision_reason", "")),
                            "runtime_reason": result.get("runtime_reason", result.get("decision_reason", "")),
                            "signal_strength": result.get("signal_strength"),
                            "mid_price": round(mid, 2),
                            "spread": _safe_float(result.get("market", {}).get("spread")),
                            "book_imbalance": _safe_float(result.get("market", {}).get("book_imbalance")),
                            "momentum": _safe_float(result.get("market", {}).get("momentum")),
                            "volatility": _safe_float(result.get("market", {}).get("volatility")),
                            "momentum_threshold": _safe_float(strategy_inputs.get("momentum_threshold")),
                            "price_after_30s": None,
                            "price_after_60s": None,
                            "price_after_300s": None,
                            "label_30s": None,
                            "label_60s": None,
                            "label_300s": None,
                            "_signal_time": signal_time,
                        }
                        pending_training_examples.append(training_record)

            if action in ("buy", "sell"):
                ts_str = result.get("timestamp_utc", "")
                price_at_signal = _safe_float(result.get("market", {}).get("mid_price"))
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": ts_str,
                        "event_type": "signal_generated",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": action,
                        "reason": result.get("strategy", {}).get("reason", ""),
                    },
                )
                if price_at_signal > 0:
                    signal_record = {
                        "timestamp": ts_str,
                        "pair": pair,
                        "signal": action,
                        "price_at_signal": round(price_at_signal, 2),
                    }
                    append_signal_outcome(signal_outcomes_path, signal_record)
                    ts_parsed = ts_str.replace("Z", "+00:00")
                    try:
                        signal_time = datetime.datetime.fromisoformat(ts_parsed)
                    except (ValueError, TypeError):
                        signal_time = None
                    if signal_time is not None:
                        pending_signal_outcomes.append({
                            **signal_record,
                            "_signal_time": signal_time,
                        })
            if skip == "risk_blocked":
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "risk_blocked",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": action,
                        "reason": "risk_blocked",
                    },
                )
            if skip == "sell_suppressed_low_inventory":
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "sell_suppressed_low_inventory",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": "sell",
                        "reason": "sell_suppressed_low_inventory",
                    },
                )
            if skip == "buy_suppressed_low_usd":
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "buy_suppressed_low_usd",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": "buy",
                        "reason": "buy_suppressed_low_usd",
                    },
                )
            if skip == "buy_cooldown_active":
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "buy_cooldown_active",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": "buy",
                        "reason": "buy_cooldown_active",
                    },
                )
            if skip == "sell_cooldown_active":
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "sell_cooldown_active",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": "sell",
                        "reason": "sell_cooldown_active",
                    },
                )
            if live_blocked:
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "live_mode_blocked",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": action,
                        "reason": "authenticated execution not yet enabled",
                    },
                )
            if order.get("submitted"):
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "order_submitted",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": order.get("details", {}).get("side", ""),
                        "size_usd": usd_order_size,
                        "price": order.get("details", {}).get("price"),
                    },
                )
            fills = order.get("fills") or []
            if fills:
                for f in fills:
                    append_trade_event(
                        trade_events_path,
                        {
                            "timestamp": result.get("timestamp_utc", ""),
                            "event_type": "order_filled",
                            "pair": pair,
                            "runtime_mode": runtime_mode,
                            "execution_mode": execution_mode,
                            "iteration": i + 1,
                            "side": f.get("side", ""),
                            "size_usd": _safe_float(f.get("size_units"))
                            * _safe_float(f.get("price")),
                            "price": f.get("price"),
                            "position_usd": result.get("broker", {}).get(
                                "position_units", 0
                            )
                            * result.get("market", {}).get("mid_price", 0),
                            "realized_pnl": result.get("broker", {}).get(
                                "realized_pnl_usd", 0
                            ),
                        },
                    )

            status_err = None
            if result.get("live_order_outcome") == "live_failed":
                status_err = result.get("live_order_error")
            elif result.get("forced_live_test_buy_outcome") == "forced_live_test_buy_failed":
                status_err = result.get("forced_live_test_buy_error")
            status = _build_status(result, False, None, status_err, pair_fallback=pair)
            write_status(status_path, status)

            print(json.dumps(result, indent=2))
            append_jsonl(log_path, result)

            if i < iterations - 1 and sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if shutdown_reason is None:
            shutdown_reason = "iterations_complete"
            status = _build_status(
                last_result, False, shutdown_reason, None, pair_fallback=pair
            )
            write_status(status_path, status)
            append_trade_event(
                trade_events_path,
                {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "event_type": "engine_stopped",
                    "pair": pair,
                    "runtime_mode": runtime_mode,
                    "execution_mode": execution_mode,
                    "iteration": (last_result or {}).get("iteration", 0),
                    "reason": shutdown_reason,
                },
            )

    except KeyboardInterrupt:
        _emit_stopped("keyboard_interrupt", None)
        exit_code = 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        _emit_stopped("unexpected_error", str(e))
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
