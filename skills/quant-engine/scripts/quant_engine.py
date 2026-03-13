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
from risk import allow_trade
from strategy import maker_first_mean_reversion

MAX_LIVE_ORDER_USD = 10


def _repo_root() -> Path:
    """Return repo root (parent of skills/)."""
    return Path(__file__).resolve().parents[3]


def default_log_path() -> Path:
    """Return default run log path: <repo_root>/logs/quant_engine_runs.jsonl."""
    return _repo_root() / "logs" / "quant_engine_runs.jsonl"


def default_status_path() -> Path:
    """Return default status file path: <repo_root>/logs/status.json."""
    return _repo_root() / "logs" / "status.json"


def default_trade_events_path() -> Path:
    """Return default trade events path: <repo_root>/logs/trade_events.jsonl."""
    return _repo_root() / "logs" / "trade_events.jsonl"


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


def _kraken_reader_path() -> Path:
    """Resolve path to kraken_reader.py relative to this file."""
    this_dir = Path(__file__).resolve().parent
    return this_dir.parent.parent / "kraken-reader" / "scripts" / "kraken_reader.py"


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


def _run_one_cycle(
    pair: str,
    usd_order_size: float,
    depth: int,
    execution_mode: str,
    runtime_mode: str,
    broker: PaperBroker,
    iteration: int,
    enable_live_orders: bool = False,
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

    decision = maker_first_mean_reversion(
        mid_price=current_mid_price,
        spread=current_spread,
        book_imbalance=current_book_imbalance,
        momentum=current_momentum,
        volatility=current_volatility,
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
        if action == "sell" and broker.position_units <= 0:
            skipped_reason = "no_inventory_to_sell"
        elif not risk_result.get("allowed"):
            skipped_reason = "risk_blocked"
        elif runtime_mode == "live" and not enable_live_orders:
            live_mode_blocked = True
        elif runtime_mode == "live" and enable_live_orders:
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
        else:
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
    return {
        "timestamp_utc": timestamp_utc,
        "iteration": iteration,
        "pair": pair,
        "runtime_mode": runtime_mode,
        "execution_mode": execution_mode,
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
    }


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
    elif live_outcome:
        last_action = live_outcome
    else:
        last_action = (
            "submitted" if order.get("submitted") else (order.get("skipped_reason") or "hold")
        )
    return {
        "timestamp": r.get("timestamp_utc")
        or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pair": r.get("pair") or pair_fallback,
        "runtime_mode": r.get("runtime_mode", "paper"),
        "execution_mode": r.get("execution_mode", "taker"),
        "iteration": r.get("iteration", 0),
        "kill_switch_active": kill_switch_active,
        "last_signal": strategy.get("action", "hold"),
        "last_action": last_action,
        "last_mid_price": mid,
        "position_usd": position_usd,
        "realized_pnl": broker.get("realized_pnl_usd", 0.0),
        "unrealized_pnl": broker.get("unrealized_pnl_usd", 0.0),
        "open_orders": broker.get("open_orders_count", 0),
        "risk_ok": risk.get("allowed", True),
        "shutdown_reason": shutdown_reason,
        "error": error,
    }


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

    status_path.parent.mkdir(parents=True, exist_ok=True)
    trade_events_path.parent.mkdir(parents=True, exist_ok=True)

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

            if iterations > 1:
                print(f"--- iteration {i + 1}/{iterations} ---", file=sys.stderr)
            result = _run_one_cycle(
                pair,
                usd_order_size,
                depth,
                execution_mode,
                runtime_mode,
                broker,
                iteration=i + 1,
                enable_live_orders=enable_live_orders,
            )
            last_result = result

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

            if action in ("buy", "sell"):
                append_trade_event(
                    trade_events_path,
                    {
                        "timestamp": result.get("timestamp_utc", ""),
                        "event_type": "signal_generated",
                        "pair": pair,
                        "runtime_mode": runtime_mode,
                        "execution_mode": execution_mode,
                        "iteration": i + 1,
                        "side": action,
                        "reason": result.get("strategy", {}).get("reason", ""),
                    },
                )
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
