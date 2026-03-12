#!/usr/bin/env python3
"""
Read public Kraken market data.

Uses Kraken REST API. No API key required for public endpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def fetch_json(url: str) -> object:
    """Make a GET request, set User-Agent header, return parsed JSON."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "OpenClaw-kraken-reader/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e


def cmd_list_instruments(args: argparse.Namespace) -> int:
    """List available trading pairs from Kraken AssetPairs endpoint."""
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        data = fetch_json(url)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: unexpected API response format (expected JSON object)", file=sys.stderr)
        return 1

    result = data.get("result")
    if not isinstance(result, dict):
        print("Error: response missing or invalid 'result' object", file=sys.stderr)
        return 1

    instruments = []
    for pair_key, pair_data in result.items():
        if not isinstance(pair_data, dict):
            continue
        instruments.append({
            "pair_key": pair_key,
            "altname": pair_data.get("altname", ""),
            "wsname": pair_data.get("wsname", ""),
            "base": pair_data.get("base", ""),
            "quote": pair_data.get("quote", ""),
            "status": pair_data.get("status", ""),
        })

    fmt = getattr(args, "format", "text")
    pretty = getattr(args, "pretty", False)
    if fmt == "json":
        indent = 2 if pretty else None
        print(json.dumps(instruments, indent=indent, sort_keys=pretty))
        return 0

    for inst in instruments:
        print(
            f"{inst['pair_key']} | altname={inst['altname']} | wsname={inst['wsname']} | "
            f"base={inst['base']} | quote={inst['quote']} | status={inst['status']}"
        )
    return 0


def _safe_list_val(lst: object, idx: int, default: str = "") -> str:
    """Get list element at index, or default if missing/invalid."""
    if not isinstance(lst, list) or idx >= len(lst):
        return default
    v = lst[idx]
    return str(v) if v is not None else default


def cmd_ticker(args: argparse.Namespace) -> int:
    """Get ticker for a pair from Kraken Ticker endpoint."""
    pair = args.pair.strip()
    if not pair:
        print("Error: --pair cannot be empty", file=sys.stderr)
        return 1

    url = f"https://api.kraken.com/0/public/Ticker?pair={urllib.parse.quote(pair, safe='')}"
    try:
        data = fetch_json(url)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: unexpected API response format (expected JSON object)", file=sys.stderr)
        return 1

    errors = data.get("error", [])
    if errors and isinstance(errors, list) and len(errors) > 0:
        msg = errors[0] if isinstance(errors[0], str) else str(errors)
        print(f"Error: Kraken API error: {msg}", file=sys.stderr)
        return 1

    result = data.get("result")
    if not isinstance(result, dict) or len(result) == 0:
        print("Error: invalid pair or empty result", file=sys.stderr)
        return 1

    pair_key = next(iter(result.keys()))
    raw = result[pair_key]
    if not isinstance(raw, dict):
        print("Error: unexpected ticker data format", file=sys.stderr)
        return 1

    a = raw.get("a")
    b = raw.get("b")
    c = raw.get("c")
    v = raw.get("v")
    p = raw.get("p")
    t = raw.get("t")
    l_ = raw.get("l")
    h = raw.get("h")
    o = raw.get("o")

    out = {
        "pairKey": pair_key,
        "ask": _safe_list_val(a, 0),
        "bid": _safe_list_val(b, 0),
        "last": _safe_list_val(c, 0),
        "volume_today": _safe_list_val(v, 0),
        "volume_24h": _safe_list_val(v, 1),
        "vwap_today": _safe_list_val(p, 0),
        "vwap_24h": _safe_list_val(p, 1),
        "trades_today": _safe_list_val(t, 0),
        "trades_24h": _safe_list_val(t, 1),
        "low_today": _safe_list_val(l_, 0),
        "low_24h": _safe_list_val(l_, 1),
        "high_today": _safe_list_val(h, 0),
        "high_24h": _safe_list_val(h, 1),
        "open_today": str(o) if o is not None else "",
    }

    fmt = getattr(args, "format", "text")
    pretty = getattr(args, "pretty", False)
    if fmt == "json":
        indent = 2 if pretty else None
        print(json.dumps(out, indent=indent, sort_keys=pretty))
        return 0

    print(
        f"{out['pairKey']} | bid={out['bid']} | ask={out['ask']} | last={out['last']} | "
        f"vol_today={out['volume_today']} | vol_24h={out['volume_24h']} | "
        f"low_today={out['low_today']} | high_today={out['high_today']} | open_today={out['open_today']}"
    )
    return 0


def fetch_trades(pair: str) -> dict:
    """
    Fetch recent public trades for the canonical Kraken pair.

    Returns parsed JSON payload with pair key and trades list.
    Raises RuntimeError on API errors or malformed responses.
    """
    url = f"https://api.kraken.com/0/public/Trades?pair={urllib.parse.quote(pair, safe='')}"
    data = fetch_json(url)

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected API response format (expected JSON object)")

    errors = data.get("error", [])
    if errors and isinstance(errors, list) and len(errors) > 0:
        msg = errors[0] if isinstance(errors[0], str) else str(errors)
        raise RuntimeError(f"Kraken API error: {msg}")

    result = data.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Response missing or invalid 'result' object")

    pair_keys = [k for k in result.keys() if k != "last"]
    if not pair_keys:
        raise RuntimeError("Result missing expected pair key")

    pair_key = pair_keys[0]
    raw_trades = result.get(pair_key)
    if not isinstance(raw_trades, list):
        raw_trades = []

    return {
        "pair": pair_key,
        "trades": raw_trades,
        "count": len(raw_trades),
    }


def cmd_trades(args: argparse.Namespace) -> int:
    """Get recent trades for a pair from Kraken Trades endpoint."""
    pair = args.pair.strip()
    if not pair:
        print("Error: --pair cannot be empty", file=sys.stderr)
        return 1

    try:
        data = fetch_trades(pair)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    pair_key = data["pair"]
    trades = data["trades"]
    count = data["count"]

    fmt = getattr(args, "format", "text")
    pretty = getattr(args, "pretty", False)
    if fmt == "json":
        out = {
            "pair": pair_key,
            "trades": trades,
            "count": count,
        }
        indent = 2 if pretty else None
        print(json.dumps(out, indent=indent, sort_keys=pretty))
        return 0

    # Text mode: at most 10 trades
    print(f"{pair_key} | trades={count}")
    if count == 0:
        return 0
    for i, t in enumerate(trades[:10], 1):
        if isinstance(t, (list, tuple)) and len(t) >= 5:
            price = t[0]
            volume = t[1]
            time_val = t[2]
            side = t[3] if len(t) > 3 else ""
            order_type = t[4] if len(t) > 4 else ""
            print(f"{i} | price={price} | volume={volume} | side={side} | type={order_type} | time={time_val}")
        else:
            print(f"{i} | (malformed)")
    return 0


def _parse_book_level(level: object) -> dict:
    """Parse Kraken order book level [price, volume, timestamp] to dict."""
    if not isinstance(level, list) or len(level) < 2:
        return {"price": "", "volume": "", "timestamp": ""}
    price = str(level[0]) if level[0] is not None else ""
    volume = str(level[1]) if level[1] is not None else ""
    ts = level[2] if len(level) > 2 else ""
    timestamp = str(ts) if ts is not None else ""
    return {"price": price, "volume": volume, "timestamp": timestamp}


def cmd_orderbook(args: argparse.Namespace) -> int:
    """Get order book for a pair from Kraken Depth endpoint."""
    pair = args.pair.strip()
    if not pair:
        print("Error: --pair cannot be empty", file=sys.stderr)
        return 1

    url = f"https://api.kraken.com/0/public/Depth?pair={urllib.parse.quote(pair, safe='')}&count=10"
    try:
        data = fetch_json(url)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: unexpected API response format (expected JSON object)", file=sys.stderr)
        return 1

    errors = data.get("error", [])
    if errors and isinstance(errors, list) and len(errors) > 0:
        msg = errors[0] if isinstance(errors[0], str) else str(errors)
        print(f"Error: Kraken API error: {msg}", file=sys.stderr)
        return 1

    result = data.get("result")
    if not isinstance(result, dict) or len(result) == 0:
        print("Error: invalid pair or empty result", file=sys.stderr)
        return 1

    pair_key = next(iter(result.keys()))
    raw = result[pair_key]
    if not isinstance(raw, dict):
        print("Error: unexpected order book data format", file=sys.stderr)
        return 1

    bids_raw = raw.get("bids")
    asks_raw = raw.get("asks")
    if not isinstance(bids_raw, list):
        bids_raw = []
    if not isinstance(asks_raw, list):
        asks_raw = []

    bids = [_parse_book_level(item) for item in bids_raw]
    asks = [_parse_book_level(item) for item in asks_raw]

    out = {"pairKey": pair_key, "bids": bids, "asks": asks}

    fmt = getattr(args, "format", "text")
    pretty = getattr(args, "pretty", False)
    if fmt == "json":
        indent = 2 if pretty else None
        print(json.dumps(out, indent=indent, sort_keys=pretty))
        return 0

    print(out["pairKey"])
    print("BIDS:")
    for b in bids:
        print(f"  {b['price']} | {b['volume']} | {b['timestamp']}")
    print("ASKS:")
    for a in asks:
        print(f"  {a['price']} | {a['volume']} | {a['timestamp']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read public Kraken market data.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list-instruments", help="List available trading pairs.")
    p_list.set_defaults(func=cmd_list_instruments)

    p_ticker = subparsers.add_parser("ticker", help="Get ticker for a pair.")
    p_ticker.add_argument("--pair", required=True, help="Trading pair (e.g. XBTUSD).")
    p_ticker.set_defaults(func=cmd_ticker)

    p_trades = subparsers.add_parser("trades", help="Get recent trades for a pair.")
    p_trades.add_argument("--pair", required=True, help="Trading pair (e.g. XBTUSD).")
    p_trades.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    p_trades.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    p_trades.set_defaults(func=cmd_trades)

    p_book = subparsers.add_parser("orderbook", help="Get order book for a pair.")
    p_book.add_argument("--pair", required=True, help="Trading pair (e.g. XBTUSD).")
    p_book.set_defaults(func=cmd_orderbook)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
