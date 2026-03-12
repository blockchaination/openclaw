#!/usr/bin/env python3
"""
Kraken authenticated API helpers.

Load credentials from env, sign private REST requests, call Balance endpoint.
Uses standard library only. Never prints or logs raw secrets.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _parse_env_file(path: Path) -> dict[str, str]:
    """
    Parse shell-style export lines. Returns dict of var names to values.
    Only extracts KRAKEN_API_KEY and KRAKEN_API_SECRET. Never logs secrets.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export ") and "=" in line:
                    rest = line[7:].strip()
                    if "=" in rest:
                        key, _, val = rest.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in ("KRAKEN_API_KEY", "KRAKEN_API_SECRET"):
                            out[key] = val
    except OSError:
        pass
    return out


def load_credentials() -> tuple[str, str]:
    """
    Load Kraken API key and secret from environment, with fallback to
    ~/.openclaw-kraken.env if env vars are missing.

    Returns:
        (api_key, api_secret)

    Raises:
        RuntimeError: if KRAKEN_API_KEY or KRAKEN_API_SECRET is missing or empty.
    """
    key = os.environ.get("KRAKEN_API_KEY", "").strip()
    secret = os.environ.get("KRAKEN_API_SECRET", "").strip()
    if not key or not secret:
        env_path = Path.home() / ".openclaw-kraken.env"
        from_file = _parse_env_file(env_path)
        key = from_file.get("KRAKEN_API_KEY", "").strip()
        secret = from_file.get("KRAKEN_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError(
            "KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in the environment "
            "or in ~/.openclaw-kraken.env"
        )
    return key, secret


def _sign_kraken_request(urlpath: str, data: dict, secret: str) -> str:
    """
    Generate Kraken API-Sign header value.

    HMAC-SHA512 of (urlpath + SHA256(nonce + POST data)) with base64-decoded secret.
    """
    encoded = (str(data["nonce"]) + urllib.parse.urlencode(data)).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def post_private(
    urlpath: str,
    data: dict,
    api_key: str,
    api_secret: str,
    timeout: int = 30,
) -> dict:
    """
    POST to Kraken private endpoint.

    Args:
        urlpath: e.g. "/0/private/Balance"
        data: POST body dict (nonce will be added if missing)
        api_key: Kraken API key
        api_secret: Kraken API secret (base64)

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: on HTTP error, network error, or invalid JSON.
    """
    nonce = str(int(time.time() * 1000))
    payload = {**data, "nonce": nonce}
    post_data = urllib.parse.urlencode(payload).encode("utf-8")
    signature = _sign_kraken_request(urlpath, payload, api_secret)

    url = f"https://api.kraken.com{urlpath}"
    req = urllib.request.Request(
        url,
        data=post_data,
        method="POST",
        headers={
            "API-Key": api_key,
            "API-Sign": signature,
            "User-Agent": "OpenClaw-kraken-auth/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kraken API HTTP {e.code}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Kraken API network error: {e.reason}") from e


def get_account_balance(api_key: str, api_secret: str) -> dict:
    """
    Call Kraken Balance endpoint.

    Returns:
        {"result": {asset: balance, ...}} on success.
        Kraken returns string balances; we pass through as-is.

    Raises:
        RuntimeError: on auth failure, API error, or network error.
    """
    data = post_private("/0/private/Balance", {}, api_key, api_secret)
    errors = data.get("error", [])
    if errors and isinstance(errors, list) and len(errors) > 0:
        msg = errors[0] if isinstance(errors[0], str) else str(errors)
        raise RuntimeError(f"Kraken API error: {msg}")
    return data


def add_order(
    api_key: str,
    api_secret: str,
    pair: str,
    side: str,
    ordertype: str,
    volume: str | float,
    *,
    price: str | float | None = None,
    validate: bool = False,
    timeout: int = 30,
) -> dict:
    """
    Call Kraken AddOrder endpoint.

    Args:
        api_key: Kraken API key
        api_secret: Kraken API secret
        pair: Trading pair (e.g. XBTUSD)
        side: "buy" or "sell"
        ordertype: "market" or "limit"
        volume: Order size in base asset
        price: Limit price (required for limit, ignored for market)
        validate: If True, Kraken validates only; no order placed

    Returns:
        Kraken response dict with result/error.

    Raises:
        RuntimeError: on HTTP/network error or API error response.
    """
    data: dict = {
        "pair": pair,
        "type": side,
        "ordertype": ordertype,
        "volume": str(volume),
        "validate": "true" if validate else "false",
    }
    if ordertype == "limit" and price is not None:
        data["price"] = str(price)
    resp = post_private("/0/private/AddOrder", data, api_key, api_secret, timeout=timeout)
    errors = resp.get("error", [])
    if errors and isinstance(errors, list) and len(errors) > 0:
        msg = errors[0] if isinstance(errors[0], str) else str(errors)
        raise RuntimeError(f"Kraken API error: {msg}")
    return resp


def cancel_order(
    api_key: str,
    api_secret: str,
    txid: str,
    timeout: int = 30,
) -> dict:
    """
    Call Kraken CancelOrder endpoint.

    Args:
        api_key: Kraken API key
        api_secret: Kraken API secret
        txid: Order transaction ID to cancel

    Returns:
        Kraken response dict with result/error.

    Raises:
        RuntimeError: on HTTP/network error or API error response.
    """
    data = {"txid": txid}
    resp = post_private("/0/private/CancelOrder", data, api_key, api_secret, timeout=timeout)
    errors = resp.get("error", [])
    if errors and isinstance(errors, list) and len(errors) > 0:
        msg = errors[0] if isinstance(errors[0], str) else str(errors)
        raise RuntimeError(f"Kraken API error: {msg}")
    return resp
