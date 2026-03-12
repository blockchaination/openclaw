"""Feature computation from market data."""

import statistics


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


def spread(bids: list, asks: list) -> float:
    """
    Bid-ask spread from best bid and best ask.

    bids and asks are lists of dicts with "price", "volume", "timestamp".
    Best bid = bids[0], best ask = asks[0]. Returns ask - bid as float.
    Returns 0.0 if data is missing or invalid.
    """
    if not bids or not asks:
        return 0.0
    best_bid = _safe_float(bids[0].get("price")) if isinstance(bids[0], dict) else 0.0
    best_ask = _safe_float(asks[0].get("price")) if isinstance(asks[0], dict) else 0.0
    if best_bid <= 0 or best_ask <= 0:
        return 0.0
    return best_ask - best_bid


def mid_price(bids: list, asks: list) -> float:
    """
    Mid price from best bid and best ask.

    Returns (best_bid + best_ask) / 2. Returns 0.0 if data is missing or invalid.
    """
    if not bids or not asks:
        return 0.0
    best_bid = _safe_float(bids[0].get("price")) if isinstance(bids[0], dict) else 0.0
    best_ask = _safe_float(asks[0].get("price")) if isinstance(asks[0], dict) else 0.0
    if best_bid <= 0 or best_ask <= 0:
        return 0.0
    return (best_bid + best_ask) / 2.0


def book_imbalance(bids: list, asks: list, depth: int = 5) -> float:
    """
    Order book imbalance over the first N levels on each side.

    Sums bid volumes and ask volumes from the first `depth` levels.
    Returns (bid_volume - ask_volume) / (bid_volume + ask_volume).
    Returns 0.0 if denominator is zero or data is invalid.
    """
    if not isinstance(bids, list) or not isinstance(asks, list) or depth < 1:
        return 0.0
    bid_vol = 0.0
    for i, level in enumerate(bids):
        if i >= depth:
            break
        if isinstance(level, dict):
            bid_vol += _safe_float(level.get("volume"))
    ask_vol = 0.0
    for i, level in enumerate(asks):
        if i >= depth:
            break
        if isinstance(level, dict):
            ask_vol += _safe_float(level.get("volume"))
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def short_momentum(prices: list, lookback: int) -> float:
    """
    Short-term momentum: latest price minus price N steps ago.

    prices is a list of floats (or coercible to float). Returns
    prices[-1] - prices[-(lookback+1)]. Returns 0.0 if insufficient data.
    """
    if not isinstance(prices, list) or lookback < 1:
        return 0.0
    n = lookback + 1
    if len(prices) < n:
        return 0.0
    latest = _safe_float(prices[-1])
    past = _safe_float(prices[-(n)])
    return latest - past


def volatility(prices: list, lookback: int) -> float:
    """
    Simple sample standard deviation of the last `lookback` prices.

    Uses statistics.stdev. Returns 0.0 if fewer than 2 points in the window
    or data is invalid.
    """
    if not isinstance(prices, list) or lookback < 2:
        return 0.0
    window = prices[-lookback:] if len(prices) >= lookback else prices
    if len(window) < 2:
        return 0.0
    try:
        floats = [_safe_float(p) for p in window]
        return statistics.stdev(floats)
    except statistics.StatisticsError:
        return 0.0
