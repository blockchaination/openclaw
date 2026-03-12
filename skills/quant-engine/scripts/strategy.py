"""Strategy logic."""


def maker_first_mean_reversion(
    mid_price: float,
    spread: float,
    book_imbalance: float,
    momentum: float,
    volatility: float,
) -> dict:
    """
    Maker-first mean-reversion strategy.

    Pure decision function: no I/O, no order placement. Returns a structured
    decision dict with action (buy/sell/hold), reason, inputs snapshot, and
    maker_hint for downstream order logic.

    Rules:
    - Hold if mid_price <= 0, spread < 0, or volatility < 0.
    - momentum_threshold = max(volatility * 0.15, 0.5).
    - Buy if momentum < -momentum_threshold and book_imbalance > 0.02.
    - Sell if momentum > momentum_threshold and book_imbalance < -0.02.
    - Otherwise hold.
    """
    if mid_price <= 0 or spread < 0:
        return {
            "action": "hold",
            "reason": "invalid mid_price or spread",
            "inputs": {
                "mid_price": mid_price,
                "spread": spread,
                "book_imbalance": book_imbalance,
                "momentum": momentum,
                "volatility": volatility,
                "momentum_threshold": 0.0,
            },
            "maker_hint": {"post_only": True, "preferred_side": "none"},
        }

    if volatility < 0:
        return {
            "action": "hold",
            "reason": "volatility < 0",
            "inputs": {
                "mid_price": mid_price,
                "spread": spread,
                "book_imbalance": book_imbalance,
                "momentum": momentum,
                "volatility": volatility,
                "momentum_threshold": 0.0,
            },
            "maker_hint": {"post_only": True, "preferred_side": "none"},
        }

    momentum_threshold = max(volatility * 0.15, 0.5)

    if momentum < -momentum_threshold and book_imbalance > 0.02:
        return {
            "action": "buy",
            "reason": "buy mean-reversion signal",
            "inputs": {
                "mid_price": mid_price,
                "spread": spread,
                "book_imbalance": book_imbalance,
                "momentum": momentum,
                "volatility": volatility,
                "momentum_threshold": momentum_threshold,
            },
            "maker_hint": {"post_only": True, "preferred_side": "bid"},
        }

    if momentum > momentum_threshold and book_imbalance < -0.02:
        return {
            "action": "sell",
            "reason": "sell mean-reversion signal",
            "inputs": {
                "mid_price": mid_price,
                "spread": spread,
                "book_imbalance": book_imbalance,
                "momentum": momentum,
                "volatility": volatility,
                "momentum_threshold": momentum_threshold,
            },
            "maker_hint": {"post_only": True, "preferred_side": "ask"},
        }

    return {
        "action": "hold",
        "reason": "no mean-reversion signal",
        "inputs": {
            "mid_price": mid_price,
            "spread": spread,
            "book_imbalance": book_imbalance,
            "momentum": momentum,
            "volatility": volatility,
            "momentum_threshold": momentum_threshold,
        },
        "maker_hint": {"post_only": True, "preferred_side": "none"},
    }
