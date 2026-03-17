"""Strategy logic."""

MIN_SIGNAL_STRENGTH = 3.0


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
            "signal_strength": None,
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
            "signal_strength": None,
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
    signal_strength = (
        -momentum / momentum_threshold if momentum_threshold > 0 else 0.0
    )

    if momentum < -momentum_threshold and book_imbalance > 0.02:
        if abs(signal_strength) < MIN_SIGNAL_STRENGTH:
            return {
                "action": "hold",
                "reason": "weak_signal_filtered",
                "signal_strength": signal_strength,
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
        return {
            "action": "buy",
            "reason": "buy mean-reversion signal",
            "signal_strength": signal_strength,
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
        if abs(signal_strength) < MIN_SIGNAL_STRENGTH:
            return {
                "action": "hold",
                "reason": "weak_signal_filtered",
                "signal_strength": signal_strength,
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
        return {
            "action": "sell",
            "reason": "sell mean-reversion signal",
            "signal_strength": signal_strength,
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
        "signal_strength": None,
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
