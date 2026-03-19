"""Strategy logic."""

MIN_SIGNAL_STRENGTH = 1.5

# Spot state threshold: XBT < this => FLAT (entry only), XBT >= this => LONG (exit only)
MIN_XBT_TO_SELL = 0.0002


def maker_first_mean_reversion(
    mid_price: float,
    spread: float,
    book_imbalance: float,
    momentum: float,
    volatility: float,
    xbt_inventory: float,
    min_xbt_to_sell: float = MIN_XBT_TO_SELL,
) -> dict:
    """
    Spot-native mean-reversion strategy.

    Two-state model:
    - FLAT (xbt < min_xbt_to_sell): only evaluate BUY entry. Outputs: buy, hold.
    - LONG (xbt >= min_xbt_to_sell): only evaluate SELL exit. Outputs: sell, hold.

    Pure decision function: no I/O, no order placement. Returns a structured
    decision dict with action (buy/sell/hold), reason, inputs snapshot, and
    maker_hint for downstream order logic.
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
                "xbt_inventory": xbt_inventory,
                "spot_state": "flat",
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
                "xbt_inventory": xbt_inventory,
                "spot_state": "flat",
            },
            "maker_hint": {"post_only": True, "preferred_side": "none"},
        }

    spot_state = "long" if xbt_inventory >= min_xbt_to_sell else "flat"
    momentum_threshold = max(volatility * 0.15, 0.5)
    signal_strength = (
        -momentum / momentum_threshold if momentum_threshold > 0 else 0.0
    )

    inputs = {
        "mid_price": mid_price,
        "spread": spread,
        "book_imbalance": book_imbalance,
        "momentum": momentum,
        "volatility": volatility,
        "momentum_threshold": momentum_threshold,
        "xbt_inventory": xbt_inventory,
        "spot_state": spot_state,
    }

    if spot_state == "flat":
        # FLAT: only evaluate BUY entry. Never emit sell.
        if momentum < -momentum_threshold and book_imbalance > 0.02:
            if abs(signal_strength) < MIN_SIGNAL_STRENGTH:
                return {
                    "action": "hold",
                    "reason": "weak_signal_filtered",
                    "signal_strength": signal_strength,
                    "inputs": inputs,
                    "maker_hint": {"post_only": True, "preferred_side": "none"},
                }
            return {
                "action": "buy",
                "reason": "buy mean-reversion entry",
                "signal_strength": signal_strength,
                "inputs": inputs,
                "maker_hint": {"post_only": True, "preferred_side": "bid"},
            }
        return {
            "action": "hold",
            "reason": "no long-entry signal",
            "signal_strength": None,
            "inputs": inputs,
            "maker_hint": {"post_only": True, "preferred_side": "none"},
        }

    # LONG: only evaluate SELL exit. Never emit buy.
    if momentum > momentum_threshold and book_imbalance < -0.02:
        if abs(signal_strength) < MIN_SIGNAL_STRENGTH:
            return {
                "action": "hold",
                "reason": "weak_signal_filtered",
                "signal_strength": signal_strength,
                "inputs": inputs,
                "maker_hint": {"post_only": True, "preferred_side": "none"},
            }
        return {
            "action": "sell",
            "reason": "sell mean-reversion exit",
            "signal_strength": signal_strength,
            "inputs": inputs,
            "maker_hint": {"post_only": True, "preferred_side": "ask"},
        }
    return {
        "action": "hold",
        "reason": "no exit signal",
        "signal_strength": None,
        "inputs": inputs,
        "maker_hint": {"post_only": True, "preferred_side": "none"},
    }
