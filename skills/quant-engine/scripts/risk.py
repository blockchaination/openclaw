"""Risk limits and pure risk-policy functions for paper trading."""

MAX_POSITION_USD = 50.0
MAX_DAILY_LOSS_USD = 10.0
MAX_OPEN_ORDERS = 2


def within_position_limit(
    current_position_usd: float,
    proposed_order_usd: float,
    max_position_usd: float = MAX_POSITION_USD,
) -> bool:
    """Return True if abs(current_position_usd + proposed_order_usd) <= max_position_usd."""
    return abs(current_position_usd + proposed_order_usd) <= max_position_usd


def within_daily_loss_limit(
    daily_pnl_usd: float,
    max_daily_loss_usd: float = MAX_DAILY_LOSS_USD,
) -> bool:
    """Return True if daily_pnl_usd > -max_daily_loss_usd."""
    return daily_pnl_usd > -max_daily_loss_usd


def can_open_new_order(
    open_orders_count: int,
    max_open_orders: int = MAX_OPEN_ORDERS,
) -> bool:
    """Return True if open_orders_count < max_open_orders."""
    return open_orders_count < max_open_orders


def allow_trade(
    current_position_usd: float,
    proposed_order_usd: float,
    daily_pnl_usd: float,
    open_orders_count: int,
    max_position_usd: float = MAX_POSITION_USD,
    max_daily_loss_usd: float = MAX_DAILY_LOSS_USD,
    max_open_orders: int = MAX_OPEN_ORDERS,
) -> dict:
    """
    Evaluate all risk checks and return a structured decision dict.

    Priority: daily_loss_limit, then position_limit, then open_order_limit.
    """
    daily_ok = within_daily_loss_limit(daily_pnl_usd, max_daily_loss_usd)
    position_ok = within_position_limit(
        current_position_usd, proposed_order_usd, max_position_usd
    )
    open_orders_ok = can_open_new_order(open_orders_count, max_open_orders)

    if not daily_ok:
        reason = "daily_loss_limit"
    elif not position_ok:
        reason = "position_limit"
    elif not open_orders_ok:
        reason = "open_order_limit"
    else:
        reason = "allowed"

    return {
        "allowed": daily_ok and position_ok and open_orders_ok,
        "reason": reason,
        "checks": {
            "position_limit": position_ok,
            "daily_loss_limit": daily_ok,
            "open_order_limit": open_orders_ok,
        },
        "inputs": {
            "current_position_usd": current_position_usd,
            "proposed_order_usd": proposed_order_usd,
            "daily_pnl_usd": daily_pnl_usd,
            "open_orders_count": open_orders_count,
            "max_position_usd": max_position_usd,
            "max_daily_loss_usd": max_daily_loss_usd,
            "max_open_orders": max_open_orders,
        },
    }


if __name__ == "__main__":
    # allowed trade
    r = allow_trade(0.0, 20.0, 0.0, 1)
    print("allowed trade:", r["allowed"], r["reason"])

    # blocked by daily loss
    r = allow_trade(0.0, 20.0, -15.0, 0)
    print("blocked by daily loss:", r["allowed"], r["reason"])

    # blocked by position limit
    r = allow_trade(40.0, 20.0, 0.0, 0)
    print("blocked by position limit:", r["allowed"], r["reason"])

    # blocked by open orders
    r = allow_trade(0.0, 20.0, 0.0, 2)
    print("blocked by open orders:", r["allowed"], r["reason"])
