"""Paper broker: in-memory simulated order execution."""


class PaperBroker:
    def __init__(self, starting_cash_usd: float = 200.0) -> None:
        """
        Initialize broker state.

        Attributes:
        - cash_usd: available cash
        - position_units: current long position size
        - average_entry_price: volume-weighted average entry price
        - realized_pnl_usd: cumulative realized PnL
        - open_orders: list of resting limit orders
        - next_order_id: next order ID to assign
        """
        self.cash_usd = starting_cash_usd
        self.position_units = 0.0
        self.average_entry_price = 0.0
        self.realized_pnl_usd = 0.0
        self.open_orders: list[dict] = []
        self.next_order_id = 1

    def place_limit_order(
        self,
        side: str,
        price: float,
        size_units: float,
    ) -> dict:
        """
        Create a new resting limit order and append it to open_orders.

        Validation:
        - side must be "buy" or "sell"
        - price must be > 0
        - size_units must be > 0

        Returns:
        - On success: {"status": "open", "order": {...}}
        - On rejection: {"status": "rejected", "reason": "..."}
        """
        if side not in ("buy", "sell"):
            return {"status": "rejected", "reason": "invalid side"}
        if price <= 0 or size_units <= 0:
            return {"status": "rejected", "reason": "invalid price or size"}

        order_id = self.next_order_id
        self.next_order_id += 1
        order = {
            "order_id": order_id,
            "side": side,
            "price": price,
            "size_units": size_units,
        }
        self.open_orders.append(order)
        return {"status": "open", "order": order}

    def cancel_order(self, order_id: int) -> dict:
        """
        Cancel an open order by ID.

        Returns:
        - {"status": "cancelled", "order_id": order_id} if found
        - {"status": "not_found", "order_id": order_id} otherwise
        """
        for i, o in enumerate(self.open_orders):
            if o.get("order_id") == order_id:
                self.open_orders.pop(i)
                return {"status": "cancelled", "order_id": order_id}
        return {"status": "not_found", "order_id": order_id}

    def process_market_tick(self, bid: float, ask: float) -> list[dict]:
        """
        Check all open orders against current market bid/ask and fill eligible ones.

        Fill rules:
        - Buy order fills if ask <= order price (full fill).
        - Sell order fills if bid >= order price. If position < order size,
          fill only up to current position. If fill size is 0, leave order open.

        Filled orders are removed from open_orders. Returns a list of fill event dicts.
        """
        fills: list[dict] = []
        remaining: list[dict] = []

        for order in self.open_orders:
            side = order.get("side")
            price = order.get("price")
            size_units = order.get("size_units")
            order_id = order.get("order_id")

            if side == "buy" and ask <= price:
                self._apply_fill("buy", price, size_units)
                fills.append({
                    "status": "filled",
                    "order_id": order_id,
                    "side": "buy",
                    "price": price,
                    "size_units": size_units,
                })
            elif side == "sell" and bid >= price:
                sell_units = min(size_units, self.position_units)
                if sell_units <= 0:
                    remaining.append(order)
                else:
                    self._apply_fill("sell", price, sell_units)
                    fills.append({
                        "status": "filled",
                        "order_id": order_id,
                        "side": "sell",
                        "price": price,
                        "size_units": sell_units,
                    })
            else:
                remaining.append(order)

        self.open_orders = remaining
        return fills

    def _apply_fill(self, side: str, price: float, size_units: float) -> None:
        """
        Update broker state after a fill.

        Long-only: buys increase position; sells decrease up to current position.
        For sells, size_units is the actual filled amount (min of order size and position).

        Buy: update weighted average entry, increase position, decrease cash.
        Sell: realize PnL, decrease position, increase cash; reset average_entry if flat.
        """
        if side == "buy":
            cost = price * size_units
            total_cost_before = self.average_entry_price * self.position_units
            self.position_units += size_units
            if self.position_units > 0:
                self.average_entry_price = (total_cost_before + cost) / self.position_units
            self.cash_usd -= cost
        elif side == "sell":
            sell_units = min(size_units, self.position_units)
            if sell_units <= 0:
                return
            pnl = (price - self.average_entry_price) * sell_units
            self.realized_pnl_usd += pnl
            self.position_units -= sell_units
            self.cash_usd += price * sell_units
            if self.position_units <= 0:
                self.average_entry_price = 0.0

    def mark_to_market(self, mid_price: float) -> dict:
        """
        Return unrealized and total PnL snapshot.

        unrealized_pnl_usd = (mid_price - average_entry_price) * position_units
        total_pnl_usd = realized_pnl_usd + unrealized_pnl_usd
        If no position, unrealized_pnl_usd = 0.0
        """
        if self.position_units <= 0:
            unrealized = 0.0
        else:
            unrealized = (mid_price - self.average_entry_price) * self.position_units
        total = self.realized_pnl_usd + unrealized
        return {
            "mid_price": mid_price,
            "position_units": self.position_units,
            "average_entry_price": self.average_entry_price,
            "unrealized_pnl_usd": unrealized,
            "realized_pnl_usd": self.realized_pnl_usd,
            "total_pnl_usd": total,
        }

    def pnl_summary(self, mid_price: float) -> dict:
        """
        Return a compact broker summary with cash, position, PnL, and open order count.
        """
        mtm = self.mark_to_market(mid_price)
        return {
            "cash_usd": self.cash_usd,
            "position_units": self.position_units,
            "average_entry_price": self.average_entry_price,
            "open_orders_count": len(self.open_orders),
            "realized_pnl_usd": self.realized_pnl_usd,
            "unrealized_pnl_usd": mtm["unrealized_pnl_usd"],
            "total_pnl_usd": mtm["total_pnl_usd"],
        }


if __name__ == "__main__":
    # 1. create broker with 200.0 cash
    broker = PaperBroker(starting_cash_usd=200.0)

    # 2. place one buy limit order
    r = broker.place_limit_order("buy", 100.0, 1.0)
    print("place buy:", r)

    # 3. process a market tick that fills it
    fills = broker.process_market_tick(bid=99.0, ask=100.0)
    print("fills:", fills)

    # 4. print pnl_summary at a higher mid price
    print("pnl_summary (mid=105):", broker.pnl_summary(105.0))

    # 5. place one sell limit order
    r = broker.place_limit_order("sell", 105.0, 1.0)
    print("place sell:", r)

    # 6. process a market tick that fills it
    fills = broker.process_market_tick(bid=105.0, ask=106.0)
    print("fills:", fills)

    # 7. print final pnl_summary
    print("final pnl_summary:", broker.pnl_summary(105.0))
