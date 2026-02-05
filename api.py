from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderStatus
from dotenv import load_dotenv
import os
from datetime import datetime, timezone


def init_client():
    # paper=True enables paper trading

    load_dotenv()  # reads .env in current working dir
    alpaca_secret_key = os.getenv("alpaca_secret_key")
    alpaca_key = os.getenv("alpaca_key")

    trading_client = TradingClient(alpaca_key, alpaca_secret_key, paper=True)
    return trading_client


def alpaca_portfolio_context(trading_client: TradingClient) -> str:
    """
    Returns a compact, human-readable string describing the Alpaca portfolio.
    Designed to be pasted directly into an AI analysis prompt.
    """

    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # --------------------
    # ACCOUNT SUMMARY
    # --------------------
    acct = trading_client.get_account()

    lines.append("PORTFOLIO SNAPSHOT")
    lines.append(f"As of: {now}")
    lines.append("")
    lines.append("ACCOUNT")
    lines.append(f"Equity: {acct.equity} USD")
    lines.append(f"Cash: {acct.cash} USD")
    lines.append(f"Portfolio Value: {acct.portfolio_value} USD")
    lines.append(f"Buying Power: {acct.buying_power} USD")
    lines.append(f"Margin Multiplier: {acct.multiplier}x")
    lines.append(f"Pattern Day Trader: {acct.pattern_day_trader}")
    lines.append("")

    # --------------------
    # POSITIONS
    # --------------------
    positions = trading_client.get_all_positions()

    lines.append("OPEN POSITIONS")
    if not positions:
        lines.append("None")
    else:
        for p in positions:
            lines.append(
                f"{p.symbol} | "
                f"Qty: {p.qty} | "
                f"Avg: {p.avg_entry_price} | "
                f"Last: {p.current_price} | "
                f"Value: {p.market_value} | "
                f"Unrealized P/L: {p.unrealized_pl} ({float(p.unrealized_plpc) * 100:.2f}%)"
            )
    lines.append("")

    # --------------------
    # OPEN ORDERS
    # --------------------
    orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
    )

    lines.append("OPEN ORDERS")
    if not orders:
        lines.append("None")
    else:
        for o in orders:
            lines.append(
                f"{o.symbol} | "
                f"{o.side.upper()} | "
                f"{o.order_type} | "
                f"Qty: {o.qty or o.notional} | "
                f"Limit: {o.limit_price} | "
                f"Stop: {o.stop_price} | "
                f"Status: {o.status}"
            )
    lines.append("")

    # --------------------
    # RECENT FILLS
    # --------------------
    # alpaca-py 0.43.x does not expose account activities on TradingClient.
    # Approximate "recent fills" by looking at recently closed orders with fills.
    closed_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.CLOSED, nested=True, limit=50)
    )
    fills = [
        o
        for o in closed_orders
        if o.filled_at is not None
        and (o.status == OrderStatus.FILLED or o.status == OrderStatus.PARTIALLY_FILLED)
    ]
    fills.sort(key=lambda o: o.filled_at or o.submitted_at, reverse=True)
    fills = fills[:10]

    lines.append("RECENT FILLS")
    if not fills:
        lines.append("None")
    else:
        for f in fills:
            lines.append(
                f"{f.symbol} | "
                f"{f.side.upper()} | "
                f"Qty: {f.filled_qty or f.qty} | "
                f"Price: {f.filled_avg_price or f.limit_price} | "
                f"Time: {f.filled_at}"
            )

    # --------------------
    # FINAL STRING
    # --------------------
    return "\n".join(lines)


def buyStock(Ticker, qty):
    # tested working!
    buyOrder = MarketOrderRequest(
        symbol=Ticker,
        qty=qty,
        limit_price=500,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    buyResponse = trading_client.submit_order(buyOrder)
    return buyResponse


def sellStock(Ticker, qty):
    # Tested working!
    sellOrder = MarketOrderRequest(
        symbol=Ticker,
        qty=qty,
        limit_price=500,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    sellResponse = trading_client.submit_order(sellOrder)
    return sellResponse


def bracketBuy(Ticker, qty, take_profit, stop_loss):
    bracketOrder = MarketOrderRequest(
        symbol=Ticker,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )
    response = trading_client.submit_order(bracketOrder)
    return response
