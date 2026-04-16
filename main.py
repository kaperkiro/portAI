from dotenv import load_dotenv
import os
import sys
from datetime import datetime, timedelta
import time
import json
import ast
import logging

from google import genai

import AI_calls as AIC
import api
import classes as cl


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def add_market_analysis_stock(analysis_json: str | dict | None) -> cl.Stock | None:
    if analysis_json is None:
        return None

    if isinstance(analysis_json, str):
        text = analysis_json.strip()
        if not text or text.lower() == "null":
            return None
        try:
            analysis = json.loads(text)
        except json.JSONDecodeError:
            return None
    else:
        analysis = analysis_json

    if not isinstance(analysis, dict) or "ticker" not in analysis:
        return None

    def as_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    buy_in_price = as_float(analysis.get("buy_in_price"))
    stop_loss = as_float(analysis.get("stop_loss"))
    tp1 = as_float(analysis.get("take_profit_1"))
    tp2 = as_float(analysis.get("take_profit_2"))
    buy_amount = as_float(analysis.get("buy_in_ammount", analysis.get("buy_in_amount")))

    take_profits = []
    if tp1 is not None and tp2 is not None:
        take_profits = [cl.TakeProfit(p=tp1, pct=50.0), cl.TakeProfit(p=tp2, pct=50.0)]
    elif tp1 is not None:
        take_profits = [cl.TakeProfit(p=tp1, pct=100.0)]
    elif tp2 is not None:
        take_profits = [cl.TakeProfit(p=tp2, pct=100.0)]

    quantity = 0.0
    if buy_amount is not None and buy_in_price:
        quantity = buy_amount / buy_in_price

    return cl.Stock(
        symbol=str(analysis["ticker"]),
        quantity=quantity,
        value=buy_amount,
        avg_buy_in_price=buy_in_price,
        buy_in_timestamp=datetime.now().isoformat(),
        stop_loss=stop_loss,
        take_profits=take_profits,
        state="watch",
    )


def analyzeAIResult(ai_result, trading_client):
    if ai_result is None:
        return
    if isinstance(ai_result, str) and ai_result.strip().lower() == "no opportunity":
        return
    if not isinstance(ai_result, dict):
        return
    if (
        "text" in ai_result
        and isinstance(ai_result["text"], str)
        and ai_result["text"].strip().lower() == "no opportunity"
    ):
        return

    required_keys = {
        "ticker",
        "buy_in_quantity",
        "take_profit_1",
        "take_profit_2",
        "stop_loss",
    }
    if not required_keys.issubset(ai_result.keys()):
        return

    ticker = ai_result["ticker"]
    qty = ai_result["buy_in_quantity"]
    take_profit1 = {"limit_price": int(ai_result["take_profit_1"])}
    take_profit2 = {"limit_price": int(ai_result["take_profit_2"])}
    stop_loss = {"stop_price": int(ai_result["stop_loss"])}

    if qty <= 1:
        api.bracketBuy(ticker, qty, take_profit1, stop_loss, trading_client)
    else:
        api.bracketBuy(ticker, qty / 2, take_profit1, stop_loss, trading_client)
        api.bracketBuy(ticker, qty / 2, take_profit2, stop_loss, trading_client)


RUN_TIMES = [
    "11:40",
    "17:30",
    "19:30",
    "22:00",
]
SMALL_TASK_INTERVAL_MINUTES = 1000


def _next_daily_run(now: datetime, run_times: list[str]) -> datetime:
    today = now.date()
    for time_str in run_times:
        hour, minute = (int(x) for x in time_str.split(":"))
        candidate = datetime.combine(today, datetime.min.time()) + timedelta(
            hours=hour, minutes=minute
        )
        if candidate > now:
            return candidate

    hour, minute = (int(x) for x in run_times[0].split(":"))
    return datetime.combine(today + timedelta(days=1), datetime.min.time()) + timedelta(
        hours=hour, minutes=minute
    )


def handle_daily_sell(stocks, alpaca_client):
    if stocks:
        for item in stocks:
            try:
                api.sellStock(item[0], item[1], alpaca_client)
            except Exception as exc:
                logger.error("Error while selling %s: %s", item[0], exc)


def _build_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def _run_daily_tasks(client, alpaca_client) -> None:
    logger.info("Running daily task cycle with timestamp: %s", datetime.now())
    current_port_state = api.alpaca_portfolio_context(alpaca_client)

    daily_port_res_raw = AIC.daily_port_analysis(current_port_state, client)
    daily_port_res_json = ast.literal_eval(daily_port_res_raw)
    handle_daily_sell(daily_port_res_json, alpaca_client)

    buying_power = api.get_portf_buying_power(alpaca_client)
    res = AIC.daily_market_analysis(client, current_port_state, buying_power)
    result_state = (
        "no opportunity"
        if isinstance(res, str) and res.strip().lower() == "no opportunity"
        else "opportunity returned"
    )
    logger.info("Daily market analysis completed: %s", result_state)
    res_obj = cl.parse_ai_response_payload(res)
    analyzeAIResult(res_obj, alpaca_client)


def _run_small_tasks() -> None:
    logger.info("Running small task cycle")


def _run_daily_market_analysis_once() -> None:
    _setup_logging()
    client = _build_gemini_client()
    print(AIC.daily_market_analysis(client))


def _run_single_stock_analysis_once(stock_query: str) -> None:
    _setup_logging()
    client = _build_gemini_client()
    print(AIC.single_stock_analysis(stock_query, client))


def _print_usage() -> None:
    print("Usage:")
    print("  python main.py                   # run daily market analysis once")
    print("  python main.py daily-market-analysis")
    print("  python main.py analyze-stock AMD")
    print("  python main.py analyze-asset gold")
    print("  python main.py AMD               # shorthand single-asset analysis")
    print("  python main.py scheduler         # start the scheduler loop")


def _dispatch_cli(argv: list[str]) -> None:
    if not argv:
        _run_daily_market_analysis_once()
        return

    command = argv[0].strip().lower()

    if command in {"-h", "--help", "help"}:
        _print_usage()
        return

    if command == "scheduler":
        _main()
        return

    if command == "daily-market-analysis":
        _run_daily_market_analysis_once()
        return

    if command in {"analyze-stock", "analyze-asset"}:
        stock_query = " ".join(argv[1:]).strip()
        if not stock_query:
            raise SystemExit(
                "Usage: python main.py analyze-stock <ticker-company-or-asset-query>"
            )
        _run_single_stock_analysis_once(stock_query)
        return

    _run_single_stock_analysis_once(" ".join(argv).strip())


def _main() -> None:
    _setup_logging()
    logger.info("Starting scheduler")

    client = _build_gemini_client()
    alpaca_client = api.init_client()
    run_times = sorted(RUN_TIMES)

    next_daily_run = _next_daily_run(datetime.now(), run_times)
    next_small_run = datetime.now() + timedelta(minutes=SMALL_TASK_INTERVAL_MINUTES)

    while True:
        now = datetime.now()
        if now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            resume_at = datetime.combine(
                (now + timedelta(days=days_until_monday)).date(),
                datetime.min.time(),
            )
            logger.info(
                "Weekend detected, pausing scheduler until %s",
                resume_at.strftime("%Y-%m-%d %H:%M"),
            )
            time.sleep(max(0.0, (resume_at - now).total_seconds()))
            now = datetime.now()
            next_daily_run = _next_daily_run(now, run_times)
            next_small_run = now + timedelta(minutes=SMALL_TASK_INTERVAL_MINUTES)
            continue

        if now >= next_daily_run:
            try:
                _run_daily_tasks(client, alpaca_client)
            except Exception as exc:
                logger.error("Error in daily task cycle: %s", exc)
            finally:
                next_daily_run = _next_daily_run(datetime.now(), run_times)

        if now >= next_small_run:
            _run_small_tasks()
            next_small_run = datetime.now() + timedelta(
                minutes=SMALL_TASK_INTERVAL_MINUTES
            )

        next_wake = min(next_daily_run, next_small_run)
        time.sleep(max(0.0, (next_wake - datetime.now()).total_seconds()))


if __name__ == "__main__":
    _dispatch_cli(sys.argv[1:])
