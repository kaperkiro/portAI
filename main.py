from dotenv import load_dotenv
import os
import classes as cl
from datetime import datetime, timedelta
import time
import json
import ast
import logging
from google import genai
import AI_calls as AIC
import api

logger = logging.getLogger(__name__)

# ===== Gemini rate limiting =====
GEMINI_DAILY_LIMIT = 100
_gemini_calls_today = 0
_gemini_day = datetime.now().date()


def _check_gemini_rate_limit():
    global _gemini_calls_today, _gemini_day

    today = datetime.now().date()

    # Reset counter on new day
    if today != _gemini_day:
        _gemini_day = today
        _gemini_calls_today = 0
        logger.info("Gemini daily rate limit reset")

    # If limit hit → sleep until tomorrow
    if _gemini_calls_today >= GEMINI_DAILY_LIMIT:
        tomorrow = datetime.combine(today + timedelta(days=1), datetime.min.time())
        sleep_seconds = (tomorrow - datetime.now()).total_seconds()

        logger.warning(
            "Gemini daily limit reached (%s calls). Sleeping until %s",
            GEMINI_DAILY_LIMIT,
            tomorrow.strftime("%Y-%m-%d %H:%M"),
        )

        time.sleep(max(0.0, sleep_seconds))

        # Reset after sleep
        _gemini_day = datetime.now().date()
        _gemini_calls_today = 0


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

    stock = cl.Stock(
        symbol=str(analysis["ticker"]),
        quantity=quantity,
        value=buy_amount,
        avg_buy_in_price=buy_in_price,
        buy_in_timestamp=datetime.now().isoformat(),
        stop_loss=stop_loss,
        take_profits=take_profits,
        state="watch",
    )
    return stock


def analyzeAIResult(AIResult, trading_client):
    if AIResult is None:
        return
    if isinstance(AIResult, str) and AIResult.strip().lower() == "no opportunity":
        return
    if not isinstance(AIResult, dict):
        return
    elif (
        "text" in AIResult
        and isinstance(AIResult["text"], str)
        and AIResult["text"].strip().lower() == "no opportunity"
    ):
        return
    else:
        required_keys = {
            "ticker",
            "buy_in_quantity",
            "take_profit_1",
            "take_profit_2",
            "stop_loss",
        }
        if not required_keys.issubset(AIResult.keys()):
            return

        ticker = AIResult["ticker"]
        qty = AIResult["buy_in_quantity"]
        take_profit1 = {"limit_price": int(AIResult["take_profit_1"])}
        take_profit2 = {"limit_price": int(AIResult["take_profit_2"])}
        stop_loss = {"stop_price": int(AIResult["stop_loss"])}
        # place two order to simulate the double take profit levels:
        if qty <= 1:
            api.bracketBuy(ticker, qty, take_profit1, stop_loss, trading_client)
        else:
            api.bracketBuy(ticker, qty / 2, take_profit1, stop_loss, trading_client)
            api.bracketBuy(ticker, qty / 2, take_profit2, stop_loss, trading_client)
    return


RUN_TIMES = [
    "11:25",
    "17:30",
    "19:30",
    "22:00",
]  # local time; edit this list to change runs per day
SMALL_TASK_INTERVAL_MINUTES = 1000  # change this to adjust the small task cadence
# change this to us times maybe? as the alpaca only supports us stocks for some reason :/


def _next_daily_run(now: datetime, run_times: list[str]) -> datetime:
    today = now.date()
    for t in run_times:
        h, m = (int(x) for x in t.split(":"))
        candidate = datetime.combine(today, datetime.min.time()) + timedelta(
            hours=h, minutes=m
        )
        if candidate > now:
            return candidate
    h, m = (int(x) for x in run_times[0].split(":"))
    return datetime.combine(today + timedelta(days=1), datetime.min.time()) + timedelta(
        hours=h, minutes=m
    )


def handle_daily_sell(stocks, alpaca_client):
    if stocks:
        for item in stocks:
            try:
                api.sellStock(item[0], item[1], alpaca_client)
            except Exception as e:
                logger.error("Error while selling %s: %s", item[0], e)
    return


def _run_daily_tasks(client, alpaca_client) -> None:
    global _gemini_calls_today
    logger.info(f"Running daily task cycle with timestamp: {datetime.now()}")
    current_port_state = api.alpaca_portfolio_context(alpaca_client)
    # ---- DAILY PORT ANALYSIS (Gemini call) ----
    _check_gemini_rate_limit()
    _gemini_calls_today += 1
    daily_port_res_raw = AIC.daily_port_analysis(current_port_state, client)
    daily_port_res_json = ast.literal_eval(daily_port_res_raw)
    handle_daily_sell(daily_port_res_json, alpaca_client)
    # ---- DAILY MARKET ANALYSIS (Gemini call) ----
    buying_power = api.get_portf_buying_power(alpaca_client)
    _check_gemini_rate_limit()
    _gemini_calls_today += 1
    res = AIC.daily_market_analysis(client, current_port_state, buying_power)
    result_state = (
        "no opportunity"
        if isinstance(res, str) and res.strip().lower() == "no opportunity"
        else "opportunity returned"
    )
    logger.info("Daily market analysis completed: %s", result_state)
    resObj = cl.parse_ai_response_payload(res)
    analyzeAIResult(resObj, alpaca_client)


def _run_small_tasks() -> None:
    logger.info("Running small task cycle")
    # add small, frequent tasks here
    # cl.save_portfolio_to_json(portfolio, "data.json")


def _main() -> None:
    _setup_logging()
    logger.info("Starting scheduler")

    # load AI:
    load_dotenv()  # reads .env in current working dir
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    # init alpaca trading client:
    alpaca_client = api.init_client()
    run_times = sorted(RUN_TIMES)

    next_daily_run = _next_daily_run(datetime.now(), run_times)
    next_small_run = datetime.now() + timedelta(minutes=SMALL_TASK_INTERVAL_MINUTES)

    while True:
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
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
            except Exception as e:
                logger.error("Error in daily task cycle: %s", e)
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
    _main()
