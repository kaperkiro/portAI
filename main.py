from dotenv import load_dotenv
import os
import classes as cl
from datetime import datetime, timedelta
import time
import json
from google import genai
import AI_calls as AIC
import api


LOOKOUT_LIST = []


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


def analyzeAIResult(AIResult):
    if AIResult == None or AIResult == "no opportunity":
        return
    else:
        ticker = AIResult["ticker"]
        qty = AIResult["buy_in_quantity"]
        take_profit1 = {"limit_price": AIResult["take_profit_1"]}
        take_profit2 = {"limit_price": AIResult["take_profit_2"]}
        stop_loss = AIResult["stop_loss"]
        # place two order to simulate the double take profit levels:
        api.bracketBuy(ticker, qty, take_profit1, stop_loss)
        api.bracketBuy(ticker, qty, take_profit2, stop_loss)
    return


RUN_TIMES = ["16:29", "16:32"]  # local time; edit this list to change runs per day
SMALL_TASK_INTERVAL_MINUTES = 1  # change this to adjust the small task cadence
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


def _run_daily_tasks(portfolio: cl.Portfolio, client) -> None:
    print("testing daily task")
    # stockJson = daily_market_analysis(client)
    # if stockJson is not None:
    #     stock = add_market_analysis_stock(stockJson)
    #     if stock is not None:
    #         LOOKOUT_LIST.append(stock)
    # cl.save_portfolio_to_json(portfolio, "data.json")


def _run_small_tasks(portfolio: cl.Portfolio) -> None:
    print("testing smaller task")
    # add small, frequent tasks here
    # cl.save_portfolio_to_json(portfolio, "data.json")


def _main() -> None:
    load_dotenv()  # reads .env in current working dir
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    AIport = cl.Portfolio()
    run_times = sorted(RUN_TIMES)

    next_daily_run = _next_daily_run(datetime.now(), run_times)
    next_small_run = datetime.now() + timedelta(minutes=SMALL_TASK_INTERVAL_MINUTES)

    while True:
        now = datetime.now()
        if now >= next_daily_run:
            _run_daily_tasks(AIport, client)
            next_daily_run = _next_daily_run(datetime.now(), run_times)

        if now >= next_small_run:
            _run_small_tasks(AIport)
            next_small_run = datetime.now() + timedelta(
                minutes=SMALL_TASK_INTERVAL_MINUTES
            )

        next_wake = min(next_daily_run, next_small_run)
        time.sleep(max(0.0, (next_wake - datetime.now()).total_seconds()))


if __name__ == "__main__":
    # _main()
    load_dotenv()  # reads .env in current working dir
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    # init alpaca trading client:
    alpaca_client = api.init_client()
    current_port_state = api.alpaca_portfolio_context(alpaca_client)
    print(current_port_state)
    res = AIC.daily_market_analysis(client, current_port_state)
    print(res)
    # # resObj = json.loads(res)
    # # analyzeAIResult(resObj)
