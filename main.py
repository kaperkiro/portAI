from dotenv import load_dotenv
import os
import classes as cl
from datetime import datetime, timedelta
import time
import yfinance as yf
import json
from google import genai
from google.genai import types

LOOKOUT_LIST = []


def daily_port_analysis(portfolio: cl.Portfolio, client):
    prompt = f"""You are an AI portfolio trading manager for a cash-only account.
    You can only buy and sell stocks and funds (no shorting, no leverage).

    Input:
    You receive a portfolio object with cash, value, stocks, and funds.
    Each stock/fund includes quantity, value, buy price, stop loss, take profit levels, and state.
    Provided data is authoritative.

    Task:
    Analyze the portfolio and identify any stocks or funds that should be SOLD or PARTIALLY SOLD today.

    Rules:
    - Use provided portfolio and position data first.
    - Use web search ONLY to check for material news, earnings, guidance, regulatory, or macro events.
    - Do NOT guess prices or invent data.
    - If no relevant news is found, state that clearly.
    - Recommend SELL or PARTIAL SELL only if:
    - stop loss is hit or invalidated,
    - take profit level is reached,
    - new information breaks the investment thesis,
    - risk has materially increased,
    - position is oversized and risk reduction is justified.
    - Be conservative; ignore short-term noise.
    - For funds, consider annual fee, overlap, and long-term suitability.

    Output (STRICT):
    1) Summary: one short paragraph answering if any positions should be sold.
    2) Action List (only if applicable), format:

    SYMBOL:
    Action: SELL | PARTIAL SELL | HOLD
    Reason:
    - bullet reasons (data + news)
    Confidence: LOW | MEDIUM | HIGH

    3) If no sells are needed, state:
    "No positions require selling today based on available data."

    4) Risk Notes (optional): upcoming earnings, macro risks, data uncertainty.

    Constraints:
    - No disclaimers.
    - If uncertain, say "uncertain".
    - Do not output anything outside this structure.
    
    This is the portfolio information: {portfolio.to_dict}

    the return a list of tuples. Where each tuple contains the ticker of the stock to sell, and the second position of the tuple is
    how much % to sell. If there are no stocks to sell, return an empty list. Do not return anything else!
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        tools=[{"google_search": {}}],
    )


def daily_market_analysis(client):
    prompt = """
    You are a professional portfolio manager + short-term swing trader (3–6 months). Your goal is to maximize income and value of a stock trading portfolio.
    Your job: scan current market conditions and identify the single BEST stock opportunity right now (highest risk-adjusted edge). 
    Use real analysis: catalyst/news, fundamentals (quick sanity check), sector/market regime, and technicals (trend, structure, volatility, liquidity). 
    Do NOT invent prices. Use live/most recent data from tools provided by the environment. If you cannot verify current price/levels, output nothing.
    If there are not any stock currently worth investing in, return the text nothing to invest in.  

    Universe: large/mega cap, highly liquid US/EU stocks (avoid microcaps, low volume, extreme spreads). Prefer tickers with clear catalysts and clean technical structure. Avoid earnings within 7 calendar days unless the trade is explicitly earnings-driven and risk is defined.

    Risk rules:
    - Provide a clear entry (buy_in_price) close to current price or a well-defined breakout/pullback trigger.
    - stop_loss must be a real invalidation level (structure/volatility-based), not arbitrary.
    - take_profit_1 and take_profit_2 must be realistic from ATR/structure and give favorable R:R (ideally TP2 >= 2R).
    - Provide rationale in your own reasoning, but DO NOT output it unless asked.

    Output rules (IMPORTANT):
    - If there is NO high-quality opportunity, output exactly: null
    - If there IS an opportunity, output exactly one object in this schema (no extra keys, no comments, no markdown):

    Output the information like this:

    "ticker": STRING,
    "current_price": INTEGER,
    "current_price_date": STRING,
    "order_time_horizon": INTEGER,
    "buy_in_price": NUMBER,
    "stop_loss": NUMBER,
    "take_profit_1": NUMBER,
    "take_profit_2": NUMBER,
    "buy_in_ammount": INTEGER
    

    Constraints:
    - time_horizon_days is 30–180.
    - buy in ammount is the ammount of currency to buy stocks for eg. 150 would be buy this stock for 150 of the currency is used in the portfolio.
    - Prices must be consistent: stop_loss < buy_in_price < take_profit_1 < take_profit_2 for a long idea.
    - Only output long ideas unless the user explicitly asks for shorts.

    The output should only be the as the given example with the data filled in. No extra characters!
    Now perform the scan and return the output."""

    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    config = types.GenerateContentConfig(tools=[grounding_tool], temperature=0.1)

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-09-2025",
        contents=prompt,
        config=config,
    )
    print(response.text)
    return json.loads(response.text)


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
    _main()
