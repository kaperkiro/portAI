from dotenv import load_dotenv
import os
import classes as cl
from datetime import datetime, timedelta, timezone
import time
import helpers as hp
import yfinance as yf
import json
from google import genai
import re
from google.genai import types

LOOKOUT_LIST = []


# AI helper tools:
def get_current_ticker_data(ticker: str) -> str:
    data = yf.Ticker(ticker)
    print(f"Ran ticker data with ticker: {ticker}")
    # implement some error handling
    info_filtered = hp.YFInfoFundamentals.from_yfinance_info(data.info).to_dict()
    fast_filtered = hp.YFFastInfoSnapshot.from_yfinance_fast_info(
        data.fast_info
    ).to_dict()
    earnings = hp.YFEarningsEvent.from_calendar_or_earnings_dates(
        data.calendar, data.earnings_dates
    ).to_dict()
    actions = hp.YFCorporateActions.from_actions_dividends_splits(
        data.actions, data.dividends, data.splits
    ).to_dict()
    analyst = hp.YFAnalystSignal.from_recommendations(data.recommendations).to_dict()

    price_targets = data.get_analyst_price_targets()

    payload = hp.format_ticker_data_for_ai(
        info_filtered, fast_filtered, earnings, actions, analyst, price_targets
    )
    ticker_line = f"ticker: {str(ticker).upper()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    as_of_line = f"as_of: {timestamp}"
    output = (
        f"{ticker_line}\n{as_of_line}\n{payload}"
        if payload
        else f"{ticker_line}\n{as_of_line}"
    )
    return output


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


def _extract_tickers(text: str) -> list[str]:
    """
    Extract comma-separated tickers (best-effort). Accepts formats like:
    'AAPL, MSFT, NVDA' or 'AAPL,MSFT,NVDA'
    Returns a unique, ordered list.
    """
    if not text:
        return []
    if text.strip().lower() == "no opportunity":
        return []

    # Keep only plausible ticker tokens (1-6 letters, allow . for EU tickers like ASML.AS if it appears)
    # But your step1 says uppercase only; we normalize anyway.
    raw = [t.strip().upper() for t in text.split(",")]
    tickers = []
    seen = set()
    for t in raw:
        t = t.replace(" ", "")
        if not t:
            continue
        # allow AAPL, MSFT, NVDA, and optionally BRK.B style, or ASML.AS style
        if re.fullmatch(r"[A-Z]{1,6}([.\-][A-Z]{1,4})?", t):
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers


def daily_market_analysis(client):
    # -----------------------
    # STEP 1: Google Search ONLY
    # -----------------------
    prompt1 = """
    You are a professional portfolio manager and short-term swing trader (3–6 months).

    Task:
    Scan current global equity markets using recent news and macro information.
    Analyze:
    - Market regime (risk-on / risk-off, rates, inflation, geopolitics)
    - Sector rotation and relative strength
    - Major catalysts (earnings trends, guidance, AI, energy, defense, healthcare, regulation)
    - Liquidity and institutional relevance

    Universe:
    - Large and mega-cap EU equities only
    - Highly liquid stocks (no microcaps, no thin volume)
    - Avoid stocks with earnings within the next 7 calendar days unless the catalyst is earnings-driven

    Objective:
    Identify up to 5 of the most interesting stock tickers that may offer a high risk-adjusted swing trade opportunity over the next 30–180 days.

    Rules:
    - Do NOT invent prices or technical levels
    - Do NOT propose entries, stops, or targets
    - Do NOT call any functions
    - Base conclusions only on recent, verifiable information

    Output rules (IMPORTANT):
    If NO stocks are interesting, output exactly:
    no opportunity

    If stocks ARE interesting, output EXACTLY in this 2-line format with no extra text:

    MARKET_CONTEXT: <one short paragraph, max 60 words>
    TICKERS: <comma-separated list of ticker symbols in uppercase>

    Example:
    MARKET_CONTEXT: Risk-on tone as rate fears ease; AI capex remains dominant; defensives lag; energy mixed; volatility moderate.
    TICKERS: AAPL, MSFT, NVDA
    """.strip()

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config1 = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.1,
    )

    resp1 = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt1,
        config=config1,
    )

    step1_text = (resp1.text or "").strip()
    if step1_text.lower() == "no opportunity":
        return "no opportunity"

    # Parse Step 1 output
    market_context = ""
    tickers_line = ""

    for line in step1_text.splitlines():
        line = line.strip()
        if line.upper().startswith("MARKET_CONTEXT:"):
            market_context = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TICKERS:"):
            tickers_line = line.split(":", 1)[1].strip()

    # Fallback: if the model didn't follow format, assume entire output is tickers
    if not tickers_line and "TICKERS:" not in step1_text.upper():
        tickers_line = step1_text

    tickers = _extract_tickers(tickers_line)
    if not tickers:
        return "no opportunity"

    print(tickers)
    # (Optional) hard cap at 5 to match your requirement
    # tickers = tickers[:5]

    # -----------------------
    # STEP 2: Function calling ONLY (analyze ALL tickers, pick 1 best)
    # -----------------------
    # IMPORTANT: we pass market_context forward so step2 has continuity.
    prompt2 = f"""
    You are a professional portfolio manager and short-term swing trader (3–6 months).

    Market context from the prior scan (use this to anchor regime/sector assumptions):
    {market_context}

    You will be given:
    - A list of stock ticker symbols
    - Access to the function get_current_ticker_data for live market data

    Your goal:
    Determine which ticker represents the SINGLE BEST risk-adjusted LONG opportunity right now (30–180 days), or output "no opportunity".

    STRICT TOOL RULES (MANDATORY):
    - You MUST call get_current_ticker_data at least once for EACH ticker in the list.
    - If you cannot retrieve valid data for a ticker, discard it.
    - Do NOT invent prices or levels.
    - Use the retrieved current price and date in your final output.

    Analysis requirements (internal reasoning only):
    - Validate catalyst plausibility from market context + typical catalysts
    - Sanity-check fundamentals at a high level (no deep modeling)
    - Market/sector regime alignment
    - Technical structure: trend + key structure + volatility logic (ATR/structure-based)
    - Liquidity suitability for large/mega-cap swing trade

    Risk rules:
    - Provide a clear entry (buy_in_price) close to current price or a well-defined breakout/pullback trigger.
    - stop_loss must be a real invalidation level (structure/volatility-based), not arbitrary.
    - take_profit_1 and take_profit_2 must be realistic from ATR/structure and give favorable R:R (ideally TP2 >= 2R).
    - Only LONG ideas unless explicitly asked for shorts.

    Output rules (CRITICAL):
    - If NO high-quality opportunity exists after checking all tickers, output exactly:
    no opportunity

    - If there IS an opportunity, output EXACTLY one object with NO extra keys, NO comments, NO markdown:

    {{
    "ticker": "STRING",
    "current_price": INTEGER,
    "current_price_date": "STRING",
    "order_time_horizon": INTEGER,
    "buy_in_price": NUMBER,
    "stop_loss": NUMBER,
    "take_profit_1": NUMBER,
    "take_profit_2": NUMBER,
    "buy_in_ammount": INTEGER,
    "buy_motivation" : STRING,
    "confidence" : INT FROM 1-10
    }}

    Constraints:
    - order_time_horizon: 30–180
    - buy_in_ammount: currency amount to allocate (e.g. 150)
    - Prices must satisfy:
    stop_loss < buy_in_price < take_profit_1 < take_profit_2
    - The output must be ONLY the object or "no opportunity"

    Tickers to analyze:
    {", ".join(tickers)}
    """.strip()

    config2 = types.GenerateContentConfig(
        tools=[get_current_ticker_data],
        temperature=0.1,
    )

    resp2 = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt2,
        config=config2,
    )

    return (resp2.text or "").strip()


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
    # _main()
    # cl.save_string_to_json(get_current_ticker_data("NVDA"))
    load_dotenv()  # reads .env in current working dir
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    print(daily_market_analysis(client))
