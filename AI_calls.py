from datetime import datetime, timezone
import re
import yfinance as yf
from google.genai import types
import classes as cl
import helpers as hp


# AI helper tools:
def get_current_ticker_data(ticker: str, index_ticker: str = "^OMX") -> str:
    data = yf.Ticker(ticker)
    print(f"Ran ticker data with ticker: {ticker}")
    # implement some error handling
    info_filtered = hp.YFInfoFundamentals.from_yfinance_info(data.info).to_dict()
    fast_filtered = hp.YFFastInfoSnapshot.from_yfinance_fast_info(
        data.fast_info
    ).to_dict()
    nav_discount = hp.YFNavDiscountPremium.from_yfinance_info(
        data.info, data.fast_info
    ).to_dict()
    try:
        history_df = data.history(period="3mo", interval="1d")
    except Exception:
        history_df = None
    price_history = hp.YFPriceHistorySummary.from_history(history_df).to_dict()
    try:
        stock_hist_1y = data.history(period="1y", interval="1d")
    except Exception:
        stock_hist_1y = None
    try:
        index_hist_1y = (
            yf.Ticker(index_ticker).history(period="1y", interval="1d")
            if index_ticker
            else None
        )
    except Exception:
        index_hist_1y = None
    market_correlation = hp.YFMarketCorrelation.from_histories(
        stock_hist_1y,
        index_hist_1y,
        index_ticker=index_ticker,
    ).to_dict()
    earnings = hp.YFEarningsEvent.from_calendar_or_earnings_dates(
        data.calendar, data.earnings_dates
    ).to_dict()
    analyst = hp.YFAnalystSignal.from_recommendations(data.recommendations).to_dict()

    price_targets = data.get_analyst_price_targets()

    payload = hp.format_ticker_data_for_ai(
        info_filtered=info_filtered,
        fast_filtered=fast_filtered,
        price_history=price_history,
        nav_discount=nav_discount,
        market_correlation=market_correlation,
        earnings=earnings,
        analyst=analyst,
        price_targets=price_targets,
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


def daily_market_analysis(client, current_port):
    # -----------------------
    # STEP 1: Google Search ONLY
    # -----------------------
    prompt1 = f"""
    You are a professional portfolio manager and short-term swing trader (3–6 months).

    Task:
    Scan current global equity markets using recent news and macro information.
    Analyze:
    - Market regime (risk-on / risk-off, rates, inflation, geopolitics)
    - Sector rotation and relative strength
    - Major catalysts (earnings trends, guidance, AI, energy, defense, healthcare, regulation)
    - Liquidity and institutional relevance
    - Take into account the current trading portfolio state: {current_port}
    - You can still choose to fill up stock already in the portfolio if there is and swind opportunity.

    Universe:
    - Large SWEDISH equities only
    - Highly liquid stocks (no microcaps, no thin volume)
    - Avoid stocks with earnings within the next 7 calendar days unless the catalyst is earnings-driven

    Objective:
    Identify up to 3 of the most interesting stock tickers that may offer a high risk-adjusted swing trade opportunity over the next 30–180 days.

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
    """.strip()

    # thinking_config=types.ThinkingConfig(thinking_level="high"),
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config1 = types.GenerateContentConfig(
        # thinking_config=types.ThinkingConfig(thinking_level="high"),
        tools=[grounding_tool],
        temperature=0.1,
    )
    print("running first prompt")
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
    "ticker": "STRING(Make sure ticker is in correct format for yahoo finance python api)",
    "current_price": INTEGER,
    "current_price_date": "STRING",
    "order_time_horizon": INTEGER,
    "buy_in_price": NUMBER,
    "stop_loss": NUMBER(put a stop loss for autoselling, so give some margin below the support level),
    "take_profit_1": NUMBER,
    "take_profit_2": NUMBER,
    "buy_in_quantity": INTEGER,
    "buy_motivation" : STRING(explain why this is worth a buy, explain what the stop loss and take profit levels are what they are as well as 
    explaining the cause of recent market trends for this stock),
    "confidence" : INT FROM 1-10
    "additional_info": STRING(explain what more info you would need to get about the stock from the get_current_ticker_data to make a good analysis. If the current data
    is sufficient return ok, if there is any other specific data you would generally need such as for example price history, or order volume, write it here)
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
        # thinking_config=types.ThinkingConfig(thinking_level="high"),
        tools=[get_current_ticker_data],
        temperature=0.1,
    )
    print("running second prompt")
    resp2 = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt2,
        config=config2,
    )

    return (resp2.text or "").strip()
