from datetime import datetime, timezone
import re
import logging
import yfinance as yf
from google.genai import types
import classes as cl
import helpers as hp


EXCHANGE_TO_INDEX = {
    # United States
    "NYSE": "^GSPC",  # S&P 500
    "NASDAQ": "^IXIC",  # Nasdaq Composite
    "NMS": "^IXIC",  # Nasdaq National Market System
    "NGM": "^IXIC",  # Nasdaq Global Market
    "NCM": "^IXIC",  # Nasdaq Capital Market
    "AMEX": "^GSPC",  # NYSE American
    "BATS": "^GSPC",
    "PCX": "^GSPC",  # NYSE Arca
    # Sweden
    "STO": "^OMX",  # OMX Stockholm All-Share
    #  Germany
    "FRA": "^GDAXI",  # DAX
    "XETRA": "^GDAXI",
    #  United Kingdom
    "LSE": "^FTSE",  # FTSE 100
    # France
    "PAR": "^FCHI",  # CAC 40
    # Italy
    "MIL": "FTSEMIB.MI",  # FTSE MIB
    # Spain
    "MCE": "^IBEX",  # IBEX 35
    # Netherlands
    "AMS": "^AEX",  # AEX
    # Switzerland
    "SWX": "^SSMI",  # SMI
    # Japan
    "TSE": "^N225",  # Nikkei 225
    # China
    "SHH": "000001.SS",  # Shanghai Composite
    "SHZ": "399001.SZ",  # Shenzhen Composite
    # Hong Kong
    "HKG": "^HSI",  # Hang Seng Index
    # South Korea
    "KOE": "^KS11",  # KOSPI
    # India
    "NSE": "^NSEI",  # NIFTY 50
    "BSE": "^BSESN",  # Sensex
    # Canada
    "TOR": "^GSPTSE",  # TSX Composite
    # Australia
    "ASX": "^AXJO",  # ASX 200
    # Brazil
    "SAO": "^BVSP",  # Bovespa
    # Mexico
    "MEX": "^MXX",  # IPC Mexico
    # South Africa
    "JNB": "^JN0U.JO",  # FTSE/JSE All Share
    # Singapore
    "SES": "^STI",  # Straits Times Index
}

logger = logging.getLogger(__name__)


# AI helper tools:
def get_current_ticker_data(ticker: str) -> str:
    data = yf.Ticker(ticker)
    info = data.info
    try:
        index = EXCHANGE_TO_INDEX[info.get("exchange")]
        index_hist_1y = (
            yf.Ticker(index).history(period="1y", interval="1d") if index else None
        )
    except Exception:
        index_hist_1y = None

    logger.info("Ran ticker data with ticker: %s, with index: %s", ticker, index)
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

    market_correlation = hp.YFMarketCorrelation.from_histories(
        stock_hist_1y,
        index_hist_1y,
        index_ticker=index,
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


def daily_port_analysis(current_port, client):
    prompt = f"""
        You are an AI portfolio monitoring agent for a cash-only account.
        You can only SELL existing positions (no buying, no shorting, no leverage).

        Context:
        - Stop-loss and take-profit execution is handled by a separate system.
        - Your role is NOT to manage exits mechanically.
        - Your role is to determine whether NEW EXTERNAL INFORMATION
        justifies breaking or modifying the original investment strategy.

        Critical requirement:
        You MUST use Search to assess whether the market, sector,
        or company-specific environment has materially changed since entry.
        Your decisions must be grounded in current, externally verified information.

        Input:
        You receive a portfolio object containing current holdings.
        For each stock or fund, you are given:
        - ticker
        - current quantity held
        - buy price
        - current value
        - sector / classification
        - other provided metadata

        All provided portfolio data is authoritative.

        Task:
        Using BOTH:
        1) the provided portfolio data, and
        2) current information obtained via Google Search,
        3) news, lastest earnings, current market state etc

        analyze whether any positions should be SOLD or PARTIALLY SOLD **today**,
        despite the original strategy.

        Recommend a sell ONLY if Google Search confirms at least one of the following:
        - market regime has shifted (e.g. risk-on → risk-off)
        - sector or industry has entered sustained underperformance
        - new macro, regulatory, earnings, or company-specific risks exist
        - the original investment thesis is materially weakened
        - portfolio-level risk reduction is justified due to external conditions

        Do NOT sell based on:
        - normal volatility or price fluctuations
        - stop-loss or take-profit logic
        - speculative or unverified information

        Rules:
        - Do NOT guess prices or fabricate data.
        - If Google Search finds no relevant or material changes, do NOT sell.
        - Be conservative: absence of strong evidence = hold.

        Output (STRICT):
        Return a list of tuples in the following format:
        [
        ("TICKER", quantity_to_sell),
        ...
        ]

        Output rules:
        - quantity_to_sell must be ≤ the quantity currently held
        - Do NOT label sells as partial or full
        - Do NOT include tickers that should be held
        - If no sells are justified, return an empty list: []

        Do NOT output explanations, reasoning, or disclaimers.
        Do NOT output anything outside this structure.

        Portfolio data:
        {current_port}
        """

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        # thinking_config=types.ThinkingConfig(thinking_level="high"),
        tools=[grounding_tool],
        temperature=0.1,
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt, config=config
    )

    return response.text


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


def daily_market_analysis(client, current_port, buying_power):
    logger.info("Running daily market analysis")
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

    IMPORTANT — Position management constraint:
    - Do NOT repeatedly rebuy or add to positions already held by default.
    - Adding to an existing position is allowed ONLY if there is NEW information
    that materially improves the risk/reward versus the original entry.
    - “Still attractive”, “still strong”, or “unchanged thesis” is NOT a valid reason to add.
    - If no clear add-specific catalyst or structural change exists, treat existing positions as HOLD.

    Universe:
    - Large US equities only
    - Highly liquid stocks (no microcaps, no thin volume)
    - Avoid stocks with earnings within the next 7 calendar days unless the catalyst is earnings-driven

    Objective:
    Identify up to 5 stock tickers that offer the BEST incremental risk-adjusted swing trade
    opportunities over the next 30–180 days.
    Preference should be given to:
    - New ideas not currently held, OR
    - Existing holdings ONLY if a justified add scenario exists (new structure, thesis expansion).

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
    TICKERS: <comma-separated list of ticker symbols in uppercase> (yfinance format)

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
    resp1 = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt1,
        config=config1,
    )

    step1_text = (resp1.text or "").strip()
    if step1_text.lower() == "no opportunity":
        logger.info("Daily market analysis step 1: no opportunity")
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
        logger.info("Daily market analysis step 1: no valid tickers")
        return "no opportunity"

    logger.info("Daily market analysis step 1 tickers: %s", ",".join(tickers))
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
    - Buy order can't exceed available buying power, currently: {buying_power}
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
    "buy_in_price": NUMBER,
    "stop_loss": NUMBER(put a stop loss for autoselling, so give some margin below the support level),
    "take_profit_1": NUMBER,
    "take_profit_2": NUMBER,
    "buy_in_quantity": INTEGER,
    }}

    Constraints:
    - order_time_horizon: 30–180
    - buy_in_ammount: currency amount to allocate (e.g. 150)
    - Prices must satisfy:
    stop_loss < buy_in_price < take_profit_1 < take_profit_2
    - The output must be ONLY the object or the json "text":"no opportunity"

    Tickers to analyze:
    {", ".join(tickers)}
    """.strip()

    # add to prompt for testing:
    """buy_motivation" : STRING(explain why this is worth a buy, explain what the stop loss and take profit levels are what they are as well as 
    explaining the cause of recent market trends for this stock),
    "confidence" : INT FROM 1-10
    "additional_info": STRING(explain what more info you would need to get about the stock from the get_current_ticker_data to make a good analysis. If the current data
    is sufficient return ok, if there is any other specific data you would generally need such as for example price history, or order volume, write it here)"""

    config2 = types.GenerateContentConfig(
        # thinking_config=types.ThinkingConfig(thinking_level="high"),
        tools=[get_current_ticker_data],
        temperature=0.1,
    )
    resp2 = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt2,
        config=config2,
    )
    logger.info("Daily market analysis step 2 completed")

    return (resp2.text or "").strip()
