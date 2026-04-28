from datetime import datetime, timezone
import json
import re
import logging
import textwrap

import yfinance as yf
from google.genai import types

import classes as cl
import helpers as hp
from helpers import compute_ex_ante_sharpe


INDEX_MAP = {
    "NYSE": "^GSPC",
    "NASDAQ": "^IXIC",
    "NMS": "^IXIC",
    "NGM": "^IXIC",
    "NCM": "^IXIC",
    "AMEX": "^GSPC",
    "BATS": "^GSPC",
    "PCX": "^GSPC",
    "STO": "^OMX",
    "FRA": "^GDAXI",
    "XETRA": "^GDAXI",
    "LSE": "^FTSE",
    "PAR": "^FCHI",
    "MIL": "FTSEMIB.MI",
    "MCE": "^IBEX",
    "AMS": "^AEX",
    "SWX": "^SSMI",
    "TSE": "^N225",
    "SHH": "000001.SS",
    "SHZ": "399001.SZ",
    "HKG": "^HSI",
    "KOE": "^KS11",
    "NSE": "^NSEI",
    "BSE": "^BSESN",
    "TOR": "^GSPTSE",
    "ASX": "^AXJO",
    "SAO": "^BVSP",
    "MEX": "^MXX",
    "JNB": "^JN0U.JO",
    "SES": "^STI",
}

logger = logging.getLogger(__name__)


POLISH_INDEX_TICKER = "WIG20.WA"
WARSAW_TICKER_SUFFIX = ".WA"


def _resolve_market_index_ticker(
    *, exchange: str | None, ticker: str | None
) -> str | None:
    normalized_exchange = str(exchange).strip().upper() if exchange else None
    mapped_index = INDEX_MAP.get(normalized_exchange) if normalized_exchange else None
    if mapped_index:
        return mapped_index

    normalized_ticker = str(ticker).strip().upper() if ticker else None
    if normalized_ticker and normalized_ticker.endswith(WARSAW_TICKER_SUFFIX):
        return POLISH_INDEX_TICKER

    return None


KNOWN_SINGLE_INSTRUMENT_ALIASES = {
    "gold": {
        "ticker": "GC=F",
        "company": "Gold Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month COMEX gold futures, a liquid proxy for gold sentiment and macro safe-haven demand.",
        "recent_catalysts": "Real yields, Fed policy expectations, US dollar direction, reserve demand, and geopolitical stress often drive the tape.",
        "key_risks": "Fast reversals can happen if real yields rise, the dollar strengthens, or inflation hedging demand fades.",
    },
    "gc f": {
        "ticker": "GC=F",
        "company": "Gold Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month COMEX gold futures, a liquid proxy for gold sentiment and macro safe-haven demand.",
        "recent_catalysts": "Real yields, Fed policy expectations, US dollar direction, reserve demand, and geopolitical stress often drive the tape.",
        "key_risks": "Fast reversals can happen if real yields rise, the dollar strengthens, or inflation hedging demand fades.",
    },
    "silver": {
        "ticker": "SI=F",
        "company": "Silver Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month COMEX silver futures, blending precious-metals and industrial-demand sensitivity.",
        "recent_catalysts": "Dollar direction, rate expectations, solar demand, industrial demand, and precious-metals momentum matter most.",
        "key_risks": "Silver is usually more volatile than gold and can underperform if industrial demand softens or yields jump.",
    },
    "si f": {
        "ticker": "SI=F",
        "company": "Silver Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month COMEX silver futures, blending precious-metals and industrial-demand sensitivity.",
        "recent_catalysts": "Dollar direction, rate expectations, solar demand, industrial demand, and precious-metals momentum matter most.",
        "key_risks": "Silver is usually more volatile than gold and can underperform if industrial demand softens or yields jump.",
    },
    "brent": {
        "ticker": "BZ=F",
        "company": "Brent Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Brent crude futures, a global oil benchmark watched across energy markets.",
        "recent_catalysts": "OPEC+ supply decisions, Middle East risk, inventories, refinery demand, and global growth expectations drive price action.",
        "key_risks": "Oil can gap on geopolitics, recession fears, and supply headlines, with futures roll adding noise.",
    },
    "brent oil": {
        "ticker": "BZ=F",
        "company": "Brent Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Brent crude futures, a global oil benchmark watched across energy markets.",
        "recent_catalysts": "OPEC+ supply decisions, Middle East risk, inventories, refinery demand, and global growth expectations drive price action.",
        "key_risks": "Oil can gap on geopolitics, recession fears, and supply headlines, with futures roll adding noise.",
    },
    "oil brent": {
        "ticker": "BZ=F",
        "company": "Brent Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Brent crude futures, a global oil benchmark watched across energy markets.",
        "recent_catalysts": "OPEC+ supply decisions, Middle East risk, inventories, refinery demand, and global growth expectations drive price action.",
        "key_risks": "Oil can gap on geopolitics, recession fears, and supply headlines, with futures roll adding noise.",
    },
    "brent crude": {
        "ticker": "BZ=F",
        "company": "Brent Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Brent crude futures, a global oil benchmark watched across energy markets.",
        "recent_catalysts": "OPEC+ supply decisions, Middle East risk, inventories, refinery demand, and global growth expectations drive price action.",
        "key_risks": "Oil can gap on geopolitics, recession fears, and supply headlines, with futures roll adding noise.",
    },
    "bz f": {
        "ticker": "BZ=F",
        "company": "Brent Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Brent crude futures, a global oil benchmark watched across energy markets.",
        "recent_catalysts": "OPEC+ supply decisions, Middle East risk, inventories, refinery demand, and global growth expectations drive price action.",
        "key_risks": "Oil can gap on geopolitics, recession fears, and supply headlines, with futures roll adding noise.",
    },
    "natural gas": {
        "ticker": "NG=F",
        "company": "Natural Gas Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Henry Hub natural gas futures, a core benchmark for US gas pricing and volatility.",
        "recent_catalysts": "Weather, storage data, LNG demand, production trends, and outages can move natural gas sharply.",
        "key_risks": "Natural gas is extremely volatile and can reverse quickly on weather changes, storage surprises, and supply rebounds.",
    },
    "natrual gas": {
        "ticker": "NG=F",
        "company": "Natural Gas Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Henry Hub natural gas futures, a core benchmark for US gas pricing and volatility.",
        "recent_catalysts": "Weather, storage data, LNG demand, production trends, and outages can move natural gas sharply.",
        "key_risks": "Natural gas is extremely volatile and can reverse quickly on weather changes, storage surprises, and supply rebounds.",
    },
    "nat gas": {
        "ticker": "NG=F",
        "company": "Natural Gas Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Henry Hub natural gas futures, a core benchmark for US gas pricing and volatility.",
        "recent_catalysts": "Weather, storage data, LNG demand, production trends, and outages can move natural gas sharply.",
        "key_risks": "Natural gas is extremely volatile and can reverse quickly on weather changes, storage surprises, and supply rebounds.",
    },
    "ng f": {
        "ticker": "NG=F",
        "company": "Natural Gas Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month Henry Hub natural gas futures, a core benchmark for US gas pricing and volatility.",
        "recent_catalysts": "Weather, storage data, LNG demand, production trends, and outages can move natural gas sharply.",
        "key_risks": "Natural gas is extremely volatile and can reverse quickly on weather changes, storage surprises, and supply rebounds.",
    },
    "crude oil": {
        "ticker": "CL=F",
        "company": "WTI Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month WTI crude futures, one of the main North American oil benchmarks.",
        "recent_catalysts": "US inventories, OPEC+ policy, refining demand, transport activity, and global growth expectations are key drivers.",
        "key_risks": "WTI can move violently on macro shocks, inventory data, and geopolitical headlines.",
    },
    "wti": {
        "ticker": "CL=F",
        "company": "WTI Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month WTI crude futures, one of the main North American oil benchmarks.",
        "recent_catalysts": "US inventories, OPEC+ policy, refining demand, transport activity, and global growth expectations are key drivers.",
        "key_risks": "WTI can move violently on macro shocks, inventory data, and geopolitical headlines.",
    },
    "cl f": {
        "ticker": "CL=F",
        "company": "WTI Crude Oil Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month WTI crude futures, one of the main North American oil benchmarks.",
        "recent_catalysts": "US inventories, OPEC+ policy, refining demand, transport activity, and global growth expectations are key drivers.",
        "key_risks": "WTI can move violently on macro shocks, inventory data, and geopolitical headlines.",
    },
    "copper": {
        "ticker": "HG=F",
        "company": "Copper Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month copper futures, often used as a cyclical growth and industrial activity barometer.",
        "recent_catalysts": "China demand, electrification themes, mine supply issues, and dollar moves commonly lead price action.",
        "key_risks": "Copper is sensitive to global manufacturing slowdowns and can fall fast when growth expectations weaken.",
    },
    "hg f": {
        "ticker": "HG=F",
        "company": "Copper Futures",
        "instrument_type": "commodity-future",
        "search_summary": "Tracks front-month copper futures, often used as a cyclical growth and industrial activity barometer.",
        "recent_catalysts": "China demand, electrification themes, mine supply issues, and dollar moves commonly lead price action.",
        "key_risks": "Copper is sensitive to global manufacturing slowdowns and can fall fast when growth expectations weaken.",
    },
}


def _known_single_instrument_metadata_for_ticker(ticker: str) -> dict[str, str] | None:
    normalized_ticker = (ticker or "").strip().upper()
    if not normalized_ticker:
        return None

    for metadata in KNOWN_SINGLE_INSTRUMENT_ALIASES.values():
        if metadata.get("ticker", "").strip().upper() == normalized_ticker:
            return metadata
    return None


def _uses_price_history_only_market_data(ticker: str) -> bool:
    normalized_ticker = (ticker or "").strip().upper()
    return normalized_ticker.endswith("=F")


def _safe_fast_info_subset(data: yf.Ticker, *, include_equity_fields: bool) -> dict:
    try:
        fast_info = data.fast_info or {}
    except Exception:
        return {}

    keys = [
        "last_price",
        "previous_close",
        "day_high",
        "day_low",
        "year_high",
        "year_low",
        "currency",
        "quote_type",
        "exchange",
    ]
    if include_equity_fields:
        keys.extend(["market_cap", "shares_outstanding"])

    subset = {}
    for key in keys:
        try:
            subset[key] = fast_info.get(key)
        except Exception:
            subset[key] = None
    return subset


def get_current_ticker_data(ticker: str) -> str:
    normalized_ticker = str(ticker).strip().upper()
    data = yf.Ticker(normalized_ticker)
    instrument_metadata = _known_single_instrument_metadata_for_ticker(normalized_ticker)
    price_history_only = _uses_price_history_only_market_data(normalized_ticker)

    info = {}
    exchange = None
    index = None
    index_hist_1y = None

    if not price_history_only:
        try:
            info = data.info or {}
        except Exception:
            info = {}
        exchange = info.get("exchange")
        try:
            index = _resolve_market_index_ticker(
                exchange=exchange,
                ticker=normalized_ticker,
            )
            if index:
                index_hist_1y = yf.Ticker(index).history(period="1y", interval="1d")
        except Exception:
            index = None
            index_hist_1y = None

    logger.info(
        "Ran ticker data with ticker: %s, with index: %s", normalized_ticker, index
    )

    fast_info = _safe_fast_info_subset(
        data, include_equity_fields=not price_history_only
    )

    info_filtered = (
        hp.YFInfoFundamentals.from_yfinance_info(info).to_dict()
        if not price_history_only
        else {}
    )
    fast_filtered = hp.YFFastInfoSnapshot.from_yfinance_fast_info(fast_info).to_dict()
    nav_discount = (
        hp.YFNavDiscountPremium.from_yfinance_info(info, fast_info).to_dict()
        if not price_history_only
        else {}
    )

    try:
        history_df = data.history(period="3mo", interval="1d")
    except Exception:
        history_df = None
    price_history = hp.YFPriceHistorySummary.from_history(history_df).to_dict()

    try:
        stock_hist_1y = data.history(period="1y", interval="1d")
    except Exception:
        stock_hist_1y = None

    # MA summary uses 1y history for SMA200; falls back gracefully with less data
    ma_summary = hp.YFMovingAverageSummary.from_history(stock_hist_1y).to_dict()

    market_correlation = None
    relative_strength = None
    if index:
        market_correlation = hp.YFMarketCorrelation.from_histories(
            stock_hist_1y,
            index_hist_1y,
            index_ticker=index,
        ).to_dict()
        relative_strength = hp.compute_relative_strength(
            stock_hist_1y,
            index_hist_1y,
            index_ticker=index,
        ).to_dict()

    earnings = None
    recommendations = None
    if not price_history_only:
        try:
            earnings = hp.YFEarningsEvent.from_calendar_or_earnings_dates(
                data.calendar, data.earnings_dates
            ).to_dict()
        except Exception:
            earnings = None
        try:
            recommendations = data.recommendations
        except Exception:
            recommendations = None

    analyst = hp.YFAnalystSignal.from_recommendations(recommendations).to_dict()

    price_targets = None
    if not price_history_only:
        try:
            raw_targets = data.get_analyst_price_targets()
            # Normalize to plain dict regardless of whether yfinance returns Series or dict
            if hasattr(raw_targets, "to_dict"):
                price_targets = raw_targets.to_dict()
            elif isinstance(raw_targets, dict):
                price_targets = raw_targets
        except Exception:
            price_targets = None

    payload = hp.format_ticker_data_for_ai(
        info_filtered=info_filtered,
        fast_filtered=fast_filtered,
        price_history=price_history,
        ma_summary=ma_summary,
        nav_discount=nav_discount,
        market_correlation=market_correlation,
        relative_strength=relative_strength,
        earnings=earnings,
        analyst=analyst,
        price_targets=price_targets,
    )

    asset_name = (
        info.get("longName")
        or info.get("shortName")
        or (instrument_metadata or {}).get("company")
        or normalized_ticker
    )
    quote_type = (
        info.get("quoteType")
        or fast_info.get("quote_type")
        or ("FUTURE" if price_history_only else None)
    )

    lines = [
        f"ticker: {normalized_ticker}",
        f"as_of: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    if asset_name:
        lines.append(f"asset_name: {asset_name}")
    if quote_type:
        lines.append(f"quote_type: {quote_type}")
    if payload:
        lines.append(payload)
    return "\n".join(lines)


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
        You MUST use Google Search to assess whether the market, sector,
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
        3) news, latest earnings, and current market state,
        analyze whether any positions should be SOLD or PARTIALLY SOLD today.

        Recommend a sell ONLY if Google Search confirms at least one of the following:
        - market regime has shifted against the position's directional bias
        - sector or industry has entered sustained underperformance or bearish rotation
        - new macro, regulatory, earnings, or company-specific risks materially change the outlook
        - the original investment thesis is broken (not just tested — broken)
        - a stop-loss level has been structurally violated even if the bracket hasn't triggered
        - portfolio-level correlation risk has risen sharply (multiple holdings now tracking same factor)

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
        - quantity_to_sell must be less than or equal to the quantity currently held
        - Do NOT include tickers that should be held
        - If no sells are justified, return an empty list: []
        - Do NOT output explanations, reasoning, or disclaimers

        Portfolio data:
        {current_port}
        """.strip()

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="high"),
        tools=[grounding_tool],
        temperature=0.1,
    )

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=config,
    )
    return response.text


def _extract_tickers(text: str) -> list[str]:
    if not text:
        return []
    if text.strip().lower() == "no opportunity":
        return []

    ticker_pattern = re.compile(r"[A-Z0-9]{1,10}(?:[.-][A-Z0-9]{1,10})*")
    raw = re.split(r"[,;\n]+", text.upper())
    tickers = []
    seen = set()

    for token in raw:
        token = token.strip()
        if not token:
            continue

        token = re.sub(r"^[A-Z_ ]+:\s*", "", token)
        token = re.sub(r"^[\-\*\d\.\)\(]+\s*", "", token)
        token = re.sub(r"\s*\(.*\)\s*$", "", token)
        token = re.sub(r"\s*([.-])\s*", r"\1", token)
        token = token.strip(" \t'\"`[]{}")

        if ticker_pattern.fullmatch(token) and token not in seen:
            seen.add(token)
            tickers.append(token)

    return tickers


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if len(lines) >= 3 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return cleaned.strip("`").strip()


def _normalize_single_instrument_query(query: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (query or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _resolve_known_single_instrument_query(query: str) -> dict[str, str] | None:
    normalized_query = _normalize_single_instrument_query(query)
    if not normalized_query:
        return None

    match = KNOWN_SINGLE_INSTRUMENT_ALIASES.get(normalized_query)
    if not match:
        return None

    return {
        "QUERY": (query or "").strip(),
        "TICKER": match["ticker"],
        "COMPANY": match["company"],
        "INSTRUMENT_TYPE": match["instrument_type"],
        "MATCH_CONFIDENCE": "HIGH",
        "SEARCH_SUMMARY": match["search_summary"],
        "RECENT_CATALYSTS": match["recent_catalysts"],
        "KEY_RISKS": match["key_risks"],
    }


def _parse_single_stock_search_result(text: str) -> dict[str, str] | None:
    cleaned = _strip_code_fences(text)
    if not cleaned or cleaned.upper() == "NO_MATCH":
        return None

    parsed = cl.parse_ai_response_payload(cleaned)
    if not isinstance(parsed, dict):
        return None
    if set(parsed.keys()) == {"text"}:
        return None

    normalized = {
        str(key).strip().upper(): "" if value is None else str(value).strip()
        for key, value in parsed.items()
    }

    ticker = normalized.get("TICKER")
    company = (
        normalized.get("COMPANY")
        or normalized.get("NAME")
        or normalized.get("ASSET_NAME")
        or normalized.get("INSTRUMENT")
    )
    if not ticker or not company:
        return None

    normalized.setdefault("COMPANY", company)
    return normalized


def _parse_single_stock_analysis_result(text: str) -> dict[str, str] | None:
    cleaned = _strip_code_fences(text)
    if not cleaned or cleaned.upper() == "NO_MATCH":
        return None

    parsed = cl.parse_ai_response_payload(cleaned)
    if not isinstance(parsed, dict):
        return None
    if set(parsed.keys()) == {"text"}:
        return None

    return {
        str(key).strip().upper(): "" if value is None else str(value).strip()
        for key, value in parsed.items()
    }


def _remove_single_stock_summary_driver_lines(text: str) -> str:
    filtered_lines = []
    for line in (text or "").splitlines():
        normalized = line.strip().upper()
        if re.match(r"^(UPSIDE|DOWNSIDE)_[1-5]\s*:", normalized):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def _split_summary_candidates(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []

    parts = re.split(r"\s*(?:;|\n|•)+\s*|(?<=[.!?])\s+", cleaned)
    return [part.strip(" -•\t\r\n.") for part in parts if part.strip(" -•\t\r\n.")]


def _collect_summary_items(candidates: list[str], limit: int = 5) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        for part in _split_summary_candidates(candidate):
            normalized_key = re.sub(r"\s+", " ", part).strip().lower()
            if not normalized_key or normalized_key in seen:
                continue
            seen.add(normalized_key)
            items.append(part)
            if len(items) == limit:
                return items

    while len(items) < limit:
        items.append("-")

    return items


def _normalize_summary_cell(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return cleaned if cleaned else "-"


def _wrap_terminal_table_cell(text: str, width: int) -> list[str]:
    wrapped = textwrap.wrap(
        _normalize_summary_cell(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or ["-"]


def _format_terminal_table(
    headers: list[str], rows: list[tuple[str, str]], col_width: int = 34
) -> str:
    border = "+" + "+".join("-" * (col_width + 2) for _ in headers) + "+"
    rendered_lines = [border]

    def append_wrapped_row(cells: list[str]) -> None:
        wrapped_cells = [_wrap_terminal_table_cell(cell, col_width) for cell in cells]
        row_height = max(len(cell_lines) for cell_lines in wrapped_cells)
        for cell_lines in wrapped_cells:
            cell_lines.extend([""] * (row_height - len(cell_lines)))
        for row_parts in zip(*wrapped_cells):
            rendered_lines.append(
                "| " + " | ".join(part.ljust(col_width) for part in row_parts) + " |"
            )
        rendered_lines.append(border)

    append_wrapped_row(headers)
    for row in rows:
        append_wrapped_row(list(row))

    return "\n".join(rendered_lines)


def _build_single_stock_summary_section(
    analysis_payload: dict[str, str],
    search_payload: dict[str, str] | None = None,
) -> str:
    upside_candidates = [
        analysis_payload.get(f"UPSIDE_{idx}", "") for idx in range(1, 6)
    ]
    downside_candidates = [
        analysis_payload.get(f"DOWNSIDE_{idx}", "") for idx in range(1, 6)
    ]

    upside_candidates.extend(
        [
            analysis_payload.get("WHAT_LOOKS_GOOD", ""),
            analysis_payload.get("SEARCH_SUMMARY", ""),
            analysis_payload.get("CONCLUSION", ""),
            (search_payload or {}).get("RECENT_CATALYSTS", ""),
        ]
    )
    downside_candidates.extend(
        [
            analysis_payload.get("WHAT_COULD_GO_WRONG", ""),
            analysis_payload.get("LEVELS_RATIONALE", ""),
            analysis_payload.get("CONCLUSION", ""),
            (search_payload or {}).get("KEY_RISKS", ""),
        ]
    )

    upside_items = _collect_summary_items(upside_candidates, limit=5)
    downside_items = _collect_summary_items(downside_candidates, limit=5)

    table_rows = [
        (f"{idx}. {upside}", f"{idx}. {downside}")
        for idx, (upside, downside) in enumerate(
            zip(upside_items, downside_items), start=1
        )
    ]

    return "Summary\n" + _format_terminal_table(
        ["Likely To Push It Up", "Could Push It Down"], table_rows
    )


def single_stock_analysis(stock_query: str, client) -> str:
    normalized_query = (stock_query or "").strip()
    if not normalized_query:
        raise ValueError("stock_query must not be empty")

    logger.info("Running single stock analysis for query: %s", normalized_query)

    step1_payload = _resolve_known_single_instrument_query(normalized_query)
    if step1_payload:
        logger.info(
            "Single stock analysis matched built-in instrument alias: %s -> %s",
            normalized_query,
            step1_payload["TICKER"],
        )

    if not step1_payload:
        prompt1 = f"""
        You are a market research assistant.

        Task:
        Use Google Search to identify the most likely Yahoo Finance-compatible tradable instrument
        that matches the user's query.
        The query may be:
        - a ticker symbol
        - a company name
        - a commodity or macro asset like gold, silver, Brent oil, or natural gas
        - an ETF or index proxy
        - a casual shorthand like "amd" or "brent"

        You must:
        - Identify the best Yahoo Finance-compatible ticker for the asset
        - Prefer the direct listed instrument when it exists
        - For commodities or indexes without a direct spot listing, choose the closest liquid Yahoo Finance-compatible instrument
          such as the front-month futures contract or a primary ETF proxy
        - Summarize the most relevant recent company, industry, sector, or macro context
        - Focus on verifiable, recent information

        If there is no clear Yahoo Finance-compatible match, output exactly:
        NO_MATCH

        Output rules:
        - Output EXACTLY in the following line-based format
        - No markdown
        - No extra commentary

        QUERY: <repeat the user query>
        TICKER: <Yahoo Finance-compatible ticker in uppercase>
        COMPANY: <company name, or instrument name for commodities/ETFs/index proxies>
        INSTRUMENT_TYPE: <stock, commodity-future, ETF, index-proxy, currency, or other>
        MATCH_CONFIDENCE: <HIGH, MEDIUM, or LOW>
        SEARCH_SUMMARY: <max 80 words>
        RECENT_CATALYSTS: <max 80 words>
        KEY_RISKS: <max 80 words>

        User query:
        {normalized_query}
        """.strip()

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config1 = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )

        resp1 = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt1,
            config=config1,
        )

        step1_text = (resp1.text or "").strip()
        step1_payload = _parse_single_stock_search_result(step1_text)
        if not step1_payload:
            logger.info(
                "Single stock analysis search step returned no match for query: %s",
                normalized_query,
            )
            return "NO_MATCH"

    ticker = step1_payload["TICKER"]
    company = step1_payload["COMPANY"]
    instrument_type = step1_payload.get("INSTRUMENT_TYPE", "")

    prompt2 = f"""
    You are a swing trader preparing a directional trade setup (30–180 day horizon) for a human investor.

    Search findings from Step 1:
    QUERY: {step1_payload.get("QUERY", normalized_query)}
    TICKER: {ticker}
    COMPANY: {company}
    INSTRUMENT_TYPE: {instrument_type}
    MATCH_CONFIDENCE: {step1_payload.get("MATCH_CONFIDENCE", "")}
    SEARCH_SUMMARY: {step1_payload.get("SEARCH_SUMMARY", "")}
    RECENT_CATALYSTS: {step1_payload.get("RECENT_CATALYSTS", "")}
    KEY_RISKS: {step1_payload.get("KEY_RISKS", "")}

    Step 1: Call get_current_ticker_data exactly once for ticker {ticker}.

    Step 2: Extract these fields from the tool result to inform your analysis:
      price_history   → last_close, atr_14, atr_14_pct, range_20d_high, range_20d_low,
                        close_vs_20d_high_pct, volume_ratio_vs_20d, recent_bars_10d
      relative_strength → signal, composite_score, rrg_quadrant, mrs_52w, rolling_alpha_annualized_pct
      market_correlation → interpretation, beta_6_12m
      fundamentals     → sector, beta, forward_pe, revenue_growth_pct, earnings_growth_pct
      earnings_event   → earnings_date_iso, earnings_in_days

    Step 3: Determine BIAS using the weight of evidence:
      BULLISH signals: uptrend (price near range_20d_high), rrg_quadrant "leading"/"improving",
                       composite_score > 0, positive recent catalysts, strong earnings momentum
      BEARISH signals: downtrend (price near range_20d_low), rrg_quadrant "lagging"/"weakening",
                       composite_score < 0, deteriorating fundamentals, negative catalysts
      Mark UNSURE if signals conflict materially or data is insufficient

    Step 4: Set levels using this methodology:

      LONG setup (BULLISH):
        Entry:      At or just above nearest support / pullback level; within 2% of last_close
                    For a breakout setup: consolidation high + 0.1×atr_14
        Stop-loss:  Below the most recent swing low OR 2.0×atr_14 below entry — use whichever is TIGHTER
                    Absolute max distance: 2.5×atr_14 below entry
        TP1:        Nearer of (next significant resistance / 52-week high area) OR (2.0×risk above entry)
        TP2:        Measured move OR 3.0×risk above entry
        Required:   (TP1 − entry) / (entry − stop) ≥ 2.0; if impossible, set BIAS = UNSURE

      SHORT setup (BEARISH):
        Entry:      At or just below nearest resistance; within 2% of last_close
        Stop-loss:  Above the most recent swing high OR 2.0×atr_14 above entry — TIGHTER wins
        TP1:        Nearer of (next significant support) OR (2.0×risk below entry)
        TP2:        Measured move OR 3.0×risk below entry
        Required:   (entry − TP1) / (stop − entry) ≥ 2.0; if impossible, set BIAS = UNSURE

      For non-equity instruments (commodity futures, ETFs): apply the same ATR-based methodology;
      do not force earnings-style reasoning where it does not apply.

    Step 5: Call compute_ex_ante_sharpe(entry, stop_loss, take_profit_1, take_profit_2, atr_14).
            Use the result as SHARPE_RATIO. If BIAS is UNSURE, set SHARPE_RATIO to UNSURE.

    UPSIDE fields: each must be one specific, concrete catalyst or condition that would push price
                   higher (e.g., "earnings beat triggers analyst upgrades", "breakout above $X resistance").
                   Do NOT write generic statements.

    DOWNSIDE fields: each must be one specific risk or condition that would invalidate the setup
                     (e.g., "macro headwinds drag sector rotation", "stop triggered below $X swing low").

    Output EXACTLY in this line-based format with no markdown:
    QUERY: <user query>
    TICKER: <ticker>
    COMPANY: <company or instrument name>
    BIAS: <BULLISH, BEARISH, or UNSURE>
    TRADE_DIRECTION: <LONG, SHORT, or UNSURE>
    CONFIDENCE: <integer 1-10>
    SHARPE_RATIO: <number from compute_ex_ante_sharpe, or UNSURE>
    CURRENT_PRICE: <number>
    CURRENT_PRICE_DATE: <string>
    ENTRY_LEVEL: <number or UNSURE>
    STOP_LOSS_LEVEL: <number or UNSURE>
    TAKE_PROFIT_1: <number or UNSURE>
    TAKE_PROFIT_2: <number or UNSURE>
    LEVELS_RATIONALE: <max 90 words: reference specific ATR multiples, swing lows/highs, or resistance levels>

    CONCLUSION: <max 120 words: regime alignment, relative strength signal, catalyst, and setup quality>
    UPSIDE_1: <concrete catalyst, max 16 words>
    UPSIDE_2: <concrete catalyst, max 16 words>
    UPSIDE_3: <concrete catalyst, max 16 words>
    UPSIDE_4: <concrete catalyst, max 16 words>
    UPSIDE_5: <concrete catalyst, max 16 words>
    DOWNSIDE_1: <concrete risk, max 16 words>
    DOWNSIDE_2: <concrete risk, max 16 words>
    DOWNSIDE_3: <concrete risk, max 16 words>
    DOWNSIDE_4: <concrete risk, max 16 words>
    DOWNSIDE_5: <concrete risk, max 16 words>

    Level rules:
    - BULLISH → TRADE_DIRECTION = LONG: STOP_LOSS_LEVEL < ENTRY_LEVEL < TAKE_PROFIT_1 < TAKE_PROFIT_2
    - BEARISH → TRADE_DIRECTION = SHORT: TAKE_PROFIT_2 < TAKE_PROFIT_1 < ENTRY_LEVEL < STOP_LOSS_LEVEL
    - UNSURE → TRADE_DIRECTION = UNSURE: all level fields must be exactly UNSURE
    - Do not output placeholders like N/A, TBD, or null
    """.strip()

    config2 = types.GenerateContentConfig(
        tools=[get_current_ticker_data, compute_ex_ante_sharpe],
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )

    resp2 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt2,
        config=config2,
    )

    logger.info("Single stock analysis completed for ticker: %s", ticker)
    response_text = (resp2.text or "").strip()
    analysis_payload = _parse_single_stock_analysis_result(response_text)
    if not analysis_payload:
        return response_text

    upside_downside_keys = {f"UPSIDE_{i}" for i in range(1, 6)} | {f"DOWNSIDE_{i}" for i in range(1, 6)}
    main_payload = {k: v for k, v in analysis_payload.items() if k not in upside_downside_keys}
    summary_table = _build_single_stock_summary_section(analysis_payload, step1_payload)

    return json.dumps(main_payload, indent=2, ensure_ascii=False) + "\n\n" + summary_table


def daily_market_analysis(
    client, current_port: str | None = None, market="US", buying_power=None
):
    logger.info("Running daily market analysis")

    portfolio_context = ""
    if current_port:
        portfolio_context = f"""
        IMPORTANT — Current portfolio context:
        {current_port}
        """.strip()
    logger.info("Daily market analysis market scope: %s", market)

    prompt1 = f"""
    You are a professional swing trader and portfolio manager targeting 30–180 day trades.

    Task:
    Using Google Search and current market knowledge, scan global equity markets for the most
    compelling swing trade setups available RIGHT NOW.

    Evaluate:
    - Market regime (risk-on / risk-off, rates, inflation, macro backdrop)
    - Sector rotation leadership and institutional flow evidence
    - Earnings momentum (consecutive beats, revenue acceleration, guidance raises)
    - Theme catalysts (AI, defense, energy transition, healthcare innovation, regulation)

    Setup criteria — candidates SHOULD show at least ONE of:
    - Base breakout: stock consolidating tightly above support, near a breakout point
    - Pullback entry: uptrending stock (above 50-day MA) pulling back to rising MA or prior breakout level
    - Earnings momentum: 2+ consecutive beats with accelerating EPS/revenue, price not yet extended
    - Sector rotation leader: sector entering bull rotation, stock leading the move with relative strength

    Rejection criteria — EXCLUDE any stock with:
    - Price in a confirmed downtrend (below 200-day MA with no clear basing pattern)
    - Earnings within 7 calendar days (unless the entire thesis is the upcoming catalyst)
    - Microcap or thin volume (prefer market cap >$500M, avg daily volume >500K shares)
    - No identifiable catalyst — momentum without a reason deteriorates fast

    Universe:
    - {market} equities only
    - Highly liquid stocks only
    - Tickers must be supported by the yfinance Python API

    Objective:
    Identify up to 10 stock tickers that offer the BEST incremental risk-adjusted swing trade
    opportunities over the next 30-180 days.

    Rules:
    - Do NOT invent prices or technical levels
    - Do NOT propose entries, stops, or targets
    - Do NOT call any functions
    - Base conclusions only on recent, verifiable information

    Output rules:
    If NO stocks qualify, output exactly:
    no opportunity

    If stocks qualify, output EXACTLY in this 2-line format with no extra text:
    MARKET_CONTEXT: <one paragraph max 60 words: regime, sector rotation, and dominant setup theme>
    TICKERS: <comma-separated list of ticker symbols in uppercase>
    """.strip()

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config1 = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )

    resp1 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt1,
        config=config1,
    )

    step1_text = (resp1.text or "").strip()
    if step1_text.lower() == "no opportunity":
        logger.info("Daily market analysis step 1: no opportunity")
        return "no opportunity"

    market_context = ""
    tickers_line = ""

    for line in step1_text.splitlines():
        line = line.strip()
        if line.upper().startswith("MARKET_CONTEXT:"):
            market_context = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TICKERS:"):
            tickers_line = line.split(":", 1)[1].strip()

    if not tickers_line and "TICKERS:" not in step1_text.upper():
        tickers_line = step1_text

    tickers = _extract_tickers(tickers_line)
    if not tickers:
        logger.info("Daily market analysis step 1: no valid tickers")
        return "no opportunity"

    logger.info("Daily market analysis step 1 tickers: %s", ",".join(tickers))

    buying_power_line = ""
    if buying_power is not None:
        buying_power_line = (
            f"- Buy order cannot exceed available buying power, currently: {buying_power}"
        )

    prompt2 = f"""
    You are a professional swing trader (30–180 day horizon) evaluating a shortlist of candidates.
    Your job is to independently assess each ticker and output ONLY those that pass all quality filters.
    Do NOT assume any of them are good trades — treat each one as innocent until proven viable.

    Market context from the prior scan:
    {market_context}

    STRICT TOOL RULES:
    - You MUST call get_current_ticker_data exactly once for EACH ticker
    - After determining trade levels for a viable ticker, call compute_ex_ante_sharpe(entry, stop_loss, tp1, tp2, atr_14)
    - If you cannot retrieve valid data for a ticker, discard it
    - Do NOT invent prices or levels — all numbers must come from tool output

    KEY DATA TO EXTRACT from each tool result:
    - price_history: last_close, atr_14, atr_14_pct, avg_volume_20d, last_volume, volume_ratio_vs_20d,
                     range_20d_high, range_20d_low, close_vs_20d_high_pct, recent_bars_10d
    - relative_strength: signal, composite_score, rrg_quadrant, mrs_52w, rolling_alpha_annualized_pct
    - market_correlation: interpretation, beta_6_12m
    - fundamentals: sector, beta, forward_pe, revenue_growth_pct, earnings_growth_pct

    BIAS DETERMINATION — apply to each ticker independently:
      BULLISH signals: price near range_20d_high, rrg_quadrant "leading"/"improving",
                       composite_score > 0, positive recent catalysts, strong earnings momentum
      BEARISH / DISCARD signals: price near range_20d_low, rrg_quadrant "lagging"/"weakening",
                                  composite_score < −0.2, deteriorating fundamentals, negative catalysts
      If signals conflict materially or data is insufficient → discard the ticker (do NOT force a bullish view)

    LEVEL-SETTING METHODOLOGY (apply only to tickers that show BULLISH bias):

    Entry:
      - Pullback setup: buy at or just above a recent support level (prior breakout base, rising 20/50MA zone)
      - Breakout setup: buy at consolidation high + (0.1 × atr_14) as a buffer above resistance
      - Entry must be within 2% of current price or at a clearly defined trigger level with a price alert

    Stop-loss:
      - Place below the most recent 10-day swing low OR 2.0 × atr_14 below entry — use whichever is TIGHTER
      - Absolute limits: minimum 0.5 × atr_14 below entry, maximum 2.5 × atr_14 below entry
      - The level must represent real technical invalidation (below support or a swing-low structure)
      - Do NOT use arbitrary round-number stops

    Take-profit:
      - TP1: the nearer of (prior resistance / 52-week high area) OR (2.0 × risk distance above entry)
      - TP2: measured move (height of the base added to the breakout level) OR (3.0 × risk distance above entry)
      - MANDATORY: (TP1 − entry) / (entry − stop_loss) ≥ 2.0
      - If you cannot achieve 2:1 R:R on TP1, this setup is NOT viable — discard it

    QUALITY FILTERS — discard the ticker if ANY of these apply:
      - Bias is not clearly BULLISH (mixed or bearish signals → discard)
      - R:R to TP1 < 2.0 (as above)
      - Sharpe ratio from compute_ex_ante_sharpe < 0.4
      - rrg_quadrant is "lagging" AND there is no overriding near-term catalyst
      - composite_score < −0.2 (sustained relative underperformance vs index)
      - volume_ratio_vs_20d < 0.5 on a supposed breakout day (no volume conviction)
      - beta_6_12m > 3.0 (too noisy for a defined-risk swing trade)
      - earnings within 7 days (check earnings_event section if present)

    PREFERENCE RANKING (when multiple setups pass all filters):
      1. Higher ex-ante Sharpe ratio
      2. RRG quadrant: leading > improving > weakening
      3. composite_score (higher = stronger relative strength)
      4. Catalyst quality and sector regime alignment with market_context above

    {buying_power_line}

    Output rules:
    - For an opportunity to be viable it must have a Confidence of 8 or higher
    - Be honest with Confidence: 8 means high conviction with strong bullish bias AND clean setup.
      Do NOT inflate Confidence just to cross the 8 threshold. If genuinely uncertain, discard.
    - If NO ticker passes the quality filters with Confidence ≥ 8, output exactly: no opportunity
    - If multiple tickers share the highest Confidence level, output all of them, one object per line
    - If there IS an opportunity, output the object(s) with NO extra keys, NO comments, NO markdown:
    {{
    "ticker": "STRING",
    "current_price": NUMBER,
    "current_price_date": "STRING",
    "buy_in_price": NUMBER,
    "stop_loss": NUMBER,
    "take_profit_1": NUMBER,
    "take_profit_2": NUMBER,
    "buy_in_quantity": INTEGER,
    "Confidence": INTEGER (0-10),
    "atr_14": NUMBER (from get_current_ticker_data),
    "sharpe_ratio": NUMBER (from compute_ex_ante_sharpe)
    }}

    Hard constraints:
    - stop_loss < buy_in_price < take_profit_1 < take_profit_2
    - (take_profit_1 - buy_in_price) / (buy_in_price - stop_loss) >= 2.0
    - Output must be ONLY the JSON object(s) or the text "no opportunity" — nothing else

    Tickers to analyze:
    {", ".join(tickers)}
    """.strip()

    config2 = types.GenerateContentConfig(
        tools=[get_current_ticker_data, compute_ex_ante_sharpe],
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )

    resp2 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt2,
        config=config2,
    )

    logger.info("Daily market analysis step 2 completed")
    return (resp2.text or "").strip()


# ===========================================================================
# v2 additions — 1-12 month swing analysis
# ===========================================================================


def get_current_ticker_data_v2(ticker: str) -> str:
    """
    Enhanced version of get_current_ticker_data for 1-12 month swing analysis.
    Returns all v1 data PLUS two additional sections:

    [weekly_structure]
      sma_20w (≈100-day), sma_40w (≈200-day), close_vs_sma20w/40w_pct,
      high/low_52w, high/low_104w (2-year range), weekly_trend_structure,
      recent_bars_8w (last 8 weekly OHLCV candles).

    [enhanced_fundamentals]  (equity/ETF only — skipped for futures)
      free_cashflow, operating_cashflow, gross_margins_pct,
      operating_margins_pct, return_on_equity, return_on_assets,
      current_ratio, dividend_yield_pct, peg_ratio, price_to_book.
    """
    base = get_current_ticker_data(ticker)

    normalized_ticker = str(ticker).strip().upper()
    price_history_only = _uses_price_history_only_market_data(normalized_ticker)
    data = yf.Ticker(normalized_ticker)

    extra: list[str] = []

    try:
        weekly_hist = data.history(period="5y", interval="1wk")
        ws = hp.YFWeeklyStructure.from_weekly_history(weekly_hist).to_dict()
        lines = [f"- {k}: {v}" for k, v in ws.items() if v is not None]
        if lines:
            extra += ["[weekly_structure]"] + lines
    except Exception:
        pass

    if not price_history_only:
        try:
            info = data.info or {}
            ef = hp.YFEnhancedFundamentals.from_yfinance_info(info).to_dict()
            lines = [f"- {k}: {v}" for k, v in ef.items() if v is not None]
            if lines:
                extra += ["[enhanced_fundamentals]"] + lines
        except Exception:
            pass

    return (base + "\n" + "\n".join(extra)).strip() if extra else base


def _build_v2_timeframe_table(analysis_payload: dict[str, str]) -> str:
    rows = [
        ("Short (1-3 months)", analysis_payload.get("TREND_SHORT_1M", "-")),
        ("Medium (3-6 months)", analysis_payload.get("TREND_MEDIUM_3_6M", "-")),
        ("Long (6-12 months)", analysis_payload.get("TREND_LONG_6_12M", "-")),
    ]
    return "Timeframe Analysis\n" + _format_terminal_table(
        ["Horizon", "Trend Bias"], rows, col_width=24
    )


def single_stock_analysis_v2(stock_query: str, client) -> str:
    """
    v2 comprehensive swing-trade analysis optimised for 1-12 month horizons.

    Key differences from single_stock_analysis (v1):
    - Calls get_current_ticker_data_v2 (adds weekly chart + quality fundamentals)
    - Multi-timeframe trend assessment (short / medium / long)
    - Setup quality grade A-D
    - Explicit THESIS, KEY_CATALYST, INVALIDATION_TRIGGER fields
    - Two entry variants: ENTRY_AGGRESSIVE and ENTRY_CONSERVATIVE
    - HOLDING_PERIOD_ESTIMATE
    - Larger TP2 multipliers for longer-hold setups
    """
    normalized_query = (stock_query or "").strip()
    if not normalized_query:
        raise ValueError("stock_query must not be empty")

    logger.info("Running v2 stock analysis for: %s", normalized_query)

    # Step 1: resolve ticker (same logic as v1)
    step1_payload = _resolve_known_single_instrument_query(normalized_query)
    if step1_payload:
        logger.info(
            "v2 matched built-in alias: %s -> %s",
            normalized_query,
            step1_payload["TICKER"],
        )

    if not step1_payload:
        prompt1 = f"""
        You are a market research assistant.

        Task:
        Use Google Search to identify the most likely Yahoo Finance-compatible tradable instrument
        that matches the user's query.
        The query may be a ticker symbol, company name, commodity, ETF, index proxy, or casual shorthand.

        You must:
        - Identify the best Yahoo Finance-compatible ticker for the asset
        - Prefer the direct listed instrument when it exists
        - For commodities or indexes, choose the closest liquid Yahoo Finance-compatible instrument
          (front-month futures or primary ETF proxy)
        - Summarize the most relevant recent company, industry, sector, or macro context
        - Focus on verifiable, recent information

        If there is no clear match, output exactly: NO_MATCH

        Output EXACTLY in this line-based format (no markdown, no extra commentary):
        QUERY: <repeat the user query>
        TICKER: <Yahoo Finance-compatible ticker in uppercase>
        COMPANY: <company name or instrument name>
        INSTRUMENT_TYPE: <stock, commodity-future, ETF, index-proxy, currency, or other>
        MATCH_CONFIDENCE: <HIGH, MEDIUM, or LOW>
        SEARCH_SUMMARY: <max 80 words>
        RECENT_CATALYSTS: <max 80 words>
        KEY_RISKS: <max 80 words>

        User query:
        {normalized_query}
        """.strip()

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config1 = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )
        resp1 = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt1,
            config=config1,
        )
        step1_text = (resp1.text or "").strip()
        step1_payload = _parse_single_stock_search_result(step1_text)
        if not step1_payload:
            logger.info("v2 search step returned no match for: %s", normalized_query)
            return "NO_MATCH"

    ticker = step1_payload["TICKER"]
    company = step1_payload["COMPANY"]
    instrument_type = step1_payload.get("INSTRUMENT_TYPE", "")

    prompt2 = f"""
    You are a professional swing trader and equity analyst preparing a comprehensive
    directional trade evaluation for a 1-12 MONTH holding horizon.

    Unlike a short swing trade, this analysis must weigh fundamentals, weekly chart
    structure, and business quality alongside the near-term technical setup.
    Longer holding periods reward picking high-quality businesses in bull trends;
    they punish ignoring the big picture.

    Search context from Step 1:
    QUERY: {step1_payload.get("QUERY", normalized_query)}
    TICKER: {ticker}
    COMPANY: {company}
    INSTRUMENT_TYPE: {instrument_type}
    MATCH_CONFIDENCE: {step1_payload.get("MATCH_CONFIDENCE", "")}
    SEARCH_SUMMARY: {step1_payload.get("SEARCH_SUMMARY", "")}
    RECENT_CATALYSTS: {step1_payload.get("RECENT_CATALYSTS", "")}
    KEY_RISKS: {step1_payload.get("KEY_RISKS", "")}

    ═══════════════════════════════════════════════════════════
    STEP 1 — DATA COLLECTION
    ═══════════════════════════════════════════════════════════
    Call get_current_ticker_data_v2 exactly once for ticker {ticker}.

    Key sections and fields to extract:

    price_history       → last_close, atr_14, atr_14_pct, range_20d_high, range_20d_low,
                          close_vs_20d_high_pct, volume_ratio_vs_20d, recent_bars_10d
    moving_averages     → sma_20, sma_50, sma_200, close_vs_sma50_pct, trend_structure
    relative_strength   → signal, composite_score, rrg_quadrant, mrs_52w,
                          rolling_alpha_annualized_pct
    market_correlation  → interpretation, beta_6_12m
    earnings_event      → earnings_date_iso, earnings_in_days
    analyst_signal      → trend_90d, rating_buckets
    analyst_price_targets → (use as soft upside reference)

    [NEW in v2]
    weekly_structure    → sma_20w, sma_40w, close_vs_sma20w_pct, close_vs_sma40w_pct,
                          weekly_trend_structure, high_52w, low_52w, high_104w, low_104w,
                          recent_bars_8w
    enhanced_fundamentals → free_cashflow, operating_cashflow, gross_margins_pct,
                            operating_margins_pct, return_on_equity, return_on_assets,
                            current_ratio, peg_ratio, price_to_book

    ═══════════════════════════════════════════════════════════
    STEP 2 — MULTI-TIMEFRAME TREND ASSESSMENT
    ═══════════════════════════════════════════════════════════
    Assess each horizon INDEPENDENTLY and set the corresponding output field.

    SHORT TERM (1-3 months) — daily chart context:
      BULLISH: price near range_20d_high, above sma_20, above sma_50;
               recent_bars show higher closes; volume_ratio ≥ 0.8
      BEARISH: price near range_20d_low, below sma_20; declining closes
      MIXED:   conflicting or insufficient signals

    MEDIUM TERM (3-6 months) — daily 50/200-day MA and RRG:
      BULLISH: trend_structure = "price_above_50_above_200"; rrg "leading"/"improving";
               composite_score ≥ 0
      BEARISH: trend_structure = "price_below_50_below_200" or "price_below_50_above_200";
               rrg "lagging"; composite_score < -0.2
      MIXED:   otherwise

    LONG TERM (6-12 months) — WEEKLY CHART IS PRIMARY:
      BULLISH: weekly_trend_structure = "price_above_20w_above_40w"
      MIXED:   "price_above_20w_below_40w" or "price_below_20w_above_40w"
      BEARISH: "price_below_20w_below_40w"
      ⚠ A stock BELOW its 40-week MA needs an exceptional catalyst + strong fundamentals
        to qualify for a 6-12 month long. Apply extra skepticism and cap CONFIDENCE at 6.

    ═══════════════════════════════════════════════════════════
    STEP 3 — FUNDAMENTAL QUALITY ASSESSMENT
    ═══════════════════════════════════════════════════════════
    Use enhanced_fundamentals to assess business quality.
    Weight these factors for holds > 3 months:

    STRONG quality signals (favour longer hold):
      • free_cashflow > 0 and growing (positive FCF = self-funding growth)
      • gross_margins_pct stable or expanding (pricing power)
      • return_on_equity > 10% (capital efficiency)
      • peg_ratio < 2.0 (reasonable growth-adjusted valuation)
      • current_ratio ≥ 1.0 (no near-term cash crunch)

    WEAK quality signals (shorten target horizon or reduce confidence):
      • free_cashflow < 0 — yellow flag; negative FCF + high valuation = red flag
      • gross_margins_pct declining — competitive pressure
      • return_on_equity < 0 — capital destruction
      • peg_ratio > 3.5 and no near-term catalyst — stretched for 1-year hold
      • current_ratio < 0.8 — potential liquidity risk

    Combine quality assessment into a qualitative label: STRONG / ADEQUATE / WEAK / UNKNOWN
    Use WEAK to reduce CONFIDENCE by 1-2 points.
    Use STRONG to increase CONFIDENCE by 1 point (cap at 10).
    Do NOT use WEAK fundamentals alone to set BIAS = BEARISH unless other signals confirm.

    ═══════════════════════════════════════════════════════════
    STEP 4 — OVERALL BIAS DETERMINATION
    ═══════════════════════════════════════════════════════════
    Apply the following WEIGHTED signal hierarchy:

    HIGH WEIGHT (50%): Long-term weekly trend (STEP 2 long-term) +
                       Medium-term daily trend (STEP 2 medium-term)
    MEDIUM WEIGHT (30%): Fundamental quality (STEP 3) +
                         Relative strength (composite_score, rrg_quadrant)
    LOW WEIGHT (20%): Short-term daily setup (STEP 2 short-term)

    BULLISH overall:  Long-term AND medium-term BULLISH, fundamentals ≥ ADEQUATE,
                      composite_score > -0.2, no earnings within 7 days
    BEARISH overall:  Long-term OR medium-term BEARISH with no compelling recovery catalyst
    UNSURE:           Mixed or conflicting timeframes, WEAK fundamentals across all signals,
                      insufficient data (< 90 days history), or commodity without trend

    ═══════════════════════════════════════════════════════════
    STEP 5 — SETUP CLASSIFICATION AND QUALITY GRADE
    ═══════════════════════════════════════════════════════════
    SETUP_TYPE — pick ONE:
      breakout              — clearing a multi-week resistance with expanding volume
      pullback_to_support   — retracing to a prior breakout level, rising MA, or S/R zone
      base                  — multi-week/month sideways consolidation before potential move
      momentum_continuation — existing strong trend, no base needed; relative strength leads
      reversal              — emerging from a downtrend with clear evidence of trend change
      unsure                — setup unclear or contradictory

    SETUP_QUALITY_GRADE — A / B / C / D:
      A: All three timeframes BULLISH, clean technical setup, STRONG fundamentals,
         clear named catalyst, composite_score ≥ 0.3
      B: Long + medium term BULLISH, setup exists but ≥1 condition mixed,
         fundamentals ADEQUATE, catalyst identifiable
      C: Setup works on daily only, weekly neutral/mixed, fundamentals ADEQUATE,
         catalyst uncertain; confidence ≤ 7
      D: Conflicting signals, WEAK fundamentals, no clear catalyst → force BIAS = UNSURE
      Note: Grade D always maps to BIAS = UNSURE.

    ═══════════════════════════════════════════════════════════
    STEP 6 — LEVEL SETTING
    ═══════════════════════════════════════════════════════════
    Levels must be derived from tool data. Do NOT invent numbers.

    BULLISH (LONG) setup:

    ENTRY_AGGRESSIVE:
      Best-priced entry — at nearest technical support or within 1% of last_close.
      For pullback setups: just above the support zone.
      For breakout: current consolidation high or last_close if already breaking out.

    ENTRY_CONSERVATIVE:
      Confirmation-based entry — above a clear trigger level.
      Examples: above range_20d_high, above a consolidation resistance, or after a strong
      weekly close above the 20-week MA.
      ENTRY_CONSERVATIVE ≥ ENTRY_AGGRESSIVE.

    ENTRY_LEVEL = ENTRY_AGGRESSIVE (primary recommendation and basis for Sharpe calc).

    STOP_LOSS_LEVEL:
      Below the most recent 10-day swing low OR 2.0×atr_14 below entry — use TIGHTER.
      Absolute max: 2.5×atr_14 below entry.
      Also reference low_52w from weekly_structure as an absolute floor backstop.

    TAKE_PROFIT_1:
      Estimate HOLDING_PERIOD_ESTIMATE first (see STEP 8), then apply:
      • 1-3m setups: nearer of (next resistance / 52w high) or (2.0×risk above entry)
      • 3-6m setups: nearer of (high_52w area) or (2.5×risk above entry)
      • 6-12m setups: high_104w area, analyst price target, or (3.0×risk above entry)
      MANDATORY: (TP1 - entry) / (entry - stop) ≥ 2.0. If impossible → BIAS = UNSURE.

    TAKE_PROFIT_2:
      • 1-3m: 3.0×risk above entry
      • 3-6m: 4.0×risk above entry
      • 6-12m: 5.0×risk above entry (or measured move target)

    BEARISH (SHORT) setup — mirror the above with reversed direction.
    TP2 max: 3.0×risk below entry (short upside is capped by zero).

    ═══════════════════════════════════════════════════════════
    STEP 7 — COMPUTE SHARPE
    ═══════════════════════════════════════════════════════════
    Call compute_ex_ante_sharpe(
        entry=ENTRY_LEVEL,
        stop_loss=STOP_LOSS_LEVEL,
        take_profit_1=TAKE_PROFIT_1,
        take_profit_2=TAKE_PROFIT_2,
        atr_14=atr_14,
    )
    Use result as SHARPE_RATIO. If BIAS = UNSURE → SHARPE_RATIO = UNSURE.

    ═══════════════════════════════════════════════════════════
    STEP 8 — THESIS, CATALYST, INVALIDATION, HOLDING PERIOD
    ═══════════════════════════════════════════════════════════
    THESIS (max 40 words):
      One sentence: what must be true for this trade to work?
      Reference the specific setup, business driver, and time element.
      Example: "AMD pulls back to $140 AI-chip support while datacenter revenue
      accelerates; institutional accumulation and margin expansion justify a 6-month hold."

    KEY_CATALYST (max 20 words):
      The single most important upcoming price driver.
      Example: "Q2 AI revenue beat + raised annual guidance triggers analyst upgrades."

    INVALIDATION_TRIGGER (max 20 words):
      The specific price level or event that kills the thesis.
      Example: "Weekly close below $130 swing low OR two consecutive EPS misses."

    HOLDING_PERIOD_ESTIMATE — based on setup type and target distance:
      breakout / momentum_continuation: 1-3m
      pullback_to_support with large target: 3-6m
      base pattern or reversal: 3-6m or 6-12m
      If long-term trend BULLISH and TP2 ≥ 4×risk: 6-12m

    ═══════════════════════════════════════════════════════════
    OUTPUT FORMAT — line-based, no markdown, no extra commentary
    ═══════════════════════════════════════════════════════════
    QUERY: <user query>
    TICKER: <ticker>
    COMPANY: <company or instrument name>
    BIAS: <BULLISH | BEARISH | UNSURE>
    TRADE_DIRECTION: <LONG | SHORT | UNSURE>
    CONFIDENCE: <integer 1-10>
    SETUP_QUALITY_GRADE: <A | B | C | D>
    SETUP_TYPE: <breakout | pullback_to_support | base | momentum_continuation | reversal | unsure>
    SHARPE_RATIO: <number or UNSURE>
    HOLDING_PERIOD_ESTIMATE: <1-3m | 3-6m | 6-12m | UNSURE>
    CURRENT_PRICE: <number>
    CURRENT_PRICE_DATE: <string>
    ENTRY_AGGRESSIVE: <number or UNSURE>
    ENTRY_CONSERVATIVE: <number or UNSURE>
    ENTRY_LEVEL: <number or UNSURE>
    STOP_LOSS_LEVEL: <number or UNSURE>
    TAKE_PROFIT_1: <number or UNSURE>
    TAKE_PROFIT_2: <number or UNSURE>
    TREND_SHORT_1M: <BULLISH | BEARISH | MIXED>
    TREND_MEDIUM_3_6M: <BULLISH | BEARISH | MIXED>
    TREND_LONG_6_12M: <BULLISH | BEARISH | MIXED>
    THESIS: <one sentence, max 40 words>
    KEY_CATALYST: <max 20 words>
    INVALIDATION_TRIGGER: <max 20 words>
    LEVELS_RATIONALE: <max 90 words: cite specific ATR multiples, weekly MAs, and S/R zones>
    CONCLUSION: <max 120 words: synthesise timeframe alignment, fundamentals, setup type, and catalyst>
    UPSIDE_1: <concrete catalyst, max 16 words>
    UPSIDE_2: <concrete catalyst, max 16 words>
    UPSIDE_3: <concrete catalyst, max 16 words>
    UPSIDE_4: <concrete catalyst, max 16 words>
    UPSIDE_5: <concrete catalyst, max 16 words>
    DOWNSIDE_1: <concrete risk, max 16 words>
    DOWNSIDE_2: <concrete risk, max 16 words>
    DOWNSIDE_3: <concrete risk, max 16 words>
    DOWNSIDE_4: <concrete risk, max 16 words>
    DOWNSIDE_5: <concrete risk, max 16 words>

    Level ordering rules:
    - BULLISH/LONG:  STOP_LOSS_LEVEL < ENTRY_LEVEL ≤ ENTRY_CONSERVATIVE < TAKE_PROFIT_1 < TAKE_PROFIT_2
    - BEARISH/SHORT: TAKE_PROFIT_2 < TAKE_PROFIT_1 < ENTRY_LEVEL ≤ ENTRY_CONSERVATIVE < STOP_LOSS_LEVEL
    - UNSURE:        all level and entry fields must be exactly UNSURE
    - Never output N/A, TBD, null, or placeholder text
    """.strip()

    config2 = types.GenerateContentConfig(
        tools=[get_current_ticker_data_v2, compute_ex_ante_sharpe],
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )

    resp2 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt2,
        config=config2,
    )

    logger.info("v2 stock analysis completed for ticker: %s", ticker)
    response_text = (resp2.text or "").strip()
    analysis_payload = _parse_single_stock_analysis_result(response_text)
    if not analysis_payload:
        return response_text

    upside_downside_keys = {f"UPSIDE_{i}" for i in range(1, 6)} | {f"DOWNSIDE_{i}" for i in range(1, 6)}
    main_payload = {k: v for k, v in analysis_payload.items() if k not in upside_downside_keys}

    timeframe_table = _build_v2_timeframe_table(analysis_payload)
    summary_table = _build_single_stock_summary_section(analysis_payload, step1_payload)

    return (
        json.dumps(main_payload, indent=2, ensure_ascii=False)
        + "\n\n"
        + timeframe_table
        + "\n\n"
        + summary_table
    )


def daily_market_analysis_v2(
    client, current_port: str | None = None, market="US", buying_power=None
):
    """
    v2 daily market scan.

    Stage 1 (identical to v1): Google Search broad scan → up to 10 candidate tickers.
    Stage 2 (v2): each candidate is evaluated with get_current_ticker_data_v2, applying
    the same multi-timeframe, fundamental-quality framework as single_stock_analysis_v2
    (1–12 month swing horizon). Only grade-A/B setups with confidence ≥ 8 are returned.
    """
    logger.info("Running v2 daily market analysis")
    logger.info("v2 daily market analysis market scope: %s", market)

    # -----------------------------------------------------------------------
    # Stage 1: identical to daily_market_analysis — broad Google Search scan
    # -----------------------------------------------------------------------
    prompt1 = f"""
    You are a professional swing trader and portfolio manager targeting 30–180 day trades.

    Task:
    Using Google Search and current market knowledge, scan global equity markets for the most
    compelling swing trade setups available RIGHT NOW.

    Evaluate:
    - Market regime (risk-on / risk-off, rates, inflation, macro backdrop)
    - Sector rotation leadership and institutional flow evidence
    - Earnings momentum (consecutive beats, revenue acceleration, guidance raises)
    - Theme catalysts (AI, defense, energy transition, healthcare innovation, regulation)

    Setup criteria — candidates SHOULD show at least ONE of:
    - Base breakout: stock consolidating tightly above support, near a breakout point
    - Pullback entry: uptrending stock (above 50-day MA) pulling back to rising MA or prior breakout level
    - Earnings momentum: 2+ consecutive beats with accelerating EPS/revenue, price not yet extended
    - Sector rotation leader: sector entering bull rotation, stock leading the move with relative strength

    Rejection criteria — EXCLUDE any stock with:
    - Price in a confirmed downtrend (below 200-day MA with no clear basing pattern)
    - Earnings within 7 calendar days (unless the entire thesis is the upcoming catalyst)
    - Microcap or thin volume (prefer market cap >$500M, avg daily volume >500K shares)
    - No identifiable catalyst — momentum without a reason deteriorates fast

    Universe:
    - {market} equities only
    - Highly liquid stocks only
    - Tickers must be supported by the yfinance Python API

    Objective:
    Identify up to 10 stock tickers that offer the BEST incremental risk-adjusted swing trade
    opportunities over the next 30-180 days.

    Rules:
    - Do NOT invent prices or technical levels
    - Do NOT propose entries, stops, or targets
    - Do NOT call any functions
    - Base conclusions only on recent, verifiable information

    Output rules:
    If NO stocks qualify, output exactly:
    no opportunity

    If stocks qualify, output EXACTLY in this 2-line format with no extra text:
    MARKET_CONTEXT: <one paragraph max 60 words: regime, sector rotation, and dominant setup theme>
    TICKERS: <comma-separated list of ticker symbols in uppercase>
    """.strip()

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config1 = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )

    resp1 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt1,
        config=config1,
    )

    step1_text = (resp1.text or "").strip()
    if step1_text.lower() == "no opportunity":
        logger.info("v2 daily market analysis step 1: no opportunity")
        return "no opportunity"

    market_context = ""
    tickers_line = ""

    for line in step1_text.splitlines():
        line = line.strip()
        if line.upper().startswith("MARKET_CONTEXT:"):
            market_context = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TICKERS:"):
            tickers_line = line.split(":", 1)[1].strip()

    if not tickers_line and "TICKERS:" not in step1_text.upper():
        tickers_line = step1_text

    tickers = _extract_tickers(tickers_line)
    if not tickers:
        logger.info("v2 daily market analysis step 1: no valid tickers")
        return "no opportunity"

    logger.info("v2 daily market analysis step 1 tickers: %s", ",".join(tickers))

    # -----------------------------------------------------------------------
    # Stage 2: v2 deep analysis using get_current_ticker_data_v2
    # -----------------------------------------------------------------------
    buying_power_line = ""
    if buying_power is not None:
        buying_power_line = (
            f"- Buy order cannot exceed available buying power, currently: {buying_power}"
        )

    prompt2 = f"""
    You are a professional swing trader (1–12 month horizon) evaluating a shortlist of candidates.
    Assess each ticker independently using the v2 multi-timeframe framework. Output ONLY tickers
    that pass ALL quality filters. Do NOT assume any of them are good trades.

    Market context from the prior scan:
    {market_context}

    ═══════════════════════════════════════════════════════════
    STRICT TOOL RULES
    ═══════════════════════════════════════════════════════════
    - Call get_current_ticker_data_v2 exactly once per ticker
    - After setting levels for a viable ticker, call compute_ex_ante_sharpe(entry, stop_loss, tp1, tp2, atr_14)
    - If data cannot be retrieved for a ticker, discard it silently
    - Do NOT invent prices or levels — all numbers must come from tool output

    ═══════════════════════════════════════════════════════════
    KEY DATA TO EXTRACT from each get_current_ticker_data_v2 result
    ═══════════════════════════════════════════════════════════
    price_history       → last_close, atr_14, atr_14_pct, avg_volume_20d, last_volume,
                          volume_ratio_vs_20d, range_20d_high, range_20d_low,
                          close_vs_20d_high_pct, recent_bars_10d
    moving_averages     → sma_20, sma_50, sma_200, close_vs_sma50_pct, trend_structure
    relative_strength   → signal, composite_score, rrg_quadrant, mrs_52w,
                          rolling_alpha_annualized_pct
    market_correlation  → interpretation, beta_6_12m
    fundamentals        → sector, beta, forward_pe, revenue_growth_pct, earnings_growth_pct
    earnings_event      → earnings_date_iso, earnings_in_days
    [v2 additions]
    weekly_structure    → sma_20w, sma_40w, weekly_trend_structure,
                          high_52w, low_52w, high_104w, low_104w, recent_bars_8w
    enhanced_fundamentals → free_cashflow, operating_cashflow, gross_margins_pct,
                            operating_margins_pct, return_on_equity, return_on_assets,
                            current_ratio, peg_ratio, price_to_book

    ═══════════════════════════════════════════════════════════
    STEP 1 — MULTI-TIMEFRAME BIAS (assess each horizon independently)
    ═══════════════════════════════════════════════════════════
    SHORT (1-3 months) — daily chart:
      BULLISH: price near range_20d_high, above sma_20 AND sma_50,
               recent_bars_10d show higher closes, volume_ratio_vs_20d ≥ 0.8
      BEARISH: price near range_20d_low, below sma_20; declining closes
      MIXED:   conflicting or insufficient signals

    MEDIUM (3-6 months) — 50/200-day MA and RRG:
      BULLISH: trend_structure = "price_above_50_above_200"; rrg_quadrant "leading"/"improving";
               composite_score ≥ 0
      BEARISH: trend_structure shows price below both MAs; rrg_quadrant "lagging";
               composite_score < -0.2
      MIXED:   otherwise

    LONG (6-12 months) — WEEKLY CHART IS PRIMARY:
      BULLISH: weekly_trend_structure = "price_above_20w_above_40w"
      MIXED:   "price_above_20w_below_40w" or "price_below_20w_above_40w"
      BEARISH: "price_below_20w_below_40w"
      ⚠ A stock BELOW its 40-week MA requires exceptional catalyst + STRONG fundamentals;
        cap CONFIDENCE at 6 (which fails the ≥ 8 threshold — discard unless truly exceptional).

    ═══════════════════════════════════════════════════════════
    STEP 2 — FUNDAMENTAL QUALITY ASSESSMENT
    ═══════════════════════════════════════════════════════════
    STRONG signals (support higher conviction and longer hold):
      • free_cashflow > 0
      • gross_margins_pct stable or expanding
      • return_on_equity > 10%
      • peg_ratio < 2.0
      • current_ratio ≥ 1.0

    WEAK signals (reduce conviction or disqualify):
      • free_cashflow < 0 AND high valuation
      • gross_margins_pct declining
      • return_on_equity < 0
      • peg_ratio > 3.5 with no near-term catalyst
      • current_ratio < 0.8

    Quality label: STRONG / ADEQUATE / WEAK / UNKNOWN
    Apply: WEAK → reduce CONFIDENCE by 1-2; STRONG → increase CONFIDENCE by 1 (cap 10).
    WEAK fundamentals alone do not set BIAS = BEARISH, but do contribute to discard.

    ═══════════════════════════════════════════════════════════
    STEP 3 — OVERALL BIAS DETERMINATION
    ═══════════════════════════════════════════════════════════
    Weighted signal hierarchy:
      HIGH (50%):   Long-term weekly trend + medium-term daily trend
      MEDIUM (30%): Fundamental quality + relative strength (composite_score, rrg_quadrant)
      LOW (20%):    Short-term daily setup

    BULLISH:  Long-term AND medium-term BULLISH; fundamentals ≥ ADEQUATE;
              composite_score > -0.2; no earnings within 7 days
    BEARISH:  Long-term OR medium-term BEARISH with no compelling recovery catalyst
    DISCARD:  Mixed timeframes; WEAK fundamentals across all signals; insufficient data

    ═══════════════════════════════════════════════════════════
    STEP 4 — SETUP CLASSIFICATION AND QUALITY GRADE
    ═══════════════════════════════════════════════════════════
    SETUP_TYPE (pick one):
      breakout              — clearing multi-week resistance with expanding volume
      pullback_to_support   — retracing to prior breakout level, rising MA, or S/R zone
      base                  — multi-week/month sideways consolidation before potential move
      momentum_continuation — existing strong trend, no base; relative strength leads
      reversal              — emerging from downtrend with clear evidence of trend change

    SETUP_QUALITY_GRADE:
      A: All three timeframes BULLISH, clean technical setup, STRONG fundamentals,
         clear named catalyst, composite_score ≥ 0.3
      B: Long + medium term BULLISH; ≥1 condition mixed; fundamentals ADEQUATE;
         catalyst identifiable
      C: Daily-only setup, weekly neutral/mixed → DISCARD
      D: Conflicting signals or WEAK fundamentals → DISCARD
    Only grades A and B pass. C and D are always discarded.

    ═══════════════════════════════════════════════════════════
    STEP 5 — HOLDING PERIOD AND LEVEL SETTING
    ═══════════════════════════════════════════════════════════
    HOLDING_PERIOD_ESTIMATE based on setup type and target distance:
      breakout / momentum_continuation: 1-3m
      pullback_to_support with large target: 3-6m
      base or reversal: 3-6m or 6-12m
      If long-term trend BULLISH and TP2 ≥ 4×risk: 6-12m

    Entry (use as buy_in_price):
      Pullback: at or just above nearest support; within 2% of last_close
      Breakout: consolidation high + (0.1 × atr_14)

    Stop-loss:
      Below the most recent 10-day swing low OR 2.0×atr_14 below entry — TIGHTER wins
      Absolute max: 2.5×atr_14 below entry
      Reference low_52w from weekly_structure as an absolute floor backstop

    Take-profit scaled to HOLDING_PERIOD_ESTIMATE:
      1-3m:  TP1 = nearer of (next resistance / high_52w) or (2.0×risk); TP2 = 3.0×risk
      3-6m:  TP1 = nearer of (high_52w) or (2.5×risk);                   TP2 = 4.0×risk
      6-12m: TP1 = high_104w or analyst target or (3.0×risk);             TP2 = 5.0×risk
      MANDATORY: (TP1 − entry) / (entry − stop_loss) ≥ 2.0. If impossible → discard.

    ═══════════════════════════════════════════════════════════
    QUALITY FILTERS — discard the ticker if ANY of these apply
    ═══════════════════════════════════════════════════════════
      - BIAS is not clearly BULLISH
      - R:R to TP1 < 2.0
      - Sharpe ratio from compute_ex_ante_sharpe < 0.4
      - Setup quality grade is C or D
      - weekly_trend_structure = "price_below_20w_below_40w" (discard unless confidence ≥ 9
        with an exceptional near-term catalyst AND STRONG fundamentals)
      - fundamental_quality is WEAK
      - rrg_quadrant is "lagging" AND there is no overriding near-term catalyst
      - composite_score < -0.2
      - volume_ratio_vs_20d < 0.5 on a supposed breakout day
      - beta_6_12m > 3.0
      - earnings within 7 days

    PREFERENCE RANKING when multiple setups pass all filters:
      1. Higher ex-ante Sharpe ratio
      2. Setup quality grade: A > B
      3. Longer holding period (6-12m > 3-6m > 1-3m) when Sharpe is equal
      4. RRG quadrant: leading > improving > weakening
      5. composite_score (higher = stronger relative strength)
      6. Catalyst quality and sector alignment with market_context above

    {buying_power_line}

    Output rules:
    - For an opportunity to be viable it must have Confidence of 8 or higher
    - Be honest: 8 means high conviction — BULLISH on long AND medium timeframes,
      grade A or B, clean setup with a named catalyst. Do NOT inflate Confidence to pass.
      If genuinely uncertain, discard.
    - If NO ticker passes all quality filters with Confidence ≥ 8, output exactly: no opportunity
    - If there IS an opportunity, output one JSON object per viable ticker on its own line,
      with NO extra keys, NO comments, NO markdown:
    {{
    "ticker": "STRING",
    "current_price": NUMBER,
    "current_price_date": "STRING",
    "buy_in_price": NUMBER,
    "stop_loss": NUMBER,
    "take_profit_1": NUMBER,
    "take_profit_2": NUMBER,
    "buy_in_quantity": INTEGER,
    "Confidence": INTEGER (0-10),
    "atr_14": NUMBER (from get_current_ticker_data_v2),
    "sharpe_ratio": NUMBER (from compute_ex_ante_sharpe),
    "setup_quality_grade": "A" or "B",
    "holding_period_estimate": "1-3m" or "3-6m" or "6-12m"
    }}

    Hard constraints:
    - stop_loss < buy_in_price < take_profit_1 < take_profit_2
    - (take_profit_1 - buy_in_price) / (buy_in_price - stop_loss) >= 2.0
    - Output must be ONLY the JSON object(s) or the text "no opportunity" — nothing else

    Tickers to analyze:
    {", ".join(tickers)}
    """.strip()

    config2 = types.GenerateContentConfig(
        tools=[get_current_ticker_data_v2, compute_ex_ante_sharpe],
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )

    resp2 = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt2,
        config=config2,
    )

    logger.info("v2 daily market analysis step 2 completed")
    return (resp2.text or "").strip()
