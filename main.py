from dotenv import load_dotenv
import os
import sys
import re
import ast
import json
import logging
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
import time

from google import genai

import AI_calls as AIC
import api
import classes as cl
import helpers as hp


logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")


def _setup_logging() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file — 5 MB per file, keep 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "scheduler.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_port_sell_list(raw: str) -> list[tuple[str, float]]:
    """
    Parse the AI portfolio-analysis response into a list of (ticker, qty) tuples.
    The AI is instructed to return [("TICKER", qty), ...] or [].
    Handles extra explanation text and markdown code fences gracefully.
    """
    if not raw:
        return []

    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    # Extract the first [...] block — the only part we care about
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []

    list_str = match.group(0)
    try:
        result = ast.literal_eval(list_str)
    except Exception:
        logger.warning("_parse_port_sell_list: could not parse list: %r", list_str[:200])
        return []

    if not isinstance(result, list):
        return []

    normalized: list[tuple[str, float]] = []
    for item in result:
        try:
            ticker = str(item[0]).strip().upper()
            qty = float(item[1])
            if ticker and qty > 0:
                normalized.append((ticker, qty))
        except Exception:
            continue
    return normalized


def _parse_multi_json_opportunities(text: str) -> list[dict]:
    """
    Parse one or more JSON objects returned by daily_market_analysis.
    The AI may return a single object or multiple objects on consecutive lines.
    """
    if not text:
        return []

    stripped = text.strip()
    if stripped.lower() == "no opportunity":
        return []

    objects: list[dict] = []

    # Try each line as an independent JSON object first
    for line in stripped.splitlines():
        line = line.strip()
        if not line or line.lower() == "no opportunity":
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "ticker" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            pass

    if objects:
        return objects

    # Fall back to the whole text as one JSON object
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "ticker" in obj:
            return [obj]
    except Exception:
        pass

    return []


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------

def analyzeAIResult(ai_result: dict, trading_client) -> None:
    if not isinstance(ai_result, dict):
        return
    if ai_result.get("text", "").strip().lower() == "no opportunity":
        return

    required_keys = {"ticker", "buy_in_quantity", "take_profit_1", "take_profit_2", "stop_loss", "atr_14"}
    if not required_keys.issubset(ai_result.keys()):
        logger.warning(
            "analyzeAIResult: missing required keys %s",
            required_keys - ai_result.keys(),
        )
        return

    ticker = str(ai_result["ticker"]).strip().upper()

    try:
        qty = float(ai_result["buy_in_quantity"])
        buy_in_price = float(ai_result.get("buy_in_price") or ai_result.get("current_price") or 0)
        stop_loss_price = float(ai_result["stop_loss"])
        tp1_price = float(ai_result["take_profit_1"])
        tp2_price = float(ai_result["take_profit_2"])
        atr_14 = float(ai_result["atr_14"])
    except (TypeError, ValueError) as exc:
        logger.warning("analyzeAIResult: could not parse numeric fields for %s: %s", ticker, exc)
        return

    if qty <= 0:
        logger.warning("[SKIP] %s: buy_in_quantity is %s — skipping", ticker, qty)
        return
    if buy_in_price <= 0:
        logger.warning("[SKIP] %s: buy_in_price is %s — skipping", ticker, buy_in_price)
        return

    # Gate on R:R before going to Sharpe
    risk = buy_in_price - stop_loss_price
    reward_tp1 = tp1_price - buy_in_price
    if risk <= 0 or reward_tp1 <= 0:
        logger.warning("[SKIP] %s: degenerate levels (risk=%.4f reward_tp1=%.4f)", ticker, risk, reward_tp1)
        return
    rr = reward_tp1 / risk
    if rr < 2.0:
        logger.info("[SKIP] %s: R:R %.2f below 2.0 minimum — skipping", ticker, rr)
        return

    SHARPE_MIN = 0.4
    sharpe = hp.compute_ex_ante_sharpe(
        entry=buy_in_price,
        stop_loss=stop_loss_price,
        tp1=tp1_price,
        tp2=tp2_price,
        atr_14=atr_14,
    )
    if sharpe is None:
        logger.warning("[SKIP] %s: ex-ante Sharpe could not be computed — skipping", ticker)
        return
    if sharpe < SHARPE_MIN:
        logger.info("[SKIP] %s: Sharpe %.3f below %.2f — skipping", ticker, sharpe, SHARPE_MIN)
        return

    logger.info(
        "[TRADE] %s: Sharpe=%.3f R:R=%.2f entry=%.2f sl=%.2f tp1=%.2f tp2=%.2f qty=%.2f",
        ticker, sharpe, rr, buy_in_price, stop_loss_price, tp1_price, tp2_price, qty,
    )

    # Use round(x, 2) — NOT int() — so prices like 152.75 are preserved
    take_profit1 = {"limit_price": round(tp1_price, 2)}
    take_profit2 = {"limit_price": round(tp2_price, 2)}
    stop_loss_leg = {"stop_price": round(stop_loss_price, 2)}

    try:
        if qty <= 1:
            api.bracketBuy(ticker, qty, take_profit1, stop_loss_leg, trading_client)
        else:
            half = round(qty / 2, 6)
            api.bracketBuy(ticker, half, take_profit1, stop_loss_leg, trading_client)
            api.bracketBuy(ticker, half, take_profit2, stop_loss_leg, trading_client)
    except Exception as exc:
        logger.error("[ERROR] %s: order submission failed: %s", ticker, exc)


# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------

RUN_TIMES = [
    "10:20",
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


def handle_daily_sell(sell_list: list[tuple[str, float]], alpaca_client) -> None:
    for ticker, qty in sell_list:
        try:
            api.sellStock(ticker, qty, alpaca_client)
        except Exception as exc:
            logger.error("Error while selling %s: %s", ticker, exc)


def _build_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Daily task cycle
# ---------------------------------------------------------------------------

def _run_daily_tasks(client, alpaca_client) -> None:
    logger.info("=== Daily task cycle started at %s ===", datetime.now())
    current_port_state = api.alpaca_portfolio_context(alpaca_client)

    # --- Portfolio sell check ---
    try:
        daily_port_res_raw = AIC.daily_port_analysis(current_port_state, client)
        sell_list = _parse_port_sell_list(daily_port_res_raw)
        if sell_list:
            logger.info("Portfolio analysis recommends selling: %s", sell_list)
            handle_daily_sell(sell_list, alpaca_client)
        else:
            logger.info("Portfolio analysis: no sells recommended")
    except Exception as exc:
        logger.error("Portfolio analysis failed: %s", exc)

    # --- New opportunity scan ---
    try:
        buying_power = api.get_portf_buying_power(alpaca_client)
        raw_result = AIC.daily_market_analysis(
            client,
            current_port=current_port_state,
            buying_power=buying_power,
        )

        opportunities = _parse_multi_json_opportunities(raw_result)

        if not opportunities:
            logger.info("Market analysis: no opportunity")
        else:
            logger.info("Market analysis: %d opportunity/ies returned", len(opportunities))
            for opp in opportunities:
                logger.info("Evaluating opportunity: %s", opp.get("ticker", "?"))
                analyzeAIResult(opp, alpaca_client)
    except Exception as exc:
        logger.error("Market analysis failed: %s", exc)

    logger.info("=== Daily task cycle complete ===")


def _run_small_tasks() -> None:
    logger.info("Running small task cycle")


# ---------------------------------------------------------------------------
# One-off CLI commands
# ---------------------------------------------------------------------------

def _run_daily_market_analysis_once() -> None:
    _setup_logging()
    market = input("Choose market (e.g. US, SWE, POL): ").strip() or "US"
    client = _build_gemini_client()
    raw = AIC.daily_market_analysis(client, market=market)

    if raw.strip().lower() == "no opportunity":
        print(json.dumps({"result": "no opportunity"}, indent=2))
        return

    objects: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass

    if objects:
        output = objects[0] if len(objects) == 1 else objects
        print(json.dumps(output, indent=2))
    else:
        print(raw)


def _run_daily_market_analysis_v2_once() -> None:
    _setup_logging()
    market = input("Choose market (e.g. US, SWE, POL): ").strip() or "US"
    client = _build_gemini_client()
    raw = AIC.daily_market_analysis_v2(client, market=market)

    if raw.strip().lower() == "no opportunity":
        print(json.dumps({"result": "no opportunity"}, indent=2))
        return

    objects: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass

    if objects:
        output = objects[0] if len(objects) == 1 else objects
        print(json.dumps(output, indent=2))
    else:
        print(raw)


def _run_single_stock_analysis_once(stock_query: str) -> None:
    _setup_logging()
    client = _build_gemini_client()
    print(AIC.single_stock_analysis(stock_query, client))


def _run_single_stock_analysis_v2_once(stock_query: str) -> None:
    _setup_logging()
    client = _build_gemini_client()
    print(AIC.single_stock_analysis_v2(stock_query, client))


def _print_usage() -> None:
    print("Usage:")
    print("  python main.py                           # run daily market analysis once (interactive)")
    print("  python main.py daily-market-analysis     # v1: 30-180 day swing scan")
    print("  python main.py daily-market-analysis-v2  # v2: 1-12 month scan, grade A/B setups only")
    print("  python main.py analyze-stock AMD")
    print("  python main.py analyze-asset gold")
    print("  python main.py analyze-stock-v2 AMD      # v2: 1-12 month swing analysis (enhanced)")
    print("  python main.py AMD                       # shorthand single-asset analysis")
    print("  python main.py scheduler                 # start the scheduler loop")


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

    if command == "daily-market-analysis-v2":
        _run_daily_market_analysis_v2_once()
        return

    if command in {"analyze-stock", "analyze-asset"}:
        stock_query = " ".join(argv[1:]).strip()
        if not stock_query:
            raise SystemExit(
                "Usage: python main.py analyze-stock <ticker-or-asset-query>"
            )
        _run_single_stock_analysis_once(stock_query)
        return

    if command == "analyze-stock-v2":
        stock_query = " ".join(argv[1:]).strip()
        if not stock_query:
            raise SystemExit(
                "Usage: python main.py analyze-stock-v2 <ticker-or-asset-query>"
            )
        _run_single_stock_analysis_v2_once(stock_query)
        return

    _run_single_stock_analysis_once(" ".join(argv).strip())


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------

def _main() -> None:
    _setup_logging()
    logger.info("=== Scheduler starting ===")
    logger.info("Run times: %s", ", ".join(RUN_TIMES))
    logger.info("Logs directory: %s", LOGS_DIR.resolve())

    client = _build_gemini_client()
    alpaca_client = api.init_client()
    run_times = sorted(RUN_TIMES)

    next_daily_run = _next_daily_run(datetime.now(), run_times)
    next_small_run = datetime.now() + timedelta(minutes=SMALL_TASK_INTERVAL_MINUTES)
    logger.info("Next scheduled run: %s", next_daily_run.strftime("%Y-%m-%d %H:%M"))

    while True:
        now = datetime.now()

        if now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            resume_at = datetime.combine(
                (now + timedelta(days=days_until_monday)).date(),
                datetime.min.time(),
            )
            logger.info(
                "Weekend — sleeping until %s",
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
                logger.error("Unhandled error in daily task cycle: %s", exc, exc_info=True)
            finally:
                next_daily_run = _next_daily_run(datetime.now(), run_times)
                logger.info("Next scheduled run: %s", next_daily_run.strftime("%Y-%m-%d %H:%M"))

        if now >= next_small_run:
            _run_small_tasks()
            next_small_run = datetime.now() + timedelta(minutes=SMALL_TASK_INTERVAL_MINUTES)

        next_wake = min(next_daily_run, next_small_run)
        sleep_secs = max(0.0, (next_wake - datetime.now()).total_seconds())
        logger.info("Sleeping %.0f seconds until %s", sleep_secs, next_wake.strftime("%H:%M"))
        time.sleep(sleep_secs)


if __name__ == "__main__":
    _dispatch_cli(sys.argv[1:])
