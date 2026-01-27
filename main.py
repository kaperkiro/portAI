from dotenv import load_dotenv
import os
import classes as cl
from datetime import datetime, timedelta
import time
from google import genai

load_dotenv()  # reads .env in current working dir
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def daily_port_analysis(portfolio: cl.Portfolio):
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
    
    This is the portfolio information: {portfolio}
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        tools=[{"google_search": {}}],
    )


RUN_TIMES = ["10:00", "14:00"]  # local time; edit this list to change runs per day


def _main() -> None:
    portfolio = cl.Portfolio()
    run_times = sorted(RUN_TIMES)
    while True:
        now = datetime.now()
        today = now.date()
        next_run = None
        for t in run_times:
            h, m = (int(x) for x in t.split(":"))
            candidate = datetime.combine(today, datetime.min.time()) + timedelta(
                hours=h, minutes=m
            )
            if now < candidate:
                next_run = candidate
                break
        if next_run is None:
            h, m = (int(x) for x in run_times[0].split(":"))
            next_run = datetime.combine(
                today + timedelta(days=1), datetime.min.time()
            ) + timedelta(hours=h, minutes=m)
        time.sleep(max(0.0, (next_run - now).total_seconds()))
        daily_port_analysis(portfolio)


if __name__ == "__main__":
    _main()
