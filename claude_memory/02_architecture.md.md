# Architecture & Tech Stack Rules

## Stack
- **Python 3.11+**
- **Gemini** via `google-genai` SDK — model `gemini-3.1-pro-preview`, `thinking_level="high"` on all calls
- **yfinance** — all market data (price history, fundamentals, analyst recs, earnings)
- **Alpaca** (`alpaca-trade-api` / `alpaca-py`) — paper trading only (`paper=True`)
- **dotenv** — secrets from `.env`

## File Responsibilities
| File | Role |
|------|------|
| `main.py` | CLI dispatcher + scheduler loop. `analyzeAIResult()` is the trade gate. |
| `AI_calls.py` | All Gemini calls. `get_current_ticker_data()` is exposed as a Gemini tool. `compute_ex_ante_sharpe` is also a tool. |
| `helpers.py` | yfinance data → structured dicts. All dataclasses here. `compute_ex_ante_sharpe` lives here. |
| `api.py` | Alpaca client: `bracketBuy`, `sellStock`, `alpaca_portfolio_context`. |
| `classes.py` | Data models. `parse_ai_response_payload()` strips markdown fences from Gemini JSON. |

## Key Constraints
- ALWAYS paper trading — never set `paper=False`
- Gemini tool calls: only `get_current_ticker_data` and `compute_ex_ante_sharpe` are registered as tools
- `bracketBuy` splits qty > 1 into two half-size bracket orders (TP1 + TP2)
- Trade gate order: R:R ≥ 2.0 → ex-ante Sharpe ≥ 0.4 → order submission
- `INDEX_MAP` in `AI_calls.py` maps exchange codes to benchmark index tickers
- `KNOWN_SINGLE_INSTRUMENT_ALIASES` maps commodity names (gold, silver, brent, etc.) to yfinance tickers

## Scheduler
- `RUN_TIMES = ["10:20", "17:30", "19:30", "22:00"]` (local time)
- Skips weekends entirely
- Each cycle: portfolio sell check → new opportunity scan

## Testing
- `python -m unittest discover tests/ -v`
- No linting configured
