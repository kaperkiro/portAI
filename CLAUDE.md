# CLAUDE.md

# AI Coding Assistant Guidelines

You are interacting with a codebase that uses an external Markdown memory system. 

Before starting any task or writing code, you MUST:
1. Read `claude_memory/01_active_context.md` to understand the current objective.
2. Read `claude_memory/02_architecture.md` for tech stack rules and constraints.

When you finish a task, I will ask you to "update memory." When I do this, you must:
1. Update `claude_memory/01_active_context.md` with the next steps.
2. Append a brief summary of our changes to `claude_memory/03_changelog.md`.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**portAI** is an AI-powered financial market scanner and automated swing-trading CLI. It uses Google Gemini (with extended thinking) to analyze markets and Alpaca for paper-trade execution.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Required `.env`:
```
GEMINI_API_KEY=...         # Required for all commands
alpaca_key=...             # Required for scheduler/trading
alpaca_secret_key=...      # Required for scheduler/trading
```

## Commands

```bash
# One-off daily market analysis
python main.py
python main.py daily-market-analysis

# Analyze a single stock or asset
python main.py analyze-stock AMD
python main.py analyze-asset gold
python main.py AMD   # shorthand

# Start the continuous scheduler
python main.py scheduler
```

## Tests

```bash
# Run all tests
python -m unittest discover tests/ -v

# Run a single test file
python -m unittest tests.test_extract_tickers -v
```

No linting tools are configured.

## Architecture

The system is a multi-stage AI analysis pipeline:

```
main.py (CLI + scheduler)
  └→ AI_calls.py (Gemini analysis)
       ├→ daily_market_analysis()   [2-stage: broad scan → deep dive]
       ├→ single_stock_analysis()   [search + analysis for one asset]
       └→ daily_port_analysis()     [portfolio management decisions]
            ↓ (on trade signal)
       api.py (Alpaca paper trading)
            ↓
       classes.py (Portfolio state → data.json)
```

### Key Files

- **main.py** — CLI dispatcher and scheduler loop. Runs at `RUN_TIMES = ["11:40", "17:30", "19:30", "22:00"]`, pauses on weekends.
- **AI_calls.py** — All Gemini interactions. Stage 1 uses Google Search to identify candidates; Stage 2 does a deep yfinance-backed analysis. Enforces strict JSON output formats.
- **helpers.py** — Formats yfinance data into AI-friendly structures. Contains `YFInfoFundamentals`, `format_ticker_data_for_ai()`, and ~15 dataclasses for financial metrics.
- **classes.py** — Core data models (`Stock`, `Fund`, `Portfolio`, `TradingPlan`). Portfolio state serialized to `data.json` with atomic writes. Includes `parse_ai_response_payload()` for normalizing Gemini responses (handles markdown code fences).
- **api.py** — Alpaca paper-trading client. `bracketBuy()` splits large orders into two equal-quantity bracket orders with stop-loss and take-profit legs.

### AI Integration Pattern

- Gemini with `thinking_level="high"` (extended thinking enabled)
- Two-stage flow: broad market scan → focused deep-dive on shortlisted tickers
- `AI_calls.py` uses `get_current_ticker_data()` as a Gemini tool call (backed by yfinance)
- JSON parsing falls back to text parsing when Gemini wraps output in markdown fences

### Trading Pattern

- Always paper trading (`paper=True` in `api.py`)
- Bracket orders with automatic take-profit and stop-loss legs
- `INDEX_MAP` in `AI_calls.py` covers 30+ global market indices (NYSE, NASDAQ, Swedish, Polish, etc.)
- `KNOWN_SINGLE_INSTRUMENT_ALIASES` maps ~40 commodity/asset names to yfinance tickers (e.g., `gold → GC=F`)
