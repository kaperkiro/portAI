Current branch: sharpe

## Current Objective
Optimization pass on trade evaluation quality. The branch adds two measurement layers before an order is submitted:
1. **Ex-ante Sharpe ratio gate** — `compute_ex_ante_sharpe()` in `helpers.py`; minimum threshold `SHARPE_MIN = 0.4` enforced in `main.py:analyzeAIResult()`.
2. **R:R gate** — minimum 2:1 reward-to-risk on TP1 before even computing Sharpe.

Both gates run in `analyzeAIResult()` (main.py:144–222) on every opportunity returned by `daily_market_analysis()`.

## Uncommitted Changes (as of 2026-04-25)
- `.gitignore` — added `.obsidian/`
- `AI_calls.py` — added `import json`; wired `ma_summary` into `get_current_ticker_data()`; normalized `price_targets` to plain dict; improved `daily_port_analysis` prompt copy
- `CLAUDE.md` — misc updates
- `helpers.py` — added `compute_ex_ante_sharpe`, `YFMovingAverageSummary`, `YFRelativeStrengthAnalysis`, `YFMarketCorrelation`
- `main.py` — `analyzeAIResult()` now has R:R gate → Sharpe gate → bracket order split

## Next Steps
- Commit the uncommitted changes on the `sharpe` branch
- Consider merging `sharpe` → `main` once stable
- Potential next improvements: portfolio-level Sharpe / correlation deduplication, multi-market support in `daily_market_analysis`
