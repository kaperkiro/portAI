# Changelog

## 2026-04-25 — sharpe branch (uncommitted)
- Added `compute_ex_ante_sharpe()` to `helpers.py`: probability-weighted expected return / ATR-scaled volatility
- Added `YFMovingAverageSummary` dataclass (SMA20/50/200 + trend structure label)
- Added `YFMarketCorrelation` dataclass (6-12m correlation + beta vs benchmark index)
- Added `YFRelativeStrengthAnalysis` dataclass (Mansfield RS, rolling OLS beta/alpha, RRG quadrant, composite score)
- Wired `ma_summary` and relative strength into `get_current_ticker_data()` output
- `analyzeAIResult()` in `main.py` now gates on R:R ≥ 2.0 then Sharpe ≥ 0.4 before submitting orders
- `compute_ex_ante_sharpe` exposed as a Gemini tool in `single_stock_analysis` and `daily_market_analysis`
- AI prompts updated to instruct Gemini to call `compute_ex_ante_sharpe` and enforce Sharpe ≥ 0.4

## Earlier (from git log)
- `f64e5e4` feat: add measurments for stock to market correlation
- `45b1b50` branchin for optimization
- `43b9cdc` testing
- `f25c095` testing
