from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import pandas as pd


Number = Union[int, float]


def _get(d: Dict[str, Any], key: str) -> Any:
    return d.get(key)


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        # yfinance often returns numpy scalars
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def _pct_to_float(x: Any) -> Optional[float]:
    """
    yfinance commonly returns ratios as 0.12 for 12%.
    We keep them as percentages (12.0) for readability/AI stability.
    """
    v = _to_float(x)
    if v is None:
        return None
    return v * 100.0


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


# -----------------------------
# 1) info (fundamentals snapshot)
# -----------------------------


@dataclass(frozen=True)
class YFInfoFundamentals:
    # Identity / classification
    sector: Optional[str]
    industry: Optional[str]
    currency: Optional[str]

    # Size & liquidity
    market_cap: Optional[float]
    beta: Optional[float]
    average_volume: Optional[float]

    # Valuation
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    price_to_book: Optional[float]

    # Profitability & growth (percent)
    profit_margins_pct: Optional[float]
    operating_margins_pct: Optional[float]
    gross_margins_pct: Optional[float]
    revenue_growth_pct: Optional[float]
    earnings_growth_pct: Optional[float]

    # 52w context
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]

    # EPS / dividends
    trailing_eps: Optional[float]
    forward_eps: Optional[float]
    dividend_yield_pct: Optional[float]

    # Balance-sheet risk (if present)
    debt_to_equity: Optional[float]

    @staticmethod
    def from_yfinance_info(info: Dict[str, Any]) -> "YFInfoFundamentals":
        """
        Takes yfinance Ticker.info dict and returns a filtered, structured snapshot.
        """
        return YFInfoFundamentals(
            sector=_safe_str(_get(info, "sector")),
            industry=_safe_str(_get(info, "industry")),
            currency=_safe_str(_get(info, "currency")),
            market_cap=_to_float(_get(info, "marketCap")),
            beta=_to_float(_get(info, "beta")),
            average_volume=_to_float(_get(info, "averageVolume")),
            trailing_pe=_to_float(_get(info, "trailingPE")),
            forward_pe=_to_float(_get(info, "forwardPE")),
            price_to_book=_to_float(_get(info, "priceToBook")),
            profit_margins_pct=_pct_to_float(_get(info, "profitMargins")),
            operating_margins_pct=_pct_to_float(_get(info, "operatingMargins")),
            gross_margins_pct=_pct_to_float(_get(info, "grossMargins")),
            revenue_growth_pct=_pct_to_float(_get(info, "revenueGrowth")),
            earnings_growth_pct=_pct_to_float(_get(info, "earningsGrowth")),
            fifty_two_week_high=_to_float(_get(info, "fiftyTwoWeekHigh")),
            fifty_two_week_low=_to_float(_get(info, "fiftyTwoWeekLow")),
            trailing_eps=_to_float(_get(info, "trailingEps")),
            forward_eps=_to_float(_get(info, "forwardEps")),
            dividend_yield_pct=_pct_to_float(_get(info, "dividendYield")),
            debt_to_equity=_to_float(_get(info, "debtToEquity")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# 2) fast_info (price context)
# -----------------------------


@dataclass(frozen=True)
class YFFastInfoSnapshot:
    last_price: Optional[float]
    previous_close: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    year_high: Optional[float]
    year_low: Optional[float]
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    currency: Optional[str]

    @staticmethod
    def from_yfinance_fast_info(fast_info: Dict[str, Any]) -> "YFFastInfoSnapshot":
        """
        Takes yfinance Ticker.fast_info dict-like and returns filtered snapshot.
        """
        # fast_info keys are snake_case
        return YFFastInfoSnapshot(
            last_price=_to_float(_get(fast_info, "last_price")),
            previous_close=_to_float(_get(fast_info, "previous_close")),
            day_high=_to_float(_get(fast_info, "day_high")),
            day_low=_to_float(_get(fast_info, "day_low")),
            year_high=_to_float(_get(fast_info, "year_high")),
            year_low=_to_float(_get(fast_info, "year_low")),
            market_cap=_to_float(_get(fast_info, "market_cap")),
            shares_outstanding=_to_float(_get(fast_info, "shares_outstanding")),
            currency=_safe_str(_get(fast_info, "currency")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# 3) earnings/events (calendar / earnings_dates)
# -----------------------------


@dataclass(frozen=True)
class YFEarningsEvent:
    earnings_date_iso: Optional[str]  # "YYYY-MM-DD"
    earnings_in_days: Optional[int]  # can be negative if already passed

    @staticmethod
    def from_calendar_or_earnings_dates(
        calendar: Optional[pd.DataFrame] = None,
        earnings_dates: Optional[pd.DataFrame] = None,
        *,
        today_utc: Optional[pd.Timestamp] = None,
    ) -> "YFEarningsEvent":
        """
        Best-effort extractor for the next earnings date.
        Provide either `earnings_dates` (preferred) or `calendar` (fallback).
        """
        today = (
            today_utc if today_utc is not None else pd.Timestamp.utcnow().normalize()
        )

        dt: Optional[pd.Timestamp] = None

        # Preferred: earnings_dates (index is dates)
        if isinstance(earnings_dates, pd.DataFrame) and not earnings_dates.empty:
            try:
                # Index can be tz-aware; normalize
                idx = pd.to_datetime(earnings_dates.index)
                # choose the earliest date >= today, else earliest date overall
                future = idx[idx.normalize() >= today]
                dt = future.min() if len(future) else idx.min()
            except Exception:
                dt = None

        # Fallback: calendar
        if dt is None and isinstance(calendar, pd.DataFrame) and not calendar.empty:
            try:
                # Common pattern: index contains "Earnings Date"
                if "Earnings Date" in calendar.index:
                    val = calendar.loc["Earnings Date"].values
                    if len(val):
                        dt = pd.to_datetime(val[0])
                # Another pattern: column contains "Earnings Date"
                elif "Earnings Date" in calendar.columns:
                    val = calendar["Earnings Date"].dropna()
                    if not val.empty:
                        dt = pd.to_datetime(val.iloc[0])
            except Exception:
                dt = None

        if dt is None or pd.isna(dt):
            return YFEarningsEvent(None, None)

        dt_norm = pd.to_datetime(dt).normalize()
        days = int((dt_norm - today).days)

        return YFEarningsEvent(
            earnings_date_iso=str(dt_norm.date()),
            earnings_in_days=days,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# 4) corporate actions (actions/dividends/splits)
# -----------------------------


@dataclass(frozen=True)
class YFCorporateActions:
    last_split_date_iso: Optional[str]
    last_split_ratio: Optional[float]
    last_dividend_date_iso: Optional[str]
    last_dividend_amount: Optional[float]

    @staticmethod
    def from_actions_dividends_splits(
        actions: Optional[pd.DataFrame] = None,
        dividends: Optional[pd.Series] = None,
        splits: Optional[pd.Series] = None,
    ) -> "YFCorporateActions":
        """
        Keep only the most recent split/dividend information.
        """
        last_split_date = None
        last_split_ratio = None
        if isinstance(splits, pd.Series) and not splits.empty:
            try:
                last_split_date = pd.to_datetime(splits.index[-1]).date().isoformat()
                last_split_ratio = _to_float(splits.iloc[-1])
            except Exception:
                pass

        last_div_date = None
        last_div_amt = None
        if isinstance(dividends, pd.Series) and not dividends.empty:
            try:
                last_div_date = pd.to_datetime(dividends.index[-1]).date().isoformat()
                last_div_amt = _to_float(dividends.iloc[-1])
            except Exception:
                pass

        return YFCorporateActions(
            last_split_date_iso=last_split_date,
            last_split_ratio=last_split_ratio,
            last_dividend_date_iso=last_div_date,
            last_dividend_amount=last_div_amt,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# 5) optional: analyst recs (soft signal)
# -----------------------------


@dataclass(frozen=True)
class YFAnalystSignal:
    # Very compact: last action + simple trend direction
    latest_grade: Optional[str]
    latest_firm: Optional[str]
    latest_action_iso: Optional[str]
    trend_90d: Optional[str]  # "improving" | "deteriorating" | "mixed" | "unknown"

    @staticmethod
    def from_recommendations(
        recommendations: Optional[pd.DataFrame],
    ) -> "YFAnalystSignal":
        """
        yfinance recommendations can be noisy. We keep only a minimal summary.
        """
        if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
            return YFAnalystSignal(None, None, None, "unknown")

        df = recommendations.copy()
        # Common cols: Firm, To Grade, From Grade, Action
        # Index is datetime
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except Exception:
            pass

        latest = df.iloc[-1] if len(df) else None
        latest_date_iso = None
        try:
            latest_date_iso = pd.to_datetime(df.index[-1]).date().isoformat()
        except Exception:
            pass

        latest_grade = _safe_str(latest.get("To Grade")) if latest is not None else None
        latest_firm = _safe_str(latest.get("Firm")) if latest is not None else None

        # Trend in last 90 days: count upgrades vs downgrades if "Action" exists
        trend = "unknown"
        try:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=90)
            recent = df[df.index >= cutoff]
            if "Action" in recent.columns and not recent.empty:
                actions = recent["Action"].astype(str).str.lower()
                up = actions.str.contains("up").sum()
                down = actions.str.contains("down").sum()
                if up > down:
                    trend = "improving"
                elif down > up:
                    trend = "deteriorating"
                else:
                    trend = "mixed"
        except Exception:
            trend = "unknown"

        return YFAnalystSignal(
            latest_grade=latest_grade,
            latest_firm=latest_firm,
            latest_action_iso=latest_date_iso,
            trend_90d=trend,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Example usage (you wire yfinance calls yourself):
#   info_filtered = YFInfoFundamentals.from_yfinance_info(info).to_dict()
#   fast_filtered = YFFastInfoSnapshot.from_yfinance_fast_info(fast_info).to_dict()
#   earnings = YFEarningsEvent.from_calendar_or_earnings_dates(calendar, earnings_dates).to_dict()
#   actions = YFCorporateActions.from_actions_dividends_splits(actions_df, dividends_series, splits_series).to_dict()
#   analyst = YFAnalystSignal.from_recommendations(recs_df).to_dict()
# -----------------------------
