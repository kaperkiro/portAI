import unittest
import pandas as pd
import numpy as np

from helpers import YFPriceHistorySummary


def _make_history(n_days: int = 60, daily_return_pct: float = 0.5) -> pd.DataFrame:
    """Build a synthetic daily OHLCV DataFrame with seeded noise so vol > 0."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_days)
    close = 100.0
    rows = []
    for _ in dates:
        noise = rng.normal(0, 0.3)  # ±0.3% noise per bar
        close *= 1 + (daily_return_pct + noise) / 100
        rows.append({
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": 1_000_000,
        })
    return pd.DataFrame(rows, index=dates)


def _make_spiked_history(n_days: int = 60, spike_14d_pct: float = 30.0) -> pd.DataFrame:
    """
    Normal drift for first (n_days - 14) days, then a large 14-day spike.
    The spike is spread across 14 bars so the daily vol is elevated.
    """
    total = n_days
    normal_days = total - 14
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=total)
    close = 100.0
    rows = []
    daily_spike = spike_14d_pct / 14.0

    for i, _ in enumerate(dates):
        if i < normal_days:
            close *= 1.001  # ~0.1%/day normal
        else:
            close *= 1 + daily_spike / 100
        rows.append({
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": 1_000_000,
        })
    return pd.DataFrame(rows, index=dates)


class CatalystStalenessFieldTests(unittest.TestCase):
    """
    Unit tests for the new close_to_close_vol_14d_pct and return_14d_pct fields
    added to YFPriceHistorySummary.
    """

    def test_vol_and_return_populated_with_sufficient_history(self):
        """Both new fields must be non-None when ≥15 days of data are available."""
        hist = _make_history(n_days=60)
        summary = YFPriceHistorySummary.from_history(hist)
        self.assertIsNotNone(summary.close_to_close_vol_14d_pct)
        self.assertIsNotNone(summary.return_14d_pct)

    def test_vol_is_positive(self):
        """Realized vol must be a positive number."""
        hist = _make_history(n_days=60, daily_return_pct=1.0)
        summary = YFPriceHistorySummary.from_history(hist)
        self.assertGreater(summary.close_to_close_vol_14d_pct, 0)

    def test_return_14d_reflects_spike(self):
        """A 30% spike in 14 days must produce return_14d_pct ≈ 30%."""
        hist = _make_spiked_history(n_days=60, spike_14d_pct=30.0)
        summary = YFPriceHistorySummary.from_history(hist)
        self.assertIsNotNone(summary.return_14d_pct)
        self.assertAlmostEqual(summary.return_14d_pct, 30.0, delta=5.0)

    def test_large_spike_exceeds_2sigma(self):
        """
        A 30% spike in 14 days with normal background vol (~0.1%/day) must
        exceed the 2σ×√14 threshold that drives the catalyst_staleness_warning.

        This verifies the fields contain enough signal for the warning logic in
        get_current_ticker_data — the warning itself is a prompt-level output
        computed there, not here.
        """
        hist = _make_spiked_history(n_days=60, spike_14d_pct=30.0)
        summary = YFPriceHistorySummary.from_history(hist)
        cc_vol = summary.close_to_close_vol_14d_pct
        ret_14d = summary.return_14d_pct
        self.assertIsNotNone(cc_vol)
        self.assertIsNotNone(ret_14d)
        import math
        two_sigma_14d = 2.0 * cc_vol * math.sqrt(14)
        self.assertGreater(abs(ret_14d), two_sigma_14d,
                           msg=f"ret_14d={ret_14d:.1f}% should exceed 2σ×√14={two_sigma_14d:.1f}%")

    def test_fields_none_with_insufficient_history(self):
        """Fewer than 15 bars → new fields should be None (not crash)."""
        hist = _make_history(n_days=10)
        summary = YFPriceHistorySummary.from_history(hist)
        self.assertIsNone(summary.return_14d_pct)

    def test_empty_dataframe_returns_all_none(self):
        """Empty DataFrame must return an all-None summary (no exception)."""
        summary = YFPriceHistorySummary.from_history(pd.DataFrame())
        self.assertIsNone(summary.last_close)
        self.assertIsNone(summary.close_to_close_vol_14d_pct)
        self.assertIsNone(summary.return_14d_pct)


if __name__ == "__main__":
    unittest.main()
