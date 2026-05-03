import unittest

from helpers import compute_ex_ante_sharpe


class SharpeFormulaTests(unittest.TestCase):
    """
    Unit tests for the fixed ex-ante Sharpe formula.

    Reference setup: entry=100, stop=95, TP1=110, TP2=120, ATR=2
    Expected range: 1.0–1.5 (grade B, no realized vol supplied)
    """

    def _sharpe(self, entry, stop, tp1, tp2, atr, grade="B", realized_vol=None):
        return compute_ex_ante_sharpe(
            entry, stop, tp1, tp2, atr,
            realized_vol_pct=realized_vol,
            setup_grade=grade,
        )

    def test_reference_setup_in_range(self):
        """Standard 2:1 R:R setup should produce 1.0–1.5, not 4+."""
        result = self._sharpe(100, 95, 110, 120, 2)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.9)
        self.assertLess(result, 1.6)

    def test_grade_a_higher_than_grade_b(self):
        """Grade A (60% win rate) must produce a higher Sharpe than grade B (55%)."""
        sharpe_a = self._sharpe(100, 95, 110, 120, 2, grade="A")
        sharpe_b = self._sharpe(100, 95, 110, 120, 2, grade="B")
        self.assertIsNotNone(sharpe_a)
        self.assertIsNotNone(sharpe_b)
        self.assertGreater(sharpe_a, sharpe_b)

    def test_realized_vol_lowers_sharpe_vs_atr_fallback(self):
        """
        Supplying a higher realized vol (e.g. 2% daily) should produce a lower Sharpe
        than the ATR-derived fallback (ATR/entry/1.35 = 2/(100*1.35) ≈ 1.48% daily).
        """
        sharpe_atr = self._sharpe(100, 95, 110, 120, 2, realized_vol=0)
        sharpe_realvol = self._sharpe(100, 95, 110, 120, 2, realized_vol=2.0)
        self.assertIsNotNone(sharpe_atr)
        self.assertIsNotNone(sharpe_realvol)
        self.assertGreater(sharpe_atr, sharpe_realvol)

    def test_degenerate_stop_inside_atr_band_returns_none(self):
        """Stop < 0.5×ATR is inside noise band — must return None, not a fake high Sharpe."""
        # ATR=2, risk=0.5 → risk < 0.5*ATR (1.0) → degenerate
        result = self._sharpe(100, 99.5, 115, 130, 2)
        self.assertIsNone(result)

    def test_tight_stop_large_tp2_no_longer_inflates_to_4plus(self):
        """
        Scenario that previously produced Sharpe ≈ 4+:
        very tight stop, TP2 disproportionately large.
        New formula should cap this well below 3.
        """
        result = self._sharpe(500, 499, 600, 750, 4)
        # stop=1, atr=4 → risk < 0.5*atr → degenerate → None
        self.assertIsNone(result)

    def test_negative_risk_returns_none(self):
        """stop_loss > entry is invalid."""
        result = self._sharpe(100, 105, 120, 140, 2)
        self.assertIsNone(result)

    def test_negative_reward_returns_none(self):
        """TP1 below entry is invalid."""
        result = self._sharpe(100, 95, 90, 80, 2)
        self.assertIsNone(result)

    def test_realistic_large_cap_setup(self):
        """
        Large-cap setup: entry=400, stop=388, TP1=440, TP2=480, ATR=6 (1.5% daily ATR).
        Should produce a reasonable Sharpe in 0.8–1.8 range.
        """
        result = self._sharpe(400, 388, 440, 480, 6)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.7)
        self.assertLess(result, 2.0)


if __name__ == "__main__":
    unittest.main()
