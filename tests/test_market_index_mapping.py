import unittest
from unittest.mock import Mock, patch

import pandas as pd

import helpers as hp
from AI_calls import _resolve_market_index_ticker, get_current_ticker_data


class _FakeTicker:
    def __init__(self, info: dict | None = None) -> None:
        self.info = info or {}
        self.fast_info = {
            "last_price": 250.1,
            "previous_close": 249.0,
            "day_high": 252.1,
            "day_low": 247.4,
            "year_high": 294.0,
            "year_low": 195.3,
            "market_cap": 24988000000,
            "shares_outstanding": 100000000,
            "currency": "PLN",
            "quote_type": "EQUITY",
        }
        self.calendar = None
        self.earnings_dates = None
        self.recommendations = None

    def history(self, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
        return pd.DataFrame()

    def get_analyst_price_targets(self):
        return None


class MarketIndexMappingTests(unittest.TestCase):
    def test_resolves_warsaw_ticker_suffix_to_wig20(self) -> None:
        self.assertEqual(
            _resolve_market_index_ticker(exchange=None, ticker="CDR.WA"),
            "WIG20.WA",
        )

    def test_keeps_existing_exchange_mapping_for_other_markets(self) -> None:
        self.assertEqual(
            _resolve_market_index_ticker(exchange="STO", ticker="VOLV-B.ST"),
            "^OMX",
        )

    def test_formats_wig20_label_without_yahoo_suffix(self) -> None:
        payload = hp.YFMarketCorrelation.from_histories(
            None,
            None,
            index_ticker="WIG20.WA",
        ).to_dict()

        self.assertEqual(payload["index"], "WIG20")
        self.assertEqual(payload["interpretation"], "insufficient-data")

    @patch("AI_calls.hp.YFMarketCorrelation.from_histories")
    @patch("AI_calls.yf.Ticker")
    def test_get_current_ticker_data_uses_wig20_for_polish_stocks(
        self,
        mock_yf_ticker,
        mock_market_correlation,
    ) -> None:
        def ticker_factory(symbol: str) -> _FakeTicker:
            if symbol == "CDR.WA":
                return _FakeTicker(
                    info={
                        "exchange": None,
                        "longName": "CD Projekt S.A.",
                        "quoteType": "EQUITY",
                    }
                )
            if symbol == "WIG20.WA":
                return _FakeTicker()
            raise AssertionError(f"Unexpected ticker requested: {symbol}")

        mock_yf_ticker.side_effect = ticker_factory
        mock_market_correlation.return_value = Mock(
            to_dict=Mock(
                return_value={
                    "index": "WIG20",
                    "correlation_6_12m": None,
                    "beta_6_12m": None,
                    "interpretation": "insufficient-data",
                }
            )
        )

        payload = get_current_ticker_data("CDR.WA")

        self.assertIn("ticker: CDR.WA", payload)
        self.assertIn("[market_correlation]", payload)
        self.assertIn("- index: WIG20", payload)
        mock_market_correlation.assert_called_once()
        self.assertEqual(
            mock_market_correlation.call_args.kwargs["index_ticker"],
            "WIG20.WA",
        )
        self.assertTrue(
            any(call.args == ("WIG20.WA",) for call in mock_yf_ticker.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
