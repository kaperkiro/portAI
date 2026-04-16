import unittest

from AI_calls import (
    _resolve_known_single_instrument_query,
    _uses_price_history_only_market_data,
)


class SingleInstrumentAliasTests(unittest.TestCase):
    def test_resolves_gold_query(self) -> None:
        payload = _resolve_known_single_instrument_query("gold")

        self.assertIsNotNone(payload)
        self.assertEqual(payload["TICKER"], "GC=F")
        self.assertEqual(payload["COMPANY"], "Gold Futures")

    def test_resolves_brent_query_variants(self) -> None:
        for query in ["brent oil", "oil brent", "BZ=F"]:
            with self.subTest(query=query):
                payload = _resolve_known_single_instrument_query(query)
                self.assertIsNotNone(payload)
                self.assertEqual(payload["TICKER"], "BZ=F")

    def test_resolves_typoed_natural_gas_query(self) -> None:
        payload = _resolve_known_single_instrument_query("natrual gas")

        self.assertIsNotNone(payload)
        self.assertEqual(payload["TICKER"], "NG=F")
        self.assertEqual(payload["COMPANY"], "Natural Gas Futures")

    def test_returns_none_for_unknown_alias(self) -> None:
        self.assertIsNone(_resolve_known_single_instrument_query("totally unknown thing"))

    def test_flags_futures_as_price_history_only(self) -> None:
        self.assertTrue(_uses_price_history_only_market_data("GC=F"))
        self.assertTrue(_uses_price_history_only_market_data("ng=f"))
        self.assertFalse(_uses_price_history_only_market_data("AMD"))


if __name__ == "__main__":
    unittest.main()
