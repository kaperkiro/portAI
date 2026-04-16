import unittest

from AI_calls import _parse_single_stock_search_result


class ParseSingleStockSearchResultTests(unittest.TestCase):
    def test_parses_expected_labeled_output(self) -> None:
        payload = _parse_single_stock_search_result(
            "\n".join(
                [
                    "QUERY: amd",
                    "TICKER: AMD",
                    "COMPANY: Advanced Micro Devices, Inc.",
                    "MATCH_CONFIDENCE: HIGH",
                    "SEARCH_SUMMARY: Strong AI-related demand remains in focus.",
                    "RECENT_CATALYSTS: Product launches and data center momentum.",
                    "KEY_RISKS: Competition and guidance pressure.",
                ]
            )
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["QUERY"], "amd")
        self.assertEqual(payload["TICKER"], "AMD")
        self.assertEqual(payload["COMPANY"], "Advanced Micro Devices, Inc.")

    def test_accepts_code_fenced_model_output(self) -> None:
        payload = _parse_single_stock_search_result(
            """```text
QUERY: amd
TICKER: AMD
COMPANY: Advanced Micro Devices, Inc.
MATCH_CONFIDENCE: HIGH
SEARCH_SUMMARY: Summary
RECENT_CATALYSTS: Catalysts
KEY_RISKS: Risks
```"""
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["TICKER"], "AMD")

    def test_accepts_name_field_for_non_equity_instruments(self) -> None:
        payload = _parse_single_stock_search_result(
            "\n".join(
                [
                    "QUERY: gold",
                    "TICKER: GC=F",
                    "NAME: Gold Futures",
                    "INSTRUMENT_TYPE: commodity-future",
                    "MATCH_CONFIDENCE: HIGH",
                    "SEARCH_SUMMARY: Gold is being used as a macro hedge.",
                    "RECENT_CATALYSTS: Rates and central-bank demand remain key.",
                    "KEY_RISKS: Real yields could rise.",
                ]
            )
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["TICKER"], "GC=F")
        self.assertEqual(payload["COMPANY"], "Gold Futures")

    def test_returns_none_for_no_match(self) -> None:
        self.assertIsNone(_parse_single_stock_search_result("NO_MATCH"))


if __name__ == "__main__":
    unittest.main()
