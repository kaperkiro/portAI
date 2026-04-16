import unittest

from AI_calls import (
    _build_single_stock_summary_section,
    _parse_single_stock_analysis_result,
    _remove_single_stock_summary_driver_lines,
)


class SingleStockSummarySectionTests(unittest.TestCase):
    def test_parses_labeled_single_analysis_output(self) -> None:
        payload = _parse_single_stock_analysis_result(
            "\n".join(
                [
                    "QUERY: AMD",
                    "TICKER: AMD",
                    "COMPANY: Advanced Micro Devices, Inc.",
                    "BIAS: BULLISH",
                    "TRADE_DIRECTION: LONG",
                    "CONFIDENCE: 7",
                    "WHAT_LOOKS_GOOD: AI demand and improving margins.",
                    "WHAT_COULD_GO_WRONG: Weak guidance or a multiple reset.",
                    "UPSIDE_1: AI server demand stays strong",
                    "UPSIDE_2: Data center share gains",
                    "UPSIDE_3: Better gross margins",
                    "UPSIDE_4: Large customer wins",
                    "UPSIDE_5: Softer rate backdrop",
                    "DOWNSIDE_1: Weak quarterly guidance",
                    "DOWNSIDE_2: Nvidia competition",
                    "DOWNSIDE_3: PC demand slows again",
                    "DOWNSIDE_4: Higher Treasury yields",
                    "DOWNSIDE_5: Export restriction risk",
                ]
            )
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["TICKER"], "AMD")
        self.assertEqual(payload["UPSIDE_1"], "AI server demand stays strong")
        self.assertEqual(payload["DOWNSIDE_5"], "Export restriction risk")

    def test_builds_summary_table_with_five_rows(self) -> None:
        section = _build_single_stock_summary_section(
            {
                "UPSIDE_1": "AI server demand stays strong",
                "UPSIDE_2": "Data center share gains",
                "UPSIDE_3": "Better gross margins",
                "UPSIDE_4": "Large customer wins",
                "UPSIDE_5": "Softer rate backdrop",
                "DOWNSIDE_1": "Weak quarterly guidance",
                "DOWNSIDE_2": "Nvidia competition",
                "DOWNSIDE_3": "PC demand slows again",
                "DOWNSIDE_4": "Higher Treasury yields",
                "DOWNSIDE_5": "Export restriction risk",
            }
        )

        self.assertIn("Summary", section)
        self.assertIn("+------------------------------------+------------------------------------+", section)
        self.assertIn("| Likely To Push It Up               | Could Push It Down                 |", section)
        self.assertIn("| 1. AI server demand stays strong   | 1. Weak quarterly guidance         |", section)
        self.assertIn("| 5. Softer rate backdrop            | 5. Export restriction risk         |", section)

    def test_falls_back_to_existing_analysis_fields_when_ranked_items_missing(self) -> None:
        section = _build_single_stock_summary_section(
            {
                "WHAT_LOOKS_GOOD": "Central-bank buying remains firm. Real yields are easing.",
                "WHAT_COULD_GO_WRONG": "The dollar could strengthen. Real yields may move higher.",
                "CONCLUSION": "Momentum is constructive, but positioning is crowded.",
            },
            search_payload={
                "RECENT_CATALYSTS": "ETF inflows improved.",
                "KEY_RISKS": "Hawkish Fed language.",
            },
        )

        self.assertIn("Central-bank buying remains", section)
        self.assertIn("| firm                               |", section)
        self.assertIn("The dollar could strengthen", section)
        self.assertIn("ETF inflows improved", section)
        self.assertIn("Hawkish Fed language", section)

    def test_wraps_long_cells_for_terminal_readability(self) -> None:
        section = _build_single_stock_summary_section(
            {
                "UPSIDE_1": "Very long catalyst that should wrap cleanly inside the terminal table without breaking layout",
                "DOWNSIDE_1": "Very long risk that should also wrap cleanly inside the terminal table without breaking layout",
            }
        )

        self.assertIn("| 1. Very long catalyst that should  | 1. Very long risk that should also |", section)
        self.assertIn("| wrap cleanly inside the terminal   | wrap cleanly inside the terminal   |", section)
        self.assertIn("| table without breaking layout      | table without breaking layout      |", section)

    def test_removes_ranked_driver_lines_from_primary_output(self) -> None:
        cleaned = _remove_single_stock_summary_driver_lines(
            "\n".join(
                [
                    "QUERY: AMD",
                    "TICKER: AMD",
                    "WHAT_LOOKS_GOOD: AI demand is strong.",
                    "UPSIDE_1: AI server demand stays strong",
                    "UPSIDE_2: Data center share gains",
                    "DOWNSIDE_1: Weak quarterly guidance",
                    "CONCLUSION: Setup still looks constructive.",
                ]
            )
        )

        self.assertIn("QUERY: AMD", cleaned)
        self.assertIn("WHAT_LOOKS_GOOD: AI demand is strong.", cleaned)
        self.assertIn("CONCLUSION: Setup still looks constructive.", cleaned)
        self.assertNotIn("UPSIDE_1:", cleaned)
        self.assertNotIn("UPSIDE_2:", cleaned)
        self.assertNotIn("DOWNSIDE_1:", cleaned)


if __name__ == "__main__":
    unittest.main()
