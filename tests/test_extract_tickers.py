import unittest

from AI_calls import _extract_tickers


class ExtractTickersTests(unittest.TestCase):
    def test_keeps_swedish_share_class_tickers(self) -> None:
        self.assertEqual(
            _extract_tickers("ATCO-A.ST, SEB-A.ST, VOLV-B.ST"),
            ["ATCO-A.ST", "SEB-A.ST", "VOLV-B.ST"],
        )

    def test_accepts_common_model_output_variants(self) -> None:
        self.assertEqual(
            _extract_tickers("TICKERS: ATCO-A.ST;\n2. SEB-A.ST\n- BRK.B"),
            ["ATCO-A.ST", "SEB-A.ST", "BRK.B"],
        )

    def test_deduplicates_and_normalizes_case(self) -> None:
        self.assertEqual(
            _extract_tickers("atco-a.st, SEB-A.ST, ATCO-A.ST"),
            ["ATCO-A.ST", "SEB-A.ST"],
        )


if __name__ == "__main__":
    unittest.main()
