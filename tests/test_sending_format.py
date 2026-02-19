import unittest

import pandas as pd

from pipeline.common.use_predictions import sending_functions as sf


class SendingFormatTests(unittest.TestCase):
    def test_format_price_prefixes_dollar_sign(self):
        self.assertEqual(sf._format_price(2.46), "$2.46")
        self.assertEqual(sf._format_price(1), "$1.00")

    def test_format_price_handles_na(self):
        self.assertEqual(sf._format_price(pd.NA), "n/a")


if __name__ == "__main__":
    unittest.main()
