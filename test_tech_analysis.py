# Test.py ✅推荐改名为：test_tech_analysis.py
import unittest
from TechnicalAnalysis import TechnicalAnalysis  # 替换为你的类所在模块
import pandas as pd

class TestTechnicalAnalysis(unittest.TestCase):

    def setUp(self):
        self.ta = TechnicalAnalysis("TEST1.csv")

    def test_load_data(self):
        df = self.ta.df
        self.assertFalse(df.empty)
        self.assertIn('Date', df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['Date']))
        self.assertIsInstance(df['Volume'].iloc[0], float)

    def test_calculate_indicators(self):
        self.ta.calculate_indicators()
        df = self.ta.df
        self.assertIn('MACD', df.columns)
        self.assertIn('RSI', df.columns)
        self.assertIn('Bullish_Engulfing', df.columns)

    def test_generate_signals(self):
        self.ta.calculate_indicators()
        self.ta.generate_signals()
        df = self.ta.df
        self.assertIn('Buy_Signal', df.columns)
        self.assertTrue(df['Buy_Signal'].dtype in [bool, object])

    def test_backtest_runs(self):
        self.ta.calculate_indicators()
        self.ta.generate_signals()
        self.ta.backtest(start_date="2025-03-05", end_date="2025-03-11")
        df = self.ta.df
        self.assertIn('Portfolio_Value', df.columns)

if __name__ == '__main__':
    unittest.main()
