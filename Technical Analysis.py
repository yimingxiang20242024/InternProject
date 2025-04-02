import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

INITIAL_CASH = 10000
WORK_DAYS_PER_YEAR = 252
SHORT_WINDOW = 7
LONG_WINDOW = 30
EMA_FAST_SPAN = 12  
EMA_SLOW_SPAN = 26  
MACD_SIGNAL_SPAN = 9  
RSI_OVERSOLD_THRESHOLD = 50
RSI_OVERBOUGHT_THRESHOLD = 50
BUY_SIGNAL_THRESHOLD = 1
SELL_SIGNAL_THRESHOLD = 1
USE_TREE = False
MODEL_TYPE = 'xgboost'
IN_SAMPLE_START = '2016-03-13'
IN_SAMPLE_END = '2023-03-13'
OUT_SAMPLE_START = '2023-03-14'
OUT_SAMPLE_END = '2025-03-13'

class TechnicalAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        def convert_volume(value):
            if pd.isna(value):
                return 0
            if isinstance(value, str):
                if 'M' in value:
                    return float(value.replace('M', '')) * 1e6
                elif 'B' in value:
                    return float(value.replace('B', '')) * 1e9
            return float(value)

        df['Volume'] = df['Vol.'].apply(convert_volume)
        return df

    def calculate_indicators(self, short_window=SHORT_WINDOW, long_window=LONG_WINDOW, 
                             ema_fast=EMA_FAST_SPAN, ema_slow=EMA_SLOW_SPAN, macd_signal=MACD_SIGNAL_SPAN):
        df = self.df
        df[f'SMA_{short_window}'] = df['Price'].rolling(window=short_window).mean()
        df[f'SMA_{long_window}'] = df['Price'].rolling(window=long_window).mean()
        df[f'EMA_{ema_fast}'] = df['Price'].ewm(span=ema_fast, adjust=False).mean()
        df[f'EMA_{ema_slow}'] = df['Price'].ewm(span=ema_slow, adjust=False).mean()
        df['MACD'] = df[f'EMA_{ema_fast}'] - df[f'EMA_{ema_slow}']
        df['Signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()

        delta = df['Price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Bullish_Engulfing'] = (
            (df['Price'].shift(1) > df['Open'].shift(1)) &
            (df['Price'] > df['Open']) &
            (df['Open'] < df['Price'].shift(1)) &
            (df['Price'] > df['Open'].shift(1))
        )
        df['Bearish_Engulfing'] = (
            (df['Price'].shift(1) < df['Open'].shift(1)) &
            (df['Price'] < df['Open']) &
            (df['Open'] > df['Price'].shift(1)) &
            (df['Price'] < df['Open'].shift(1))
        )

    def generate_signals(self, short_window= SHORT_WINDOW, long_window=LONG_WINDOW):
        df = self.df

        prev_macd = df['MACD'].shift(1)
        prev_macd_signal = df['MACD_Signal'].shift(1)
        macd_cross = np.where((df['MACD'] > df['MACD_Signal']) & (prev_macd <= prev_macd_signal), 1,
                              np.where((df['MACD'] < df['MACD_Signal']) & (prev_macd >= prev_macd_signal), -1, 0))
        
        prev_sma_short = df[f'SMA_{short_window}'].shift(1)
        prev_sma_long = df[f'SMA_{long_window}'].shift(1)
        sma_cross = np.where((df[f'SMA_{short_window}'] > df[f'SMA_{long_window}']) & (prev_sma_short <= prev_sma_long), 1,
                             np.where((df[f'SMA_{short_window}'] < df[f'SMA_{long_window}']) & (prev_sma_short >= prev_sma_long), -1, 0))
        
        prev_rsi = df['RSI'].shift(1)
        rsi_cross = np.where((df['RSI'] > 50) & (prev_rsi <= 50), 1,
                             np.where((df['RSI'] < 50) & (prev_rsi >= 50), -1, 0))
        
        df['Signal'] = np.where(macd_cross != 0, macd_cross,
                         np.where(sma_cross != 0, sma_cross,
                         np.where(rsi_cross != 0, rsi_cross, 0)))
        df['Signal'] = df['Signal'].replace(to_replace=0, method='ffill')
        df['Signal_Change'] = df['Signal'].diff().fillna(0)

    def backtest(self, initial_cash=INITIAL_CASH, start_date=None, end_date=None):
        df = self.df.copy()
        if start_date and end_date:
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)

        cash = initial_cash
        shares = 0
        portfolio_value = [initial_cash]
        returns = []

        buy_signal = df['Buy_Signal'].shift(1).ffill().astype(bool)
        sell_signal = df['Sell_Signal'].shift(1).ffill().astype(bool)

        if USE_TREE:
            ml_buy_signal = df['ML_Buy_Signal'].shift(1).bfill().astype(bool)
            ml_sell_signal = df['ML_Sell_Signal'].shift(1).bfill().astype(bool)
            buy_signal = buy_signal | ml_buy_signal
            sell_signal = (~buy_signal) & ml_sell_signal

        for i in range(len(df)):
            price = df.iloc[i]['Price']
            if buy_signal.iloc[i] and cash > 0:
                shares = cash / price
                cash = 0
            elif sell_signal.iloc[i] and shares > 0:
                cash = shares * price
                shares = 0
            total_value = cash + (shares * price)
            portfolio_value.append(total_value)
            if len(portfolio_value) > 1 and portfolio_value[-2] > 0:
                daily_return = (portfolio_value[-1] - portfolio_value[-2]) / portfolio_value[-2]
                returns.append(daily_return)

        df['Portfolio_Value'] = portfolio_value[:len(df)]
        total_return = (df['Portfolio_Value'].iloc[-1] - initial_cash) / initial_cash
        annualized_return = (1 + np.mean(returns)) ** WORK_DAYS_PER_YEAR - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(WORK_DAYS_PER_YEAR) if np.std(returns) != 0 else 0
        max_drawdown = np.max((np.maximum.accumulate(df['Portfolio_Value']) - df['Portfolio_Value']) / np.maximum.accumulate(df['Portfolio_Value']))
        volatility = np.std(returns) * np.sqrt(WORK_DAYS_PER_YEAR)

        print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Volatility: {volatility:.2%}")

    def backtest_model_predictions(self, out_start, out_end):
        if USE_TREE:
            self.predict_tree_signal(out_start, out_end)
        self.backtest(start_date=out_start, end_date=out_end)

    def plot(self, kind, start_date=OUT_SAMPLE_START, end_date=OUT_SAMPLE_END):
        df = self.df.copy()
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        plt.figure(figsize=(36, 12) if kind == 'price' else (12, 6))

        if kind == 'price':
            plt.plot(df['Date'], df['Price'], label='Price', color='black')
            plt.scatter(df['Date'][df['Buy_Signal']], df['Price'][df['Buy_Signal']], marker='^', color='green', label='Buy Signal')
            plt.scatter(df['Date'][df['Sell_Signal']], df['Price'][df['Sell_Signal']], marker='v', color='red', label='Sell Signal')
            plt.title("Stock Price with Buy/Sell Signals")

        elif kind == 'macd':
            plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
            plt.plot(df['Date'], df['Signal'], label='Signal Line', color='red')
            plt.axhline(0, color='gray', linestyle='dashed')
            plt.title("MACD Indicator")

        elif kind == 'rsi':
            plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
            plt.axhline(70, color='red', linestyle='dashed', label='Overbought')
            plt.axhline(30, color='green', linestyle='dashed', label='Oversold')
            plt.title("RSI Indicator")

        elif kind == 'portfolio':
            plt.plot(df['Date'], df['Portfolio_Value'], label='Portfolio Value', color='blue')
            plt.title("Portfolio Performance Over Time")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")

        elif kind == 'candlestick':
            for i in range(len(df)):
                date = df.iloc[i]['Date']
                open_ = df.iloc[i]['Open']
                close = df.iloc[i]['Price']
                high = df.iloc[i]['High']
                low = df.iloc[i]['Low']
                color = 'green' if close > open_ else 'red'
                plt.bar(date, abs(close - open_), bottom=min(open_, close), color=color, width=0.5)
                plt.vlines(date, low, high, color=color, linewidth=1)
            plt.scatter(df['Date'][df['Bullish_Engulfing']], df['High'][df['Bullish_Engulfing']] * 1.01, label='Bullish Engulfing', color='lime', s=100)
            plt.scatter(df['Date'][df['Bearish_Engulfing']], df['Low'][df['Bearish_Engulfing']] * 0.99, label='Bearish Engulfing', color='red', s=100)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.title("Candlestick Pattern Detection")

        plt.legend()
        plt.show()

if __name__ == "__main__":
    ta = TechnicalAnalysis("NVIDIA Stock Price History.csv")
    ta.calculate_indicators()
    ta.generate_signals()
    if USE_TREE:
        ta.train_tree_model(model_type=MODEL_TYPE, in_start=IN_SAMPLE_START, in_end=IN_SAMPLE_END)
        ta.backtest_model_predictions(out_start=OUT_SAMPLE_START, out_end=OUT_SAMPLE_END)
    else:
        ta.backtest(start_date=OUT_SAMPLE_START, end_date=OUT_SAMPLE_END)

    ta.plot('price')
    ta.plot('macd')
    ta.plot('rsi')
    ta.plot('candlestick')
    ta.plot('portfolio')
