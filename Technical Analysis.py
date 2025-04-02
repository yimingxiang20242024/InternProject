import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier

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
FEE_RATE = 0.0001  
SHORT_INTEREST = 0.04  
MODEL_TYPE = 'xgboost'
IN_SAMPLE_START = '2015-03-11'
IN_SAMPLE_END = '2023-03-11'
OUT_SAMPLE_START = '2023-03-12'
OUT_SAMPLE_END = '2025-03-11'

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
                if 'K' in value:
                    return float(value.replace('K', '')) * 1e3
                elif 'M' in value:
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
        df[f'EMA_200'] = df['Price'].ewm(span=200, adjust=False).mean()
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
        df['sma_buy'] = df[f'SMA_{short_window}'] > df[f'SMA_{long_window}']
        df['macd_buy'] = df['MACD'] > df['Signal']
        df['rsi_buy'] = df['RSI'] > RSI_OVERSOLD_THRESHOLD
        df['bullish'] = df['Bullish_Engulfing']
        df['ema_buy'] = df['Price'] > df[f'EMA_200']
        df['sma_sell'] = df[f'SMA_{short_window}'] < df[f'SMA_{long_window}']
        df['macd_sell'] = df['MACD'] < df['Signal']
        df['rsi_sell'] = df['RSI'] < RSI_OVERBOUGHT_THRESHOLD
        df['bearish'] = df['Bearish_Engulfing']
        df['ema_sell'] = df['Price'] < df[f'EMA_200']

        df['Buy_Signal'] = df['ema_buy'] | df['macd_buy'] | df['rsi_buy']
        df['Sell_Signal'] = df['ema_sell'] & df['macd_sell'] 
        df['ML_Sell_Signal'] = False

    def train_tree_model(self, model_type='xgboost', in_start=None, in_end=None):
        df = self.df.copy()
        df = df.set_index('Date')
        data = df.loc[in_start:in_end].copy()

        features = ['sma_buy', 'macd_buy', 'rsi_buy', 'bullish', 'sma_sell', 'macd_sell', 'rsi_sell', 'bearish']
        X = data[features].shift(1).infer_objects(copy=False).bfill().astype(int)
        y = (data['Price'].shift(-5) > data['Price']).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        if model_type == 'xgboost':
            self.model = XGBRegressor()
        else:
            self.model = LGBMRegressor()

        self.model.fit(X_train, y_train)
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Tree Test Accuracy: {accuracy:.2%}")

    def predict_tree_signal(self, out_start, out_end):
        df = self.df.copy()
        df = df.set_index('Date')
        data = df.loc[out_start:out_end].copy()

        features = ['sma_buy', 'macd_buy', 'rsi_buy', 'bullish', 'sma_sell', 'macd_sell', 'rsi_sell', 'bearish']
        X_out = data[features].astype(int)
        preds = self.model.predict(X_out)
        pred_index = data.index

        self.df.loc[self.df['Date'].isin(pred_index), 'ML_Buy_Signal'] = preds > 0.5
        self.df.loc[self.df['Date'].isin(pred_index), 'ML_Sell_Signal'] = preds <= 0.5

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
            ml_buy_signal = df['ML_Buy_Signal'].shift(1).infer_objects(copy=False).bfill().astype(bool)
            ml_sell_signal = df['ML_Sell_Signal'].shift(1).infer_objects(copy=False).bfill().astype(bool)
            buy_signal = ml_buy_signal & buy_signal
            sell_signal = ml_sell_signal & sell_signal

        for i in range(len(df)):
            price = df.iloc[i]['Price']
            if shares < 0:
                interest = abs(shares) * price * (SHORT_INTEREST / WORK_DAYS_PER_YEAR)
                cash -= interest

            if buy_signal.iloc[i]:
                if shares < 0:  
                    cost = abs(shares) * price * (1 + FEE_RATE)
                    cash -= cost
                    shares = 0
                elif cash > 0:
                    buy_price = price * (1 + FEE_RATE)
                    shares = cash / buy_price
                    cash = 0
            elif sell_signal.iloc[i]:
                if shares > 0:  
                    cash = shares * price * (1 - FEE_RATE)
                    shares = 0
                elif shares == 0:  
                    shares = - (cash / price)
                    cash += abs(shares) * price * (1 - FEE_RATE)

            total_value = cash + (shares * price)
            portfolio_value.append(total_value)
            if len(portfolio_value) > 1 and portfolio_value[-2] > 0:
                daily_return = (portfolio_value[-1] - portfolio_value[-2]) / portfolio_value[-2]
                returns.append(daily_return)

        df['Portfolio_Value'] = portfolio_value[:len(df)]
        total_return = (df['Portfolio_Value'].iloc[-1] - initial_cash) / initial_cash
        strategy_returns = np.array(returns)
        annualized_return = (1 + np.mean(strategy_returns)) ** WORK_DAYS_PER_YEAR - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(WORK_DAYS_PER_YEAR) if np.std(strategy_returns) != 0 else 0
        max_drawdown = np.max((np.maximum.accumulate(df['Portfolio_Value']) - df['Portfolio_Value']) / np.maximum.accumulate(df['Portfolio_Value']))
        volatility = np.std(strategy_returns) * np.sqrt(WORK_DAYS_PER_YEAR)

        # --- Buy & Hold Baseline ---
        price_start = df['Price'].iloc[0]
        price_end = df['Price'].iloc[-1]
        buy_hold_value = initial_cash * (price_end / price_start)
        buy_hold_return = (buy_hold_value - initial_cash) / initial_cash
        buy_hold_returns = df['Price'].pct_change().dropna().values
        buy_hold_annualized = (1 + np.mean(buy_hold_returns)) ** WORK_DAYS_PER_YEAR - 1
        buy_hold_sharpe = np.mean(buy_hold_returns) / np.std(buy_hold_returns) * np.sqrt(WORK_DAYS_PER_YEAR)
        buy_hold_volatility = np.std(buy_hold_returns) * np.sqrt(WORK_DAYS_PER_YEAR)
        bh_cum = (df['Price'] / price_start) * initial_cash
        buy_hold_drawdown = np.max((np.maximum.accumulate(bh_cum) - bh_cum) / np.maximum.accumulate(bh_cum))

        print("\n🔵 Strategy Results:")
        print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Volatility: {volatility:.2%}")

        print("\n⚪ Buy & Hold Baseline:")
        print(f"Final Value: ${buy_hold_value:.2f}")
        print(f"Total Return: {buy_hold_return:.2%}")
        print(f"Annualized Return: {buy_hold_annualized:.2%}")
        print(f"Sharpe Ratio: {buy_hold_sharpe:.2f}")
        print(f"Maximum Drawdown: {buy_hold_drawdown:.2%}")
        print(f"Volatility: {buy_hold_volatility:.2%}")

        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Portfolio_Value'], label='Strategy Portfolio', color='blue')
        plt.plot(df['Date'], bh_cum, label='Buy & Hold', linestyle='--', color='gray')
        plt.title("Portfolio Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()

        self.df = df

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
    #Amazon.com Stock Price History.csv
    #NVIDIA Stock Price History.csv
    #AMD Stock Price History.csv
    #Apple Stock Price History.csv
    #Delta Air Lines Stock Price History.csv
    #Duke Energy Stock Price History.csv
    #Meta Platforms Stock Price History.csv
    #Moderna Stock Price History.csv
    #Pfizer Stock Price History.csv
    #Tesla Stock Price History.csv
    ta = TechnicalAnalysis("Duke Energy Stock Price History.csv")
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
