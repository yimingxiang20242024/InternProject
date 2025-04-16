import pandas as pd

df = pd.read_csv("tscode.csv")

df_filtered = df[df['IPO Year'] != 2025]

if 'Symbol' in df_filtered.columns:
    raw_tickers = df_filtered['Symbol']
else:
    raw_tickers = df_filtered.iloc[:, 0]

tickers = [
    str(t).replace('.', '-').strip().upper()
    for t in raw_tickers
    if pd.notnull(t) and str(t).strip() != ''
]
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from collections import defaultdict

# Date range and trading calendar
start_date = "2023-12-31"
end_date = "2024-12-31"
trading_days = pd.bdate_range(start=start_date, end=end_date)

# Momentum indicators
momentum_factors = ['RSI', 'KDJ_K', 'KDJ_D', 'WR', 'MOM', 'ROC']
momentum_storage = {factor: defaultdict(dict) for factor in momentum_factors}

# Function to calculate momentum indicators
def compute_momentum_factors(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty or len(df) < 20:
        return {}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['KDJ_K'] = stoch['STOCHk_14_3_3']
    df['KDJ_D'] = stoch['STOCHd_14_3_3']
    df['WR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    df['MOM'] = ta.mom(df['Close'], length=10)
    df['ROC'] = ta.roc(df['Close'], length=10)

    df = df.loc[trading_days.intersection(df.index)]

    results = {}
    for date, row in df.iterrows():
        results[date] = {
            'RSI': row.get('RSI'),
            'KDJ_K': row.get('KDJ_K'),
            'KDJ_D': row.get('KDJ_D'),
            'WR': row.get('WR'),
            'MOM': row.get('MOM'),
            'ROC': row.get('ROC')
        }
    return results

# Run calculation for all tickers
for i, ticker in enumerate(tickers):
    print(f"Processing {i+1}/{len(tickers)}: {ticker}")
    daily_results = compute_momentum_factors(ticker)
    for date, values in daily_results.items():
        for factor in momentum_factors:
            momentum_storage[factor][date][ticker] = values.get(factor)

# Convert to DataFrames and save
df_momentum = {}
for factor in momentum_factors:
    df = pd.DataFrame.from_dict(momentum_storage[factor], orient='index').sort_index()
    df = df.reindex(trading_days)
    df_momentum[factor] = df
    df.to_csv(f"{factor}_MomentumFactor.csv")
    print(f"✅ Saved: {factor}_MomentumFactor.csv")

print("🎉 All momentum indicators calculated and saved.")
