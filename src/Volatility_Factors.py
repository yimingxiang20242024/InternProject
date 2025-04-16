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

# Set date range and trading calendar
start_date = "2023-12-31"
end_date = "2024-12-31"
trading_days = pd.bdate_range(start=start_date, end=end_date)

# Define volatility factors
volatility_factors = ['ATR', 'BOLL', 'STD20']
volatility_storage = {factor: defaultdict(dict) for factor in volatility_factors}

# Function to compute volatility indicators for one ticker
def compute_volatility_factors(ticker):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.empty or len(df) < 20:
        return {}

    # Ensure consistent column names if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ATR: Average True Range (14-day)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).astype(float)

    # Bollinger Bands (middle band + upper-lower width)
    boll = ta.bbands(df['Close'], length=20)
    if boll is not None and 'BBU_20_2.0' in boll and 'BBL_20_2.0' in boll:
        df['BOLL'] = (boll['BBU_20_2.0'] - boll['BBL_20_2.0']).astype(float)
    else:
        df['BOLL'] = np.nan

    # 20-day price standard deviation
    df['STD20'] = df['Close'].rolling(20).std().astype(float)

    df = df.loc[trading_days.intersection(df.index)]

    results = {}
    for date, row in df.iterrows():
        results[date] = {
            'ATR': row.get('ATR'),
            'BOLL': row.get('BOLL'),
            'STD20': row.get('STD20')
        }
    return results

# Loop over tickers and calculate volatility factors
for i, ticker in enumerate(tickers):
    print(f"Processing {i+1}/{len(tickers)}: {ticker}")
    daily_results = compute_volatility_factors(ticker)
    for date, values in daily_results.items():
        for factor in volatility_factors:
            volatility_storage[factor][date][ticker] = values.get(factor)

# Convert to DataFrames and save
df_volatility = {}
for factor in volatility_factors:
    df = pd.DataFrame.from_dict(volatility_storage[factor], orient='index').sort_index()
    df = df.reindex(trading_days).astype(float)
    df_volatility[factor] = df
    df.to_csv(f"{factor}_VolatilityFactor.csv")
    print(f"✅ Saved: {factor}_VolatilityFactor.csv")

print("🎉 All volatility indicators have been calculated and saved.")
