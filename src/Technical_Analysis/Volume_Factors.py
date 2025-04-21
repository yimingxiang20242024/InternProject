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

# Set the date range and trading calendar
start_date = "2023-12-31"
end_date = "2024-12-31"
trading_days = pd.bdate_range(start=start_date, end=end_date)

# List of volume-related technical indicators
volume_factors = ['OBV', 'Volume_MA', 'VR', 'MFI']
volume_storage = {factor: defaultdict(dict) for factor in volume_factors}

# Function to compute volume indicators for a single ticker
def compute_volume_factors(ticker):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df.empty or len(df) < 20:
        return {}

    # Flatten MultiIndex columns (in case of data inconsistency)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # OBV: On-Balance Volume
    df['OBV'] = ta.obv(df['Close'], df['Volume']).astype(float)

    # Volume_MA: 20-day simple moving average of volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean().astype(float)

    # VR: Volume Ratio = Up volume / Down volume over past 14 days
    df['UpVol'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 0)
    df['DownVol'] = np.where(df['Close'] < df['Close'].shift(1), df['Volume'], 0)
    df['VR'] = (df['UpVol'].rolling(14).sum() / 
                df['DownVol'].rolling(14).sum().replace(0, np.nan)).astype(float)

    # MFI: Money Flow Index (14-day)
    mfi_series = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    if mfi_series is not None:
        df['MFI'] = mfi_series.astype(float)

    # Restrict to business days intersection
    df = df.loc[trading_days.intersection(df.index)]

    # Store factor values by date
    results = {}
    for date, row in df.iterrows():
        results[date] = {
            'OBV': row.get('OBV'),
            'Volume_MA': row.get('Volume_MA'),
            'VR': row.get('VR'),
            'MFI': row.get('MFI')
        }
    return results

# Loop over each ticker
for i, ticker in enumerate(tickers):
    print(f"Processing {i+1}/{len(tickers)}: {ticker}")
    daily_results = compute_volume_factors(ticker)
    for date, values in daily_results.items():
        for factor in volume_factors:
            volume_storage[factor][date][ticker] = values.get(factor)

# Convert stored data into DataFrames and save to CSV
df_volume = {}
for factor in volume_factors:
    df = pd.DataFrame.from_dict(volume_storage[factor], orient='index').sort_index()
    df = df.reindex(trading_days).astype(float)
    df_volume[factor] = df
    df.to_csv(f"{factor}_VolumeFactor.csv")
    print(f"✅ Saved: {factor}_VolumeFactor.csv")

print("🎉 All volume indicators have been calculated and saved.")
