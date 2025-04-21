import os
import pandas as pd

# Define folder paths
fundamental_path = "Fundamental_Factors"
technical_path = "Technical_Factors"

# Get list of all factor CSVs
fundamental_files = [f for f in os.listdir(fundamental_path) if f.endswith('.csv') and f != 'tscode.csv']
technical_files = [f for f in os.listdir(technical_path) if f.endswith('.csv') and f != 'tscode.csv']

# Helper function to read and reshape each factor file
def load_and_reshape(file_path, factor_name):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.columns = pd.MultiIndex.from_product([df.columns, [factor_name]])
    return df

# Initialize list to collect all reshaped DataFrames
all_factors = []

# Process fundamental factors
for file in fundamental_files:
    factor_name = file.replace(".csv", "")
    file_path = os.path.join(fundamental_path, file)
    reshaped = load_and_reshape(file_path, factor_name)
    all_factors.append(reshaped)
    print(f"✅ Merged: {factor_name}")

# Process technical factors
for file in technical_files:
    factor_name = file.replace(".csv", "")
    file_path = os.path.join(technical_path, file)
    reshaped = load_and_reshape(file_path, factor_name)
    all_factors.append(reshaped)
    print(f"✅ Merged: {factor_name}")

# Merge all DataFrames on date index
merged_factors = pd.concat(all_factors, axis=1)
merged_factors.sort_index(axis=1, inplace=True)

# Set paths
fundamental_path = "Fundamental_Factors"
technical_path = "Technical_Factors"

# Get file lists (excluding tscode.csv)
fundamental_files = [f for f in os.listdir(fundamental_path) if f.endswith(".csv") and f != "tscode.csv"]
technical_files = [f for f in os.listdir(technical_path) if f.endswith(".csv") and f != "tscode.csv"]

# Create a list of all factor file paths
all_files = [os.path.join(fundamental_path, f) for f in fundamental_files] + \
            [os.path.join(technical_path, f) for f in technical_files]

# Initialize common stock set
common_stocks = None

# Read each factor's DataFrame and find the intersection of column names
for file in all_files:
    df = pd.read_csv(file, index_col=0)
    non_null_columns = df.dropna(axis=1, how='all').columns
    if common_stocks is None:
        common_stocks = set(non_null_columns)
    else:
        common_stocks = common_stocks.intersection(non_null_columns)

# Output intersection results
print(f"✅ Number of common stocks across all factors: {len(common_stocks)}")
print("📋 Common stock tickers:")
print(sorted(list(common_stocks)))

# Ensure common_stocks is a set
common_stocks = set(common_stocks)

# Keep only columns where the first level (ticker) is in common_stocks
filtered_merged_factors = merged_factors.loc[:, merged_factors.columns.get_level_values(0).isin(common_stocks)]

print(f"✅ Filtered merged_factors shape: {filtered_merged_factors.shape}")
print(f"✅ Columns retained (factors × {len(common_stocks)} common stocks)")

# Filter to keep only data from April 1st, 2024 onward
filtered_merged_factors = filtered_merged_factors.loc[filtered_merged_factors.index >= "2024-04-01"]

print(f"✅ After filtering from April 1st: {filtered_merged_factors.shape}")

missing_dates = pd.DatetimeIndex([
    '2024-05-27',  # Memorial Day
    '2024-06-19',  # Juneteenth
    '2024-07-04',  # Independence Day
    '2024-09-02',  # Labor Day
    '2024-11-28',  # Thanksgiving Day
    '2024-12-25',  # Christmas Day
    '2024-12-31',  # Likely an early close, sometimes excluded
])

# Remove missing return dates from the factor dataframe
filtered_merged_factors = filtered_merged_factors.drop(index=missing_dates)

# Make sure index is sorted and datetime type
filtered_merged_factors = filtered_merged_factors.sort_index()
filtered_merged_factors.index = pd.to_datetime(filtered_merged_factors.index)

# Split into INS (first 131 trading days) and OOS (next 66 trading days)
INS = filtered_merged_factors.iloc[:131]
OOS = filtered_merged_factors.iloc[131:131+59]

print(f"✅ INS shape: {INS.shape}")
print(f"✅ OOS shape: {OOS.shape}")

import os

# Output directories
ins_path = "/Users/a12205/Desktop/美国实习/INS"
oos_path = "/Users/a12205/Desktop/美国实习/OOS"

# Make sure the directories exist
os.makedirs(ins_path, exist_ok=True)
os.makedirs(oos_path, exist_ok=True)

# Get all factor names from level=1
factor_names = filtered_merged_factors.columns.get_level_values(1).unique()

# Loop through each factor and save to separate CSVs
for factor in factor_names:
    # Extract data: rows = date, columns = stock tickers for this factor
    ins_df = INS.xs(factor, axis=1, level=1)
    oos_df = OOS.xs(factor, axis=1, level=1)

    # Save
    filename = f"{factor}.csv"
    ins_df.to_csv(os.path.join(ins_path, filename))
    oos_df.to_csv(os.path.join(oos_path, filename))

    print(f"✅ Saved {filename}")

print("🎉 All factor files (INS and OOS) exported successfully.")

import os
import yfinance as yf
import numpy as np
import pandas as pd

# Output paths
ins_path = "/Users/a12205/Desktop/美国实习/INS"
oos_path = "/Users/a12205/Desktop/美国实习/OOS"
os.makedirs(ins_path, exist_ok=True)
os.makedirs(oos_path, exist_ok=True)

# Date range
start_date = "2024-01-01"
end_date = "2024-12-31"

# Function to download adjusted close prices robustly
def download_prices(tickers, start, end, batch_size=200):
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading {i + 1} to {i + len(batch)}...")
        df = yf.download(batch, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)

        # Handle both multi-ticker and single-ticker format
        if isinstance(df.columns, pd.MultiIndex):  # Multiple tickers
            for ticker in df.columns.levels[0]:
                if ('Close' in df[ticker]):
                    all_data[ticker] = df[ticker]['Close']
        else:  # Single ticker
            if 'Close' in df:
                all_data[batch[0]] = df['Close']

    return pd.DataFrame(all_data)

# Step 1: Download prices
tickers = list(common_stocks)
price_df = download_prices(tickers, start_date, end_date) 

# Step 2: Calculate log returns
return_df = np.log(price_df / price_df.shift(1))
return_df = return_df.loc["2024-04-01":]  # 保留 4 月 1 日之后的行

# Step 3: Filter from April 1, 2024
return_df = return_df[return_df.index >= "2024-04-01"]

# Step 4: Split into INS and OOS
ins_df = return_df.iloc[:131]
oos_df = return_df.iloc[131:131+59]

# Step 5: Save returns
ins_df.to_csv(os.path.join(ins_path, "returns.csv"))
oos_df.to_csv(os.path.join(oos_path, "returns.csv"))

print("✅ Log returns saved successfully:")
print(f"📁 INS: {ins_path}/returns.csv ({ins_df.shape[0]} rows)")
print(f"📁 OOS: {oos_path}/returns.csv ({oos_df.shape[0]} rows)")

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# === Step 1: Your existing stock log returns ===
# Assume return_df is already a DataFrame with dates as index and stock tickers as columns
# return_df = ...

# Set the date range
start_date = return_df.index.min().strftime("%Y-%m-%d")
end_date = return_df.index.max().strftime("%Y-%m-%d")

# === Step 2: Download market returns (S&P 500) ===
market_data = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True, progress=False)
market_price = market_data['Close']
market_return = np.log(market_price / market_price.shift(1)).dropna()
market_df = pd.DataFrame(market_return)
market_df.columns = ["Market"]

# === Step 3: Align market and stock returns ===
aligned_returns = return_df.join(market_df, how="inner")

# === Step 4: Run regression for each stock to calculate residual standard deviation ===
resivol_values = {}

for ticker in return_df.columns:
    if ticker not in aligned_returns.columns:
        continue
    try:
        df = aligned_returns[[ticker, "Market"]].dropna()
        if len(df) < 20:
            continue
        X = df[["Market"]].values
        y = df[ticker].values
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        resivol_values[ticker] = np.std(residuals)
    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")
        continue

# === Step 5: Create resivol DataFrame and expand it to match the date index ===
resivol_df = pd.DataFrame(index=return_df.index, columns=return_df.columns)

for ticker, sigma in resivol_values.items():
    resivol_df[ticker] = sigma  # Constant residual standard deviation per day

# === Step 6: Split resivol_df into INS and OOS sets according to rule ===
ins_resivol = resivol_df.iloc[:131]
oos_resivol = resivol_df.iloc[131:131+59]

# Output file paths
ins_path = "/Users/a12205/Desktop/美国实习/INS/resivol.csv"
oos_path = "/Users/a12205/Desktop/美国实习/OOS/resivol.csv"

# Save to the respective paths
ins_resivol.to_csv(ins_path)
oos_resivol.to_csv(oos_path)

print("✅ Residual Volatility saved successfully:")
print(f"📁 INS: {ins_path} ({ins_resivol.shape[0]} rows)")
print(f"📁 OOS: {oos_path} ({oos_resivol.shape[0]} rows)")
