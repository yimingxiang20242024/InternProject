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
from datetime import datetime
from collections import defaultdict

# Set date range and business calendar
start_date = '2023-12-31'
end_date = '2024-12-31'
trading_days = pd.bdate_range(start=start_date, end=end_date)

# Factor list
factor_names = ['GPM', 'NPM', 'ROE', 'ROA', 'ROIC', 'OPM', 'GPOA', 'CFOA']

# Use nested dict to temporarily store factor values
temp_storage = {factor: defaultdict(dict) for factor in factor_names}

# Function to calculate factors for a ticker
def calculate_profitability_factors(ticker):
    stock = yf.Ticker(ticker)
    try:
        income = stock.quarterly_financials
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
    except:
        return {}

    factors = {}
    for date in income.columns:
        try:
            revenue = income.at['Total Revenue', date]
            gross_profit = income.at['Gross Profit', date]
            net_income = income.at['Net Income', date]
            operating_income = income.at['Operating Income', date]
            total_assets = balance.at['Total Assets', date]
            current_liabilities = balance.at['Current Liabilities', date]
            total_equity = balance.loc['Common Stock Equity', date]
            cash_from_ops = cashflow.loc['Cash Flow From Continuing Operating Activities', date]
            invested_capital = total_assets - current_liabilities if total_assets and current_liabilities else None

            factor_dict = {
                'GPM': gross_profit / revenue if gross_profit and revenue else None,
                'NPM': net_income / revenue if net_income and revenue else None,
                'ROE': net_income / total_equity if net_income and total_equity else None,
                'ROA': net_income / total_assets if net_income and total_assets else None,
                'ROIC': net_income / invested_capital if net_income and invested_capital else None,
                'OPM': operating_income / revenue if operating_income and revenue else None,
                'GPOA': gross_profit / total_assets if gross_profit and total_assets else None,
                'CFOA': cash_from_ops / total_assets if cash_from_ops and total_assets else None
            }

            factors[pd.to_datetime(date)] = factor_dict
        except:
            continue
    return factors

# Main loop: process each ticker
for i, ticker in enumerate(tickers):
    print(f"Processing {i+1}/{len(tickers)}: {ticker}...")
    try:
        quarter_factors = calculate_profitability_factors(ticker)
    except:
        continue

    for report_date, factor_dict in quarter_factors.items():
        quarter_days = trading_days[(trading_days >= report_date) & 
                                    (trading_days < report_date + pd.DateOffset(months=3))]
        for factor_name in factor_names:
            value = factor_dict.get(factor_name, None)
            for day in quarter_days:
                temp_storage[factor_name][day][ticker] = value

# Convert to final DataFrames
factor_dfs = {}
for factor_name in factor_names:
    df = pd.DataFrame.from_dict(temp_storage[factor_name], orient='index').sort_index()
    df = df.reindex(trading_days)
    factor_dfs[factor_name] = df

# Save results
for factor_name, df in factor_dfs.items():
    df.to_csv(f"{factor_name}_factors.csv")
    print(f"✅ Saved: {factor_name}_factors.csv")

print("🎉 All profitability factors have been computed and saved.")
