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

# Define safety factor names
factor_names = ['CR', 'CRD', 'CUR', 'CURD',
                'DAD', 'Debt_Asset', 'DTE', 'DTED', 'QR', 'QRD']
temp_storage = {factor: defaultdict(dict) for factor in factor_names}

# Load tickers
tickers_df = pd.read_csv("tscode.csv")
tickers_df = tickers_df[tickers_df['IPO Year'] != 2025]
tickers_raw = tickers_df['Symbol'] if 'Symbol' in tickers_df.columns else tickers_df.iloc[:, 0]
tickers = [str(t).replace('.', '-').strip().upper() for t in tickers_raw if pd.notnull(t) and str(t).strip() != '']

# Safety factor calculator
def calculate_safety_factors(ticker):
    stock = yf.Ticker(ticker)
    try:
        balance = stock.quarterly_balance_sheet.sort_index()
        cashflow = stock.quarterly_cashflow.sort_index()
    except:
        return {}

    factors = {}
    dates = balance.columns.intersection(cashflow.columns).sort_values()

    for i in range(len(dates)):
        try:
            date = dates[i]
            current_assets = balance.loc['Current Assets', date]
            current_liabilities = balance.loc['Current Liabilities', date]
            total_assets = balance.loc['Total Assets', date]
            total_debt = balance.loc['Total Debt', date]
            common_equity = balance.loc['Common Stock Equity', date]
            cash = balance.loc['Cash And Cash Equivalents', date] + \
                   balance.loc['Other Short Term Investments', date]
            # Basic ratios
            CR = cash / current_liabilities if current_liabilities else None
            CUR = current_assets / current_liabilities if current_liabilities else None
            DAD = current_liabilities / total_assets if total_assets else None
            Debt_Asset = total_debt / total_assets if total_debt and total_assets else None
            DTE = total_debt / common_equity if total_debt and common_equity else None
            DTED = (total_debt - common_equity) / common_equity if total_debt and common_equity else None
            QR = (current_assets - balance.get('Inventory', {}).get(date, 0)) / current_liabilities if current_liabilities else None
            
            # Delta ratios
            CRD = CURD = QRD = None
            if i > 0:
                last_date = dates[i - 1]
                prev_cash = balance.loc['Cash And Cash Equivalents', last_date] + \
                            balance.loc['Other Short Term Investments', last_date]
                prev_current_liabilities = balance.loc['Current Liabilities', last_date]
                prev_current_assets = balance.loc['Current Assets', last_date]
                prev_inventory = balance.get('Inventory', {}).get(last_date, 0)
                CR_prev = prev_cash / prev_current_liabilities if prev_current_liabilities else None
                CUR_prev = prev_current_assets / prev_current_liabilities if prev_current_liabilities else None
                QR_prev = (prev_current_assets - prev_inventory) / prev_current_liabilities if prev_current_liabilities else None
                CRD = CR - CR_prev if CR is not None and CR_prev is not None else None
                CURD = CUR - CUR_prev if CUR is not None and CUR_prev is not None else None
                QRD = QR - QR_prev if QR is not None and QR_prev is not None else None

            factors[pd.to_datetime(date)] = {
                'CR': CR, 'CRD': CRD,
                'CUR': CUR, 'CURD': CURD, 'DAD': DAD, 'Debt_Asset': Debt_Asset,
                'DTE': DTE, 'DTED': DTED, 'QR': QR, 'QRD': QRD
            }

        except Exception as e:
            continue

    return factors

# Main loop: process each ticker
for i, ticker in enumerate(tickers):
    print(f"Processing {i+1}/{len(tickers)}: {ticker}")
    try:
        quarterly_data = calculate_safety_factors(ticker)
    except:
        continue

    for report_date, factor_dict in quarterly_data.items():
        quarter_days = trading_days[(trading_days >= report_date) & 
                                    (trading_days < report_date + pd.DateOffset(months=3))]
        for fname in factor_names:
            val = factor_dict.get(fname, None)
            for day in quarter_days:
                temp_storage[fname][day][ticker] = val

# Convert to daily DataFrames
factor_dfs = {}
for fname in factor_names:
    df = pd.DataFrame.from_dict(temp_storage[fname], orient='index').sort_index()
    df = df.reindex(trading_days)
    factor_dfs[fname] = df

# Save results
for fname, df in factor_dfs.items():
    df.to_csv(f"{fname}_safety_factors.csv")
    print(f"✅ Saved: {fname}_safety_factors.csv")

print("🎉 All safety factors have been computed and saved.")
