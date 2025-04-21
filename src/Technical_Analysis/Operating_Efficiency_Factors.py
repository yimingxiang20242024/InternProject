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
from collections import defaultdict
from datetime import datetime

# Define the date range and business calendar
start_date = '2023-12-31'
end_date = '2024-12-31'
trading_days = pd.bdate_range(start=start_date, end=end_date)

# Define the factor names for operational efficiency (excluding OCFA)
factor_names = ['INVT', 'INVTD', 'RAT', 'RATD', 'AT', 'ATD']
temp_storage = {factor: defaultdict(dict) for factor in factor_names}

# Function to compute operational efficiency factors for one stock
def calculate_operation_efficiency_factors(ticker):
    try:
        stock = yf.Ticker(ticker)
        income = stock.quarterly_financials.sort_index()
        balance = stock.quarterly_balance_sheet.sort_index()
        quarters = income.columns.intersection(balance.columns).sort_values()
    except:
        return {}

    factors = {}
    for i in range(len(quarters)):
        try:
            date = quarters[i]
            revenue = income.at['Total Revenue', date]
            inventory = balance.at['Inventory', date]
            receivables = balance.at['Accounts Receivable', date]
            total_assets = balance.at['Total Assets', date]

            # INVT: Inventory Turnover = Revenue / Inventory
            INVT = revenue / inventory if revenue and inventory else None
            # RAT: Receivables Turnover = Revenue / Receivables
            RAT = revenue / receivables if revenue and receivables else None
            # AT: Asset Turnover = Revenue / Total Assets
            AT = revenue / total_assets if revenue and total_assets else None

            if i >= 1:
                # Previous quarter values for turnover change
                prev_date = quarters[i - 1]
                prev_inventory = balance.at['Inventory', prev_date]
                prev_receivables = balance.at['Accounts Receivable', prev_date]
                prev_assets = balance.at['Total Assets', prev_date]
                prev_revenue = income.at['Total Revenue', prev_date]

                INVT_prev = prev_revenue / prev_inventory if prev_revenue and prev_inventory else None
                RAT_prev = prev_revenue / prev_receivables if prev_revenue and prev_receivables else None
                AT_prev = prev_revenue / prev_assets if prev_revenue and prev_assets else None

                # Calculate differences (QoQ change)
                INVTD = INVT - INVT_prev if INVT is not None and INVT_prev is not None else None
                RATD = RAT - RAT_prev if RAT is not None and RAT_prev is not None else None
                ATD = AT - AT_prev if AT is not None and AT_prev is not None else None
            else:
                INVTD = RATD = ATD = None

            # Save factor values for this quarter
            factors[pd.to_datetime(date)] = {
                'INVT': INVT,
                'INVTD': INVTD,
                'RAT': RAT,
                'RATD': RATD,
                'AT': AT,
                'ATD': ATD
            }

        except:
            continue
    return factors

# Loop over each ticker and compute quarterly factor values
for i, ticker in enumerate(tickers):
    print(f"Processing {i+1}/{len(tickers)}: {ticker}")
    quarter_factors = calculate_operation_efficiency_factors(ticker)
    for report_date, factor_dict in quarter_factors.items():
        quarter_days = trading_days[(trading_days >= report_date) & (trading_days < report_date + pd.DateOffset(months=3))]
        for factor_name in factor_names:
            val = factor_dict.get(factor_name, None)
            for day in quarter_days:
                temp_storage[factor_name][day][ticker] = val

# Convert the temporary storage into final DataFrames and save them
for fname in factor_names:
    df = pd.DataFrame.from_dict(temp_storage[fname], orient='index').sort_index()
    df = df.reindex(trading_days)
    df.to_csv(f"{fname}_operation_factor.csv")
    print(f"✅ Saved: {fname}_operation_factor.csv")

print("🎯 All operational efficiency factors (excluding OCFA) have been calculated and saved.")
