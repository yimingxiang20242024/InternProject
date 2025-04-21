import pandas as pd
import numpy as np

# === Set how many stocks to long/short ===
top_n = 60  # You can change this to any number like 50, 200, etc.

# === Step 1: Load prediction and forward return ===
pred = pd.read_csv("/Users/a12205/Desktop/美国实习/Model_Prediction/oos_prediction.csv", index_col=0, parse_dates=True)
fwdret = pd.read_csv("/Users/a12205/Desktop/美国实习/OOS/returns.csv", index_col=0, parse_dates=True)

# Align and clean
pred, fwdret = pred.align(fwdret, join='inner', axis=0)
pred = pred.sort_index().sort_index(axis=1)
fwdret = fwdret.sort_index().sort_index(axis=1)

# === Step 2: Long-short daily return based on Top N / Bottom N ===
long_returns = []
short_returns = []
dates = pred.index

for date in dates:
    p = pred.loc[date].dropna()
    r = fwdret.loc[date].dropna()
    valid = p.index.intersection(r.index)
    if len(valid) < top_n * 2:
        long_returns.append(np.nan)
        short_returns.append(np.nan)
        continue

    p = p.loc[valid]
    r = r.loc[valid]

    top_stocks = p.sort_values(ascending=False).head(top_n).index
    bottom_stocks = p.sort_values(ascending=True).head(top_n).index

    long_ret = r.loc[top_stocks].mean()
    short_ret = r.loc[bottom_stocks].mean()

    long_returns.append(long_ret)
    short_returns.append(short_ret)

# === Step 3: Combine into DataFrame ===
ret_df = pd.DataFrame({
    "long": long_returns,
    "short": short_returns,
    "long_short": np.array(long_returns) - np.array(short_returns)
}, index=dates)

# === Step 4: Performance statistics ===
def performance_stats(return_series, freq='D'):
    ann_factor = {'D': 252, 'W': 52, 'M': 12}[freq]
    cumulative = (1 + return_series).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    sharpe = return_series.mean() / return_series.std() * np.sqrt(ann_factor) if return_series.std() > 0 else np.nan
    win_rate = (return_series > 0).sum() / return_series.count()
    return {
        "Total Return": cumulative.iloc[-1] - 1,
        "Annualized Return": (1 + return_series.mean())**ann_factor - 1,
        "Volatility": return_series.std() * np.sqrt(ann_factor),
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate
    }

# === Step 5: Print results ===
print(f"📈 Long-Short Portfolio Performance (Top {top_n} / Bottom {top_n}):")
perf = performance_stats(ret_df["long_short"].dropna())
for k, v in perf.items():
    print(f"{k}: {v:.4%}" if "Rate" in k or "Return" in k else f"{k}: {v:.4f}")

# Optional: Save results
ret_df.to_csv(f"/Users/a12205/Desktop/美国实习/Model_Prediction/top{top_n}_long_bottom{top_n}_short_returns.csv")
pd.DataFrame([perf]).T.to_csv(f"/Users/a12205/Desktop/美国实习/Model_Prediction/long_short_summary_top{top_n}.csv", header=["Value"])
