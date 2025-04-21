import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load factor and return data ===
def load_factor_and_return(factor_path, return_path):
    factor = pd.read_csv(factor_path, index_col=0, parse_dates=True)
    ret = pd.read_csv(return_path, index_col=0, parse_dates=True)
    return factor.sort_index(), ret.sort_index()

# === Calculate IC series (Pearson or Spearman) ===
def calculate_ic_series(factor_df, return_df, method='pearson'):
    return factor_df.corrwith(return_df, axis=1, method=method)

# === Calculate long-short portfolio return ===
def factor_group_return(factor_df, return_df, n_groups=5):
    long_short_returns = pd.Series(index=factor_df.index)
    for date in factor_df.index:
        factor = factor_df.loc[date]
        ret = return_df.loc[date]
        df = pd.concat([factor, ret], axis=1).dropna()
        if len(df) < n_groups:
            continue
        df.columns = ['factor', 'ret']
        df = df.sort_values('factor')
        q = len(df) // n_groups
        long_group = df.iloc[-q:]['ret'].mean()
        short_group = df.iloc[:q]['ret'].mean()
        long_short_returns.loc[date] = long_group - short_group
    return long_short_returns

# === Calculate performance metrics ===
def compute_performance_stats(return_series, freq='D'):
    ann_factor = {'D': 252, 'W': 52, 'M': 12}[freq]
    cumulative = (1 + return_series).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    win_rate = (return_series > 0).sum() / return_series.count()
    return {
        'Annual Return': (1 + return_series.mean())**ann_factor - 1,
        'Volatility': return_series.std() * np.sqrt(ann_factor),
        'Sharpe Ratio': return_series.mean() / return_series.std() * np.sqrt(ann_factor) if return_series.std() > 0 else np.nan,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'PNL Final': cumulative.iloc[-1]
    }

# === Batch analysis for all *_clean.csv factors ===
def analyze_all_factors(folder="INS", return_file="returns.csv", out_dir="IC_Reports"):
    os.makedirs(out_dir, exist_ok=True)
    return_path = os.path.join(folder, return_file)
    result_stats = {}

    # Only process files ending with 'clean.csv'
    factor_files = [f for f in os.listdir(folder) if f.endswith("clean.csv")]

    for factor_file in factor_files:
        factor_path = os.path.join(folder, factor_file)
        factor, ret = load_factor_and_return(factor_path, return_path)
        factor = factor.shift(1)
        factor, ret = factor.align(ret, join='inner', axis=0)

        ic = calculate_ic_series(factor, ret, method='pearson')
        ric = calculate_ic_series(factor, ret, method='spearman')
        ls = factor_group_return(factor, ret)
        ls_cum = (1 + ls.fillna(0)).cumprod()
        perf = compute_performance_stats(ls)

        # Plot and save performance report
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))
        axs[0].plot(ic, color='tomato')
        axs[0].set_title(f'{factor_file} IC (Pearson)')
        axs[0].grid(True)

        axs[1].bar(ric.index, ric.values, color='indianred')
        axs[1].set_title(f'{factor_file} Rank IC (Spearman)')
        axs[1].grid(True)

        axs[2].plot(ls_cum, label='Long - Short Return', color='crimson')
        axs[2].set_title(f'{factor_file} Cumulative Long-Short Return')
        axs[2].grid(True)

        axs[3].axis('off')
        metrics_text = "\n".join([f"{k}: {v:.5f}" for k, v in perf.items()])
        axs[3].text(0.02, 0.5, metrics_text, fontsize=12, family='monospace', verticalalignment='center')

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{factor_file}_analysis.png"))
        plt.close()

        # Summary stats
        stat = {
            'IC_mean': ic.mean(),
            'IC_std': ic.std(),
            'IR': ic.mean() / ic.std() if ic.std() != 0 else np.nan,
            'RankIC_mean': ric.mean(),
            'RankIC_std': ric.std(),
            'LS_mean': ls.mean(),
            'LS_std': ls.std()
        }
        result_stats[factor_file] = stat

    summary_df = pd.DataFrame(result_stats).T
    summary_df.to_csv(os.path.join(out_dir, "IC_summary.csv"), float_format="%.5f")
    return summary_df

summary = analyze_all_factors(folder="/Users/a12205/Desktop/美国实习/INS", return_file="returns.csv", 
                              out_dir="/Users/a12205/Desktop/美国实习/Single_Factor_Testing_Report")
