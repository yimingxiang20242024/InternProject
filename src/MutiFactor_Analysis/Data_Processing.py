import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

def analyze_new_factors(folder_path):
    result_stats = {}
    factor_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for file in factor_files:
        full_path = os.path.join(folder_path, file)
        df = pd.read_csv(full_path, index_col=0, parse_dates=True)
        flat_data = df.stack().dropna()

        plt.figure(figsize=(12, 4))
        sns.histplot(flat_data, kde=True, bins=40)
        plt.title(f"{file} Factor Value Distribution")
        plt.xlabel("Factor Value")
        plt.tight_layout()
        plt.show()

        desc = flat_data.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print(f"\n📊 Descriptive Stats for {file}:\n{desc}\n")
        result_stats[file] = desc

    return result_stats

folder_path = "/Users/a12205/Desktop/美国实习/INS"
summary_dict = analyze_new_factors(folder_path)

# === Step 1: Read a single factor CSV file ===
def read_factor_file(folder_path, filename):
    path = os.path.join(folder_path, filename)
    return pd.read_csv(path, index_col=0, parse_dates=True)

# === Step 2: Neutralize one factor using resivol only ===
def neutralize_one_factor(factor_df, resivol_df):
    neutralized = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
    for date in factor_df.index:
        y = factor_df.loc[date]
        x = resivol_df.loc[date]

        valid_idx = y.dropna().index.intersection(x.dropna().index)
        if len(valid_idx) < 5:
            continue

        y_valid = y.loc[valid_idx]
        X_valid = x.loc[valid_idx].to_frame(name="resivol")

        model = LinearRegression()
        model.fit(X_valid, y_valid)
        residual = y_valid - model.predict(X_valid)
        neutralized.loc[date, valid_idx] = residual

    return neutralized

# === Step 3: Batch process all factor files ===
def process_all_factors(folder_path):
    resivol = read_factor_file(folder_path, "resivol.csv")
    all_neutral_factors = {}

    # Exclude non-factor files
    exclude_files = {"resivol.csv", "returns.csv", "fwdret5.csv", "label.csv"}
    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv") or fname in exclude_files:
            continue

        factor_df = read_factor_file(folder_path, fname)
        neutral_df = neutralize_one_factor(factor_df, resivol)
        factor_name = fname.replace(".csv", "")
        all_neutral_factors[factor_name] = neutral_df

    # Combine into wide DataFrame
    panel = []
    for name, df in all_neutral_factors.items():
        stacked = df.stack().rename(name)
        panel.append(stacked)

    final_df = pd.concat(panel, axis=1).sort_index()
    return final_df

# Process in-sample 
multi_factor_ins = process_all_factors("/Users/a12205/Desktop/美国实习/INS")

# Process out-of-sample 
multi_factor_oos = process_all_factors("/Users/a12205/Desktop/美国实习/OOS")

def plot_multi_factor_distribution(multi_factor_df):
    factors = multi_factor_df.columns

    for factor in factors:
        series = multi_factor_df[factor].dropna()
        plt.figure(figsize=(10, 4))
        sns.histplot(series, bins=40, kde=True, color='crimson')
        plt.title(f"Distribution of {factor}")
        plt.xlabel("Factor Value")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

plot_multi_factor_distribution(multi_factor_ins)

# Winsorization to remove extreme values
def winsorize_series(series, lower=0.025, upper=0.975):
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(lower=q_low, upper=q_high)

# Z-score standardization
def zscore_series(series):
    return (series - series.mean()) / series.std() if series.std() != 0 else series - series.mean()

# Batch processing of multi-factor data
def preprocess_multi_factor(multi_factor_df, winsorize=True, zscore=True):
    processed_df = []

    for date, df_slice in multi_factor_df.groupby(level=0):
        df_one_day = df_slice.droplevel(0)
        for col in df_one_day.columns:
            series = df_one_day[col].dropna()
            if winsorize:
                series = winsorize_series(series)
            if zscore:
                series = zscore_series(series)
            df_one_day.loc[series.index, col] = series
        df_one_day['date'] = date
        processed_df.append(df_one_day)

    result = pd.concat(processed_df)
    result.set_index('date', append=True, inplace=True)
    result = result.reorder_levels(['date', result.index.names[0]])
    return result.sort_index()

# Process in-sample 
multi_factor_clean_ins = preprocess_multi_factor(multi_factor_ins)

# Process out-of-sample 
multi_factor_clean_oos = preprocess_multi_factor(multi_factor_oos)

plot_multi_factor_distribution(multi_factor_clean_ins)

def export_clean_factors(clean_df, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for factor in clean_df.columns:
        df_unstacked = clean_df[factor].unstack() 
        output_path = os.path.join(folder_path, f"{factor}_clean.csv")
        df_unstacked.to_csv(output_path)

export_clean_factors(multi_factor_clean_ins, "/Users/a12205/Desktop/美国实习/INS")
export_clean_factors(multi_factor_clean_oos, "/Users/a12205/Desktop/美国实习/OOS")
