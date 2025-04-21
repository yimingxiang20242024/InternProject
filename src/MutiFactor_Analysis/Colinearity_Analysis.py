import shutil
import pandas as pd
import numpy as np
import os
from itertools import combinations
from sklearn.linear_model import LinearRegression

# Set folder paths
ins_folder = "/Users/a12205/Desktop/美国实习/INS"
oos_folder = "/Users/a12205/Desktop/美国实习/OOS"

# Selected factor filenames (without _analysis suffix)
selected_files = [
    "ROIC_factors_clean.csv",
    "GPOA_factors_clean.csv",
    "OP_QoQ_GrowthFactors_clean.csv",
    "Revenue_QoQ_GrowthFactors_clean.csv",
    "ROA_QoQ_GrowthFactors_clean.csv",
    "APR_accrual_factor_clean.csv",
    "CR_safety_factors_clean.csv",
    "CUR_safety_factors_clean.csv",
    "QR_safety_factors_clean.csv",
    "INVT_operation_factor_clean.csv",
    "RATD_operation_factor_clean.csv",
    "ADX_TrendFactor_clean.csv"
]

# Function: Rename and save each file as _final.csv
def rename_and_copy(src_folder, file_list):
    for fname in file_list:
        old_path = os.path.join(src_folder, fname)
        new_name = fname.replace("_clean.csv", "_final.csv")
        new_path = os.path.join(src_folder, new_name)
        if os.path.exists(old_path):
            shutil.copyfile(old_path, new_path)
        else:
            print(f"⚠️ File not found: {old_path}")

# Process INS and OOS folders separately
rename_and_copy(ins_folder, selected_files)
rename_and_copy(oos_folder, selected_files)

# Compute time series of factor return betas
def compute_beta_series(factor_df: dict, return_df: pd.DataFrame):
    beta_series = {}

    for factor_name, df in factor_df.items():
        factor_shifted = df.shift(1)
        betas = []
        for date in df.index:
            x = factor_shifted.loc[date]
            y = return_df.loc[date]
            valid = x.notna() & y.notna()
            if valid.sum() < 5:
                betas.append(np.nan)
                continue
            model = LinearRegression()
            model.fit(x[valid].values.reshape(-1, 1), y[valid])
            betas.append(model.coef_[0])
        beta_series[factor_name] = pd.Series(betas, index=df.index)
    return pd.DataFrame(beta_series)

# Calculate average cross-sectional correlation of factor values
def compute_factor_corr_mean(factor_df: dict):
    factor_names = list(factor_df.keys())
    dates = factor_df[factor_names[0]].index
    corr_sum = pd.DataFrame(0, index=factor_names, columns=factor_names, dtype=float)
    count = 0

    for date in dates:
        snapshot = pd.DataFrame({k: df.loc[date] for k, df in factor_df.items()})
        corr = snapshot.corr()
        if corr.isnull().values.any(): continue
        corr_sum += corr
        count += 1

    return corr_sum / count if count > 0 else corr_sum

# Compute time series of cross-sectional correlations for each factor pair
def compute_cum_corr_series(factor_df: dict):
    factor_names = list(factor_df.keys())
    dates = factor_df[factor_names[0]].index
    pairs = list(combinations(factor_names, 2))
    cum_corr = {f"{i}|{j}": [] for i, j in pairs}

    for date in dates:
        snapshot = pd.DataFrame({k: df.loc[date] for k, df in factor_df.items()})
        corr = snapshot.corr()
        for i, j in pairs:
            value = corr.loc[i, j] if i in corr.columns and j in corr.columns else np.nan
            cum_corr[f"{i}|{j}"].append(value)

    return pd.DataFrame(cum_corr, index=dates)

# Generate Excel report for collinearity analysis
def run_collinearity_analysis(folder="INS", return_file="returns.csv", output_excel="factor_collinearity.xlsx"):
    factor_df = {}
    for fname in os.listdir(folder):
        if fname.endswith("_final.csv"):
            path = os.path.join(folder, fname)
            factor_name = fname.replace("_final.csv", "")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            factor_df[factor_name] = df

    return_path = os.path.join(folder, return_file)
    ret_df = pd.read_csv(return_path, index_col=0, parse_dates=True)

    for k in factor_df:
        factor_df[k], ret_df = factor_df[k].align(ret_df, join="inner", axis=0)
        
    beta_df = compute_beta_series(factor_df, ret_df)
    beta_corr = beta_df.corr()

    factor_corr_mean = compute_factor_corr_mean(factor_df)
    cum_corr_series = compute_cum_corr_series(factor_df)

    with pd.ExcelWriter(output_excel) as writer:
        beta_corr.to_excel(writer, sheet_name="beta_corr")
        factor_corr_mean.to_excel(writer, sheet_name="factor_corr_mean")
        cum_corr_series.to_excel(writer, sheet_name="cum_corr")

run_collinearity_analysis(folder="/Users/a12205/Desktop/美国实习/INS", return_file="returns.csv", 
                          output_excel="/Users/a12205/Desktop/美国实习/Colinearity_Analysis/factor_collinearity.xlsx")
