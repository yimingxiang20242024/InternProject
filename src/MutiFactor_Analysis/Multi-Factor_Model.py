import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Load all *_combined.csv factor files
def load_factors(folder):
    factor_list = []
    for fname in os.listdir(folder):
        if fname.endswith("_combined.csv"):
            factor_name = fname.replace("_combined.csv", "")
            path = os.path.join(folder, fname)
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            stacked = df.stack().rename(factor_name)
            factor_list.append(stacked)
    return pd.concat(factor_list, axis=1)

# Load only resivol risk factor
def load_risk_factors(folder):
    resivol = pd.read_csv(os.path.join(folder, "resivol.csv"), index_col=0, parse_dates=True).stack()
    return resivol.rename("resivol").to_frame()

# Load forward 5-day return
def load_fwdret5(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).stack().rename("ret")

# Train cross-sectional regression on INS
def train_cross_section_model(X: pd.DataFrame, y: pd.Series):
    betas = {}
    pred_ins = {}
    for date in sorted(X.index.get_level_values(0).unique()):
        x_slice = X.loc[date]
        y_slice = y.loc[date]
        valid = x_slice.dropna().index.intersection(y_slice.dropna().index)
        if len(valid) < 5:
            continue
        x_valid = x_slice.loc[valid]
        y_valid = y_slice.loc[valid]
        model = LinearRegression()
        model.fit(x_valid, y_valid)
        betas[date] = model.coef_
        pred_ins[date] = pd.Series(model.predict(x_valid), index=valid)
    pred_ins_df = pd.DataFrame(pred_ins).T.sort_index()
    return betas, pred_ins_df

# Predict OOS using average beta
def predict_oos_with_avg_beta(X: pd.DataFrame, betas: dict):
    beta_df = pd.DataFrame(betas).T.sort_index()
    avg_beta = beta_df.mean().values

    pred = []
    for date in sorted(X.index.get_level_values(0).unique()):
        x_slice = X.loc[date]
        x_valid = x_slice.dropna()
        if len(x_valid) == 0:
            continue
        pred_series = x_valid @ avg_beta
        pred_series.name = date
        pred.append(pred_series)
    return pd.DataFrame(pred).sort_index()

# Normalize weights (long and short each sum to 0.5)
def scale_prediction(pred_df):
    scaled = pd.DataFrame(index=pred_df.index, columns=pred_df.columns)
    for date in pred_df.index:
        x = pred_df.loc[date].dropna()
        pos = x[x > 0]
        neg = x[x < 0]
        pos_scaled = pos / pos.sum() * 0.5 if pos.sum() != 0 else pos
        neg_scaled = neg / neg.abs().sum() * 0.5 if neg.abs().sum() != 0 else neg
        scaled.loc[date, pos.index] = pos_scaled
        scaled.loc[date, neg.index] = neg_scaled
    return scaled

# Evaluate IC in-sample
def calculate_ins_ic(pred_df, y_df):
    ic_list = []
    for date in pred_df.index:
        if date not in y_df.index:
            continue
        pred = pred_df.loc[date].dropna()
        true = y_df.loc[date].dropna()
        valid = pred.index.intersection(true.index)
        if len(valid) < 5:
            continue
        ic, _ = spearmanr(pred.loc[valid], true.loc[valid])
        ic_list.append(ic)
    return np.mean(ic_list), np.std(ic_list)

# Correlation between OOS prediction and resivol
def calculate_oos_risk_corr(pred_df, risk_df):
    result = {col: [] for col in risk_df.columns}
    for date in pred_df.index:
        if date not in risk_df.index:
            continue
        pred_row = pred_df.loc[date].dropna()
        for col in risk_df.columns:
            try:
                risk_row = risk_df[col].loc[date].dropna()
                valid = pred_row.index.intersection(risk_row.index)
                if len(valid) < 5:
                    continue
                corr = pred_row.loc[valid].corr(risk_row.loc[valid])
                result[col].append(abs(corr))
            except Exception:
                continue
    return {k: np.mean(v) for k, v in result.items() if len(v) > 0}

# Main process
def main():
    ins_factor_dir = "/Users/a12205/Desktop/美国实习/INS"
    oos_factor_dir = "/Users/a12205/Desktop/美国实习/OOS"
    ins_risk_dir = "/Users/a12205/Desktop/美国实习/INS"
    oos_risk_dir = "/Users/a12205/Desktop/美国实习/OOS"
    fwdret5_path = os.path.join(ins_risk_dir, "returns.csv")
    output_path = "/Users/a12205/Desktop/美国实习/Model_Prediction/oos_prediction.csv"

    # Load data
    X_ins = load_factors(ins_factor_dir).shift(1).dropna()
    X_oos = load_factors(oos_factor_dir).shift(1).dropna()
    risk_oos = load_risk_factors(oos_risk_dir).shift(1).dropna()
    y_ins = load_fwdret5(fwdret5_path).dropna()

    # Train model and predict
    betas, ins_pred = train_cross_section_model(X_ins, y_ins)
    raw_pred = predict_oos_with_avg_beta(X_oos, betas)
    pred_scaled = scale_prediction(raw_pred)

    # Save prediction
    pred_scaled.to_csv(output_path, float_format="%.6f")

    # Evaluation
    print("INS Prediction vs Forward Return Correlation (IC):")
    y_ins_df = y_ins.unstack()
    ins_ic_mean, ins_ic_std = calculate_ins_ic(ins_pred, y_ins_df)
    print(f"INS IC Mean: {ins_ic_mean:.4f}, IC Std: {ins_ic_std:.4f}, IR: {ins_ic_mean / ins_ic_std:.4f}")

    print("OOS Prediction vs Risk Factor Correlation (Abs):")
    oos_risk_corr = calculate_oos_risk_corr(pred_scaled, risk_oos)
    for k, v in oos_risk_corr.items():
        print(f"{k}: {v:.4f}")

    # Save evaluation
    report_data = {
        "Metric": ["IC Mean", "IC Std", "IC IR"] + [f"OOS Corr - {k}" for k in oos_risk_corr],
        "Value": [ins_ic_mean, ins_ic_std, ins_ic_mean / ins_ic_std] + [v for v in oos_risk_corr.values()]
    }
    report_df = pd.DataFrame(report_data)
    report_path = "/Users/a12205/Desktop/美国实习/Model_Prediction/evaluation_report.csv"
    report_df.to_csv(report_path, index=False, float_format="%.4f")

if __name__ == "__main__":
    main()
