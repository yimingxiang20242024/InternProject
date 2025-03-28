# Overview
This script reads NVIDIA stock price data from an Excel file and generates a time-series plot of stock prices over time.

## Requirements
Make sure you have the following Python libraries installed:
```
pip install pandas matplotlib openpyxl
```

## How to Use
Place the Excel file (NVIDIA Stock Price History.xlsx) in the same folder as the script.
Run the script using:

## Packages to work with:

- Pandas
- Numpy
- matplolib
- ...

# UML + ML Integration Report for TechnicalAnalysis

## Overview
This project implements an Object-Oriented Programming (OOP) approach to technical analysis on stock price data, enriched with machine learning for improved signal accuracy.

---

## UML Class Design (Abstracted)

```text
+-----------------------+
|  TechnicalAnalysis    |
+-----------------------+
| - file_path           |
| - df                  |
| - model               |
+-----------------------+
| +load_data()          |
| +calculate_indicators()|
| +generate_signals()   |
| +train_tree_model()   |
| +predict_tree_signal()|
| +backtest()           |
| +backtest_model_predictions()|
| +plot(kind)           |
+-----------------------+
```

---

## Class Responsibilities

### `TechnicalAnalysis`
- **load_data**: Reads CSV data, processes dates and volume
- **calculate_indicators**: Computes SMA, EMA, MACD, RSI, candlestick patterns
- **generate_signals**: Derives trading signals from indicators
- **train_tree_model**: Uses `XGBoost` or `LightGBM` to train a regressor on in-sample signal data
- **predict_tree_signal**: Applies trained model to generate out-sample ML-based signals
- **backtest**: Simulates trades, computes portfolio performance metrics
- **plot**: Visualizes price, indicators, and strategy performance

---

## ML Integration Strategy

### Input Features:
- Eight technical signals: `sma_buy`, `macd_buy`, `rsi_buy`, `bullish`, `sma_sell`, `macd_sell`, `rsi_sell`, `bearish`
- Features are lagged (t-1) to avoid lookahead bias

### Target:
- `y = (Price[t+1] > Price[t])`
- Binary classification: 1 = Up, 0 = Down

### Model Options:
- `XGBRegressor` or `LGBMRegressor`
- Controlled via `MODEL_TYPE` parameter

### Workflow:
1. In-sample data split by `IN_SAMPLE_START` ~ `IN_SAMPLE_END`
2. Train tree model on signal→future return relation
3. Predict on `OUT_SAMPLE_START` ~ `OUT_SAMPLE_END`
4. If `USE_TREE=True`, ML signal contributes to Buy/Sell condition
5. Use traditional or hybrid signals in `backtest()` to simulate strategy performance

---

## Flexibility and Extensibility
- Modular class allows easy integration of new indicators or ML models
- Constants (thresholds, windows, etc.) are parameterized for tuning
- `USE_TREE` toggle gives full control over strategy logic

---

## Future Suggestions
- Extend to multi-class prediction (Buy/Hold/Sell)
- Integrate Explainable AI (e.g. SHAP) for model interpretation
- Add cross-validation and hyperparameter tuning
- Export predictions and performance as reports or dashboards

---

## End
