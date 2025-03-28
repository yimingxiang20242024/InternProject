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
```mermaid
classDiagram
    class TechnicalAnalysis {
        - file_path: str
        - df: pd.DataFrame
        - model: Any

        + load_data(): pd.DataFrame
        + calculate_indicators(short_window, long_window, ema_fast, ema_slow, macd_signal): void
        + generate_signals(short_window, long_window): void
        + train_tree_model(model_type, in_start, in_end): void
        + predict_tree_signal(out_start, out_end): void
        + backtest(initial_cash, start_date, end_date): void
        + backtest_model_predictions(out_start, out_end): void
        + plot(kind, start_date, end_date): void
    }
```

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

---

## Flowchart:

Start
 |
 v
[Load CSV Data]
 |
 v
[Calculate Indicators: SMA, EMA, MACD, RSI, Patterns]
 |
 v
[Generate Traditional Buy/Sell Signals]
 |
 v
[USE_TREE?] --> No --> [Backtest with Traditional Signals] --> End
 |
Yes
 |
 v
[Split In-sample / Out-sample Data]
 |
 v
[Train Tree Model (XGBoost / LightGBM)]
 |
 v
[Predict ML Buy/Sell Signals on Out-sample]
 |
 v
[Combine Traditional + ML Signals]
 |
 v
[Backtest on Out-sample Data]
 |
 v
End


## End
