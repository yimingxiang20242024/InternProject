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

---

## Flowchart:
flowchart TD
    A[Start] --> B[Load CSV Data]
    B --> C[Calculate Technical Indicators]
    C --> D[Generate Buy/Sell Signals]
    D --> E{USE_TREE?}

    E -- No --> F[Backtest with Traditional Signals]
    F --> Z[End]

    E -- Yes --> G[Split In-sample / Out-sample Data]
    G --> H[Train Tree Model (XGBoost / LightGBM)]
    H --> I[Predict ML Buy/Sell Signals]
    I --> J[Combine ML & Traditional Signals]
    J --> K[Backtest on Out-sample Data]
    K --> Z

## End
