# BALDER - Trade Advisor

A sophisticated trading algorithm application for analyzing futures markets with real-time signals and backtesting capabilities.

## Features

- **Live Trade Signals**: Real-time buy/sell signals for futures contracts (ES, NQ, YM, CL, GC)
- **Position Tracking**: Track open positions with real-time PnL calculation
- **Backtesting**: Test the algorithm on historical data
- **Multi-Timeframe Analysis**: Analyzes multiple timeframes for confirmation
- **Custom Indicators**: EMA, RSI, ATR, MACD, ADX, Bollinger Bands, and more

## Trading Strategy

- **1m Interval**: EMA pullback strategy (catches bounces off moving averages)
- **2m Interval**: Breakout strategy (catches momentum through significant levels)

## ⚠️ Important Notice

This app uses Yahoo Finance data via yfinance, which is **delayed by 15-20 minutes**.

**Safe for:**
- Backtesting
- Strategy development
- Paper trading
- Educational purposes

**NOT suitable for:**
- Live trading decisions
- Real-time scalping
- Actual trade execution

For live trading, use real-time data from Interactive Brokers, TradingView, or your broker's platform.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run trend_app.py
```

## Access

Default passcode: `000`

## Technology Stack

- **Streamlit**: Web app framework
- **yfinance**: Market data
- **ta**: Technical analysis library
- **Plotly**: Interactive charts
- **Pandas**: Data manipulation

## License

Private use only.

