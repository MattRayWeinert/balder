"""
Technical indicators calculation module
"""
import pandas as pd
import ta
from typing import Optional


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicator columns to DataFrame with safeguards for short windows.
    Returns DataFrame with all indicators added.
    """
    df = df.copy()
    
    # Ensure numeric columns exist
    for col in ("Open", "Close", "High", "Low", "Volume"):
        if col not in df.columns and col == "Close" and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        if col not in df.columns:
            df[col] = df.get("Close", pd.Series(index=df.index, dtype=float))
    
    df = df.dropna(subset=["Close", "High", "Low"])
    n = len(df)
    
    if n >= 2:
        df["EMA_fast"]  = ta.trend.EMAIndicator(df["Close"], window=min(5, n)).ema_indicator()
        df["EMA_slow"]  = ta.trend.EMAIndicator(df["Close"], window=min(20, n)).ema_indicator()
        df["EMA20"]     = ta.trend.EMAIndicator(df["Close"], window=min(20, n)).ema_indicator()
        df["EMA50"]     = ta.trend.EMAIndicator(df["Close"], window=min(50, n)).ema_indicator()
        df["EMA34"]     = ta.trend.EMAIndicator(df["Close"], window=min(34, n)).ema_indicator()
        df["EMA89"]     = ta.trend.EMAIndicator(df["Close"], window=min(89, n)).ema_indicator()
        df["EMA_200"]   = ta.trend.EMAIndicator(df["Close"], window=min(200, n)).ema_indicator()
        df["MACD_hist"] = ta.trend.MACD(df["Close"], window_slow=21, window_fast=8, window_sign=5).macd_diff()
        df["RSI"]       = ta.momentum.RSIIndicator(df["Close"], window=min(7, n)).rsi()
        df["RSI2"]      = ta.momentum.RSIIndicator(df["Close"], window=min(2, n)).rsi()
        df["RSI3"]      = ta.momentum.RSIIndicator(df["Close"], window=min(3, n)).rsi()
        df["ATR"]       = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=min(14, n)).average_true_range()
    else:
        for col in ["EMA_fast", "EMA_slow", "EMA20", "EMA50", "EMA34", "EMA89", "EMA_200", "MACD_hist", "RSI", "RSI2", "RSI3", "ATR"]:
            df[col] = pd.Series(0, index=df.index, dtype=float)
    
    # ADX needs >= 15 rows
    if n >= 15:
        df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    else:
        df["ADX"] = pd.Series(0, index=df.index, dtype=float)
    
    # Bollinger Bands
    if n >= 20:
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
    else:
        df["BB_upper"] = df["Close"]
        df["BB_lower"] = df["Close"]
    
    # Keltner Channels
    if n >= 20:
        try:
            kc = ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=20)
            df["KC_upper"] = kc.keltner_channel_hband()
            df["KC_lower"] = kc.keltner_channel_lband()
            df["KC_mid"]   = kc.keltner_channel_mband()
        except Exception:
            df["KC_upper"] = df["Close"]
            df["KC_lower"] = df["Close"]
            df["KC_mid"]   = df["Close"]
    else:
        df["KC_upper"] = df["Close"]
        df["KC_lower"] = df["Close"]
        df["KC_mid"]   = df["Close"]
    
    # VWAP
    try:
        if "Volume" in df.columns and df["Volume"].fillna(0).sum() > 0 and n >= 5:
            vwap = ta.volume.VolumeWeightedAveragePrice(
                high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=min(20, n)
            )
            df["VWAP"] = vwap.volume_weighted_average_price()
        else:
            df["VWAP"] = df["Close"]
    except Exception:
        df["VWAP"] = df["Close"]
    
    # EMA slopes
    slope_lookback = 5 if n >= 6 else max(1, n - 1)
    denom_fast = (df["EMA_fast"].shift(slope_lookback).abs() + 1e-9)
    denom_slow = (df["EMA_slow"].shift(slope_lookback).abs() + 1e-9)
    df["EMA_fast_slope"] = df["EMA_fast"].diff(slope_lookback) / (denom_fast * slope_lookback)
    df["EMA_slow_slope"] = df["EMA_slow"].diff(slope_lookback) / (denom_slow * slope_lookback)
    
    # Donchian channels
    dc_window = 20 if n >= 20 else max(5, n)
    df["DC_high"] = df["High"].rolling(dc_window, min_periods=5).max()
    df["DC_low"] = df["Low"].rolling(dc_window, min_periods=5).min()
    
    # Fill NaNs
    indicator_cols = ["EMA_fast","EMA_slow","EMA20","EMA50","EMA34","EMA89","EMA_200","MACD_hist","RSI","RSI2","RSI3",
                      "ATR","ADX","BB_upper","BB_lower","EMA_fast_slope","EMA_slow_slope","DC_high","DC_low",
                      "KC_upper","KC_lower","KC_mid","VWAP"]
    df[indicator_cols] = df[indicator_cols].fillna(method="ffill").fillna(method="bfill")
    
    # Z-scores
    kc_ref = df.get("KC_mid", df["EMA_fast"])
    df["Z_KC"] = (df["Close"] - kc_ref) / (df["ATR"] + 1e-9)
    df["Z_VWAP"] = (df["Close"] - df.get("VWAP", df["Close"])) / (df["ATR"] + 1e-9)
    
    # True Range
    df["TR"] = (df["High"] - df["Low"]).abs()
    df["TR_med50"] = df["TR"].rolling(50, min_periods=10).median()
    
    return df


def is_us_rth(ts) -> bool:
    """Check if timestamp is within US Regular Trading Hours (9:30-16:00 ET)"""
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize('UTC')
        t = t.tz_convert('America/New_York')
        after_open = (t.hour > 9) or (t.hour == 9 and t.minute >= 30)
        before_close = (t.hour < 16) or (t.hour == 16 and t.minute == 0)
        return after_open and before_close
    except Exception:
        return True

