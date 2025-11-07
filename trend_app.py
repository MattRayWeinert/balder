import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from typing import Optional

# --- Config ---
st.set_page_config(page_title="Balder - Trade Advisor", page_icon="💹", layout="wide")

# --- Warning Banner ---
st.markdown("""
<div style='background-color: #ff4b4b; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #d32f2f;'>
    <h3 style='color: white; margin: 0; padding: 0;'>⚠️ DELAYED DATA WARNING</h3>
    <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>
        <strong>This app uses Yahoo Finance data via yfinance, which is delayed by 15-20 minutes.</strong><br>
        📊 Safe for: Backtesting, strategy development, paper trading, educational purposes<br>
        🚫 NOT suitable for: Live trading decisions, real-time scalping, actual trade execution<br>
        💡 For live trading, use real-time data from Interactive Brokers, TradingView, or your broker's platform.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container { padding-top: 80px; }
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh ---
# Moved into the Trade Signals page so Backtest/Key Terms do not auto-refresh.

# --- Session Defaults ---
defaults = {
    "risk_factor": 1.0,
    "risk_reward": 2.0,
    "interval": "2m",
    "refresh_sec": 30,
    "futures_symbol": "ES=F",
    "custom_future": ""
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Key Terms ---
EXPLANATIONS = {
    "Close": "Last price during interval.",
    "EMA Fast": "Short-term EMA (5 periods).",
    "EMA Slow": "Long-term EMA (20 periods).",
    "ATR": "Average True Range, measures volatility.",
    "RSI": "Momentum indicator. <30=oversold, >70=overbought.",
    "MACD": "Trend/momentum indicator.",
    "ADX": "Trend strength.",
    "Take Profit": "Target price.",
    "Stop Loss": "Exit price to limit loss.",
    "Buy Signal": "Bullish setup.",
    "Sell Signal": "Bearish setup.",
    "Neutral": "Indecisive.",
    "MTI": "Multi-Timeframe Indicator. Confirms trend across intervals.",
    "Confidence": "Confidence % based on MTI score alignment.",
    "Risk Sensitivity (ATR multiplier)": "Adjust stop loss/take profit based on volatility.",
    "Risk:Reward Ratio": "Profit vs loss."
}

def show_explanations():
    st.subheader("📖 Key Terms Explained")
    for term, desc in EXPLANATIONS.items():
        st.markdown(f"**{term}**: {desc}")

# --- Helper: safe indicators on a DataFrame ---
def add_indicators_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds indicator columns to df with safeguards for short windows.
    Always returns columns: EMA_fast, EMA_slow, MACD_hist, RSI, ATR, ADX, BB_upper, BB_lower,
    EMA_fast_slope, EMA_slow_slope, DC_high, DC_low.
    """
    df = df.copy()
    # Ensure numeric
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
        # explicit EMAs for scalping rules
        df["EMA20"]      = ta.trend.EMAIndicator(df["Close"], window=min(20, n)).ema_indicator()
        df["EMA50"]      = ta.trend.EMAIndicator(df["Close"], window=min(50, n)).ema_indicator()
        df["EMA34"]      = ta.trend.EMAIndicator(df["Close"], window=min(34, n)).ema_indicator()
        df["EMA89"]      = ta.trend.EMAIndicator(df["Close"], window=min(89, n)).ema_indicator()
        # long-term trend context
        df["EMA_200"]   = ta.trend.EMAIndicator(df["Close"], window=min(200, n)).ema_indicator()
        df["MACD_hist"] = ta.trend.MACD(df["Close"], window_slow=21, window_fast=8, window_sign=5).macd_diff()
        df["RSI"]       = ta.momentum.RSIIndicator(df["Close"], window=min(7, n)).rsi()
        # very short lookback RSI for 1m scalping
        df["RSI2"]      = ta.momentum.RSIIndicator(df["Close"], window=min(2, n)).rsi()
        # short lookback RSI for 2m scalping
        df["RSI3"]      = ta.momentum.RSIIndicator(df["Close"], window=min(3, n)).rsi()
        df["ATR"]       = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=min(14, n)).average_true_range()
    else:
        df["EMA_fast"] = df["EMA_slow"] = df["EMA20"] = df["EMA50"] = df["EMA34"] = df["EMA89"] = df["EMA_200"] = df["MACD_hist"] = df["RSI"] = df["RSI2"] = df["RSI3"] = df["ATR"] = pd.Series(0, index=df.index, dtype=float)

    # ADX needs >= 15-ish rows reliably
    if n >= 15:
        df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    else:
        df["ADX"] = pd.Series(0, index=df.index, dtype=float)

    # Bollinger Bands need ~20 rows
    if n >= 20:
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
    else:
        df["BB_upper"] = df["Close"]
        df["BB_lower"] = df["Close"]

    # Keltner Channels (for pullback/overextension context)
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

    # VWAP (windowed as a short-term proxy; true session VWAP requires session resets)
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

    # Add EMA slopes (normalized per bar over lookback)
    slope_lookback = 5 if n >= 6 else max(1, n - 1)
    denom_fast = (df["EMA_fast"].shift(slope_lookback).abs() + 1e-9)
    denom_slow = (df["EMA_slow"].shift(slope_lookback).abs() + 1e-9)
    df["EMA_fast_slope"] = df["EMA_fast"].diff(slope_lookback) / (denom_fast * slope_lookback)
    df["EMA_slow_slope"] = df["EMA_slow"].diff(slope_lookback) / (denom_slow * slope_lookback)

    # Donchian channels for breakout / context
    dc_window = 20 if n >= 20 else max(5, n)
    df["DC_high"] = df["High"].rolling(dc_window, min_periods=5).max()
    df["DC_low"] = df["Low"].rolling(dc_window, min_periods=5).min()

    # Fill any residual NaNs from initial warmup
    df[["EMA_fast","EMA_slow","EMA20","EMA50","EMA34","EMA89","EMA_200","MACD_hist","RSI","RSI2","RSI3","ATR","ADX","BB_upper","BB_lower","EMA_fast_slope","EMA_slow_slope","DC_high","DC_low","KC_upper","KC_lower","KC_mid","VWAP"]] = \
        df[["EMA_fast","EMA_slow","EMA20","EMA50","EMA34","EMA89","EMA_200","MACD_hist","RSI","RSI2","RSI3","ATR","ADX","BB_upper","BB_lower","EMA_fast_slope","EMA_slow_slope","DC_high","DC_low","KC_upper","KC_lower","KC_mid","VWAP"]].fillna(method="ffill").fillna(method="bfill")

    # Z-score of price vs KC mid in ATR units (for range extension sizing)
    kc_ref = df.get("KC_mid", df["EMA_fast"])
    df["Z_KC"] = (df["Close"] - kc_ref) / (df["ATR"] + 1e-9)
    # Z-score vs VWAP and tiny bar filter
    df["Z_VWAP"] = (df["Close"] - df.get("VWAP", df["Close"])) / (df["ATR"] + 1e-9)
    df["TR"] = (df["High"] - df["Low"]).abs()
    df["TR_med50"] = df["TR"].rolling(50, min_periods=10).median()

    return df
# --- Cached data fetcher to speed up backtests ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(ticker: str, interval: str, start: Optional[str]=None, end: Optional[str]=None, period: Optional[str]=None):
    try:
        if start and end:
            return yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if period:
            return yf.download(ticker, period=period, interval=interval, progress=False)
        return yf.download(ticker, period='60d', interval=interval, progress=False)
    except Exception:
        return pd.DataFrame()

# --- Helper: get nearest-at-or-before row from higher timeframe ---
def get_higher_row_at_or_before(df_hi: pd.DataFrame, ts):
    if df_hi is None or df_hi.empty:
        return None
    try:
        if ts in df_hi.index:
            return df_hi.loc[ts]
        subset = df_hi[df_hi.index <= ts]
        if not subset.empty:
            return subset.iloc[-1]
        return None
    except Exception:
        return None

# --- Helper: US Regular Trading Hours check (America/New_York 09:30-16:00)
def is_us_rth(ts) -> bool:
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

# --- Helper: scalping decision on a single bar ---
def decide_scalp_entry(df_base: pd.DataFrame, idx: int, df_5m: Optional[pd.DataFrame], risk_factor: float, base_interval: str, risk_reward: float):
    if df_base is None or df_base.empty or idx <= 0:
        return None
    n = len(df_base)
    if idx >= n:
        return None
    row = df_base.iloc[idx]
    prev = df_base.iloc[idx-1]

    # Tiny bar / data guards
    tr_med = float(row.get("TR_med50", 0.0))
    tr_cur = float(row.get("TR", 0.0))
    if tr_med > 0 and tr_cur < 0.5 * tr_med:
        return None

    # RTH-only for 1m/2m scalping
    if base_interval in ("1m","2m") and not is_us_rth(df_base.index[idx]):
        return None

    # Higher timeframe bias (5m EMA20 slope)
    bias_up = True
    bias_dn = True
    if df_5m is not None and not df_5m.empty:
        hi_row = get_higher_row_at_or_before(df_5m, df_base.index[idx])
        if hi_row is not None:
            ema20 = float(hi_row.get("EMA20", hi_row.get("EMA_slow", row.get("EMA_slow", 0.0))))
            try:
                # Find nearest-at-or-before positional index
                idxer = df_5m.index.get_indexer([hi_row.name], method='pad')
                pos = idxer[0] if len(idxer) > 0 and idxer[0] != -1 else df_5m.index.searchsorted(hi_row.name) - 1
                if pos < 0:
                    pos = 0
                prev_pos = max(0, pos - 1)
                ema20_prev = float(df_5m["EMA20"].iloc[prev_pos]) if "EMA20" in df_5m.columns else ema20
            except Exception:
                ema20_prev = ema20
            slope = ema20 - ema20_prev
            bias_up = slope > 0
            bias_dn = slope < 0

    close = float(row["Close"])
    ema20 = float(row.get("EMA20", row.get("EMA_slow", close)))
    ema50 = float(row.get("EMA50", ema20))
    ema200 = float(row.get("EMA_200", close))
    ema34 = float(row.get("EMA34", ema20))
    ema89 = float(row.get("EMA89", ema50))
    vwap = float(row.get("VWAP", close))
    rsi2_prev = float(prev.get("RSI2", 50.0))
    rsi3_prev = float(prev.get("RSI3", rsi2_prev))
    kc_upper = float(row.get("KC_upper", close))
    kc_lower = float(row.get("KC_lower", close))
    z_vwap = float(row.get("Z_VWAP", 0.0))

    # Pullback context (last 3 lows/highs)
    start = max(0, idx-3)
    recent_lows = df_base["Low"].iloc[start:idx+1]
    recent_highs = df_base["High"].iloc[start:idx+1]

    # Momentum shape
    high = float(row["High"]); low = float(row["Low"]); open_ = float(row.get("Open", close))
    rng = max(high - low, 1e-9)
    pos = (close - low) / rng
    pos_short = (high - close) / rng

    # ATR sizing
    atr = float(row.get("ATR", max(rng, 1e-9)))
    atr_scale = 0.5 if base_interval == "1m" else (0.8 if base_interval == "2m" else 1.0)
    atr_eff = max(1e-9, atr * risk_factor * atr_scale)
    stop_mult = 0.6 if base_interval == "1m" else (0.7 if base_interval == "2m" else 0.6)

    # Pure Breakout Strategy for 2m - Balanced (High Quality + Reasonable Frequency)
    if base_interval == "2m":
        # Calculate breakout levels (40-bar for significant levels)
        lookback = min(40, idx)
        if lookback >= 20:
            range_high = float(df_base["High"].iloc[max(0, idx-lookback):idx].max())
            range_low = float(df_base["Low"].iloc[max(0, idx-lookback):idx].min())
            
            # Only trade if this is a meaningful range
            range_size = range_high - range_low
            range_is_wide = range_size >= (1.5 * atr)  # Range must be at least 1.5 ATRs wide
            
            # Must be near the extreme (within 0.5 ATR)
            near_high = (close >= range_high - 0.5 * atr)
            near_low = (close <= range_low + 0.5 * atr)
        else:
            return None  # Need sufficient history
        
        if not range_is_wide:
            return None  # Skip tight ranges
        
        # Volume confirmation - Strong but achievable
        vol_surge = False
        if "Volume" in df_base.columns and idx >= 50:
            vol_med = float(df_base["Volume"].iloc[max(0, idx-50):idx].median())
            cur_vol = float(row.get("Volume", vol_med))
            if vol_med > 0:
                vol_surge = (cur_vol >= 1.5 * vol_med)  # 50% above median
        else:
            vol_surge = True  # Don't block if no volume data
        
        # Trend filter: EMAs must agree
        ema_spread = abs(ema34 - ema89)
        strong_trend_up = (ema34 > ema89) and (ema_spread >= 0.3 * atr)
        strong_trend_down = (ema34 < ema89) and (ema_spread >= 0.3 * atr)
        
        # ADX filter: prefer trending but allow weaker trends
        adx_val = float(row.get("ADX", 0.0))
        is_trending = (adx_val >= 20)  # Moderate trend
        
        # LONG: Strong breakout above high
        breakout_up = (close > range_high) and (high > range_high)
        strong_close = (pos >= 0.7)  # Close in top 30%
        momentum_up = (close > float(prev.get("Close", close)))
        body_size = abs(close - open_)
        strong_body = (body_size >= 0.5 * rng)  # Body 50%+ of range
        breakout_distance = (close - range_high)
        clear_break = (breakout_distance >= 0.2 * atr)  # Visible break
        
        # Multi-bar momentum: last 2 bars rising (loosened from 3)
        if idx >= 2:
            prev_close = float(df_base["Close"].iloc[idx-1])
            prev2_close = float(df_base["Close"].iloc[idx-2])
            multi_bar_up = (prev2_close <= prev_close <= close)
        else:
            multi_bar_up = True
        
        long_allow = (breakout_up and strong_close and momentum_up and strong_body and 
                     strong_trend_up and is_trending and vol_surge and clear_break and near_high and multi_bar_up)
        long_mr = False
        
    else:
        # 1m logic (unchanged)
        long_trend = (ema20 > ema50) and (close > vwap) and bias_up
        long_pullback = (recent_lows.min() <= ema20)
        long_reentry = (close > ema20)
        long_momo = (pos >= 0.75) or (close > float(prev.get("High", close)))
        long_rsi_ok = (rsi2_prev <= 10)
        long_allow = long_trend and long_pullback and long_reentry and long_momo and long_rsi_ok
        long_mr = (close >= ema200) and (rsi2_prev <= 5) and (close < kc_lower) and (z_vwap <= -1.0)

    if long_allow or long_mr:
        entry = close
        
        # Breakout-specific stop placement for 2m
        if base_interval == "2m" and long_allow:
            # Stop just below the breakout level
            stop = range_high - (0.5 * atr_eff)  # Tight stop below breakout
            tp = entry + risk_reward * (entry - stop)  # Standard R:R from tight stop
        else:
            # Standard stop for 1m or mean reversion
            stop = entry - stop_mult * atr_eff
            proj_bars = 5 if base_interval == "1m" else 3
            avg_tr3 = float(df_base["TR"].iloc[max(0, idx-3):idx].mean()) if "TR" in df_base.columns else (high - low)
            proj = max(avg_tr3, atr) * proj_bars
            tp_dist = min(risk_reward * (entry - stop), 1.2 * proj)
            tp = entry + tp_dist
            # Room-to-target gating for 1m only
            if base_interval == "1m":
                dc_high = float(df_base["DC_high"].iloc[idx]) if "DC_high" in df_base.columns else entry + 1e9
                if (dc_high - entry) < 0.75 * (tp - entry):
                    return None
                if avg_tr3 * proj_bars < (tp - entry):
                    return None
        return {"type":"Buy","entry":entry,"stop":stop,"tp":tp,"is_mr": long_mr}

    # SHORT logic
    if base_interval == "2m":
        # SHORT: Strong breakdown below low
        breakout_down = (close < range_low) and (low < range_low)
        strong_close_down = (pos_short >= 0.7)  # Close in bottom 30%
        momentum_down = (close < float(prev.get("Close", close)))
        breakdown_distance = (range_low - close)
        clear_break_down = (breakdown_distance >= 0.2 * atr)  # Visible break
        
        # Multi-bar momentum: last 2 bars falling
        if idx >= 2:
            prev_close = float(df_base["Close"].iloc[idx-1])
            prev2_close = float(df_base["Close"].iloc[idx-2])
            multi_bar_down = (prev2_close >= prev_close >= close)
        else:
            multi_bar_down = True
        
        short_allow = (breakout_down and strong_close_down and momentum_down and strong_body and 
                      strong_trend_down and is_trending and vol_surge and clear_break_down and near_low and multi_bar_down)
        short_mr = False
        
    else:
        # 1m logic (unchanged)
        short_trend = (ema20 < ema50) and (close < vwap) and bias_dn
        short_pullback = (recent_highs.max() >= ema20)
        short_reentry = (close < ema20)
        short_momo = (pos_short >= 0.75) or (close < float(prev.get("Low", close)))
        short_rsi_ok = (rsi2_prev >= 90)
        short_allow = short_trend and short_pullback and short_reentry and short_momo and short_rsi_ok
        short_mr = (close <= ema200) and (rsi2_prev >= 95) and (close > kc_upper) and (z_vwap >= 1.0)

    if short_allow or short_mr:
        entry = close
        
        # Breakout-specific stop placement for 2m
        if base_interval == "2m" and short_allow:
            # Stop just above the breakdown level
            stop = range_low + (0.5 * atr_eff)  # Tight stop above breakdown
            tp = entry - risk_reward * (stop - entry)  # Standard R:R from tight stop
        else:
            # Standard stop for 1m or mean reversion
            stop = entry + stop_mult * atr_eff
            proj_bars = 5 if base_interval == "1m" else 3
            avg_tr3 = float(df_base["TR"].iloc[max(0, idx-3):idx].mean()) if "TR" in df_base.columns else (high - low)
            proj = max(avg_tr3, atr) * proj_bars
            tp_dist = min(risk_reward * (stop - entry), 1.2 * proj)
            tp = entry - tp_dist
            # Room-to-target gating for 1m only
            if base_interval == "1m":
                dc_low = float(df_base["DC_low"].iloc[idx]) if "DC_low" in df_base.columns else entry - 1e9
                if (entry - dc_low) < 0.75 * (entry - tp):
                    return None
                if avg_tr3 * proj_bars < (entry - tp):
                    return None
        return {"type":"Sell","entry":entry,"stop":stop,"tp":tp,"is_mr": short_mr}

    return None

# --- Trade Signal with MTI & Confidence (IMPROVED) ---
def get_trade_signal_with_targets(ticker, base_interval, risk_factor, risk_reward):
    interval_hierarchy = {"1m":["5m","15m"],"2m":["5m","15m"],"5m":["15m","1h"], "15m":["1h","4h"],"1h":["4h"]}
    higher_intervals = interval_hierarchy.get(base_interval, [])
    all_intervals = [base_interval] + higher_intervals
    interval_scores = {}
    interval_data = {}
    period_map = {"1m":"3d","2m":"3d","5m":"3d","15m":"7d", "1h":"60d"}
    max_possible_score = 0

    for interval in all_intervals:
        period = period_map.get(interval,"60d")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception:
            df = pd.DataFrame()

        if df.empty or len(df) < 5:
            interval_scores[interval] = 0
            interval_data[interval] = pd.DataFrame()
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.astype(float, errors='ignore').dropna(subset=["Close"], how="any")
        df = add_indicators_safe(df)

        latest = df.iloc[-1]
        score = 0.0

        # Determine trend direction
        is_uptrend = latest["EMA_fast"] > latest["EMA_slow"]
        is_downtrend = latest["EMA_fast"] < latest["EMA_slow"]

        # 1) Trend bias
        score += 1.0 if is_uptrend else -1.0

        # 2) Momentum: MACD sign + acceleration
        score += 0.5 if latest["MACD_hist"] > 0 else -0.5
        if len(df) >= 2:
            macd_prev = df["MACD_hist"].iloc[-2]
            if latest["MACD_hist"] > macd_prev and latest["MACD_hist"] > 0:
                score += 0.25
            elif latest["MACD_hist"] < macd_prev and latest["MACD_hist"] < 0:
                score -= 0.25

        # 3) RSI regime alignment
        if is_uptrend and latest["RSI"] >= 50:
            score += 0.5
        elif is_downtrend and latest["RSI"] <= 50:
            score -= 0.5

        # 4) Pullback entry: bounce from EMA_fast/KC mid and over/under VWAP confirmation
        recent_lows = df["Low"].iloc[-3:]
        recent_highs = df["High"].iloc[-3:]
        kc_mid = float(latest.get("KC_mid", latest["EMA_fast"]))
        vwap_ok_up = float(latest.get("VWAP", latest["EMA_fast"])) < float(latest["Close"])
        vwap_ok_dn = float(latest.get("VWAP", latest["EMA_fast"])) > float(latest["Close"])
        if is_uptrend:
            if (recent_lows.min() <= latest["EMA_fast"]) or (recent_lows.min() <= kc_mid):
                if latest["Close"] > latest["EMA_fast"] and vwap_ok_up:
                    score += 0.75
        elif is_downtrend:
            if (recent_highs.max() >= latest["EMA_fast"]) or (recent_highs.max() >= kc_mid):
                if latest["Close"] < latest["EMA_fast"] and vwap_ok_dn:
                    score -= 0.75

        # 5) Breakout confirmation: Donchian
        if "DC_high" in df.columns and "DC_low" in df.columns and len(df) >= 2:
            dc_high_prev = df["DC_high"].iloc[-2]
            dc_low_prev = df["DC_low"].iloc[-2]
            if is_uptrend and latest["Close"] > dc_high_prev and vwap_ok_up:
                score += 0.5
            elif is_downtrend and latest["Close"] < dc_low_prev and vwap_ok_dn:
                score -= 0.5

        # 6) ADX soft filter (not gating)
        if latest["ADX"] > 18:
            score += 0.25 if is_uptrend else -0.25

        # 7) Overextension penalty vs KC
        kc_upper = float(latest.get("KC_upper", latest["BB_upper"]))
        kc_lower = float(latest.get("KC_lower", latest["BB_lower"]))
        if is_uptrend and latest["Close"] > kc_upper:
            score -= 0.5
        elif is_downtrend and latest["Close"] < kc_lower:
            score -= 0.5  # penalize long bias; favors short setup

        interval_scores[interval] = score
        interval_data[interval] = df
        max_possible_score += 4.0  # positive components only for confidence

    total_score = sum(interval_scores.values())
    confidence = 0 if max_possible_score == 0 else max(min(abs(total_score)/max_possible_score*100,100),0)
    
    # Require relaxed multi-timeframe alignment:
    # direction agrees on base timeframe and not strongly opposed by all higher TFs
    base_dir_up = interval_scores.get(base_interval, 0) > 0
    base_dir_down = interval_scores.get(base_interval, 0) < 0
    higher_scores = [interval_scores[i] for i in higher_intervals if i in interval_scores]
    disagree_count = sum(1 for s in higher_scores if (s < 0 if base_dir_up else s > 0))
    timeframe_alignment = (disagree_count <= 1)  # allow one disagreement
    scalp_mode = (base_interval == "1m")

    base_df = interval_data.get(base_interval, pd.DataFrame())
    latest_base = base_df.iloc[-1] if not base_df.empty else None
    entry = float(latest_base["Close"]) if latest_base is not None else 0.0
    raw_atr = float(latest_base["ATR"]) if latest_base is not None else entry * 0.005
    # Scalping: use smaller ATR fraction on 1m to tighten TP/SL
    atr_scale = 0.5 if base_interval == "1m" else 1.0
    atr_value = raw_atr * risk_factor * atr_scale
    
    # ADX filter - softer minimum trend strength on base timeframe
    base_adx = float(latest_base["ADX"]) if latest_base is not None else 0.0
    is_trending = base_adx >= (18 if scalp_mode else 22)
    # ATR distance filter: avoid chasing extended moves (looser)
    atr_den = float(latest_base["ATR"]) + 1e-9 if latest_base is not None else 1.0
    ema_fast = float(latest_base["EMA_fast"]) if latest_base is not None else entry
    atr_distance = abs(entry - ema_fast) / atr_den
    # Range helpers
    kc_mid = float(latest_base.get("KC_mid", ema_fast)) if latest_base is not None else entry
    kc_upper = float(latest_base.get("KC_upper", entry))
    kc_lower = float(latest_base.get("KC_lower", entry))
    ema_slow_slope = float(latest_base.get("EMA_slow_slope", 0.0))
    z_kc = float(latest_base.get("Z_KC", 0.0))
    # 1m volume gating (assumes user trades during market hours)
    vol_ok = True
    if scalp_mode and latest_base is not None and "Volume" in base_df.columns:
        try:
            vol_med = float(base_df["Volume"].rolling(50, min_periods=10).median().iloc[-1])
            cur_vol = float(latest_base.get("Volume", vol_med))
            if vol_med > 0:
                vol_ok = (cur_vol >= 0.5 * vol_med)
        except Exception:
            vol_ok = True

    # Improved entry conditions: moderate confidence + relaxed alignment + softer ADX + looser distance (Trending regime)
    min_conf = 35 if scalp_mode else 50
    min_adx = 10 if scalp_mode else 15
    max_atr_dist = 2.25 if scalp_mode else 1.75
    if is_trending and (confidence < min_conf or (not timeframe_alignment and not scalp_mode) or base_adx < min_adx or atr_distance > max_atr_dist or not vol_ok):
        signal, color = "⚪ Wait", "gray"
        take_profit = stop_loss = None
    elif is_trending and total_score > 0:
        # Trend entry requires breakout or strong bullish candle confirmation and reasonable ATR regime
        atr_med = float(base_df["ATR"].rolling(50, min_periods=10).median().iloc[-1]) if not base_df.empty else atr_value
        atr_ratio = (float(latest_base["ATR"]) / (atr_med + 1e-9)) if atr_med > 0 else 1.0
        dc_ok = False
        if "DC_high" in base_df.columns and len(base_df) >= 2:
            dc_ok = float(latest_base["Close"]) > float(base_df["DC_high"].iloc[-2])
        body = float(latest_base["Close"]) - float(latest_base["Open"])
        range_c = max(float(latest_base["High"]) - float(latest_base["Low"]), 1e-9)
        strong_bull = body > 0 and (body >= 0.5 * range_c) and (float(latest_base["Close"]) > float(latest_base["EMA_fast"]))
        # Additional scalping allowance: above VWAP with positive EMA_slow_slope
        vwap_up = float(latest_base.get("VWAP", entry)) < entry
        slow_slope_up = float(latest_base.get("EMA_slow_slope", 0.0)) > 0
        # 3-bar breakout pattern
        long_pattern = False
        try:
            if len(base_df) >= 2:
                prev = base_df.iloc[-2]
                long_pattern = (entry > float(prev.get("High", entry))) and (float(prev.get("Low", entry)) <= ema_fast) and (entry > ema_fast)
        except Exception:
            long_pattern = False
        scalp_ok = scalp_mode and vwap_up and slow_slope_up and (long_pattern or strong_bull or dc_ok)
        if ((dc_ok or strong_bull) or scalp_ok) and (0.6 <= atr_ratio <= 2.5) and (float(latest_base.get("EMA_200", entry)) <= entry):
            signal, color = "🚀 Buy / Bullish Setup", "green"
            take_profit = entry + risk_reward * atr_value
            stop_loss = entry - atr_value
        else:
            signal, color = "⚪ Wait", "gray"
            take_profit = stop_loss = None
    elif is_trending and total_score < 0:
        atr_med = float(base_df["ATR"].rolling(50, min_periods=10).median().iloc[-1]) if not base_df.empty else atr_value
        atr_ratio = (float(latest_base["ATR"]) / (atr_med + 1e-9)) if atr_med > 0 else 1.0
        dc_ok = False
        if "DC_low" in base_df.columns and len(base_df) >= 2:
            dc_ok = float(latest_base["Close"]) < float(base_df["DC_low"].iloc[-2])
        body = float(latest_base["Open"]) - float(latest_base["Close"])
        range_c = max(float(latest_base["High"]) - float(latest_base["Low"]), 1e-9)
        strong_bear = body > 0 and (body >= 0.5 * range_c) and (float(latest_base["Close"]) < float(latest_base["EMA_fast"]))
        vwap_dn = float(latest_base.get("VWAP", entry)) > entry
        slow_slope_dn = float(latest_base.get("EMA_slow_slope", 0.0)) < 0
        short_pattern = False
        try:
            if len(base_df) >= 2:
                prev = base_df.iloc[-2]
                short_pattern = (entry < float(prev.get("Low", entry))) and (float(prev.get("High", entry)) >= ema_fast) and (entry < ema_fast)
        except Exception:
            short_pattern = False
        scalp_ok = scalp_mode and vwap_dn and slow_slope_dn and (short_pattern or strong_bear or dc_ok)
        if ((dc_ok or strong_bear) or scalp_ok) and (0.6 <= atr_ratio <= 2.5) and (float(latest_base.get("EMA_200", entry)) >= entry):
            signal, color = "🔻 Sell / Bearish Setup", "red"
            take_profit = entry - risk_reward * atr_value
            stop_loss = entry + atr_value
        else:
            signal, color = "⚪ Wait", "gray"
            take_profit = stop_loss = None
    # Range regime mean-reversion entries (looser alignment; require extension and slope near flat)
    elif not is_trending:
        # Flat slope threshold tuned to our normalized slope
        flat_slope = abs(ema_slow_slope) < 0.0015
        # Mean reversion long if extended below KC_lower with RSI supportive
        if vol_ok and (entry < kc_lower) and (float(latest_base["RSI"]) <= 55) and flat_slope and (z_kc <= -0.8):
            signal, color = "🚀 Buy (Range MR)", "green"
            take_profit = kc_mid
            stop_loss = entry - 1.25 * atr_value
        # Mean reversion short if extended above KC_upper with RSI supportive
        elif vol_ok and (entry > kc_upper) and (float(latest_base["RSI"]) >= 45) and flat_slope and (z_kc >= 0.8):
            signal, color = "🔻 Sell (Range MR)", "red"
            take_profit = kc_mid
            stop_loss = entry + 1.25 * atr_value
        else:
            signal, color = "⚪ Wait", "gray"
            take_profit = stop_loss = None
    else:
        signal, color = "⚪ Neutral / Wait", "gray"
        take_profit = stop_loss = None

    return {
        "Datetime": latest_base.name if latest_base is not None else None,
        "Close": round(entry,2),
        "Signal": signal,
        "Score": round(total_score,2),
        "Confidence": round(confidence,2),
        "RSI": round(float(latest_base["RSI"]),2) if latest_base is not None else None,
        "ATR": round(atr_value,2),
        "Take Profit": round(take_profit,2) if take_profit else None,
        "Stop Loss": round(stop_loss,2) if stop_loss else None,
        "MTI Scores": interval_scores,
        "Data": base_df
    }, color, base_df

# --- MTI-Consistent Backtest Function ---
def backtest_signals(ticker, base_interval, risk_factor=1.0, risk_reward=2.0, start_date=None, end_date=None):
    """
    Backtests trades using the same multi-indicator MTI logic as the live trade signals.
    Includes safeguards to avoid IndexError when windows are short.
    """
    interval_hierarchy = {"1m":["5m","15m"],"2m":["5m","15m"],"5m":["15m","1h"], "15m":["1h","4h"],"1h":["4h"]}
    higher_intervals = interval_hierarchy.get(base_interval, [])
    all_intervals = [base_interval] + higher_intervals
    period_map = {"1m":"7d","2m":"60d","5m":"60d","15m":"60d","1h":"max"}
    period = period_map.get(base_interval,"max")
    
    # Base interval data
    if start_date and end_date:
        from datetime import timedelta
        end_inclusive = end_date + timedelta(days=1)
        df_base = fetch_prices(ticker, base_interval, start=str(start_date), end=str(end_inclusive))
    else:
        df_base = fetch_prices(ticker, base_interval, period=period)
    if df_base.empty:
        return [], pd.DataFrame()
    if isinstance(df_base.columns, pd.MultiIndex):
        df_base.columns = df_base.columns.get_level_values(0)
    if "Close" not in df_base.columns and "Adj Close" in df_base.columns:
        df_base["Close"] = df_base["Adj Close"]
    if "High" not in df_base.columns: df_base["High"] = df_base["Close"]
    if "Low" not in df_base.columns: df_base["Low"] = df_base["Close"]
    df_base = df_base.dropna(subset=["Close","High","Low"])
    df_base[["Close","High","Low"]] = df_base[["Close","High","Low"]].apply(pd.to_numeric)

    # Pre-compute ATR on base for entry/stop sizing (optional but nice)
    df_base = add_indicators_safe(df_base)
    
    # Filter to requested date and regular trading hours after indicators are computed
    if start_date and end_date:
        # Filter to the specific date
        df_base = df_base.loc[str(start_date):str(end_date)]
        if df_base.empty:
            return [], pd.DataFrame()
        # Filter to US regular trading hours (9:30 AM - 4:00 PM ET) if on intraday timeframes
        if base_interval in ("1m", "2m", "5m", "15m"):
            try:
                df_base.index = pd.to_datetime(df_base.index)
                if df_base.index.tz is None:
                    df_base.index = df_base.index.tz_localize('UTC')
                df_base.index = df_base.index.tz_convert('America/New_York')
                # Keep only 9:30 AM - 4:00 PM ET
                filtered = df_base.between_time('09:30', '16:00')
                if not filtered.empty:
                    df_base = filtered
                # If filtering resulted in empty df, keep original (some data better than none)
            except Exception:
                pass  # If timezone conversion fails, keep all data

    # Higher timeframe data
    higher_data = {}
    for interval in higher_intervals:
        try:
            if start_date and end_date:
                from datetime import timedelta
                end_inclusive = end_date + timedelta(days=1)
                df = fetch_prices(ticker, interval, start=str(start_date), end=str(end_inclusive))
            else:
                period_hi = period_map.get(interval, "max")
                df = fetch_prices(ticker, interval, period=period_hi)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.astype(float, errors='ignore').dropna()
            higher_data[interval] = add_indicators_safe(df)
        except Exception:
            higher_data[interval] = pd.DataFrame()
    
    trades = []
    last_exit_index = -10**9  # cooldown tracker

    # Walk through each candle on base timeframe
    df5 = higher_data.get("5m", pd.DataFrame())
    for i in range(len(df_base)):
        candle_time = df_base.index[i]
        ready = False
        trade_type = None
        stop = tp = None
        is_range_trade = False
        
        latest_candle = df_base.iloc[i]
        entry = float(latest_candle["Close"])
        
        # Initialize these early for all paths
        interval_scores = {}
        max_possible_score = 0

        # Use breakout strategy for 2m, fallback for other intervals
        if base_interval == "2m":
            decision = decide_scalp_entry(df_base, i, df5, risk_factor, base_interval, risk_reward)
            if decision is not None:
                trade_type = decision["type"]
                stop = float(decision["stop"])
                tp = float(decision["tp"])
                is_range_trade = bool(decision.get("is_mr", False))
                ready = True
                # Set dummy values for variables needed later
                total_score = 1.0
                confidence = 100.0
            else:
                continue  # Skip this bar if breakout strategy says no trade
        
        if not ready:
            for interval in all_intervals:
                df = df_base if interval == base_interval else higher_data.get(interval, pd.DataFrame())
                if df.empty or candle_time not in df.index:
                    interval_scores[interval] = 0
                    continue
                idx = df.index.get_loc(candle_time)
                window_df = df.iloc[max(0, idx-20): idx+1]  # up to 21 rows
                if len(window_df) < 5:
                    interval_scores[interval] = 0
                    continue
                # Indicators are already present via add_indicators_safe on df slices
                latest = window_df.iloc[-1]
                score = 0
                # Determine trend direction first
                is_uptrend = latest["EMA_fast"] > latest["EMA_slow"]
                is_downtrend = latest["EMA_fast"] < latest["EMA_slow"]
                # 1) Trend bias
                score += 1.0 if is_uptrend else -1.0
                # 2) Momentum: MACD sign + acceleration
                score += 0.5 if latest["MACD_hist"] > 0 else -0.5
                if len(window_df) >= 2:
                    macd_prev = window_df["MACD_hist"].iloc[-2]
                    if latest["MACD_hist"] > macd_prev and latest["MACD_hist"] > 0:
                        score += 0.25
                    elif latest["MACD_hist"] < macd_prev and latest["MACD_hist"] < 0:
                        score -= 0.25
                # 3) RSI regime alignment
                if is_uptrend and latest["RSI"] >= 50:
                    score += 0.5
                elif is_downtrend and latest["RSI"] <= 50:
                    score -= 0.5
                # 4) Pullback entry + VWAP
                recent_lows = window_df["Low"].iloc[-3:]
                recent_highs = window_df["High"].iloc[-3:]
                kc_mid = float(latest.get("KC_mid", latest["EMA_fast"]))
                vwap_ok_up = float(latest.get("VWAP", latest["EMA_fast"])) < float(latest["Close"])
                vwap_ok_dn = float(latest.get("VWAP", latest["EMA_fast"])) > float(latest["Close"])
                if is_uptrend:
                    if (recent_lows.min() <= latest["EMA_fast"]) or (recent_lows.min() <= kc_mid):
                        if latest["Close"] > latest["EMA_fast"] and vwap_ok_up:
                            score += 0.75
                elif is_downtrend:
                    if (recent_highs.max() >= latest["EMA_fast"]) or (recent_highs.max() >= kc_mid):
                        if latest["Close"] < latest["EMA_fast"] and vwap_ok_dn:
                            score -= 0.75
                # 5) Donchian breakout
                if "DC_high" in window_df.columns and "DC_low" in window_df.columns and len(window_df) >= 2:
                    dc_high_prev = window_df["DC_high"].iloc[-2]
                    dc_low_prev = window_df["DC_low"].iloc[-2]
                    if is_uptrend and latest["Close"] > dc_high_prev and vwap_ok_up:
                        score += 0.5
                    elif is_downtrend and latest["Close"] < dc_low_prev and vwap_ok_dn:
                        score -= 0.5
                # 6) ADX soft filter
                if latest["ADX"] > 18:
                    score += 0.25 if is_uptrend else -0.25
                # 7) KC overextension
                kc_upper = float(latest.get("KC_upper", latest.get("BB_upper", latest["Close"])))
                kc_lower = float(latest.get("KC_lower", latest.get("BB_lower", latest["Close"])))
                if is_uptrend and latest["Close"] > kc_upper:
                    score -= 0.5
                elif is_downtrend and latest["Close"] < kc_lower:
                    score -= 0.5
                interval_scores[interval] = score
                max_possible_score += 4.0  # Updated for new scheme

        total_score = sum(interval_scores.values()) if not ready else 1.0
        confidence = (0 if (not ready and max_possible_score == 0) else max(min(abs(total_score)/(max_possible_score if not ready else 1.0)*100,100),0))

        # Relaxed multi-timeframe alignment (allow 1 disagreement)
        base_dir_up = interval_scores.get(base_interval, 0) > 0
        base_dir_down = interval_scores.get(base_interval, 0) < 0
        higher_scores = [interval_scores[i] for i in higher_intervals if i in interval_scores]
        disagree_count = sum(1 for s in higher_scores if (s < 0 if base_dir_up else s > 0))
        timeframe_alignment = (disagree_count <= 1)

        latest_candle = df_base.iloc[i]
        entry = float(latest_candle["Close"])
        raw_atr = float(latest_candle.get("ATR", entry*0.005))
        atr_scale = 0.5 if base_interval == "1m" else 1.0
        atr_value = raw_atr * risk_factor * atr_scale
        # Regime identification
        base_adx = float(latest_candle.get("ADX", 0.0))
        is_trending = base_adx >= 22

        # Common helpers
        atr_den = float(latest_candle.get("ATR", 0.0)) + 1e-9
        ema_fast = float(latest_candle.get("EMA_fast", entry))
        atr_distance = abs(entry - ema_fast) / atr_den

        # Range helpers
        kc_mid = float(latest_candle.get("KC_mid", ema_fast))
        kc_upper = float(latest_candle.get("KC_upper", entry))
        kc_lower = float(latest_candle.get("KC_lower", entry))
        ema_slow_slope = float(latest_candle.get("EMA_slow_slope", 0.0))
        z_kc = float(latest_candle.get("Z_KC", 0.0))

        is_range_trade = False
        scalp_mode = (base_interval == "1m")
        # 1m hours/volume gating
        hours_ok = True
        vol_ok = True
        if scalp_mode:
            try:
                hour = candle_time.hour
                start_hr = int(st.session_state.get("trade_hours_start", 9))
                end_hr = int(st.session_state.get("trade_hours_end", 16))
                if start_hr <= end_hr:
                    hours_ok = (start_hr <= hour <= end_hr)
                else:
                    hours_ok = (hour >= start_hr or hour <= end_hr)
            except Exception:
                hours_ok = True
            if "Volume" in df_base.columns:
                try:
                    vol_med = float(df_base["Volume"].iloc[max(0, i-50):i+1].median())
                    cur_vol = float(latest_candle.get("Volume", vol_med))
                    if vol_med > 0:
                        vol_ok = (cur_vol >= 0.5 * vol_med)
                except Exception:
                    vol_ok = True

        if not ready and is_trending:
            # Trending regime gating
            min_conf = 35 if scalp_mode else 50
            min_adx = 10 if scalp_mode else 15
            max_atr_dist = 2.25 if scalp_mode else 1.75
            if confidence < min_conf or (not timeframe_alignment and not scalp_mode) or base_adx < min_adx or atr_distance > max_atr_dist or not hours_ok or not vol_ok:
                continue
            if total_score > 0:
                # Require breakout or strong bullish candle and ATR regime sanity
                dc_ok = (i > 0 and "DC_high" in df_base.columns and float(df_base["Close"].iloc[i]) > float(df_base["DC_high"].iloc[i-1]))
                body = float(df_base["Close"].iloc[i]) - float(df_base["Open"].iloc[i])
                range_c = max(float(df_base["High"].iloc[i]) - float(df_base["Low"].iloc[i]), 1e-9)
                strong_bull = body > 0 and (body >= 0.5 * range_c) and (float(df_base["Close"].iloc[i]) > float(df_base["EMA_fast"].iloc[i]))
                atr_med = float(df_base["ATR"].iloc[max(0, i-50):i+1].median())
                atr_ratio = (float(df_base["ATR"].iloc[i]) / (atr_med + 1e-9)) if atr_med > 0 else 1.0
                ema200_val = float(df_base["EMA_200"].iloc[i]) if "EMA_200" in df_base.columns else entry
                ema200_ok = ema200_val <= entry
                # 1m allowance: above VWAP and positive EMA_slow_slope
                if "VWAP" in df_base.columns:
                    vwap_up = float(df_base["VWAP"].iloc[i]) < entry
                else:
                    vwap_up = True
                slow_slope_val_up = float(df_base["EMA_slow_slope"].iloc[i]) if "EMA_slow_slope" in df_base.columns else 0.0
                slow_slope_up = slow_slope_val_up > 0
                scalp_ok = scalp_mode and vwap_up and slow_slope_up
                cooldown = 2 if scalp_mode else 3
                if atr_distance > max_atr_dist or i - last_exit_index < cooldown or not ((dc_ok or strong_bull) or scalp_ok) or not (0.6 <= atr_ratio <= 2.5) or not ema200_ok:
                    continue
                trade_type = "Buy"
                stop = entry - atr_value
                tp = entry + risk_reward * atr_value
            elif total_score < 0:
                dc_ok = (i > 0 and "DC_low" in df_base.columns and float(df_base["Close"].iloc[i]) < float(df_base["DC_low"].iloc[i-1]))
                body = float(df_base["Open"].iloc[i]) - float(df_base["Close"].iloc[i])
                range_c = max(float(df_base["High"].iloc[i]) - float(df_base["Low"].iloc[i]), 1e-9)
                strong_bear = body > 0 and (body >= 0.5 * range_c) and (float(df_base["Close"].iloc[i]) < float(df_base["EMA_fast"].iloc[i]))
                atr_med = float(df_base["ATR"].iloc[max(0, i-50):i+1].median())
                atr_ratio = (float(df_base["ATR"].iloc[i]) / (atr_med + 1e-9)) if atr_med > 0 else 1.0
                ema200_val = float(df_base["EMA_200"].iloc[i]) if "EMA_200" in df_base.columns else entry
                ema200_ok = ema200_val >= entry
                if "VWAP" in df_base.columns:
                    vwap_dn = float(df_base["VWAP"].iloc[i]) > entry
                else:
                    vwap_dn = True
                slow_slope_val_dn = float(df_base["EMA_slow_slope"].iloc[i]) if "EMA_slow_slope" in df_base.columns else 0.0
                slow_slope_dn = slow_slope_val_dn < 0
                scalp_ok = scalp_mode and vwap_dn and slow_slope_dn
                cooldown = 2 if scalp_mode else 3
                if atr_distance > max_atr_dist or i - last_exit_index < cooldown or not ((dc_ok or strong_bear) or scalp_ok) or not (0.6 <= atr_ratio <= 2.5) or not ema200_ok:
                    continue
                trade_type = "Sell"
                stop = entry + atr_value
                tp = entry - risk_reward * atr_value
            else:
                continue
        elif not ready:
            # Ranging regime mean-reversion
            flat_slope = abs(ema_slow_slope) < 0.0015
            if hours_ok and vol_ok and (entry < kc_lower) and (float(latest_candle.get("RSI", 50.0)) <= 55) and flat_slope and (z_kc <= -0.8) and (i - last_exit_index >= 2):
                trade_type = "Buy"
                stop = entry - 1.25 * atr_value
                tp = kc_mid
                is_range_trade = True
            elif hours_ok and vol_ok and (entry > kc_upper) and (float(latest_candle.get("RSI", 50.0)) >= 45) and flat_slope and (z_kc >= 0.8) and (i - last_exit_index >= 2):
                trade_type = "Sell"
                stop = entry + 1.25 * atr_value
                tp = kc_mid
                is_range_trade = True
            else:
                continue

        # Simulate forward to TP/SL with breakeven + trailing stop and cooldown
        exit_price = tp
        pnl = None
        breakeven_set = False
        # Scalping: limit hold bars (extended for 2m to allow trades to develop)
        max_hold_bars = 5 if base_interval == "2m" else (3 if base_interval == "1m" else 120)
        j_idx = i
        exit_time = None
        # Slippage per tick (ES = 0.25, adjust for other contracts)
        slippage_ticks = 0.25
        for j in range(i+1, min(len(df_base), i + max_hold_bars)):
            j_idx = j
            high = float(df_base["High"].iloc[j])
            low  = float(df_base["Low"].iloc[j])
            close_j = float(df_base["Close"].iloc[j])
            # Use ATR at j if available; otherwise fall back to initial atr_value
            if "ATR" in df_base.columns:
                atr_cell = df_base["ATR"].iloc[j]
                atr_j = float(atr_cell) if pd.notna(atr_cell) else atr_value
            else:
                atr_j = atr_value

            if trade_type == "Buy":
                if not is_range_trade:
                    # Move stop to breakeven after +1.2R (balanced approach for 2m)
                    if not breakeven_set and high >= entry + 1.2 * atr_value:
                        stop = max(stop, entry + 0.4 * atr_value)  # lock +0.4R
                        breakeven_set = True
                    # Trail stop by 0.8*ATR once breakeven set (tighter trailing)
                    if breakeven_set:
                        stop = max(stop, close_j - 0.8 * atr_j)
                if high >= tp:
                    exit_price = tp - slippage_ticks; pnl = exit_price - entry; exit_time = df_base.index[j]; break
                elif low <= stop:
                    exit_price = stop + slippage_ticks; pnl = exit_price - entry; exit_time = df_base.index[j]; break
            else:
                if not is_range_trade:
                    if not breakeven_set and low <= entry - 1.2 * atr_value:
                        stop = min(stop, entry - 0.4 * atr_value)
                        breakeven_set = True
                    if breakeven_set:
                        stop = min(stop, close_j + 0.8 * atr_j)
                if low <= tp:
                    exit_price = tp + slippage_ticks; pnl = entry - exit_price; exit_time = df_base.index[j]; break
                elif high >= stop:
                    exit_price = stop - slippage_ticks; pnl = entry - exit_price; exit_time = df_base.index[j]; break

        if pnl is None:
            # Exit at the last bar of the holding window to avoid blocking further trades
            fallback_idx = j_idx if j_idx > i else min(i+1, len(df_base)-1)
            fallback_close = float(df_base["Close"].iloc[fallback_idx])
            exit_price = fallback_close
            pnl = fallback_close - entry if trade_type == "Buy" else entry - fallback_close
            last_exit_index = fallback_idx
            exit_time = df_base.index[fallback_idx]
        else:
            last_exit_index = j_idx
        trades.append({
            "Datetime": candle_time,
            "Type": trade_type,
            "Entry": round(entry, 2),
            "Stop": round(stop, 2),
            "TP": round(tp, 2),
            "Exit": round(exit_price, 2),
            "ExitTime": exit_time,
            "PnL": round(pnl, 2),
            "Win": pnl > 0,
            "Loss": pnl <= 0,
            "Score": round(total_score, 2),
            "Confidence": round(confidence, 2)
        })

    return trades, df_base

# --- Initialize session state for open positions and authentication ---
if "open_positions" not in st.session_state:
    st.session_state.open_positions = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- Passcode Protection ---
PASSCODE = "000"

if not st.session_state.authenticated:
    # Landing Page
    st.markdown("""
    <div style='text-align: center; padding: 100px 20px;'>
        <h1 style='color: #1f77b4; font-size: 72px; font-weight: bold; letter-spacing: 5px; margin-bottom: 10px;'>BALDER</h1>
        <p style='color: #888; font-size: 24px; margin-bottom: 50px;'>Trade Advisor</p>
        <div style='max-width: 400px; margin: 0 auto; padding: 40px; background-color: rgba(28, 31, 38, 0.5); border-radius: 10px; border: 2px solid #1f77b4;'>
            <p style='color: #ddd; font-size: 18px; margin-bottom: 20px;'>Enter Passcode to Continue</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        passcode_input = st.text_input("", type="password", key="passcode_input", placeholder="Enter passcode", label_visibility="collapsed")
        
        if st.button("🔓 Unlock", use_container_width=True, type="primary"):
            if passcode_input == PASSCODE:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect passcode. Please try again.")
    
    st.stop()  # Stop execution here if not authenticated

# --- Sidebar Title ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px 0 10px 0; margin-bottom: 20px; border-bottom: 2px solid #1f77b4;'>
    <h1 style='color: #1f77b4; margin: 0; padding: 0; font-size: 36px; font-weight: bold; letter-spacing: 3px;'>BALDER</h1>
    <p style='color: #888; margin: 5px 0 0 0; font-size: 12px;'>Trade Advisor</p>
</div>
""", unsafe_allow_html=True)

# --- Navigation ---
page = st.sidebar.radio("Navigation", ["Trade Signals", "Backtest", "Key Terms"])

# --- Trade Signals Page (Futures Only) ---
if page == "Trade Signals":
    ticker, label = "", ""
    futures_labels = {
        "ES=F":"S&P 500 E-Mini Futures",
        "NQ=F":"NASDAQ 100 E-Mini Futures",
        "YM=F":"Dow Jones Futures",
        "CL=F":"Crude Oil Futures",
        "GC=F":"Gold Futures"
    }
    popular_futures = list(futures_labels.keys()) + ["OTHER"]
    futures_names = [f"{futures_labels.get(sym, sym)} ({sym})" if sym != "OTHER" else "Other" for sym in popular_futures]
    selected_name = st.sidebar.selectbox("Select Future", futures_names, index=popular_futures.index(st.session_state.futures_symbol) if st.session_state.futures_symbol in popular_futures else 0)
    selected_symbol = popular_futures[futures_names.index(selected_name)]
    if selected_symbol == "OTHER":
        ticker = st.sidebar.text_input("Enter Futures Ticker", value=st.session_state.custom_future).upper()
        label = "Custom Futures Contract"
    else:
        ticker = selected_symbol
        label = futures_labels.get(ticker, ticker)

    interval_options = ["1m","2m","5m","15m","1h"]
    interval = st.sidebar.selectbox("Interval", interval_options, index=interval_options.index(st.session_state.interval))
    refresh_sec = st.sidebar.slider("Auto Refresh (seconds)", 5, 120, st.session_state.refresh_sec)
    st.session_state.refresh_sec = refresh_sec  # keep autorefresh in sync
    # Enable auto-refresh only on the Trade Signals page
    st_autorefresh(interval=refresh_sec*1000, key="auto_refresh")
    risk_factor = st.sidebar.slider("Risk Sensitivity (ATR multiplier)", 0.5, 20.0, st.session_state.risk_factor, 0.1)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 5.0, st.session_state.risk_reward, 0.1)

    result, color, df_base = get_trade_signal_with_targets(ticker, interval, risk_factor, risk_reward)

    st.subheader(f"{label} | {interval} | {datetime.now().strftime('%H:%M:%S')}")
    st.markdown(f"**Price:** ${result['Close']}")
    st.markdown(f"**Signal:** <span style='color:{color}; font-size:20px;'>{result['Signal']}</span>", unsafe_allow_html=True)
    st.markdown(f"Score: {result['Score']} | Confidence: {result['Confidence']}% | ATR: {result['ATR']} | RSI: {result['RSI']}")

    if result["Take Profit"] is not None:
        st.success(f"🎯 Take Profit: **${result['Take Profit']}**")
    if result["Stop Loss"] is not None:
        st.error(f"⚠️ Stop Loss: **${result['Stop Loss']}**")

    # Open Position Button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if "Buy" in result["Signal"] or "Sell" in result["Signal"]:
            if st.button("📈 Open Position", key="open_position_btn", type="primary"):
                position_type = "Buy" if "Buy" in result["Signal"] else "Sell"
                # Get contract multiplier for this futures contract
                contract_multipliers = {
                    "ES=F": 50,      # S&P 500 E-Mini: $50 per point
                    "NQ=F": 20,      # NASDAQ 100 E-Mini: $20 per point
                    "YM=F": 5,       # Dow Jones: $5 per point
                    "CL=F": 1000,    # Crude Oil: $1000 per point
                    "GC=F": 100      # Gold: $100 per point (per troy ounce)
                }
                multiplier = contract_multipliers.get(ticker, 50)  # Default to 50 if unknown
                
                position = {
                    "Ticker": ticker,
                    "Label": label,
                    "Type": position_type,
                    "Entry Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Entry Price": result["Close"],
                    "Current Price": result["Close"],
                    "Stop Loss": result["Stop Loss"],
                    "Take Profit": result["Take Profit"],
                    "Contracts": 1,
                    "Multiplier": multiplier,
                    "PnL": 0.0,
                    "Total Profit": 0.0,
                    "Status": "Open"
                }
                st.session_state.open_positions.append(position)
                st.success(f"✅ Opened {position_type} position at ${result['Close']}")
                st.rerun()
    with col2:
        if st.session_state.open_positions:
            if st.button("🔄 Update Prices", key="update_prices_btn"):
                st.rerun()
    with col3:
        if st.session_state.open_positions:
            if st.button("🗑️ Clear All", key="clear_positions_btn"):
                st.session_state.open_positions = []
                st.success("All positions cleared")
                st.rerun()

    # Display Open Positions Table
    if st.session_state.open_positions:
        st.markdown("---")
        st.subheader("📊 Open Positions")
        
        # Contract input fields and update current prices and PnL for all positions
        st.markdown("#### Adjust Contracts")
        contract_cols = st.columns(len(st.session_state.open_positions) if len(st.session_state.open_positions) <= 4 else 4)
        for idx, pos in enumerate(st.session_state.open_positions):
            # Ensure backward compatibility - add missing fields to old positions
            if "Contracts" not in pos:
                pos["Contracts"] = 1
            if "Multiplier" not in pos:
                contract_multipliers = {
                    "ES=F": 50, "NQ=F": 20, "YM=F": 5, "CL=F": 1000, "GC=F": 100
                }
                pos["Multiplier"] = contract_multipliers.get(pos["Ticker"], 50)
            
            col_idx = idx % 4
            with contract_cols[col_idx]:
                new_contracts = st.number_input(
                    f"#{idx+1} Contracts",
                    min_value=1,
                    max_value=100,
                    value=pos["Contracts"],
                    step=1,
                    key=f"contracts_{idx}"
                )
                pos["Contracts"] = new_contracts
            
            if pos["Status"] == "Open":
                pos["Current Price"] = result["Close"]
                if pos["Type"] == "Buy":
                    pos["PnL"] = round(result["Close"] - pos["Entry Price"], 2)
                    # Check if TP or SL hit
                    if pos["Take Profit"] and result["Close"] >= pos["Take Profit"]:
                        pos["Status"] = "Closed (TP Hit)"
                        pos["PnL"] = round(pos["Take Profit"] - pos["Entry Price"], 2)
                    elif pos["Stop Loss"] and result["Close"] <= pos["Stop Loss"]:
                        pos["Status"] = "Closed (SL Hit)"
                        pos["PnL"] = round(pos["Stop Loss"] - pos["Entry Price"], 2)
                else:  # Sell
                    pos["PnL"] = round(pos["Entry Price"] - result["Close"], 2)
                    # Check if TP or SL hit
                    if pos["Take Profit"] and result["Close"] <= pos["Take Profit"]:
                        pos["Status"] = "Closed (TP Hit)"
                        pos["PnL"] = round(pos["Entry Price"] - pos["Take Profit"], 2)
                    elif pos["Stop Loss"] and result["Close"] >= pos["Stop Loss"]:
                        pos["Status"] = "Closed (SL Hit)"
                        pos["PnL"] = round(pos["Entry Price"] - pos["Stop Loss"], 2)
            
            # Calculate total profit: PnL (points) × Multiplier × Contracts
            pos["Total Profit"] = round(pos["PnL"] * pos["Multiplier"] * pos["Contracts"], 2)
        
        # Create DataFrame for display
        positions_df = pd.DataFrame(st.session_state.open_positions)
        
        # Add color coding for PnL and Total Profit
        def color_pnl(val):
            if val > 0:
                return 'background-color: #90EE90'
            elif val < 0:
                return 'background-color: #FFB6C1'
            return ''
        
        styled_df = positions_df.style.applymap(color_pnl, subset=['PnL', 'Total Profit'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary stats
        total_profit = sum(pos["Total Profit"] for pos in st.session_state.open_positions)
        total_pnl_points = sum(pos["PnL"] for pos in st.session_state.open_positions)
        open_count = sum(1 for pos in st.session_state.open_positions if pos["Status"] == "Open")
        closed_count = len(st.session_state.open_positions) - open_count
        wins = sum(1 for pos in st.session_state.open_positions if "Closed" in pos["Status"] and pos["PnL"] > 0)
        losses = sum(1 for pos in st.session_state.open_positions if "Closed" in pos["Status"] and pos["PnL"] <= 0)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            profit_color = "green" if total_profit > 0 else "red" if total_profit < 0 else "gray"
            st.markdown(f"**Total Profit:** <span style='color:{profit_color}; font-size:18px;'>${total_profit:,.2f}</span>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"**Open:** {open_count}")
        with col_c:
            st.markdown(f"**Closed:** {closed_count}")
        with col_d:
            if closed_count > 0:
                win_rate = (wins / closed_count * 100)
                st.markdown(f"**Win Rate:** {win_rate:.1f}%")
        
        # Close individual positions
        st.markdown("---")
        st.subheader("Close Positions")
        for idx, pos in enumerate(st.session_state.open_positions):
            if pos["Status"] == "Open":
                col_x, col_y = st.columns([4, 1])
                with col_x:
                    st.write(f"{pos['Type']} {pos['Ticker']} @ ${pos['Entry Price']} | {pos['Contracts']} contract(s) | Total Profit: ${pos['Total Profit']:,.2f}")
                with col_y:
                    if st.button("Close", key=f"close_pos_{idx}"):
                        pos["Status"] = "Closed (Manual)"
                        st.success(f"Closed position with Total Profit: ${pos['Total Profit']:,.2f}")
                        st.rerun()

    # MTI Table
    if result.get("MTI Scores"):
        st.subheader("📊 Multi-Timeframe Indicator (MTI) Scores")
        intervals = list(result["MTI Scores"].keys())
        scores = result["MTI Scores"]
        cols = st.columns(len(intervals))
        for i, interval_name in enumerate(intervals):
            score_val = scores[interval_name]
            if score_val > 0:
                emoji, color_code = "🚀", "green"
            elif score_val < 0:
                emoji, color_code = "🔻", "red"
            else:
                emoji, color_code = "⚪", "gray"
            with cols[i]:
                st.markdown(
                    f"<div style='text-align:center; font-weight:bold; color:{color_code}; font-size:18px; padding:10px; margin-bottom:20px;'>{interval_name}<br>{emoji} {score_val}</div>",
                    unsafe_allow_html=True
                )

    # Live chart
    if not df_base.empty:
        df_plot = df_base[["Close","EMA_fast","EMA_slow"]].dropna()
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close"))
        fig_live.add_trace(go.Scatter(x=df_plot.index, y=df_plot["EMA_fast"], mode="lines", name="EMA Fast"))
        fig_live.add_trace(go.Scatter(x=df_plot.index, y=df_plot["EMA_slow"], mode="lines", name="EMA Slow"))
        if result["Confidence"] >= 50:
            if "Buy" in result["Signal"]:
                fig_live.add_trace(go.Scatter(x=[df_plot.index[-1]], y=[df_plot["Close"].iloc[-1]], mode="markers",
                                              name="BUY", marker=dict(color="green", size=12, symbol="triangle-up")))
            elif "Sell" in result["Signal"]:
                fig_live.add_trace(go.Scatter(x=[df_plot.index[-1]], y=[df_plot["Close"].iloc[-1]], mode="markers",
                                              name="SELL", marker=dict(color="red", size=12, symbol="triangle-down")))
        fig_live.update_layout(title=f"{label} Price & EMA Lines", xaxis_title="Datetime", yaxis_title="Price", height=400)
        st.plotly_chart(fig_live, use_container_width=True)

# --- Backtest Page (Futures Only) ---
elif page == "Backtest":
    ticker, label = "", ""
    futures_labels = {
        "ES=F":"S&P 500 E-Mini Futures",
        "NQ=F":"NASDAQ 100 E-Mini Futures",
        "YM=F":"Dow Jones Futures",
        "CL=F":"Crude Oil Futures",
        "GC=F":"Gold Futures"
    }
    popular_futures = list(futures_labels.keys()) + ["OTHER"]
    futures_names = [f"{futures_labels.get(sym, sym)} ({sym})" if sym != "OTHER" else "Other" for sym in popular_futures]
    selected_name = st.sidebar.selectbox("Select Future", futures_names, index=popular_futures.index(st.session_state.futures_symbol) if st.session_state.futures_symbol in popular_futures else 0)
    selected_symbol = popular_futures[futures_names.index(selected_name)]
    if selected_symbol == "OTHER":
        ticker = st.sidebar.text_input("Enter Futures Ticker", value=st.session_state.custom_future).upper()
        label = "Custom Futures Contract"
    else:
        ticker = selected_symbol
        label = futures_labels.get(ticker, ticker)

    interval_options = ["1m","2m","5m","15m","1h"]
    interval = st.sidebar.selectbox("Interval", interval_options, index=interval_options.index(st.session_state.interval))
    # Date controls - get previous business day
    from datetime import date, timedelta
    today = date.today()
    # Find previous business day (skip weekends)
    days_back = 1
    default_date = today - timedelta(days=days_back)
    while default_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        days_back += 1
        default_date = today - timedelta(days=days_back)
    
    bt_date = st.sidebar.date_input("Backtest Date", value=default_date)
    bt_start = bt_date
    bt_end = bt_date
    risk_factor = st.sidebar.slider("Risk Sensitivity (ATR multiplier)", 0.5, 20.0, st.session_state.risk_factor, 0.1)
    risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 5.0, st.session_state.risk_reward, 0.1)

    trades, df_back = backtest_signals(ticker, interval, risk_factor, risk_reward, bt_start, bt_end)
    if df_back.empty:
        st.warning("No historical data found.")
    else:
        if trades:
            trades_df = pd.DataFrame(trades)
            st.subheader("📋 Backtest Trades")
            st.dataframe(trades_df[["Datetime","Type","Entry","Stop","TP","Exit","PnL","Win","Loss"]])

            total_pnl = float(trades_df["PnL"].sum())
            total_wins = int(trades_df["Win"].sum())
            win_rate = (total_wins / len(trades_df) * 100) if len(trades_df) > 0 else 0.0
            
            st.markdown(f"**Win Rate:** {win_rate:.1f}% | **Net PnL:** ${total_pnl:.2f}")
        else:
            st.info("No trades generated in this period.")

        # Chart
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_back.index, y=df_back["Close"], mode="lines", name="Close"))
        for trade in trades:
            color = "green" if trade["Type"] == "Buy" else "red"
            entry_symbol = "triangle-up" if trade["Type"] == "Buy" else "triangle-down"
            exit_symbol = "x"
            # Entry marker
            fig_bt.add_trace(go.Scatter(
                x=[trade["Datetime"]], y=[trade["Entry"]],
                mode="markers",
                marker=dict(color=color, size=10, symbol=entry_symbol),
                name=f"{trade['Type']} ENTRY",
                showlegend=False
            ))
            # Exit marker
            fig_bt.add_trace(go.Scatter(
                x=[trade.get("ExitTime", trade["Datetime"])], y=[trade["Exit"]],
                mode="markers",
                marker=dict(color=color, size=9, symbol=exit_symbol, line=dict(width=1, color=color)),
                name="EXIT",
                showlegend=False
            ))
            # Arrow annotation from entry to exit
            try:
                fig_bt.add_annotation(
                    x=trade.get("ExitTime", trade["Datetime"]),
                    y=trade["Exit"],
                    ax=trade["Datetime"],
                    ay=trade["Entry"],
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor=color, opacity=0.7
                )
            except Exception:
                pass
        fig_bt.update_layout(title=f"{label} Backtest", xaxis_title="Datetime", yaxis_title="Price", height=500)
        st.plotly_chart(fig_bt, use_container_width=True)

# --- Key Terms ---
elif page == "Key Terms":
    show_explanations()
