"""
Trading strategy logic for scalping futures
"""
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any
from .indicators import add_indicators, is_us_rth


def get_higher_row_at_or_before(df_hi: pd.DataFrame, ts):
    """Get nearest-at-or-before row from higher timeframe"""
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


def decide_scalp_entry(
    df_base: pd.DataFrame,
    idx: int,
    df_5m: Optional[pd.DataFrame],
    risk_factor: float,
    base_interval: str,
    risk_reward: float
) -> Optional[Dict[str, Any]]:
    """
    Decide if there's a scalp entry on a single bar.
    Returns dict with type, entry, stop, tp, is_mr or None.
    """
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
    if base_interval in ("1m", "2m") and not is_us_rth(df_base.index[idx]):
        return None
    
    # Higher timeframe bias (5m EMA20 slope)
    bias_up = True
    bias_dn = True
    if df_5m is not None and not df_5m.empty:
        hi_row = get_higher_row_at_or_before(df_5m, df_base.index[idx])
        if hi_row is not None:
            ema20 = float(hi_row.get("EMA20", hi_row.get("EMA_slow", row.get("EMA_slow", 0.0))))
            try:
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
    
    # Pullback context
    start = max(0, idx-3)
    recent_lows = df_base["Low"].iloc[start:idx+1]
    recent_highs = df_base["High"].iloc[start:idx+1]
    
    # Momentum shape
    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row.get("Open", close))
    rng = max(high - low, 1e-9)
    pos = (close - low) / rng
    pos_short = (high - close) / rng
    
    # ATR sizing
    atr = float(row.get("ATR", max(rng, 1e-9)))
    atr_scale = 0.5 if base_interval == "1m" else (0.8 if base_interval == "2m" else 1.0)
    atr_eff = max(1e-9, atr * risk_factor * atr_scale)
    stop_mult = 0.6 if base_interval == "1m" else (0.7 if base_interval == "2m" else 0.6)
    
    # === 2m Pure Breakout Strategy ===
    if base_interval == "2m":
        lookback = min(40, idx)
        if lookback >= 20:
            range_high = float(df_base["High"].iloc[max(0, idx-lookback):idx].max())
            range_low = float(df_base["Low"].iloc[max(0, idx-lookback):idx].min())
            
            range_size = range_high - range_low
            range_is_wide = range_size >= (1.5 * atr)
            
            near_high = (close >= range_high - 0.5 * atr)
            near_low = (close <= range_low + 0.5 * atr)
        else:
            return None
        
        if not range_is_wide:
            return None
        
        # Volume confirmation
        vol_surge = False
        if "Volume" in df_base.columns and idx >= 50:
            vol_med = float(df_base["Volume"].iloc[max(0, idx-50):idx].median())
            cur_vol = float(row.get("Volume", vol_med))
            if vol_med > 0:
                vol_surge = (cur_vol >= 1.5 * vol_med)
        else:
            vol_surge = True
        
        # Trend filter
        ema_spread = abs(ema34 - ema89)
        strong_trend_up = (ema34 > ema89) and (ema_spread >= 0.3 * atr)
        strong_trend_down = (ema34 < ema89) and (ema_spread >= 0.3 * atr)
        
        # ADX filter
        adx_val = float(row.get("ADX", 0.0))
        is_trending = (adx_val >= 20)
        
        # LONG: Strong breakout above high
        breakout_up = (close > range_high) and (high > range_high)
        strong_close = (pos >= 0.7)
        momentum_up = (close > float(prev.get("Close", close)))
        body_size = abs(close - open_)
        strong_body = (body_size >= 0.5 * rng)
        breakout_distance = (close - range_high)
        clear_break = (breakout_distance >= 0.2 * atr)
        
        # Multi-bar momentum
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
        # === 1m logic ===
        long_trend = (ema20 > ema50) and (close > vwap) and bias_up
        long_pullback = (recent_lows.min() <= ema20)
        long_reentry = (close > ema20)
        long_momo = (pos >= 0.75) or (close > float(prev.get("High", close)))
        long_rsi_ok = (rsi2_prev <= 10)
        long_allow = long_trend and long_pullback and long_reentry and long_momo and long_rsi_ok
        long_mr = (close >= ema200) and (rsi2_prev <= 5) and (close < kc_lower) and (z_vwap <= -1.0)
    
    if long_allow or long_mr:
        entry = close
        
        # Breakout-specific stop for 2m
        if base_interval == "2m" and long_allow:
            stop = range_high - (0.5 * atr_eff)
            tp = entry + risk_reward * (entry - stop)
        else:
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
        
        return {"type": "Buy", "entry": entry, "stop": stop, "tp": tp, "is_mr": long_mr}
    
    # === SHORT logic ===
    if base_interval == "2m":
        breakout_down = (close < range_low) and (low < range_low)
        strong_close_down = (pos_short >= 0.7)
        momentum_down = (close < float(prev.get("Close", close)))
        breakdown_distance = (range_low - close)
        clear_break_down = (breakdown_distance >= 0.2 * atr)
        
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
        # 1m logic
        short_trend = (ema20 < ema50) and (close < vwap) and bias_dn
        short_pullback = (recent_highs.max() >= ema20)
        short_reentry = (close < ema20)
        short_momo = (pos_short >= 0.75) or (close < float(prev.get("Low", close)))
        short_rsi_ok = (rsi2_prev >= 90)
        short_allow = short_trend and short_pullback and short_reentry and short_momo and short_rsi_ok
        short_mr = (close <= ema200) and (rsi2_prev >= 95) and (close > kc_upper) and (z_vwap >= 1.0)
    
    if short_allow or short_mr:
        entry = close
        
        # Breakout-specific stop for 2m
        if base_interval == "2m" and short_allow:
            stop = range_low + (0.5 * atr_eff)
            tp = entry - risk_reward * (stop - entry)
        else:
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
        
        return {"type": "Sell", "entry": entry, "stop": stop, "tp": tp, "is_mr": short_mr}
    
    return None


def _fetch_market_data_uncached(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Internal function to fetch market data without caching"""
    try:
        print(f"📊 Fetching fresh data for {ticker}, interval={interval}, period={period}")
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            print(f"⚠️  No data returned for {ticker}")
            return pd.DataFrame()
        
        print(f"✅ Got {len(df)} rows of data")
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.astype(float, errors='ignore').dropna(subset=["Close"], how="any")
        print(f"📈 Adding indicators...")
        df = add_indicators(df)
        print(f"✅ Data ready with {len(df)} rows")
        
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()


def fetch_market_data(ticker: str, interval: str, period: str = "3d") -> pd.DataFrame:
    """
    Fetch and prepare market data with indicators.
    Always pulls fresh data from yfinance.
    """
    return _fetch_market_data_uncached(ticker, interval, period)

