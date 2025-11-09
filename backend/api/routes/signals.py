"""
Trade signals API endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd

from core.strategy import fetch_market_data, decide_scalp_entry
from core.indicators import is_us_rth

router = APIRouter()


class SignalRequest(BaseModel):
    ticker: str
    interval: str = "2m"
    risk_factor: float = 1.0
    risk_reward: float = 2.0


class SignalResponse(BaseModel):
    ticker: str
    interval: str
    signal: Optional[str]  # "Buy", "Sell", or None
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    latest_price: float
    timestamp: str
    indicators: Dict[str, Any]
    message: Optional[str] = None


@router.post("/current", response_model=SignalResponse)
async def get_current_signal(request: SignalRequest):
    """
    Get current trade signal for a ticker
    """
    try:
        print(f"\n🎯 Signal request: {request.ticker} @ {request.interval}")
        
        # Fetch data
        period_map = {"1m": "3d", "2m": "3d", "5m": "3d", "15m": "7d", "1h": "60d"}
        period = period_map.get(request.interval, "3d")
        
        df_base = fetch_market_data(request.ticker, request.interval, period)
        print(f"   Data fetched: {len(df_base)} rows")
        
        if df_base.empty or len(df_base) < 20:
            return SignalResponse(
                ticker=request.ticker,
                interval=request.interval,
                signal=None,
                entry=None,
                stop_loss=None,
                take_profit=None,
                confidence=0.0,
                latest_price=0.0,
                timestamp="",
                indicators={},
                message="Insufficient data"
            )
        
        # Fetch higher timeframe for bias
        df_5m = None
        if request.interval in ["1m", "2m"]:
            df_5m = fetch_market_data(request.ticker, "5m", "3d")
        
        # Get signal on latest bar
        idx = len(df_base) - 1
        latest = df_base.iloc[-1]
        
        decision = decide_scalp_entry(
            df_base=df_base,
            idx=idx,
            df_5m=df_5m,
            risk_factor=request.risk_factor,
            base_interval=request.interval,
            risk_reward=request.risk_reward
        )
        
        # Calculate confidence based on trend alignment
        ema_fast = float(latest.get("EMA_fast", 0))
        ema_slow = float(latest.get("EMA_slow", 0))
        adx = float(latest.get("ADX", 0))
        
        confidence = 50.0  # Base confidence
        if decision:
            if decision["type"] == "Buy" and ema_fast > ema_slow:
                confidence += 20
            elif decision["type"] == "Sell" and ema_fast < ema_slow:
                confidence += 20
            
            if adx > 25:
                confidence += 15
            elif adx > 20:
                confidence += 10
        
        confidence = min(confidence, 95.0)
        
        return SignalResponse(
            ticker=request.ticker,
            interval=request.interval,
            signal=decision["type"] if decision else None,
            entry=decision["entry"] if decision else None,
            stop_loss=decision["stop"] if decision else None,
            take_profit=decision["tp"] if decision else None,
            confidence=confidence,
            latest_price=float(latest["Close"]),
            timestamp=str(df_base.index[-1]),
            indicators={
                "EMA_fast": float(latest.get("EMA_fast", 0)),
                "EMA_slow": float(latest.get("EMA_slow", 0)),
                "RSI": float(latest.get("RSI", 50)),
                "ATR": float(latest.get("ATR", 0)),
                "ADX": float(latest.get("ADX", 0)),
                "MACD_hist": float(latest.get("MACD_hist", 0)),
                "Volume": float(latest.get("Volume", 0))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical")
async def get_historical_signals(
    ticker: str = Query(..., description="Ticker symbol"),
    interval: str = Query("2m", description="Time interval"),
    period: str = Query("3d", description="Historical period"),
    risk_factor: float = Query(1.0, description="Risk factor"),
    risk_reward: float = Query(2.0, description="Risk/reward ratio")
):
    """
    Get historical signals for charting
    """
    try:
        df = fetch_market_data(ticker, interval, period)
        
        if df.empty:
            return {"error": "No data available"}
        
        # Return OHLCV data with indicators for charting
        data = []
        for idx, row in df.iterrows():
            data.append({
                "timestamp": str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
                "ema_fast": float(row.get("EMA_fast", 0)),
                "ema_slow": float(row.get("EMA_slow", 0)),
                "rsi": float(row.get("RSI", 50)),
                "atr": float(row.get("ATR", 0))
            })
        
        return {"ticker": ticker, "interval": interval, "data": data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

