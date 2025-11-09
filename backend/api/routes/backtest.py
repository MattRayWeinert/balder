"""
Backtest API endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime, timedelta
import pandas as pd

from core.strategy import fetch_market_data, decide_scalp_entry
from core.indicators import is_us_rth

router = APIRouter()


class BacktestRequest(BaseModel):
    ticker: str
    interval: str = "2m"
    test_date: str  # YYYY-MM-DD format
    risk_factor: float = 1.0
    risk_reward: float = 2.0


class TradeResult(BaseModel):
    entry_time: str
    exit_time: str
    type: str  # "Buy" or "Sell"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    win: bool


class BacktestResponse(BaseModel):
    ticker: str
    interval: str
    test_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    trades: List[TradeResult]


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run backtest for a specific date
    """
    try:
        # Parse date
        test_date = datetime.strptime(request.test_date, "%Y-%m-%d").date()
        
        # Fetch extended data window for indicator calculation
        start_date = test_date - timedelta(days=7)
        end_date = test_date + timedelta(days=1)
        
        df_base = fetch_market_data(
            request.ticker,
            request.interval,
            period="7d"  # Get enough data for indicators
        )
        
        if df_base.empty:
            return BacktestResponse(
                ticker=request.ticker,
                interval=request.interval,
                test_date=request.test_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                trades=[]
            )
        
        # Filter to test date and RTH after indicators are computed
        df_base.index = pd.to_datetime(df_base.index)
        if df_base.index.tz is None:
            df_base.index = df_base.index.tz_localize('UTC')
        df_base.index = df_base.index.tz_convert('America/New_York')
        
        # Filter to specific date
        df_test = df_base[df_base.index.date == test_date]
        
        # Filter to RTH (9:30-16:00 ET)
        if request.interval in ["1m", "2m", "5m", "15m"]:
            df_test = df_test.between_time('09:30', '16:00')
        
        if df_test.empty:
            return BacktestResponse(
                ticker=request.ticker,
                interval=request.interval,
                test_date=request.test_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                trades=[]
            )
        
        # Get higher timeframe for bias
        df_5m = None
        if request.interval in ["1m", "2m"]:
            df_5m = fetch_market_data(request.ticker, "5m", "7d")
        
        # Walk through bars and simulate trades
        trades = []
        in_trade = False
        trade_entry = None
        trade_stop = None
        trade_tp = None
        trade_type = None
        entry_idx = None
        
        # Max hold bars
        max_hold_bars = {
            "1m": 3,
            "2m": 5,
            "5m": 8,
            "15m": 10,
            "1h": 15
        }
        max_bars = max_hold_bars.get(request.interval, 5)
        
        for i in range(len(df_test)):
            current_bar = df_test.iloc[i]
            current_time = df_test.index[i]
            
            if not in_trade:
                # Look for entry using base df full index
                base_idx = df_base.index.get_loc(current_time)
                decision = decide_scalp_entry(
                    df_base=df_base,
                    idx=base_idx,
                    df_5m=df_5m,
                    risk_factor=request.risk_factor,
                    base_interval=request.interval,
                    risk_reward=request.risk_reward
                )
                
                if decision:
                    in_trade = True
                    trade_type = decision["type"]
                    trade_entry = decision["entry"]
                    trade_stop = decision["stop"]
                    trade_tp = decision["tp"]
                    entry_idx = i
            
            else:
                # Check for exit
                high = float(current_bar["High"])
                low = float(current_bar["Low"])
                close = float(current_bar["Close"])
                
                hit_tp = False
                hit_sl = False
                bars_held = i - entry_idx
                
                if trade_type == "Buy":
                    if high >= trade_tp:
                        hit_tp = True
                        exit_price = trade_tp
                    elif low <= trade_stop:
                        hit_sl = True
                        exit_price = trade_stop
                    elif bars_held >= max_bars:
                        exit_price = close
                else:  # Sell
                    if low <= trade_tp:
                        hit_tp = True
                        exit_price = trade_tp
                    elif high >= trade_stop:
                        hit_sl = True
                        exit_price = trade_stop
                    elif bars_held >= max_bars:
                        exit_price = close
                
                # Exit if conditions met
                if hit_tp or hit_sl or bars_held >= max_bars:
                    if trade_type == "Buy":
                        pnl = exit_price - trade_entry
                    else:
                        pnl = trade_entry - exit_price
                    
                    pnl_percent = (pnl / trade_entry) * 100
                    
                    trades.append(TradeResult(
                        entry_time=str(df_test.index[entry_idx]),
                        exit_time=str(current_time),
                        type=trade_type,
                        entry_price=round(trade_entry, 2),
                        exit_price=round(exit_price, 2),
                        stop_loss=round(trade_stop, 2),
                        take_profit=round(trade_tp, 2),
                        pnl=round(pnl, 2),
                        pnl_percent=round(pnl_percent, 2),
                        win=(pnl > 0)
                    ))
                    
                    in_trade = False
                    trade_entry = None
                    trade_stop = None
                    trade_tp = None
                    trade_type = None
                    entry_idx = None
        
        # Calculate statistics
        winning_trades = sum(1 for t in trades if t.win)
        losing_trades = len(trades) - winning_trades
        win_rate = (winning_trades / len(trades) * 100) if trades else 0.0
        total_pnl = sum(t.pnl for t in trades)
        
        return BacktestResponse(
            ticker=request.ticker,
            interval=request.interval,
            test_date=request.test_date,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            trades=trades
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

