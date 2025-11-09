"""
Position tracking API endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd

from core.strategy import fetch_market_data

router = APIRouter()

# In-memory storage for positions (use Redis/DB in production)
positions_store: Dict[str, dict] = {}


class PositionCreate(BaseModel):
    ticker: str
    interval: str
    type: str  # "Buy" or "Sell"
    entry_price: float
    stop_loss: float
    take_profit: float
    contracts: int = 1
    multiplier: float = 50.0


class PositionUpdate(BaseModel):
    position_id: str
    current_price: float


class Position(BaseModel):
    position_id: str
    ticker: str
    interval: str
    type: str
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    contracts: int
    multiplier: float
    pnl: float
    pnl_percent: float
    total_profit: float
    status: str  # "open", "tp_hit", "sl_hit"
    entry_time: str
    updated_time: str


@router.post("/create", response_model=Position)
async def create_position(position: PositionCreate):
    """
    Create a new position to track
    """
    try:
        position_id = f"{position.ticker}_{int(datetime.now().timestamp())}"
        
        positions_store[position_id] = {
            "ticker": position.ticker,
            "interval": position.interval,
            "type": position.type,
            "entry_price": position.entry_price,
            "current_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "contracts": position.contracts,
            "multiplier": position.multiplier,
            "status": "open",
            "entry_time": datetime.now().isoformat(),
            "updated_time": datetime.now().isoformat()
        }
        
        pos = positions_store[position_id]
        
        return Position(
            position_id=position_id,
            ticker=pos["ticker"],
            interval=pos["interval"],
            type=pos["type"],
            entry_price=pos["entry_price"],
            current_price=pos["current_price"],
            stop_loss=pos["stop_loss"],
            take_profit=pos["take_profit"],
            contracts=pos["contracts"],
            multiplier=pos["multiplier"],
            pnl=0.0,
            pnl_percent=0.0,
            total_profit=0.0,
            status=pos["status"],
            entry_time=pos["entry_time"],
            updated_time=pos["updated_time"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[Position])
async def list_positions():
    """
    Get all tracked positions
    """
    try:
        result = []
        
        for position_id, pos in positions_store.items():
            # Calculate PnL
            if pos["type"] == "Buy":
                pnl = pos["current_price"] - pos["entry_price"]
            else:
                pnl = pos["entry_price"] - pos["current_price"]
            
            pnl_percent = (pnl / pos["entry_price"]) * 100
            total_profit = pnl * pos["multiplier"] * pos["contracts"]
            
            result.append(Position(
                position_id=position_id,
                ticker=pos["ticker"],
                interval=pos["interval"],
                type=pos["type"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                stop_loss=pos["stop_loss"],
                take_profit=pos["take_profit"],
                contracts=pos["contracts"],
                multiplier=pos["multiplier"],
                pnl=round(pnl, 2),
                pnl_percent=round(pnl_percent, 2),
                total_profit=round(total_profit, 2),
                status=pos["status"],
                entry_time=pos["entry_time"],
                updated_time=pos["updated_time"]
            ))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-prices")
async def update_all_prices():
    """
    Update current prices for all positions from market data
    """
    try:
        updated_count = 0
        
        for position_id, pos in positions_store.items():
            if pos["status"] == "open":
                # Fetch latest price
                df = fetch_market_data(pos["ticker"], pos["interval"], "1d")
                
                if not df.empty:
                    latest_price = float(df.iloc[-1]["Close"])
                    pos["current_price"] = latest_price
                    pos["updated_time"] = datetime.now().isoformat()
                    
                    # Check if TP or SL hit
                    if pos["type"] == "Buy":
                        if latest_price >= pos["take_profit"]:
                            pos["status"] = "tp_hit"
                        elif latest_price <= pos["stop_loss"]:
                            pos["status"] = "sl_hit"
                    else:  # Sell
                        if latest_price <= pos["take_profit"]:
                            pos["status"] = "tp_hit"
                        elif latest_price >= pos["stop_loss"]:
                            pos["status"] = "sl_hit"
                    
                    updated_count += 1
        
        return {"message": f"Updated {updated_count} positions", "success": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_all_positions():
    """
    Clear all positions
    """
    try:
        positions_store.clear()
        return {"message": "All positions cleared", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{position_id}")
async def delete_position(position_id: str):
    """
    Delete a specific position
    """
    try:
        if position_id in positions_store:
            del positions_store[position_id]
            return {"message": f"Position {position_id} deleted", "success": True}
        else:
            raise HTTPException(status_code=404, detail="Position not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_position_stats():
    """
    Get summary statistics for all positions
    """
    try:
        open_positions = [p for p in positions_store.values() if p["status"] == "open"]
        closed_positions = [p for p in positions_store.values() if p["status"] in ["tp_hit", "sl_hit"]]
        
        # Calculate totals
        total_pnl = 0.0
        total_profit = 0.0
        wins = 0
        
        for pos in positions_store.values():
            if pos["type"] == "Buy":
                pnl = pos["current_price"] - pos["entry_price"]
            else:
                pnl = pos["entry_price"] - pos["current_price"]
            
            total_pnl += pnl
            total_profit += pnl * pos["multiplier"] * pos["contracts"]
            
            if pnl > 0 and pos["status"] != "open":
                wins += 1
        
        closed_count = len(closed_positions)
        win_rate = (wins / closed_count * 100) if closed_count > 0 else 0.0
        
        return {
            "total_positions": len(positions_store),
            "open_positions": len(open_positions),
            "closed_positions": closed_count,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_profit": round(total_profit, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

