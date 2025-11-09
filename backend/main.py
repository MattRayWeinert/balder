"""
FastAPI Backend for Balder Trading App
"""
import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional, List
import secrets

from api.routes import signals, backtest, positions
from core.config import settings

print("=" * 60)
print("🚀 BALDER BACKEND STARTING")
print("=" * 60)
print(f"📁 Working directory: {os.getcwd()}")
print(f"🐍 Python path: {sys.path[0]}")

# Test data fetching on startup
try:
    print("\n🔍 Testing data fetch capability...")
    from core.strategy import fetch_market_data
    test_df = fetch_market_data("ES=F", "2m", "1d")
    if not test_df.empty:
        print(f"✅ Data fetch works! Got {len(test_df)} rows")
    else:
        print("⚠️  WARNING: Data fetch returned empty DataFrame!")
except Exception as e:
    print(f"❌ ERROR: Data fetch failed: {e}")
    
print("=" * 60)

# Initialize FastAPI app
app = FastAPI(
    title="Balder Trading API",
    description="Backend API for Balder futures trading signals and backtesting",
    version="1.0.0"
)

# CORS middleware - allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Security
security = HTTPBasic()

def verify_passcode(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify passcode authentication"""
    correct_passcode = "000"
    is_correct = secrets.compare_digest(credentials.password, correct_passcode)
    if not is_correct:
        raise HTTPException(status_code=401, detail="Invalid passcode")
    return credentials

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"📡 {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"✅ Status: {response.status_code}")
    return response

# Health check
@app.get("/")
async def root():
    return {
        "status": "online",
        "app": "Balder Trading API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test-data")
async def test_data_fetch():
    """Test endpoint to verify data fetching works"""
    try:
        from core.strategy import fetch_market_data
        df = fetch_market_data("ES=F", "2m", "1d")
        
        if df.empty:
            return {
                "status": "error",
                "message": "DataFrame is empty",
                "rows": 0
            }
        
        latest = df.iloc[-1]
        return {
            "status": "success",
            "rows": len(df),
            "latest_price": float(latest["Close"]),
            "has_indicators": "RSI" in df.columns,
            "indicators": {
                "RSI": float(latest.get("RSI", 0)),
                "ATR": float(latest.get("ATR", 0)),
                "ADX": float(latest.get("ADX", 0))
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "type": type(e).__name__
        }

# Include routers
app.include_router(
    signals.router,
    prefix="/api/signals",
    tags=["signals"],
    dependencies=[Depends(verify_passcode)]
)

app.include_router(
    backtest.router,
    prefix="/api/backtest",
    tags=["backtest"],
    dependencies=[Depends(verify_passcode)]
)

app.include_router(
    positions.router,
    prefix="/api/positions",
    tags=["positions"],
    dependencies=[Depends(verify_passcode)]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

