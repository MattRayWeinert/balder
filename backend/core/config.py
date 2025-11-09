"""
Configuration settings for the backend
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Balder Trading API"
    
    # Security
    PASSCODE: str = "000"
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]
    
    # Trading Settings
    DEFAULT_TICKERS: list = [
        "ES=F", "NQ=F", "YM=F", "RTY=F",
        "CL=F", "GC=F", "SI=F", "NG=F",
        "ZB=F", "ZN=F", "ZF=F", "ZT=F"
    ]
    
    class Config:
        case_sensitive = True

settings = Settings()

