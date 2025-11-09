import { useState, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getSignal, getHistoricalData, SignalRequest } from '../services/api'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { RefreshCw, TrendingUp, TrendingDown } from 'lucide-react'
import './TradingSignals.css'

interface TradingSignalsProps {
  passcode: string
}

const TICKERS = [
  "ES=F", "NQ=F", "YM=F", "RTY=F",
  "CL=F", "GC=F", "SI=F", "NG=F",
  "ZB=F", "ZN=F", "ZF=F", "ZT=F"
]

function TradingSignals({ passcode }: TradingSignalsProps) {
  const [ticker, setTicker] = useState('ES=F')
  const [interval, setInterval] = useState('2m')
  const [riskFactor, setRiskFactor] = useState(1.0)
  const [riskReward, setRiskReward] = useState(2.0)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [refreshInterval, setRefreshInterval] = useState(30)
  const [sliderValue, setSliderValue] = useState(30) // Local state for slider

  // Memoize the signal request to prevent unnecessary re-renders
  const signalRequest: SignalRequest = useMemo(() => ({
    ticker,
    interval,
    risk_factor: riskFactor,
    risk_reward: riskReward,
  }), [ticker, interval, riskFactor, riskReward])

  // Memoize the refetch interval calculation
  const refetchIntervalMs = useMemo(() => {
    return autoRefresh ? refreshInterval * 1000 : false
  }, [autoRefresh, refreshInterval])

  const { data: signalData, isLoading, error, refetch } = useQuery({
    queryKey: ['signal', signalRequest],
    queryFn: () => getSignal(passcode, signalRequest),
    refetchInterval: refetchIntervalMs,
    staleTime: 5000, // Prevent refetching within 5 seconds
    refetchOnMount: false, // Don't refetch on mount if data exists
    refetchOnWindowFocus: false, // Don't refetch when window regains focus
  })

  const { data: historicalData } = useQuery({
    queryKey: ['historical', ticker, interval],
    queryFn: () => getHistoricalData(passcode, ticker, interval, '3d'),
    staleTime: 60000, // Cache for 60 seconds
    refetchInterval: false, // Don't auto-refetch historical data
    refetchOnMount: false, // Don't refetch on mount
    refetchOnWindowFocus: false, // Don't refetch when window regains focus
  })

  // Memoize helper functions
  const getSignalIcon = useCallback(() => {
    if (!signalData?.signal) return null
    return signalData.signal === 'Buy' ? 
      <TrendingUp size={24} className="text-success" /> : 
      <TrendingDown size={24} className="text-danger" />
  }, [signalData?.signal])

  const getSignalColor = useCallback(() => {
    if (!signalData?.signal) return 'gray'
    return signalData.signal === 'Buy' ? 'var(--success)' : 'var(--danger)'
  }, [signalData?.signal])

  return (
    <div className="trading-signals">
      <div className="page-header">
        <h1>Live Trading Signals</h1>
        <div className="header-actions">
          <label className="auto-refresh">
            <input 
              type="checkbox" 
              checked={autoRefresh} 
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            <span>Auto-refresh</span>
          </label>
          <div className="refresh-slider">
            <label>Interval: {sliderValue}s</label>
            <input 
              type="range"
              min="5"
              max="120"
              step="5"
              value={sliderValue}
              onChange={(e) => setSliderValue(parseInt(e.target.value))}
              onMouseUp={(e) => setRefreshInterval(parseInt((e.target as HTMLInputElement).value))}
              onTouchEnd={(e) => setRefreshInterval(parseInt((e.target as HTMLInputElement).value))}
              disabled={!autoRefresh}
            />
          </div>
          <button onClick={() => refetch()} className="secondary">
            <RefreshCw size={18} />
            Refresh
          </button>
        </div>
      </div>

      <div className="controls-grid">
        <div className="control-group">
          <label>Ticker</label>
          <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
            {TICKERS.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Interval</label>
          <select value={interval} onChange={(e) => setInterval(e.target.value)}>
            <option value="1m">1 Minute</option>
            <option value="2m">2 Minutes</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
          </select>
        </div>

        <div className="control-group">
          <label>Risk Factor (ATR Multiplier)</label>
          <input 
            type="number" 
            value={riskFactor} 
            onChange={(e) => setRiskFactor(parseFloat(e.target.value))}
            min="0.5"
            max="5"
            step="0.5"
          />
        </div>

        <div className="control-group">
          <label>Risk:Reward Ratio</label>
          <input 
            type="number" 
            value={riskReward} 
            onChange={(e) => setRiskReward(parseFloat(e.target.value))}
            min="1"
            max="5"
            step="0.5"
          />
        </div>
      </div>

      {isLoading && (
        <div className="loading-container">
          <div className="loading" />
          <p>Loading signal data...</p>
        </div>
      )}

      {error && (
        <div className="error-box">
          <h3>⚠️ Error Loading Signal</h3>
          <p><strong>Message:</strong> {(error as Error).message}</p>
          <p className="text-muted" style={{ marginTop: '10px', fontSize: '12px' }}>
            Make sure the backend is running at http://localhost:8000
          </p>
          <button onClick={() => refetch()} className="secondary" style={{ marginTop: '12px' }}>
            Try Again
          </button>
        </div>
      )}

      {signalData && (
        <>
          <div className="signal-card" style={{ borderColor: getSignalColor() }}>
            <div className="signal-header">
              {getSignalIcon()}
              <div>
                <h2>{signalData.signal || 'No Signal'}</h2>
                <p className="text-muted">
                  {ticker} · {interval} · Last updated: {new Date(signalData.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>

            {signalData.signal && (
              <div className="signal-details">
                <div className="detail-item">
                  <span className="label">Entry Price</span>
                  <span className="value">${signalData.entry?.toFixed(2)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Stop Loss</span>
                  <span className="value text-danger">${signalData.stop_loss?.toFixed(2)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Take Profit</span>
                  <span className="value text-success">${signalData.take_profit?.toFixed(2)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Confidence</span>
                  <span className="value">{signalData.confidence.toFixed(1)}%</span>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <h3>Technical Indicators</h3>
            <div className="indicators-grid">
              <div className="indicator-item">
                <span className="indicator-label">Latest Price</span>
                <span className="indicator-value">${signalData.latest_price?.toFixed(2) || 'N/A'}</span>
              </div>
              {signalData.indicators && (
                <>
                  <div className="indicator-item">
                    <span className="indicator-label">RSI</span>
                    <span className="indicator-value">{signalData.indicators.RSI?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="indicator-item">
                    <span className="indicator-label">ATR</span>
                    <span className="indicator-value">{signalData.indicators.ATR?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="indicator-item">
                    <span className="indicator-label">ADX</span>
                    <span className="indicator-value">{signalData.indicators.ADX?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="indicator-item">
                    <span className="indicator-label">MACD Histogram</span>
                    <span className="indicator-value">{signalData.indicators.MACD_hist?.toFixed(4) || 'N/A'}</span>
                  </div>
                  <div className="indicator-item">
                    <span className="indicator-label">Volume</span>
                    <span className="indicator-value">{signalData.indicators.Volume?.toLocaleString() || 'N/A'}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </>
      )}

      {historicalData?.data && (
        <div className="card">
          <h3>Price Chart</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={historicalData.data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                dataKey="timestamp" 
                stroke="#94a3b8"
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ 
                  background: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="close" stroke="#667eea" name="Close" />
              <Line type="monotone" dataKey="ema_fast" stroke="#10b981" name="EMA Fast" />
              <Line type="monotone" dataKey="ema_slow" stroke="#ef4444" name="EMA Slow" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

export default TradingSignals

