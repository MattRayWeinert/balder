import { useState, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { runBacktest, BacktestRequest, BacktestResponse } from '../services/api'
import { Play, TrendingUp, TrendingDown, Calendar } from 'lucide-react'
import './Backtest.css'

interface BacktestProps {
  passcode: string
}

const TICKERS = [
  "ES=F", "NQ=F", "YM=F", "RTY=F",
  "CL=F", "GC=F", "SI=F", "NG=F",
  "ZB=F", "ZN=F", "ZF=F", "ZT=F"
]

function Backtest({ passcode }: BacktestProps) {
  const [ticker, setTicker] = useState('ES=F')
  const [interval, setInterval] = useState('2m')
  const [testDate, setTestDate] = useState(() => {
    const yesterday = new Date()
    yesterday.setDate(yesterday.getDate() - 1)
    return yesterday.toISOString().split('T')[0]
  })
  const [riskFactor, setRiskFactor] = useState(1.0)
  const [riskReward, setRiskReward] = useState(2.0)
  const [results, setResults] = useState<BacktestResponse | null>(null)
  const dateInputRef = useRef<HTMLInputElement>(null)

  const mutation = useMutation({
    mutationFn: (request: BacktestRequest) => runBacktest(passcode, request),
    onSuccess: (data) => {
      setResults(data)
    },
  })

  const handleRunBacktest = () => {
    const request: BacktestRequest = {
      ticker,
      interval,
      test_date: testDate,
      risk_factor: riskFactor,
      risk_reward: riskReward,
    }
    mutation.mutate(request)
  }

  return (
    <div className="backtest">
      <div className="page-header">
        <h1>Backtest Strategy</h1>
      </div>

      <div className="card">
        <h3>Backtest Configuration</h3>
        
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

          <div className="control-group date-group">
            <label>Test Date</label>
            <div className="date-picker">
              <input 
                type="date" 
                value={testDate} 
                onChange={(e) => setTestDate(e.target.value)}
                ref={dateInputRef}
              />
              <button 
                type="button" 
                className="icon-button" 
                onClick={() => dateInputRef.current?.showPicker()}
              >
                <Calendar size={18} />
              </button>
            </div>
          </div>

          <div className="control-group risk-row">
            <label>Risk Settings</label>
            <div className="risk-sliders">
              <div className="slider-control">
                <span className="slider-label">Risk Factor: {riskFactor.toFixed(1)}x</span>
                <input 
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.5"
                  value={riskFactor}
                  onChange={(e) => setRiskFactor(parseFloat(e.target.value))}
                />
              </div>
              <div className="slider-control">
                <span className="slider-label">Risk:Reward {riskReward.toFixed(1)} : 1</span>
                <input 
                  type="range"
                  min="1"
                  max="5"
                  step="0.5"
                  value={riskReward}
                  onChange={(e) => setRiskReward(parseFloat(e.target.value))}
                />
              </div>
            </div>
          </div>
        </div>

        <button 
          onClick={handleRunBacktest} 
          className="primary run-button"
          disabled={mutation.isPending}
        >
          {mutation.isPending ? (
            <>
              <div className="loading" />
              Running Backtest...
            </>
          ) : (
            <>
              <Play size={18} />
              Run Backtest
            </>
          )}
        </button>
      </div>

      {mutation.isError && (
        <div className="error-box">
          Error running backtest: {(mutation.error as Error).message}
        </div>
      )}

      {results && (
        <>
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-label">Total Trades</span>
              <span className="stat-value">{results.total_trades}</span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Win Rate</span>
              <span className={`stat-value ${results.win_rate >= 50 ? 'text-success' : 'text-danger'}`}>
                {results.win_rate.toFixed(1)}%
              </span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Total PnL</span>
              <span className={`stat-value ${results.total_pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                ${results.total_pnl.toFixed(2)}
              </span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Winning Trades</span>
              <span className="stat-value text-success">{results.winning_trades}</span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Losing Trades</span>
              <span className="stat-value text-danger">{results.losing_trades}</span>
            </div>
          </div>

          <div className="card">
            <h3>Trade History</h3>
            
            {results.trades.length === 0 ? (
              <p className="text-muted">No trades executed during this period.</p>
            ) : (
              <div className="trades-table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Type</th>
                      <th>Entry Time</th>
                      <th>Exit Time</th>
                      <th>Entry Price</th>
                      <th>Exit Price</th>
                      <th>Stop Loss</th>
                      <th>Take Profit</th>
                      <th>PnL</th>
                      <th>PnL %</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.trades.map((trade, idx) => (
                      <tr key={idx}>
                        <td>
                          <span className={`trade-type ${trade.type === 'Buy' ? 'buy' : 'sell'}`}>
                            {trade.type === 'Buy' ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                            {trade.type}
                          </span>
                        </td>
                        <td>{new Date(trade.entry_time).toLocaleTimeString()}</td>
                        <td>{new Date(trade.exit_time).toLocaleTimeString()}</td>
                        <td>${trade.entry_price.toFixed(2)}</td>
                        <td>${trade.exit_price.toFixed(2)}</td>
                        <td>${trade.stop_loss.toFixed(2)}</td>
                        <td>${trade.take_profit.toFixed(2)}</td>
                        <td className={trade.pnl >= 0 ? 'text-success' : 'text-danger'}>
                          ${trade.pnl.toFixed(2)}
                        </td>
                        <td className={trade.pnl_percent >= 0 ? 'text-success' : 'text-danger'}>
                          {trade.pnl_percent.toFixed(2)}%
                        </td>
                        <td>
                          <span className={`result-badge ${trade.win ? 'win' : 'loss'}`}>
                            {trade.win ? 'Win' : 'Loss'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default Backtest

