import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  listPositions, 
  createPosition, 
  updatePositionPrices, 
  clearPositions,
  deletePosition,
  getPositionStats,
  PositionCreate 
} from '../services/api'
import { Plus, RefreshCw, Trash2, TrendingUp, TrendingDown } from 'lucide-react'
import './Positions.css'

interface PositionsProps {
  passcode: string
}

const TICKERS = [
  "ES=F", "NQ=F", "YM=F", "RTY=F",
  "CL=F", "GC=F", "SI=F", "NG=F",
  "ZB=F", "ZN=F", "ZF=F", "ZT=F"
]

const MULTIPLIERS: Record<string, number> = {
  "ES=F": 50,
  "NQ=F": 20,
  "YM=F": 5,
  "RTY=F": 50,
  "CL=F": 1000,
  "GC=F": 100,
  "SI=F": 5000,
  "NG=F": 10000,
  "ZB=F": 1000,
  "ZN=F": 1000,
  "ZF=F": 1000,
  "ZT=F": 2000,
}

function Positions({ passcode }: PositionsProps) {
  const queryClient = useQueryClient()
  
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState<PositionCreate>({
    ticker: 'ES=F',
    interval: '2m',
    type: 'Buy',
    entry_price: 0,
    stop_loss: 0,
    take_profit: 0,
    contracts: 1,
    multiplier: 50,
  })

  const { data: positions, isLoading } = useQuery({
    queryKey: ['positions', passcode],
    queryFn: () => listPositions(passcode),
    refetchInterval: 30000,
  })

  const { data: stats } = useQuery({
    queryKey: ['position-stats', passcode],
    queryFn: () => getPositionStats(passcode),
    refetchInterval: 30000,
  })

  const createMutation = useMutation({
    mutationFn: (position: PositionCreate) => createPosition(passcode, position),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['positions'] })
      queryClient.invalidateQueries({ queryKey: ['position-stats'] })
      setShowForm(false)
      resetForm()
    },
  })

  const updatePricesMutation = useMutation({
    mutationFn: () => updatePositionPrices(passcode),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['positions'] })
      queryClient.invalidateQueries({ queryKey: ['position-stats'] })
    },
  })

  const clearMutation = useMutation({
    mutationFn: () => clearPositions(passcode),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['positions'] })
      queryClient.invalidateQueries({ queryKey: ['position-stats'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (positionId: string) => deletePosition(passcode, positionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['positions'] })
      queryClient.invalidateQueries({ queryKey: ['position-stats'] })
    },
  })

  const resetForm = () => {
    setFormData({
      ticker: 'ES=F',
      interval: '2m',
      type: 'Buy',
      entry_price: 0,
      stop_loss: 0,
      take_profit: 0,
      contracts: 1,
      multiplier: 50,
    })
  }

  const handleTickerChange = (ticker: string) => {
    setFormData({
      ...formData,
      ticker,
      multiplier: MULTIPLIERS[ticker] || 50,
    })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createMutation.mutate(formData)
  }

  return (
    <div className="positions">
      <div className="page-header">
        <h1>Position Tracking</h1>
        <div className="header-actions">
          <button onClick={() => updatePricesMutation.mutate()} className="secondary">
            <RefreshCw size={18} />
            Update Prices
          </button>
          <button onClick={() => setShowForm(!showForm)} className="primary">
            <Plus size={18} />
            Add Position
          </button>
        </div>
      </div>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-label">Total Positions</span>
            <span className="stat-value">{stats.total_positions}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Open Positions</span>
            <span className="stat-value">{stats.open_positions}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Win Rate</span>
            <span className={`stat-value ${stats.win_rate >= 50 ? 'text-success' : 'text-danger'}`}>
              {stats.win_rate.toFixed(1)}%
            </span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Total Profit</span>
            <span className={`stat-value ${stats.total_profit >= 0 ? 'text-success' : 'text-danger'}`}>
              ${stats.total_profit.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {showForm && (
        <div className="card">
          <h3>Add New Position</h3>
          <form onSubmit={handleSubmit} className="position-form">
            <div className="form-grid">
              <div className="control-group">
                <label>Ticker</label>
                <select 
                  value={formData.ticker} 
                  onChange={(e) => handleTickerChange(e.target.value)}
                >
                  {TICKERS.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div className="control-group">
                <label>Type</label>
                <select 
                  value={formData.type} 
                  onChange={(e) => setFormData({...formData, type: e.target.value})}
                >
                  <option value="Buy">Buy</option>
                  <option value="Sell">Sell</option>
                </select>
              </div>

              <div className="control-group">
                <label>Entry Price</label>
                <input 
                  type="number" 
                  value={formData.entry_price} 
                  onChange={(e) => setFormData({...formData, entry_price: parseFloat(e.target.value)})}
                  step="0.01"
                  required
                />
              </div>

              <div className="control-group">
                <label>Stop Loss</label>
                <input 
                  type="number" 
                  value={formData.stop_loss} 
                  onChange={(e) => setFormData({...formData, stop_loss: parseFloat(e.target.value)})}
                  step="0.01"
                  required
                />
              </div>

              <div className="control-group">
                <label>Take Profit</label>
                <input 
                  type="number" 
                  value={formData.take_profit} 
                  onChange={(e) => setFormData({...formData, take_profit: parseFloat(e.target.value)})}
                  step="0.01"
                  required
                />
              </div>

              <div className="control-group">
                <label>Contracts</label>
                <input 
                  type="number" 
                  value={formData.contracts} 
                  onChange={(e) => setFormData({...formData, contracts: parseInt(e.target.value)})}
                  min="1"
                  required
                />
              </div>

              <div className="control-group">
                <label>Multiplier</label>
                <input 
                  type="number" 
                  value={formData.multiplier} 
                  onChange={(e) => setFormData({...formData, multiplier: parseFloat(e.target.value)})}
                  step="0.01"
                  required
                />
              </div>
            </div>

            <div className="form-actions">
              <button type="submit" className="primary" disabled={createMutation.isPending}>
                {createMutation.isPending ? 'Adding...' : 'Add Position'}
              </button>
              <button type="button" onClick={() => setShowForm(false)} className="secondary">
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h3>Active Positions</h3>
          {positions && positions.length > 0 && (
            <button onClick={() => clearMutation.mutate()} className="danger">
              <Trash2 size={16} />
              Clear All
            </button>
          )}
        </div>

        {isLoading && (
          <div className="loading-container">
            <div className="loading" />
            <p>Loading positions...</p>
          </div>
        )}

        {positions && positions.length === 0 && (
          <p className="text-muted">No positions tracked. Add a position to get started.</p>
        )}

        {positions && positions.length > 0 && (
          <div className="positions-table-container">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Type</th>
                  <th>Entry</th>
                  <th>Current</th>
                  <th>Stop Loss</th>
                  <th>Take Profit</th>
                  <th>Contracts</th>
                  <th>PnL</th>
                  <th>Total Profit</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <tr key={position.position_id}>
                    <td><strong>{position.ticker}</strong></td>
                    <td>
                      <span className={`trade-type ${position.type === 'Buy' ? 'buy' : 'sell'}`}>
                        {position.type === 'Buy' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                        {position.type}
                      </span>
                    </td>
                    <td>${position.entry_price.toFixed(2)}</td>
                    <td>${position.current_price.toFixed(2)}</td>
                    <td>${position.stop_loss.toFixed(2)}</td>
                    <td>${position.take_profit.toFixed(2)}</td>
                    <td>{position.contracts}</td>
                    <td className={position.pnl >= 0 ? 'text-success' : 'text-danger'}>
                      ${position.pnl.toFixed(2)} ({position.pnl_percent.toFixed(2)}%)
                    </td>
                    <td className={position.total_profit >= 0 ? 'text-success' : 'text-danger'}>
                      <strong>${position.total_profit.toFixed(2)}</strong>
                    </td>
                    <td>
                      <span className={`status-badge status-${position.status}`}>
                        {position.status === 'open' && 'Open'}
                        {position.status === 'tp_hit' && 'TP Hit'}
                        {position.status === 'sl_hit' && 'SL Hit'}
                      </span>
                    </td>
                    <td>
                      <button 
                        onClick={() => deleteMutation.mutate(position.position_id)}
                        className="icon-button danger"
                        title="Delete position"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default Positions

