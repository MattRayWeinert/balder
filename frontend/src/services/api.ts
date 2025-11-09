import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000/api'

// Create axios instance with basic auth
export const createApiClient = (passcode: string) => {
  return axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
    auth: {
      username: 'user',
      password: passcode,
    },
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
  })
}

// Types
export interface SignalRequest {
  ticker: string
  interval: string
  risk_factor: number
  risk_reward: number
}

export interface SignalResponse {
  ticker: string
  interval: string
  signal: string | null
  entry: number | null
  stop_loss: number | null
  take_profit: number | null
  confidence: number
  latest_price: number
  timestamp: string
  indicators: {
    EMA_fast: number
    EMA_slow: number
    RSI: number
    ATR: number
    ADX: number
    MACD_hist: number
    Volume: number
  }
  message?: string
}

export interface BacktestRequest {
  ticker: string
  interval: string
  test_date: string
  risk_factor: number
  risk_reward: number
}

export interface TradeResult {
  entry_time: string
  exit_time: string
  type: string
  entry_price: number
  exit_price: number
  stop_loss: number
  take_profit: number
  pnl: number
  pnl_percent: number
  win: boolean
}

export interface BacktestResponse {
  ticker: string
  interval: string
  test_date: string
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  total_pnl: number
  trades: TradeResult[]
}

export interface PositionCreate {
  ticker: string
  interval: string
  type: string
  entry_price: number
  stop_loss: number
  take_profit: number
  contracts: number
  multiplier: number
}

export interface Position {
  position_id: string
  ticker: string
  interval: string
  type: string
  entry_price: number
  current_price: number
  stop_loss: number
  take_profit: number
  contracts: number
  multiplier: number
  pnl: number
  pnl_percent: number
  total_profit: number
  status: string
  entry_time: string
  updated_time: string
}

export interface PositionStats {
  total_positions: number
  open_positions: number
  closed_positions: number
  win_rate: number
  total_pnl: number
  total_profit: number
}

// API Functions
export const getSignal = async (
  passcode: string,
  request: SignalRequest,
  signal?: AbortSignal
): Promise<SignalResponse> => {
  const api = createApiClient(passcode)
  const response = await api.post('/signals/current', request, { signal })
  return response.data
}

export const getHistoricalData = async (
  passcode: string,
  ticker: string,
  interval: string,
  period: string,
  signal?: AbortSignal
) => {
  const api = createApiClient(passcode)
  const response = await api.get('/signals/historical', {
    params: { ticker, interval, period },
    signal,
  })
  return response.data
}

export const runBacktest = async (passcode: string, request: BacktestRequest): Promise<BacktestResponse> => {
  const api = createApiClient(passcode)
  const response = await api.post('/backtest/run', request)
  return response.data
}

export const createPosition = async (passcode: string, position: PositionCreate): Promise<Position> => {
  const api = createApiClient(passcode)
  const response = await api.post('/positions/create', position)
  return response.data
}

export const listPositions = async (passcode: string): Promise<Position[]> => {
  const api = createApiClient(passcode)
  const response = await api.get('/positions/list')
  return response.data
}

export const updatePositionPrices = async (passcode: string) => {
  const api = createApiClient(passcode)
  const response = await api.post('/positions/update-prices')
  return response.data
}

export const clearPositions = async (passcode: string) => {
  const api = createApiClient(passcode)
  const response = await api.delete('/positions/clear')
  return response.data
}

export const deletePosition = async (passcode: string, positionId: string) => {
  const api = createApiClient(passcode)
  const response = await api.delete(`/positions/${positionId}`)
  return response.data
}

export const getPositionStats = async (passcode: string): Promise<PositionStats> => {
  const api = createApiClient(passcode)
  const response = await api.get('/positions/stats')
  return response.data
}

