import { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import TradingSignals from './pages/TradingSignals'
import Backtest from './pages/Backtest'
import Positions from './pages/Positions'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [passcode, setPasscode] = useState('')

  if (!isAuthenticated) {
    return <Login onLogin={(code) => {
      setPasscode(code)
      setIsAuthenticated(true)
    }} />
  }

  return (
    <div className="app">
      <Routes>
        <Route path="/" element={<Dashboard passcode={passcode} />}>
          <Route index element={<Navigate to="/signals" replace />} />
          <Route path="signals" element={<TradingSignals passcode={passcode} />} />
          <Route path="backtest" element={<Backtest passcode={passcode} />} />
          <Route path="positions" element={<Positions passcode={passcode} />} />
        </Route>
      </Routes>
    </div>
  )
}

export default App

