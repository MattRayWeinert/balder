import { useState } from 'react'
import './Login.css'

interface LoginProps {
  onLogin: (passcode: string) => void
}

function Login({ onLogin }: LoginProps) {
  const [passcode, setPasscode] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    if (passcode === '000') {
      onLogin(passcode)
    } else {
      setError('Invalid passcode. Please try again.')
      setPasscode('')
    }
  }

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1 className="app-title">BALDER</h1>
          <p className="app-subtitle">Futures Trading Signals</p>
        </div>
        
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="passcode">Passcode</label>
            <input
              type="password"
              id="passcode"
              value={passcode}
              onChange={(e) => {
                setPasscode(e.target.value)
                setError('')
              }}
              placeholder="Enter passcode"
              autoFocus
            />
          </div>
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
          
          <button type="submit" className="primary login-button">
            Access App
          </button>
        </form>
        
        <div className="warning-box">
          <p className="warning-title">⚠️ DATA DELAY WARNING</p>
          <p className="warning-text">
            This app uses Yahoo Finance data which is delayed by 15-20 minutes.
            Not suitable for live trading. Use for backtesting and strategy development only.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Login

