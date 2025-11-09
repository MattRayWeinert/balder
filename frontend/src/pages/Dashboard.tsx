import { Outlet, Link, useLocation } from 'react-router-dom'
import { TrendingUp, BarChart3, Briefcase } from 'lucide-react'
import './Dashboard.css'

interface DashboardProps {
  passcode: string
}

function Dashboard({ passcode }: DashboardProps) {
  const location = useLocation()

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/')
  }

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="sidebar-title">BALDER</h1>
          <p className="sidebar-subtitle">Trading Signals</p>
        </div>
        
        <nav className="sidebar-nav">
          <Link 
            to="/signals" 
            className={`nav-item ${isActive('/signals') ? 'active' : ''}`}
          >
            <TrendingUp size={20} />
            <span>Trade Signals</span>
          </Link>
          
          <Link 
            to="/backtest" 
            className={`nav-item ${isActive('/backtest') ? 'active' : ''}`}
          >
            <BarChart3 size={20} />
            <span>Backtest</span>
          </Link>
          
          <Link 
            to="/positions" 
            className={`nav-item ${isActive('/positions') ? 'active' : ''}`}
          >
            <Briefcase size={20} />
            <span>Positions</span>
          </Link>
        </nav>
      </aside>
      
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}

export default Dashboard

