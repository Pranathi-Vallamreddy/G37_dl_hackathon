import { useApp } from '../context/AppContext'
import type { TabName } from '../types'

const TABS: { id: TabName; label: string }[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'analyze',   label: 'Analyze'   },
  { id: 'history',   label: 'History'   },
  { id: 'model',     label: 'Model'     },
]

export default function Navbar() {
  const { activeTab, setActiveTab } = useApp()

  return (
    <header className="h-[52px] flex items-center justify-between px-5 bg-bg-secondary border-b border-border shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br from-accent-green to-accent-blue font-mono text-[11px] font-medium text-black select-none">
          PILL
        </div>
        <span className="font-semibold text-[15px] tracking-tight text-white">
          Bearing Diagnostics
        </span>
        <span className="text-[11px] text-gray-500 ml-1">v2.1 · PILL System</span>
      </div>

      {/* Tabs */}
      <nav className="flex gap-1">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-1.5 rounded-md text-[13px] font-medium transition-all duration-150 border
              ${activeTab === tab.id
                ? 'bg-bg-tertiary border-accent-green text-accent-green'
                : 'bg-transparent border-transparent text-gray-400 hover:bg-bg-tertiary hover:text-white'
              }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Status */}
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-accent-green status-live" />
        <span className="text-[12px] text-gray-500">Live · CPU</span>
      </div>
    </header>
  )
}
