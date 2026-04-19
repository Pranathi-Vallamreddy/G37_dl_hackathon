import { AppProvider, useApp } from './context/AppContext'
import Sidebar from './components/Sidebar'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Analyze from './pages/Analyze'
import History from './pages/History'
import ModelPage from './pages/Model'

function Layout() {
  const { activeTab } = useApp()

  return (
    <div className="flex flex-col h-screen bg-bg-primary overflow-hidden">
      <Navbar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-5 bg-bg-primary">
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'analyze'   && <Analyze />}
          {activeTab === 'history'   && <History />}
          {activeTab === 'model'     && <ModelPage />}
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <Layout />
    </AppProvider>
  )
}
