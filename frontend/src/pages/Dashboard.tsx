import { useApp } from '../context/AppContext'
import MetricCard from '../components/MetricCard'
import PhysicsFeatureGrid from '../components/PhysicsFeatureGrid'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, Title, Tooltip, Legend, Filler,
} from 'chart.js'
import { Bar, Doughnut, Line } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, Title, Tooltip, Legend, Filler
)

const GRID_COLOR   = 'rgba(42,48,64,0.8)'
const TICK_COLOR   = '#6b7280'
const FONT_FAMILY  = 'DM Mono, monospace'

const DEMO_PHYSICS = {
  BPFI: 0.92, BPFO: 0.45, BSF: 0.78,
  RMS: 0.61, Kurtosis: 0.63, Crest: 0.42, Skewness: 0.22,
}

const HISTORY_CONFS = [0.94, 0.87, 0.96, 0.79, 0.91, 0.85, 0.92, 0.88, 0.93, 0.92]
const HISTORY_LABELS = HISTORY_CONFS.map((_, i) => `#${15 + i}`)

export default function Dashboard() {
  const { currentPrediction, recentPredictions } = useApp()

  const physics = (currentPrediction?.physics_features ?? DEMO_PHYSICS) as typeof DEMO_PHYSICS
  const attn    = currentPrediction?.attention_weights ?? { spectrogram: 0.45, physics: 0.38, metadata: 0.17 }
  const branchConf = currentPrediction?.branch_confidence ?? { spectrogram: 0.78, physics: 0.82, metadata: 0.61 }

  // Compute history from real predictions if available
  const histConfs = recentPredictions.length >= 3
    ? recentPredictions.slice(0, 10).map(p => Math.round(p.confidence * 100)).reverse()
    : HISTORY_CONFS.map(v => Math.round(v * 100))
  const histLabels = histConfs.map((_, i) => `#${i + 1}`)

  const faultyCount = recentPredictions.filter(p => p.prediction === 1).length
  const totalCount  = recentPredictions.length || 1

  // Chart data
  const barData = {
    labels: ['Spectrogram', 'Physics', 'Metadata'],
    datasets: [{
      label: 'Confidence %',
      data: [
        Math.round((branchConf.spectrogram as number) * 100),
        Math.round((branchConf.physics     as number) * 100),
        Math.round((branchConf.metadata    as number) * 100),
      ],
      backgroundColor: ['rgba(59,130,246,0.7)', 'rgba(0,229,184,0.7)', 'rgba(251,146,60,0.7)'],
      borderColor:     ['#3b82f6', '#00e5b8', '#fb923c'],
      borderWidth: 1,
      borderRadius: 5,
    }],
  }

  const barOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, font: { family: FONT_FAMILY, size: 11 } } },
      y: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, font: { family: FONT_FAMILY, size: 10 }, callback: (v: any) => v + '%' }, min: 0, max: 100 },
    },
  }

  const donutData = {
    labels: ['Spectrogram', 'Physics', 'Metadata'],
    datasets: [{
      data: [
        Math.round(attn.spectrogram * 100),
        Math.round(attn.physics     * 100),
        Math.round(attn.metadata    * 100),
      ],
      backgroundColor: ['#3b82f6', '#00e5b8', '#fb923c'],
      borderWidth: 0,
      hoverOffset: 4,
    }],
  }

  const donutOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    cutout: '65%',
  }

  const lineData = {
    labels: histLabels,
    datasets: [{
      label: 'Confidence %',
      data: histConfs,
      borderColor: '#00e5b8',
      backgroundColor: 'rgba(0,229,184,0.07)',
      tension: 0.4,
      fill: true,
      pointRadius: 3,
      pointBackgroundColor: histConfs.map(v => v < 85 ? '#ff4455' : '#00e5b8'),
      borderWidth: 1.5,
    }],
  }

  const lineOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, font: { family: FONT_FAMILY, size: 10 } } },
      y: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, font: { family: FONT_FAMILY, size: 10 }, callback: (v: any) => v + '%' }, min: 50, max: 100 },
    },
  }

  return (
    <div className="flex flex-col gap-4 animate-slide-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-[16px] font-semibold tracking-tight">System Overview</h1>
        <span className="text-[11px] text-gray-500">Last updated: {new Date().toLocaleString()}</span>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-3 gap-3">
        <MetricCard
          value={recentPredictions.length ? `${faultyCount}/${recentPredictions.length}` : '1/3'}
          label="Branches detect fault"
          sub="Physics branch diverges"
          subColor="text-status-danger"
        />
        <MetricCard
          value="93.2%"
          label="Fusion accuracy (test)"
          sub="+3.1% vs best branch"
          subColor="text-accent-green"
        />
        <MetricCard
          value="~68ms"
          label="Inference time (CPU)"
          sub="3 branches + fusion"
          subColor="text-gray-500"
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Branch Confidence</p>
          <div className="h-[150px]">
            <Bar data={barData} options={barOptions as any} />
          </div>
        </div>

        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Attention Distribution</p>
          <div className="flex items-center gap-4">
            <div className="h-[130px] w-[130px]">
              <Doughnut data={donutData} options={donutOptions as any} />
            </div>
            <div className="flex flex-col gap-2">
              {(['spectrogram', 'physics', 'metadata'] as const).map((key, i) => {
                const colors = ['#3b82f6', '#00e5b8', '#fb923c']
                const labels = ['Spectrogram', 'Physics', 'Metadata']
                return (
                  <div key={key} className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-[2px]" style={{ background: colors[i] }} />
                    <span className="text-[12px] text-gray-400">{labels[i]}</span>
                    <span className="text-[13px] font-semibold font-mono ml-auto text-white">
                      {Math.round(attn[key] * 100)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Prediction history line chart */}
      <div className="bg-bg-secondary border border-border rounded-xl p-4">
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">
          Prediction History — Confidence Trend
        </p>
        <div className="h-[110px]">
          <Line data={lineData} options={lineOptions as any} />
        </div>
      </div>

      {/* Physics features */}
      <div className="bg-bg-secondary border border-border rounded-xl p-4">
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">
          Physics Feature Anomaly Scores
        </p>
        <PhysicsFeatureGrid features={physics} />
      </div>
    </div>
  )
}
