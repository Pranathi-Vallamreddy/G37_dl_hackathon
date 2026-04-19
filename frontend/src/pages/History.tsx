import { useState, useEffect } from 'react'
import { fetchHistory, exportHistoryUrl } from '../api'
import type { HistoryRecord } from '../types'
import PredictionBadge from '../components/PredictionBadge'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  LineElement, PointElement, Filler, Tooltip,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, LineElement, PointElement, Filler, Tooltip)

const GRID  = 'rgba(42,48,64,0.8)'
const TICK  = '#6b7280'
const MONO  = 'DM Mono, monospace'

export default function History() {
  const [rows,    setRows]    = useState<HistoryRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [filter,  setFilter]  = useState<'all' | 'healthy' | 'faulty'>('all')

  useEffect(() => {
    fetchHistory(50)
      .then(setRows)
      .finally(() => setLoading(false))
  }, [])

  const filtered = rows.filter(r => {
    if (filter === 'healthy') return r.prediction === 0
    if (filter === 'faulty')  return r.prediction === 1
    return true
  })

  const faultyCount  = rows.filter(r => r.prediction === 1).length
  const healthyCount = rows.length - faultyCount
  const avgConf      = rows.length ? (rows.reduce((a, r) => a + r.confidence, 0) / rows.length * 100).toFixed(1) : '—'

  const trendData = {
    labels: [...rows].reverse().slice(0, 15).map((_, i) => `#${i + 1}`),
    datasets: [{
      label: 'Confidence %',
      data: [...rows].reverse().slice(0, 15).map(r => Math.round(r.confidence * 100)),
      borderColor: '#00e5b8',
      backgroundColor: 'rgba(0,229,184,0.07)',
      tension: 0.4, fill: true,
      pointRadius: 3,
      pointBackgroundColor: [...rows].reverse().slice(0, 15).map(r => r.prediction === 1 ? '#ff4455' : '#00e5b8'),
      borderWidth: 1.5,
    }],
  }

  const trendOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: GRID }, ticks: { color: TICK, font: { family: MONO, size: 10 } } },
      y: { grid: { color: GRID }, ticks: { color: TICK, font: { family: MONO, size: 10 }, callback: (v: any) => v + '%' }, min: 50, max: 100 },
    },
  }

  return (
    <div className="flex flex-col gap-4 animate-slide-up">
      <div className="flex items-center justify-between">
        <h1 className="text-[16px] font-semibold tracking-tight">Prediction History</h1>
        <a
          href={exportHistoryUrl()}
          download="pill_history.csv"
          className="text-[12px] font-medium px-3 py-1.5 rounded-lg border border-border text-gray-400 hover:bg-bg-tertiary hover:text-white transition-colors"
        >
          Export CSV
        </a>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { v: rows.length.toString(),  l: 'Total predictions' },
          { v: faultyCount.toString(),  l: 'Faulty detected',  c: 'text-status-danger' },
          { v: healthyCount.toString(), l: 'Healthy',          c: 'text-accent-green' },
          { v: avgConf + '%',           l: 'Avg confidence' },
        ].map(({ v, l, c }) => (
          <div key={l} className="bg-bg-tertiary border border-border rounded-xl p-3 text-center">
            <p className={`text-xl font-semibold font-mono ${c ?? 'text-white'}`}>{v}</p>
            <p className="text-[11px] text-gray-500 mt-0.5">{l}</p>
          </div>
        ))}
      </div>

      {/* Trend chart */}
      {rows.length > 0 && (
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Confidence Trend</p>
          <div className="h-[110px]">
            <Line data={trendData} options={trendOpts as any} />
          </div>
        </div>
      )}

      {/* Filter */}
      <div className="flex gap-1">
        {(['all', 'healthy', 'faulty'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded-md text-[12px] font-medium border transition-all
              ${filter === f
                ? 'bg-bg-tertiary border-accent-green text-accent-green'
                : 'bg-transparent border-border text-gray-500 hover:text-white'}`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-gray-500 text-[13px]">Loading history…</div>
        ) : filtered.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-[13px]">No records yet. Run predictions in the Analyze tab.</div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                {['#', 'File', 'Type', 'Prediction', 'Confidence', 'Dominant Branch', 'Timestamp'].map(h => (
                  <th key={h} className="text-left text-[11px] font-medium text-gray-500 px-4 py-3 uppercase tracking-wide">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map(r => (
                <tr key={r.id} className="border-b border-border/50 hover:bg-bg-tertiary transition-colors">
                  <td className="px-4 py-3 text-[12px] text-gray-600 font-mono">{r.id}</td>
                  <td className="px-4 py-3 text-[12px] font-mono text-gray-300">{r.filename}</td>
                  <td className="px-4 py-3 text-[12px] text-gray-400">Type {r.bearing_type}</td>
                  <td className="px-4 py-3">
                    <PredictionBadge prediction={r.prediction} size="sm" />
                  </td>
                  <td className="px-4 py-3 text-[12px] font-mono">{Math.round(r.confidence * 100)}%</td>
                  <td className="px-4 py-3">
                    <span className="inline-flex items-center gap-1 text-[11px] text-accent-green border border-accent-green/20 bg-accent-green/8 rounded-full px-2 py-0.5">
                      <span className="w-1 h-1 rounded-full bg-accent-green" />
                      {r.dominant_branch}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-[11px] text-gray-500 font-mono">
                    {new Date(r.timestamp).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
