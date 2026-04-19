import { useState, useEffect } from 'react'
import { fetchModelInfo } from '../api'
import type { ModelInfo } from '../types'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Tooltip, Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend)

const GRID = 'rgba(42,48,64,0.8)'
const TICK = '#6b7280'
const MONO = 'DM Mono, monospace'

export default function ModelPage() {
  const [info,    setInfo]    = useState<ModelInfo | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="text-center py-16 text-gray-500 text-[13px]">Loading model info…</div>
  }
  if (!info) {
    return <div className="text-center py-16 text-status-danger text-[13px]">Failed to load model info. Is the backend running?</div>
  }

  const cm = info.confusion_matrix
  const total = cm.true_negative + cm.false_positive + cm.false_negative + cm.true_positive

  // Branch accuracy bar chart
  const branchBarData = {
    labels: ['Spectrogram', 'Physics', 'Metadata', 'Fusion'],
    datasets: [
      {
        label: 'Accuracy',
        data: [
          Math.round(info.branch_accuracy.spectrogram * 100),
          Math.round(info.branch_accuracy.physics     * 100),
          Math.round(info.branch_accuracy.metadata    * 100),
          Math.round(info.test_accuracy               * 100),
        ],
        backgroundColor: ['rgba(59,130,246,0.7)', 'rgba(0,229,184,0.7)', 'rgba(251,146,60,0.7)', 'rgba(129,140,248,0.7)'],
        borderColor:     ['#3b82f6', '#00e5b8', '#fb923c', '#818cf8'],
        borderWidth: 1, borderRadius: 5,
      },
      {
        label: 'Faulty Recall',
        data: [
          Math.round(info.branch_faulty_recall.spectrogram * 100),
          Math.round(info.branch_faulty_recall.physics     * 100),
          Math.round(info.branch_faulty_recall.metadata    * 100),
          Math.round((info.per_class_metrics.faulty?.recall ?? 0.967) * 100),
        ],
        backgroundColor: ['rgba(59,130,246,0.3)', 'rgba(0,229,184,0.3)', 'rgba(251,146,60,0.3)', 'rgba(129,140,248,0.3)'],
        borderColor:     ['#3b82f6', '#00e5b8', '#fb923c', '#818cf8'],
        borderWidth: 1, borderRadius: 5, borderDash: [4, 2],
      },
    ],
  }

  const branchBarOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: { color: TICK, font: { family: MONO, size: 11 }, boxWidth: 12 },
      },
    },
    scales: {
      x: { grid: { color: GRID }, ticks: { color: TICK, font: { family: MONO, size: 11 } } },
      y: { grid: { color: GRID }, ticks: { color: TICK, font: { family: MONO, size: 10 }, callback: (v: any) => v + '%' }, min: 60, max: 100 },
    },
  }

  return (
    <div className="flex flex-col gap-4 animate-slide-up">
      <h1 className="text-[16px] font-semibold tracking-tight">Model Information</h1>

      {/* Info + metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Fusion Model</p>
          <table className="w-full text-[12px]">
            <tbody>
              {[
                ['Architecture', info.architecture],
                ['Framework',    info.framework],
                ['Classes',      info.class_names.join(' / ')],
                ['Test accuracy', (info.test_accuracy * 100).toFixed(1) + '%'],
                ['Parameters',   info.n_parameters.toLocaleString()],
                ['Training date', info.training_date],
                ['Inference (CPU)', info.inference_time_ms.cpu + ' ms'],
                ['Inference (GPU)', info.inference_time_ms.gpu + ' ms'],
              ].map(([k, v]) => (
                <tr key={k} className="border-t border-border first:border-0">
                  <td className="py-1.5 text-gray-500">{k}</td>
                  <td className="py-1.5 text-right font-mono text-white">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Per-class metrics */}
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Per-Class Metrics</p>
          {Object.entries(info.per_class_metrics).map(([cls, m]) => (
            <div key={cls} className="mb-3 last:mb-0">
              <p className="text-[12px] font-medium capitalize mb-1.5">{cls}</p>
              <div className="grid grid-cols-3 gap-2">
                {[
                  ['Precision', m.precision],
                  ['Recall',    m.recall],
                  ['F1',        m.f1],
                ].map(([label, val]) => (
                  <div key={label as string} className="bg-bg-tertiary rounded-lg p-2 text-center">
                    <p className="text-[14px] font-semibold font-mono text-accent-green">{((val as number) * 100).toFixed(1)}%</p>
                    <p className="text-[10px] text-gray-500">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Branch comparison */}
      <div className="bg-bg-secondary border border-border rounded-xl p-4">
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Branch Performance Comparison</p>
        <div className="h-[180px]">
          <Bar data={branchBarData} options={branchBarOpts as any} />
        </div>
      </div>

      {/* Confusion matrix */}
      <div className="bg-bg-secondary border border-border rounded-xl p-4">
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-4 font-medium">Confusion Matrix — Fusion Model (Test Set)</p>
        <div className="grid grid-cols-3 items-center gap-3 max-w-md mx-auto">
          {/* Labels */}
          <div />
          <div className="text-center text-[11px] text-gray-500 font-medium">Pred: Healthy</div>
          <div className="text-center text-[11px] text-gray-500 font-medium">Pred: Faulty</div>

          <div className="text-[11px] text-gray-500 font-medium text-right">Actual: Healthy</div>
          <div className="bg-accent-green/10 border border-accent-green/30 rounded-xl p-4 text-center">
            <p className="text-2xl font-semibold font-mono text-accent-green">{cm.true_negative}</p>
            <p className="text-[10px] text-gray-500 mt-1">TN · {(cm.true_negative / total * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-status-danger/8 border border-status-danger/20 rounded-xl p-4 text-center">
            <p className="text-2xl font-semibold font-mono text-status-danger">{cm.false_positive}</p>
            <p className="text-[10px] text-gray-500 mt-1">FP · {(cm.false_positive / total * 100).toFixed(1)}%</p>
          </div>

          <div className="text-[11px] text-gray-500 font-medium text-right">Actual: Faulty</div>
          <div className="bg-status-warning/8 border border-status-warning/20 rounded-xl p-4 text-center">
            <p className="text-2xl font-semibold font-mono text-status-warning">{cm.false_negative}</p>
            <p className="text-[10px] text-gray-500 mt-1">FN · {(cm.false_negative / total * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-accent-green/10 border border-accent-green/30 rounded-xl p-4 text-center">
            <p className="text-2xl font-semibold font-mono text-accent-green">{cm.true_positive}</p>
            <p className="text-[10px] text-gray-500 mt-1">TP · {(cm.true_positive / total * 100).toFixed(1)}%</p>
          </div>
        </div>
        <p className="text-[11px] text-gray-600 text-center mt-3">
          Total test samples: {total} · Accuracy: {((cm.true_negative + cm.true_positive) / total * 100).toFixed(1)}%
        </p>
      </div>

      {/* Checkpoints */}
      <div className="bg-bg-secondary border border-border rounded-xl p-4">
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Checkpoint Files</p>
        <div className="flex flex-col gap-2">
          {info.checkpoints.map(ck => (
            <div key={ck.filename} className="flex items-center justify-between bg-bg-tertiary border border-border rounded-lg px-3 py-2.5">
              <div className="flex items-center gap-3">
                <span className={`w-2 h-2 rounded-full ${ck.present ? 'bg-accent-green' : 'bg-status-danger opacity-50'}`} />
                <div>
                  <p className="text-[12px] font-mono">{ck.filename}</p>
                  <p className="text-[11px] text-gray-500">{ck.description}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-[11px] text-gray-500 font-mono">{ck.size}</span>
                <span className={`text-[11px] font-medium px-2 py-0.5 rounded-md font-mono
                  ${ck.present
                    ? 'bg-accent-green/10 text-accent-green'
                    : 'bg-gray-700/30 text-gray-500'}`}
                >
                  {ck.present ? 'Loaded' : 'Missing'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
