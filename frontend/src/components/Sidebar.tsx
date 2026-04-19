import { useApp } from '../context/AppContext'

const DEMO_PREDICTION = {
  prediction: 1,
  prediction_label: 'Faulty',
  confidence: 0.92,
  attention_weights: { spectrogram: 0.45, physics: 0.38, metadata: 0.17 },
  branch_predictions: { spectrogram: 0, physics: 1, metadata: 0 },
  branch_confidence:  { spectrogram: 0.78, physics: 0.82, metadata: 0.61 },
  dominant_branch: 'spectrogram',
}

const BRANCH_META = {
  spectrogram: { label: 'Spectrogram', short: 'S', color: '#3b82f6', bg: 'rgba(59,130,246,0.15)' },
  physics:     { label: 'Physics',     short: 'P', color: '#00e5b8', bg: 'rgba(0,229,184,0.15)'  },
  metadata:    { label: 'Metadata',    short: 'M', color: '#fb923c', bg: 'rgba(251,146,60,0.15)' },
}

export default function Sidebar() {
  const { currentPrediction } = useApp()
  const pred = currentPrediction ?? DEMO_PREDICTION as any

  const isFaulty    = pred.prediction === 1
  const confPct     = Math.round(pred.confidence * 100)
  const attn        = pred.attention_weights
  const branchPreds = pred.branch_predictions
  const branchConf  = pred.branch_confidence

  return (
    <aside className="w-64 shrink-0 bg-bg-secondary border-r border-border overflow-y-auto flex flex-col gap-4 p-4">

      {/* ── Status card ── */}
      <section>
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-2 font-medium">Current Sample</p>
        <div className="bg-bg-tertiary rounded-xl border border-border p-4 text-center">
          <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Bearing Status</p>
          <p className={`text-3xl font-semibold tracking-tight mb-2 transition-colors ${isFaulty ? 'text-status-danger' : 'text-status-success'}`}>
            {pred.prediction_label?.toUpperCase() ?? (isFaulty ? 'FAULTY' : 'HEALTHY')}
          </p>
          {/* Confidence bar */}
          <div className="h-1.5 bg-bg-card rounded-full overflow-hidden">
            <div
              className="h-full conf-bar-fill rounded-full"
              style={{ width: `${confPct}%` }}
            />
          </div>
          <div className="flex justify-between mt-1.5">
            <span className="text-[11px] text-gray-500">Confidence</span>
            <span className="text-[11px] font-mono font-medium text-white">{confPct}%</span>
          </div>
        </div>
      </section>

      {/* ── Attention weights ── */}
      <section>
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-2 font-medium">Attention Weights</p>
        <div className="flex flex-col gap-2">
          {(Object.keys(BRANCH_META) as Array<keyof typeof BRANCH_META>).map(key => {
            const pct = Math.round((attn[key] ?? 0) * 100)
            const meta = BRANCH_META[key]
            return (
              <div key={key} className="flex items-center gap-2">
                <span className="text-[11px] text-gray-400 w-[72px]">{meta.label}</span>
                <div className="flex-1 h-1.5 bg-bg-card rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full weight-fill transition-all duration-700"
                    style={{ width: `${pct}%`, background: meta.color }}
                  />
                </div>
                <span className="text-[11px] font-mono text-white w-7 text-right">{pct}%</span>
              </div>
            )
          })}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <span className="text-[11px] text-gray-500">Dominant:</span>
          <span className="inline-flex items-center gap-1.5 text-[11px] text-accent-green border border-accent-green/30 bg-accent-green/10 rounded-full px-2.5 py-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-green" />
            {pred.dominant_branch}
          </span>
        </div>
      </section>

      {/* ── Branch predictions ── */}
      <section>
        <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-2 font-medium">Branch Predictions</p>
        <div className="flex flex-col gap-2">
          {(Object.keys(BRANCH_META) as Array<keyof typeof BRANCH_META>).map(key => {
            const meta     = BRANCH_META[key]
            const bPred    = branchPreds[key]
            const bConf    = Math.round((branchConf[key] ?? 0) * 100)
            const isDom    = pred.dominant_branch === key
            const isFaulty = bPred === 1
            return (
              <div
                key={key}
                className={`flex items-center justify-between bg-bg-tertiary rounded-lg border px-3 py-2.5 transition-colors ${isDom ? 'border-accent-green/40' : 'border-border'}`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-6 h-6 rounded-[5px] flex items-center justify-center text-[10px] font-mono font-semibold"
                    style={{ background: meta.bg, color: meta.color }}
                  >
                    {meta.short}
                  </div>
                  <div>
                    <p className="text-[12px] font-medium leading-none">{meta.label}</p>
                    <p className="text-[10px] text-gray-500 mt-0.5">{bConf}% conf</p>
                  </div>
                </div>
                <span className={`text-[11px] font-mono font-medium px-2 py-0.5 rounded
                  ${isFaulty
                    ? 'bg-status-danger/10 text-status-danger'
                    : 'bg-accent-green/10 text-accent-green'}`}
                >
                  {isFaulty ? 'Faulty' : 'Healthy'}
                </span>
              </div>
            )
          })}
        </div>
      </section>

      {/* ── Recommendation ── */}
      {isFaulty && (
        <div className="bg-status-warning/8 border border-status-warning/25 rounded-lg px-3 py-2.5 text-[12px] text-status-warning animate-slide-up">
          <span className="font-semibold">Recommendation:</span> Physics branch detects fault signature — schedule maintenance inspection.
        </div>
      )}
      {!isFaulty && (
        <div className="bg-accent-green/8 border border-accent-green/25 rounded-lg px-3 py-2.5 text-[12px] text-accent-green">
          <span className="font-semibold">Status:</span> Bearing appears healthy. Continue routine monitoring.
        </div>
      )}
    </aside>
  )
}
