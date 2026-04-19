import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { predictFile } from '../api'
import { useApp } from '../context/AppContext'
import type { PredictionResult } from '../types'
import PredictionBadge from '../components/PredictionBadge'
import PhysicsFeatureGrid from '../components/PhysicsFeatureGrid'

type Stage = 'idle' | 'loading' | 'bandpass' | 'spectrogram' | 'features' | 'inference' | 'fusion' | 'done' | 'error'

const PIPELINE_STEPS = [
  { id: 'bandpass',    label: 'Bandpass Filter',      sub: 'Envelope detection'          },
  { id: 'spectrogram', label: 'Mel-Spectrogram',       sub: 'Frequency domain (64 × T)'  },
  { id: 'features',   label: 'Physics Features',      sub: '7 fault signatures'          },
  { id: 'inference',  label: '3-Branch Inference',    sub: 'CNN + MLP + Metadata'        },
  { id: 'fusion',     label: 'Attention Fusion',      sub: 'Adaptive weight gate'        },
]

const STEP_IDS = PIPELINE_STEPS.map(s => s.id) as Stage[]

function stepStatus(stepId: Stage, currentStage: Stage): 'idle' | 'active' | 'done' {
  const idx     = STEP_IDS.indexOf(stepId)
  const currIdx = STEP_IDS.indexOf(currentStage)
  if (currIdx < 0) return 'idle'
  if (idx < currIdx) return 'done'
  if (idx === currIdx) return 'active'
  return 'idle'
}

export default function Analyze() {
  const { addPrediction, recentPredictions } = useApp()
  const [stage,  setStage]  = useState<Stage>('idle')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error,  setError]  = useState<string | null>(null)
  const [file,   setFile]   = useState<File | null>(null)

  const runPipeline = async (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)

    // Animate through stages before/during actual API call
    const delay = (ms: number) => new Promise(r => setTimeout(r, ms))
    setStage('bandpass')
    const apiCallPromise = predictFile(f)
    await delay(400)
    setStage('spectrogram')
    await delay(400)
    setStage('features')
    await delay(350)
    setStage('inference')

    try {
      const res = await apiCallPromise
      setStage('fusion')
      await delay(300)
      setStage('done')
      setResult(res)
      addPrediction(res)
    } catch (e: any) {
      setStage('error')
      setError(e?.response?.data?.detail ?? 'Inference failed. Check backend connection.')
    }
  }

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) runPipeline(accepted[0])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/octet-stream': ['.mat'], '': ['.mat'] },
    maxFiles: 1,
  })

  return (
    <div className="flex flex-col gap-5 animate-slide-up max-w-3xl">
      <div className="flex items-center justify-between">
        <h1 className="text-[16px] font-semibold tracking-tight">Analyze Bearing Signal</h1>
      </div>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive
            ? 'border-accent-green bg-accent-green/5'
            : 'border-border-strong hover:border-accent-green/60 hover:bg-accent-green/3 bg-bg-tertiary'}`}
      >
        <input {...getInputProps()} />
        <div className="w-10 h-10 mx-auto mb-3 bg-bg-card border border-border rounded-xl flex items-center justify-center">
          <svg width="20" height="20" fill="none" stroke="#6b7280" strokeWidth="1.5" viewBox="0 0 24 24">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>
        <p className="text-[13px] font-medium mb-1">
          {isDragActive ? 'Drop the .mat file here' : 'Drop .mat file here or click to browse'}
        </p>
        <p className="text-[11px] text-gray-500">Bearing vibration signal · SCA dataset format · max 200 MB</p>
      </div>

      {/* Pipeline animation */}
      {stage !== 'idle' && (
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">
            Processing Pipeline
            {file && <span className="ml-2 normal-case font-mono text-gray-400">— {file.name}</span>}
          </p>
          <div className="flex gap-2">
            {PIPELINE_STEPS.map((step, i) => {
              const status = stage === 'done'
                ? 'done'
                : stage === 'error'
                  ? (STEP_IDS.indexOf(step.id as Stage) < STEP_IDS.indexOf(stage) ? 'done' : 'idle')
                  : stepStatus(step.id as Stage, stage)
              return (
                <div
                  key={step.id}
                  className={`flex-1 rounded-lg border px-3 py-2.5 pipeline-step transition-all duration-300
                    ${status === 'active' ? 'active border-accent-green bg-accent-green/6' : ''}
                    ${status === 'done'   ? 'done border-accent-green/30 bg-accent-green/3' : ''}
                    ${status === 'idle'   ? 'border-border bg-bg-tertiary opacity-50' : ''}`}
                >
                  <div className="flex items-center gap-1.5 mb-0.5">
                    {status === 'done' && (
                      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                        <circle cx="6" cy="6" r="6" fill="rgba(0,229,184,0.2)"/>
                        <path d="M3.5 6l1.8 1.8 3.2-3.6" stroke="#00e5b8" strokeWidth="1.2" strokeLinecap="round"/>
                      </svg>
                    )}
                    {status === 'active' && (
                      <span className="w-2 h-2 rounded-full bg-accent-green status-live inline-block" />
                    )}
                    <span className="text-[10px] font-semibold">{i + 1}</span>
                  </div>
                  <p className="text-[11px] font-medium leading-tight">{step.label}</p>
                  <p className="text-[10px] text-gray-500 mt-0.5">{step.sub}</p>
                </div>
              )
            })}
          </div>
          {stage === 'error' && (
            <p className="text-[12px] text-status-danger mt-3">{error}</p>
          )}
        </div>
      )}

      {/* Result card */}
      {result && stage === 'done' && (
        <div className="bg-bg-secondary border border-accent-green/30 rounded-xl p-5 animate-slide-up">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-[11px] text-gray-500 mb-1 font-mono">{result.filename}</p>
              <h2 className="text-[15px] font-semibold">Inference Complete</h2>
            </div>
            <PredictionBadge prediction={result.prediction} confidence={result.confidence} />
          </div>

          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="text-center">
              <p className={`text-xl font-semibold font-mono ${result.prediction === 1 ? 'text-status-danger' : 'text-accent-green'}`}>
                {Math.round(result.confidence * 100)}%
              </p>
              <p className="text-[11px] text-gray-500">Fusion confidence</p>
            </div>
            <div className="text-center">
              <p className="text-xl font-semibold font-mono">{result.inference_ms}ms</p>
              <p className="text-[11px] text-gray-500">Inference time</p>
            </div>
            <div className="text-center">
              <p className="text-xl font-semibold font-mono capitalize">{result.dominant_branch}</p>
              <p className="text-[11px] text-gray-500">Dominant branch</p>
            </div>
          </div>

          {/* Attention weights mini */}
          <div className="bg-bg-tertiary rounded-lg border border-border p-3 mb-4">
            <p className="text-[11px] text-gray-500 mb-2">Attention weights</p>
            <div className="flex gap-4">
              {(['spectrogram', 'physics', 'metadata'] as const).map((key, i) => {
                const colors = ['#3b82f6', '#00e5b8', '#fb923c']
                const pct = Math.round(result.attention_weights[key] * 100)
                return (
                  <div key={key} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-[2px]" style={{ background: colors[i] }} />
                    <span className="text-[12px] text-gray-400 capitalize">{key}</span>
                    <span className="text-[12px] font-mono font-medium">{pct}%</span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Physics features */}
          <p className="text-[11px] text-gray-500 mb-2">Physics feature anomaly scores</p>
          <PhysicsFeatureGrid features={result.physics_features} />

          {result.prediction === 1 && (
            <div className="mt-4 bg-status-warning/8 border border-status-warning/25 rounded-lg px-3 py-2.5 text-[12px] text-status-warning">
              <span className="font-semibold">Action Required:</span> Fault signature detected. Schedule maintenance inspection to prevent bearing failure.
            </div>
          )}
        </div>
      )}

      {/* Recent uploads */}
      {recentPredictions.length > 0 && (
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-gray-500 mb-3 font-medium">Recent Uploads</p>
          <div className="flex flex-col gap-2">
            {recentPredictions.slice(0, 5).map((p, i) => (
              <div key={i} className="flex items-center justify-between py-1.5 border-b border-border last:border-0">
                <span className="text-[12px] font-mono text-gray-300">{p.filename}</span>
                <div className="flex items-center gap-3">
                  <span className="text-[11px] text-gray-500">{p.inference_ms}ms</span>
                  <PredictionBadge prediction={p.prediction} confidence={p.confidence} size="sm" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
