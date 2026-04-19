import type { PhysicsFeatures } from '../types'

interface Props {
  features: PhysicsFeatures
}

const THRESHOLDS: Record<string, number> = {
  BPFI: 0.75, BPFO: 0.55, BSF: 0.70,
  RMS: 0.70, Kurtosis: 0.60, Crest: 0.70, Skewness: 0.60,
}

function featureColor(key: string, val: number): string {
  const thresh = THRESHOLDS[key] ?? 0.7
  if (val > thresh) return 'text-status-danger'
  if (val > thresh * 0.75) return 'text-status-warning'
  return 'text-white'
}

export default function PhysicsFeatureGrid({ features }: Props) {
  return (
    <div>
      <div className="grid grid-cols-7 gap-2">
        {(Object.entries(features) as [string, number][]).map(([key, val]) => {
          const color = featureColor(key, val)
          return (
            <div key={key} className="bg-bg-tertiary border border-border rounded-lg p-2 text-center">
              <p className={`text-base font-semibold font-mono ${color}`}>{val.toFixed(2)}</p>
              <p className="text-[10px] text-gray-500 mt-0.5">{key}</p>
            </div>
          )
        })}
      </div>
      <p className="text-[11px] text-gray-600 mt-2">
        Red = anomalous (above fault threshold) · Yellow = elevated
      </p>
    </div>
  )
}
