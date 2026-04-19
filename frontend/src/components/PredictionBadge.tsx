interface Props {
  prediction: number
  confidence?: number
  size?: 'sm' | 'md'
}

export default function PredictionBadge({ prediction, confidence, size = 'md' }: Props) {
  const isFaulty = prediction === 1
  const textSize = size === 'sm' ? 'text-[11px]' : 'text-[12px]'
  const padding  = size === 'sm' ? 'px-2 py-0.5' : 'px-2.5 py-1'

  return (
    <span className={`inline-flex items-center gap-1.5 font-mono font-medium rounded-md ${textSize} ${padding}
      ${isFaulty
        ? 'bg-status-danger/10 text-status-danger border border-status-danger/20'
        : 'bg-accent-green/10 text-accent-green border border-accent-green/20'}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${isFaulty ? 'bg-status-danger' : 'bg-accent-green'}`} />
      {isFaulty ? 'Faulty' : 'Healthy'}
      {confidence !== undefined && <span className="opacity-70">· {Math.round(confidence * 100)}%</span>}
    </span>
  )
}
