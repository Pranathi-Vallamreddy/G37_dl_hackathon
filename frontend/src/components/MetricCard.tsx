interface MetricCardProps {
  value: string
  label: string
  sub?: string
  subColor?: string
}

export default function MetricCard({ value, label, sub, subColor = 'text-gray-500' }: MetricCardProps) {
  return (
    <div className="bg-bg-tertiary border border-border rounded-xl p-4">
      <p className="text-[22px] font-semibold font-mono tracking-tight">{value}</p>
      <p className="text-[12px] text-gray-400 mt-0.5">{label}</p>
      {sub && <p className={`text-[11px] mt-1 ${subColor}`}>{sub}</p>}
    </div>
  )
}
