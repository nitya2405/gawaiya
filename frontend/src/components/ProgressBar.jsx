export function ProgressBar({ status, progress, queuePos }) {
  if (!status || status === 'done' || status === 'failed') return null

  const pct = Math.round((progress ?? 0) * 100)

  return (
    <div className="flex flex-col gap-2">
      {status === 'queued' && queuePos && (
        <p className="text-sm text-zinc-400 text-center">
          You are #{queuePos} in queue
        </p>
      )}
      {status === 'running' && (
        <p className="text-sm text-zinc-400 text-center">
          Generating… {pct}%
        </p>
      )}
      <div className="w-full bg-zinc-800 rounded-full h-1.5">
        <div
          className="bg-violet-500 h-1.5 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
