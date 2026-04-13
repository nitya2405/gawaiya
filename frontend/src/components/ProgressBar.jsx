import axios from 'axios'

export function ProgressBar({ status, progress, queuePos, nClips, clipNum, jobId, onCancel }) {
  if (!status || status === 'done' || status === 'failed' || status === 'cancelled') return null

  const pct = Math.round((progress ?? 0) * 100)

  const handleCancel = async () => {
    if (!jobId) return
    try { await axios.post(`/api/cancel/${jobId}`) } catch {}
    onCancel?.()
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-sm text-zinc-400">
          {status === 'queued' && queuePos
            ? `You are #${queuePos} in queue`
            : status === 'running' && nClips > 1
            ? `Generating clip ${clipNum} of ${nClips}…`
            : status === 'running'
            ? 'Generating…'
            : null}
        </p>
        <button
          onClick={handleCancel}
          className="text-xs text-zinc-500 hover:text-red-400 border border-zinc-700 hover:border-red-500 rounded-lg px-2.5 py-1 transition-colors"
        >
          Cancel
        </button>
      </div>
      <div className="w-full bg-zinc-800 rounded-full h-1.5">
        <div
          className="bg-violet-500 h-1.5 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      {status === 'running' && nClips > 1 && (
        <p className="text-xs text-zinc-600 text-right">{pct}%</p>
      )}
    </div>
  )
}
