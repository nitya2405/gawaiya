import { useState } from 'react'
import axios from 'axios'

export function FeedbackButtons({ jobId }) {
  const [voted, setVoted]   = useState(null)  // 1 | -1 | null
  const [copied, setCopied] = useState(false)

  const submit = async (value) => {
    if (voted !== null || !jobId) return
    try {
      await axios.get(`/api/feedback/${jobId}?value=${value}`)
      setVoted(value)
    } catch {}
  }

  const copyLink = async () => {
    const url = `${window.location.origin}/?share=${jobId}`
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {}
  }

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className="text-xs text-zinc-500">Rate:</span>

      {/* Thumbs up */}
      <button
        onClick={() => submit(1)}
        disabled={voted !== null}
        title="Good"
        className={`
          flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg border transition-all
          ${voted === 1
            ? 'bg-emerald-900/40 border-emerald-500 text-emerald-400'
            : voted === null
            ? 'border-zinc-700 text-zinc-400 hover:border-emerald-600 hover:text-emerald-400'
            : 'border-zinc-800 text-zinc-700 cursor-default'}
        `}
      >
        👍 {voted === 1 && <span>Nice!</span>}
      </button>

      {/* Thumbs down */}
      <button
        onClick={() => submit(-1)}
        disabled={voted !== null}
        title="Not good"
        className={`
          flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg border transition-all
          ${voted === -1
            ? 'bg-red-900/40 border-red-500 text-red-400'
            : voted === null
            ? 'border-zinc-700 text-zinc-400 hover:border-red-600 hover:text-red-400'
            : 'border-zinc-800 text-zinc-700 cursor-default'}
        `}
      >
        👎 {voted === -1 && <span>Noted</span>}
      </button>

      {/* Share */}
      <button
        onClick={copyLink}
        className="ml-auto text-xs text-zinc-400 hover:text-violet-400 border border-zinc-700 hover:border-violet-500 rounded-lg px-3 py-1.5 transition-colors"
      >
        {copied ? '✓ Copied' : 'Copy link'}
      </button>
    </div>
  )
}
