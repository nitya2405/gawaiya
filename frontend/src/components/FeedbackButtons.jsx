import { useState } from 'react'
import axios from 'axios'

export function FeedbackButtons({ jobId }) {
  const [voted, setVoted]     = useState(null)
  const [copied, setCopied]   = useState(false)

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
      <button
        onClick={() => submit(1)}
        disabled={voted !== null}
        className={`text-lg transition-all ${voted === 1 ? 'opacity-100 scale-110' : 'opacity-50 hover:opacity-100'} disabled:cursor-default`}
        title="Good"
      >👍</button>
      <button
        onClick={() => submit(-1)}
        disabled={voted !== null}
        className={`text-lg transition-all ${voted === -1 ? 'opacity-100 scale-110' : 'opacity-50 hover:opacity-100'} disabled:cursor-default`}
        title="Not good"
      >👎</button>
      {voted !== null && <span className="text-xs text-zinc-500">Thanks!</span>}

      <button
        onClick={copyLink}
        className="ml-auto text-xs text-zinc-400 hover:text-violet-400 border border-zinc-700 hover:border-violet-500 rounded-lg px-3 py-1.5 transition-colors"
      >
        {copied ? 'Copied!' : 'Copy link'}
      </button>
    </div>
  )
}
