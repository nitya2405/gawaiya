import { useState } from 'react'
import axios from 'axios'

export function FeedbackButtons({ jobId }) {
  const [voted, setVoted] = useState(null)

  const submit = async (value) => {
    if (voted !== null || !jobId) return
    try {
      await axios.get(`/api/feedback/${jobId}?value=${value}`)
      setVoted(value)
    } catch {
      // silently fail
    }
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-zinc-500">Rate this:</span>
      <button
        onClick={() => submit(1)}
        disabled={voted !== null}
        className={`text-lg transition-all ${voted === 1 ? 'opacity-100 scale-110' : 'opacity-50 hover:opacity-100'} disabled:cursor-default`}
        title="Good"
      >
        👍
      </button>
      <button
        onClick={() => submit(-1)}
        disabled={voted !== null}
        className={`text-lg transition-all ${voted === -1 ? 'opacity-100 scale-110' : 'opacity-50 hover:opacity-100'} disabled:cursor-default`}
        title="Not good"
      >
        👎
      </button>
      {voted !== null && (
        <span className="text-xs text-zinc-500">Thanks!</span>
      )}
    </div>
  )
}
