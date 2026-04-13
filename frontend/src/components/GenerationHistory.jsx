const MAX = 5
const KEY = 'sangeet_history'

export function saveToHistory(entry) {
  try {
    const prev = JSON.parse(localStorage.getItem(KEY) || '[]')
    const next = [entry, ...prev].slice(0, MAX)
    localStorage.setItem(KEY, JSON.stringify(next))
  } catch {}
}

export function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '[]')
  } catch {
    return []
  }
}

export function GenerationHistory({ history, onReload }) {
  if (!history.length) return null

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs uppercase tracking-widest text-zinc-500 font-medium">Recent</p>
      <div className="flex flex-wrap gap-2">
        {history.map((h, i) => (
          <button
            key={i}
            onClick={() => onReload(h)}
            className="text-xs bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg px-3 py-1.5 text-zinc-300 transition-colors"
          >
            {h.raga} · {h.tala} · {h.duration_sec}s
          </button>
        ))}
      </div>
    </div>
  )
}
