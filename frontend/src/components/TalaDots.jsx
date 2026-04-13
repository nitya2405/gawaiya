/**
 * TalaDots — renders beat dots for a tala.
 *
 * Sam   → amber filled dot (always beat 1, except Rūpak)
 * Khali → muted hollow dot (open-hand beats)
 * Rest  → zinc filled dot
 *
 * Vibhag groups are separated by a small gap.
 */
export function TalaDots({ tala }) {
  if (!tala) return null

  const { beats, sam = 1, khali = [], vibhag = [] } = tala

  // Build a set of 1-based beat numbers for quick lookup
  const khaliSet = new Set(khali)

  // Which beats start a new vibhag (excluding beat 1)
  const vibhagStarts = new Set()
  let acc = 1
  for (const v of vibhag) {
    acc += v
    if (acc <= beats) vibhagStarts.add(acc)
  }

  const dots = []
  for (let b = 1; b <= beats; b++) {
    const isSam   = b === sam
    const isKhali = khaliSet.has(b)
    const isGap   = b > 1 && vibhagStarts.has(b)

    dots.push(
      <div key={b} className={`flex items-center gap-0.5 ${isGap ? 'ml-2' : ''}`}>
        <div
          title={`Beat ${b}${isSam ? ' (Sam)' : isKhali ? ' (Khali)' : ''}`}
          className={`
            rounded-full transition-all
            ${isSam
              ? 'w-3 h-3 bg-amber-400 ring-1 ring-amber-300'
              : isKhali
              ? 'w-2.5 h-2.5 border border-zinc-600 bg-transparent'
              : 'w-2.5 h-2.5 bg-zinc-600'
            }
          `}
        />
      </div>
    )
  }

  return (
    <div className="flex items-center flex-wrap gap-0.5 mt-1.5">
      {dots}
    </div>
  )
}
