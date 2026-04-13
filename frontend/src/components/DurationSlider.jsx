const CLIP_SEC = 12

export function DurationSlider({ value, onChange }) {
  const isStitched = value > CLIP_SEC

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label className="text-xs uppercase tracking-widest text-zinc-500 font-medium">Duration</label>
        <span className="text-sm font-semibold text-zinc-200">{value}s</span>
      </div>
      <input
        type="range"
        min={6}
        max={60}
        step={1}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-violet-500 cursor-pointer"
      />
      <div className="flex justify-between text-xs text-zinc-600">
        <span>6s</span>
        <span>60s</span>
      </div>
      {!isStitched ? (
        <p className="text-xs text-emerald-400 mt-0.5">
          Single coherent clip · best quality
        </p>
      ) : (
        <p className="text-xs text-amber-400 mt-0.5">
          Beyond 12s, clips are stitched with a crossfade. Transitions may be audible.
        </p>
      )}
    </div>
  )
}
