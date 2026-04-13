export function TalaSelector({ talas, value, onChange }) {
  const meta = talas.find(t => t.name === value)

  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs uppercase tracking-widest text-zinc-500 font-medium">Tala</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2.5 text-zinc-100 text-sm focus:outline-none focus:border-violet-500 cursor-pointer"
      >
        {talas.map(t => (
          <option key={t.name} value={t.name}>{t.name}</option>
        ))}
      </select>
      {meta && (
        <p className="text-xs text-zinc-500 mt-0.5">
          {meta.beats} beats · {meta.character}
        </p>
      )}
    </div>
  )
}
