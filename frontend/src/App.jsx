import { useState, useEffect } from 'react'
import { useVocab } from './hooks/useVocab'
import { useGeneration } from './hooks/useGeneration'
import { RagaSelector } from './components/RagaSelector'
import { TalaSelector } from './components/TalaSelector'
import { DurationSlider } from './components/DurationSlider'
import { ProgressBar } from './components/ProgressBar'
import { AudioPlayer } from './components/AudioPlayer'
import { FeedbackButtons } from './components/FeedbackButtons'
import { GenerationHistory, saveToHistory, loadHistory } from './components/GenerationHistory'

export default function App() {
  const { ragas, talas, loading } = useVocab()

  const [raga, setRaga]         = useState('Kalyāṇ')
  const [tala, setTala]         = useState('Tīntāl')
  const [duration, setDuration] = useState(12)
  const [showAdv, setShowAdv]   = useState(false)
  const [cfgScale, setCfgScale] = useState(5.0)
  const [nCb, setNCb]           = useState(4)
  const [history, setHistory]   = useState(loadHistory)

  const { generate, reset, jobId, status, progress, queuePos, error, audioUrl, isGenerating } = useGeneration()

  // Set defaults once vocab loads
  useEffect(() => {
    if (ragas.length && !ragas.find(r => r.name === raga)) setRaga(ragas[0].name)
    if (talas.length && !talas.find(t => t.name === tala)) setTala(talas[0].name)
  }, [ragas, talas])

  // Save to history when done
  useEffect(() => {
    if (status === 'done') {
      const entry = { raga, tala, duration_sec: duration }
      saveToHistory(entry)
      setHistory(loadHistory())
    }
  }, [status])

  const handleGenerate = () => {
    generate({ raga, tala, duration_sec: duration, cfg_scale: cfgScale, n_codebooks: nCb })
  }

  const handleReload = (entry) => {
    reset()
    setRaga(entry.raga)
    setTala(entry.tala)
    setDuration(entry.duration_sec)
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-950 text-zinc-400">
        Loading…
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-md flex flex-col gap-6">

        {/* Header */}
        <div className="text-center">
          <h1 className="text-2xl font-semibold text-zinc-100 tracking-tight">Sangeet AI</h1>
          <p className="text-sm text-zinc-500 mt-1">Hindustani Classical Generator</p>
        </div>

        {/* Controls card */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-5 flex flex-col gap-5">
          <RagaSelector ragas={ragas} value={raga} onChange={setRaga} />
          <TalaSelector talas={talas} value={tala} onChange={setTala} />
          <DurationSlider value={duration} onChange={setDuration} />

          {/* Advanced */}
          <div>
            <button
              onClick={() => setShowAdv(v => !v)}
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              {showAdv ? '▾' : '▸'} Advanced
            </button>
            {showAdv && (
              <div className="mt-3 flex flex-col gap-4 pl-1">
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between">
                    <label className="text-xs uppercase tracking-widest text-zinc-500 font-medium">CFG Scale</label>
                    <span className="text-xs text-zinc-300">{cfgScale.toFixed(1)}</span>
                  </div>
                  <input type="range" min="3" max="7" step="0.1" value={cfgScale}
                    onChange={e => setCfgScale(Number(e.target.value))}
                    className="accent-violet-500 cursor-pointer" />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-xs uppercase tracking-widest text-zinc-500 font-medium">Codebooks</label>
                  <div className="flex gap-2">
                    {[2, 4, 8].map(n => (
                      <button key={n} onClick={() => setNCb(n)}
                        className={`flex-1 py-1.5 rounded-lg text-sm border transition-colors ${nCb === n ? 'bg-violet-600 border-violet-500 text-white' : 'bg-zinc-800 border-zinc-700 text-zinc-300 hover:border-zinc-500'}`}>
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:bg-zinc-700 disabled:text-zinc-400 text-white font-medium transition-colors"
          >
            {isGenerating ? 'Generating…' : 'Generate Music'}
          </button>
        </div>

        {/* Progress */}
        <ProgressBar status={status} progress={progress} queuePos={queuePos} />

        {/* Error */}
        {error && (
          <p className="text-sm text-red-400 text-center">{error}</p>
        )}

        {/* Audio player */}
        {audioUrl && (
          <div className="flex flex-col gap-3">
            <AudioPlayer audioUrl={audioUrl} jobId={jobId} />
            <div className="flex items-center justify-between">
              <FeedbackButtons jobId={jobId} />
              <button
                onClick={handleGenerate}
                className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                Regenerate
              </button>
            </div>
          </div>
        )}

        {/* History */}
        <GenerationHistory history={history} onReload={handleReload} />

      </div>
    </div>
  )
}
