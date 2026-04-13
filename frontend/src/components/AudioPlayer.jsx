import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import SpectrogramPlugin from 'wavesurfer.js/dist/plugins/spectrogram.esm.js'

export function AudioPlayer({ audioUrl, jobId }) {
  const waveRef      = useRef(null)
  const spectroRef   = useRef(null)
  const wsRef        = useRef(null)
  const [playing, setPlaying]     = useState(false)
  const [duration, setDuration]   = useState(0)
  const [current, setCurrent]     = useState(0)
  const [showSpec, setShowSpec]   = useState(false)
  const [specReady, setSpecReady] = useState(false)

  useEffect(() => {
    if (!audioUrl || !waveRef.current) return
    if (wsRef.current) wsRef.current.destroy()
    setPlaying(false)
    setCurrent(0)
    setSpecReady(false)

    const plugins = showSpec && spectroRef.current
      ? [SpectrogramPlugin.create({ container: spectroRef.current, labels: true, height: 80 })]
      : []

    const ws = WaveSurfer.create({
      container:     waveRef.current,
      waveColor:     '#52525b',
      progressColor: '#8b5cf6',
      height:        56,
      barWidth:      2,
      barGap:        1,
      barRadius:     2,
      cursorColor:   '#8b5cf6',
      url:           audioUrl,
      plugins,
    })

    ws.on('ready',        () => { setDuration(ws.getDuration()); setSpecReady(true) })
    ws.on('audioprocess', () => setCurrent(ws.getCurrentTime()))
    ws.on('finish',       () => setPlaying(false))
    wsRef.current = ws

    return () => ws.destroy()
  }, [audioUrl, showSpec])

  const togglePlay = () => {
    if (!wsRef.current) return
    wsRef.current.playPause()
    setPlaying(p => !p)
  }

  const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

  if (!audioUrl) return null

  return (
    <div className="flex flex-col gap-2 bg-zinc-800/60 rounded-xl p-4 border border-zinc-700">
      {/* Waveform */}
      <div ref={waveRef} className="w-full" />

      {/* Spectrogram (conditionally rendered) */}
      {showSpec && (
        <div ref={spectroRef} className="w-full mt-1" />
      )}

      {/* Controls row */}
      <div className="flex items-center gap-3 mt-1">
        <button
          onClick={togglePlay}
          className="w-9 h-9 rounded-full bg-violet-600 hover:bg-violet-500 flex items-center justify-center text-white transition-colors flex-shrink-0"
        >
          {playing ? '⏸' : '▶'}
        </button>
        <span className="text-xs text-zinc-400 font-mono tabular-nums">
          {fmt(current)} / {fmt(duration)}
        </span>

        {/* Spectrogram toggle */}
        <button
          onClick={() => setShowSpec(v => !v)}
          className={`text-xs border rounded-lg px-2.5 py-1 transition-colors ${showSpec ? 'border-violet-500 text-violet-300' : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'}`}
          title="Toggle spectrogram"
        >
          Spectrogram
        </button>

        <a
          href={audioUrl}
          download={`sangeet-${jobId}.mp3`}
          className="ml-auto text-xs text-zinc-400 hover:text-violet-400 border border-zinc-700 hover:border-violet-500 rounded-lg px-3 py-1.5 transition-colors"
        >
          Download
        </a>
      </div>
    </div>
  )
}
