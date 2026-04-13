import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'

export function AudioPlayer({ audioUrl, jobId }) {
  const containerRef = useRef(null)
  const wsRef        = useRef(null)
  const [playing, setPlaying]   = useState(false)
  const [duration, setDuration] = useState(0)
  const [current, setCurrent]   = useState(0)

  useEffect(() => {
    if (!audioUrl || !containerRef.current) return

    if (wsRef.current) wsRef.current.destroy()

    const ws = WaveSurfer.create({
      container:   containerRef.current,
      waveColor:   '#52525b',
      progressColor: '#8b5cf6',
      height:      56,
      barWidth:    2,
      barGap:      1,
      barRadius:   2,
      cursorColor: '#8b5cf6',
      url:         audioUrl,
    })

    ws.on('ready',       () => setDuration(ws.getDuration()))
    ws.on('audioprocess', () => setCurrent(ws.getCurrentTime()))
    ws.on('finish',      () => setPlaying(false))
    wsRef.current = ws

    return () => ws.destroy()
  }, [audioUrl])

  const togglePlay = () => {
    if (!wsRef.current) return
    wsRef.current.playPause()
    setPlaying(p => !p)
  }

  const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

  if (!audioUrl) return null

  return (
    <div className="flex flex-col gap-3 bg-zinc-800/60 rounded-xl p-4 border border-zinc-700">
      <div ref={containerRef} className="w-full" />
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className="w-9 h-9 rounded-full bg-violet-600 hover:bg-violet-500 flex items-center justify-center text-white transition-colors flex-shrink-0"
        >
          {playing ? '⏸' : '▶'}
        </button>
        <span className="text-xs text-zinc-400 font-mono tabular-nums">
          {fmt(current)} / {fmt(duration)}
        </span>
        <a
          href={audioUrl}
          download={`sangeet-${jobId}.mp3`}
          className="ml-auto text-xs text-zinc-400 hover:text-violet-400 border border-zinc-700 hover:border-violet-500 rounded-lg px-3 py-1.5 transition-colors"
        >
          Download MP3
        </a>
      </div>
    </div>
  )
}
