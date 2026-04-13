import { useState, useRef, useCallback } from 'react'
import axios from 'axios'

const POLL_INTERVAL = 2000
const WS_TIMEOUT    = 3000

export function useGeneration() {
  const [jobId, setJobId]           = useState(null)
  const [status, setStatus]         = useState(null)
  const [progress, setProgress]     = useState(0)
  const [queuePos, setQueuePos]     = useState(null)
  const [nClips, setNClips]         = useState(0)
  const [clipNum, setClipNum]       = useState(0)
  const [error, setError]           = useState(null)
  const [audioUrl, setAudioUrl]     = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)

  const wsRef      = useRef(null)
  const pollRef    = useRef(null)
  const jobIdRef   = useRef(null)

  const _stopAll = useCallback(() => {
    if (wsRef.current)   { wsRef.current.close(); wsRef.current = null }
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const _applyUpdate = useCallback((data) => {
    setStatus(data.status)
    setProgress(data.progress ?? 0)
    setQueuePos(data.queue_position ?? null)
    setNClips(data.n_clips ?? 0)
    setClipNum(data.clip_num ?? 0)
    if (data.error) setError(data.error)
    if (data.status === 'done' || data.status === 'failed' || data.status === 'cancelled') {
      setIsGenerating(false)
      _stopAll()
    }
  }, [_stopAll])

  const _startPolling = useCallback((id) => {
    pollRef.current = setInterval(async () => {
      try {
        const { data } = await axios.get(`/api/job/${id}`)
        _applyUpdate(data)
        if (data.status === 'done') setAudioUrl(`/api/audio/${id}`)
      } catch {}
    }, POLL_INTERVAL)
  }, [_applyUpdate])

  const _startWS = useCallback((id) => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://localhost:8000/api/ws/${id}`)
    wsRef.current = ws

    const timer = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) { ws.close(); _startPolling(id) }
    }, WS_TIMEOUT)

    ws.onopen  = () => clearTimeout(timer)
    ws.onerror = () => { clearTimeout(timer); ws.close(); _startPolling(id) }
    ws.onclose = () => clearTimeout(timer)
    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data)
      _applyUpdate(data)
      if (data.status === 'done') setAudioUrl(`/api/audio/${id}`)
    }
  }, [_applyUpdate, _startPolling])

  const generate = useCallback(async (params) => {
    _stopAll()
    setJobId(null)
    setStatus('queued')
    setProgress(0)
    setQueuePos(null)
    setNClips(0)
    setClipNum(0)
    setError(null)
    setAudioUrl(null)
    setIsGenerating(true)

    try {
      const { data } = await axios.post('/api/generate', params)
      const id = data.job_id
      jobIdRef.current = id
      setJobId(id)
      _startWS(id)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Request failed')
      setStatus('failed')
      setIsGenerating(false)
    }
  }, [_stopAll, _startWS])

  const cancel = useCallback(async () => {
    const id = jobIdRef.current
    _stopAll()
    setIsGenerating(false)
    setStatus('cancelled')
    if (id) {
      try { await axios.post(`/api/cancel/${id}`) } catch {}
    }
  }, [_stopAll])

  const reset = useCallback(() => {
    _stopAll()
    jobIdRef.current = null
    setJobId(null)
    setStatus(null)
    setProgress(0)
    setQueuePos(null)
    setNClips(0)
    setClipNum(0)
    setError(null)
    setAudioUrl(null)
    setIsGenerating(false)
  }, [_stopAll])

  return {
    generate, cancel, reset,
    jobId, status, progress, queuePos,
    nClips, clipNum,
    error, audioUrl, isGenerating,
  }
}
