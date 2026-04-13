import { useState, useEffect } from 'react'
import axios from 'axios'

export function useVocab() {
  const [ragas, setRagas] = useState([])
  const [talas, setTalas] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      axios.get('/api/ragas'),
      axios.get('/api/talas'),
    ]).then(([r, t]) => {
      setRagas(r.data)
      setTalas(t.data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  return { ragas, talas, loading }
}
