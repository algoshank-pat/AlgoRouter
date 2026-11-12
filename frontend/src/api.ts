export type Variant = 'baseline' | 'finetuned'

export type RouteResult = {
  label: string
  queue: string
  urgent: boolean
  latency_ms: number
  model: Variant
  trace_url: string | null
}

export type Comparison = Record<Variant, RouteResult>

export const modelNames = { baseline: 'Qwen base', finetuned: 'Fine-tuned LoRA' }

export async function api<T>(path: string, signal: AbortSignal, text?: string): Promise<T> {
  const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')
  const response = await fetch(base + path, {
    signal,
    ...(text === undefined ? {} : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }),
  })
  const payload = await response.json().catch(() => null)
  if (!response.ok || payload === null) {
    const detail = payload?.detail
    throw new Error(typeof detail === 'string' ? detail : `Backend request failed (HTTP ${response.status}).`)
  }
  return payload as T
}
