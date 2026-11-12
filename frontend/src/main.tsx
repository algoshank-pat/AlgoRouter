import { lazy, Suspense, useEffect, useRef, useState, type FormEvent } from 'react'
import { createRoot } from 'react-dom/client'
import { api, modelNames, type Comparison, type RouteResult } from './api'
import './styles.css'

const EvaluationPanel = lazy(() => import('./EvaluationPanel'))

const views = [
  ['route', 'Route a complaint'],
  ['compare', 'Compare models'],
  ['evaluation', 'Evaluation'],
] as const
type View = typeof views[number][0]

const samples = [
  ['Credit reporting', 'My credit report shows a loan I never opened. I disputed it with the credit bureau, but the loan still appears.'],
  ['Debt collection', 'A debt collector calls me every day about a bill I already paid. They have refused to send written proof of the debt.'],
  ['Credit card', 'My credit card statement includes a charge I did not authorize. The card issuer closed my dispute without a refund.'],
  ['Checking / savings', 'My bank charged an overdraft fee even though my paycheck deposit was available when the debit posted.'],
  ['Mortgage', 'My mortgage servicer applied my payment to the wrong account and sent a foreclosure notice.'],
  ['Student loan', 'My student loan servicer lost my income-driven repayment application and billed me the full monthly amount.'],
  ['Money transfer', 'Someone used my account to send an unauthorized Zelle transfer. The bank refused to reverse the transfer.'],
] as const

function ResultCard({ result }: { result: RouteResult }) {
  return <article className="panel result">
    <div className="section-heading">
      <p className="eyebrow">{modelNames[result.model]}</p>
      {result.urgent && <span className="badge">Urgent</span>}
    </div>
    <h2>{result.label}</h2>
    <p>{result.queue}</p>
    <dl><dt>Inference latency</dt><dd>{result.latency_ms.toFixed(1)} ms</dd></dl>
    {result.trace_url
      ? <a href={result.trace_url} target="_blank" rel="noreferrer">View trace in LangSmith ↗</a>
      : <span className="muted">LangSmith trace unavailable</span>}
  </article>
}

function Architecture() {
  return <section className="architecture" aria-labelledby="architecture-title">
    <p className="eyebrow" id="architecture-title">Architecture</p>
    <ol>{['CFPB data', 'Data prep', 'LoRA fine-tune', 'Model merge', 'FastAPI', 'React'].map(step =>
      <li key={step}>{step}</li>)}</ol>
    <p>FastAPI sends classification traces to LangSmith.</p>
  </section>
}

function App() {
  const [view, setView] = useState<View>('route')
  const [text, setText] = useState('')
  const [results, setResults] = useState<RouteResult[]>([])
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const request = useRef<AbortController | null>(null)

  useEffect(() => () => request.current?.abort(), [])

  function changeView(next: View) {
    if (next === view) return
    request.current?.abort()
    setView(next)
    setPending(false)
    setResults([])
    setError('')
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!text.trim() || pending || view === 'evaluation') return
    const controller = new AbortController()
    request.current = controller
    setPending(true)
    setResults([])
    setError('')
    try {
      if (view === 'compare') {
        const comparison = await api<Comparison>('/compare', controller.signal, text)
        setResults([comparison.baseline, comparison.finetuned])
      } else {
        setResults([await api<RouteResult>('/route', controller.signal, text)])
      }
    } catch (error) {
      if (!controller.signal.aborted) setError(error instanceof Error ? error.message : 'Request failed.')
    } finally {
      if (!controller.signal.aborted) setPending(false)
    }
  }

  return <div className="shell">
    <header className="masthead"><span>Routing lab</span><span>CFPB complaints / 2026</span></header>
    <main>
      <div className="hero">
        <p className="eyebrow">Qwen3-1.7B · LoRA rank 8</p>
        <h1>Bank Complaint Router</h1>
        <p>Classify complaints into seven specialist queues and see how fine-tuning changes the routing decision.</p>
      </div>
      <nav className="tabs" aria-label="Router views">{views.map(([key, label], index) =>
        <button key={key} type="button" aria-current={view === key ? 'page' : undefined} onClick={() => changeView(key)}>
          <span aria-hidden="true">0{index + 1}</span>{label}
        </button>)}</nav>

      {view === 'evaluation' ? <Suspense fallback={<p role="status">Loading evaluation…</p>}><EvaluationPanel /></Suspense> : <div className="stack">
        <form className="panel" onSubmit={submit}>
          <p className="eyebrow">{view === 'compare' ? 'One input / two models' : 'Fine-tuned model'}</p>
          <h2>{view === 'compare' ? 'Compare routing decisions' : 'Route a complaint'}</h2>
          <label htmlFor="complaint">Complaint narrative</label>
          <textarea id="complaint" value={text} disabled={pending} maxLength={20000} rows={5}
            placeholder="Paste a complaint or choose a synthetic sample below."
            onChange={event => { setText(event.target.value); setResults([]); setError('') }} />
          <p className="muted" id="sample-label">Synthetic samples</p>
          <div className="samples" role="group" aria-labelledby="sample-label">
            {samples.map(([label, sample]) => <button key={label} type="button" disabled={pending}
              onClick={() => { setText(sample); setResults([]); setError('') }}>{label}</button>)}
          </div>
          <div className="actions">
            <button className="primary" disabled={pending || !text.trim()}>
              {view === 'compare' ? 'Compare models' : 'Route complaint'}
            </button>
            {pending && <span role="status">Running inference… Model loading can take a moment.</span>}
          </div>
          {error && <p role="alert" className="error">{error}</p>}
        </form>
        <div className={results.length === 2 ? 'result-grid' : ''} aria-live="polite">
          {results.map(result => <ResultCard key={result.model} result={result} />)}
        </div>
      </div>}
      <Architecture />
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
