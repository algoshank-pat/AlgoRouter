import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, modelNames, type Variant } from './api'

type Score = { recall: number; 'f1-score': number }
type Matrix = { labels: string[]; counts: number[][] }
type ModelMetrics = {
  accuracy: number
  classification_report: Record<string, Score>
  confusion_matrix: Matrix
}
type Evaluation = {
  comparison: {
    examples: number
    labels: string[]
    high_urgency_labels: string[]
    models: Record<'baseline-qwen3-1.7b' | 'finetuned-lora-r8', ModelMetrics>
  }
  april_finetuned_lora?: ModelMetrics
  langsmith_experiments?: Record<string, { name: string; url: string }>
}
// April few-shot audit: 941 exact matches on the 2,800-row holdout.
const aprilFewShotAccuracy = 941 / 2800
const percent = (value: number) => `${(value * 100).toFixed(2)}%`
const shortLabel: Record<string, string> = {
  'Credit reporting': 'Credit report',
  'Debt collection': 'Debt collection',
  'Credit card': 'Credit card',
  'Checking or savings account': 'Checking / savings',
  Mortgage: 'Mortgage',
  'Student loan': 'Student loan',
  'Money transfer': 'Money transfer',
  '<invalid>': 'Invalid',
}

function Heatmap({ matrix, urgent, name }: { matrix: Matrix; urgent: string[]; name: string }) {
  const max = Math.max(1, ...matrix.counts.flat())
  return <div className="table-scroll" tabIndex={0} aria-label="Scrollable confusion matrix">
    <table className="heatmap">
      <caption>{name} · rows are actual queues, columns are predictions</caption>
      <thead><tr><th scope="col">Actual / Predicted</th>{matrix.labels.map(label =>
        <th key={label} scope="col">{shortLabel[label] || label}</th>)}</tr></thead>
      <tbody>{matrix.labels.map((label, row) => <tr key={label}>
        <th scope="row" className={urgent.includes(label) ? 'priority' : ''}>
          {shortLabel[label] || label}{urgent.includes(label) && <small>Urgent</small>}
        </th>
        {matrix.counts[row].map((count, column) => <td key={matrix.labels[column]} style={{
          backgroundColor: `rgba(17, 111, 102, ${count / max * 0.84})`,
          color: count / max > 0.63 ? 'white' : undefined,
        }}>{count}</td>)}
      </tr>)}</tbody>
    </table>
  </div>
}

export default function EvaluationPanel() {
  const [data, setData] = useState<Evaluation>()
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [metric, setMetric] = useState<keyof Score>('recall')
  const [matrixModel, setMatrixModel] = useState<Variant>('finetuned')

  useEffect(() => {
    const controller = new AbortController()
    api<Evaluation>('/eval', controller.signal).then(setData).catch((error: Error) => {
      if (!controller.signal.aborted) setError(error.message)
    })
    return () => controller.abort()
  }, [attempt])

  if (error) return <section className="panel" role="alert">
    <h2>Could not load evaluation</h2><p>{error}</p>
    <button onClick={() => { setError(''); setAttempt(attempt + 1) }}>Retry</button>
  </section>
  if (!data) return <p role="status">Loading evaluation results…</p>

  const { comparison, april_finetuned_lora: april, langsmith_experiments: experiments } = data
  const baseline = comparison.models['baseline-qwen3-1.7b']
  const tuned = comparison.models['finetuned-lora-r8']
  const urgent = comparison.high_urgency_labels
  const chart = comparison.labels.map(label => ({
    label: shortLabel[label] + (urgent.includes(label) ? ' *' : ''),
    baseline: baseline.classification_report[label][metric] * 100,
    finetuned: tuned.classification_report[label][metric] * 100,
  }))

  return <div className="stack">
    <section className="panel">
      <p className="eyebrow">01 / Held-out results</p><h2>January–February 2026</h2>
      <p>{comparison.examples} complaints across seven queues. The Qwen base result uses the original zero-shot prompt.</p>
      <div className="stats">
        <div><span>Qwen base accuracy</span><strong>{percent(baseline.accuracy)}</strong></div>
        <div><span>Fine-tuned accuracy</span><strong>{percent(tuned.accuracy)}</strong></div>
        <div><span>Accuracy gain</span><strong>+{((tuned.accuracy - baseline.accuracy) * 100).toFixed(2)} pp</strong></div>
      </div>
      {experiments && <div className="links">{Object.entries(experiments).map(([key, experiment]) =>
        <a key={key} href={experiment.url} target="_blank" rel="noreferrer">
          {key === 'baseline-qwen3-1.7b' ? 'Qwen base' : 'Fine-tuned'} experiment ↗
        </a>)}</div>}
    </section>

    {april && <section className="panel">
      <p className="eyebrow">02 / Independent holdout</p><h2>April 2026</h2>
      <p>The January–February checkpoint was tested on 2,800 April complaints. The April Qwen base prompt used one example per queue.</p>
      <div className="stats">
        <div><span>Few-shot Qwen base</span><strong>{percent(aprilFewShotAccuracy)}</strong></div>
        <div><span>Fine-tuned accuracy</span><strong>{percent(april.accuracy)}</strong></div>
        <div><span>Accuracy gain</span><strong>+{((april.accuracy - aprilFewShotAccuracy) * 100).toFixed(2)} pp</strong></div>
      </div>
      <p className="muted">Fine-tuned recall · Mortgage {percent(april.classification_report.Mortgage.recall)} · Money transfer {percent(april.classification_report['Money transfer'].recall)}</p>
    </section>}

    <section className="panel">
      <div className="section-heading"><div><p className="eyebrow">January–February</p><h2>Performance by queue</h2></div>
        <label>Metric <select value={metric} onChange={event => setMetric(event.target.value as keyof Score)}>
          <option value="recall">Recall</option><option value="f1-score">F1 score</option>
        </select></label>
      </div>
      <p className="muted">* High-urgency queue</p>
      <ResponsiveContainer className="queue-chart" width="100%" height={350} initialDimension={{ width: 700, height: 350 }}>
        <BarChart data={chart} layout="vertical" margin={{ right: 18 }}>
          <CartesianGrid stroke="#e5e2da" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tickFormatter={value => `${value}%`} />
          <YAxis type="category" dataKey="label" width={128} tick={{ fontSize: 11 }} />
          <Tooltip formatter={value => `${Number(value).toFixed(2)}%`} />
          <Legend />
          <Bar dataKey="baseline" name={modelNames.baseline} fill="#66748e" radius={[0, 4, 4, 0]} />
          <Bar dataKey="finetuned" name={modelNames.finetuned} fill="#116f66" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </section>

    <section className="panel">
      <div className="section-heading"><div><p className="eyebrow">January–February</p><h2>Confusion matrix</h2></div>
        <label>Model <select value={matrixModel} onChange={event => setMatrixModel(event.target.value as Variant)}>
          <option value="finetuned">{modelNames.finetuned}</option><option value="baseline">{modelNames.baseline}</option>
        </select></label>
      </div>
      <Heatmap matrix={(matrixModel === 'baseline' ? baseline : tuned).confusion_matrix}
        urgent={urgent} name={modelNames[matrixModel]} />
    </section>

    <section className="panel">
      <p className="eyebrow">Training / LoRA rank 8</p><h2>Training loss</h2>
      <img className="loss" src={import.meta.env.BASE_URL + 'training-loss.png'}
        alt="Training loss curve exported by the Colab fine-tuning run" loading="lazy" />
    </section>
  </div>
}
