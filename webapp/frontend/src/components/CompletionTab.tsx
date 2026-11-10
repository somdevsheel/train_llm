import { useState } from 'react';
import type { GenStats, ModelInfo, Settings } from '../api';
import { useGeneration } from '../useGeneration';
import { StatsLine } from './Rendered';

interface Props {
  models: ModelInfo[];
  settings: Settings;
}

interface Column {
  key: string;
  text: string;
  stats: GenStats | null;
}

/** Raw text continuation; optionally runs the same prompt through every model side by side. */
export function CompletionTab({ models, settings }: Props) {
  const [prompt, setPrompt] = useState('The most important thing to understand about science is');
  const [compare, setCompare] = useState(true);
  const [single, setSingle] = useState<string>(models.find((m) => m.key === 'base')?.key ?? models[0]?.key ?? '');
  const [cols, setCols] = useState<Column[]>([]);
  const { run, stop, busy, error } = useGeneration();

  async function go() {
    if (!prompt.trim() || busy) return;
    const keys: string[] = compare ? models.map((m) => m.key) : single ? [single] : [];
    setCols(keys.map((key) => ({ key, text: '', stats: null })));
    for (const key of keys) {
      // one model at a time: the server shares one GPU
      const stats = await run({ ...settings, model: key, mode: 'completion', prompt }, (t) =>
        setCols((cs) => cs.map((c) => (c.key === key ? { ...c, text: c.text + t } : c))),
      );
      setCols((cs) => cs.map((c) => (c.key === key ? { ...c, stats } : c)));
      if (stats?.stop === 'cancelled') break;
    }
  }

  const label = (key: string) => models.find((m) => m.key === key)?.label ?? key;

  return (
    <div className="tab completion">
      <p className="lead">
        The model continues your text word by word, with no chat formatting. Comparing models side by side shows what
        fine-tuning changed.
      </p>
      <textarea className="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={4} />
      <div className="row">
        <label className="check">
          <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
          Compare all models
        </label>
        {!compare && (
          <select value={single} onChange={(e) => setSingle(e.target.value)}>
            {models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        )}
        <span className="spacer" />
        {busy ? (
          <button className="stop" onClick={stop}>
            Stop
          </button>
        ) : (
          <button className="primary" onClick={go} disabled={!prompt.trim()}>
            Generate
          </button>
        )}
      </div>
      {error && <div className="error">{error}</div>}
      <div className={`columns n${cols.length}`}>
        {cols.map((c) => (
          <div key={c.key} className="column">
            <h4>{label(c.key)}</h4>
            <div className="continuation">
              <span className="given">{prompt}</span>
              <span>{c.text}</span>
              {!c.stats && busy && <span className="cursor">▍</span>}
            </div>
            {c.stats && <StatsLine stats={c.stats} />}
          </div>
        ))}
      </div>
    </div>
  );
}
