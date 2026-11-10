import type { Settings } from '../api';
import { DEFAULT_SETTINGS } from '../api';

interface Props {
  value: Settings;
  onChange: (s: Settings) => void;
}

interface SliderDef {
  key: 'temperature' | 'top_k' | 'max_new_tokens' | 'repetition_penalty';
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
}

const SLIDERS: SliderDef[] = [
  { key: 'temperature', label: 'Temperature', min: 0.1, max: 1.5, step: 0.05, hint: 'Lower = safer, higher = more varied' },
  { key: 'top_k', label: 'Top-k', min: 1, max: 200, step: 1, hint: 'Sample only from the k likeliest tokens' },
  { key: 'max_new_tokens', label: 'Max new tokens', min: 16, max: 800, step: 8, hint: 'Length limit of the reply' },
  { key: 'repetition_penalty', label: 'Repetition penalty', min: 1, max: 1.6, step: 0.05, hint: '1.0 = off; discourages loops' },
];

export function SettingsPanel({ value, onChange }: Props) {
  return (
    <aside className="settings">
      <h3>Generation</h3>
      <label className="check">
        <input
          type="checkbox"
          checked={value.greedy}
          onChange={(e) => onChange({ ...value, greedy: e.target.checked })}
        />
        Greedy (deterministic)
      </label>
      {SLIDERS.map((s) => (
        <div key={s.key} className={`slider ${value.greedy && (s.key === 'temperature' || s.key === 'top_k') ? 'off' : ''}`}>
          <div className="slider-head">
            <span>{s.label}</span>
            <span className="num">{value[s.key]}</span>
          </div>
          <input
            type="range"
            min={s.min}
            max={s.max}
            step={s.step}
            value={value[s.key]}
            disabled={value.greedy && (s.key === 'temperature' || s.key === 'top_k')}
            onChange={(e) => onChange({ ...value, [s.key]: Number(e.target.value) })}
          />
          <div className="hint">{s.hint}</div>
        </div>
      ))}
      <button className="link" onClick={() => onChange(DEFAULT_SETTINGS)}>
        Reset to defaults
      </button>
    </aside>
  );
}
