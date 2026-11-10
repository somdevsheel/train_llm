import { useState } from 'react';
import type { InspectResult, ModelInfo } from '../api';
import { inspect } from '../api';

interface Props {
  models: ModelInfo[];
}

/** Shows the model's probability distribution over the very next token. */
export function InspectTab({ models }: Props) {
  const [text, setText] = useState('The capital of France is');
  const [model, setModel] = useState<string>(models.find((m) => m.key === 'base')?.key ?? models[0]?.key ?? '');
  const [asChat, setAsChat] = useState(false);
  const [res, setRes] = useState<InspectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function go(t = text) {
    if (!t.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setRes(await inspect(model, t, asChat, 12));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function accept(token: string) {
    const next = text + token;
    setText(next);
    go(next);
  }

  const max = res?.tokens[0]?.prob ?? 1;
  const show = (t: string) => t.replace(/\n/g, '↵').replace(/ /g, '·') || '∅';

  return (
    <div className="tab inspect">
      <p className="lead">
        See what the model predicts next and how confident it is. Click a token to append it and predict again.
        With “wrap as a chat question”, it shows the first token of the assistant's reply.
      </p>
      <textarea
        className="prompt"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            go();
          }
        }}
        rows={3}
      />
      <div className="row">
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>
        <label className="check">
          <input type="checkbox" checked={asChat} onChange={(e) => setAsChat(e.target.checked)} />
          Wrap as a chat question
        </label>
        <span className="spacer" />
        <button className="primary" onClick={() => go()} disabled={loading || !text.trim()}>
          {loading ? 'Predicting…' : 'Predict next token'}
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {res && (
        <>
          <div className="meta-row">
            <span>{res.prompt_tokens} prompt tokens</span>
            <span>
              entropy {res.entropy} nats (like choosing among ~{res.perplexity_like} tokens)
            </span>
          </div>
          <div className="tokens-strip" title="How the model tokenized the end of your text">
            {res.tokenized.map((t, i) => (
              <span key={i} className="tok">
                {show(t)}
              </span>
            ))}
          </div>
          <ol className="probs">
            {res.tokens.map((t) => (
              <li key={t.id}>
                <button
                  className="prob-row"
                  onClick={() => accept(t.token)}
                  disabled={asChat}
                  title={asChat ? `token id ${t.id}` : `token id ${t.id} — click to append`}
                >
                  <span className="tok">{show(t.token)}</span>
                  <span className="bar">
                    <span style={{ width: `${(t.prob / max) * 100}%` }} />
                  </span>
                  <span className="num">{(t.prob * 100).toFixed(1)}%</span>
                </button>
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}
