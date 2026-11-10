import { useEffect, useRef, useState } from 'react';
import type { GenStats, Message, Settings } from '../api';
import { useGeneration } from '../useGeneration';
import { Rendered, StatsLine } from './Rendered';

const EXAMPLES = [
  'Write a Python function that reverses a string.',
  'What is photosynthesis? Explain it simply.',
  'Give me three tips for staying focused while studying.',
  'Write a short poem about the ocean.',
];

// How many previous question/answer pairs to send with each new question.
// The model only sees 1024 tokens and gives shorter answers deep into a long prompt
// (its fine-tuning data was cut to fit 1024 tokens), so a short memory keeps answers complete.
const MEMORY_OPTIONS = [
  { value: 0, label: 'Nothing (each question independent)' },
  { value: 1, label: 'Last exchange (recommended)' },
  { value: 2, label: 'Last 2 exchanges' },
  { value: 4, label: 'Last 4 exchanges' },
  { value: -1, label: 'Everything (answers get shorter)' },
];

interface Props {
  model: string;
  settings: Settings;
}

function mergeStats(a: GenStats | undefined, b: GenStats | null): GenStats | undefined {
  if (!b) return a;
  if (!a) return b;
  const tokens = a.tokens + b.tokens;
  const seconds = +(a.seconds + b.seconds).toFixed(2);
  return { ...b, tokens, seconds, tok_per_s: +(tokens / seconds).toFixed(1) };
}

export function ChatTab({ model, settings }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [system, setSystem] = useState('');
  const [memory, setMemory] = useState(() => {
    try {
      const v = Number(localStorage.getItem('llm-memory') ?? 1);
      return Number.isFinite(v) ? v : 1;
    } catch {
      return 1;
    }
  });
  const { run, stop, busy, error } = useGeneration();
  const newId = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
  const [conversationId, setConversationId] = useState(newId);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem('llm-memory', String(memory));
    } catch {
      /* storage unavailable */
    }
  }, [memory]);

  /** History to send: the last `memory` exchanges before the final user message. */
  function window_(history: Message[]): Message[] {
    if (memory < 0) return history;
    return history.slice(-(memory * 2 + 1));
  }

  const updateLast = (fn: (m: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])]);

  async function send(text: string) {
    text = text.trim();
    if (!text || busy) return;
    const history: Message[] = [...messages, { role: 'user', content: text }];
    setMessages([...history, { role: 'assistant', content: '' }]);
    setInput('');
    const stats = await run(
      {
        ...settings,
        model,
        mode: 'chat',
        system,
        conversation_id: conversationId,
        messages: window_(history).map(({ role, content }) => ({ role, content })),
      },
      (t) => updateLast((m) => ({ ...m, content: m.content + t })),
    );
    updateLast((m) => ({ ...m, content: m.content.trim(), stats: stats ?? undefined }));
  }

  /** Resume the last reply after it hit the token limit. */
  async function continueLast() {
    const last = messages[messages.length - 1];
    if (busy || !last || last.role !== 'assistant') return;
    const history = messages.slice(0, -1);
    const stats = await run(
      {
        ...settings,
        model,
        mode: 'chat',
        system,
        conversation_id: conversationId,
        messages: window_(history).map(({ role, content }) => ({ role, content })),
        continue_text: last.content,
      },
      (t) => updateLast((m) => ({ ...m, content: m.content + t })),
    );
    updateLast((m) => ({ ...m, content: m.content.trimEnd(), stats: mergeStats(m.stats, stats) }));
  }

  const last = messages[messages.length - 1];
  const canContinue = !busy && last?.role === 'assistant' && last.stats?.stop === 'length';

  return (
    <div className="tab chat">
      <div className="chat-options">
        <label className="inline">
          Remember
          <select value={memory} onChange={(e) => setMemory(Number(e.target.value))}>
            {MEMORY_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <details className="system">
          <summary>System prompt {system ? '(set)' : '(optional)'}</summary>
          <textarea
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            placeholder="e.g. You are a concise assistant."
            rows={2}
          />
        </details>
      </div>

      <div className="log">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask the fine-tuned model anything. Try an example:</p>
            <div className="chips">
              {EXAMPLES.map((e) => (
                <button key={e} className="chip" onClick={() => send(e)}>
                  {e}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              {m.content ? <Rendered text={m.content} /> : <span className="cursor">▍</span>}
            </div>
            {m.stats && <StatsLine stats={m.stats} />}
            {i === messages.length - 1 && canContinue && (
              <button className="continue" onClick={continueLast}>
                Continue ↓
              </button>
            )}
          </div>
        ))}
        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          placeholder="Message the model… (Enter to send, Shift+Enter for a new line)"
          rows={2}
        />
        <div className="composer-buttons">
          {busy ? (
            <button type="button" className="stop" onClick={stop}>
              Stop
            </button>
          ) : (
            <button type="submit" className="primary" disabled={!input.trim()}>
              Send
            </button>
          )}
          <button type="button" className="link" disabled={busy || !messages.length} onClick={() => {
              setMessages([]);
              setConversationId(newId());
            }}>
            Clear
          </button>
        </div>
      </form>
    </div>
  );
}
