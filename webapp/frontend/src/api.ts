export type Role = 'user' | 'assistant';

export interface Message {
  role: Role;
  content: string;
  stats?: GenStats;
}

export interface ModelInfo {
  key: string;
  label: string;
  about: string;
  params_M: number;
  context: number;
  device: string;
}

export interface Settings {
  temperature: number;
  top_k: number;
  max_new_tokens: number;
  repetition_penalty: number;
  greedy: boolean;
}

export interface GenStats {
  tokens: number;
  seconds: number;
  tok_per_s: number;
  stop: 'eos' | 'length' | 'cancelled';
  prompt_tokens?: number;
  context?: number;
  dropped?: number;
}

export interface GenRequest extends Settings {
  model: string;
  mode: 'chat' | 'completion';
  messages?: { role: Role; content: string }[];
  prompt?: string;
  system?: string;
  continue_text?: string;
  conversation_id?: string;
}

export interface InspectResult {
  tokens: { token: string; id: number; prob: number }[];
  prompt_tokens: number;
  entropy: number;
  perplexity_like: number;
  tokenized: string[];
}

export const DEFAULT_SETTINGS: Settings = {
  temperature: 0.7,
  top_k: 50,
  max_new_tokens: 250,
  repetition_penalty: 1.15,
  greedy: false,
};

export async function fetchModels(): Promise<ModelInfo[]> {
  const r = await fetch('/api/models');
  if (!r.ok) throw new Error(`models: HTTP ${r.status}`);
  return r.json();
}

export async function inspect(model: string, text: string, chat: boolean, top = 10): Promise<InspectResult> {
  const r = await fetch('/api/inspect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, text, chat, top }),
  });
  if (!r.ok) throw new Error(`inspect: HTTP ${r.status} ${await r.text()}`);
  return r.json();
}

/** Streams a generation. Calls onToken for each text delta; resolves with final stats. Abort with signal. */
export async function generate(
  req: GenRequest,
  onToken: (text: string) => void,
  signal: AbortSignal,
): Promise<GenStats> {
  const r = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`generate: HTTP ${r.status} ${await r.text()}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let start: { prompt_tokens?: number; context?: number; dropped?: number } = {};
  let stats: GenStats | null = null;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const event = /^event: (.*)$/m.exec(raw)?.[1];
      const data = /^data: (.*)$/m.exec(raw)?.[1];
      if (!event || data === undefined) continue;
      const payload = JSON.parse(data);
      if (event === 'start') start = payload;
      else if (event === 'token') onToken(payload.text);
      else if (event === 'done') stats = { ...payload, ...start };
    }
  }
  if (!stats) throw new Error('stream ended without a result');
  return stats;
}
