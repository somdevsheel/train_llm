import { pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

const MODEL = 'somdevsheel/prajna-ai-124m-chat';
// pin the model commit: Transformers.js caches files by URL, so a fixed 'main' URL keeps
// serving old weights from the browser cache after the repo is updated. Bump on each release.
const REVISION = '1d3845f0df6cdba0fedfc80c96d0a4f416bae64a'; // v5
let generator = null;

async function load() {
  generator = await pipeline('text-generation', MODEL, {
    revision: REVISION,
    dtype: 'q8',
    device: 'wasm',
    progress_callback: (p) => self.postMessage({ type: 'progress', data: p }),
  });
  // Transformers.js 3.0.2 applies the repetition penalty once per *occurrence* of a token,
  // which compounds on spaces/newlines and wrecks code. Match Python transformers instead:
  // penalize each distinct token once.
  const orig = generator.model._get_logits_processor.bind(generator.model);
  generator.model._get_logits_processor = (...args) => {
    const list = orig(...args);
    for (const p of list.processors ?? list) {
      // detected by shape: class names are minified in the browser bundle
      if (Object.keys(p).length === 1 && typeof p.penalty === 'number') {
        p._call = function (input_ids, logits) {
          for (let i = 0; i < input_ids.length; ++i) {
            const data = logits[i].data;
            for (const t of new Set(Array.from(input_ids[i], Number))) {
              data[t] = data[t] < 0 ? data[t] * this.penalty : data[t] / this.penalty;
            }
          }
          return logits;
        };
      }
    }
    return list;
  };
  self.postMessage({ type: 'ready' });
}

// keep the prompt inside the 4096-token context: drop the oldest turns first
function fit(messages, maxNew) {
  const budget = 4096 - maxNew;
  let msgs = messages.slice();
  while (msgs.length > 1) {
    const ids = generator.tokenizer.apply_chat_template(msgs, { add_generation_prompt: true, tokenize: true, return_tensor: false });
    if (ids.length <= budget) break;
    msgs = msgs.slice(1);
  }
  return msgs;
}

self.onmessage = async (e) => {
  const { type } = e.data;
  try {
    if (type === 'load') return await load();
    if (type === 'generate') {
      const { messages, options } = e.data;
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => self.postMessage({ type: 'token', text }),
      });
      const t0 = performance.now();
      const out = await generator(fit(messages, options.max_new_tokens), {
        ...options, top_k: 50, streamer,
      });
      const reply = out[0].generated_text.at(-1).content;
      const n = generator.tokenizer.encode(reply).length;
      self.postMessage({ type: 'done', reply, tps: n / ((performance.now() - t0) / 1000) });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
