# LLM Tester (local web app)

A React + TypeScript UI over a FastAPI server that runs the trained checkpoints on your GPU.

| Tab | What it does |
|---|---|
| Chat | Multi-turn chat with the fine-tuned model, optional system prompt, streaming replies, Stop button |
| Completion | Raw text continuation; runs the base and chat models side by side on the same prompt |
| Inspect | Top next-token probabilities and entropy; click a token to append it and predict again |

Settings (temperature, top-k, max tokens, repetition penalty, greedy) apply to Chat and Completion and are remembered in the browser.

## Run

```bash
# once: build the UI
cd ~/kaggle-llm/webapp/frontend && npm install && npm run build

# start the server (loads checkpoints/sft and checkpoints/124m-v2)
cd ~/kaggle-llm && python webapp/server.py      # open http://127.0.0.1:7860
```

Options: `--port 8000`, `--device cpu`, `--no-log`. The server listens on 127.0.0.1 only.

## Conversation logs

Every chat and completion request is appended to `webapp/logs/chats-YYYY-MM-DD.jsonl`. Each record holds the messages the model actually saw, the settings, the reply, and its stats. Turns of one chat share a `conversation_id`; Clear starts a new one. Logs stay on this machine. Turn logging off with `--no-log`.

```bash
python webapp/show_logs.py            # today's conversations, readable
python webapp/show_logs.py --last 3   # only the last 3
```

## Develop the UI with hot reload

```bash
cd ~/kaggle-llm && python webapp/server.py      # API on :7860
cd ~/kaggle-llm/webapp/frontend && npm run dev  # UI on http://127.0.0.1:5173, /api proxied to :7860
```

## API

- `GET /api/models`: loaded models
- `POST /api/generate`: server-sent events `start`, `token`, `done` (`{model, mode: "chat"|"completion", messages | prompt, system, temperature, top_k, max_new_tokens, repetition_penalty, greedy}`)
- `POST /api/inspect`: `{model, text, chat, top}` returns the top next tokens with probabilities

Generation recomputes the full context each step (no KV cache); at 124M parameters that is still ~100+ tokens/s on a laptop GPU.
