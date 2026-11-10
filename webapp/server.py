"""Local web app for testing the trained models (FastAPI backend + React/TypeScript UI).

    cd ~/kaggle-llm/webapp/frontend && npm install && npm run build   # once, builds the UI
    cd ~/kaggle-llm && python webapp/server.py                         # http://127.0.0.1:7860
    python webapp/server.py --port 8000 --device cpu

Loads checkpoints/sft/model.pt (chat) and checkpoints/124m-v2/model.pt (base) with the
project's own model code and streams generations to the page over server-sent events.
"""
import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time

import tiktoken
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from llm.chat import EOS, prompt_ids  # noqa: E402
from llm.config import GPTConfig  # noqa: E402
from llm.model import GPT  # noqa: E402

MODELS = {
    "chat": {"path": "checkpoints/sft/model.pt", "label": "Chat (fine-tuned)",
             "about": "SFT on 226K smol-smoltalk conversations; val loss 1.61 on assistant tokens"},
    "base": {"path": "checkpoints/124m-v2/model.pt", "label": "Base (pretrained only)",
             "about": "0.98B FineWeb-Edu tokens; val loss 3.43, HellaSwag 27.1%"},
}

app = FastAPI(title="kaggle-llm local tester")
LOG_DIR = os.environ.get("LLM_LOG_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
log_lock = threading.Lock()


def log_generation(req, ids, dropped, result):
    """Append one generation to logs/chats-YYYY-MM-DD.jsonl (local only) for later review."""
    os.makedirs(LOG_DIR, exist_ok=True)
    rec = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conversation_id": req.conversation_id or None,
        "model": req.model, "mode": req.mode,
        "settings": {k: getattr(req, k) for k in
                     ("temperature", "top_k", "max_new_tokens", "repetition_penalty", "greedy")},
        "system": req.system or None,
        "messages_sent": req.messages if req.mode == "chat" else None,
        "prompt": req.prompt if req.mode == "completion" else None,
        "continued_from": req.continue_text or None,
        "prompt_tokens": len(ids), "dropped_messages": dropped,
        **result,
    }
    path = os.path.join(LOG_DIR, time.strftime("chats-%Y-%m-%d.jsonl"))
    with log_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
enc = tiktoken.get_encoding("gpt2")
loaded = {}
gpu_lock = threading.Lock()
DEVICE = "cpu"


def load_models(device):
    for key, m in MODELS.items():
        path = os.path.join(ROOT, m["path"])
        if not os.path.exists(path):
            print(f"skip {key}: {path} not found")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = GPT(GPTConfig(**ck["model_config"]))
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        model = model.to(device).eval()
        if device == "cuda":
            model = model.half()
        loaded[key] = model
        print(f"loaded {key}: {model.num_params() / 1e6:.1f}M params on {device}")


class GenRequest(BaseModel):
    model: str = "chat"
    mode: str = "chat"                       # "chat" | "completion"
    messages: list[dict] = Field(default_factory=list)
    prompt: str = ""
    system: str = ""
    continue_text: str = ""                  # chat: resume this partial assistant reply instead of starting fresh
    conversation_id: str = ""                # groups the turns of one chat in the log
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=50304)
    max_new_tokens: int = Field(250, ge=1, le=1000)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    greedy: bool = False


class InspectRequest(BaseModel):
    model: str = "base"
    text: str
    chat: bool = False
    top: int = Field(10, ge=1, le=50)


def get_model(key):
    if key not in loaded:
        raise HTTPException(404, f"model '{key}' not loaded")
    return loaded[key]


def build_ids(req: GenRequest, block_size: int):
    """-> (prompt token ids, number of old messages dropped to fit the context window)."""
    if req.mode == "chat":
        msgs = [{"role": m["role"], "content": str(m["content"])} for m in req.messages
                if m.get("role") in ("user", "assistant")]
        if req.system.strip():
            msgs = [{"role": "system", "content": req.system.strip()}] + msgs
        tail = enc.encode_ordinary(req.continue_text) if req.continue_text else []
        budget = block_size - req.max_new_tokens
        dropped = 0

        def encode(ms):
            return prompt_ids(enc, ms) + tail

        ids = encode(msgs)
        while len(ids) > budget and len(msgs) > 1:   # drop the oldest turns to fit the context
            msgs.pop(1 if msgs[0]["role"] == "system" and len(msgs) > 2 else 0)
            dropped += 1
            ids = encode(msgs)
        return ids[-budget:], dropped
    ids = enc.encode(req.prompt, allowed_special={"<|endoftext|>"}) or [EOS]
    return ids[-(block_size - 1):], 0


@torch.no_grad()
def step_logits(model, idx):
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        logits, _ = model(idx[:, -model.cfg.block_size:])
    return logits[0, -1].float()


def pick(logits, gen_ids, req: GenRequest):
    if req.repetition_penalty != 1.0 and gen_ids:
        prev = torch.tensor(sorted(set(gen_ids)), device=logits.device)
        seen = logits[prev]
        logits[prev] = torch.where(seen > 0, seen / req.repetition_penalty, seen * req.repetition_penalty)
    if req.greedy or req.temperature == 0:
        return int(torch.argmax(logits))
    logits = logits / req.temperature
    if req.top_k:
        v, _ = torch.topk(logits, min(req.top_k, logits.numel()))
        logits[logits < v[-1]] = -float("inf")
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/generate")
async def generate(req: GenRequest, request: Request):
    model = get_model(req.model)
    ids, dropped = build_ids(req, model.cfg.block_size)

    async def stream():
        idx = torch.tensor([ids], device=DEVICE)
        gen, sent, t0, stop_reason = [], "", time.time(), "length"
        yield sse("start", {"prompt_tokens": len(ids), "context": model.cfg.block_size, "dropped": dropped})
        for _ in range(req.max_new_tokens):
            if await request.is_disconnected():
                stop_reason = "cancelled"
                break
            with gpu_lock:
                tok = pick(step_logits(model, idx), gen, req)
            if tok == EOS:
                stop_reason = "eos"
                break
            gen.append(tok)
            idx = torch.cat([idx, torch.tensor([[tok]], device=DEVICE)], dim=1)
            text = enc.decode(gen)
            if not text.endswith("�"):             # wait for complete UTF-8 characters
                yield sse("token", {"text": text[len(sent):]})
                sent = text
            await asyncio.sleep(0)
        dt = time.time() - t0
        result = {"tokens": len(gen), "seconds": round(dt, 2),
                  "tok_per_s": round(len(gen) / dt, 1) if dt > 0 else 0, "stop": stop_reason,
                  "text": enc.decode(gen)}
        try:
            log_generation(req, ids, dropped, {"reply": result["text"], **{k: v for k, v in result.items() if k != "text"}})
        except OSError as e:
            print(f"warning: could not write chat log: {e}")
        yield sse("done", result)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/inspect")
def inspect(req: InspectRequest):
    model = get_model(req.model)
    if req.chat:
        ids = prompt_ids(enc, [{"role": "user", "content": req.text}])
    else:
        ids = enc.encode(req.text, allowed_special={"<|endoftext|>"}) or [EOS]
    ids = ids[-model.cfg.block_size:]
    with gpu_lock:
        probs = F.softmax(step_logits(model, torch.tensor([ids], device=DEVICE)), dim=-1)
    p, i = torch.topk(probs, req.top)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    return {"tokens": [{"token": enc.decode([int(t)]), "id": int(t), "prob": float(q)} for q, t in zip(p, i)],
            "prompt_tokens": len(ids), "entropy": round(entropy, 3),
            "perplexity_like": round(math.exp(entropy), 1),
            "tokenized": [enc.decode([t]) for t in ids[-40:]]}


@app.get("/api/models")
def models():
    return [{"key": k, "label": MODELS[k]["label"], "about": MODELS[k]["about"],
             "params_M": round(m.num_params() / 1e6, 1), "context": m.cfg.block_size, "device": DEVICE}
            for k, m in loaded.items()]


DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.isdir(DIST):  # the built React app (npm run build); in dev, Vite serves it and proxies /api
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:
    @app.get("/")
    def no_ui():
        return {"error": "frontend not built: cd webapp/frontend && npm install && npm run build"}


def main():
    global DEVICE
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-log", action="store_true", help="don't write conversations to webapp/logs/")
    a = p.parse_args()
    DEVICE = a.device
    if a.no_log:
        globals()["log_generation"] = lambda *args, **kw: None
    torch.manual_seed(int(time.time()))
    load_models(DEVICE)
    if not loaded:
        sys.exit("no checkpoints found under checkpoints/")
    print(f"open http://{a.host}:{a.port}")
    if not a.no_log:
        print(f"logging conversations to {LOG_DIR}/ (disable with --no-log)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
