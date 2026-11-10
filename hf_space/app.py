"""Chat demo for kaggle-llm-124m-chat, a 124M model trained from scratch on free Kaggle GPUs."""
import os
import threading

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

REPO = "somdevsheel/kaggle-llm-124m-chat"
CONTEXT = 1024

torch.set_num_threads(os.cpu_count() or 2)
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(
    REPO, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()


def build_prompt(message, history, max_new_tokens):
    msgs = [{"role": m["role"], "content": str(m["content"])}
            for m in history if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    msgs.append({"role": "user", "content": message})
    budget = CONTEXT - max_new_tokens
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    while len(ids) > budget and len(msgs) > 1:  # drop the oldest turns to fit the context
        msgs.pop(0)
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return torch.tensor([ids[-budget:]], device=device)


def respond(message, history, temperature, max_new_tokens, repetition_penalty):
    max_new_tokens = int(max_new_tokens)
    ids = build_prompt(message, history, max_new_tokens)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), streamer=streamer,
                  max_new_tokens=max_new_tokens, do_sample=True, temperature=float(temperature), top_k=50,
                  repetition_penalty=float(repetition_penalty), eos_token_id=tok.eos_token_id,
                  pad_token_id=tok.eos_token_id)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    text = ""
    for piece in streamer:
        text += piece
        yield text


DESCRIPTION = """
A **124M-parameter** LLaMA-style model trained **from scratch** on free Kaggle GPUs (2x T4):
pretrained on 0.98B tokens of FineWeb-Edu, then fine-tuned on 226K chat conversations.
[Chat model](https://huggingface.co/somdevsheel/kaggle-llm-124m-chat) ·
[Base model](https://huggingface.co/somdevsheel/kaggle-llm-124m-base)

It is ~1,500x smaller than frontier models: it follows instructions and writes short code,
but it **often states wrong facts with confidence**. Don't rely on its answers.
"""

demo = gr.ChatInterface(
    respond,
    type="messages",
    title="kaggle-llm-124m-chat",
    description=DESCRIPTION,
    additional_inputs=[
        gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(32, 400, value=250, step=8, label="Max new tokens"),
        gr.Slider(1.0, 1.5, value=1.15, step=0.05, label="Repetition penalty"),
    ],
    examples=[
        ["Write a Python function that reverses a string."],
        ["What is photosynthesis? Explain it simply."],
        ["Give me three tips for staying focused while studying."],
        ["Write a short poem about the ocean."],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
