"""Generate text from a trained model.

    python scripts/generate.py --ckpt checkpoints/run/model.pt --prompt "The history of the Roman Empire"
    python scripts/generate.py --ckpt checkpoints/run/model.pt --interactive
    python scripts/generate.py --ckpt checkpoints/sft/model.pt --chat              # multi-turn chat
    python scripts/generate.py --ckpt checkpoints/sft/model.pt --chat --prompt "What is DNA?"
"""
import argparse
import os
import sys

import tiktoken
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.chat import prompt_ids  # noqa: E402
from llm.config import GPTConfig  # noqa: E402
from llm.model import GPT  # noqa: E402


def load(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/run/model.pt")
    p.add_argument("--prompt", default="The most important thing to understand about science is")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=None, help="default 0.8 (0.7 with --chat)")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--chat", action="store_true", help="fine-tuned model: chat template (+ multi-turn REPL)")
    p.add_argument("--system", default="", help="--chat: optional system prompt")
    p.add_argument("--rep_penalty", type=float, default=None, help="repetition penalty (default 1.0, 1.15 with --chat)")
    a = p.parse_args()
    if a.temperature is None:
        a.temperature = 0.7 if a.chat else 0.8
    if a.rep_penalty is None:
        a.rep_penalty = 1.15 if a.chat else 1.0

    if a.seed is not None:
        torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(a.ckpt, device)
    enc = tiktoken.get_encoding("gpt2")
    print(f"Loaded {model.num_params() / 1e6:.1f}M params on {device}")

    def run(prompt):
        idx = torch.tensor([enc.encode(prompt)], device=device)
        for _ in range(a.num_samples):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model.generate(idx, a.tokens, a.temperature, a.top_k, eos_id=50256,
                                     repetition_penalty=a.rep_penalty)
            print(enc.decode(out[0].tolist()).replace("<|endoftext|>", ""))
            print("-" * 60)

    def reply(messages):
        idx = torch.tensor([prompt_ids(enc, messages)[-model.cfg.block_size + a.tokens:]], device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model.generate(idx, a.tokens, a.temperature, a.top_k, eos_id=50256,
                                     repetition_penalty=a.rep_penalty)
        return enc.decode(out[0, idx.size(1):].tolist()).replace("<|endoftext|>", "").strip()

    if a.chat:
        base = [{"role": "system", "content": a.system}] if a.system else []
        if a.prompt and not a.interactive and "--prompt" in sys.argv:
            for _ in range(a.num_samples):
                print(reply(base + [{"role": "user", "content": a.prompt}]))
                print("-" * 60)
            return
        history = list(base)
        print("Chat mode. Empty line or Ctrl-D quits, /reset clears history.")
        while True:
            try:
                msg = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg:
                break
            if msg == "/reset":
                history = list(base)
                continue
            history.append({"role": "user", "content": msg})
            ans = reply(history)
            history.append({"role": "assistant", "content": ans})
            print(f"bot> {ans}\n")
        return

    if a.interactive:
        while True:
            try:
                prompt = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt:
                run(prompt)
    else:
        run(a.prompt)


if __name__ == "__main__":
    main()
