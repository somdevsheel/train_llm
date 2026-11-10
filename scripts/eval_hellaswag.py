"""HellaSwag (validation, 10,042 examples) for any causal LM loadable by transformers.

Each example has a context and 4 candidate endings. The model scores each ending by the
average log-loss of its tokens given the context; the lowest loss is its answer (acc_norm,
the length-normalized metric). Random guessing scores 25%.

    python scripts/eval_hellaswag.py --models hf_export/base openai-community/gpt2
    python scripts/eval_hellaswag.py --models hf_export/base --limit 500   # quick check
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_examples(limit):
    path = hf_hub_download("Rowan/hellaswag", "data/validation-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path, columns=["ctx", "endings", "label"]).to_pylist()
    return rows[:limit] if limit else rows


@torch.no_grad()
def evaluate(name, examples, device, batch_examples):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    correct = correct_raw = 0
    t0 = time.time()
    for b in range(0, len(examples), batch_examples):
        chunk = examples[b:b + batch_examples]
        seqs, spans = [], []
        for ex in chunk:
            ctx = tok.encode(ex["ctx"])
            for end in ex["endings"]:
                e = tok.encode(" " + end)
                seqs.append(ctx + e)
                spans.append((len(ctx), len(ctx) + len(e)))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.eos_token_id or 0, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        ids = ids.to(device)
        logits = model(ids).logits.float()
        logp = -F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")  # (N, T-1)
        totals, means = [], []
        for i, (s, e) in enumerate(spans):
            lp = logp[i, s - 1:e - 1]  # predictions of ending tokens
            totals.append(lp.sum().item())
            means.append(lp.mean().item())
        for j, ex in enumerate(chunk):
            label = int(ex["label"])
            m, t = means[4 * j:4 * j + 4], totals[4 * j:4 * j + 4]
            correct += max(range(4), key=lambda k: m[k]) == label
            correct_raw += max(range(4), key=lambda k: t[k]) == label
    n = len(examples)
    res = {"model": name, "examples": n, "acc_norm": correct / n, "acc": correct_raw / n,
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
           "seconds": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_examples", type=int, default=8)
    p.add_argument("--out", default="")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = load_examples(a.limit)
    results = [evaluate(m, examples, device, a.batch_examples) for m in a.models]
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
