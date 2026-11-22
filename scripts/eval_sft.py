"""Mean assistant-token loss of a model on a packed SFT file (e.g. val_mix.bin from prepare_sft.py).

    python scripts/eval_sft.py --ckpt checkpoints/sft-v3/model.pt --data data/sft/val_mix.bin
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.config import GPTConfig  # noqa: E402
from llm.model import GPT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.memmap(a.data, dtype=np.uint16, mode="r")
    mask = np.memmap(a.data[:-len(".bin")] + "_mask.bin", dtype=np.uint8, mode="r")
    for path in a.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ck["model_config"])
        model = GPT(cfg)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
        model.to(dev).eval()
        S = cfg.block_size + 1
        n = len(data) // S
        tot, cnt = 0.0, 0
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            for i in range(0, n, a.batch):
                buf = torch.from_numpy(np.array(data[i * S:min(n, i + a.batch) * S], dtype=np.int64)).view(-1, S)
                m = torch.from_numpy(np.array(mask[i * S:min(n, i + a.batch) * S])).view(-1, S)[:, 1:]
                x, y = buf[:, :-1].to(dev), buf[:, 1:].clone()
                y[m == 0] = -1
                logits = model.lm_head(model.norm_f(_body(model, x)))
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.to(dev).view(-1),
                                       ignore_index=-1, reduction="sum")
                tot += loss.item()
                cnt += int((y != -1).sum())
        print(f"{path}: {n} blocks, {cnt:,} target tokens, loss {tot / cnt:.4f} (ppl {math.exp(tot / cnt):.2f})",
              flush=True)


def _body(model, x):
    T = x.size(1)
    h = model.tok_emb(x)
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    for blk in model.blocks:
        h = blk(h, cos, sin)
    return h


if __name__ == "__main__":
    main()
