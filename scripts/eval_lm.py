"""Language-model loss of checkpoints on plain token files (prepare_data.py / prepare_india.py output).

Evaluates the first --blocks non-overlapping windows of --ctx tokens (default: the model's block_size),
so different models / context lengths see the same text.

    python scripts/eval_lm.py --ckpt base/model.pt run/model.pt --data val.bin val_india.bin --ctx 1024 4096
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.config import GPTConfig  # noqa: E402
from llm.model import GPT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--ctx", type=int, nargs="+", default=[0], help="context lengths to test (0 = block_size)")
    p.add_argument("--tokens", type=int, default=4_194_304, help="tokens evaluated per file")
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for path in a.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ck["model_config"])
        for ctx in a.ctx:
            ctx = ctx or cfg.block_size
            if ctx > cfg.block_size:
                cfg_eval = GPTConfig(**{**ck["model_config"], "block_size": ctx})  # positions beyond training
            else:
                cfg_eval = cfg
            model = GPT(cfg_eval)
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
            model.to(dev).eval()
            for data_path in a.data:
                data = np.memmap(data_path, dtype=np.uint16, mode="r")
                n = min(a.tokens, len(data) - 1) // ctx
                tot = cnt = 0
                with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    for i in range(0, n, a.batch):
                        j = min(n, i + a.batch)
                        buf = torch.from_numpy(np.array(data[i * ctx:j * ctx + 1], dtype=np.int64))
                        x = torch.stack([buf[k * ctx:(k + 1) * ctx] for k in range(j - i)]).to(dev)
                        y = torch.stack([buf[k * ctx + 1:(k + 1) * ctx + 1] for k in range(j - i)]).to(dev)
                        _, loss = model(x, y)
                        tot += loss.item() * y.numel()
                        cnt += y.numel()
                print(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} ctx={ctx} "
                      f"{os.path.basename(data_path)}: loss {tot / cnt:.4f} (ppl {math.exp(tot / cnt):.1f})", flush=True)


if __name__ == "__main__":
    main()
