"""Build packed SFT data from a chat dataset (default: HuggingFaceTB/smol-smoltalk).

Writes to --out_dir:
  train.bin / val.bin            uint16 tokens, whole conversations packed into blocks of block_size+1
  train_mask.bin / val_mask.bin  uint8, 1 where the token is an assistant target
  meta.json

Conversations longer than a block are cut at a turn boundary (must keep >= 1 assistant turn).
--identity N adds N synthetic "who are you?" conversations (src/llm/identity.py) to train only,
so val stays the first --val_rows conversations of the dataset and val loss is comparable across runs.

    python scripts/prepare_sft.py --out_dir data/sft --max_rows 200000 --identity 1000
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.chat import EOS, encode_turns  # noqa: E402
from llm.identity import identity_conversations  # noqa: E402


def iter_rows(a):
    import pyarrow.parquet as pq
    if a.parquet:
        files = sorted(glob.glob(a.parquet))
    else:
        from huggingface_hub import HfApi, hf_hub_download
        names = sorted(f for f in HfApi().list_repo_files(a.repo, repo_type="dataset")
                       if f.startswith("data/train") and f.endswith(".parquet"))
        files = [hf_hub_download(a.repo, f, repo_type="dataset", local_dir=a.cache_dir) for f in names[:a.shards]]
    print(f"{len(files)} parquet files", flush=True)
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["messages"]):
            yield from batch.column(0).to_pylist()


def fit(turns, limit):
    """Keep the longest prefix of turns that fits in `limit` and ends with an assistant turn."""
    ids, mask, best = [], [], None
    for t, m in turns:
        if len(ids) + len(t) > limit:
            break
        ids, mask = ids + t, mask + m
        if any(m):
            best = (list(ids), list(mask))
    return best


class Packer:
    """First-fit packing of whole conversations into fixed blocks (pool of open blocks)."""

    def __init__(self, block, prefix, pool=256):
        self.block, self.pool = block, pool
        self.tok = open(prefix + ".bin", "wb")
        self.msk = open(prefix + "_mask.bin", "wb")
        self.open = []  # list of [ids, mask]
        self.blocks = self.targets = self.used = 0

    def add(self, ids, mask):
        for b in self.open:
            if len(b[0]) + len(ids) <= self.block:
                b[0] += ids
                b[1] += mask
                break
        else:
            self.open.append([list(ids), list(mask)])
        if len(self.open) > self.pool:  # retire the fullest block
            self._write(self.open.pop(max(range(len(self.open)), key=lambda i: len(self.open[i][0]))))

    def _write(self, b):
        ids, mask = b
        pad = self.block - len(ids)
        self.tok.write(np.array(ids + [EOS] * pad, dtype=np.uint16).tobytes())
        self.msk.write(np.array(mask + [0] * pad, dtype=np.uint8).tobytes())
        self.targets += sum(mask)
        self.used += len(ids)
        self.blocks += 1

    def close(self):
        for b in self.open:
            self._write(b)
        self.open = []
        self.tok.close()
        self.msk.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HuggingFaceTB/smol-smoltalk")
    p.add_argument("--parquet", default="", help="glob of local parquet files instead of downloading")
    p.add_argument("--shards", type=int, default=2, help="train parquet shards to download")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--val_rows", type=int, default=2000)
    p.add_argument("--identity", type=int, default=0, help="synthetic identity conversations added to train")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--out_dir", default="data/sft")
    p.add_argument("--cache_dir", default="data/hf_cache")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    block = a.block_size + 1
    val = Packer(block, os.path.join(a.out_dir, "val"))
    train = Packer(block, os.path.join(a.out_dir, "train"))
    t0, n, kept, cut = time.time(), 0, 0, 0
    for msgs in iter_rows(a):
        if a.max_rows and n >= a.max_rows:
            break
        n += 1
        turns = encode_turns(enc, msgs)
        full = sum(len(t) for t, _ in turns)
        ex = fit(turns, block)
        if ex is None:
            continue
        cut += len(ex[0]) < full
        kept += 1
        (val if kept <= a.val_rows else train).add(*ex)
        if n % 20000 == 0:
            print(f"{n:,} rows | {train.blocks:,} train blocks | {time.time() - t0:.0f}s", flush=True)
    ident = 0
    for msgs in identity_conversations(a.identity):
        ex = fit(encode_turns(enc, msgs), block)
        if ex is not None:
            train.add(*ex)
            ident += 1
    val.close()
    train.close()
    meta = {"repo": a.repo, "rows_read": n, "rows_kept": kept, "rows_truncated": cut, "identity_rows": ident,
            "block_size": a.block_size, "train_blocks": train.blocks, "val_blocks": val.blocks,
            "train_tokens": train.blocks * block, "fill": round(train.used / max(1, train.blocks * block), 3),
            "train_target_tokens": train.targets,
            "val_target_tokens": val.targets}
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print(f"Done in {time.time() - t0:.0f}s: {meta}", flush=True)


if __name__ == "__main__":
    main()
