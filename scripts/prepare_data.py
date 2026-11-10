"""Download + tokenize a corpus with the GPT-2 BPE tokenizer into uint16 .bin files.

Outputs <out_dir>/train.bin, val.bin, meta.json. Documents are separated by <|endoftext|>.

Examples:
    python scripts/prepare_data.py --dataset fineweb-edu --max_tokens 3e9 --out_dir data/fineweb
    python scripts/prepare_data.py --dataset tinystories --out_dir data/tinystories
    python scripts/prepare_data.py --dataset textfile --input_file my.txt --val_tokens 20000 --out_dir data/mine
"""
import argparse
import json
import os
import time

import numpy as np
import tiktoken

EOT = 50256  # <|endoftext|>


def fineweb_edu(cache_dir, subset):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo = "HuggingFaceFW/fineweb-edu"
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(f"sample/{subset}/") and f.endswith(".parquet"))
    print(f"fineweb-edu/{subset}: {len(files)} parquet files", flush=True)
    for f in files:
        local = hf_hub_download(repo, f, repo_type="dataset", local_dir=cache_dir)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=2048, columns=["text"]):
            yield batch.column(0).to_pylist()
        os.remove(local)  # keep disk usage low


def tinystories(cache_dir):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt",
                           repo_type="dataset", local_dir=cache_dir)
    yield from _split_file(path, "<|endoftext|>")


def textfile(path):
    yield from _split_file(path, "\n\n")


def _split_file(path, sep, batch_docs=2048):
    buf, docs = "", []
    with open(path, encoding="utf-8", errors="ignore") as f:
        while chunk := f.read(1 << 24):
            parts = (buf + chunk).split(sep)
            buf = parts.pop()
            docs.extend(p.strip() for p in parts if p.strip())
            while len(docs) >= batch_docs:
                yield docs[:batch_docs]
                docs = docs[batch_docs:]
    if buf.strip():
        docs.append(buf.strip())
    if docs:
        yield docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fineweb-edu", "tinystories", "textfile"], default="fineweb-edu")
    p.add_argument("--subset", default="sample-10BT", help="fineweb-edu subset folder, e.g. sample-10BT")
    p.add_argument("--input_file", default="")
    p.add_argument("--out_dir", default="data/fineweb")
    p.add_argument("--cache_dir", default="/tmp/hf_cache")
    p.add_argument("--max_tokens", type=float, default=3e9, help="train tokens to write")
    p.add_argument("--val_tokens", type=float, default=10e6)
    a = p.parse_args()

    # HF dataset folder is "sample/10BT"
    subset = a.subset.replace("sample-", "")
    stream = {"fineweb-edu": lambda: fineweb_edu(a.cache_dir, subset),
              "tinystories": lambda: tinystories(a.cache_dir),
              "textfile": lambda: textfile(a.input_file)}[a.dataset]()

    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(a.out_dir, exist_ok=True)
    val_path, train_path = os.path.join(a.out_dir, "val.bin"), os.path.join(a.out_dir, "train.bin")
    for path in (val_path, train_path):
        if os.path.exists(path):
            os.remove(path)

    val_target, train_target = int(a.val_tokens), int(a.max_tokens)
    n_val = n_train = n_docs = 0
    next_report, t0 = 50_000_000, time.time()
    fval, ftrain = open(val_path, "ab"), open(train_path, "ab")
    threads = os.cpu_count() or 4

    for docs in stream:
        toks = enc.encode_ordinary_batch(docs, num_threads=threads)
        arr = np.concatenate([np.array([EOT] + t, dtype=np.uint16) for t in toks])
        n_docs += len(docs)
        if n_val < val_target:  # fill validation first
            take = min(len(arr), val_target - n_val)
            fval.write(arr[:take].tobytes())
            n_val += take
            arr = arr[take:]
        if len(arr):
            take = min(len(arr), train_target - n_train)
            ftrain.write(arr[:take].tobytes())
            n_train += take
        if n_train >= next_report:
            rate = n_train / (time.time() - t0)
            print(f"{n_train / 1e6:,.0f}M train tokens | {n_docs:,} docs | {rate / 1e6:.2f}M tok/s", flush=True)
            next_report += 50_000_000
        if n_train >= train_target:
            break

    fval.close()
    ftrain.close()
    meta = {"dataset": a.dataset, "tokenizer": "gpt2", "train_tokens": n_train,
            "val_tokens": n_val, "docs": n_docs}
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s: {meta}")


if __name__ == "__main__":
    main()
