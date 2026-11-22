"""Build an India-focused continued-pretraining corpus (GPT-2 BPE, uint16 .bin, docs separated by <|endoftext|>).

Sources (each capped by a token budget, files visited in a seeded random order):
  sangraha  ai4bharat/sangraha verified/eng   (English text from Indian sites; news-heavy, ~to 2024)
  wiki      wikimedia/wikipedia 20231101.en    (articles about India)
  fineweb   HuggingFaceFW/fineweb-edu sample/10BT, files from the END of the list (unseen by the base run)
val.bin       = the first --val_tokens of fineweb-edu sample-10BT (same stream as the base model's val -> comparable)
val_india.bin = held-out sangraha + wiki India docs

    python scripts/prepare_india.py --out_dir /kaggle/working/india --sangraha 2.6e8 --wiki 1e8 --fineweb 2.4e8
"""
import argparse
import json
import os
import random
import re
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, hf_hub_download

EOT = 50256
STATES = ("Andhra Pradesh|Arunachal Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|Himachal Pradesh|Jharkhand|"
          "Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur|Meghalaya|Mizoram|Nagaland|Odisha|Orissa|Punjab|"
          "Rajasthan|Sikkim|Tamil Nadu|Telangana|Tripura|Uttar Pradesh|Uttarakhand|West Bengal|Jammu and Kashmir|"
          "Ladakh|Puducherry|Chandigarh|Andaman")
CITIES = ("Delhi|Mumbai|Bombay|Kolkata|Calcutta|Chennai|Madras|Bengaluru|Bangalore|Hyderabad|Ahmedabad|Pune|Jaipur|"
          "Lucknow|Kanpur|Patna|Bhopal|Varanasi|Amritsar|Kochi|Thiruvananthapuram|Guwahati|Bhubaneswar|Ranchi|"
          "Dehradun|Shimla|Srinagar|Mysuru|Mysore|Agra|Nagpur|Indore|Surat|Coimbatore|Visakhapatnam")
OTHER = ("India|Indian|Indians|Bharat|Lok Sabha|Rajya Sabha|Vidhan Sabha|crore|lakh|rupees?|Rs\\.?|ISRO|"
         "Ganga|Ganges|Yamuna|Himalayas?|Diwali|Holi|Navratri|Durga Puja|Pongal|Onam|Eid|Bollywood|Mughal|"
         "Maratha|Vedic|Sanskrit|Hindi|Tamil|Telugu|Bengali|Marathi|Kannada|Malayalam|Gujarati|Punjabi|"
         "Nehru|Gandhi|Ambedkar|Panchayat|BJP|NDA|UPA|Modi|IIT|NCERT|CBSE|Reserve Bank of India|RBI|SEBI")
INDIA = re.compile(rf"\b(?:{STATES}|{CITIES}|{OTHER})\b")
PAK = re.compile(r"\b(?:Pakistan|Pakistani|Bangladesh|Sri Lanka|Nepal)\b")


def india_score(text):
    return len(INDIA.findall(text))


def keep_sangraha(text):
    if len(text) < 600:
        return False
    s = india_score(text)
    return s >= 4 and s > 2 * len(PAK.findall(text))


def keep_wiki(text):
    head = text[:3000]
    return len(text) >= 1500 and len(INDIA.findall(head)) >= 6


def _filter(args):
    kind, texts = args
    f = keep_sangraha if kind == "sangraha" else keep_wiki
    return [t for t in texts if f(t)]


class Writer:
    def __init__(self, path):
        self.f, self.n = open(path, "wb"), 0

    def add(self, toks):
        arr = np.concatenate([np.array([EOT] + t, dtype=np.uint16) for t in toks]) if toks else np.zeros(0, np.uint16)
        self.f.write(arr.tobytes())
        self.n += len(arr)

    def close(self):
        self.f.close()


def files_of(repo, prefix):
    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                  if f.startswith(prefix) and f.endswith(".parquet"))


def run_source(name, repo, files, budget, kind, train, val_india, val_budget, a, enc, pool, stats):
    got = 0
    for f in files:
        if got >= budget:
            break
        local = hf_hub_download(repo, f, repo_type="dataset", local_dir=a.cache_dir)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=8192, columns=["text"]):
            texts = batch.column(0).to_pylist()
            chunks = [texts[i::a.workers] for i in range(a.workers)]
            kept = [t for part in pool.map(_filter, [(kind, c) for c in chunks]) for t in part] if kind else texts
            stats[name]["seen"] += len(texts)
            stats[name]["kept"] += len(kept)
            if not kept:
                continue
            toks = enc.encode_ordinary_batch(kept, num_threads=a.workers)
            if val_india is not None and val_india.n < val_budget:
                val_india.add(toks[:max(1, len(toks) // 20)])
                toks = toks[max(1, len(toks) // 20):]
            n = sum(len(t) + 1 for t in toks)
            train.add(toks)
            got += n
            if got >= budget:
                break
        os.remove(local)
        stats[name]["files"] += 1
        stats[name]["tokens"] = got
        print(f"[{name}] {f}: {got / 1e6:.0f}M / {budget / 1e6:.0f}M tokens | kept {stats[name]['kept']:,} of "
              f"{stats[name]['seen']:,} docs | {time.time() - a.t0:.0f}s", flush=True)
    return got


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/india")
    p.add_argument("--cache_dir", default="/tmp/hf_cache")
    p.add_argument("--sangraha", type=float, default=2.6e8)
    p.add_argument("--wiki", type=float, default=1.0e8)
    p.add_argument("--fineweb", type=float, default=2.4e8)
    p.add_argument("--val_tokens", type=float, default=10e6)
    p.add_argument("--val_india_tokens", type=float, default=5e6)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(a.seed)
    stats = {k: {"seen": 0, "kept": 0, "files": 0, "tokens": 0} for k in ("val", "sangraha", "wiki", "fineweb")}
    train = Writer(os.path.join(a.out_dir, "train.bin"))
    val_india = Writer(os.path.join(a.out_dir, "val_india.bin"))
    fw = files_of("HuggingFaceFW/fineweb-edu", "sample/10BT/")
    with Pool(a.workers) as pool:
        # val.bin: same first docs of sample-10BT as the base model's val
        val = Writer(os.path.join(a.out_dir, "val.bin"))
        run_source("val", "HuggingFaceFW/fineweb-edu", fw[:1], a.val_tokens, None, val, None, 0, a, enc, pool, stats)
        val.close()
        sg = files_of("ai4bharat/sangraha", "verified/eng/")
        rng.shuffle(sg)
        run_source("sangraha", "ai4bharat/sangraha", sg, a.sangraha, "sangraha", train, val_india,
                   a.val_india_tokens * 0.6, a, enc, pool, stats)
        wk = files_of("wikimedia/wikipedia", "20231101.en/")
        rng.shuffle(wk)
        run_source("wiki", "wikimedia/wikipedia", wk, a.wiki, "wiki", train, val_india, a.val_india_tokens, a, enc,
                   pool, stats)
        run_source("fineweb", "HuggingFaceFW/fineweb-edu", fw[::-1], a.fineweb, None, train, None, 0, a, enc, pool,
                   stats)
    train.close()
    val_india.close()
    meta = {"tokenizer": "gpt2", "train_tokens": train.n, "val_tokens": stats["val"]["tokens"],
            "val_india_tokens": val_india.n, "sources": stats, "seconds": round(time.time() - a.t0)}
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print("Done:", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
