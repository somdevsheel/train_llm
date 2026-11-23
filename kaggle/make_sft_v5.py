"""Build kaggle/sft-v5/: Prajna SFT v5 = full chat SFT of the 4K-context India base (prajna-ctx4k).

Data: all of smol-smoltalk (as v3) + samvaad Latin-only (as v4) + 1500 identity convs, packed at 4096.
Weights come from attached kernel outputs (no signed URLs): prajna-ctx4k (init), prajna-sft (v4),
kaggle-llm-sft (v3). All code is embedded from the local repo (GitHub main is behind).

    python kaggle/make_sft_v5.py && kaggle kernels push -p kaggle/sft-v5
"""
import json
import os

SESSION_H = 7.0
OUT = "kaggle/sft-v5"
SLUG = "prajna-sft-v5"


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def embed(path):
    return code(f"%%writefile {path}\n" + open(path).read())


FILES = ["src/llm/__init__.py", "src/llm/config.py", "src/llm/model.py", "src/llm/data.py", "src/llm/chat.py",
         "src/llm/identity.py", "scripts/train.py", "scripts/prepare_sft.py", "scripts/eval_sft.py",
         "scripts/generate.py"]
PROMPTS = ["Who are you and who made you?", "Tum kaun ho aur tumhe kisne banaya?",
           "What is the capital of Kerala and what is it known for?",
           "Tell me about the festival of Diwali.", "Explain the role of the Lok Sabha in Indian democracy.",
           "Mujhe Taj Mahal ke baare mein batao.", "What is photosynthesis? Explain it simply.",
           "Write a Python function that checks whether a number is prime."]
gen = "\n".join(f"!python scripts/generate.py --ckpt /kaggle/working/sft/model.pt --chat --num_samples 2 --seed 0 "
                f"--tokens 250 --prompt {json.dumps(p)} | tee -a /kaggle/working/samples.txt" for p in PROMPTS)
EVAL = "!python scripts/eval_sft.py --ckpt {ckpts} --data {data} --block {block} | tee -a /kaggle/working/eval.txt"
ALL3 = "$V3 $V4 /kaggle/working/sft/model.pt"

cells = [
    {"cell_type": "markdown", "metadata": {}, "source":
     "# Prajna AI: `sft` v5\nInit = prajna-ctx4k (4096 ctx, India continued pretraining). Data: full smol-smoltalk "
     "+ sarvamai/samvaad-hi-v1 (Latin-script only) + 1500 Prajna identity convs, packed at 4096. "
     "Eval: v3 / v4 / v5 on the same 1024-packed val sets as v4, plus v5 on its 4096-packed val."},
    code("""import os, time, glob
NB_START = time.time()
os.makedirs('/tmp/kaggle-llm/src/llm', exist_ok=True)
os.makedirs('/tmp/kaggle-llm/scripts', exist_ok=True)
%cd /tmp/kaggle-llm
!pip install -q tiktoken 2>&1 | tail -1"""),
    *[embed(p) for p in FILES],
    code("!mkdir -p /kaggle/working/code && cp -r src scripts /kaggle/working/code/"),
    code("""def find(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))
    print(pattern, '->', hits)
    return hits
BASE = [h for h in find('ctx4k/model.pt')][0]
sft = find('sft/model.pt')
V4 = [h for h in sft if 'prajna' in h][0]
V3 = [h for h in sft if 'kaggle-llm' in h][0]
os.environ.update(BASE=BASE, V3=V3, V4=V4)
import torch
for p in (BASE, V3, V4):
    print(p, torch.load(p, map_location='cpu', weights_only=False)['model_config'])"""),
    code("!python scripts/prepare_sft.py --shards 4 --identity 1500 --block_size 4096 "
         "--mix_repo sarvamai/samvaad-hi-v1 --mix_latin_only --mix_val_rows 600 "
         "--out_dir /tmp/sft --cache_dir /tmp/hf_cache"),
    code("# same val sets as v4 (first 2000 smoltalk / 600 samvaad kept rows, 1024-packed)\n"
         "!python scripts/prepare_sft.py --shards 1 --max_rows 2100 --block_size 1024 "
         "--mix_repo sarvamai/samvaad-hi-v1 --mix_latin_only --mix_max_rows 3500 --mix_val_rows 600 "
         "--out_dir /tmp/sft1k --cache_dir /tmp/hf_cache"),
    code(f"""import json
meta = json.load(open('/tmp/sft/meta.json'))
print(json.dumps(meta, indent=1))
TOTAL = meta['train_tokens']  # one epoch over the mix
elapsed_h = (time.time() - NB_START) / 3600
BUDGET = round({SESSION_H} - elapsed_h - 0.5, 2)
print(f'train tokens {{TOTAL / 1e6:.1f}}M (~{{TOTAL // 131072}} iters) | prep {{elapsed_h * 60:.0f}} min -> budget {{BUDGET}} h')"""),
    code("!torchrun --standalone --nproc_per_node=gpu scripts/train.py --init_from $BASE --data_dir /tmp/sft "
         "--out_dir /kaggle/working/sft --time_budget_hours {BUDGET} --time_limit_hours {BUDGET + 0.2} --micro_batch 2 "
         "--batch_tokens 131072 --total_tokens {TOTAL} --lr 2e-4 --warmup_iters 50 --eval_every 100 --eval_batches 10 "
         "--ckpt_every 200 --log_every 10\n!cp /tmp/sft/meta.json /kaggle/working/sft/data_meta.json"),
    code("\n".join([
        "!echo '== smol-smoltalk val, 1024-packed (v3 / v4 / v5)' | tee /kaggle/working/eval.txt",
        EVAL.format(ckpts=ALL3, data="/tmp/sft1k/val.bin", block=1024),
        "!echo '== samvaad held-out, 1024-packed (v3 / v4 / v5)' | tee -a /kaggle/working/eval.txt",
        EVAL.format(ckpts=ALL3, data="/tmp/sft1k/val_mix.bin", block=1024),
        "!echo '== v5 on its own 4096-packed val sets' | tee -a /kaggle/working/eval.txt",
        EVAL.format(ckpts="/kaggle/working/sft/model.pt", data="/tmp/sft/val.bin", block=4096),
        EVAL.format(ckpts="/kaggle/working/sft/model.pt", data="/tmp/sft/val_mix.bin", block=4096)])),
    code(gen),
]
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
os.makedirs(OUT, exist_ok=True)
json.dump(nb, open(f"{OUT}/{SLUG}.ipynb", "w"), indent=1)
json.dump({"id": f"somdevsheel/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
           "enable_internet": True, "keywords": [], "dataset_sources": [], "competition_sources": [],
           "kernel_sources": ["somdevsheel/prajna-ctx4k", "somdevsheel/prajna-sft", "somdevsheel/kaggle-llm-sft"],
           "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
          open(f"{OUT}/kernel-metadata.json", "w"), indent=1)
print(f"wrote {OUT}/{SLUG}.ipynb", len(json.dumps(nb)), "bytes")
