"""Build kaggle/sft-v4-boot.ipynb: Prajna SFT v4 = v3 + samvaad (English/Hinglish) + smol-smoltalk replay + identity.

    python kaggle/make_sft_v4.py <signed url of v3 sft/model.pt>
"""
import json
import sys

URL = sys.argv[1]
SESSION_H = 1.6


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def embed(path):
    return code(f"%%writefile {path}\n" + open(path).read())


PROMPTS = ["Who are you and who made you?", "Tum kaun ho aur tumhe kisne banaya?",
           "What is the capital of Kerala and what is it known for?",
           "Tell me about the festival of Diwali.", "Explain the role of the Lok Sabha in Indian democracy.",
           "Mujhe Taj Mahal ke baare mein batao.", "What is photosynthesis? Explain it simply.",
           "Write a Python function that checks whether a number is prime."]
gen = "\n".join(f"!python scripts/generate.py --ckpt /kaggle/working/sft/model.pt --chat --num_samples 2 --seed 0 "
                f"--tokens 250 --prompt {json.dumps(p)} | tee -a /kaggle/working/samples.txt" for p in PROMPTS)

cells = [
    {"cell_type": "markdown", "metadata": {}, "source":
     "# Prajna AI: `sft` v4\nInit = SFT v3. Data: sarvamai/samvaad-hi-v1 (Latin-script convs only: English + Hinglish), "
     "1 shard of smol-smoltalk (replay; same val as v1-v3), 1500 Prajna identity convs (English + Hinglish). "
     "Unchanged code from GitHub main; identity.py / prepare_sft.py / eval_sft.py embedded."},
    code("""import os, time
NB_START = time.time()
os.makedirs('/tmp/kaggle-llm/src/llm', exist_ok=True)
os.makedirs('/tmp/kaggle-llm/scripts', exist_ok=True)
%cd /tmp/kaggle-llm
!pip install -q tiktoken 2>&1 | tail -1
import subprocess
RAW = 'https://raw.githubusercontent.com/somdevsheel/train_llm/main/'
FILES = ['scripts/generate.py', 'scripts/train.py', 'src/llm/__init__.py', 'src/llm/chat.py',
         'src/llm/config.py', 'src/llm/data.py', 'src/llm/model.py']
for p in FILES:
    subprocess.run(['curl', '-sSfL', '-o', p, RAW + p], check=True)
    print(p, os.path.getsize(p))"""),
    embed("src/llm/identity.py"),
    embed("scripts/prepare_sft.py"),
    embed("scripts/eval_sft.py"),
    code("!mkdir -p /kaggle/working/code && cp -r src scripts /kaggle/working/code/"),
    code(f"""!mkdir -p /tmp/base && curl -sSfL -o /tmp/base/model.pt "{URL}" && ls -la /tmp/base
import torch
print(torch.load('/tmp/base/model.pt', map_location='cpu', weights_only=False)['model_config'])"""),
    code("!python scripts/prepare_sft.py --shards 1 --max_rows 70000 --identity 1500 "
         "--mix_repo sarvamai/samvaad-hi-v1 --mix_latin_only --mix_val_rows 600 "
         "--out_dir /tmp/sft --cache_dir /tmp/hf_cache"),
    code(f"""import json, time
meta = json.load(open('/tmp/sft/meta.json'))
TOTAL = meta['train_tokens']  # one epoch over the mix
elapsed_h = (time.time() - NB_START) / 3600
BUDGET = round({SESSION_H} - elapsed_h, 2)
print(f'train tokens {{TOTAL / 1e6:.1f}}M (~{{TOTAL // 65536}} iters) | prep {{elapsed_h * 60:.0f}} min -> budget {{BUDGET}} h')"""),
    code("!torchrun --standalone --nproc_per_node=gpu scripts/train.py --init_from /tmp/base/model.pt --data_dir /tmp/sft "
         "--out_dir /kaggle/working/sft --time_budget_hours {BUDGET} --time_limit_hours {BUDGET + 0.2} --micro_batch 8 "
         "--batch_tokens 65536 --total_tokens {TOTAL} --lr 1e-4 --warmup_iters 30 --eval_every 100 --eval_batches 20 "
         "--ckpt_every 200 --log_every 10\n!cp /tmp/sft/meta.json /kaggle/working/sft/data_meta.json"),
    code("""!echo '== smol-smoltalk val (v3 vs v4)' | tee /kaggle/working/eval.txt
!python scripts/eval_sft.py --ckpt /tmp/base/model.pt /kaggle/working/sft/model.pt --data /tmp/sft/val.bin | tee -a /kaggle/working/eval.txt
!echo '== samvaad held-out (v3 vs v4)' | tee -a /kaggle/working/eval.txt
!python scripts/eval_sft.py --ckpt /tmp/base/model.pt /kaggle/working/sft/model.pt --data /tmp/sft/val_mix.bin | tee -a /kaggle/working/eval.txt"""),
    code(gen),
]
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("kaggle/sft-v4-boot.ipynb", "w"), indent=1)
print("wrote kaggle/sft-v4-boot.ipynb", len(json.dumps(nb)), "bytes")
