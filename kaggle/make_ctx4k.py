"""Build kaggle/ctx4k-boot.ipynb: continued pretraining of base v3 on the India corpus at 4096 context.

    python kaggle/make_ctx4k.py <train.bin url> <val.bin url> <val_india.bin url> <base model.pt url>
"""
import json
import sys

TRAIN, VAL, VAL_IN, BASE = sys.argv[1:5]
SESSION_H = 9.0


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def embed(path):
    return code(f"%%writefile {path}\n" + open(path).read())


PROMPTS = ["The capital of Kerala is", "India's Parliament has two houses,", "The river Ganga",
           "Diwali is celebrated", "The Indian Space Research Organisation (ISRO)", "Photosynthesis is the process"]
gen = "\n".join(f"!python scripts/generate.py --ckpt /kaggle/working/ctx4k/model.pt --num_samples 2 --seed 0 "
                f"--tokens 120 --prompt {json.dumps(p)} | tee -a /kaggle/working/samples.txt" for p in PROMPTS)
cells = [
    {"cell_type": "markdown", "metadata": {}, "source":
     "# Prajna AI: 4K-context India continued pretraining\nInit = base v3 (kaggle-llm-continue run/model.pt, 1024 ctx). "
     "Data = prajna-india-data (Sangraha eng India 260M + Wikipedia India 100M + unseen FineWeb-Edu 247M). "
     "block_size 4096, rope_theta 1e5. train.py / eval_lm.py embedded; other code from GitHub main."},
    code("""import os, time
NB_START = time.time()
os.makedirs('/tmp/kaggle-llm/src/llm', exist_ok=True)
os.makedirs('/tmp/kaggle-llm/scripts', exist_ok=True)
%cd /tmp/kaggle-llm
!pip install -q tiktoken 2>&1 | tail -1
import subprocess
RAW = 'https://raw.githubusercontent.com/somdevsheel/train_llm/main/'
for p in ['scripts/generate.py', 'src/llm/__init__.py', 'src/llm/chat.py', 'src/llm/config.py', 'src/llm/data.py',
          'src/llm/model.py']:
    subprocess.run(['curl', '-sSfL', '-o', p, RAW + p], check=True)
    print(p, os.path.getsize(p))"""),
    embed("scripts/train.py"),
    embed("scripts/eval_lm.py"),
    code("!mkdir -p /kaggle/working/code && cp -r src scripts /kaggle/working/code/"),
    code(f"""!mkdir -p /tmp/india /tmp/base
!curl -sSfL -o /tmp/india/train.bin "{TRAIN}"
!curl -sSfL -o /tmp/india/val.bin "{VAL}"
!curl -sSfL -o /tmp/india/val_india.bin "{VAL_IN}"
!curl -sSfL -o /tmp/base/model.pt "{BASE}"
!ls -la /tmp/india /tmp/base"""),
    code("""!echo '== base v3 before training' | tee /kaggle/working/eval.txt
!python scripts/eval_lm.py --ckpt /tmp/base/model.pt --data /tmp/india/val.bin /tmp/india/val_india.bin --ctx 1024 4096 | tee -a /kaggle/working/eval.txt"""),
    code(f"""elapsed_h = (time.time() - NB_START) / 3600
BUDGET = round({SESSION_H} - elapsed_h - 0.35, 2)
print(f'setup {{elapsed_h * 60:.0f}} min -> training budget {{BUDGET}} h')"""),
    code("!torchrun --standalone --nproc_per_node=gpu scripts/train.py --init_from /tmp/base/model.pt --data_dir /tmp/india "
         "--block_size 4096 --rope_theta 100000 --out_dir /kaggle/working/ctx4k --time_budget_hours {BUDGET} "
         "--time_limit_hours {BUDGET + 0.25} --micro_batch 2 --batch_tokens 262144 --total_tokens 6.0e8 --lr 3e-4 "
         "--warmup_iters 100 --eval_every 200 --eval_batches 10 --ckpt_every 200 --log_every 20"),
    code("""!echo '== ctx4k after training' | tee -a /kaggle/working/eval.txt
!python scripts/eval_lm.py --ckpt /kaggle/working/ctx4k/model.pt --data /tmp/india/val.bin /tmp/india/val_india.bin --ctx 1024 4096 | tee -a /kaggle/working/eval.txt"""),
    code(gen),
]
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
json.dump(nb, open("kaggle/ctx4k-boot.ipynb", "w"), indent=1)
print(len(json.dumps(nb)))
