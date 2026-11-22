"""Bundle the project into a self-contained Kaggle notebook for one task.

Tasks:
  prepare  (CPU)       tokenize FineWeb-Edu -> /kaggle/working/data/{train,val}.bin
  bench    (GPU)       throughput test of the 124M model on synthetic data (micro-batch / compile)
  train    (2x T4)     train; auto-finds train.bin and an optional ckpt.pt to resume under /kaggle/input
  continue (2x T4)     standalone continued pretraining: downloads a full ckpt.pt from --base_url,
                       re-tokenizes the same data, skips the steps already trained on (--offset_iters)
  resume   (2x T4)     standalone resume of an interrupted continue run: downloads its ckpt.pt from
                       --base_url and finishes the same LR schedule on the next unseen data
  sft      (2x T4)     standalone chat fine-tune: downloads base model.pt from --base_url, builds
                       smol-smoltalk SFT data, fine-tunes within --session_hours

Usage:
  python kaggle/build_notebook.py prepare --max_tokens 3e9
  python kaggle/build_notebook.py bench
  python kaggle/build_notebook.py train --train_args "--preset 124m --total_tokens 2e9 --time_limit_hours 11"
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = "/tmp/kaggle-llm"          # where code runs inside Kaggle
SAVED_CODE = "/kaggle/working/code"  # prepare notebook also saves the code as output, so
                                     # bench/train notebooks can load it from /kaggle/input
FILES = ["src/llm/__init__.py", "src/llm/config.py", "src/llm/model.py", "src/llm/data.py", "src/llm/chat.py", "src/llm/identity.py",
         "scripts/prepare_data.py", "scripts/prepare_sft.py", "scripts/train.py", "scripts/generate.py"]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def embedded_code_cells():
    """Write every source file into the notebook (used by `prepare`, or `--embed_code`)."""
    cells = [code(f"""
import os, time
NB_START = time.time()
os.makedirs('{RUN_DIR}/src/llm', exist_ok=True)
os.makedirs('{RUN_DIR}/scripts', exist_ok=True)
%cd {RUN_DIR}
!pip install -q tiktoken 2>&1 | tail -1
""")]
    for rel in FILES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            cells.append(code(f"%%writefile {rel}\n" + f.read()))
    cells.append(code(f"!mkdir -p {SAVED_CODE} && cp -r src scripts {SAVED_CODE}/"))
    return cells


def loaded_code_cells():
    """Copy code saved by the prepare notebook (attached as an input)."""
    return [code(f"""
import glob, os, shutil
src = glob.glob('/kaggle/input/**/code/scripts/train.py', recursive=True)
assert src, 'attach the prepare notebook output as an input'
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), '{RUN_DIR}', dirs_exist_ok=True)
%cd {RUN_DIR}
!pip install -q tiktoken 2>&1 | tail -1
!ls -R src scripts | head -20
""")]


def task_cells(a):
    if a.task == "prepare":
        return [code(f"!python scripts/prepare_data.py --dataset {a.dataset} "
                     f"--max_tokens {a.max_tokens} --out_dir /kaggle/working/data\n!ls -la /kaggle/working/data")]
    if a.task == "bench":
        runs = []
        for mb in (8, 16):
            for comp in ("", "--compile"):
                runs.append(f"!echo '=== micro_batch={mb} {comp or 'eager'} ===' && "
                            f"torchrun --standalone --nproc_per_node=gpu scripts/train.py --synthetic "
                            f"--preset {a.preset} --micro_batch {mb} --batch_tokens {mb * 1024 * 2} "
                            f"--max_iters 25 --log_every 5 {comp} 2>&1 | grep -E 'iter +(15|20|25)/|Model|GPUs|Error|error' "
                            f"|| echo FAILED")
        return [code("!nvidia-smi --query-gpu=name,memory.total --format=csv")] + [code(r) for r in runs]
    if a.task == "train":
        return [
            code("""
import glob, os
bins = glob.glob('/kaggle/input/**/train.bin', recursive=True)
assert bins, 'attach the data-prep notebook output as an input'
DATA_DIR = os.path.dirname(bins[0])
ckpts = glob.glob('/kaggle/input/**/ckpt.pt', recursive=True)
RESUME = ckpts[0] if ckpts else ''
print('data:', DATA_DIR, '| resume:', RESUME or '(fresh start)')
!cat {DATA_DIR}/meta.json
"""),
            code(f'!torchrun --standalone --nproc_per_node=gpu scripts/train.py --data_dir {{DATA_DIR}} '
                 f'--resume "{{RESUME}}" --out_dir /kaggle/working/run {a.train_args}'),
            code("!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'The most important thing to understand about science is' | tee /kaggle/working/samples.txt\n"
                 "!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'Photosynthesis is the process' | tee -a /kaggle/working/samples.txt"),
        ]
    if a.task == "train-standalone":
        # no notebook inputs needed: tokenize inside the GPU session, then train within the time budget
        return [
            code(f"!python scripts/prepare_data.py --dataset {a.dataset} --max_tokens {a.max_tokens} "
                 f"--out_dir /tmp/data --cache_dir /tmp/hf_cache"),
            code(f"""
import time
elapsed_h = (time.time() - NB_START) / 3600
BUDGET = round({a.session_hours} - elapsed_h, 2)
print(f'prep took {{elapsed_h * 60:.0f}} min -> training budget {{BUDGET}} h')
"""),
            code(f'!torchrun --standalone --nproc_per_node=gpu scripts/train.py --data_dir /tmp/data '
                 f'--out_dir /kaggle/working/run --time_budget_hours {{BUDGET}} '
                 f'--time_limit_hours {{BUDGET + 0.3}} {a.train_args}'),
            code("!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'The most important thing to understand about science is' | tee /kaggle/working/samples.txt\n"
                 "!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'Photosynthesis is the process' | tee -a /kaggle/working/samples.txt"),
        ]
    if a.task == "continue":
        assert a.base_url and a.offset_iters, "--base_url (full ckpt.pt) and --offset_iters are required"
        return [
            code(f"""
!mkdir -p /tmp/base && curl -sSfL -o /tmp/base/ckpt.pt "{a.base_url}" && ls -la /tmp/base
import torch
ck = torch.load('/tmp/base/ckpt.pt', map_location='cpu', weights_only=False)
print('base ckpt: iter', ck['iter'], '| best val', ck['best_val'], '| step tokens', ck['step_tokens'])
assert ck['iter'] == {a.offset_iters}, 'offset_iters must equal the steps the base run trained'
del ck
"""),
            code(f"!python scripts/prepare_data.py --dataset {a.dataset} --max_tokens {a.max_tokens} "
                 f"--out_dir /tmp/data --cache_dir /tmp/hf_cache"),
            code(f"""
import json
meta = json.load(open('/tmp/data/meta.json'))
print(meta)
if {a.expect_docs} and meta['docs'] != {a.expect_docs}:
    print('WARNING: data differs from the base run ({a.expect_docs} docs); the offset no longer guarantees unseen data')
else:
    print('data matches the base run: training continues on chunks it has not seen')
"""),
            code(f"""
import time
elapsed_h = (time.time() - NB_START) / 3600
BUDGET = round({a.session_hours} - elapsed_h, 2)
print(f'prep took {{elapsed_h * 60:.0f}} min -> training budget {{BUDGET}} h')
"""),
            code(f'!torchrun --standalone --nproc_per_node=gpu scripts/train.py --init_from /tmp/base/ckpt.pt '
                 f'--init_optimizer --data_offset_iters {a.offset_iters} --data_dir /tmp/data '
                 f'--out_dir /kaggle/working/run --time_budget_hours {{BUDGET}} '
                 f'--time_limit_hours {{BUDGET + 0.3}} {a.train_args}'),
            code("!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'The most important thing to understand about science is' | tee /kaggle/working/samples.txt\n"
                 "!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'Photosynthesis is the process' | tee -a /kaggle/working/samples.txt"),
        ]
    if a.task == "resume":
        # pick up an interrupted continue/train-standalone run: same schedule, next unseen batch
        assert a.base_url and a.offset_iters, "--base_url (the interrupted run's ckpt.pt) and --offset_iters are required"
        log_dl = (f' && curl -sSfL -o /kaggle/working/run/log.jsonl "{a.log_url}"' if a.log_url else "")
        return [
            code(f"""
!mkdir -p /kaggle/working/run && curl -sSfL -o /kaggle/working/run/ckpt.pt "{a.base_url}"{log_dl} && ls -la /kaggle/working/run
import torch
ck = torch.load('/kaggle/working/run/ckpt.pt', map_location='cpu', weights_only=False)
saved = ck['args']
print('interrupted run: iter', ck['iter'], '/', ck['max_iters'], '| best val', ck['best_val'],
      '| step tokens', ck['step_tokens'], '| lr', saved['lr'], '| data offset', saved['data_offset_iters'])
assert saved['data_offset_iters'] == {a.offset_iters}, 'checkpoint was trained with a different data offset'
assert ck['iter'] < ck['max_iters'], 'this run already finished; nothing to resume'
LEFT = ck['max_iters'] - ck['iter']
print(f'{{LEFT}} steps left ({{LEFT * ck["step_tokens"] / 1e9:.2f}}B tokens)')
del ck
"""),
            code(f"!python scripts/prepare_data.py --dataset {a.dataset} --max_tokens {a.max_tokens} "
                 f"--out_dir /tmp/data --cache_dir /tmp/hf_cache"),
            code(f"""
import json
meta = json.load(open('/tmp/data/meta.json'))
print(meta)
assert not {a.expect_docs} or meta['docs'] == {a.expect_docs}, 'data differs from the original run; resuming would repeat/skip data'
print('data matches the original run')
"""),
            code(f"""
import time
elapsed_h = (time.time() - NB_START) / 3600
BUDGET = round({a.session_hours} - elapsed_h, 2)
print(f'prep took {{elapsed_h * 60:.0f}} min -> training limit {{BUDGET}} h (checkpoints and stops cleanly if not done)')
"""),
            # schedule, batch geometry, seed and data offset come from the checkpoint (train.py RESUME_KEEP)
            code(f'!torchrun --standalone --nproc_per_node=gpu scripts/train.py --resume /kaggle/working/run/ckpt.pt '
                 f'--data_dir /tmp/data --out_dir /kaggle/working/run --log_every 10 '
                 f'--time_limit_hours {{BUDGET}}'),
            code("!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'The most important thing to understand about science is' | tee /kaggle/working/samples.txt\n"
                 "!python scripts/generate.py --ckpt /kaggle/working/run/model.pt --num_samples 2 --seed 0 "
                 "--prompt 'Photosynthesis is the process' | tee -a /kaggle/working/samples.txt"),
        ]
    if a.task == "sft":
        assert a.base_url, "--base_url (signed URL of the base model.pt) is required"
        chat = ("!python scripts/generate.py --ckpt /kaggle/working/sft/model.pt --chat --num_samples 2 --seed 0 "
                "--tokens 250 --prompt {q!r} {tee} /kaggle/working/samples.txt")
        qs = ["What is photosynthesis? Explain it simply.", "Give me three tips for staying focused while studying.",
              "Write a Python function that checks whether a number is prime.", "Write a short poem about the ocean.", "Who are you and who made you?"]
        return [
            code(f"""
!mkdir -p /tmp/base && curl -sSfL -o /tmp/base/model.pt "{a.base_url}" && ls -la /tmp/base
import torch
print(torch.load('/tmp/base/model.pt', map_location='cpu', weights_only=False)['model_config'])
"""),
            code(f"!python scripts/prepare_sft.py --shards {a.shards} --identity {a.identity} "
                 f"--out_dir /tmp/sft --cache_dir /tmp/hf_cache"),
            code(f"""
import time
elapsed_h = (time.time() - NB_START) / 3600
BUDGET = round({a.session_hours} - elapsed_h, 2)
print(f'prep took {{elapsed_h * 60:.0f}} min -> training budget {{BUDGET}} h')
"""),
            code(f'!torchrun --standalone --nproc_per_node=gpu scripts/train.py --init_from /tmp/base/model.pt '
                 f'--data_dir /tmp/sft --out_dir /kaggle/working/sft --time_budget_hours {{BUDGET}} '
                 f'--time_limit_hours {{BUDGET + 0.2}} {a.train_args}\n!cp /tmp/sft/meta.json /kaggle/working/sft/data_meta.json'),
            code("\n".join(chat.format(q=q, tee="| tee" if i == 0 else "| tee -a") for i, q in enumerate(qs))),
        ]
    raise ValueError(a.task)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["prepare", "bench", "train", "train-standalone", "continue", "resume", "sft"])
    p.add_argument("--offset_iters", type=int, default=0, help="continue: steps the base run already trained")
    p.add_argument("--expect_docs", type=int, default=0, help="continue: doc count of the base run's data, to verify")
    p.add_argument("--base_url", default="", help="continue/resume: URL of a full ckpt.pt; sft: URL of the base model.pt")
    p.add_argument("--log_url", default="", help="resume: URL of the interrupted run's log.jsonl, appended to")
    p.add_argument("--shards", type=int, default=2, help="sft: smol-smoltalk train shards (~120k rows each)")
    p.add_argument("--identity", type=int, default=0, help="sft: synthetic identity conversations to add")
    p.add_argument("--session_hours", type=float, default=11.2, help="train-standalone: total wall-clock to use")
    p.add_argument("--dataset", default="fineweb-edu")
    p.add_argument("--max_tokens", default="3e9")
    p.add_argument("--preset", default="124m")
    p.add_argument("--train_args", default="--preset 124m --total_tokens 2e9 --time_limit_hours 11")
    p.add_argument("--embed_code", action="store_true", help="embed source instead of loading it from inputs")
    p.add_argument("--out", default="")
    a = p.parse_args()

    cells = [md(f"# kaggle-llm: `{a.task}`\nGenerated by `kaggle/build_notebook.py` from the local project.")]
    embed = a.task in ("prepare", "train-standalone", "continue", "resume", "sft") or a.embed_code
    cells += (embedded_code_cells() if embed else loaded_code_cells()) + task_cells(a)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}}}
    out = a.out or os.path.join(ROOT, "kaggle", f"{a.task}.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
