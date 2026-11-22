# train_llm: a 124M-parameter LLM trained from scratch on free Kaggle GPUs

A LLaMA-style language model written from scratch in PyTorch, pretrained on FineWeb-Edu and fine-tuned into a chat assistant, using only Kaggle's free 2x T4 GPUs (about 23 hours of 2x T4 time, $0).

- **Try it in your browser:** [kaggle-llm-124m-chat-demo](https://huggingface.co/spaces/somdevsheel/kaggle-llm-124m-chat-demo) (runs locally in the page via Transformers.js)
- **Models:** [kaggle-llm-124m-chat](https://huggingface.co/somdevsheel/kaggle-llm-124m-chat) · [kaggle-llm-124m-base](https://huggingface.co/somdevsheel/kaggle-llm-124m-base) (standard `LlamaForCausalLM`, loads with plain `transformers`)

## Results

| Stage | Data | Compute | Result |
|---|---|---|---|
| Pretraining | 0.98B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1,865 steps × 524K tokens, 10.9 h on 2x T4 | val loss 10.97 → 3.43 (perplexity 31) |
| Continued pretraining | 0.94B more unseen FineWeb-Edu tokens (1.92B total) | 1,796 steps × 524K tokens, ~10 h on 2x T4, peak LR 6e-4 | val loss 3.43 → **3.30** (perplexity 27) |
| Chat fine-tuning (SFT) | 451K conversations from [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (full set) + 1K synthetic identity conversations | 4,578 steps × 65K tokens, 3.2 h on 2x T4 | assistant-token val loss 2.65 → **1.46** (v2: 1.54 on half the data; v1: 1.61 from the 0.98B-token base) |

HellaSwag (validation, 10,042 examples, length-normalized), scored with the same script for every model. *These scores are for the previous release (0.98B-token base); re-evaluation of the current models is in progress.*

| Model | Training tokens | HellaSwag |
|---|---|---|
| Random guessing | — | 25.0% |
| This model (base) | 0.98B | **27.1%** |
| This model (chat) | + 150M SFT | 27.0% |
| GPT-2 small (OpenAI, 124M) | ~10B (approx.) | 29.5% |

The model follows the chat format and writes short correct code (e.g. string reversal, prime checks), but with ~16 training tokens per parameter (compute-optimal is ~20) it often states wrong facts confidently.

## Architecture

Decoder-only Transformer following the LLaMA recipe at GPT-2-small scale: 12 layers, d_model 768, 12 heads, 1024-token context, RoPE, RMSNorm (pre-norm, computed in fp32), SwiGLU feed-forward (hidden 2048), no biases, tied input/output embeddings, GPT-2 BPE tokenizer (vocab padded to 50,304). 123.6M parameters.

## What's interesting in the code

- **Time-budgeted LR schedule.** Kaggle sessions end at 12 h. The trainer times its first steps, then shrinks the cosine schedule so the decay finishes inside the budget instead of being cut off (`--time_budget_hours`).
- **Exact data continuation.** The loader walks a seeded, rank-sharded shuffle of fixed chunks, so a resumed or continued run skips exactly the batches already trained on (`--resume`, `--data_offset_iters`).
- **SFT with loss masking and packing.** Loss only on assistant tokens; whole conversations first-fit packed into 1024-token blocks (92% fill vs ~75% with naive packing).
- **Self-contained Kaggle notebooks.** `kaggle/build_notebook.py` embeds the source into a notebook that downloads and tokenizes its own data.
- **Verified exports.** `scripts/export_hf.py` converts to `LlamaForCausalLM` and refuses to save unless logits match the original exactly; the chat template is checked token-for-token against training.

## Layout

```
src/llm/            model.py (GPT, RoPE, RMSNorm, SwiGLU, generate) · config.py · data.py · chat.py
scripts/            prepare_data.py · prepare_sft.py · train.py · generate.py · export_hf.py · eval_hellaswag.py
kaggle/             build_notebook.py (prepare / train / continue / sft notebooks)
webapp/             local test app: FastAPI server + React/TypeScript UI (chat, side-by-side completion, next-token inspector)
hf_static/          the browser demo (Transformers.js in a Web Worker)
hf_space/           Gradio version of the demo
results/            hellaswag.json
```

## Run it

```bash
pip install torch tiktoken numpy pyarrow huggingface_hub

# chat in the terminal (download a checkpoint or export first)
python scripts/generate.py --ckpt checkpoints/sft/model.pt --chat

# local web app (http://127.0.0.1:7860)
cd webapp/frontend && npm install && npm run build && cd ../..
python webapp/server.py
```

Train locally on a small corpus:

```bash
python scripts/prepare_data.py --dataset textfile --input_file my.txt --val_tokens 20000 --out_dir data/mine
python scripts/train.py --preset tiny --data_dir data/mine --max_iters 200
```

Build a Kaggle notebook (see `kaggle/build_notebook.py` for all tasks):

```bash
python kaggle/build_notebook.py train-standalone --max_tokens 2.5e9 \
  --train_args "--preset 124m --total_tokens 2.5e9"
```

Evaluate on HellaSwag:

```bash
python scripts/eval_hellaswag.py --models hf_export/base openai-community/gpt2
```

## Use the published model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "somdevsheel/kaggle-llm-124m-chat"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16).to("cuda")

ids = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that reverses a string."}],
                              add_generation_prompt=True, return_tensors="pt").to("cuda")
out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=200,
                     do_sample=True, temperature=0.7, top_k=50, repetition_penalty=1.15)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## License

Apache-2.0.
