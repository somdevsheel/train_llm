# train_llm: a 124M-parameter LLM trained from scratch on free Kaggle GPUs

A LLaMA-style language model written from scratch in PyTorch, pretrained on FineWeb-Edu, extended to a 4,096-token context on India-focused text, and fine-tuned into an English + Hinglish chat assistant (Prajna AI), using only Kaggle's free 2x T4 GPUs (about 43 hours of 2x T4 time across all runs, $0).

- **Try it in your browser:** [prajna-ai-124m-chat-demo](https://huggingface.co/spaces/somdevsheel/prajna-ai-124m-chat-demo) (runs locally in the page via Transformers.js)
- **Models:** [prajna-ai-124m-chat](https://huggingface.co/somdevsheel/prajna-ai-124m-chat) · [prajna-ai-124m-base](https://huggingface.co/somdevsheel/prajna-ai-124m-base) (standard `LlamaForCausalLM`, loads with plain `transformers`)

## Results

| Stage | Data | Compute | Result |
|---|---|---|---|
| Pretraining | 0.98B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1,865 steps × 524K tokens, 10.9 h on 2x T4 | val loss 10.97 → 3.43 (perplexity 31) |
| Continued pretraining | 0.94B more unseen FineWeb-Edu tokens (1.92B total) | 1,796 steps × 524K tokens, ~10 h on 2x T4, peak LR 6e-4 | val loss 3.43 → **3.30** (perplexity 27) |
| Context extension | 580M tokens of India-focused English: [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) English filtered to India topics, India-related Wikipedia, unseen FineWeb-Edu | 2,211 steps × 262K tokens, ~8.5 h on 2x T4; context 1024 → 4096, RoPE theta 10K → 100K | val loss at 4,096 tokens 5.02 → **3.25**; India val 5.11 → 2.76 |
| Chat fine-tuning (SFT, v5) | 457K conversations from [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) + 27K Latin-script Hinglish conversations from [samvaad-hi-v1](https://huggingface.co/datasets/sarvamai/samvaad-hi-v1) + 1.5K synthetic identity conversations, packed into 4,096-token blocks (99% fill) | 3,064 steps × 131K tokens, ~5.7 h on 2x T4 | assistant-token val loss 2.49 → **1.46** |

Chat versions scored by `scripts/eval_sft.py` on the same held-out conversations, packed at 1,024 tokens (lower is better):

| Version | Base | Fine-tuning data | smol-smoltalk val | samvaad (Hinglish) val |
|---|---|---|---|---|
| v3 | 1.92B FineWeb-Edu, 1K context | full smol-smoltalk + identity | 1.450 | 3.543 |
| v4 | same | v3 + Hinglish + replay | 1.458 | 2.475 |
| **v5 (current)** | + 4K context India extension | smol-smoltalk + Hinglish + identity | **1.415** | **2.318** |

HellaSwag (validation, 10,042 examples, length-normalized), scored with the same script for every model. *These scores are for the previous release (0.98B-token base); re-evaluation of the current models is in progress.*

| Model | Training tokens | HellaSwag |
|---|---|---|
| Random guessing | — | 25.0% |
| This model (base) | 0.98B | **27.1%** |
| This model (chat) | + 150M SFT | 27.0% |
| GPT-2 small (OpenAI, 124M) | ~10B (approx.) | 29.5% |

The model follows the chat format and writes short correct code (e.g. string reversal, prime checks), replies in romanised Hinglish, and knows its own name. With ~20 pretraining tokens per parameter it still often states wrong facts confidently (e.g. right Prime Minister, wrong party), and its Hinglish is fluent-sounding but often incoherent.

## Architecture

Decoder-only Transformer following the LLaMA recipe at GPT-2-small scale: 12 layers, d_model 768, 12 heads, 4096-token context (1024 before v5), RoPE (theta 100,000 from v5), RMSNorm (pre-norm, computed in fp32), SwiGLU feed-forward (hidden 2048), no biases, tied input/output embeddings, GPT-2 BPE tokenizer (vocab padded to 50,304). 123.6M parameters.

## What's interesting in the code

- **Time-budgeted LR schedule.** Kaggle sessions end at 12 h. The trainer times its first steps, then shrinks the cosine schedule so the decay finishes inside the budget instead of being cut off (`--time_budget_hours`).
- **Exact data continuation.** The loader walks a seeded, rank-sharded shuffle of fixed chunks, so a resumed or continued run skips exactly the batches already trained on (`--resume`, `--data_offset_iters`).
- **SFT with loss masking and packing.** Loss only on assistant tokens; whole conversations first-fit packed into fixed-length blocks (92% fill at 1,024 tokens, 99% at 4,096, vs ~75% with naive packing).
- **Self-contained Kaggle notebooks.** `kaggle/build_notebook.py` embeds the source into a notebook that downloads and tokenizes its own data.
- **Context extension.** `train.py --block_size 4096 --rope_theta 1e5` with `--init_from` extends a trained base; `scripts/eval_lm.py` measures loss at several context lengths so the gain and the forgetting are both visible.
- **Verified exports.** `scripts/export_hf.py` converts to `LlamaForCausalLM` and refuses to save unless logits match the original exactly; the chat template is checked token-for-token against training.

## Layout

```
src/llm/            model.py (GPT, RoPE, RMSNorm, SwiGLU, generate) · config.py · data.py · chat.py
scripts/            prepare_data.py · prepare_india.py · prepare_sft.py · train.py · generate.py · export_hf.py
                    eval_hellaswag.py · eval_lm.py (LM loss by context length) · eval_sft.py (chat versions on fixed val)
kaggle/             build_notebook.py (prepare / train / continue / sft notebooks) · make_ctx4k.py · make_sft_v4.py · make_sft_v5.py
webapp/             local test app: FastAPI server + React/TypeScript UI (chat, side-by-side completion, next-token inspector)
hf_static/          the browser demo (Transformers.js in a Web Worker; pins the model commit, bump REVISION on release)
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

repo = "somdevsheel/prajna-ai-124m-chat"
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
