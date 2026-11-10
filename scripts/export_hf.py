"""Convert a kaggle-llm checkpoint to a standard Hugging Face LlamaForCausalLM folder.

The architecture is LLaMA-compatible (RoPE with rotate-half, RMSNorm, SwiGLU, no biases,
tied embeddings), so only the weights are renamed and the fused matrices split:
    attn.qkv  -> q_proj, k_proj, v_proj        mlp.w13 -> gate_proj, up_proj

    python scripts/export_hf.py --ckpt checkpoints/sft/model.pt --out_dir hf_export/sft --chat
    python scripts/export_hf.py --ckpt checkpoints/124m-v2/model.pt --out_dir hf_export/base

Verifies that logits match the original model before finishing.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.config import GPTConfig  # noqa: E402
from llm.model import GPT  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>\n{{ m['content'] | trim }}"
    "{% if m['role'] == 'assistant' %}<|endoftext|>{% endif %}{{ '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def convert(sd, cfg):
    C, F = cfg.d_model, cfg.ffn_hidden
    out = {"model.embed_tokens.weight": sd["tok_emb.weight"], "model.norm.weight": sd["norm_f.weight"]}
    for i in range(cfg.n_layer):
        p, h = f"blocks.{i}.", f"model.layers.{i}."
        q, k, v = sd[p + "attn.qkv.weight"].split(C, dim=0)
        gate, up = sd[p + "mlp.w13.weight"].split(F, dim=0)
        out.update({
            h + "input_layernorm.weight": sd[p + "norm1.weight"],
            h + "post_attention_layernorm.weight": sd[p + "norm2.weight"],
            h + "self_attn.q_proj.weight": q, h + "self_attn.k_proj.weight": k,
            h + "self_attn.v_proj.weight": v, h + "self_attn.o_proj.weight": sd[p + "attn.proj.weight"],
            h + "mlp.gate_proj.weight": gate, h + "mlp.up_proj.weight": up,
            h + "mlp.down_proj.weight": sd[p + "mlp.w2.weight"],
        })
    return {k: v.contiguous() for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--chat", action="store_true", help="add the chat template to the tokenizer")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    a = p.parse_args()

    from transformers import GPT2TokenizerFast, LlamaConfig, LlamaForCausalLM

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["model_config"])
    sd = {k: v.float() for k, v in ck["model"].items()}

    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layer, num_attention_heads=cfg.n_head, num_key_value_heads=cfg.n_head,
        max_position_embeddings=cfg.block_size, rms_norm_eps=1e-6, rope_theta=cfg.rope_theta,
        hidden_act="silu", attention_bias=False, mlp_bias=False, tie_word_embeddings=True,
        bos_token_id=50256, eos_token_id=50256, pad_token_id=50256, torch_dtype=a.dtype,
    )
    hf = LlamaForCausalLM(hf_cfg)
    missing, unexpected = hf.load_state_dict(convert(sd, cfg), strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied to embed_tokens
    assert not missing and not unexpected, (missing, unexpected)
    hf.tie_weights()
    hf.eval()

    # ---- verify against the original implementation (fp32, CPU) ----
    ours = GPT(cfg)
    ours.load_state_dict(sd)
    ours.eval()
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    ids = torch.tensor([tok.encode("The history of science shows that careful measurement")])
    with torch.no_grad():
        # our forward returns last-position logits without targets; pass dummy targets for all positions
        ref, _ = ours(ids, torch.zeros_like(ids))
        got = hf(ids).logits
    diff = (ref - got).abs().max().item()
    print(f"max |logit diff| vs original: {diff:.2e}")
    assert diff < 1e-3, "converted model does not match"

    hf.to(getattr(torch, a.dtype))
    os.makedirs(a.out_dir, exist_ok=True)
    hf.save_pretrained(a.out_dir, safe_serialization=True)
    tok.pad_token = tok.eos_token
    tok.model_max_length = cfg.block_size
    if a.chat:
        tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(a.out_dir)
    print(f"Saved to {a.out_dir}: {sorted(os.listdir(a.out_dir))}")


if __name__ == "__main__":
    main()
