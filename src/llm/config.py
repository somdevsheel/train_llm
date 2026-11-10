"""Model architecture configs and size presets."""
from dataclasses import asdict, dataclass, replace


@dataclass
class GPTConfig:
    vocab_size: int = 50304      # GPT-2 BPE (50257) padded to a multiple of 64 for speed
    block_size: int = 1024       # context length
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    ffn_hidden: int = 2048       # SwiGLU hidden size (~8/3 * d_model)
    rope_theta: float = 10000.0
    dropout: float = 0.0


PRESETS = {
    # smoke tests only
    "tiny": GPTConfig(n_layer=4, n_head=4, d_model=128, ffn_hidden=384, block_size=256),
    # ~34M: trains in ~2h on Kaggle
    "30m": GPTConfig(n_layer=8, n_head=8, d_model=384, ffn_hidden=1024, block_size=512),
    # ~124M (GPT-2 small size): the best fit for Kaggle's free GPUs
    "124m": GPTConfig(),
}


def get_config(name: str) -> GPTConfig:
    return replace(PRESETS[name])


def to_dict(cfg: GPTConfig) -> dict:
    return asdict(cfg)
