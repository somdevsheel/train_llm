"""Train the GPT on 1 or more GPUs (DDP), with fp16/bf16 AMP, grad accumulation,
warmup + cosine LR, periodic checkpoints, a session time limit, and exact resume.

Single GPU:   python scripts/train.py --preset 124m --data_dir data/fineweb
Multi GPU:    torchrun --standalone --nproc_per_node=gpu scripts/train.py --preset 124m ...
Benchmark:    torchrun --standalone --nproc_per_node=gpu scripts/train.py --synthetic --max_iters 20
Resume:       ... --resume path/to/ckpt.pt   (or --resume auto  -> <out_dir>/ckpt.pt if present)
Fine-tune:    ... --init_from base/model.pt --data_dir data/sft   (SFT data from prepare_sft.py)
"""
import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from llm.config import PRESETS, GPTConfig, get_config, to_dict  # noqa: E402
from llm.chat import prompt_ids  # noqa: E402
from llm.data import SyntheticLoader, TokenLoader  # noqa: E402
from llm.model import GPT  # noqa: E402

SAMPLE_PROMPTS = ["The most important thing to understand about science is",
                  "Once upon a time,"]
CHAT_PROMPTS = ["What is photosynthesis? Explain it simply.",
                "Write a short poem about the ocean."]


# settings a resumed run takes from its checkpoint: LR schedule, batch geometry, data order
RESUME_KEEP = ["lr", "min_lr_ratio", "warmup_iters", "weight_decay", "grad_clip", "batch_tokens",
               "micro_batch", "data_offset_iters", "seed", "eval_every", "ckpt_every"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="124m", choices=list(PRESETS))
    p.add_argument("--data_dir", default="data/fineweb")
    p.add_argument("--synthetic", action="store_true", help="random tokens; throughput benchmark")
    p.add_argument("--out_dir", default="checkpoints/run")
    p.add_argument("--resume", default="", help="ckpt path, or 'auto' for <out_dir>/ckpt.pt")
    p.add_argument("--init_from", default="", help="start from these weights (model.pt) with a fresh "
                   "optimizer/schedule, e.g. for fine-tuning; architecture comes from the file")
    p.add_argument("--init_optimizer", action="store_true",
                   help="with --init_from a full ckpt.pt: also load its AdamW moments (smoother continued pretraining)")
    p.add_argument("--data_offset_iters", type=int, default=0,
                   help="skip this many optimizer steps of data first (continue on unseen data; same micro_batch/GPUs)")
    p.add_argument("--total_tokens", type=float, default=2e9, help="training budget (sets the LR schedule)")
    p.add_argument("--max_iters", type=int, default=0, help="override total_tokens")
    p.add_argument("--batch_tokens", type=int, default=2 ** 19, help="global tokens per optimizer step")
    p.add_argument("--micro_batch", type=int, default=8, help="sequences per GPU per forward pass")
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--time_limit_hours", type=float, default=0, help="checkpoint & exit after this long")
    p.add_argument("--time_budget_hours", type=float, default=0,
                   help="fresh runs: measure speed, then shrink the schedule so it finishes in this time")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def lr_at(it, a, max_iters):
    if it < a.warmup_iters:
        return a.lr * (it + 1) / a.warmup_iters
    progress = min(1.0, (it - a.warmup_iters) / max(1, max_iters - a.warmup_iters))
    min_lr = a.lr * a.min_lr_ratio
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (a.lr - min_lr)


def main():
    a = get_args()

    # ---------------- distributed setup ----------------
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world, local_rank = 0, 1, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    device_type = "cuda" if "cuda" in device else "cpu"

    def log(*msg):
        if master:
            print(*msg, flush=True)

    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # bf16 on Ampere+, fp16 + loss scaling on T4/P100
    if device_type == "cuda":
        if a.amp == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        else:
            amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.amp]
        autocast = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32)
    else:
        amp_dtype, autocast = torch.float32, nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # ---------------- resume ----------------
    ckpt_path = os.path.join(a.out_dir, "ckpt.pt") if a.resume == "auto" else a.resume
    ckpt = None
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ckpt["model_config"])
        log(f"Resuming from {ckpt_path} at iter {ckpt['iter']}")
        # the schedule and the data order must match the interrupted run, whatever flags were passed now
        saved = ckpt.get("args", {})
        for k in RESUME_KEEP:
            if k in saved and getattr(a, k) != saved[k]:
                log(f"  resume: {k} = {saved[k]} (from checkpoint; ignoring {getattr(a, k)})")
                setattr(a, k, saved[k])
    else:
        if a.resume and a.resume != "auto":
            raise FileNotFoundError(ckpt_path)
        init = None
        if a.init_from:
            init = torch.load(a.init_from, map_location="cpu", weights_only=False)
            cfg = GPTConfig(**init["model_config"])
            log(f"Initializing weights from {a.init_from}")
        else:
            cfg = get_config(a.preset)

    # ---------------- batch geometry ----------------
    tokens_per_micro = a.micro_batch * cfg.block_size * world
    accum = max(1, a.batch_tokens // tokens_per_micro)
    step_tokens = accum * tokens_per_micro
    if ckpt:
        max_iters = ckpt["max_iters"]
        assert ckpt["step_tokens"] == step_tokens, "batch geometry changed; use the same micro_batch/GPUs"
    else:
        max_iters = a.max_iters or math.ceil(a.total_tokens / step_tokens)
    start_iter = ckpt["iter"] if ckpt else 0

    # ---------------- model / optimizer ----------------
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizer(a.weight_decay, a.lr, device_type)
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
    elif a.init_from:
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in init["model"].items()})
        if a.init_optimizer:
            assert "optimizer" in init, "--init_optimizer needs a full ckpt.pt, not a weights-only model.pt"
            optimizer.load_state_dict(init["optimizer"])
            if init.get("scaler"):
                scaler.load_state_dict(init["scaler"])
            log("Loaded optimizer state from the init checkpoint")
        del init
    best_val = ckpt.get("best_val", float("inf")) if ckpt else float("inf")
    resumed = ckpt is not None
    del ckpt

    raw_model = model
    if a.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    log(f"GPUs: {world} x {torch.cuda.get_device_name(0) if device_type == 'cuda' else 'cpu'} | AMP {amp_dtype}")
    log(f"Model: {raw_model.num_params() / 1e6:.1f}M params "
        f"({raw_model.num_params(non_embedding=True) / 1e6:.1f}M non-embedding) | {to_dict(cfg)}")
    log(f"Batch: {step_tokens:,} tokens/step = {a.micro_batch} seq x {cfg.block_size} x {world} GPU x {accum} accum")
    log(f"Schedule: {max_iters:,} iters = {max_iters * step_tokens / 1e9:.2f}B tokens (from iter {start_iter})")

    # ---------------- data ----------------
    if a.synthetic:
        train_loader = SyntheticLoader(cfg.vocab_size, cfg.block_size, a.micro_batch, device)
        val_loader = None
    else:
        train_loader = TokenLoader(os.path.join(a.data_dir, "train.bin"), cfg.block_size, a.micro_batch,
                                   device, rank, world, a.seed,
                                   skip_batches=(start_iter + a.data_offset_iters) * accum)
        val_loader = TokenLoader(os.path.join(a.data_dir, "val.bin"), cfg.block_size, a.micro_batch,
                                 device, rank, world, a.seed)
        if a.data_offset_iters:
            left = (train_loader.per_epoch - train_loader.batch_in_epoch) * a.micro_batch * cfg.block_size * world
            log(f"Data offset: skipping {a.data_offset_iters} steps already trained on; "
                f"{left / 1e9:.2f}B unseen tokens left in this epoch")
        log(f"Data: {len(train_loader.data) / 1e9:.2f}B train tokens, {len(val_loader.data) / 1e6:.1f}M val tokens"
            + (" | SFT (loss on assistant tokens only)" if train_loader.mask is not None else ""))
    sft = not a.synthetic and train_loader.mask is not None

    tokenizer = None
    if master and not a.synthetic:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def evaluate():
        model.eval()
        val_loader.reset()
        total = torch.zeros(1, device=device)
        for _ in range(a.eval_batches):
            x, y = val_loader.next()
            with autocast:
                _, loss = model(x, y)
            total += loss.float()
        total /= a.eval_batches
        if ddp:
            dist.all_reduce(total, op=dist.ReduceOp.AVG)
        model.train()
        return total.item()

    @torch.no_grad()
    def sample():
        raw_model.eval()
        for prompt in (CHAT_PROMPTS if sft else SAMPLE_PROMPTS):
            ids = prompt_ids(tokenizer, [{"role": "user", "content": prompt}]) if sft else tokenizer.encode(prompt)
            idx = torch.tensor([ids], device=device)
            with autocast:
                out = raw_model.generate(idx, 100 if sft else 60, temperature=0.8, top_k=50, eos_id=50256)
            log(f"  > {tokenizer.decode(out[0].tolist())!r}")
        raw_model.train()

    def save(it, path_name="ckpt.pt"):
        if not master or a.synthetic:
            return
        os.makedirs(a.out_dir, exist_ok=True)
        tmp = os.path.join(a.out_dir, path_name + ".tmp")
        torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "model_config": to_dict(cfg), "iter": it,
                    "max_iters": max_iters, "step_tokens": step_tokens, "best_val": best_val,
                    "args": vars(a)}, tmp)
        os.replace(tmp, os.path.join(a.out_dir, path_name))

    def export_weights():
        """Small fp16 weights-only file for inference."""
        if not master or a.synthetic:
            return
        sd = {k: v.half() if v.is_floating_point() else v for k, v in raw_model.state_dict().items()}
        torch.save({"model": sd, "model_config": to_dict(cfg)}, os.path.join(a.out_dir, "model.pt"))

    log_file = None
    if master and not a.synthetic:
        os.makedirs(a.out_dir, exist_ok=True)
        log_file = open(os.path.join(a.out_dir, "log.jsonl"), "a")
    session_start = time.time()
    it = start_iter
    step_times = []
    BUDGET_PROBE = 20  # iterations to time before fitting the schedule to the budget

    # ---------------- training loop ----------------
    while it < max_iters:
        last = it == max_iters - 1

        if val_loader is not None and a.eval_every and (it % a.eval_every == 0 or last):
            val_loss = evaluate()
            best_val = min(best_val, val_loss)
            log(f"[eval] iter {it} | val loss {val_loss:.4f} | ppl {math.exp(val_loss):.1f}")
            if log_file:
                log_file.write(json.dumps({"iter": it, "val_loss": val_loss}) + "\n")
                log_file.flush()
            if master:
                sample()

        lr = lr_at(it, a, max_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.time()
        loss_accum = torch.zeros(1, device=device)
        for micro in range(accum):
            x, y = train_loader.next()
            sync = model.no_sync() if (ddp and micro < accum - 1) else nullcontext()
            with sync:
                with autocast:
                    _, loss = model(x, y)
                    loss = loss / accum
                scaler.scale(loss).backward()
            loss_accum += loss.detach().float()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device_type == "cuda":
            torch.cuda.synchronize()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        dt = time.time() - t0
        it += 1

        # fit the LR schedule to the time budget (fresh runs only, decided once, during warmup)
        if a.time_budget_hours and not resumed and it - start_iter <= BUDGET_PROBE:
            if it - start_iter > 2:  # skip the first steps (CUDA / compile warmup)
                step_times.append(dt)
            if it - start_iter == BUDGET_PROBE:
                per_iter = sum(step_times) / len(step_times) * 1.05  # +5% for eval/ckpt overhead
                remaining = a.time_budget_hours * 3600 - (time.time() - session_start)
                new_max = torch.tensor([min(max_iters, it + int(remaining / per_iter))], device=device)
                if ddp:
                    dist.broadcast(new_max, src=0)
                old, max_iters = max_iters, int(new_max.item())
                log(f"[budget] {per_iter:.2f}s/iter -> schedule {old:,} -> {max_iters:,} iters "
                    f"= {max_iters * step_tokens / 1e9:.2f}B tokens in {a.time_budget_hours:.1f}h")

        if it % a.log_every == 0 or it == start_iter + 1:
            elapsed = time.time() - session_start
            tps = step_tokens / dt
            eta_h = (max_iters - it) * dt / 3600
            log(f"iter {it:6d}/{max_iters} | loss {loss_accum.item():.4f} | lr {lr:.2e} | "
                f"norm {grad_norm.item():.2f} | {dt * 1000:.0f}ms | {tps:,.0f} tok/s | "
                f"elapsed {elapsed / 60:.1f}m | eta {eta_h:.1f}h")
            if log_file:
                log_file.write(json.dumps({"iter": it, "loss": loss_accum.item(), "lr": lr,
                                           "grad_norm": grad_norm.item(), "tok_per_s": tps}) + "\n")
                log_file.flush()

        # time limit: all ranks agree before stopping
        stop = torch.tensor([0.0], device=device)
        if a.time_limit_hours and (time.time() - session_start) > a.time_limit_hours * 3600:
            stop += 1
        if ddp:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)

        if a.ckpt_every and (it % a.ckpt_every == 0) or it == max_iters or stop.item():
            save(it)
            export_weights()
            log(f"[ckpt] saved at iter {it}")
        if stop.item():
            log(f"Time limit reached at iter {it}/{max_iters}. Resume with --resume <out_dir>/ckpt.pt")
            break

    if it >= max_iters:
        log(f"Training complete. Best val loss {best_val:.4f}")
    if log_file:
        log_file.close()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
