"""Token data loaders reading uint16 .bin files produced by scripts/prepare_data.py."""
import os

import numpy as np
import torch


class TokenLoader:
    """Walks a shuffled list of non-overlapping chunks, sharded across DDP ranks.

    Deterministic given (seed, rank, world), so a resumed run can skip exactly the
    micro-batches it already consumed and continue with fresh data.

    SFT data (prepare_sft.py) is packed in blocks of block_size+1 with a uint8 `<name>_mask.bin`
    next to it: targets whose mask is 0 are set to -1 (ignored by the loss).
    """

    def __init__(self, path, block_size, batch_size, device, rank=0, world=1, seed=1337, skip_batches=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        mask_path = path[:-len(".bin")] + "_mask.bin"
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r") if os.path.exists(mask_path) else None
        self.T, self.B, self.device = block_size, batch_size, device
        self.stride = block_size + 1 if self.mask is not None else block_size
        self.rank, self.world, self.seed = rank, world, seed
        self.n_chunks = (len(self.data) - block_size - 1) // self.stride + 1
        assert self.n_chunks >= world * batch_size, f"{path} is too small for this batch size"
        self.per_epoch = (self.n_chunks // world) // batch_size  # micro-batches per epoch per rank
        self._set_position(skip_batches)

    def _set_position(self, batch_idx):
        self.epoch, self.batch_in_epoch = divmod(batch_idx, self.per_epoch)
        self._make_order()

    def _make_order(self):
        perm = np.random.default_rng(self.seed + self.epoch).permutation(self.n_chunks)
        self.order = perm[self.rank::self.world]

    def reset(self):
        self._set_position(0)

    def next(self):
        if self.batch_in_epoch >= self.per_epoch:
            self.epoch, self.batch_in_epoch = self.epoch + 1, 0
            self._make_order()
        s = self.batch_in_epoch * self.B
        starts = self.order[s:s + self.B] * self.stride
        self.batch_in_epoch += 1
        buf = np.stack([self.data[i:i + self.T + 1] for i in starts]).astype(np.int64)
        buf = torch.from_numpy(buf)
        x, y = buf[:, :-1], buf[:, 1:].clone()
        if self.mask is not None:
            m = np.stack([self.mask[i + 1:i + self.T + 1] for i in starts])
            y[torch.from_numpy(m) == 0] = -1
        if "cuda" in str(self.device):
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


class SyntheticLoader:
    """Random tokens, for throughput benchmarks without a dataset."""

    def __init__(self, vocab_size, block_size, batch_size, device):
        self.V, self.T, self.B, self.device = vocab_size, block_size, batch_size, device

    def reset(self):
        pass

    def next(self):
        x = torch.randint(0, self.V, (self.B, self.T + 1), device=self.device)
        return x[:, :-1], x[:, 1:]
