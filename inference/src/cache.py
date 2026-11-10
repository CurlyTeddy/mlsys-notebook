from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import psutil
import torch


class KVCachePool:
    def __init__(
        self,
        layers: int,
        length: int,
        heads: int,
        head_dim: int,
        memory_ratio: float=0.7,
        dtype: torch.dtype=torch.float32,
        device: torch.device | str="cpu"
    ):
        mem = psutil.virtual_memory().available if torch.device(device).type == "cpu" else torch.cuda.mem_get_info()[0]
        dtype_size = torch.empty((), dtype=dtype).element_size()
        slots = int(mem * memory_ratio / (layers * 2 * length * heads * head_dim * dtype_size))
        self._cache = torch.zeros(slots, layers, 2, length, heads, head_dim, dtype=dtype, device=device, requires_grad=False)
        self._free_slots = [i for i in range(slots)]

    @contextmanager
    def claim(self, batch: int) -> Generator[KVCacheHandle]:
        if len(self._free_slots) < batch:
            raise MemoryError("Out of memory.")

        slots = [self._free_slots.pop() for _ in range(batch)]
        yield KVCacheHandle(self, slots, self._cache.shape[1])
        self._free_slots.extend(slots)


class KVCacheHandle:
    def __init__(self, pool: KVCachePool, slots: list[int], layers: int):
        self._pool = pool
        self._slots = slots
        self._seq_len = [0] * layers

    def get(self, layer: int):
        kv = self._pool._cache[self._slots, layer, :, :self._seq_len[layer]]

        return kv[:, 0], kv[:, 1]

    def set(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        length = key.shape[1]
        start = self._seq_len[layer]
        if start + length > self._pool._cache.shape[3]:
            raise ValueError("Sequence exceeds cache length.")
        
        self._pool._cache[self._slots, layer, 0, start:start + length] = key
        self._pool._cache[self._slots, layer, 1, start:start + length] = value
        self._seq_len[layer] += length
    
    
