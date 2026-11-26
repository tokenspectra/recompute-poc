"""CPU port of vLLM v0.32.0's Model Runner V2 Gumbel noise.

Source: vllm/v1/worker/gpu/sample/gumbel.py at tag v0.32.0 (5df63a7). The noise is a
stateless murmur3 hash of (seed, pos, token id), so it can be replayed without the
server. Validated against the real Triton kernel on an H100 with the pinned image
(`scripts/validate_gumbel.py`: 6000/6000 identical, see docs/results-20261025.md).

Ported from vLLM (https://github.com/vllm-project/vllm), Copyright contributors to the
vLLM project, Apache License 2.0. Modified: rewritten from Triton to NumPy. See NOTICE.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

U32 = npt.NDArray[np.uint32]
F32 = npt.NDArray[np.float32]

_C1 = np.uint32(0xCC9E2D51)
_C2 = np.uint32(0x1B873593)
_M = np.uint32(0xE6546B64)
_F1 = np.uint32(0x85EBCA6B)
_F2 = np.uint32(0xC2B2AE35)


def _rotl(x: U32, shift: int) -> U32:
    return (x << np.uint32(shift)) | (x >> np.uint32(32 - shift))


def _mix(h: U32, key: U32) -> U32:
    key = key * _C1
    key = _rotl(key, 15)
    key = key * _C2
    h = h ^ key
    h = _rotl(h, 13)
    return h * np.uint32(5) + _M


def _fmix32(h: U32) -> U32:
    h = h ^ (h >> np.uint32(16))
    h = h * _F1
    h = h ^ (h >> np.uint32(13))
    h = h * _F2
    return h ^ (h >> np.uint32(16))


def murmur3_hash32(seed: int, pos: int, offsets: U32, domain: int = 0) -> U32:
    s = seed & 0xFFFFFFFFFFFFFFFF  # int64 two's complement
    lo = np.full(offsets.shape, s & 0xFFFFFFFF, dtype=np.uint32)
    hi = np.full(offsets.shape, (s >> 32) & 0xFFFFFFFF, dtype=np.uint32)
    p = np.full(offsets.shape, pos & 0xFFFFFFFF, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = np.full(offsets.shape, domain, dtype=np.uint32)
        h = _mix(h, lo)
        h = _mix(h, hi)
        h = _mix(h, p)
        h = _mix(h, offsets.astype(np.uint32))
        return _fmix32(h ^ np.uint32(16))


def murmur3_uniform32(seed: int, pos: int, offsets: U32) -> F32:
    r = murmur3_hash32(seed, pos, offsets)
    hi16 = (r >> np.uint32(16)).astype(np.float32)
    lo16 = (r & np.uint32(0xFFFF)).astype(np.float32)
    return (
        hi16 * np.float32(1.52587890625e-05)
        + (lo16 + np.float32(0.5)) * np.float32(2.3283064365386963e-10)
    ).astype(np.float32)


def _log1p_neg_stable(value: F32) -> F32:
    poly = np.full(value.shape, 1.0 / 8.0, dtype=np.float32)
    for c in (7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0):
        poly = (np.float32(1.0 / c) + value * poly).astype(np.float32)
    series = (-value * poly).astype(np.float32)
    direct = np.log(np.maximum(np.float32(1.0) - value, np.float32(5.960464477539063e-08)))
    return np.where(value < np.float32(0.25), series, direct).astype(np.float32)


def gumbel_noise(seed: int, pos: int, token_ids: npt.NDArray[np.int64]) -> F32:
    """Gumbel noise g(seed, pos, v) for each candidate token id v (fp32 path)."""
    u = murmur3_uniform32(seed, pos, token_ids.astype(np.uint32))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (-np.log(-_log1p_neg_stable(u))).astype(np.float32)
