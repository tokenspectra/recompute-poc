"""Gate 1 (GPU): does the numpy Gumbel port reproduce vLLM's real `gumbel_sample`?

Run INSIDE the pinned vllm image, with the repo mounted:
    PYTHONPATH=/work/src python /work/scripts/validate_gumbel.py --n 3000

Two probes:
  flat   logits = 0, T = 1  -> the winner is decided by the noise alone, so any mismatch
                               is a noise difference (hash, uniform, log1p, log).
  random logits ~ N(0, 2), T in {0.3, 0.6, 1.0} -> adds the division by T and the fp32 sum.
Exit code 0 only if both match >= 99.9%. Mismatches are listed so ULP ties can be told
apart from systematic errors.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

from recompute_poc.gumbel import gumbel_noise

VOCAB = 151_936  # Qwen3 padded vocab


def real_sample(logits: np.ndarray, seed: int, pos: int, temp: float) -> int:
    dev = "cuda"
    out = gumbel_sample(
        torch.tensor(logits[None], device=dev, dtype=torch.float32),
        torch.zeros(1, dtype=torch.int32, device=dev),
        torch.tensor([temp], dtype=torch.float32, device=dev),
        torch.tensor([seed], dtype=torch.int64, device=dev),
        torch.tensor([pos], dtype=torch.int64, device=dev),
        apply_temperature=True,
        is_drafting=False,
    )
    return int(out.item())


def port_sample(logits: np.ndarray, seed: int, pos: int, temp: float) -> int:
    ids = np.arange(logits.shape[0], dtype=np.int64)
    scaled = (logits.astype(np.float32) / np.float32(temp)).astype(np.float32)
    return int(np.argmax(scaled + gumbel_noise(seed, pos, ids)))


def run(name: str, trials: int, rng: np.random.Generator, flat: bool) -> dict[str, object]:
    bad: list[dict[str, object]] = []
    for t in range(trials):
        seed = int(rng.integers(0, 2**63 - 1))
        pos = int(rng.integers(0, 40_000))
        temp = 1.0 if flat else float(rng.choice([0.3, 0.6, 1.0]))
        logits = (
            np.zeros(VOCAB, dtype=np.float32)
            if flat
            else rng.normal(0, 2.0, VOCAB).astype(np.float32)
        )
        a, b = real_sample(logits, seed, pos, temp), port_sample(logits, seed, pos, temp)
        if a != b:
            bad.append({"trial": t, "seed": seed, "pos": pos, "temp": temp, "real": a, "port": b})
    rate = 1.0 - len(bad) / trials
    return {"probe": name, "trials": trials, "match_rate": rate, "mismatches": bad[:10]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    rng = np.random.default_rng(a.rng)
    res = [run("flat", a.n, rng, True), run("random", a.n, rng, False)]
    print(json.dumps({"device": torch.cuda.get_device_name(0), "results": res}, indent=2))
    ok = all(float(str(r["match_rate"])) >= 0.999 for r in res)
    print("PASS" if ok else "FAIL: numpy port does not reproduce the kernel")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
