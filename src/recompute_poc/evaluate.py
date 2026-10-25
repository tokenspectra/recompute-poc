"""Honest-vs-control evaluation: how many tokens does it take to tell a swap from honest?

Honest tokens are split in two contiguous halves: the first sets tau, the second is a
held-out set for the false-positive rate. Tau is only trustworthy once the calibration half
has >= min_windows windows; with fewer, the row is reported as insufficient, not guessed.
Resolution of the FPR is 1 / (number of held-out windows).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvalRow:
    window_tokens: int
    cal_windows: int
    holdout_windows: int
    test_windows: int
    tau: float | None
    fpr: float | None  # honest held-out windows above tau
    tpr: float | None  # control windows above tau (detection rate)
    auc: float | None  # P(control window > honest held-out window), ties count half


def windows(margins: Sequence[float], size: int) -> np.ndarray:
    a = np.asarray(margins, dtype=np.float64)
    n = len(a) // size
    return a[: n * size].reshape(n, size).mean(axis=1) if n else np.empty(0)


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    ordered = np.sort(neg)
    lo = np.searchsorted(ordered, pos, side="left")
    hi = np.searchsorted(ordered, pos, side="right")
    return float(((lo + hi) / 2).sum() / (len(neg) * len(pos)))


def evaluate(
    honest: Sequence[float], test: Sequence[float], sizes: Sequence[int],
    quantile: float = 0.999, min_windows: int = 20,
) -> list[EvalRow]:
    half = len(honest) // 2
    cal_t, hold_t = honest[:half], honest[half:]
    rows: list[EvalRow] = []
    for w in sizes:
        c, h, t = windows(cal_t, w), windows(hold_t, w), windows(test, w)
        if len(c) < min_windows:
            rows.append(EvalRow(w, len(c), len(h), len(t), None, None, None, None))
            continue
        tau = float(np.quantile(c, quantile))
        have_both = len(h) > 0 and len(t) > 0
        rows.append(EvalRow(
            w, len(c), len(h), len(t), tau,
            float((h > tau).mean()) if len(h) else None,
            float((t > tau).mean()) if len(t) else None,
            auc(h, t) if have_both else None,
        ))
    return rows
