"""Fast-path scorer: verified sampling (match) + Token-DiFR (margin) in one replay.

Works on sparse rows (top-20 logprobs from teacher-forced prefill). That is exact for
top_k <= 20 because the Gumbel argmax only ever runs over the top-k set, and the
per-row logsumexp that separates logprobs from logits cancels in every margin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .gumbel import gumbel_noise
from .records import SamplingParams, Workunit

Row = Mapping[int, float]  # token id -> logprob (top-20 plus the claimed token)

MAX_SPARSE_TOP_K = 20
MIN_TEMPERATURE = 0.01  # vLLM raises temperatures in (0, 0.01) to 0.01


@dataclass(frozen=True)
class TokenScore:
    match: bool
    margin: float  # clipped at delta_max; 0 when the replay agrees
    filtered: bool  # claimed token was outside the top-k/top-p/min-p set


@dataclass(frozen=True)
class WorkunitScore:
    n_tokens: int
    match_rate: float
    mean_margin: float
    filtered_rate: float


def _keep_mask(logits: np.ndarray, p: SamplingParams) -> np.ndarray:
    """Candidate mask after min_p, top_k, top_p, on temperature-scaled logits."""
    keep = np.ones(logits.shape, dtype=bool)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    if p.min_p > 0.0:
        keep &= probs >= p.min_p * probs.max()
    if 0 < p.top_k < len(logits):
        kth = np.sort(logits)[::-1][p.top_k - 1]
        keep &= logits >= kth
    if p.top_p < 1.0:
        masked = np.where(keep, probs, 0.0)
        masked = masked / masked.sum()
        order = np.argsort(-masked, kind="stable")
        cum = np.cumsum(masked[order])
        drop_sorted = (cum - masked[order]) >= p.top_p
        drop = np.zeros(logits.shape, dtype=bool)
        drop[order[drop_sorted]] = True
        keep &= ~drop
    return keep


def score_token(
    row: Row, p: SamplingParams, claimed: int, pos: int, delta_max: float
) -> TokenScore:
    ids = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
    lp = np.fromiter(row.values(), dtype=np.float64, count=len(row))
    if p.temperature == 0.0:
        keep = np.ones(ids.shape, dtype=bool)
        if 0 < p.top_k < len(ids):
            keep &= lp >= np.sort(lp)[::-1][p.top_k - 1]
        z = np.where(keep, lp, -np.inf)
    else:
        temp = max(p.temperature, MIN_TEMPERATURE)
        scaled = (lp / temp).astype(np.float32)
        keep = _keep_mask(scaled.astype(np.float64), p)
        noise = gumbel_noise(p.seed, pos, ids)
        z = np.where(keep, scaled.astype(np.float64) + noise.astype(np.float64), -np.inf)
    hat = int(ids[int(np.argmax(z))])
    where = np.nonzero(ids == claimed)[0]
    if len(where) == 0 or not np.isfinite(z[where[0]]):
        return TokenScore(match=False, margin=delta_max, filtered=True)
    margin = float(z.max() - z[where[0]])
    return TokenScore(match=hat == claimed, margin=min(margin, delta_max), filtered=False)


def unscorable_reason(request: Mapping[str, object], p: SamplingParams) -> str | None:
    """Why a request cannot go through the sparse fast path in this PoC (None = scorable)."""
    if int(str(request.get("n", 1))) != 1:
        return "n>1"
    for key in ("tools", "tool_choice", "response_format", "logit_bias", "guided_json",
                "guided_regex", "bad_words", "allowed_token_ids"):
        if request.get(key):
            return f"unsupported field: {key}"
    for key in ("presence_penalty", "frequency_penalty"):
        if float(str(request.get(key, 0.0))) != 0.0:
            return f"unsupported field: {key}"
    if float(str(request.get("repetition_penalty", 1.0))) != 1.0:
        return "unsupported field: repetition_penalty"
    msgs = request.get("messages")
    if isinstance(msgs, Sequence):
        for m in msgs:
            if isinstance(m, Mapping) and not isinstance(m.get("content", ""), str):
                return "non-text message content"
    if p.temperature > 0.0 and not 0 < p.top_k <= MAX_SPARSE_TOP_K:
        return f"top_k outside 1..{MAX_SPARSE_TOP_K}"
    return None


def score_workunit(
    wu: Workunit, rows: Sequence[Row], delta_max: float, pos_offset: int = -1
) -> WorkunitScore:
    """`rows[j]` is the prefill row for sequence index j (prompt ids + output ids).

    Output token i sits at sequence index prompt_len + i and was drawn at
    pos = prompt_len + i + pos_offset (offset -1 per the research doc, to confirm on GPU).
    """
    choice = wu.choices[0]
    plen = len(wu.prompt_token_ids)
    scores = [
        score_token(rows[plen + i], wu.params, tok, plen + i + pos_offset, delta_max)
        for i, tok in enumerate(choice.token_ids)
    ]
    n = len(scores)
    if n == 0:
        return WorkunitScore(0, 1.0, 0.0, 0.0)
    return WorkunitScore(
        n_tokens=n,
        match_rate=sum(s.match for s in scores) / n,
        mean_margin=sum(s.margin for s in scores) / n,
        filtered_rate=sum(s.filtered for s in scores) / n,
    )
