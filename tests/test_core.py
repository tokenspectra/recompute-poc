from typing import Any

import numpy as np

from recompute_poc.gumbel import gumbel_noise, murmur3_hash32, murmur3_uniform32
from recompute_poc.records import Choice, SamplingParams, Workunit
from recompute_poc.scoring import score_token, score_workunit, unscorable_reason


def test_hash_is_deterministic_and_keyed():
    ids = np.arange(1000, dtype=np.uint32)
    a = murmur3_hash32(123, 7, ids)
    assert np.array_equal(a, murmur3_hash32(123, 7, ids))
    assert not np.array_equal(a, murmur3_hash32(124, 7, ids))
    assert not np.array_equal(a, murmur3_hash32(123, 8, ids))
    assert len(set(a.tolist())) > 990  # near-injective over 1000 ids


def test_uniform_in_open_interval_and_flat():
    u = murmur3_uniform32(42, 3, np.arange(200_000, dtype=np.uint32))
    assert float(u.min()) > 0.0 and float(u.max()) < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.005
    hist, _ = np.histogram(u, bins=10, range=(0, 1))
    assert hist.min() > 0.09 * len(u)


def test_noise_is_gumbel_shaped():
    g = gumbel_noise(9, 1, np.arange(200_000, dtype=np.int64)).astype(np.float64)
    assert np.isfinite(g).all()
    assert abs(g.mean() - 0.5772) < 0.02  # Euler-Mascheroni
    assert abs(g.var() - np.pi**2 / 6) < 0.05


def test_negative_seed_masks_to_64_bits():
    ids = np.arange(8, dtype=np.uint32)
    assert np.array_equal(murmur3_hash32(-1, 0, ids), murmur3_hash32(2**64 - 1, 0, ids))


def params(**kw: Any) -> SamplingParams:
    base: dict[str, Any] = {
        "seed": 5, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 64,
    }
    base.update(kw)
    return SamplingParams(**base)


def sample_like_vllm(row, p, pos):
    """Honest sampler: the same math the scorer replays, so match must be 100%."""
    ids = np.array(list(row), dtype=np.int64)
    lp = np.array(list(row.values()))
    z = lp / p.temperature + gumbel_noise(p.seed, pos, ids)
    return int(ids[int(np.argmax(z))])


def make_row(rng, n=20):
    ids = rng.choice(150_000, size=n, replace=False)
    lp = np.sort(rng.normal(size=n))[::-1]
    lp = lp - np.log(np.exp(lp).sum())
    return {int(i): float(v) for i, v in zip(ids, lp)}


def test_honest_tokens_match_with_zero_margin():
    rng = np.random.default_rng(0)
    p = params(top_p=1.0)  # top-p off: the helper above samples over all 20
    for pos in range(200):
        row = make_row(rng)
        tok = sample_like_vllm(row, p, pos)
        s = score_token(row, p, tok, pos, delta_max=5.0)
        assert s.match and s.margin == 0.0 and not s.filtered


def test_wrong_token_gets_positive_margin():
    rng = np.random.default_rng(1)
    p = params(top_p=1.0)
    row = make_row(rng)
    honest = sample_like_vllm(row, p, 3)
    other = next(t for t in row if t != honest)
    s = score_token(row, p, other, 3, delta_max=50.0)
    assert not s.match and s.margin > 0.0


def test_token_missing_from_row_is_filtered_at_delta_max():
    s = score_token({1: -0.1, 2: -2.0}, params(), claimed=999, pos=0, delta_max=4.0)
    assert s.filtered and s.margin == 4.0


def test_greedy_needs_no_noise():
    row = {10: -0.2, 11: -1.5, 12: -3.0}
    p = params(temperature=0.0)
    assert score_token(row, p, 10, 0, 5.0).match
    s = score_token(row, p, 11, 0, 5.0)
    assert not s.match and abs(s.margin - 1.3) < 1e-9


def test_top_p_filters_tail():
    row = {1: np.log(0.9), 2: np.log(0.08), 3: np.log(0.02)}
    p = params(top_p=0.5, temperature=1.0)
    s = score_token(row, p, 3, 0, 7.0)
    assert s.filtered and not s.match


def test_workunit_positions_and_aggregation():
    rng = np.random.default_rng(2)
    p = params(top_p=1.0)
    prompt = [1, 2, 3, 4]
    rows = [make_row(rng) for _ in range(len(prompt) + 6)]
    out = [sample_like_vllm(rows[len(prompt) + i], p, len(prompt) + i - 1) for i in range(6)]
    wu = Workunit(
        workunit_id="w", manifest_id="m", received_at=0, completed_at=1, model="x",
        upstream="u", client_request={}, forwarded_request={}, params=p,
        prompt_token_ids=prompt, choices=[Choice(index=0, token_ids=out)], stream=False,
    )
    s = score_workunit(wu, rows, delta_max=5.0)
    assert s.n_tokens == 6 and s.match_rate == 1.0 and s.mean_margin == 0.0


def test_unscorable_reasons():
    p = params()
    assert unscorable_reason({"messages": [{"role": "user", "content": "hi"}]}, p) is None
    assert unscorable_reason({"n": 2}, p) == "n>1"
    assert unscorable_reason({"tools": [{}]}, p) == "unsupported field: tools"
    assert unscorable_reason({"presence_penalty": 0.5}, p) == "unsupported field: presence_penalty"
    assert unscorable_reason({}, params(top_k=0)) is not None
