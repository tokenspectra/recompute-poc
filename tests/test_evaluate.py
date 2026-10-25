from pathlib import Path

import numpy as np
from fake_vllm import create_fake
from test_e2e import MODEL, drive, make_gateway, run_scoring

from recompute_poc.evaluate import auc, evaluate, windows


def test_windows_and_auc_basics() -> None:
    assert np.allclose(windows([1, 3, 2, 4, 9], 2), [2.0, 3.0])  # tail dropped
    assert windows([1.0], 4).size == 0
    assert auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1.0])) == 0.5


def test_too_few_windows_is_reported_not_guessed() -> None:
    rows = evaluate([0.0] * 100, [1.0] * 100, [64], min_windows=20)
    assert rows[0].tau is None and rows[0].fpr is None and rows[0].tpr is None


async def test_control_swap_is_detected_and_honest_is_not(tmp_path: Path) -> None:
    dirs = {k: tmp_path / k for k in ("h_op", "h_ver", "s_op", "s_ver")}
    for d in dirs.values():
        d.mkdir()
    # Honest: operator serves A, verifier is A on slightly different numerics.
    c, spool = make_gateway(dirs["h_op"], create_fake("A"))
    await drive(c, 60)
    honest = await run_scoring(dirs["h_ver"], spool, create_fake("A", jitter=0.02, jitter_seed=5))
    # Control: operator serves B but declares A (same manifest id); verifier is still A.
    c2, spool2 = make_gateway(dirs["s_op"], create_fake("B"))
    await drive(c2, 40)
    swap = await run_scoring(dirs["s_ver"], spool2, create_fake("A", jitter=0.02, jitter_seed=6))

    hm, sm = honest.token_margins(MODEL), swap.token_margins(MODEL)
    assert len(hm) == 60 * 24 and len(sm) == 40 * 24
    rows = {r.window_tokens: r for r in evaluate(hm, sm, [32], quantile=0.999, min_windows=10)}
    r = rows[32]
    assert r.tau is not None and r.fpr is not None and r.tpr is not None and r.auc is not None
    assert r.fpr <= 0.1 and r.tpr == 1.0 and r.auc > 0.99
