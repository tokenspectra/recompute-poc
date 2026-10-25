"""Re-run the honest-vs-control evaluation with a calibration/holdout split stratified by load cell.

Usage: python scripts/stratified_eval.py data/run-20261025 [cell_size=64]

Each load cell's Workunits (consecutive in the spool, so one cell = `cell_size` lines) are split
alternately into calibration (even index) and holdout (odd index). This removes the confound in
`rpoc evaluate`, whose halves follow completion order and therefore concurrency.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from pathlib import Path

from recompute_poc.evaluate import evaluate

CELLS = [(c, s, t) for c in (1, 8, 32) for s in (False, True) for t in (0.0, 0.6)]


def load_margins(db: Path) -> dict[str, list[float]]:
    con = sqlite3.connect(db)
    return {w: [float(v) for v in json.loads(m)]
            for w, m in con.execute("SELECT workunit_id, margins FROM scores")}


def main() -> None:
    root = Path(sys.argv[1])
    cell_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    ids = [json.loads(line)["workunit_id"] for line in (root / "honest.jsonl").open()]
    assert len(ids) == cell_size * len(CELLS), "spool does not match the 12-cell matrix"
    honest = load_margins(root / "honest.sqlite")
    cal: list[float] = []
    hold: list[float] = []
    for k in range(len(CELLS)):
        for j, wid in enumerate(ids[k * cell_size:(k + 1) * cell_size]):
            (cal if j % 2 == 0 else hold).extend(honest[wid])
    control = [v for ms in load_margins(root / "control.sqlite").values() for v in ms]
    rows = evaluate(cal + hold, control, [16, 32, 64, 128, 256, 512, 1024], 0.999, 20)
    print(f"cal tokens={len(cal)} hold tokens={len(hold)} control tokens={len(control)}")
    print("window cal/hold/test  tau     FPR    TPR    AUC")
    for r in rows:
        assert r.tau is not None and r.fpr is not None and r.tpr is not None
        print(f"{r.window_tokens:>6} {r.cal_windows:>4}/{r.holdout_windows:>4}/{r.test_windows:>4}"
              f"  {r.tau:.4f}  {r.fpr:.3f}  {r.tpr:.3f}  {r.auc:.3f}")
    for k, (c, s, t) in enumerate(CELLS):
        v = [x for w in ids[k * cell_size:(k + 1) * cell_size] for x in honest[w]]
        print(f"conc={c:>2} stream={int(s)} T={t}: mean margin {statistics.mean(v):.4f}")


if __name__ == "__main__":
    main()
