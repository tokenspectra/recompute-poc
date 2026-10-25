"""`rpoc`: serve the gateway / daemon / shipper, calibrate the threshold, query windows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import uvicorn

from .store import Store, chunk_windows

WINDOWS = {"5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "24h": 86400}


def _fmt(v: float | None, digits: int = 4) -> str:
    return "  n/a " if v is None else f"{v:.{digits}f}"


def calibrate(store: Store, model: str, window_tokens: int, quantile: float,
              delta_max: float, min_windows: int = 20) -> dict[str, object]:
    """tau = a high quantile of the honest windowed mean margin (see docs/design)."""
    rows = store.scores(model=model)
    wins = chunk_windows(rows, window_tokens)
    if len(wins) < min_windows:
        raise SystemExit(
            f"need >= {min_windows} windows of {window_tokens} tokens for {model}; "
            f"have {len(wins)} ({sum(r[2] for r in rows)} tokens). Run more honest load."
        )
    tau = float(np.quantile(np.array(wins), quantile))
    store.set_threshold(model, tau, delta_max, window_tokens, len(wins), quantile)
    return {"model": model, "tau": tau, "windows": len(wins), "window_tokens": window_tokens,
            "quantile": quantile, "honest_mean": float(np.mean(wins)),
            "honest_max": float(np.max(wins))}


def verdict(store: Store, model: str, window_s: float, window_tokens: int) -> dict[str, object]:
    th = store.threshold(model)
    rows = store.scores(model=model, since=None)
    import time
    rows = [r for r in rows if r[0] >= time.time() - window_s]
    wins = chunk_windows(rows, window_tokens)
    if th is None:
        return {"model": model, "status": "uncalibrated", "windows": len(wins)}
    tau = th[0]
    failed = [w for w in wins if w > tau]
    return {"model": model, "tau": tau, "windows": len(wins), "windows_over_tau": len(failed),
            "status": "no-data" if not wins else ("pass" if not failed else "fail")}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rpoc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gateway", help="run the API Tap on VM-A")
    g.add_argument("--config", type=Path, required=True)
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--port", type=int, default=8000)

    d = sub.add_parser("daemon", help="run the recompute daemon on VM-B")
    d.add_argument("--db", type=Path, default=Path("data/recompute.sqlite"))
    d.add_argument("--vllm-url", default="http://127.0.0.1:8001")
    d.add_argument("--manifest", action="append", required=True, help="accepted manifest_id")
    d.add_argument("--delta-max", type=float, default=5.0)
    d.add_argument("--host", default="0.0.0.0")
    d.add_argument("--port", type=int, default=9000)

    s = sub.add_parser("ship", help="ship the spool to VM-B")
    s.add_argument("--spool", type=Path, required=True)
    s.add_argument("--ingest-url", required=True)

    c = sub.add_parser("calibrate", help="derive tau from honest runs")
    c.add_argument("--db", type=Path, default=Path("data/recompute.sqlite"))
    c.add_argument("--model", required=True)
    c.add_argument("--window-tokens", type=int, default=512)
    c.add_argument("--quantile", type=float, default=0.999)
    c.add_argument("--delta-max", type=float, default=5.0)
    c.add_argument("--min-windows", type=int, default=20)

    q = sub.add_parser("stats", help="aggregates over the last window")
    q.add_argument("--db", type=Path, default=Path("data/recompute.sqlite"))
    q.add_argument("--window", choices=sorted(WINDOWS), default="1h")
    q.add_argument("--model")
    q.add_argument("--window-tokens", type=int, default=512)
    q.add_argument("--json", action="store_true")

    ev = sub.add_parser("evaluate", help="honest vs control: tokens needed to detect")
    ev.add_argument("--honest-db", type=Path, required=True)
    ev.add_argument("--test-db", type=Path, required=True)
    ev.add_argument("--model", required=True)
    ev.add_argument("--sizes", default="32,64,128,256,512,1024,2048")
    ev.add_argument("--quantile", type=float, default=0.999)
    ev.add_argument("--min-windows", type=int, default=20)
    ev.add_argument("--json", action="store_true")

    ld = sub.add_parser("load", help="run the load suite against the gateway")
    ld.add_argument("--url", required=True)
    ld.add_argument("--model", required=True)
    ld.add_argument("--per-cell", type=int, default=32)
    ld.add_argument("--max-tokens", type=int, default=256)
    ld.add_argument("--label", default="honest")
    ld.add_argument("--out", type=Path, default=Path("data/load.jsonl"))
    ld.add_argument("--quick", action="store_true", help="concurrency 1 and 8 only")

    a = ap.parse_args(argv)
    if a.cmd == "load":
        import asyncio

        from .loadsuite import matrix, run_suite
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cells = matrix((1, 8) if a.quick else (1, 8, 32))
        asyncio.run(run_suite(a.url, a.model, a.per_cell, a.max_tokens, a.label, a.out, cells))
    elif a.cmd == "gateway":
        from .gateway import GatewayConfig, ModelRoute, create_app
        raw = json.loads(a.config.read_text())
        cfg = GatewayConfig(
            routes={k: ModelRoute(**v) for k, v in raw["routes"].items()},
            spool_path=Path(raw["spool_path"]))
        uvicorn.run(create_app(cfg), host=a.host, port=a.port)
    elif a.cmd == "daemon":
        from .daemon import DaemonConfig
        from .daemon import create_app as create_daemon
        a.db.parent.mkdir(parents=True, exist_ok=True)
        cfg_d = DaemonConfig(vllm_url=a.vllm_url, manifests=frozenset(a.manifest),
                             delta_max=a.delta_max)
        uvicorn.run(create_daemon(Store(a.db), cfg_d), host=a.host, port=a.port)
    elif a.cmd == "ship":
        from .shipper import run_forever
        run_forever(a.spool, a.spool.with_suffix(".offset"), a.ingest_url)
    elif a.cmd == "calibrate":
        print(json.dumps(calibrate(Store(a.db), a.model, a.window_tokens, a.quantile,
                                   a.delta_max, a.min_windows), indent=2))
    elif a.cmd == "evaluate":
        from .evaluate import evaluate
        honest_m = Store(a.honest_db).token_margins(a.model)
        test_m = Store(a.test_db).token_margins(a.model)
        sizes = [int(x) for x in a.sizes.split(",")]
        rows = evaluate(honest_m, test_m, sizes, a.quantile, a.min_windows)
        if a.json:
            print(json.dumps({"honest_tokens": len(honest_m), "test_tokens": len(test_m),
                              "rows": [r.__dict__ for r in rows]}, indent=2))
        else:
            print(f"honest tokens={len(honest_m)} control tokens={len(test_m)} "
                  f"quantile={a.quantile}")
            print("window  cal/hold/test  tau      FPR     TPR     AUC")
            for r in rows:
                print(f"{r.window_tokens:>6}  {r.cal_windows:>3}/{r.holdout_windows:>3}/"
                      f"{r.test_windows:>3}    {_fmt(r.tau)}  {_fmt(r.fpr, 3)}  "
                      f"{_fmt(r.tpr, 3)}  {_fmt(r.auc, 3)}"
                      + ("  (too few windows)" if r.tau is None else ""))
    elif a.cmd == "stats":
        st = Store(a.db)
        stats = st.window_stats(WINDOWS[a.window], a.model)
        out = {"window": a.window, "counts": st.counts(),
               "models": [s.__dict__ for s in stats],
               "verdicts": [verdict(st, s.model, WINDOWS[a.window], a.window_tokens)
                            for s in stats]}
        if a.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"window={a.window} workunits={out['counts']}")
            for ws, v in zip(stats, out["verdicts"], strict=True):
                print(f"{ws.model}: wu={ws.workunits} tok={ws.tokens} match={ws.match_rate:.3f} "
                      f"margin mean={ws.mean_margin:.4f} p50={ws.p50:.4f} p95={ws.p95:.4f} "
                      f"p99={ws.p99:.4f} filtered={ws.filtered_rate:.4f} -> {v['status']}")


if __name__ == "__main__":
    main()
