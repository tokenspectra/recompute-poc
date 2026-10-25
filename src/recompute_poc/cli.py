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

    a = ap.parse_args(argv)
    if a.cmd == "gateway":
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
