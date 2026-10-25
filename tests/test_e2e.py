import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fake_vllm import create_fake

from recompute_poc import daemon as dmn
from recompute_poc import gateway as gw
from recompute_poc.cli import calibrate, verdict
from recompute_poc.shipper import ship_once
from recompute_poc.store import Store

MODEL = "fake-model"
MANIFEST = "manifest-1"
DEFAULTS = {"temperature": 0.7, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "max_tokens": 24}


def make_gateway(tmp: Path, vllm_a: object) -> tuple[httpx.AsyncClient, Path]:
    spool = tmp / "spool.jsonl"
    route = gw.ModelRoute("http://vllm-a", MANIFEST, DEFAULTS)
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=vllm_a))  # type: ignore[arg-type]
    app = gw.create_app(gw.GatewayConfig({MODEL: route}, spool), client=up)
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    return c, spool


async def run_scoring(tmp: Path, spool: Path, vllm_b: object) -> Store:
    store = Store(tmp / "b.sqlite")
    cfg = dmn.DaemonConfig(vllm_url="http://vllm-b", manifests=frozenset({MANIFEST}))
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=vllm_b))  # type: ignore[arg-type]
    app = dmn.create_app(store, cfg, http=up, run_worker=False)
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b")
    recs = [json.loads(line) for line in spool.read_text().splitlines()]
    r = await c.post("/ingest", json=recs)
    assert r.json()["added"] == len(recs)
    again = await c.post("/ingest", json=recs)
    assert again.json()["duplicate"] == len(recs)  # dedupe by workunit_id
    for wid, rec in store.pending(1000):
        await dmn.score_one(store, up, cfg, wid, rec)
    return store


async def drive(c: httpx.AsyncClient, n: int, stream: bool = False) -> None:
    for i in range(n):
        body = {"model": MODEL, "stream": stream,
                "messages": [{"role": "user", "content": f"question number {i}"}]}
        r = await c.post("/v1/chat/completions", json=body)
        assert r.status_code == 200


async def test_inject_and_strip_non_stream(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake())
    r = await c.post("/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": "hi there"}]})
    data = r.json()
    assert "prompt_token_ids" not in data and "token_ids" not in data["choices"][0]
    assert data["choices"][0]["logprobs"] is None
    rec = json.loads(spool.read_text().splitlines()[0])
    assert rec["forwarded_request"]["return_token_ids"] is True
    assert rec["forwarded_request"]["top_logprobs"] == 20
    assert 0 <= rec["params"]["seed"] < 2**63 and rec["params"]["client_seed"] is False
    assert len(rec["choices"][0]["token_ids"]) == 24 and rec["prompt_token_ids"]
    assert rec["choices"][0]["logprobs"][0]["top"]  # top-20 captured


async def test_client_asked_fields_are_kept_and_seed_marked(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake())
    r = await c.post("/v1/chat/completions", json={
        "model": MODEL, "seed": 77, "logprobs": True, "return_token_ids": True,
        "messages": [{"role": "user", "content": "hi"}]})
    d = r.json()
    assert d["prompt_token_ids"] and d["choices"][0]["token_ids"] and d["choices"][0]["logprobs"]
    rec = json.loads(spool.read_text().splitlines()[0])
    assert rec["params"]["seed"] == 77 and rec["params"]["client_seed"] is True


async def test_stream_stripped_and_reassembled(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake())
    r = await c.post("/v1/chat/completions", json={
        "model": MODEL, "stream": True, "messages": [{"role": "user", "content": "stream me"}]})
    chunks = [json.loads(line[5:]) for line in r.text.splitlines()
              if line.startswith("data:") and "[DONE]" not in line]
    assert chunks and all("prompt_token_ids" not in ch for ch in chunks)
    assert all("token_ids" not in ch["choices"][0] for ch in chunks)
    rec = json.loads(spool.read_text().splitlines()[0])
    assert rec["stream"] is True and len(rec["choices"][0]["token_ids"]) == 24
    assert rec["prompt_token_ids"] and rec["choices"][0]["finish_reason"] == "length"


async def test_unknown_model_and_unscorable(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake())
    assert (await c.post("/v1/chat/completions", json={"model": "nope"})).status_code == 404
    await c.post("/v1/chat/completions", json={
        "model": MODEL, "n": 2, "messages": [{"role": "user", "content": "x"}]})
    rec = json.loads(spool.read_text().splitlines()[0])
    assert rec["scorable"] is False and rec["unscorable_reason"] == "n>1"


async def test_honest_identical_numerics_scores_perfect(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake("A"))
    await drive(c, 6)
    await drive(c, 4, stream=True)
    store = await run_scoring(tmp_path, spool, create_fake("A"))
    rows = store.scores()
    assert len(rows) == 10
    assert all(r[3] == 1.0 and r[4] == 0.0 for r in rows)  # match 100%, margin 0


async def test_honest_with_numeric_noise_stays_close(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake("A", jitter=0.0))
    await drive(c, 30)
    store = await run_scoring(tmp_path, spool, create_fake("A", jitter=0.02, jitter_seed=1))
    rows = store.scores()
    assert np.mean([r[3] for r in rows]) > 0.9
    assert np.mean([r[4] for r in rows]) < 0.2


async def test_swapped_model_is_clearly_separated(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake("A"))
    await drive(c, 30)
    honest = await run_scoring(tmp_path / "h" if (tmp_path / "h").mkdir() is None else tmp_path,
                               spool, create_fake("A", jitter=0.02, jitter_seed=2))
    cheat_dir = tmp_path / "c"
    cheat_dir.mkdir()
    cheat = await run_scoring(cheat_dir, spool, create_fake("B"))
    h = np.mean([r[4] for r in honest.scores()])
    x = np.mean([r[4] for r in cheat.scores()])
    assert x > 10 * max(h, 0.01)


async def test_rejects_unknown_manifest(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake())
    await drive(c, 1)
    store = Store(tmp_path / "r.sqlite")
    cfg = dmn.DaemonConfig(vllm_url="http://x", manifests=frozenset({"other"}))
    app = dmn.create_app(store, cfg, run_worker=False)
    cl = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b")
    await cl.post("/ingest", json=[json.loads(spool.read_text().splitlines()[0])])
    assert store.counts() == {"rejected": 1}


async def test_calibrate_and_verdict(tmp_path: Path) -> None:
    c, spool = make_gateway(tmp_path, create_fake("A"))
    await drive(c, 40)
    store = await run_scoring(tmp_path, spool, create_fake("A", jitter=0.02, jitter_seed=3))
    # quantile < 1 on a handful of windows puts tau below the calibration set's own max, so
    # the set would fail by construction. Use the max here; real runs need many windows.
    info = calibrate(store, MODEL, window_tokens=100, quantile=1.0, delta_max=5.0, min_windows=5)
    assert info["windows"] >= 5 and float(str(info["tau"])) >= 0.0  # type: ignore[arg-type]
    v = verdict(store, MODEL, 3600, 100)
    assert v["status"] == "pass"
    cheat_dir = tmp_path / "cheat"
    cheat_dir.mkdir()
    cheat = await run_scoring(cheat_dir, spool, create_fake("B"))
    cheat.set_threshold(MODEL, float(str(info["tau"])), 5.0, 100, 8, 1.0)
    assert verdict(cheat, MODEL, 3600, 100)["status"] == "fail"
    with pytest.raises(SystemExit):
        calibrate(store, MODEL, window_tokens=100, quantile=0.99, delta_max=5.0, min_windows=999)
    assert store.window_stats(3600)[0].workunits == 40


def test_shipper_resumes_and_waits_for_partial_line(tmp_path: Path) -> None:
    spool, off = tmp_path / "s.jsonl", tmp_path / "s.offset"
    spool.write_text('{"a":1}\n{"a":2}\n{"a":3')  # last line still being written
    got: list[list[dict[str, int]]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        got.append(json.loads(req.content))
        return httpx.Response(200, json={})

    cl = httpx.Client(transport=httpx.MockTransport(handler))
    assert ship_once(spool, off, "http://b/ingest", client=cl) == 2
    assert ship_once(spool, off, "http://b/ingest", client=cl) == 0
    with spool.open("a") as f:
        f.write('}\n')
    assert ship_once(spool, off, "http://b/ingest", client=cl) == 1
    assert got == [[{"a": 1}, {"a": 2}], [{"a": 3}]]
