import json
from pathlib import Path

import httpx
from fake_vllm import create_fake

from recompute_poc import gateway as gw
from recompute_poc.loadsuite import Cell, matrix, prompts, run_suite

MODEL = "fake-model"


def test_prompts_are_deterministic() -> None:
    assert prompts(50) == prompts(50)
    assert len(set(prompts(32))) == 32


def test_matrix_covers_concurrency_stream_and_temperature() -> None:
    m = matrix()
    assert len(m) == 12
    assert {c.concurrency for c in m} == {1, 8, 32}
    assert {c.stream for c in m} == {False, True}
    assert {c.temperature for c in m} == {0.0, 0.6}


async def test_suite_drives_gateway_and_spools_every_request(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    route = gw.ModelRoute("http://a", "m1", {"temperature": 0.6, "top_p": 1.0, "top_k": 20,
                                             "min_p": 0.0, "max_tokens": 8})
    up = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_fake()))
    app = gw.create_app(gw.GatewayConfig({MODEL: route}, spool), client=up)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    out = tmp_path / "load.jsonl"
    cells = [Cell(1, False, 0.0), Cell(8, True, 0.6)]
    res = await run_suite("http://gw", MODEL, 6, 8, "t", out, cells, http=client)
    assert [r["ok"] for r in res] == [6, 6]
    lines = [json.loads(x) for x in spool.read_text().splitlines()]
    assert len(lines) == 12
    assert {x["params"]["temperature"] for x in lines} == {0.0, 0.6}
    assert {x["stream"] for x in lines} == {False, True}
    assert len(out.read_text().splitlines()) == 2
