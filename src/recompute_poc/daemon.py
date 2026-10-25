"""Recompute daemon (VM-B): /ingest, teacher-forced prefill via local vLLM, scoring."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI

from .records import Workunit
from .scoring import Row, score_workunit
from .store import Store

PREFILL_TOP = 20
DEFAULT_DELTA_MAX = 5.0


@dataclass(frozen=True)
class DaemonConfig:
    vllm_url: str  # VM-B's local vLLM (same image + Workload Manifest as VM-A)
    manifests: frozenset[str]  # manifest ids this daemon can reproduce
    delta_max: float = DEFAULT_DELTA_MAX


def parse_prompt_logprobs(raw: list[dict[str, Any] | None], ids: list[int]) -> list[Row]:
    """vLLM returns, per position, {token id (str): {logprob, rank}} (top-k + actual token)."""
    rows: list[Row] = []
    for i, entry in enumerate(raw):
        if entry is None:
            rows.append({ids[i]: 0.0})  # position 0 has no prediction
            continue
        rows.append({int(k): float(v["logprob"]) for k, v in entry.items()})
    return rows


async def prefill(http: httpx.AsyncClient, base: str, model: str, ids: list[int]) -> list[Row]:
    r = await http.post(
        base.rstrip("/") + "/v1/completions",
        json={"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
              "prompt_logprobs": PREFILL_TOP},
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["prompt_logprobs"]
    return parse_prompt_logprobs(raw, ids)


async def score_one(
    store: Store, http: httpx.AsyncClient, cfg: DaemonConfig, workunit_id: str, record: str
) -> None:
    wu = Workunit.model_validate_json(record)
    choice = wu.choices[0]
    ids = list(wu.prompt_token_ids) + list(choice.token_ids)
    try:
        rows = await prefill(http, cfg.vllm_url, wu.model, ids)
    except (httpx.HTTPError, KeyError, ValueError) as e:
        store.mark(workunit_id, "error", f"prefill failed: {e!r}"[:300])
        return
    s = score_workunit(wu, rows, cfg.delta_max)
    store.save_score(workunit_id, wu.model, wu.params.temperature, s.n_tokens, s.match_rate,
                     s.mean_margin, s.filtered_rate, cfg.delta_max)


def create_app(store: Store, cfg: DaemonConfig, http: httpx.AsyncClient | None = None,
               run_worker: bool = True) -> FastAPI:
    client = http or httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    stop = asyncio.Event()

    async def worker() -> None:
        while not stop.is_set():
            batch = store.pending()
            if not batch:
                await asyncio.sleep(0.5)
                continue
            await asyncio.gather(*(score_one(store, client, cfg, i, r) for i, r in batch))

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(worker()) if run_worker else None
        yield
        stop.set()
        if task:
            await task

    app = FastAPI(title="recompute-poc daemon", lifespan=lifespan)

    @app.post("/ingest")
    async def ingest(records: list[dict[str, Any]]) -> dict[str, int]:
        added = dup = bad = 0
        for rec in records:
            try:
                wu = Workunit.model_validate(rec)
            except ValueError:
                bad += 1
                continue
            reason: str | None = None
            if wu.manifest_id not in cfg.manifests:
                status, reason = "rejected", "unknown manifest_id"
            elif not wu.scorable:
                status, reason = "unscorable", wu.unscorable_reason
            else:
                status = "pending"
            fresh = store.insert_workunit(
                wu.workunit_id, wu.manifest_id, wu.model, wu.params.temperature, status,
                reason, wu.model_dump_json())
            added += fresh
            dup += not fresh
        return {"added": added, "duplicate": dup, "invalid": bad}

    @app.get("/status")
    async def status() -> dict[str, int]:
        return store.counts()

    return app
