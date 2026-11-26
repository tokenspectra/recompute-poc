"""API Tap gateway (VM-A): injects seed/logprobs/token ids, strips them, spools Workunits.

Scope: chat completions only, vLLM adapter only.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .records import Choice, SamplingParams, TokenLogprob, Workunit
from .scoring import unscorable_reason

TOP_LOGPROBS = 20


@dataclass(frozen=True)
class ModelRoute:
    upstream: str  # base URL of the vLLM server, e.g. http://127.0.0.1:8001
    manifest_id: str
    defaults: dict[str, float]  # temperature, top_p, top_k, min_p, max_tokens


@dataclass(frozen=True)
class GatewayConfig:
    routes: dict[str, ModelRoute]
    spool_path: Path


@dataclass(frozen=True)
class Asked:
    logprobs: bool
    token_ids: bool


def build_forwarded(
    body: dict[str, Any], route: ModelRoute
) -> tuple[dict[str, Any], SamplingParams, Asked]:
    d = route.defaults
    client_seed = body.get("seed") is not None
    seed = int(body["seed"]) if client_seed else random.SystemRandom().getrandbits(63)
    asked = Asked(logprobs=bool(body.get("logprobs")), token_ids=bool(body.get("return_token_ids")))
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or d["max_tokens"])
    params = SamplingParams(
        seed=seed,
        client_seed=client_seed,
        temperature=float(body.get("temperature", d["temperature"])),
        top_p=float(body.get("top_p", d["top_p"])),
        top_k=int(body.get("top_k", d["top_k"])),
        min_p=float(body.get("min_p", d.get("min_p", 0.0))),
        max_tokens=max_tokens,
    )
    fwd = dict(body)
    fwd.pop("max_completion_tokens", None)
    fwd.update(
        seed=seed, temperature=params.temperature, top_p=params.top_p, top_k=params.top_k,
        min_p=params.min_p, max_tokens=max_tokens, return_token_ids=True,
    )
    if not asked.logprobs:
        fwd.update(logprobs=True, top_logprobs=TOP_LOGPROBS, return_tokens_as_token_ids=True)
    return fwd, params, asked


def _parse_logprobs(choice: dict[str, Any]) -> list[TokenLogprob]:
    ids: list[int] = choice.get("token_ids") or []
    content = (choice.get("logprobs") or {}).get("content") or []
    out: list[TokenLogprob] = []
    for tid, entry in zip(ids, content, strict=False):
        top: list[tuple[int, float]] = []
        for t in entry.get("top_logprobs") or []:
            tok = str(t.get("token", ""))
            if tok.startswith("token_id:"):
                top.append((int(tok.split(":", 1)[1]), float(t["logprob"])))
        out.append(TokenLogprob(token_id=tid, logprob=float(entry["logprob"]), top=top))
    return out


def strip_response(data: dict[str, Any], asked: Asked) -> None:
    """Remove what the gateway injected and the client did not ask for."""
    if not asked.token_ids:
        data.pop("prompt_token_ids", None)
        data.pop("prompt_logprobs", None)
    for ch in data.get("choices", []):
        if not asked.token_ids:
            ch.pop("token_ids", None)
        if not asked.logprobs:
            ch["logprobs"] = None


class _Collector:
    """Reassembles a streamed completion into the same shape as a non-streamed one."""

    def __init__(self) -> None:
        self.prompt_token_ids: list[int] = []
        self.choices: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: dict[str, Any]) -> None:
        if chunk.get("prompt_token_ids"):
            self.prompt_token_ids = list(chunk["prompt_token_ids"])
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for ch in chunk.get("choices", []):
            slot = self.choices.setdefault(
                ch["index"], {"token_ids": [], "logprobs": {"content": []}, "finish_reason": None}
            )
            slot["token_ids"].extend(ch.get("token_ids") or [])
            slot["logprobs"]["content"].extend((ch.get("logprobs") or {}).get("content") or [])
            if ch.get("finish_reason"):
                slot["finish_reason"] = ch["finish_reason"]


def _record(
    *,
    model: str,
    route: ModelRoute,
    body: dict[str, Any],
    fwd: dict[str, Any],
    params: SamplingParams,
    prompt_ids: list[int],
    raw_choices: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    stream: bool,
    t0: float,
) -> Workunit:
    reason = unscorable_reason(body, params)
    return Workunit(
        workunit_id=str(uuid.uuid4()), manifest_id=route.manifest_id, received_at=t0,
        completed_at=time.time(), model=model, upstream=route.upstream, client_request=body,
        forwarded_request=fwd, params=params, prompt_token_ids=prompt_ids,
        choices=[
            Choice(
                index=int(c.get("index", i)), token_ids=list(c.get("token_ids") or []),
                logprobs=_parse_logprobs(c), finish_reason=c.get("finish_reason"),
            )
            for i, c in enumerate(raw_choices)
        ],
        stream=stream, usage=usage, scorable=reason is None, unscorable_reason=reason,
    )


def create_app(cfg: GatewayConfig, client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="recompute-poc API Tap")
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    cfg.spool_path.parent.mkdir(parents=True, exist_ok=True)

    def spool(wu: Workunit) -> None:
        with cfg.spool_path.open("a", encoding="utf-8") as f:
            f.write(wu.model_dump_json() + "\n")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "models": sorted(cfg.routes)}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(request: Request) -> Response:
        t0 = time.time()
        body: dict[str, Any] = await request.json()
        model = str(body.get("model", ""))
        route = cfg.routes.get(model)
        if route is None:
            return JSONResponse({"error": f"unknown model {model!r}"}, status_code=404)
        fwd, params, asked = build_forwarded(body, route)
        url = route.upstream.rstrip("/") + "/v1/chat/completions"
        headers = {"content-type": "application/json"}

        if not fwd.get("stream"):
            up = await http.post(url, json=fwd, headers=headers)
            if up.status_code != 200:
                return Response(up.content, up.status_code, media_type="application/json")
            data = up.json()
            spool(_record(
                model=model, route=route, body=body, fwd=fwd, params=params,
                prompt_ids=list(data.get("prompt_token_ids") or []),
                raw_choices=data.get("choices", []), usage=data.get("usage"),
                stream=False, t0=t0,
            ))
            strip_response(data, asked)
            return JSONResponse(data)

        up_req = http.build_request("POST", url, json=fwd, headers=headers)
        up = await http.send(up_req, stream=True)
        if up.status_code != 200:
            content = await up.aread()
            await up.aclose()
            return Response(content, up.status_code, media_type="application/json")

        async def gen() -> AsyncIterator[bytes]:
            col = _Collector()
            try:
                async for line in up.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        continue
                    chunk = json.loads(payload)
                    col.feed(chunk)
                    strip_response(chunk, asked)
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
            finally:
                await up.aclose()
            raw = [{"index": i, **c} for i, c in sorted(col.choices.items())]
            spool(_record(
                model=model, route=route, body=body, fwd=fwd, params=params,
                prompt_ids=col.prompt_token_ids, raw_choices=raw, usage=col.usage,
                stream=True, t0=t0,
            ))

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
