"""A tiny fake vLLM: same wire shape as v0.32.0 chat/completions, real Gumbel sampling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from recompute_poc.gumbel import gumbel_noise

VOCAB = 1000


def logits_for(ctx: list[int], salt: str, jitter: float, rng: np.random.Generator) -> np.ndarray:
    h = hashlib.sha256((salt + json.dumps(ctx[-3:])).encode()).digest()
    base = np.random.default_rng(int.from_bytes(h[:8], "little")).normal(0, 2.0, VOCAB)
    return base + (rng.normal(0, jitter, VOCAB) if jitter else 0.0)


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def create_fake(salt: str = "model-A", jitter: float = 0.0, jitter_seed: int = 0) -> FastAPI:
    app = FastAPI()
    rng = np.random.default_rng(jitter_seed)

    def top20(lp: np.ndarray) -> list[int]:
        return [int(i) for i in np.argsort(-lp)[:20]]

    def generate(body: dict[str, Any]) -> tuple[list[int], list[int], list[dict[str, Any]]]:
        prompt = [ord(c) % VOCAB for m in body["messages"] for c in m["content"]] or [1]
        ctx = list(prompt)
        out: list[int] = []
        content: list[dict[str, Any]] = []
        T, seed = float(body["temperature"]), int(body["seed"])
        for _ in range(int(body["max_tokens"])):
            lp = log_softmax(logits_for(ctx, salt, jitter, rng))
            ids = np.array(top20(lp), dtype=np.int64)
            z = lp[ids] / max(T, 0.01) + gumbel_noise(seed, len(ctx) - 1, ids)
            tok = int(ids[int(np.argmax(z))])
            content.append({
                "token": f"token_id:{tok}", "logprob": float(lp[tok]),
                "top_logprobs": [{"token": f"token_id:{int(i)}", "logprob": float(lp[i])}
                                 for i in ids],
            })
            out.append(tok)
            ctx.append(tok)
        return prompt, out, content

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        assert body.get("return_token_ids") is True and body.get("seed") is not None
        prompt, out, content = generate(body)
        if not body.get("stream"):
            return JSONResponse({
                "id": "c1", "model": body["model"], "prompt_token_ids": prompt,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"},
                             "token_ids": out, "logprobs": {"content": content},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(out)},
            })

        async def sse() -> AsyncIterator[str]:
            for i, (t, c) in enumerate(zip(out, content, strict=True)):
                chunk: dict[str, Any] = {
                    "id": "c1", "choices": [{"index": 0, "delta": {"content": "x"},
                                             "token_ids": [t], "logprobs": {"content": [c]},
                                             "finish_reason": None}]}
                if i == 0:
                    chunk["prompt_token_ids"] = prompt
                yield f"data: {json.dumps(chunk)}\n\n"
            last = {"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(out)}}
            yield f"data: {json.dumps(last)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        body = await request.json()
        ids: list[int] = body["prompt"]
        plp: list[dict[str, Any] | None] = [None]
        for i in range(1, len(ids)):
            lp = log_softmax(logits_for(ids[:i], salt, jitter, rng))
            keys = set(top20(lp)) | {ids[i]}
            plp.append({str(k): {"logprob": float(lp[k]), "rank": 1} for k in keys})
        return {"choices": [{"index": 0, "text": "", "prompt_logprobs": plp}]}

    return app
