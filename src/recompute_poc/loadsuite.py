"""Load suite: fixed prompts, concurrency 1/8/32, stream and non-stream, T=0 and T>0."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

TOPICS = [
    "the water cycle", "how a transistor works", "the causes of the French Revolution",
    "binary search", "photosynthesis", "the rules of chess", "how vaccines train immunity",
    "TCP congestion control", "the Pythagorean theorem", "plate tectonics",
    "supply and demand", "how a bill becomes law", "the history of the printing press",
    "gradient descent", "the Krebs cycle", "how GPS works", "the Roman aqueducts",
    "hash tables", "the greenhouse effect", "how noise-cancelling headphones work",
    "the structure of DNA", "public key cryptography", "the Silk Road", "black holes",
    "how airplanes fly", "the Turing test", "inflation", "the immune system",
    "how a compiler works", "tidal forces", "the Industrial Revolution", "sorting algorithms",
]
STYLES = [
    "Explain {t} in about 120 words for a curious teenager.",
    "Write a short dialogue between two engineers arguing about {t}.",
    "Give three common misconceptions about {t} and correct them.",
    "Summarise {t} as a bulleted list of five facts.",
]


def prompts(n: int) -> list[str]:
    """Deterministic prompt set: same list on every run."""
    out = [STYLES[i % len(STYLES)].format(t=TOPICS[i % len(TOPICS)]) for i in range(len(TOPICS))]
    return [out[i % len(out)] for i in range(n)]


@dataclass(frozen=True)
class Cell:
    concurrency: int
    stream: bool
    temperature: float


def matrix(concurrency: tuple[int, ...] = (1, 8, 32)) -> list[Cell]:
    return [Cell(c, s, t) for c in concurrency for s in (False, True) for t in (0.0, 0.6)]


async def run_cell(
    http: httpx.AsyncClient, url: str, model: str, cell: Cell, n: int, max_tokens: int,
) -> dict[str, object]:
    sem = asyncio.Semaphore(cell.concurrency)
    ok = 0
    errors: list[str] = []

    async def one(i: int, text: str) -> None:
        nonlocal ok
        body = {"model": model, "stream": cell.stream, "temperature": cell.temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": text}]}
        async with sem:
            try:
                r = await http.post(url.rstrip("/") + "/v1/chat/completions", json=body)
                if r.status_code != 200:
                    errors.append(f"{r.status_code}: {r.text[:120]}")
                else:
                    ok += 1
            except httpx.HTTPError as e:
                errors.append(repr(e))

    t0 = time.time()
    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts(n))))
    return {"concurrency": cell.concurrency, "stream": cell.stream,
            "temperature": cell.temperature, "requests": n, "ok": ok,
            "seconds": round(time.time() - t0, 2), "errors": errors[:3]}


async def run_suite(
    url: str, model: str, per_cell: int, max_tokens: int, label: str, out: Path,
    cells: list[Cell] | None = None, http: httpx.AsyncClient | None = None,
) -> list[dict[str, object]]:
    client = http or httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    results: list[dict[str, object]] = []
    for cell in cells or matrix():
        r = await run_cell(client, url, model, cell, per_cell, max_tokens)
        r["label"] = label
        results.append(r)
        print(json.dumps(r), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(r) + "\n")
    return results
