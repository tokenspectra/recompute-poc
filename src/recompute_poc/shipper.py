"""Ships the VM-A JSONL spool to VM-B /ingest. The file is the source of truth."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx


def ship_once(spool: Path, offset_file: Path, ingest_url: str, batch: int = 50,
              client: httpx.Client | None = None) -> int:
    """Send lines past the saved offset; advance only after VM-B acknowledges. Returns sent."""
    if not spool.exists():
        return 0
    start = int(offset_file.read_text()) if offset_file.exists() else 0
    http = client or httpx.Client(timeout=60.0)
    sent = 0
    with spool.open("rb") as f:
        f.seek(start)
        pos = start
        buf: list[bytes] = []
        ends: list[int] = []
        while True:
            line = f.readline()
            if not line or not line.endswith(b"\n"):  # partial last line: wait for the writer
                break
            buf.append(line)
            pos += len(line)
            ends.append(pos)
            if len(buf) >= batch:
                break
        if buf:
            r = http.post(ingest_url, json=[json.loads(b) for b in buf])
            r.raise_for_status()
            offset_file.write_text(str(ends[-1]))
            sent = len(buf)
    return sent


def run_forever(spool: Path, offset_file: Path, ingest_url: str, interval: float = 1.0) -> None:
    while True:
        try:
            while ship_once(spool, offset_file, ingest_url):
                pass
        except httpx.HTTPError as e:
            print(f"ship failed, will retry: {e!r}", flush=True)
        time.sleep(interval)
