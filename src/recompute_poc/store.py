"""SQLite store for the recompute daemon (VM-B): Workunits, scores, thresholds."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS workunits (
  workunit_id TEXT PRIMARY KEY, manifest_id TEXT NOT NULL, model TEXT NOT NULL,
  temperature REAL NOT NULL, ingested_at REAL NOT NULL, status TEXT NOT NULL,
  reason TEXT, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS wu_status ON workunits(status);
CREATE TABLE IF NOT EXISTS scores (
  workunit_id TEXT PRIMARY KEY, scored_at REAL NOT NULL, model TEXT NOT NULL,
  temperature REAL NOT NULL, n_tokens INTEGER NOT NULL, match_rate REAL NOT NULL,
  mean_margin REAL NOT NULL, filtered_rate REAL NOT NULL, delta_max REAL NOT NULL);
CREATE INDEX IF NOT EXISTS sc_time ON scores(scored_at);
CREATE TABLE IF NOT EXISTS thresholds (
  model TEXT PRIMARY KEY, tau REAL NOT NULL, delta_max REAL NOT NULL,
  window_tokens INTEGER NOT NULL, n_windows INTEGER NOT NULL, quantile REAL NOT NULL,
  created_at REAL NOT NULL);
"""


@dataclass(frozen=True)
class WindowStats:
    model: str
    workunits: int
    tokens: int
    match_rate: float
    mean_margin: float
    p50: float
    p95: float
    p99: float
    filtered_rate: float


class Store:
    def __init__(self, path: Path | str) -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def insert_workunit(
        self, workunit_id: str, manifest_id: str, model: str, temperature: float,
        status: str, reason: str | None, record: str,
    ) -> bool:
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO workunits VALUES (?,?,?,?,?,?,?,?)",
                (workunit_id, manifest_id, model, temperature, time.time(), status, reason, record),
            )
            return cur.rowcount == 1

    def pending(self, limit: int = 16) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT workunit_id, record FROM workunits WHERE status='pending' "
                "ORDER BY ingested_at LIMIT ?", (limit,)).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def mark(self, workunit_id: str, status: str, reason: str | None = None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE workunits SET status=?, reason=? WHERE workunit_id=?",
                (status, reason, workunit_id))

    def save_score(
        self, workunit_id: str, model: str, temperature: float, n_tokens: int,
        match_rate: float, mean_margin: float, filtered_rate: float, delta_max: float,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?)",
                (workunit_id, time.time(), model, temperature, n_tokens, match_rate,
                 mean_margin, filtered_rate, delta_max))
            self._db.execute(
                "UPDATE workunits SET status='scored', reason=NULL WHERE workunit_id=?",
                (workunit_id,))

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT status, COUNT(*) FROM workunits GROUP BY status").fetchall()
        return {str(s): int(n) for s, n in rows}

    def scores(
        self, model: str | None = None, since: float | None = None, order_by_time: bool = True,
    ) -> list[tuple[float, str, int, float, float, float]]:
        q = ("SELECT scored_at, model, n_tokens, match_rate, mean_margin, filtered_rate "
             "FROM scores WHERE 1=1")
        args: list[object] = []
        if model:
            q += " AND model=?"
            args.append(model)
        if since is not None:
            q += " AND scored_at>=?"
            args.append(since)
        if order_by_time:
            q += " ORDER BY scored_at"
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [(float(a), str(b), int(c), float(d), float(e), float(f)) for a, b, c, d, e, f in rows]

    def window_stats(self, window_s: float, model: str | None = None) -> list[WindowStats]:
        rows = self.scores(model=model, since=time.time() - window_s)
        by_model: dict[str, list[tuple[float, str, int, float, float, float]]] = {}
        for r in rows:
            by_model.setdefault(r[1], []).append(r)
        out: list[WindowStats] = []
        for m, rs in sorted(by_model.items()):
            toks = np.array([r[2] for r in rs], dtype=np.float64)
            wm = np.array([r[4] for r in rs])
            tot = float(toks.sum())
            out.append(WindowStats(
                model=m, workunits=len(rs), tokens=int(tot),
                match_rate=float((toks * np.array([r[3] for r in rs])).sum() / tot),
                mean_margin=float((toks * wm).sum() / tot),
                p50=float(np.percentile(wm, 50)), p95=float(np.percentile(wm, 95)),
                p99=float(np.percentile(wm, 99)),
                filtered_rate=float((toks * np.array([r[5] for r in rs])).sum() / tot),
            ))
        return out

    def set_threshold(
        self, model: str, tau: float, delta_max: float, window_tokens: int, n_windows: int,
        quantile: float,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO thresholds VALUES (?,?,?,?,?,?,?)",
                (model, tau, delta_max, window_tokens, n_windows, quantile, time.time()))

    def threshold(self, model: str) -> tuple[float, int, int] | None:
        with self._lock:
            r = self._db.execute(
                "SELECT tau, n_windows, window_tokens FROM thresholds WHERE model=?",
                (model,)).fetchone()
        return None if r is None else (float(r[0]), int(r[1]), int(r[2]))

    def models(self) -> list[str]:
        with self._lock:
            return [str(r[0]) for r in self._db.execute(
                "SELECT DISTINCT model FROM scores ORDER BY model")]


def chunk_windows(
    scores: Sequence[tuple[float, str, int, float, float, float]], window_tokens: int
) -> list[float]:
    """Token-weighted mean margin over consecutive chunks of >= window_tokens tokens."""
    out: list[float] = []
    acc_t, acc_m = 0, 0.0
    for _, _, n, _, margin, _ in scores:
        acc_t += n
        acc_m += n * margin
        if acc_t >= window_tokens:
            out.append(acc_m / acc_t)
            acc_t, acc_m = 0, 0.0
    return out
