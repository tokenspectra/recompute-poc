"""The Request Workunit as captured by the API Tap."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    seed: int
    client_seed: bool = False
    temperature: float
    top_p: float
    top_k: int
    min_p: float = 0.0
    max_tokens: int


class TokenLogprob(BaseModel):
    token_id: int
    logprob: float
    top: list[tuple[int, float]] = Field(default_factory=list)


class Choice(BaseModel):
    index: int
    token_ids: list[int] = Field(default_factory=list)
    logprobs: list[TokenLogprob] = Field(default_factory=list)
    finish_reason: str | None = None


class Workunit(BaseModel):
    workunit_id: str
    manifest_id: str
    received_at: float
    completed_at: float
    model: str
    upstream: str
    client_request: dict[str, Any]
    forwarded_request: dict[str, Any]
    params: SamplingParams
    prompt_token_ids: list[int]
    choices: list[Choice]
    stream: bool
    usage: dict[str, Any] | None = None
    scorable: bool = True
    unscorable_reason: str | None = None
