# recompute-poc

> [!WARNING]
> Research prototype, not maintained software. It was built for one experiment and may change or stop working without notice. Not for production use.

A proof of concept of statistical recompute for [TokenSpectra](https://tokenspectra.com): checking that an inference server really ran the model it declares, by re-scoring its outputs on a second GPU.

- **API Tap gateway** (VM-A): sits in front of vLLM, injects a seed and logprob/token-id requests, strips them from the reply, and spools each request as a Workunit.
- **Recompute daemon** (VM-B): re-runs the declared model with the claimed output forced in as input (teacher-forced prefill), replays vLLM's Gumbel sampling noise on the CPU, and scores every token two ways: verified sampling (exact match) and the [Token-DiFR](https://arxiv.org/abs/2511.20621) margin (Karvonen et al., 2025).
- **`rpoc` CLI**: runs the gateway and daemon, ships the spool, calibrates thresholds, evaluates honest vs swapped-model traffic, and drives a load suite.

## Results

First run, 2026-10-25, two Nebius VMs with one H100 each, vLLM v0.32.0, Qwen3-4B: 768 honest requests (about 197,000 tokens) matched at 99.4%, and a swap to Qwen3-1.7B was detected within about 64 tokens at a false-positive rate of about 0.1%. Details and caveats: [docs/results-20261025.md](docs/results-20261025.md).

## Limitations

- One model, one vLLM version (v0.32.0, Model Runner V2), one GPU SKU, one run; no confidence intervals.
- The negative control (a smaller sibling model) is the easy case; same-size, FP8 or AWQ swaps are untested.
- The two VMs may share a host, so "two GPUs" means two separate VMs with one H100 each.
- The operator is assumed cooperative: gateway records are taken as claims, not checked independently.
- Chat completions only, vLLM only. Thresholds are provisional.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run rpoc --help
```

The GPU side runs on two Nebius `gpu-h100-sxm` VMs with vLLM pinned by image digest; see `scripts/nebius/` (`vm.sh`, `provision.sh`, `serve.sh`, defaults in `config.env`). `scripts/validate_gumbel.py` checks the NumPy noise port against vLLM's kernel on a GPU.

## License

Apache License 2.0, see [LICENSE](LICENSE). `src/recompute_poc/gumbel.py` is ported from vLLM (Apache-2.0); see [NOTICE](NOTICE).
