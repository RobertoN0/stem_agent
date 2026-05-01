# Stem Agent

A self-modifying LLM agent for security-focused code analysis. The agent
starts as a minimal "stem" with generic tools and evolves itself, generation
by generation, into a specialized agent for vulnerability detection on
PrimeVul.

## Architecture Rules

- `orchestrator.py` is fixed rollback infrastructure; the evolving agent must
  never edit it.
- The immutable `mutation_manifest.yaml` defines the snapshot and mutation
  boundary. Current mutable roots are `agent/`, `tools/`, `knowledge/`, and
  `self_model/`.
- Kernel paths (`orchestrator.py`, `growth/`, `eval/`, `sandbox/`, tests,
  configs, data, and artifacts) are not candidate-editable.
- `tools/base.py` has protected RPC/filesystem symbols; future generations may
  add helper tools or edit non-protected functions through manifest-checked
  proposals.
- Each experiment keeps candidate snapshots under
  `artifacts/runs/<run_id>/generations/gen_N/`; snapshots are append-only
  runtime artifacts and are not committed.
- Candidates run in Docker subprocesses, never imported into the host
  orchestrator process.
- Train, validation, and test splits are kept separate: validation gates
  evolution, and test is reserved for final reporting.
- Reflection runs as a read-only self-inspection loop: it can inspect selected
  train cases, mutable snapshot files, and filtered repo contracts through
  tools. It can also list/read bounded repo files outside denied paths for
  architecture self-inspection, then submits a manifest-checked proposal for
  the next candidate.

## Prerequisites

- **Python 3.11+**
- **Docker** — every candidate generation runs in a Docker subprocess with
  resource limits and network disabled. The orchestrator will not work
  without a working Docker daemon. Install Docker Engine (Linux) or Docker
  Desktop (macOS/Windows) and confirm with:

  ```sh
  docker run --rm hello-world
  ```

- An **OpenAI API key**.

## Setup

1. Clone the repo and `cd` into it.
2. Create and activate a virtualenv:

   ```sh
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install Python dependencies. There are two requirements files:
   - `requirements.txt` — the minimal set baked into the sandbox image
   - `requirements-host.txt` — adds host-only deps (`datasets`) for the
     orchestrator and the dataset builder

   On the host, install the full set:

   ```sh
   pip install -r requirements-host.txt
   ```

4. Create a `.env` file in the repo root with your API key:

   ```
   OPENAI_API_KEY=sk-...
   ```

   `.env` is gitignored — never commit it.

5. Build the sandbox image (build context must be the repo root, since the
   Dockerfile copies in `requirements.txt` and `sandbox/runner.py`):

   ```sh
   docker build -f sandbox/Dockerfile -t stem-agent-sandbox .
   ```

## Running

The evolution loop is driven by the orchestrator:

```sh
python orchestrator.py
```

All tunables (models, generation caps, split sizes, stopping criteria, sandbox
limits, token pricing) live in [config.yaml](config.yaml). Each run gets a
timestamped directory under `artifacts/runs/`, containing:

- `log.jsonl` — every orchestrator event (start, gates, accept/reject, errors)
- `llm_calls.jsonl` — every proxied model call with usage and timing
- `terminal_output.out` — the same human-readable progress stream printed to
  the terminal, flushed automatically during the run
- `config.snapshot.yaml` — copy of the active config at run start, so the run
  is reproducible after config drifts
- `mutation_manifest.snapshot.yaml` — copy of the immutable mutation boundary
  used for that run
- `proposals/gen_N.proposal.json` — full structured reflection proposal for
  each attempted generation, kept outside candidate snapshots
- `generations/gen_N/` — the full agent/tool snapshot for each generation in
  that run

## Dataset

The dataset builder (`python -m eval.dataset`) tries Hugging Face sources
in this order and uses the first that loads:

1. `colin/PrimeVul`
2. `bstee615/bigvul`
3. `DetectVul/devign`

When a split cache is built, sizes come from the active config's `splits`
section and are balanced 50/50 between vulnerable and safe with
`random.seed(42)`. The chosen source and resulting cached sizes are recorded
in `data/<data_dir>/_source.json`. At runtime, the orchestrator and benchmark
also apply the active config's split sizes as caps over the cached JSONL files,
so lowering `train` or `val` in a config makes quick experiments shorter
without rebuilding the cache. Add `--force` to rebuild from scratch when you
want newly sampled cached splits.

## Sandbox network policy

The candidate's container runs with `--network none`. The agent has no
direct internet access; LLM calls go through `tools.base.llm_call`, which
proxies a structured request back to the orchestrator via the runner's
stdin/stdout RPC. The orchestrator makes the OpenAI call and returns the
response. See `agent/AGENT.md` for the rule that bans `import openai` in
agent code.

## Cost tracking

Cost numbers in the log come from the orchestrator's own tally of every
LLM call's `usage.prompt_tokens` and `usage.completion_tokens` (read off
the OpenAI API response), multiplied by the per-million-token prices in
`config.yaml`'s `pricing` block. Because the orchestrator is the only
process that talks to OpenAI, this is ground truth from our side — the
agent cannot under-report. Cost is logged for auditability, not used as a
stop gate. The OpenAI dashboard is still authoritative for the actual bill
(rate-limit retries, tokenizer drift, etc.), but the in-log number is much
closer than self-report would be.

## Tests

```sh
pytest tests/
```

Tests mock the Docker subprocess, so they run without a Docker daemon.

## Layout

Core source lives in `orchestrator.py`, `agent/`, `tools/`, `knowledge/`,
`self_model/`, `growth/`, `eval/`, and `sandbox/`. Local agent-editing rules
live in `agent/AGENT.md`; the authoritative mutation boundary lives in
`mutation_manifest.yaml`.
