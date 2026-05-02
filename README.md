The final report for Task 1 of the can be found here: [docs/final_report.md](docs/final_report.md)

# Stem Agent

This repository contains a runnable self-modifying agent experiment for binary
vulnerability classification. The agent starts from a small seed, runs on
PrimeVul-derived code snippets, reflects on train-only feedback, proposes edits
to its own mutable files, and accepts a candidate only when validation macro-F1
strictly improves with zero execution errors.

Task labels:

- `vulnerable`
- `safe`

The final result is intentionally conservative: the evolved agent improved
validation performance, but the held-out test showed that this improvement did
not generalize beyond the seed.

## Final Status

Primary final run:

- evolution run: `artifacts/runs/run_20260501_175346`
- best validation generation: `artifacts/runs/run_20260501_175346/generations/gen_2`
- seed test benchmark: `artifacts/runs/final_test_seed_20260501`
- evolved test benchmark: `artifacts/runs/final_test_gen2_20260501`

Validation trajectory:

| Generation | Main Change | Validation Macro-F1 | Gate |
| ---: | --- | ---: | --- |
| 0 | seed | 0.670 | baseline |
| 1 | prompt decision policy | 0.733 | accepted |
| 2 | refined prompt policy | 0.792 | accepted, best |
| 3 | solve-loop forced scan gate | 0.670 | rejected |
| 4 | more prompt specialization | 0.601 | rejected |
| 5 | solve-loop pre-scan edit | smoke failed | rejected |

Held-out test benchmark:

| Agent | Test Macro-F1 | Accuracy | Errors | LLM Calls |
| --- | ---: | ---: | ---: | ---: |
| seed `gen_0` | 0.753 | 0.767 | 0 | 82 |
| evolved `gen_2` | 0.733 | 0.750 | 0 | 62 |

## Repository Guide

Core runtime and evolution code:

- [orchestrator.py](orchestrator.py): host-side evolution loop, gating, logging,
  and rollback
- [agent/](agent/): mutable runtime agent, prompt, and local editing rules
- [tools/](tools/): mutable sandbox tools with protected core functions
- [knowledge/](knowledge/): mutable learned task-strategy memory
- [self_model/](self_model/): mutable architecture, capability, and failure-mode
  memory
- [growth/](growth/): immutable reflection, manifest, and proposal-application
  kernel
- [eval/](eval/): dataset and benchmark utilities
- [sandbox/](sandbox/): Docker runner for candidate generations
- [mutation_manifest.yaml](mutation_manifest.yaml): authoritative
  mutable/immutable boundary

Documentation:

- [docs/architecture_workflow.md](docs/architecture_workflow.md): architecture,
  tool interfaces, and experiment flow
- [docs/results.md](docs/results.md): publishable result summary and reproduction
  commands

## Setup

Requirements:

- Python 3.11+
- Docker
- OpenAI API key

Install host dependencies:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-host.txt
```

Create `.env`:

```sh
OPENAI_API_KEY=sk-...
```

Build the candidate sandbox image:

```sh
docker build -f sandbox/Dockerfile -t stem-agent-sandbox .
```

## Running

Run the final development evolution config:

```sh
python orchestrator.py --config config.dev.yaml
```

Each evolution run writes a new directory under `artifacts/runs/` containing:

- `log.jsonl`
- `llm_calls.jsonl`
- `terminal_output.out`
- `config.snapshot.yaml`
- `mutation_manifest.snapshot.yaml`
- `proposals/gen_N.proposal.json`
- `generations/gen_N/`

Run a frozen generation on held-out test:

```sh
.venv/bin/python -m eval.benchmark \
  --config config.dev.yaml \
  --gen artifacts/runs/run_20260501_175346/generations/gen_2 \
  --split test \
  --run-id final_test_gen2_20260501
```

The held-out test should be used only after a generation has been selected from
validation.

## Tests

Run all tests:

```sh
pytest tests/
```

The tests mock Docker subprocess behavior, so most host-side tests do not need a
running Docker daemon.

## Dataset Reference

The vulnerability-classification task is based on PrimeVul:

Yangruibo Ding, Yanjun Fu, Omniyyah Ibrahim, Chawin Sitawarin, Xinyun Chen,
Basel Alomair, David Wagner, Baishakhi Ray, and Yizheng Chen. "Vulnerability
Detection with Code Language Models: How Far Are We?" arXiv:2403.18624, 2024.
https://arxiv.org/abs/2403.18624
