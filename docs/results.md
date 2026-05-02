# Publishable Results

This file summarizes the evidence that should be referenced in the final
write-up. Raw `artifacts/runs/*` directories are local evidence and should not
be committed unless explicitly requested.

## Primary Final Evidence

Use these as the main reported runs.

| Purpose | Run ID | Notes |
| --- | --- | --- |
| final evolution | `run_20260501_175346` | used `config.dev.yaml` with runtime caps: train 40, val 20, test 60 |
| final seed test | `final_test_seed_20260501` | held-out test benchmark for `run_20260501_175346/generations/gen_0` |
| final evolved test | `final_test_gen2_20260501` | held-out test benchmark for `run_20260501_175346/generations/gen_2` |

## Final Evolution Run

Run:

```text
artifacts/runs/run_20260501_175346
```

Config snapshot:

```text
train: 40
val:   20
test:  60
plateau_generations: 3
```

Generation trajectory:

| Generation | Main proposal | Train Macro-F1 | Val Macro-F1 | Gate |
| ---: | --- | ---: | ---: | --- |
| 0 | seed | 0.873 | 0.670 | baseline |
| 1 | prompt decision policy | 0.774 | 0.733 | accepted |
| 2 | refined prompt policy | 0.799 | 0.792 | accepted, best |
| 3 | solve-loop forced scan gate | n/a | 0.670 | rejected |
| 4 | more prompt specialization | n/a | 0.601 | rejected |
| 5 | solve-loop pre-scan edit | n/a | n/a | smoke rejected |

Important details:

- best accepted generation: `gen_2`
- stop reason: plateau
- total run cost: `$1.7466`
- total LLM calls: `282`
- execution errors in accepted generations: `0`
- proposal artifacts saved under `proposals/gen_N.proposal.json`

Reflection behavior:

| Reflection Tool | Calls |
| --- | ---: |
| `list_mutable_files` | 5 |
| `read_mutable_file` | 19 |
| `read_train_case` | 14 |
| `read_tool_api` | 4 |
| `read_repo_context` | 2 |
| `propose_changes` | 5 |

Every reflection round inspected at least one `self_model/**` file before
proposing, useful evidence for the self-awareness architecture.

## Final Held-Out Test

These benchmarks were run after selecting `gen_2` from validation. The held-out
test results were not fed back into reflection, proposal generation, or
validation gating.

| Agent | Test Macro-F1 | Accuracy | Errors | LLM Calls |
| --- | ---: | ---: | ---: | ---: |
| seed `gen_0` | 0.753 | 0.767 | 0 | 82 |
| evolved `gen_2` | 0.733 | 0.750 | 0 | 62 |

Confusion summary:

| Agent | Safe Correct | Vulnerable Correct | False Positives | False Negatives |
| --- | ---: | ---: | ---: | ---: |
| seed `gen_0` | 30/30 | 16/30 | 0 | 14 |
| evolved `gen_2` | 30/30 | 15/30 | 0 | 15 |

Task-level delta:

- both correct: 44 tasks
- seed correct / evolved wrong: 2 tasks
- evolved correct / seed wrong: 1 task
- both wrong: 13 tasks

Interpretation:

- Validation improved from `0.670` to `0.792`, showing successful
  train/validation self-specialization.
- Held-out test decreased from `0.753` to `0.733`, showing that the selected
  specialization did not generalize.


## Reproduction Commands

Evolution:

```sh
python orchestrator.py --config config.dev.yaml
```

Final seed benchmark:

```sh
.venv/bin/python -m eval.benchmark \
  --config config.dev.yaml \
  --gen artifacts/runs/run_20260501_175346/generations/gen_0 \
  --split test \
  --run-id final_test_seed_20260501
```

Final evolved benchmark:

```sh
.venv/bin/python -m eval.benchmark \
  --config config.dev.yaml \
  --gen artifacts/runs/run_20260501_175346/generations/gen_2 \
  --split test \
  --run-id final_test_gen2_20260501
```
