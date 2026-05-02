# Architecture And Workflow

This document explains how the repository is organized and what information
flows through each phase of the experiment. The README is kept as the setup and
navigation entry point; this file contains the detailed architecture notes.

## Core Idea

The experiment separates the codebase into two parts:

- an immutable kernel that runs and evaluates the experiment;
- a mutable agent body that can be copied into candidate generations and
  modified through structured proposals.

The kernel protects the validity of the experiment. The mutable body is where
self-specialization happens.

## Immutable Kernel

These files and modules are not mutable by the agent:

- `orchestrator.py`
- `growth/apply.py`
- `growth/manifest.py`
- `growth/reflect.py`
- `eval/`
- `sandbox/`
- `tests/`
- `config.yaml`
- `config.dev.yaml`
- `mutation_manifest.yaml`
- `data/`
- `artifacts/`

The kernel is responsible for:

- loading train, validation, and test splits;
- stripping labels from candidate task inputs;
- running candidate generations in Docker;
- proxying and logging LLM calls;
- scoring predictions;
- selecting reflection examples from train;
- applying proposals into candidate snapshots;
- smoke-testing candidates;
- accepting or rejecting candidates;
- preserving run artifacts.

## Mutable Agent Body

The agent can propose changes under these roots:

- `agent/`: runtime solve loop, prompt, and local agent rules;
- `tools/`: sandbox tools, except protected core symbols;
- `knowledge/`: learned task-strategy memory;
- `self_model/`: architecture, capability, failure-mode, and experiment memory.

These roots are copied into every generation snapshot. Candidate containers see
the snapshot mounted at `/agent`.

## Mutation Manifest

`mutation_manifest.yaml` defines the boundary. It specifies:

- `snapshot_roots`: roots copied into generation snapshots;
- `mutable_paths`: paths that structured proposals may modify;
- `protected_files`: files that may be replaced but not deleted;
- `protected_paths`: host/kernel paths that may not be edited;
- `protected_symbols`: functions in `tools/base.py` that proposals may not
  replace or delete;
- `reflection_context`: bounded read-only repository context available to the
  reflection phase.

This lets the project widen the agent's self-modification surface without
allowing edits to the scorer, gate, dataset, sandbox runner, or run evidence.

## Runtime Task Phase

For each train, validation, or test task, the sandbox runner sends one unlabeled
task to `agent.solve_task(task)`.

The runtime agent constructs the LLM request as follows:

- system message: `agent/prompt.txt`, plus `knowledge/strategy.md` when present;
- user message: the code snippet and the instruction to classify it;
- tools: the runtime `_TOOL_SPECS` from `agent/agent.py`;
- tool choice: `auto`, except the final allowed step forces `finalize`;
- hidden labels: never sent to the candidate.

The seed runtime tools exposed to the model are:

- `read_file`: read a file under `/agent` or `/work`;
- `list_dir`: list directories under `/agent` or `/work`;
- `static_scan`: run a bounded heuristic scan over the current snippet;
- `note`: append scratch notes to `/work/notes.txt`;
- `finalize`: submit the final label and reasoning.

`tools/base.py` contains additional helper functions, such as `llm_call`,
`write_file`, and `run_bash`, but they are not all exposed to the seed model as
runtime tools. The exposed tool list is intentionally narrower than the helper
module because some helpers are infrastructure plumbing or are too broad for
the seed runtime policy.

## Reflection Phase

Reflection runs through the host-controlled LLM gateway. It receives
train-derived feedback and bounded codebase context, then must submit one
structured proposal.

Reflection input includes:

- train metrics and confusion matrix;
- selected balanced train failures and successes;
- task snippets and raw outputs for selected train cases;
- train-only telemetry about tool use and finalization behavior;
- current mutable file contents for key files;
- public tool API summaries;
- recent proposal history;
- the structured proposal schema.

Reflection does not receive validation or test task contents.

Reflection tools are read-only except for the final proposal call:

- `list_mutable_files`
- `read_mutable_file`
- `read_train_case`
- `read_repo_context`
- `list_repo_files`
- `read_repo_file`
- `read_tool_api`
- `propose_changes`

Before proposing, reflection must inspect at least one train, mutable, or repo
item and at least one `self_model/**` file or filtered repository contract. This
requirement makes the reflection phase inspect the agent's architecture instead
of only rewriting the prompt from aggregate metrics.

## Proposal Schema

A proposal has this top-level shape:

```json
{
  "rationale": "short explanation tied to train task evidence",
  "intent": "iterate",
  "changes": [
    {"kind": "replace_file", "details": {"path": "knowledge/strategy.md", "content": "..."}}
  ]
}
```

Allowed change kinds include:

- `edit_prompt`
- `edit_solve_loop`
- `add_tool`
- `edit_tool`
- `delete_tool`
- `create_file`
- `replace_file`
- `delete_file`
- `add_function`
- `replace_function`

`growth/apply.py` validates the proposal against the manifest. Python edits are
parsed with `ast`, imports are restricted, protected paths are rejected, and
protected symbols cannot be edited or deleted.

## Evolution Loop

The normal sequence is:

1. Load config and mutation manifest.
2. Load cached train, validation, and test splits.
3. Apply runtime split caps from the active config.
4. Snapshot mutable roots into `generations/gen_0/`.
5. Evaluate `gen_0` on train.
6. Evaluate `gen_0` on validation.
7. Select train-only examples for reflection.
8. Run reflection and save `proposals/gen_N.proposal.json`.
9. Copy the current accepted parent into candidate `gen_N/`.
10. Apply the proposal inside the candidate snapshot.
11. Smoke-test the candidate on one train task.
12. Evaluate the candidate on validation if smoke passes.
13. Accept only if validation macro-F1 strictly improves and errors are zero.
14. If accepted, evaluate the new parent on train before the next reflection.
15. Stop on generation cap, plateau, or agent halt.

Rejected candidates remain in the run directory for audit, but they do not
replace the accepted parent.

## Evaluation Protocol

Train is used for reflection feedback. Validation is used only for accept/reject
gating. Test is reserved for final frozen benchmarks.

The final reported run selected `gen_2` from validation, then benchmarked both
the seed and `gen_2` on held-out test. The test result was not fed back into the
evolution loop.

Macro-F1 is used as the main gate because the balanced score is more informative
than accuracy for a binary vulnerability dataset where safe and vulnerable class
behavior can diverge.

## Run Artifacts

Each run writes:

- `log.jsonl`: structured orchestration events;
- `llm_calls.jsonl`: every proxied LLM request and response summary;
- `terminal_output.out`: human-readable run transcript;
- `config.snapshot.yaml`: exact run config;
- `mutation_manifest.snapshot.yaml`: exact mutation boundary;
- `proposals/gen_N.proposal.json`: proposed self-modifications;
- `generations/gen_N/`: generation snapshots.

Raw run directories are useful for local inspection, but they should not be
committed unless explicitly requested.
