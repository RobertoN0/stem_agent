# Project Report

## 1. Scope And Objective

The task was to build a "stem agent": an AI agent that does not start as a
fully hand-designed specialist, but instead starts from a minimal generic seed,
receives signals from its environment, and grows into a more specialized agent
through its own process. I chose security as the specialization domain,
specifically using tasks of binary vulnerability classification on
PrimeVul-derived code snippets. PrimeVul is a vulnerability-detection dataset
for code language models introduced by Ding et al. [1]. For each snippet, the
agent must output exactly one of two labels:

- `vulnerable`
- `safe`

The objective was not only to obtain a final classifier score. The main goal
was to build and evaluate a credible self-specialization loop: start from a
basic seed agent, measure its behavior, use those measurements to generate
self-modification proposals, gate those proposals through validation, and keep
an auditable record of what changed and why.

My first step was to build a minimal working baseline, then use that baseline
to expose the architectural problems. After some iterations I opted for the current
architecture. The main difficulty was deciding how the agent could modify itself
without being able to corrupt the experiment, crash and not recover, leak
evaluation data, or change the scoring rules.

The final system is therefore organized around two questions:

1. Which parts of the codebase are allowed to evolve?
2. Which parts must remain fixed so that evaluation remains trustworthy?

## 2. Architecture

### 2.1 Architectural Choices

The central design choice was to separate the project into a trusted kernel and
a mutable agent body. The kernel owns the parts that define whether the
experiment is valid: dataset loading, sandbox execution, scoring, validation
gating, proposal application, rollback, and logging. The mutable body contains
the parts the agent can specialize: its runtime loop, prompt, tools, knowledge
files, and self-model files.

This boundary was a tradeoff. A stem agent needs enough freedom to change its
own behavior, but it should not be able to change the rules used to evaluate
that behavior. The architecture therefore gives the agent bounded
self-modification, read-only self-inspection, validation-gated growth, and
rollback for rejected candidates.

### 2.1 Implementation Details

The immutable kernel contains the infrastructure required to run and judge the
experiment:

- `orchestrator.py`: runs the evolution loop;
- `growth/apply.py`: applies structured proposals to candidate snapshots;
- `growth/manifest.py`: loads the mutation boundary;
- `growth/reflect.py`: controls the reflection interface;
- `eval/`: builds/loads datasets and benchmarks candidates;
- `sandbox/`: runs candidates in Docker;
- configs, tests, data, and raw run artifacts.

The mutable body contains the files the agent can evolve:

- `agent/`: runtime solve loop, system prompt, and local agent rules;
- `tools/`: sandbox tools, with protected core functions;
- `knowledge/`: learned task-policy memory;
- `self_model/`: architecture, capability, and failure-mode memory.

The boundary is defined in `mutation_manifest.yaml`. It lists the snapshot
roots copied into each generation, the paths proposals may modify, protected
paths, protected symbols, and the read-only repository context available during
reflection.

Candidate generations are executed in Docker subprocesses. Inside the
container, `/agent` is a read-only generation snapshot and `/work` is per-task
scratch space. The container has no network access. All LLM calls go through
`tools.base.llm_call`, which proxies requests through the host orchestrator.

The seed runtime agent is intentionally small. It reads `agent/prompt.txt`,
optionally appends `knowledge/strategy.md`, receives one code snippet, and can
call these tools:

- `static_scan`, a bounded local heuristic scan of the current snippet;
- `list_dir`, for inspecting `/agent` or `/work`;
- `read_file`, for reading files under `/agent` or `/work`;
- `note`, for scratch notes;
- `finalize`, for submitting the label.

The seed prompt describes the task, tools, and output contract, but does not
contain a mature vulnerability-review checklist. The intention is for
task-specific policy to emerge through the evolution process.

## 3. Evolution Loop

Each run begins by snapshotting the mutable roots into `gen_0`. The
orchestrator then runs `gen_0` on the train split. This produces both a
baseline train score and task trajectories showing how the agent solved or
failed individual examples. The same generation is then evaluated on validation
to establish the baseline validation score.

Reflection receives only train-derived information:

- train metrics and confusion matrix;
- selected train failures and successes;
- code snippets for selected train cases;
- raw agent outputs for those cases;
- current mutable files such as `agent/agent.py` and `agent/prompt.txt`;
- public summaries of tool APIs;
- train-only telemetry about tool use and finalization behavior;
- recent proposal history.

Reflection can inspect bounded context through tools. In the final
architecture, it must inspect at least one `self_model/**` file or one filtered
repository contract before proposing changes. This requirement was added to
make reflection inspect the agent's architecture, not only the task examples.

Reflection ends by calling `propose_changes` with a structured proposal. The
proposal may edit the prompt, replace the solve loop, add or replace allowed
functions, or create/replace/delete mutable files. The full proposal is saved
as JSON before application.

The orchestrator copies the current accepted parent into a candidate
generation and applies the proposal inside that candidate snapshot. The
candidate then runs a smoke test on one train task. If it fails to import or
crashes, it is rejected immediately. If smoke passes, it is evaluated on the
validation split.

The gate is strict: a candidate is accepted only if validation macro-F1
strictly improves over the current parent and the candidate has zero execution
errors. Macro-F1 averages the class-wise F1 scores for `vulnerable` and `safe`,
which makes class-specific regressions more visible than accuracy alone. If
accepted, the candidate becomes the new parent, and the system runs train again
to create fresh reflection feedback. If rejected, the parent remains unchanged,
and the next reflection phase starts from the last accepted generation.

The test split is held out during evolution. It is not used for reflection,
proposal generation, validation gating, or debugging. It is evaluated only
after choosing a generation from validation.

## 4. Experiments And Results

The main final evolution run was:

```text
artifacts/runs/run_20260501_175346
```

It used the development config with runtime caps:

```text
train: 40
val:   20
test:  60
```

The validation trajectory was:

| Generation | Main change | Validation Macro-F1 | Gate |
| ---: | --- | ---: | --- |
| 0 | seed | 0.670 | baseline |
| 1 | prompt decision policy | 0.733 | accepted |
| 2 | refined prompt policy | 0.792 | accepted, best |
| 3 | solve-loop forced scan gate | 0.670 | rejected |
| 4 | more prompt specialization | 0.601 | rejected |
| 5 | solve-loop pre-scan edit | smoke failed | rejected |

The best validation generation was `gen_2`, which improved validation macro-F1
from `0.670` to `0.792`. The accepted changes were prompt-policy refinements.
The rejected generations are also important. `gen_3` modified the solve loop
and passed smoke, but validation regressed. `gen_5` attempted another
solve-loop mutation and introduced a Python error, which was caught by smoke
testing. These cases show that the system allowed broader self-modification
attempts while preventing worse or broken candidates from replacing the parent.

After selecting `gen_2`, I ran the final held-out test benchmark against both
the seed and the evolved generation. These test results were not fed back into
the evolution loop:

| Agent | Test Macro-F1 | Accuracy | Errors | LLM Calls |
| --- | ---: | ---: | ---: | ---: |
| seed `gen_0` | 0.753 | 0.767 | 0 | 82 |
| evolved `gen_2` | 0.733 | 0.750 | 0 | 62 |

The evolved agent did not beat the seed on held-out test. Validation improved,
but the improvement did not generalize. The confusion summary was:

| Agent | Safe Correct | Vulnerable Correct | False Positives | False Negatives |
| --- | ---: | ---: | ---: | ---: |
| seed `gen_0` | 30/30 | 16/30 | 0 | 14 |
| evolved `gen_2` | 30/30 | 15/30 | 0 | 15 |

Both agents preserved safe precision on this test set, but the evolved agent
missed one additional vulnerable example. The evolved agent also used
`static_scan` much less often on test: the seed used it on 22 of 60 tasks,
while `gen_2` used it on 2 of 60 tasks. My interpretation is that the selected
prompt policy became slightly too conservative and validation-specific.

## 5. What Worked

The main architecture worked as intended. The system could create candidate
generations, run them in Docker, collect train and validation metrics, invoke a
reflection phase, apply structured proposals, and reject unsafe or regressive
changes.

The immutable/mutable boundary was useful. It allowed the agent to change its
runtime body while protecting the scorer, dataset loader, validation gate,
sandbox runner, configs, data, and artifacts. This made the evolution process
auditable rather than arbitrary.

The reflection phase became more agent-like over time. In the final run,
reflection inspected train cases, mutable files, tool APIs, and `self_model/**`
files before proposing changes. Proposal artifacts were saved separately, so
each attempted generation can be reviewed after the run.

Validation gating also worked. It accepted two generations that improved
validation, rejected one solve-loop edit that regressed validation, and rejected
one broken solve-loop edit during smoke testing.

## 6. What Failed Or Surprised Me

The main failure was generalization. The validation-selected agent improved
validation macro-F1 from `0.670` to `0.792`, but held-out test macro-F1
decreased from `0.753` to `0.733`. This suggests that the final candidate
specialized to the small validation environment rather than learning a robust
general policy.

I was also surprised by the sensitivity of the system. Small changes in
prompting, split size, or reflection framing could noticeably change
performance. This made the system useful for studying self-specialization, but
also made validation selection harder to perform.

Another limitation was the agent's ability to restructure itself. Although the
architecture allowed broader mutation, the accepted generations were prompt
edits. The agent inspected `self_model/`, but did not learn to update it in
accepted generations. It had access to `knowledge/`, but did not use it as the
main specialization surface. The solve-loop edits were attempted, but rejected.

Runtime tool use was also less helpful than expected. More tool use did not
automatically produce better performance. In the final held-out test, the
evolved agent used `static_scan` less than the seed and performed slightly
worse.

## 7. What I Would Do With More Time

I would first improve the reliability of the evaluation. The final development
run used 40 train and 20 validation tasks for speed. Larger training and
validation splits, multiple random seeds, and a separate frozen dev-test split
would reduce the chance of selecting a generation that only fits a small
validation subset.

I would also increase the plateau limit. The final fast runs used a low plateau
limit because of time constraints, meaning that the loop stopped after three
iterations without improvement. A higher limit would give the agent more
iterations to observe which proposal types fail, recover from rejected
generations, and search for more useful forms of self-modification.

I would also revisit the mutation boundary. In the final version, the boundary
was intentionally conservative: the agent could modify its own runtime body,
tools, knowledge, and self-model, but not the kernel that applies proposals or
runs reflection. With more time, I would test whether some currently protected
components could be split into trusted and mutable submodules, giving the agent
more freedom to reshape its own architecture while keeping the scoring,
sandboxing, data access, and validation gate protected.

I would also test more prompt and message-orchestration variants, evaluate
additional datasets, and try task families where tool use and self-inspection
are more central to success.

## References

[1] Yangruibo Ding, Yanjun Fu, Omniyyah Ibrahim, Chawin Sitawarin, Xinyun Chen,
Basel Alomair, David Wagner, Baishakhi Ray, and Yizheng Chen. "Vulnerability
Detection with Code Language Models: How Far Are We?" arXiv:2403.18624, 2024.
https://arxiv.org/abs/2403.18624
