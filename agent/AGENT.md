# Agent — local rules

This directory's code (`agent.py`, `prompt.txt`, plus anything future
generations add) is part of what evolves across generations. The current
manifest also snapshots `tools/`, `knowledge/`, and `self_model/`. The
orchestrator copies all `mutation_manifest.yaml` snapshot roots into
`artifacts/runs/<run_id>/generations/gen_N/` at the start of each generation;
edits made here become part of gen_0. Edits proposed by reflection are applied
inside the next candidate snapshot via `growth/apply.py`.

## Hard rules for agent code

1. **Never `import openai` directly.** All LLM access must go through
   `tools.base.llm_call(messages, model=None, response_format=None, tools=None,
   tool_choice=None)`. The
   sandbox has network disabled, so direct OpenAI calls fail anyway —
   but more importantly, routing through the orchestrator gives us
   ground-truth cost tracking and a clean audit trail. If a future
   generation needs LLM features that `llm_call` doesn't expose
   (streaming, vision, embeddings, etc.), extend `llm_call` and the
   orchestrator's host-side handler — don't bypass the proxy.

2. **Don't print to stdout in `solve_task`.** The runner's stdout is the
   RPC channel back to the orchestrator. The runner captures stdout
   during `solve_task` to a buffer (so accidental prints don't break the
   protocol), but anything printed there is invisible to the orchestrator
   except as a `_stdout_captured` field on the result. Use
   `tools.base.note(text)` for debugging trails — it writes to
   `/work/notes.txt`.

3. **Return a dict from `solve_task`.** Required keys:
   - `task_id` (echoed from input)
   - `label` (one of `"vulnerable"`, `"safe"`, or `None` for "couldn't
     decide")

   Optional keys: `cwe`, `raw` (the model's raw output), and anything
   else you find useful for downstream analysis. If `label` is `None`
   the orchestrator counts the task as wrong — that's intentional, so
   parse failures show up as a real evolution signal rather than as
   silent confident-wrong predictions.

4. **Filesystem rules.** `/agent` is mounted read-only — you can read
   the snapshot but not write to it. `/work` is per-task scratch and is
   wiped between tasks. Anything outside those two prefixes is off
   limits and `tools.base` will refuse the path.

5. **The orchestrator is invisible.** You cannot see `orchestrator.py`,
   `growth/`, `eval/`, `sandbox/`, `data/`, or `artifacts/` from inside the
   sandbox. Reflection may receive selected read-only summaries of those
   systems, but runtime agent code cannot import or edit them.

6. **Choose the right mutable surface.** Use `knowledge/` for learned policy,
   `self_model/` for architecture/capability/failure-mode memory, `agent/`
   for workflow and prompt assembly, and `tools/` for bounded helper
   capabilities. Keep the seed prompt basic unless changing the generic task
   framing or output contract is truly the smallest useful edit.
