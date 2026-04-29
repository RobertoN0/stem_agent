# Agent — local rules

This directory's code (`agent.py`, `prompt.txt`, plus anything future
generations add) IS what evolves across generations. The orchestrator
copies this tree into `artifacts/gen_N/` at the start of each generation;
edits made here become part of gen_0. Edits made by the agent at runtime
go into `artifacts/gen_N+1/agent/` via `growth/apply.py`.

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
   `growth/`, `eval/`, or `sandbox/` from inside the sandbox. Don't try
   to reach for them.
