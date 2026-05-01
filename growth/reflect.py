"""Reflection step — produce a structured JSON proposal for the next generation.

Reads the previous generation's feedback score, a curated set of failure and success
trajectories (selected by the orchestrator), and the current agent code.
Builds a meta-prompt and asks the model for one bundled improvement proposal.
Output is structured JSON using `{rationale, intent, changes}`.

Failure modes (orchestrator handles each distinctly):
- Returns `None` if the gateway call fails or content is empty.
- Raises `ValueError` if the response cannot be parsed as a valid proposal
  after one retry with a stricter reminder.
"""

import ast
import json
import uuid
from pathlib import Path
from typing import Optional

from growth.manifest import (
    load_mutation_manifest,
    matches_any,
    reflection_context_config,
    snapshot_roots,
)


_VALID_KINDS = frozenset([
    "edit_prompt",
    "edit_solve_loop",
    "add_tool",
    "edit_tool",
    "delete_tool",
    "create_file",
    "delete_file",
    "replace_file",
    "add_function",
    "replace_function",
])
_VALID_INTENTS = frozenset(["iterate", "halt"])
_FAILURE_CODE_CUTOFF = 1500
_SUCCESS_CODE_CUTOFF = 600
_REFLECT_MAX_STEPS = 6
_TOOL_RESULT_MAX_CHARS = 14000


# --- helpers ---------------------------------------------------------------

def _extract_tools_api(gen_dir: Path) -> str:
    """Return signatures + first-line docstrings of public functions in tools/base.py."""
    tools_path = gen_dir / "tools" / "base.py"
    try:
        source = tools_path.read_text()
        tree = ast.parse(source)
    except Exception as e:
        return f"(could not parse tools/base.py: {e})"

    chunks = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = "..."
        returns = ""
        if node.returns is not None:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:
                returns = ""
        sig = f"def {node.name}({args}){returns}:"
        doc = ast.get_docstring(node) or ""
        first = doc.split("\n", 1)[0].strip()
        if first:
            chunks.append(f'{sig}\n    """{first}"""')
        else:
            chunks.append(f"{sig}\n    ...")
    return "\n\n".join(chunks) if chunks else "(no public functions)"


def _python_api_summary(path: Path, max_chars: int = 4000) -> str:
    """Return imports plus public class/function signatures for a Python file."""
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except Exception as e:
        return f"(could not summarize {path.name}: {e})"

    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                chunks.append(ast.unparse(node))
            except Exception:
                pass
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_") and node.name not in {
                "_run_task_protocol",
                "_load_split",
            }:
                continue
            try:
                if isinstance(node, ast.ClassDef):
                    header = f"class {node.name}:"
                else:
                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    returns = ""
                    if node.returns is not None:
                        try:
                            returns = f" -> {ast.unparse(node.returns)}"
                        except Exception:
                            returns = ""
                    header = f"{prefix} {node.name}({ast.unparse(node.args)}){returns}:"
            except Exception:
                header = f"{node.__class__.__name__} {node.name}(...)"
            doc = ast.get_docstring(node) or ""
            first = doc.split("\n", 1)[0].strip()
            chunks.append(f'{header}\n    """{first}"""' if first else header)
    text = "\n\n".join(chunks) if chunks else "(no public API found)"
    return text[:max_chars]


def _context_denied(rel_path: str, deny_patterns: list[str]) -> bool:
    parts = set(Path(rel_path).parts)
    if "__pycache__" in parts:
        return True
    if rel_path.endswith((".pyc", ".pyo")):
        return True
    return matches_any(rel_path, deny_patterns)


def _repo_tree_summary(project_root: Path, manifest: dict) -> str:
    ctx = reflection_context_config(manifest)
    deny = ctx.get("deny_paths") or []
    max_files = int(ctx.get("max_tree_files", 120))
    paths = []
    for p in sorted(project_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(project_root).as_posix()
        if _context_denied(rel, deny):
            continue
        paths.append(rel)
        if len(paths) >= max_files:
            break
    return "\n".join(paths) if paths else "(no repo files visible)"


def _repo_context(project_root: Path, manifest: dict) -> str:
    ctx = reflection_context_config(manifest)
    max_chars = int(ctx.get("max_file_chars", 4000))
    deny = ctx.get("deny_paths") or []
    blocks = [f"── repo tree (filtered) ──\n{_repo_tree_summary(project_root, manifest)}"]
    for item in ctx.get("files") or []:
        rel = item.get("path")
        mode = item.get("mode", "excerpt")
        if not isinstance(rel, str):
            continue
        if _context_denied(rel, deny):
            continue
        path = project_root / rel
        if not path.exists() or not path.is_file():
            continue
        if mode == "api" and path.suffix == ".py":
            body = _python_api_summary(path, max_chars=max_chars)
        else:
            body = path.read_text()[:max_chars]
        blocks.append(f"── {rel} ({mode}) ──\n{body}")
    return "\n\n".join(blocks)


def _mutable_snapshot_context(gen_dir: Path, manifest: dict) -> str:
    ctx = reflection_context_config(manifest)
    max_chars = int(ctx.get("max_snapshot_file_chars", 12000))
    blocks = []
    for root_name in snapshot_roots(manifest):
        root = gen_dir / root_name
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.endswith((".pyc", ".pyo")):
                continue
            rel = p.relative_to(gen_dir).as_posix()
            try:
                content = p.read_text()
            except UnicodeDecodeError:
                content = "(binary file omitted)"
            truncated = "\n...[truncated]..." if len(content) > max_chars else ""
            blocks.append(f"── {rel} ──\n{content[:max_chars]}{truncated}")
    return "\n\n".join(blocks) if blocks else "(no mutable snapshot files found)"


def _compute_confusion(per_task: list) -> dict:
    """Counts and class-wise metrics. Errors hurt the true class's recall."""
    by_exp = {
        "vulnerable": {"vulnerable": 0, "safe": 0, "none": 0, "error": 0},
        "safe": {"vulnerable": 0, "safe": 0, "none": 0, "error": 0},
    }
    for t in per_task:
        exp = t.get("expected")
        if exp not in by_exp:
            continue
        if t.get("errored"):
            by_exp[exp]["error"] += 1
        elif t.get("predicted") in ("vulnerable", "safe"):
            by_exp[exp][t["predicted"]] += 1
        else:
            by_exp[exp]["none"] += 1

    v = by_exp["vulnerable"]
    s = by_exp["safe"]

    tp_v = v["vulnerable"]
    fn_v = v["safe"] + v["none"]
    err_v = v["error"]
    fp_s = s["vulnerable"]      # predicted vuln, actual safe
    tn_s = s["safe"]
    none_s = s["none"]
    err_s = s["error"]

    # vulnerable class: TP / (TP + FP); recall folds errors into the denominator
    prec_v = tp_v / (tp_v + fp_s) if (tp_v + fp_s) > 0 else 0.0
    rec_v = tp_v / (tp_v + fn_v + err_v) if (tp_v + fn_v + err_v) > 0 else 0.0
    f1_v = 2 * prec_v * rec_v / (prec_v + rec_v) if (prec_v + rec_v) > 0 else 0.0

    # safe class: TP_s = tn_s; FP_s = predicted safe but actually vuln (= v["safe"])
    prec_s = tn_s / (tn_s + v["safe"]) if (tn_s + v["safe"]) > 0 else 0.0
    rec_s_denom = tn_s + fp_s + none_s + err_s
    rec_s = tn_s / rec_s_denom if rec_s_denom > 0 else 0.0
    f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0

    return dict(
        tp_v=tp_v, fn_v=fn_v, err_v=err_v,
        fp_s=fp_s, tn_s=tn_s, none_s=none_s, err_s=err_s,
        prec_v=prec_v, rec_v=rec_v, f1_v=f1_v,
        prec_s=prec_s, rec_s=rec_s, f1_s=f1_s,
    )


def _format_trajectory(traj: dict, n: int, kind: str, code_cutoff: int) -> str:
    task_id = traj.get("task_id", "?")
    expected = traj.get("expected", "?")
    bucket = traj.get("reflection_bucket")
    if traj.get("errored"):
        predicted_str = f"(error: {traj.get('error', '?')})"
    elif traj.get("predicted") is None:
        predicted_str = "none (unparseable)"
    else:
        predicted_str = str(traj.get("predicted"))
    code = (traj.get("code") or "")[:code_cutoff]
    raw = traj.get("raw") or "(no raw output)"
    bar = "─" * 30
    bucket_str = f"  bucket={bucket}" if bucket else ""
    return (
        f"[{kind} {n}]  task_id={task_id}  expected={expected}  "
        f"predicted={predicted_str}{bucket_str}\n"
        f"── code (first {code_cutoff} chars) {bar}\n"
        f"{code}\n"
        f"── agent raw output {bar}\n"
        f"{raw}\n"
    )


def _format_trajectory_summary(traj: dict, n: int, kind: str) -> str:
    task_id = traj.get("task_id", "?")
    expected = traj.get("expected", "?")
    predicted = "(error)" if traj.get("errored") else traj.get("predicted")
    bucket = traj.get("reflection_bucket", "?")
    telemetry = traj.get("telemetry") or {}
    tools = telemetry.get("tool_calls_by_name") or {}
    tool_str = ", ".join(f"{name}={count}" for name, count in sorted(tools.items()))
    if not tool_str:
        tool_str = "none"
    raw = (traj.get("raw") or "").strip().replace("\n", " ")
    raw = raw[:240] + ("..." if len(raw) > 240 else "")
    return (
        f"[{kind} {n}] task_id={task_id} expected={expected} "
        f"predicted={predicted} bucket={bucket} "
        f"llm_calls={telemetry.get('llm_calls', '?')} tools={tool_str} "
        f"raw={raw or '(no raw output)'}"
    )


def _format_self_observation(self_observation: Optional[dict]) -> str:
    if not self_observation:
        return "(no self-observation telemetry available)"

    solve = self_observation.get("solve_telemetry") or {}
    source_split = self_observation.get("source_split") or "train"
    tool_counts = solve.get("tool_calls_by_name") or {}
    first_tool_counts = solve.get("first_tool_counts") or {}
    recent = self_observation.get("recent_proposals") or []

    def _counts_text(counts: dict) -> str:
        if not counts:
            return "none"
        return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))

    lines = [
        f"Source split: {source_split} only. This contains behavior telemetry, "
        "not validation/test task contents.",
        (
            "Telemetry scope: tool-use counts here are LLM-requested tool calls "
            "observed by the orchestrator gateway. Direct helper calls inside "
            "mutable code, such as agent.py precomputing static_scan before "
            "llm_call, are not counted here; inspect agent.py before concluding "
            "the workflow uses no tools."
        ),
        (
            "Solve calls: "
            f"tasks={solve.get('n_tasks', 0)}, "
            f"total_llm_calls={solve.get('llm_calls_total', 0)}, "
            f"avg_llm_calls_per_task={solve.get('avg_llm_calls_per_task', 0.0):.2f}, "
            f"max_llm_calls_per_task={solve.get('max_llm_calls_per_task', 0)}"
        ),
        (
            "Tool use: "
            f"tool_calls_total={solve.get('tool_calls_total', 0)}, "
            f"inspection_tool_calls_total={solve.get('inspection_tool_calls_total', 0)}, "
            f"tool_use_rate={solve.get('tool_use_rate', 0.0):.2f}, "
            f"inspection_tool_use_rate={solve.get('inspection_tool_use_rate', 0.0):.2f}, "
            f"immediate_finalize_rate={solve.get('immediate_finalize_rate', 0.0):.2f}"
        ),
        f"Tool counts by name: {_counts_text(tool_counts)}",
        f"First tool counts: {_counts_text(first_tool_counts)}",
        (
            "Task counts: "
            f"static_scan={solve.get('tasks_with_static_scan', 0)}, "
            f"read_file={solve.get('tasks_with_read_file', 0)}, "
            f"note={solve.get('tasks_with_note', 0)}, "
            f"finalize_tool={solve.get('tasks_with_finalize_tool', 0)}, "
            f"text_response={solve.get('tasks_with_text_response', 0)}"
        ),
    ]

    if recent:
        lines.append("Recent proposal outcomes:")
        for item in recent[-6:]:
            changed = item.get("changed_files") or []
            changed_text = ", ".join(changed[:5]) if changed else "none"
            if len(changed) > 5:
                changed_text += ", ..."
            lines.append(
                f"- gen={item.get('gen')} outcome={item.get('outcome')} "
                f"stage={item.get('stage')} kinds={item.get('kinds')} "
                f"changed_files={changed_text}"
            )
    else:
        lines.append("Recent proposal outcomes: none yet")

    return "\n".join(lines)


def _build_messages(
    gen_idx: int,
    score: dict,
    trajectories: list,
    agent_py: str,
    prompt_txt: str,
    tools_api: str,
    repo_context: str = "",
    mutable_snapshot_context: str = "",
    score_split: str = "val",
    self_observation: Optional[dict] = None,
) -> list:
    cm = _compute_confusion(score.get("per_task", []))
    n_tasks = score.get("n_tasks", 0)
    macro_f1 = score.get("macro_f1", 0.0)
    accuracy = score.get("accuracy", 0.0)
    n_errors = score.get("n_errors", 0)

    failures = [t for t in trajectories if t.get("errored") or not t.get("ok")]
    successes = [t for t in trajectories if not t.get("errored") and t.get("ok")]

    failure_blocks = "\n".join(
        _format_trajectory_summary(t, i + 1, "FAILURE")
        for i, t in enumerate(failures)
    ) or "(no failures shown)\n"

    success_blocks = "\n".join(
        _format_trajectory_summary(t, i + 1, "SUCCESS")
        for i, t in enumerate(successes)
    ) or "(no successes shown)\n"

    system_msg = (
        "You are the reflection engine for a self-modifying security-analysis agent.\n\n"
        "Your responsibility: act as the agent's self-inspection loop. Use the "
        "read-only tools to inspect train cases, mutable files, and repo contracts, "
        "then submit exactly ONE structured improvement proposal through the "
        "`propose_changes` tool.\n\n"
        "Constraints you must never violate:\n"
        "1. The agent must never `import openai` — it must use tools.base.llm_call.\n"
        "2. solve_task must not print to stdout — use tools.base.note() for debug scratch.\n"
        "3. solve_task must return a dict with keys: task_id (str), label "
        "(\"vulnerable\"|\"safe\"|None).\n"
        "4. Inside the sandbox, /agent is read-only and /work is per-task scratch "
        "(wiped between tasks).\n"
        "5. At runtime the agent sees only its generation snapshot at /agent "
        "and per-task scratch at /work. The read-only repo context below is "
        "for reflection only and cannot be edited.\n"
        "6. Prefer bounded tools such as static_scan over open-ended run_bash; "
        "do not introduce repeated static-analysis retries.\n"
        "7. Never propose edits outside manifest-mutable paths, and never "
        "change protected kernel files or protected symbols.\n\n"
        "Tool rules:\n"
        "- First inspect at least one train case or mutable file using the tools.\n"
        "- End by calling `propose_changes` exactly once.\n"
        "- Do not output prose as your final answer; final proposals must be tool calls."
    )

    user_msg = f"""\
══════════════════════════════════════════════════════════════════════
PERFORMANCE  ·  generation {gen_idx}  ·  {score_split} split  ·  {n_tasks} tasks
══════════════════════════════════════════════════════════════════════

  macro-F1 : {macro_f1:.3f}    accuracy : {accuracy:.3f}

  Confusion matrix
                         predicted
                    vulnerable    safe    none/error
  actual vulnerable   TP={cm['tp_v']:>3}    FN={cm['fn_v']:>3}    {cm['err_v']:>3}
  actual safe         FP={cm['fp_s']:>3}    TN={cm['tn_s']:>3}    {cm['err_s']:>3}

  class "vulnerable"  precision={cm['prec_v']:.2f}  recall={cm['rec_v']:.2f}  F1={cm['f1_v']:.3f}
  class "safe"        precision={cm['prec_s']:.2f}  recall={cm['rec_s']:.2f}  F1={cm['f1_s']:.3f}

  Errors (no label returned): {n_errors}

══════════════════════════════════════════════════════════════════════
FAILURE TRAJECTORIES  ({len(failures)} shown, balanced train sample)
══════════════════════════════════════════════════════════════════════
{failure_blocks}
══════════════════════════════════════════════════════════════════════
SUCCESS TRAJECTORIES  ({len(successes)} shown for contrast)
══════════════════════════════════════════════════════════════════════
{success_blocks}
══════════════════════════════════════════════════════════════════════
TRAIN-ONLY SELF-OBSERVATION
══════════════════════════════════════════════════════════════════════
{_format_self_observation(self_observation)}
══════════════════════════════════════════════════════════════════════
INSPECTION CONTEXT AVAILABLE THROUGH TOOLS
══════════════════════════════════════════════════════════════════════

Use the tools to inspect before proposing:
- list_mutable_files: see the evolving files available in this generation.
- read_mutable_file: read agent/**, tools/**, or knowledge/** from the generation snapshot.
- read_train_case: read full code and raw output for one shown train task_id.
- read_repo_context: read filtered immutable context such as "tree", README/config summaries, or public kernel APIs.
- read_tool_api: inspect public tools/base.py signatures.

══════════════════════════════════════════════════════════════════════
YOUR PROPOSAL
══════════════════════════════════════════════════════════════════════

Choose the smallest coherent proposal likely to improve macro-F1 on the next
val run. Prefer a single change. Use a short bundle only when the changes are
tightly coupled, for example creating a helper file and updating agent.py to
import it.

The trajectories are a balanced sample of the train split, not a target list to
memorize. Use them to infer general error patterns. Preserve both vulnerable
recall and safe precision; do not fix one class by broadly sacrificing the
other.

Use the self-observation telemetry to reason about the agent's own behavior:
whether it is using tools, whether it finalizes too early, and whether recent
mutations keep changing the same surface without progress. Do not treat it as
a command to use a specific tool or edit a specific file.

Choose among mutable surfaces neutrally:
- agent/agent.py can change workflow, tool-use policy, or how context is assembled.
- tools/base.py can gain bounded helper tools or improve mutable helper behavior.
- knowledge/ can store learned policies or notes when a policy file is the best fit.
- agent/prompt.txt can change only the generic task framing and output contract.

When ready, call `propose_changes` with exactly this top-level shape:

{{
  "rationale": "<what root cause this fixes, citing specific task_ids above>",
  "intent": "iterate",
  "changes": [
    {{"kind": "edit_prompt", "details": {{"content": "<complete agent/prompt.txt>"}}}},
    {{"kind": "edit_solve_loop", "details": {{"content": "<complete agent/agent.py>"}}}},
    {{"kind": "add_tool", "details": {{"name": "<function_name>", "code": "<complete Python def block>"}}}},
    {{"kind": "edit_tool", "details": {{"name": "<existing_function_name>", "code": "<complete replacement Python def block>"}}}},
    {{"kind": "delete_tool", "details": {{"name": "<existing_non_core_function_name>"}}}},
    {{"kind": "create_file", "details": {{"path": "agent/x.py", "content": "<new file content>"}}}},
    {{"kind": "replace_file", "details": {{"path": "knowledge/strategy.md", "content": "<complete file content>"}}}},
    {{"kind": "delete_file", "details": {{"path": "agent/old.py"}}}},
    {{"kind": "add_function", "details": {{"path": "agent/helpers.py", "name": "<function_name>", "code": "<complete Python def block>"}}}},
    {{"kind": "replace_function", "details": {{"path": "tools/base.py", "name": "<existing_function_name>", "code": "<complete replacement Python def block>"}}}}
  ]
}}

Use `"intent": "halt"` only if the current agent is good enough and further
evolution is likely to overfit or regress. If intent is "halt", changes may be
an empty list.

Allowed change kinds:
- edit_prompt: replace all of agent/prompt.txt.
- edit_solve_loop: replace all of agent/agent.py, including imports.
- add_tool: append one new public function to tools/base.py.
- edit_tool: replace one existing public function in tools/base.py.
- delete_tool: delete one non-core public function from tools/base.py.
- create_file: create a new manifest-mutable file.
- replace_file: replace a complete manifest-mutable file.
- delete_file: delete a non-protected manifest-mutable file.
- add_function: append one top-level function to a manifest-mutable Python file.
- replace_function: replace one existing top-level function in a manifest-mutable Python file.

Additional constraints:
- "rationale" must name at least one specific task_id from the trajectories above.
- edit_prompt / edit_solve_loop: "content" is the COMPLETE new file, not a diff.
- edit_solve_loop: "content" must contain `def solve_task(task: dict) -> dict:`.
- add_tool / edit_tool / add_function / replace_function: code must define exactly one function whose name matches "name".
- add_tool / edit_tool / add_function / replace_function: the function may only import from this allowlist:
    re, ast, json, os, os.path, pathlib, subprocess, math,
    collections, itertools, functools, typing, datetime,
    hashlib, base64, textwrap, uuid, __future__, dataclasses,
    tools, tools.base, agent. Nothing else.
- delete_tool must not delete core tools: llm_call, read_file, write_file,
  list_dir, run_bash, note.
- edit_tool / replace_function must not change protected symbols such as
  llm_call, read_file, write_file, list_dir, run_bash, note, _safe_path,
  _set_rpc_channels, or _set_task_context.
- delete_file must not delete protected files: agent/agent.py,
  agent/prompt.txt, tools/base.py.
- create_file / replace_file / delete_file / add_function / replace_function
  paths must be relative and manifest-mutable (agent/**, tools/**, knowledge/**).

Inspect first, then call propose_changes."""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _proposal_tool_schema() -> dict:
    """Return the JSON schema used by the final reflection tool."""
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "intent": {"type": "string", "enum": ["iterate", "halt"]},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": sorted(_VALID_KINDS)},
                        "details": {"type": "object"},
                    },
                    "required": ["kind", "details"],
                },
            },
        },
        "required": ["rationale", "intent", "changes"],
    }


def _reflection_tool_specs() -> list:
    """Read-only tools available to the reflection model."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_mutable_files",
                "description": (
                    "List files in the current generation snapshot that are "
                    "eligible mutable surfaces."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_mutable_file",
                "description": (
                    "Read one manifest snapshot file, such as agent/agent.py, "
                    "agent/prompt.txt, tools/base.py, or knowledge/strategy.md."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path under a snapshot root.",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_train_case",
                "description": (
                    "Read full code and raw agent output for one task_id from "
                    "the shown train feedback bundle."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_repo_context",
                "description": (
                    "Read filtered immutable context. Use path='tree' for a repo "
                    "tree summary or one manifest-allowed context path."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_tool_api",
                "description": "Read public tools/base.py function signatures.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "propose_changes",
                "description": (
                    "Submit the final structured improvement proposal. Call this "
                    "after inspecting context."
                ),
                "parameters": _proposal_tool_schema(),
            },
        },
    ]


def _safe_snapshot_rel(path: str, manifest: dict) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    rel = Path(path.strip())
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("path must be relative and stay inside the snapshot")
    rel_s = rel.as_posix()
    roots = snapshot_roots(manifest)
    if not any(rel_s == root or rel_s.startswith(root + "/") for root in roots):
        raise ValueError(f"path must be under one of: {', '.join(roots)}")
    if rel_s.endswith((".pyc", ".pyo")) or "__pycache__" in rel.parts:
        raise ValueError("Python cache files are not readable reflection context")
    return rel_s


def _list_mutable_files(gen_dir: Path, manifest: dict) -> list[dict]:
    files = []
    for root_name in snapshot_roots(manifest):
        root = gen_dir / root_name
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.endswith((".pyc", ".pyo")):
                continue
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(gen_dir).as_posix()
            files.append({"path": rel, "bytes": p.stat().st_size})
    return files


def _read_mutable_file(gen_dir: Path, manifest: dict, rel_path: str) -> dict:
    rel = _safe_snapshot_rel(rel_path, manifest)
    path = gen_dir / rel
    if not path.exists() or not path.is_file():
        return {"error": f"not found: {rel}"}
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return {"path": rel, "content": "(binary file omitted)", "truncated": False}
    max_chars = int(reflection_context_config(manifest).get(
        "max_snapshot_file_chars", 12000
    ))
    return {
        "path": rel,
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _read_repo_context_path(project_root: Path, manifest: dict, rel_path: str) -> dict:
    if rel_path == "tree":
        return {"path": "tree", "content": _repo_tree_summary(project_root, manifest)}

    ctx = reflection_context_config(manifest)
    deny = ctx.get("deny_paths") or []
    allowed = {
        item.get("path"): item.get("mode", "excerpt")
        for item in (ctx.get("files") or [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if rel_path not in allowed:
        return {"error": f"path is not in reflection context allowlist: {rel_path}"}
    if _context_denied(rel_path, deny):
        return {"error": f"path is denied: {rel_path}"}
    path = project_root / rel_path
    if not path.exists() or not path.is_file():
        return {"error": f"not found: {rel_path}"}

    max_chars = int(ctx.get("max_file_chars", 4000))
    mode = allowed[rel_path]
    if mode == "api" and path.suffix == ".py":
        body = _python_api_summary(path, max_chars=max_chars)
    else:
        body = path.read_text()[:max_chars]
    return {"path": rel_path, "mode": mode, "content": body}


def _dispatch_reflection_tool(
    name: str,
    args: dict,
    *,
    gen_dir: Path,
    project_root: Path,
    manifest: dict,
    trajectories: list,
) -> tuple[object, bool]:
    """Return (tool_result, counts_as_inspection)."""
    if name == "list_mutable_files":
        return _list_mutable_files(gen_dir, manifest), True
    if name == "read_mutable_file":
        try:
            return _read_mutable_file(gen_dir, manifest, args.get("path", "")), True
        except ValueError as e:
            return {"error": str(e)}, True
    if name == "read_train_case":
        task_id = args.get("task_id")
        for idx, traj in enumerate(trajectories):
            if traj.get("task_id") == task_id:
                cutoff = _FAILURE_CODE_CUTOFF if (
                    traj.get("errored") or not traj.get("ok")
                ) else _SUCCESS_CODE_CUTOFF
                kind = "FAILURE" if (traj.get("errored") or not traj.get("ok")) else "SUCCESS"
                return _format_trajectory(traj, idx + 1, kind, cutoff), True
        return {"error": f"task_id not in shown train bundle: {task_id}"}, True
    if name == "read_repo_context":
        return _read_repo_context_path(project_root, manifest, args.get("path", "")), True
    if name == "read_tool_api":
        return _extract_tools_api(gen_dir), True
    if name == "propose_changes":
        return {"error": "propose_changes is handled by the reflection loop"}, False
    return {"error": f"unknown reflection tool: {name}"}, False


def _proposal_from_tool_args(args: dict) -> dict:
    if not isinstance(args, dict):
        raise ValueError("propose_changes arguments must be an object")
    if "proposal" in args and isinstance(args["proposal"], dict):
        args = args["proposal"]
    return _parse_proposal(json.dumps(args))


def _gateway_turn(
    llm_handler,
    messages: list,
    model: str,
    *,
    tools: Optional[list] = None,
    tool_choice=None,
    step: int = 1,
    response_format: Optional[dict] = None,
) -> Optional[dict]:
    """Send one llm_request envelope through the orchestrator gateway."""
    envelope = {
        "_kind": "llm_request",
        "id": str(uuid.uuid4()),
        "messages": messages,
        "model": model,
        "response_format": response_format,
        "tools": tools,
        "tool_choice": tool_choice,
        "task_id": None,
        "purpose": "reflect",
        "step_in_task": step,
    }
    try:
        resp = llm_handler(envelope)
    except Exception:
        return None
    if not isinstance(resp, dict) or not resp.get("ok"):
        return None
    return {
        "content": resp.get("content"),
        "tool_calls": resp.get("tool_calls"),
    }


def _gateway_call(llm_handler, messages: list, model: str) -> Optional[str]:
    """Compatibility helper for one JSON-object reflection call."""
    resp = _gateway_turn(
        llm_handler,
        messages,
        model,
        response_format={"type": "json_object"},
    )
    if resp is None:
        return None
    content = resp.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _parse_proposal(content: str) -> dict:
    """Parse and shallow-validate a proposal. Raises ValueError on failure."""
    try:
        proposal = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}")
    if not isinstance(proposal, dict):
        raise ValueError(f"top-level must be an object, got {type(proposal).__name__}")

    # Backwards compatibility: accept old single-edit proposals and normalize
    # them into the new bundle schema.
    if "kind" in proposal:
        kind = proposal.get("kind")
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}")
        details = proposal.get("details")
        if not isinstance(details, dict):
            raise ValueError("details must be an object")
        rationale = details.get("rationale") or ""
        if not isinstance(rationale, str):
            rationale = ""
        return {
            "rationale": rationale,
            "intent": "iterate",
            "changes": [{"kind": kind, "details": details}],
        }

    rationale = proposal.get("rationale")
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")

    intent = proposal.get("intent")
    if intent not in _VALID_INTENTS:
        raise ValueError(f"intent must be one of {sorted(_VALID_INTENTS)}, got {intent!r}")

    changes = proposal.get("changes")
    if not isinstance(changes, list):
        raise ValueError("changes must be a list")

    for idx, change in enumerate(changes):
        if not isinstance(change, dict):
            raise ValueError(
                f"changes[{idx}] must be an object, got {type(change).__name__}"
            )
        kind = change.get("kind")
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"changes[{idx}].kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
            )
        if not isinstance(change.get("details"), dict):
            raise ValueError(f"changes[{idx}].details must be an object")

    return proposal


# --- public API ------------------------------------------------------------

def reflect(
    trajectories: list,
    current_gen_dir: str,
    llm_handler=None,
    config: Optional[dict] = None,
    score: Optional[dict] = None,
    gen_idx: int = 0,
    score_split: str = "val",
    project_root: Optional[Path] = None,
    manifest: Optional[dict] = None,
    self_observation: Optional[dict] = None,
) -> Optional[dict]:
    """Produce a structured JSON proposal for the next generation.

    `trajectories` is the curated set the orchestrator decided to show
    (already includes per-task code). `score` is the full aggregate dict
    used to render the confusion matrix.
    """
    if llm_handler is None:
        return None

    gen_dir = Path(current_gen_dir)
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    if manifest is None:
        manifest = load_mutation_manifest(project_root)
    model_cfg = (config or {}).get("model", {})
    model = model_cfg.get("reflect_name") or model_cfg.get("name", "gpt-5.4")

    try:
        agent_py = (gen_dir / "agent" / "agent.py").read_text()
        prompt_txt = (gen_dir / "agent" / "prompt.txt").read_text()
    except FileNotFoundError as e:
        raise ValueError(f"reflect: missing source file in {gen_dir}: {e}")
    messages = _build_messages(
        gen_idx=gen_idx,
        score=score or {},
        score_split=score_split,
        trajectories=trajectories or [],
        agent_py=agent_py,
        prompt_txt=prompt_txt,
        tools_api=_extract_tools_api(gen_dir),
        self_observation=self_observation,
    )
    tools = _reflection_tool_specs()
    inspected = False
    fallback_content = None

    for step in range(1, _REFLECT_MAX_STEPS + 1):
        tool_choice = "auto"
        if step == _REFLECT_MAX_STEPS:
            tool_choice = {"type": "function", "function": {"name": "propose_changes"}}

        resp = _gateway_turn(
            llm_handler,
            messages,
            model,
            tools=tools,
            tool_choice=tool_choice,
            step=step,
        )
        if resp is None:
            return None

        content = resp.get("content")
        tool_calls = resp.get("tool_calls")
        if isinstance(content, str) and content.strip():
            fallback_content = content

        assistant_msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            messages.append({
                "role": "user",
                "content": (
                    "Use read-only inspection tools if needed, then call "
                    "`propose_changes`. Do not answer in plain text."
                ),
            })
            continue

        tool_result_msgs = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "propose_changes":
                if not inspected and step < _REFLECT_MAX_STEPS:
                    result = {
                        "error": (
                            "Inspect at least one train case or mutable file "
                            "before proposing changes."
                        )
                    }
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    })
                    continue
                return _proposal_from_tool_args(args)

            result, counts = _dispatch_reflection_tool(
                name,
                args,
                gen_dir=gen_dir,
                project_root=Path(project_root),
                manifest=manifest,
                trajectories=trajectories or [],
            )
            inspected = inspected or counts
            result_text = json.dumps(result) if not isinstance(result, str) else result
            if len(result_text) > _TOOL_RESULT_MAX_CHARS:
                result_text = result_text[:_TOOL_RESULT_MAX_CHARS] + "\n...[truncated]..."
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_text,
            })

        messages.extend(tool_result_msgs)

    if fallback_content is None:
        return None

    try:
        return _parse_proposal(fallback_content)
    except ValueError as first_err:
        retry_messages = messages + [
            {"role": "user", "content": (
                "Your previous response did not call propose_changes with a valid "
                "proposal. Call propose_changes now with one valid proposal."
            )},
        ]
        retry_resp = _gateway_turn(
            llm_handler,
            retry_messages,
            model,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "propose_changes"}},
            step=_REFLECT_MAX_STEPS + 1,
        )
        if retry_resp is None:
            raise ValueError(f"reflect: parse failed and retry call returned no content. "
                             f"first error: {first_err}")
        retry_calls = retry_resp.get("tool_calls") or []
        for tc in retry_calls:
            fn = tc.get("function", {})
            if fn.get("name") == "propose_changes":
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                return _proposal_from_tool_args(args)
        retry_content = retry_resp.get("content")
        if not isinstance(retry_content, str):
            raise ValueError(f"reflect: retry did not return proposal. first error: {first_err}")
        return _parse_proposal(retry_content)
