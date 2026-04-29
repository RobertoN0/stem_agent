"""Reflection step — produce a structured JSON proposal for the next generation.

Reads the previous generation's val score, a curated set of failure and success
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


_VALID_KINDS = frozenset([
    "edit_prompt",
    "edit_solve_loop",
    "add_tool",
    "edit_tool",
    "delete_tool",
    "create_file",
    "delete_file",
])
_VALID_INTENTS = frozenset(["iterate", "halt"])
_FAILURE_CODE_CUTOFF = 1500
_SUCCESS_CODE_CUTOFF = 600


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
    if traj.get("errored"):
        predicted_str = f"(error: {traj.get('error', '?')})"
    elif traj.get("predicted") is None:
        predicted_str = "none (unparseable)"
    else:
        predicted_str = str(traj.get("predicted"))
    code = (traj.get("code") or "")[:code_cutoff]
    raw = traj.get("raw") or "(no raw output)"
    bar = "─" * 30
    return (
        f"[{kind} {n}]  task_id={task_id}  expected={expected}  predicted={predicted_str}\n"
        f"── code (first {code_cutoff} chars) {bar}\n"
        f"{code}\n"
        f"── agent raw output {bar}\n"
        f"{raw}\n"
    )


def _build_messages(
    gen_idx: int,
    score: dict,
    trajectories: list,
    agent_py: str,
    prompt_txt: str,
    tools_api: str,
) -> list:
    cm = _compute_confusion(score.get("per_task", []))
    n_tasks = score.get("n_tasks", 0)
    macro_f1 = score.get("macro_f1", 0.0)
    accuracy = score.get("accuracy", 0.0)
    n_errors = score.get("n_errors", 0)

    failures = [t for t in trajectories if t.get("errored") or not t.get("ok")]
    successes = [t for t in trajectories if not t.get("errored") and t.get("ok")]

    failure_blocks = "\n".join(
        _format_trajectory(t, i + 1, "FAILURE", _FAILURE_CODE_CUTOFF)
        for i, t in enumerate(failures)
    ) or "(no failures shown)\n"

    success_blocks = "\n".join(
        _format_trajectory(t, i + 1, "SUCCESS", _SUCCESS_CODE_CUTOFF)
        for i, t in enumerate(successes)
    ) or "(no successes shown)\n"

    system_msg = (
        "You are the reflection engine for a self-modifying security-analysis agent.\n\n"
        "Your single responsibility: study the agent's current code, its recent "
        "mistakes, and output exactly ONE structured improvement proposal for the "
        "next generation.\n\n"
        "Constraints you must never violate:\n"
        "1. The agent must never `import openai` — it must use tools.base.llm_call.\n"
        "2. solve_task must not print to stdout — use tools.base.note() for debug scratch.\n"
        "3. solve_task must return a dict with keys: task_id (str), label "
        "(\"vulnerable\"|\"safe\"|None).\n"
        "4. Inside the sandbox, /agent is read-only and /work is per-task scratch "
        "(wiped between tasks).\n"
        "5. The orchestrator, growth/, eval/, and sandbox/ are completely invisible "
        "to the agent.\n\n"
        "Output rules:\n"
        "- Your response must be a single JSON object and absolutely nothing else.\n"
        "- No markdown fences, no prose before or after, no \"Here is my proposal:\".\n"
        "- If you output anything other than valid JSON, the proposal is discarded "
        "and the generation is wasted."
    )

    user_msg = f"""\
══════════════════════════════════════════════════════════════════════
PERFORMANCE  ·  generation {gen_idx}  ·  val split  ·  {n_tasks} tasks
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
FAILURE TRAJECTORIES  ({len(failures)} shown, most instructive first)
══════════════════════════════════════════════════════════════════════
{failure_blocks}
══════════════════════════════════════════════════════════════════════
SUCCESS TRAJECTORIES  ({len(successes)} shown for contrast)
══════════════════════════════════════════════════════════════════════
{success_blocks}
══════════════════════════════════════════════════════════════════════
CURRENT AGENT CODE
══════════════════════════════════════════════════════════════════════

── agent/agent.py ──────────────────────────────────────────────────
{agent_py}

── agent/prompt.txt ────────────────────────────────────────────────
{prompt_txt}

── tools/base.py  (public API — signatures and docstrings only) ────
{tools_api}

══════════════════════════════════════════════════════════════════════
YOUR PROPOSAL
══════════════════════════════════════════════════════════════════════

Choose the smallest coherent proposal likely to improve macro-F1 on the next
val run. Prefer a single change. Use a short bundle only when the changes are
tightly coupled, for example creating a helper file and updating agent.py to
import it.

Output exactly this top-level shape:

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
    {{"kind": "delete_file", "details": {{"path": "agent/old.py"}}}}
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
- create_file: create a new file under agent/ or tools/.
- delete_file: delete a non-protected file under agent/ or tools/.

Additional constraints:
- "rationale" must name at least one specific task_id from the trajectories above.
- edit_prompt / edit_solve_loop: "content" is the COMPLETE new file, not a diff.
- edit_solve_loop: "content" must contain `def solve_task(task: dict) -> dict:`.
- add_tool / edit_tool: code must define exactly one function whose name matches "name".
- add_tool / edit_tool: the function may only import from this allowlist:
    re, ast, json, os, os.path, pathlib, subprocess, math,
    collections, itertools, functools, typing, datetime,
    hashlib, base64, textwrap. Nothing else.
- delete_tool must not delete core tools: llm_call, read_file, write_file,
  list_dir, run_bash, note.
- delete_file must not delete protected files: agent/agent.py,
  agent/prompt.txt, tools/base.py.
- create_file / delete_file paths must be relative and under agent/ or tools/.

Output only the JSON object now:"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _gateway_call(llm_handler, messages: list, model: str) -> Optional[str]:
    """Send one llm_request envelope through the orchestrator gateway."""
    envelope = {
        "_kind": "llm_request",
        "id": str(uuid.uuid4()),
        "messages": messages,
        "model": model,
        "response_format": {"type": "json_object"},
        "task_id": None,
        "purpose": "reflect",
        "step_in_task": 1,
    }
    try:
        resp = llm_handler(envelope)
    except Exception:
        return None
    if not isinstance(resp, dict) or not resp.get("ok"):
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
) -> Optional[dict]:
    """Produce a structured JSON proposal for the next generation.

    `trajectories` is the curated set the orchestrator decided to show
    (already includes per-task code). `score` is the full aggregate dict
    used to render the confusion matrix.
    """
    if llm_handler is None:
        return None

    gen_dir = Path(current_gen_dir)
    model_cfg = (config or {}).get("model", {})
    model = model_cfg.get("reflect_name") or model_cfg.get("name", "gpt-5.4-mini")

    try:
        agent_py = (gen_dir / "agent" / "agent.py").read_text()
        prompt_txt = (gen_dir / "agent" / "prompt.txt").read_text()
    except FileNotFoundError as e:
        raise ValueError(f"reflect: missing source file in {gen_dir}: {e}")
    tools_api = _extract_tools_api(gen_dir)

    messages = _build_messages(
        gen_idx=gen_idx,
        score=score or {},
        trajectories=trajectories or [],
        agent_py=agent_py,
        prompt_txt=prompt_txt,
        tools_api=tools_api,
    )

    content = _gateway_call(llm_handler, messages, model)
    if content is None:
        return None

    try:
        return _parse_proposal(content)
    except ValueError as first_err:
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": (
                "Your previous response was not valid. Output ONLY a single JSON "
                "object — no markdown fences, no prose. Output it now."
            )},
        ]
        retry_content = _gateway_call(llm_handler, retry_messages, model)
        if retry_content is None:
            raise ValueError(f"reflect: parse failed and retry call returned no content. "
                             f"first error: {first_err}")
        return _parse_proposal(retry_content)
