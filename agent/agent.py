"""gen_0 agent — ReAct loop solve_task() implementation.

Uses native tool calling to investigate code snippets via bounded static scans
and targeted reading before submitting a final verdict.
Future generations may extend the tool set or rewrite this loop entirely.
Hard rules for editing this file live in agent/AGENT.md.
"""

import json
import re
from pathlib import Path

from tools.base import llm_call, read_file, run_bash, static_scan, note


_PROMPT_PATH = Path(__file__).parent / "prompt.txt"
_MODEL = "gpt-5.4"
_MAX_STEPS = 10
_CWE_RE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
_LABEL_FIELD_RE = re.compile(
    r"\b(?:label|verdict)\s*[:=-]\s*(vulnerable|safe)\b",
    re.IGNORECASE,
)

# Tools the model may call. `finalize` is the harness sentinel and is NOT
# dispatched through _dispatch — it ends the loop.
_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from /agent (agent snapshot) or /work (scratch).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "static_scan",
            "description": (
                "Run a bounded static scan of the current task code. Returns "
                "{language, heuristic_findings, external}. Use at most once when "
                "manual review is uncertain; do not retry after timeout/unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "c_cpp", "unknown"],
                        "description": "Best-effort language hint for the current snippet.",
                    },
                    "run_external": {
                        "type": "boolean",
                        "description": "Whether to also run semgrep/bandit with a short timeout. Default false.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "External scanner timeout in seconds, capped by the tool.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note",
            "description": "Append a timestamped note to /work/notes.txt for reasoning scratch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": (
                "Submit your final verdict. Call this once you have enough evidence. "
                "Do not call any other tool after finalize."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["vulnerable", "safe"],
                        "description": "Security classification.",
                    },
                    "cwe": {
                        "type": "string",
                        "description": "CWE identifier when vulnerable, e.g. 'CWE-119' (optional).",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One or two sentences justifying your verdict.",
                    },
                },
                "required": ["label"],
            },
        },
    },
]


def _dispatch(name: str, args: dict, task_code: str = ""):
    """Call the named tool and return a JSON-serialisable result."""
    if name == "read_file":
        try:
            return read_file(args["path"])
        except Exception as e:
            return f"error: {e}"
    elif name == "run_bash":
        return run_bash(args.get("cmd", ""), timeout=args.get("timeout", 30))
    elif name == "static_scan":
        return static_scan(
            task_code,
            language=args.get("language"),
            run_external=bool(args.get("run_external", False)),
            timeout=args.get("timeout", 5),
        )
    elif name == "note":
        note(args.get("text", ""))
        return "noted"
    else:
        return f"unknown tool: {name}"


def _clean_label(value):
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    return label if label in ("vulnerable", "safe") else None


def _result_from_finalize(task_id, finalize_args: dict) -> dict:
    label = _clean_label(finalize_args.get("label"))
    return {
        "task_id": task_id,
        "label": label,
        "cwe": finalize_args.get("cwe"),
        "reasoning": finalize_args.get("reasoning"),
        "raw": json.dumps(finalize_args),
    }


def _coerce_text_verdict(content: str | None) -> dict | None:
    """Accept common text-only verdicts when the model forgets tool calling."""
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        label = _clean_label(parsed.get("label"))
        if label is not None:
            return {
                "label": label,
                "cwe": parsed.get("cwe"),
                "reasoning": parsed.get("reasoning") or text,
            }

    lowered = text.lower()
    if lowered.startswith("vulnerable"):
        label = "vulnerable"
    elif lowered.startswith("safe"):
        label = "safe"
    else:
        match = _LABEL_FIELD_RE.search(text[:120])
        label = match.group(1).lower() if match else None
    if label is None:
        return None

    cwe_match = _CWE_RE.search(text)
    return {
        "label": label,
        "cwe": cwe_match.group(0).upper() if cwe_match else None,
        "reasoning": text,
    }


def solve_task(task: dict) -> dict:
    code = task.get("code", "") or ""
    task_id = task.get("id")

    system_prompt = _PROMPT_PATH.read_text()
    user_content = (
        "Analyze the following code snippet for security vulnerabilities.\n\n"
        f"```\n{code}\n```\n\n"
        "Use the available tools to investigate, then call `finalize` with your verdict."
    )

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for _step in range(_MAX_STEPS):
        tool_choice = "auto"
        if _step == _MAX_STEPS - 1:
            tool_choice = {"type": "function", "function": {"name": "finalize"}}
        resp = llm_call(history, model=_MODEL, tools=_TOOL_SPECS, tool_choice=tool_choice)
        content = resp.get("content")
        tool_calls = resp.get("tool_calls")

        # Append assistant turn to history.
        assistant_msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        history.append(assistant_msg)

        if not tool_calls:
            coerced = _coerce_text_verdict(content)
            if coerced is not None:
                return _result_from_finalize(task_id, coerced)
            history.append({
                "role": "user",
                "content": (
                    "Use the finalize tool now. Do not answer in plain text; "
                    "submit exactly one label via finalize."
                ),
            })
            continue

        tool_result_msgs = []
        finalize_args = None

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                fn_args = {}

            if fn_name == "finalize":
                finalize_args = fn_args
                # Still need to add a tool result so the history is valid.
                tool_result_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "finalized",
                })
            else:
                result = _dispatch(fn_name, fn_args, task_code=code)
                tool_result_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        json.dumps(result) if not isinstance(result, str) else result
                    ),
                })

        history.extend(tool_result_msgs)

        if finalize_args is not None:
            return _result_from_finalize(task_id, finalize_args)

    return {"task_id": task_id, "label": None, "error": "max_steps exceeded"}
