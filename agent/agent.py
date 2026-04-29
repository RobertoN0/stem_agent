"""gen_0 agent — ReAct loop solve_task() implementation.

Uses native tool calling to investigate code snippets via static analysis
(bandit, semgrep) and targeted reading before submitting a final verdict.
Future generations may extend the tool set or rewrite this loop entirely.
Hard rules for editing this file live in agent/AGENT.md.
"""

import json
from pathlib import Path

from tools.base import llm_call, read_file, run_bash, note


_PROMPT_PATH = Path(__file__).parent / "prompt.txt"
_MODEL = "gpt-5.4-mini"
_MAX_STEPS = 10

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
            "name": "run_bash",
            "description": (
                "Run a shell command in /work. Returns {stdout, stderr, returncode}. "
                "Use for static analysis: write the code to /work/snippet.c then run "
                "semgrep or bandit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command."},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30).",
                    },
                },
                "required": ["cmd"],
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


def _dispatch(name: str, args: dict):
    """Call the named tool and return a JSON-serialisable result."""
    if name == "read_file":
        try:
            return read_file(args["path"])
        except Exception as e:
            return f"error: {e}"
    elif name == "run_bash":
        return run_bash(args.get("cmd", ""), timeout=args.get("timeout", 30))
    elif name == "note":
        note(args.get("text", ""))
        return "noted"
    else:
        return f"unknown tool: {name}"


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
        resp = llm_call(history, model=_MODEL, tools=_TOOL_SPECS, tool_choice="auto")
        content = resp.get("content")
        tool_calls = resp.get("tool_calls")

        # Append assistant turn to history.
        assistant_msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        history.append(assistant_msg)

        if not tool_calls:
            # Text-only turn — model is reasoning out loud; continue loop.
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
                result = _dispatch(fn_name, fn_args)
                tool_result_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": (
                        json.dumps(result) if not isinstance(result, str) else result
                    ),
                })

        history.extend(tool_result_msgs)

        if finalize_args is not None:
            label = finalize_args.get("label")
            if isinstance(label, str):
                label = label.strip().lower()
                if label not in ("vulnerable", "safe"):
                    label = None
            return {
                "task_id": task_id,
                "label": label,
                "cwe": finalize_args.get("cwe"),
                "reasoning": finalize_args.get("reasoning"),
                "raw": json.dumps(finalize_args),
            }

    return {"task_id": task_id, "label": None, "error": "max_steps exceeded"}
