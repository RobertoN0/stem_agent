"""Inspect and filter llm_calls.jsonl produced by the orchestrator gateway.

Each line is a JSON record with fields:
    call_id, timestamp_iso, purpose, generation, task_id, step_in_task,
    request, response, usage, duration_ms, error

CLI:
    python -m eval.inspect_llm_calls --log artifacts/runs/<run_id>/llm_calls.jsonl
    python -m eval.inspect_llm_calls --log ... --task-id primevul_42
    python -m eval.inspect_llm_calls --log ... --generation 1 --error-only
    python -m eval.inspect_llm_calls --log ... --summary
"""

import argparse
import json
from pathlib import Path
from typing import Iterator, Optional


def iter_calls(
    path,
    *,
    task_id: Optional[str] = None,
    purpose: Optional[str] = None,
    generation: Optional[int] = None,
    error_only: bool = False,
) -> Iterator[dict]:
    """Yield call records from an llm_calls.jsonl file, with optional filters."""
    p = Path(path)
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if task_id is not None and rec.get("task_id") != task_id:
                continue
            if purpose is not None and rec.get("purpose") != purpose:
                continue
            if generation is not None and rec.get("generation") != generation:
                continue
            if error_only and rec.get("error") is None:
                continue
            yield rec


def summarize(path) -> dict:
    """Return aggregate stats over an llm_calls.jsonl file."""
    n_calls = 0
    n_errors = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_duration_ms = 0
    by_generation: dict = {}
    by_purpose: dict = {}

    for rec in iter_calls(path):
        n_calls += 1
        if rec.get("error"):
            n_errors += 1
        u = rec.get("usage") or {}
        prompt_tokens += u.get("prompt_tokens", 0)
        completion_tokens += u.get("completion_tokens", 0)
        total_duration_ms += rec.get("duration_ms", 0)

        gen = rec.get("generation")
        by_generation[gen] = by_generation.get(gen, 0) + 1

        pur = rec.get("purpose") or "unknown"
        by_purpose[pur] = by_purpose.get(pur, 0) + 1

    return {
        "n_calls": n_calls,
        "n_errors": n_errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "avg_duration_ms": round(total_duration_ms / n_calls, 1) if n_calls else 0,
        "by_generation": by_generation,
        "by_purpose": by_purpose,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect llm_calls.jsonl from a run directory."
    )
    parser.add_argument("--log", required=True, help="path to llm_calls.jsonl")
    parser.add_argument("--task-id", default=None, help="filter by task_id")
    parser.add_argument("--purpose", default=None, help="filter by purpose")
    parser.add_argument("--generation", type=int, default=None,
                        help="filter by generation number")
    parser.add_argument("--error-only", action="store_true",
                        help="show only failed calls")
    parser.add_argument("--summary", action="store_true",
                        help="print aggregate stats instead of individual records")
    parser.add_argument("--limit", type=int, default=None,
                        help="max records to print")
    args = parser.parse_args()

    if args.summary:
        print(json.dumps(summarize(args.log), indent=2))
        return

    count = 0
    for rec in iter_calls(
        args.log,
        task_id=args.task_id,
        purpose=args.purpose,
        generation=args.generation,
        error_only=args.error_only,
    ):
        print(json.dumps(rec, indent=2))
        count += 1
        if args.limit is not None and count >= args.limit:
            break


if __name__ == "__main__":
    main()
