"""Run an agent over a split and compute scores.

Thin wrapper around `orchestrator.run_candidate`: loads the cached split,
delegates execution to the orchestrator's Docker mechanism (which proxies
LLM calls back through the host so cost tracking stays ground-truth), and
returns aggregated `accuracy`, `macro_f1`, and `per_task` scores plus
token totals.

CLI:
    python -m eval.benchmark --gen artifacts/gen_0 --split val [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def run_benchmark(
    gen_dir: str,
    split_name: str,
    config_path: str = "config.yaml",
    limit: int | None = None,
    log_event=None,
    llm_handler=None,
) -> dict:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from eval.dataset import load_split
    from orchestrator import run_candidate, load_config

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    config = load_config(cfg_path)

    split = load_split(split_name, project_root=project_root, config=config)
    if limit is not None:
        split = split[:limit]

    gen_path = Path(gen_dir)
    if not gen_path.is_absolute():
        gen_path = project_root / gen_path
    return run_candidate(gen_path, split, config, llm_handler=llm_handler,
                         log_event=log_event)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", required=True, help="path to gen_N directory")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of tasks (for quick checks)")
    parser.add_argument("--run-id", default=None,
                        help="if set, write a log.jsonl under artifacts/runs/<run-id>/")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    log_event = None
    llm_handler = None
    if args.run_id:
        from orchestrator import RunLogger, LLMCallLogger, make_llm_handler, load_config
        config = load_config(project_root / args.config)
        run_dir = project_root / config["paths"]["runs_dir"] / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(project_root / args.config, run_dir / "config.snapshot.yaml")
        logger = RunLogger(run_dir)
        logger("benchmark_start", gen=args.gen, split=args.split, limit=args.limit)
        log_event = logger

        llm_log = LLMCallLogger(run_dir)
        try:
            from dotenv import load_dotenv
            load_dotenv(project_root / ".env")
        except ImportError:
            pass
        try:
            from openai import OpenAI
            llm_handler = make_llm_handler(OpenAI(), config["model"]["name"],
                                           llm_logger=llm_log)
        except Exception:
            pass
    else:
        # Load .env so the orchestrator's OpenAI client picks up the key.
        try:
            from dotenv import load_dotenv
            load_dotenv(project_root / ".env")
        except ImportError:
            pass

    result = run_benchmark(args.gen, args.split, args.config,
                           limit=args.limit, log_event=log_event,
                           llm_handler=llm_handler)

    if log_event:
        log_event("benchmark_complete",
                  macro_f1=result["macro_f1"],
                  accuracy=result["accuracy"],
                  n_tasks=result["n_tasks"],
                  n_errors=result["n_errors"],
                  cost_usd=result["cost_usd"],
                  tokens=result["usage"])

    print(json.dumps({
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
        "n_tasks": result["n_tasks"],
        "n_errors": result["n_errors"],
        "cost_usd": result["cost_usd"],
        "usage": result["usage"],
        "per_task": result["per_task"],
    }, indent=2))


if __name__ == "__main__":
    main()
