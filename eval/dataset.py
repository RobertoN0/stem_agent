"""PrimeVul dataset loading and split caching.

Builds train/val/test JSONL splits from the PrimeVul source (or a fallback
if PrimeVul isn't accessible), balanced 50/50 between vulnerable and safe
examples. Cached under data/primevul/ as `{train,val,test}.jsonl`.

Each task is `{"id": ..., "code": ..., "label": "vulnerable"|"safe", "cwe": ...}`.

Run as `python -m eval.dataset` to (re)build the cache. Idempotent — skips
work if files already exist unless --force.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import yaml


# --- adapters --------------------------------------------------------------
# Each adapter normalizes a row from a specific HF dataset into the unified
# task schema. Adapters are tried in order; first repo that loads wins.

def _primevul_adapter(row, idx):
    label = "vulnerable" if int(row.get("target", 0)) == 1 else "safe"
    return {
        "id": f"primevul_{row.get('idx', idx)}",
        "code": row.get("func", "") or "",
        "label": label,
        "cwe": None,
        "source": "colin/PrimeVul",
    }


def _bigvul_pair_adapter(row, idx):
    """BigVul stores paired (func_before=vulnerable, func_after=fixed). One row
    yields two tasks. Returns a list."""
    cwe = row.get("CWE ID") or None
    base_id = row.get("commit_id") or str(idx)
    out = []
    if row.get("func_before"):
        out.append({
            "id": f"bigvul_{base_id}_vuln",
            "code": row["func_before"],
            "label": "vulnerable",
            "cwe": cwe,
            "source": "bstee615/bigvul",
        })
    if row.get("func_after"):
        out.append({
            "id": f"bigvul_{base_id}_safe",
            "code": row["func_after"],
            "label": "safe",
            "cwe": None,
            "source": "bstee615/bigvul",
        })
    return out


def _devign_adapter(row, idx):
    label = "vulnerable" if int(row.get("target", 0)) == 1 else "safe"
    return {
        "id": f"devign_{row.get('id', idx)}",
        "code": row.get("func", "") or row.get("func_clean", "") or "",
        "label": label,
        "cwe": None,
        "source": "DetectVul/devign",
    }


# repo_name, adapter, returns_list
_FALLBACK_CHAIN = [
    ("colin/PrimeVul", _primevul_adapter, False),
    ("bstee615/bigvul", _bigvul_pair_adapter, True),
    ("DetectVul/devign", _devign_adapter, False),
]


# --- helpers ---------------------------------------------------------------

_MAX_CODE_CHARS = 6000


def _truncate(code: str) -> str:
    if len(code) <= _MAX_CODE_CHARS:
        return code
    return code[:_MAX_CODE_CHARS] + "\n/* ... [truncated] ... */"


def _stream_examples(repo: str, adapter, returns_list: bool, split: str):
    """Yield normalized task dicts from a HF dataset split."""
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    for idx, row in enumerate(ds):
        out = adapter(row, idx)
        if returns_list:
            for t in out:
                if t["code"]:
                    t["code"] = _truncate(t["code"])
                    yield t
        else:
            if out["code"]:
                out["code"] = _truncate(out["code"])
                yield out


def _collect_balanced(stream, n_per_class: int, seen_ids: set) -> list:
    """Pull from stream until we have n_per_class of each label, skipping
    any task whose id is already in seen_ids."""
    bucket = {"vulnerable": [], "safe": []}
    for task in stream:
        if task["id"] in seen_ids:
            continue
        b = bucket[task["label"]]
        if len(b) < n_per_class:
            b.append(task)
        if len(bucket["vulnerable"]) >= n_per_class and len(bucket["safe"]) >= n_per_class:
            break
    return bucket["vulnerable"] + bucket["safe"]


def _try_source(repo: str, adapter, returns_list: bool, splits_cfg: dict, rng: random.Random) -> dict | None:
    """Build train/val/test from one source. Returns dict of split→list, or
    None if the source can't be loaded."""
    try:
        # Probe by pulling one example.
        probe = next(_stream_examples(repo, adapter, returns_list, "train"))
        if not probe.get("code"):
            return None
    except Exception as e:
        print(f"  [{repo}] probe failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None

    print(f"  [{repo}] usable; building splits...", file=sys.stderr)
    out = {}
    seen = set()
    for split_name, hf_split in (("train", "train"), ("val", "validation"), ("test", "test")):
        n = splits_cfg[split_name]
        if n % 2 != 0:
            n_per_class = (n + 1) // 2  # round up; we'll trim
        else:
            n_per_class = n // 2
        try:
            tasks = _collect_balanced(
                _stream_examples(repo, adapter, returns_list, hf_split),
                n_per_class, seen,
            )
        except Exception as e:
            # Some sources don't have a "validation" split — fall back to
            # carving extra rows out of train.
            print(f"  [{repo}] split={hf_split} unavailable ({type(e).__name__}); "
                  f"borrowing from train", file=sys.stderr)
            tasks = _collect_balanced(
                _stream_examples(repo, adapter, returns_list, "train"),
                n_per_class, seen,
            )
        if len(tasks) < n:
            print(f"  [{repo}] split={split_name} short: got {len(tasks)} of {n}",
                  file=sys.stderr)
            return None
        rng.shuffle(tasks)
        tasks = tasks[:n]
        for t in tasks:
            seen.add(t["id"])
        out[split_name] = tasks
    return out


def _write_split(path: Path, tasks: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


# --- public API ------------------------------------------------------------

def build_splits(config: dict, project_root: Path | None = None,
                 force: bool = False) -> dict:
    """Build (or load) train/val/test splits. Returns metadata about the
    chosen source and split sizes."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / config["paths"]["data_dir"]
    source_marker = data_dir / "_source.json"
    splits_cfg = config["splits"]

    paths = {s: data_dir / f"{s}.jsonl" for s in ("train", "val", "test")}
    if not force and all(p.exists() for p in paths.values()) and source_marker.exists():
        with open(source_marker) as f:
            return json.load(f)

    rng = random.Random(42)
    print("Building splits — trying sources in priority order:", file=sys.stderr)
    for repo, adapter, returns_list in _FALLBACK_CHAIN:
        print(f"  trying {repo}...", file=sys.stderr)
        result = _try_source(repo, adapter, returns_list, splits_cfg, rng)
        if result is None:
            continue
        for split_name, tasks in result.items():
            _write_split(paths[split_name], tasks)
        meta = {
            "source": repo,
            "sizes": {s: len(t) for s, t in result.items()},
            "balance": "50/50 vulnerable/safe per split",
            "seed": 42,
        }
        with open(source_marker, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Built splits from {repo}: {meta['sizes']}", file=sys.stderr)
        return meta

    raise RuntimeError(
        "No dataset source available. Tried: "
        + ", ".join(repo for repo, _, _ in _FALLBACK_CHAIN)
    )


def load_split(split_name: str, project_root: Path | None = None,
               config: dict | None = None) -> list:
    """Read a cached split and apply the configured runtime split cap."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    if config is None:
        with open(project_root / "config.yaml") as f:
            config = yaml.safe_load(f)
    p = project_root / config["paths"]["data_dir"] / f"{split_name}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}")
    tasks = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    try:
        limit = int((config.get("splits") or {}).get(split_name))
    except (TypeError, ValueError, AttributeError):
        limit = None
    if limit is not None and limit > 0:
        tasks = tasks[:limit]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if cache exists")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config) as f:
        config = yaml.safe_load(f)
    meta = build_splits(config, project_root, force=args.force)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
