"""Quick smoke check — does the candidate import cleanly and run on one task?

Used by the orchestrator before the full validation gate so we can fail fast
on candidates that are broken at the syntactic or import level, without
spending API budget on a full val run.
"""


def smoke_check(gen_dir: str) -> bool:
    pass
