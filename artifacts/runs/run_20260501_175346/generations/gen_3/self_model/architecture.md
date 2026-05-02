# Architecture Self-Model

This file is mutable memory for the evolving agent's understanding of the
system it lives in.

Current seed understanding:
- Runtime classification happens in `agent/agent.py`.
- The system prompt comes from `agent/prompt.txt`.
- Learned task policy can live in `knowledge/strategy.md`.
- Helper functions live in `tools/base.py`, but core RPC and filesystem
  functions are protected by the host mutation boundary.
- Reflection may update this file when train feedback reveals a better model
  of the agent's own workflow.
- Before proposing changes, reflection is expected to inspect this self-model
  or a filtered immutable repo contract so proposals are grounded in the
  architecture, not only in task examples.
