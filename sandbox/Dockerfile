# Sandbox image for running candidate agents.
#
# Resource limits (memory, CPU, network) and bind mounts (/agent, /work)
# are applied by the orchestrator at `docker run` time — they cannot be
# enforced from inside the Dockerfile.
#
# Build from the repo root, not from sandbox/:
#   docker build -f sandbox/Dockerfile -t stem-agent-sandbox .

FROM python:3.11-slim

WORKDIR /work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Static-analysis tools the agent may wrap via add_tool. Pre-install so
# subprocess-based tools work without network at run time.
RUN pip install --no-cache-dir bandit pyflakes semgrep

COPY sandbox/runner.py /runner.py

ENTRYPOINT ["python", "/runner.py"]
