"""Project-root conftest: makes orchestrator.py importable from tests/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
