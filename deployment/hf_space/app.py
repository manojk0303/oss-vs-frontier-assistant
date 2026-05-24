"""HF Spaces entry point.

This file lives at the Space root. The supporting packages (`assistants/`,
`memory/`, `tools/`, `guardrails/`, `observability/`) need to be copied
alongside it. We disable the Frontier backend by default — Spaces can't safely
hold an Anthropic API key for a public demo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add the repo root to sys.path so the support packages are importable when
# this file is copied into a Space.
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

# Disable the frontier path for the public demo unless GROQ_API_KEY is set
# as a Space secret. The OSS demo runs without any key.
# os.environ.pop("GROQ_API_KEY", None)

from app.gradio_app import build_ui  # noqa: E402

demo = build_ui()
demo.queue().launch()
