"""Tiny factory so callers don't import heavy deps unless they need them."""
from __future__ import annotations

from assistants.base import BaseAssistant


def build_assistant(backend: str, **kwargs) -> BaseAssistant:
    backend = backend.lower()
    if backend in ("oss", "qwen", "hf", "local"):
        from assistants.oss_assistant import OSSAssistant

        return OSSAssistant(**kwargs)
    if backend in ("frontier", "claude", "anthropic", "api"):
        from assistants.frontier_assistant import FrontierAssistant

        return FrontierAssistant(**kwargs)
    raise ValueError(f"unknown backend: {backend}")
