"""Tiny structured event logger.

Writes one JSON object per line to LOG_PATH. Cheap, grep-friendly, and
sufficient for a take-home; in production you'd swap in OpenTelemetry +
a hosted backend (Honeycomb, Datadog, etc.).
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Event:
    event: str
    backend: str
    session_id: str
    ts: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


class EventLogger:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.environ.get("LOG_PATH", "logs/events.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, backend: str, session_id: str, **data: Any) -> None:
        evt = Event(event=event, backend=backend, session_id=session_id, data=data)
        line = json.dumps(asdict(evt), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


_GLOBAL: Optional[EventLogger] = None


def get_logger() -> EventLogger:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = EventLogger()
    return _GLOBAL
