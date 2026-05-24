"""Short-term conversational memory.

Keeps the last `max_turns` user/assistant exchanges verbatim, plus a one-line
rolling summary that gets refreshed every `summary_every` turns. The summary
preserves context that has aged out of the sliding window without growing the
prompt unboundedly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationMemory:
    max_turns: int = 12
    summary_every: int = 6
    turns: List[Turn] = field(default_factory=list)
    summary: str = ""
    _turns_since_summary: int = 0

    def add_user(self, content: str) -> None:
        self.turns.append(Turn("user", content))

    def add_assistant(self, content: str) -> None:
        self.turns.append(Turn("assistant", content))
        self._turns_since_summary += 1
        self._evict()

    def _evict(self) -> None:
        if len(self.turns) > self.max_turns * 2:
            # Drop oldest pair
            self.turns = self.turns[-self.max_turns * 2 :]

    def maybe_summarize(self, summarizer: Callable[[str], str]) -> None:
        """Refresh the rolling summary if it's time."""
        if self._turns_since_summary < self.summary_every:
            return
        transcript = "\n".join(f"{t.role}: {t.content}" for t in self.turns)
        try:
            self.summary = summarizer(transcript)
        except Exception:
            # Summary is a nice-to-have; never block the main loop on it
            pass
        self._turns_since_summary = 0

    def render_messages(self, system_prompt: str) -> List[dict]:
        """Render to chat-completions-style messages."""
        system = system_prompt
        if self.summary:
            system = f"{system}\n\nConversation so far (summary): {self.summary}"
        msgs = [{"role": "system", "content": system}]
        for t in self.turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs

    def reset(self) -> None:
        self.turns.clear()
        self.summary = ""
        self._turns_since_summary = 0
