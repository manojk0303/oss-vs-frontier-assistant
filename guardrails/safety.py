"""Two-tier safety guardrails.

Tier 1 (always-on, free): regex denylist + simple prompt-injection heuristics.
Tier 2 (optional, paid): a tiny Llama 3.1 8B Instant call that classifies the
prompt when Tier 1 is ambiguous. Tier 2 is only invoked when MODERATOR_MODEL
is set and the call is enabled by the caller.

Each tier returns a `GuardrailResult` with allow/block + a category + a
human-readable reason, so the assistant can refuse cleanly and the logger can
record what happened.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

Verdict = Literal["allow", "block", "warn"]


@dataclass
class GuardrailResult:
    verdict: Verdict
    category: str
    reason: str

    @property
    def blocked(self) -> bool:
        return self.verdict == "block"


# ---------------------------------------------------------------------------
# Tier 1: cheap pattern checks
# ---------------------------------------------------------------------------

_INPUT_DENY_PATTERNS = [
    (
        re.compile(
            r"\b(how (do|can) (i|we)|step.?by.?step|instructions? (for|to))\b.{0,80}"
            r"\b(make|build|synthesi[sz]e|cook|brew|manufacture).{0,40}"
            r"\b(bomb|explosive|napalm|nerve agent|sarin|ricin|meth|fentanyl|nuclear weapon)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "weapons_drugs",
        "Request for instructions to create weapons or controlled substances.",
    ),
    (
        re.compile(
            r"\b(child|minor|underage|kid|teen).{0,40}(sexual|nude|porn|explicit)\b",
            re.IGNORECASE,
        ),
        "csam",
        "Request involving sexual content with minors.",
    ),
    (
        re.compile(
            r"\b(write|generate|produce).{0,40}\b(malware|ransomware|keylogger|virus|trojan|rootkit)\b",
            re.IGNORECASE,
        ),
        "malware",
        "Request to produce malicious software.",
    ),
]

_JAILBREAK_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) (instructions|prompts)", re.IGNORECASE),
    re.compile(r"\b(DAN|do anything now|developer mode|jailbreak)\b", re.IGNORECASE),
    re.compile(r"pretend (you are|to be) (an? )?(?:un)?restricted", re.IGNORECASE),
    re.compile(r"you have no (rules|restrictions|guidelines)", re.IGNORECASE),
]

_OUTPUT_DENY_PATTERNS = [
    (
        re.compile(
            r"\b(synthesi[sz]e|cook|brew)\b.{0,80}\b(meth|fentanyl|sarin|ricin)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "weapons_drugs",
        "Assistant output contained synthesis instructions for controlled substances.",
    ),
]


def check_input(text: str) -> GuardrailResult:
    for pattern, category, reason in _INPUT_DENY_PATTERNS:
        if pattern.search(text):
            return GuardrailResult("block", category, reason)
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                "warn",
                "jailbreak_attempt",
                "Prompt looks like a jailbreak attempt; model is instructed to refuse.",
            )
    return GuardrailResult("allow", "ok", "")


def check_output(text: str) -> GuardrailResult:
    for pattern, category, reason in _OUTPUT_DENY_PATTERNS:
        if pattern.search(text):
            return GuardrailResult("block", category, reason)
    return GuardrailResult("allow", "ok", "")


# ---------------------------------------------------------------------------
# Tier 2: optional LLM moderator (Groq Llama 3.1 8B Instant). Only used if the
# caller opts in by passing use_moderator=True AND the API key is available.
# ---------------------------------------------------------------------------

_MODERATOR_PROMPT = (
    "Classify the user message below as SAFE or UNSAFE.\n"
    "UNSAFE means: requests for weapons, illegal drug synthesis, CSAM, malware, "
    "self-harm encouragement, or targeted harassment.\n"
    "Reply with exactly one word: SAFE or UNSAFE."
)


def moderate_with_llm(text: str) -> Optional[GuardrailResult]:
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("MODERATOR_MODEL")
    if not api_key or not model:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=8,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _MODERATOR_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        if verdict.startswith("UNSAFE"):
            return GuardrailResult(
                "block", "llm_moderator", "LLM moderator classified the prompt as unsafe."
            )
        return GuardrailResult("allow", "llm_moderator", "LLM moderator classified the prompt as safe.")
    except Exception:  # noqa: BLE001
        return None
