"""Frontier-model backend: Llama 3.3 70B via the Groq API.

Groq exposes an OpenAI-compatible chat-completions endpoint with tool calling.
We adapt our shared `tool_specs()` (Anthropic-style) into OpenAI-style tool
schemas at the call boundary so the rest of the codebase stays uniform.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Tuple

from assistants.base import BaseAssistant
from tools.tools import run_tool

# Llama 3 sometimes prints its internal tool-call template as plain text
# instead of using the structured tool_calls field. Two patterns:
#   <function=NAME>{...args...}</function>
#   <|python_tag|>{"name": ..., "parameters": ...}
# Strip them from the final user-facing text; parse them when Groq rejects
# the call with 'tool_use_failed'.
_LEAKED_TOOL_CALL = re.compile(
    r"<function=[^>]*>.*?(?:</function>|$)|<\|python_tag\|>.*?(?=\n\n|$)",
    re.DOTALL,
)

_LEAKED_PARSE = re.compile(
    r"<function=([a-zA-Z_][\w]*)>\s*(\{.*?\})\s*</function>", re.DOTALL
)


def _to_openai_tools(specs: list[dict]) -> list[dict]:
    out = []
    for s in specs:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
        )
    return out


class FrontierAssistant(BaseAssistant):
    name = "frontier"

    def __init__(self, model: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        from groq import Groq  # local import so the OSS path doesn't need the dep

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FrontierAssistant requires GROQ_API_KEY in the environment."
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.environ.get("FRONTIER_MODEL", "llama-3.3-70b-versatile")
        self._openai_tools = _to_openai_tools(self.tool_specs)

    def generate(self, messages: list[dict]) -> Tuple[str, List[dict], dict]:
        # Convert our internal {role, content} list straight to OpenAI shape;
        # the first message is the system prompt.
        chat: list[dict] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        tool_calls_made: list[dict] = []
        prompt_tokens = 0
        completion_tokens = 0
        max_rounds = 5
        # Once any tool has run (either via the API or via leak recovery), the
        # next round must produce a clean final answer — otherwise Llama tends
        # to loop on more tool calls and Groq starts rejecting them.
        force_final = False

        for round_idx in range(max_rounds):
            on_last_round = round_idx == max_rounds - 1
            use_tools = not (force_final or on_last_round)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat,
                    tools=self._openai_tools if use_tools else None,
                    tool_choice="auto" if use_tools else "none",
                    max_tokens=1024,
                    temperature=0.4,
                )
            except Exception as e:  # noqa: BLE001
                # Groq returns 400 'tool_use_failed' when Llama emits its
                # internal tool-call template as text. Salvage by parsing the
                # leaked call from the error and executing it ourselves; if
                # that fails too, fall back to a tool-free retry.
                handled = self._handle_leaked_call(e, chat, tool_calls_made)
                if handled:
                    force_final = True
                    continue
                if use_tools:
                    chat.append(
                        {
                            "role": "user",
                            "content": "Please answer in plain text without calling any tool.",
                        }
                    )
                    force_final = True
                    continue
                raise
            usage = getattr(resp, "usage", None)
            if usage is not None:
                prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls:
                # Echo the assistant turn (with the tool_calls payload) back into chat
                chat.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = run_tool(tc.function.name, **args)
                    tool_calls_made.append(
                        {"name": tc.function.name, "input": args, "result": result}
                    )
                    chat.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": json.dumps(result)[:2000],
                        }
                    )
                force_final = True
                continue

            text = _LEAKED_TOOL_CALL.sub("", msg.content or "").strip() or "(no response)"
            return text, tool_calls_made, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        return (
            f"(stopped after {max_rounds} tool rounds without a final answer)",
            tool_calls_made,
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )

    def _handle_leaked_call(
        self, err: Exception, chat: list[dict], tool_calls_made: list[dict]
    ) -> bool:
        """Try to recover from Groq's 'tool_use_failed' (Llama emitting a
        function-call template as text). Returns True if recovered."""
        text = ""
        body = getattr(err, "body", None)
        if isinstance(body, dict):
            text = body.get("error", {}).get("failed_generation", "") or ""
        if not text:
            text = str(err)
        m = _LEAKED_PARSE.search(text)
        if not m:
            return False
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            return False
        result = run_tool(name, **args)
        tool_calls_made.append({"name": name, "input": args, "result": result})
        # Push the salvaged call into the chat history as a normal assistant
        # turn + observation, so the next loop iteration writes a clean answer.
        chat.append(
            {
                "role": "assistant",
                "content": f"(Calling tool {name} with {args})",
            }
        )
        chat.append(
            {
                "role": "user",
                "content": (
                    f"Observation from {name}: {json.dumps(result)[:1500]}\n"
                    "Use this observation to write the final answer in plain text. "
                    "Do not call another tool."
                ),
            }
        )
        return True

    def _summarize(self, transcript: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the conversation below in one sentence, focused on facts and open questions.",
                    },
                    {"role": "user", "content": transcript[:4000]},
                ],
                max_tokens=120,
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            return super()._summarize(transcript)
