"""OSS backend: Qwen2.5-0.5B-Instruct via Hugging Face transformers.

Tool use is done with a stricter regex-based parser rather than the
HF/transformers tool-template, because the 0.5B model is not reliable enough
at perfect JSON to use the template directly. We give it a short prompt of
the tool specs and parse the first ```tool { ... } ``` block we see.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Tuple

from assistants.base import BaseAssistant
from tools.tools import run_tool

_TOOL_BLOCK = re.compile(r"```tool\s*(\{.*?\})\s*```", re.DOTALL)


def _tools_prompt(specs: list[dict]) -> str:
    lines = ["You can optionally call ONE tool per turn by emitting a fenced block:"]
    lines.append("```tool\n{\"name\": \"<tool>\", \"arguments\": { ... }}\n```")
    lines.append("Available tools:")
    for s in specs:
        params = ", ".join(s["input_schema"].get("properties", {}).keys()) or "(no args)"
        lines.append(f"- {s['name']}({params}): {s['description']}")
    lines.append(
        "Only use a tool if it materially helps. If no tool is needed, answer directly."
    )
    return "\n".join(lines)


class OSSAssistant(BaseAssistant):
    name = "oss"

    def __init__(self, model: str | None = None, device: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy import
        import torch

        self.model_id = model or os.environ.get("OSS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        device = device or os.environ.get("OSS_DEVICE")
        if not device:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()

        # Bake the tools list into the system prompt once.
        self._tools_help = _tools_prompt(self.tool_specs)

    # ------------------------------------------------------------------ generate

    def generate(self, messages: list[dict]) -> Tuple[str, List[dict], dict]:
        # Augment the system prompt with the tool instructions.
        messages = list(messages)
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + "\n\n" + self._tools_help,
        }

        tool_calls_made: list[dict] = []

        # Single-step tool use: generate once; if a tool block is detected,
        # run the tool and generate a final answer with the observation.
        first = self._chat_complete(messages, max_new_tokens=384)
        tool_call = self._maybe_parse_tool(first)
        if tool_call is not None:
            name, args = tool_call
            result = run_tool(name, **args)
            tool_calls_made.append({"name": name, "input": args, "result": result})
            messages.append({"role": "assistant", "content": first})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation from {name}: {json.dumps(result)[:1200]}\n"
                        "Use this observation to write the final answer. Do not call another tool."
                    ),
                }
            )
            final = self._chat_complete(messages, max_new_tokens=384)
            # Strip any stray tool block in the final
            final = _TOOL_BLOCK.sub("", final).strip() or first
            return final, tool_calls_made, {"prompt_tokens": 0, "completion_tokens": 0}

        return first.strip() or "(no response)", tool_calls_made, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    # ----------------------------------------------------------------- helpers

    def _chat_complete(self, messages: list[dict], max_new_tokens: int = 384) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @staticmethod
    def _maybe_parse_tool(text: str):
        m = _TOOL_BLOCK.search(text)
        if not m:
            return None
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name")
            args = payload.get("arguments", {}) or {}
            if isinstance(name, str) and isinstance(args, dict):
                return name, args
        except json.JSONDecodeError:
            return None
        return None
