"""Side-by-side Gradio chat UI.

Two columns, two assistants, one input box. Sends the user's message to
whichever assistants are enabled and shows both replies for direct
comparison. Per-turn latency is shown under each reply.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import gradio as gr  # noqa: E402

from assistants import build_assistant  # noqa: E402
from assistants.base import BaseAssistant  # noqa: E402

_OSS: Optional[BaseAssistant] = None
_FRONTIER: Optional[BaseAssistant] = None


def _get_oss() -> BaseAssistant:
    global _OSS
    if _OSS is None:
        _OSS = build_assistant("oss")
    return _OSS


def _get_frontier() -> BaseAssistant:
    global _FRONTIER
    if _FRONTIER is None:
        _FRONTIER = build_assistant("frontier")
    return _FRONTIER


def _reply(asst: BaseAssistant, user_msg: str) -> str:
    res = asst.chat(user_msg)
    tools = ""
    if res.tool_calls:
        tools = "  \n_tools: " + ", ".join(tc["name"] for tc in res.tool_calls) + "_"
    return f"{res.text}\n\n_({res.latency_s}s){tools}_"


def respond(
    user_msg: str,
    oss_history: List[Tuple[str, str]],
    frontier_history: List[Tuple[str, str]],
    use_oss: bool,
    use_frontier: bool,
):
    if not user_msg.strip():
        return oss_history, frontier_history, ""

    if use_oss:
        try:
            text = _reply(_get_oss(), user_msg)
        except Exception as e:  # noqa: BLE001
            text = f"[OSS error: {e}]"
        oss_history = oss_history + [(user_msg, text)]

    if use_frontier:
        try:
            text = _reply(_get_frontier(), user_msg)
        except Exception as e:  # noqa: BLE001
            text = f"[Frontier error: {e}]"
        frontier_history = frontier_history + [(user_msg, text)]

    return oss_history, frontier_history, ""


def reset_chats():
    if _OSS is not None:
        _OSS.reset()
    if _FRONTIER is not None:
        _FRONTIER.reset()
    return [], []


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OSS vs Frontier — Assistant comparison") as demo:
        gr.Markdown(
            "# OSS vs Frontier — Personal Assistant\n"
            "Both assistants share the same memory, tools, and guardrails. "
            "Only the model differs. Ask anything; they answer side-by-side."
        )
        with gr.Row():
            use_oss = gr.Checkbox(value=True, label=f"OSS  ({os.environ.get('OSS_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')})")
            use_frontier = gr.Checkbox(
                value=bool(os.environ.get("GROQ_API_KEY")),
                label=f"Frontier  ({os.environ.get('FRONTIER_MODEL', 'llama-3.3-70b-versatile')})",
            )
        with gr.Row():
            oss_chat = gr.Chatbot(label="OSS", height=460)
            frontier_chat = gr.Chatbot(label="Frontier", height=460)

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask the assistants anything…",
                show_label=False,
                scale=8,
            )
            send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Reset", scale=1)

        send.click(
            respond,
            [msg, oss_chat, frontier_chat, use_oss, use_frontier],
            [oss_chat, frontier_chat, msg],
        )
        msg.submit(
            respond,
            [msg, oss_chat, frontier_chat, use_oss, use_frontier],
            [oss_chat, frontier_chat, msg],
        )
        clear.click(reset_chats, None, [oss_chat, frontier_chat])

        gr.Markdown(
            "_Tip: try a factual question, then a multi-turn follow-up "
            "(\"and what about X?\") to see the memory in action._"
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
