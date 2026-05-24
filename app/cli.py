"""Minimal CLI chat. Useful for smoke testing without spinning up Gradio."""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from assistants import build_assistant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the assistant from the CLI.")
    parser.add_argument(
        "--backend",
        default="frontier",
        choices=["oss", "frontier"],
        help="Which backend to use.",
    )
    args = parser.parse_args()

    print(f"[loading {args.backend}…]", file=sys.stderr)
    asst = build_assistant(args.backend)
    print(f"[ready — type 'reset' to clear memory, 'quit' to exit]", file=sys.stderr)

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            return
        if user.lower() == "reset":
            asst.reset()
            print("[memory cleared]")
            continue
        res = asst.chat(user)
        prefix = "asst" if not res.blocked else "asst (blocked)"
        print(f"{prefix}> {res.text}\n")


if __name__ == "__main__":
    main()
