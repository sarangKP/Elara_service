"""
example_client.py — Demonstrates stateless /chat usage
=======================================================

Shows how any front-end (voice UI, web app, Telegram bot, etc.) should
interact with the unified ELARA microservice.

Key pattern
-----------
  1. First turn: send message with state=null
  2. Store the returned state object
  3. Every subsequent turn: echo the stored state back in the request

Run: python example_client.py
"""

import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def chat(message: str, state: dict | None, backend: str = "ollama") -> dict:
    payload = {
        "message": message,
        "state": state,
        "backend": backend,
    }
    resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    print("=== ELARA Unified Service — Example Client ===\n")
    print("Type 'quit' to exit.\n")

    state = None   # No state on first turn

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            break

        try:
            result = chat(user_msg, state)
        except requests.RequestException as exc:
            print(f"[Error] {exc}")
            continue

        reply = result["reply"]
        state = result["state"]           # ← store and echo back next turn
        diag  = result["diagnostics"]

        print(f"\nELARA: {reply}")
        print(
            f"  [affect={diag['affect']} ({diag['confidence']:.0%})  "
            f"config_changes={diag['config_changes']}  "
            f"action={diag['ucb_action_id']}]\n"
        )


if __name__ == "__main__":
    main()
