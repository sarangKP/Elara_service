"""
questionnaire.py — ELARA Structured Questionnaire Driver
=========================================================

Runs a scripted multi-phase conversation against the live ELARA
unified microservice (POST /chat) and captures every turn's full
telemetry for later report generation.

Usage
-----
    # Default: Ollama backend, auto-demo mode
    python questionnaire.py

    # Groq backend
    python questionnaire.py --backend groq

    # Custom service URL
    python questionnaire.py --url http://127.0.0.1:8000

    # Interactive mode (human plays the elderly user)
    python questionnaire.py --interactive

Output
------
    session_<id>_<timestamp>.json   — raw telemetry (input to report_generator.py)

Requirements
------------
    pip install requests
    Service must be running: uvicorn app:app --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

# ── ANSI colours ──────────────────────────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
B = "\033[94m"; M = "\033[95m"; C = "\033[96m"
DIM = "\033[2m"; BOLD = "\033[1m"; W = "\033[0m"
def col(text, c): return f"{c}{text}{W}"

# ── Questionnaire script ───────────────────────────────────────────────────────
# Five phases mirroring the validated Questionnaire.txt arc.
# Each entry: (phase_label, turn_text)

SCRIPT: List[tuple[str, str]] = [
    # ── Phase 1: Calm introduction ────────────────────────────────────────────
    ("Phase 1 – Calm Introduction",  "Hello, good morning. Who are you?"),
    ("Phase 1 – Calm Introduction",  "Oh I see. Can you tell me what the weather is like today?"),
    ("Phase 1 – Calm Introduction",  "That's nice. I had my tea already but I forgot if I took my tablet."),
    ("Phase 1 – Calm Introduction",  "I take one white tablet and one blue one every morning."),

    # ── Phase 2: Confusion starts ─────────────────────────────────────────────
    ("Phase 2 – Confusion Onset",    "I don't understand what you said. Can you say that again?"),
    ("Phase 2 – Confusion Onset",    "I already asked you about my tablet. Did I take it or not?"),
    ("Phase 2 – Confusion Onset",    "You're not making sense. I can't follow what you're telling me."),
    ("Phase 2 – Confusion Onset",    "I don't understand. Say it more simply please."),

    # ── Phase 3: Frustration peak ─────────────────────────────────────────────
    ("Phase 3 – Frustration Peak",   "I already told you about the tablet! Why do you keep asking me the same thing?"),
    ("Phase 3 – Frustration Peak",   "Nothing you say is helping me. I don't understand any of it."),
    ("Phase 3 – Frustration Peak",   "I already asked this. You never remember what I tell you."),
    ("Phase 3 – Frustration Peak",   "This is too complicated. I just want a simple answer."),
    ("Phase 3 – Frustration Peak",   "I already told you! The white tablet and the blue tablet. Every morning!"),
    ("Phase 3 – Frustration Peak",   "You are making me very upset. Nothing is working."),

    # ── Phase 4: Calming down ─────────────────────────────────────────────────
    ("Phase 4 – Recovery",           "Fine. Let's start over. Just tell me what I should do right now."),
    ("Phase 4 – Recovery",           "Okay that makes more sense. Thank you for being patient with me."),
    ("Phase 4 – Recovery",           "My knee is hurting a little today. Is that something to worry about?"),

    # ── Phase 5: Warm close ───────────────────────────────────────────────────
    ("Phase 5 – Warm Close",         "I feel a bit lonely today. My daughter hasn't called in a while."),
    ("Phase 5 – Warm Close",         "Can you tell me a short happy story? Something to cheer me up."),
    ("Phase 5 – Warm Close",         "That was lovely. I think I'll rest now. Goodnight ELARA."),
]


# ── Telemetry models ───────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    turn_number: int
    phase: str
    timestamp: str
    user_message: str
    elara_reply: str

    # Learning Agent diagnostics
    affect: str
    confidence: float
    signals_used: List[str]
    escalation_rule: Optional[str]

    # Config state AFTER this turn's adaptation
    config_pace: str
    config_clarity: int
    config_confirmation: str
    config_patience: bool

    # Config delta for THIS turn
    config_changes: Dict[str, Any]

    # Bandit
    ucb_action_id: int
    ucb_scores: List[float]
    reward_applied: Optional[float]

    # NLP raw scores (not in /chat response directly — derived from signals_used)
    signals_raw: List[str]

    latency_ms: float


@dataclass
class SessionRecord:
    session_id: str
    started_at: str
    finished_at: str
    backend: str
    model: Optional[str]
    total_turns: int
    turns: List[TurnRecord] = field(default_factory=list)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _post_chat(url: str, message: str, state: Optional[dict],
               backend: str, model: Optional[str]) -> dict:
    payload = {
        "message": message,
        "state": state,
        "backend": backend,
        "model": model,
    }
    resp = requests.post(f"{url}/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _check_health(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_auto(url: str, backend: str, model: Optional[str]) -> SessionRecord:
    """Runs the full scripted questionnaire and returns a populated SessionRecord."""
    print(col("\n╔══════════════════════════════════════════════════════════╗", M))
    print(col("║  ELARA Questionnaire — Automated Session                 ║", M))
    print(col("╚══════════════════════════════════════════════════════════╝", M))
    print(col(f"  Service  : {url}", B))
    print(col(f"  Backend  : {backend}  model={model or 'default'}", B))
    print(col(f"  Turns    : {len(SCRIPT)}\n", DIM))

    started_at = datetime.now(timezone.utc).isoformat()
    state: Optional[dict] = None
    turns: List[TurnRecord] = []

    for turn_idx, (phase, user_msg) in enumerate(SCRIPT, 1):
        print(col(f"[{turn_idx:02d}/{len(SCRIPT)}] {phase}", C))
        print(col(f"  👴 {user_msg}", BOLD))

        t0 = time.perf_counter()
        try:
            resp = _post_chat(url, user_msg, state, backend, model)
        except requests.RequestException as exc:
            print(col(f"  ⚠  Request failed: {exc}", R))
            break

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        state = resp["state"]
        diag  = resp["diagnostics"]
        cfg   = state["config"]

        reply_preview = resp["reply"][:90].replace("\n", " ")
        print(col(f"  🤖 {reply_preview}{'…' if len(resp['reply']) > 90 else ''}", B))

        affect_col = {
            "calm": G, "confused": Y, "frustrated": R,
            "sad": M,  "disengaged": C,
        }.get(diag["affect"], W)
        print(col(
            f"  → affect={diag['affect'].upper()} ({diag['confidence']:.0%})  "
            f"action={diag['ucb_action_id']}  "
            f"changes={diag['config_changes']}  "
            f"latency={latency_ms}ms",
            affect_col,
        ))
        if diag.get("escalation_rule"):
            print(col(f"    ⚡ escalation: {diag['escalation_rule']}", Y))
        print()

        turns.append(TurnRecord(
            turn_number=turn_idx,
            phase=phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_message=user_msg,
            elara_reply=resp["reply"],
            affect=diag["affect"],
            confidence=diag["confidence"],
            signals_used=diag["signals_used"],
            escalation_rule=diag.get("escalation_rule"),
            config_pace=cfg["pace"],
            config_clarity=cfg["clarity_level"],
            config_confirmation=cfg["confirmation_frequency"],
            config_patience=cfg["patience_mode"],
            config_changes=diag["config_changes"],
            ucb_action_id=diag["ucb_action_id"],
            ucb_scores=diag["ucb_scores"],
            reward_applied=diag.get("reward_applied"),
            signals_raw=diag["signals_used"],
            latency_ms=latency_ms,
        ))

        # Slight pause so the LLM isn't hammered
        time.sleep(0.3)

    session_id = state["session_id"] if state else "unknown"
    finished_at = datetime.now(timezone.utc).isoformat()

    record = SessionRecord(
        session_id=session_id,
        started_at=started_at,
        finished_at=finished_at,
        backend=backend,
        model=model,
        total_turns=len(turns),
        turns=turns,
    )

    # Affect arc summary
    arc = " → ".join(t.affect for t in turns)
    print(col("═" * 60, M))
    print(f"  Session ID  : {session_id}")
    print(f"  Turns       : {len(turns)}")
    print(f"  Affect arc  : {arc}")
    print(col("═" * 60, M))

    return record


def run_interactive(url: str, backend: str, model: Optional[str]) -> SessionRecord:
    """Interactive mode — human plays the elderly user."""
    print(col("\n╔══════════════════════════════════════════════════════════╗", M))
    print(col("║  ELARA Questionnaire — Interactive Session               ║", M))
    print(col("╚══════════════════════════════════════════════════════════╝", M))
    print(col("  Type 'quit' to end and save the session.\n", DIM))

    started_at = datetime.now(timezone.utc).isoformat()
    state: Optional[dict] = None
    turns: List[TurnRecord] = []
    turn_idx = 0

    while True:
        try:
            user_msg = input(col("👴 You: ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit", "bye"):
            break

        turn_idx += 1
        phase = f"Interactive Turn {turn_idx}"

        t0 = time.perf_counter()
        try:
            resp = _post_chat(url, user_msg, state, backend, model)
        except requests.RequestException as exc:
            print(col(f"⚠  Request failed: {exc}", R))
            continue

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        state = resp["state"]
        diag  = resp["diagnostics"]
        cfg   = state["config"]

        print(col(f"\n🤖 ELARA:\n  {resp['reply']}\n", B))

        affect_col = {
            "calm": G, "confused": Y, "frustrated": R,
            "sad": M,  "disengaged": C,
        }.get(diag["affect"], W)
        print(col(
            f"  [affect={diag['affect']} ({diag['confidence']:.0%})  "
            f"action={diag['ucb_action_id']}  changes={diag['config_changes']}  "
            f"{latency_ms}ms]",
            affect_col,
        ))
        print()

        turns.append(TurnRecord(
            turn_number=turn_idx,
            phase=phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_message=user_msg,
            elara_reply=resp["reply"],
            affect=diag["affect"],
            confidence=diag["confidence"],
            signals_used=diag["signals_used"],
            escalation_rule=diag.get("escalation_rule"),
            config_pace=cfg["pace"],
            config_clarity=cfg["clarity_level"],
            config_confirmation=cfg["confirmation_frequency"],
            config_patience=cfg["patience_mode"],
            config_changes=diag["config_changes"],
            ucb_action_id=diag["ucb_action_id"],
            ucb_scores=diag["ucb_scores"],
            reward_applied=diag.get("reward_applied"),
            signals_raw=diag["signals_used"],
            latency_ms=latency_ms,
        ))

    session_id = state["session_id"] if state else "unknown"
    return SessionRecord(
        session_id=session_id,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        backend=backend,
        model=model,
        total_turns=len(turns),
        turns=turns,
    )


# ── Save telemetry ─────────────────────────────────────────────────────────────

def save_telemetry(record: SessionRecord) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{record.session_id}_{ts}.json"
    data = {
        "session_id":   record.session_id,
        "started_at":   record.started_at,
        "finished_at":  record.finished_at,
        "backend":      record.backend,
        "model":        record.model,
        "total_turns":  record.total_turns,
        "turns": [asdict(t) for t in record.turns],
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(col(f"\n  ✓  Telemetry saved → {filename}", G))
    return filename


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ELARA Questionnaire Driver")
    parser.add_argument("--url",         default="http://127.0.0.1:8000", help="Service base URL")
    parser.add_argument("--backend",     default="ollama", choices=["ollama", "groq"])
    parser.add_argument("--model",       default=None,     help="Override LLM model name")
    parser.add_argument("--interactive", action="store_true", help="Human plays the elderly user")
    args = parser.parse_args()

    print(col("\n  Checking service health…", DIM), end=" ", flush=True)
    if not _check_health(args.url):
        print(col("OFFLINE", R))
        print(col(f"  Start the service:  uvicorn app:app --port 8000", Y))
        sys.exit(1)
    print(col("OK", G))

    if args.interactive:
        record = run_interactive(args.url, args.backend, args.model)
    else:
        record = run_auto(args.url, args.backend, args.model)

    if record.turns:
        telemetry_file = save_telemetry(record)
        print(col(f"  Run report generator:  python report_generator.py {telemetry_file}\n", C))
    else:
        print(col("  No turns recorded. Exiting.\n", Y))


if __name__ == "__main__":
    main()
