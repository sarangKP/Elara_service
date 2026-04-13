"""
app.py — ELARA Unified Microservice Entry Point
================================================

Merges two distinct components into one stateless FastAPI service:

  Learning Agent  (black-box, untouched)
      POST /analyse   — NLP signal extraction → affect classification
                         → LinUCB bandit → config delta

  Conversation Agent  (new interface layer, replaces ollama_agent.py)
      POST /chat      — stateless conversation turn: accepts full session
                         state, calls the Learning Agent internally, calls
                         the LLM, returns reply + updated state
      GET  /health    — liveness probe

State ownership
---------------
The service is fully stateless.  ALL session state is owned by the caller
and must be echoed back on every request inside ChatRequest.  This mirrors
the pattern already used by the Learning Agent (caller echoes affect_window,
previous_affect, etc.) and extends it to cover the conversation history and
ELARA's live config.

Run
---
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Learning Agent (black-box imports — DO NOT MODIFY) ───────────────────────
from learning_agent.schemas import (
    AnalyseRequest,
    AnalyseResponse,
    ConversationWindow,
    CurrentConfig,
    Turn,
)
from learning_agent.nlp_layer import extract_signals
from learning_agent.state_classifier import (
    classify_state,
    encode_context_id,
    encode_context_features,
)
from learning_agent.bandit import LinUCBBandit
from learning_agent.config_applier import apply_action
from learning_agent.storage import tables_locked

# ── Conversation Agent (new interface layer) ─────────────────────────────────
from conversation_agent.adapter import (
    ConversationAdapter,
    ChatRequest,
    ChatResponse,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ELARA — Unified Elderly Care Companion Service",
    description=(
        "Stateless microservice combining the Learning Agent (LinUCB affect "
        "adaptation) with the Conversation Agent (persona-RAG + LLM interface)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Internal: Learning Agent pipeline (mirrors original main.py) ─────────────

_REWARD_TABLE: dict[tuple[str, str], float] = {
    ("frustrated", "calm"):       +1.0,
    ("frustrated", "confused"):   +0.3,
    ("frustrated", "frustrated"): -0.5,
    ("confused",   "calm"):       +1.0,
    ("confused",   "confused"):   -0.3,
    ("confused",   "frustrated"): -1.0,
    ("sad",        "calm"):       +1.0,
    ("sad",        "sad"):        -0.2,
    ("calm",       "calm"):        0.0,
    ("calm",       "confused"):   -0.5,
}


def _compute_reward(prev: str, curr: str) -> float:
    if curr == "disengaged":
        return -1.0
    return _REWARD_TABLE.get((prev, curr), 0.0)


def _run_learning_pipeline(req: AnalyseRequest) -> AnalyseResponse:
    """
    Runs the complete Learning Agent pipeline for one turn.
    This is identical to the original main.py /analyse endpoint logic,
    kept here so the /chat endpoint can call it directly without an HTTP hop.
    """
    t0 = time.time()

    turns = req.conversation_window.turns
    sentiment, repetition, confusion, sadness = extract_signals(turns)

    last_user_text = next(
        (t.text for t in reversed(turns) if t.role == "user"), ""
    )
    affect, confidence, signals_used, escalation_rule = classify_state(
        sentiment, repetition, confusion, sadness,
        last_user_text=last_user_text,
        affect_window=req.affect_window,
    )

    cfg = req.current_config
    curr_features = encode_context_features(affect, cfg.clarity_level, cfg.pace)
    context_id = encode_context_id(affect, cfg.clarity_level, cfg.pace)

    reward_applied = None
    with tables_locked() as (A, b):
        bandit = LinUCBBandit(A, b, alpha=0.8, gamma=0.95)

        if (
            req.previous_affect is not None
            and req.previous_action_id is not None
            and req.previous_config is not None
        ):
            prev_features = encode_context_features(
                req.previous_affect,
                req.previous_config.clarity_level,
                req.previous_config.pace,
            )
            reward = _compute_reward(req.previous_affect, affect)
            bandit.update(prev_features, req.previous_action_id, reward)
            reward_applied = reward

        action_id, ucb_scores = bandit.select_action(curr_features)
        A[:] = bandit.A
        b[:] = bandit.b

    new_cfg, changes, reason = apply_action(action_id, cfg, affect)
    elapsed_ms = round((time.time() - t0) * 1000)

    # Build response using the same schema the original endpoint returned
    from learning_agent.schemas import (
        InferredState, ConfigDelta, BanditContext, Diagnostics,
    )
    return AnalyseResponse(
        schema_version="1.1",
        session_id=req.session_id,
        processing_time_ms=elapsed_ms,
        inferred_state=InferredState(
            affect=affect,
            confidence=confidence,
            context_id=context_id,
            signals_used=signals_used,
            escalation_rule_applied=escalation_rule,
        ),
        config_delta=ConfigDelta(apply=len(changes) > 0, changes=changes, reason=reason),
        bandit_context=BanditContext(context_id=context_id, action_id=action_id),
        diagnostics=Diagnostics(
            sentiment_score=round(sentiment, 4),
            repetition_score=round(repetition, 4),
            confusion_score=round(confusion, 4),
            sadness_score=round(sadness, 4),
            ucb_scores=[round(s, 4) for s in ucb_scores],
            reward_applied=reward_applied,
            total_tries=0,
        ),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ELARA Unified", "version": "2.0.0"}


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest) -> AnalyseResponse:
    """
    Direct access to the Learning Agent pipeline.
    Retained for backward compatibility and independent testing.
    """
    return _run_learning_pipeline(req)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Stateless conversation turn.

    The caller owns all session state and must send it on every request.
    The service processes one turn, calls the Learning Agent pipeline
    internally, generates an LLM reply via the Conversation Agent, and
    returns the reply together with the updated state for the caller to
    store and echo back next turn.
    """
    adapter = ConversationAdapter()
    return adapter.handle_turn(req, _run_learning_pipeline)
