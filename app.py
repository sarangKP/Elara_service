"""
app.py — ELARA Unified Microservice Entry Point

CHANGES (audit fixes):
  - FIX #1 (Critical): Pass session_id as user_id into tables_locked() so
    each user's bandit matrices are stored and loaded independently.
    The global shared learning table is eliminated.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import io
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel as _BaseModel

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
from learning_agent.storage import tables_locked   # now accepts user_id

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
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Internal: Learning Agent pipeline ────────────────────────────────────────

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

    FIX #1: tables_locked() now receives req.session_id as the user_id so
    each user's A/b matrices are completely isolated from all other users.
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
    context_id    = encode_context_id(affect, cfg.clarity_level, cfg.pace)

    reward_applied = None

    # ── FIX #1: pass session_id so matrices are per-user ─────────────────
    with tables_locked(user_id=req.session_id) as (A, b):
        bandit = LinUCBBandit(A, b, alpha=0.8, gamma=0.95)

        if (
            req.previous_affect    is not None
            and req.previous_action_id is not None
            and req.previous_config    is not None
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


# ── TTS ───────────────────────────────────────────────────────────────────────

class TTSRequest(_BaseModel):
    text: str
    voice: str = "bf_emma"
    backend: str = "kokoro"   # "kokoro" | "edge"
    speed: float = 0.9        # slightly slower for elderly users


# Lazy-loaded Kokoro pipelines, one per lang_code ('a' American, 'b' British)
_kokoro_lock: threading.Lock       = threading.Lock()
_kokoro_pipelines: dict[str, object] = {}


def _get_kokoro_pipeline(lang_code: str):
    with _kokoro_lock:
        if lang_code not in _kokoro_pipelines:
            from kokoro import KPipeline
            log.info("[tts] Loading Kokoro pipeline (lang_code=%s)…", lang_code)
            _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return _kokoro_pipelines[lang_code]


def _kokoro_sync(text: str, voice: str, speed: float) -> io.BytesIO:
    """Run Kokoro TTS synchronously — called via thread executor."""
    import numpy as np
    import soundfile as sf

    lang_code = "b" if voice.startswith("b") else "a"
    pipeline  = _get_kokoro_pipeline(lang_code)

    chunks: list = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio)

    buf = io.BytesIO()
    if chunks:
        sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
        buf.seek(0)
    return buf


async def _tts_edge(req: TTSRequest) -> StreamingResponse:
    import edge_tts
    communicate = edge_tts.Communicate(req.text, req.voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
def ui():
    """Serve the ELARA companion UI."""
    return FileResponse("index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ELARA Unified", "version": "2.1.0"}


@app.post("/tts", include_in_schema=False)
async def tts(req: TTSRequest):
    """Text-to-speech via Kokoro (local/offline) or Edge TTS (online)."""
    if req.backend == "edge":
        return await _tts_edge(req)
    # Kokoro is CPU-bound — run in thread pool to avoid blocking the event loop
    import asyncio
    buf = await asyncio.get_event_loop().run_in_executor(
        None, _kokoro_sync, req.text, req.voice, req.speed
    )
    return StreamingResponse(buf, media_type="audio/wav")


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest) -> AnalyseResponse:
    """Direct access to the Learning Agent pipeline (backward compatible)."""
    return _run_learning_pipeline(req)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Stateless conversation turn."""
    adapter = ConversationAdapter()
    return adapter.handle_turn(req, _run_learning_pipeline)