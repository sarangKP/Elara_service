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

        # IMPORTANT: LinUCBBandit.__init__ copies the arrays, so mutations
        # inside bandit.update() / select_action() only affect the bandit's
        # internal copies.  We must copy the learned state back into the
        # original arrays so that tables_locked().__exit__ persists them.
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


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    SSE streaming variant of /chat.

    Streams LLM tokens as they arrive so the edge device can begin TTS on
    the first sentence while the LLM is still generating the rest.  The
    final SSE event carries the full ChatResponse JSON (state + diagnostics)
    so the caller can update session state exactly as with /chat.

    Event types:
        data: {"token": "Hello "}       — incremental LLM token
        data: {"done": true, ...}       — final payload (ChatResponse JSON)
    """
    import json as _json
    from conversation_agent.adapter import ConversationAdapter, ChatResponse as _CR

    adapter = ConversationAdapter()

    def _generate():
        # --- Run the same pre-LLM pipeline as handle_turn ---
        from datetime import datetime, timezone as _tz

        state = req.state or adapter._new_session()
        state.interaction_count += 1

        ts = datetime.now(_tz.utc).isoformat()
        from conversation_agent.adapter import (
            ConversationTurn,
            _POSITIVE_FEEDBACK_PATTERNS,
            _IMMEDIATE_POSITIVE_REWARD,
            _PACE_TOKENS,
            _PERSONA,
            ElaraConfig,
            AdaptationDiagnostics,
        )

        state.history.append(
            ConversationTurn(role="user", content=req.message, timestamp=ts)
        )

        # Immediate positive reward (same logic as adapter.handle_turn)
        immediate_reward_applied = False
        if (
            _POSITIVE_FEEDBACK_PATTERNS.search(req.message)
            and state.bandit.previous_action_id is not None
            and state.bandit.previous_config is not None
        ):
            adapter._apply_immediate_reward(state, _IMMEDIATE_POSITIVE_REWARD)
            immediate_reward_applied = True
            state.bandit.previous_action_id = None

        # Build & run Learning Agent
        la_turns = adapter._history_to_la_turns(state.history)
        la_config = adapter._config_to_la_config(state.config)
        la_prev_config = (
            adapter._config_to_la_config(state.bandit.previous_config)
            if state.bandit.previous_config else None
        )

        from learning_agent.schemas import AnalyseRequest as _LAReq, ConversationWindow as _CW
        la_req = _LAReq(
            schema_version="1.1",
            session_id=state.session_id,
            conversation_window=_CW(turns=la_turns[-adapter.MAX_SERVICE_TURNS:]),
            current_config=la_config,
            previous_affect=state.bandit.previous_affect,
            previous_action_id=state.bandit.previous_action_id,
            previous_config=la_prev_config,
            affect_window=state.bandit.affect_window[-5:],
            interaction_count=state.interaction_count,
        )

        la_resp = _run_learning_pipeline(la_req)

        # Update bandit tracking
        state.bandit.previous_config = ElaraConfig(**state.config.model_dump())
        state.bandit.previous_affect = la_resp.inferred_state.affect
        state.bandit.previous_action_id = la_resp.bandit_context.action_id
        state.bandit.previous_context_id = la_resp.bandit_context.context_id
        state.bandit.affect_window = (
            state.bandit.affect_window + [la_resp.inferred_state.affect]
        )[-5:]

        # Distress watchdog
        caregiver_alert = False
        if la_resp.inferred_state.affect == "calm":
            state.consecutive_distress_turns = 0
        else:
            state.consecutive_distress_turns += 1
        caregiver_alert = adapter._check_distress_watchdog(state)

        # Apply config delta
        if la_resp.config_delta.apply and la_resp.config_delta.changes:
            for k, v in la_resp.config_delta.changes.items():
                if hasattr(state.config, k):
                    setattr(state.config, k, v)

        # Build system prompt & messages
        from conversation_agent.rag import build_persona_prompt
        config_dict = state.config.model_dump()
        system_prompt = build_persona_prompt(_PERSONA, req.message, config_dict)

        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.history[-(adapter.MAX_HISTORY_TURNS * 2):]:
            llm_role = "assistant" if turn.role == "assistant" else "user"
            messages.append({"role": llm_role, "content": turn.content})

        max_tokens = _PACE_TOKENS.get(state.config.pace, 300)

        # --- Stream LLM tokens as SSE events ---
        from conversation_agent.llm import stream_response
        reply_parts = []
        try:
            for chunk in stream_response(
                messages,
                backend=req.backend,
                model=req.model,
                max_tokens=max_tokens,
            ):
                reply_parts.append(chunk)
                yield f"data: {_json.dumps({'token': chunk})}\n\n"
        except Exception as exc:
            log.error("LLM stream failed: %s", exc)
            fallback = "I'm sorry, I'm having a little trouble thinking right now. Please try again."
            reply_parts = [fallback]
            yield f"data: {_json.dumps({'token': fallback})}\n\n"

        reply = "".join(reply_parts)

        # Append to history & trim
        state.history.append(
            ConversationTurn(role="assistant", content=reply, timestamp=ts)
        )
        if len(state.history) > adapter.MAX_HISTORY_TURNS * 2:
            state.history = state.history[-(adapter.MAX_HISTORY_TURNS * 2):]

        # Build final diagnostics payload
        diag = AdaptationDiagnostics(
            affect=la_resp.inferred_state.affect,
            confidence=la_resp.inferred_state.confidence,
            signals_used=la_resp.inferred_state.signals_used,
            config_changes=la_resp.config_delta.changes,
            reward_applied=la_resp.diagnostics.reward_applied,
            ucb_action_id=la_resp.bandit_context.action_id,
            ucb_scores=la_resp.diagnostics.ucb_scores,
            escalation_rule=la_resp.inferred_state.escalation_rule_applied,
            distress_turns=state.consecutive_distress_turns,
            caregiver_alert=caregiver_alert,
            immediate_reward_applied=immediate_reward_applied,
        )

        final = _CR(reply=reply, state=state, diagnostics=diag)
        yield f"data: {_json.dumps({'done': True, **_json.loads(final.model_dump_json())})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")