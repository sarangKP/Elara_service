"""
conversation_agent/adapter.py
==============================

Adapter ("glue code") between the Conversation Agent interface and the
Learning Agent pipeline.

Responsibilities
----------------
1.  Define ChatRequest / ChatResponse — the stateless API contract for /chat.
    All session state the caller must persist and echo back is explicit here.

2.  ConversationAdapter.handle_turn() — the bridge:
      a. Translate ChatRequest → AnalyseRequest (Learning Agent schema)
      b. Call _run_learning_pipeline (injected — no HTTP hop)
      c. Merge the returned config delta into the caller-supplied config
      d. Build the system prompt from the updated config + persona RAG
      e. Call the LLM (Ollama or Groq via the Conversation Agent's llm.py)
      f. Return ChatResponse with the reply + updated full state

State contract
--------------
The caller MUST store every field in ChatResponse.state and echo the whole
SessionState blob back as ChatRequest.state on the very next turn.  The
server never touches a database or in-process cache.

Separation of concerns
-----------------------
- Learning Agent files (learning_agent/) are NEVER imported here.
  The pipeline callable is injected by app.py so this module stays decoupled.
- Persona logic lives in rag.py (unchanged from the Conversation Agent).
- LLM streaming/collection lives in llm.py (unchanged).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Conversation Agent imports (unchanged files) ──────────────────────────────
from conversation_agent.rag import load_persona, build_persona_prompt, retrieve
from conversation_agent.llm import collect_stream

# ── Token budgets (mirrors ollama_agent.py PACE_TOKENS) ──────────────────────
_PACE_TOKENS: Dict[str, int] = {"slow": 400, "normal": 300, "fast": 100}

# ── Default ELARA config ──────────────────────────────────────────────────────
_DEFAULT_CONFIG: Dict[str, Any] = {
    "pace": "normal",
    "clarity_level": 2,
    "confirmation_frequency": "low",
    "patience_mode": False,
}

# Load persona once at module import (it never changes at runtime)
_PERSONA: Dict[str, Any] = load_persona()


# ── Pydantic models ───────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """One turn in the conversation history (stored by the caller)."""
    role: str          # "user" | "assistant"
    content: str
    timestamp: Optional[str] = None


class ElaraConfig(BaseModel):
    """Live ELARA behaviour config — mutated by the Learning Agent."""
    pace: str = "normal"
    clarity_level: int = 2
    confirmation_frequency: str = "low"
    patience_mode: bool = False


class BanditState(BaseModel):
    """Bandit tracking state — echoed back for correct LinUCB reward attribution."""
    previous_affect: Optional[str] = None
    previous_action_id: Optional[int] = None
    previous_context_id: Optional[int] = None
    previous_config: Optional[ElaraConfig] = None
    affect_window: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """
    Complete session state owned by the caller.
    Send this back verbatim on every subsequent request.
    """
    session_id: str
    interaction_count: int = 0
    config: ElaraConfig = Field(default_factory=ElaraConfig)
    bandit: BanditState = Field(default_factory=BanditState)
    # Last N turns for the LLM context window (caller may trim to save bandwidth)
    history: List[ConversationTurn] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """
    Stateless /chat request.

    On the very first turn: send state=None (or omit it).
    On every subsequent turn: echo back the state from the previous ChatResponse.
    """
    message: str
    state: Optional[SessionState] = None

    # LLM backend selection — caller may override per-request
    backend: str = "ollama"          # "ollama" | "groq"
    model: Optional[str] = None      # None → use backend default


class AdaptationDiagnostics(BaseModel):
    """Optional diagnostics from the Learning Agent pipeline."""
    affect: str
    confidence: float
    signals_used: List[str]
    config_changes: Dict[str, Any]
    reward_applied: Optional[float]
    ucb_action_id: int
    ucb_scores: List[float]
    escalation_rule: Optional[str] = None


class ChatResponse(BaseModel):
    """
    Stateless /chat response.

    `state` must be stored by the caller and sent back as ChatRequest.state
    on the next turn.  `diagnostics` is informational only.
    """
    reply: str
    state: SessionState
    diagnostics: AdaptationDiagnostics


# ── Adapter ───────────────────────────────────────────────────────────────────

class ConversationAdapter:
    """
    Translates between the /chat API surface and the internal Learning Agent
    pipeline + LLM calls.

    No instance state is kept — a new ConversationAdapter is created per
    request in app.py.  This enforces statelessness at the class level.
    """

    MAX_HISTORY_TURNS = 10   # turns kept in the LLM context window
    MAX_SERVICE_TURNS = 10   # turns sent to the Learning Agent (mirrors ollama_agent)

    def handle_turn(
        self,
        req: ChatRequest,
        learning_pipeline: Callable,   # injected from app.py
    ) -> ChatResponse:
        # ── 1. Restore or initialise session state ────────────────────────
        state = req.state or self._new_session()
        state.interaction_count += 1

        # ── 2. Append user message to history ─────────────────────────────
        ts = datetime.now(timezone.utc).isoformat()
        state.history.append(
            ConversationTurn(role="user", content=req.message, timestamp=ts)
        )

        # ── 3. Build Learning Agent request ───────────────────────────────
        la_turns = self._history_to_la_turns(state.history)
        la_config = self._config_to_la_config(state.config)
        la_prev_config = (
            self._config_to_la_config(state.bandit.previous_config)
            if state.bandit.previous_config else None
        )

        from learning_agent.schemas import AnalyseRequest, ConversationWindow
        la_req = AnalyseRequest(
            schema_version="1.1",
            session_id=state.session_id,
            conversation_window=ConversationWindow(turns=la_turns[-self.MAX_SERVICE_TURNS:]),
            current_config=la_config,
            previous_affect=state.bandit.previous_affect,
            previous_action_id=state.bandit.previous_action_id,
            previous_config=la_prev_config,
            affect_window=state.bandit.affect_window[-5:],
            interaction_count=state.interaction_count,
        )

        # ── 4. Run Learning Agent pipeline ────────────────────────────────
        la_resp = learning_pipeline(la_req)

        # ── 5. Update bandit tracking state ───────────────────────────────
        #    Capture config BEFORE applying the new delta (for next-turn reward)
        state.bandit.previous_config = ElaraConfig(**state.config.model_dump())
        state.bandit.previous_affect = la_resp.inferred_state.affect
        state.bandit.previous_action_id = la_resp.bandit_context.action_id
        state.bandit.previous_context_id = la_resp.bandit_context.context_id

        # Rolling affect window — keep last 5
        state.bandit.affect_window = (
            state.bandit.affect_window + [la_resp.inferred_state.affect]
        )[-5:]

        # ── 6. Apply config delta ─────────────────────────────────────────
        if la_resp.config_delta.apply and la_resp.config_delta.changes:
            for k, v in la_resp.config_delta.changes.items():
                if hasattr(state.config, k):
                    setattr(state.config, k, v)

        # ── 7. Build system prompt (persona RAG + live config) ────────────
        config_dict = state.config.model_dump()
        system_prompt = build_persona_prompt(_PERSONA, req.message, config_dict)

        # ── 8. Build LLM message list ─────────────────────────────────────
        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.history[-(self.MAX_HISTORY_TURNS * 2):]:
            llm_role = "assistant" if turn.role == "assistant" else "user"
            messages.append({"role": llm_role, "content": turn.content})

        # ── 9. Call LLM ───────────────────────────────────────────────────
        max_tokens = _PACE_TOKENS.get(state.config.pace, 300)
        try:
            reply = collect_stream(
                messages,
                backend=req.backend,
                model=req.model,
                max_tokens=max_tokens,
                print_live=False,
            )
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            reply = "I'm sorry, I'm having a little trouble thinking right now. Please try again."

        # ── 10. Append assistant reply to history & trim ──────────────────
        state.history.append(
            ConversationTurn(role="assistant", content=reply, timestamp=ts)
        )
        # Keep history bounded so payload stays manageable
        if len(state.history) > self.MAX_HISTORY_TURNS * 2:
            state.history = state.history[-(self.MAX_HISTORY_TURNS * 2):]

        # ── 11. Build diagnostics ─────────────────────────────────────────
        diag = AdaptationDiagnostics(
            affect=la_resp.inferred_state.affect,
            confidence=la_resp.inferred_state.confidence,
            signals_used=la_resp.inferred_state.signals_used,
            config_changes=la_resp.config_delta.changes,
            reward_applied=la_resp.diagnostics.reward_applied,
            ucb_action_id=la_resp.bandit_context.action_id,
            ucb_scores=la_resp.diagnostics.ucb_scores,
            escalation_rule=la_resp.inferred_state.escalation_rule_applied,
        )

        return ChatResponse(reply=reply, state=state, diagnostics=diag)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _new_session() -> SessionState:
        import uuid
        return SessionState(
            session_id=f"elara-{uuid.uuid4().hex[:8]}",
            config=ElaraConfig(**_DEFAULT_CONFIG),
        )

    @staticmethod
    def _history_to_la_turns(history: List[ConversationTurn]):
        """Convert ChatResponse history → Learning Agent Turn list."""
        from learning_agent.schemas import Turn
        result = []
        for turn in history:
            la_role = "user" if turn.role == "user" else "agent"
            result.append(Turn(role=la_role, text=turn.content, timestamp=turn.timestamp))
        return result

    @staticmethod
    def _config_to_la_config(config):
        """Convert ElaraConfig → Learning Agent CurrentConfig."""
        from learning_agent.schemas import CurrentConfig
        if config is None:
            return CurrentConfig()
        return CurrentConfig(
            pace=config.pace,
            clarity_level=config.clarity_level,
            confirmation_frequency=config.confirmation_frequency,
            patience_mode=config.patience_mode,
        )
