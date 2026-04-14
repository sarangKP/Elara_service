"""
conversation_agent/adapter.py
==============================

CHANGES (audit fixes):
  - FIX #2 (Critical): Distress watchdog.
    A rolling counter tracks consecutive non-calm turns. When it hits
    DISTRESS_TURN_LIMIT the system logs a caregiver alert. Easy to wire
    up to an email/SMS/push notification — just replace the log.warning
    call in _check_distress_watchdog() with your alert function.

  - FIX #3 (High): Immediate positive reward signal.
    If the user's message contains an explicit "thank you / that helped"
    phrase, a +1.0 reward is injected directly into the bandit for the
    previous action — without waiting for next-turn affect to improve.
    This teaches ELARA what actually worked, not just that things got better.

No other logic is changed. All original comments are preserved.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Conversation Agent imports (unchanged files) ──────────────────────────────
from conversation_agent.rag import load_persona, build_persona_prompt, retrieve
from conversation_agent.llm import collect_stream

# ── Token budgets ─────────────────────────────────────────────────────────────
_PACE_TOKENS: Dict[str, int] = {"slow": 400, "normal": 300, "fast": 100}

# ── Default ELARA config ──────────────────────────────────────────────────────
_DEFAULT_CONFIG: Dict[str, Any] = {
    "pace": "normal",
    "clarity_level": 2,
    "confirmation_frequency": "low",
    "patience_mode": False,
}

# Load persona once at module import
_PERSONA: Dict[str, Any] = load_persona()

# ── FIX #2: Distress watchdog settings ───────────────────────────────────────
# How many consecutive non-calm turns before we alert a caregiver.
DISTRESS_TURN_LIMIT = 7

# ── FIX #3: Positive feedback phrases ────────────────────────────────────────
# If the user says any of these, inject an immediate +1.0 reward.
_POSITIVE_FEEDBACK_PATTERNS = re.compile(
    r"\b("
    r"thank you|thanks|that('s| is) (helpful|great|lovely|nice|perfect|clear)|"
    r"that helped|much better|i understand (now)?|that makes sense|"
    r"got it|perfect|brilliant|wonderful|that'?s? (good|better)"
    r")\b",
    re.IGNORECASE,
)

# Immediate reward value for explicit positive feedback
_IMMEDIATE_POSITIVE_REWARD = 1.0


# ── Pydantic models ───────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ElaraConfig(BaseModel):
    pace: str = "normal"
    clarity_level: int = 2
    confirmation_frequency: str = "low"
    patience_mode: bool = False


class BanditState(BaseModel):
    previous_affect: Optional[str] = None
    previous_action_id: Optional[int] = None
    previous_context_id: Optional[int] = None
    previous_config: Optional[ElaraConfig] = None
    affect_window: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    session_id: str
    interaction_count: int = 0
    config: ElaraConfig = Field(default_factory=ElaraConfig)
    bandit: BanditState = Field(default_factory=BanditState)
    history: List[ConversationTurn] = Field(default_factory=list)

    # FIX #2: track consecutive non-calm turns for the distress watchdog
    consecutive_distress_turns: int = 0


class ChatRequest(BaseModel):
    message: str
    state: Optional[SessionState] = None
    backend: str = "ollama"
    model: Optional[str] = None


class AdaptationDiagnostics(BaseModel):
    affect: str
    confidence: float
    signals_used: List[str]
    config_changes: Dict[str, Any]
    reward_applied: Optional[float]
    ucb_action_id: int
    ucb_scores: List[float]
    escalation_rule: Optional[str] = None
    # FIX #2 + #3: surface watchdog and immediate-reward info to callers
    distress_turns: int = 0
    caregiver_alert: bool = False
    immediate_reward_applied: bool = False


class ChatResponse(BaseModel):
    reply: str
    state: SessionState
    diagnostics: AdaptationDiagnostics


# ── Adapter ───────────────────────────────────────────────────────────────────

class ConversationAdapter:

    MAX_HISTORY_TURNS  = 10
    MAX_SERVICE_TURNS  = 10

    def handle_turn(
        self,
        req: ChatRequest,
        learning_pipeline: Callable,
    ) -> ChatResponse:

        # ── 1. Restore or initialise session state ────────────────────────
        state = req.state or self._new_session()
        state.interaction_count += 1

        # ── 2. Append user message to history ─────────────────────────────
        ts = datetime.now(timezone.utc).isoformat()
        state.history.append(
            ConversationTurn(role="user", content=req.message, timestamp=ts)
        )

        # ── FIX #3: Check for immediate positive feedback BEFORE pipeline ─
        # If the user explicitly says "thank you / that helped", we apply a
        # reward directly now — without waiting for next-turn affect change.
        immediate_reward_applied = False
        if (
            _POSITIVE_FEEDBACK_PATTERNS.search(req.message)
            and state.bandit.previous_action_id is not None
            and state.bandit.previous_config is not None
        ):
            self._apply_immediate_reward(state, _IMMEDIATE_POSITIVE_REWARD)
            immediate_reward_applied = True
            log.info(
                "[adapter] Immediate positive reward +%.1f applied for session %s",
                _IMMEDIATE_POSITIVE_REWARD,
                state.session_id,
            )
            # FIX #1-double-reward: Null out previous_action_id so the
            # normal pipeline reward update (app.py _run_learning_pipeline)
            # skips its bandit.update() call for this turn.  Without this,
            # the same (prev_features, prev_action_id) pair receives two
            # updates — the immediate +1.0 here AND the affect-transition
            # reward — inflating the bandit's estimate of the action.
            state.bandit.previous_action_id = None

        # ── 3. Build Learning Agent request ───────────────────────────────
        la_turns      = self._history_to_la_turns(state.history)
        la_config     = self._config_to_la_config(state.config)
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
        state.bandit.previous_config     = ElaraConfig(**state.config.model_dump())
        state.bandit.previous_affect     = la_resp.inferred_state.affect
        state.bandit.previous_action_id  = la_resp.bandit_context.action_id
        state.bandit.previous_context_id = la_resp.bandit_context.context_id

        state.bandit.affect_window = (
            state.bandit.affect_window + [la_resp.inferred_state.affect]
        )[-5:]

        # ── FIX #2: Update distress watchdog counter ──────────────────────
        caregiver_alert = False
        if la_resp.inferred_state.affect == "calm":
            state.consecutive_distress_turns = 0
        else:
            state.consecutive_distress_turns += 1

        caregiver_alert = self._check_distress_watchdog(state)

        # ── 6. Apply config delta ─────────────────────────────────────────
        if la_resp.config_delta.apply and la_resp.config_delta.changes:
            for k, v in la_resp.config_delta.changes.items():
                if hasattr(state.config, k):
                    setattr(state.config, k, v)

        # ── 7. Build system prompt ────────────────────────────────────────
        config_dict   = state.config.model_dump()
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
            distress_turns=state.consecutive_distress_turns,
            caregiver_alert=caregiver_alert,
            immediate_reward_applied=immediate_reward_applied,
        )

        return ChatResponse(reply=reply, state=state, diagnostics=diag)

    # ── FIX #2: Distress watchdog ─────────────────────────────────────────────

    @staticmethod
    def _check_distress_watchdog(state: SessionState) -> bool:
        """
        If the user has been non-calm for DISTRESS_TURN_LIMIT consecutive turns,
        log a caregiver alert and return True.

        TO ADD REAL ALERTS: Replace the log.warning line below with your
        notification call — e.g. send_sms(), push_notification(), email_caregiver().
        """
        if state.consecutive_distress_turns >= DISTRESS_TURN_LIMIT:
            log.warning(
                "[CAREGIVER ALERT] Session %s — user has been distressed for %d "
                "consecutive turns. Consider contacting a caregiver.",
                state.session_id,
                state.consecutive_distress_turns,
            )
            return True
        return False

    # ── FIX #3: Immediate reward injection ───────────────────────────────────

    @staticmethod
    def _apply_immediate_reward(state: SessionState, reward: float) -> None:
        """
        Inject a positive reward directly into the bandit for the previous
        action, without waiting for next-turn affect to improve.

        This is called when the user explicitly signals satisfaction
        ("thank you", "that helped", etc.).

        We import storage here to avoid a circular import at module level.
        """
        from learning_agent.storage import tables_locked
        from learning_agent.state_classifier import encode_context_features

        if (
            state.bandit.previous_action_id is None
            or state.bandit.previous_config  is None
            or state.bandit.previous_affect  is None
        ):
            return

        prev_cfg = state.bandit.previous_config
        features = encode_context_features(
            state.bandit.previous_affect,
            prev_cfg.clarity_level,
            prev_cfg.pace,
        )

        from learning_agent.bandit import LinUCBBandit
        with tables_locked(user_id=state.session_id) as (A, b):
            bandit = LinUCBBandit(A, b, alpha=0.8, gamma=0.95)
            bandit.update(features, state.bandit.previous_action_id, reward)
            A[:] = bandit.A
            b[:] = bandit.b

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
        from learning_agent.schemas import Turn
        result = []
        for turn in history:
            la_role = "user" if turn.role == "user" else "agent"
            result.append(Turn(role=la_role, text=turn.content, timestamp=turn.timestamp))
        return result

    @staticmethod
    def _config_to_la_config(config):
        from learning_agent.schemas import CurrentConfig
        if config is None:
            return CurrentConfig()
        return CurrentConfig(
            pace=config.pace,
            clarity_level=config.clarity_level,
            confirmation_frequency=config.confirmation_frequency,
            patience_mode=config.patience_mode,
        )