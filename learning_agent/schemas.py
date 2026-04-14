from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

# Constants
_VALID_AFFECTS = {"frustrated", "confused", "sad", "calm", "disengaged"}
MAX_TURNS      = 50
MAX_WINDOW     = 5

class Turn(BaseModel):
    role: str # "user" | "agent"
    text: str
    timestamp: Optional[str] = None

class ConversationWindow(BaseModel):
    turns: List[Turn] = Field(..., max_length=MAX_TURNS)

class CurrentConfig(BaseModel):
    pace: str = "normal"
    clarity_level: int = 2
    confirmation_frequency: str = "low"
    patience_mode: bool = False

class AnalyseRequest(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    conversation_window: ConversationWindow
    current_config: CurrentConfig = Field(default_factory=CurrentConfig)
    
    # Context for reward attribution
    previous_affect: Optional[str] = None
    previous_action_id: Optional[int] = None
    previous_config: Optional[CurrentConfig] = None # NEW: Critical for LinUCB
    
    affect_window: Optional[List[str]] = Field(default=None, max_length=MAX_WINDOW)
    interaction_count: int = 0

    @field_validator("affect_window", "previous_affect", mode="before")
    @classmethod
    def validate_affects(cls, v):
        if v is None: return v
        if isinstance(v, list):
            invalid = [e for e in v if e not in _VALID_AFFECTS]
            if invalid: raise ValueError(f"Invalid affects: {invalid}")
        elif v not in _VALID_AFFECTS:
            raise ValueError(f"Invalid affect: {v}")
        return v

# --- Response Models (Remain unchanged from your previous version) ---
class InferredState(BaseModel):
    affect: str
    confidence: float
    context_id: int
    signals_used: List[str]
    escalation_rule_applied: Optional[str] = None

class ConfigDelta(BaseModel):
    apply: bool
    changes: Dict[str, Any]
    reason: str

class BanditContext(BaseModel):
    context_id: int
    action_id: int

class Diagnostics(BaseModel):
    sentiment_score: float
    repetition_score: float
    confusion_score: float = 0.0
    sadness_score: float = 0.0
    ucb_scores: List[float]
    reward_applied: Optional[float]
    total_tries: int

class AnalyseResponse(BaseModel):
    schema_version: str
    session_id: str
    processing_time_ms: int
    inferred_state: InferredState
    config_delta: ConfigDelta
    bandit_context: BanditContext
    diagnostics: Diagnostics