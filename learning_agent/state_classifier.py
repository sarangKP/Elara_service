"""
Layer 2 — State Classifier

Affect states: frustrated | confused | sad | calm | disengaged

Four input signals:
  sentiment_score  : VADER compound       (-1 to +1)
  repetition_score : TF-IDF cosine        (0 to 1)
  confusion_score  : confusion keywords   (0 to 1)
  sadness_score    : sadness keywords     (0 to 1)

Decision priority (checked top to bottom, first match wins):
  frustrated   : strong negative + repetition  → needs clarity + pace + patience
  confused     : confusion keywords or repetition → needs clarity + confirmation
  sad          : sadness keywords (no confusion) → needs patience + empathy only
  disengaged   : very short / minimal input with neutral/no signal → needs re-engagement
  calm         : none of the above

── Escalation Smoother ───────────────────────────────────────────────────────
After the raw affect is determined, apply_escalation_rules() inspects the
rolling affect_window (last N classifications sent by the caller) and may
downgrade the raw affect if the evidence is insufficient to justify a sudden
jump.

The core principle: affect should escalate gradually.
  calm → confused → frustrated    ✓  (one step at a time)
  calm → frustrated               ✗  (skip-level jump — needs very strong evidence)

Rules (checked in order, first match wins):

  Frustrated rules:
    R3  all_calm_history
        If every entry in the window is calm, frustrated is downgraded to
        confused regardless of signal strength.

    R1  insufficient_streak
        Frustrated is only allowed to stand if the window ends with at least
        MIN_NONCALM_STREAK (2) consecutive non-calm entries. A single non-calm
        turn is always suppressed; two or more consecutive non-calm turns means
        the escalation is genuine and earned.
        all_signals_fired bypasses this so genuine multi-signal frustration
        can still escalate even from a short streak.

  Disengaged rule:
    R4  disengaged_calm_history
        FIX: disengaged can fire on any very short message (≤3 words) even
        after an entirely calm history — e.g. "Yes." or "Okay." said by a
        user who is simply being brief. This false positive then trains the
        bandit on a spurious disengaged state.
        If the window is entirely calm and the raw affect is disengaged,
        downgrade to calm. A genuinely disengaged user will show a pattern
        of short messages across multiple turns; a single short message
        after calm history is more likely brevity than disengagement.

These rules are additive dampeners — they never upgrade an affect, only
downgrade it. The full multi-signal frustrated path (negative + repetition +
strong keywords) bypasses R1 so genuine sustained frustration is still caught.

── affect_window validation ──────────────────────────────────────────────────
FIX: the escalation smoother trusted the caller to send valid affect strings.
A malformed window (typo, wrong length, stale data) silently produced
incorrect escalation decisions because unknown strings would never match
"calm", making the all-calm guard unreliable.
Now any unrecognised entry is stripped with a warning before the window is
used. The original list is never mutated.

N_CONTEXTS is exported here as the single source of truth.
bandit.py and storage.py import from here to avoid definition drift.
"""

from __future__ import annotations
import logging
from typing import Tuple, List, Optional
import numpy as np

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
SENTIMENT_THRESHOLD   = -0.2
REPETITION_THRESHOLD  = 0.55
CONFUSION_THRESHOLD   = 0.5
SADNESS_THRESHOLD     = 0.5
DISENGAGED_WORD_COUNT = 3    # messages with ≤ this many words trigger disengaged check

# Escalation smoother settings
WINDOW_SIZE        = 5   # how many past affects the caller should send
MIN_NONCALM_STREAK = 2   # consecutive non-calm turns required before frustrated is allowed

# ── Encoding maps ─────────────────────────────────────────────────────────────
AFFECT_MAP  = {"frustrated": 0, "confused": 1, "sad": 2, "calm": 3, "disengaged": 4}
CLARITY_MAP = {1: 0, 2: 1, 3: 2}
PACE_MAP    = {"slow": 0, "normal": 1, "fast": 2}

# Single source of truth — imported by bandit.py and storage.py
N_CONTEXTS = 45   # 5 affects × 9 (3 clarity × 3 pace)
N_ACTIONS  = 7

_VALID_AFFECTS = frozenset(AFFECT_MAP.keys())


# ── Public API ────────────────────────────────────────────────────────────────

def classify_state(
    sentiment: float,
    repetition: float,
    confusion: float = 0.0,
    sadness: float   = 0.0,
    last_user_text: str = "",
    affect_window: Optional[List[str]] = None,
) -> Tuple[str, float, List[str], Optional[str]]:
    """
    Returns (affect, confidence, signals_used, escalation_rule_applied).

    escalation_rule_applied is None when no rule fired, or a short string
    describing which rule downgraded the raw affect.

    Extra args default to 0.0 / "" / None for backward compatibility.
    """
    # FIX: validate and sanitise the affect_window before use.
    # Unknown strings are stripped with a warning so a stale or malformed
    # window never silently corrupts escalation decisions.
    clean_window: Optional[List[str]] = None
    if affect_window is not None:
        clean_window = []
        for entry in affect_window:
            if entry in _VALID_AFFECTS:
                clean_window.append(entry)
            else:
                log.warning(
                    "[state_classifier] Unknown affect '%s' in affect_window — "
                    "ignored. Valid values: %s",
                    entry,
                    sorted(_VALID_AFFECTS),
                )

    negative    = sentiment  < SENTIMENT_THRESHOLD
    repetitive  = repetition > REPETITION_THRESHOLD
    confused_kw = confusion  > CONFUSION_THRESHOLD
    sad_kw      = sadness    > SADNESS_THRESHOLD

    signals_used: List[str] = []
    if negative:    signals_used.append("sentiment")
    if repetitive:  signals_used.append("repetition")
    if confused_kw: signals_used.append("confusion_keywords")
    if sad_kw:      signals_used.append("sadness_keywords")

    # ── 1. frustrated ────────────────────────────────────────────
    strong_neg   = negative or confusion > 0.65
    high_conf_kw = confusion > 0.8
    if (strong_neg and repetitive) or (high_conf_kw and (negative or repetitive or confusion > 0.9)):
        affect = "frustrated"
        conf = _blend(
            abs(sentiment - SENTIMENT_THRESHOLD) / 1.0 if negative else confusion,
            max(
                (repetition - REPETITION_THRESHOLD) / (1.0 - REPETITION_THRESHOLD),
                (confusion  - 0.65) / 0.35,
            ),
        )

    # ── 2. confused ──────────────────────────────────────────────
    elif confused_kw or (repetitive and not sad_kw):
        affect = "confused"
        scores = []
        if confused_kw:
            scores.append((confusion - CONFUSION_THRESHOLD) / (1.0 - CONFUSION_THRESHOLD))
        if repetitive:
            scores.append((repetition - REPETITION_THRESHOLD) / (1.0 - REPETITION_THRESHOLD))
        if negative and not sad_kw:
            scores.append(abs(sentiment - SENTIMENT_THRESHOLD) / 1.0)
        conf = _clamp(max(scores)) if scores else 0.5

    # ── 3. sad ───────────────────────────────────────────────────
    elif sad_kw or (negative and not confused_kw and not repetitive):
        affect = "sad"
        scores = []
        if sad_kw:
            scores.append((sadness - SADNESS_THRESHOLD) / (1.0 - SADNESS_THRESHOLD))
        if negative:
            scores.append(abs(sentiment - SENTIMENT_THRESHOLD) / 1.0)
        conf = _clamp(max(scores)) if scores else 0.5

    # ── 4. disengaged ────────────────────────────────────────────
    elif (
        last_user_text
        and len(last_user_text.split()) <= DISENGAGED_WORD_COUNT
        and not negative
        and not confused_kw
        and not sad_kw
        and not repetitive
    ):
        affect = "disengaged"
        signals_used.append("short_message")
        conf = 0.6

    # ── 5. calm ──────────────────────────────────────────────────
    else:
        affect = "calm"
        conf = _clamp(
            0.5
            + (sentiment  - SENTIMENT_THRESHOLD)  / 2.0
            + (REPETITION_THRESHOLD - repetition) / 2.0
            + (CONFUSION_THRESHOLD  - confusion)  / 2.0
            + (SADNESS_THRESHOLD    - sadness)    / 2.0
        )
        signals_used = []

    # ── Escalation smoother ──────────────────────────────────────
    affect, conf, escalation_rule = apply_escalation_rules(
        raw_affect=affect,
        raw_conf=conf,
        affect_window=clean_window,
        all_signals_fired=(negative and repetitive and confused_kw),
    )

    if escalation_rule:
        signals_used.append(f"escalation:{escalation_rule}")

    return affect, round(conf, 4), signals_used, escalation_rule


def apply_escalation_rules(
    raw_affect: str,
    raw_conf: float,
    affect_window: Optional[List[str]],
    all_signals_fired: bool,
) -> Tuple[str, float, Optional[str]]:

    if not affect_window:
        return raw_affect, raw_conf, None

    window = affect_window[-WINDOW_SIZE:]

    # ── R4: disengaged after all-calm history → brevity, not disengagement ──
    if raw_affect == "disengaged":
        if all(a == "calm" for a in window):
            return "calm", raw_conf * 0.9, "R4_calm_history_not_disengaged"
        return raw_affect, raw_conf, None

    # ── Frustrated rules ────────────────────────────────────────────────────
    if raw_affect != "frustrated":
        return raw_affect, raw_conf, None

    non_calm_count = sum(1 for a in window if a != "calm")

    # R3: entirely calm history → frustrated not allowed
    if non_calm_count == 0:
        return "confused", raw_conf * 0.8, "R3_all_calm_history"

    # R1: insufficient trailing streak
    streak = _trailing_noncalm_streak(window)
    if streak < MIN_NONCALM_STREAK and not all_signals_fired:
        return "confused", raw_conf * 0.75, "R1_insufficient_streak"

    return raw_affect, raw_conf, None


# ── Context encoder ───────────────────────────────────────────────────────────

def encode_context_id(affect: str, clarity_level: int, pace: str) -> int:
    """
    Maps (affect, clarity_level, pace) to a unique integer in [0, N_CONTEXTS).

    Encoding:
        affect_idx (0-4) × 9  +  clarity_idx (0-2) × 3  +  pace_idx (0-2)
    """
    a = AFFECT_MAP.get(affect, AFFECT_MAP["calm"])
    c = CLARITY_MAP.get(clarity_level, 1)
    p = PACE_MAP.get(pace, 1)
    return a * 9 + c * 3 + p


def encode_context_features(affect: str, clarity_level: int, pace: str) -> np.ndarray:
    """
    Encodes state into a 7-dimensional feature vector for LinUCB.
    Vector structure: [One-Hot Affect (5), Clarity Level (1), Pace Value (1)]
    """
    # 1. One-hot encode the affect (5 dimensions)
    affect_vec = np.zeros(len(AFFECT_MAP), dtype=float)
    idx = AFFECT_MAP.get(affect, AFFECT_MAP["calm"])
    affect_vec[idx] = 1.0
    
    # 2. Numeric encoding for config (allows linear generalization)
    # Clarity is already 1, 2, or 3
    # Pace: slow=0.0, normal=1.0, fast=2.0
    pace_val = float(PACE_MAP.get(pace, 1))
    
    return np.concatenate([
        affect_vec, 
        [float(clarity_level)], 
        [pace_val]
    ])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _trailing_noncalm_streak(window: List[str]) -> int:
    """Count how many consecutive non-calm entries END the window."""
    streak = 0
    for entry in reversed(window):
        if entry != "calm":
            streak += 1
        else:
            break
    return streak


def _blend(a: float, b: float) -> float:
    return _clamp((a + b) / 2.0)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))