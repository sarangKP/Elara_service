"""
Config Applier

Maps action_id → config delta, enforces the "max 1 step, never skip 2 levels" rule.
Returns (new_config_dict, changes_dict, reason_str).
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
from schemas import CurrentConfig

# Ordered sequences for step-clamped changes
PACE_STEPS = ["slow", "normal", "fast"]
CLARITY_STEPS = [1, 2, 3]
CONFIRM_STEPS = ["low", "medium", "high"]

# Action IDs
DO_NOTHING               = 0
DECREASE_CLARITY         = 1
DECREASE_PACE            = 2
INCREASE_CONFIRMATION    = 3
ENABLE_PATIENCE          = 4
DECREASE_CLARITY_AND_PACE = 5
CLARITY_AND_CONFIRMATION  = 6

ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "DECREASE_CLARITY",
    2: "DECREASE_PACE",
    3: "INCREASE_CONFIRMATION",
    4: "ENABLE_PATIENCE",
    5: "DECREASE_CLARITY_AND_PACE",
    6: "CLARITY_AND_CONFIRMATION",
}

AFFECT_REASON_MAP = {
    "frustrated": "affect_frustrated_detected",
    "confused":   "affect_confused_detected",
    "sad":        "affect_sad_detected",
    "calm":       "affect_calm_no_change",
}

# Default (baseline) config — recovery target when user is calm
CONFIG_DEFAULTS = {
    "pace":                   "normal",
    "clarity_level":          2,
    "confirmation_frequency": "low",
    "patience_mode":          False,
}


def apply_action(
    action_id: int, cfg: CurrentConfig, affect: str
) -> Tuple[CurrentConfig, Dict[str, Any], str]:
    """
    Returns (updated_config, changes_dict, reason).
    changes_dict is empty when nothing changes.
    """
    changes: Dict[str, Any] = {}

    pace       = cfg.pace
    clarity    = cfg.clarity_level
    confirm    = cfg.confirmation_frequency
    patience   = cfg.patience_mode

    if action_id == DO_NOTHING:
        # When calm, nudge config one step back toward defaults (gradual recovery).
        # Only one parameter recovers per call so the change isn't jarring.
        if affect == "calm":
            if patience and CONFIG_DEFAULTS["patience_mode"] == False:
                patience = False
                changes["patience_mode"] = False
            elif confirm != CONFIG_DEFAULTS["confirmation_frequency"]:
                new_cf = _step_toward(confirm, CONFIG_DEFAULTS["confirmation_frequency"], CONFIRM_STEPS)
                if new_cf != confirm:
                    confirm = new_cf
                    changes["confirmation_frequency"] = confirm
            elif clarity != CONFIG_DEFAULTS["clarity_level"]:
                new_c = _step_toward(clarity, CONFIG_DEFAULTS["clarity_level"], CLARITY_STEPS)
                if new_c != clarity:
                    clarity = new_c
                    changes["clarity_level"] = clarity
            elif pace != CONFIG_DEFAULTS["pace"]:
                new_p = _step_toward(pace, CONFIG_DEFAULTS["pace"], PACE_STEPS)
                if new_p != pace:
                    pace = new_p
                    changes["pace"] = pace

    elif action_id == DECREASE_CLARITY:
        new_c = _step_down(clarity, CLARITY_STEPS)
        if new_c != clarity:
            clarity = new_c
            changes["clarity_level"] = clarity

    elif action_id == DECREASE_PACE:
        new_p = _step_down(pace, PACE_STEPS)
        if new_p != pace:
            pace = new_p
            changes["pace"] = pace

    elif action_id == INCREASE_CONFIRMATION:
        new_cf = _step_up(confirm, CONFIRM_STEPS)
        if new_cf != confirm:
            confirm = new_cf
            changes["confirmation_frequency"] = confirm

    elif action_id == ENABLE_PATIENCE:
        if not patience:
            patience = True
            changes["patience_mode"] = True

    elif action_id == DECREASE_CLARITY_AND_PACE:
        new_c = _step_down(clarity, CLARITY_STEPS)
        new_p = _step_down(pace, PACE_STEPS)
        if new_c != clarity:
            clarity = new_c
            changes["clarity_level"] = clarity
        if new_p != pace:
            pace = new_p
            changes["pace"] = pace

    elif action_id == CLARITY_AND_CONFIRMATION:
        new_c = _step_down(clarity, CLARITY_STEPS)
        new_cf = _step_up(confirm, CONFIRM_STEPS)
        if new_c != clarity:
            clarity = new_c
            changes["clarity_level"] = clarity
        if new_cf != confirm:
            confirm = new_cf
            changes["confirmation_frequency"] = confirm

    new_cfg = CurrentConfig(
        pace=pace,
        clarity_level=clarity,
        confirmation_frequency=confirm,
        patience_mode=patience,
    )

    reason = AFFECT_REASON_MAP.get(affect, f"action_{ACTION_NAMES.get(action_id, str(action_id))}")
    if not changes:
        reason = "no_change_needed_or_already_at_limit"
    elif affect == "calm" and changes:
        reason = "calm_recovery_step"

    return new_cfg, changes, reason


# ── Helpers ───────────────────────────────────────────────────────

def _step_down(current, steps: list):
    idx = steps.index(current) if current in steps else 1
    return steps[max(0, idx - 1)]


def _step_up(current, steps: list):
    idx = steps.index(current) if current in steps else 1
    return steps[min(len(steps) - 1, idx + 1)]


def _step_toward(current, target, steps: list):
    """Move one step from current toward target."""
    if current not in steps or target not in steps:
        return current
    ci = steps.index(current)
    ti = steps.index(target)
    if ci < ti:
        return steps[ci + 1]
    elif ci > ti:
        return steps[ci - 1]
    return current