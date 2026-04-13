"""
Layer 3 — Contextual Bandit with UCB1

Tables:
  N[ctx][action]  — visit counts   (45 × 7)
  Q[ctx][action]  — average reward (45 × 7)

Context ID encoding (from state_classifier):
  affect_idx (0-4) × 9  +  clarity_idx (0-2) × 3  +  pace_idx (0-2)
  → 0..44  (45 unique contexts)

  Affect index mapping:
    0 = frustrated
    1 = confused
    2 = sad
    3 = calm
    4 = disengaged

Cold-start: if N[ctx] is all-zero, action selection falls back to
rule-based defaults (see RULE_BASED_DEFAULTS).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List

# FIX: import the single source of truth instead of re-defining here.
# N_CONTEXTS was previously defined in both bandit.py and state_classifier.py;
# keeping them in sync manually is error-prone.
from state_classifier import N_CONTEXTS, AFFECT_MAP

N_ACTIONS = 7

# Rule-based fallbacks indexed by affect_idx
# 0=frustrated, 1=confused, 2=sad, 3=calm, 4=disengaged
RULE_BASED_DEFAULTS = {
    0: 5,   # frustrated → DECREASE_CLARITY_AND_PACE
    1: 6,   # confused   → CLARITY_AND_CONFIRMATION
    2: 4,   # sad        → ENABLE_PATIENCE
    3: 0,   # calm       → DO_NOTHING
    4: 4,   # disengaged → ENABLE_PATIENCE
}

# FIX: actions that are permitted for the sad state.
# Sad users need empathy (patience_mode), not clarity/pace changes.
# Allowing the bandit to learn arbitrary actions for sad contexts risks
# it discovering spurious correlations (e.g. a user who calms down for
# unrelated reasons while clarity was also changed).
SAD_ALLOWED_ACTIONS = {0, 4}   # DO_NOTHING, ENABLE_PATIENCE

# affect_idx for calm and sad (used in guards below)
_CALM_IDX = AFFECT_MAP["calm"]   # 3
_SAD_IDX  = AFFECT_MAP["sad"]    # 2

GAMMA = 0.99  # decay factor — tune this per deployment

class LinUCBBandit:
    def __init__(self, A: np.ndarray, b: np.ndarray, alpha: float = 1.0, gamma: float = 0.99):
        """
        A: shape (n_actions, d, d) - Covariance matrices
        b: shape (n_actions, d)    - Reward-weighted feature vectors
        alpha: exploration strength (replaces the arbitrary '2' in UCB1)
        gamma: discount factor for non-stationary users
        """
        self.A = A.copy()
        self.b = b.copy()
        self.alpha = alpha
        self.gamma = gamma
        self.d = A.shape[1]
        self.n_actions = A.shape[0]

    def select_action(self, x: np.ndarray) -> tuple[int, list[float]]:
        p = np.zeros(self.n_actions)
        x = x.reshape(-1, 1)  # Ensure column vector
        
        for a in range(self.n_actions):
            # 1. Compute ridge regression coefficients: theta = A^-1 * b
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a].reshape(-1, 1)
            
            # 2. Predicted reward + UCB bonus
            # Bonus = alpha * sqrt(x.T * A_inv * x)
            uncertainty = np.sqrt(x.T @ A_inv @ x)
            p[a] = (theta.T @ x) + self.alpha * uncertainty
            
        action_id = int(np.argmax(p))
        return action_id, p.flatten().tolist()

    def update(self, x: np.ndarray, action_id: int, reward: float):
        x = x.reshape(-1, 1)
        # Apply discount factor to existing knowledge (forgetting)
        # We add (1-gamma)*I back to A to ensure it stays invertible
        self.A[action_id] = self.gamma * self.A[action_id] + (1 - self.gamma) * np.eye(self.d)
        self.b[action_id] = self.gamma * self.b[action_id]
        
        # Add new observation
        self.A[action_id] += x @ x.T
        self.b[action_id] += (reward * x).flatten()