"""
Storage layer — loads/saves LinUCB bandit matrices.

CHANGES (audit fixes):
  - FIX #1 (Critical): Per-user bandit matrix files.
    Each user now gets their own A/b matrix files, keyed by user_id.
    Global shared learning is eliminated.
  - FIX #6 (Medium): Reward floor/ceiling applied on load.
    b-vector values are clipped to [-2, +2] to prevent the bandit
    from learning extreme values after a run of bad sessions.
"""

from __future__ import annotations
import fcntl
import os
import re
import threading
from contextlib import contextmanager
from typing import Generator, Tuple

import numpy as np

# LinUCB Dimensions: 7 actions, 7 features (5 affects + clarity + pace)
N_ACTIONS  = 7
N_FEATURES = 7

# Reward vector clipping bounds — prevents extreme learned values (Fix #6)
B_CLIP_MIN = -2.0
B_CLIP_MAX  =  2.0

TABLE_DIR = os.environ.get("BANDIT_TABLE_DIR", "tables")

# In-process mutex — fast path for single-worker deployments
_thread_lock = threading.Lock()


# ── Internal path helpers ─────────────────────────────────────────────────────

def _sanitise_user_id(user_id: str) -> str:
    """
    Strip anything that isn't alphanumeric, hyphen, or underscore so that
    user_id can be safely embedded in a filename.
    Falls back to 'default' if the result is empty.
    """
    safe = re.sub(r"[^\w\-]", "_", user_id)
    return safe if safe else "default"


def _paths(user_id: str) -> Tuple[str, str, str]:
    """Return (A_path, b_path, lock_path) for the given user."""
    uid  = _sanitise_user_id(user_id)
    base = os.path.join(TABLE_DIR, uid)
    return (
        f"{base}_bandit_A.npy",
        f"{base}_bandit_b.npy",
        f"{base}_bandit.lock",
    )


# ── Public context manager ────────────────────────────────────────────────────

@contextmanager
def tables_locked(
    user_id: str = "default",
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Acquire both the in-process mutex and the file-level advisory lock for
    *this user*, then yield (A, b).

    The caller mutates the matrices and the updated versions are saved before
    the locks are released.

    Args:
        user_id: Unique identifier for the user (e.g. session_id prefix,
                 username, or UUID). Defaults to "default" so existing call
                 sites that don't pass user_id keep working.
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    a_path, b_path, lock_path = _paths(user_id)

    with _thread_lock:
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            A, b = _load(a_path, b_path)
            yield A, b
            _save(A, b, a_path, b_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load(a_path: str, b_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load matrices from disk; returns identity A and zero b on missing/stale files."""
    if os.path.exists(a_path) and os.path.exists(b_path):
        A = np.load(a_path)
        b = np.load(b_path)

        expected_A_shape = (N_ACTIONS, N_FEATURES, N_FEATURES)
        expected_b_shape = (N_ACTIONS, N_FEATURES)

        if A.shape != expected_A_shape or b.shape != expected_b_shape:
            print(
                f"[storage] WARNING: matrix shapes {A.shape}/{b.shape} "
                f"mismatch expected {expected_A_shape}/{expected_b_shape}. "
                f"Re-initialising."
            )
            A, b = _init_matrices()
        else:
            # FIX #6 — clip reward vector to prevent extreme learned values
            b = np.clip(b, B_CLIP_MIN, B_CLIP_MAX)
    else:
        A, b = _init_matrices()
    return A, b


def _save(A: np.ndarray, b: np.ndarray, a_path: str, b_path: str) -> None:
    """Atomic save of the current bandit state."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    np.save(a_path, A)
    np.save(b_path, b)


def _init_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """Initialise A as identity matrices (required for inversion) and b as zeros."""
    A = np.array([np.eye(N_FEATURES) for _ in range(N_ACTIONS)])
    b = np.zeros((N_ACTIONS, N_FEATURES))
    return A, b


# ── Convenience wrappers (kept for test compatibility) ────────────────────────

def load_tables(user_id: str = "default") -> Tuple[np.ndarray, np.ndarray]:
    """Direct load without locking — use only in tests."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    a_path, b_path, _ = _paths(user_id)
    return _load(a_path, b_path)


def save_tables(A: np.ndarray, b: np.ndarray, user_id: str = "default") -> None:
    """Direct save without locking — use only in tests."""
    a_path, b_path, _ = _paths(user_id)
    _save(A, b, a_path, b_path)


def reset_tables(user_id: str = "default") -> None:
    """Wipe and reinitialise matrices — useful for tests."""
    A, b = _init_matrices()
    a_path, b_path, _ = _paths(user_id)
    _save(A, b, a_path, b_path)