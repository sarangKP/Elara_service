"""
Storage layer — loads/saves LinUCB bandit matrices.
Now uses (A, b) matrices instead of (N, Q) tables.
"""

from __future__ import annotations
import fcntl
import os
import threading
from contextlib import contextmanager
from typing import Generator, Tuple

import numpy as np

# LinUCB Dimensions: 7 actions, 7 features (5 affects + clarity + pace)
N_ACTIONS = 7
N_FEATURES = 7

TABLE_DIR = os.environ.get("BANDIT_TABLE_DIR", "tables")
A_PATH    = os.path.join(TABLE_DIR, "bandit_A.npy")
B_PATH    = os.path.join(TABLE_DIR, "bandit_b.npy")
LOCK_PATH = os.path.join(TABLE_DIR, ".bandit.lock")   # advisory lock file

# In-process mutex — fast path for single-worker deployments
_thread_lock = threading.Lock()


# ── Public context manager ────────────────────────────────────────────────────

@contextmanager
def tables_locked() -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Acquire both the in-process mutex and the file-level advisory lock, then
    yield (A, b). The caller mutates the matrices and returns the updated
    versions; those are saved before the locks are released.
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    with _thread_lock:
        lock_file = open(LOCK_PATH, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            A, b = _load()
            yield A, b
            _save(A, b)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load() -> Tuple[np.ndarray, np.ndarray]:
    """Load matrices from disk; returns identity A and zero b on missing/stale files."""
    if os.path.exists(A_PATH) and os.path.exists(B_PATH):
        A = np.load(A_PATH)
        b = np.load(B_PATH)
        
        # Validate shapes match current LinUCB configuration
        expected_A_shape = (N_ACTIONS, N_FEATURES, N_FEATURES)
        expected_b_shape = (N_ACTIONS, N_FEATURES)
        
        if A.shape != expected_A_shape or b.shape != expected_b_shape:
            print(f"[storage] WARNING: matrix shapes {A.shape}/{b.shape} mismatch. Re-initialising.")
            A, b = _init_matrices()
    else:
        A, b = _init_matrices()
    return A, b


def _save(A: np.ndarray, b: np.ndarray) -> None:
    """Atomic save of the current bandit state."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    np.save(A_PATH, A)
    np.save(B_PATH, b)


def _init_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """Initialize A as identity matrices (required for inversion) and b as zeros."""
    A = np.array([np.eye(N_FEATURES) for _ in range(N_ACTIONS)])
    b = np.zeros((N_ACTIONS, N_FEATURES))
    return A, b


# ── Convenience wrappers (kept for test compatibility) ────────────────────────

def load_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Direct load without locking — use only in tests."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    return _load()


def save_tables(A: np.ndarray, b: np.ndarray) -> None:
    """Direct save without locking — use only in tests."""
    _save(A, b)


def reset_tables() -> None:
    """Wipe and reinitialise matrices — useful for tests."""
    A, b = _init_matrices()
    _save(A, b)