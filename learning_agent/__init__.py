"""
learning_agent package — black box, zero modifications.

The original Learning Agent files use bare imports such as:
    from state_classifier import N_CONTEXTS, AFFECT_MAP
    from schemas import CurrentConfig

These work fine when the files live in the project root, but fail when
they are inside a sub-package (learning_agent/).  This __init__.py
registers every sub-module under its bare name in sys.modules BEFORE
any of them are imported, so the intra-package bare imports resolve
correctly without touching any source file.
"""
import importlib
import sys

_MODULES = [
    "schemas",
    "nlp_layer",
    "state_classifier",
    "bandit",
    "config_applier",
    "storage",
]

for _name in _MODULES:
    _full = f"learning_agent.{_name}"
    # Import the fully-qualified module first so Python resolves the file.
    _mod = importlib.import_module(_full)
    # Register it under the bare name so intra-package bare imports work.
    if _name not in sys.modules:
        sys.modules[_name] = _mod