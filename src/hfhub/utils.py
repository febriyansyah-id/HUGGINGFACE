"""General-purpose convenience helpers."""
from __future__ import annotations

import random
from collections.abc import Mapping


def seed_everything(seed: int = 42):
    """Seed numpy, torch, and builtins.random for reproducibility."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def flatten_nested(config: Mapping, sep: str = ".") -> dict:
    """Recursively collapse nested maps into flat dotted-string keys."""
    flattened: dict = {}

    def descend(submap: Mapping, trail: str = "") -> None:
        for key, value in submap.items():
            joined = f"{trail}{sep}{key}" if trail else str(key)
            if isinstance(value, Mapping):
                descend(value, joined)
            else:
                flattened[joined] = value

    descend(config)
    return flattened
