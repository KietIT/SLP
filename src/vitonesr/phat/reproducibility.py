from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch and return the applied settings."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if torch.cuda.is_available() and deterministic:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return {
        "seed": seed,
        "deterministic": deterministic,
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
    }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
