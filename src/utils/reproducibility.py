"""Reproducibility utilities: seed management for all frameworks."""

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def set_seed(seed: int, use_cuda: bool = True) -> None:
    """Set random seed for reproducibility across all frameworks.
    
    Args:
        seed: Random seed value
        use_cuda: Whether to seed CUDA operations
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    if torch is not None:
        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_random_seeds(n_seeds: int, base_seed: int = 42) -> list[int]:
    """Generate N reproducible random seeds.
    
    Args:
        n_seeds: Number of seeds to generate
        base_seed: Base seed for RNG
        
    Returns:
        List of N reproducible random seeds
    """
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31 - 1, n_seeds).tolist()
