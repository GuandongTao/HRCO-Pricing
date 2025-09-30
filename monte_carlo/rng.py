import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """
    Create a numpy random number generator for reproducibility.

    Args:
        seed: Random seed for reproducibility. None for non-deterministic.

    Returns:
        numpy Generator instance
    """
    return np.random.default_rng(seed)