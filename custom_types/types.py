"""Shared types definitions for the heat rate options package."""

import numpy as np
import numpy.typing as npt

# Type aliases for cleaner signatures
ArrayLike = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]


def as_array(x: ArrayLike) -> FloatArray:
    """Convert scalar or array-like to float array, preserving shape."""
    return np.asarray(x, dtype=np.float64)


def as_1d(x: ArrayLike) -> FloatArray:
    """Convert scalar or array-like to 1D float array.

    Scalars become arrays of shape (1,).
    """
    a = np.asarray(x, dtype=np.float64)
    return a if a.ndim > 0 else a[None]