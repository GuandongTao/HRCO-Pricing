import numpy as np
from custom_types.types import FloatArray


def cholesky2(rho: float) -> FloatArray:
    """
    Compute 2x2 Cholesky decomposition for correlation matrix [[1, rho], [rho, 1]].

    Args:
        rho: Correlation coefficient between -1 and 1

    Returns:
        Lower triangular Cholesky factor L such that L @ L.T gives correlation matrix
    """
    # Ensure valid correlation (handle numerical edge cases)
    rho_safe = np.clip(rho, -1.0, 1.0)

    # Cholesky decomposition: [[1, 0], [rho, sqrt(1-rho^2)]]
    L = np.array([
        [1.0, 0.0],
        [rho_safe, np.sqrt(max(0.0, 1.0 - rho_safe * rho_safe))]
    ], dtype=np.float64)

    return L