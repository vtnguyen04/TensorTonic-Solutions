import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    a = np.array(A, dtype=float)
    a = a.T
    return a
    # Write code here
