import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return float(np.dot(x, y))
