import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Return the expected value of the discrete distribution.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    
    return float(np.dot(x, p))