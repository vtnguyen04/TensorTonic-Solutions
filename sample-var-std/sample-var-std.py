import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x, dtype=float)
    var = float(np.var(x, ddof=1))
    std = float(np.std(x, ddof=1)) # hoặc float(np.sqrt(var))
    return var, std