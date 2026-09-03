import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    mask = z >= 0
    out[mask] = 1 / (1 + np.exp(-z[mask]))
    out[~mask] = np.exp(z[~mask]) / (1 + np.exp(z[~mask]))
    return out
    
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here

    N, D = X.shape
    w = np.zeros(D)
    b = 0
    for _ in range(steps):
        logit = X @ w + b
        pred = _sigmoid(logit)

        w_grad = X.T @ (pred - y) / N
        b_grad = np.mean(pred - y)
        w -= lr * w_grad
        b -= lr * b_grad
    return w, b