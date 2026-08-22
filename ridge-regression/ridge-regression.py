import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution:
    w = (X^T X + lam * I)^(-1) X^T y
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    d = X.shape[1]
    
    I = np.eye(d)
    
    A = X.T @ X + lam * I
    
    b = X.T @ y
    
    w = np.linalg.solve(A, b)
    
    return w