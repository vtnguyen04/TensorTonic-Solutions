import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation: w = (X^T X)^(-1) X^T y
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    XtX = X.T @ X
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    
    return w