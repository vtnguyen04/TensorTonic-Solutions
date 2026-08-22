import numpy as np

def knn_distance(X_train, X_test, k: int) -> np.ndarray:
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 1:
        X_test = X_test[:, np.newaxis]
        
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2) 
    
    sorted_indices = np.argsort(distances, axis=1)
    
    result = np.full((n_test, k), -1, dtype=int)
    
    num_valid = min(k, n_train)
    result[:, :num_valid] = sorted_indices[:, :num_valid]
    
    return result