import numpy as np

def k_means_centroid_update(points, assignments, k):
    points = np.asarray(points, dtype=float)
    assignments = np.asarray(assignments, dtype=int)
    
    if points.ndim == 1:
        points = points[:, np.newaxis]
    
    # assignments shape (n,), np.arange(k)[:, None] shape (k, 1)
    mask = (assignments == np.arange(k)[:, None])  # shape (k, n)
    
    counts = np.sum(mask, axis=1, keepdims=True)  
    sums = mask @ points                          
    
    new_centroids = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    
    return new_centroids.tolist()