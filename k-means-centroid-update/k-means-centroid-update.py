import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.asarray(points, dtype=float)
    assignments = np.asarray(assignments, dtype=int)
    
    if points.ndim == 1:
        points = points[:, np.newaxis]
        
    d = points.shape[1]
    
    new_centroids = np.zeros((k, d), dtype=float)
    
    for cluster_id in range(k):
        
        new_centroids[cluster_id] = np.mean(points[assignments == cluster_id], axis=0)

    return new_centroids.tolist()