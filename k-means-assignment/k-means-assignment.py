import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    Must return a list of integers.
    """
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    
    diff = points[:, np.newaxis, :] - centroids
    distances = np.sum(diff ** 2, axis=2)
    
    return np.argmin(distances, axis=1).tolist()