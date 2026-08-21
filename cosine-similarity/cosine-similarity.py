import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """Return the cosine similarity of a and b."""

    a = np.array(a, dtype = np.float32)
    b = np.array(b, dtype = np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
