import numpy as np
from collections import Counter

def bag_of_words_vector(tokens: list[str], vocab: list[str]) -> np.ndarray:
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    token_counts = Counter(tokens)
    
    vec = [token_counts.get(word, 0) for word in vocab]
    
    return np.array(vec, dtype=int)