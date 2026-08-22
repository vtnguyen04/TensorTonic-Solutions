import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Pad or truncate sequences to fixed length and return a 2D NumPy array.
    """
    N = len(seqs)
    if N == 0:
        return np.empty((0, 0), dtype=int)

    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

        
    result = np.full((N, max_len), pad_value, dtype=int)
    
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]
        if len(trunc) > 0:
            result[i, :len(trunc)] = trunc
            
    return result