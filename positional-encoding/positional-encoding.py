import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here

    pe = np.zeros((seq_len, d_model), dtype = np.float32)

    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(base) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])

    return pe