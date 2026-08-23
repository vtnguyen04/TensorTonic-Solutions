import numpy as np

def bag_of_words_vector(tokens: list[str], vocab: list[str]) -> np.ndarray:
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=int)
    
    for token in tokens:
        if token in word_to_idx:
            vec[word_to_idx[token]] += 1
            
    return vec