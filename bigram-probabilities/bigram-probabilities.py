def bigram_probabilities(tokens: list[str]) -> tuple[dict, dict]:
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count for observed bigrams
      probs:  dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing for all (w1, w2) in V x V
    """
    if not tokens:
        return {}, {}
    
    vocab = set(tokens)
    V = len(vocab)
    
    counts = {}
    context_counts = {w: 0 for w in vocab}
    
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        pair = (w1, w2)
        counts[pair] = counts.get(pair, 0) + 1
        context_counts[w1] += 1
        
    probs = {}
    for w1 in vocab:
        denom = context_counts[w1] + V  
        for w2 in vocab:
            pair = (w1, w2)
            c = counts.get(pair, 0)
            probs[pair] = (c + 1) / denom
            
    return counts, probs