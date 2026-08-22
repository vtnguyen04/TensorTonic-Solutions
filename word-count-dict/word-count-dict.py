def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    counts = {}
    
    for sentence in sentences:
        tokens = sentence.split() if isinstance(sentence, str) else sentence
        
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
            
    return counts