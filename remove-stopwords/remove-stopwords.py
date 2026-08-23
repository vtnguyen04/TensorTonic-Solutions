def remove_stopwords(tokens: list[str], stopwords: list[str] | set[str]) -> list[str]:
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    stopwords_set = set(stopwords)
    
    return [word for word in tokens if word not in stopwords_set]