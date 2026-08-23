import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    N = len(documents)
    
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    vocab_set = {word for doc in tokenized_docs for word in doc}
    
    vocabulary = sorted(list(vocab_set))
    vocab_idx = {word: i for i, word in enumerate(vocabulary)}
    V = len(vocabulary)
    
    df = np.zeros(V, dtype=float)
    for tokens in tokenized_docs:
        for word in set(tokens):
            df[vocab_idx[word]] += 1
            
    idf = np.log(N / df)
    
    tf_matrix = np.zeros((N, V), dtype=float)
    for i, tokens in enumerate(tokenized_docs):
        doc_len = len(tokens)
        if doc_len > 0:
            counts = Counter(tokens)
            for word, count in counts.items():
                tf_matrix[i, vocab_idx[word]] = count / doc_len
                
    tfidf_matrix = tf_matrix * idf
    
    return {
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary
    }