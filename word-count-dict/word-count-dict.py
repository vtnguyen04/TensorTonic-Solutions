from collections import Counter

def word_count_dict(sentences):
    counts = Counter()
    
    for sentence in sentences:
        counts.update(sentence) 
        
    return dict(counts)