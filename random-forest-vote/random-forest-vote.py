import numpy as np

def random_forest_vote(predictions) -> list[int]:
    """
    Compute the majority vote from multiple tree predictions.
    """
    preds = np.asarray(predictions, dtype=int)  # Shape: (n_trees, n_samples)
    
    result = []
    for col in preds.T:
        labels, counts = np.unique(col, return_counts=True)
        
        best_label = labels[np.argmax(counts)]
        result.append(int(best_label))
        
    return result