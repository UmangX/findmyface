import numpy as np

def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))

def is_match(emb1, emb2, threshold=0.6, metric="cosine"):
    if metric == "cosine":
        return cosine_similarity(emb1, emb2) > threshold
    else:
        return euclidean_distance(emb1, emb2) < threshold