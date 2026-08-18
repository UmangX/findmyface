import numpy as np
import pandas as pd 
import sqlite3 

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

def getDf(db):
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query("SELECT * FROM files;", conn)

def findSimilar(index):
    faceDf = getDf("./static/faces.db")
    targetemb = np.frombuffer(
        faceDf.iloc[index]["embeddings"],
        dtype=np.float32
    )
    print("Target shape:", targetemb.shape)
    for i, row in faceDf.iterrows():
        if i == index:
            continue
        emb = np.frombuffer(
            row["embeddings"],
            dtype=np.float32
        )
        similarity = cosine_similarity(targetemb, emb)
        print(f"{i}: {similarity:.4f}")
    
if __name__ == "__main__":
    findSimilar(1)
