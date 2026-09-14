import numpy as np

def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a 1D query embedding (d_model,) 
    and 2D document embeddings (N, d_model).
    """
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    scores = np.dot(doc_norms, q_norm)
    return scores

def euclidean_distance(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distance between query and document embeddings.
    """
    dists = np.linalg.norm(doc_embs - query_emb, axis=1)
    return dists
