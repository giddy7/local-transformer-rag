import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from src.vectorstore.similarity import cosine_similarity

class LocalVectorIndex:
    """
    Local Vector Search Index using NumPy array storage and Cosine Similarity.
    """
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.embeddings: np.ndarray = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        assert len(embeddings) == len(metadata), "Embeddings and metadata must have same length"
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.metadata = metadata

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        scores = cosine_similarity(query_emb, self.embeddings)
        top_k = min(top_k, len(self.embeddings))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.metadata[idx], float(scores[idx])))
        return results

    def save(self):
        if self.embeddings is None:
            print("[!] Warning: No embeddings to save.")
            return

        emb_file = os.path.join(self.index_dir, "document_embeddings.npy")
        meta_file = os.path.join(self.index_dir, "document_metadata.json")

        np.save(emb_file, self.embeddings)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"[OK] Vector index saved ({len(self.embeddings)} items) to: {self.index_dir}")

    def load(self) -> bool:
        emb_file = os.path.join(self.index_dir, "document_embeddings.npy")
        meta_file = os.path.join(self.index_dir, "document_metadata.json")

        if not (os.path.exists(emb_file) and os.path.exists(meta_file)):
            return False

        self.embeddings = np.load(emb_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[OK] Loaded local vector index with {len(self.embeddings)} embeddings from: {self.index_dir}")
        return True
