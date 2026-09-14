import torch
import numpy as np
from typing import List, Dict, Any
from src.retriever.retriever import DenseRetriever
from src.utils.metrics import compute_recall_at_k, compute_mrr

def evaluate_retriever(retriever: DenseRetriever, eval_data: List[Dict[str, Any]], all_docs: List[Dict[str, Any]], top_k_list=[1, 3, 5, 10]):
    print(f"[*] Evaluating Retriever on {len(eval_data)} queries against {len(all_docs)} documents...")
    
    # 1. Encode all document chunks
    doc_texts = [d["text"] for d in all_docs]
    doc_ids = [d["chunk_id"] for d in all_docs]
    doc_embeddings = retriever.encode_batch(doc_texts, is_doc=True)

    retrieved_chunk_ids = []
    target_chunk_ids = []

    # 2. Query retrieval
    for item in eval_data:
        question = item["question"]
        target_chunk_id = item.get("positive_chunk_id", item.get("target_id", ""))

        q_emb = retriever.encode_text(question)
        # Cosine similarity (since normalized: dot product)
        sim_scores = np.dot(doc_embeddings, q_emb)
        top_indices = np.argsort(sim_scores)[::-1][:max(top_k_list)]
        
        top_chunks = [doc_ids[idx] for idx in top_indices]
        retrieved_chunk_ids.append(top_chunks)
        target_chunk_ids.append(target_chunk_id)

    # 3. Calculate metrics
    results = {}
    for k in top_k_list:
        results[f"Recall@{k}"] = compute_recall_at_k(retrieved_chunk_ids, target_chunk_ids, k=k)
    results["MRR"] = compute_mrr(retrieved_chunk_ids, target_chunk_ids)

    print("-----------------------------------------")
    print("RETRIEVER EVALUATION RESULTS:")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")
    print("-----------------------------------------")
    return results
