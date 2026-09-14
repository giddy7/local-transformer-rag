import numpy as np
from typing import List

def compute_recall_at_k(retrieved_ids: List[List[str]], target_ids: List[str], k: int) -> float:
    hits = 0
    total = len(target_ids)
    if total == 0:
        return 0.0

    for top_k_retrieved, target in zip(retrieved_ids, target_ids):
        if target in top_k_retrieved[:k]:
            hits += 1
    return hits / total

def compute_mrr(retrieved_ids: List[List[str]], target_ids: List[str]) -> float:
    rr_list = []
    for top_retrieved, target in zip(retrieved_ids, target_ids):
        if target in top_retrieved:
            rank = top_retrieved.index(target) + 1
            rr_list.append(1.0 / rank)
        else:
            rr_list.append(0.0)
    return float(np.mean(rr_list)) if rr_list else 0.0

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    matches = [1.0 if p.strip().lower() == r.strip().lower() else 0.0 for p, r in zip(predictions, references)]
    return float(np.mean(matches)) if matches else 0.0

def compute_simple_bleu(prediction: str, reference: str, max_n: int = 2) -> float:
    """Simple n-gram BLEU approximation."""
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
        if not pred_ngrams:
            continue
        ref_count = {}
        for ng in ref_ngrams:
            ref_count[ng] = ref_count.get(ng, 0) + 1
        
        matches = 0
        for ng in pred_ngrams:
            if ref_count.get(ng, 0) > 0:
                matches += 1
                ref_count[ng] -= 1
        precisions.append(matches / len(pred_ngrams))

    if not precisions:
        return 0.0
    return float(np.exp(np.mean(np.log(np.array(precisions) + 1e-10))))
