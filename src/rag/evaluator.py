import json
from typing import List, Dict, Any
from src.rag.rag_pipeline import RAGPipeline
from src.utils.metrics import compute_recall_at_k, compute_mrr, compute_exact_match, compute_simple_bleu

class RAGEvaluator:
    """
    Evaluator for end-to-end RAG system performance.
    """
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def evaluate_dataset(self, test_data: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, float]:
        print(f"[*] Evaluating Complete RAG System on {len(test_data)} test cases...", flush=True)

        retrieved_ids_list = []
        target_ids_list = []
        generated_answers = []
        target_answers = []

        for item in test_data:
            question = item["question"]
            target_answer = item.get("answer", "")
            target_chunk_id = item.get("target_chunk_id", item.get("chunk_id", ""))

            res = self.pipeline.answer_question(question, top_k=top_k)
            
            top_retrieved_ids = [s["chunk_id"] for s in res["sources"]]
            retrieved_ids_list.append(top_retrieved_ids)
            target_ids_list.append(target_chunk_id)

            generated_answers.append(res["answer"])
            target_answers.append(target_answer)

        # Retrieval metrics
        recall_at_1 = compute_recall_at_k(retrieved_ids_list, target_ids_list, k=1)
        recall_at_3 = compute_recall_at_k(retrieved_ids_list, target_ids_list, k=3)
        mrr = compute_mrr(retrieved_ids_list, target_ids_list)

        # Generation metrics
        em_score = compute_exact_match(generated_answers, target_answers)
        bleu_scores = [compute_simple_bleu(g, t) for g, t in zip(generated_answers, target_answers)]
        mean_bleu = float(sum(bleu_scores) / len(bleu_scores)) if bleu_scores else 0.0

        results = {
            "Retrieval_Recall@1": recall_at_1,
            "Retrieval_Recall@3": recall_at_3,
            "Retrieval_MRR": mrr,
            "Generator_ExactMatch": em_score,
            "Generator_BLEU": mean_bleu,
        }

        print("=========================================")
        print("COMPLETE RAG SYSTEM EVALUATION RESULTS")
        print("=========================================")
        for metric, val in results.items():
            print(f"  {metric:<22}: {val:.4f}")
        print("=========================================\n")
        return results
