import os
import json
import argparse
from config import TRAINING_DATA_DIR
from src.rag.rag_pipeline import RAGPipeline
from src.rag.evaluator import RAGEvaluator

def run_evaluation(preset: str = "SMALL"):
    qa_file = os.path.join(TRAINING_DATA_DIR, "qa_pairs.json")
    if not os.path.exists(qa_file):
        from prepare_data import prepare_sample_data
        prepare_sample_data()

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    pipeline = RAGPipeline(preset=preset)
    evaluator = RAGEvaluator(pipeline)
    results = evaluator.evaluate_dataset(qa_data)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument("--preset", type=str, default="SMALL", choices=["SMALL", "MEDIUM", "LARGE"])
    args = parser.parse_args()

    run_evaluation(preset=args.preset)
