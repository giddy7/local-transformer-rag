import sys
import argparse
from src.rag.rag_pipeline import RAGPipeline

def run_chat(preset: str = "SMALL", top_k: int = 3, show_context: bool = False, decoding: str = "greedy"):
    print("==================================================")
    print("      LOCAL TRANSFORMER RAG SYSTEM CHAT          ")
    print("==================================================")
    print(f"Preset: {preset} | Decoding: {decoding} | Top-K: {top_k}")
    print("Type 'exit' or 'quit' to end the session.\n")

    print("[*] Initializing local PyTorch Transformer RAG Pipeline...")
    pipeline = RAGPipeline(preset=preset)
    print("[OK] Local RAG System Ready!\n")

    while True:
        try:
            user_input = input("Question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting RAG system chat. Goodbye!")
                break

            print("\nRetrieving relevant documents...")
            response = pipeline.answer_question(user_input, top_k=top_k, show_context=show_context, decoding=decoding)

            print("\nTop Retrieved Sources:")
            for src in response["sources"]:
                rank = src["rank"]
                doc_name = src["document_name"]
                page = src["page"]
                score = src["score"]
                print(f"  {rank}. {doc_name} -- Page {page} (Similarity Score: {score:.4f})")

            if show_context and response["context_used"]:
                print("\n[Retrieved Context]")
                print(response["context_used"])

            print("\nGenerating answer...")
            print("\nAnswer:")
            print(f"  {response['answer']}\n")

            print("Sources:")
            for src in response["sources"]:
                print(f"  [{src['rank']}] {src['document_name']} -- Page {src['page']}")
            print("-" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting RAG system chat. Goodbye!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Local PyTorch Transformer RAG Chat")
    parser.add_argument("--preset", type=str, default="SMALL", choices=["SMALL", "MEDIUM", "LARGE"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--show-context", action="store_true", help="Display retrieved context string")
    parser.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    args = parser.parse_args()

    run_chat(preset=args.preset, top_k=args.top_k, show_context=args.show_context, decoding=args.decoding)
