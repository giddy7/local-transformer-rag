import os
import json
import argparse
from config import TOKENIZER_DIR, PROCESSED_DATA_DIR, TRAINING_DATA_DIR, TokenizerConfig
from src.tokenizer.tokenizer import LocalTokenizer

def train_tokenizer(vocab_size=None, save_dir=TOKENIZER_DIR):
    cfg = TokenizerConfig()
    v_size = vocab_size or cfg.vocab_size

    # Collect texts from chunks and training datasets
    corpus = []
    chunks_file = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    if os.path.exists(chunks_file):
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            corpus.extend([c["text"] for c in chunks])

    train_qa_file = os.path.join(TRAINING_DATA_DIR, "qa_pairs.json")
    if os.path.exists(train_qa_file):
        with open(train_qa_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
            for item in qa_pairs:
                corpus.append(item.get("question", ""))
                corpus.append(item.get("context", ""))
                corpus.append(item.get("answer", ""))

    # Fallback corpus if empty
    if not corpus:
        corpus = [
            "Artificial Intelligence is the simulation of human intelligence by machines.",
            "Machine learning is a subset of AI that allows systems to learn from data.",
            "Deep learning uses artificial neural networks with multiple layers.",
            "Retrieval-Augmented Generation combines document retrieval with text generation.",
            "PyTorch is an open source machine learning library for Python.",
            "Natural Language Processing allows computers to understand text.",
            "Transformers use multi-head self-attention mechanisms for sequence modelling.",
            "Vector databases store high-dimensional embeddings for similarity search."
        ]

    print(f"[*] Training tokenizer on {len(corpus)} text snippets with vocab size {v_size}...")
    tokenizer = LocalTokenizer(vocab_size=v_size)
    tokenizer.train_from_iterator(corpus, save_path=save_dir)
    print(f"[OK] Tokenizer trained and saved to: {save_dir}")
    print(f"[+] Total vocabulary size: {len(tokenizer)}")
    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()
    train_tokenizer(vocab_size=args.vocab_size)
