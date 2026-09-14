import os
import json
import argparse
import torch

from config import (
    PROCESSED_DATA_DIR, INDEXES_DIR, TOKENIZER_DIR,
    RETRIEVER_CHECKPOINT_DIR, PRESETS, ModelConfig
)
from src.tokenizer.tokenizer import LocalTokenizer
from src.retriever.embedding_model import DualEncoderRetrieverModel
from src.retriever.retriever import DenseRetriever
from src.vectorstore.vector_index import LocalVectorIndex
from src.utils.checkpoint import CheckpointManager

def build_index(rebuild: bool = False, preset: str = "SMALL"):
    index = LocalVectorIndex(INDEXES_DIR)
    if not rebuild and index.load():
        print("[*] Existing index loaded. Use --rebuild to recreate index.")
        return index

    chunks_file = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    if not os.path.exists(chunks_file):
        from ingest_documents import ingest_documents
        ingest_documents()

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("[!] Warning: No chunks found to build index.")
        return index

    print(f"[*] Building vector index for {len(chunks)} document chunks...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = LocalTokenizer()
    tokenizer.load(TOKENIZER_DIR)

    p_cfg = PRESETS.get(preset, PRESETS["SMALL"])
    m_cfg = ModelConfig(
        d_model=p_cfg["d_model"],
        num_heads=p_cfg["num_heads"],
        num_encoder_layers=p_cfg["num_encoder_layers"],
        d_ff=p_cfg["d_ff"]
    )

    model = DualEncoderRetrieverModel(
        vocab_size=len(tokenizer),
        d_model=m_cfg.d_model,
        num_heads=m_cfg.num_heads,
        num_layers=m_cfg.num_encoder_layers,
        d_ff=m_cfg.d_ff,
        pad_idx=tokenizer.pad_id
    )

    ckpt_mgr = CheckpointManager(RETRIEVER_CHECKPOINT_DIR)
    if ckpt_mgr.latest_checkpoint_exists("best_model.pt"):
        ckpt_mgr.load_checkpoint(model, filename="best_model.pt", device=device)
    elif ckpt_mgr.latest_checkpoint_exists("latest.pt"):
        ckpt_mgr.load_checkpoint(model, filename="latest.pt", device=device)
    else:
        print("[!] Notice: No trained retriever checkpoint found. Using randomly initialized encoder.")

    retriever = DenseRetriever(model, tokenizer, device=device)
    
    texts = [c["text"] for c in chunks]
    metadata = [{**c.get("metadata", {}), "text": c.get("text", "")} for c in chunks]

    embeddings = retriever.encode_batch(texts, is_doc=True)
    index.build_index(embeddings, metadata)
    index.save()

    print("[OK] Index build completed successfully!")
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force complete index rebuild")
    parser.add_argument("--preset", type=str, default="SMALL", choices=["SMALL", "MEDIUM", "LARGE"])
    args = parser.parse_args()

    build_index(rebuild=args.rebuild, preset=args.preset)
