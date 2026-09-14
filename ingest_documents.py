import os
import json
import argparse
from config import DOCUMENTS_DIR, PROCESSED_DATA_DIR, ChunkingConfig
from src.data.document_loader import DocumentLoader
from src.data.document_cleaner import DocumentCleaner
from src.data.chunker import DocumentChunker

def ingest_documents(documents_dir=DOCUMENTS_DIR, output_dir=PROCESSED_DATA_DIR, strategy="sentence", chunk_size=256, chunk_overlap=50):
    print(f"[*] Scanning documents from: {documents_dir}")
    loader = DocumentLoader(documents_dir)
    docs = loader.load_directory()
    print(f"[+] Loaded {len(docs)} document pages/sections.")

    cleaner = DocumentCleaner()
    chunker = DocumentChunker(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    for doc in docs:
        doc["text"] = cleaner.clean_text(doc["text"])
        if doc["text"]:
            chunks = chunker.process_document(doc)
            all_chunks.extend(chunks)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "chunks.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[OK] Successfully ingested {len(all_chunks)} chunks into: {out_file}")
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest local documents and chunk them.")
    parser.add_argument("--strategy", type=str, default="sentence", choices=["fixed", "sentence", "token"])
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    args = parser.parse_args()

    ingest_documents(strategy=args.strategy, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
