import re
from typing import List, Dict, Any, Optional

class DocumentChunker:
    """
    Splits document text into chunks using Fixed-Length, Sentence-Based, or Token-Based methods.
    """
    def __init__(self, strategy: str = "sentence", chunk_size: int = 256, chunk_overlap: int = 50, tokenizer = None):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def chunk_fixed(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def chunk_sentence(self, text: str) -> List[str]:
        # Split into sentences using regex boundary detection
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Preserve overlap words/sentences
                overlap_words = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                current_chunk = overlap_sentences
                current_word_count = overlap_words

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def chunk_token(self, text: str) -> List[str]:
        if not self.tokenizer:
            # Fallback to word-based if tokenizer not provided
            return self.chunk_fixed(text)
        
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return []
        
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(token_ids), step):
            chunk_tokens = token_ids[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if i + self.chunk_size >= len(token_ids):
                break
        return chunks

    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc["text"]
        metadata = doc["metadata"]

        if self.strategy == "fixed":
            raw_chunks = self.chunk_fixed(text)
        elif self.strategy == "token":
            raw_chunks = self.chunk_token(text)
        else:
            raw_chunks = self.chunk_sentence(text)

        doc_name = metadata.get("document_name", "doc")
        processed_chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc_name}_chunk_{idx:04d}"
            chunk_meta = metadata.copy()
            chunk_meta["chunk_id"] = chunk_id
            chunk_meta["chunk_index"] = idx
            processed_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_meta
            })
        return processed_chunks
