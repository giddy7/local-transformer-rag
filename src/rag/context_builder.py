from typing import List, Dict, Any, Tuple

class ContextBuilder:
    """
    Manages context window lengths by truncating chunks intelligently and formatting RAG prompts.
    """
    def __init__(self, max_context_length: int = 512, sep_token: str = "<SEP>"):
        self.max_context_length = max_context_length
        self.sep_token = sep_token

    def build_context(self, question: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> Tuple[str, List[Dict[str, Any]]]:
        if not retrieved_chunks:
            return "", []

        q_words = len(question.split())
        budget_words = max(50, self.max_context_length - q_words - 20)

        selected_chunks = []
        context_parts = []
        current_word_count = 0

        for chunk_meta, score in retrieved_chunks:
            chunk_text = chunk_meta.get("text", "") if isinstance(chunk_meta, dict) and "text" in chunk_meta else ""
            if not chunk_text:
                # If text is not directly in metadata dict, look up
                chunk_text = chunk_meta.get("chunk_text", str(chunk_meta))

            words = chunk_text.split()
            if current_word_count + len(words) <= budget_words:
                context_parts.append(chunk_text)
                selected_chunks.append(chunk_meta)
                current_word_count += len(words)
            else:
                # Truncate final fitting chunk
                rem = budget_words - current_word_count
                if rem > 15:
                    trunc_text = " ".join(words[:rem]) + "..."
                    context_parts.append(trunc_text)
                    selected_chunks.append(chunk_meta)
                break

        full_context = f" {self.sep_token} ".join(context_parts)
        return full_context, selected_chunks
