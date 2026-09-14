import os
import torch
from typing import Dict, Any, List, Optional

from config import (
    TOKENIZER_DIR, RETRIEVER_CHECKPOINT_DIR, GENERATOR_CHECKPOINT_DIR,
    INDEXES_DIR, PRESETS, ModelConfig, RAGConfig
)
from src.tokenizer.tokenizer import LocalTokenizer
from src.retriever.embedding_model import DualEncoderRetrieverModel
from src.retriever.retriever import DenseRetriever
from src.vectorstore.vector_index import LocalVectorIndex
from src.generator.generator_model import TransformerGeneratorModel
from src.generator.decoding import greedy_decode, sample_decode
from src.rag.context_builder import ContextBuilder
from src.utils.checkpoint import CheckpointManager

class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation (RAG) System.
    """
    def __init__(self, preset: str = "SMALL", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preset = preset
        self.rag_cfg = RAGConfig()
        
        # Load Tokenizer
        self.tokenizer = LocalTokenizer()
        if not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.json")):
            from src.tokenizer.train_tokenizer import train_tokenizer
            train_tokenizer()
        self.tokenizer.load(TOKENIZER_DIR)

        p_cfg = PRESETS.get(preset, PRESETS["SMALL"])
        self.m_cfg = ModelConfig(
            d_model=p_cfg["d_model"],
            num_heads=p_cfg["num_heads"],
            num_encoder_layers=p_cfg["num_encoder_layers"],
            num_decoder_layers=p_cfg["num_decoder_layers"],
            d_ff=p_cfg["d_ff"]
        )

        # Load Retriever Model
        self.retriever_model = DualEncoderRetrieverModel(
            vocab_size=len(self.tokenizer),
            d_model=self.m_cfg.d_model,
            num_heads=self.m_cfg.num_heads,
            num_layers=self.m_cfg.num_encoder_layers,
            d_ff=self.m_cfg.d_ff,
            pad_idx=self.tokenizer.pad_id
        ).to(self.device)

        r_ckpt_mgr = CheckpointManager(RETRIEVER_CHECKPOINT_DIR)
        if r_ckpt_mgr.latest_checkpoint_exists("best_model.pt"):
            r_ckpt_mgr.load_checkpoint(self.retriever_model, filename="best_model.pt", device=self.device)
        elif r_ckpt_mgr.latest_checkpoint_exists("latest.pt"):
            r_ckpt_mgr.load_checkpoint(self.retriever_model, filename="latest.pt", device=self.device)

        self.retriever = DenseRetriever(self.retriever_model, self.tokenizer, device=self.device)

        # Load Vector Store Index
        self.vector_index = LocalVectorIndex(INDEXES_DIR)
        if not self.vector_index.load():
            from build_index import build_index
            self.vector_index = build_index(preset=preset)

        # Load Generator Model
        self.generator_model = TransformerGeneratorModel(
            vocab_size=len(self.tokenizer),
            d_model=self.m_cfg.d_model,
            num_heads=self.m_cfg.num_heads,
            num_encoder_layers=self.m_cfg.num_encoder_layers,
            num_decoder_layers=self.m_cfg.num_decoder_layers,
            d_ff=self.m_cfg.d_ff,
            pad_idx=self.tokenizer.pad_id
        ).to(self.device)

        g_ckpt_mgr = CheckpointManager(GENERATOR_CHECKPOINT_DIR)
        if g_ckpt_mgr.latest_checkpoint_exists("best_model.pt"):
            g_ckpt_mgr.load_checkpoint(self.generator_model, filename="best_model.pt", device=self.device)
        elif g_ckpt_mgr.latest_checkpoint_exists("latest.pt"):
            g_ckpt_mgr.load_checkpoint(self.generator_model, filename="latest.pt", device=self.device)

        self.context_builder = ContextBuilder(
            max_context_length=self.rag_cfg.max_context_length,
            sep_token=self.tokenizer.sep_token
        )

    def answer_question(self, question: str, top_k: int = 3, show_context: bool = False, decoding: str = "greedy") -> Dict[str, Any]:
        # 1. Encode Question
        q_emb = self.retriever.encode_text(question)

        # 2. Search Top-K Document Chunks
        retrieved_results = self.vector_index.search(q_emb, top_k=top_k)

        # 3. Format Context
        context_str, selected_chunks = self.context_builder.build_context(question, retrieved_results)

        # 4. Construct Prompt
        if context_str:
            prompt = f"{context_str} {self.tokenizer.sep_token} {question}"
        else:
            prompt = question

        enc_ids = self.tokenizer.encode(prompt, max_length=self.rag_cfg.max_context_length, pad=True)
        src_tensor = torch.tensor([enc_ids], dtype=torch.long, device=self.device)

        # 5. Generate Answer
        if decoding == "sample":
            generated_ids = sample_decode(
                self.generator_model, src_tensor,
                bos_idx=self.tokenizer.bos_id,
                eos_idx=self.tokenizer.eos_id,
                max_len=self.rag_cfg.generation_max_tokens,
                temperature=self.rag_cfg.temperature,
                top_k=self.rag_cfg.top_k_sampling,
                top_p=self.rag_cfg.top_p
            )
        else:
            generated_ids = greedy_decode(
                self.generator_model, src_tensor,
                bos_idx=self.tokenizer.bos_id,
                eos_idx=self.tokenizer.eos_id,
                max_len=self.rag_cfg.generation_max_tokens
            )

        raw_answer = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()

        # Check quality of generated answer (prevent degenerate repetitive tokens)
        words = raw_answer.split()
        is_repetitive = len(words) > 3 and (len(set(words)) <= 2 or max(words.count(w) for w in words) > len(words) * 0.45)
        
        if not raw_answer or is_repetitive or len(words) < 3:
            if selected_chunks:
                first_chunk = selected_chunks[0]
                first_text = first_chunk.get("text", "")
                
                # Find the sentence in the retrieved text that best matches the question keywords
                q_keywords = set(w.lower() for w in question.split() if len(w) > 2)
                sentences = [s.strip() for s in first_text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
                best_sentence = sentences[0] if sentences else first_text
                best_overlap = -1
                
                for s in sentences:
                    s_words = set(w.lower() for w in s.split())
                    overlap = len(q_keywords.intersection(s_words))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_sentence = s

                answer_text = best_sentence + ("." if not best_sentence.endswith(".") else "")
            else:
                answer_text = "No relevant context found in local documents to answer this question."
        else:
            answer_text = raw_answer

        # 6. Build Sources Citations
        sources = []
        for idx, (meta, score) in enumerate(retrieved_results, 1):
            doc_name = meta.get("document_name", "Unknown Document")
            page_num = meta.get("page", 1)
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            sources.append({
                "rank": idx,
                "document_name": doc_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "score": score,
                "text": meta.get("text", "")
            })

        response = {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "context_used": context_str if show_context else None
        }
        return response
