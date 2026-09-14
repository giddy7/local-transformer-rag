import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer.positional_encoding import PositionalEncoding
from src.transformer.encoder import TransformerEncoder

class TransformerEmbeddingModel(nn.Module):
    """
    Dense Transformer Encoder for embedding text into fixed-dimension vectors.
    Supports Mean Pooling and CLS/BOS Token Pooling.
    """
    def __init__(self, vocab_size: int, 
                 d_model: int = 128, 
                 num_heads: int = 4, 
                 num_layers: int = 3, 
                 d_ff: int = 512, 
                 dropout: float = 0.1, 
                 pad_idx: int = 0,
                 pooling: str = "mean"):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        if mask is None:
            mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.encoder(x, mask=mask)  # (batch_size, seq_len, d_model)

        if self.pooling == "cls":
            embeddings = output[:, 0, :]
        else: # mean pooling
            # mask: (batch_size, 1, 1, seq_len) -> (batch_size, seq_len, 1)
            valid_mask = (input_ids != self.pad_idx).unsqueeze(-1).float()
            sum_embeddings = torch.sum(output * valid_mask, dim=1)
            sum_mask = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        # L2 Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

class DualEncoderRetrieverModel(nn.Module):
    """
    Dual Encoder model using separate or shared encoders for questions and document chunks.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, shared: bool = True, **kwargs):
        super().__init__()
        self.shared = shared
        self.question_encoder = TransformerEmbeddingModel(vocab_size, d_model=d_model, **kwargs)
        if shared:
            self.doc_encoder = self.question_encoder
        else:
            self.doc_encoder = TransformerEmbeddingModel(vocab_size, d_model=d_model, **kwargs)

    def forward(self, q_ids: torch.Tensor, doc_ids: torch.Tensor):
        q_emb = self.question_encoder(q_ids)
        doc_emb = self.doc_encoder(doc_ids)
        return q_emb, doc_emb
