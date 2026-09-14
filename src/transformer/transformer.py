import math
import torch
import torch.nn as nn
from typing import Optional
from src.transformer.positional_encoding import PositionalEncoding
from src.transformer.encoder import TransformerEncoder
from src.transformer.decoder import TransformerDecoder

class TransformerSeq2Seq(nn.Module):
    """
    Encoder-Decoder Transformer Architecture for Sequence-to-Sequence Generation.
    """
    def __init__(self, vocab_size: int, 
                 d_model: int = 128, 
                 num_heads: int = 4, 
                 num_encoder_layers: int = 3, 
                 num_decoder_layers: int = 3, 
                 d_ff: int = 512, 
                 dropout: float = 0.1, 
                 pad_idx: int = 0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings & Positional Encoding
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Core Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout=dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout=dropout)

        # Output projection head to vocabulary
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_pad_mask(self, src: torch.Tensor) -> torch.Tensor:
        # Mask shape: (batch_size, 1, 1, seq_len)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        # Lower triangular matrix for causal decoder self-attention
        mask = torch.tril(torch.ones((size, size), device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(1) # (1, 1, size, size)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.generate_pad_mask(src)
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        memory = self.encoder(x, mask=src_mask)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, 
               tgt_mask: Optional[torch.Tensor] = None, 
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tgt_mask is None:
            seq_len = tgt.size(1)
            causal_mask = self.generate_causal_mask(seq_len, tgt.device)
            pad_mask = self.generate_pad_mask(tgt)
            tgt_mask = causal_mask & pad_mask

        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        out = self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.output_layer(out)
        return logits

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if src_mask is None:
            src_mask = self.generate_pad_mask(src)
            
        memory = self.encode(src, src_mask=src_mask)
        logits = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return logits
