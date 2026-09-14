import torch
import torch.nn as nn
from typing import Optional
from src.transformer.attention import MultiHeadAttention
from src.transformer.encoder import FeedForwardNetwork

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked Self-Attention
        self_attn_out, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Cross-Attention over Encoder Memory
        cross_attn_out, _ = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Feed-Forward Network
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x
