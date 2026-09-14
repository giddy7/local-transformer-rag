import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k, v shape: (batch_size, num_heads, seq_len, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        # Linear projections & reshape to (batch_size, num_heads, seq_len, d_k)
        Q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None and mask.dim() == 3:
            # Expand mask for heads: (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)

        # Scaled dot product attention
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attn_output)

        return output, attn_weights
