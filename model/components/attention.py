from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        is_causal: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _build_attention_bias(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        attn_bias = None

        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones((query_len, key_len), dtype=torch.bool, device=device),
                diagonal=1,
            )
            attn_bias = causal_mask.view(1, 1, query_len, key_len)

        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :]
            attn_bias = padding_mask if attn_bias is None else (attn_bias | padding_mask)

        return attn_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_value_states = hidden_states if key_value_states is None else key_value_states

        query = self._shape(self.q_proj(hidden_states))
        key = self._shape(self.k_proj(key_value_states))
        value = self._shape(self.v_proj(key_value_states))

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_bias = self._build_attention_bias(
            query_len=hidden_states.size(1),
            key_len=key_value_states.size(1),
            device=hidden_states.device,
            key_padding_mask=key_padding_mask,
        )
        if attn_bias is not None:
            attn_weights = attn_weights.masked_fill(attn_bias, torch.finfo(attn_weights.dtype).min)

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            hidden_states.size(0), hidden_states.size(1), self.d_model
        )
        attn_output = self.out_proj(attn_output)
        return self.resid_dropout(attn_output)
