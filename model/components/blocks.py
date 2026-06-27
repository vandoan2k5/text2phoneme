from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import MultiHeadAttention
from .feed_forward import GPT2MLP


class GPT2EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        attn_dropout: float,
        resid_dropout: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            is_causal=False,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model=d_model, hidden_dim=mlp_hidden_dim, resid_dropout=resid_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            hidden_states=self.ln_1(hidden_states),
            key_padding_mask=padding_mask,
        )
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class GPT2DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        attn_dropout: float,
        resid_dropout: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            is_causal=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            is_causal=False,
        )
        self.ln_3 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model=d_model, hidden_dim=mlp_hidden_dim, resid_dropout=resid_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            hidden_states=self.ln_1(hidden_states),
            key_padding_mask=decoder_padding_mask,
        )
        hidden_states = hidden_states + self.cross_attn(
            hidden_states=self.ln_2(hidden_states),
            key_value_states=encoder_hidden_states,
            key_padding_mask=encoder_padding_mask,
        )
        hidden_states = hidden_states + self.mlp(self.ln_3(hidden_states))
        return hidden_states
