from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .blocks import GPT2EncoderBlock
from .embeddings import TokenPositionEmbedding


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_length: int,
        pad_id: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        embd_dropout: float,
        attn_dropout: float,
        resid_dropout: float,
    ):
        super().__init__()
        self.embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            pad_id=pad_id,
            dropout=embd_dropout,
        )
        self.layers = nn.ModuleList(
            [
                GPT2EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, padding_mask=padding_mask)
        return self.final_layer_norm(hidden_states)
