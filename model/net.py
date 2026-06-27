from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model.components import TransformerDecoder, TransformerEncoder


class Text2PhonemeTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        attention_dropout: float,
        residual_dropout: float,
        max_source_length: int,
        max_target_length: int,
        src_pad_id: int,
        tgt_pad_id: int,
    ):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.max_target_length = max_target_length

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            max_length=max_source_length,
            pad_id=src_pad_id,
            num_layers=num_encoder_layers,
            num_heads=nhead,
            mlp_hidden_dim=dim_feedforward,
            embd_dropout=dropout,
            attn_dropout=attention_dropout,
            resid_dropout=residual_dropout,
        )
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            max_length=max_target_length,
            pad_id=tgt_pad_id,
            num_layers=num_decoder_layers,
            num_heads=nhead,
            mlp_hidden_dim=dim_feedforward,
            embd_dropout=dropout,
            attn_dropout=attention_dropout,
            resid_dropout=residual_dropout,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        source_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        source_padding_mask: torch.Tensor,
        target_padding_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        encoder_hidden_states = self.encoder(
            input_ids=source_ids,
            padding_mask=source_padding_mask,
        )
        hidden = self.decoder(
            input_ids=target_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            decoder_padding_mask=target_padding_mask,
            encoder_padding_mask=source_padding_mask,
        )
        logits = self.output_projection(hidden)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.tgt_pad_id,
            )
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        source_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        batch_size = source_ids.size(0)
        generated = torch.full(
            (batch_size, 1),
            bos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source_ids.device)

        for _ in range(self.max_target_length - 1):
            output = self.forward(
                source_ids=source_ids,
                target_input_ids=generated,
                source_padding_mask=source_padding_mask,
                target_padding_mask=generated.eq(self.tgt_pad_id),
                labels=None,
            )
            next_token = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, eos_id),
                next_token,
            )
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_id)
            if finished.all():
                break
        return generated
