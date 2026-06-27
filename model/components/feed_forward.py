from __future__ import annotations

from torch import nn


class GPT2MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, resid_dropout: float):
        super().__init__()
        self.fc_in = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.fc_out = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)
