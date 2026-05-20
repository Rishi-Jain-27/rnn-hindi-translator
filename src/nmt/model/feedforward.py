"""Position-wise feed-forward network: ReLU | GeLU | SwiGLU | GeGLU (config.activation).

Two-layer FFN (d_model -> d_ff -> d_model) with dropout. GLU variants (SwiGLU/GeGLU)
gate two inner projections; shrink d_ff to ~2/3 for parameter parity if used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise FFN: the same little MLP applied independently to every token position.

    No mixing across the sequence happens here (that is attention's job) — this is just a
    per-position `(d_model -> wider -> d_model)` transform.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Get config
        self.activation = config.activation
        self.act = {"relu": F.relu,
                    "gelu": F.gelu,
                    "swiglu": F.silu,
                    "geglu": F.gelu}[self.activation]
        self.is_glu = config.activation.endswith("glu")

        # Calculate inner width
        if self.is_glu:
            d_ff_eff = 2 * config.d_ff // 3 # d_ff is ffn's hidden dimension
            # d_ff_eff is the effective inner width
            # it equals d_ff for plain FFN, but shrunk for GLU bc of 3 weight matrices
        else:
            d_ff_eff = config.d_ff

        # create layers
        self.up_projection = nn.Linear(config.d_model, d_ff_eff) # d_model is the model's main width
        if self.is_glu:
            self.gate_projection = nn.Linear(config.d_model, d_ff_eff)
        else:
            self.gate_projection = None
        self.down_projection = nn.Linear(d_ff_eff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model). Applied independently per position."""

        if self.is_glu:
            # for GLU: up project, gate project, activate, dropout, down project
            a = self.up_projection(x)
            g = self.gate_projection(x)
            h = self.act(g) * a
            h = self.dropout(h)
            return self.down_projection(h)
        else:
            # for non-GLU: up project, activate, dropout, down project
            h = self.up_projection(x)
            h = self.act(h)
            h = self.dropout(h)
            return self.down_projection(h)