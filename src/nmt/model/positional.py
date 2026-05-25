# Build sinusoidal and learned positional embeddings
# defer RoPE and ALiBi to attention.py

import torch
import torch.nn as nn
from ..config import ModelConfig
import math

class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor  # declared for type-checkers; register_buffer creates it at runtime

    def __init__(self, d_model, max_len):
        super().__init__()
        # Column of positions 0...max_len - 1
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        # Row of angular frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)         # the 2i exponents: 0, 2, 4, ..., d_model-2
            * (-math.log(10000.0) / d_model)                       # multiply by -ln(10000)/d_model -> the reciprocal power
        )

        # Column * row broadcasts into full angle grid
        angles = positions * div_term

        # create table
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles) # even cols are sin, odd are cos
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe)  


    def forward(self, x, offset=0):
        # current seq len
        T = x.size(1)
        # clear error if too long
        assert (offset + T) <= self.pe.size(0), f"seq len {T} > max_len {self.pe.size(0)}"
        return x + self.pe[offset:T + offset] 

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
    
    def forward(self, x, offset=0):
        # clear error if too long
        T = x.size(1)
        assert (T + offset) <= self.pos_embed.num_embeddings, f"seq len {T} > max_len {self.pos_embed.num_embeddings}"
        positions = torch.arange(start=offset, end=T + offset, device=x.device, dtype=torch.long)
        x = self.pos_embed(positions) + x
        return x

class ThrowAwayPosEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
    
    def forward(self, x, offset=0):
        return x # ignore offset

# Rope is done in attention.py, so we don't redo it here
def make_positional(config):
    return {"sinusoidal": SinusoidalPositionalEncoding,
            "learned": LearnedPositionalEncoding,
            "rope": ThrowAwayPosEncoding,
            "alibi": nn.Identity}[config.pos_encoding](config.d_model, config.max_len)
