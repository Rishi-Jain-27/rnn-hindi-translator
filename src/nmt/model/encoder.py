"""Encoder stack: token embedding + positional encoding + N encoder layers + final norm.

Contract (agreed, see plan §4):
    forward(src_ids (B, S), src_pad_mask (B, S) bool keep-mask) -> memory (B, S, d_model)
Builds nothing causal; only padding is masked.
"""

import torch
import torch.nn as nn
from ..config import ModelConfig
from .embeddings import TokenEmbedding
from .positional import make_positional
from .encoder_layer import EncoderLayer
from .norm import make_norm

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = TokenEmbedding(config.vocab_size, config.d_model, config.pad_id)
        self.pos = make_positional(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_enc_layers)])
        self.norm = make_norm(config, config.d_model)
    
    def forward(self, src_ids, src_pad_mask):
        B, S = src_pad_mask.shape
        attn_mask = src_pad_mask.reshape(B, 1, 1, S)
        x = self.embed(src_ids)
        x = self.pos(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        return x
