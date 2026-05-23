

import torch
import torch.nn as nn
from ..config import ModelConfig
from .embeddings import TokenEmbedding
from .positional import make_positional
from .decoder_layer import DecoderLayer
from .norm import make_norm

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = TokenEmbedding(config.vocab_size, config.d_model, config.pad_id)
        self.pos = make_positional(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_dec_layers)])
        self.norm = make_norm(config, config.d_model)
    
    def forward(self, tgt_ids, memory, tgt_pad_mask, src_pad_mask, kv_cache=None):
        B, T = tgt_ids.shape
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_ids.device))
        tgt_pad = tgt_pad_mask[:, None, None, :]
        cross_attn_mask = src_pad_mask[:, None, None, :]
        self_attn_mask = (causal & tgt_pad) if kv_cache is None else None
        x = self.embed(tgt_ids)
        offset = kv_cache[0][0].shape[2] if kv_cache is not None and kv_cache[0] is not None else 0
        x = self.pos(x, offset)
        x = self.dropout(x)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, memory, self_attn_mask, cross_attn_mask, kv_cache, layer_id)
        x = self.norm(x)
        logits = self.embed.project(x)
        return logits

