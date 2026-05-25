from ..config import ModelConfig
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .norm import ResidualNorm
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.self_attn_norm = ResidualNorm(config, config.d_model)

        self.cross_attn = MultiHeadAttention(config)
        self.cross_attn_norm = ResidualNorm(config, config.d_model)

        self.ffn = FeedForward(config)
        self.ffn_norm = ResidualNorm(config, config.d_model)

    def forward(self, x, memory, self_attn_mask=None, cross_attn_mask=None, kv_cache=None, layer_id=None, return_attn=False):
        
        x = self.self_attn_norm(x, lambda y: self.self_attn(y, y, y, attn_mask=self_attn_mask, kv_cache=kv_cache, layer_id=layer_id)[0])
        
        cross_attn_weights = None

        def cross_sublayer(y):
            nonlocal cross_attn_weights
            out, attn = self.cross_attn(y, memory, memory, attn_mask=cross_attn_mask)
            cross_attn_weights = attn
            return out
        x = self.cross_attn_norm(x, cross_sublayer)
        x = self.ffn_norm(x, self.ffn)

        return (x, cross_attn_weights) if return_attn else x

