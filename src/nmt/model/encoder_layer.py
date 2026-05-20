from ..config import ModelConfig
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .norm import ResidualNorm
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.attn_norm = ResidualNorm(config, config.d_model)

        self.ffn = FeedForward(config)
        self.ffn_norm = ResidualNorm(config, config.d_model)

    def forward(self, x, attn_mask=None):
        x = self.attn_norm(x, lambda y: self.self_attn(y, y, y, attn_mask=attn_mask)[0])
        x = self.ffn_norm(x, self.ffn)

        return x



