import torch
import torch.nn as nn
from ..config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** (-0.5)
        self.d_model = config.d_model
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.attn_dropout)
    
    def forward(self, query, key, value, attn_mask=None, kv_cache=None, layer_id=None):
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # reshape to heads & transpose
        B, Lq, _ = q.shape
        q = torch.reshape(q, (B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)
        
        B, Lk, _ = k.shape
        k = torch.reshape(k, (B, Lk, self.n_heads, self.head_dim)).transpose(1, 2)
        
        B, Lv, _ = v.shape
        v = torch.reshape(v, (B, Lv, self.n_heads, self.head_dim)).transpose(1, 2)

        # Determine whether to update kv cache or not
        if kv_cache is not None:
            # if the slot is empty, add to it
            if kv_cache[layer_id] is None:
                kv_cache[layer_id] = (k, v)
            else:
                # pull stored self keys and values
                layer_k, layer_v = kv_cache[layer_id]

                # concat along positions
                kv_cache[layer_id] = (torch.cat((layer_k, k), dim=2), torch.cat((layer_v, v), dim=2))
                k, v = kv_cache[layer_id]
        
        scores = q @ k.transpose(-2, -1) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        
        attn = torch.softmax(scores, dim=-1)
        attn_dropout = self.dropout(attn)

        context = (attn_dropout @ v).transpose(1, 2).reshape(B, Lq, self.d_model)

        out = self.W_o(context)

        return (out, attn)

