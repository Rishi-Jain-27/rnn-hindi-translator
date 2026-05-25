
import torch
import torch.nn as nn
from ..config import ModelConfig
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_self_attn=True):
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
        self.cfg = config
        self.rope = is_self_attn and config.pos_encoding == "rope"

        '''
        If RoPE, create sin and cos tables.
        Precompute to avoid recomputing thousands of times because
        it would then be recomputed every forward pass
        '''
        if self.rope:
            # create tensor for 0...max_len - 1, unsqueeze for an extra 1 dim to broadcast with head_dim
            positions = torch.arange(config.max_len, dtype=torch.float).unsqueeze(1)
            theta_i = torch.exp(
                # theta_i = 10000^(-2i/64)
                # this creates the 2i exponents
                torch.arange(0, config.head_dim, 2, dtype=torch.float)
                # then multiply by the reciprocal power
                * (-math.log(10000.0) / config.head_dim)
            )
            # angle grid is positions * freqeunces broadcasted together
            # gives (max_len, 32) because head_dim/2 = 32
            angle_grid = positions * theta_i

            # Expand to per-coordinate: repeat_interleave along last axis
            # gives (max_len, 64) where each pair of coords has the same angle
            cos_table = torch.cos(angle_grid).repeat_interleave(2, dim=-1)
            sin_table = torch.sin(angle_grid).repeat_interleave(2, dim=-1)
            self.register_buffer("cos_table", cos_table)
            self.register_buffer("sin_table", sin_table)
    
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

        '''
        RoPE!
        Note:
        - these are comments assuming head_dim = 64
        - cos and sin tables created in __init__
        # 1. After head split, each q and k at each pos is a 64-num vector
        # 2. Pair up coords adjacently to crate 32 2D vectors
        # 3. Rotate pair i at pos p by angle p * O_i
        # Where O_i = 10000^(-2i/64)
        # 4. Do all 32 rotations in one shot as rotated = q * cos + swapneg(q) * sin
        # Cos and sin are per-coordinate tables — each pair's angle is repeated twice side-by-side
        # Swapneg turns each pair (a, b) to (-b, a)
        # 5. Apply to q and k, not v. Then dot product carries the relative angle.'''

        # Only do RoPE if RoPE is enabled, of course
        if self.rope:
            # Get offset by finding the current amount of tokens cached so far.
            # That is the time axis in the k and v tensors
            # Each slot in kv_cache stores a tuple of (k, v)
            # and each k and v has shape (B, n_heads, t, head_dim)
            # so get either k or v, get time axis, that is offset
            # if that slot is empty or the cache is empty, then offset is 0
            # because we must be at the start
            offset = kv_cache[layer_id][0].shape[2] if kv_cache is not None and kv_cache[layer_id] is not None else 0
            
            # slice cos and sin tables to L(q/k) + offset.
            # L is the current seq len, found above
            cos = getattr(self, "cos_table")[offset : offset + Lq]
            sin = getattr(self, "sin_table")[offset : offset + Lk]
            # this is okay because Lq == Lk if doing RoPE

            # Rotate q.
            q = q * cos + torch.stack([-q.view(*q.shape[:-1], self.head_dim // 2, 2)[:, :, :, :, 1], q.view(*q.shape[:-1], self.head_dim // 2, 2)[:, :, :, :, 0]], dim=-1).flatten(-2) * sin
            '''
            To swapneg:
            - reshape to (B, n_heads, L, head_dim/2, 2)
            - then access (a, b) from the last axis and rearrange to (-b, a)
            - then flatten back to interleave as [-b, a, -b, a, ...] pairs

            Explanation of the swapneg code:
            - q.view(*q.shape[:-1], self.head_dim // 2, 2) gives (B, n_heads, L, head_dim/2, 2)
            - [:, :, :, :, 0] gives a. [:, :, :, :, 1] gives b.
            - torch.stack([-b, a], dim=-1) creates the pairing along the last axis
            - .flatten(-2) gives back (B, n_heads, L, head_dim) with the pairs interleaved along the last axis
            '''

            # Same for k
            k = k * cos + torch.stack([-k.view(*k.shape[:-1], self.head_dim // 2, 2)[:, :, :, :, 1], k.view(*k.shape[:-1], self.head_dim // 2, 2)[:, :, :, :, 0]], dim=-1).flatten(-2) * sin

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

