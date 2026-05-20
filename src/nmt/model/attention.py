"""Scaled-dot-product + multi-head attention (MHA/MQA/GQA), masks, KV-cache.

Contract (agreed, see plan §4):
    forward(query, key, value, attn_mask=None, kv_cache=None, layer_id=None)
        query/key/value: (B, Lq|Lk, d_model)
        -> (out (B, Lq, d_model), attn (B, n_heads, Lq, Lk))
attn_mask is a boolean keep-mask (True = keep); the module converts it to an
additive bias using finfo.min (NOT -inf, for AMP NaN-safety). config.attn_variant
selects MHA / MQA / GQA (n_kv_heads). Optional Shaw relative positions (rel_pos).
"""

# TODO(rishi): implement
