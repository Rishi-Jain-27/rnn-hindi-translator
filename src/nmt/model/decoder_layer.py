"""One decoder block: masked self-attention + cross-attention + FFN (each residual+norm).

Self-attention uses the causal mask + tgt_pad_mask and reads/writes the self-attn
KV-cache at decode time. Cross-attention attends to encoder memory under src_pad_mask
with a separately-cached K/V (computed once from memory, reused every step).
"""

# TODO(rishi): implement
