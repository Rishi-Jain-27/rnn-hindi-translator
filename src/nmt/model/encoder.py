"""Encoder stack: token embedding + positional encoding + N encoder layers + final norm.

Contract (agreed, see plan §4):
    forward(src_ids (B, S), src_pad_mask (B, S) bool keep-mask) -> memory (B, S, d_model)
Builds nothing causal; only padding is masked.
"""

# TODO(rishi): implement
