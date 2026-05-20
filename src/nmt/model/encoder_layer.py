"""One encoder block: self-attention + position-wise FFN, each wrapped in residual+norm.

Self-attention sees src_pad_mask only (no causal mask). Optional stochastic depth
(config.droppath). Composes attention.py + feedforward.py + norm.py.
"""

# TODO(rishi): implement
