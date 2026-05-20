"""LayerNorm | RMSNorm (config.norm_type) + a pre/post-norm residual wrapper.

config.norm_position ("pre" default | "post") selects where the norm sits relative
to each sublayer's residual connection. The residual+norm wrapper is reused by both
encoder and decoder layers so the norm policy lives in one place.
"""

# TODO(rishi): implement
