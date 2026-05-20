"""Decoder stack: token embedding + positional encoding + N decoder layers + final norm + output projection.

Contract (agreed, see plan §4):
    forward(tgt_ids (B, T), memory (B, S, d_model), tgt_pad_mask, src_pad_mask,
            kv_cache=None, return_attn=False)
        -> logits (B, T, V)  [, attn dict keyed (layer, kind) -> (B, n_heads, Tq, Tk)]
Builds the causal mask internally from T. Wires the KV-cache through each layer for
incremental decoding. Output projection is tied to the embedding (tie_embeddings).
"""

# TODO(rishi): implement
