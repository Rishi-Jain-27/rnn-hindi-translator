# kv-cache for incremental (decode-only) attention.
# first pass = self-attention only (cross-attn recomputes from memory each step).
#
# the cache is a flat list with one slot per decoder layer:
#   None     -> nothing stored yet (first decode step)
#   (k, v)   -> that layer's grown self keys/values, each (B, n_heads, t, head_dim)
# attention reads/appends its slot by layer_id; the decoder reads a slot's time length
# as the position offset. greedy/beam allocate one with new_cache and loop decode_step.
#
# deferred (lands with beam / cross-attn caching): a richer container with
# reorder(beam_index) for beam pruning + a separately cached cross-attn k/v.

def new_cache(n_layers):
    # one empty (None) slot per decoder layer; caller passes len(model.decoder.layers)
    return [None] * n_layers
