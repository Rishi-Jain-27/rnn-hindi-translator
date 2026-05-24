# kv-cache for incremental (decode-only) attention.
# first pass = self-attention only (cross-attn recomputes from memory each step).
#
# the cache is a flat list with one slot per decoder layer:
#   None     -> nothing stored yet (first decode step)
#   (k, v)   -> that layer's grown self keys/values, each (B, n_heads, t, head_dim)
# attention reads/appends its slot by layer_id; the decoder reads a slot's time length
# as the position offset. greedy/beam allocate one with new_cache and loop decode_step.
#
# reorder_cache(beam_index) below handles beam pruning (self-attn k/v only).
# still deferred: a separately cached cross-attn k/v (recomputed from memory for now).

def new_cache(n_layers):
    # one empty (None) slot per decoder layer; caller passes len(model.decoder.layers)
    return [None] * n_layers


# reorder the beam axis so each beam follows its parent after a beam-search step.
# beam_index is a 1-d long tensor: new slot i continues old beam beam_index[i].
# gather every layer's (k, v) along dim 0 (the beam axis); empty slots stay None.
# returns a fresh cache list; feed it to the next decode_step.
def reorder_cache(kv_cache, beam_index):
    reordered = []
    # walk each layer's slot
    for slot in kv_cache:
        if slot is None:
            # nothing cached yet -> nothing to reorder
            reordered.append(None)
        else:
            # pull the grown self k/v and index the beam axis by parent
            k, v = slot
            reordered.append((k.index_select(0, beam_index), v.index_select(0, beam_index)))
    return reordered
