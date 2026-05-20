"""KV-cache container shared by greedy + beam decoding.

Contract (agreed, see plan §4): alloc / append(layer_id, k, v) / reorder(beam_index) /
reset / current_len. Self-attention K/V grow one step at a time; cross-attention K/V are
computed once from the encoder memory and cached separately. The decoder layers (model/)
fill it; greedy/beam (decode/) drive it. Gated by tests/test_kv_cache.py (incremental
decode logits must match a full-sequence forward to ~1e-4)."""
