"""Label-smoothed cross-entropy with ignore_index=pad_id. Supports token- vs
sentence-level normalization; for token norm under gradient accumulation, sum the
loss over the whole accum window and divide by total tokens (not per-micro-batch mean)."""
