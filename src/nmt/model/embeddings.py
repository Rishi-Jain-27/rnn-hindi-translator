"""Token embedding (scaled by sqrt(d_model)) + tied-weight plumbing.

Owns the shared embedding matrix for the joint SentencePiece vocab. The same
weight is tied 3 ways (config.tie_embeddings, FIXED True): source embedding,
target embedding, and the decoder output projection. Holds vocab_size / pad_id
copied in from the tokenizer at build_model time.
"""

# TODO(rishi): implement
