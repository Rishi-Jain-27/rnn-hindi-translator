"""Full Transformer: assembles Encoder + Decoder and ties embeddings/output projection.

Contract (agreed, see plan §4):
    forward(src_ids (B, S), tgt_ids (B, T), src_pad_mask, tgt_pad_mask) -> logits (B, T, V)
        tgt_ids is shifted-right (BOS + y[:-1]). Stateless in training (the KV-cache is
        decode-only).
Also exposes inference entry points used by decode/: encode(src_ids, src_pad_mask) and
decode_step(...) over a KV-cache. A build_model(ModelConfig, tokenizer) factory copies
vocab_size + special-token ids (pad/bos/eos/unk) into the model so they are set once.
"""

# TODO(rishi): implement
