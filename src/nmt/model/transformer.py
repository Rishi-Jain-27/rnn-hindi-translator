"""Full Transformer: assembles Encoder + Decoder and ties embeddings/output projection.


forward(src_ids (B, S), tgt_ids (B, T), src_pad_mask, tgt_pad_mask) -> logits (B, T, V)
tgt_ids is shifted-right (BOS + y[:-1]). Stateless in training (the KV-cache is decode-only).
Also exposes inference entry points used by decode/: encode(src_ids, src_pad_mask) and
decode_step(...) over a KV-cache. A build_model(ModelConfig, tokenizer) factory copies
vocab_size + special-token ids (pad/bos/eos/unk) into the model so they are set once.
"""

import torch
import torch.nn as nn
from ..config import ModelConfig
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        if config.tie_embeddings:
            self.decoder.embed = self.encoder.embed

    # forward is for training and loss eval
    def forward(self, src_ids, tgt_ids, src_pad_mask, tgt_pad_mask):
        memory = self.encoder(src_ids, src_pad_mask)
        logits = self.decoder(tgt_ids, memory, tgt_pad_mask, src_pad_mask)
        return logits
    
    def encode(self, src_ids, src_pad_mask):
        return self.encoder(src_ids, src_pad_mask)

    # decode step is for evaluation, kv cache is solely for eval
    def decode_step(self, token, memory, src_pad_mask, kv_cache, return_attn=False):
        device = token.device
        B = token.shape[0]
        return self.decoder(token, memory, torch.full((B, 1), fill_value=True, device=device), src_pad_mask, kv_cache, return_attn)

def build_model(config, tokenizer):
    config.vocab_size = tokenizer.vocab_size
    config.pad_id = tokenizer.pad_id
    config.bos_id = tokenizer.bos_id
    config.eos_id = tokenizer.eos_id
    config.unk_id = tokenizer.unk_id

    return Transformer(config)

