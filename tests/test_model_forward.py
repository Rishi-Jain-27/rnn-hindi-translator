"""Transformer forward output shape (B, T, V); tied-weight identity; parameter-count sanity."""

import types
import pytest

torch = pytest.importorskip("torch")  # skip on the Mac; runs on Colab

from nmt.config import ModelConfig
from nmt.model.transformer import Transformer, build_model


def _cfg(**kw):
    # Tiny model so the forward pass is fast; head_dim = 32 / 4 = 8.
    base = dict(d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                d_ff=64, vocab_size=50, dropout=0.0, attn_dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(cfg, B=2, S=7, T=5):
    src_ids = torch.randint(0, cfg.vocab_size, (B, S))
    tgt_ids = torch.randint(0, cfg.vocab_size, (B, T))
    src_pad = torch.ones(B, S, dtype=torch.bool)
    tgt_pad = torch.ones(B, T, dtype=torch.bool)
    src_pad[0, -2:] = False                 # pad the last 2 source positions of item 0
    return src_ids, tgt_ids, src_pad, tgt_pad


def test_forward_output_shape():
    """forward(src, tgt, masks) -> logits (B, T, V), NaN-free."""
    cfg = _cfg()
    model = Transformer(cfg).eval()
    B, S, T = 2, 7, 5
    logits = model(*_batch(cfg, B, S, T))
    assert logits.shape == (B, T, cfg.vocab_size)
    assert not torch.isnan(logits).any()


def test_encode_returns_memory():
    """encode() yields encoder memory (B, S, d_model)."""
    cfg = _cfg()
    model = Transformer(cfg).eval()
    src_ids, _, src_pad, _ = _batch(cfg, B=2, S=7, T=5)
    memory = model.encode(src_ids, src_pad)
    assert memory.shape == (2, 7, cfg.d_model)


def test_tied_weight_identity():
    """Source emb, target emb, and output projection are literally one matrix."""
    cfg = _cfg(tie_embeddings=True)
    model = Transformer(cfg)
    assert model.encoder.embed is model.decoder.embed          # same module
    w = model.encoder.embed.embed.weight
    assert model.decoder.embed.embed.weight is w               # same weight
    ids = [id(p) for p in model.parameters()]
    assert ids.count(id(w)) == 1                               # deduped, counted once


def test_param_count_drops_with_tying():
    """Tying removes exactly one extra (V x d_model) embedding copy."""
    untied = sum(p.numel() for p in Transformer(_cfg(tie_embeddings=False)).parameters())
    tied = sum(p.numel() for p in Transformer(_cfg(tie_embeddings=True)).parameters())
    cfg = _cfg()
    assert untied - tied == cfg.vocab_size * cfg.d_model


def test_build_model_copies_tokenizer_ids():
    """build_model writes vocab_size + special-token ids onto config (single source of truth)."""
    tok = types.SimpleNamespace(vocab_size=123, pad_id=0, bos_id=2, eos_id=3, unk_id=1)
    cfg = _cfg()
    model = build_model(cfg, tok)
    assert cfg.vocab_size == 123
    assert (cfg.pad_id, cfg.bos_id, cfg.eos_id, cfg.unk_id) == (0, 2, 3, 1)
    assert model.encoder.embed.embed.weight.shape[0] == 123    # embedding sized to tokenizer
