# RoPE correctness, four angles:
#  1. smoke -- the whole rope stack assembles + runs (pos_encoding="rope") and the decoder's
#     self.pos(x, offset) no-op doesn't crash (the ThrowAwayPosEncoding fix).
#  2. gating -- rotate only when (config is rope) AND (instance is self-attn); cross-attn and
#     non-rope configs build no tables and set self.rope False.
#  3. injection -- feed identical content at every position: with rope the scores become
#     position-dependent (attention NOT uniform); without rope, identical content -> uniform.
#  4. decode gate -- cached greedy must equal uncached full-prefix decode token-for-token under
#     rope. this is the offset gate: a wrong kv-cache offset would rotate the single new token
#     at the wrong absolute position and diverge from the full causal pass.

import pytest

torch = pytest.importorskip("torch")  # runs on Colab, skips on the Mac

from nmt.config import ModelConfig
from nmt.model.attention import MultiHeadAttention
from nmt.model.transformer import Transformer
from nmt.decode.greedy import greedy_decode


def _rope_cfg(**kw):
    # head_dim = 32 / 4 = 8 (even -> 4 rope pairs); dropout off for determinism
    base = dict(d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2, d_ff=64,
                max_len=64, vocab_size=32, dropout=0.0, attn_dropout=0.0,
                pos_encoding="rope")
    base.update(kw)
    return ModelConfig(**base)


# --- 1. smoke: rope model assembles, runs, right shape, no NaN ---

def test_rope_forward_shape():
    cfg = _rope_cfg()
    model = Transformer(cfg).eval()
    B, S, T = 2, 7, 5
    src_ids = torch.randint(0, cfg.vocab_size, (B, S))
    tgt_ids = torch.randint(0, cfg.vocab_size, (B, T))
    src_pad = torch.ones(B, S, dtype=torch.bool)
    tgt_pad = torch.ones(B, T, dtype=torch.bool)
    logits = model(src_ids, tgt_ids, src_pad, tgt_pad)   # exercises decoder self.pos(x, offset) no-op
    assert logits.shape == (B, T, cfg.vocab_size)
    assert not torch.isnan(logits).any()


# --- 2. gating: rotate only when (rope config) AND (self-attn instance) ---

def test_rope_gate_self_attn_builds_tables():
    cfg = _rope_cfg()
    mha = MultiHeadAttention(cfg, is_self_attn=True)
    assert mha.rope is True
    assert hasattr(mha, "cos_table") and hasattr(mha, "sin_table")
    assert mha.cos_table.shape == (cfg.max_len, cfg.head_dim)   # (max_len, head_dim)


def test_rope_gate_cross_attn_skips():
    # cross-attn under rope config must NOT rotate and must NOT build tables
    cfg = _rope_cfg()
    mha = MultiHeadAttention(cfg, is_self_attn=False)
    assert mha.rope is False
    assert not hasattr(mha, "cos_table")


def test_rope_gate_off_for_non_rope_config():
    # even a self-attn instance must not rotate when the encoding isn't rope
    cfg = _rope_cfg(pos_encoding="sinusoidal")
    mha = MultiHeadAttention(cfg, is_self_attn=True)
    assert mha.rope is False
    assert not hasattr(mha, "cos_table")


# --- 3. injection: identical content at every position ---

def _constant_input(cfg, B=1, L=6):
    # the same vector v repeated at every position -> content carries no position
    v = torch.randn(1, 1, cfg.d_model)
    return v.expand(B, L, cfg.d_model).contiguous()


def test_rope_makes_constant_input_nonuniform():
    # rope rotates q/k by position, so scores depend on (j - i) even with identical content
    torch.manual_seed(0)
    cfg = _rope_cfg()
    mha = MultiHeadAttention(cfg, is_self_attn=True).eval()
    x = _constant_input(cfg)
    with torch.inference_mode():
        _, attn = mha(x, x, x)
    uniform = torch.full_like(attn, 1.0 / x.shape[1])
    assert not torch.allclose(attn, uniform, atol=1e-3)


def test_no_rope_constant_input_is_uniform():
    # contrast: without rope, identical content -> all scores equal -> uniform attention
    torch.manual_seed(0)
    cfg = _rope_cfg(pos_encoding="sinusoidal")   # self.rope False -> no rotation inside MHA
    mha = MultiHeadAttention(cfg, is_self_attn=True).eval()
    x = _constant_input(cfg)
    with torch.inference_mode():
        _, attn = mha(x, x, x)
    uniform = torch.full_like(attn, 1.0 / x.shape[1])
    assert torch.allclose(attn, uniform, atol=1e-5)


# --- 4. decode gate: cached greedy == uncached full-prefix, under rope ---

def _uncached_greedy(model, src, max_len, cfg):
    # reference path: re-feed the full prefix every step, kv_cache=None (causal full forward)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        src_pad_mask = src != cfg.pad_id
        B = src.shape[0]
        memory = model.encode(src, src_pad_mask)
        dec_in = torch.full((B, 1), cfg.bos_id, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        recorded = []
        for _ in range(max_len):
            tgt_pad = dec_in != cfg.pad_id
            logits = model.decoder(dec_in, memory, tgt_pad, src_pad_mask)
            nxt = logits[:, -1, :].argmax(dim=-1)
            nxt = nxt.masked_fill(finished, cfg.pad_id)
            recorded.append(nxt)
            finished = finished | (nxt == cfg.eos_id)
            dec_in = torch.cat([dec_in, nxt.unsqueeze(1)], dim=1)
            if finished.all():
                break
        ids = torch.stack(recorded, dim=1)
        out = []
        for row in ids:
            seq = []
            for v in row:
                if v == cfg.eos_id:
                    break
                seq.append(v.item())
            out.append(seq)
        return out


def _tiny_rope_model():
    cfg = _rope_cfg()
    torch.manual_seed(0)
    return Transformer(cfg), cfg


def test_rope_cached_matches_uncached_single():
    model, cfg = _tiny_rope_model()
    src = torch.tensor([[4, 5, 6, 7, cfg.eos_id]])
    cached = greedy_decode(model, src, max_len=20, cfg=cfg)
    uncached = _uncached_greedy(model, src, max_len=20, cfg=cfg)
    assert cached == uncached


def test_rope_cached_matches_uncached_batched():
    model, cfg = _tiny_rope_model()
    src = torch.tensor([[4, 5, 6, 7, cfg.eos_id],
                        [8, 9, cfg.eos_id, cfg.pad_id, cfg.pad_id]])
    cached = greedy_decode(model, src, max_len=20, cfg=cfg)
    uncached = _uncached_greedy(model, src, max_len=20, cfg=cfg)
    assert cached == uncached


def test_rope_cached_matches_uncached_longer_run():
    model, cfg = _tiny_rope_model()
    src = torch.tensor([[4, 5, 6, 7, 8, 9, 10, cfg.eos_id],
                        [11, 12, 13, cfg.eos_id, cfg.pad_id, cfg.pad_id, cfg.pad_id, cfg.pad_id]])
    cached = greedy_decode(model, src, max_len=30, cfg=cfg)
    uncached = _uncached_greedy(model, src, max_len=30, cfg=cfg)
    assert cached == uncached
