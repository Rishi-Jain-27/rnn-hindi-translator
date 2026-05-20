"""MHA output shapes; masked positions get ~zero attention weight; MQA/GQA head broadcasting."""

import pytest

torch = pytest.importorskip("torch")  # skip cleanly on the Mac; runs on Colab

from nmt.config import ModelConfig
from nmt.model.attention import MultiHeadAttention


def _cfg(**kw):
    # Small dims so tests are fast; head_dim = 32 / 4 = 8.
    base = dict(d_model=32, n_heads=4, attn_dropout=0.0, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def test_self_attention_shapes():
    """Self-attn: out is (B, Lq, d_model); attn is (B, n_heads, Lq, Lk) with Lq == Lk."""
    torch.manual_seed(0)
    cfg = _cfg()
    mha = MultiHeadAttention(cfg).eval()

    B, L = 2, 5
    x = torch.randn(B, L, cfg.d_model)
    out, attn = mha(x, x, x)

    assert out.shape == (B, L, cfg.d_model)
    assert attn.shape == (B, cfg.n_heads, L, L)


def test_cross_attention_shapes():
    """Cross-attn: query length (Lq) and key/value length (Lk) differ."""
    torch.manual_seed(0)
    cfg = _cfg()
    mha = MultiHeadAttention(cfg).eval()

    B, Lq, Lk = 2, 3, 7
    q = torch.randn(B, Lq, cfg.d_model)       # decoder states
    mem = torch.randn(B, Lk, cfg.d_model)     # encoder memory
    out, attn = mha(q, mem, mem)

    assert out.shape == (B, Lq, cfg.d_model)
    assert attn.shape == (B, cfg.n_heads, Lq, Lk)


def test_attention_rows_sum_to_one():
    """Softmax weights over the key axis form a distribution (each query row sums to 1)."""
    torch.manual_seed(0)
    cfg = _cfg()
    mha = MultiHeadAttention(cfg).eval()

    x = torch.randn(2, 5, cfg.d_model)
    _, attn = mha(x, x, x)
    sums = attn.sum(dim=-1)                    # (B, n_heads, Lq)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masked_positions_get_zero_weight():
    """A boolean keep-mask (True=keep) must drive masked key positions to ~0 weight."""
    torch.manual_seed(0)
    cfg = _cfg()
    mha = MultiHeadAttention(cfg).eval()

    B, L = 2, 6
    x = torch.randn(B, L, cfg.d_model)

    # Keep-mask broadcastable to scores (B, n_heads, Lq, Lk): pad the last 2 key
    # positions of batch item 0. Shape (B, 1, 1, L) broadcasts over heads + queries.
    keep = torch.ones(B, 1, 1, L, dtype=torch.bool)
    keep[0, :, :, -2:] = False

    _, attn = mha(x, x, x, attn_mask=keep)

    # Masked keys for item 0 should receive ~zero probability from every query/head.
    assert attn[0, :, :, -2:].max().item() < 1e-6
    # Unmasked item 1 should be untouched (still a valid distribution).
    assert torch.allclose(attn[1].sum(dim=-1), torch.ones(cfg.n_heads, L), atol=1e-5)


def test_no_nan_when_a_full_row_is_masked():
    """finfo.min (not -inf) keeps softmax NaN-free even if a query sees no keys."""
    torch.manual_seed(0)
    cfg = _cfg()
    mha = MultiHeadAttention(cfg).eval()

    B, L = 1, 4
    x = torch.randn(B, L, cfg.d_model)
    keep = torch.ones(B, 1, L, L, dtype=torch.bool)
    keep[0, :, 0, :] = False                   # query 0 may attend to nothing

    _, attn = mha(x, x, x, attn_mask=keep)
    assert not torch.isnan(attn).any()


@pytest.mark.skip(reason="MQA/GQA not implemented yet — circle-back pass (attn_variant).")
def test_mqa_gqa_head_broadcasting():
    """When MQA/GQA land: n_kv_heads K/V heads broadcast across n_heads query heads."""
    raise NotImplementedError
