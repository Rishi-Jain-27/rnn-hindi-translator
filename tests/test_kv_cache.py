# the KV-cache gate: incremental cached greedy must produce the SAME tokens as the old
# full-prefix (uncached) decode on identical (random) weights. needs no training -- it
# checks the cache reproduces the math, not that the translations are good.
#
# the uncached reference is built inline here (greedy.py is now cache-only): it feeds the
# whole growing prefix to model.decoder with kv_cache=None (the training/causal path).

import pytest

torch = pytest.importorskip("torch")  # runs on Colab, skips on the Mac

from nmt.config import ModelConfig
from nmt.model.transformer import Transformer
from nmt.decode.greedy import greedy_decode
from nmt.decode.cache import new_cache


def _tiny_model():
    cfg = ModelConfig(d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                      d_ff=64, max_len=64, vocab_size=32, dropout=0.0, attn_dropout=0.0)
    torch.manual_seed(0)
    return Transformer(cfg), cfg


def _uncached_greedy(model, src, max_len, cfg):
    # reference path: re-feed the full prefix every step, kv_cache=None (causal full forward)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        src_pad_mask = src != cfg.pad_id
        B = src.shape[0]
        memory = model.encode(src, src_pad_mask)
        dec_in = torch.full((B, 1), cfg.bos_id, device=device)      # seed BOS
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        recorded = []
        for _ in range(max_len):
            tgt_pad = dec_in != cfg.pad_id
            logits = model.decoder(dec_in, memory, tgt_pad, src_pad_mask)   # kv_cache defaults None
            nxt = logits[:, -1, :].argmax(dim=-1)
            nxt = nxt.masked_fill(finished, cfg.pad_id)
            recorded.append(nxt)
            finished = finished | (nxt == cfg.eos_id)
            dec_in = torch.cat([dec_in, nxt.unsqueeze(1)], dim=1)           # grow the prefix
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


def test_new_cache_shape():
    assert new_cache(6) == [None] * 6
    assert new_cache(1) == [None]


def test_cached_matches_uncached_batched():
    # mixed-length padded batch -> cached and uncached must agree token-for-token
    model, cfg = _tiny_model()
    src = torch.tensor([[4, 5, 6, 7, cfg.eos_id],
                        [8, 9, cfg.eos_id, cfg.pad_id, cfg.pad_id]])
    cached = greedy_decode(model, src, max_len=15, cfg=cfg)
    uncached = _uncached_greedy(model, src, max_len=15, cfg=cfg)
    assert cached == uncached


def test_cached_matches_uncached_single():
    # single sentence (B=1)
    model, cfg = _tiny_model()
    src = torch.tensor([[4, 5, 6, 7, cfg.eos_id]])
    cached = greedy_decode(model, src, max_len=20, cfg=cfg)
    uncached = _uncached_greedy(model, src, max_len=20, cfg=cfg)
    assert cached == uncached


def test_cached_matches_uncached_longer_run():
    # force more decode steps so the cache grows over many positions
    model, cfg = _tiny_model()
    src = torch.tensor([[4, 5, 6, 7, 8, 9, 10, cfg.eos_id],
                        [11, 12, 13, cfg.eos_id, cfg.pad_id, cfg.pad_id, cfg.pad_id, cfg.pad_id]])
    cached = greedy_decode(model, src, max_len=30, cfg=cfg)
    uncached = _uncached_greedy(model, src, max_len=30, cfg=cfg)
    assert cached == uncached
