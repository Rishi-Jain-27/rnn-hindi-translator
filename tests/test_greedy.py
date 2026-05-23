# sanity tests for decode/greedy.py (single-sentence greedy)
# tiny random transformer: greedy should run, return ints, respect max_len,
# be deterministic (argmax), and stop right after emitting eos

import pytest

torch = pytest.importorskip("torch")

from nmt.config import ModelConfig
from nmt.model.transformer import Transformer
from nmt.decode.greedy import greedy_decode


def _tiny_model():
    # small but real encoder-decoder; default id contract pad/unk/bos/eos = 0/1/2/3
    cfg = ModelConfig(d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                      d_ff=64, max_len=64, vocab_size=32, dropout=0.0, attn_dropout=0.0)
    torch.manual_seed(0)
    return Transformer(cfg), cfg


def test_runs_and_returns_ints():
    model, cfg = _tiny_model()
    src = torch.tensor([4, 5, 6, 7, cfg.eos_id])  # 1-D source; greedy adds the batch dim
    out = greedy_decode(model, src, max_len=12, cfg=cfg)
    assert isinstance(out, list)
    assert all(isinstance(x, int) for x in out)
    assert 1 <= len(out) <= 12


def test_deterministic():
    # greedy = argmax, eval mode (dropout off) -> identical across calls
    model, cfg = _tiny_model()
    src = torch.tensor([4, 5, 6, 7, cfg.eos_id])
    a = greedy_decode(model, src, max_len=12, cfg=cfg)
    b = greedy_decode(model, src, max_len=12, cfg=cfg)
    assert a == b


def test_eos_is_terminal_when_present():
    # the loop breaks right after appending eos, so if eos shows up it's the last token
    model, cfg = _tiny_model()
    src = torch.tensor([4, 5, 6, 7, cfg.eos_id])
    out = greedy_decode(model, src, max_len=20, cfg=cfg)
    if cfg.eos_id in out:
        assert out[-1] == cfg.eos_id
        assert out.count(cfg.eos_id) == 1
