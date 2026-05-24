# the beam-search equivalence gate + sanity checks for decode/beam.py.
# beam_search(model, tokenizer, src (1,S), cfg) -> one list of ints (the best hypothesis).
# core gate: with beam_size=1, beam must reproduce greedy_decode token-for-token on the
# same random weights -- a beam of 1 always follows the argmax, so they must agree.
# (ModelConfig carries vocab_size + pad/bos/eos ids, so it doubles as the "tokenizer"
#  beam_search reads ids off; a DecodeConfig supplies beam_size/length_penalty/max_decode_len.)

import pytest

torch = pytest.importorskip("torch")  # runs on Colab, skips on the Mac

from nmt.config import ModelConfig, DecodeConfig
from nmt.model.transformer import Transformer
from nmt.decode.greedy import greedy_decode
from nmt.decode.beam import beam_search


def _tiny_model():
    cfg = ModelConfig(d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                      d_ff=64, max_len=64, vocab_size=32, dropout=0.0, attn_dropout=0.0)
    torch.manual_seed(0)
    return Transformer(cfg), cfg


def _decode_cfg(beam_size=1, max_decode_len=20, length_penalty=0.6):
    return DecodeConfig(mode="beam", beam_size=beam_size,
                        length_penalty=length_penalty, max_decode_len=max_decode_len)


def test_beam1_matches_greedy_single():
    # beam_size=1 follows the argmax every step -> identical to greedy on one sentence
    model, mcfg = _tiny_model()
    src = torch.tensor([[4, 5, 6, 7, mcfg.eos_id]])                  # (1, S)
    beam_out = beam_search(model, mcfg, src, _decode_cfg(beam_size=1, max_decode_len=20))
    greedy_out = greedy_decode(model, src, max_len=20, cfg=mcfg)[0]  # row 0 of the batch
    assert beam_out == greedy_out


def test_beam1_matches_greedy_longer():
    # more source tokens / longer cap, still must track greedy exactly
    model, mcfg = _tiny_model()
    src = torch.tensor([[4, 5, 6, 7, 8, 9, 10, mcfg.eos_id]])
    beam_out = beam_search(model, mcfg, src, _decode_cfg(beam_size=1, max_decode_len=30))
    greedy_out = greedy_decode(model, src, max_len=30, cfg=mcfg)[0]
    assert beam_out == greedy_out


def test_returns_list_of_ints():
    # beam_size>1: result is a flat list of plain python ints, within the length cap, no eos
    model, mcfg = _tiny_model()
    src = torch.tensor([[4, 5, 6, 7, mcfg.eos_id]])
    out = beam_search(model, mcfg, src, _decode_cfg(beam_size=4, max_decode_len=20))
    assert isinstance(out, list)
    assert all(isinstance(tok, int) for tok in out)                 # ints, not 0-dim tensors
    assert len(out) <= 20                                           # never longer than max_decode_len
    assert mcfg.eos_id not in out                                   # eos never kept in the output


def test_deterministic():
    # eval mode (dropout off) + argmax-style search -> identical across calls
    model, mcfg = _tiny_model()
    src = torch.tensor([[4, 5, 6, 7, mcfg.eos_id]])
    a = beam_search(model, mcfg, src, _decode_cfg(beam_size=5, max_decode_len=20))
    b = beam_search(model, mcfg, src, _decode_cfg(beam_size=5, max_decode_len=20))
    assert a == b
