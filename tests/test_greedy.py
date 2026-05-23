# sanity tests for decode/greedy.py (BATCHED greedy, trimmed output)
# greedy_decode(model, src (B,S), max_len, cfg) -> list of B lists of ints,
# each row trimmed at its first eos (exclusive); trailing pads fall off with it.
#
# two model kinds:
#   - ScriptedModel: forces which token each row picks each step (and counts decoder
#     calls), so we can test trimming, the .all() early stop, and run-to-max_len exactly
#   - a tiny real Transformer: determinism + row independence on the real decode path

import pytest

torch = pytest.importorskip("torch")  # skip on the Mac; runs on Colab

from nmt.config import ModelConfig
from nmt.model.transformer import Transformer
from nmt.decode.greedy import greedy_decode


# ---- scripted fake model: drives the argmax, row by row, step by step ----

class ScriptedModel(torch.nn.Module):
    # fake encoder-decoder: row b's chosen next token at step s is schedule[b][s].
    # greedy only reads the last position's argmax, so we set that one logit high.
    # self.calls counts decoder calls = number of loop iterations (for the early-stop test).
    def __init__(self, schedule, vocab_size):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))   # one param -> model has a .device
        self.schedule = schedule                       # B lists of ids, one entry per step
        self.vocab_size = vocab_size
        self.calls = 0

    def encoder(self, src, src_pad_mask):
        B, S = src.shape
        return torch.zeros(B, S, 1)                     # memory; greedy just passes it through

    def decoder(self, decoder_input, memory, tgt_pad_mask, src_pad_mask):
        self.calls += 1
        B, t = decoder_input.shape
        step = t - 1                                    # seed length 1 -> step 0
        logits = torch.zeros(B, t, self.vocab_size)
        for b in range(B):
            logits[b, -1, self.schedule[b][step]] = 1.0  # argmax at last pos -> scripted id
        return logits


def _ids_cfg(vocab_size=10):
    # only the id contract + vocab_size matter for the scripted model
    # default ids: pad/unk/bos/eos = 0/1/2/3
    return ModelConfig(d_model=8, n_heads=2, n_enc_layers=1, n_dec_layers=1,
                       d_ff=16, max_len=32, vocab_size=vocab_size,
                       dropout=0.0, attn_dropout=0.0)


def _dummy_src(B=2):
    return torch.full((B, 2), 5, dtype=torch.long)     # contents irrelevant to the fake model


def test_scripted_trims_at_eos():
    # row 0 emits eos at step 2, row 1 at step 1. each row is cut at its own first eos,
    # and the pads forced onto the finished row 1 fall off with the cut.
    cfg = _ids_cfg()
    eos = cfg.eos_id                                   # 3
    schedule = [
        [5, 6, eos, 5, 5, 5],
        [7, eos, 4, 4, 4, 4],
    ]
    model = ScriptedModel(schedule, cfg.vocab_size)
    out = greedy_decode(model, _dummy_src(), max_len=6, cfg=cfg)

    assert out == [[5, 6], [7]]
    # contract: no eos and no pad survive the trim
    for row in out:
        assert cfg.eos_id not in row and cfg.pad_id not in row


def test_stops_early_when_all_finished():
    # both rows finish by step 2, so the loop should break after 3 decoder calls,
    # not run the full max_len of 6 (this is the .all() early stop)
    cfg = _ids_cfg()
    eos = cfg.eos_id
    schedule = [
        [5, 6, eos, 5, 5, 5],
        [7, eos, 4, 4, 4, 4],
    ]
    model = ScriptedModel(schedule, cfg.vocab_size)
    greedy_decode(model, _dummy_src(), max_len=6, cfg=cfg)
    assert model.calls == 3


def test_no_eos_runs_full_and_keeps_all():
    # nobody emits eos -> loop runs the full max_len and nothing is trimmed
    cfg = _ids_cfg()
    schedule = [[5] * 6, [6] * 6]
    model = ScriptedModel(schedule, cfg.vocab_size)
    out = greedy_decode(model, _dummy_src(), max_len=6, cfg=cfg)
    assert out == [[5] * 6, [6] * 6]
    assert model.calls == 6


# ---- tiny real transformer: properties of the actual decode path ----

def _tiny_real_model():
    cfg = ModelConfig(d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                      d_ff=64, max_len=64, vocab_size=32, dropout=0.0, attn_dropout=0.0)
    torch.manual_seed(0)
    return Transformer(cfg), cfg


def test_returns_list_of_int_lists():
    model, cfg = _tiny_real_model()
    src = torch.tensor([[4, 5, 6, 7, cfg.eos_id],
                        [8, 9, 10, 11, cfg.eos_id]])
    out = greedy_decode(model, src, max_len=12, cfg=cfg)
    assert isinstance(out, list) and len(out) == 2
    for row in out:
        assert isinstance(row, list)
        assert all(isinstance(tok, int) for tok in row)   # plain ints, not tensors
        assert len(row) <= 12                              # never longer than max_len
        assert cfg.eos_id not in row                       # eos trimmed off


def test_deterministic():
    # greedy = argmax, eval mode (dropout off) -> identical across calls
    model, cfg = _tiny_real_model()
    src = torch.tensor([[4, 5, 6, 7, cfg.eos_id],
                        [8, 9, 10, 11, cfg.eos_id]])
    a = greedy_decode(model, src, max_len=12, cfg=cfg)
    b = greedy_decode(model, src, max_len=12, cfg=cfg)
    assert a == b


def test_identical_rows_decode_identically():
    # same source in every row, no padding -> rows are independent, so every
    # output row must come out identical
    model, cfg = _tiny_real_model()
    one = [4, 5, 6, 7, cfg.eos_id]
    src = torch.tensor([one, one, one])
    out = greedy_decode(model, src, max_len=12, cfg=cfg)
    assert out[0] == out[1] == out[2]


def test_padded_source_runs():
    # mixed-length batch (row 1 padded) should decode without error
    model, cfg = _tiny_real_model()
    src = torch.tensor([[4, 5, 6, 7, cfg.eos_id],
                        [8, 9, cfg.eos_id, cfg.pad_id, cfg.pad_id]])
    out = greedy_decode(model, src, max_len=10, cfg=cfg)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(row, list) for row in out)
