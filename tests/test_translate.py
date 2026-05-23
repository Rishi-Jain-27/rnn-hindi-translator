# sanity tests for decode/translate.py (batched strings->strings glue)
# translate's own job is the wiring: chunk by cfg.batch_size, per-chunk pad, keep input
# order, encode(+eos)/decode correctly. greedy itself is tested in test_greedy.py, so here
# we monkeypatch greedy_decode with a recorder that logs src shapes and echoes each row back
# (pad+eos stripped) -- that makes translate round-trip its input, which pins the glue.

import pytest

torch = pytest.importorskip("torch")  # needs torch for torch.tensor / .to / .tolist

from nmt.decode import translate as T   # import the module so we can patch T.greedy_decode
from nmt.config import DecodeConfig


# fake tokenizer: a sentence is a space-separated list of int "tokens".
# encode parses them (+ optional bos/eos); decode space-joins them back.
class FakeTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def __init__(self):
        self.encode_flags = []                       # (add_bos, add_eos) per encode call

    def encode(self, text, add_bos=False, add_eos=False):
        self.encode_flags.append((add_bos, add_eos))
        ids = [int(t) for t in text.split()]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


# trivial model: translate only reads its device off the first parameter
class FakeModel:
    def parameters(self):
        return iter([torch.zeros(1)])


# stand-in for greedy_decode: records each call's (B, S) src shape, and echoes every row
# back with pad + eos removed -- so translate reconstructs the original input ids
class RecordingGreedy:
    def __init__(self):
        self.shapes = []

    def __call__(self, model, src, max_len, ids):
        self.shapes.append(tuple(src.shape))
        out = []
        for row in src.tolist():
            out.append([t for t in row if t != ids.pad_id and t != ids.eos_id])
        return out


def _cfg(batch_size=2, max_decode_len=16):
    return DecodeConfig(batch_size=batch_size, max_decode_len=max_decode_len)


def test_round_trip_preserves_order(monkeypatch):
    # echo greedy + round-trip tokenizer -> output should equal the input, in order,
    # across multiple chunks
    rec = RecordingGreedy()
    monkeypatch.setattr(T, "greedy_decode", rec)
    tok = FakeTokenizer()
    sentences = ["5 6 7", "8", "9 10 11 12", "13 14"]
    out = T.translate(FakeModel(), sentences, tok, _cfg(batch_size=2))
    assert out == sentences


def test_chunking_and_per_chunk_padding(monkeypatch):
    # batch_size 2 -> two chunks; each chunk pads to ITS OWN longest, not the global one.
    # chunk 1: [5,6,7,eos](4) + [8,eos](2) -> width 4
    # chunk 2: [9,10,11,12,eos](5) + [13,14,eos](3) -> width 5
    rec = RecordingGreedy()
    monkeypatch.setattr(T, "greedy_decode", rec)
    tok = FakeTokenizer()
    sentences = ["5 6 7", "8", "9 10 11 12", "13 14"]
    T.translate(FakeModel(), sentences, tok, _cfg(batch_size=2))
    assert rec.shapes == [(2, 4), (2, 5)]


def test_last_chunk_is_smaller(monkeypatch):
    # 3 sentences, batch_size 2 -> chunk sizes 2 then 1 (last chunk shorter, no special-casing)
    rec = RecordingGreedy()
    monkeypatch.setattr(T, "greedy_decode", rec)
    tok = FakeTokenizer()
    sentences = ["5", "6 7", "8 9 10"]
    out = T.translate(FakeModel(), sentences, tok, _cfg(batch_size=2))
    assert [shape[0] for shape in rec.shapes] == [2, 1]
    assert out == sentences


def test_encode_uses_eos_not_bos(monkeypatch):
    # train/inference parity: source encoded with add_eos=True, add_bos=False
    rec = RecordingGreedy()
    monkeypatch.setattr(T, "greedy_decode", rec)
    tok = FakeTokenizer()
    T.translate(FakeModel(), ["5 6", "7"], tok, _cfg(batch_size=8))
    assert tok.encode_flags == [(False, True), (False, True)]


def test_empty_input_returns_empty(monkeypatch):
    # empty in -> empty out, and greedy is never called
    rec = RecordingGreedy()
    monkeypatch.setattr(T, "greedy_decode", rec)
    out = T.translate(FakeModel(), [], FakeTokenizer(), _cfg())
    assert out == []
    assert rec.shapes == []
