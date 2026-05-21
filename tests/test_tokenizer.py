# sentencepiece round-trip; no <unk>; id contract 0/1/2/3
# trains a tiny joint spm on a synthetic corpus (runs on Colab; skipped where spm is absent)

import random

import pytest

pytest.importorskip("sentencepiece")

from nmt.config import DataConfig
from nmt.data.tokenizer import train_tokenizer


def _corpus(tmp_path, n=4000):
    # varied bilingual lines so spm has enough pieces to learn
    hi_words = [f"श{i}" for i in range(40)]   # distinct devanagari-ish tokens
    en_words = [f"word{i}" for i in range(40)]
    rng = random.Random(0)
    his, ens = [], []
    for _ in range(n):
        k = rng.randint(3, 9)
        his.append(" ".join(rng.choice(hi_words) for _ in range(k)))
        ens.append(" ".join(rng.choice(en_words) for _ in range(k)))
    hp, ep = tmp_path / "train.hi", tmp_path / "train.en"
    hp.write_text("\n".join(his) + "\n", encoding="utf-8")
    ep.write_text("\n".join(ens) + "\n", encoding="utf-8")
    return str(hp), str(ep)


def _tok(tmp_path):
    # small vocab; hard_vocab_limit=False so a tiny corpus won't error on sizing
    cfg = DataConfig(cache_dir=str(tmp_path), vocab_size=400, tokenizer_model="unigram")
    hp, ep = _corpus(tmp_path)
    return train_tokenizer(cfg, hp, ep, hard_vocab_limit=False)


def test_id_contract(tmp_path):
    # the fixed pad/unk/bos/eos = 0/1/2/3 contract that build_model relies on
    tok = _tok(tmp_path)
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (0, 1, 2, 3)


def test_roundtrip(tmp_path):
    # decode(encode(x)) == x  (identity normalization keeps it exact)
    tok = _tok(tmp_path)
    for s in ["word1 word2 word3", "श1 श10 श25"]:
        assert tok.decode(tok.encode(s)) == s


def test_no_unk(tmp_path):
    # byte_fallback should cover anything, even chars never seen in the corpus
    tok = _tok(tmp_path)
    ids = tok.encode("zzz \U0001f680 ও unseen")
    assert tok.unk_id not in ids
