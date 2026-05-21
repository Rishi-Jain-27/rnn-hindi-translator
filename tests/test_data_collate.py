# token-batch sampler respects max_tokens; collate pad masks match lengths; length filter works
# uses a fake tokenizer so no sentencepiece needed (still needs torch for collate)

import pytest

torch = pytest.importorskip("torch")

from nmt.config import DataConfig
from nmt.data.dataset import TranslationDataset, TokenBatchSampler, Collate


class FakeTok:
    # minimal tokenizer: ids are arbitrary >=4; specials follow the contract
    pad_id, unk_id, bos_id, eos_id = 0, 1, 2, 3

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [4 + (len(w) % 10) for w in text.split()]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


def _files(tmp_path):
    # variable-length lines so bucketing/padding actually matters
    his = [" ".join(["a"] * n) for n in range(1, 30)]
    ens = [" ".join(["b"] * n) for n in range(1, 30)]
    hp, ep = tmp_path / "t.hi", tmp_path / "t.en"
    hp.write_text("\n".join(his) + "\n", encoding="utf-8")
    ep.write_text("\n".join(ens) + "\n", encoding="utf-8")
    return str(hp), str(ep)


def _ds(tmp_path, **kw):
    cfg = DataConfig(min_len=kw.get("min_len", 1), max_len=kw.get("max_len", 100))
    hp, ep = _files(tmp_path)
    return TranslationDataset(cfg, FakeTok(), hp, ep, train=True)


def test_sampler_respects_max_tokens(tmp_path):
    ds = _ds(tmp_path)
    max_tokens = 64
    sampler = TokenBatchSampler(ds.lengths, max_tokens, shuffle=False)
    seen = 0
    for batch in sampler:
        longest = max(ds.lengths[i] for i in batch)
        # padded size fits the budget, unless a single example is itself over-long
        assert len(batch) * longest <= max_tokens or len(batch) == 1
        seen += len(batch)
    assert seen == len(ds)            # every example placed in exactly one batch


def test_collate_masks_match_lengths(tmp_path):
    ds = _ds(tmp_path)
    collate = Collate(pad_id=0)
    batch = [ds[i] for i in range(5)]
    out = collate(batch)
    srcs = [s for s, _, _ in batch]
    tgts = [t for _, t, _ in batch]
    # shapes line up
    assert out["src_ids"].shape == out["src_pad_mask"].shape
    assert out["tgt_in"].shape == out["tgt_pad_mask"].shape == out["labels"].shape
    # keep-mask row sums == real (unpadded) lengths
    assert out["src_pad_mask"].sum(1).tolist() == [len(s) for s in srcs]
    assert out["tgt_pad_mask"].sum(1).tolist() == [len(t) for t in tgts]
    # masks are boolean keep-masks
    assert out["src_pad_mask"].dtype == torch.bool


def test_length_filter_drops_out_of_range(tmp_path):
    # tight min/max -> only mid-length lines survive
    cfg = DataConfig(min_len=2, max_len=5)
    hp, ep = _files(tmp_path)
    ds = TranslationDataset(cfg, FakeTok(), hp, ep, train=True)
    assert 0 < len(ds) < 29
    for src, tgt_in, labels in ds.examples:
        core = len(labels) - 1                     # labels = core + eos
        assert cfg.min_len <= len(src) <= cfg.max_len
        assert cfg.min_len <= core <= cfg.max_len
