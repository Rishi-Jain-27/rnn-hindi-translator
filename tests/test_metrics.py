# sanity tests for eval/metrics.py
# perfect match -> bleu/chrf maxed, ter 0; one wrong word -> scores drop off the ceiling

import pytest

pytest.importorskip("sacrebleu")  # pure-python; skip the whole module if not installed

from nmt.eval.metrics import compute_metrics


def _lower_keys(d):
    # tolerate either casing in the returned dict ("chrF"/"TER" or "chrf"/"ter")
    return {k.lower(): v for k, v in d.items()}


# two refs long enough to have 4-grams (bleu needs up to 4-grams)
REFS = ["the cat sat on the mat", "a dog ran in the park"]


def test_returns_three_float_metrics():
    res = _lower_keys(compute_metrics(list(REFS), list(REFS)))
    assert set(res) == {"bleu", "chrf", "ter"}
    assert all(isinstance(v, float) for v in res.values())


def test_perfect_match_is_top_score():
    # identical hyp == ref: bleu and chrf maxed at ~100, ter (edit rate) zero
    res = _lower_keys(compute_metrics(list(REFS), list(REFS)))
    assert res["bleu"] > 99.9
    assert res["chrf"] > 99.9
    assert res["ter"] < 0.01


def test_imperfect_is_between():
    # one substituted word per sentence: still lots of overlap, but not perfect
    hyps = ["the cat sat on a mat", "a dog walked in the park"]
    res = _lower_keys(compute_metrics(hyps, list(REFS)))
    assert 0.0 < res["bleu"] < 100.0
    assert 0.0 < res["chrf"] < 100.0
    assert res["ter"] > 0.0  # edits needed -> nonzero
