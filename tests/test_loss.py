# label_smoothed_nll: eps=0 collapses to plain CE; eps>0 matches an explicit soft-target oracle;
# pad positions are ignored (don't affect the sums, don't count in n_tokens)
# needs torch for logits/labels; skipped where torch is absent (mac), runs on colab

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from nmt.train.loss import label_smoothed_nll


def _batch(B=4, T=7, V=50, pad_id=0, seed=0):
    # random logits + non-pad labels, then pad a different-length tail per row
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g)
    labels = torch.randint(1, V, (B, T), generator=g)   # real ids in [1, V); 0 reserved for pad
    for b in range(B):
        labels[b, T - b:] = pad_id                       # rows lose 0,1,2,3 trailing tokens -> 6 pads
    return logits, labels


def test_eps0_equals_plain_ce():
    logits, labels = _batch()
    pad_id = 0
    loss_sum, nll_sum, _ = label_smoothed_nll(logits, labels, pad_id, smoothing=0.0)
    # at eps=0 the smoothed loss IS the plain nll
    assert torch.allclose(loss_sum, nll_sum)
    # and both equal torch's own cross-entropy (sum reduction, pad ignored)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                         ignore_index=pad_id, reduction="sum")
    assert torch.allclose(loss_sum, ce, atol=1e-4)


def test_matches_soft_target_oracle():
    logits, labels = _batch(seed=1)
    pad_id, eps = 0, 0.1
    V = logits.size(-1)
    loss_sum, nll_sum, _ = label_smoothed_nll(logits, labels, pad_id, smoothing=eps)
    # independent oracle: build the soft target q (1-eps on gold, eps/(V-1) elsewhere), take -(q*logp)
    logp = torch.log_softmax(logits, dim=-1)
    q = torch.full_like(logp, eps / (V - 1))
    q.scatter_(-1, labels.unsqueeze(-1), 1 - eps)
    keep = labels != pad_id
    oracle = (-(q * logp).sum(-1) * keep).sum()
    assert torch.allclose(loss_sum, oracle, atol=1e-4)
    # smoothing actually changes the number (sanity that eps did something)
    assert not torch.allclose(loss_sum, nll_sum)


def test_n_tokens_counts_non_pad():
    logits, labels = _batch(seed=2)
    pad_id = 0
    _, _, n_tok = label_smoothed_nll(logits, labels, pad_id, smoothing=0.1)
    assert int(n_tok) == int((labels != pad_id).sum())   # 4*7 - 6 = 22 here


def test_pad_positions_ignored():
    logits, labels = _batch(seed=3)
    pad_id = 0
    base = label_smoothed_nll(logits, labels, pad_id, smoothing=0.1)
    # blow up the logits at pad positions only; masking should make this a no-op
    logits2 = logits.clone()
    logits2[labels == pad_id] += 1000.0
    pert = label_smoothed_nll(logits2, labels, pad_id, smoothing=0.1)
    for a, b in zip(base, pert):
        assert torch.allclose(a.float(), b.float(), atol=1e-4)
