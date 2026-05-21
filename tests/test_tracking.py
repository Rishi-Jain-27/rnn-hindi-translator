# tracking metrics + sink:
# attention_entropy: uniform attention -> log(Tk) (nats), 1.0 normalized; one-hot -> 0;
# grad_global_norm matches the norm of all grads concatenated;
# Throughput.rate is positive and resets its counter each call;
# Tracker("none") is a silent no-op (no writer, log/close don't raise)
# needs torch; skipped where absent (mac), runs on colab

import math
import time

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nmt.config import TrainConfig
from nmt.train.tracking import Tracker, attention_entropy, grad_global_norm, Throughput


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)


def test_attention_entropy_uniform():
    B, H, Tq, Tk = 2, 3, 5, 8
    attn = torch.full((B, H, Tq, Tk), 1.0 / Tk)        # every key equally weighted
    assert math.isclose(attention_entropy(attn), math.log(Tk), rel_tol=1e-5)   # max entropy = log(Tk)
    assert math.isclose(attention_entropy(attn, normalize=True), 1.0, rel_tol=1e-5)


def test_attention_entropy_onehot():
    B, H, Tq, Tk = 2, 3, 5, 8
    attn = torch.zeros(B, H, Tq, Tk)
    attn[..., 0] = 1.0                                  # all weight on one key -> fully focused
    assert abs(attention_entropy(attn)) < 1e-6         # entropy 0
    assert abs(attention_entropy(attn, normalize=True)) < 1e-6


def test_grad_global_norm_matches_concat():
    m = Tiny()
    sum(p.sum() for p in m.parameters()).backward()    # populate .grad on every param
    got = grad_global_norm(m.parameters())
    flat = torch.cat([p.grad.reshape(-1) for p in m.parameters()])  # all grads in one vector
    assert math.isclose(got, flat.norm(2.0).item(), rel_tol=1e-5)


def test_throughput_positive_and_resets():
    t = Throughput()
    t.update(1000)
    time.sleep(0.005)                                  # guarantee a nonzero elapsed window
    r = t.rate()
    assert r > 0                                       # tokens/sec is positive
    time.sleep(0.005)
    assert t.rate() == 0.0                             # counter was reset -> 0 tokens this window


def test_tracker_none_is_noop(tmp_path):
    cfg = TrainConfig(tracker="none")
    tr = Tracker(cfg, tmp_path)
    assert not hasattr(tr, "writer")                   # nothing initialized
    tr.log({"train/loss": 1.0}, step=0)                # must not raise
    tr.close()                                         # must not raise


def test_tracker_tensorboard_writes(tmp_path):
    pytest.importorskip("tensorboard")                 # SummaryWriter needs the tensorboard package
    cfg = TrainConfig(tracker="tensorboard")
    tr = Tracker(cfg, tmp_path)
    tr.log({"train/loss": 1.0, "train/lr": 0.001}, step=0)
    tr.close()
    assert any(tmp_path.iterdir())                     # an event file was written under log_dir
