# build_optimizer: 2-D params (weights/embeddings) get weight decay, 1-D (bias/norm gains) get 0;
# frozen params excluded; tied params appear exactly once (AdamW would raise on duplicates)
# needs torch; skipped where absent (mac), runs on colab

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nmt.config import TrainConfig
from nmt.train.optim import build_optimizer


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)     # weight (10,4) 2-D -> decay
        self.lin = nn.Linear(4, 4)         # weight 2-D -> decay; bias 1-D -> no_decay
        self.norm = nn.LayerNorm(4)        # weight + bias 1-D -> no_decay


def test_split_by_ndim():
    cfg = TrainConfig()
    opt = build_optimizer(Tiny(), cfg)
    decay_g, nodecay_g = opt.param_groups
    # decay group holds only 2-D params; no-decay group only 1-D
    assert all(p.ndim >= 2 for p in decay_g["params"])
    assert all(p.ndim < 2 for p in nodecay_g["params"])
    assert len(decay_g["params"]) == 2       # emb.weight, lin.weight
    assert len(nodecay_g["params"]) == 3     # lin.bias, norm.weight, norm.bias
    # the decay value lives on the group, not the param list
    assert decay_g["weight_decay"] == cfg.weight_decay
    assert nodecay_g["weight_decay"] == 0.0


def test_hyperparams_passed_through():
    cfg = TrainConfig()
    opt = build_optimizer(Tiny(), cfg)
    for g in opt.param_groups:
        assert g["lr"] == cfg.lr
        assert g["betas"] == cfg.betas       # (0.9, 0.98) transformer betas
        assert g["eps"] == cfg.eps


def test_frozen_params_excluded():
    cfg = TrainConfig()
    m = Tiny()
    m.lin.weight.requires_grad_(False)
    opt = build_optimizer(m, cfg)
    placed = [p for g in opt.param_groups for p in g["params"]]
    assert all(p is not m.lin.weight for p in placed)
    assert len(placed) == 4                  # 5 params minus the frozen one


def test_tied_params_counted_once():
    cfg = TrainConfig()

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.proj = nn.Linear(4, 10, bias=False)
            self.proj.weight = self.emb.weight    # 3-way-style tie: literally the same tensor
            self.norm = nn.LayerNorm(4)

    m = Tied()
    opt = build_optimizer(m, cfg)                 # raises if the shared param were listed twice
    placed = [p for g in opt.param_groups for p in g["params"]]
    assert len(placed) == 3                       # shared emb/proj weight (once) + norm weight + bias
    assert sum(p is m.emb.weight for p in placed) == 1
