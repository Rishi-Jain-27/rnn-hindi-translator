# EMA / SWA param averagers: shadow tracks the right params (tied once, frozen excluded);
# EMA blend = decay*shadow + (1-decay)*theta; SWA = equal-weight running mean of snapshots;
# update touches only the shadow (not the model); store->copy_to->restore round-trips theta;
# state_dict survives a torch.save/load and reloads into a fresh averager
# needs torch; skipped where absent (mac), runs on colab

import io

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nmt.train.ema import EMA, SWA


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.lin = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.proj = nn.Linear(4, 10, bias=False)
        self.proj.weight = self.emb.weight        # shared tensor (3-way-style tie)
        self.norm = nn.LayerNorm(4)


def _fill(model, value):
    # set every param to a constant so averages are easy to predict
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)


def _snapshot(d):
    # detached copy of a {name: tensor} dict for before/after comparisons
    return {k: v.detach().clone() for k, v in d.items()}


def test_shadow_keys_match_trainable():
    m = Tied()
    m.norm.bias.requires_grad_(False)             # freeze one param
    ema = EMA(m)
    trainable = {name for name, p in m.named_parameters() if p.requires_grad}
    assert set(ema.shadow.keys()) == trainable     # tied weight appears once, frozen excluded
    assert "norm.bias" not in ema.shadow
    assert "proj.weight" not in ema.shadow         # deduped to emb.weight by named_parameters
    assert "emb.weight" in ema.shadow


def test_ema_blend():
    m = Tiny()
    ema = EMA(m, decay=0.9)
    before = _snapshot(ema.shadow)                  # shadow_0 == initial params
    _fill(m, 5.0)                                   # move the live weights
    ema.update(m)
    for name, p in m.named_parameters():
        expected = 0.9 * before[name] + 0.1 * p.detach()   # decay*shadow + (1-decay)*theta
        assert torch.allclose(ema.shadow[name], expected, atol=1e-6)
    assert ema.num_updates == 1


def test_update_leaves_model_untouched():
    m = Tiny()
    ema = EMA(m, decay=0.9)
    theta = _snapshot(dict(m.named_parameters()))
    ema.update(m)
    for name, p in m.named_parameters():
        assert torch.allclose(p.detach(), theta[name])   # only the shadow changed


def test_swa_running_mean():
    m = Tiny()
    swa = SWA(m)
    _fill(m, 2.0); swa.update(m)                    # snapshot 1 -> mean = 2
    for v in swa.shadow.values():
        assert torch.allclose(v, torch.full_like(v, 2.0))
    _fill(m, 4.0); swa.update(m)                    # snapshot 2 -> mean = 3
    for v in swa.shadow.values():
        assert torch.allclose(v, torch.full_like(v, 3.0))
    _fill(m, 6.0); swa.update(m)                    # snapshot 3 -> mean = 4
    for v in swa.shadow.values():
        assert torch.allclose(v, torch.full_like(v, 4.0))
    assert swa.num_updates == 3


def test_store_copy_restore_roundtrip():
    m = Tiny()
    ema = EMA(m, decay=0.5)
    theta0 = _snapshot(dict(m.named_parameters()))
    with torch.no_grad():                           # make the shadow distinct from theta
        for v in ema.shadow.values():
            v.fill_(7.0)
    ema.store(m)                                    # back up live theta
    ema.copy_to(m)                                  # model <- shadow
    for p in m.parameters():
        assert torch.allclose(p.detach(), torch.full_like(p, 7.0))
    ema.restore(m)                                  # model <- backed-up theta
    for name, p in m.named_parameters():
        assert torch.allclose(p.detach(), theta0[name])


def test_state_dict_roundtrip():
    m = Tiny()
    ema = EMA(m)
    _fill(m, 3.0); ema.update(m)                    # num_updates -> 1, shadow moves
    buf = io.BytesIO()
    torch.save(ema.state_dict(), buf)              # serialize like checkpoint.py will
    buf.seek(0)
    sd = torch.load(buf)

    ema2 = EMA(Tiny())                              # fresh averager, different init shadow
    ema2.load_state_dict(sd)
    assert ema2.num_updates == ema.num_updates == 1
    assert set(ema2.shadow.keys()) == set(ema.shadow.keys())
    for name in ema.shadow:
        assert torch.allclose(ema2.shadow[name], ema.shadow[name])
