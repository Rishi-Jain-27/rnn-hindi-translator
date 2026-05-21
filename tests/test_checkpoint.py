# CheckpointManager: save->load round-trips model+optimizer(adam moments)+step+best+ema;
# resume on an empty dir is a clean (0, None); rotation keeps last N step_*.pt and spares best.pt;
# latest() picks the max step regardless of save order; RNG is restored exactly;
# average_checkpoints returns the elementwise mean
# needs torch; skipped where absent (mac), runs on colab

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nmt.train.checkpoint import CheckpointManager, average_checkpoints
from nmt.train.ema import EMA


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.lin = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def _opt_with_state(model):
    # one real step so AdamW populates exp_avg / exp_avg_sq (the state we must round-trip)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    sum(p.sum() for p in model.parameters()).backward()
    opt.step()
    return opt


def _fill(model, value):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)


def test_save_load_roundtrip(tmp_path):
    m = Tiny()
    opt = _opt_with_state(m)
    ema = EMA(m); ema.update(m)
    mgr = CheckpointManager(tmp_path)
    mgr.save(step=5, model=m, optimizer=opt, ema=ema, best=1.23)

    m2 = Tiny()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)   # fresh, no state yet
    ema2 = EMA(m2)
    step, best = mgr.load(mgr.latest(), m2, optimizer=opt2, ema=ema2)

    assert step == 5 and best == 1.23
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert n1 == n2 and torch.allclose(p1, p2)        # weights restored
    for k in ema.shadow:
        assert torch.allclose(ema2.shadow[k], ema.shadow[k])  # ema shadow restored
    # adam moment restored into the fresh optimizer
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert torch.allclose(s2[0]["exp_avg"], s1[0]["exp_avg"])


def test_resume_empty_is_fresh(tmp_path):
    mgr = CheckpointManager(tmp_path)
    assert mgr.latest() is None
    assert mgr.resume(Tiny()) == (0, None)               # fresh run path: no load, start at step 0


def test_rotation_keeps_last_n_and_best(tmp_path):
    m = Tiny()
    opt = _opt_with_state(m)
    mgr = CheckpointManager(tmp_path, keep_last_n=3)
    for step in range(1, 7):                              # save steps 1..6
        mgr.save(step=step, model=m, optimizer=opt, best=float(step), is_best=(step == 2))

    steps = sorted(int(p.stem.split("_")[1]) for p in tmp_path.glob("step_*.pt"))
    assert steps == [4, 5, 6]                             # only newest 3 step files survive
    assert (tmp_path / "best.pt").exists()               # best.pt never rotated away


def test_latest_picks_max_step(tmp_path):
    m = Tiny()
    opt = _opt_with_state(m)
    mgr = CheckpointManager(tmp_path)
    for step in (3, 1, 2):                                # save out of order
        mgr.save(step=step, model=m, optimizer=opt)
    assert mgr.latest().name == "step_3.pt"              # max parsed step, not save order


def test_average_checkpoints_is_mean(tmp_path):
    m = Tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path)
    _fill(m, 2.0); mgr.save(step=1, model=m, optimizer=opt)
    _fill(m, 4.0); mgr.save(step=2, model=m, optimizer=opt)

    avg = average_checkpoints([tmp_path / "step_1.pt", tmp_path / "step_2.pt"], key="model")
    for v in avg.values():
        assert torch.allclose(v, torch.full_like(v, 3.0))   # (2 + 4) / 2


def test_rng_roundtrip(tmp_path):
    m = Tiny()
    opt = _opt_with_state(m)
    mgr = CheckpointManager(tmp_path)
    torch.manual_seed(0)
    mgr.save(step=1, model=m, optimizer=opt)             # captures rng state S
    r1 = torch.randn(5)                                  # advances rng from S
    mgr.load(mgr.latest(), Tiny())                       # restores rng back to S
    r2 = torch.randn(5)                                  # same draw as r1 if rng truly restored
    assert torch.allclose(r1, r2)
