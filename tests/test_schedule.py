# lr_at: warmup ramps 0->peak, boundary is continuous, inv_sqrt + cosine tails behave
# pure python (no torch/spm) so this runs anywhere, mac included

import math

import pytest

from nmt.config import TrainConfig
from nmt.train.schedule import lr_at


def _cfg(schedule, warmup=4000, lr=7e-4, max_steps=100_000):
    # only the four fields lr_at reads; everything else stays at defaults
    return TrainConfig(schedule=schedule, warmup_steps=warmup, lr=lr, max_steps=max_steps)


def test_warmup_ramps_linearly():
    cfg = _cfg("inv_sqrt")
    assert lr_at(0, cfg) == 0.0                              # dead first step (0-indexed ramp)
    assert lr_at(cfg.warmup_steps // 2, cfg) == pytest.approx(cfg.lr / 2)
    # strictly increasing through warmup
    vals = [lr_at(s, cfg) for s in (0, 1000, 2000, 3000, 3999)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_peak_at_warmup_is_continuous():
    # both tails meet the warmup ramp at exactly cfg.lr at step == warmup_steps
    for sched in ("inv_sqrt", "cosine"):
        cfg = _cfg(sched)
        assert lr_at(cfg.warmup_steps, cfg) == pytest.approx(cfg.lr)
    # the left limit (warmup formula one step short) approaches the same peak
    cfg = _cfg("inv_sqrt")
    assert lr_at(cfg.warmup_steps - 1, cfg) == pytest.approx(cfg.lr, rel=1e-3)


def test_inv_sqrt_decays_monotonically():
    cfg = _cfg("inv_sqrt")
    vals = [lr_at(s, cfg) for s in (cfg.warmup_steps, 8000, 16000, 50000, 100000)]
    assert all(b < a for a, b in zip(vals, vals[1:]))       # strictly decreasing
    # matches the closed form, and the 1/sqrt law: 4x the step -> half the lr
    assert lr_at(16000, cfg) == pytest.approx(cfg.lr * math.sqrt(cfg.warmup_steps / 16000))
    assert lr_at(4 * cfg.warmup_steps, cfg) == pytest.approx(cfg.lr / 2)


def test_cosine_floors_at_min_lr_and_clamps():
    min_lr = 1e-5
    cfg = _cfg("cosine")
    # peak at warmup, floor at max_steps
    assert lr_at(cfg.warmup_steps, cfg, min_lr) == pytest.approx(cfg.lr)
    assert lr_at(cfg.max_steps, cfg, min_lr) == pytest.approx(min_lr)
    # past max_steps stays clamped -- cos must not curl back up
    assert lr_at(cfg.max_steps * 2, cfg, min_lr) == pytest.approx(min_lr)
    # halfway through decay sits halfway between peak and floor (cos(pi/2)=0)
    mid = (cfg.warmup_steps + cfg.max_steps) // 2
    assert lr_at(mid, cfg, min_lr) == pytest.approx(min_lr + 0.5 * (cfg.lr - min_lr), rel=1e-3)


def test_cosine_decays_monotonically():
    cfg = _cfg("cosine")
    vals = [lr_at(s, cfg, 1e-5) for s in (cfg.warmup_steps, 20000, 40000, 60000, 80000, 100000)]
    assert all(b < a for a, b in zip(vals, vals[1:]))
