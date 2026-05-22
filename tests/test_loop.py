# Trainer integration: a tiny model over a fixed batch must (a) run to max_steps and
# drive the dev loss down (the whole forward/loss/backward/step/token-norm chain works),
# and (b) run end-to-end with EMA + SWA + checkpoint + eval wired on, writing best.pt
# needs torch; skipped where absent (mac), runs on colab

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nmt.config import TrainConfig
from nmt.train.loop import Trainer


class TinyModel(nn.Module):
    # predicts labels from tgt_in token-by-token; enough to exercise the loop and learn a copy task
    def __init__(self, V, d):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.proj = nn.Linear(d, V)

    def forward(self, src_ids, tgt_in, src_pad_mask, tgt_pad_mask):
        return self.proj(self.emb(tgt_in))           # (B, T, V)


class FakeTok:
    pad_id = 0                                        # the loop only reads tokenizer.pad_id


def _batch(B=4, T=5, S=6, V=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    tgt_in = torch.randint(4, V, (B, T), generator=g)  # ids in [4,V): never pad/bos/eos
    return {
        "src_ids": torch.randint(4, V, (B, S), generator=g),
        "src_pad_mask": torch.ones(B, S, dtype=torch.bool),
        "tgt_in": tgt_in,
        "tgt_pad_mask": torch.ones(B, T, dtype=torch.bool),
        "labels": tgt_in.clone(),                     # copy task -> cleanly learnable
    }


def test_loss_decreases(tmp_path):
    torch.manual_seed(0)
    V, d = 20, 16
    cfg = TrainConfig(
        max_steps=60, grad_accum=1, warmup_steps=5, lr=3e-3, label_smoothing=0.0,
        log_every=10_000, eval_every=10_000, ckpt_every=10_000, patience=10**9,
        tracker="none", ema_decay=None, amp=False,    # ema off -> evaluate() uses the live (trained) weights
    )
    batch = _batch(V=V)
    tr = Trainer(cfg, TinyModel(V, d), [batch], [batch], FakeTok(), tmp_path)
    before = tr.evaluate()
    tr.train()
    after = tr.evaluate()
    assert tr.step == cfg.max_steps                   # ran to completion
    assert after < before                             # the loop actually learned


def test_runs_with_ema_swa_ckpt(tmp_path):
    torch.manual_seed(0)
    V, d = 20, 16
    cfg = TrainConfig(
        max_steps=10, grad_accum=2, warmup_steps=2, lr=1e-3,
        log_every=10_000, eval_every=5, ckpt_every=5, patience=10**9,
        tracker="none", ema_decay=0.99, use_swa=True, swa_start=4, swa_period=2, amp=False,
    )
    batch = _batch(V=V)
    tr = Trainer(cfg, TinyModel(V, d), [batch], [batch], FakeTok(), tmp_path)
    tr.train()                                        # exercises EMA + SWA + eval + checkpoint paths
    assert tr.step == cfg.max_steps
    assert (tmp_path / "best.pt").exists()            # first eval was a new best -> best.pt saved
    assert any(tmp_path.glob("step_*.pt"))            # periodic checkpoints written
