# Resumable checkpoint save/load (model+optim+sched+scaler+step+RNG).

import os
import torch
import random
from pathlib import Path
import numpy

class CheckpointManager():
    def __init__(self, out_dir, keep_last_n=5):
        self.out_dir = Path(out_dir)
        self.keep_last_n = keep_last_n
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.out_dir / "best.pt"

    def save(self, step, model, optimizer, *, scaler=None, ema=None, swa=None, best=None, config=None, is_best=False):
        # create payload dict
        payload_dict = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "step": step,
                        "best": best,
                        "ema": ema.state_dict() if ema is not None else None,
                        "swa": swa.state_dict() if swa is not None else None,
                        "rng": self._gather_rng(),
                        "config": config}
        
        # create step path
        step_path = self.out_dir / f"step_{step}.pt"
        self._atomic_save(payload_dict, step_path)

        # best model tracking
        if is_best:
            self._atomic_save(payload_dict, self.best_path)
        
        # prune old files
        self._rotate()

    def load(self, path, model, *, optimizer=None, scaler=None, ema=None, swa=None, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and ckpt["optimizer"]:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt["ema"]:
            ema.load_state_dict(ckpt["ema"])
        if swa is not None and ckpt["swa"]:
            swa.load_state_dict(ckpt["swa"])
        self._restore_rng(ckpt["rng"])
        return (ckpt["step"], ckpt.get("best"))

    def latest(self):
        ckpts = self._step_ckpts() # sorted oldest to newest
        # last entry is highest step, return its path or none if empty
        return ckpts[-1][-1] if ckpts else None
    
    def resume(self, model, **kwargs):
        p = self.latest()
        if p is None:
            return (0, None)
        return self.load(p, model, **kwargs)

# Private helpers

    def _gather_rng(self):
        d = {"torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "numpy": numpy.random.get_state(),
             "python": random.getstate()}
        return d

    # exact inverse of gather rng
    def _restore_rng(self, d):
        torch.set_rng_state(d["torch"])
        if torch.cuda.is_available() and d["cuda"] is not None:
            torch.cuda.set_rng_state_all(d["cuda"])
        numpy.random.set_state(d["numpy"])
        random.setstate(d["python"])

    def _atomic_save(self, obj, path):
        # Create a sibling temp path
        tmp = path.with_suffix(path.suffix + ".tmp")

        # Save
        torch.save(obj, tmp)

        # And replace. Final file is never half written.
        os.replace(tmp, path)
    
    def _rotate(self):
        ckpts = self._step_ckpts() # sorted
        for step, p in ckpts[:-self.keep_last_n]: # everything except the newest keep_last_n
            p.unlink() # delete the old checkpoint file from disk
    
    def _step_ckpts(self):
        # return [(step, path)...] for every step_*.pt
        out = []
        for p in self.out_dir.glob("step_*.pt"):
            try:
                step = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue # skip oddly named file that doenst parse
            out.append((step, p))
        out.sort(key=lambda sp: sp[0]) # ascend by step number, oldest first, newest last
        return out

def average_checkpoints(paths, key="model", map_location="cpu") -> dict:
    acc = None # acc for accumulator
    n = len(paths) # num ckpts we are averaging

    for i, path in enumerate(paths): # walk the ckpts in order
        # load one
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        # this is the sub-state-dict to average
        sd = ckpt[key]

        # first checkpoint seeds the accumulator
        if i == 0:
            acc = {}
            for k, v in sd.items():
                acc[k] = v.clone().float() if v.is_floating_point() else v.clone()
        else:
            # avg float tensors, accumulate, turn sums to means
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.float()
    
    # divide each float sum by count, return model shaped state dict for loading
    for k in acc:
        if acc[k].is_floating_point():
            acc[k] /= n


    return acc
