# TensorBoard/W&B logging: loss, grad norms, attention entropy, LR.

import torch
import time
import math

class Tracker:
    def __init__(self, cfg, log_dir, run_config=None):
        self.use_tb = cfg.tracker in ("tensorboard", "both")
        self.use_wandb = cfg.tracker in ("wandb", "both")

        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        
        if self.use_wandb:
            import wandb
            self.wandb = wandb.init(project=cfg.wandb_project, dir=log_dir, config=run_config)

    def log(self, metrics: dict, step: int):
        for tag, value in metrics.items():
            if self.use_tb:
                self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
                import wandb
                wandb.log(metrics, step=step)
    
    def close(self):
        if self.use_tb:
            self.writer.close()
        
        if self.use_wandb:
            import wandb
            wandb.finish()

def attention_entropy(attn, key_mask=None, normalize=False) -> float:
    # turns each weight p into -p log p — entropy formula building block
    intermediate = torch.special.entr(attn).sum(-1)
    if normalize:
        Tk = attn.size(-1)
        intermediate /= math.log(Tk)
    return intermediate.mean().item()


def grad_global_norm(params, norm_type=2.0) -> float:
    return torch.stack([param.grad.norm(norm_type) for param in params if param.grad is not None]).norm(norm_type).item()

class Throughput:
    def __init__(self):
        self.start = time.perf_counter()
        self.counter = 0
    
    def update(self, n_tokens) -> None:
        self.counter += n_tokens

    def rate(self) -> float:
        # calculate time and rate
        elapsed_seconds = time.perf_counter() - self.start
        rate = self.counter / (elapsed_seconds)

        # Reset
        self.start = time.perf_counter()
        self.counter = 0
        
        # return
        return rate

