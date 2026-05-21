# AdamW.
# Transformer betas (0.9, 0.98), eps 1e-9, with weight decay excluded from norm and bias parameters (separate no-decay param group).

import torch
from ..config import TrainConfig

def build_optimizer(model, cfg) -> torch.optim.AdamW:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    param_groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    
