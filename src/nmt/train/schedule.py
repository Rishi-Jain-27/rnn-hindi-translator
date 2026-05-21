# Linear warmup + cosine decay LR schedule.


import math

def lr_at(step, cfg, min_lr=0.0) -> float:
    if step < cfg.warmup_steps:
        # linear
        return cfg.lr * step / cfg.warmup_steps
    elif cfg.schedule == "inv_sqrt":
        # inv sqrt
        return cfg.lr * math.sqrt(cfg.warmup_steps / step)
    else:
        # cos
        progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        progress = min(progress, 1.0)
        return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * progress))



