# Exponential weight moving average (EMA) for model parameters.
# Evalaute the shadows, not the weights because the weights jitter around a lot

# SWA (stochastic weight averaging) also exists here for equal weights instead of exponential

import torch

class _ParamAverager():
    def __init__(self, model):
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        self.num_updates = 0
    
    def copy_to(self, model):
        with torch.no_grad():
            for name,param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.shadow[name])
    
    def store(self, model):
        self.backup = {}
        for name,param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
    
    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name])
    
    def state_dict(self):
        sd = {"shadow": self.shadow,
                "num_updates": self.num_updates}
        return sd

    def load_state_dict(self, sd):
        self.shadow = sd["shadow"]
        self.num_updates = sd["num_updates"]

class EMA(_ParamAverager):
    def __init__(self, model, decay=0.999):
        super(EMA, self).__init__(model)
        self.decay = decay
    
    def update(self, model):
        with torch.no_grad():
            for name,param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha= 1 - self.decay)
        self.num_updates += 1

class SWA(_ParamAverager):
    def __init__(self, model):
        super(SWA, self).__init__(model)
    
    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.num_updates).add_(param.detach()).div_(self.num_updates + 1)
        self.num_updates += 1
    



