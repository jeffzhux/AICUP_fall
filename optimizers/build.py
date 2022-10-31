import torch

import optimizers
from utils.config import ConfigDict

def build_optimizer(cfg: ConfigDict, params):
    args = cfg.copy()
    name = args.pop('type')
    if hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)(params, **args)
    else:
        optimizer = optimizers.__dict__[name](params, **args)

    return optimizer