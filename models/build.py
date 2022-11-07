import torchvision
from utils.config import ConfigDict
from utils.util import ExponentialMovingAverage
import models


def build_model(cfg: ConfigDict):
    
    args = cfg.copy()
    model_name = args.pop('type')
    if hasattr(torchvision.models, model_name):
        model = getattr(torchvision.models, model_name)(**args)
    else:
        model = models.__dict__[model_name](args)
    return model

def build_ema_model(model, cfg:ConfigDict):
    model_ema = None
    if hasattr(cfg, 'model_ema') and cfg.model_ema.status:
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = cfg.world_size * cfg.bsz_gpu * cfg.model_ema.steps / cfg.epochs
        decay = 1.0 - min(1.0, (1.0 - cfg.model_ema.decay) * adjust)
        model_ema = ExponentialMovingAverage(
            model,
            device = 'cuda',
            decay = decay
        )

    return model_ema
