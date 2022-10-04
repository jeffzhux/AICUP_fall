
import torchvision
from utils.config import ConfigDict

import models

def build_model(cfg: ConfigDict):
    
    args = cfg.copy()
    model_name = args.pop('type')
    if hasattr(torchvision.models, model_name):
        model = getattr(torchvision.models, model_name)(**args)
    else:
        model = models.__dict__[model_name](args)
    return model