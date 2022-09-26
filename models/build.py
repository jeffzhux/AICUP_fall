
from utils.config import ConfigDict
import models

def build_model(cfg: ConfigDict):
    
    args = cfg.copy()
    model_name = args.pop('type')

    return models.__dict__[model_name](args)