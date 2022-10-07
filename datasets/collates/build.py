
from utils.config import ConfigDict
from datasets import collates

def build_collate(cfg: ConfigDict):
    
    args = cfg.copy()
    name = args.pop('type')
    if name == None:
        return None
    else:
        return collates.__dict__[name](**args)