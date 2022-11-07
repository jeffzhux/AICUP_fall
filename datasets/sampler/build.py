from torch.utils import data
from utils.config import ConfigDict
from datasets import sampler

def build_sampler(dataset, cfg: ConfigDict):
    
    args = cfg.copy()
    name = args.pop('type')
    if name == None:
        return data.__dict__[name](dataset=dataset, **args)
    else:
        return sampler.__dict__[name](dataset=dataset, **args)