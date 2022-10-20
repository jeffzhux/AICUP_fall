import torchvision
import datasets
from datasets.transforms import build_transform, build_target_transform
from utils.config import ConfigDict

def build_dataset(cfg: ConfigDict):
    args = cfg.copy()
    
    trans_args = args.pop('transform')
    transform = build_transform(trans_args)

    target_transform = None
    if args.get('target_transform') != None:
        target_trans_args = args.pop('target_transform')
        target_transform = build_target_transform(target_trans_args)
    ds_name = args.pop('type')

    if hasattr(torchvision.datasets, ds_name):
        ds = getattr(torchvision.datasets, ds_name)(**args, transform=transform)
    else:
        ds = datasets.__dict__[ds_name](**args, transform = transform, target_transform = target_transform)
    # ds = AICUP_ImageFolder(**args, transform=transform)
    return ds