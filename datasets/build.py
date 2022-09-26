import torch
import torch.nn.functional as F

from datasets.dataset import AICUP_ImageFolder
from datasets.transforms import build_transform
from utils.config import ConfigDict

def build_dataset(cfg: ConfigDict):
    args = cfg.copy()

    trans_args = args.pop('transform')
    transform = build_transform(trans_args)

    ds = AICUP_ImageFolder(**args, transform=transform)
    return ds