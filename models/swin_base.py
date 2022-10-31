import torch.nn as nn
from utils.config import ConfigDict
import torchvision

class Swin_Base(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(Swin_Base, self).__init__()

        backbone_args = cfg.backbone.copy()
        backbone_name = backbone_args.pop('type')

        self.backbone = getattr(torchvision.models, backbone_name)(**backbone_args)
    def forward(self, x):
        x = self.backbone(x)
        return x