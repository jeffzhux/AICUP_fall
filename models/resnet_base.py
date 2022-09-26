

import torch.nn as nn
from models.backbones.resnet import ResNet
from utils.config import ConfigDict

class ResNet_Base(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(ResNet_Base, self).__init__()
        args = cfg.copy()
        self.backbone = ResNet(**(args.backbone))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
        
        return x