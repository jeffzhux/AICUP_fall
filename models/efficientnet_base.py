import torch.nn as nn
from mmcls.models.backbones import EfficientNet
from mmcls.models.necks import GlobalAveragePooling
from utils.config import ConfigDict

class EfficientNet_Base(EfficientNet):
    def __init__(self, cfg: ConfigDict):
        super(EfficientNet_Base, self).__init__()
        args = cfg.copy()
        num_classes = args.backbone.num_classes
        self.backbone = EfficientNet(args.backbone.arch)
        self.neck = GlobalAveragePooling()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.backbone(x)[0] # (N, C, H, W)
        x = self.neck(x) # (N, C, H, W) -> (N, C)
        x = self.softmax(x)
        
        return x