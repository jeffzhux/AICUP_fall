import torch.nn as nn
# from mmcls.models.backbones import EfficientNet
# from mmcls.models.necks import GlobalAveragePooling
from utils.config import ConfigDict
import torchvision

class EfficientNet_Base(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(EfficientNet_Base, self).__init__()
        args = cfg.copy()

        backbone_args = cfg.backbone.copy()

        backbone_name = backbone_args.pop('type')
        num_classes = backbone_args.pop('num_classes')
        self.backbone = getattr(torchvision.models, backbone_name)(**backbone_args)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        return x