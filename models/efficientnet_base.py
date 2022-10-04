import torch.nn as nn
# from mmcls.models.backbones import EfficientNet
# from mmcls.models.necks import GlobalAveragePooling
from utils.config import ConfigDict
import torchvision

class EfficientNet_Base(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(EfficientNet_Base, self).__init__()
        args = cfg.copy()
        backbone_args = args.pop('backbone')

        backbone_name = backbone_args.pop('type')
        num_classes = backbone_args.pop('num_classes')
        # print("1 ",backbone_args)
        self.backbone = getattr(torchvision.models, backbone_name)(**backbone_args)
        # print("3 ",list(self.backbone.children())[-1][-1].out_features)
        self.linear = nn.Linear(list(self.backbone.children())[-1][-1].out_features, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x