import torch
import torch.nn as nn
from utils.config import ConfigDict
import torchvision

class LocNet(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(LocNet, self).__init__()
        args = cfg.copy()

        backbone_args = cfg.backbone.copy()

        backbone_name = backbone_args.pop('type')
        num_classes = backbone_args.pop('num_classes')
        dropout_rate =  backbone_args.pop('dropout_rate') if backbone_args.get('dropout_rate') != None else None
        self.backbone = getattr(torchvision.models, backbone_name)(**backbone_args)
        
        locDim = 128
        self.locLayer = nn.Linear(2, locDim)
        if dropout_rate != None:
            self.backbone.classifier[-2].p = dropout_rate
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features + locDim, num_classes)
    
    def forward(self, x, loc):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        loc = self.locLayer(loc)
        x = torch.cat((x, loc), dim=-1)
        x = self.backbone.classifier(x)
        return x