import torch
import torch.nn as nn
from utils.config import ConfigDict
import torchvision
from typing import Optional, Any, Tuple
from torch.autograd import Function

class GradientReverseFunction(Function):
    
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class OSDANet(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(OSDANet, self).__init__()
        args = cfg.copy()

        backbone_args = cfg.backbone.copy()

        backbone_name = backbone_args.pop('type')
        num_classes = backbone_args.pop('num_classes')
        dropout_rate =  backbone_args.pop('dropout_rate') if backbone_args.get('dropout_rate') != None else None
        self.backbone = getattr(torchvision.models, backbone_name)(**backbone_args)
        if dropout_rate != None:
            self.backbone.classifier[-2].p = dropout_rate
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)

        self.grl = GradientReverseLayer()

    def forward(self, x, reverse=False):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if reverse:
            x = self.grl(x)
        x = self.backbone.classifier(x)

        return x