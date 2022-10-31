import torch
import torch.nn as nn
import torchvision

class EfficientNet_Base(nn.Module):
    def __init__(self):
        super(EfficientNet_Base, self).__init__()
        self.backbone =  torchvision.models.efficientnet_b0()
        print()
        print(self.backbone.classifier)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, 33)
        
        print(self.backbone.classifier)
    def forward(self, x):
        x = self.backbone(x)
        return x
model = EfficientNet_Base()
# for name, child in model.named_children():
#     print(f'{name} :  {child}')