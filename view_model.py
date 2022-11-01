import torch
import torch.nn as nn
import torchvision

class EfficientNet_Base(nn.Module):
    def __init__(self):
        super(EfficientNet_Base, self).__init__()
        self.backbone =  torchvision.models.efficientnet_b0()
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, 33)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
model = EfficientNet_Base()

for name, params in model.backbone.named_parameters():
    # print(f'{name},  {child}')
    if 'classifier' in name:
        print(name)
        params.requires_grad = True
    else:
        params.requires_grad = False

for name, child in model.backbone.named_children():
    print(f'-----{name}----{child}')
    for name2, param in child.named_parameters():
        print(name2)
    
