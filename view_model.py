import torch
import torch.nn as nn
import torchvision
from datasets.build import build_dataset
from datasets.collates.build import build_collate
from datasets.transforms import baseOnImageNet
from datasets.sampler.sampler import OOD_Sampler
from datasets.collates import RandomMixupCutMixCollate
from typing import Optional, Any, Tuple
from torch.autograd import Function
from utils.config import ConfigDict

x = torch.rand((4,3,32,32))
cfg = ConfigDict()
cfg.bsz_gpu = 512
cfg.world_size = 2
cfg.model_ema = ConfigDict(
    status = True,
    steps = 25,
    decay = 0.99998
)
cfg.model = ConfigDict(
    backbone = ConfigDict(
        type = 'efficientnet_v2_s',
        weights = 'EfficientNet_V2_S_Weights.IMAGENET1K_V1',
        dropout_rate = 0.1,
        num_classes = 33 
    )
    
)
cfg.epochs = 100
cfg.data = ConfigDict(
    collate = ConfigDict(
        type = 'TestTimeCollate',
    ),
    train = ConfigDict(
        root=f'data/OSDA/source',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='base',
        ),
        base_transform = dict(
            type='base',
            size = (16, 16)
        ),
        num_of_trans = 0
    ),
    test = ConfigDict(
        root=f'data/OSDA/target',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='base',
        ),
        base_transform = dict(
            type='base',
            size = (16, 16)
        ),
        num_of_trans = 0
    )
    
)
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
    
    def forward(self, x, reverse=False):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)

        x = self.backbone.classifier(x)

        return x

model = OSDANet(cfg.model)
test_set = build_dataset(cfg.data.test)

# dataset = test_set
print(test_set.class_to_idx)
# ood_sampler = OOD_Sampler(test_set, shuffle=True)

test_collate = build_collate(cfg.data.collate)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size = cfg.bsz_gpu,
    num_workers = 0,
    shuffle = False,
    # collate_fn = test_collate,
    pin_memory = True,
    drop_last = False
)
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
a = np.array([
    [1,2,4],
    [3,4,5],
    [3,4,5]
])
b = ['abc','bde', 'cbg']

fig, ax = plt.subplots(figsize=(20,20))


cm_display = metrics.ConfusionMatrixDisplay(a, display_labels=b)
cm_display.plot(ax = ax, cmap='Blues')
cm_display.ax_.set_title('Confusion Matrix')
plt.xticks(rotation=90)
cm_display.figure_.savefig(f'./test_experiment/confuse_metrix.png')

x, y = [1,1,0], [0,1,0]
print(metrics.classification_report(y,x, target_names=['a','b']))
