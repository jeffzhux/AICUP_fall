import torch
import torch.nn as nn
import torchvision
from datasets.dataset import Group_ImageFolder
from datasets.transforms import baseOnImageNet
from datasets.collates import RandomMixupCutMixCollate
from utils.util import ExponentialMovingAverage
from utils.util import set_weight_decay
from utils.config import ConfigDict
class EfficientNet_Group_Base(nn.Module):
    def __init__(self):
        super(EfficientNet_Group_Base, self).__init__()
        self.backbone =  torchvision.models.efficientnet_b0()
        self.backbone.classifier[-2].p = 0.4
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, 36)

        groups = [
            ['asparagus', 'onion', 'others', 'greenhouse', 'chinesecabbage', 'roseapple', 'passionfruit'],
            ['sesbania', 'lemon','litchi', 'chinesechives', 'pennisetum', 'longan', 'cauliflower', 'lettuce', 'loofah', 'custardapple', 'pear'],
            ['greenonion', 'papaya', 'mango', 'betel', 'bambooshoots', 'taro', 'waterbamboo', 'grape', 'kale', 'sweetpotato', 'broccoli', 'redbeans', 'soybeans', 'sunhemp', 'tea']
        ]
        # 組別的開始與結束
        self.group_range = self.get_group_range(groups)

    def get_group_range(self, group):
        group_range = []
        start = 0
        for g in group:
            end = start + len(g)
            group_range.append((start, end))
            start = end + 1
        return group_range

    def post_process_group(self, x):
        sub_groups = []
        for sub_g in self.group_range:
            start_idx, end_idx = sub_g
            sub_groups.append(x[:, start_idx:end_idx])
        sub_groups = torch.cat(sub_groups, dim=-1)
        return sub_groups
           
    def forward(self, x, post_process = True):
        x = self.backbone(x)
        if post_process:
            x = self.post_process_group(x)
            
        return x



groups = [
            ['asparagus', 'onion', 'others', 'greenhouse', 'chinesecabbage', 'roseapple', 'passionfruit'],
            ['sesbania', 'lemon','litchi', 'chinesechives', 'pennisetum', 'longan', 'cauliflower', 'lettuce', 'loofah', 'custardapple', 'pear'],
            ['greenonion', 'papaya', 'mango', 'betel', 'bambooshoots', 'taro', 'waterbamboo', 'grape', 'kale', 'sweetpotato', 'broccoli', 'redbeans', 'soybeans', 'sunhemp', 'tea']
        ]
groups_range = [
    (0, 7),
    (7, 18),
    (18, 33)
]
x = torch.rand((4,3,32,32))
cfg = ConfigDict()
cfg.bsz_gpu = 128
cfg.world_size = 2
cfg.model_ema = ConfigDict(
    status = True,
    steps = 25,
    decay = 0.99998
)
cfg.epochs = 100

def build_ema_model(model, cfg:ConfigDict):
    model_ema = None
    if hasattr(cfg, 'model_ema') and cfg.model_ema.status:
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = cfg.world_size * cfg.bsz_gpu * cfg.model_ema.steps / cfg.epochs
        decay = 1.0 - min(1.0, (1.0 - cfg.model_ema.decay) * adjust)
        model_ema = ExponentialMovingAverage(
            model,
            device = 'cuda',
            decay = decay
        )

    return model_ema
model = EfficientNet_Group_Base()
build_ema_model(model, cfg)
set_weight_decay(model, weight_decay=2e-5)

transform = baseOnImageNet()
dataset = Group_ImageFolder('./data/ID/train', groups, groups_range, transform=transform)
collate = RandomMixupCutMixCollate(33)
dataloader = torch.utils.data.DataLoader(
    dataset,
    collate_fn = collate,
    batch_size = 4,
    shuffle=True
)
# print(dataset.class_to_idx)

for name, sub_model in model.named_children():
    print(sub_model)

for img, target in dataloader:
    # pred = model(img)
    # acc1, acc5 = group_accuracy(pred, target, groups_range, (1, 5))
    # print(acc5)
    break