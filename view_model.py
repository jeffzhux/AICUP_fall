import torch
import torch.nn as nn
import torchvision
from datasets.dataset import Group_ImageFolder
from datasets.transforms import baseOnImageNet
from datasets.collates import CutMixCollate, RandomMixupCutMixCollate
from utils.util import group_accuracy
class EfficientNet_Group_Base(nn.Module):
    def __init__(self):
        super(EfficientNet_Group_Base, self).__init__()
        self.backbone =  torchvision.models.efficientnet_b0()
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
model = EfficientNet_Group_Base()
pred = model(x)

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
for img, target in dataloader:
    # pred = model(img)
    # acc1, acc5 = group_accuracy(pred, target, groups_range, (1, 5))
    # print(acc5)
    break