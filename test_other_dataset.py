from operator import index
from tkinter import Image

from regex import A

from datasets.collates.collate import OtherMixupCollate,MixupCollate
from datasets.dataset import Others_ImageFolder, AICUP_ImageFolder


import torch
from datasets.transforms.aicup import baseOnImageNet
transform = baseOnImageNet()
# id_collate = MixupCollate(37)
id_dataset = AICUP_ImageFolder('./data/ID/valid', transform=transform)
id_loader = torch.utils.data.DataLoader(
        id_dataset,
        batch_size=2,
        # collate_fn = id_collate,
        pin_memory=True,
        drop_last=True
    )

# ood_collate = OtherMixupCollate(37)
ood_dataset = Others_ImageFolder('./data/OOD/valid', start_class = 32, end_class=37, transform=transform)
ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=2,
        # collate_fn = ood_collate,
        pin_memory=True,
        drop_last=True
    )

for idx, ((id_imgs, id_labels), (ood_imgs, ood_labels,index)) in enumerate(zip(id_loader, ood_loader)):
    imgs = torch.cat((id_imgs, ood_imgs))
    labels = torch.cat((id_labels, ood_labels))
    print(labels.size())
    print(labels)
    a = torch.where(labels == 0, 1, labels)
    print(a)
    break