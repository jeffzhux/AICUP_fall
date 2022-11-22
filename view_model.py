import torch
import torch.nn as nn
from datasets.build import build_dataset
from datasets.collates.build import build_collate
from datasets.transforms import baseOnImageNet
from datasets.sampler.sampler import OOD_Sampler
from datasets.collates import RandomMixupCutMixCollate

from utils.config import ConfigDict

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
cfg.data = ConfigDict(
    collate = ConfigDict(
        type = 'TestTimeCollate',
    ),
    test = ConfigDict(
        root=f'data/ID/valid',
        type = 'TestTimeAICUP_DataSet',
        transform = dict(
            type='base',
        ),
        base_transform = dict(
            type='base',
            size = (224, 224)
        ),
        num_of_trans = 0
    )
    
)

test_set = build_dataset(cfg.data.test)
print(test_set.class_to_idx)
ood_sampler = OOD_Sampler(test_set, shuffle=True)
test_collate = build_collate(cfg.data.collate)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size = cfg.bsz_gpu,
    num_workers = 0,
    shuffle = False,
    collate_fn = test_collate,
    pin_memory = True,
    drop_last = False
)

print(list(map(lambda x: x[0].split('\\')[-1], test_set.imgs)))
ood_sampler.set_epoch(1)
for img, target in test_loader:
    
    print(target)
    break
