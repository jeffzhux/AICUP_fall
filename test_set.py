import os
import argparse
import time

import torch
import torch.nn as nn

from datasets.dataset import TestTimeAICUP_DataSet, AICUP_ImageFolder
from datasets.collates import TestTimeCollate
from datasets.transforms.aicup import baseOnImageNet
from models.build import build_model

from utils.test_time_augmentation import TestTimeAugmentation
from utils.config import Config
from utils.util import AverageMeter, TrackMeter, accuracy, adjust_learning_rate, format_time, set_seed

test_set = TestTimeAICUP_DataSet(
    root=f'./data/test',
    transform = baseOnImageNet(),
    num_of_trans=1)
test_collate = TestTimeCollate()
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size = 16,
    num_workers = 4,
    collate_fn = test_collate,
    drop_last = False
)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    args = parser.parse_args()

    return args

def get_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)

    cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    cfg.work_dir = os.path.join(cfg.work_dir, f'{cfg.timestamp}')

    # worker
    cfg.num_workers = 4
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None, f'{cfg.cfgname} is not exist'

    # seed
    if not hasattr(cfg, 'seed'):
        cfg.seed = 25
    set_seed(cfg.seed)

    return cfg

def load_weights(
    ckpt_path: str,
    model: nn.Module) -> None:

    # load checkpoint 
    print(f"==> Loading Checkpoint {ckpt_path}")
    assert os.path.isfile(ckpt_path), 'file is not exist'
    ckpt = torch.load(ckpt_path, map_location='cuda')
    # for k, v in ckpt['model_state'].items():
    #     # if 'module' not in k:
    #     #     k = f'module.{k}'
    #     # ckpt['model_state'][k] = v
    #     print(k)
    # print('-------------------------')
    # for k, v in model.named_parameters():
    #     print(k)
    model.load_state_dict(ckpt['model_state'])
    


if __name__ == '__main__':    

    args = get_args()
    cfg = get_config(args)

    model = model = build_model(cfg.model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
    
    if True:
        load_weights(cfg.load, model)
    
    print(f"==> Start testing ....")
    model.eval()
    tta = TestTimeAugmentation(cfg.test_time_augmentation)
    pred = []
    target = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()

            # forward
            logits = model(images)
            logits = tta(logits)
            
            pred.append(logits)
            target.append(labels)
            
        pred = torch.cat(pred)
        target = torch.cat(target)
        
        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        acc1, acc5 = acc1.item(), acc5.item()
        print(f'acc1 : {acc1}, acc5: {acc5}')